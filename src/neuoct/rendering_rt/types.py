"""Batched ray-tracing value types shared by the offline NeuOct ray tracer.

The tracer is a small, self-contained CPU-friendly offline renderer.  Every
module exchanges only the two dataclasses defined here plus :class:`Material`,
so a primitive implemented against a mesh and a primitive implemented against
the NeuOct local scene can be combined by the scene without either one knowing
about the other.

Every batch carries exactly one device.  :class:`MaterialBatch`, :class:`Rays`
and :class:`RayHit` each elect a *leading* field (``base_color``, ``origins``
and ``hit`` respectively) and coerce every other field onto that field's device,
so a batch can never mix CPU and CUDA tensors.  Indexing helpers move the index
onto the batch device, which makes gathering with CUDA index tensors work.
Rays are *always* stored with unit-norm directions; this makes the ray parameter
``t`` a metric distance and lets primitives share it without re-normalising.
``t_min`` and ``t_max`` are stored per ray but may be created from python
scalars; both are expanded onto the ray device.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
from torch import Tensor
from torch.nn import functional as F

# Background defaults used by safe miss constructors.  They are deliberately
# finite and deterministic so a miss can always be shaded without NaN checks.
MISS_NORMAL = (0.0, 0.0, 1.0)
MISS_ALBEDO = (0.0, 0.0, 0.0)
BACKGROUND_MATERIAL_COLOR = (1.0, 1.0, 1.0)


def resolve_device(spec: torch.device | str | int | None = None) -> torch.device:
    """Resolve a device spec, validating ``cuda:N`` before any work starts.

    Accepts ``None`` (CPU), ``"cpu"``/``"cpu:N"``, ``"cuda"``, ``"cuda:N"`` and a
    bare CUDA index.  Failing here -- rather than deep inside the first tensor
    operation -- turns ``--device cuda:3`` on a two-GPU host into an actionable
    message.  ``"cuda"`` resolves to ``cuda:0`` so downstream code always has an
    explicit index.
    """
    if spec is None:
        return torch.device("cpu")
    device = torch.device("cuda", int(spec)) if isinstance(spec, int) else torch.device(spec)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device {str(device)!r} was requested but torch.cuda.is_available() is False"
            )
        index = 0 if device.index is None else int(device.index)
        count = int(torch.cuda.device_count())
        if not 0 <= index < count:
            raise RuntimeError(
                f"CUDA device index {index} is out of range; this host exposes {count} device(s)"
            )
        return torch.device("cuda", index)
    if device.type == "cpu":
        # ``cpu:0`` and ``cpu`` are the same device; normalise so equality holds.
        return torch.device("cpu")
    raise ValueError(f"unsupported device type {device.type!r}; use 'cpu' or 'cuda[:N]'")


def _as_tensor(value: object, *, shape: tuple[int, ...], device, dtype, name: str) -> Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    return tensor


@dataclass(frozen=True)
class Material:
    """A deterministic, physically-motivated surface description.

    ``base_color`` is stored in **linear** space.  ``base_color_srgb`` records
    where the value came from so a caller can round-trip it for debugging.
    ``thin_walled`` documents that transmission is modelled as a zero-thickness
    interface (enter and leave the surface at the same point); it is not a
    closed solid medium.
    """

    base_color: Tensor = field(default_factory=lambda: torch.ones(3))
    roughness: float = 0.5
    metallic: float = 0.0
    specular: float = 0.04
    transmission: float = 0.0
    ior: float = 1.5
    thin_walled: bool = True
    emission: Tensor = field(default_factory=lambda: torch.zeros(3))
    base_color_srgb: bool = False
    texture: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_color",
            torch.as_tensor(self.base_color, dtype=torch.float32).reshape(3),
        )
        object.__setattr__(
            self,
            "emission",
            torch.as_tensor(self.emission, dtype=torch.float32).reshape(3),
        )
        for name in ("roughness", "metallic", "specular", "transmission", "ior"):
            value = float(getattr(self, name))
            if not torch.isfinite(torch.tensor(value)):
                raise ValueError(f"Material.{name} must be finite")
        if not 0.0 <= self.roughness <= 1.0:
            raise ValueError("Material.roughness must be in [0, 1]")
        if not 0.0 <= self.metallic <= 1.0:
            raise ValueError("Material.metallic must be in [0, 1]")
        if self.specular < 0.0:
            raise ValueError("Material.specular must be non-negative")
        if not 0.0 <= self.transmission <= 1.0:
            raise ValueError("Material.transmission must be in [0, 1]")
        if self.ior <= 0.0:
            raise ValueError("Material.ior must be positive")

    def to(self, device: torch.device | str) -> "Material":
        return replace(
            self,
            base_color=self.base_color.to(device),
            emission=self.emission.to(device),
        )

    @property
    def f0(self) -> Tensor:
        """Normal-incidence Fresnel reflectance (dielectric + metal mix)."""
        dielectric = self.specular * (1.0 - self.metallic)
        metal = self.base_color * self.metallic
        return (dielectric + metal).clamp(0.0, 1.0)


@dataclass(frozen=True)
class MaterialBatch:
    """Per-ray material parameters used by the integrator.

    A single :class:`Material` cannot describe a scene whose winning primitive
    changes per ray, so the scene flattens the winning material fields into this
    batch.  ``thin_walled`` and ``base_color_srgb`` are stored as boolean masks.
    """

    base_color: Tensor
    roughness: Tensor
    metallic: Tensor
    specular: Tensor
    transmission: Tensor
    ior: Tensor
    thin_walled: Tensor
    emission: Tensor
    base_color_srgb: Tensor | None = None
    texture: object | None = None

    def __post_init__(self) -> None:
        base_color = torch.as_tensor(self.base_color, dtype=torch.float32)
        if base_color.ndim != 2 or base_color.shape[1] != 3:
            raise ValueError(
                f"MaterialBatch.base_color must have shape (N, 3), got {tuple(base_color.shape)}"
            )
        count = int(base_color.shape[0])
        # ``base_color`` defines the batch device; the remaining fields follow it.
        device = base_color.device
        object.__setattr__(self, "base_color", base_color)
        emission = torch.as_tensor(self.emission, dtype=torch.float32, device=device)
        if emission.shape != (count, 3):
            raise ValueError(f"MaterialBatch.emission must have shape ({count}, 3)")
        object.__setattr__(self, "emission", emission)
        for name in ("roughness", "metallic", "specular", "transmission", "ior"):
            value = torch.as_tensor(getattr(self, name), dtype=torch.float32, device=device)
            if value.shape != (count,):
                raise ValueError(f"MaterialBatch.{name} must have shape ({count},)")
            object.__setattr__(self, name, value)
        thin = torch.as_tensor(self.thin_walled, dtype=torch.bool, device=device)
        if thin.shape != (count,):
            raise ValueError(f"MaterialBatch.thin_walled must have shape ({count},)")
        object.__setattr__(self, "thin_walled", thin)
        srgb = self.base_color_srgb
        if srgb is not None:
            srgb = torch.as_tensor(srgb, dtype=torch.bool, device=device)
            if srgb.shape != (count,):
                raise ValueError("MaterialBatch.base_color_srgb must have shape (N,)")
            object.__setattr__(self, "base_color_srgb", srgb)

    @property
    def num_rays(self) -> int:
        return int(self.roughness.shape[0])

    @property
    def device(self) -> torch.device:
        return self.base_color.device

    @classmethod
    def from_material(cls, material: Material, count: int, *, device=None) -> "MaterialBatch":
        device = material.base_color.device if device is None else device
        return cls(
            base_color=material.base_color.to(device).expand(count, 3).clone(),
            roughness=torch.full((count,), float(material.roughness), device=device),
            metallic=torch.full((count,), float(material.metallic), device=device),
            specular=torch.full((count,), float(material.specular), device=device),
            transmission=torch.full((count,), float(material.transmission), device=device),
            ior=torch.full((count,), float(material.ior), device=device),
            thin_walled=torch.full((count,), bool(material.thin_walled), dtype=torch.bool, device=device),
            emission=material.emission.to(device).expand(count, 3).clone(),
            base_color_srgb=torch.full(
                (count,), bool(material.base_color_srgb), dtype=torch.bool, device=device
            ),
            texture=material.texture,
        )

    def index(self, index: Tensor) -> "MaterialBatch":
        index = torch.as_tensor(index, device=self.device)
        return MaterialBatch(
            base_color=self.base_color[index],
            roughness=self.roughness[index],
            metallic=self.metallic[index],
            specular=self.specular[index],
            transmission=self.transmission[index],
            ior=self.ior[index],
            thin_walled=self.thin_walled[index],
            emission=self.emission[index],
            base_color_srgb=(
                None if self.base_color_srgb is None else self.base_color_srgb[index]
            ),
            texture=None,
        )

    def to(self, device: torch.device | str) -> "MaterialBatch":
        return MaterialBatch(
            base_color=self.base_color.to(device),
            roughness=self.roughness.to(device),
            metallic=self.metallic.to(device),
            specular=self.specular.to(device),
            transmission=self.transmission.to(device),
            ior=self.ior.to(device),
            thin_walled=self.thin_walled.to(device),
            emission=self.emission.to(device),
            base_color_srgb=(
                None if self.base_color_srgb is None else self.base_color_srgb.to(device)
            ),
            texture=self.texture,
        )


@dataclass(frozen=True)
class Rays:
    """A batch of unit-direction rays with per-ray clipping bounds.

    ``t_min``/``t_max`` accept python scalars or tensors with shape ``(N,)`` or
    broadcastable to it.  The constructor normalises ``directions`` in place
    (copying the input) and expands both clipping bounds to ``(N,)``.
    """

    origins: Tensor
    directions: Tensor
    t_min: Tensor = 0.0
    t_max: Tensor = float("inf")

    def __post_init__(self) -> None:
        origins = torch.as_tensor(self.origins, dtype=torch.float32)
        directions = torch.as_tensor(self.directions, dtype=torch.float32)
        if origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError(f"Rays.origins must have shape (N, 3), got {tuple(origins.shape)}")
        if directions.shape != origins.shape:
            raise ValueError(
                "Rays.directions must match origins shape "
                f"{(tuple(origins.shape))}, got {tuple(directions.shape)}"
            )
        norms = directions.norm(dim=-1, keepdim=True)
        if bool((norms <= 1.0e-12).any()):
            raise ValueError("Rays.directions must be non-zero")
        # Only renormalise when the input is not already unit.  Dividing an
        # exactly-unit vector by its float32 norm perturbs every component by
        # ~1e-7, and the NeuOct checkpoint is evaluated in fp16 where such a
        # perturbation can flip a rounding decision and move a predicted normal
        # by ~1e-3.  Skipping the no-op division is what makes the strict
        # primary-ray parity test reproducible; the stored directions are still
        # unit to well within 1e-6.
        if not bool(((norms - 1.0).abs() <= 1.0e-6).all()):
            directions = directions / norms.clamp_min(1.0e-12)
        count = origins.shape[0]
        # Scalar or tensor bounds must end up on the ray device: a CUDA batch
        # built from python floats would otherwise carry CPU clipping bounds and
        # fail on the first device-indexed gather.
        t_min = torch.as_tensor(self.t_min, dtype=torch.float32, device=origins.device).reshape(-1)
        t_max = torch.as_tensor(self.t_max, dtype=torch.float32, device=origins.device).reshape(-1)
        for name, value in (("t_min", t_min), ("t_max", t_max)):
            if value.numel() not in (1, count):
                raise ValueError(
                    f"Rays.{name} must be a scalar or have {count} element(s), "
                    f"got {value.numel()}"
                )
        t_min = t_min.expand(count).clone()
        t_max = t_max.expand(count).clone()
        if bool((t_min > t_max).any()):
            raise ValueError("Rays.t_min must not exceed t_max")
        object.__setattr__(self, "origins", origins)
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "t_min", t_min)
        object.__setattr__(self, "t_max", t_max)

    @property
    def num_rays(self) -> int:
        return int(self.origins.shape[0])

    @property
    def device(self) -> torch.device:
        return self.origins.device

    @property
    def dtype(self) -> torch.dtype:
        return self.origins.dtype

    def at(self, t: Tensor | float) -> Tensor:
        """Return ``origin + t * direction`` for a scalar or per-ray ``t``."""
        t = torch.as_tensor(t, device=self.device, dtype=self.dtype)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return self.origins + t * self.directions

    def index(self, index: Tensor) -> "Rays":
        index = torch.as_tensor(index, device=self.device)
        return Rays(
            self.origins[index],
            self.directions[index],
            self.t_min[index],
            self.t_max[index],
        )

    def to(self, device: torch.device | str) -> "Rays":
        return Rays(
            self.origins.to(device),
            self.directions.to(device),
            self.t_min.to(device),
            self.t_max.to(device),
        )


@dataclass(frozen=True)
class RayHit:
    """Per-ray intersection result.

    ``albedo`` is the **linear** surface colour ready for shading.
    ``albedo_srgb`` keeps the model-native sRGB colour when the primitive can
    provide it (the NeuOct renderer predicts sRGB; mesh base colour factors are
    linear).  A miss has ``hit=False`` and finite, deterministic fields.

    ``geometric_normal`` / ``front_face`` are optional and only populated by
    primitives that can report the two-sided orientation of a *closed* surface
    (ordinary meshes).  ``normal`` may be flipped to face the ray for shading,
    whereas ``geometric_normal`` preserves the winding, and ``front_face`` is
    ``True`` when the ray arrives against that winding.  Closed-dielectric
    refraction needs exactly this state to pick ``eta_i / eta_t``; primitives
    that cannot provide it (NeuOct) leave both ``None``, which is what forces
    the documented thin-walled transmission approximation there.
    """

    hit: Tensor
    t: Tensor
    position: Tensor
    normal: Tensor
    barycentric: Tensor
    triangle_index: Tensor
    primitive_id: Tensor
    albedo: Tensor
    albedo_srgb: Tensor
    hit_probability: Tensor
    uv: Tensor | None = None
    material: Material | None = None
    materials: MaterialBatch | None = None
    geometric_normal: Tensor | None = None
    front_face: Tensor | None = None

    def __post_init__(self) -> None:
        hit = torch.as_tensor(self.hit, dtype=torch.bool)
        if hit.ndim != 1:
            raise ValueError(f"RayHit.hit must be one-dimensional, got {tuple(hit.shape)}")
        count = int(hit.shape[0])
        # ``hit`` defines the batch device; every other field is coerced onto it
        # so a hit batch can never mix CPU and CUDA tensors.
        device = hit.device
        object.__setattr__(self, "hit", hit)
        for name in ("t", "hit_probability"):
            value = torch.as_tensor(getattr(self, name), dtype=torch.float32, device=device)
            if value.shape != (count,):
                raise ValueError(f"RayHit.{name} must have shape ({count},)")
            object.__setattr__(self, name, value)
        for name in ("position", "normal", "albedo", "albedo_srgb"):
            value = torch.as_tensor(getattr(self, name), dtype=torch.float32, device=device)
            if value.shape != (count, 3):
                raise ValueError(f"RayHit.{name} must have shape ({count}, 3)")
            object.__setattr__(self, name, value)
        barycentric = torch.as_tensor(self.barycentric, dtype=torch.float32, device=device)
        if barycentric.shape != (count, 3):
            raise ValueError(f"RayHit.barycentric must have shape ({count}, 3)")
        object.__setattr__(self, "barycentric", barycentric)
        for name in ("triangle_index", "primitive_id"):
            value = torch.as_tensor(getattr(self, name), dtype=torch.long, device=device)
            if value.shape != (count,):
                raise ValueError(f"RayHit.{name} must have shape ({count},)")
            object.__setattr__(self, name, value)
        if self.uv is not None:
            uv = torch.as_tensor(self.uv, dtype=torch.float32, device=device)
            if uv.shape != (count, 2):
                raise ValueError(f"RayHit.uv must have shape ({count}, 2)")
            object.__setattr__(self, "uv", uv)
        for name in ("geometric_normal",):
            value = getattr(self, name)
            if value is not None:
                value = torch.as_tensor(value, dtype=torch.float32, device=device)
                if value.shape != (count, 3):
                    raise ValueError(f"RayHit.{name} must have shape ({count}, 3)")
                object.__setattr__(self, name, value)
        if self.front_face is not None:
            value = torch.as_tensor(self.front_face, dtype=torch.bool, device=device)
            if value.shape != (count,):
                raise ValueError(f"RayHit.front_face must have shape ({count},)")
            object.__setattr__(self, "front_face", value)

    @property
    def num_rays(self) -> int:
        return int(self.hit.shape[0])

    @property
    def device(self) -> torch.device:
        return self.hit.device

    @classmethod
    def miss(cls, count: int, *, device=None, dtype=torch.float32) -> "RayHit":
        """Build a fully-populated miss batch with no NaN/Inf fields."""
        device = torch.device("cpu") if device is None else device
        return cls(
            hit=torch.zeros(count, dtype=torch.bool, device=device),
            t=torch.full((count,), float("inf"), dtype=dtype, device=device),
            position=torch.zeros(count, 3, dtype=dtype, device=device),
            normal=torch.tensor(MISS_NORMAL, dtype=dtype, device=device).expand(count, 3).clone(),
            barycentric=torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
            .expand(count, 3)
            .clone(),
            triangle_index=torch.full((count,), -1, dtype=torch.long, device=device),
            primitive_id=torch.full((count,), -1, dtype=torch.long, device=device),
            albedo=torch.zeros(count, 3, dtype=dtype, device=device),
            albedo_srgb=torch.zeros(count, 3, dtype=dtype, device=device),
            hit_probability=torch.zeros(count, dtype=dtype, device=device),
            uv=None,
            material=None,
            materials=None,
            geometric_normal=None,
            front_face=None,
        )

    def index(self, index: Tensor) -> "RayHit":
        index = torch.as_tensor(index, device=self.device)
        return RayHit(
            hit=self.hit[index],
            t=self.t[index],
            position=self.position[index],
            normal=self.normal[index],
            barycentric=self.barycentric[index],
            triangle_index=self.triangle_index[index],
            primitive_id=self.primitive_id[index],
            albedo=self.albedo[index],
            albedo_srgb=self.albedo_srgb[index],
            hit_probability=self.hit_probability[index],
            uv=None if self.uv is None else self.uv[index],
            material=self.material,
            materials=None if self.materials is None else self.materials.index(index),
            geometric_normal=(
                None if self.geometric_normal is None else self.geometric_normal[index]
            ),
            front_face=None if self.front_face is None else self.front_face[index],
        )

    def to(self, device: torch.device | str) -> "RayHit":
        return RayHit(
            hit=self.hit.to(device),
            t=self.t.to(device),
            position=self.position.to(device),
            normal=self.normal.to(device),
            barycentric=self.barycentric.to(device),
            triangle_index=self.triangle_index.to(device),
            primitive_id=self.primitive_id.to(device),
            albedo=self.albedo.to(device),
            albedo_srgb=self.albedo_srgb.to(device),
            hit_probability=self.hit_probability.to(device),
            uv=None if self.uv is None else self.uv.to(device),
            material=None if self.material is None else self.material.to(device),
            materials=None if self.materials is None else self.materials.to(device),
            geometric_normal=(
                None if self.geometric_normal is None else self.geometric_normal.to(device)
            ),
            front_face=None if self.front_face is None else self.front_face.to(device),
        )


def gather_ray_directions(pins: Tensor, pouts: Tensor) -> Tensor:
    """Unit directions from entry/exit points (used by the NeuOct model)."""
    if pins.shape != pouts.shape:
        raise ValueError("pins and pouts must share a shape")
    return F.normalize(pouts - pins, dim=-1, eps=1.0e-8)


def stable_lexsort(*keys: Tensor) -> Tensor:
    """Indices that sort ``keys[0]`` primarily, then ``keys[1]``, and so on.

    ``torch.lexsort`` is not available on every supported torch build, so this
    folds successive *stable* sorts from the least significant key to the most
    significant one.  All keys must be one-dimensional and share a length.
    """
    if not keys:
        raise ValueError("stable_lexsort requires at least one key")
    count = int(keys[0].numel())
    for key in keys:
        if key.ndim != 1 or int(key.numel()) != count:
            raise ValueError("stable_lexsort keys must be one-dimensional and equal length")
    order = torch.arange(count, device=keys[0].device)
    for key in reversed(keys):
        order = order[torch.argsort(key[order], stable=True)]
    return order
