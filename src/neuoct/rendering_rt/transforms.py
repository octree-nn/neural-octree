"""Coordinate transforms for the NeuOct ray tracer.

There are two distinct spaces:

``object``
    The checkpoint's canonical object frame.  The packed NeuOct inference
    bundle and its camera captures both live in this frame (Z-up for the
    Objaverse orbit checkpoints).

``octree``
    The integer voxel frame used by ``LocalSparseOctree`` and the renderer.
    The audited mapping is::

        p_octree = p_object * point_scale * ray_scale * 2**depth + 2**(depth-1)

    Because the scale factor is uniform, directions and normals only need
    normalising after the (linear) scaling, and ray parameters scale by the
    same constant.

``scene``
    The world frame the offline tracer renders in (Y-up "chess" for the
    placements shipped under ``configs/raytracing``).  A :class:`SimilarityTransform`
    maps object <-> scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
from torch import Tensor
from torch.nn import functional as F

# Standard Z-up (Blender/Objaverse) -> Y-up (glTF/chess) rotation about X by -90 deg.
Z_UP_TO_Y_UP: tuple[tuple[float, ...], ...] = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, -1.0, 0.0),
)

_ORTHONORMAL_TOL = 1.0e-4


def _normalize_directions(values: Tensor, eps: float = 1.0e-8) -> Tensor:
    return F.normalize(values, dim=-1, eps=eps)


@dataclass(frozen=True)
class SimilarityTransform:
    """A validated proper similarity transform ``p_scene = s R p_object + t``.

    ``rotation`` must be a proper orthonormal 3x3 matrix (det ``+1``); a
    reflection is rejected because it would flip surface normals under the
    inverse-transpose.  ``scale`` must be positive and uniform.
    """

    scale: float = 1.0
    rotation: Tensor = field(default_factory=lambda: torch.eye(3, dtype=torch.float32))
    translation: Tensor = field(default_factory=lambda: torch.zeros(3, dtype=torch.float32))
    name: str = "object_to_scene"

    def __post_init__(self) -> None:
        rotation = torch.as_tensor(self.rotation, dtype=torch.float32).reshape(3, 3)
        translation = torch.as_tensor(self.translation, dtype=torch.float32).reshape(3)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "scale", float(self.scale))
        self.validate()

    def validate(self) -> None:
        if not torch.isfinite(torch.tensor(self.scale)) or self.scale <= 0.0:
            raise ValueError("SimilarityTransform.scale must be finite and positive")
        # The identity must live on the rotation's own device/dtype: a CUDA
        # placement (the normal case when the NeuOct model runs on GPU) would
        # otherwise fail with a CPU/CUDA mismatch inside ``allclose``.
        identity = self.rotation.transpose(0, 1) @ self.rotation
        reference = torch.eye(
            3, dtype=identity.dtype, device=identity.device
        )
        if not torch.allclose(identity, reference, atol=_ORTHONORMAL_TOL):
            raise ValueError("SimilarityTransform.rotation must be orthonormal")
        determinant = float(torch.linalg.det(self.rotation.detach().cpu()))
        if abs(determinant - 1.0) > _ORTHONORMAL_TOL:
            raise ValueError(
                "SimilarityTransform.rotation must be a proper rotation "
                f"(det = +1), got det = {determinant:.6f}"
            )
        if tuple(self.translation.shape) != (3,):
            raise ValueError("SimilarityTransform.translation must have shape (3,)")

    def _aligned(self, value: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(rotation, translation)`` on ``value``'s device and dtype.

        The transform is stored once (normally on CPU float32), but callers may
        hand it scene-space rays that already live on a CUDA device.  Moving the
        small 3x3/3-element operands here keeps ``transform_points`` and friends
        usable from any device without a CPU/CUDA mismatch.
        """
        device = value.device
        dtype = value.dtype if value.is_floating_point() else self.rotation.dtype
        return (
            self.rotation.to(device=device, dtype=dtype),
            self.translation.to(device=device, dtype=dtype),
        )

    @property
    def matrix(self) -> Tensor:
        matrix = torch.eye(4, dtype=self.rotation.dtype, device=self.rotation.device)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    def transform_points(self, points: Tensor) -> Tensor:
        points = torch.as_tensor(points)
        rotation, translation = self._aligned(points)
        return self.scale * (points @ rotation.transpose(0, 1)) + translation

    def inverse_points(self, points: Tensor) -> Tensor:
        points = torch.as_tensor(points)
        rotation, translation = self._aligned(points)
        local = (points - translation) / self.scale
        return local @ rotation

    def transform_directions(self, directions: Tensor) -> Tensor:
        directions = torch.as_tensor(directions)
        rotation, _ = self._aligned(directions)
        return _normalize_directions(directions @ rotation.transpose(0, 1))

    def inverse_directions(self, directions: Tensor) -> Tensor:
        directions = torch.as_tensor(directions)
        rotation, _ = self._aligned(directions)
        return _normalize_directions(directions @ rotation)

    def transform_normals(self, normals: Tensor) -> Tensor:
        # Inverse-transpose of ``s R`` is ``(1/s) R``; after normalising, only R remains.
        normals = torch.as_tensor(normals)
        rotation, _ = self._aligned(normals)
        return _normalize_directions(normals @ rotation.transpose(0, 1))

    def inverse_normals(self, normals: Tensor) -> Tensor:
        normals = torch.as_tensor(normals)
        rotation, _ = self._aligned(normals)
        return _normalize_directions(normals @ rotation)

    def inverse(self) -> "SimilarityTransform":
        rotation_t = self.rotation.transpose(0, 1)
        return SimilarityTransform(
            scale=1.0 / self.scale,
            rotation=rotation_t,
            translation=-(rotation_t @ self.translation) / self.scale,
            name=f"inverse({self.name})",
        )

    def compose(self, other: "SimilarityTransform") -> "SimilarityTransform":
        """Return ``self(other(p))`` (apply ``other`` first)."""
        return SimilarityTransform(
            scale=self.scale * other.scale,
            rotation=self.rotation @ other.rotation,
            translation=self.scale * (self.rotation @ other.translation) + self.translation,
            name=f"{self.name}({other.name})",
        )

    def is_identity(self, *, atol: float = 1.0e-6) -> bool:
        """True when the transform maps every point to itself.

        Used as a fast path so an unplaced object is handed to the network with
        bit-identical rays: applying ``R = I`` and renormalising perturbs every
        direction in the last float32 bits, and the fp16 checkpoint can turn
        that into a visible difference on knife-edge pixels.
        """
        rotation, translation = self.rotation, self.translation
        if abs(self.scale - 1.0) > atol:
            return False
        if bool((translation.abs() > atol).any()):
            return False
        identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
        return bool(torch.allclose(rotation, identity, atol=atol))

    def to(self, device: torch.device | str) -> "SimilarityTransform":
        return replace(
            self,
            rotation=self.rotation.to(device),
            translation=self.translation.to(device),
        )

    @classmethod
    def identity(cls, name: str = "identity") -> "SimilarityTransform":
        return cls(name=name)

    @classmethod
    def z_up_to_y_up(
        cls,
        scale: float = 1.0,
        translation: Tensor | tuple[float, float, float] = (0.0, 0.0, 0.0),
        name: str = "z_up_to_y_up",
    ) -> "SimilarityTransform":
        return cls(
            scale=scale,
            rotation=torch.tensor(Z_UP_TO_Y_UP, dtype=torch.float32),
            translation=torch.as_tensor(translation, dtype=torch.float32),
            name=name,
        )


@dataclass(frozen=True)
class OctreeTransform:
    """Exact NeuOct object-frame <-> integer-octree mapping.

    ``linear`` is ``point_scale * ray_scale * 2**depth`` and ``offset`` is
    ``2**(depth-1)``; both are exposed so callers never re-derive the constants.
    """

    depth: int
    point_scale: float = 0.99
    ray_scale: float = 0.5

    def __post_init__(self) -> None:
        if int(self.depth) <= 0:
            raise ValueError("OctreeTransform.depth must be positive")
        if self.point_scale <= 0.0 or self.ray_scale <= 0.0:
            raise ValueError("point_scale and ray_scale must be positive")
        object.__setattr__(self, "depth", int(self.depth))
        object.__setattr__(self, "point_scale", float(self.point_scale))
        object.__setattr__(self, "ray_scale", float(self.ray_scale))

    @property
    def scale(self) -> int:
        return 1 << self.depth

    @property
    def linear(self) -> float:
        return self.point_scale * self.ray_scale * float(self.scale)

    @property
    def offset(self) -> float:
        return float(self.scale) * 0.5

    def object_to_octree(self, points: Tensor) -> Tensor:
        # The multiplication order mirrors the deployed ``SparseRenderer``
        # (``origins * (point_scale * ray_scale) * scale + scale / 2``) bit for
        # bit; folding the two scalars into one ``linear`` first can differ by an
        # ulp, which is enough to move an fp16 network prediction.
        return (
            torch.as_tensor(points) * (self.point_scale * self.ray_scale) * float(self.scale)
            + self.offset
        )

    def octree_to_object(self, points: Tensor) -> Tensor:
        return (torch.as_tensor(points) - self.offset) / self.linear

    def object_to_octree_directions(self, directions: Tensor) -> Tensor:
        return _normalize_directions(torch.as_tensor(directions) * self.linear)

    def octree_to_object_directions(self, directions: Tensor) -> Tensor:
        return _normalize_directions(torch.as_tensor(directions))

    def object_to_octree_normals(self, normals: Tensor) -> Tensor:
        # Uniform scale: the inverse-transpose normalises back to the same vector.
        return _normalize_directions(torch.as_tensor(normals) * self.linear)

    def octree_to_object_normals(self, normals: Tensor) -> Tensor:
        return _normalize_directions(torch.as_tensor(normals))

    def object_to_octree_t(self, t: Tensor | float) -> Tensor:
        return torch.as_tensor(t) * self.linear

    def octree_to_object_t(self, t: Tensor | float) -> Tensor:
        return torch.as_tensor(t) / self.linear

    def as_dict(self) -> dict[str, float | int]:
        return {
            "depth": self.depth,
            "point_scale": self.point_scale,
            "ray_scale": self.ray_scale,
            "scale": self.scale,
            "linear": self.linear,
            "offset": self.offset,
        }
