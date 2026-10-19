"""A small batched offline path/direct integrator.

Phase one supports a constant-colour environment (the extensibility point for
an HDRI later), a configurable maximum depth, and three deterministic sampling
modes:

``direct``
    Shade the primary hit with ``emission + diffuse_albedo * environment``.
``one_bounce``
    One secondary ray whose terminal hit contributes the direct term.
``path``
    Up to ``max_depth`` secondary rays; each surface adds emission and the
    terminal hit adds the direct term.

Sampling is driven by :class:`~.rng.CounterRNG`, so a render is reproducible and
independent of the batch/chunk sizes.  There is no UI and no screenshot path.
Rendering is internally **sample-batched**: :meth:`Integrator.render` walks the
sample axis in blocks of ``sample_batch`` and accumulates, so peak path tensors
are bounded by ``num_pixels * sample_batch`` rather than by
``num_pixels * samples``.  Every random number is addressed by
``(pixel, sample, dimension)`` and never drawn from a running counter, so the
batching order cannot change a single sample.

Branch weighting
----------------
Each non-terminal surface hit picks one of three lobes -- transmission,
reflection or diffuse -- from a single uniform, using the material's own
transmission/metallic/specular fractions as the branch probabilities.  The
throughput multiplier is ``lobe / p(selected branch)`` (clamped, and zero when a
branch is unreachable), so the one-sample estimator is unbiased for the sum of
the lobes instead of being crushed by the selection probability.  Reflection
uses the angle-dependent Schlick Fresnel reflectance at the per-channel ``F0``;
for a metallic surface that ``F0`` is the *combined* learned/texture surface
colour, not the bare specular tint.

Transmission has two clearly separated models:

``thin-walled`` (``Material.thin_walled``)
    Ordinary mesh sheets retain the glTF-style zero-thickness straight-through
    model.  NeuOct has no winding-derived closed-surface state, so its learned
    normal is used for a two-sided Snell interface instead; the path state tracks
    entry/exit so the transmitted ray bends while remaining transparent.  This
    is still an approximation because the learned representation does not expose
    a watertight interior.  Such paths are counted in
    ``RenderBuffers.thin_walled_transmission_paths`` and the approximation is
    declared in the manifest.
``closed dielectric`` (``thin_walled=False`` on an ordinary mesh)
    Real Snell refraction with total internal reflection, driven by the mesh's
    winding-derived ``front_face`` / ``geometric_normal`` state, so ``eta_i`` and
    ``eta_t`` are correct on both entry and exit (see
    :func:`~.sampling.snell_refract`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .camera import Camera
from .color import linear_to_display
from .rng import CounterRNG
from .sampling import (
    cosine_hemisphere_sample,
    dielectric_f0,
    ggx_reflection,
    refract,
    schlick_fresnel,
    snell_refract,
)
from .scene import Scene
from .types import MaterialBatch, RayHit, Rays

_MODES = ("direct", "one_bounce", "path")
# Lower clamp for the importance-sampling division.  A branch with zero
# probability is never selected, so this only guards floating-point noise.
_MIN_BRANCH_PROB = 1.0e-4


class Environment(ABC):
    """Radiance source; subclassed by HDRI environments in phase two."""

    @abstractmethod
    def radiance(self, directions: Tensor) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    @abstractmethod
    def ambient(self) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def to(self, device: torch.device | str) -> "Environment":
        return self


class ConstantEnvironment(Environment):
    def __init__(self, color: Tensor | tuple[float, ...] | float = 0.0) -> None:
        color = torch.as_tensor(color, dtype=torch.float32)
        if color.ndim == 0:
            color = color.expand(3)
        color = color.reshape(3)
        if bool((color < 0.0).any()):
            raise ValueError("Environment colour must be non-negative")
        self.color = color

    @property
    def ambient(self) -> Tensor:
        return self.color

    def radiance(self, directions: Tensor) -> Tensor:
        directions = torch.as_tensor(directions)
        return self.color.to(directions.device).expand_as(directions)

    def to(self, device: torch.device | str) -> "ConstantEnvironment":
        return ConstantEnvironment(self.color.to(device))


@dataclass
class RenderBuffers:
    color: Tensor
    albedo: Tensor
    normal: Tensor
    depth: Tensor
    hit: Tensor
    environment: Environment
    mode: str
    samples: int
    max_depth: int
    # --- phase-two AOVs and diagnostics (optional: phase-one callers unaffected)
    primitive_id: Tensor | None = None
    coverage: dict[str, int] = field(default_factory=dict)
    sample_batch: int = 0
    thin_walled_transmission_paths: int = 0
    closed_transmission_paths: int = 0
    total_internal_reflection_paths: int = 0
    secondary_hits_by_primitive: dict[str, int] = field(default_factory=dict)
    declared_primitives: dict[str, str] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.color.shape[0]), int(self.color.shape[1]))

    @property
    def device(self) -> torch.device:
        return self.color.device

    def to_display(self, *, mode: str = "aces", exposure: float = 1.0, encoded: bool = True) -> Tensor:
        return linear_to_display(self.color, mode=mode, exposure=exposure, encoded=encoded)


class Integrator:
    def __init__(
        self,
        scene: Scene,
        *,
        environment: Environment | None = None,
        mode: str = "path",
        max_depth: int = 4,
        samples: int = 1,
        seed: int = 0,
        self_hit_eps: float = 1.0e-4,
        background: Tensor | tuple[float, ...] | None = None,
        sample_batch: int = 0,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"Integrator.mode must be one of {_MODES}, got {mode!r}")
        if samples <= 0:
            raise ValueError("Integrator.samples must be positive")
        if max_depth < 0:
            raise ValueError("Integrator.max_depth must be non-negative")
        if mode == "direct":
            max_depth = 0
        elif mode == "one_bounce":
            max_depth = min(max_depth, 1)
        self.scene = scene
        self.environment = environment if environment is not None else ConstantEnvironment(0.0)
        self.mode = mode
        self.max_depth = int(max_depth)
        self.samples = int(samples)
        self.seed = int(seed)
        self.self_hit_eps = float(self_hit_eps)
        # 0 means "one batch"; otherwise walk the sample axis in blocks so peak
        # path tensors stay bounded independently of the total sample count.
        requested = int(sample_batch)
        self.sample_batch = int(samples) if requested <= 0 else max(1, min(requested, int(samples)))
        self.background = (
            None if background is None else torch.as_tensor(background, dtype=torch.float32).reshape(3)
        )

    # ------------------------------------------------------------------ render
    @torch.no_grad()
    def render(
        self,
        camera: Camera,
        *,
        pixel_mask: Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> RenderBuffers:
        """Render ``camera``; ``device`` overrides the camera's own device."""
        primary = camera.generate_rays(device=device)
        count = primary.num_rays
        ray_device = primary.device
        samples = self.samples
        height, width = camera.image_shape
        mask = camera.pixel_mask().to(device=ray_device)
        if pixel_mask is not None:
            mask = mask & pixel_mask.to(ray_device)
        mask_flat = mask.reshape(-1)

        radiance_sum = torch.zeros(count, 3, device=ray_device)
        albedo_sum = torch.zeros(count, 3, device=ray_device)
        normal_sum = torch.zeros(count, 3, device=ray_device)
        depth_sum = torch.zeros(count, device=ray_device)
        hit_any = torch.zeros(count, dtype=torch.bool, device=ray_device)
        primitive_id_image = torch.full((count,), -1, dtype=torch.long, device=ray_device)
        coverage: dict[str, int] = {}
        secondary_counts: dict[str, int] = {}
        thin_walled_paths = 0
        closed_paths = 0
        tir_paths = 0

        for begin in range(0, samples, self.sample_batch):
            block = min(self.sample_batch, samples - begin)
            total = count * block
            ray_ids = torch.arange(total, device=ray_device)
            pixel_index = torch.div(ray_ids, block, rounding_mode="floor")
            sample_index = (ray_ids % block) + begin
            origins = primary.origins.repeat_interleave(block, dim=0)
            directions = primary.directions.repeat_interleave(block, dim=0)

            radiance = torch.zeros(total, 3, device=ray_device)
            active = mask_flat.repeat_interleave(block).clone()
            throughput = torch.ones(total, 3, device=ray_device)
            primary_albedo = torch.zeros(total, 3, device=ray_device)
            primary_normal = torch.zeros(total, 3, device=ray_device)
            primary_depth = torch.zeros(total, device=ray_device)
            primary_hit = torch.zeros(total, dtype=torch.bool, device=ray_device)
            primary_primitive = torch.full((total,), -1, dtype=torch.long, device=ray_device)
            inside = torch.zeros(total, dtype=torch.bool, device=ray_device)
            rng = CounterRNG(seed=self.seed)

            for bounce in range(self.max_depth + 1):
                if not bool(active.any()):
                    break
                active_index = torch.nonzero(active, as_tuple=False).squeeze(1)
                rays = Rays(
                    origins[active_index],
                    directions[active_index],
                    t_min=self.self_hit_eps,
                    t_max=float("inf"),
                )
                hit = self.scene.intersect(rays)
                local_miss = ~hit.hit
                if bool(local_miss.any()):
                    miss_index = active_index[local_miss]
                    radiance[miss_index] += throughput[miss_index] * self.environment.radiance(
                        directions[miss_index]
                    )
                local_hit = hit.hit
                if bounce == 0:
                    recorded = active_index
                    primary_albedo[recorded] = hit.albedo
                    primary_normal[recorded] = hit.normal
                    primary_depth[recorded] = torch.where(
                        hit.hit, hit.t, torch.full_like(hit.t, float("inf"))
                    )
                    primary_hit[recorded] = hit.hit
                    primary_primitive[recorded] = hit.primitive_id
                if not bool(local_hit.any()):
                    active[active_index] = False
                    continue
                hit_index = active_index[local_hit]
                if bounce > 0:
                    for value, amount in zip(
                        *[tensor.tolist() for tensor in torch.unique(hit.primitive_id[local_hit], return_counts=True)]
                    ):
                        key = str(int(value))
                        secondary_counts[key] = secondary_counts.get(key, 0) + int(amount)
                hit = hit.index(local_hit)
                materials = self._material_batch(hit, hit_index.numel(), ray_device)
                radiance[hit_index] += throughput[hit_index] * materials.emission
                if bounce >= self.max_depth:
                    # A terminal path must still respect transparent materials.
                    # The old unconditional diffuse fallback made a fully
                    # transmissive NeuOct surface re-introduce its learned
                    # albedo whenever a ray reached the depth limit.
                    p_transmission, _, p_diffuse = self._branch_probabilities(materials)
                    terminal_normal = torch.where(
                        ((directions[hit_index] * hit.normal).sum(-1, keepdim=True) > 0.0),
                        -hit.normal,
                        hit.normal,
                    )
                    terminal_cos = (
                        directions[hit_index] * terminal_normal
                    ).sum(-1).abs().clamp(0.0, 1.0)
                    terminal_f0 = dielectric_f0(materials.ior)
                    terminal_transmission = (
                        p_transmission.unsqueeze(-1)
                        * torch.ones_like(materials.base_color)
                        * (1.0 - schlick_fresnel(terminal_cos, terminal_f0)).unsqueeze(-1)
                    )
                    terminal_diffuse = p_diffuse.unsqueeze(-1) * self._diffuse_albedo(
                        hit, materials
                    )
                    # A terminal diffuse path can use the deterministic ambient
                    # approximation, but a transparent path must preserve the
                    # incident direction.  Sampling only ``ambient`` here made
                    # glass collapse the HDRI to a nearly uniform white patch.
                    radiance[hit_index] += (
                        throughput[hit_index]
                        * terminal_diffuse
                        * self.environment.ambient.to(ray_device)
                    )
                    radiance[hit_index] += (
                        throughput[hit_index]
                        * terminal_transmission
                        * self.environment.radiance(directions[hit_index])
                    )
                    active[hit_index] = False
                    continue
                new_direction, weight, inside_update, stats = self._sample_branch(
                    hit,
                    materials,
                    directions[hit_index],
                    bounce=bounce,
                    pixel_index=pixel_index[hit_index],
                    sample_index=sample_index[hit_index],
                    rng=rng,
                    inside=inside[hit_index],
                )
                thin_walled_paths += stats["thin_walled"]
                closed_paths += stats["closed"]
                tir_paths += stats["tir"]
                throughput[hit_index] = throughput[hit_index] * weight
                directions[hit_index] = new_direction
                # A single self-hit offset along the *new* direction; TIR also
                # uses this path, so no branch can double-offset the origin.
                origins[hit_index] = hit.position + new_direction * self.self_hit_eps
                inside[hit_index] = inside_update
                active[active_index] = False
                active[hit_index] = True

            radiance_sum.index_add_(0, pixel_index, radiance)
            albedo_sum.index_add_(0, pixel_index, primary_albedo)
            normal_sum.index_add_(0, pixel_index, primary_normal)
            depth_sum.index_add_(0, pixel_index, primary_depth)
            hit_any |= self._scatter_any(pixel_index, primary_hit, count)
            update = primary_primitive >= 0
            if bool(update.any()):
                rows = pixel_index[update]
                primitive_id_image.index_copy_(0, rows, primary_primitive[update])
                values, amounts = torch.unique(primary_primitive[update], return_counts=True)
                for value, amount in zip(values.tolist(), amounts.tolist()):
                    key = str(int(value))
                    coverage[key] = coverage.get(key, 0) + int(amount)

        invalid = ~mask_flat
        radiance_sum[invalid] = 0.0
        if self.background is not None:
            radiance_sum[invalid] = self.background.to(ray_device)

        declared = {
            str(primitive.primitive_id): type(primitive).__name__ for primitive in self.scene
        }
        return RenderBuffers(
            color=(radiance_sum / float(samples)).reshape(height, width, 3),
            albedo=(albedo_sum / float(samples)).reshape(height, width, 3),
            normal=(normal_sum / float(samples)).reshape(height, width, 3),
            depth=(depth_sum / float(samples)).reshape(height, width),
            hit=hit_any.reshape(height, width),
            environment=self.environment,
            mode=self.mode,
            samples=samples,
            max_depth=self.max_depth,
            primitive_id=primitive_id_image.reshape(height, width),
            coverage=coverage,
            sample_batch=self.sample_batch,
            thin_walled_transmission_paths=thin_walled_paths,
            closed_transmission_paths=closed_paths,
            total_internal_reflection_paths=tir_paths,
            secondary_hits_by_primitive=secondary_counts,
            declared_primitives=declared,
        )

    @staticmethod
    def _scatter_any(pixel_index: Tensor, values: Tensor, count: int) -> Tensor:
        """OR-reduce per-ray booleans back onto their pixels."""
        result = torch.zeros(count, dtype=torch.bool, device=values.device)
        if bool(values.any()):
            result.index_put_(
                (pixel_index[values],), torch.ones_like(pixel_index[values], dtype=torch.bool)
            )
        return result

    # ------------------------------------------------------------------ shading
    @staticmethod
    def _material_batch(hit: RayHit, count: int, device) -> MaterialBatch:
        if hit.materials is not None:
            return hit.materials
        from .materials import MaterialResolver

        return MaterialBatch.from_material(
            hit.material or MaterialResolver().resolve_neuoct(), count, device=device
        )

    @staticmethod
    def _surface_color(hit: RayHit, materials: MaterialBatch) -> Tensor:
        """Combined learned/texture surface colour, tinted exactly once."""
        return (hit.albedo * materials.base_color).clamp(0.0, 1.0)

    @classmethod
    def _diffuse_albedo(cls, hit: RayHit, materials: MaterialBatch) -> Tensor:
        """Diffuse lobe: the combined surface colour, minus the metal fraction.

        NeuOct's ``hit.albedo`` is the *learned* diffuse colour, while the
        material ``base_color`` is a tint (white by default).  Shading with the
        tint alone would silently replace the prediction with white; meshes are
        made to behave identically by storing the texture (not the tint) in
        ``hit.albedo``.  Metals have no diffuse lobe, hence ``(1 - metallic)``.
        """
        return cls._surface_color(hit, materials) * (1.0 - materials.metallic).unsqueeze(-1)

    @staticmethod
    def _branch_probabilities(materials: MaterialBatch) -> tuple[Tensor, Tensor, Tensor]:
        """Sampling probabilities for ``(transmission, reflection, diffuse)``.

        Transmission takes ``transmission``; reflection takes the metal plus
        dielectric-specular fraction, capped so it cannot overlap transmission;
        diffuse takes the remainder.  The three values are simultaneously the
        branch distribution *and* the importance weights the one-sample
        estimator divides by, so they always sum to one.
        """
        p_transmission = materials.transmission.clamp(0.0, 1.0)
        dielectric_specular = materials.specular.clamp_min(0.0) * (1.0 - materials.metallic)
        p_reflection = (materials.metallic + dielectric_specular).clamp(0.0, 1.0)
        p_reflection = torch.minimum(p_reflection, 1.0 - p_transmission)
        p_diffuse = (1.0 - p_transmission - p_reflection).clamp_min(0.0)
        return p_transmission, p_reflection, p_diffuse

    def _sample_branch(
        self,
        hit: RayHit,
        materials: MaterialBatch,
        incident: Tensor,
        *,
        bounce: int,
        pixel_index: Tensor,
        sample_index: Tensor,
        rng: CounterRNG,
        inside: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, int]]:
        """Sample one lobe: ``(direction, weight, inside, stats)``.

        The multiplier is ``lobe / p(selected branch)`` -- the probability
        compensation that makes the one-sample estimator unbiased for the sum of
        the lobes -- clamped so a zero-probability branch can never produce an
        infinite weight.  Transmission is Snell refraction with total internal
        reflection whenever the hit exposes closed-surface state, and the
        documented zero-thickness sheet otherwise.
        """
        base = bounce * 16
        u_branch = rng.uniform_index(pixel_index, sample_index=sample_index, dimension=base + 0)
        u1 = rng.uniform_index(pixel_index, sample_index=sample_index, dimension=base + 1)
        u2 = rng.uniform_index(pixel_index, sample_index=sample_index, dimension=base + 2)
        p_transmission, p_reflection, p_diffuse = self._branch_probabilities(materials)

        transmission_region = u_branch < p_transmission
        reflection_region = (~transmission_region) & (u_branch < p_transmission + p_reflection)

        # Shade against a normal that faces the incoming ray, otherwise the
        # cosine/GGX samples can land in the far hemisphere.
        normal = torch.where(
            ((incident * hit.normal).sum(-1, keepdim=True) > 0.0), -hit.normal, hit.normal
        )
        cosine = cosine_hemisphere_sample(normal, u1, u2)
        reflected = ggx_reflection(incident, normal, materials.roughness, u1, u2)
        surface_color = self._surface_color(hit, materials)
        # Metallic F0 is the *combined* surface colour (learned NeuOct albedo or
        # mesh base colour/texture), so a mirror NeuOct keeps its identity.
        f0 = (
            materials.specular.clamp_min(0.0).unsqueeze(-1) * (1.0 - materials.metallic.unsqueeze(-1))
            + surface_color * materials.metallic.unsqueeze(-1)
        ).clamp(0.0, 1.0)
        cos_i = (incident * normal).sum(-1).abs().clamp(0.0, 1.0)
        fresnel = schlick_fresnel(cos_i.unsqueeze(-1), f0).clamp(0.0, 1.0)

        # --- transmission ------------------------------------------------------
        stats = {"thin_walled": 0, "closed": 0, "tir": 0}
        geometric = hit.geometric_normal
        front_face = hit.front_face
        closed_available = geometric is not None and front_face is not None
        if closed_available:
            snell = snell_refract(incident, geometric, front_face, ior=materials.ior)
            closed = (~materials.thin_walled).unsqueeze(-1)
            tir = snell.tir.unsqueeze(-1)
            # On total internal reflection the interface is a perfect mirror.
            snell_weight = torch.where(tir, torch.ones_like(snell.fresnel).unsqueeze(-1),
                                       (1.0 - snell.fresnel).unsqueeze(-1))
            transmitted = torch.where(closed, snell.direction, incident)
            transmittance = torch.where(
                closed, snell_weight, (1.0 - fresnel)
            )
            stats["closed"] = int((closed & transmission_region.unsqueeze(-1)).sum())
            stats["tir"] = int((tir & closed & transmission_region.unsqueeze(-1)).sum())
            stats["thin_walled"] = int(
                ((~closed) & transmission_region.unsqueeze(-1)).sum()
            )
            # Winding-derived entry/exit state: entering leaves the path inside.
            inside_update = torch.where(
                closed.squeeze(-1) & transmission_region, front_face, inside
            )
        else:
            # NeuOct has no watertight winding state, but it does provide a
            # learned normal. Use that normal for a two-sided Snell interface;
            # ``inside`` supplies the incident/transmitted medium ratio on the
            # next interface. This preserves angular bending while treating the
            # representation as transparent rather than diffuse.
            ior = materials.ior.clamp_min(1.0)
            eta = torch.where(inside, ior, ior.reciprocal())
            thin_snell = refract(incident, normal, eta)
            cos_interface = (incident * normal).sum(-1).abs().clamp(0.0, 1.0)
            thin_f0 = dielectric_f0(ior)
            thin_fresnel = schlick_fresnel(cos_interface, thin_f0).clamp(0.0, 1.0)
            transmitted = thin_snell.direction
            # ``refract`` returns the reflection direction on TIR; preserve all
            # energy there rather than attenuating it as if transmission won.
            transmittance = torch.where(
                thin_snell.tir, torch.ones_like(thin_fresnel), 1.0 - thin_fresnel
            ).unsqueeze(-1)
            stats["thin_walled"] = int(transmission_region.sum())
            inside_update = torch.where(
                transmission_region & ~thin_snell.tir, ~inside, inside
            )

        # Transmission is transparent transport, not a diffuse lobe: neither
        # learned NeuOct albedo nor a diffuse/base-colour texture attenuates the
        # transmitted radiance.  Material ``base_color`` remains available to
        # the diffuse and metallic lobes, but transparent paths are uncoloured.
        transmission_tint = torch.ones_like(materials.base_color)
        transmission_weight = p_transmission.unsqueeze(-1) * transmission_tint * transmittance
        new_direction = torch.where(
            reflection_region.unsqueeze(-1),
            reflected,
            torch.where(transmission_region.unsqueeze(-1), transmitted, cosine),
        )
        lobe_weight = torch.where(
            reflection_region.unsqueeze(-1),
            fresnel,
            torch.where(transmission_region.unsqueeze(-1), transmission_weight, self._diffuse_albedo(hit, materials)),
        )
        selected_probability = torch.where(
            reflection_region,
            p_reflection,
            torch.where(transmission_region, p_transmission, p_diffuse),
        )
        # Probability-compensated one-sample estimator: E[lobe / p] = lobe.
        safe_probability = selected_probability.clamp_min(_MIN_BRANCH_PROB)
        inverse_probability = torch.where(
            selected_probability > 0.0,
            safe_probability.reciprocal(),
            torch.zeros_like(selected_probability),
        )
        weight = lobe_weight * inverse_probability.unsqueeze(-1)
        if not bool(torch.isfinite(weight).all()):
            raise RuntimeError("non-finite branch weight; probability compensation is broken")
        return new_direction, weight, inside_update, stats


def render_scene(
    scene: Scene,
    camera: Camera,
    *,
    environment: Environment | None = None,
    mode: str = "path",
    max_depth: int = 4,
    samples: int = 1,
    seed: int = 0,
    pixel_mask: Tensor | None = None,
    device: torch.device | str | None = None,
    sample_batch: int = 0,
) -> RenderBuffers:
    """Convenience wrapper mirroring :class:`Integrator` construction."""
    integrator = Integrator(
        scene,
        environment=environment,
        mode=mode,
        max_depth=max_depth,
        samples=samples,
        seed=seed,
        sample_batch=sample_batch,
    )
    return integrator.render(camera, pixel_mask=pixel_mask, device=device)
