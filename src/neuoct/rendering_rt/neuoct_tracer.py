"""Direct NeuOct ray traversal with model inference.

This module never extracts a mesh.  It enumerates ray/voxel segment candidates
with the *existing* ``neuoct.geometry.ray_traversal.enumerate_ray_voxel_pairs``
helper, runs the checkpoint's renderer directly on each candidate segment, and
picks the nearest accepted hit per ray.  Candidate segments that lie behind the
ray origin are discarded (a ray starting inside a voxel produces one segment
whose entry plane is behind the origin), and ``t_min`` / ``t_max`` are honoured
so a bounce ray can skip the surface it just left.

The model colour is sRGB; :class:`NeuOctRayTracer` decodes it to linear for
shading while retaining ``albedo_srgb``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .color import srgb_to_linear
from .transforms import OctreeTransform, SimilarityTransform
from .types import Material, MaterialBatch, RayHit, Rays


@dataclass
class OctreeTraceResult:
    """Per-ray nearest hit expressed in **octree** coordinates."""

    hit: Tensor
    t: Tensor
    position: Tensor
    normal: Tensor
    albedo_srgb: Tensor
    hit_probability: Tensor
    material: Material | None = None
    candidate_count: int = 0
    accepted_count: int = 0

    @property
    def num_rays(self) -> int:
        return int(self.hit.shape[0])

    @classmethod
    def miss(cls, count: int, *, device=None, dtype=torch.float32) -> "OctreeTraceResult":
        device = torch.device("cpu") if device is None else device
        return cls(
            hit=torch.zeros(count, dtype=torch.bool, device=device),
            t=torch.full((count,), float("inf"), dtype=dtype, device=device),
            position=torch.zeros(count, 3, dtype=dtype, device=device),
            normal=torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
            .expand(count, 3)
            .clone(),
            albedo_srgb=torch.zeros(count, 3, dtype=dtype, device=device),
            hit_probability=torch.zeros(count, dtype=dtype, device=device),
        )


def _model_device(model) -> "torch.device | None":
    """Best-effort device of a checkpoint-backed model, or ``None`` if unknown.

    The tracer must run the model on the same device as the ray batch; reading
    the device here lets :func:`trace_octree` fail with an actionable message
    instead of a bare CUDA/CPU mismatch from deep inside the network.
    """
    field = getattr(model, "feature_field", None)
    for owner in (field, model):
        if owner is None:
            continue
        for name in ("features", "vertex_coordinates"):
            device = getattr(getattr(owner, name, None), "device", None)
            if device is not None:
                return torch.device(device)
    return None


def _require_same_device(value, expected: torch.device, *, name: str) -> None:
    device = getattr(value, "device", None)
    if device is None:
        return
    device = torch.device(device)
    if device != expected:
        raise ValueError(f"{name} lives on {device} but the rays live on {expected}")


def _predict_candidates(model, pins: Tensor, pouts: Tensor, voxel_indices: Tensor):
    """Run the checkpoint renderer on one candidate chunk.

    Mirrors ``SparseRenderer``: gather the corner features (and optional
    voxel-local features / base colours) from the model's own feature field and
    call ``renderer_model`` directly.  This keeps colour-skip checkpoints
    working without duplicating network code.

    ``pins``/``pouts`` must be the **voxel-local** entry/exit points, exactly the
    values the deployed evaluator feeds the network (the training loop targets
    ``pairs.positions - pairs.voxel_coordinates``).  Passing absolute octree
    coordinates here silently changes every prediction.
    """
    feature_field = model.feature_field
    renderer = model.renderer_model
    query_features = feature_field.gather(voxel_indices)
    voxel_features = None
    if int(getattr(renderer, "voxel_feature_dim", 0)):
        voxel_features = feature_field.gather_voxels(voxel_indices)
    base_colors = None
    if float(getattr(renderer, "color_skip_weight", 0.0)) > 0.0:
        gather_colors = getattr(feature_field, "gather_colors", None)
        if gather_colors is None:
            raise RuntimeError("Renderer requires voxel colours but the feature field lacks them")
        base_colors = gather_colors(voxel_indices)
    directions = F.normalize(pouts - pins, dim=-1, eps=1.0e-8)
    return renderer(
        pins,
        directions,
        query_features,
        pouts,
        voxel_features=voxel_features,
        base_colors=base_colors,
    ).final


def _first_per_ray(labels: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``(selected_candidate_indices, selected_ray_labels)``.

    Selection is the **first accepted candidate in enumeration order** per ray,
    which is exactly what the deployed ``SparseRenderer`` does: it processes
    candidates grouped by ray and drops a ray as soon as one candidate clears
    the visibility threshold.  ``enumerate_ray_voxel_pairs`` emits a ray's
    segments in traversal order, so this is also the front-to-back choice for
    every ray whose network predictions are ordered -- but ordering by the
    *predicted* ``t`` instead would silently disagree with the deployed
    evaluator whenever the network predicts a nearer position in a later
    segment, which is what the strict parity test catches.
    """
    order = torch.argsort(labels, stable=True)
    sorted_labels = labels[order]
    first = torch.cat(
        [
            torch.ones(1, dtype=torch.bool, device=labels.device),
            sorted_labels[1:] != sorted_labels[:-1],
        ]
    )
    selected = order[first]
    return selected, labels[selected]


@torch.no_grad()
def trace_octree(
    origins: Tensor,
    directions: Tensor,
    model,
    octree,
    *,
    depth: int,
    hit_threshold: float = 0.5,
    ray_chunk_size: int = 16384,
    candidate_chunk_size: int = 65536,
    t_min: Tensor | float = 0.0,
    t_max: Tensor | float = float("inf"),
    self_hit_eps: float = 1.0e-4,
    material: Material | None = None,
) -> OctreeTraceResult:
    """Trace unit rays given in octree coordinates and return nearest hits.

    ``origins``/``directions`` are octree-space ``(N, 3)`` tensors.  Directions
    are normalised.  ``t_min``/``t_max`` are octree-space parameters (scalar or
    per-ray).
    """
    origins = torch.as_tensor(origins, dtype=torch.float32)
    directions = torch.as_tensor(directions, dtype=torch.float32)
    if origins.ndim != 2 or origins.shape[1] != 3:
        raise ValueError("origins must have shape (N, 3)")
    if directions.shape != origins.shape:
        raise ValueError("directions must match origins shape")
    if not 0.0 <= hit_threshold < 1.0:
        raise ValueError("hit_threshold must be in [0, 1)")
    norms = directions.norm(dim=-1, keepdim=True)
    # Skip the no-op renormalisation of already-unit directions: it perturbs
    # every component by ~1e-7, which changes the octree plane parameters (and
    # therefore ``pins``/``pouts``) enough to move an fp16 prediction.  The
    # deployed ``SparseRenderer`` feeds its ray directions straight through.
    if not bool(((norms - 1.0).abs() <= 1.0e-6).all()):
        directions = F.normalize(directions, dim=-1, eps=1.0e-12)
    ray_count = origins.shape[0]
    if ray_count == 0:
        return OctreeTraceResult.miss(0, device=origins.device, dtype=origins.dtype)
    _require_same_device(octree, origins.device, name="the octree")
    _require_same_device(_model_device(model), origins.device, name="the model")

    # Lazy import: keeps the tracer importable without the OCNN/triton stack.
    from neuoct.geometry.ray_traversal import (
        enumerate_ray_voxel_pairs,
        grouped_interleave_indices,
    )

    scale = 1 << int(depth)
    pairs = enumerate_ray_voxel_pairs(
        origins,
        directions,
        octree,
        scale=scale,
        depth=int(depth),
        batch_id=0,
        ray_label_offset=0,
        ray_chunk_size=int(ray_chunk_size),
    )
    if pairs.ray_labels.numel() == 0:
        return OctreeTraceResult.miss(ray_count, device=origins.device, dtype=origins.dtype)

    labels = pairs.ray_labels.long()
    voxel_coordinates = pairs.voxel_coordinates.to(origins.dtype)
    # ``pairs.pins``/``pouts`` are *voxel-local* and are what the deployed
    # ``SparseRenderer`` feeds the checkpoint; the absolute entry/exit points are
    # local + voxel coordinate.  Keep both: mixing them up offsets every
    # predicted position by a voxel coordinate.
    local_pins = pairs.pins.to(origins.dtype)
    local_pouts = pairs.pouts.to(origins.dtype)
    voxel_indices = pairs.voxel_indices.long()

    t_min_tensor = torch.as_tensor(t_min, dtype=origins.dtype, device=origins.device).expand(ray_count)
    t_max_tensor = torch.as_tensor(t_max, dtype=origins.dtype, device=origins.device).expand(ray_count)
    effective_min = torch.maximum(t_min_tensor, torch.full_like(t_min_tensor, float(self_hit_eps)))
    lower = effective_min[labels]
    upper = t_max_tensor[labels]

    # Exact parity with the deployed evaluator.  ``SparseRenderer`` runs the
    # network on the *whole* candidate set, ordered by
    # ``grouped_interleave_indices`` (every ray's first segment, then every ray's
    # second, ...), in chunks of ``max_batch_size``.  A fused fp16 network is not
    # obliged to return bit-identical rows for a different batch composition, so
    # pruning or re-ordering candidates here silently perturbs predictions --
    # which is exactly what the strict primary-ray parity test measures.  Order
    # and batch the candidates the same way instead.
    if labels.numel() > 1 and not bool((labels[1:] >= labels[:-1]).all()):
        order = torch.argsort(labels, stable=True)
        labels = labels[order]
        local_pins = local_pins[order]
        local_pouts = local_pouts[order]
        voxel_coordinates = voxel_coordinates[order]
        voxel_indices = voxel_indices[order]
        lower = lower[order]
        upper = upper[order]
    order = grouped_interleave_indices(labels)
    labels = labels[order]
    local_pins = local_pins[order]
    local_pouts = local_pouts[order]
    voxel_coordinates = voxel_coordinates[order]
    voxel_indices = voxel_indices[order]
    lower = lower[order]
    upper = upper[order]

    prediction_position = torch.empty(labels.numel(), 3, dtype=origins.dtype, device=origins.device)
    prediction_normal = torch.empty_like(prediction_position)
    prediction_color = torch.empty_like(prediction_position)
    prediction_prob = torch.empty(labels.numel(), dtype=origins.dtype, device=origins.device)
    for begin in range(0, labels.numel(), int(candidate_chunk_size)):
        end = min(begin + int(candidate_chunk_size), labels.numel())
        stage = _predict_candidates(
            model,
            local_pins[begin:end],
            local_pouts[begin:end],
            voxel_indices[begin:end],
        )
        prediction_position[begin:end] = stage.position.to(origins.dtype)
        prediction_normal[begin:end] = stage.normal.to(origins.dtype)
        prediction_color[begin:end] = stage.color.to(origins.dtype)
        prediction_prob[begin:end] = stage.hit_probability.to(origins.dtype)

    global_position = prediction_position + voxel_coordinates
    t_hit = ((global_position - origins[labels]) * directions[labels]).sum(-1)
    # Acceptance mirrors the deployed ``SparseRenderer``: a candidate is a hit
    # when the network's visibility probability clears the threshold and the
    # resulting position lies inside the ray's own [t_min, t_max] window.
    accepted = prediction_prob > hit_threshold
    accepted &= t_hit >= lower
    accepted &= t_hit <= upper
    candidate_count = int(labels.numel())
    if not bool(accepted.any()):
        return OctreeTraceResult.miss(
            ray_count, device=origins.device, dtype=origins.dtype
        )
    labels = labels[accepted]
    t_hit = t_hit[accepted]
    global_position = global_position[accepted]
    prediction_normal = prediction_normal[accepted]
    prediction_color = prediction_color[accepted]
    prediction_prob = prediction_prob[accepted]

    selected, selected_labels = _first_per_ray(labels)
    hit = torch.zeros(ray_count, dtype=torch.bool, device=origins.device)
    t_result = torch.full((ray_count,), float("inf"), dtype=origins.dtype, device=origins.device)
    position_result = torch.zeros(ray_count, 3, dtype=origins.dtype, device=origins.device)
    normal_result = torch.tensor([0.0, 0.0, 1.0], dtype=origins.dtype, device=origins.device).expand(
        ray_count, 3
    ).clone()
    color_result = torch.zeros(ray_count, 3, dtype=origins.dtype, device=origins.device)
    prob_result = torch.zeros(ray_count, dtype=origins.dtype, device=origins.device)
    hit[selected_labels] = True
    t_result[selected_labels] = t_hit[selected]
    position_result[selected_labels] = global_position[selected]
    normal_result[selected_labels] = prediction_normal[selected]
    color_result[selected_labels] = prediction_color[selected]
    prob_result[selected_labels] = prediction_prob[selected]
    return OctreeTraceResult(
        hit=hit,
        t=t_result,
        position=position_result,
        normal=normal_result,
        albedo_srgb=color_result,
        hit_probability=prob_result,
        material=material,
        candidate_count=candidate_count,
        accepted_count=int(selected_labels.numel()),
    )


class NeuOctRayTracer:
    """Scene-space wrapper around :func:`trace_octree`.

    It applies the external similarity transform (object <-> scene) and the
    exact octree mapping, and returns :class:`RayHit` records in scene space.
    """

    def __init__(
        self,
        model,
        octree,
        *,
        depth: int,
        point_scale: float = 0.99,
        ray_scale: float = 0.5,
        object_to_scene: SimilarityTransform | None = None,
        hit_threshold: float = 0.5,
        ray_chunk_size: int = 16384,
        candidate_chunk_size: int = 65536,
        self_hit_eps: float = 1.0e-4,
        material: Material | None = None,
        primitive_id: int = 0,
    ) -> None:
        self.model = model
        self.octree = octree
        self.octree_transform = OctreeTransform(depth, point_scale, ray_scale)
        self.object_to_scene = object_to_scene or SimilarityTransform.identity()
        self.hit_threshold = float(hit_threshold)
        self.ray_chunk_size = int(ray_chunk_size)
        self.candidate_chunk_size = int(candidate_chunk_size)
        self.self_hit_eps = float(self_hit_eps)
        self.material = material
        self.primitive_id = int(primitive_id)

    @property
    def linear(self) -> float:
        return self.octree_transform.linear

    def self_hit_epsilon_octree(self, similarity: SimilarityTransform | None = None) -> float:
        """Scene-space self-hit epsilon expressed in octree units.

        ``self_hit_eps`` is a *scene-space* length, so the exact chain is
        ``scene_t / similarity.scale * octree.linear``: the similarity scale maps
        scene lengths to object lengths, and ``linear`` maps object lengths to
        octree units.  Dividing only by ``linear`` (or only by ``scale``) makes
        the rejection radius depend on the placement scale.
        """
        transform = self.object_to_scene if similarity is None else similarity
        return self.self_hit_eps / float(transform.scale) * self.octree_transform.linear

    @torch.no_grad()
    def trace(self, rays: Rays) -> RayHit:
        if rays.num_rays == 0:
            return RayHit.miss(0, device=rays.device, dtype=rays.dtype)
        # Move the (small) similarity transform once onto the ray device so the
        # whole public path works on CUDA without a CPU/CUDA mismatch.
        similarity = self.object_to_scene.to(rays.device)
        # An unplaced object must reach the network with bit-identical rays.
        # Round-tripping through ``R = I`` and renormalising perturbs every
        # component in the last float32 bits, and the fp16 checkpoint can turn
        # that into a ~1e-3 difference on knife-edge pixels; skipping the no-op
        # transform is both faster and what makes strict parity reproducible.
        identity = similarity.is_identity()
        if identity:
            object_origins = rays.origins
            object_directions = rays.directions
        else:
            object_origins = similarity.inverse_points(rays.origins)
            object_directions = similarity.inverse_directions(rays.directions)
        octree_origins = self.octree_transform.object_to_octree(object_origins)
        # Ray parameters: t_octree = linear * t_object = linear * t_scene / scale.
        t_min_object = rays.t_min / similarity.scale
        t_max_object = rays.t_max / similarity.scale
        t_min_octree = self.octree_transform.object_to_octree_t(t_min_object)
        t_max_octree = self.octree_transform.object_to_octree_t(t_max_object)
        # ``self_hit_eps`` is a scene-space length, so it must also be divided by
        # the similarity scale before the octree linear factor is applied;
        # dropping the division breaks scale-aware self-hit rejection.
        self_hit_eps_octree = self.self_hit_epsilon_octree(similarity)
        result = trace_octree(
            octree_origins,
            object_directions,
            self.model,
            self.octree,
            depth=self.octree_transform.depth,
            hit_threshold=self.hit_threshold,
            ray_chunk_size=self.ray_chunk_size,
            candidate_chunk_size=self.candidate_chunk_size,
            t_min=t_min_octree,
            t_max=t_max_octree,
            self_hit_eps=self_hit_eps_octree,
            material=self.material,
        )
        object_position = self.octree_transform.octree_to_object(result.position)
        if identity:
            scene_position = object_position
            scene_normal = result.normal
            scene_t = result.t / self.linear
        else:
            scene_position = similarity.transform_points(object_position)
            scene_normal = similarity.transform_normals(result.normal)
            scene_t = result.t * similarity.scale / self.linear
        linear_albedo = srgb_to_linear(result.albedo_srgb)
        primitive_id = torch.full_like(result.hit.long(), self.primitive_id)
        materials = (
            None
            if self.material is None
            else MaterialBatch.from_material(self.material, result.num_rays, device=result.hit.device)
        )
        return RayHit(
            hit=result.hit,
            t=scene_t,
            position=scene_position,
            normal=scene_normal,
            barycentric=torch.tensor([1.0, 0.0, 0.0], device=result.hit.device, dtype=torch.float32)
            .expand(result.num_rays, 3)
            .clone(),
            triangle_index=torch.full((result.num_rays,), -1, dtype=torch.long, device=result.hit.device),
            primitive_id=primitive_id,
            albedo=linear_albedo,
            albedo_srgb=result.albedo_srgb,
            hit_probability=result.hit_probability,
            uv=None,
            material=result.material,
            materials=materials,
        )
