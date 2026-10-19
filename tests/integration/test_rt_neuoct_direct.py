"""Integration: direct NeuOct primary rays vs the existing ``SparseRenderer``.

This test loads the real packed checkpoint and renders the *same* primary rays
through two code paths:

* ``SparseRenderer.render_frame`` -- the deployed evaluator path, and
* ``NeuOctPrimitive.intersect`` -- the new offline tracer, driven by the actual
  ``LocalSceneModel``/``LocalSparseOctree`` (no mesh extraction).

It is skipped unless a CUDA device and the checkpoint are present. Set
NEUOCT_RT_CHECKPOINT and NEUOCT_RT_OBJECT, then run::

    python -m pytest tests/integration/test_rt_neuoct_direct.py -q

The numerical assertions are deliberately tight (exact hit mask, 1e-4 on
positions/normals/colour); the two paths must agree, not merely correlate.
"""

from __future__ import annotations

from pathlib import Path
import os

import pytest
import torch

CHECKPOINT = Path(
    os.environ.get("NEUOCT_RT_CHECKPOINT", "checkpoints/example_object.pt")
)
OBJECT_KEY = os.environ.get("NEUOCT_RT_OBJECT", "objaverse/01_5f9711ec59f5411fafaa8515289a6643")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA device is required"),
    pytest.mark.skipif(not CHECKPOINT.exists(), reason=f"checkpoint {CHECKPOINT} is unavailable"),
]


def test_direct_primary_rays_match_sparse_renderer() -> None:
    from neuoct.data.camera import CameraIntrinsics, generate_rays
    from neuoct.engine.local_checkpoint import load_local_inference_checkpoint
    from neuoct.rendering.renderer import SparseRenderer
    from neuoct.rendering.views import generate_orbit_views
    from neuoct.rendering_rt import (
        NeuOctPrimitive,
        OctreeTransform,
        Rays,
        SimilarityTransform,
    )

    device = torch.device("cuda")
    model, octree, payload = load_local_inference_checkpoint(
        CHECKPOINT, device=device, expected_object=OBJECT_KEY
    )
    model.eval()
    depth = int(payload["query_depth"])
    point_scale, ray_scale = 0.99, 0.5

    renderer = SparseRenderer(
        model.renderer_model,
        model.feature_field,
        octree,
        depth=depth,
        point_scale=point_scale,
        ray_scale=ray_scale,
        camera_convention="neuoct",
        hit_threshold=0.5,
    )
    primitive = NeuOctPrimitive(
        model,
        octree,
        depth=depth,
        point_scale=point_scale,
        ray_scale=ray_scale,
        object_to_scene=SimilarityTransform.identity(),
        hit_threshold=0.5,
        # The parity claim is only meaningful if the network is fed the same
        # candidate batches.  ``SparseRenderer`` uses ``max_batch_size`` (its
        # default is 262144), so the offline tracer is asked for the same chunk
        # size here; the package default (65536) is left untouched.
        candidate_chunk_size=262144,
    )

    width = height = 64
    intrinsics = CameraIntrinsics.from_horizontal_fov(0.7, width, height)
    camera_to_world = generate_orbit_views(
        1, radius=2.5, camera_convention="neuoct", up_axis="y", device=device
    )[0]

    rendered = renderer.render_frame(camera_to_world, intrinsics)
    origins, directions = generate_rays(intrinsics, camera_to_world, convention="neuoct")
    rays = Rays(origins, directions, t_min=1.0e-4, t_max=1.0e9)
    traced = primitive.intersect(rays)

    rendered_hit = rendered.hit.reshape(height, width).reshape(-1)
    assert torch.equal(traced.hit, rendered_hit)

    selected = traced.hit
    assert bool(selected.any())
    octree_transform = OctreeTransform(depth, point_scale, ray_scale)
    traced_position_octree = octree_transform.object_to_octree(traced.position[selected])
    rendered_position = rendered.position.reshape(-1, 3)[selected]
    assert torch.allclose(traced_position_octree, rendered_position, atol=1.0e-4)
    assert torch.allclose(traced.normal[selected], rendered.normal.reshape(-1, 3)[selected], atol=1.0e-4)
    assert torch.allclose(
        traced.albedo_srgb[selected], rendered.color.reshape(-1, 3)[selected], atol=1.0e-4
    )
    # Scene-space metric distance must correspond to the rendered octree point
    # through the exact linear mapping (t_octree = linear * t_scene).
    origin_octree = octree_transform.object_to_octree(origins[selected])
    octree_t = ((rendered_position - origin_octree) * directions[selected]).sum(-1)
    expected_scene_t = octree_t / octree_transform.linear
    assert torch.allclose(traced.t[selected], expected_scene_t, atol=1.0e-4)
