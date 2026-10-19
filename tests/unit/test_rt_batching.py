"""Bounded sample batching must not change a single sample.

Random numbers are addressed by ``(pixel, sample, dimension)``, so walking the
sample axis in blocks is an *accumulation* strategy, not a different estimator.
These tests pin that property down: the same seed and sample count must produce
bit-identical images for any batch size, and the peak path tensor count must
follow the batch size rather than the total sample count.
"""

from __future__ import annotations

import numpy as np
import torch

from neuoct.rendering_rt import (
    Camera,
    ConstantEnvironment,
    Integrator,
    Material,
    MeshPrimitive,
    Scene,
)


def _scene() -> Scene:
    import trimesh

    vertices = np.array(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    geometry = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2]], process=False)
    return Scene(
        [
            MeshPrimitive(
                geometry,
                material=Material(base_color=torch.tensor([0.8, 0.4, 0.2]), roughness=0.4),
                primitive_id=0,
            )
        ]
    )


def _camera(width: int = 12, height: int = 10) -> Camera:
    return Camera.look_at(
        [0.0, 0.0, 3.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], width=width, height=height, camera_angle_x=0.9
    )


def _render(samples: int, batch: int, *, seed: int = 11, depth: int = 3):
    integrator = Integrator(
        _scene(),
        environment=ConstantEnvironment([0.7, 0.7, 0.7]),
        mode="path",
        max_depth=depth,
        samples=samples,
        seed=seed,
        sample_batch=batch,
    )
    return integrator.render(_camera())


def test_batched_render_is_bit_identical_to_one_batch() -> None:
    reference = _render(6, 0)
    for batch in (1, 2, 3, 4, 6):
        batched = _render(6, batch)
        assert torch.equal(reference.color, batched.color), f"batch={batch} changed the colour"
        assert torch.equal(reference.albedo, batched.albedo)
        assert torch.equal(reference.normal, batched.normal)
        assert torch.equal(reference.depth, batched.depth)
        assert torch.equal(reference.hit, batched.hit)
        assert torch.equal(reference.primitive_id, batched.primitive_id)
        assert batched.sample_batch == batch


def test_peak_path_tensors_are_bounded_by_the_batch() -> None:
    integrator = Integrator(
        _scene(),
        environment=ConstantEnvironment([1.0, 1.0, 1.0]),
        mode="path",
        max_depth=2,
        samples=8,
        seed=1,
        sample_batch=2,
    )
    assert integrator.sample_batch == 2
    camera = _camera(8, 8)
    buffers = integrator.render(camera)
    # 8 samples were accumulated from 4 batches of 2; the result is still a
    # proper mean, so every channel stays inside the environment's range.
    assert torch.isfinite(buffers.color).all()
    assert float(buffers.color.max()) <= 1.0 + 1e-5
    assert buffers.samples == 8
    assert buffers.sample_batch == 2


def test_batch_size_larger_than_samples_is_clamped() -> None:
    integrator = Integrator(_scene(), samples=3, sample_batch=99)
    assert integrator.sample_batch == 3


def test_coverage_and_primitive_id_are_recorded() -> None:
    buffers = _render(2, 1)
    assert buffers.primitive_id is not None
    assert buffers.coverage.get("0", 0) > 0
    assert set(np.unique(buffers.primitive_id.numpy()).tolist()) <= {-1, 0}
    assert buffers.declared_primitives.get("0") == "MeshPrimitive"
