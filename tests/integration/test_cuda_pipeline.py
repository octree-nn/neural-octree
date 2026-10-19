from pathlib import Path

import pytest
import torch

from neuoct.config.schema import DataConfig
from neuoct.data.camera import CameraIntrinsics, generate_camera_directions
from neuoct.data.manifest import ManifestEntry
from neuoct.data.preprocess import DeviceBatchPreprocessor
from neuoct.data.sample import RawSample
from neuoct.geometry.octree_adapter import build_input_features
from neuoct.losses import IntersectionObjective, IntersectionTarget
from neuoct.models import NeuralIntersectionModel
from neuoct.rendering import SparseRenderer

pytestmark = pytest.mark.gpu


def _sample(batch_id: int, size: int = 16) -> RawSample:
    intrinsics = CameraIntrinsics.from_horizontal_fov(torch.pi / 3, size, size)
    transform = torch.eye(4)
    transform[2, 3] = -1.0
    directions = generate_camera_directions(intrinsics, device="cpu")
    point_positions = (directions * 1.0) + transform[:3, 3]
    point_positions = point_positions.reshape(-1, 3).clamp(-0.95, 0.95)
    color = torch.full((size, size, 3), 128 + batch_id, dtype=torch.uint8)
    normal = torch.zeros((size, size, 3), dtype=torch.uint8)
    normal[..., 2] = 255
    return RawSample(
        entry=ManifestEntry("category", f"object-{batch_id}", 0),
        point_positions=point_positions,
        point_colors=color.reshape(-1, 3).float() / 255.0,
        intrinsics=intrinsics,
        camera_to_world=transform,
        color_image=color,
        depth_image=torch.ones(size, size),
        normal_image=normal,
        valid_depth=torch.ones(size, size, dtype=torch.bool),
        depth_min=0.0,
        depth_max=2.0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_batch_two_forward_and_backward_are_finite():
    config = DataConfig(
        image_root=Path("."),
        pointcloud_root=Path("."),
        manifest=Path("manifest.txt"),
        image_width=16,
        image_height=16,
        take=2,
    )
    samples = [_sample(0), _sample(1)]
    preprocessor = DeviceBatchPreprocessor(config, "cuda")
    prepared = preprocessor(samples)
    assert prepared.batch_size == 2
    assert prepared.ray_labels.max() >= 16 * 16
    input_features = build_input_features(prepared.octree, "LPF")
    model = NeuralIntersectionModel(9, 7).cuda().train()
    with torch.autocast("cuda", enabled=True):
        prediction = model(
            input_features,
            prepared.octree,
            prepared.octree.depth,
            prepared.pins,
            prepared.directions,
            prepared.voxel_indices,
        )
        target = IntersectionTarget(
            prepared.local_positions, prepared.normals, prepared.colors, prepared.hits
        )
        result = IntersectionObjective().cuda()(
            prediction,
            target,
            prepared.ray_labels,
            batch_size=2,
            height=16,
            width=16,
        )
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    model.eval()
    with torch.no_grad():
        _, field = model.extract_scene_features(
            input_features, prepared.octree, prepared.octree.depth
        )
        rendered = SparseRenderer(
            model.renderer_model, field, prepared.octree, depth=7
        ).render_frame(samples[0].camera_to_world, samples[0].intrinsics, batch_id=0)
    assert rendered.color.shape == (16, 16, 3)
    assert torch.isfinite(rendered.color).all()
    preprocessor._build_octree(samples[0])
    assert preprocessor.cache_hits >= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_query_depth_six_forward_is_finite():
    config = DataConfig(
        image_root=Path("."),
        pointcloud_root=Path("."),
        manifest=Path("manifest.txt"),
        query_depth=6,
        image_width=16,
        image_height=16,
        take=1,
    )
    prepared = DeviceBatchPreprocessor(config, "cuda")([_sample(0)])
    features = build_input_features(prepared.octree, "LPF")
    model = NeuralIntersectionModel(9, 6).cuda().eval()
    with torch.no_grad():
        prediction = model(
            features,
            prepared.octree,
            prepared.octree.depth,
            prepared.pins,
            prepared.directions,
            prepared.voxel_indices,
        )
    assert len(prediction.stages) == 3
    assert torch.isfinite(prediction.final.position).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_material_fp32_stays_finite_for_large_features_under_amp():
    from neuoct.models.intersection_renderer import SplitVoxelRendererModel

    model = SplitVoxelRendererModel(
        f_dim=4,
        hidden_dim=16,
        n_iters=2,
        material_fp32=True,
    ).cuda().train()
    with torch.no_grad():
        model.geometry_embedding.weight.zero_()
        model.material_embedding.weight.fill_(1.0e4)
    count = 32
    pin = torch.full((count, 3), 0.5, device="cuda")
    direction = torch.nn.functional.normalize(torch.randn(count, 3, device="cuda"), dim=1)
    vertex_features = torch.full((count, 2, 2, 2, 4), 100.0, device="cuda")
    with torch.autocast("cuda", enabled=True):
        prediction = model(pin, direction, vertex_features)
        loss = prediction.final.color.mean()
    loss.backward()
    assert torch.isfinite(prediction.final.color).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def _compiled_fixture():
    from neuoct.geometry.local_octree import LocalSparseOctree
    from neuoct.geometry.vertex_index import compute_vertex_index
    from neuoct.models.local_scene import LocalSceneModel

    octree = LocalSparseOctree(
        torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.int32),
        2,
        "cuda",
    )
    index = compute_vertex_index(octree, 2, include_batch=True)

    def build():
        torch.manual_seed(7)
        return LocalSceneModel(
            index,
            feature_dim=4,
            hidden_dim=16,
            n_iters=2,
            geometry_only=True,
            sampling_mode="uniform",
        ).cuda()

    torch.manual_seed(3)
    pins = torch.rand(64, 3, device="cuda")
    pouts = torch.rand(64, 3, device="cuda")
    directions = torch.nn.functional.normalize(pouts - pins, dim=1)
    voxel_indices = torch.zeros(64, dtype=torch.long, device="cuda")
    return build, pins, directions, voxel_indices, pouts


def _step(model, pins, directions, voxel_indices, pouts):
    model.zero_grad(set_to_none=True)
    prediction = model(pins, directions, voxel_indices, pouts)
    loss = (
        prediction.final.position.square().mean()
        + prediction.final.normal.square().mean()
        + prediction.final.hit_probability.mean()
    )
    loss.backward()
    grads = {
        # ``torch.compile`` wraps the renderer in OptimizedModule, so normalize
        # the parameter names before comparing against the eager model.
        name.replace("renderer_model._orig_mod.", "renderer_model."): parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return prediction.final, loss.detach(), grads


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_compiled_renderer_matches_eager_and_roundtrips(tmp_path):
    from neuoct.engine.local_checkpoint import (
        load_local_checkpoint,
        save_local_checkpoint,
    )

    build, pins, directions, voxel_indices, pouts = _compiled_fixture()
    eager = build()
    eager_final, eager_loss, eager_grads = _step(
        eager, pins, directions, voxel_indices, pouts
    )

    compiled = build()
    compiled.cudagraph_step_begin = False
    compiled.renderer_model = torch.compile(
        compiled.renderer_model, mode="default", dynamic=True
    )
    # Warm the compiled path up before the step that is compared.
    for _ in range(3):
        _step(compiled, pins, directions, voxel_indices, pouts)
    compiled_final, compiled_loss, compiled_grads = _step(
        compiled, pins, directions, voxel_indices, pouts
    )

    torch.testing.assert_close(compiled_loss, eager_loss, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(
        compiled_final.position, eager_final.position, rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(
        compiled_final.normal, eager_final.normal, rtol=1e-4, atol=1e-5
    )
    assert set(compiled_grads) == set(eager_grads)
    for name, value in eager_grads.items():
        torch.testing.assert_close(
            compiled_grads[name], value, rtol=1e-3, atol=1e-5
        )

    # The compiled module is wrapped in OptimizedModule; its checkpoint must load
    # back into the plain renderer and reproduce the same prediction.
    checkpoint = save_local_checkpoint(
        tmp_path / "compiled.pt",
        compiled,
        object_key="category/object",
        query_depth=2,
        epoch=1,
    )
    loaded, _ = load_local_checkpoint(
        checkpoint, device="cuda", expected_object="category/object"
    )
    restored = loaded(pins, directions, voxel_indices, pouts).final
    torch.testing.assert_close(
        restored.position, compiled_final.position, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        restored.normal, compiled_final.normal, rtol=1e-5, atol=1e-6
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cudagraph_compiled_renderer_survives_repeated_invocations():
    """``reduce-overhead`` needs an explicit step boundary between invocations."""
    build, pins, directions, voxel_indices, pouts = _compiled_fixture()
    model = build()
    model.cudagraph_step_begin = True
    model.renderer_model = torch.compile(
        model.renderer_model, mode="reduce-overhead", dynamic=True
    )
    for _ in range(4):
        final, _, _ = _step(model, pins, directions, voxel_indices, pouts)
    assert torch.isfinite(final.position).all()
    assert torch.isfinite(final.normal).all()
