import json

import torch

from neuoct.config.schema import DataConfig
from neuoct.data.camera import CameraIntrinsics
from neuoct.data.dataset import ShapeNetMultiViewDataset
from neuoct.engine.local_checkpoint import (
    _normalized_model_state_dict,
    load_local_checkpoint,
    load_local_inference_checkpoint,
    save_local_checkpoint,
    save_local_inference_checkpoint,
)
from neuoct.engine.local_optimize import (
    _cache_rank_order,
    _pair_balanced_view_order,
    _rank_view_order,
)
from neuoct.geometry.local_octree import LocalSparseOctree
from neuoct.geometry.vertex_index import VertexIndex, compute_vertex_index
from neuoct.models.local_scene import LocalSceneModel
from neuoct.rendering import SparseRenderer


def _vertex_index() -> VertexIndex:
    coordinates = torch.tensor(
        [[0, x, y, z] for x in range(2) for y in range(2) for z in range(2)],
        dtype=torch.int32,
    )
    return VertexIndex(coordinates, torch.arange(8).reshape(1, 8))


def test_local_scene_forward_backward_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(3)
    model = LocalSceneModel(_vertex_index(), feature_dim=4, hidden_dim=16, n_iters=2)
    pins = torch.rand(5, 3)
    directions = torch.nn.functional.normalize(torch.randn(5, 3), dim=1)
    prediction = model(pins, directions, torch.zeros(5, dtype=torch.long))
    loss = sum(stage.position.square().mean() for stage in prediction.stages)
    loss.backward()
    assert model.feature_field.features.grad is not None
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = save_local_checkpoint(
        tmp_path / "scene.pt",
        model,
        object_key="category/object",
        query_depth=7,
        epoch=2,
        optimizer=optimizer,
    )
    loaded, payload = load_local_checkpoint(checkpoint, expected_object="category/object")
    assert payload["epoch"] == 2
    assert loaded.feature_field.num_vertices == 8
    for expected, actual in zip(model.state_dict().values(), loaded.state_dict().values()):
        torch.testing.assert_close(expected, actual)


def test_geometry_only_scene_has_no_material_parameters_and_roundtrips(tmp_path):
    model = LocalSceneModel(
        _vertex_index(),
        feature_dim=4,
        hidden_dim=16,
        n_iters=2,
        geometry_only=True,
    )
    parameter_names = {name for name, _ in model.named_parameters()}
    assert not any("material" in name for name in parameter_names)
    assert not any("init_color" in name for name in parameter_names)
    pins = torch.rand(5, 3)
    directions = torch.nn.functional.normalize(torch.randn(5, 3), dim=1)
    prediction = model(pins, directions, torch.zeros(5, dtype=torch.long))
    assert torch.count_nonzero(prediction.final.color) == 0
    checkpoint = save_local_checkpoint(
        tmp_path / "geometry.pt",
        model,
        object_key="category/object",
        query_depth=7,
        epoch=1,
    )
    loaded, payload = load_local_checkpoint(checkpoint, expected_object="category/object")
    assert payload["model_config"]["geometry_only"] is True
    assert loaded.renderer_model.geometry_only is True
    assert not any("material" in name for name, _ in loaded.named_parameters())


def test_object_dataset_enumerates_all_transform_frames(tmp_path):
    image_root = tmp_path / "images"
    point_root = tmp_path / "points"
    object_dir = image_root / "category" / "object"
    object_dir.mkdir(parents=True)
    point_root.mkdir()
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("category/object/0\n", encoding="utf-8")
    (object_dir / "transforms.json").write_text(
        json.dumps({"frames": [{"file_path": "0.png"}, {"file_path": "1.png"}]}),
        encoding="utf-8",
    )
    config = DataConfig(image_root=image_root, pointcloud_root=point_root, manifest=manifest)
    dataset = ShapeNetMultiViewDataset.for_object(config, "category/object")
    assert [entry.frame_id for entry in dataset.entries] == [0, 1]


def test_rank_view_order_has_equal_steps_and_marks_padding():
    shards = [
        _rank_view_order(list(range(10)), rank=rank, world_size=4, views_per_step=2)
        for rank in range(4)
    ]
    assert {len(shard) for shard in shards} == {4}
    assert sum(is_real for shard in shards for _, is_real in shard) == 10
    assert all(len(shard) % 2 == 0 for shard in shards)


def test_cache_rank_order_balances_views_and_covers_all_real_views():
    shards = [
        _cache_rank_order(11, rank=rank, world_size=3, views_per_step=2)
        for rank in range(3)
    ]
    assert {len(shard) for shard in shards} == {4}
    real_views = [view for shard in shards for view, is_real in shard if is_real]
    assert sorted(real_views) == list(range(11))
    assert all(len(shard) % 2 == 0 for shard in shards)


def test_pair_balanced_view_order_covers_views_and_balances_pairs():
    order = list(range(10))
    pair_counts = {index: (index + 1) * 100 for index in order}
    shards = [
        _pair_balanced_view_order(
            order,
            pair_counts,
            rank=rank,
            world_size=4,
            views_per_step=2,
        )
        for rank in range(4)
    ]
    assert {len(shard) for shard in shards} == {4}
    assert all(len(shard) % 2 == 0 for shard in shards)
    real_views = [view for shard in shards for view, is_real in shard if is_real]
    assert sorted(real_views) == order
    assert sum(is_real for shard in shards for _, is_real in shard) == len(order)
    loads = [
        sum(pair_counts[view] for view, is_real in shard if is_real)
        for shard in shards
    ]
    assert max(loads) - min(loads) <= max(pair_counts.values())


def test_pair_balanced_view_order_rejects_missing_pair_count():
    try:
        _pair_balanced_view_order([0, 1], {0: 1}, rank=0, world_size=2, views_per_step=1)
    except ValueError as exc:
        assert "missing view indices" in str(exc)
    else:
        raise AssertionError("missing pair counts should fail fast")


def test_pair_balanced_view_order_reduces_per_step_stragglers():
    order = [0, 1, 2, 3]
    pair_counts = {0: 1000, 1: 10, 2: 900, 3: 20}
    baseline = [
        _rank_view_order(order, rank=rank, world_size=2, views_per_step=1)
        for rank in range(2)
    ]
    balanced = [
        _pair_balanced_view_order(
            order,
            pair_counts,
            rank=rank,
            world_size=2,
            views_per_step=1,
        )
        for rank in range(2)
    ]

    def total_step_spread(shards):
        return sum(
            max(pair_counts[shards[rank][step][0]] for rank in range(2))
            - min(pair_counts[shards[rank][step][0]] for rank in range(2))
            for step in range(2)
        )

    assert total_step_spread(balanced) < total_step_spread(baseline)


def test_compiled_renderer_state_keys_are_normalized():
    class CompiledState:
        def state_dict(self):
            return {
                "feature_field.features": torch.ones(2),
                "renderer_model._orig_mod.geometry_embedding.weight": torch.zeros(3),
            }

    state = _normalized_model_state_dict(CompiledState())
    assert "renderer_model.geometry_embedding.weight" in state
    assert all("_orig_mod" not in key for key in state)


def test_local_sparse_octree_search_returns_sorted_voxel_positions():
    octree = LocalSparseOctree(
        torch.tensor([[1, 0, 2], [0, 3, 1], [0, 0, 0]], dtype=torch.int32), 2, "cpu"
    )
    x, y, z, batch = octree.xyzb(2)
    assert torch.equal(torch.stack([x, y, z], dim=1), torch.tensor([[0, 0, 0], [0, 3, 1], [1, 0, 2]]))
    query = torch.tensor(
        [[0, 0, 0, 0], [1, 0, 2, 0], [2, 0, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]
    )
    assert torch.equal(octree.search_xyzb(query, 2), torch.tensor([0, 2, -1, -1, -1]))
    assert torch.equal(batch, torch.zeros(3, dtype=torch.int64))


def test_inference_checkpoint_rebuilds_exact_vertex_mapping(tmp_path):
    torch.manual_seed(11)
    voxels = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.int32)
    octree = LocalSparseOctree(voxels, 2, "cpu")
    index = compute_vertex_index(octree, 2, include_batch=True)
    model = LocalSceneModel(
        index,
        feature_dim=4,
        hidden_dim=12,
        n_iters=2,
        voxel_feature_dim=3,
        position_encoding_frequencies=2,
    )
    resume = save_local_checkpoint(
        tmp_path / "resume.pt", model, object_key="category/object", query_depth=2, epoch=3
    )
    inference = save_local_inference_checkpoint(
        tmp_path / "inference.pt",
        model,
        voxel_coordinates=voxels,
        object_key="category/object",
        query_depth=2,
    )
    loaded_resume, _ = load_local_checkpoint(resume, expected_object="category/object")
    loaded, loaded_octree, payload = load_local_inference_checkpoint(
        inference, expected_object="category/object"
    )
    assert payload["schema_version"] == 2
    assert payload["voxel_coordinates"].dtype == torch.uint8
    assert torch.equal(loaded_octree.voxel_coordinates, octree.voxel_coordinates)
    assert torch.equal(loaded.feature_field.vertex_coordinates, model.feature_field.vertex_coordinates)
    assert torch.equal(loaded.feature_field.voxel_vertices, model.feature_field.voxel_vertices)
    for expected, actual in zip(loaded_resume.renderer_model.state_dict().values(), loaded.renderer_model.state_dict().values()):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    torch.testing.assert_close(
        loaded.feature_field.features, model.feature_field.features, rtol=0, atol=0
    )
    torch.testing.assert_close(
        loaded.feature_field.voxel_features,
        model.feature_field.voxel_features,
        rtol=0,
        atol=0,
    )


def test_schema_v1_and_v2_render_identically(tmp_path):
    torch.manual_seed(17)
    voxels = torch.cartesian_prod(torch.arange(4), torch.arange(4), torch.arange(4)).to(torch.int32)
    octree = LocalSparseOctree(voxels, 2, "cpu")
    index = compute_vertex_index(octree, 2, include_batch=True)
    model = LocalSceneModel(
        index,
        feature_dim=4,
        hidden_dim=12,
        n_iters=2,
        voxel_feature_dim=3,
        position_encoding_frequencies=2,
    )
    resume = save_local_checkpoint(
        tmp_path / "resume.pt", model, object_key="category/object", query_depth=2, epoch=1
    )
    inference = save_local_inference_checkpoint(
        tmp_path / "inference.pt",
        model,
        voxel_coordinates=voxels,
        object_key="category/object",
        query_depth=2,
    )
    old_model, _ = load_local_checkpoint(resume, expected_object="category/object")
    new_model, new_octree, _ = load_local_inference_checkpoint(
        inference, expected_object="category/object"
    )
    camera = CameraIntrinsics(width=4, height=4, fx=4.0, fy=4.0, cx=2.0, cy=2.0)
    transform = torch.eye(4)
    old_renderer = SparseRenderer(old_model.renderer_model, old_model.feature_field, octree, depth=2)
    new_renderer = SparseRenderer(new_model.renderer_model, new_model.feature_field, new_octree, depth=2)
    old_output = old_renderer.render_frame(transform, camera)
    new_output = new_renderer.render_frame(transform, camera)
    for expected, actual in zip(
        (old_output.position, old_output.normal, old_output.color, old_output.hit),
        (new_output.position, new_output.normal, new_output.color, new_output.hit),
    ):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_segment_position_stays_between_voxel_entry_and_exit():
    voxels = torch.tensor([[0, 0, 0]], dtype=torch.int32)
    octree = LocalSparseOctree(voxels, 1, "cpu")
    index = compute_vertex_index(octree, 1, include_batch=True)
    model = LocalSceneModel(
        index,
        feature_dim=4,
        hidden_dim=12,
        n_iters=2,
        segment_position=True,
    )
    pin = torch.tensor([[0.0, 0.25, 0.5]])
    pout = torch.tensor([[1.0, 0.25, 0.5]])
    direction = torch.tensor([[1.0, 0.0, 0.0]])
    prediction = model(pin, direction, torch.tensor([0]), pout)

    for stage in prediction.stages:
        assert 0.0 <= stage.position[0, 0] <= 1.0
        torch.testing.assert_close(stage.position[0, 1:], pin[0, 1:])


def test_fourier_position_encoding_has_expected_values_and_shape():
    model = LocalSceneModel(
        _vertex_index(),
        feature_dim=4,
        hidden_dim=16,
        n_iters=1,
        position_encoding_frequencies=2,
    )
    positions = torch.tensor([[0.0, 0.5, 1.0]])
    encoded = model.renderer_model.encode_position(positions)
    assert encoded.shape == (1, 12)
    expected_phase = positions.unsqueeze(-1) * torch.tensor([1.0, 2.0]) * torch.pi
    expected = torch.cat([torch.sin(expected_phase), torch.cos(expected_phase)], dim=-1).flatten(1)
    torch.testing.assert_close(encoded, expected)


def test_voxel_residual_and_fourier_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(23)
    voxels = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int32)
    octree = LocalSparseOctree(voxels, 1, "cpu")
    index = compute_vertex_index(octree, 1, include_batch=True)
    model = LocalSceneModel(
        index,
        feature_dim=4,
        hidden_dim=12,
        n_iters=2,
        geometry_only=True,
        segment_position=True,
        voxel_feature_dim=3,
        position_encoding_frequencies=2,
    )
    pins = torch.rand(4, 3)
    pouts = torch.ones(4, 3)
    directions = torch.nn.functional.normalize(pouts - pins, dim=1)
    voxel_indices = torch.tensor([0, 1, 0, 1])
    expected = model(pins, directions, voxel_indices, pouts).final
    loss = expected.normal.square().mean() + expected.position.square().mean()
    loss.backward()
    assert model.feature_field.voxel_features is not None
    assert model.feature_field.voxel_features.grad is not None

    resume = save_local_checkpoint(
        tmp_path / "resume_detail.pt",
        model,
        object_key="category/object",
        query_depth=1,
        epoch=1,
    )
    inference = save_local_inference_checkpoint(
        tmp_path / "inference_detail.pt",
        model,
        voxel_coordinates=voxels.flip(0),
        object_key="category/object",
        query_depth=1,
        storage_dtype="fp16",
    )
    loaded_resume, resume_payload = load_local_checkpoint(resume)
    loaded_inference, _, inference_payload = load_local_inference_checkpoint(inference)
    assert resume_payload["model_config"]["voxel_feature_dim"] == 3
    assert inference_payload["model_config"]["position_encoding_frequencies"] == 2
    assert inference_payload["vertex_features"].dtype == torch.float16
    for loaded in (loaded_resume, loaded_inference):
        actual = loaded(pins, directions, voxel_indices, pouts).final
        tolerance = 0.0 if loaded is loaded_resume else 2.0e-3
        torch.testing.assert_close(actual.position, expected.position, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(actual.normal, expected.normal, atol=tolerance, rtol=tolerance)


def test_old_model_config_without_detail_fields_still_loads(tmp_path):
    model = LocalSceneModel(_vertex_index(), feature_dim=4, hidden_dim=12, n_iters=1)
    checkpoint = save_local_checkpoint(
        tmp_path / "old_style.pt",
        model,
        object_key="category/object",
        query_depth=7,
        epoch=1,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["model_config"].pop("voxel_feature_dim")
    payload["model_config"].pop("position_encoding_frequencies")
    torch.save(payload, checkpoint)
    loaded, _ = load_local_checkpoint(checkpoint)
    assert loaded.feature_field.voxel_feature_dim == 0
    assert loaded.renderer_model.position_encoding_frequencies == 0


def _sampling_inputs(count: int = 5):
    pin = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.15, 0.4], [0.0, 0.5, 0.5]])
    pout = torch.tensor([[0.9, 0.2, 0.3], [0.6, 0.15, 0.8], [0.4, 0.5, 0.5]])
    pins = pin.repeat((count + 2) // 3, 1)[:count]
    pouts = pout.repeat((count + 2) // 3, 1)[:count]
    directions = torch.nn.functional.normalize(pouts - pins, dim=1)
    return pins, directions, pouts


def test_sampling_modes_share_parameters_and_stage_counts():
    torch.manual_seed(5)
    modes = {"recurrent": 4, "uniform": 4, "single": 1}
    parameter_shapes = {}
    pins, directions, pouts = _sampling_inputs()
    voxel_indices = torch.zeros(pins.shape[0], dtype=torch.long)
    for mode, n_iters in modes.items():
        torch.manual_seed(5)
        model = LocalSceneModel(
            _vertex_index(),
            feature_dim=4,
            hidden_dim=16,
            n_iters=n_iters,
            geometry_only=True,
            sampling_mode=mode,
        )
        prediction = model(pins, directions, voxel_indices, pouts)
        expected_stages = n_iters if mode == "recurrent" else 1
        assert len(prediction.stages) == expected_stages
        assert model.renderer_model.sampling_mode == mode
        parameter_shapes[mode] = {
            name: tuple(value.shape) for name, value in model.named_parameters()
        }
    assert parameter_shapes["uniform"] == parameter_shapes["recurrent"]
    assert parameter_shapes["single"] == parameter_shapes["recurrent"]


def test_sampling_modes_backpropagate_to_vertex_features_and_renderer():
    pins, directions, pouts = _sampling_inputs()
    voxel_indices = torch.zeros(pins.shape[0], dtype=torch.long)
    for mode, n_iters in (("recurrent", 4), ("uniform", 4), ("single", 1)):
        torch.manual_seed(7)
        model = LocalSceneModel(
            _vertex_index(),
            feature_dim=4,
            hidden_dim=16,
            n_iters=n_iters,
            geometry_only=True,
            sampling_mode=mode,
        )
        prediction = model(pins, directions, voxel_indices, pouts)
        loss = (
            prediction.final.position.square().mean()
            + prediction.final.normal.square().mean()
            + prediction.final.hit_probability.mean()
        )
        loss.backward()
        assert model.feature_field.features.grad is not None
        missing = [name for name, value in model.named_parameters() if value.grad is None]
        assert missing == []


def test_uniform_sampling_spans_entry_and_exit_endpoints():
    model = LocalSceneModel(
        _vertex_index(),
        feature_dim=4,
        hidden_dim=16,
        n_iters=4,
        geometry_only=True,
        sampling_mode="uniform",
    )
    pin = torch.tensor([[0.0, 0.25, 0.5], [0.2, 0.5, 0.1]])
    pout = torch.tensor([[1.0, 0.25, 0.5], [0.8, 0.5, 0.4]])
    queries = model.renderer_model.sampled_query_positions(pin, pout)
    assert queries.shape == (2, 4, 3)
    torch.testing.assert_close(queries[:, 0], pin)
    torch.testing.assert_close(queries[:, -1], pout)
    spacing = (pout - pin) / 3
    for index in range(1, 4):
        torch.testing.assert_close(queries[:, index], pin + spacing * index)


def test_single_sampling_queries_only_the_entry_point():
    model = LocalSceneModel(
        _vertex_index(),
        feature_dim=4,
        hidden_dim=16,
        n_iters=4,
        geometry_only=True,
        sampling_mode="single",
    )
    pin = torch.tensor([[0.1, 0.2, 0.3]])
    pout = torch.tensor([[0.9, 0.2, 0.3]])
    queries = model.renderer_model.sampled_query_positions(pin, pout)
    assert queries.shape == (1, 1, 3)
    torch.testing.assert_close(queries[:, 0], pin)


def test_uniform_sampling_requires_exit_point():
    model = LocalSceneModel(
        _vertex_index(),
        feature_dim=4,
        hidden_dim=16,
        n_iters=4,
        geometry_only=True,
        sampling_mode="uniform",
    )
    pins, directions, _ = _sampling_inputs()
    voxel_indices = torch.zeros(pins.shape[0], dtype=torch.long)
    try:
        model(pins, directions, voxel_indices, None)
    except ValueError as exc:
        assert "pout is required" in str(exc)
    else:
        raise AssertionError("uniform sampling without pout should fail fast")


def test_sampling_mode_survives_checkpoint_roundtrips(tmp_path):
    voxels = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int32)
    octree = LocalSparseOctree(voxels, 1, "cpu")
    index = compute_vertex_index(octree, 1, include_batch=True)
    pins, directions, pouts = _sampling_inputs(4)
    voxel_indices = torch.tensor([0, 1, 0, 1])
    for mode, n_iters in (("uniform", 4), ("single", 1)):
        torch.manual_seed(13)
        model = LocalSceneModel(
            index,
            feature_dim=4,
            hidden_dim=16,
            n_iters=n_iters,
            geometry_only=True,
            sampling_mode=mode,
        )
        expected = model(pins, directions, voxel_indices, pouts).final
        resume = save_local_checkpoint(
            tmp_path / f"{mode}_resume.pt",
            model,
            object_key="category/object",
            query_depth=1,
            epoch=1,
        )
        inference = save_local_inference_checkpoint(
            tmp_path / f"{mode}_inference.pt",
            model,
            voxel_coordinates=voxels,
            object_key="category/object",
            query_depth=1,
        )
        loaded_resume, resume_payload = load_local_checkpoint(resume)
        loaded_inference, _, inference_payload = load_local_inference_checkpoint(inference)
        assert resume_payload["model_config"]["sampling_mode"] == mode
        assert inference_payload["model_config"]["sampling_mode"] == mode
        assert loaded_resume.renderer_model.sampling_mode == mode
        assert loaded_inference.renderer_model.sampling_mode == mode
        for loaded in (loaded_resume, loaded_inference):
            actual = loaded(pins, directions, voxel_indices, pouts).final
            torch.testing.assert_close(actual.position, expected.position, rtol=0, atol=0)
            torch.testing.assert_close(actual.normal, expected.normal, rtol=0, atol=0)
            torch.testing.assert_close(
                actual.hit_probability, expected.hit_probability, rtol=0, atol=0
            )


def test_old_checkpoints_default_to_recurrent_sampling(tmp_path):
    model = LocalSceneModel(_vertex_index(), feature_dim=4, hidden_dim=12, n_iters=2)
    checkpoint = save_local_checkpoint(
        tmp_path / "pre_sampling_mode.pt",
        model,
        object_key="category/object",
        query_depth=7,
        epoch=1,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["model_config"].pop("sampling_mode")
    torch.save(payload, checkpoint)
    loaded, _ = load_local_checkpoint(checkpoint)
    assert loaded.renderer_model.sampling_mode == "recurrent"


def test_endpoint_modes_share_parameter_shapes():
    shapes = {}
    for mode in ("recurrent_endpoints", "uniform_endpoints"):
        torch.manual_seed(5)
        model = LocalSceneModel(
            _vertex_index(),
            feature_dim=4,
            hidden_dim=16,
            n_iters=3,
            geometry_only=True,
            sampling_mode=mode,
        )
        shapes[mode] = {
            name: tuple(value.shape) for name, value in model.named_parameters()
        }
    assert shapes["recurrent_endpoints"] == shapes["uniform_endpoints"]
    embedding_dim = model.renderer_model.geometry_embedding.in_features
    # 3 features + 3 positions + direction + normal + hit (no position encoding).
    assert embedding_dim == 3 * 4 + 9 + 3 + 3 + 1


def test_recurrent_endpoints_starts_at_midpoint_and_adapts():
    torch.manual_seed(9)
    model = LocalSceneModel(
        _vertex_index(),
        feature_dim=4,
        hidden_dim=16,
        n_iters=3,
        geometry_only=True,
        sampling_mode="recurrent_endpoints",
    )
    recorded: list[torch.Tensor] = []
    original = model.renderer_model.get_scene_feature_for_points

    def spy(vertex_features, positions):
        recorded.append(positions.detach().clone())
        return original(vertex_features, positions)

    model.renderer_model.get_scene_feature_for_points = spy
    pins, directions, pouts = _sampling_inputs(2)
    prediction = model(pins, directions, torch.zeros(2, dtype=torch.long), pouts)
    assert len(prediction.stages) == 3
    # entry, exit, then one query per stage
    assert len(recorded) == 5
    torch.testing.assert_close(recorded[0], pins)
    torch.testing.assert_close(recorded[1], pouts)
    torch.testing.assert_close(recorded[2], (pins + pouts) * 0.5)
    assert not torch.allclose(recorded[3], recorded[2])
    assert not torch.allclose(recorded[4], recorded[3])


def test_uniform_endpoints_uses_fixed_interior_grid():
    model = LocalSceneModel(
        _vertex_index(),
        feature_dim=4,
        hidden_dim=16,
        n_iters=3,
        geometry_only=True,
        sampling_mode="uniform_endpoints",
    )
    pin = torch.tensor([[0.0, 0.25, 0.5], [0.2, 0.5, 0.1]])
    pout = torch.tensor([[1.0, 0.25, 0.5], [0.8, 0.5, 0.4]])
    queries = model.renderer_model.endpoint_query_positions(pin, pout)
    assert queries.shape == (2, 3, 3)
    for index, fraction in enumerate((0.25, 0.5, 0.75)):
        torch.testing.assert_close(queries[:, index], pin + (pout - pin) * fraction)
    prediction = model(pin, torch.nn.functional.normalize(pout - pin, dim=1), torch.zeros(2, dtype=torch.long), pout)
    assert len(prediction.stages) == 3


def test_uniform_concat_flattens_all_points_into_one_input():
    torch.manual_seed(31)
    model = LocalSceneModel(
        _vertex_index(),
        feature_dim=4,
        hidden_dim=16,
        n_iters=4,
        geometry_only=True,
        sampling_mode="uniform_concat",
    )
    renderer = model.renderer_model
    # 4 points x (feature 4 + position 3) + direction 3 + normal 3 + hit 1
    assert renderer.geometry_embedding.in_features == 4 * 7 + 3 + 3 + 1
    pins, directions, pouts = _sampling_inputs(3)
    voxel_indices = torch.zeros(pins.shape[0], dtype=torch.long)
    prediction = model(pins, directions, voxel_indices, pouts)
    assert len(prediction.stages) == 1
    loss = (
        prediction.final.position.square().mean()
        + prediction.final.normal.square().mean()
        + prediction.final.hit_probability.mean()
    )
    loss.backward()
    missing = [name for name, value in model.named_parameters() if value.grad is None]
    assert missing == []


def test_uniform_concat_endpoints_input_shape_and_single_pass():
    model = LocalSceneModel(
        _vertex_index(),
        feature_dim=4,
        hidden_dim=16,
        n_iters=3,
        geometry_only=True,
        sampling_mode="uniform_concat_endpoints",
    )
    renderer = model.renderer_model
    # entry + exit + 3 interior points, each (feature 4 + position 3), plus
    # direction, normal and hit.
    assert renderer.geometry_embedding.in_features == 5 * 7 + 3 + 3 + 1
    pins, directions, pouts = _sampling_inputs(2)
    voxel_indices = torch.zeros(pins.shape[0], dtype=torch.long)
    prediction = model(pins, directions, voxel_indices, pouts)
    assert len(prediction.stages) == 1


def test_uniform_concat_checkpoint_roundtrip(tmp_path):
    voxels = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int32)
    octree = LocalSparseOctree(voxels, 1, "cpu")
    index = compute_vertex_index(octree, 1, include_batch=True)
    pins, directions, pouts = _sampling_inputs(4)
    voxel_indices = torch.tensor([0, 1, 0, 1])
    for mode, n_iters in (("uniform_concat", 4), ("uniform_concat_endpoints", 3)):
        torch.manual_seed(33)
        model = LocalSceneModel(
            index,
            feature_dim=4,
            hidden_dim=16,
            n_iters=n_iters,
            geometry_only=True,
            sampling_mode=mode,
        )
        expected = model(pins, directions, voxel_indices, pouts).final
        checkpoint = save_local_checkpoint(
            tmp_path / f"{mode}.pt",
            model,
            object_key="category/object",
            query_depth=1,
            epoch=1,
        )
        loaded, payload = load_local_checkpoint(checkpoint)
        assert payload["model_config"]["sampling_mode"] == mode
        assert loaded.renderer_model.sampling_mode == mode
        actual = loaded(pins, directions, voxel_indices, pouts).final
        torch.testing.assert_close(actual.position, expected.position, rtol=0, atol=0)
        torch.testing.assert_close(actual.normal, expected.normal, rtol=0, atol=0)


def test_recurrent_pool_matches_uniform_architecture_and_adapts():
    torch.manual_seed(41)
    pool = LocalSceneModel(
        _vertex_index(),
        feature_dim=4,
        hidden_dim=16,
        n_iters=2,
        geometry_only=True,
        sampling_mode="recurrent_pool",
    )
    torch.manual_seed(41)
    fixed = LocalSceneModel(
        _vertex_index(),
        feature_dim=4,
        hidden_dim=16,
        n_iters=4,
        geometry_only=True,
        sampling_mode="uniform",
    )
    # Pooled fusion shares the encoder, so parameter shapes are independent of
    # the number of sampled points and must match the uniform pool exactly.
    assert {
        name: tuple(value.shape) for name, value in pool.named_parameters()
    } == {name: tuple(value.shape) for name, value in fixed.named_parameters()}
    # One point at a time: feature 4 + direction 3 + position 3 + normal 3 + hit 1
    assert pool.renderer_model.geometry_embedding.in_features == 4 + 3 + 3 + 3 + 1

    recorded: list[torch.Tensor] = []
    original = pool.renderer_model.get_scene_feature_for_points

    def spy(vertex_features, positions):
        recorded.append(positions.detach().clone())
        return original(vertex_features, positions)

    pool.renderer_model.get_scene_feature_for_points = spy
    pins, directions, pouts = _sampling_inputs(3)
    voxel_indices = torch.zeros(pins.shape[0], dtype=torch.long)
    prediction = pool(pins, directions, voxel_indices, pouts)
    assert len(prediction.stages) == 1
    # entry, exit, then one query per adaptive iteration
    assert len(recorded) == 4
    torch.testing.assert_close(recorded[0], pins)
    torch.testing.assert_close(recorded[1], pouts)
    torch.testing.assert_close(recorded[2], (pins + pouts) * 0.5)
    assert not torch.allclose(recorded[3], recorded[2])


def test_recurrent_pool_backpropagates_and_roundtrips(tmp_path):
    voxels = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int32)
    octree = LocalSparseOctree(voxels, 1, "cpu")
    index = compute_vertex_index(octree, 1, include_batch=True)
    pins, directions, pouts = _sampling_inputs(4)
    voxel_indices = torch.tensor([0, 1, 0, 1])
    torch.manual_seed(43)
    model = LocalSceneModel(
        index,
        feature_dim=4,
        hidden_dim=16,
        n_iters=2,
        geometry_only=True,
        sampling_mode="recurrent_pool",
    )
    prediction = model(pins, directions, voxel_indices, pouts)
    loss = (
        prediction.final.position.square().mean()
        + prediction.final.normal.square().mean()
        + prediction.final.hit_probability.mean()
    )
    loss.backward()
    missing = [name for name, value in model.named_parameters() if value.grad is None]
    assert missing == []
    expected = prediction.final
    checkpoint = save_local_checkpoint(
        tmp_path / "recurrent_pool.pt",
        model,
        object_key="category/object",
        query_depth=1,
        epoch=1,
    )
    loaded, payload = load_local_checkpoint(checkpoint)
    assert payload["model_config"]["sampling_mode"] == "recurrent_pool"
    assert loaded.renderer_model.sampling_mode == "recurrent_pool"
    actual = loaded(pins, directions, voxel_indices, pouts).final
    torch.testing.assert_close(actual.position, expected.position, rtol=0, atol=0)
    torch.testing.assert_close(actual.normal, expected.normal, rtol=0, atol=0)


def test_endpoint_modes_backpropagate_and_roundtrip(tmp_path):
    voxels = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int32)
    octree = LocalSparseOctree(voxels, 1, "cpu")
    index = compute_vertex_index(octree, 1, include_batch=True)
    pins, directions, pouts = _sampling_inputs(4)
    voxel_indices = torch.tensor([0, 1, 0, 1])
    for mode in ("recurrent_endpoints", "uniform_endpoints"):
        torch.manual_seed(21)
        model = LocalSceneModel(
            index,
            feature_dim=4,
            hidden_dim=16,
            n_iters=3,
            geometry_only=True,
            sampling_mode=mode,
        )
        prediction = model(pins, directions, voxel_indices, pouts)
        loss = (
            prediction.final.position.square().mean()
            + prediction.final.normal.square().mean()
            + prediction.final.hit_probability.mean()
        )
        loss.backward()
        missing = [name for name, value in model.named_parameters() if value.grad is None]
        assert missing == []
        expected = prediction.final
        resume = save_local_checkpoint(
            tmp_path / f"{mode}_resume.pt",
            model,
            object_key="category/object",
            query_depth=1,
            epoch=1,
        )
        loaded, payload = load_local_checkpoint(resume)
        assert payload["model_config"]["sampling_mode"] == mode
        assert loaded.renderer_model.sampling_mode == mode
        actual = loaded(pins, directions, voxel_indices, pouts).final
        torch.testing.assert_close(actual.position, expected.position, rtol=0, atol=0)
        torch.testing.assert_close(actual.normal, expected.normal, rtol=0, atol=0)



def test_single_endpoints_uses_entry_and_exit_only():
    torch.manual_seed(51)
    model = LocalSceneModel(
        _vertex_index(),
        feature_dim=4,
        hidden_dim=16,
        n_iters=1,
        geometry_only=True,
        sampling_mode="single_endpoints",
    )
    renderer = model.renderer_model
    # entry block + exit block, each (feature 4 + position 3), plus direction,
    # normal and hit.
    assert renderer.geometry_embedding.in_features == 2 * 7 + 3 + 3 + 1
    recorded: list[torch.Tensor] = []
    original = renderer.get_scene_feature_for_points

    def spy(vertex_features, positions):
        recorded.append(positions.detach().clone())
        return original(vertex_features, positions)

    renderer.get_scene_feature_for_points = spy
    pins, directions, pouts = _sampling_inputs(3)
    voxel_indices = torch.zeros(pins.shape[0], dtype=torch.long)
    prediction = model(pins, directions, voxel_indices, pouts)
    assert len(prediction.stages) == 1
    # Exactly two lookups: entry then exit. No interior query.
    assert len(recorded) == 2
    torch.testing.assert_close(recorded[0], pins)
    torch.testing.assert_close(recorded[1], pouts)
    loss = (
        prediction.final.position.square().mean()
        + prediction.final.normal.square().mean()
        + prediction.final.hit_probability.mean()
    )
    loss.backward()
    missing = [name for name, value in model.named_parameters() if value.grad is None]
    assert missing == []


def test_single_endpoints_checkpoint_roundtrip(tmp_path):
    voxels = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int32)
    octree = LocalSparseOctree(voxels, 1, "cpu")
    index = compute_vertex_index(octree, 1, include_batch=True)
    pins, directions, pouts = _sampling_inputs(4)
    voxel_indices = torch.tensor([0, 1, 0, 1])
    torch.manual_seed(53)
    model = LocalSceneModel(
        index,
        feature_dim=4,
        hidden_dim=16,
        n_iters=1,
        geometry_only=True,
        sampling_mode="single_endpoints",
    )
    expected = model(pins, directions, voxel_indices, pouts).final
    checkpoint = save_local_checkpoint(
        tmp_path / "single_endpoints.pt",
        model,
        object_key="category/object",
        query_depth=1,
        epoch=1,
    )
    loaded, payload = load_local_checkpoint(checkpoint)
    assert payload["model_config"]["sampling_mode"] == "single_endpoints"
    actual = loaded(pins, directions, voxel_indices, pouts).final
    torch.testing.assert_close(actual.position, expected.position, rtol=0, atol=0)
    torch.testing.assert_close(actual.normal, expected.normal, rtol=0, atol=0)
