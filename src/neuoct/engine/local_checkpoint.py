from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neuoct.geometry.local_octree import LocalSparseOctree
from neuoct.geometry.vertex_index import VertexIndex, compute_vertex_index
from neuoct.models.local_scene import LocalSceneModel

LOCAL_CHECKPOINT_SCHEMA_VERSION = 1
LOCAL_INFERENCE_SCHEMA_VERSION = 2


def _normalized_model_state_dict(model: LocalSceneModel) -> dict[str, torch.Tensor]:
    """Return checkpoint keys independent of whether the renderer is compiled.

    ``torch.compile`` wraps the renderer in ``OptimizedModule`` and exposes its
    parameters under ``renderer_model._orig_mod.*``.  Checkpoints are consumed
    by the ordinary ``SplitVoxelRendererModel`` during evaluation, so remove
    that implementation detail at the serialization boundary.
    """
    normalized: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        key = key.replace("renderer_model._orig_mod.", "renderer_model.")
        if key in normalized:
            raise ValueError(f"Duplicate normalized model state key: {key}")
        normalized[key] = value.detach().cpu()
    return normalized


def _renderer_config(model: LocalSceneModel) -> dict[str, Any]:
    renderer = getattr(model.renderer_model, "_orig_mod", model.renderer_model)
    return {
        "feature_dim": model.feature_field.feature_dim,
        "voxel_feature_dim": model.feature_field.voxel_feature_dim,
        "hidden_dim": int(renderer.geometry_embedding.out_features),
        "n_iters": int(renderer.n_iters),
        "geometry_only": bool(getattr(renderer, "geometry_only", False)),
        "segment_position": bool(getattr(renderer, "segment_position", False)),
        "position_encoding_frequencies": int(
            getattr(renderer, "position_encoding_frequencies", 0)
        ),
        "sampling_mode": str(getattr(renderer, "sampling_mode", "recurrent")),
    }


def get_local_checkpoint_schema(path: str | Path) -> int:
    """Read only the schema marker without constructing a model."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "schema_version" not in payload:
        raise ValueError("Local checkpoint must contain a schema_version")
    return int(payload["schema_version"])


def _packed_coordinate_tensor(coordinates: torch.Tensor, depth: int) -> torch.Tensor:
    if coordinates.numel() == 0:
        raise ValueError("At least one coordinate is required")
    maximum = int(coordinates.max())
    if maximum < 256 and depth <= 8:
        return coordinates.to(torch.uint8)
    if maximum < 65536 and depth <= 16:
        return coordinates.to(torch.uint16)
    return coordinates.to(torch.int32)


def save_local_inference_checkpoint(
    path: str | Path,
    model: LocalSceneModel,
    *,
    voxel_coordinates: torch.Tensor,
    object_key: str,
    query_depth: int,
    storage_dtype: str = "fp32",
) -> Path:
    """Save an inference bundle that rebuilds topology from leaf voxels."""
    if storage_dtype not in ("fp32", "fp16"):
        raise ValueError("storage_dtype must be 'fp32' or 'fp16'")
    coordinates = torch.as_tensor(voxel_coordinates, dtype=torch.long)
    if coordinates.ndim == 2 and coordinates.shape[1] == 4:
        if bool((coordinates[:, 0] != 0).any()):
            raise ValueError("Inference local bundles only support batch id 0")
        coordinates = coordinates[:, 1:]
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("voxel_coordinates must have shape (N, 3) or (N, 4)")
    local_octree = LocalSparseOctree(coordinates, query_depth, "cpu")
    index = compute_vertex_index(local_octree, query_depth, include_batch=True)
    old_coordinates = model.feature_field.vertex_coordinates.detach().cpu().long()
    if old_coordinates.ndim != 2 or old_coordinates.shape[1] != 4:
        raise ValueError("Model vertex coordinates must have shape (N, 4)")
    if bool((old_coordinates[:, 0] != 0).any()):
        raise ValueError("Inference local bundles only support batch id 0")
    old_voxel_vertices = model.feature_field.voxel_vertices.detach().cpu().long()
    if old_voxel_vertices.ndim != 2 or old_voxel_vertices.shape[1] != 8:
        raise ValueError("Model voxel-to-vertex mapping must have shape (N, 8)")
    old_voxel_xyz = old_coordinates[old_voxel_vertices].amin(dim=1)
    old_voxel_xyz = old_voxel_xyz[:, 1:]
    voxel_scale = 1 << query_depth
    old_voxel_keys = (
        old_voxel_xyz[:, 0] * voxel_scale * voxel_scale
        + old_voxel_xyz[:, 1] * voxel_scale
        + old_voxel_xyz[:, 2]
    )
    new_voxel_xyz = local_octree.voxel_coordinates.cpu().long()
    new_voxel_keys = (
        new_voxel_xyz[:, 0] * voxel_scale * voxel_scale
        + new_voxel_xyz[:, 1] * voxel_scale
        + new_voxel_xyz[:, 2]
    )
    if not torch.equal(torch.sort(old_voxel_keys).values, torch.sort(new_voxel_keys).values):
        raise ValueError("Saved voxel coordinates do not match the model voxel topology")
    # Vertices are on the q-depth grid, whose upper boundary is ``scale``.
    # Use a base of scale + 1 so boundary coordinates cannot collide with a
    # neighboring interior coordinate in the flattened lookup key.
    key_base = (1 << query_depth) + 1
    old_xyz = old_coordinates[:, 1:]
    old_keys = (
        old_xyz[:, 0] * key_base * key_base
        + old_xyz[:, 1] * key_base
        + old_xyz[:, 2]
    )
    sorted_old_keys, sorted_old_rows = torch.sort(old_keys)
    if bool((sorted_old_keys[1:] == sorted_old_keys[:-1]).any()):
        raise ValueError("Model vertex coordinates must be unique")
    new_xyz = index.coordinates[:, 1:].cpu().long()
    new_keys = (
        new_xyz[:, 0] * key_base * key_base
        + new_xyz[:, 1] * key_base
        + new_xyz[:, 2]
    )
    new_positions = torch.searchsorted(sorted_old_keys, new_keys)
    valid = new_positions < sorted_old_keys.numel()
    safe = new_positions.clamp_max(sorted_old_keys.numel() - 1)
    valid &= sorted_old_keys[safe] == new_keys
    if not bool(valid.all()):
        raise ValueError("Saved voxel coordinates do not match the model vertex field")
    old_rows = sorted_old_rows[new_positions]
    vertex_features = model.feature_field.features.detach().cpu()[old_rows]
    storage_type = torch.float16 if storage_dtype == "fp16" else torch.float32
    renderer_state = {
        key.removeprefix("renderer_model."): value.to(storage_type)
        for key, value in _normalized_model_state_dict(model).items()
        if key.startswith("renderer_model.")
    }
    voxel_features = None
    if model.feature_field.voxel_features is not None:
        sorted_old_voxel_keys, sorted_old_voxel_rows = torch.sort(old_voxel_keys)
        new_voxel_positions = torch.searchsorted(sorted_old_voxel_keys, new_voxel_keys)
        valid_voxels = new_voxel_positions < sorted_old_voxel_keys.numel()
        safe_voxels = new_voxel_positions.clamp_max(sorted_old_voxel_keys.numel() - 1)
        valid_voxels &= sorted_old_voxel_keys[safe_voxels] == new_voxel_keys
        if not bool(valid_voxels.all()):
            raise ValueError("Saved voxel coordinates do not match the model voxel features")
        old_voxel_rows = sorted_old_voxel_rows[new_voxel_positions]
        voxel_features = model.feature_field.voxel_features.detach().cpu()[old_voxel_rows]
    payload: dict[str, Any] = {
        "schema_version": LOCAL_INFERENCE_SCHEMA_VERSION,
        "kind": "neuoct_local_inference",
        "object_key": str(object_key),
        "query_depth": int(query_depth),
        "model_config": _renderer_config(model),
        "coordinate_order": "x,y,z",
        "voxel_coordinates": _packed_coordinate_tensor(
            local_octree.voxel_coordinates.cpu(), query_depth
        ),
        "vertex_features": vertex_features.to(storage_type),
        "storage_dtype": storage_dtype,
        "renderer_state_dict": renderer_state,
    }
    if voxel_features is not None:
        payload["voxel_features"] = voxel_features.to(storage_type)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return output


def save_local_checkpoint(
    path: str | Path,
    model: LocalSceneModel,
    *,
    object_key: str,
    query_depth: int,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    """Save an object-optimization checkpoint with its complete scene topology."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    field = model.feature_field
    payload: dict[str, Any] = {
        "schema_version": LOCAL_CHECKPOINT_SCHEMA_VERSION,
        "kind": "neuoct_local_scene",
        "object_key": str(object_key),
        "query_depth": int(query_depth),
        "epoch": int(epoch),
        "model_config": _renderer_config(model),
        "vertex_coordinates": field.vertex_coordinates.detach().cpu(),
        "voxel_vertices": field.voxel_vertices.detach().cpu(),
        "model_state_dict": _normalized_model_state_dict(model),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if config is not None:
        payload["config"] = config
    torch.save(payload, output)
    return output


def load_local_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    expected_object: str | None = None,
) -> tuple[LocalSceneModel, dict[str, Any]]:
    """Load and reconstruct a local scene model from a self-contained checkpoint."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Local checkpoint must contain a dictionary")
    required = {
        "schema_version",
        "kind",
        "object_key",
        "query_depth",
        "model_config",
        "vertex_coordinates",
        "voxel_vertices",
        "model_state_dict",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Local checkpoint is missing fields: {sorted(missing)}")
    if int(payload["schema_version"]) != LOCAL_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported local checkpoint schema: {payload['schema_version']}")
    if payload["kind"] != "neuoct_local_scene":
        raise ValueError(f"Unsupported checkpoint kind: {payload['kind']!r}")
    if expected_object is not None and payload["object_key"] != expected_object:
        raise ValueError(
            f"Checkpoint object {payload['object_key']!r} does not match {expected_object!r}"
        )
    model_config = payload["model_config"]
    if not isinstance(model_config, dict):
        raise ValueError("model_config must be a dictionary")
    coordinates = torch.as_tensor(payload["vertex_coordinates"], device=device, dtype=torch.int32)
    voxel_vertices = torch.as_tensor(payload["voxel_vertices"], device=device, dtype=torch.long)
    vertex_index = VertexIndex(coordinates, voxel_vertices)
    model = LocalSceneModel(
        vertex_index,
        feature_dim=int(model_config["feature_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        n_iters=int(model_config["n_iters"]),
        geometry_only=bool(model_config.get("geometry_only", False)),
        segment_position=bool(model_config.get("segment_position", False)),
        voxel_feature_dim=int(model_config.get("voxel_feature_dim", 0)),
        position_encoding_frequencies=int(
            model_config.get("position_encoding_frequencies", 0)
        ),
        sampling_mode=str(model_config.get("sampling_mode", "recurrent")),
    ).to(device)
    state = payload["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError("model_state_dict must be a dictionary")
    model.load_state_dict(state, strict=True)
    return model, payload


def load_local_inference_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    expected_object: str | None = None,
) -> tuple[LocalSceneModel, LocalSparseOctree, dict[str, Any]]:
    """Load schema-v2 local inference data and rebuild its topology."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Local inference checkpoint must contain a dictionary")
    required = {
        "schema_version",
        "kind",
        "object_key",
        "query_depth",
        "model_config",
        "coordinate_order",
        "voxel_coordinates",
        "vertex_features",
        "renderer_state_dict",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Local inference checkpoint is missing fields: {sorted(missing)}")
    if int(payload["schema_version"]) != LOCAL_INFERENCE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported local inference schema: {payload['schema_version']}")
    if payload["kind"] != "neuoct_local_inference":
        raise ValueError(f"Unsupported checkpoint kind: {payload['kind']!r}")
    if payload["coordinate_order"] != "x,y,z":
        raise ValueError(f"Unsupported coordinate order: {payload['coordinate_order']!r}")
    storage_dtype = payload.get("storage_dtype", "fp32")
    if storage_dtype not in ("fp32", "fp16"):
        raise ValueError(f"Unsupported inference storage dtype: {storage_dtype!r}")
    if expected_object is not None and payload["object_key"] != expected_object:
        raise ValueError(
            f"Checkpoint object {payload['object_key']!r} does not match {expected_object!r}"
        )
    model_config = payload["model_config"]
    if not isinstance(model_config, dict):
        raise ValueError("model_config must be a dictionary")
    coordinates = torch.as_tensor(payload["voxel_coordinates"], device=device, dtype=torch.long)
    octree = LocalSparseOctree(coordinates, int(payload["query_depth"]), device)
    index = compute_vertex_index(octree, octree.depth, include_batch=True)
    vertex_features = torch.as_tensor(payload["vertex_features"], device=device, dtype=torch.float32)
    if vertex_features.shape != (index.num_vertices, int(model_config["feature_dim"])):
        raise ValueError(
            "vertex_features shape does not match dynamically rebuilt vertex topology: "
            f"got {tuple(vertex_features.shape)}, expected "
            f"({index.num_vertices}, {int(model_config['feature_dim'])})"
        )
    model = LocalSceneModel(
        index,
        feature_dim=int(model_config["feature_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        n_iters=int(model_config["n_iters"]),
        geometry_only=bool(model_config.get("geometry_only", False)),
        segment_position=bool(model_config.get("segment_position", False)),
        voxel_feature_dim=int(model_config.get("voxel_feature_dim", 0)),
        position_encoding_frequencies=int(
            model_config.get("position_encoding_frequencies", 0)
        ),
        sampling_mode=str(model_config.get("sampling_mode", "recurrent")),
    ).to(device)
    with torch.no_grad():
        model.feature_field.features.copy_(vertex_features)
        voxel_feature_dim = int(model_config.get("voxel_feature_dim", 0))
        if voxel_feature_dim:
            if "voxel_features" not in payload:
                raise ValueError("Inference checkpoint is missing voxel_features")
            voxel_features = torch.as_tensor(
                payload["voxel_features"], device=device, dtype=torch.float32
            )
            expected_shape = (index.voxel_vertices.shape[0], voxel_feature_dim)
            if voxel_features.shape != expected_shape:
                raise ValueError(
                    "voxel_features shape does not match rebuilt voxel topology: "
                    f"got {tuple(voxel_features.shape)}, expected {expected_shape}"
                )
            if model.feature_field.voxel_features is None:
                raise RuntimeError("Model unexpectedly has no voxel-local feature table")
            model.feature_field.voxel_features.copy_(voxel_features)
    renderer_state = payload["renderer_state_dict"]
    if not isinstance(renderer_state, dict):
        raise ValueError("renderer_state_dict must be a dictionary")
    model.renderer_model.load_state_dict(renderer_state, strict=True)
    return model, octree, payload
