from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from neuoct.models.vertex_features import VertexFeatureField

FEATURE_SCHEMA_VERSION = 1


def save_feature_export(
    path: str | Path,
    *,
    entries: list[str],
    query_depth: int,
    voxel_coordinates: Tensor,
    voxel_features: Tensor,
    field: VertexFeatureField,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "coordinate_order": "batch,x,y,z",
        "entries": entries,
        "query_depth": int(query_depth),
        "voxel_coordinates": voxel_coordinates.detach().to("cpu", torch.int32),
        "voxel_features": voxel_features.detach().cpu(),
        "vertex_coordinates": field.vertex_index.coordinates.detach().to("cpu", torch.int32),
        "vertex_features": field.features.detach().cpu(),
        "voxel_vertex_indices": field.vertex_index.voxel_vertices.detach().to("cpu", torch.int64),
    }
    torch.save(payload, output)
    return output


def load_feature_export(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Feature export must be a dictionary")
    required = {
        "schema_version",
        "coordinate_order",
        "entries",
        "query_depth",
        "voxel_coordinates",
        "voxel_features",
        "vertex_coordinates",
        "vertex_features",
        "voxel_vertex_indices",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Feature export is missing keys: {sorted(missing)}")
    if payload["schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported feature schema {payload['schema_version']}; "
            f"expected {FEATURE_SCHEMA_VERSION}."
        )
    if payload["coordinate_order"] != "batch,x,y,z":
        raise ValueError(f"Unsupported coordinate order: {payload['coordinate_order']}")
    return payload
