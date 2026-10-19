from __future__ import annotations

from pathlib import Path

import torch

from neuoct.data.dataset import ShapeNetMultiViewDataset
from neuoct.data.preprocess import DeviceBatchPreprocessor

from .local_checkpoint import (
    load_local_checkpoint,
    save_local_inference_checkpoint,
)
from .local_optimize import LocalOptimizationConfig


def pack_local_checkpoint(
    config_path: str | Path,
    checkpoint: str | Path,
    *,
    object_key: str,
    output: str | Path,
    data_split: str = "train",
    device: str = "cuda",
    storage_dtype: str = "fp32",
) -> Path:
    """Convert a schema-v1 resume checkpoint into a schema-v2 inference bundle.

    The source point cloud is used only while packing to recover the occupied
    q-depth voxels. The resulting file contains those voxel coordinates and no
    registered topology buffers or optimizer state.
    """
    holder = LocalOptimizationConfig.from_yaml(
        config_path,
        object_key=object_key,
        output=output,
        data_split=data_split,
        epochs=1,
        views=1,
        device=device,
    )
    resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
    model, payload = load_local_checkpoint(
        checkpoint, device=resolved_device, expected_object=object_key
    )
    if int(payload["query_depth"]) != holder.data.query_depth:
        raise ValueError("Checkpoint and packing configuration query depths differ")

    dataset = ShapeNetMultiViewDataset.for_object(holder.data, object_key, take=1)
    preprocessor = DeviceBatchPreprocessor(holder.data, resolved_device)
    octree = preprocessor.build_octree(dataset[0])
    x, y, z, batch = octree.xyzb(holder.data.query_depth, nempty=True)
    voxel_coordinates = torch.stack([batch, x, y, z], dim=1)
    packed = save_local_inference_checkpoint(
        output,
        model,
        voxel_coordinates=voxel_coordinates,
        object_key=object_key,
        query_depth=holder.data.query_depth,
        storage_dtype=storage_dtype,
    )
    return packed
