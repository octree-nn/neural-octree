from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

VERTEX_OFFSETS = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
)


@dataclass(frozen=True)
class VertexIndex:
    """Unique batch-aware vertices and the eight vertex ids of every occupied voxel."""

    coordinates: Tensor
    voxel_vertices: Tensor

    @property
    def num_vertices(self) -> int:
        return int(self.coordinates.shape[0])


def compute_vertex_index(octree: object, depth: int, *, include_batch: bool = True) -> VertexIndex:
    x, y, z, batch = octree.xyzb(depth, nempty=True)
    voxel_xyz = torch.stack([x, y, z], dim=1)
    offsets = torch.tensor(VERTEX_OFFSETS, device=voxel_xyz.device, dtype=voxel_xyz.dtype)
    vertex_xyz = voxel_xyz.unsqueeze(1) + offsets.unsqueeze(0)
    if include_batch:
        vertex_batch = batch.reshape(-1, 1, 1).expand(-1, 8, 1).to(voxel_xyz.dtype)
        keyed_vertices = torch.cat([vertex_batch, vertex_xyz], dim=-1)
    else:
        keyed_vertices = vertex_xyz
    coordinates, inverse = torch.unique(
        keyed_vertices.reshape(-1, keyed_vertices.shape[-1]), dim=0, return_inverse=True
    )
    return VertexIndex(coordinates=coordinates, voxel_vertices=inverse.reshape(-1, 8))
