from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from neuoct.geometry.vertex_index import VertexIndex, compute_vertex_index


@dataclass(frozen=True)
class VertexFeatureField:
    vertex_index: VertexIndex
    features: Tensor
    voxel_features: Tensor | None = None
    voxel_colors: Tensor | None = None

    def gather(self, voxel_indices: Tensor) -> Tensor:
        voxel_vertices = self.vertex_index.voxel_vertices[voxel_indices.long()]
        return self.features[voxel_vertices.reshape(-1, 2, 2, 2)]

    def gather_voxels(self, voxel_indices: Tensor) -> Tensor:
        if self.voxel_features is None:
            raise RuntimeError("This feature field does not contain voxel-local features")
        return self.voxel_features[voxel_indices.long()]

    def gather_colors(self, voxel_indices: Tensor) -> Tensor:
        if self.voxel_colors is None:
            raise RuntimeError("This feature field does not contain voxel colors")
        return self.voxel_colors[voxel_indices.long()]


class VertexFeatureAggregator:
    """Aggregates the eight corner slots emitted for every occupied voxel."""

    @staticmethod
    def from_voxel_corner_features(
        octree: object,
        depth: int,
        voxel_features: Tensor,
        *,
        voxel_local_features: Tensor | None = None,
        voxel_colors: Tensor | None = None,
    ) -> VertexFeatureField:
        vertex_index = compute_vertex_index(octree, depth, include_batch=True)
        voxel_vertices = vertex_index.voxel_vertices
        if voxel_features.shape[0] != voxel_vertices.shape[0]:
            raise ValueError(
                f"Voxel feature rows ({voxel_features.shape[0]}) do not match occupied voxels "
                f"({voxel_vertices.shape[0]}) at depth {depth}."
            )
        if voxel_features.shape[1] % 8 != 0:
            raise ValueError(
                f"Voxel feature channels must be divisible by 8, got {voxel_features.shape[1]}"
            )
        feature_channels = voxel_features.shape[1] // 8
        num_vertices = vertex_index.num_vertices
        slots = torch.zeros(
            (num_vertices * 8, feature_channels + 1),
            dtype=voxel_features.dtype,
            device=voxel_features.device,
        )
        corner_ids = torch.arange(8, device=voxel_features.device).unsqueeze(0)
        slot_indices = voxel_vertices * 8 + corner_ids
        reshaped = voxel_features.reshape(-1, feature_channels)
        weighted = torch.cat(
            [
                reshaped,
                torch.ones((reshaped.shape[0], 1), device=reshaped.device, dtype=reshaped.dtype),
            ],
            dim=1,
        )
        slots[slot_indices.reshape(-1)] = weighted
        slots = slots.reshape(num_vertices, 8, feature_channels + 1)
        weight_sum = slots[..., -1:].sum(dim=1)
        if bool((weight_sum == 0).any()):
            raise RuntimeError("A unique vertex received no voxel-corner features")
        features = slots[..., :-1].sum(dim=1) / weight_sum
        if voxel_local_features is not None and voxel_local_features.shape[0] != voxel_vertices.shape[0]:
            raise ValueError(
                "Voxel-local feature rows must match occupied voxels: "
                f"{voxel_local_features.shape[0]} vs {voxel_vertices.shape[0]}"
            )
        if voxel_colors is not None and voxel_colors.shape != (voxel_vertices.shape[0], 3):
            raise ValueError(
                "Voxel colors must have shape "
                f"({voxel_vertices.shape[0]}, 3), got {tuple(voxel_colors.shape)}"
            )
        return VertexFeatureField(
            vertex_index=vertex_index,
            features=features,
            voxel_features=voxel_local_features,
            voxel_colors=voxel_colors,
        )
