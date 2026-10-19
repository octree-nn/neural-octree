from __future__ import annotations

from collections.abc import Sequence

import torch
from ocnn.octree import Octree
from torch import Tensor

from .vertex_index import compute_vertex_index


class NeuOctOctree(Octree):
    """Thin OCNN adapter containing only NeuOct-specific convenience methods."""

    def compute_vertices(self, target_depth: int = 5) -> tuple[int, Tensor]:
        index = compute_vertex_index(self, target_depth, include_batch=True)
        return index.num_vertices, index.voxel_vertices

    def get_split(self, depth: int, nempty: bool = False) -> Tensor:
        children = self.children[depth]
        if children is None:
            count = int(self.nnum_nempty[depth] if nempty else self.nnum[depth])
            return torch.zeros(count, dtype=torch.int32, device=self.device)
        split = (children >= 0).to(torch.int32)
        return split[self.nempty_mask(depth)] if nempty else split


def merge_octree_batch(octrees: Sequence[NeuOctOctree]) -> NeuOctOctree:
    if not octrees:
        raise ValueError("At least one octree is required")
    merged = NeuOctOctree.init_like(octrees[0])
    merged.merge_octrees(list(octrees))
    return merged


def build_input_features(octree: Octree, feature_spec: str) -> Tensor:
    """Builds N/F/L/D/P features with the legacy channel order."""
    depth = octree.depth
    spec = feature_spec.upper()
    features: list[Tensor] = []
    if "N" in spec:
        if octree.normals[depth] is None:
            raise ValueError("Feature spec requests normals, but the octree has no normals")
        features.append(octree.normals[depth])
    if "F" in spec:
        if octree.features[depth] is None:
            raise ValueError("Feature spec requests point features, but the octree has none")
        features.append(octree.features[depth])
    local_points: Tensor | None = None
    if "L" in spec or "D" in spec:
        if octree.points[depth] is None:
            raise ValueError("Feature spec requests local points, but the octree has none")
        local_points = octree.points[depth].frac() - 0.5
    if "L" in spec:
        features.append(local_points)
    if "D" in spec:
        if octree.normals[depth] is None:
            raise ValueError("Feature D requires octree normals")
        features.append((local_points * octree.normals[depth]).sum(dim=1, keepdim=True))
    if "P" in spec:
        if octree.points[depth] is None:
            raise ValueError("Feature P requires octree points")
        scale = 2 ** (1 - depth)
        features.append(octree.points[depth] * scale - 1.0)
    if not features:
        raise ValueError(f"Feature spec {feature_spec!r} selects no channels")
    return torch.cat(features, dim=1)
