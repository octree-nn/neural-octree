from __future__ import annotations

import torch
from torch import Tensor


class LocalSparseOctree:
    """Leaf-coordinate octree adapter used by inference-only local bundles.

    Local scene rendering only queries occupied cells at the renderer depth. A
    sorted leaf table therefore provides the same ``xyzb``/``search_xyzb``
    contract needed by ray traversal without rebuilding an OCNN point octree.
    """

    def __init__(self, voxel_coordinates: Tensor, depth: int, device: torch.device | str):
        coordinates = torch.as_tensor(voxel_coordinates, device=device, dtype=torch.long)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("voxel_coordinates must have shape (N, 3)")
        if coordinates.shape[0] == 0:
            raise ValueError("At least one occupied voxel is required")
        if depth <= 0:
            raise ValueError("depth must be positive")
        scale = 1 << depth
        if bool((coordinates < 0).any()) or bool((coordinates >= scale).any()):
            raise ValueError(f"voxel coordinates must be in [0, {scale})")
        keys = self._coords_to_keys(coordinates, scale)
        sorted_keys, order = torch.sort(keys)
        if bool((sorted_keys[1:] == sorted_keys[:-1]).any()):
            raise ValueError("voxel_coordinates must be unique")
        self.depth = int(depth)
        self.scale = scale
        self.device = coordinates.device
        self.voxel_coordinates = coordinates[order].contiguous()
        self.voxel_keys = sorted_keys.contiguous()

    @staticmethod
    def _coords_to_keys(coordinates: Tensor, scale: int) -> Tensor:
        return coordinates[:, 0] * scale * scale + coordinates[:, 1] * scale + coordinates[:, 2]

    @property
    def num_voxels(self) -> int:
        return int(self.voxel_coordinates.shape[0])

    def construct_all_neigh(self) -> None:
        """Compatibility no-op; local traversal does not query neighbor tables."""

    def xyzb(self, depth: int, nempty: bool = True) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        del nempty
        if int(depth) != self.depth:
            raise ValueError(f"LocalSparseOctree only supports depth {self.depth}, got {depth}")
        x, y, z = self.voxel_coordinates.unbind(dim=1)
        batch = torch.zeros_like(x)
        return x, y, z, batch

    def search_xyzb(self, query: Tensor, depth: int, nempty: bool = True) -> Tensor:
        del nempty
        if int(depth) != self.depth:
            raise ValueError(f"LocalSparseOctree only supports depth {self.depth}, got {depth}")
        if query.ndim != 2 or query.shape[1] != 4:
            raise ValueError("query must have shape (N, 4) = x,y,z,batch")
        query_coordinates = query[:, :3].long()
        query_batch = query[:, 3].long()
        in_bounds = ((query_coordinates >= 0) & (query_coordinates < self.scale)).all(dim=1)
        keys = self._coords_to_keys(query_coordinates, self.scale)
        positions = torch.searchsorted(self.voxel_keys, keys)
        valid = positions < self.voxel_keys.numel()
        safe_positions = positions.clamp_max(self.voxel_keys.numel() - 1)
        valid &= self.voxel_keys[safe_positions] == keys
        valid &= in_bounds
        valid &= query_batch == 0
        return torch.where(valid, positions, torch.full_like(positions, -1))
