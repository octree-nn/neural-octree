from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class LocalPositionEncoder:
    depth: int

    def encode(self, global_positions: Tensor) -> Tensor:
        return global_positions - global_positions.floor()

    def decode(self, local_positions: Tensor, voxel_coordinates: Tensor) -> Tensor:
        return voxel_coordinates.to(local_positions.dtype) + local_positions
