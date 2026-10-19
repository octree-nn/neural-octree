from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .camera import CameraIntrinsics
from .manifest import ManifestEntry


@dataclass
class RawSample:
    entry: ManifestEntry
    point_positions: Tensor
    point_colors: Tensor
    intrinsics: CameraIntrinsics
    camera_to_world: Tensor
    color_image: Tensor
    depth_image: Tensor
    normal_image: Tensor
    valid_depth: Tensor
    depth_min: float
    depth_max: float

    @property
    def image_shape(self) -> tuple[int, int]:
        return self.intrinsics.height, self.intrinsics.width


@dataclass
class PreparedBatch:
    octree: object
    pins: Tensor
    pouts: Tensor
    voxel_indices: Tensor
    hits: Tensor
    positions: Tensor
    normals: Tensor
    colors: Tensor
    ray_labels: Tensor
    batch_size: int
    image_height: int
    image_width: int
    entries: tuple[ManifestEntry, ...]

    @property
    def device(self) -> torch.device:
        return self.pins.device

    @property
    def directions(self) -> Tensor:
        return torch.nn.functional.normalize(self.pouts - self.pins, dim=-1)

    @property
    def local_positions(self) -> Tensor:
        return self.positions - self.positions.floor()
