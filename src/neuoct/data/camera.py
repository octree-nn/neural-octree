from __future__ import annotations

from dataclasses import dataclass
from math import tan

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_horizontal_fov(
        cls, camera_angle_x: float, width: int, height: int
    ) -> "CameraIntrinsics":
        focal = 0.5 * width / tan(0.5 * float(camera_angle_x))
        return cls(width=width, height=height, fx=focal, fy=focal, cx=width / 2, cy=height / 2)

    def matrix(self, *, device: torch.device | str, dtype: torch.dtype = torch.float32) -> Tensor:
        return torch.tensor(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )


def generate_camera_directions(
    intrinsics: CameraIntrinsics,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    convention: str = "neuoct",
) -> Tensor:
    x, y = torch.meshgrid(
        torch.arange(intrinsics.width, device=device, dtype=dtype) + 0.5,
        torch.arange(intrinsics.height, device=device, dtype=dtype) + 0.5,
        indexing="xy",
    )
    if convention == "neuoct":
        components = [
            -(x - intrinsics.cx) / intrinsics.fx,
            -(y - intrinsics.cy) / intrinsics.fy,
            torch.ones_like(x),
        ]
    elif convention == "legacy_blender":
        components = [
            (x - intrinsics.cx) / intrinsics.fx,
            -(y - intrinsics.cy) / intrinsics.fy,
            -torch.ones_like(x),
        ]
    else:
        raise ValueError(f"Unsupported camera convention: {convention!r}")
    return torch.stack(components, dim=-1)


def generate_rays(
    intrinsics: CameraIntrinsics,
    camera_to_world: Tensor,
    *,
    convention: str = "neuoct",
) -> tuple[Tensor, Tensor]:
    directions = generate_camera_directions(
        intrinsics,
        device=camera_to_world.device,
        dtype=camera_to_world.dtype,
        convention=convention,
    )
    rotation = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]
    if convention == "legacy_blender":
        world_directions = F.normalize(directions @ rotation.T, dim=-1)
    else:
        world_directions = F.normalize(directions @ rotation, dim=-1)
    world_directions = world_directions.reshape(-1, 3)
    origins = translation.expand(world_directions.shape[0], 3)
    return origins, world_directions


def unproject_rgbd(
    depth: Tensor,
    normal_rgb: Tensor,
    color_rgb: Tensor,
    intrinsics: CameraIntrinsics,
    camera_to_world: Tensor,
    *,
    convention: str = "neuoct",
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError(
            f"Depth shape {tuple(depth.shape)} does not match camera "
            f"{intrinsics.height}x{intrinsics.width}."
        )
    directions = generate_camera_directions(
        intrinsics,
        device=depth.device,
        dtype=depth.dtype,
        convention=convention,
    )
    rotation = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]
    if convention == "legacy_blender":
        world_directions = directions @ rotation.T
        points = (directions * depth.unsqueeze(-1)) @ rotation.T + translation
    else:
        world_directions = directions @ rotation
        points = (directions * depth.unsqueeze(-1)) @ rotation + translation
    ray_dirs = F.normalize(world_directions, dim=-1).reshape(-1, 3)
    origins = translation.expand(ray_dirs.shape[0], 3)
    if normal_rgb.is_floating_point():
        normals = normal_rgb.reshape(-1, 3).float()
    else:
        normals = normal_rgb.reshape(-1, 3).float() / 255.0 * 2.0 - 1.0
    colors = color_rgb.reshape(-1, 3).float() / 255.0
    return origins, ray_dirs, points.reshape(-1, 3), normals, colors
