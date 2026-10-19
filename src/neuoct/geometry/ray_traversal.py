from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class RayVoxelPairs:
    pins: Tensor
    pouts: Tensor
    voxel_indices: Tensor
    ray_labels: Tensor
    voxel_coordinates: Tensor
    hits: Tensor | None = None
    positions: Tensor | None = None
    normals: Tensor | None = None
    colors: Tensor | None = None

    @classmethod
    def concatenate(cls, chunks: list["RayVoxelPairs"]) -> "RayVoxelPairs":
        if not chunks:
            raise ValueError("Cannot concatenate an empty pair list")

        def cat_optional(name: str) -> Tensor | None:
            values = [getattr(chunk, name) for chunk in chunks]
            if values[0] is None:
                return None
            return torch.cat(values, dim=0)

        return cls(
            pins=torch.cat([chunk.pins for chunk in chunks], dim=0),
            pouts=torch.cat([chunk.pouts for chunk in chunks], dim=0),
            voxel_indices=torch.cat([chunk.voxel_indices for chunk in chunks], dim=0),
            ray_labels=torch.cat([chunk.ray_labels for chunk in chunks], dim=0),
            voxel_coordinates=torch.cat([chunk.voxel_coordinates for chunk in chunks], dim=0),
            hits=cat_optional("hits"),
            positions=cat_optional("positions"),
            normals=cat_optional("normals"),
            colors=cat_optional("colors"),
        )


def safe_inverse(values: Tensor, eps: float = 1.0e-10) -> Tensor:
    signs = torch.where(values < 0, -torch.ones_like(values), torch.ones_like(values))
    safe_values = torch.where(values.abs() < eps, signs * eps, values)
    return safe_values.reciprocal()


def grouped_interleave_indices(sorted_labels: Tensor, *, stable: bool = True) -> Tensor:
    if sorted_labels.ndim != 1:
        raise ValueError(f"Expected one-dimensional labels, got {tuple(sorted_labels.shape)}")
    if sorted_labels.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=sorted_labels.device)
    if bool((sorted_labels[1:] < sorted_labels[:-1]).any()):
        raise ValueError("grouped_interleave_indices requires sorted labels")
    _, counts = torch.unique_consecutive(sorted_labels, return_counts=True)
    starts = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=sorted_labels.device), counts.cumsum(0)[:-1]]
    )
    indices = torch.arange(sorted_labels.numel(), device=sorted_labels.device)
    relative = indices - torch.repeat_interleave(starts, counts)
    return torch.argsort(relative, stable=stable)


def _plane_parameters(origins: Tensor, directions: Tensor, scale: int) -> Tensor:
    planes = torch.arange(scale + 1, dtype=origins.dtype, device=origins.device)
    parameters = (planes.view(1, -1, 1) - origins.unsqueeze(1)) * safe_inverse(
        directions
    ).unsqueeze(1)
    return parameters.reshape(origins.shape[0], -1).sort(dim=-1).values


def _occupied_segments(
    origins: Tensor,
    directions: Tensor,
    octree: object,
    *,
    scale: int,
    depth: int,
    batch_id: int,
    ray_label_offset: int,
    segment_mask: Tensor | None = None,
) -> RayVoxelPairs:
    parameters = _plane_parameters(origins, directions, scale)
    xyz = origins[:, None, :] + parameters.unsqueeze(-1) * directions[:, None, :]
    pins = xyz[:, :-1, :]
    pouts = xyz[:, 1:, :]
    labels = torch.arange(origins.shape[0], device=origins.device, dtype=torch.long)
    labels = labels[:, None].expand(-1, pins.shape[1]) + ray_label_offset
    in_bounds = (
        (pins >= -0.001) & (pins <= scale + 0.001) & (pouts >= -0.001) & (pouts <= scale + 0.001)
    ).all(dim=-1)
    selected = in_bounds if segment_mask is None else in_bounds & segment_mask
    pins = pins[selected]
    pouts = pouts[selected]
    labels = labels[selected]
    query_xyz = ((pins + pouts) * 0.5).long()
    batch_column = torch.full(
        (query_xyz.shape[0], 1), batch_id, dtype=torch.long, device=query_xyz.device
    )
    indices = octree.search_xyzb(
        torch.cat([query_xyz, batch_column], dim=1), depth=depth, nempty=True
    )
    valid = indices >= 0
    query_xyz = query_xyz[valid]
    return RayVoxelPairs(
        pins=pins[valid] - query_xyz,
        pouts=pouts[valid] - query_xyz,
        voxel_indices=indices[valid].long(),
        ray_labels=labels[valid],
        voxel_coordinates=query_xyz,
    )


def enumerate_ray_voxel_pairs(
    origins: Tensor,
    directions: Tensor,
    octree: object,
    *,
    scale: int,
    depth: int,
    batch_id: int = 0,
    ray_label_offset: int = 0,
    ray_chunk_size: int = 16384,
) -> RayVoxelPairs:
    chunks = [
        _occupied_segments(
            origins[start : start + ray_chunk_size],
            directions[start : start + ray_chunk_size],
            octree,
            scale=scale,
            depth=depth,
            batch_id=batch_id,
            ray_label_offset=ray_label_offset + start,
        )
        for start in range(0, origins.shape[0], ray_chunk_size)
    ]
    return RayVoxelPairs.concatenate(chunks)


def _query_supervised_chunk(
    origins: Tensor,
    directions: Tensor,
    intersections: Tensor,
    octree: object,
    *,
    scale: int,
    depth: int,
    batch_id: int,
    ray_label_offset: int,
) -> RayVoxelPairs:
    parameters = _plane_parameters(origins, directions, scale)
    points = intersections[:, 1:4]
    delta = points - origins
    t_hit = (delta * directions).sum(dim=1, keepdim=True) / (
        directions.square().sum(dim=1, keepdim=True) + 1.0e-12
    )
    before_hit = parameters[:, :-1] <= t_hit
    pairs = _occupied_segments(
        origins,
        directions,
        octree,
        scale=scale,
        depth=depth,
        batch_id=batch_id,
        ray_label_offset=ray_label_offset,
        segment_mask=before_hit,
    )
    # Recover selected hit labels using the same occupancy lookup. A segment contains the
    # supervised point iff its local entry/exit interval contains that point.
    global_ray = pairs.ray_labels - ray_label_offset
    local_position = intersections[global_ray, 1:4] - pairs.voxel_coordinates
    segment_min = torch.minimum(pairs.pins, pairs.pouts) - 1.0e-5
    segment_max = torch.maximum(pairs.pins, pairs.pouts) + 1.0e-5
    contains_point = ((local_position >= segment_min) & (local_position <= segment_max)).all(dim=1)
    contains_point &= intersections[global_ray, 0] != 0
    pairs.hits = contains_point.to(torch.long)
    pairs.positions = intersections[global_ray, 1:4]
    normals = intersections[global_ray, 4:7].clone()
    view_directions = F.normalize(pairs.pouts - pairs.pins, dim=-1)
    wrong_orientation = (normals * view_directions).sum(dim=-1) > 0
    normals[wrong_orientation] = -normals[wrong_orientation]
    pairs.normals = normals
    pairs.colors = intersections[global_ray, 7:10]
    return pairs


def query_supervised_pairs(
    origins: Tensor,
    directions: Tensor,
    intersections: Tensor,
    octree: object,
    *,
    scale: int,
    depth: int,
    batch_id: int = 0,
    ray_label_offset: int = 0,
    ray_chunk_size: int = 16384,
) -> RayVoxelPairs:
    chunks: list[RayVoxelPairs] = []
    for start in range(0, origins.shape[0], ray_chunk_size):
        stop = min(start + ray_chunk_size, origins.shape[0])
        chunks.append(
            _query_supervised_chunk(
                origins[start:stop],
                directions[start:stop],
                intersections[start:stop],
                octree,
                scale=scale,
                depth=depth,
                batch_id=batch_id,
                ray_label_offset=ray_label_offset + start,
            )
        )
    return RayVoxelPairs.concatenate(chunks)
