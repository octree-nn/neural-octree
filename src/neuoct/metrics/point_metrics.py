from __future__ import annotations

import torch
from torch import Tensor


def _mean_nearest_squared_distance(
    source: Tensor, target: Tensor, *, source_chunk: int, target_chunk: int
) -> Tensor:
    minima: list[Tensor] = []
    for source_start in range(0, source.shape[0], source_chunk):
        source_part = source[source_start : source_start + source_chunk]
        nearest = torch.full(
            (source_part.shape[0],), torch.inf, dtype=source.dtype, device=source.device
        )
        for target_start in range(0, target.shape[0], target_chunk):
            target_part = target[target_start : target_start + target_chunk]
            distances = (source_part[:, None, :] - target_part[None, :, :]).square().sum(dim=-1)
            nearest = torch.minimum(nearest, distances.min(dim=1).values)
        minima.append(nearest)
    return torch.cat(minima).mean()


def symmetric_chamfer_distance(
    points_a: Tensor,
    points_b: Tensor,
    *,
    source_chunk: int = 2048,
    target_chunk: int = 2048,
) -> Tensor:
    """Symmetric squared Chamfer distance without constructing the full NxM matrix."""
    if points_a.ndim != 2 or points_b.ndim != 2 or points_a.shape[1] != points_b.shape[1]:
        raise ValueError("Point tensors must have shape [N, D] and [M, D]")
    if points_a.shape[0] == 0 or points_b.shape[0] == 0:
        raise ValueError("Chamfer distance requires two non-empty point sets")
    if points_a.device != points_b.device:
        raise ValueError("Point tensors must be on the same device")
    return _mean_nearest_squared_distance(
        points_a, points_b, source_chunk=source_chunk, target_chunk=target_chunk
    ) + _mean_nearest_squared_distance(
        points_b, points_a, source_chunk=source_chunk, target_chunk=target_chunk
    )
