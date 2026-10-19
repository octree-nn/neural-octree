"""Mesh-derived surface-occupancy point clouds.

``scripts/build_mesh_occupancy.py`` writes one ``<uid>.npz`` per Objaverse object
holding every query-depth grid cell the *mesh* surface touches (exact GPU
triangle/box separating-axis rasterisation, the same guarantee as kaolin's
``unbatched_mesh_to_spc``).  The back-projected point clouds that the pipeline
normally consumes only contain what the 8 source depth maps happened to see, so
self-occluded surface has no octree node at all and the renderer cannot hit it
(those pixels are the ``MISSING`` holes).

This module turns those files back into extra input points so a dataset can
*union* them with the original point cloud:

  * ``cells``    int16 ``[N, 3]`` integer cell indices on the ``2**query_depth`` grid
  * ``offsets``  uint8 ``[N, 3]`` in-cell position, quantised to 1/256 of a cell
  * ``colors``   uint8 ``[N, 3]`` albedo, nearest-neighbour from the source cloud

The reconstructed positions come back in the pipeline **world** frame, which is
what ``pointcloud.pth`` stores, so they can simply be concatenated with the
back-projected point cloud before ``DeviceBatchPreprocessor`` applies
``point_scale``.  Internally the files are quantised on the ``2**query_depth``
grid and reconstructed in the octree input frame that
``ocnn.Octree.build_octree`` normalises from ``[-1, 1]`` to ``[0, 2**depth]``
(``world * point_scale``); the final divide by ``point_scale`` undoes that.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

__all__ = ["OccupancyCache", "occupancy_points_world"]

# 1/256 of a cell is the quantisation used when the files were written.
_OFFSET_LEVELS = 256.0


def _cap_cells(cells: torch.Tensor, file_depth: int, query_depth: int,
               max_cells: int) -> torch.Tensor:
    """Indices keeping at most ``max_cells`` cells without losing query-depth coverage.

    Cells are grouped by their ``query_depth`` parent and a strided subset is kept
    from each group, so every parent keeps at least one child.  The query-depth
    occupancy -- and therefore the renderer's candidate count and the absence of
    holes -- is untouched; only the sub-cell detail of the very largest meshes is
    thinned.  Needed because a handful of Objaverse meshes reach ~8M depth-9 cells,
    where the scene U-Net alone no longer fits in 24 GB.
    """
    shift = int(file_depth) - int(query_depth)
    if shift <= 0:
        return torch.arange(cells.shape[0], device=cells.device)
    parent = cells.long() >> shift
    scale = 1 << int(query_depth)
    key = (parent[:, 0] * scale + parent[:, 1]) * scale + parent[:, 2]
    order = torch.argsort(key, stable=True)
    key_sorted = key[order]
    count = key_sorted.shape[0]
    changed = torch.nonzero(key_sorted[1:] != key_sorted[:-1]).reshape(-1) + 1
    first = torch.cat([torch.zeros(1, dtype=torch.long, device=cells.device), changed])
    counts = torch.diff(torch.cat([first, torch.tensor([count], device=cells.device)]))
    rank = torch.arange(count, device=cells.device) - torch.repeat_interleave(first, counts)
    groups = int(first.numel())
    per_group = max(1, max_cells // max(groups, 1))
    stride = torch.repeat_interleave(((counts + per_group - 1) // per_group).clamp_min(1), counts)
    return order[(rank % stride) == 0]


def occupancy_points_world(
    path: str | Path,
    *,
    query_depth: int,
    ray_scale: float,
    point_scale: float,
    max_cells: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(positions, colors)`` in the pipeline world frame.

    ``positions`` are the per-cell surface representatives (the mean of the mesh
    samples that landed in the cell, or the cell centre when a cell was only
    grazed) and ``colors`` are floats in ``[0, 1]``.  ``max_cells > 0`` thins the
    largest objects as described in :func:`_cap_cells`.
    """
    blob = np.load(path)
    # files written before ``depth`` was recorded were all built at the query depth
    file_depth = int(blob["depth"]) if "depth" in blob.files else int(query_depth)
    raw_cells = torch.from_numpy(np.asarray(blob["cells"])).to(device)
    offsets = torch.from_numpy(np.asarray(blob["offsets"])).to(device)
    colors = torch.from_numpy(np.asarray(blob["colors"])).to(device)
    if max_cells > 0 and raw_cells.shape[0] > max_cells:
        keep = _cap_cells(raw_cells, file_depth, query_depth, max_cells)
        raw_cells, offsets, colors = raw_cells[keep], offsets[keep], colors[keep]
    cells = raw_cells.to(torch.float32)
    offsets = offsets.to(torch.float32)
    scale = 1 << file_depth
    half = float(scale >> 1)
    fraction = (offsets + 0.5) / _OFFSET_LEVELS
    # grid -> octree input frame (world * point_scale) -> world
    positions = (cells + fraction - half) / (float(ray_scale) * scale * float(point_scale))
    return positions, colors.to(torch.float32) / 255.0


class OccupancyCache:
    """Small per-process LRU cache for mesh-derived occupancy files."""

    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self._items: OrderedDict[Path, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    def load(
        self,
        path: str | Path,
        *,
        query_depth: int,
        ray_scale: float,
        point_scale: float,
        max_cells: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        resolved = Path(path).resolve()
        if resolved in self._items:
            self._items.move_to_end(resolved)
            positions, colors = self._items[resolved]
            return positions.clone(), colors.clone()
        positions, colors = occupancy_points_world(
            resolved,
            query_depth=query_depth,
            ray_scale=ray_scale,
            point_scale=point_scale,
            max_cells=max_cells,
        )
        if self.capacity > 0:
            self._items[resolved] = (positions, colors)
            self._items.move_to_end(resolved)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return positions.clone(), colors.clone()
