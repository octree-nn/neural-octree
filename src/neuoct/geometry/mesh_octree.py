"""Build an ocnn octree from a triangle mesh (GPU conservative rasterization).

The mesh must already be in the pipeline frame -- export it with
``scripts/objaverse/blender_render.py --dump-mesh`` so no registration is needed.
Every grid cell that any triangle intersects becomes an occupied cell, which
removes the coverage gaps the point-cloud octree has (that point cloud is
back-projected from 8 views, so self-occluded surface is missing).

The octree input convention matters: ocnn's ``Octree.build_octree`` normalises its
input from ``[-1, 1]`` to ``[0, 2**depth]`` (``octree.py``, ``bbmin=-1, bbmax=1``),
while ``search_xyzb`` takes coordinates in ``[0, 2**query_depth)``.  A depth-``d``
cell centre therefore corresponds to the octree-space value
``(cell + 0.5 - 2**(d-1)) / (ray_scale * 2**d)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ocnn.octree import Points

from .octree_adapter import NeuOctOctree

__all__ = [
    "conservative_surface_voxels",
    "load_mesh_npz",
    "build_mesh_octree",
]


def _axis_separated(v: torch.Tensor, axis: torch.Tensor, half: float,
                    eps: float) -> torch.Tensor:
    """Separating-axis test with the box centred at the origin.

    ``v`` is (M, 3, 3) holding the triangle vertices relative to the cell centre.
    Returns True where the axis proves separation (so the pair can be dropped).
    """
    proj = (v * axis[:, None, :]).sum(-1)
    radius = half * axis.abs().sum(-1)
    return (proj.min(1).values > radius + eps) | (proj.max(1).values < -radius - eps)


def conservative_surface_voxels(vertices: torch.Tensor, faces: torch.Tensor,
                                resolution: int, *, eps: float = 1.0e-4,
                                max_pairs: int = 40_000_000,
                                chunk: int = 4_000_000) -> torch.Tensor:
    """Integer cells of a ``resolution**3`` grid touched by the mesh.

    ``vertices`` must already live in the grid frame ``[0, resolution]``.
    Implements the same guarantee as kaolin's ``unbatched_mesh_to_spc``
    (validated to agree with it to within 0-0.016 % of the cell count).
    """
    triangle = vertices[faces]
    lo = torch.floor(triangle.amin(dim=1)).clamp_(0, resolution - 1).long()
    hi = torch.floor(triangle.amax(dim=1)).clamp_(0, resolution - 1).long()
    ext = hi - lo + 1
    size = ext.prod(dim=1)
    total = int(size.sum())
    if total > max_pairs:
        raise RuntimeError(f"too many candidate pairs: {total}")

    tri_idx = torch.repeat_interleave(torch.arange(triangle.shape[0], device=triangle.device), size)
    starts = torch.repeat_interleave(size.cumsum(0) - size, size)
    offset = torch.arange(total, device=triangle.device) - starts
    sx, sy, sz = ext[tri_idx, 0], ext[tri_idx, 1], ext[tri_idx, 2]
    cells = torch.stack([lo[tri_idx, 0] + offset % sx,
                         lo[tri_idx, 1] + (offset // sx) % sy,
                         lo[tri_idx, 2] + offset // (sx * sy)], dim=1)

    half = 0.5
    kept: list[torch.Tensor] = []
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        c = cells[start:stop]
        t = triangle[tri_idx[start:stop]]
        v = t - (c.to(t.dtype) + 0.5)[:, None, :]
        separated = torch.zeros(v.shape[0], dtype=torch.bool, device=v.device)
        for a in range(3):                                    # 3 box axes
            axis = torch.zeros_like(v[:, 0])
            axis[:, a] = 1.0
            separated |= _axis_separated(v, axis, half, eps)
        e0, e1, e2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 1], v[:, 0] - v[:, 2]
        separated |= _axis_separated(v, torch.cross(e0, e1, dim=1), half, eps)
        for i in range(3):                                    # 9 edge x box-axis
            box_axis = torch.zeros_like(v[:, 0])
            box_axis[:, i] = 1.0
            for edge in (e0, e1, e2):
                separated |= _axis_separated(v, torch.cross(edge, box_axis, dim=1), half, eps)
        kept.append(c[~separated])
    if not kept:
        return torch.zeros((0, 3), dtype=torch.long, device=vertices.device)
    return torch.unique(torch.cat(kept), dim=0)


def load_mesh_npz(path: str | Path, device: str | torch.device = "cuda"):
    payload = np.load(str(path))
    vertices = torch.from_numpy(np.asarray(payload["vertices"], dtype=np.float32)).to(device)
    faces = torch.from_numpy(np.asarray(payload["faces"], dtype=np.int64)).to(device)
    colors = None
    if "colors" in payload:
        candidate = np.asarray(payload["colors"], dtype=np.float32)
        if candidate.size and float(np.abs(candidate).max()) > 1.0e-6:
            colors = torch.from_numpy(candidate).to(device)
    return vertices, faces, colors


def build_mesh_octree(mesh_path: str | Path, *, query_depth: int, octree_depth: int = 9,
                      full_depth: int = 4, ray_scale: float = 0.5,
                      point_scale: float = 0.99, device: str | torch.device = "cuda",
                      color_source: tuple[torch.Tensor, torch.Tensor] | None = None,
                      samples_per_cell: int = 24, max_samples: int = 8_000_000):
    """Return ``(octree, cells, stats)`` for a pipeline-frame mesh.

    ``color_source`` is an optional ``(positions, colors)`` pair in the pipeline
    input frame (``world * point_scale``); per-cell colours are taken from the
    nearest source point, which matches how the point-cloud octree gets its
    features.  Without it, mesh vertex colours or a neutral grey are used.
    """
    device = torch.device(device)
    vertices, faces, mesh_colors = load_mesh_npz(mesh_path, device)
    scale = 1 << int(query_depth)
    half = scale >> 1
    world_scale = point_scale * ray_scale

    grid = vertices * world_scale * scale + half
    cells = conservative_surface_voxels(grid, faces, scale)
    if cells.shape[0] == 0:
        raise RuntimeError(f"conservative rasterisation produced no cells for {mesh_path}")

    # ocnn input space: build_octree() rescales [-1, 1] -> [0, 2**depth]
    centres = (cells.to(torch.float32) + 0.5 - half) / (ray_scale * scale)

    # ocnn stores octree.points[d] as the *mean* input point per node and the
    # L/P features are derived from it.  Feeding bare cell centres would put the
    # surface point at the cell centre, which is a large shift from training
    # (where it is the mean of the sampled surface points inside the cell).
    # So sample the mesh densely and use the per-cell mean surface point.
    cell_keys = (cells[:, 0] * scale + cells[:, 1]) * scale + cells[:, 2]
    order = torch.argsort(cell_keys)
    sorted_keys = cell_keys[order]
    positions = centres.clone()
    cell_colors = None
    if samples_per_cell > 0:
        n_samples = int(min(cells.shape[0] * samples_per_cell, max_samples))
        v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        area = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=1).norm(dim=1)
        prob = area / area.sum().clamp_min(1e-12)
        pick = torch.multinomial(prob, n_samples, replacement=True)
        r1 = torch.rand(n_samples, device=device).sqrt()
        r2 = torch.rand(n_samples, device=device)
        a, b, c = (1 - r1).unsqueeze(1), (r1 * (1 - r2)).unsqueeze(1), (r1 * r2).unsqueeze(1)
        samples = a * v0[pick] + b * v1[pick] + c * v2[pick]
        sample_grid = samples * world_scale * scale + half
        sample_cells = torch.floor(sample_grid).long().clamp_(0, scale - 1)
        sample_keys = (sample_cells[:, 0] * scale + sample_cells[:, 1]) * scale + sample_cells[:, 2]
        # keep only samples that fall in the conservative cell set
        keep = torch.isin(sample_keys, sorted_keys)
        samples, sample_keys = samples[keep], sample_keys[keep]
        if samples.shape[0] > 0:
            uniq, inverse = torch.unique(sample_keys, return_inverse=True)
            count = torch.zeros(uniq.shape[0], device=device)
            total = torch.zeros((uniq.shape[0], 3), device=device)
            count.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float32))
            total.index_add_(0, inverse, samples)
            mean = total / count.clamp_min(1.0).unsqueeze(-1)
            slot = torch.searchsorted(sorted_keys, uniq)
            # samples live in the world frame; the octree input frame is world*point_scale
            positions[order[slot]] = mean * point_scale
            if mesh_colors is not None:
                vc = mesh_colors[faces[:, 0]] * a + mesh_colors[faces[:, 1]] * b + mesh_colors[faces[:, 2]] * c
                vc = vc[keep]
                ctotal = torch.zeros((uniq.shape[0], mesh_colors.shape[1]), device=device)
                ctotal.index_add_(0, inverse, vc)
                cell_colors = torch.zeros((cells.shape[0], mesh_colors.shape[1]), device=device)
                cell_colors[order[slot]] = ctotal / count.clamp_min(1.0).unsqueeze(-1)

    if color_source is not None:
        # nearest source point per cell, via a CPU KD-tree (a GPU cdist would need
        # len(cells) x len(points) memory)
        from scipy.spatial import cKDTree

        source_positions, source_colors = color_source
        tree = cKDTree(source_positions.detach().cpu().numpy())
        _, index = tree.query(centres.detach().cpu().numpy(), k=1)
        colors = source_colors[torch.from_numpy(index).to(source_colors.device)].float()
    elif cell_colors is not None:
        colors = cell_colors
    elif mesh_colors is not None:
        # average vertex colours per cell
        counts = torch.zeros(cells.shape[0], device=device)
        colors = torch.zeros((cells.shape[0], mesh_colors.shape[1]), device=device)
        cell_of_vertex = torch.floor(
            vertices * world_scale * scale + half).long().clamp_(0, scale - 1)
        keys = (cell_of_vertex[:, 0] * scale + cell_of_vertex[:, 1]) * scale + cell_of_vertex[:, 2]
        cell_keys = (cells[:, 0] * scale + cells[:, 1]) * scale + cells[:, 2]
        lookup = {int(k): i for i, k in enumerate(cell_keys.tolist())}
        for vertex_index, key in enumerate(keys.tolist()):
            target = lookup.get(int(key))
            if target is not None:
                colors[target] += mesh_colors[vertex_index]
                counts[target] += 1.0
        colors = colors / counts.clamp_min(1.0).unsqueeze(-1)
    else:
        colors = torch.full((cells.shape[0], 3), 0.5, device=device)

    points = Points(positions.to(device), features=colors.to(device))
    octree = NeuOctOctree(depth=int(octree_depth), full_depth=int(full_depth), device=device)
    octree.build_octree(points)
    octree.construct_all_neigh()
    stats = {
        "mesh": str(mesh_path),
        "mesh_vertices": int(vertices.shape[0]),
        "mesh_faces": int(faces.shape[0]),
        "cells": int(cells.shape[0]),
        "samples_per_cell": int(samples_per_cell),
        "query_depth": int(query_depth),
        "occupied_cells_at_query_depth": int(octree.nnum_nempty[int(query_depth)]),
    }
    return octree, cells, stats
