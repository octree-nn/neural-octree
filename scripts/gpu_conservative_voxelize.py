"""Conservative triangle/AABB voxelization kernels used by build_mesh_occupancy."""
from __future__ import annotations
import torch

def _axis_separated(v: torch.Tensor, axis: torch.Tensor, half: torch.Tensor,
                    eps: float) -> torch.Tensor:
    """Separating-axis test with the box centred at the origin.

    v: (M,3,3) vertices relative to the cell centre, axis: (M,3), half: scalar.
    Returns True where the axis proves separation (so the pair can be dropped).
    """
    proj = (v * axis[:, None, :]).sum(-1)                      # (M,3)
    radius = half * axis.abs().sum(-1)                         # (M,)
    return (proj.min(1).values > radius + eps) | (proj.max(1).values < -radius - eps)


def _subdivide_large_triangles(tri: torch.Tensor, resolution: int,
                               cell_budget: int, max_rounds: int = 10,
                               max_triangles: int = 12_000_000) -> torch.Tensor:
    """Split triangles whose AABB exceeds ``cell_budget`` cells into four children.

    A triangle is exactly tiled by its three edge-midpoint children, so the union of
    the children covers the same surface and the conservative cell set is unchanged.
    This is what makes meshes built from a handful of huge triangles tractable: a
    triangle spanning the whole ``resolution^3`` grid has an AABB of ``resolution**3``
    cells, but after a few rounds its children have small AABBs, cutting the
    candidate-pair count by two to three orders of magnitude.

    Each split halves the AABB volume of a triangle, so its separating-axis cost also
    halves; the recursion stops at ``cell_budget`` cells, at ``max_rounds``, or when
    the triangle count would pass ``max_triangles``.
    """
    for _ in range(max_rounds):
        lo = torch.floor(tri.amin(dim=1)).clamp_(0, resolution - 1).long()
        hi = torch.floor(tri.amax(dim=1)).clamp_(0, resolution - 1).long()
        size = (hi - lo + 1).prod(dim=1)
        big = size > cell_budget
        if not bool(big.any()):
            break
        keep = tri[~big]
        block = tri[big]
        if keep.shape[0] + 4 * block.shape[0] > max_triangles:
            break
        m01 = (block[:, 0] + block[:, 1]) * 0.5
        m12 = (block[:, 1] + block[:, 2]) * 0.5
        m20 = (block[:, 2] + block[:, 0]) * 0.5
        children = torch.stack(
            [
                torch.stack([block[:, 0], m01, m20], dim=1),
                torch.stack([m01, block[:, 1], m12], dim=1),
                torch.stack([m20, m12, block[:, 2]], dim=1),
                torch.stack([m01, m12, m20], dim=1),
            ],
            dim=1,
        ).reshape(-1, 3, 3)
        tri = torch.cat([keep, children], dim=0)
    return tri


def _rasterise_block(tri: torch.Tensor, eps: float, resolution: int) -> torch.Tensor:
    """Conservative cells for one block of triangles (kept pairs only)."""
    lo = torch.floor(tri.amin(dim=1)).clamp_(0, resolution - 1).long()
    hi = torch.floor(tri.amax(dim=1)).clamp_(0, resolution - 1).long()
    ext = hi - lo + 1
    size = ext.prod(dim=1)
    total = int(size.sum())
    tri_idx = torch.repeat_interleave(torch.arange(tri.shape[0], device=tri.device), size)
    starts = torch.repeat_interleave(size.cumsum(0) - size, size)
    offset = torch.arange(total, device=tri.device) - starts
    sx, sy, sz = ext[tri_idx, 0], ext[tri_idx, 1], ext[tri_idx, 2]
    cells = torch.stack([lo[tri_idx, 0] + offset % sx,
                         lo[tri_idx, 1] + (offset // sx) % sy,
                         lo[tri_idx, 2] + offset // (sx * sy)], dim=1)
    half = 0.5
    separated = torch.zeros(cells.shape[0], dtype=torch.bool, device=cells.device)
    centre = cells.to(tri.dtype) + 0.5
    v = tri[tri_idx] - centre[:, None, :]
    for a in range(3):
        axis = torch.zeros_like(v[:, 0])
        axis[:, a] = 1.0
        separated |= _axis_separated(v, axis, half, eps)
    v0, v1, v2 = v[:, 0], v[:, 1], v[:, 2]
    e0, e1, e2 = v1 - v0, v2 - v1, v0 - v2
    normal = torch.cross(e0, e1, dim=1)
    separated |= _axis_separated(v, normal, half, eps)
    for i in range(3):
        box_axis = torch.zeros_like(v[:, 0])
        box_axis[:, i] = 1.0
        for edge in (e0, e1, e2):
            separated |= _axis_separated(v, torch.cross(edge, box_axis, dim=1), half, eps)
    return cells[~separated]


def conservative_surface_voxels(vertices: torch.Tensor, faces: torch.Tensor,
                                resolution: int, *, eps: float = 1.0e-4,
                                max_pairs: int = 40_000_000,
                                chunk: int = 2_000_000,
                                max_triangles: int = 12_000_000,
                                subdivide_cells: int = 256) -> torch.Tensor:
    """Return the (N,3) integer cells of a `resolution^3` grid touched by the mesh.

    `vertices` must already live in the grid frame [0, resolution].

    Two guards keep this usable on real-world meshes:

    * triangles whose AABB is larger than ``subdivide_cells`` are split into four
      children first (exact, see :func:`_subdivide_large_triangles`);
    * the surviving candidates are rasterised in blocks of at most ``chunk`` pairs,
      so the peak device memory no longer scales with the whole mesh.
    """
    tri = vertices[faces]                                      # (F,3,3)
    if subdivide_cells > 0:
        tri = _subdivide_large_triangles(tri, resolution, subdivide_cells,
                                         max_triangles=max_triangles)

    lo = torch.floor(tri.amin(dim=1)).clamp_(0, resolution - 1).long()
    hi = torch.floor(tri.amax(dim=1)).clamp_(0, resolution - 1).long()
    size = (hi - lo + 1).prod(dim=1)
    total = int(size.sum())
    if total > max_pairs:
        raise RuntimeError(f"too many candidate pairs: {total}")

    kept: list[torch.Tensor] = []
    triangles = int(tri.shape[0])
    # Block boundaries chosen so each block holds at most `chunk` candidate pairs.
    # Computed from a cumulative sum rather than a per-triangle Python loop: meshes
    # reach tens of millions of faces and that loop dominated the whole build.
    cumulative = torch.cumsum(size, dim=0)
    if total > chunk:
        edges = torch.arange(chunk, total, chunk, device=size.device)
        bounds = torch.searchsorted(cumulative, edges, right=True).tolist()
    else:
        bounds = []
    bounds = sorted({0, *bounds, triangles})
    for start, end in zip(bounds, bounds[1:]):
        if end <= start:
            continue
        kept.append(_rasterise_block(tri[start:end], eps, resolution))
    if not kept:
        return torch.zeros((0, 3), dtype=torch.long, device=vertices.device)
    return torch.unique(torch.cat(kept), dim=0)

