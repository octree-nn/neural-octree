"""Triangle accelerators: a real Embree backend plus a deterministic NumPy BVH.

Two interchangeable nearest-hit accelerators implement the same
:class:`BVHIntersection` contract:

``EmbreeBVH``
    Uses ``embreex`` through trimesh's ``ray_pyembree`` intersector.  This is
    the production backend for the real N-BVH chess scene (~1.5M triangles after
    flattening the 49 glTF instances), where a Python-level BVH traversal is far
    too slow.  The mesh is uploaded once; every query is a batched native call.
``TriangleBVH``
    A pure ``numpy`` median-split BVH with an explicit stack and Moller-Trumbore
    leaf tests.  It is the deterministic reference used by the unit tests, the
    fallback when Embree is unavailable, and the definition of the tie-breaking
    rule (nearest ``t``, then lowest triangle index).

:func:`make_accelerator` picks Embree when it is importable and falls back to
NumPy otherwise; :func:`embree_status` reports which one is in use so a render
manifest can state the backend that actually produced the image instead of
claiming one it never used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - exercised only when embreex is installed
    import embreex as _embreex  # type: ignore

    EMBREE_AVAILABLE = True
except Exception:  # pragma: no cover - the common case on CPU-only hosts
    _embreex = None
    EMBREE_AVAILABLE = False

try:  # pragma: no cover - trimesh is a hard dependency on the render host
    from trimesh.ray.ray_pyembree import RayMeshIntersector as _RayMeshIntersector
except Exception:  # pragma: no cover
    _RayMeshIntersector = None

_EMBREE_USABLE = bool(EMBREE_AVAILABLE and _RayMeshIntersector is not None)

_DET_EPS = 1.0e-12
_INV_EPS = 1.0e-10


def embree_status() -> dict[str, object]:
    """Report which accelerators are actually available on this host."""
    return {
        "embree_available": EMBREE_AVAILABLE,
        "embree_usable": _EMBREE_USABLE,
        "backend": "embree" if _EMBREE_USABLE else "numpy",
        "numpy_fallback": "always available and used for unit tests",
        "note": (
            "make_accelerator() prefers embree and falls back to the deterministic "
            "numpy BVH when embreex/trimesh's embree bridge is missing"
        ),
    }


@dataclass
class BVHIntersection:
    hit: np.ndarray
    t: np.ndarray
    u: np.ndarray
    v: np.ndarray
    triangle_index: np.ndarray

    def __len__(self) -> int:
        return int(self.hit.shape[0])


def _safe_inverse(values: np.ndarray) -> np.ndarray:
    safe = np.where(np.abs(values) < _INV_EPS, np.copysign(_INV_EPS, values), values)
    return 1.0 / safe


class TriangleBVH:
    """Deterministic median-split BVH over a triangle soup."""

    backend = "numpy"

    def __init__(
        self,
        vertices,
        faces,
        *,
        leaf_size: int = 4,
        max_depth: int | None = None,
    ) -> None:
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (V, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        if faces.shape[0] == 0:
            raise ValueError("faces must be non-empty")
        if int(faces.min()) < 0 or int(faces.max()) >= vertices.shape[0]:
            raise ValueError("face indices out of range")
        if leaf_size <= 0:
            raise ValueError("leaf_size must be positive")
        self.vertices = vertices
        self.faces = faces
        self.leaf_size = int(leaf_size)
        self.max_depth = int(max_depth) if max_depth is not None else 64
        self.triangles = vertices[faces]  # (T, 3, 3)
        self.centroids = self.triangles.mean(axis=1)
        self.triangle_bounds_min = self.triangles.min(axis=1)
        self.triangle_bounds_max = self.triangles.max(axis=1)
        self._build()

    @property
    def num_triangles(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.node_bounds_min.shape[0])

    def _build(self) -> None:
        count = self.num_triangles
        bounds_min: list[np.ndarray] = []
        bounds_max: list[np.ndarray] = []
        left: list[int] = []
        right: list[int] = []
        start: list[int] = []
        node_count: list[int] = []
        order = np.arange(count, dtype=np.int64)

        def new_node(lo: np.ndarray, hi: np.ndarray) -> int:
            index = len(bounds_min)
            bounds_min.append(lo)
            bounds_max.append(hi)
            left.append(-1)
            right.append(-1)
            start.append(0)
            node_count.append(0)
            return index

        order_work = order

        def build(begin: int, end: int, depth: int) -> int:
            lo = self.triangle_bounds_min[order_work[begin:end]].min(axis=0)
            hi = self.triangle_bounds_max[order_work[begin:end]].max(axis=0)
            node = new_node(lo, hi)
            span = end - begin
            if span <= self.leaf_size or depth >= self.max_depth:
                start[node] = begin
                node_count[node] = span
                return node
            centroids = self.centroids[order_work[begin:end]]
            extent = centroids.max(axis=0) - centroids.min(axis=0)
            axis = int(np.argmax(extent))
            local = order_work[begin:end]
            keys = self.centroids[local, axis]
            # Stable, deterministic: primary key centroid coordinate, secondary
            # key triangle index.  numpy's lexsort uses the last key first.
            sorted_local = local[np.lexsort((local, keys))]
            order_work[begin:end] = sorted_local
            if bool((extent <= 0.0).all()):
                # Degenerate split; make a leaf rather than recursing forever.
                start[node] = begin
                node_count[node] = span
                return node
            mid = begin + span // 2
            left[node] = build(begin, mid, depth + 1)
            right[node] = build(mid, end, depth + 1)
            return node

        build(0, count, 0)
        self.order = order_work
        self.node_bounds_min = np.asarray(bounds_min, dtype=np.float64)
        self.node_bounds_max = np.asarray(bounds_max, dtype=np.float64)
        self.node_left = np.asarray(left, dtype=np.int64)
        self.node_right = np.asarray(right, dtype=np.int64)
        self.node_start = np.asarray(start, dtype=np.int64)
        self.node_count = np.asarray(node_count, dtype=np.int64)

    def _bounds_for(self, begin: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        indices = self.order[begin:end]
        return (
            self.triangle_bounds_min[indices].min(axis=0),
            self.triangle_bounds_max[indices].max(axis=0),
        )

    def intersect(
        self,
        origins,
        directions,
        *,
        t_min=0.0,
        t_max=float("inf"),
        ray_chunk_size: int = 4096,
    ) -> BVHIntersection:
        origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        if origins.shape != directions.shape:
            raise ValueError("origins and directions must share shape (R, 3)")
        ray_count = origins.shape[0]
        if ray_chunk_size <= 0:
            raise ValueError("ray_chunk_size must be positive")
        t_min_array = np.broadcast_to(np.asarray(t_min, dtype=np.float64), (ray_count,))
        t_max_array = np.broadcast_to(np.asarray(t_max, dtype=np.float64), (ray_count,))
        hit = np.zeros(ray_count, dtype=bool)
        t_out = np.full(ray_count, np.inf, dtype=np.float64)
        u_out = np.zeros(ray_count, dtype=np.float64)
        v_out = np.zeros(ray_count, dtype=np.float64)
        tri_out = np.full(ray_count, -1, dtype=np.int64)
        for begin in range(0, ray_count, ray_chunk_size):
            end = min(begin + ray_chunk_size, ray_count)
            for index in range(begin, end):
                result = self._intersect_one(
                    origins[index],
                    directions[index],
                    float(t_min_array[index]),
                    float(t_max_array[index]),
                )
                if result is not None:
                    hit[index] = True
                    t_out[index], u_out[index], v_out[index], tri_out[index] = result
        return BVHIntersection(hit=hit, t=t_out, u=u_out, v=v_out, triangle_index=tri_out)

    def _intersect_one(self, origin, direction, t_min, t_max):
        inverse = _safe_inverse(direction)
        best_t = float(t_max)
        best_u = 0.0
        best_v = 0.0
        best_tri = -1
        stack = [0]
        while stack:
            node = stack.pop()
            if not self._slab_hit(node, origin, direction, inverse, t_min, best_t):
                continue
            span = int(self.node_count[node])
            if span > 0:
                begin = int(self.node_start[node])
                candidates = self.order[begin : begin + span]
                result = self._triangle_leaf(origin, direction, candidates, t_min, best_t)
                if result is not None:
                    t_hit, u_hit, v_hit, tri = result
                    if t_hit < best_t or (t_hit == best_t and (best_tri < 0 or tri < best_tri)):
                        best_t, best_u, best_v, best_tri = t_hit, u_hit, v_hit, tri
            else:
                stack.append(int(self.node_left[node]))
                stack.append(int(self.node_right[node]))
        if best_tri < 0:
            return None
        return best_t, best_u, best_v, best_tri

    def _slab_hit(self, node, origin, direction, inverse, t_min, t_max) -> bool:
        lo = (self.node_bounds_min[node] - origin) * inverse
        hi = (self.node_bounds_max[node] - origin) * inverse
        near = np.minimum(lo, hi).max()
        far = np.maximum(lo, hi).min()
        return bool(near <= far and far >= t_min and near <= t_max)

    def _triangle_leaf(self, origin, direction, candidates, t_min, t_max):
        triangles = self.triangles[candidates]
        edge1 = triangles[:, 1, :] - triangles[:, 0, :]
        edge2 = triangles[:, 2, :] - triangles[:, 0, :]
        pvec = np.cross(direction, edge2)
        det = np.einsum("ij,ij->i", edge1, pvec)
        valid = np.abs(det) > _DET_EPS
        inv_det = np.where(valid, 1.0 / np.where(valid, det, 1.0), 0.0)
        tvec = origin - triangles[:, 0, :]
        u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
        qvec = np.cross(tvec, edge1)
        v = (qvec @ direction) * inv_det
        t = np.einsum("ij,ij->i", edge2, qvec) * inv_det
        valid &= (u >= -1.0e-9) & (v >= -1.0e-9) & (u + v <= 1.0 + 1.0e-9)
        valid &= (t >= t_min) & (t <= t_max)
        if not bool(valid.any()):
            return None
        candidate_indices = np.nonzero(valid)[0]
        candidate_t = t[candidate_indices]
        candidate_tri = candidates[candidate_indices]
        # Deterministic lexicographic (t, triangle index) minimum.
        winner = np.lexsort((candidate_tri, candidate_t))[0]
        selected = candidate_indices[winner]
        return float(t[selected]), float(u[selected]), float(v[selected]), int(candidates[selected])

    def leaf_ranges(self) -> list[tuple[int, int]]:
        """Return ``(start, count)`` for every leaf; useful for diagnostics."""
        leaves = []
        for node in range(self.num_nodes):
            span = int(self.node_count[node])
            if span > 0:
                leaves.append((int(self.node_start[node]), span))
        return leaves


class EmbreeBVH:
    """Native Embree nearest-hit accelerator for a triangle soup.

    Queries go through trimesh's ``ray_pyembree`` intersector, which wraps
    ``embreex``; the mesh is uploaded once at construction.  The returned record
    is the same :class:`BVHIntersection` the NumPy BVH produces, so a scene can
    switch backends without changing a single shading path.

    Ray parameters are handled by shifting each origin to ``origin + t_min *
    direction`` before the query, which turns Embree's "first hit in front of
    the origin" into the ``[t_min, t_max]`` window every caller expects, and
    ``t`` is recomputed against the *original* vertices so nothing depends on
    the library's internal scaling.  Barycentric coordinates are recovered with
    the same Moller-Trumbore formulation the NumPy backend uses, which makes the
    two backends agree to floating-point precision on shared hits.
    """

    backend = "embree"

    def __init__(self, vertices, faces, *, leaf_size: int = 32, **_ignored) -> None:
        if not _EMBREE_USABLE:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "EmbreeBVH requires 'embreex' and trimesh's ray_pyembree bridge; "
                "install embreex or use make_accelerator(..., prefer_embree=False)"
            )
        import trimesh

        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (V, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        if faces.shape[0] == 0:
            raise ValueError("faces must be non-empty")
        if int(faces.min()) < 0 or int(faces.max()) >= vertices.shape[0]:
            raise ValueError("face indices out of range")
        self.vertices = vertices
        self.faces = faces
        self.leaf_size = int(leaf_size)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        # The scene is already in metres and well conditioned, so no
        # normalisation is needed; keeping the original frame avoids any
        # scaling round-trip in the ray parameters.
        self._intersector = _RayMeshIntersector(mesh, scale_to_box=False)
        self.triangles = vertices[faces]

    @property
    def num_triangles(self) -> int:
        return int(self.faces.shape[0])

    @property
    def num_nodes(self) -> int:
        return 0  # Embree owns its own hierarchy

    def intersect(
        self,
        origins,
        directions,
        *,
        t_min=0.0,
        t_max=float("inf"),
        ray_chunk_size: int = 262144,
    ) -> BVHIntersection:
        origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        if origins.shape != directions.shape:
            raise ValueError("origins and directions must share shape (R, 3)")
        ray_count = origins.shape[0]
        if ray_chunk_size <= 0:
            raise ValueError("ray_chunk_size must be positive")
        t_min_array = np.broadcast_to(np.asarray(t_min, dtype=np.float64), (ray_count,))
        t_max_array = np.broadcast_to(np.asarray(t_max, dtype=np.float64), (ray_count,))
        hit = np.zeros(ray_count, dtype=bool)
        t_out = np.full(ray_count, np.inf, dtype=np.float64)
        u_out = np.zeros(ray_count, dtype=np.float64)
        v_out = np.zeros(ray_count, dtype=np.float64)
        tri_out = np.full(ray_count, -1, dtype=np.int64)
        if ray_count == 0:
            return BVHIntersection(hit=hit, t=t_out, u=u_out, v=v_out, triangle_index=tri_out)
        for begin in range(0, ray_count, ray_chunk_size):
            end = min(begin + ray_chunk_size, ray_count)
            chunk_origins = origins[begin:end]
            chunk_directions = directions[begin:end]
            chunk_min = t_min_array[begin:end]
            # Shift to the t_min plane so Embree's nearest hit is the nearest
            # hit *inside* the requested window.
            shifted = chunk_origins + chunk_min[:, None] * chunk_directions
            index_tri, index_ray = self._intersects_id(shifted, chunk_directions)
            if index_ray.size == 0:
                continue
            index_tri = np.asarray(index_tri, dtype=np.int64)
            index_ray = np.asarray(index_ray, dtype=np.int64)
            triangles = self.triangles[index_tri]
            local_origin = shifted[index_ray]
            local_direction = chunk_directions[index_ray]
            edge1 = triangles[:, 1] - triangles[:, 0]
            edge2 = triangles[:, 2] - triangles[:, 0]
            pvec = np.cross(local_direction, edge2)
            det = np.einsum("ij,ij->i", edge1, pvec)
            safe = np.where(np.abs(det) < _DET_EPS, 1.0, det)
            inv_det = 1.0 / safe
            tvec = local_origin - triangles[:, 0]
            u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
            qvec = np.cross(tvec, edge1)
            v = np.einsum("ij,ij->i", qvec, local_direction) * inv_det
            local_t = np.einsum("ij,ij->i", edge2, qvec) * inv_det
            absolute_t = chunk_min[index_ray] + local_t
            keep = (absolute_t >= 0.0) & (absolute_t <= t_max_array[begin:end][index_ray])
            if not bool(keep.any()):
                continue
            rows = begin + index_ray[keep]
            hit[rows] = True
            t_out[rows] = absolute_t[keep]
            u_out[rows] = u[keep]
            v_out[rows] = v[keep]
            tri_out[rows] = index_tri[keep]
        return BVHIntersection(hit=hit, t=t_out, u=u_out, v=v_out, triangle_index=tri_out)

    def _intersects_id(self, origins: np.ndarray, directions: np.ndarray):
        """Call trimesh's intersector across its several return signatures."""
        result = self._intersector.intersects_id(
            origins, directions, multiple_hits=False, return_locations=False
        )
        if isinstance(result, tuple) and len(result) >= 2:
            return result[0], result[1]
        raise RuntimeError(  # pragma: no cover - defensive
            f"unexpected trimesh intersects_id return value: {type(result)!r}"
        )

    def leaf_ranges(self) -> list[tuple[int, int]]:  # pragma: no cover - diagnostics
        return []


def make_accelerator(vertices, faces, *, prefer_embree: bool = True, leaf_size: int = 32):
    """Build the best available accelerator, reporting nothing it cannot do."""
    if prefer_embree and _EMBREE_USABLE:
        return EmbreeBVH(vertices, faces, leaf_size=leaf_size)
    return TriangleBVH(vertices, faces, leaf_size=leaf_size)


def accelerator_name(accelerator: object) -> str:
    """Backend label for a manifest or log line."""
    return str(getattr(accelerator, "backend", "numpy"))
