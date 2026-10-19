from __future__ import annotations

import numpy as np
import pytest

from neuoct.rendering_rt import TriangleBVH, embree_status


def _two_triangles() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.5],
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    return vertices, faces


def test_moller_trumbore_nearest_hit_and_barycentrics() -> None:
    vertices, faces = _two_triangles()
    bvh = TriangleBVH(vertices, faces)
    result = bvh.intersect([[0.25, 0.25, 1.0]], [[0.0, 0.0, -1.0]], t_min=0.0, t_max=np.inf)
    assert bool(result.hit[0])
    assert result.t[0] == pytest.approx(0.5, abs=1.0e-9)
    assert result.u[0] == pytest.approx(0.25, abs=1.0e-9)
    assert result.v[0] == pytest.approx(0.25, abs=1.0e-9)
    assert result.triangle_index[0] == 1
    # The status must name a recognised accelerator.  ``embree`` is the label
    # ``make_accelerator``/``accelerator_name`` report for the native backend
    # (``embreex`` is the Python package that provides it); ``numpy`` is the
    # deterministic fallback this reference BVH always implements.
    assert embree_status()["backend"] in ("numpy", "embree", "embreex")


def test_miss_and_clipping() -> None:
    vertices, faces = _two_triangles()
    bvh = TriangleBVH(vertices, faces)
    miss = bvh.intersect([[2.0, 2.0, 1.0]], [[0.0, 0.0, -1.0]])
    assert not bool(miss.hit[0])
    assert np.isinf(miss.t[0])
    clipped = bvh.intersect([[0.25, 0.25, 1.0]], [[0.0, 0.0, -1.0]], t_min=0.0, t_max=0.4)
    assert not bool(clipped.hit[0])
    near = bvh.intersect([[0.25, 0.25, 1.0]], [[0.0, 0.0, -1.0]], t_min=0.6, t_max=np.inf)
    assert bool(near.hit[0]) and near.triangle_index[0] == 0


def test_backface_and_parallel_rays() -> None:
    vertices, faces = _two_triangles()
    bvh = TriangleBVH(vertices, faces)
    backface = bvh.intersect([[0.25, 0.25, -1.0]], [[0.0, 0.0, 1.0]])
    assert bool(backface.hit[0])
    assert backface.triangle_index[0] == 0
    parallel = bvh.intersect([[2.0, 2.0, 0.25]], [[1.0, 0.0, 0.0]])
    assert not bool(parallel.hit[0])


def test_chunking_and_leaf_size_do_not_change_results() -> None:
    vertices, faces = _two_triangles()
    rng = np.random.default_rng(0)
    origins = np.concatenate(
        [rng.uniform(-0.2, 1.2, size=(128, 2)), rng.uniform(0.1, 2.0, size=(128, 1))], axis=1
    )
    directions = np.tile(np.array([[0.0, 0.0, -1.0]]), (128, 1))
    reference = TriangleBVH(vertices, faces, leaf_size=4).intersect(
        origins, directions, ray_chunk_size=4096
    )
    chunked = TriangleBVH(vertices, faces, leaf_size=4).intersect(
        origins, directions, ray_chunk_size=7
    )
    assert np.array_equal(reference.hit, chunked.hit)
    assert np.allclose(reference.t[reference.hit], chunked.t[reference.hit])
    assert np.array_equal(reference.triangle_index, chunked.triangle_index)

    small_leaves = TriangleBVH(vertices, faces, leaf_size=1).intersect(origins, directions)
    assert np.array_equal(reference.hit, small_leaves.hit)
    assert np.array_equal(reference.triangle_index, small_leaves.triangle_index)
    assert np.allclose(reference.u[reference.hit], small_leaves.u[reference.hit])


def test_split_is_deterministic() -> None:
    vertices, faces = _two_triangles()
    first = TriangleBVH(vertices, faces, leaf_size=1)
    second = TriangleBVH(vertices, faces, leaf_size=1)
    assert np.array_equal(first.node_bounds_min, second.node_bounds_min)
    assert np.array_equal(first.node_bounds_max, second.node_bounds_max)
    assert np.array_equal(first.node_left, second.node_left)
    assert np.array_equal(first.order, second.order)


def test_build_validates_inputs() -> None:
    with pytest.raises(ValueError):
        TriangleBVH(np.zeros((3, 3)), np.zeros((0, 3), dtype=np.int64))
    with pytest.raises(ValueError):
        TriangleBVH(np.zeros((3, 3)), np.array([[0, 1, 9]], dtype=np.int64))
