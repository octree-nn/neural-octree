"""The Embree backend and its NumPy fallback must agree hit for hit.

``make_accelerator`` prefers Embree and falls back to the deterministic NumPy
BVH, so the two are exercised on the same geometry here.  The Embree comparison
is skipped where ``embreex`` is not installed (the CPU-only host and the unit
test environment), while the fallback contract itself is always checked.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuoct.rendering_rt import (
    EmbreeBVH,
    TriangleBVH,
    accelerator_name,
    embree_status,
    make_accelerator,
)


def _icosphere(subdivisions: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """A closed triangulated sphere; enough triangles to have a real hierarchy."""
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    return (
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
    )


def _rays(count: int = 512, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    origins = np.tile(np.array([0.0, 0.0, 4.0]), (count, 1)) + rng.normal(0.0, 0.05, (count, 3))
    return origins, directions


def test_status_reports_the_real_backend() -> None:
    status = embree_status()
    assert status["backend"] in ("embree", "numpy")
    assert isinstance(status["embree_available"], bool)
    assert isinstance(status["embree_usable"], bool)
    # The label must match what make_accelerator actually returns.
    vertices, faces = _icosphere()
    accelerator = make_accelerator(vertices, faces)
    assert accelerator_name(accelerator) == status["backend"]


def test_numpy_fallback_is_selectable() -> None:
    vertices, faces = _icosphere()
    accelerator = make_accelerator(vertices, faces, prefer_embree=False)
    assert isinstance(accelerator, TriangleBVH)
    assert accelerator_name(accelerator) == "numpy"
    origins, directions = _rays(64)
    result = accelerator.intersect(origins, directions)
    assert result.hit.any()


def test_embree_matches_numpy_hit_for_hit() -> None:
    if not embree_status()["embree_usable"]:
        pytest.skip("embreex/trimesh embree bridge is unavailable on this host")
    vertices, faces = _icosphere(subdivisions=3)
    embree = EmbreeBVH(vertices, faces)
    numpy = TriangleBVH(vertices, faces, leaf_size=4)
    origins, directions = _rays(1024)
    reference = numpy.intersect(origins, directions)
    accelerated = embree.intersect(origins, directions)
    assert np.array_equal(reference.hit, accelerated.hit)
    both = reference.hit & accelerated.hit
    assert both.any()
    assert np.allclose(reference.t[both], accelerated.t[both], atol=1e-9)
    # The same surface point, hence the same triangle, for every shared hit.
    assert np.array_equal(reference.triangle_index[both], accelerated.triangle_index[both])
    assert np.allclose(reference.u[both], accelerated.u[both], atol=1e-9)
    assert np.allclose(reference.v[both], accelerated.v[both], atol=1e-9)


def test_embree_respects_the_t_window() -> None:
    if not embree_status()["embree_usable"]:
        pytest.skip("embreex/trimesh embree bridge is unavailable on this host")
    vertices, faces = _icosphere()
    embree = EmbreeBVH(vertices, faces)
    numpy = TriangleBVH(vertices, faces, leaf_size=4)
    origins, directions = _rays(256)
    # Clip to a shell inside the sphere: some rays keep a hit, some lose it.
    t_min, t_max = 3.0, 4.6
    reference = numpy.intersect(origins, directions, t_min=t_min, t_max=t_max)
    accelerated = embree.intersect(origins, directions, t_min=t_min, t_max=t_max)
    assert np.array_equal(reference.hit, accelerated.hit)
    both = reference.hit
    if both.any():
        assert np.all(accelerated.t[both] >= t_min - 1e-9)
        assert np.all(accelerated.t[both] <= t_max + 1e-9)
        assert np.allclose(reference.t[both], accelerated.t[both], atol=1e-9)
    # A window entirely in front of the sphere must miss everywhere.
    empty = embree.intersect(origins, directions, t_min=0.0, t_max=1.0)
    assert not empty.hit.any()
    assert np.all(np.isinf(empty.t))
    assert np.all(empty.triangle_index == -1)


def test_embree_rejects_an_empty_mesh() -> None:
    if not embree_status()["embree_usable"]:
        pytest.skip("embreex/trimesh embree bridge is unavailable on this host")
    with pytest.raises(ValueError):
        EmbreeBVH(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
