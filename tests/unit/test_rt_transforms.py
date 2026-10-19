from __future__ import annotations

import math

import pytest
import torch

from neuoct.rendering_rt import (
    Z_UP_TO_Y_UP,
    OctreeTransform,
    SimilarityTransform,
)


def test_octree_mapping_matches_audited_formula() -> None:
    transform = OctreeTransform(depth=9, point_scale=0.99, ray_scale=0.5)
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 0.5], [-0.25, 0.75, 1.5]])
    expected = points * (0.99 * 0.5 * 2**9) + 2**8
    assert torch.allclose(transform.object_to_octree(points), expected, atol=1.0e-4)
    assert transform.scale == 512
    assert math.isclose(transform.linear, 0.99 * 0.5 * 512, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(transform.offset, 256.0)


def test_octree_round_trip_for_points_directions_normals_and_t() -> None:
    transform = OctreeTransform(depth=7, point_scale=0.9, ray_scale=0.4)
    points = torch.randn(16, 3)
    round_trip = transform.octree_to_object(transform.object_to_octree(points))
    assert torch.allclose(round_trip, points, atol=1.0e-5)

    directions = torch.nn.functional.normalize(torch.randn(16, 3), dim=-1)
    mapped = transform.object_to_octree_directions(directions * 5.0)
    assert torch.allclose(mapped, directions, atol=1.0e-6)
    assert torch.allclose(mapped.norm(dim=-1), torch.ones(16), atol=1.0e-5)

    normals = torch.nn.functional.normalize(torch.randn(16, 3), dim=-1)
    mapped_normals = transform.object_to_octree_normals(normals)
    assert torch.allclose(mapped_normals, normals, atol=1.0e-6)
    assert torch.allclose(mapped_normals.norm(dim=-1), torch.ones(16), atol=1.0e-5)


def test_octree_t_scales_linearly() -> None:
    transform = OctreeTransform(depth=8, point_scale=0.99, ray_scale=0.5)
    t = torch.tensor([0.0, 1.0, 2.5])
    assert torch.allclose(transform.object_to_octree_t(t), t * transform.linear)
    assert torch.allclose(transform.octree_to_object_t(transform.object_to_octree_t(t)), t)


def test_similarity_validation_rejects_bad_transforms() -> None:
    with pytest.raises(ValueError):
        SimilarityTransform(scale=0.0)
    with pytest.raises(ValueError):
        SimilarityTransform(scale=-1.0)
    with pytest.raises(ValueError):
        # Reflection.
        SimilarityTransform(rotation=torch.diag(torch.tensor([1.0, 1.0, -1.0])))
    with pytest.raises(ValueError):
        # Not orthonormal.
        SimilarityTransform(rotation=torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]))


def test_similarity_round_trip_and_composition() -> None:
    transform = SimilarityTransform.z_up_to_y_up(
        scale=0.5, translation=torch.tensor([1.0, 2.0, 3.0])
    )
    points = torch.randn(8, 3)
    assert torch.allclose(transform.inverse_points(transform.transform_points(points)), points, atol=1.0e-5)
    composed = transform.compose(transform.inverse())
    assert torch.allclose(composed.matrix, torch.eye(4), atol=1.0e-5)


def test_similarity_maps_z_up_to_y_up() -> None:
    transform = SimilarityTransform.z_up_to_y_up()
    assert torch.allclose(
        transform.rotation, torch.tensor(Z_UP_TO_Y_UP, dtype=torch.float32)
    )
    direction = torch.tensor([[0.0, 0.0, 1.0]])
    assert torch.allclose(transform.transform_directions(direction), torch.tensor([[0.0, 1.0, 0.0]]), atol=1.0e-6)
    # Normals follow the same proper rotation (uniform scale does not flip them).
    assert torch.allclose(transform.transform_normals(direction), torch.tensor([[0.0, 1.0, 0.0]]), atol=1.0e-6)
    assert torch.allclose(transform.inverse_directions(transform.transform_directions(direction)), direction, atol=1.0e-6)


def test_octree_transform_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError):
        OctreeTransform(depth=0)
    with pytest.raises(ValueError):
        OctreeTransform(depth=4, point_scale=0.0)
