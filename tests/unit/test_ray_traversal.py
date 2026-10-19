import torch

from neuoct.geometry.ray_traversal import grouped_interleave_indices, safe_inverse


def test_grouped_interleave_matches_documented_order():
    labels = torch.tensor([0, 0, 1, 1, 1, 3, 4, 4])
    indices = grouped_interleave_indices(labels)
    assert indices.tolist() == [0, 2, 5, 6, 1, 3, 7, 4]


def test_safe_inverse_is_finite_and_preserves_sign():
    values = torch.tensor([-2.0, -0.0, 0.0, 4.0])
    inverse = safe_inverse(values)
    assert torch.isfinite(inverse).all()
    assert inverse[0] < 0 and inverse[-1] > 0
