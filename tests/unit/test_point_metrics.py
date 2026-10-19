import torch

from neuoct.metrics import symmetric_chamfer_distance


def test_chunked_chamfer_matches_full_cdist():
    torch.manual_seed(2)
    points_a = torch.rand(11, 3)
    points_b = torch.rand(7, 3)
    distances = torch.cdist(points_a, points_b).square()
    expected = distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean()
    actual = symmetric_chamfer_distance(points_a, points_b, source_chunk=3, target_chunk=4)
    torch.testing.assert_close(actual, expected)
