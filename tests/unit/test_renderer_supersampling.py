import torch

from neuoct.rendering.output import RenderOutput
from neuoct.rendering.renderer import SparseRenderer


def test_supersampling_fills_majority_hit_and_averages_valid_geometry():
    hit = torch.tensor([[True, True], [True, False]])
    position = torch.zeros((2, 2, 3))
    position[hit] = torch.tensor([2.0, 4.0, 6.0])
    normal = torch.zeros((2, 2, 3))
    normal[hit] = torch.tensor([0.0, 0.0, 1.0])
    color = torch.ones((2, 2, 3))

    output = SparseRenderer._downsample(RenderOutput(position, normal, color, hit, 8), 2)

    assert output.hit.shape == (1, 1)
    assert output.hit.item()
    torch.testing.assert_close(output.position[0, 0], torch.tensor([2.0, 4.0, 6.0]))
    torch.testing.assert_close(output.normal[0, 0], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(output.color[0, 0], torch.ones(3))


def test_supersampling_rejects_subpixel_minority_hit():
    hit = torch.tensor([[True, False], [False, False]])
    output = SparseRenderer._downsample(
        RenderOutput(
            torch.ones((2, 2, 3)),
            torch.ones((2, 2, 3)),
            torch.zeros((2, 2, 3)),
            hit,
            8,
        ),
        2,
    )

    assert not output.hit.item()
    torch.testing.assert_close(output.position, torch.zeros_like(output.position))
    torch.testing.assert_close(output.normal, torch.zeros_like(output.normal))
