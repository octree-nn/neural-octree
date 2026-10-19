import torch

from neuoct.metrics import legacy_local_view_metrics
from neuoct.rendering import RenderOutput


def test_legacy_local_metrics_match_known_errors():
    rendered = RenderOutput(
        position=torch.tensor([[[5.0, 5.0, 5.0]]]),
        normal=torch.tensor([[[1.0, 0.0, 0.0]]]),
        color=torch.zeros(1, 1, 3),
        hit=torch.ones(1, 1, dtype=torch.bool),
        scale=10,
    )
    metrics = legacy_local_view_metrics(
        rendered,
        target_points=torch.tensor([[[0.2, 0.0, 0.0]]]),
        target_normals=torch.tensor([[[0.0, 1.0, 0.0]]]),
        target_colors=torch.ones(1, 1, 3),
        target_valid=torch.ones(1, 1, dtype=torch.bool),
        point_scale=1.0,
        quantize=False,
    )
    torch.testing.assert_close(metrics["legacy_distance_e4"], torch.tensor(1000.0))
    torch.testing.assert_close(metrics["legacy_angle_error_deg"], torch.tensor(90.0))
    torch.testing.assert_close(metrics["legacy_color_psnr"], torch.tensor(0.0))


def test_legacy_local_metrics_zero_invalid_pixels_like_original_script():
    rendered = RenderOutput(
        position=torch.tensor([[[2.5, 2.0, 2.0], [4.0, 4.0, 4.0]]]),
        normal=torch.tensor([[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]]),
        color=torch.tensor([[[0.25, 0.5, 0.75], [1.0, 1.0, 1.0]]]),
        hit=torch.tensor([[True, True]]),
        scale=4,
    )
    metrics = legacy_local_view_metrics(
        rendered,
        target_points=torch.tensor([[[0.25, 0.0, 0.0], [9.0, 9.0, 9.0]]]),
        target_normals=torch.tensor([[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]]),
        target_colors=torch.tensor([[[0.25, 0.5, 0.75], [0.0, 0.0, 0.0]]]),
        target_valid=torch.tensor([[True, False]]),
        point_scale=1.0,
        quantize=False,
    )
    torch.testing.assert_close(metrics["legacy_distance_e4"], torch.tensor(0.0), atol=1.0e-4, rtol=0)
    assert float(metrics["legacy_angle_error_deg"]) < 0.02
    assert torch.isinf(metrics["legacy_color_psnr"])


def test_local_metrics_default_to_target_foreground_only():
    rendered = RenderOutput(
        position=torch.tensor([[[2.5, 2.0, 2.0], [0.0, 0.0, 0.0]]]),
        normal=torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
        color=torch.zeros(1, 2, 3),
        hit=torch.tensor([[True, True]]),
        scale=4,
    )
    kwargs = {
        "target_points": torch.tensor([[[0.25, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        "target_normals": torch.tensor([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]),
        "target_colors": torch.zeros(1, 2, 3),
        "target_valid": torch.tensor([[True, False]]),
        "point_scale": 1.0,
        "quantize": False,
        "include_color": False,
    }
    foreground = legacy_local_view_metrics(rendered, **kwargs)
    full_image = legacy_local_view_metrics(rendered, **kwargs, foreground_only=False)
    torch.testing.assert_close(foreground["legacy_distance_e4"], torch.tensor(0.0))
    torch.testing.assert_close(foreground["legacy_angle_error_deg"], torch.tensor(0.0))
    assert float(full_image["legacy_angle_error_deg"]) > 20.0
