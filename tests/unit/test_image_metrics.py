import torch

from neuoct.metrics import (
    color_detail_metrics,
    geometry_noise_metrics,
    hit_topology_metrics,
    normal_detail_metrics,
)


def test_geometry_noise_metrics_separate_interior_holes_and_normal_outliers():
    target_hit = torch.ones((5, 5), dtype=torch.bool)
    predicted_hit = target_hit.clone()
    predicted_hit[2, 2] = False
    target_normal = torch.zeros((5, 5, 3))
    target_normal[..., 2] = 1.0
    predicted_normal = target_normal.clone()
    predicted_normal[1, 1] = torch.tensor([1.0, 0.0, 0.0])

    metrics = geometry_noise_metrics(
        predicted_normal,
        target_normal,
        predicted_hit,
        target_hit,
    )

    torch.testing.assert_close(metrics["interior_hit_recall"], torch.tensor(8.0 / 9.0))
    torch.testing.assert_close(
        metrics["masked_normal_angle_error_deg"], torch.tensor(90.0 / 24.0)
    )
    torch.testing.assert_close(metrics["normal_outlier_20deg_rate"], torch.tensor(1.0 / 24.0))


def test_normal_detail_metrics_focus_on_high_gradient_foreground():
    target_hit = torch.ones((4, 4), dtype=torch.bool)
    predicted_hit = target_hit.clone()
    predicted_hit[1, 2] = False
    target_normal = torch.zeros((4, 4, 3))
    target_normal[..., 2] = 1.0
    target_normal[:, 2:] = torch.tensor([1.0, 0.0, 0.0])
    predicted_normal = target_normal.clone()
    predicted_normal[2, 1] = torch.tensor([0.0, 1.0, 0.0])

    metrics = normal_detail_metrics(
        predicted_normal,
        target_normal,
        predicted_hit,
        target_hit,
        detail_fraction=0.5,
    )

    assert metrics["normal_gradient_mae"] > 0
    assert 0 < metrics["high_detail_hit_recall"] < 1
    assert metrics["high_detail_normal_angle_deg"] >= 0
    assert torch.isfinite(metrics["high_detail_normal_psnr"])


def test_normal_detail_metrics_reject_invalid_fraction():
    normal = torch.zeros((2, 2, 3))
    hit = torch.ones((2, 2), dtype=torch.bool)
    try:
        normal_detail_metrics(normal, normal, hit, hit, detail_fraction=0.0)
    except ValueError as exc:
        assert "detail_fraction" in str(exc)
    else:
        raise AssertionError("invalid detail fraction should fail fast")


def test_hit_topology_metrics_expose_small_holes_and_remote_floaters():
    target = torch.zeros((9, 9), dtype=torch.bool)
    target[2:7, 2:7] = True
    prediction = target.clone()
    prediction[4, 4] = False
    prediction[0, 0] = True
    metrics = hit_topology_metrics(prediction, target, neighborhood=3)
    assert metrics["interior_hole_rate"] > 0
    assert metrics["remote_floater_rate"] > 0
    assert 0 < metrics["hit_boundary_f1"] < 1


def test_color_detail_metrics_focus_on_texture_and_rare_color():
    target = torch.full((4, 4, 3), 0.2)
    target[:, 2:] = torch.tensor([1.0, 0.0, 1.0])
    prediction = target.clone()
    prediction[:, 2:] = 0.5
    metrics = color_detail_metrics(
        prediction, target, torch.ones((4, 4), dtype=torch.bool), detail_fraction=0.5
    )
    assert metrics["color_gradient_mae"] > 0
    assert metrics["high_detail_color_psnr"] < 20
    assert metrics["rare_color_psnr"] < 20
