from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def image_psnr(prediction: Tensor, target: Tensor) -> Tensor:
    mse = (prediction - target).square().mean()
    return -10.0 * torch.log10(mse + 1.0e-8)


def masked_psnr(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    expanded = mask.unsqueeze(-1).expand_as(prediction)
    if not bool(expanded.any()):
        return torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    mse = (prediction[expanded] - target[expanded]).square().mean()
    return -10.0 * torch.log10(mse + 1.0e-8)


def hit_classification_metrics(prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
    prediction = prediction.bool()
    target = target.bool()
    true_positive = (prediction & target).sum().float()
    false_positive = (prediction & ~target).sum().float()
    false_negative = (~prediction & target).sum().float()
    precision = true_positive / (true_positive + false_positive + 1.0e-8)
    recall = true_positive / (true_positive + false_negative + 1.0e-8)
    iou = true_positive / (true_positive + false_positive + false_negative + 1.0e-8)
    return {"hit_precision": precision, "hit_recall": recall, "hit_iou": iou}


def hit_topology_metrics(
    predicted_hit: Tensor,
    target_hit: Tensor,
    *,
    neighborhood: int = 5,
) -> dict[str, Tensor]:
    """Measure visually salient interior holes, remote floaters and boundaries."""
    if neighborhood <= 0 or neighborhood % 2 == 0:
        raise ValueError("neighborhood must be a positive odd integer")
    predicted = predicted_hit.bool()
    target = target_hit.bool()
    if predicted.ndim == 2:
        predicted = predicted.unsqueeze(0)
        target = target.unsqueeze(0)
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("hit maps must have matching (H, W) or (B, H, W) shapes")
    predicted_float = predicted.float().unsqueeze(1)
    target_float = target.float().unsqueeze(1)
    target_interior = (
        F.avg_pool2d(target_float, kernel_size=3, stride=1, padding=1) > 0.999
    ).squeeze(1)
    target_neighborhood = F.max_pool2d(
        target_float,
        kernel_size=neighborhood,
        stride=1,
        padding=neighborhood // 2,
    ).squeeze(1) > 0
    holes = target_interior & ~predicted
    floaters = predicted & ~target_neighborhood
    hole_rate = holes.sum().float() / target_interior.sum().clamp_min(1)
    floater_rate = floaters.sum().float() / predicted.sum().clamp_min(1)

    predicted_eroded = (
        -F.max_pool2d(-predicted_float, 3, stride=1, padding=1) > 0.999
    ).squeeze(1)
    target_eroded = (
        -F.max_pool2d(-target_float, 3, stride=1, padding=1) > 0.999
    ).squeeze(1)
    predicted_boundary = predicted & ~predicted_eroded
    target_boundary = target & ~target_eroded
    predicted_boundary_neighborhood = F.max_pool2d(
        predicted_boundary.float().unsqueeze(1), 3, stride=1, padding=1
    ).squeeze(1) > 0
    target_boundary_neighborhood = F.max_pool2d(
        target_boundary.float().unsqueeze(1), 3, stride=1, padding=1
    ).squeeze(1) > 0
    boundary_precision = (
        (predicted_boundary & target_boundary_neighborhood).sum().float()
        / predicted_boundary.sum().clamp_min(1)
    )
    boundary_recall = (
        (target_boundary & predicted_boundary_neighborhood).sum().float()
        / target_boundary.sum().clamp_min(1)
    )
    boundary_f1 = 2 * boundary_precision * boundary_recall / (
        boundary_precision + boundary_recall
    ).clamp_min(1.0e-8)
    return {
        "interior_hole_rate": hole_rate,
        "remote_floater_rate": floater_rate,
        "hit_boundary_f1": boundary_f1,
    }


def color_detail_metrics(
    predicted_color: Tensor,
    target_color: Tensor,
    target_hit: Tensor,
    *,
    detail_fraction: float = 0.25,
) -> dict[str, Tensor]:
    """Measure texture-edge fidelity and uncommon/high-chroma material colors."""
    if not 0.0 < detail_fraction <= 1.0:
        raise ValueError("detail_fraction must be in (0, 1]")
    target_hit = target_hit.bool()
    if not bool(target_hit.any()):
        zero = torch.zeros((), device=target_color.device)
        return {
            "color_gradient_mae": zero,
            "high_detail_color_psnr": zero,
            "rare_color_psnr": zero,
        }
    predicted = predicted_color.float()
    target = target_color.float()
    detail_score = torch.zeros_like(target_hit, dtype=torch.float32)
    gradient_numerator = torch.zeros((), device=target.device)
    gradient_count = torch.zeros((), device=target.device)

    dx_valid = target_hit[:, 1:] & target_hit[:, :-1]
    dx_target = target[:, 1:] - target[:, :-1]
    dx_error = ((predicted[:, 1:] - predicted[:, :-1]) - dx_target).abs().mean(dim=-1)
    if bool(dx_valid.any()):
        gradient_numerator += dx_error[dx_valid].sum()
        gradient_count += dx_valid.sum()
    dx_score = dx_target.abs().mean(dim=-1) * dx_valid
    detail_score[:, 1:] += dx_score
    detail_score[:, :-1] += dx_score

    dy_valid = target_hit[1:] & target_hit[:-1]
    dy_target = target[1:] - target[:-1]
    dy_error = ((predicted[1:] - predicted[:-1]) - dy_target).abs().mean(dim=-1)
    if bool(dy_valid.any()):
        gradient_numerator += dy_error[dy_valid].sum()
        gradient_count += dy_valid.sum()
    dy_score = dy_target.abs().mean(dim=-1) * dy_valid
    detail_score[1:] += dy_score
    detail_score[:-1] += dy_score
    gradient_mae = gradient_numerator / gradient_count.clamp_min(1.0)

    foreground_detail = detail_score[target_hit]
    detail_threshold = torch.quantile(foreground_detail, 1.0 - detail_fraction)
    high_detail = target_hit & (detail_score >= detail_threshold)
    detail_mse = (predicted[high_detail] - target[high_detail]).square().mean()

    foreground = target[target_hit]
    center = foreground.median(dim=0).values
    color_distance = (target - center).abs().mean(dim=-1)
    rare_threshold = torch.quantile(color_distance[target_hit], 1.0 - detail_fraction)
    rare = target_hit & (color_distance >= rare_threshold)
    rare_mse = (predicted[rare] - target[rare]).square().mean()
    return {
        "color_gradient_mae": gradient_mae,
        "high_detail_color_psnr": -10.0 * torch.log10(detail_mse + 1.0e-8),
        "rare_color_psnr": -10.0 * torch.log10(rare_mse + 1.0e-8),
    }


def geometry_noise_metrics(
    predicted_normal: Tensor,
    target_normal: Tensor,
    predicted_hit: Tensor,
    target_hit: Tensor,
    *,
    normal_outlier_degrees: float = 20.0,
) -> dict[str, Tensor]:
    """Measure visually salient interior holes and foreground normal outliers.

    Hit IoU is dominated by the many correct pixels in a silhouette.  This
    diagnostic separately measures target-foreground pixels whose full 3x3
    neighborhood is also foreground, so isolated interior misses are not
    hidden by the boundary.  Normal statistics are evaluated only where both
    target and prediction report a hit; misses remain visible in the interior
    recall metric instead of being converted to an arbitrary normal error.
    """
    predicted_hit = predicted_hit.bool()
    target_hit = target_hit.bool()
    target_float = target_hit.float().unsqueeze(0).unsqueeze(0)
    neighborhood = F.avg_pool2d(target_float, kernel_size=3, stride=1, padding=1)
    interior = target_hit & (neighborhood[0, 0] >= 1.0 - 1.0e-6)
    interior_count = interior.sum().float()
    interior_recall = (predicted_hit & interior).sum().float() / (interior_count + 1.0e-8)

    overlap = predicted_hit & target_hit
    overlap_count = overlap.sum().float()
    predicted = F.normalize(predicted_normal.float(), dim=-1, eps=1.0e-8)
    target = F.normalize(target_normal.float(), dim=-1, eps=1.0e-8)
    cosine = (predicted * target).sum(dim=-1).clamp(-1.0, 1.0)
    angles = torch.acos(cosine) / torch.pi * 180.0
    angle_mean = angles[overlap].sum() / (overlap_count + 1.0e-8)
    outlier_rate = (
        ((angles > float(normal_outlier_degrees)) & overlap).sum().float()
        / (overlap_count + 1.0e-8)
    )
    return {
        "interior_hit_recall": interior_recall,
        "masked_normal_angle_error_deg": angle_mean,
        "normal_outlier_20deg_rate": outlier_rate,
    }


def normal_detail_metrics(
    predicted_normal: Tensor,
    target_normal: Tensor,
    predicted_hit: Tensor,
    target_hit: Tensor,
    *,
    detail_fraction: float = 0.25,
) -> dict[str, Tensor]:
    """Measure normal fidelity specifically around target high-frequency detail.

    Detail is ranked by the sum of valid horizontal/vertical target-normal
    differences at each foreground pixel. Gradient error is measured on edges
    where both endpoints are visible in target and prediction. The high-detail
    angle is conditioned on a predicted hit, while a separate recall exposes
    detail pixels that the renderer misses entirely.
    """
    if not 0.0 < detail_fraction <= 1.0:
        raise ValueError("detail_fraction must be in (0, 1]")
    predicted_hit = predicted_hit.bool()
    target_hit = target_hit.bool()
    predicted = F.normalize(predicted_normal.float(), dim=-1, eps=1.0e-8)
    target = F.normalize(target_normal.float(), dim=-1, eps=1.0e-8)
    detail_score = torch.zeros_like(target_hit, dtype=torch.float32)

    target_dx_valid = target_hit[:, 1:] & target_hit[:, :-1]
    target_dx = (target[:, 1:] - target[:, :-1]).norm(dim=-1)
    detail_score[:, 1:] += target_dx * target_dx_valid
    detail_score[:, :-1] += target_dx * target_dx_valid
    target_dy_valid = target_hit[1:] & target_hit[:-1]
    target_dy = (target[1:] - target[:-1]).norm(dim=-1)
    detail_score[1:] += target_dy * target_dy_valid
    detail_score[:-1] += target_dy * target_dy_valid

    gradient_numerator = torch.zeros((), device=target.device)
    gradient_count = torch.zeros((), device=target.device)
    predicted_dx_valid = predicted_hit[:, 1:] & predicted_hit[:, :-1]
    dx_valid = target_dx_valid & predicted_dx_valid
    if bool(dx_valid.any()):
        dx_error = (
            (predicted[:, 1:] - predicted[:, :-1])
            - (target[:, 1:] - target[:, :-1])
        ).abs().mean(dim=-1)
        gradient_numerator += dx_error[dx_valid].sum()
        gradient_count += dx_valid.sum()
    predicted_dy_valid = predicted_hit[1:] & predicted_hit[:-1]
    dy_valid = target_dy_valid & predicted_dy_valid
    if bool(dy_valid.any()):
        dy_error = (
            (predicted[1:] - predicted[:-1])
            - (target[1:] - target[:-1])
        ).abs().mean(dim=-1)
        gradient_numerator += dy_error[dy_valid].sum()
        gradient_count += dy_valid.sum()
    gradient_mae = gradient_numerator / gradient_count.clamp_min(1.0)

    if not bool(target_hit.any()):
        zero = torch.zeros((), device=target.device)
        return {
            "normal_gradient_mae": gradient_mae,
            "high_detail_hit_recall": zero,
            "high_detail_normal_angle_deg": zero,
            "high_detail_normal_psnr": zero,
        }
    foreground_scores = detail_score[target_hit]
    threshold = torch.quantile(foreground_scores, 1.0 - detail_fraction)
    high_detail = target_hit & (detail_score >= threshold)
    high_detail_count = high_detail.sum().float()
    high_detail_overlap = high_detail & predicted_hit
    overlap_count = high_detail_overlap.sum().float()
    high_detail_recall = overlap_count / high_detail_count.clamp_min(1.0)
    cosine = (predicted * target).sum(dim=-1).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.acos(cosine))
    angle = angles[high_detail_overlap].sum() / overlap_count.clamp_min(1.0)
    encoded_error = (
        ((predicted + 1.0) * 0.5 - (target + 1.0) * 0.5).square().mean(dim=-1)
    )
    mse = encoded_error[high_detail_overlap].sum() / overlap_count.clamp_min(1.0)
    psnr = -10.0 * torch.log10(mse + 1.0e-8)
    return {
        "normal_gradient_mae": gradient_mae,
        "high_detail_hit_recall": high_detail_recall,
        "high_detail_normal_angle_deg": angle,
        "high_detail_normal_psnr": psnr,
    }
