from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from neuoct.rendering.output import RenderOutput


def _match_image_shape(value: Tensor, reference: Tensor, name: str) -> Tensor:
    if value.shape == reference.shape:
        return value
    if value.numel() == reference.numel():
        return value.reshape(reference.shape)
    raise ValueError(f"{name} shape {tuple(value.shape)} does not match {tuple(reference.shape)}")


def _erode_mask(mask: Tensor, iterations: int = 1) -> Tensor:
    """4-neighbour erosion, matching the project's 1-px foreground convention."""
    m = mask
    for _ in range(int(iterations)):
        m = (m & torch.roll(m, 1, -2) & torch.roll(m, -1, -2)
             & torch.roll(m, 1, -1) & torch.roll(m, -1, -1))
    return m


def legacy_local_view_metrics(
    rendered: RenderOutput,
    target_points: Tensor,
    target_normals: Tensor,
    target_colors: Tensor,
    target_valid: Tensor,
    *,
    point_scale: float = 0.99,
    ray_scale: float = 0.5,
    quantize: bool = True,
    include_color: bool = True,
    foreground_only: bool = True,
) -> dict[str, Tensor]:
    """Compute the metrics used by ``NeuOct_local/metrics.py``.

    ``target_points`` are world-space points from RGB-D unprojection. By
    default, predictions pass through the same uint8 tile encoding and
    decoding used by the old script; ``quantize=False`` is available for a
    higher-precision diagnostic. Invalid pixels follow the legacy convention
    before normalisation. By default, scores are averaged over target
    foreground pixels only; ``foreground_only=False`` restores the old
    full-image convention.
    """
    if point_scale <= 0:
        raise ValueError("point_scale must be positive")
    if ray_scale <= 0:
        raise ValueError("ray_scale must be positive")
    target_points = _match_image_shape(target_points, rendered.position, "target_points").float()
    target_normals = _match_image_shape(
        target_normals, rendered.normal, "target_normals"
    ).float()
    target_colors = _match_image_shape(target_colors, rendered.color, "target_colors").float()
    target_valid = _match_image_shape(target_valid, rendered.hit, "target_valid").bool()

    if quantize:
        tiles = rendered.uint8_tiles()
        predicted_points = tiles["position"].float() / 255.0 - 0.5
        predicted_normals = tiles["normal"].float() / 127.5 - 1.0
        predicted_colors = tiles["color"].float() / 255.0
        predicted_hit = tiles["hit"][..., 0] >= 127
    else:
        predicted_hit = rendered.hit.bool()
        predicted_points = rendered.position.float() / float(rendered.scale) - 0.5
        predicted_normals = rendered.normal.float().clone()
        predicted_colors = rendered.color.float().clone()
    predicted_invalid = ~predicted_hit
    predicted_points[predicted_invalid] = 0.0
    predicted_normals[predicted_invalid] = 1.0
    predicted_colors[predicted_invalid] = 0.0

    target_points = target_points * (float(point_scale) * float(ray_scale))
    target_normals = target_normals.clone()
    target_colors = target_colors.clone()
    target_invalid = ~target_valid
    target_points[target_invalid] = 0.0
    target_normals[target_invalid] = 1.0
    target_colors[target_invalid] = 0.0

    predicted_normals = F.normalize(predicted_normals, dim=-1)
    target_normals = F.normalize(target_normals, dim=-1)
    score_mask = target_valid if foreground_only else torch.ones_like(target_valid)
    if bool(score_mask.any()):
        point_errors = (predicted_points - target_points).square().sum(dim=-1).sqrt()
        point_distance = point_errors[score_mask].mean() * 10000.0
        cosine = (predicted_normals * target_normals).sum(dim=-1).clamp(-1.0, 1.0)
        angle_error = torch.acos(cosine[score_mask]).mean() / torch.pi * 180.0
    else:
        zero = rendered.position.sum() * 0.0
        point_distance = zero
        angle_error = zero
    # Same quantities as the VQAD / N-BVH rows: orientation-aligned normal angle
    # (arccos|cos|) and a 1-px eroded foreground mask.  Evaluated in this
    # pipeline's own (self-consistent) frame, so the geometric values are
    # directly comparable with the other rows' unified numbers.
    eroded = _erode_mask(score_mask, 1)
    if bool(eroded.any()):
        unified_distance = point_errors[eroded].mean() * 10000.0
        unified_angle = (
            torch.acos(cosine.abs().clamp(0.0, 1.0)[eroded]).mean() / torch.pi * 180.0
        )
    else:
        zero = rendered.position.sum() * 0.0
        unified_distance = zero
        unified_angle = zero
    metrics = {
        "legacy_distance_e4": point_distance,
        "legacy_angle_error_deg": angle_error,
        "unified_distance_e4": unified_distance,
        "unified_normal_angle_error_deg": unified_angle,
    }
    # Union caliber: foreground is the *prediction* foreground OR the target
    # foreground, so a floater (prediction-only) pixel is scored with its own
    # error instead of being hidden behind a pixel both sides call background.
    # This is the ``union = prediction.hit | ground_truth.hit`` protocol of the
    # ShapeNet / Objaverse comparison tables; the angle is the oriented
    # ``arccos(cos)`` there, with the sign-agnostic ``arccos|cos|`` reported
    # alongside for diagnosis.
    union = predicted_hit | target_valid
    if bool(union.any()):
        union_distance = point_errors[union].mean() * 10000.0
        union_angle = torch.acos(cosine[union]).mean() / torch.pi * 180.0
        union_angle_abs = (
            torch.acos(cosine.abs().clamp(0.0, 1.0)[union]).mean() / torch.pi * 180.0
        )
    else:
        zero = rendered.position.sum() * 0.0
        union_distance = zero
        union_angle = zero
        union_angle_abs = zero
    metrics.update(
        {
            "union_distance_e4": union_distance,
            "union_normal_angle_error_deg": union_angle,
            "union_normal_angle_abs_deg": union_angle_abs,
        }
    )
    if foreground_only:
        metrics.update(
            {
                "foreground_distance_e4": point_distance,
                "foreground_normal_angle_error_deg": angle_error,
            }
        )
    if include_color:
        color_errors = (predicted_colors - target_colors).square().mean(dim=-1)
        color_mse = (
            color_errors[score_mask].mean()
            if bool(score_mask.any())
            else rendered.color.sum() * 0.0
        )
        metrics["legacy_color_psnr"] = torch.where(
            color_mse == 0,
            torch.full_like(color_mse, torch.inf),
            -10.0 * torch.log10(color_mse),
        )
    return metrics
