from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from neuoct.models.outputs import IntersectionStage


def remap_rays(
    labels: Tensor,
    values: Tensor,
    *,
    batch_size: int,
    height: int,
    width: int,
    fill: float = 0.0,
) -> Tensor:
    channels = values.shape[-1]
    image = torch.full(
        (batch_size * height * width, channels),
        fill,
        dtype=values.dtype,
        device=values.device,
    )
    image[labels.long()] = values
    return image.reshape(batch_size, height, width, channels)


def remap_hit_mask(
    labels: Tensor, hits: Tensor, *, batch_size: int, height: int, width: int
) -> Tensor:
    mask = torch.zeros(batch_size * height * width, dtype=torch.bool, device=hits.device)
    hit_labels = labels[hits.bool()].long()
    mask[hit_labels] = True
    return mask.reshape(batch_size, height, width)


def first_predicted_hit_maps(
    stage: IntersectionStage,
    labels: Tensor,
    *,
    batch_size: int,
    height: int,
    width: int,
) -> tuple[Tensor, Tensor, Tensor]:
    hit_indices = torch.nonzero(stage.hit_probability > 0.5, as_tuple=False).flatten()
    output_shape = (batch_size, height, width, 3)
    if hit_indices.numel() == 0:
        normal = torch.zeros(output_shape, device=stage.normal.device, dtype=stage.normal.dtype)
        color = torch.zeros(output_shape, device=stage.color.device, dtype=stage.color.dtype)
        mask = torch.zeros((batch_size, height, width), device=stage.color.device, dtype=torch.bool)
        return normal, color, mask
    hit_labels = labels[hit_indices].long()
    order = torch.argsort(hit_labels, stable=True)
    sorted_labels = hit_labels[order]
    first = torch.cat(
        [
            torch.ones(1, dtype=torch.bool, device=labels.device),
            sorted_labels[1:] != sorted_labels[:-1],
        ]
    )
    selected_indices = hit_indices[order[first]]
    selected_labels = labels[selected_indices].long()
    normal = remap_rays(
        selected_labels,
        stage.normal[selected_indices],
        batch_size=batch_size,
        height=height,
        width=width,
    )
    color = remap_rays(
        selected_labels,
        stage.color[selected_indices],
        batch_size=batch_size,
        height=height,
        width=width,
    )
    mask = torch.zeros(batch_size * height * width, dtype=torch.bool, device=labels.device)
    mask[selected_labels] = True
    return normal, color, mask.reshape(batch_size, height, width)


def color_smoothness(color: Tensor, mask: Tensor | None = None) -> Tensor:
    image = color.permute(0, 3, 1, 2)
    channels = image.shape[1]
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).reshape(1, 1, 3, 3)
    grad_x = F.conv2d(image, sobel_x.expand(channels, 1, 3, 3), padding=1, groups=channels)
    grad_y = F.conv2d(image, sobel_y.expand(channels, 1, 3, 3), padding=1, groups=channels)
    magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-8)
    if mask is None:
        return magnitude.mean()
    expanded = mask[:, None].expand_as(magnitude)
    count = expanded.sum()
    if int(count) == 0:
        return magnitude.sum() * 0.0
    return (magnitude * expanded).sum() / count


def color_constancy(prediction: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(prediction.mean(dim=(0, 1, 2)), target.mean(dim=(0, 1, 2)))


def masked_color_constancy(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Match foreground color statistics without dilution from the background."""
    valid = mask.bool().unsqueeze(-1).expand_as(prediction)
    count = valid.sum(dim=(0, 1, 2)).clamp_min(1)
    predicted_mean = (prediction * valid).sum(dim=(0, 1, 2)) / count
    target_mean = (target * valid).sum(dim=(0, 1, 2)) / count
    return F.mse_loss(predicted_mean, target_mean)


def masked_ssim_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Compute SSIM over the foreground support and ignore background-only windows."""
    from torchmetrics.functional.image import structural_similarity_index_measure

    mask_float = mask.float().unsqueeze(1)
    prediction_chw = prediction.permute(0, 3, 1, 2)
    target_chw = target.permute(0, 3, 1, 2)
    _, similarity = structural_similarity_index_measure(
        prediction_chw,
        target_chw,
        data_range=1.0,
        reduction="none",
        return_full_image=True,
    )
    valid_windows = F.avg_pool2d(mask_float, kernel_size=11, stride=1, padding=5)
    valid_windows = (valid_windows > 0.5).expand_as(similarity)
    if not bool(valid_windows.any()):
        return similarity.sum() * 0.0
    return 1.0 - similarity[valid_windows].mean()


def edge_aware_color_gradient_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    detail_scale: float = 0.0,
) -> Tensor:
    """Preserve foreground texture/color edges while avoiding silhouette gradients."""

    def axis_loss(
        predicted_delta: Tensor, target_delta: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        error = (predicted_delta - target_delta).abs().mean(dim=-1)
        detail = target_delta.abs().mean(dim=-1)
        if detail_scale > 0 and bool(valid.any()):
            reference = torch.quantile(detail[valid], 0.9).clamp_min(1.0e-4)
            weights = 1.0 + float(detail_scale) * (detail / reference).clamp(0.0, 1.0)
        else:
            weights = torch.ones_like(error)
        return (error * weights * valid).sum(), (weights * valid).sum()

    mask = mask.bool()
    dx_num, dx_den = axis_loss(
        prediction[:, :, 1:] - prediction[:, :, :-1],
        target[:, :, 1:] - target[:, :, :-1],
        mask[:, :, 1:] & mask[:, :, :-1],
    )
    dy_num, dy_den = axis_loss(
        prediction[:, 1:] - prediction[:, :-1],
        target[:, 1:] - target[:, :-1],
        mask[:, 1:] & mask[:, :-1],
    )
    return (dx_num + dy_num) / (dx_den + dy_den).clamp_min(1.0e-8)


def soft_hit_maps(
    hit_probability: Tensor,
    labels: Tensor,
    *,
    batch_size: int,
    height: int,
    width: int,
) -> Tensor:
    """Aggregate pair probabilities with the same any-hit semantics as rendering."""
    pixel_count = batch_size * height * width
    labels = labels.long()
    probability = hit_probability.float()
    aggregated = torch.zeros(pixel_count, device=labels.device, dtype=probability.dtype)
    aggregated.scatter_reduce_(
        0,
        labels,
        probability,
        reduce="amax",
        include_self=True,
    )
    return aggregated.reshape(batch_size, height, width)


def hit_map_regularization(
    predicted_probability: Tensor,
    target_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Penalize silhouette holes, isolated floaters and boundary disagreement."""
    probability = predicted_probability.float().clamp(1.0e-6, 1.0 - 1.0e-6)
    target = target_mask.float()
    interior = F.avg_pool2d(target.unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0.999
    exterior = F.max_pool2d(target.unsqueeze(1), 3, stride=1, padding=1).squeeze(1) < 0.001
    hole_loss = (
        -probability[interior].log().mean()
        if bool(interior.any())
        else probability.sum() * 0.0
    )
    floater_loss = (
        -(1.0 - probability[exterior]).log().mean()
        if bool(exterior.any())
        else probability.sum() * 0.0
    )
    target_dx = target[:, :, 1:] - target[:, :, :-1]
    target_dy = target[:, 1:] - target[:, :-1]
    predicted_dx = probability[:, :, 1:] - probability[:, :, :-1]
    predicted_dy = probability[:, 1:] - probability[:, :-1]
    boundary_loss = 0.5 * (
        (predicted_dx - target_dx).abs().mean()
        + (predicted_dy - target_dy).abs().mean()
    )
    return hole_loss, floater_loss, boundary_loss


def hit_tversky_loss(
    predicted_probability: Tensor,
    target_mask: Tensor,
    *,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 1.0,
) -> Tensor:
    """Soft silhouette overlap loss with explicit false-negative control.

    ``beta > alpha`` makes missing foreground pixels more expensive than
    background false positives. This protects thin structures and holes while
    the separate hard-floater term controls rare remote positives.
    """
    if alpha < 0.0 or beta < 0.0 or alpha + beta <= 0.0:
        raise ValueError("alpha and beta must be non-negative with a positive sum")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    probability = predicted_probability.float().clamp(1.0e-6, 1.0 - 1.0e-6)
    target = target_mask.float()
    reduce_dims = tuple(range(1, probability.ndim))
    true_positive = (probability * target).sum(dim=reduce_dims)
    false_positive = (probability * (1.0 - target)).sum(dim=reduce_dims)
    false_negative = ((1.0 - probability) * target).sum(dim=reduce_dims)
    smooth = 1.0e-6
    score = (true_positive + smooth) / (
        true_positive + float(alpha) * false_positive + float(beta) * false_negative + smooth
    )
    return (1.0 - score).pow(float(gamma)).mean()


def soft_hit_boundary_f1_loss(
    predicted_probability: Tensor,
    target_mask: Tensor,
    *,
    tolerance: int = 1,
) -> Tensor:
    """Differentiable boundary-F1 loss for silhouette topology.

    A tolerance band makes the loss robust to one-pixel rasterization shifts,
    while boundaries introduced by interior holes or remote floaters remain
    unmatched and reduce precision. The construction mirrors the boundary-F1
    validation metric without thresholding the prediction.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    probability = predicted_probability.float().clamp(1.0e-6, 1.0 - 1.0e-6)
    target = target_mask.float()
    if probability.shape != target.shape or probability.ndim != 3:
        raise ValueError("hit maps must have matching (B, H, W) shapes")

    def inner_boundary(value: Tensor) -> Tensor:
        value_4d = value.unsqueeze(1)
        eroded = -F.max_pool2d(-value_4d, 3, stride=1, padding=1)
        return (value_4d - eroded).clamp_min(0.0).squeeze(1)

    predicted_boundary = inner_boundary(probability)
    target_boundary = inner_boundary(target)
    if tolerance > 0:
        kernel_size = 2 * tolerance + 1
        predicted_neighborhood = F.max_pool2d(
            predicted_boundary.unsqueeze(1),
            kernel_size,
            stride=1,
            padding=tolerance,
        ).squeeze(1)
        target_neighborhood = F.max_pool2d(
            target_boundary.unsqueeze(1),
            kernel_size,
            stride=1,
            padding=tolerance,
        ).squeeze(1)
    else:
        predicted_neighborhood = predicted_boundary
        target_neighborhood = target_boundary

    reduce_dims = tuple(range(1, probability.ndim))
    epsilon = 1.0e-6
    precision = (predicted_boundary * target_neighborhood).sum(dim=reduce_dims) / (
        predicted_boundary.sum(dim=reduce_dims) + epsilon
    )
    recall = (target_boundary * predicted_neighborhood).sum(dim=reduce_dims) / (
        target_boundary.sum(dim=reduce_dims) + epsilon
    )
    boundary_f1 = 2.0 * precision * recall / (precision + recall + epsilon)
    return (1.0 - boundary_f1).mean()


def hard_floater_loss(
    predicted_probability: Tensor,
    target_mask: Tensor,
    *,
    fraction: float = 0.001,
) -> Tensor:
    """Penalize the highest-confidence remote background predictions.

    Averaging over every exterior pixel makes a handful of visible floaters
    contribute almost nothing. Top-k mining keeps the loss focused on those
    rare high-confidence errors while ignoring the target boundary band. A
    Brier penalty keeps both the value and probability-space gradient bounded;
    ``-log(1-p)`` was unstable for near-certain negatives under AMP.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    probability = predicted_probability.float().clamp(1.0e-6, 1.0 - 1.0e-6)
    target = target_mask.float()
    exterior = F.max_pool2d(target.unsqueeze(1), 3, stride=1, padding=1).squeeze(1) < 0.001
    values = probability[exterior]
    if values.numel() == 0:
        return probability.sum() * 0.0
    count = max(1, int(values.numel() * float(fraction)))
    hard = values.topk(min(count, values.numel()), sorted=False).values
    return hard.square().mean()


def color_detail_weights_for_pairs(
    colors: Tensor,
    hits: Tensor,
    labels: Tensor,
    *,
    batch_size: int,
    height: int,
    width: int,
    detail_scale: float,
    rarity_scale: float,
) -> Tensor | None:
    """Upweight texture edges and rare foreground colors without changing geometry weights."""
    if detail_scale <= 0 and rarity_scale <= 0:
        return None
    dense_color = remap_rays(
        labels,
        colors.float(),
        batch_size=batch_size,
        height=height,
        width=width,
    )
    valid = remap_hit_mask(
        labels,
        hits,
        batch_size=batch_size,
        height=height,
        width=width,
    )
    dense_weights = torch.ones_like(valid, dtype=torch.float32)
    for batch_id in range(batch_size):
        batch_valid = valid[batch_id]
        if not bool(batch_valid.any()):
            continue
        if detail_scale > 0:
            score = torch.zeros_like(batch_valid, dtype=torch.float32)
            dx_valid = batch_valid[:, 1:] & batch_valid[:, :-1]
            dx = (
                dense_color[batch_id, :, 1:] - dense_color[batch_id, :, :-1]
            ).abs().mean(dim=-1) * dx_valid
            score[:, 1:] += dx
            score[:, :-1] += dx
            dy_valid = batch_valid[1:] & batch_valid[:-1]
            dy = (
                dense_color[batch_id, 1:] - dense_color[batch_id, :-1]
            ).abs().mean(dim=-1) * dy_valid
            score[1:] += dy
            score[:-1] += dy
            reference = torch.quantile(score[batch_valid], 0.9).clamp_min(1.0e-4)
            dense_weights[batch_id] += float(detail_scale) * (
                score / reference
            ).clamp(0.0, 1.0)
        if rarity_scale > 0:
            foreground = dense_color[batch_id][batch_valid]
            center = foreground.median(dim=0).values
            distance = (dense_color[batch_id] - center).abs().mean(dim=-1)
            reference = torch.quantile(distance[batch_valid], 0.9).clamp_min(1.0e-4)
            dense_weights[batch_id] += float(rarity_scale) * (
                distance / reference
            ).clamp(0.0, 1.0)
    pair_weights = dense_weights.reshape(-1)[labels.long()]
    positive = hits.bool()
    if bool(positive.any()):
        pair_weights = pair_weights / pair_weights[positive].mean().clamp_min(1.0e-6)
    return pair_weights


def normal_gradient_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    detail_scale: float = 0.0,
) -> Tensor:
    """Match foreground normal gradients, emphasizing high-curvature pixels."""
    prediction = F.normalize(prediction.float(), dim=-1, eps=1.0e-8)
    target = F.normalize(target.float(), dim=-1, eps=1.0e-8)
    mask = mask.bool()

    def axis_loss(
        predicted_delta: Tensor, target_delta: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        error = (predicted_delta - target_delta).abs().mean(dim=-1)
        detail = target_delta.norm(dim=-1)
        if detail_scale > 0 and bool(valid.any()):
            reference = detail[valid].mean().clamp_min(1.0e-6)
            weights = 1.0 + float(detail_scale) * (detail / reference).clamp(max=4.0)
        else:
            weights = torch.ones_like(error)
        numerator = (error * weights * valid).sum()
        denominator = (weights * valid).sum()
        return numerator, denominator

    dx_num, dx_den = axis_loss(
        prediction[:, :, 1:] - prediction[:, :, :-1],
        target[:, :, 1:] - target[:, :, :-1],
        mask[:, :, 1:] & mask[:, :, :-1],
    )
    dy_num, dy_den = axis_loss(
        prediction[:, 1:] - prediction[:, :-1],
        target[:, 1:] - target[:, :-1],
        mask[:, 1:] & mask[:, :-1],
    )
    denominator = dx_den + dy_den
    return (dx_num + dy_num) / denominator.clamp_min(1.0e-8)


def _masked_pyramid_downsample(
    image: Tensor,
    mask: Tensor,
    *,
    normalize_vectors: bool = False,
) -> tuple[Tensor, Tensor]:
    """Downsample an NHWC foreground image without bleeding in the background."""
    mask_float = mask.float().unsqueeze(1)
    image_chw = image.permute(0, 3, 1, 2)
    support = F.avg_pool2d(mask_float, kernel_size=2, stride=2)
    pooled = F.avg_pool2d(image_chw * mask_float, kernel_size=2, stride=2)
    pooled = pooled / support.clamp_min(1.0e-6)
    valid = support[:, 0] >= 0.75
    pooled = pooled.permute(0, 2, 3, 1)
    if normalize_vectors:
        pooled = F.normalize(pooled, dim=-1, eps=1.0e-8)
    return pooled, valid


def multiscale_normal_gradient_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    detail_scale: float = 0.0,
    levels: int = 3,
) -> Tensor:
    """Match normal variation at progressively larger image-space scales.

    The native-resolution term remains available separately through
    ``normal_gradient_loss``. This pyramid starts at half resolution so it
    adds medium- and coarse-scale curvature supervision without simply
    increasing the existing local gradient weight.
    """
    if levels < 2:
        raise ValueError("levels must be at least 2")
    prediction = F.normalize(prediction.float(), dim=-1, eps=1.0e-8)
    target = F.normalize(target.float(), dim=-1, eps=1.0e-8)
    current_mask = mask.bool()
    losses: list[Tensor] = []
    weights: list[float] = []
    for level in range(1, levels):
        if min(prediction.shape[1:3]) < 2:
            break
        prediction, predicted_valid = _masked_pyramid_downsample(
            prediction, current_mask, normalize_vectors=True
        )
        target, target_valid = _masked_pyramid_downsample(
            target, current_mask, normalize_vectors=True
        )
        current_mask = predicted_valid & target_valid
        losses.append(
            normal_gradient_loss(
                prediction,
                target,
                current_mask,
                detail_scale=detail_scale,
            )
        )
        weights.append(1.0 / float(2**level))
    if not losses:
        return prediction.sum() * 0.0
    weight_tensor = torch.tensor(weights, device=prediction.device, dtype=prediction.dtype)
    return (torch.stack(losses) * weight_tensor).sum() / weight_tensor.sum()


def multiscale_color_laplacian_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    levels: int = 3,
    eps: float = 1.0e-3,
) -> Tensor:
    """Match a masked Laplacian pyramid to preserve foreground color detail."""
    if levels < 1:
        raise ValueError("levels must be positive")
    prediction = prediction.float()
    target = target.float()
    current_mask = mask.bool()
    losses: list[Tensor] = []
    weights: list[float] = []
    for level in range(levels):
        if min(prediction.shape[1:3]) < 2:
            break
        prediction_low, prediction_valid = _masked_pyramid_downsample(
            prediction, current_mask
        )
        target_low, target_valid = _masked_pyramid_downsample(target, current_mask)
        output_size = prediction.shape[1:3]
        prediction_blur = F.interpolate(
            prediction_low.permute(0, 3, 1, 2),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        target_blur = F.interpolate(
            target_low.permute(0, 3, 1, 2),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        valid = current_mask.unsqueeze(-1).expand_as(prediction)
        if bool(valid.any()):
            difference = (prediction - prediction_blur) - (target - target_blur)
            robust = torch.sqrt(difference.square() + eps * eps) - eps
            losses.append(robust[valid].mean())
            weights.append(1.0 / float(2**level))
        prediction = prediction_low
        target = target_low
        current_mask = prediction_valid & target_valid
    if not losses:
        return prediction.sum() * 0.0
    weight_tensor = torch.tensor(weights, device=prediction.device, dtype=prediction.dtype)
    return (torch.stack(losses) * weight_tensor).sum() / weight_tensor.sum()


def normal_detail_weights_for_pairs(
    normals: Tensor,
    hits: Tensor,
    labels: Tensor,
    *,
    batch_size: int,
    height: int,
    width: int,
    scale: float,
) -> Tensor | None:
    """Map dense target-normal detail scores back to ray/voxel supervision pairs."""
    if scale <= 0:
        return None
    dense_normal = remap_rays(
        labels,
        normals.float(),
        batch_size=batch_size,
        height=height,
        width=width,
    )
    dense_normal = F.normalize(dense_normal, dim=-1, eps=1.0e-8)
    valid = remap_hit_mask(
        labels,
        hits,
        batch_size=batch_size,
        height=height,
        width=width,
    )
    score = torch.zeros_like(valid, dtype=torch.float32)
    dx_valid = valid[:, :, 1:] & valid[:, :, :-1]
    dx = (dense_normal[:, :, 1:] - dense_normal[:, :, :-1]).norm(dim=-1) * dx_valid
    score[:, :, 1:] += dx
    score[:, :, :-1] += dx
    dy_valid = valid[:, 1:] & valid[:, :-1]
    dy = (dense_normal[:, 1:] - dense_normal[:, :-1]).norm(dim=-1) * dy_valid
    score[:, 1:] += dy
    score[:, :-1] += dy
    dense_weights = torch.ones_like(score)
    for batch_id in range(batch_size):
        batch_valid = valid[batch_id]
        if bool(batch_valid.any()):
            reference = torch.quantile(score[batch_id][batch_valid], 0.9).clamp_min(1.0e-6)
            dense_weights[batch_id] += float(scale) * (
                score[batch_id] / reference
            ).clamp(0.0, 1.0)
    pair_weights = dense_weights.reshape(-1)[labels.long()]
    positive = hits.bool()
    if bool(positive.any()):
        pair_weights = pair_weights / pair_weights[positive].mean().clamp_min(1.0e-6)
    return pair_weights
