from __future__ import annotations

from dataclasses import dataclass
from math import exp

import torch
from torch import Tensor, nn
from torchmetrics.functional.image import structural_similarity_index_measure

from neuoct.models.outputs import IntersectionPrediction, IntersectionStage

from .image import (
    color_smoothness,
    edge_aware_color_gradient_loss,
    first_predicted_hit_maps,
    hard_floater_loss,
    hit_map_regularization,
    hit_tversky_loss,
    masked_color_constancy,
    masked_ssim_loss,
    multiscale_color_laplacian_loss,
    multiscale_normal_gradient_loss,
    normal_gradient_loss,
    remap_hit_mask,
    remap_rays,
    soft_hit_boundary_f1_loss,
    soft_hit_maps,
)
from .intersection import IntersectionTarget, PNCHLoss


@dataclass(frozen=True)
class ObjectiveWeights:
    ssim_normal: float = 0.3
    ssim_color: float = 1.0
    smoothness: float = 0.2
    constancy: float = 0.1
    normal_gradient: float = 0.0
    normal_multiscale: float = 0.0
    color_gradient: float = 0.0
    color_laplacian: float = 0.0
    hit_hole: float = 0.0
    hit_floater: float = 0.0
    hit_floater_hard: float = 0.0
    hit_boundary: float = 0.0
    hit_boundary_f1: float = 0.0
    hit_tversky: float = 0.0
    ray_distance: float = 0.0


@dataclass(frozen=True)
class ObjectiveResult:
    loss: Tensor
    predicted_normal: Tensor
    predicted_color: Tensor
    predicted_hit_mask: Tensor
    target_normal: Tensor
    target_color: Tensor
    target_hit_mask: Tensor
    components: dict[str, Tensor]


class IntersectionObjective(nn.Module):
    def __init__(
        self,
        weights: ObjectiveWeights | None = None,
        *,
        include_color: bool = True,
        positive_geometry: bool = False,
        hit_loss_mode: str = "bce",
        focal_gamma: float = 2.0,
        hit_positive_weight: float = 1.0,
        hit_term_weight: float = 1.0,
        normal_cosine_weight: float = 0.0,
        stage_weight_mode: str = "legacy",
        final_stage_weight: float = 0.5,
        masked_normal_ssim: bool = False,
        masked_color_ssim: bool = False,
        normal_detail_scale: float = 0.0,
        color_detail_scale: float = 0.0,
        color_loss_mode: str = "relative_mse",
        color_weight: float = 2.0,
        hit_floater_hard_fraction: float = 0.001,
        hit_boundary_f1_tolerance: int = 1,
        hit_tversky_alpha: float = 0.3,
        hit_tversky_beta: float = 0.7,
        hit_tversky_gamma: float = 1.0,
    ):
        super().__init__()
        if stage_weight_mode not in ("legacy", "final_emphasis", "fixed"):
            raise ValueError(
                "stage_weight_mode must be 'legacy', 'final_emphasis', or 'fixed'"
            )
        if not 0.0 < final_stage_weight <= 1.0:
            raise ValueError("final_stage_weight must be in (0, 1]")
        self.weights = weights or ObjectiveWeights()
        self.include_color = bool(include_color)
        self.stage_weight_mode = stage_weight_mode
        self.final_stage_weight = float(final_stage_weight)
        self.masked_normal_ssim = bool(masked_normal_ssim)
        self.masked_color_ssim = bool(masked_color_ssim)
        self.normal_detail_scale = float(normal_detail_scale)
        self.color_detail_scale = float(color_detail_scale)
        self.hit_floater_hard_fraction = float(hit_floater_hard_fraction)
        if hit_boundary_f1_tolerance < 0:
            raise ValueError("hit_boundary_f1_tolerance must be non-negative")
        self.hit_boundary_f1_tolerance = int(hit_boundary_f1_tolerance)
        if hit_tversky_alpha < 0 or hit_tversky_beta < 0:
            raise ValueError("hit_tversky alpha and beta must be non-negative")
        if hit_tversky_alpha + hit_tversky_beta <= 0:
            raise ValueError("hit_tversky alpha and beta must have a positive sum")
        if hit_tversky_gamma <= 0:
            raise ValueError("hit_tversky gamma must be positive")
        self.hit_tversky_alpha = float(hit_tversky_alpha)
        self.hit_tversky_beta = float(hit_tversky_beta)
        self.hit_tversky_gamma = float(hit_tversky_gamma)
        self.intersection_loss = PNCHLoss(
            include_color=self.include_color,
            positive_geometry=positive_geometry,
            hit_loss_mode=hit_loss_mode,
            focal_gamma=focal_gamma,
            hit_positive_weight=hit_positive_weight,
            hit_term_weight=hit_term_weight,
            normal_cosine_weight=normal_cosine_weight,
            color_loss_mode=color_loss_mode,
            color_weight=color_weight,
        )

    @staticmethod
    def _stage_weights(
        epoch: int,
        total_epochs: int,
        count: int,
        device: torch.device,
        *,
        mode: str = "legacy",
        final_stage_weight: float = 0.5,
    ) -> Tensor:
        progress = (epoch / max(total_epochs, 1) - 0.5) * 2.0 + 0.5

        def sigmoid(value: float) -> float:
            return 1.0 / (1.0 + exp(-6.0 * (value - 0.5)))

        mu = (sigmoid(progress) - 0.5) / (2.0 * sigmoid(1.0) - 1.0) + 0.5
        positions = torch.linspace(0.0, 1.0, count, device=device)
        weights = torch.exp(-0.5 * (positions - mu).square()) + 0.5
        legacy = weights / weights.sum()
        if mode == "fixed":
            return torch.full_like(legacy, 1.0 / count)
        if mode == "legacy" or count == 1:
            return legacy
        earlier = torch.arange(1, count, device=device, dtype=legacy.dtype)
        emphasized = torch.empty_like(legacy)
        emphasized[:-1] = earlier / earlier.sum() * (1.0 - final_stage_weight)
        emphasized[-1] = final_stage_weight
        progress = min(max(float(epoch) / max(float(total_epochs), 1.0), 0.0), 1.0)
        blend = min(max((progress - 0.25) / 0.5, 0.0), 1.0)
        return legacy * (1.0 - blend) + emphasized * blend

    def forward(
        self,
        prediction: IntersectionPrediction,
        target: IntersectionTarget,
        labels: Tensor,
        *,
        batch_size: int,
        height: int,
        width: int,
        epoch: int = 0,
        total_epochs: int = 1,
        training: bool = True,
        sample_weights: Tensor | None = None,
        color_sample_weights: Tensor | None = None,
    ) -> ObjectiveResult:
        device_type = target.position.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            float_prediction = IntersectionPrediction(
                tuple(
                    IntersectionStage(
                        stage.position.float(),
                        stage.normal.float(),
                        stage.color.float(),
                        stage.hit_probability.float(),
                    )
                    for stage in prediction.stages
                )
            )
            float_target = IntersectionTarget(
                target.position.float(),
                target.normal.float(),
                target.color.float(),
                target.hit,
                None if target.ray_pin is None else target.ray_pin.float(),
                None if target.ray_pout is None else target.ray_pout.float(),
            )
            return self._forward_impl(
                float_prediction,
                float_target,
                labels,
                batch_size=batch_size,
                height=height,
                width=width,
                epoch=epoch,
                total_epochs=total_epochs,
                training=training,
                sample_weights=sample_weights,
                color_sample_weights=color_sample_weights,
            )

    def _forward_impl(
        self,
        prediction: IntersectionPrediction,
        target: IntersectionTarget,
        labels: Tensor,
        *,
        batch_size: int,
        height: int,
        width: int,
        epoch: int = 0,
        total_epochs: int = 1,
        training: bool = True,
        sample_weights: Tensor | None = None,
        color_sample_weights: Tensor | None = None,
    ) -> ObjectiveResult:
        target_normal = remap_rays(
            labels,
            target.normal,
            batch_size=batch_size,
            height=height,
            width=width,
        )
        if self.include_color:
            target_color = remap_rays(
                labels,
                target.color,
                batch_size=batch_size,
                height=height,
                width=width,
            )
        else:
            target_color = torch.zeros(
                (batch_size, height, width, 3),
                device=labels.device,
                dtype=target.position.dtype,
            )
        target_hit_mask = remap_hit_mask(
            labels, target.hit, batch_size=batch_size, height=height, width=width
        )
        stages = prediction.stages if training else (prediction.final,)
        weights = self._stage_weights(
            epoch,
            total_epochs,
            len(stages),
            labels.device,
            mode=self.stage_weight_mode,
            final_stage_weight=self.final_stage_weight,
        )
        total = torch.zeros((), device=labels.device, dtype=target.position.dtype)
        components = {
            "intersection": torch.zeros_like(total),
            "ssim_normal": torch.zeros_like(total),
            "ssim_color": torch.zeros_like(total),
            "smoothness": torch.zeros_like(total),
            "constancy": torch.zeros_like(total),
            "normal_gradient": torch.zeros_like(total),
            "normal_multiscale": torch.zeros_like(total),
            "color_gradient": torch.zeros_like(total),
            "color_laplacian": torch.zeros_like(total),
            "hit_hole": torch.zeros_like(total),
            "hit_floater": torch.zeros_like(total),
            "hit_floater_hard": torch.zeros_like(total),
            "hit_boundary": torch.zeros_like(total),
            "hit_boundary_f1": torch.zeros_like(total),
            "hit_tversky": torch.zeros_like(total),
            "ray_distance": torch.zeros_like(total),
        }
        predicted_normal = predicted_color = predicted_hit_mask = None
        for stage_weight, stage in zip(weights, stages, strict=True):
            predicted_normal, predicted_color, predicted_hit_mask = first_predicted_hit_maps(
                stage,
                labels,
                batch_size=batch_size,
                height=height,
                width=width,
            )
            if self.masked_normal_ssim:
                normal_mask = target_hit_mask.unsqueeze(-1)
                predicted_ssim = (predicted_normal + 1.0) * 0.5 * normal_mask
                target_ssim = (target_normal + 1.0) * 0.5 * normal_mask
            else:
                predicted_ssim = predicted_normal
                target_ssim = target_normal
            normal_ssim = 1.0 - structural_similarity_index_measure(
                predicted_ssim.permute(0, 3, 1, 2),
                target_ssim.permute(0, 3, 1, 2),
                data_range=1.0,
            )
            intersection = self.intersection_loss(
                stage, target, sample_weights, color_sample_weights
            )
            ray_distance = torch.zeros_like(intersection)
            if (
                self.weights.ray_distance > 0
                and target.ray_pin is not None
                and target.ray_pout is not None
            ):
                segment = target.ray_pout - target.ray_pin
                segment_sq = segment.square().sum(dim=-1).clamp_min(1.0e-8)
                pred_t = ((stage.position - target.ray_pin) * segment).sum(dim=-1) / segment_sq
                target_t = ((target.position - target.ray_pin) * segment).sum(dim=-1) / segment_sq
                positive = target.hit.bool()
                if bool(positive.any()):
                    distance_weights = (
                        torch.ones_like(target_t)
                        if sample_weights is None
                        else sample_weights.to(target_t)
                    )
                    distance_weights = distance_weights[positive]
                    ray_distance = (
                        (pred_t[positive] - target_t[positive]).abs() * distance_weights
                    ).sum() / distance_weights.sum().clamp_min(1.0e-8)
            normal_gradient = normal_gradient_loss(
                predicted_normal,
                target_normal,
                target_hit_mask & predicted_hit_mask,
                detail_scale=self.normal_detail_scale,
            )
            normal_multiscale = (
                multiscale_normal_gradient_loss(
                    predicted_normal,
                    target_normal,
                    target_hit_mask & predicted_hit_mask,
                    detail_scale=self.normal_detail_scale,
                )
                if self.weights.normal_multiscale > 0
                else torch.zeros_like(intersection)
            )
            if self.include_color:
                if self.masked_color_ssim:
                    color_ssim = masked_ssim_loss(
                        predicted_color, target_color, target_hit_mask
                    )
                else:
                    color_ssim = 1.0 - structural_similarity_index_measure(
                        predicted_color.permute(0, 3, 1, 2),
                        target_color.permute(0, 3, 1, 2),
                        data_range=1.0,
                    )
                smoothness = color_smoothness(predicted_color, predicted_hit_mask)
                constancy = masked_color_constancy(
                    predicted_color, target_color, target_hit_mask
                )
                color_gradient = edge_aware_color_gradient_loss(
                    predicted_color,
                    target_color,
                    target_hit_mask & predicted_hit_mask,
                    detail_scale=self.color_detail_scale,
                )
                color_laplacian = (
                    multiscale_color_laplacian_loss(
                        predicted_color,
                        target_color,
                        target_hit_mask & predicted_hit_mask,
                    )
                    if self.weights.color_laplacian > 0
                    else torch.zeros_like(intersection)
                )
            else:
                color_ssim = torch.zeros_like(intersection)
                smoothness = torch.zeros_like(intersection)
                constancy = torch.zeros_like(intersection)
                color_gradient = torch.zeros_like(intersection)
                color_laplacian = torch.zeros_like(intersection)
            predicted_hit_probability = soft_hit_maps(
                stage.hit_probability,
                labels,
                batch_size=batch_size,
                height=height,
                width=width,
            )
            hit_hole, hit_floater, hit_boundary = hit_map_regularization(
                predicted_hit_probability, target_hit_mask
            )
            hit_tversky = (
                hit_tversky_loss(
                    predicted_hit_probability,
                    target_hit_mask,
                    alpha=self.hit_tversky_alpha,
                    beta=self.hit_tversky_beta,
                    gamma=self.hit_tversky_gamma,
                )
                if self.weights.hit_tversky > 0
                else torch.zeros_like(intersection)
            )
            hit_boundary_f1 = (
                soft_hit_boundary_f1_loss(
                    predicted_hit_probability,
                    target_hit_mask,
                    tolerance=self.hit_boundary_f1_tolerance,
                )
                if self.weights.hit_boundary_f1 > 0
                else torch.zeros_like(intersection)
            )
            hit_floater_hard = (
                hard_floater_loss(
                    predicted_hit_probability,
                    target_hit_mask,
                    fraction=self.hit_floater_hard_fraction,
                )
                if self.weights.hit_floater_hard > 0
                else torch.zeros_like(intersection)
            )
            if training and self.include_color:
                stage_loss = (
                    intersection
                    + self.weights.ssim_normal * normal_ssim
                    + self.weights.ssim_color * color_ssim
                    + self.weights.smoothness * smoothness
                    + self.weights.constancy * constancy
                    + self.weights.normal_gradient * normal_gradient
                    + self.weights.normal_multiscale * normal_multiscale
                    + self.weights.color_gradient * color_gradient
                    + self.weights.color_laplacian * color_laplacian
                    + self.weights.hit_hole * hit_hole
                    + self.weights.hit_floater * hit_floater
                    + self.weights.hit_floater_hard * hit_floater_hard
                    + self.weights.hit_boundary * hit_boundary
                    + self.weights.hit_boundary_f1 * hit_boundary_f1
                    + self.weights.hit_tversky * hit_tversky
                    + self.weights.ray_distance * ray_distance
                )
            elif self.include_color:
                stage_loss = (
                    intersection
                    + self.weights.ssim_normal * normal_ssim
                    + self.weights.ssim_color * color_ssim
                    + self.weights.normal_gradient * normal_gradient
                    + self.weights.normal_multiscale * normal_multiscale
                    + self.weights.color_gradient * color_gradient
                    + self.weights.color_laplacian * color_laplacian
                    + self.weights.hit_hole * hit_hole
                    + self.weights.hit_floater * hit_floater
                    + self.weights.hit_floater_hard * hit_floater_hard
                    + self.weights.hit_boundary * hit_boundary
                    + self.weights.hit_boundary_f1 * hit_boundary_f1
                    + self.weights.hit_tversky * hit_tversky
                    + self.weights.ray_distance * ray_distance
                )
            else:
                stage_loss = (
                    intersection
                    + self.weights.ssim_normal * normal_ssim
                    + self.weights.normal_gradient * normal_gradient
                    + self.weights.normal_multiscale * normal_multiscale
                    + self.weights.hit_hole * hit_hole
                    + self.weights.hit_floater * hit_floater
                    + self.weights.hit_floater_hard * hit_floater_hard
                    + self.weights.hit_boundary * hit_boundary
                    + self.weights.hit_boundary_f1 * hit_boundary_f1
                    + self.weights.hit_tversky * hit_tversky
                    + self.weights.ray_distance * ray_distance
                )
            total = total + stage_weight * stage_loss
            components["intersection"] += stage_weight * intersection
            components["ssim_normal"] += stage_weight * normal_ssim
            components["ssim_color"] += stage_weight * color_ssim
            components["smoothness"] += stage_weight * smoothness
            components["constancy"] += stage_weight * constancy
            components["normal_gradient"] += stage_weight * normal_gradient
            components["normal_multiscale"] += stage_weight * normal_multiscale
            components["color_gradient"] += stage_weight * color_gradient
            components["color_laplacian"] += stage_weight * color_laplacian
            components["hit_hole"] += stage_weight * hit_hole
            components["hit_floater"] += stage_weight * hit_floater
            components["hit_floater_hard"] += stage_weight * hit_floater_hard
            components["hit_boundary"] += stage_weight * hit_boundary
            components["hit_boundary_f1"] += stage_weight * hit_boundary_f1
            components["hit_tversky"] += stage_weight * hit_tversky
            components["ray_distance"] += stage_weight * ray_distance
        return ObjectiveResult(
            loss=total,
            predicted_normal=predicted_normal,
            predicted_color=predicted_color,
            predicted_hit_mask=predicted_hit_mask,
            target_normal=target_normal,
            target_color=target_color,
            target_hit_mask=target_hit_mask,
            components=components,
        )
