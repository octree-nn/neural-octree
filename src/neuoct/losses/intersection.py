from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neuoct.models.outputs import IntersectionStage


@dataclass(frozen=True)
class IntersectionTarget:
    position: Tensor
    normal: Tensor
    color: Tensor
    hit: Tensor
    ray_pin: Tensor | None = None
    ray_pout: Tensor | None = None


class PNCHLoss(nn.Module):
    """Legacy-compatible position, normal, relative color and hit loss."""

    def __init__(
        self,
        *,
        include_color: bool = True,
        positive_geometry: bool = False,
        hit_loss_mode: str = "bce",
        focal_gamma: float = 2.0,
        hit_positive_weight: float = 1.0,
        hit_term_weight: float = 1.0,
        normal_cosine_weight: float = 0.0,
        color_loss_mode: str = "relative_mse",
        color_weight: float = 2.0,
    ) -> None:
        super().__init__()
        if hit_loss_mode not in ("bce", "balanced_bce", "balanced_focal"):
            raise ValueError(
                "hit_loss_mode must be 'bce', 'balanced_bce' or 'balanced_focal'"
            )
        if focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        if hit_positive_weight <= 0:
            raise ValueError("hit_positive_weight must be positive")
        if hit_term_weight < 0:
            raise ValueError("hit_term_weight must be non-negative")
        if normal_cosine_weight < 0:
            raise ValueError("normal_cosine_weight must be non-negative")
        if color_loss_mode not in ("relative_mse", "mse", "charbonnier"):
            raise ValueError(
                "color_loss_mode must be 'relative_mse', 'mse' or 'charbonnier'"
            )
        if color_weight < 0:
            raise ValueError("color_weight must be non-negative")
        self.include_color = bool(include_color)
        self.positive_geometry = bool(positive_geometry)
        self.hit_loss_mode = hit_loss_mode
        self.focal_gamma = float(focal_gamma)
        self.hit_positive_weight = float(hit_positive_weight)
        self.hit_term_weight = float(hit_term_weight)
        self.normal_cosine_weight = float(normal_cosine_weight)
        self.color_loss_mode = color_loss_mode
        self.color_weight = float(color_weight)

    def _hit_loss(self, probability: Tensor, mask: Tensor) -> Tensor:
        # BCE's CUDA kernel asserts when its input is outside [0, 1].  A
        # sigmoid output can only violate that contract through NaN/Inf (or a
        # tiny numerical overshoot), and the asynchronous CUDA assertion hides
        # the actual offending stage.  Return a NaN sentinel before calling
        # BCE so the solver's synchronized non-finite-step handling can log and
        # skip the update instead of aborting the whole DDP job.
        probability = probability.float()
        if not bool(torch.isfinite(probability).all()):
            return probability.sum() * 0.0
        probability = probability.clamp(1.0e-6, 1.0 - 1.0e-6)
        if self.hit_loss_mode == "bce":
            element_loss = F.binary_cross_entropy(probability, mask, reduction="none")
            element_weight = torch.where(
                mask.bool(),
                torch.full_like(element_loss, self.hit_positive_weight),
                torch.ones_like(element_loss),
            )
            return (element_loss * element_weight).sum() / element_weight.sum().clamp_min(
                1.0e-8
            )
        element_loss = F.binary_cross_entropy(probability, mask, reduction="none")
        if self.hit_loss_mode == "balanced_focal":
            correct_probability = torch.where(mask.bool(), probability, 1.0 - probability)
            element_loss = element_loss * (1.0 - correct_probability).pow(self.focal_gamma)
        positive = mask.bool()
        negative = ~positive
        class_losses = []
        if bool(positive.any()):
            class_losses.append(element_loss[positive].mean())
        if bool(negative.any()):
            class_losses.append(element_loss[negative].mean())
        if not class_losses:
            return element_loss.mean()
        if bool(positive.any()) and bool(negative.any()):
            positive_loss = element_loss[positive].mean()
            negative_loss = element_loss[negative].mean()
            weight = self.hit_positive_weight
            return (weight * positive_loss + negative_loss) / (weight + 1.0)
        return class_losses[0]

    @staticmethod
    def _weighted_l1(prediction: Tensor, target: Tensor, weights: Tensor) -> Tensor:
        error = (prediction - target).abs().mean(dim=-1)
        return (error * weights).sum() / weights.sum().clamp_min(1.0e-8)

    @staticmethod
    def _weighted_mse(prediction: Tensor, target: Tensor, weights: Tensor) -> Tensor:
        error = (prediction - target).square().mean(dim=-1)
        return (error * weights).sum() / weights.sum().clamp_min(1.0e-8)

    @staticmethod
    def _weighted_charbonnier(
        prediction: Tensor, target: Tensor, weights: Tensor, eps: float = 1.0e-3
    ) -> Tensor:
        error = torch.sqrt((prediction - target).square() + eps * eps) - eps
        error = error.mean(dim=-1)
        return (error * weights).sum() / weights.sum().clamp_min(1.0e-8)

    def forward(
        self,
        prediction: IntersectionStage,
        target: IntersectionTarget,
        sample_weights: Tensor | None = None,
        color_sample_weights: Tensor | None = None,
    ) -> Tensor:
        mask = target.hit.float()
        mask_vector = mask.unsqueeze(-1)
        positive = mask.bool()
        weights = (
            torch.ones_like(mask)
            if sample_weights is None
            else sample_weights.to(device=mask.device, dtype=mask.dtype)
        )
        if weights.shape != mask.shape:
            raise ValueError(
                f"sample_weights shape {tuple(weights.shape)} does not match {tuple(mask.shape)}"
            )
        color_weights = (
            weights
            if color_sample_weights is None
            else color_sample_weights.to(device=mask.device, dtype=mask.dtype)
        )
        if color_weights.shape != mask.shape:
            raise ValueError(
                "color_sample_weights shape "
                f"{tuple(color_weights.shape)} does not match {tuple(mask.shape)}"
            )
        if self.positive_geometry and bool(positive.any()):
            positive_weights = weights[positive]
            position_loss = self._weighted_l1(
                prediction.position[positive], target.position[positive], positive_weights
            )
            normal_loss = self._weighted_l1(
                prediction.normal[positive], target.normal[positive], positive_weights
            )
        elif self.positive_geometry:
            position_loss = prediction.position.sum() * 0.0
            normal_loss = prediction.normal.sum() * 0.0
        else:
            position_loss = F.l1_loss(
                prediction.position * mask_vector, target.position * mask_vector
            )
            normal_loss = F.l1_loss(
                prediction.normal * mask_vector, target.normal * mask_vector
            )
        hit_loss = self._hit_loss(prediction.hit_probability, mask)
        loss = position_loss + normal_loss + self.hit_term_weight * hit_loss
        if self.normal_cosine_weight > 0 and bool(positive.any()):
            cosine = F.cosine_similarity(
                prediction.normal[positive], target.normal[positive], dim=-1
            )
            cosine_error = 1.0 - cosine
            positive_weights = weights[positive]
            cosine_loss = (cosine_error * positive_weights).sum() / positive_weights.sum().clamp_min(
                1.0e-8
            )
            loss = loss + self.normal_cosine_weight * cosine_loss
        if self.include_color and bool(positive.any()):
            positive_color_weights = color_weights[positive]
            predicted_color = prediction.color[positive]
            target_color = target.color[positive]
            if self.color_loss_mode == "relative_mse":
                color_numerator = self._weighted_mse(
                    predicted_color, target_color, positive_color_weights
                )
                color_denominator = self._weighted_mse(
                    target_color, torch.zeros_like(target_color), positive_color_weights
                )
                color_loss = color_numerator / (color_denominator + 1.0e-8)
            elif self.color_loss_mode == "mse":
                color_loss = self._weighted_mse(
                    predicted_color, target_color, positive_color_weights
                )
            else:
                color_loss = self._weighted_charbonnier(
                    predicted_color, target_color, positive_color_weights
                )
            loss = loss + self.color_weight * color_loss
        return loss
