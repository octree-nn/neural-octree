from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class IntersectionStage:
    position: Tensor
    normal: Tensor
    color: Tensor
    hit_probability: Tensor

    def as_tuple(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.position, self.normal, self.color, self.hit_probability


@dataclass(frozen=True)
class IntersectionPrediction:
    stages: tuple[IntersectionStage, ...]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("IntersectionPrediction requires at least one stage")

    @property
    def final(self) -> IntersectionStage:
        return self.stages[-1]
