from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping) and "model_dict" in payload:
        payload = payload["model_dict"]
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("Checkpoint does not contain a model state_dict")
    return payload


def load_model_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    state = dict(_state_dict(payload))
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
