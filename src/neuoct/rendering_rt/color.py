"""Colour-space conversion and tone mapping utilities.

The tracer keeps all shading in **linear** space and only encodes to sRGB when
an 8-bit image is written.  The two conversions here are the exact piecewise
IEC 61966-2-1 (sRGB) curves, not the common gamma-2.2 approximation, so a
render can be checked against the NeuOct colour target losslessly.
"""

from __future__ import annotations

import torch
from torch import Tensor

# IEC 61966-2-1 constants.
_SRGB_LINEAR_CUTOFF = 0.04045
_SRGB_ENCODE_CUTOFF = 0.0031308
_SRGB_LINEAR_SLOPE = 12.92
_SRGB_OFFSET = 0.055
_SRGB_SCALE = 1.055
_SRGB_GAMMA = 2.4


def srgb_to_linear(value: Tensor) -> Tensor:
    """Exact IEC sRGB -> linear conversion (element-wise)."""
    value = torch.as_tensor(value)
    low = value / _SRGB_LINEAR_SLOPE
    high = torch.pow(((value + _SRGB_OFFSET) / _SRGB_SCALE).clamp_min(0.0), _SRGB_GAMMA)
    return torch.where(value <= _SRGB_LINEAR_CUTOFF, low, high)


def linear_to_srgb(value: Tensor) -> Tensor:
    """Exact linear -> IEC sRGB conversion (element-wise)."""
    value = torch.as_tensor(value)
    low = value * _SRGB_LINEAR_SLOPE
    high = _SRGB_SCALE * torch.pow(value.clamp_min(0.0), 1.0 / _SRGB_GAMMA) - _SRGB_OFFSET
    return torch.where(value <= _SRGB_ENCODE_CUTOFF, low, high)


def reinhard(value: Tensor) -> Tensor:
    """Per-channel Reinhard tone map ``c / (1 + c)`` (maps [0, inf) -> [0, 1))."""
    value = torch.as_tensor(value)
    return value / (1.0 + value)


def aces(value: Tensor) -> Tensor:
    """Narkowicz ACES filmic approximation (clamped to [0, 1])."""
    value = torch.as_tensor(value)
    numerator = value * (2.51 * value + 0.03)
    denominator = value * (2.43 * value + 0.59) + 0.14
    return (numerator / denominator.clamp_min(1.0e-8)).clamp(0.0, 1.0)


_TONEMAPPERS = {
    "none": lambda value: value,
    "linear": lambda value: value,
    "reinhard": reinhard,
    "aces": aces,
    "clamp": lambda value: value.clamp(0.0, 1.0),
}


def tonemap(value: Tensor, mode: str = "aces") -> Tensor:
    """Apply a named tone mapper, returning linear display values."""
    try:
        mapper = _TONEMAPPERS[mode]
    except KeyError as error:
        raise ValueError(
            f"Unknown tone mapper {mode!r}; choose from {sorted(_TONEMAPPERS)}"
        ) from error
    return mapper(torch.as_tensor(value))


def linear_to_display(
    value: Tensor,
    *,
    mode: str = "aces",
    exposure: float = 1.0,
    encoded: bool = True,
) -> Tensor:
    """Full linear-HDR -> display pipeline used by :mod:`neuoct.rendering_rt.io`."""
    exposed = torch.as_tensor(value) * float(exposure)
    mapped = tonemap(exposed, mode)
    return linear_to_srgb(mapped) if encoded else mapped
