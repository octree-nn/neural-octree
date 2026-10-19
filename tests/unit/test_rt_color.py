from __future__ import annotations

import pytest
import torch

from neuoct.rendering_rt import (
    aces,
    linear_to_display,
    linear_to_srgb,
    reinhard,
    srgb_to_linear,
    tonemap,
)


def test_srgb_to_linear_known_values() -> None:
    assert torch.allclose(srgb_to_linear(torch.tensor(0.0)), torch.tensor(0.0))
    # Below the piecewise cutoff the curve is linear.
    assert torch.allclose(srgb_to_linear(torch.tensor(0.04)), torch.tensor(0.04 / 12.92))
    # Exact 0.5 -> 0.21404114...
    expected = ((0.5 + 0.055) / 1.055) ** 2.4
    assert torch.allclose(srgb_to_linear(torch.tensor(0.5)), torch.tensor(expected), atol=1.0e-6)
    assert torch.allclose(srgb_to_linear(torch.tensor(1.0)), torch.tensor(1.0), atol=1.0e-6)


def test_linear_to_srgb_is_exact_inverse() -> None:
    values = torch.linspace(0.0, 1.0, 257)
    round_trip = linear_to_srgb(srgb_to_linear(values))
    assert torch.allclose(round_trip, values, atol=1.0e-5)
    assert torch.allclose(srgb_to_linear(linear_to_srgb(values)), values, atol=1.0e-5)


def test_srgb_functions_reject_non_positive_pow_domain() -> None:
    # Negative inputs must not produce NaNs.
    values = torch.tensor([-0.5, -0.1])
    assert torch.isfinite(srgb_to_linear(values)).all()
    assert torch.isfinite(linear_to_srgb(values)).all()


def test_reinhard_properties() -> None:
    assert torch.allclose(reinhard(torch.tensor(0.0)), torch.tensor(0.0))
    assert torch.allclose(reinhard(torch.tensor(1.0)), torch.tensor(0.5))
    assert torch.allclose(reinhard(torch.tensor(3.0)), torch.tensor(0.75))
    values = torch.linspace(0.0, 10.0, 64)
    assert bool((reinhard(values)[1:] >= reinhard(values)[:-1]).all())


def test_aces_is_finite_and_bounded() -> None:
    values = torch.tensor([0.0, 0.18, 1.0, 8.0, 100.0])
    mapped = aces(values)
    assert torch.isfinite(mapped).all()
    assert bool((mapped >= 0.0).all()) and bool((mapped <= 1.0).all())
    assert mapped[0].item() == pytest.approx(0.0, abs=1.0e-6)


def test_tonemap_dispatch_and_errors() -> None:
    values = torch.tensor([0.5])
    assert torch.allclose(tonemap(values, "none"), values)
    assert torch.allclose(tonemap(values, "clamp"), values)
    with pytest.raises(ValueError):
        tonemap(values, "does-not-exist")


def test_linear_to_display_applies_exposure_before_tonemap() -> None:
    value = torch.tensor([0.25])
    encoded = linear_to_display(value, mode="reinhard", exposure=2.0, encoded=True)
    expected = linear_to_srgb(reinhard(value * 2.0))
    assert torch.allclose(encoded, expected, atol=1.0e-6)
