from __future__ import annotations

import math

import pytest
import torch

from neuoct.rendering_rt import (
    CounterRNG,
    cosine_hemisphere_sample,
    ggx_half_vector,
    ggx_reflection,
    orthonormal_basis,
    reflect,
    refract,
    schlick_fresnel,
    thin_walled_refract,
)


def test_reflect_matches_mirror_formula() -> None:
    direction = torch.tensor([[0.0, -1.0, 0.0], [1.0, -1.0, 0.0]])
    normal = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    reflected = reflect(direction, normal)
    assert torch.allclose(reflected[0], torch.tensor([0.0, 1.0, 0.0]), atol=1.0e-6)
    # Grazing reflection reverses only the normal component.
    expected = direction[1] - 2.0 * torch.dot(direction[1], normal[1]) * normal[1]
    assert torch.allclose(reflected[1], expected / expected.norm(), atol=1.0e-6)


def test_schlick_fresnel_endpoints() -> None:
    f0 = 0.04
    assert schlick_fresnel(torch.tensor(1.0), f0).item() == pytest.approx(f0, abs=1.0e-7)
    assert schlick_fresnel(torch.tensor(0.0), f0).item() == pytest.approx(1.0, abs=1.0e-7)
    middle = schlick_fresnel(torch.tensor(0.5), f0).item()
    assert f0 < middle < 1.0


def test_orthonormal_basis_orientation_deterministic() -> None:
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    tangent, bitangent = orthonormal_basis(normal)
    assert torch.allclose((tangent * normal).sum(-1), torch.zeros(1), atol=1.0e-6)
    assert torch.allclose((bitangent * normal).sum(-1), torch.zeros(1), atol=1.0e-6)
    twice = orthonormal_basis(normal)
    assert torch.allclose(tangent, twice[0]) and torch.allclose(bitangent, twice[1])


def test_cosine_hemisphere_sample_is_in_hemisphere_and_centred() -> None:
    count = 65536
    normal = torch.tensor([0.0, 1.0, 0.0]).expand(count, 3)
    u1, u2 = CounterRNG(seed=3).uniform_pair(torch.arange(count))
    samples = cosine_hemisphere_sample(normal, u1, u2)
    assert torch.allclose(samples.norm(dim=-1), torch.ones(count), atol=1.0e-5)
    assert bool(((samples * normal).sum(-1) >= -1.0e-6).all())
    mean = samples.mean(dim=0)
    # The mean of cosine-weighted unit directions has magnitude 2/3.
    assert torch.allclose(mean, torch.tensor([0.0, 2.0 / 3.0, 0.0]), atol=0.02)


def test_ggx_half_vector_and_reflection() -> None:
    count = 256
    normal = torch.tensor([0.0, 0.0, 1.0]).expand(count, 3)
    u1, u2 = CounterRNG(seed=11).uniform_pair(torch.arange(count))
    half = ggx_half_vector(normal, 0.5, u1, u2)
    assert torch.allclose(half.norm(dim=-1), torch.ones(count), atol=1.0e-5)
    assert bool(((half * normal).sum(-1) >= -1.0e-6).all())
    smooth = ggx_half_vector(normal, 1.0e-3, u1, u2)
    assert torch.allclose(smooth, normal, atol=2.0e-2)
    incident = -normal
    reflected = ggx_reflection(incident, normal, 0.4, u1, u2)
    assert torch.allclose(reflected.norm(dim=-1), torch.ones(count), atol=1.0e-5)
    assert bool(((reflected * normal).sum(-1) >= -1.0e-6).all())


def test_refract_normal_incidence_and_snell() -> None:
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    straight = refract(torch.tensor([[0.0, 0.0, -1.0]]), normal, 1.0 / 1.5)
    assert not bool(straight.tir[0])
    assert torch.allclose(straight.direction, torch.tensor([[0.0, 0.0, -1.0]]), atol=1.0e-6)
    # eta == 1 is a no-op.
    same = refract(torch.tensor([[0.5, 0.0, -math.sqrt(0.75)]]), normal, 1.0)
    assert torch.allclose(same.direction, torch.tensor([[0.5, 0.0, -math.sqrt(0.75)]]), atol=1.0e-6)


def test_refract_total_internal_reflection_falls_back_to_reflection() -> None:
    # Dense -> rare with an incidence angle beyond the critical angle.
    direction = torch.tensor([[math.sqrt(0.75), 0.0, -0.5]])
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    result = refract(direction, normal, 1.5)
    assert bool(result.tir[0])
    expected = reflect(direction, normal)
    assert torch.allclose(result.direction, expected, atol=1.0e-6)
    assert result.cos_theta[0].item() == pytest.approx(0.0, abs=1.0e-7)


def test_thin_walled_refract_keeps_direction_and_weights_transmission() -> None:
    direction = torch.tensor([[0.0, 0.0, -1.0]])
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    transmitted, weight = thin_walled_refract(direction, normal, f0=0.04, transmission=1.0)
    assert torch.allclose(transmitted, direction, atol=1.0e-6)
    assert weight[0].item() == pytest.approx(1.0 - 0.04, abs=1.0e-6)
