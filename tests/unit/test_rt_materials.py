from __future__ import annotations

import types

import pytest
import torch

from neuoct.rendering_rt import MaterialResolver
from neuoct.rendering_rt.color import srgb_to_linear


def _mesh(**material_fields) -> types.SimpleNamespace:
    material = types.SimpleNamespace(**material_fields)
    return types.SimpleNamespace(visual=types.SimpleNamespace(material=material))


def test_neuoct_audited_defaults() -> None:
    material = MaterialResolver().resolve_neuoct()
    assert material.roughness == pytest.approx(0.5)
    assert material.metallic == pytest.approx(0.0)
    assert material.specular == pytest.approx(0.04)
    assert material.transmission == pytest.approx(0.0)
    assert material.ior == pytest.approx(1.5)
    assert material.thin_walled is True
    assert torch.allclose(material.base_color, torch.ones(3))
    assert torch.allclose(material.emission, torch.zeros(3))


def test_neuoct_overrides_and_unknown_key() -> None:
    material = MaterialResolver().resolve_neuoct(
        {"roughness": 0.1, "metallic": 1.0, "ior": 1.33, "base_color": [0.2, 0.4, 0.6]}
    )
    assert material.roughness == pytest.approx(0.1)
    assert material.metallic == pytest.approx(1.0)
    assert material.ior == pytest.approx(1.33)
    assert torch.allclose(material.base_color, torch.tensor([0.2, 0.4, 0.6]))
    with pytest.raises(ValueError):
        MaterialResolver().resolve_neuoct({"not_a_field": 1.0})


def test_mesh_base_color_factor_is_linear() -> None:
    mesh = _mesh(baseColorFactor=[0.2, 0.4, 0.6, 1.0], roughnessFactor=0.3, metallicFactor=0.7)
    material = MaterialResolver().resolve_mesh(mesh)
    assert torch.allclose(material.base_color, torch.tensor([0.2, 0.4, 0.6]), atol=1.0e-6)
    assert material.roughness == pytest.approx(0.3)
    assert material.metallic == pytest.approx(0.7)
    assert material.base_color_srgb is False


def test_mesh_named_color_is_decoded_from_srgb() -> None:
    mesh = _mesh(diffuse=[128.0, 64.0, 32.0])
    material = MaterialResolver().resolve_mesh(mesh)
    expected = srgb_to_linear(torch.tensor([128.0, 64.0, 32.0]) / 255.0)
    assert torch.allclose(material.base_color, expected, atol=1.0e-6)
    assert material.base_color_srgb is True


def test_mesh_missing_fields_use_deterministic_defaults() -> None:
    mesh = types.SimpleNamespace(visual=None)
    material = MaterialResolver().resolve_mesh(mesh)
    assert torch.allclose(material.base_color, torch.ones(3))
    assert material.roughness == pytest.approx(0.5)
    assert material.ior == pytest.approx(1.5)
    assert material.thin_walled is True


def test_mesh_byte_factor_is_scaled_and_override_wins() -> None:
    mesh = _mesh(baseColorFactor=[255, 128, 0, 255])
    material = MaterialResolver().resolve_mesh(mesh)
    assert torch.allclose(material.base_color, torch.tensor([1.0, 128.0 / 255.0, 0.0]), atol=1.0e-5)
    overridden = MaterialResolver().resolve_mesh(mesh, {"base_color": [0.1, 0.2, 0.3]})
    assert torch.allclose(overridden.base_color, torch.tensor([0.1, 0.2, 0.3]), atol=1.0e-6)
