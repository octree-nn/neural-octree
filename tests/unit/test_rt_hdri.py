"""HDRI environment: exact N-BVH direction lookup, sampling, rotation and cap."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from neuoct.rendering_rt import (
    DEFAULT_CHANNEL_CAP,
    LatLongHDRI,
    direction_to_uv,
    rotation_y_matrix,
    uv_to_direction,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _uv(x: float, y: float, z: float, rotation_deg: float = 0.0) -> tuple[float, float]:
    uv = direction_to_uv(torch.tensor([[x, y, z]], dtype=torch.float64), rotation_deg)
    return float(uv[0, 0]), float(uv[0, 1])


def test_analytic_rotation_zero_mappings() -> None:
    # -Z is the centre of the map.
    assert _uv(0.0, 0.0, -1.0) == pytest.approx((0.5, 0.5), abs=1e-12)
    # +X is a quarter turn to the right of centre.
    assert _uv(1.0, 0.0, 0.0) == pytest.approx((0.75, 0.5), abs=1e-12)
    # -X is the mirror image.
    assert _uv(-1.0, 0.0, 0.0) == pytest.approx((0.25, 0.5), abs=1e-12)
    # +Y is the top pole, -Y the bottom.  ``u`` is a singularity *at* the pole
    # (N-BVH's atan2(0, -0) returns pi there too), so only ``v`` is asserted
    # exactly; just off the pole ``u`` is well defined again.
    assert _uv(0.0, 1.0, 0.0)[1] == pytest.approx(0.0, abs=1e-12)
    assert _uv(0.0, -1.0, 0.0)[1] == pytest.approx(1.0, abs=1e-12)
    assert _uv(1e-9, 1.0 - 1e-12, 0.0)[0] == pytest.approx(0.75, abs=1e-6)


def test_analytic_rotation_ninety_degrees() -> None:
    # A right-handed +90 deg rotation about +Y sends world +X to the centre...
    assert _uv(1.0, 0.0, 0.0, 90.0) == pytest.approx((0.5, 0.5), abs=1e-12)
    # ... world -Z to (x' = -1, z' = 0), i.e. u = 0.25 ...
    assert _uv(0.0, 0.0, -1.0, 90.0) == pytest.approx((0.25, 0.5), abs=1e-12)
    # ... and world +Z to the opposite side of the map.
    assert _uv(0.0, 0.0, 1.0, 90.0) == pytest.approx((0.75, 0.5), abs=1e-12)
    # The rotation must not touch the poles.
    assert _uv(0.0, 1.0, 0.0, 90.0)[1] == pytest.approx(0.0, abs=1e-12)
    assert _uv(1e-9, -1.0, 0.0, 90.0)[1] == pytest.approx(1.0, abs=1e-12)


def test_analytic_rotation_285_degrees_matches_nbvh_config() -> None:
    # cos(285) = 0.2588190451, sin(285) = -0.9659258263.  World +X maps to
    # (cos285, 0, -sin285) = (0.2588, 0, 0.9659), i.e. u = 1 - 15/360.
    u, v = _uv(1.0, 0.0, 0.0, 285.0)
    assert u == pytest.approx(1.0 - 15.0 / 360.0, abs=1e-9)
    assert v == pytest.approx(0.5, abs=1e-12)
    # glm::rotate(285 deg, +Y) in row-major form.
    expected = np.array(
        [
            [math.cos(math.radians(285.0)), 0.0, math.sin(math.radians(285.0))],
            [0.0, 1.0, 0.0],
            [-math.sin(math.radians(285.0)), 0.0, math.cos(math.radians(285.0))],
        ]
    )
    assert np.allclose(rotation_y_matrix(285.0), expected, atol=1e-12)


def test_uv_to_direction_is_the_inverse_mapping() -> None:
    directions = torch.tensor(
        [[0.3, 0.5, -0.8], [-0.7, -0.2, 0.6], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    directions = directions / directions.norm(dim=-1, keepdim=True)
    for rotation in (0.0, 90.0, 285.0):
        uv = direction_to_uv(directions, rotation)
        restored = uv_to_direction(uv)
        rotated = torch.as_tensor(rotation_y_matrix(rotation), dtype=torch.float64)
        # ``uv_to_direction`` returns the direction in the *environment* frame,
        # i.e. the world direction already rotated by the configured angle.
        assert torch.allclose(restored, directions @ rotated.transpose(0, 1), atol=1e-9)


def _gradient_image(width: int = 4, height: int = 2) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[..., 0] = np.linspace(0.0, 1.0, width)[None, :]
    image[..., 1] = np.linspace(0.0, 1.0, height)[:, None]
    image[..., 2] = 0.25
    return image


def test_synthetic_sampling_is_bilinear_and_clamped() -> None:
    image = _gradient_image(4, 2)
    environment = LatLongHDRI.from_array(image, rotation_deg=0.0, strength=1.0)
    # Centre of texel (0, 0) in normalized coordinates.
    uv = torch.tensor([[0.5 / 4.0, 0.5 / 2.0]])
    sampled = environment.sample_uv(uv)
    assert torch.allclose(sampled[0], torch.tensor(image[0, 0]), atol=1e-6)
    # Exactly between the two top texels -> the average.
    uv = torch.tensor([[1.0 / 4.0, 0.5 / 2.0]])
    sampled = environment.sample_uv(uv)
    assert torch.allclose(sampled[0], torch.tensor((image[0, 0] + image[0, 1]) / 2.0), atol=1e-6)
    # u = 0 clamps to the first column; u = 1 clamps to the last.
    clamp_low = environment.sample_uv(torch.tensor([[0.0, 0.5]]))[0]
    clamp_high = environment.sample_uv(torch.tensor([[1.0, 0.5]]))[0]
    assert torch.allclose(clamp_low, torch.tensor((image[0, 0] + image[1, 0]) / 2.0), atol=1e-6)
    assert torch.allclose(clamp_high, torch.tensor((image[0, 3] + image[1, 3]) / 2.0), atol=1e-6)


def test_strength_and_nbvh_channel_cap() -> None:
    image = np.full((2, 4, 3), 60.0, dtype=np.float32)
    capped = LatLongHDRI.from_array(image, strength=1.0)
    assert torch.allclose(capped.sample_uv(torch.tensor([[0.5, 0.5]]))[0], torch.full((3,), 60.0))
    scaled = LatLongHDRI.from_array(image, strength=4.0)
    # 60 * 4 = 240 must be clamped to N-BVH's 100 firefly cap.
    assert torch.allclose(
        scaled.sample_uv(torch.tensor([[0.5, 0.5]]))[0], torch.full((3,), DEFAULT_CHANNEL_CAP)
    )


def test_radiance_uses_the_rotated_world_direction() -> None:
    # A map whose red channel encodes u; the radiance of a world direction must
    # equal the sample at the *rotated* uv.
    image = _gradient_image(8, 4)
    for rotation in (0.0, 90.0, 285.0):
        environment = LatLongHDRI.from_array(image, rotation_deg=rotation)
        directions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float32)
        radiance = environment.radiance(directions)
        expected_uv = direction_to_uv(directions, rotation)
        assert torch.allclose(radiance, environment.sample_uv(expected_uv), atol=1e-6)


def test_missing_openexr_reports_actionable_error(tmp_path) -> None:
    from neuoct.rendering_rt import load_exr_rgb

    with pytest.raises(FileNotFoundError):
        load_exr_rgb(tmp_path / "absent.exr")
    # If OpenEXR is unavailable the loader must say what to install; if it is
    # available the call must still fail loudly on a non-EXR file.
    broken = tmp_path / "broken.exr"
    broken.write_bytes(b"not an exr")
    with pytest.raises(Exception) as error:
        load_exr_rgb(broken)
    message = str(error.value)
    assert "OpenEXR" in message or "exr" in message.lower()


def test_exr_channel_layouts_are_both_supported() -> None:
    # Lat-long EXRs are written either as three planar channels or as a single
    # interleaved RGB channel; both must assemble to (H, W, 3).
    from neuoct.rendering_rt.hdri import _assemble_rgb

    height, width = 3, 5
    red = np.full((height, width), 0.25, dtype=np.float32)
    green = np.full((height, width), 0.5, dtype=np.float32)
    blue = np.full((height, width), 0.75, dtype=np.float32)
    planar = _assemble_rgb({"R": red, "G": green, "B": blue}, "planar.exr")
    assert planar.shape == (height, width, 3)
    assert np.allclose(planar[..., 0], 0.25)
    assert np.allclose(planar[..., 2], 0.75)

    interleaved = np.stack([red, green, blue], axis=-1)
    from_interleaved = _assemble_rgb({"RGB": interleaved}, "interleaved.exr")
    assert np.allclose(from_interleaved, planar)
    # Channel-first storage is handled too.
    channel_first = _assemble_rgb({"RGB": np.moveaxis(interleaved, -1, 0)}, "cf.exr")
    assert np.allclose(channel_first, planar)
    # RGBA drops alpha.
    rgba = np.concatenate([interleaved, np.ones((height, width, 1), dtype=np.float32)], axis=-1)
    assert np.allclose(_assemble_rgb({"RGBA": rgba}, "rgba.exr"), planar)
    with pytest.raises(ValueError):
        _assemble_rgb({"Z": red}, "depth.exr")


def test_ambient_is_cosine_weighted_upper_hemisphere() -> None:
    # A map that is bright only in the top half: the cosine-weighted +Y average
    # must be strictly between the two halves.
    image = np.zeros((4, 8, 3), dtype=np.float32)
    image[:2] = 1.0
    environment = LatLongHDRI.from_array(image)
    ambient = environment.ambient
    assert 0.0 < float(ambient[0]) < 1.0
    assert float(ambient[0]) == pytest.approx(1.0, abs=0.35)
