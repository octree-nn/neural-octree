from __future__ import annotations

import math

import torch

from neuoct.rendering_rt import Camera, look_at_matrix


def _c2w(translation=(0.0, 0.0, 0.0)):
    matrix = torch.eye(4)
    matrix[:3, 3] = torch.tensor(translation, dtype=torch.float32)
    return matrix


def test_perspective_center_and_corner_directions() -> None:
    center = Camera(width=1, height=1, fx=1.0, fy=1.0, cx=0.5, cy=0.5, camera_to_world=_c2w())
    assert torch.allclose(center.camera_directions().reshape(3), torch.tensor([0.0, 0.0, 1.0]), atol=1.0e-6)

    corner = Camera(width=2, height=2, fx=2.0, fy=2.0, cx=1.0, cy=1.0, camera_to_world=_c2w())
    direction = corner.camera_directions()[0, 0]
    expected = torch.tensor([0.25, 0.25, 1.0])
    expected = expected / expected.norm()
    assert torch.allclose(direction, expected, atol=1.0e-6)


def test_perspective_directions_are_unit_and_frame_is_documented() -> None:
    camera = Camera(width=5, height=4, fx=3.0, fy=3.0, cx=2.0, cy=1.5, camera_to_world=_c2w())
    directions = camera.camera_directions()
    assert torch.allclose(directions.norm(dim=-1), torch.ones(4, 5), atol=1.0e-5)
    rays, mask = camera.primary_rays()
    assert rays.num_rays == 20
    assert mask.shape == (4, 5)
    assert bool(mask.all())
    # The neuoct frame stores the basis as rows and uses +Z forward.
    assert hasattr(camera, "camera_to_world")


def test_legacy_blender_center_points_along_minus_z() -> None:
    camera = Camera(
        width=1,
        height=1,
        fx=1.0,
        fy=1.0,
        cx=0.5,
        cy=0.5,
        camera_to_world=_c2w(),
        camera_convention="legacy_blender",
    )
    assert torch.allclose(camera.camera_directions().reshape(3), torch.tensor([0.0, 0.0, -1.0]), atol=1.0e-6)


def test_fisheye_center_is_optical_axis_and_angle_matches_equidistant() -> None:
    camera = Camera.from_fisheye_fov(math.pi, 9, 9, _c2w())
    direction = camera.camera_directions()[4, 4]
    assert torch.allclose(direction, torch.tensor([0.0, 0.0, 1.0]), atol=1.0e-6)
    rays, mask = camera.primary_rays()
    assert rays.num_rays == 81
    assert bool(mask[4, 4])
    # For every valid pixel the incidence angle equals the equidistant theta.
    x, y = camera._pixel_coordinates()
    u = (x + 0.5 - camera.cx) / camera.fx
    v = (y + 0.5 - camera.cy) / camera.fy
    theta = torch.sqrt(u * u + v * v)
    directions = camera.camera_directions()
    angle = torch.acos(directions[..., 2].clamp(-1.0, 1.0))
    valid = camera.pixel_mask()
    assert torch.allclose(angle[valid], theta[valid], atol=1.0e-5)


def test_fisheye_masks_outside_disk() -> None:
    camera = Camera.from_fisheye_fov(math.pi / 2.0, 8, 8, _c2w())
    mask = camera.pixel_mask()
    assert mask.shape == (8, 8)
    assert bool(mask[3:5, 3:5].all())
    assert not bool(mask[0, 0])
    assert not bool(mask[7, 7])
    # Masked rays still exist (finite) but are not valid output pixels.
    rays, _ = camera.primary_rays()
    assert torch.isfinite(rays.origins).all()
    assert torch.isfinite(rays.directions).all()


def test_camera_rays_scale_rotation_and_clipping() -> None:
    matrix = look_at_matrix([0.0, 0.0, 3.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    camera = Camera.from_horizontal_fov(0.7, 4, 4, matrix, near=0.1, far=50.0)
    rays = camera.generate_rays()
    assert torch.allclose(rays.origins, torch.tensor([0.0, 0.0, 3.0]).expand(16, 3), atol=1.0e-6)
    assert torch.allclose(rays.t_min, torch.full((16,), 0.1))
    assert torch.allclose(rays.t_max, torch.full((16,), 50.0))
    assert torch.allclose(rays.directions.norm(dim=-1), torch.ones(16), atol=1.0e-5)
    # Forward direction is -Z in world (camera at +Z looking at origin).
    assert rays.directions[:, 2].max() < 0.0
