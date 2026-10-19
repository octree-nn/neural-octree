import pytest
import torch

from neuoct.data.camera import CameraIntrinsics, generate_rays
from neuoct.rendering.views import generate_orbit_views


@pytest.mark.parametrize("convention", ["neuoct", "legacy_blender"])
@pytest.mark.parametrize("up_axis", ["x", "y", "z"])
def test_orbit_center_rays_look_at_target(convention: str, up_axis: str) -> None:
    poses = generate_orbit_views(
        16, radius=2.5, camera_convention=convention, up_axis=up_axis, device="cpu"
    )
    intrinsics = CameraIntrinsics(width=3, height=3, fx=2.0, fy=2.0, cx=1.5, cy=1.5)
    center = 4
    for pose in poses:
        _, directions = generate_rays(intrinsics, pose, convention=convention)
        expected = torch.nn.functional.normalize(-pose[:3, 3], dim=0)
        torch.testing.assert_close(directions[center], expected, atol=1.0e-5, rtol=1.0e-5)


@pytest.mark.parametrize("convention", ["neuoct", "legacy_blender"])
def test_orbit_poses_are_rigid_and_nonduplicated(convention: str) -> None:
    poses = generate_orbit_views(32, radius=3.0, camera_convention=convention, device="cpu")
    rotation = poses[:, :3, :3]
    identity = torch.eye(3).expand_as(rotation)
    torch.testing.assert_close(rotation.transpose(1, 2) @ rotation, identity, atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(32), atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(
        poses[:, :3, 3].norm(dim=1), torch.full((32,), 3.0), atol=1.0e-5, rtol=1.0e-5
    )
    assert not torch.allclose(poses[0], poses[-1])


def test_orbit_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="num_views"):
        generate_orbit_views(0, device="cpu")
    with pytest.raises(ValueError, match="camera_convention"):
        generate_orbit_views(4, camera_convention="unknown", device="cpu")
    with pytest.raises(ValueError, match="up_axis"):
        generate_orbit_views(4, up_axis="unknown", device="cpu")


def test_z_up_orbit_rotates_in_xy_plane() -> None:
    elevation = 0.2
    poses = generate_orbit_views(
        16, radius=2.5, elevation_radians=elevation, up_axis="z", device="cpu"
    )
    positions = poses[:, :3, 3]
    torch.testing.assert_close(
        positions[:, 2],
        torch.full((16,), 2.5 * torch.sin(torch.tensor(elevation))),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    torch.testing.assert_close(
        positions[:, :2].norm(dim=1),
        torch.full((16,), 2.5 * torch.cos(torch.tensor(elevation))),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
