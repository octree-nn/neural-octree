import torch

from neuoct.data.camera import CameraIntrinsics, generate_rays, unproject_rgbd


def test_center_ray_and_identity_unprojection():
    intrinsics = CameraIntrinsics.from_horizontal_fov(torch.pi / 2, width=3, height=3)
    transform = torch.eye(4)
    origins, directions = generate_rays(intrinsics, transform)
    center = 4
    torch.testing.assert_close(origins[center], torch.zeros(3))
    torch.testing.assert_close(directions[center], torch.tensor([0.0, 0.0, 1.0]))
    depth = torch.ones(3, 3)
    normal = torch.full((3, 3, 3), 127, dtype=torch.uint8)
    color = torch.full((3, 3, 3), 255, dtype=torch.uint8)
    _, _, points, normals, colors = unproject_rgbd(depth, normal, color, intrinsics, transform)
    torch.testing.assert_close(points[center], torch.tensor([0.0, 0.0, 1.0]))
    assert normals.shape == colors.shape == (9, 3)
    torch.testing.assert_close(colors[center], torch.ones(3))


def test_legacy_blender_center_ray_and_unprojection():
    intrinsics = CameraIntrinsics.from_horizontal_fov(torch.pi / 2, width=3, height=3)
    transform = torch.eye(4)
    origins, directions = generate_rays(
        intrinsics, transform, convention="legacy_blender"
    )
    center = 4
    torch.testing.assert_close(origins[center], torch.zeros(3))
    torch.testing.assert_close(directions[center], torch.tensor([0.0, 0.0, -1.0]))
    depth = torch.ones(3, 3)
    normal = torch.full((3, 3, 3), 127, dtype=torch.uint8)
    color = torch.zeros((3, 3, 3), dtype=torch.uint8)
    _, _, points, _, _ = unproject_rgbd(
        depth,
        normal,
        color,
        intrinsics,
        transform,
        convention="legacy_blender",
    )
    torch.testing.assert_close(points[center], torch.tensor([0.0, 0.0, -1.0]))


def test_float_normals_are_not_uint8_rescaled():
    intrinsics = CameraIntrinsics.from_horizontal_fov(torch.pi / 2, width=3, height=3)
    expected = torch.tensor([0.25, -0.5, 0.75])
    normal = expected.expand(3, 3, 3).clone()
    _, _, _, normals, _ = unproject_rgbd(
        torch.ones(3, 3),
        normal,
        torch.zeros((3, 3, 3), dtype=torch.uint8),
        intrinsics,
        torch.eye(4),
    )
    torch.testing.assert_close(normals[4], expected)
