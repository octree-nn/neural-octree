"""Legacy N-BVH matrix camera adapter and the equidistant fisheye frame.

The flat matrix is copied verbatim from the real N-BVH chess config
(``configs/nodes_32940_hash_18_lvl_8_tsteps_10000_neural.json``) together with
its ``yfov`` of 45 degrees.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from neuoct.rendering_rt import Camera, load_rt_config

# GLM column-major WORLD-TO-CAMERA view matrix, exactly as stored on disk.
LEGACY_MATRIX = [
    0.28359198570251465, -0.5326340198516846, 0.7974169850349426, 0.0,
    0.0, 0.8315569758415222, 0.5554370284080505, 0.0,
    -0.9589440226554871, -0.15751700103282928, 0.23582300543785095, 0.0,
    0.00184299994725734, 0.050255998969078064, -0.7192429900169373, 1.0,
]
NBVH_YFOV_RAD = 0.7853981852531433  # 45 degrees

# Audited camera-to-world values for that matrix.
EXPECTED_ORIGIN = [0.59978395, 0.35770471, 0.17929818]
EXPECTED_FORWARD = [-0.79741966, -0.55543891, -0.23582380]
EXPECTED_UP = [-0.53263507, 0.83155897, -0.15751810]
EXPECTED_RIGHT = [0.28359213, 0.0, -0.95894601]


def _camera(width: int = 64, height: int = 64, *, projection: str = "perspective", fov_deg: float = 180.0):
    return Camera.from_legacy_matrix(
        LEGACY_MATRIX,
        width=width,
        height=height,
        yfov=NBVH_YFOV_RAD,
        projection=projection,
        fisheye_fov=math.radians(fov_deg),
    )


def test_flat_matrix_is_column_major_world_to_camera() -> None:
    camera = _camera()
    # Reshaping with order='F' and inverting must reproduce the audited pose.
    view = np.asarray(LEGACY_MATRIX, dtype=np.float64).reshape((4, 4), order="F")
    c2w = np.linalg.inv(view)
    assert np.allclose(camera.camera_to_world.numpy(), c2w, atol=1e-6)
    # The adapter must not treat the raw translation column as the origin: that
    # translation is [.001843, .050256, -.719243], the origin is far from it.
    assert not np.allclose(c2w[:3, 3], [0.001843, 0.050256, -0.719243], atol=1e-3)


def test_audited_camera_origin_forward_up_and_right() -> None:
    camera = _camera()
    matrix = camera.camera_to_world.numpy()
    origin = matrix[:3, 3]
    right = matrix[:3, 0]
    up = matrix[:3, 1]
    backward = matrix[:3, 2]
    assert np.allclose(origin, EXPECTED_ORIGIN, atol=1e-5)
    assert np.allclose(-backward, EXPECTED_FORWARD, atol=1e-5)
    assert np.allclose(up, EXPECTED_UP, atol=1e-5)
    assert np.allclose(right, EXPECTED_RIGHT, atol=1e-5)
    # Proper rotation, no reflection; local -Z is forward for legacy_blender.
    # (The adapter inverts a float32 view matrix, so det is +1 to float32
    # precision rather than to machine epsilon.)
    assert np.linalg.det(matrix[:3, :3]) == pytest.approx(1.0, abs=1e-4)
    assert camera.camera_convention == "legacy_blender"
    # The principal point is the optical axis: use an odd image size so pixel
    # (h//2, w//2) really is the centre pixel.
    odd = _camera(65, 65)
    centre = odd.camera_directions()[32, 32]
    assert torch.allclose(centre, torch.tensor([0.0, 0.0, -1.0]), atol=1e-6)


def test_vertical_fov_matches_nbvh_image_plane() -> None:
    width = height = 512
    camera = _camera(width, height)
    # N-BVH: img_plane_size.y = 2*tan(fovy/2), img_plane_size.x = y * aspect.
    half_height = math.tan(0.5 * NBVH_YFOV_RAD)
    assert camera.fy == pytest.approx(0.5 * height / half_height, rel=1e-6)
    assert camera.fx == pytest.approx(camera.fy, rel=1e-12)
    # The top-centre ray's offset from the optical axis is exactly half the image
    # plane height; the pixel *centre* is half a pixel inside the image edge, so
    # the angle is atan((height/2 - 0.5) / fy) rather than exactly fovy/2.
    top = camera.camera_directions()[0, width // 2]
    # The pixel centre is half a pixel inside the image edge.
    assert float(top[0]) == pytest.approx(0.0, abs=2.0 / camera.fx)
    tan_angle = abs(float(top[1])) / abs(float(top[2]))
    assert tan_angle == pytest.approx((height / 2.0 - 0.5) / camera.fy, rel=1e-5)
    # And that is fovy/2 up to the half-pixel offset.
    assert math.atan(tan_angle) == pytest.approx(0.5 * NBVH_YFOV_RAD, rel=2e-3)


def test_legacy_fisheye_y_sign_matches_perspective() -> None:
    fisheye = _camera(64, 64, projection="equidistant", fov_deg=180.0)
    # Small theta: the equidistant direction must agree in sign with the
    # perspective direction of the same pixel, i.e. image row -> -Y.
    x, y = fisheye._pixel_coordinates()
    u = (x + 0.5 - fisheye.cx) / fisheye.fx
    v = (y + 0.5 - fisheye.cy) / fisheye.fy
    theta = torch.sqrt(u * u + v * v)
    directions = fisheye.camera_directions()
    near_axis = theta < 0.05
    expected = torch.stack([u, -v, -torch.ones_like(u)], dim=-1)
    expected = expected / expected.norm(dim=-1, keepdim=True)
    assert torch.allclose(directions[near_axis], expected[near_axis], atol=1e-4)
    # The sign convention holds across the whole disk, not just near the axis.
    assert bool((torch.sign(directions[..., 0]) == torch.sign(u))[theta > 0.1].all())
    assert bool((torch.sign(directions[..., 1]) == torch.sign(-v))[theta > 0.1].all())
    # A pixel in the lower half of the image must look *downwards* (-Y).
    assert float(directions[56, 32, 1]) < 0.0
    assert float(directions[8, 32, 1]) > 0.0


def test_fisheye_principal_center_center_ray_and_edge_angle() -> None:
    fov_deg = 180.0
    camera = _camera(65, 65, projection="equidistant", fov_deg=fov_deg)
    center_y, center_x = 32, 32
    centre = camera.camera_directions()[center_y, center_x]
    assert torch.allclose(centre, torch.tensor([0.0, 0.0, -1.0]), atol=1e-6)
    x, y = camera._pixel_coordinates()
    u = (x + 0.5 - camera.cx) / camera.fx
    v = (y + 0.5 - camera.cy) / camera.fy
    theta = torch.sqrt(u * u + v * v)
    directions = camera.camera_directions()
    angle = torch.acos((-directions[..., 2]).clamp(-1.0, 1.0))
    mask = camera.pixel_mask()
    assert torch.allclose(angle[mask], theta[mask], atol=1e-5)
    assert mask[center_y, center_x]
    # The disk radius in pixels is exactly half the shorter image dimension, so
    # the *geometric* disk edge is at fov/2 and the outermost included pixel
    # centre sits half a pixel inside it.
    assert camera.fx * (math.radians(fov_deg) / 2.0) == pytest.approx(65 / 2.0, rel=1e-9)
    assert float(theta[mask].max()) == pytest.approx(
        math.radians(fov_deg) / 2.0, abs=1.0 / camera.fx
    )
    edge_direction = directions[center_y, 0]
    edge_angle = math.acos(max(-1.0, min(1.0, -float(edge_direction[2]))))
    assert edge_angle == pytest.approx((65 // 2) / camera.fx, rel=1e-6)
    assert edge_angle < math.radians(fov_deg) / 2.0
    assert not bool(mask[0, 0])


def test_config_matrix_camera_round_trips(tmp_path) -> None:
    config_path = tmp_path / "matrix_camera.yaml"
    numbers = ", ".join(repr(value) for value in LEGACY_MATRIX)
    config_path.write_text(
        "version: 2\n"
        "scene:\n"
        "  similarity:\n"
        "    scale: 1.0\n"
        "    rotation: [[1,0,0],[0,1,0],[0,0,1]]\n"
        "    translation: [0,0,0]\n"
        "camera:\n"
        "  type: perspective\n"
        "  layout: matrix\n"
        "  yfov_deg: 45.0\n"
        f"  matrix: [{numbers}]\n"
        "render:\n"
        "  width: 32\n"
        "  height: 32\n"
        "primitives: []\n",
        encoding="utf-8",
    )
    config = load_rt_config(config_path)
    assert config["camera"]["layout"] == "matrix"
    assert config["camera"]["convention"] == "legacy_blender"
    assert config["camera"]["yfov_deg"] == pytest.approx(45.0)
    assert len(config["camera"]["matrix"]) == 16
