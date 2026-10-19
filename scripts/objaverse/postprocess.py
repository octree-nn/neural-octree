#!/usr/bin/env python3
"""Convert Blender outputs into NeuOct normals and colored point clouds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import OpenEXR
import torch
from PIL import Image


def _seed(uid: str) -> int:
    return int.from_bytes(hashlib.sha256(uid.encode("utf-8")).digest()[:8], "little")


def read_normal(path: Path) -> np.ndarray:
    channels = OpenEXR.File(str(path)).channels()
    if "RGB" not in channels:
        raise ValueError(f"EXR has no RGB channel: {path}")
    normal = np.asarray(channels["RGB"].pixels, dtype=np.float32)
    if normal.ndim != 3 or normal.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 EXR normals, got {normal.shape}: {path}")
    return normal


def save_normal_png(exr_path: Path, png_path: Path, valid: np.ndarray) -> None:
    normal = read_normal(exr_path)
    if normal.shape[:2] != valid.shape:
        raise ValueError(f"Normal/depth shape mismatch: {exr_path}")
    lengths = np.linalg.norm(normal, axis=-1, keepdims=True)
    foreground = valid & (lengths[..., 0] > 1.0e-6)
    normal[foreground] /= lengths[foreground]
    normal[~foreground] = 0
    encoded = np.rint(np.clip(normal * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
    encoded[~valid] = 127
    Image.fromarray(encoded, mode="RGB").save(png_path, compress_level=6)


def frame_points(
    depth_raw: np.ndarray,
    albedo: np.ndarray,
    frame: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    valid = depth_raw != 65535
    height, width = depth_raw.shape
    depth_info = frame["depth"]
    depth_min = float(depth_info["min"])
    depth_max = float(depth_info["max"])
    depth = depth_min + depth_raw.astype(np.float32) * (depth_max - depth_min) / 65535.0
    focal = 0.5 * width / np.tan(0.5 * float(frame["camera_angle_x"]))
    x = (np.arange(width, dtype=np.float32) + 0.5 - width * 0.5) / focal
    y = -(np.arange(height, dtype=np.float32) + 0.5 - height * 0.5) / focal
    xx, yy = np.meshgrid(x, y)
    directions = np.stack((xx, yy, -np.ones_like(xx)), axis=-1)
    camera_points = directions[valid] * depth[valid, None]
    camera_to_world = np.asarray(frame["transform_matrix"], dtype=np.float32)
    points = camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    colors = albedo[..., :3][valid].astype(np.float32) / 255.0
    return points, colors


def composite_albedo(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
    return np.rint(rgb).astype(np.uint8)


def postprocess(
    image_dir: Path,
    pointcloud_path: Path,
    *,
    uid: str,
    point_count: int,
    keep_intermediates: bool = True,
) -> dict[str, object]:
    metadata = json.loads((image_dir / "transforms.json").read_text(encoding="utf-8"))
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames in {image_dir / 'transforms.json'}")
    point_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    foreground_fractions: list[float] = []
    for frame in frames:
        stem = Path(str(frame["file_path"])).stem
        depth_path = image_dir / f"{stem}_depth.png"
        normal_exr_path = image_dir / f"{stem}_normal.exr"
        normal_png_path = image_dir / f"{stem}_normal.png"
        albedo_path = image_dir / f"{stem}_albedo.png"
        with Image.open(depth_path) as image:
            depth_raw = np.asarray(image).copy()
        if depth_raw.dtype != np.uint16:
            raise ValueError(f"Depth must be uint16, got {depth_raw.dtype}: {depth_path}")
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[..., 0]
        valid = depth_raw != 65535
        foreground_fractions.append(float(valid.mean()))
        if valid.sum() == 0:
            raise ValueError(f"Empty depth frame: {depth_path}")
        save_normal_png(normal_exr_path, normal_png_path, valid)
        albedo = composite_albedo(albedo_path)
        primary_path = image_dir / f"{stem}.png"
        if keep_intermediates:
            primary_path.replace(image_dir / f"{stem}_beauty.png")
        Image.fromarray(albedo, mode="RGB").save(primary_path, compress_level=6)
        points, colors = frame_points(depth_raw, albedo, frame)
        point_parts.append(points)
        color_parts.append(colors)

    points = np.concatenate(point_parts, axis=0)
    colors = np.concatenate(color_parts, axis=0)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    points = points[finite]
    colors = colors[finite]
    in_bounds = np.abs(points).max(axis=1) <= 1.01
    points = points[in_bounds]
    colors = colors[in_bounds]
    if len(points) < min(point_count, 4096):
        raise ValueError(f"Only {len(points)} valid surface samples for {uid}")
    rng = np.random.default_rng(_seed(uid))
    replace = len(points) < point_count
    indices = rng.choice(len(points), size=point_count, replace=replace)
    positions_tensor = torch.from_numpy(points[indices].astype(np.float32, copy=False)).contiguous()
    colors_tensor = torch.from_numpy(colors[indices].astype(np.float32, copy=False)).contiguous()
    pointcloud_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointcloud_path.with_name(pointcloud_path.name + ".tmp")
    torch.save((positions_tensor, colors_tensor), temporary)
    os.replace(temporary, pointcloud_path)

    if not keep_intermediates:
        for path in image_dir.iterdir():
            if path.name.endswith("_normal.exr") or path.name.endswith("_albedo.png"):
                path.unlink()
    return {
        "uid": uid,
        "views": len(frames),
        "points": int(point_count),
        "source_surface_points": int(len(points)),
        "position_min": float(positions_tensor.min()),
        "position_max": float(positions_tensor.max()),
        "color_min": float(colors_tensor.min()),
        "color_max": float(colors_tensor.max()),
        "foreground_fraction_min": min(foreground_fractions),
        "foreground_fraction_max": max(foreground_fractions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--pointcloud", type=Path, required=True)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--point-count", type=int, default=200_000)
    parser.add_argument("--remove-intermediates", action="store_true")
    args = parser.parse_args()
    report = postprocess(
        args.image_dir,
        args.pointcloud,
        uid=args.uid,
        point_count=args.point_count,
        keep_intermediates=not args.remove_intermediates,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
