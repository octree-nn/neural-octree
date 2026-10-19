from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from neuoct.data.dataset import ShapeNetMultiViewDataset
from neuoct.data.preprocess import DeviceBatchPreprocessor
from neuoct.geometry.octree_adapter import build_input_features
from neuoct.rendering import SparseRenderer, generate_orbit_views, save_render_output

from .feedforward_evaluate import load_feedforward_config, load_feedforward_model
from .local_orbit import _video_codec_spec


def saved_frame_views(views: int, frame_count: int = 8) -> set[int]:
    """Evenly spaced view indices whose full-resolution PNG tiles are kept.

    Diagnostics (hole/leak forensics) need several distinct viewpoints rather than
    just the first and the middle one, so the default keeps eight of them.
    """
    if views <= 0:
        raise ValueError("views must be positive")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    count = min(int(frame_count), int(views))
    if count == 1:
        return {0}
    indices = np.linspace(0, views - 1, count)
    return {int(round(value)) for value in indices}


@torch.no_grad()
def render_feedforward_orbit(
    config_path: str | Path,
    checkpoint: str | Path,
    *,
    object_key: str,
    output: str | Path,
    data_split: str = "test",
    views: int = 120,
    fps: int = 24,
    device: str = "cuda",
    radius: float = 2.5,
    elevation_degrees: float = 18.0,
    hit_threshold: float = 0.5,
    orient_normals: bool = False,
    video_codec: str = "mp4v",
    frame_count: int = 8,
    mesh_octree: str | Path | None = None,
) -> dict[str, object]:
    if views <= 0 or fps <= 0:
        raise ValueError("views and fps must be positive")
    data, model_config, _ = load_feedforward_config(config_path, data_split=data_split)
    resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = load_feedforward_model(checkpoint, model_config, device=resolved_device)
    dataset = ShapeNetMultiViewDataset.for_object(data, object_key, take=1)
    sample = dataset[0]
    preprocessor = DeviceBatchPreprocessor(data, resolved_device)
    mesh_stats = None
    if mesh_octree is not None:
        # coverage-complete octree built from the pipeline-frame mesh instead of
        # the (8-view back-projected) point cloud
        from neuoct.geometry.mesh_octree import build_mesh_octree

        octree, _cells, mesh_stats = build_mesh_octree(
            mesh_octree,
            query_depth=data.query_depth,
            octree_depth=data.octree_depth,
            full_depth=data.full_depth,
            ray_scale=data.ray_scale,
            point_scale=data.point_scale,
            device=resolved_device,
            color_source=(
                sample.point_positions.to(resolved_device) * data.point_scale,
                sample.point_colors.to(resolved_device).float()
                / (255.0 if float(sample.point_colors.max()) > 1.5 else 1.0),
            ),
        )
        print(f"[INFO] mesh octree: {mesh_stats}", flush=True)
    else:
        octree = preprocessor.build_octree(sample)
    input_features = build_input_features(octree, model_config.feature_spec)
    _, field = model.extract_scene_features(input_features, octree, octree.depth)
    renderer = SparseRenderer(
        model.renderer_model,
        field,
        octree,
        depth=data.query_depth,
        point_scale=data.point_scale,
        ray_scale=data.ray_scale,
        camera_convention=data.camera_convention,
        hit_threshold=hit_threshold,
        orient_normals_to_camera=orient_normals,
    )
    orbit = generate_orbit_views(
        views,
        radius=radius,
        elevation_radians=torch.deg2rad(
            torch.tensor(elevation_degrees, device=resolved_device)
        ),
        camera_convention=data.camera_convention,
        up_axis=data.orbit_up_axis,
        device=resolved_device,
    )
    output_root = Path(output)
    output_root.mkdir(parents=True, exist_ok=True)
    frame_dir = output_root / "frames"
    fourcc_name, extension = _video_codec_spec(video_codec)
    color_path = output_root / f"orbit_color{extension}"
    composite_path = output_root / f"orbit_composite{extension}"
    width, height = data.image_width, data.image_height
    color_writer = cv2.VideoWriter(
        str(color_path), cv2.VideoWriter_fourcc(*fourcc_name), fps, (width, height)
    )
    composite_writer = cv2.VideoWriter(
        str(composite_path),
        cv2.VideoWriter_fourcc(*fourcc_name),
        fps,
        (width * 4, height),
    )
    if not color_writer.isOpened() or not composite_writer.isOpened():
        color_writer.release()
        composite_writer.release()
        raise RuntimeError(f"OpenCV could not open a {video_codec} video writer")
    started = perf_counter()
    frame_views = saved_frame_views(views, frame_count)
    try:
        for view_id, transform in enumerate(orbit):
            rendered = renderer.render_frame(transform, sample.intrinsics)
            color = np.asarray(rendered.uint8_tiles()["color"].cpu())
            composite = np.asarray(rendered.composite().cpu())
            color_writer.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            composite_writer.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            if view_id in frame_views:
                save_render_output(rendered, frame_dir, f"orbit_{view_id:03d}")
            print(f"orbit={view_id + 1}/{views} object={object_key}", flush=True)
    finally:
        color_writer.release()
        composite_writer.release()
    report = {
        "schema_version": 1,
        "config": str(Path(config_path).resolve()),
        "checkpoint": str(Path(checkpoint).resolve()),
        "object_key": object_key,
        "view_count": views,
        "fps": fps,
        "duration_seconds": views / fps,
        "frame_size": [width, height],
        "up_axis": data.orbit_up_axis,
        "hit_threshold": hit_threshold,
        "orient_normals": orient_normals,
        "video_codec": video_codec.lower(),
        "mesh_octree": mesh_stats,
        "frame_count": len(frame_views),
        "frame_views": sorted(frame_views),
        "render_seconds": perf_counter() - started,
        "color_video": str(color_path),
        "composite_video": str(composite_path),
        "color_video_bytes": color_path.stat().st_size,
        "composite_video_bytes": composite_path.stat().st_size,
    }
    (output_root / "orbit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
