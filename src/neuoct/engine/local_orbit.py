from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from neuoct.data.dataset import ShapeNetMultiViewDataset
from neuoct.data.preprocess import DeviceBatchPreprocessor
from neuoct.geometry.vertex_index import compute_vertex_index
from neuoct.rendering import SparseRenderer, generate_orbit_views, save_render_output

from .local_checkpoint import (
    get_local_checkpoint_schema,
    load_local_checkpoint,
    load_local_inference_checkpoint,
)
from .local_optimize import LocalOptimizationConfig


def _video_codec_spec(codec: str) -> tuple[str, str]:
    normalized = codec.lower()
    if normalized == "mp4v":
        return "mp4v", ".mp4"
    if normalized == "ffv1":
        return "FFV1", ".mkv"
    raise ValueError("video_codec must be 'mp4v' or 'ffv1'")


@torch.no_grad()
def render_local_orbit(
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
    up_axis: str | None = None,
    hit_threshold: float = 0.5,
    orient_normals: bool = False,
    supersample_factor: int = 1,
    supersample_hit_fraction: float = 0.5,
    video_codec: str = "mp4v",
) -> dict[str, object]:
    if views <= 0 or fps <= 0:
        raise ValueError("views and fps must be positive")
    holder = LocalOptimizationConfig.from_yaml(
        config_path,
        object_key=object_key,
        output=output,
        data_split=data_split,
        epochs=1,
        views=1,
        device=device,
    )
    resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
    dataset = ShapeNetMultiViewDataset.for_object(holder.data, object_key, take=1)
    schema = get_local_checkpoint_schema(checkpoint)
    if schema == 2:
        model, octree, payload = load_local_inference_checkpoint(
            checkpoint, device=resolved_device, expected_object=object_key
        )
    else:
        preprocessor = DeviceBatchPreprocessor(holder.data, resolved_device)
        octree = preprocessor.build_octree(dataset[0])
        index = compute_vertex_index(octree, holder.data.query_depth, include_batch=True)
        model, payload = load_local_checkpoint(
            checkpoint, device=resolved_device, expected_object=object_key
        )
        if not torch.equal(model.feature_field.vertex_coordinates, index.coordinates.int()):
            raise ValueError("Orbit octree vertices differ from checkpoint topology")
        if not torch.equal(model.feature_field.voxel_vertices, index.voxel_vertices.long()):
            raise ValueError("Orbit voxel-to-vertex mapping differs from checkpoint topology")
    if int(payload["query_depth"]) != holder.data.query_depth:
        raise ValueError("Checkpoint and orbit configuration query depths differ")
    model_config = payload.get("model_config", {})
    geometry_only = bool(
        isinstance(model_config, dict) and model_config.get("geometry_only", False)
    )
    model.eval()
    renderer = SparseRenderer(
        model.renderer_model,
        model.feature_field,
        octree,
        depth=holder.data.query_depth,
        point_scale=holder.data.point_scale,
        ray_scale=holder.data.ray_scale,
        camera_convention=holder.data.camera_convention,
        hit_threshold=hit_threshold,
        orient_normals_to_camera=orient_normals,
        supersample_factor=supersample_factor,
        supersample_hit_fraction=supersample_hit_fraction,
    )
    output_root = Path(output)
    output_root.mkdir(parents=True, exist_ok=True)
    frame_dir = output_root / "frames"
    resolved_up_axis = holder.data.orbit_up_axis if up_axis is None else up_axis
    orbit = generate_orbit_views(
        views,
        radius=radius,
        elevation_radians=torch.deg2rad(torch.tensor(elevation_degrees, device=resolved_device)),
        camera_convention=holder.data.camera_convention,
        up_axis=resolved_up_axis,
        device=resolved_device,
    )
    height, width = holder.data.image_height, holder.data.image_width
    primary_name = "normal" if geometry_only else "color"
    fourcc_name, extension = _video_codec_spec(video_codec)
    primary_path = output_root / f"orbit_{primary_name}{extension}"
    composite_path = output_root / f"orbit_composite{extension}"
    primary_writer = cv2.VideoWriter(
        str(primary_path), cv2.VideoWriter_fourcc(*fourcc_name), fps, (width, height)
    )
    composite_width = width * (3 if geometry_only else 4)
    composite_writer = cv2.VideoWriter(
        str(composite_path), cv2.VideoWriter_fourcc(*fourcc_name), fps, (composite_width, height)
    )
    if not primary_writer.isOpened() or not composite_writer.isOpened():
        primary_writer.release()
        composite_writer.release()
        raise RuntimeError(f"OpenCV could not open a {video_codec} video writer")
    encode_started = perf_counter()
    try:
        for view_id, transform in enumerate(orbit):
            rendered = renderer.render_frame(transform, dataset[0].intrinsics)
            tiles = rendered.uint8_tiles()
            primary = np.asarray(tiles[primary_name].cpu())
            composite = np.asarray(
                rendered.composite(include_color=not geometry_only).cpu()
            )
            primary_writer.write(cv2.cvtColor(primary, cv2.COLOR_RGB2BGR))
            composite_writer.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            if view_id in (0, views // 2):
                save_render_output(
                    rendered,
                    frame_dir,
                    f"orbit_{view_id:03d}",
                    include_color=not geometry_only,
                )
    finally:
        primary_writer.release()
        composite_writer.release()
    encode_elapsed = perf_counter() - encode_started
    report = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "object_key": object_key,
        "view_count": views,
        "fps": fps,
        "duration_seconds": views / fps,
        "frame_size": [width, height],
        "geometry_only": geometry_only,
        "up_axis": resolved_up_axis,
        "hit_threshold": hit_threshold,
        "orient_normals": orient_normals,
        "supersample_factor": supersample_factor,
        "supersample_hit_fraction": supersample_hit_fraction,
        "video_codec": video_codec.lower(),
        "video_encode_seconds": encode_elapsed,
        "primary_video_bytes": primary_path.stat().st_size,
        "composite_video_bytes": composite_path.stat().st_size,
        "primary_video": str(primary_path),
        "composite_video": str(composite_path),
    }
    report[f"{primary_name}_video"] = str(primary_path)
    (output_root / "orbit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
