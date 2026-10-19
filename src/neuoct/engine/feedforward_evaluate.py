from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from neuoct.config.schema import DataConfig, ModelConfig
from neuoct.data.camera import unproject_rgbd
from neuoct.data.dataset import ShapeNetMultiViewDataset
from neuoct.data.preprocess import DeviceBatchPreprocessor
from neuoct.engine.checkpoint import load_model_checkpoint
from neuoct.geometry.octree_adapter import build_input_features
from neuoct.metrics import (
    color_detail_metrics,
    geometry_noise_metrics,
    hit_classification_metrics,
    hit_topology_metrics,
    image_psnr,
    legacy_local_view_metrics,
    masked_psnr,
    normal_detail_metrics,
)
from neuoct.models import NeuralIntersectionModel
from neuoct.rendering import SparseRenderer, save_render_output


@dataclass(frozen=True)
class FeedforwardEvaluationResult:
    output: Path
    view_count: int
    object_count: int
    metrics: dict[str, float]


def _merge_config(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    """Recursively merge YAML mappings, with ``override`` taking precedence."""
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_config_with_bases(config_path: Path, seen: set[Path] | None = None) -> dict[str, object]:
    """Load a config using the same BASE semantics as THSolver."""
    path = config_path.expanduser().resolve()
    active = set() if seen is None else set(seen)
    if path in active:
        raise ValueError(f"Cyclic BASE config reference: {path}")
    active.add(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    bases = payload.get("BASE", [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list):
        raise ValueError(f"BASE must be a list in {path}")
    merged: dict[str, object] = {}
    for base in bases:
        base_path = Path(str(base)).expanduser()
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = _merge_config(merged, _load_config_with_bases(base_path, active))
    payload = dict(payload)
    payload.pop("BASE", None)
    return _merge_config(merged, payload)


def load_feedforward_config(
    config_path: str | Path, *, data_split: str = "test"
) -> tuple[DataConfig, ModelConfig, dict[str, object]]:
    payload = _load_config_with_bases(Path(config_path))
    data_payload = payload.get("DATA")
    model_payload = payload.get("MODEL")
    if not isinstance(data_payload, dict) or data_split not in data_payload:
        raise ValueError(f"Configuration does not define DATA.{data_split}")
    if not isinstance(model_payload, dict):
        raise ValueError("Configuration does not define MODEL")
    data = DataConfig.from_flags(data_payload[data_split])
    model = ModelConfig.from_flags(model_payload)
    data.validate(require_paths=True)
    model.validate()
    return data, model, payload


def load_feedforward_model(
    checkpoint: str | Path,
    config: ModelConfig,
    *,
    device: torch.device,
) -> NeuralIntersectionModel:
    model = NeuralIntersectionModel(
        config.input_channels,
        config.query_depth,
        renderer_hidden_dim=config.renderer_hidden_dim,
        renderer_iterations=config.renderer_iterations,
        segment_position=config.segment_position,
        voxel_feature_dim=config.voxel_feature_dim,
        color_skip_weight=config.color_skip_weight,
        material_fp32=config.material_fp32,
        norm_type=config.norm_type,
        norm_groups=config.norm_groups,
        sampling_mode=config.sampling_mode,
    ).to(device)
    load_model_checkpoint(model, checkpoint, device=device, strict=True)
    return model.eval()


def _save_comparison(
    path: Path,
    target_color: torch.Tensor,
    predicted_color: torch.Tensor,
    target_normal: torch.Tensor,
    predicted_normal: torch.Tensor,
    target_hit: torch.Tensor,
    predicted_hit: torch.Tensor,
) -> None:
    hit_target = target_hit.unsqueeze(-1).expand_as(target_color).float()
    hit_prediction = predicted_hit.unsqueeze(-1).expand_as(target_color).float()
    tiles = [
        target_color,
        predicted_color,
        (target_normal + 1.0) * 0.5 * hit_target,
        (predicted_normal + 1.0) * 0.5 * hit_prediction,
        hit_target,
        hit_prediction,
    ]
    image = (torch.cat(tiles, dim=1).clamp(0.0, 1.0) * 255.0).round().byte()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image.cpu().numpy())).save(path)


def _mean_metrics(rows: list[dict[str, float | str | int]]) -> dict[str, float]:
    excluded = {"entry", "object_key", "frame_id"}
    names = [name for name in rows[0] if name not in excluded]
    return {
        name: sum(float(row[name]) for row in rows) / len(rows)
        for name in names
    }


def _component_hit_metrics(
    predicted_hit: torch.Tensor,
    target_hit: torch.Tensor,
) -> dict[str, float]:
    """Count disconnected predicted islands and enclosed holes in image space."""
    predicted = np.asarray(predicted_hit.detach().cpu(), dtype=np.uint8)
    target = np.asarray(target_hit.detach().cpu(), dtype=np.uint8)
    target_dilated = cv2.dilate(target, np.ones((3, 3), np.uint8), iterations=1)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        predicted, connectivity=8
    )
    floater_components = 0
    floater_pixels = 0
    for component_id in range(1, component_count):
        component = labels == component_id
        if not bool((component & target_dilated.astype(bool)).any()):
            floater_components += 1
            floater_pixels += int(stats[component_id, cv2.CC_STAT_AREA])

    missing = ((target > 0) & (predicted == 0)).astype(np.uint8)
    missing_count, missing_labels, missing_stats, _ = cv2.connectedComponentsWithStats(
        missing, connectivity=8
    )
    hole_components = 0
    hole_pixels = 0
    height, width = target.shape
    for component_id in range(1, missing_count):
        component = missing_labels == component_id
        rows, columns = np.nonzero(component)
        touches_image_border = bool(
            (rows == 0).any()
            or (rows == height - 1).any()
            or (columns == 0).any()
            or (columns == width - 1).any()
        )
        boundary = cv2.dilate(component.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        enclosed_by_prediction = bool((predicted[boundary & ~component] > 0).all())
        if not touches_image_border and enclosed_by_prediction:
            hole_components += 1
            hole_pixels += int(missing_stats[component_id, cv2.CC_STAT_AREA])
    predicted_pixels = max(int(predicted.sum()), 1)
    target_pixels = max(int(target.sum()), 1)
    return {
        "floater_component_count": float(floater_components),
        "floater_component_pixel_rate": floater_pixels / predicted_pixels,
        "enclosed_hole_count": float(hole_components),
        "enclosed_hole_pixel_rate": hole_pixels / target_pixels,
    }


@torch.no_grad()
def evaluate_feedforward_checkpoint(
    config_path: str | Path,
    checkpoint: str | Path,
    *,
    output: str | Path,
    data_split: str = "test",
    device: str = "cuda",
    hit_threshold: float = 0.5,
    max_views: int = -1,
    save_renders: int = 8,
    orient_normals: bool = False,
) -> FeedforwardEvaluationResult:
    data, model_config, _ = load_feedforward_config(config_path, data_split=data_split)
    resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = load_feedforward_model(checkpoint, model_config, device=resolved_device)
    dataset = ShapeNetMultiViewDataset(data)
    count = len(dataset) if max_views < 1 else min(len(dataset), max_views)
    preprocessor = DeviceBatchPreprocessor(data, resolved_device)
    output_root = Path(output)
    output_root.mkdir(parents=True, exist_ok=True)
    per_view: list[dict[str, float | str | int]] = []
    object_rows: dict[str, list[dict[str, float | str | int]]] = defaultdict(list)
    active_object = ""
    renderer: SparseRenderer | None = None
    started = perf_counter()

    for view_index in range(count):
        sample = dataset[view_index]
        if sample.entry.object_key != active_object:
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
            active_object = sample.entry.object_key
        if renderer is None:
            raise RuntimeError("Feedforward renderer was not initialized")

        rendered = renderer.render_frame(sample.camera_to_world, sample.intrinsics)
        target_color = sample.color_image.to(resolved_device).float() / 255.0
        normal_image = sample.normal_image.to(resolved_device)
        target_normal = (
            normal_image.float()
            if normal_image.is_floating_point()
            else normal_image.float() / 255.0 * 2.0 - 1.0
        )
        target_hit = sample.valid_depth.to(resolved_device)
        _, _, target_points, target_normals, target_colors = unproject_rgbd(
            sample.depth_image.to(resolved_device),
            sample.normal_image.to(resolved_device),
            sample.color_image.to(resolved_device),
            sample.intrinsics,
            sample.camera_to_world.to(resolved_device),
            convention=data.camera_convention,
        )
        hit_metrics = hit_classification_metrics(rendered.hit, target_hit)
        topology_metrics = hit_topology_metrics(rendered.hit, target_hit)
        component_metrics = _component_hit_metrics(rendered.hit, target_hit)
        noise_metrics = geometry_noise_metrics(
            rendered.normal,
            target_normal,
            rendered.hit,
            target_hit,
        )
        detail_metrics = normal_detail_metrics(
            rendered.normal,
            target_normal,
            rendered.hit,
            target_hit,
        )
        color_metrics = color_detail_metrics(rendered.color, target_color, target_hit)
        legacy_metrics = legacy_local_view_metrics(
            rendered,
            target_points,
            target_normals,
            target_colors,
            target_hit,
            point_scale=data.point_scale,
            ray_scale=data.ray_scale,
            include_color=True,
            foreground_only=True,
        )
        metrics: dict[str, float | str | int] = {
            "entry": sample.entry.relative_name,
            "object_key": sample.entry.object_key,
            "frame_id": sample.entry.frame_id,
            "color_psnr": float(image_psnr(rendered.color, target_color).cpu()),
            "masked_color_psnr": float(
                masked_psnr(rendered.color, target_color, target_hit).cpu()
            ),
            "masked_normal_psnr": float(
                masked_psnr(
                    (rendered.normal + 1.0) * 0.5,
                    (target_normal + 1.0) * 0.5,
                    target_hit,
                ).cpu()
            ),
            **{name: float(value.cpu()) for name, value in hit_metrics.items()},
            **{name: float(value.cpu()) for name, value in topology_metrics.items()},
            **component_metrics,
            **{name: float(value.cpu()) for name, value in noise_metrics.items()},
            **{name: float(value.cpu()) for name, value in detail_metrics.items()},
            **{name: float(value.cpu()) for name, value in color_metrics.items()},
            **{name: float(value.cpu()) for name, value in legacy_metrics.items()},
        }
        per_view.append(metrics)
        object_rows[sample.entry.object_key].append(metrics)
        if view_index < save_renders:
            render_dir = output_root / "render" / sample.entry.category / sample.entry.object_id
            save_render_output(rendered, render_dir, f"{sample.entry.frame_id:03d}")
            _save_comparison(
                render_dir / f"{sample.entry.frame_id:03d}_comparison.png",
                target_color,
                rendered.color,
                target_normal,
                rendered.normal,
                target_hit,
                rendered.hit,
            )
        print(
            f"view={view_index + 1}/{count} entry={sample.entry.relative_name} "
            f"masked_color_psnr={float(metrics['masked_color_psnr']):.3f} "
            f"hit_iou={float(metrics['hit_iou']):.3f}",
            flush=True,
        )

    per_object = {
        object_key: _mean_metrics(rows)
        for object_key, rows in object_rows.items()
    }
    view_mean = _mean_metrics(per_view)
    object_mean = {
        name: sum(row[name] for row in per_object.values()) / len(per_object)
        for name in next(iter(per_object.values()))
    }
    report = {
        "schema_version": 1,
        "config": str(Path(config_path).resolve()),
        "checkpoint": str(Path(checkpoint).resolve()),
        "data_split": data_split,
        "manifest": str(data.manifest.resolve()),
        "view_count": len(per_view),
        "object_count": len(per_object),
        "hit_threshold": hit_threshold,
        "orient_normals": orient_normals,
        "elapsed_seconds": perf_counter() - started,
        "mean": view_mean,
        "object_mean": object_mean,
        "per_object": per_object,
        "per_view": per_view,
    }
    (output_root / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return FeedforwardEvaluationResult(
        output_root,
        len(per_view),
        len(per_object),
        view_mean,
    )
