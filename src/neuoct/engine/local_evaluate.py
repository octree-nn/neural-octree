from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from neuoct.data.camera import unproject_rgbd
from neuoct.data.dataset import ShapeNetMultiViewDataset
from neuoct.data.preprocess import DeviceBatchPreprocessor
from neuoct.geometry.vertex_index import compute_vertex_index
from neuoct.metrics import (
    geometry_noise_metrics,
    hit_classification_metrics,
    image_psnr,
    legacy_local_view_metrics,
    masked_psnr,
    normal_detail_metrics,
)
from neuoct.rendering import SparseRenderer, save_render_output

from .local_checkpoint import (
    get_local_checkpoint_schema,
    load_local_checkpoint,
    load_local_inference_checkpoint,
)
from .local_optimize import LocalOptimizationConfig


@dataclass(frozen=True)
class LocalEvaluationResult:
    output: Path
    view_count: int
    metrics: dict[str, float]


def _save_comparison(
    path: Path,
    target_color: torch.Tensor,
    predicted_color: torch.Tensor,
    target_normal: torch.Tensor,
    predicted_normal: torch.Tensor,
    target_hit: torch.Tensor,
    predicted_hit: torch.Tensor,
    *,
    include_color: bool = True,
) -> None:
    hit_target = target_hit.unsqueeze(-1).expand_as(target_color).float()
    hit_prediction = predicted_hit.unsqueeze(-1).expand_as(target_color).float()
    tiles = []
    if include_color:
        tiles.extend([target_color, predicted_color])
    tiles.extend(
        [
            (target_normal + 1.0) * 0.5 * hit_target,
            (predicted_normal + 1.0) * 0.5 * hit_prediction,
            hit_target,
            hit_prediction,
        ]
    )
    image = (torch.cat(tiles, dim=1).clamp(0.0, 1.0) * 255.0).round().byte()
    Image.fromarray(np.asarray(image.cpu().numpy())).save(path)


@torch.no_grad()
def evaluate_local_checkpoint(
    config_path: str | Path,
    checkpoint: str | Path,
    *,
    object_key: str,
    output: str | Path,
    data_split: str = "test",
    views: int = -1,
    device: str = "cuda",
    hit_threshold: float = 0.5,
    orient_normals: bool = False,
    supersample_factor: int = 1,
    supersample_hit_fraction: float = 0.5,
    topology_diagnostics: bool = False,
    foreground_only_metrics: bool = True,
    normal_source: str | None = None,
) -> LocalEvaluationResult:
    """Render and score a local checkpoint on a configured set of object views."""
    holder = LocalOptimizationConfig.from_yaml(
        config_path,
        object_key=object_key,
        output=output,
        data_split=data_split,
        epochs=1,
        views=views,
        device=device,
        normal_source=normal_source,
    )
    resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
    dataset = ShapeNetMultiViewDataset.for_object(holder.data, object_key, take=views)
    preprocessor = DeviceBatchPreprocessor(holder.data, resolved_device)
    schema = get_local_checkpoint_schema(checkpoint)
    if schema == 2:
        model, octree, payload = load_local_inference_checkpoint(
            checkpoint, device=resolved_device, expected_object=object_key
        )
    else:
        octree = preprocessor.build_octree(dataset[0])
        current_index = compute_vertex_index(octree, holder.data.query_depth, include_batch=True)
        model, payload = load_local_checkpoint(
            checkpoint, device=resolved_device, expected_object=object_key
        )
        if not torch.equal(model.feature_field.vertex_coordinates, current_index.coordinates.int()):
            raise ValueError("Evaluation octree vertices differ from the checkpoint topology")
        if not torch.equal(model.feature_field.voxel_vertices, current_index.voxel_vertices.long()):
            raise ValueError("Evaluation voxel-to-vertex mapping differs from the checkpoint topology")
    if int(payload["query_depth"]) != holder.data.query_depth:
        raise ValueError("Checkpoint and evaluation query depths differ")
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
    per_view: list[dict[str, float | int]] = []
    for view_index in range(len(dataset)):
        sample = dataset[view_index]
        rendered = renderer.render_frame(sample.camera_to_world, sample.intrinsics)
        save_render_output(
            rendered,
            output_root / "render",
            f"{view_index:03d}",
            include_color=not geometry_only,
        )
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
            convention=holder.data.camera_convention,
        )
        hit_metrics = hit_classification_metrics(rendered.hit, target_hit)
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
        legacy_metrics = legacy_local_view_metrics(
            rendered,
            target_points,
            target_normals,
            target_colors,
            target_hit,
            point_scale=holder.data.point_scale,
            ray_scale=holder.data.ray_scale,
            include_color=not geometry_only,
            foreground_only=foreground_only_metrics,
        )
        metrics: dict[str, float | int] = {
            "view": view_index,
            "masked_normal_psnr": float(
                masked_psnr(
                    (rendered.normal + 1.0) * 0.5,
                    (target_normal + 1.0) * 0.5,
                    target_hit,
                ).cpu()
            ),
            **{name: float(value.cpu()) for name, value in hit_metrics.items()},
            **{name: float(value.cpu()) for name, value in noise_metrics.items()},
            **{name: float(value.cpu()) for name, value in detail_metrics.items()},
            **{name: float(value.cpu()) for name, value in legacy_metrics.items()},
        }
        if topology_diagnostics:
            pairs = preprocessor.prepare_sample(sample, octree)
            if pairs.hits is None:
                raise RuntimeError("Topology diagnostics require supervised hit labels")
            topology_hit = torch.zeros_like(target_hit).reshape(-1)
            positive_labels = torch.unique(pairs.ray_labels[pairs.hits.bool()].long())
            topology_hit[positive_labels] = True
            topology_hit = topology_hit.reshape_as(target_hit)
            topology_target = topology_hit & target_hit
            target_count = target_hit.sum().float()
            topology_count = topology_target.sum().float()
            metrics.update(
                {
                    "topology_recall": float(
                        (topology_count / (target_count + 1.0e-8)).cpu()
                    ),
                    "model_hit_recall_given_topology": float(
                        (
                            (rendered.hit & topology_target).sum().float()
                            / (topology_count + 1.0e-8)
                        ).cpu()
                    ),
                }
            )
        if not geometry_only:
            metrics.update(
                {
                    "color_psnr": float(image_psnr(rendered.color, target_color).cpu()),
                    "masked_color_psnr": float(
                        masked_psnr(rendered.color, target_color, target_hit).cpu()
                    ),
                }
            )
        per_view.append(metrics)
        _save_comparison(
            output_root / "render" / f"{view_index:03d}_comparison.png",
            target_color,
            rendered.color,
            target_normal,
            rendered.normal,
            target_hit,
            rendered.hit,
            include_color=not geometry_only,
        )
        quality_name = "masked_normal_psnr" if geometry_only else "masked_color_psnr"
        print(
            f"view={view_index}/{len(dataset) - 1} "
            f"{quality_name}={float(metrics[quality_name]):.3f} "
            f"hit_iou={float(metrics['hit_iou']):.3f}",
            flush=True,
        )
    names = [name for name in per_view[0] if name != "view"]
    mean = {name: sum(float(row[name]) for row in per_view) / len(per_view) for name in names}
    report = {
        "object_key": object_key,
        "checkpoint": str(Path(checkpoint).resolve()),
        "data_split": data_split,
        "view_count": len(dataset),
        "geometry_only": geometry_only,
        "hit_threshold": hit_threshold,
        "orient_normals": orient_normals,
        "supersample_factor": supersample_factor,
        "supersample_hit_fraction": supersample_hit_fraction,
        "topology_diagnostics": topology_diagnostics,
        "foreground_only_metrics": foreground_only_metrics,
        "normal_source": holder.data.normal_source,
        "mean": mean,
        "per_view": per_view,
    }
    (output_root / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return LocalEvaluationResult(output_root, len(dataset), mean)
