from __future__ import annotations

import json
import math
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Any

import torch
import yaml
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from neuoct.config.schema import DataConfig
from neuoct.data.dataset import ShapeNetMultiViewDataset
from neuoct.data.preprocess import DeviceBatchPreprocessor
from neuoct.data.sample import RawSample
from neuoct.geometry.ray_traversal import RayVoxelPairs
from neuoct.geometry.vertex_index import compute_vertex_index
from neuoct.losses import IntersectionObjective, IntersectionTarget, ObjectiveWeights
from neuoct.models.local_scene import LocalSceneModel
from neuoct.rendering import SparseRenderer, save_render_output

from .local_checkpoint import save_local_checkpoint


@dataclass(frozen=True)
class LocalOptimizationConfig:
    data: DataConfig
    object_key: str
    output: Path
    epochs: int = 1
    views: int = -1
    ray_batch_size: int = 65536
    feature_dim: int = 32
    hidden_dim: int = 128
    n_iters: int = 8
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.0e-3
    lr_schedule: str = "cosine"
    lr_min: float = 1.0e-5
    warmup_epochs: int = 0
    lr_step_epochs: int = 0
    lr_gamma: float = 0.5
    init_std: float = 0.01
    seed: int = 0
    device: str = "cuda"
    amp: bool = False
    render: bool = False
    max_steps: int = -1
    checkpoint_every: int = 10
    views_per_step: int = 1
    cache_supervision: bool = False
    pair_balanced_ddp: bool = False
    ddp_static: bool = False
    tf32: bool = False
    compile_renderer: bool = False
    compile_mode: str = "default"
    geometry_only: bool = False
    positive_geometry_loss: bool = False
    hit_loss_mode: str = "bce"
    focal_gamma: float = 2.0
    normal_cosine_weight: float = 0.0
    stage_weight_mode: str = "legacy"
    final_stage_weight: float = 0.5
    segment_position: bool = False
    voxel_feature_dim: int = 0
    position_encoding_frequencies: int = 0
    normal_gradient_weight: float = 0.0
    detail_weight_scale: float = 0.0
    masked_normal_ssim: bool = False
    sampling_mode: str = "recurrent"

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.views == 0 or self.views < -1:
            raise ValueError("views must be -1 or positive")
        if self.ray_batch_size == 0 or self.ray_batch_size < -1:
            raise ValueError("ray_batch_size must be -1 or positive")
        if self.feature_dim <= 0 or self.hidden_dim <= 0 or self.n_iters <= 0:
            raise ValueError("feature_dim, hidden_dim and n_iters must be positive")
        if self.voxel_feature_dim < 0 or self.position_encoding_frequencies < 0:
            raise ValueError(
                "voxel_feature_dim and position_encoding_frequencies must be non-negative"
            )
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or positive")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if self.views_per_step <= 0:
            raise ValueError("views_per_step must be positive")
        if self.hit_loss_mode not in ("bce", "balanced_bce", "balanced_focal"):
            raise ValueError(
                "hit_loss_mode must be 'bce', 'balanced_bce' or 'balanced_focal'"
            )
        if self.focal_gamma < 0 or self.normal_cosine_weight < 0:
            raise ValueError("focal_gamma and normal_cosine_weight must be non-negative")
        if self.normal_gradient_weight < 0 or self.detail_weight_scale < 0:
            raise ValueError("normal gradient and detail weights must be non-negative")
        if self.stage_weight_mode not in ("legacy", "final_emphasis"):
            raise ValueError("stage_weight_mode must be 'legacy' or 'final_emphasis'")
        if not 0.0 < self.final_stage_weight <= 1.0:
            raise ValueError("final_stage_weight must be in (0, 1]")
        if self.sampling_mode not in (
            "recurrent",
            "uniform",
            "single",
            "recurrent_endpoints",
            "uniform_endpoints",
            "uniform_concat",
            "uniform_concat_endpoints",
            "recurrent_pool",
            "single_endpoints",
        ):
            raise ValueError(
                "sampling_mode must be 'recurrent', 'uniform', 'single', "
                "'recurrent_endpoints', 'uniform_endpoints', 'uniform_concat', "
                "'uniform_concat_endpoints', 'recurrent_pool' or 'single_endpoints'"
            )
        if self.compile_mode not in (
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ):
            raise ValueError(
                "compile_mode must be 'default', 'reduce-overhead', 'max-autotune' "
                "or 'max-autotune-no-cudagraphs'"
            )
        if self.compile_mode in ("reduce-overhead", "max-autotune"):
            # Measured 2026-09-15 on the remote 4090 (Armadillo depth 7, F=16/H=194, ws=1):
            #   RuntimeError: Error: accessing tensor output of CUDAGraphs that has been
            #   overwritten by a subsequent run.  (intersection_renderer.py:800, forward)
            # The renderer is invoked once per ray chunk, so CUDA-graph replay reuses the
            # output buffer before the previous chunk's backward has consumed it.  Fail
            # fast here instead of dying ~90 s into a multi-hour run.
            raise ValueError(
                f"compile_mode={self.compile_mode!r} captures CUDA graphs, which is "
                "unsupported for this renderer (it is invoked once per ray chunk, so a "
                "graph output is overwritten before its backward runs). Use 'default' "
                "or 'max-autotune-no-cudagraphs'."
            )
        if self.lr_schedule not in (
            "cosine",
            "linear",
            "step",
            "constant",
            "warmup_cosine",
        ):
            raise ValueError(
                "lr_schedule must be 'cosine', 'linear', 'step', 'constant' "
                "or 'warmup_cosine'"
            )
        if self.lr_min < 0:
            raise ValueError("lr_min must be non-negative")
        if self.lr_min > self.learning_rate:
            raise ValueError("lr_min must not exceed learning_rate")
        if self.warmup_epochs < 0 or self.warmup_epochs >= self.epochs:
            raise ValueError("warmup_epochs must be in [0, epochs)")
        if self.lr_schedule == "step" and self.lr_step_epochs <= 0:
            raise ValueError("lr_step_epochs must be positive for the 'step' schedule")
        if not 0.0 < self.lr_gamma <= 1.0:
            raise ValueError("lr_gamma must be in (0, 1]")

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        object_key: str,
        output: str | Path,
        data_split: str = "train",
        device: str = "cuda",
        amp: bool = False,
        epochs: int = 1,
        views: int = -1,
        ray_batch_size: int = 65536,
        feature_dim: int = 32,
        hidden_dim: int = 128,
        n_iters: int = 8,
        learning_rate: float = 2.0e-3,
        weight_decay: float = 1.0e-3,
        lr_schedule: str = "cosine",
        lr_min: float = 1.0e-5,
        warmup_epochs: int = 0,
        lr_step_epochs: int = 0,
        lr_gamma: float = 0.5,
        render: bool = False,
        max_steps: int = -1,
        checkpoint_every: int = 10,
        views_per_step: int = 1,
        cache_supervision: bool = False,
        pair_balanced_ddp: bool = False,
        ddp_static: bool = False,
        tf32: bool = False,
        compile_renderer: bool = False,
        compile_mode: str = "default",
        geometry_only: bool = False,
        positive_geometry_loss: bool = False,
        hit_loss_mode: str = "bce",
        focal_gamma: float = 2.0,
        normal_cosine_weight: float = 0.0,
        stage_weight_mode: str = "legacy",
        final_stage_weight: float = 0.5,
        segment_position: bool = False,
        voxel_feature_dim: int = 0,
        position_encoding_frequencies: int = 0,
        normal_gradient_weight: float = 0.0,
        detail_weight_scale: float = 0.0,
        masked_normal_ssim: bool = False,
        normal_source: str | None = None,
        sampling_mode: str = "recurrent",
    ) -> "LocalOptimizationConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            root = yaml.safe_load(handle)
        if not isinstance(root, dict) or not isinstance(root.get("DATA"), dict):
            raise ValueError("Configuration must contain a DATA mapping")
        if data_split not in ("train", "test"):
            raise ValueError("data_split must be 'train' or 'test'")
        data_mapping = root["DATA"].get(data_split)
        if not isinstance(data_mapping, dict):
            raise ValueError(f"Configuration must contain DATA.{data_split}")
        data = DataConfig(
            image_root=Path(str(data_mapping.get("location", ""))).expanduser(),
            pointcloud_root=Path(str(data_mapping.get("pointcloud_root", ""))).expanduser(),
            manifest=Path(str(data_mapping.get("filelist", ""))).expanduser(),
            # BUG FIX (2026-09-15): ``occupancy_root`` / ``occupancy_max_cells`` were never
            # forwarded here, so setting them in the YAML was a SILENT no-op on the
            # single-object path -- the MISSING-hole fix could not be switched on at all.
            # Measured: with occupancy_q7 set, coord_bytes/feature_shape were byte-identical
            # to the run without it.  ``DataConfig.from_flags`` always parsed these keys, so
            # only this hand-rolled constructor was lossy.
            occupancy_root=(
                Path(str(data_mapping["occupancy_root"])).expanduser()
                if str(data_mapping.get("occupancy_root", "") or "").strip()
                else None
            ),
            occupancy_max_cells=int(data_mapping.get("occupancy_max_cells", 0) or 0),
            octree_depth=int(data_mapping.get("depth", 9)),
            full_depth=int(data_mapping.get("full_depth", 4)),
            query_depth=int(data_mapping.get("q_depth", 7)),
            image_width=int(data_mapping.get("image_width", 512)),
            image_height=int(data_mapping.get("image_height", 512)),
            point_scale=float(data_mapping.get("point_scale", 0.99)),
            ray_scale=float(data_mapping.get("ray_scale", 0.5)),
            camera_convention=str(data_mapping.get("camera_convention", "neuoct")),
            orbit_up_axis=str(data_mapping.get("orbit_up_axis", "y")),
            normal_source=(
                str(data_mapping.get("normal_source", "png"))
                if normal_source is None
                else normal_source
            ),
            object_cache_size=int(data_mapping.get("object_cache_size", 8)),
            octree_cache_size=int(data_mapping.get("octree_cache_size", 2)),
            take=-1,
        )
        return cls(
            data=data,
            object_key=object_key,
            output=Path(output),
            epochs=epochs,
            views=views,
            ray_batch_size=ray_batch_size,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            n_iters=n_iters,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_schedule=lr_schedule,
            lr_min=lr_min,
            warmup_epochs=warmup_epochs,
            lr_step_epochs=lr_step_epochs,
            lr_gamma=lr_gamma,
            device=device,
            amp=amp,
            render=render,
            max_steps=max_steps,
            checkpoint_every=checkpoint_every,
            views_per_step=views_per_step,
            cache_supervision=cache_supervision,
            pair_balanced_ddp=pair_balanced_ddp,
            ddp_static=ddp_static,
            tf32=tf32,
            compile_renderer=compile_renderer,
            compile_mode=compile_mode,
            geometry_only=geometry_only,
            positive_geometry_loss=positive_geometry_loss,
            hit_loss_mode=hit_loss_mode,
            focal_gamma=focal_gamma,
            normal_cosine_weight=normal_cosine_weight,
            stage_weight_mode=stage_weight_mode,
            final_stage_weight=final_stage_weight,
            segment_position=segment_position,
            voxel_feature_dim=voxel_feature_dim,
            position_encoding_frequencies=position_encoding_frequencies,
            normal_gradient_weight=normal_gradient_weight,
            detail_weight_scale=detail_weight_scale,
            masked_normal_ssim=masked_normal_ssim,
            sampling_mode=sampling_mode,
        )


@dataclass(frozen=True)
class LocalOptimizationResult:
    checkpoint: Path
    epochs: int
    steps: int
    final_loss: float
    elapsed_seconds: float
    object_key: str
    view_count: int
    vertex_count: int
    world_size: int
    effective_batch_size: int


def _device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(value)


def _distributed_device(value: str) -> tuple[torch.device, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size <= 1:
        return _device(value), 0, 1, False
    if not torch.cuda.is_available() or not value.startswith("cuda"):
        raise RuntimeError("Multi-process local optimization requires CUDA/NCCL")
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        dist.init_process_group(backend="nccl", init_method="env://")
    return torch.device("cuda", local_rank), rank, world_size, owns_process_group


def _rank_view_order(
    order: list[int], *, rank: int, world_size: int, views_per_step: int
) -> list[tuple[int, bool]]:
    """Pad and shard views so every DDP rank executes the same number of steps."""
    per_rank = ceil(len(order) / (world_size * views_per_step)) * views_per_step
    total = per_rank * world_size
    padded = [(index, True) for index in order]
    for index in range(total - len(order)):
        padded.append((order[index % len(order)], False))
    return padded[rank:total:world_size]


def _pair_balanced_view_order(
    order: list[int],
    pair_counts: dict[int, int],
    *,
    rank: int,
    world_size: int,
    views_per_step: int,
) -> list[tuple[int, bool]]:
    """Shard views with a greedy pair-count-balanced DDP assignment.

    Every rank receives the same number of view slots (and therefore the same
    number of optimizer steps).  Real views are assigned exactly once.  If the
    view count is not divisible by ``world_size * views_per_step``, duplicate
    entries are appended as padding and marked ``False`` just like
    :func:`_rank_view_order`.

    ``order`` is an epoch-shuffled list.  Views with similar pair counts are
    placed in the same optimizer step, then greedily partitioned across ranks.
    This matters more than balancing only the epoch total because DDP waits for
    the slowest rank at every backward synchronization.  Step order still
    follows an epoch-dependent key derived from ``order``.
    """
    if world_size <= 0 or views_per_step <= 0:
        raise ValueError("world_size and views_per_step must be positive")
    if not order:
        return []
    missing = [index for index in order if index not in pair_counts]
    if missing:
        raise ValueError(f"pair_counts missing view indices: {missing[:5]}")

    slots_per_rank = ceil(len(order) / (world_size * views_per_step)) * views_per_step
    total_slots = slots_per_rank * world_size
    position = {view_index: offset for offset, view_index in enumerate(order)}
    items = [
        (view_index, True)
        for view_index in sorted(
            order, key=lambda index: (-int(pair_counts[index]), position[index])
        )
    ]
    # Padding entries still participate in forward/backward to keep DDP
    # collectives aligned.  Reusing the cheapest views minimizes padding cost.
    padding_order = sorted(order, key=lambda index: (int(pair_counts[index]), position[index]))
    for offset in range(total_slots - len(items)):
        items.append((padding_order[offset % len(padding_order)], False))

    step_width = world_size * views_per_step
    rank_total_pairs = [0 for _ in range(world_size)]
    rank_steps: list[list[list[tuple[int, bool]]]] = [[] for _ in range(world_size)]
    step_keys: list[int] = []
    for step_start in range(0, len(items), step_width):
        step_items = items[step_start : step_start + step_width]
        bins: list[list[tuple[int, bool]]] = [[] for _ in range(world_size)]
        bin_pairs = [0 for _ in range(world_size)]
        for item in step_items:
            candidates = [
                candidate
                for candidate in range(world_size)
                if len(bins[candidate]) < views_per_step
            ]
            target = min(
                candidates,
                key=lambda candidate: (bin_pairs[candidate], len(bins[candidate]), candidate),
            )
            bins[target].append(item)
            bin_pairs[target] += int(pair_counts[item[0]])

        # Rotate the heaviest per-step bins toward the ranks with the lightest
        # accumulated epoch load, improving both step and total balance.
        bin_order = sorted(range(world_size), key=lambda item: (-bin_pairs[item], item))
        rank_order = sorted(
            range(world_size), key=lambda item: (rank_total_pairs[item], item)
        )
        for target_rank, source_bin in zip(rank_order, bin_order):
            group = sorted(
                bins[source_bin],
                key=lambda item: (not item[1], position[item[0]]),
            )
            rank_steps[target_rank].append(group)
            rank_total_pairs[target_rank] += bin_pairs[source_bin]
        step_keys.append(min(position[item[0]] for item in step_items if item[1]))

    step_order = sorted(range(len(step_keys)), key=step_keys.__getitem__)
    return [item for step_index in step_order for item in rank_steps[rank][step_index]]


def _chunk_slices(length: int, chunk: int) -> list[slice]:
    if length <= 0:
        return []
    if chunk < 0 or chunk >= length:
        return [slice(0, length)]
    return [slice(start, min(start + chunk, length)) for start in range(0, length, chunk)]


def _as_config_dict(config: LocalOptimizationConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["data"] = asdict(config.data)
    payload["output"] = str(config.output)
    for key, value in list(payload["data"].items()):
        if isinstance(value, Path):
            payload["data"][key] = str(value)
    return payload


@dataclass(frozen=True)
class PreparedView:
    """Model-independent supervision cached for one fixed camera view."""

    pairs: RayVoxelPairs
    directions: torch.Tensor
    target_positions: torch.Tensor
    detail_weights: torch.Tensor | None = None


def _detail_weights_for_pairs(
    sample: RawSample,
    pairs: RayVoxelPairs,
    *,
    device: torch.device,
    scale: float,
) -> torch.Tensor | None:
    if scale <= 0:
        return None
    normal_image = sample.normal_image.to(device)
    normals = (
        normal_image.float()
        if normal_image.is_floating_point()
        else normal_image.float() / 255.0 * 2.0 - 1.0
    )
    normals = torch.nn.functional.normalize(normals, dim=-1, eps=1.0e-8)
    valid = sample.valid_depth.to(device).bool()
    score = torch.zeros_like(valid, dtype=torch.float32)
    dx_valid = valid[:, 1:] & valid[:, :-1]
    dx = (normals[:, 1:] - normals[:, :-1]).norm(dim=-1) * dx_valid
    score[:, 1:] += dx
    score[:, :-1] += dx
    dy_valid = valid[1:] & valid[:-1]
    dy = (normals[1:] - normals[:-1]).norm(dim=-1) * dy_valid
    score[1:] += dy
    score[:-1] += dy
    if not bool(valid.any()):
        return torch.ones_like(pairs.hits, dtype=torch.float32)
    reference = torch.quantile(score[valid], 0.9).clamp_min(1.0e-6)
    dense_weights = 1.0 + float(scale) * (score / reference).clamp(0.0, 1.0)
    pair_weights = dense_weights.reshape(-1)[pairs.ray_labels.long()]
    if pairs.hits is not None and bool(pairs.hits.bool().any()):
        pair_weights = pair_weights / pair_weights[pairs.hits.bool()].mean().clamp_min(1.0e-6)
    return pair_weights


def _prepare_view(
    sample: RawSample,
    preprocessor: DeviceBatchPreprocessor,
    octree: object,
    *,
    geometry_only: bool = False,
    detail_weight_scale: float = 0.0,
) -> PreparedView:
    pairs = preprocessor.prepare_sample(sample, octree)
    if (
        pairs.hits is None
        or pairs.positions is None
        or pairs.normals is None
        or pairs.colors is None
    ):
        raise RuntimeError("Supervision preprocessing did not return complete targets")
    if int(pairs.pins.shape[0]) == 0:
        raise RuntimeError("View produced no supervised ray/voxel pairs")
    directions = torch.nn.functional.normalize(pairs.pouts - pairs.pins, dim=1)
    target_positions = pairs.positions - pairs.voxel_coordinates
    if geometry_only and pairs.colors is not None:
        pairs.colors.zero_()
    detail_weights = _detail_weights_for_pairs(
        sample,
        pairs,
        device=pairs.pins.device,
        scale=detail_weight_scale,
    )
    return PreparedView(pairs, directions, target_positions, detail_weights)


def _cache_rank_order(
    num_views: int, *, rank: int, world_size: int, views_per_step: int
) -> list[tuple[int, bool]]:
    """Return a fixed, balanced view shard used by the supervision cache."""
    if num_views <= 0:
        raise ValueError("num_views must be positive")
    order = list(range(num_views))
    return _rank_view_order(order, rank=rank, world_size=world_size, views_per_step=views_per_step)


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer, config: LocalOptimizationConfig
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build the per-epoch LR schedule selected by ``config.lr_schedule``.

    The training loop calls ``scheduler.step()`` exactly once per epoch (epoch = one full
    pass over the training views), so every schedule below is expressed in epochs.

    ``cosine`` is deliberately the **original** ``CosineAnnealingLR(T_max=epochs,
    eta_min=1e-5)`` object, byte-for-byte the behaviour every recorded campaign number was
    produced with -- changing the factory must not perturb that path.

    * ``cosine``        cosine decay from ``learning_rate`` to ``lr_min`` over ``epochs``
    * ``linear``        linear decay from ``learning_rate`` to ``lr_min`` over ``epochs``
    * ``step``          multiply by ``lr_gamma`` every ``lr_step_epochs`` epochs
    * ``constant``      no decay (stays at ``learning_rate``)
    * ``warmup_cosine`` linear warmup over ``warmup_epochs``, then cosine to ``lr_min``
    """
    epochs = int(config.epochs)
    lr0 = float(config.learning_rate)
    lr_min = float(config.lr_min)
    schedule = config.lr_schedule

    if schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr_min
        )
    if schedule == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _epoch: 1.0)
    if schedule == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(config.lr_step_epochs), gamma=float(config.lr_gamma)
        )

    if schedule == "linear":
        span = max(epochs - 1, 1)

        def factor(epoch: int) -> float:
            # epoch index 1 is the first recorded epoch -> start exactly at lr0
            frac = min(max(epoch - 1, 0), span) / span
            return (lr0 + (lr_min - lr0) * frac) / lr0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    # warmup_cosine
    warmup = max(int(config.warmup_epochs), 0)
    decay_epochs = max(epochs - warmup, 1)
    # NOTE on indexing: the training loop calls `scheduler.step()` *before* recording the
    # epoch's LR, so the LR of epoch e (1-based) is the factor at epoch index e.  The warmup
    # therefore ramps lr0*1/W .. lr0 over the first W recorded epochs.

    def factor(epoch: int) -> float:
        if warmup > 0 and epoch <= warmup:
            return max(epoch, 1) / warmup
        progress = min(max(epoch - warmup, 0), decay_epochs) / decay_epochs
        value = lr_min + (lr0 - lr_min) * (1.0 + math.cos(math.pi * progress)) / 2.0
        return value / lr0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def optimize_object(config: LocalOptimizationConfig) -> LocalOptimizationResult:
    """Optimize one scene and persist a reloadable local checkpoint."""
    config.validate()
    config.data.validate(require_paths=True)
    device, rank, world_size, owns_process_group = _distributed_device(config.device)
    is_master = rank == 0
    if config.tf32 and device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    dataset = ShapeNetMultiViewDataset.for_object(
        config.data, config.object_key, take=config.views
    )
    preprocessor = DeviceBatchPreprocessor(config.data, device)
    first_sample = dataset[0]
    octree = preprocessor.build_octree(first_sample)
    vertex_index = compute_vertex_index(octree, config.data.query_depth, include_batch=True)
    model = LocalSceneModel(
        vertex_index,
        feature_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        n_iters=config.n_iters,
        init_std=config.init_std,
        geometry_only=config.geometry_only,
        segment_position=config.segment_position,
        voxel_feature_dim=config.voxel_feature_dim,
        position_encoding_frequencies=config.position_encoding_frequencies,
        sampling_mode=config.sampling_mode,
    ).to(device)
    if config.compile_renderer:
        if device.type != "cuda":
            raise ValueError("compile_renderer requires CUDA")
        # ``cudagraph_step_begin`` only matters for CUDA-graph modes; those are rejected in
        # ``LocalOptimizationConfig.validate()`` because the renderer is invoked per ray
        # chunk and a replayed graph overwrites an output its backward still needs.
        model.cudagraph_step_begin = False
        model.renderer_model = torch.compile(
            model.renderer_model, mode=config.compile_mode, dynamic=True
        )
    train_model: LocalSceneModel | DistributedDataParallel
    if world_size > 1:
        ddp_options: dict[str, Any] = {}
        if config.ddp_static:
            ddp_options = {
                "broadcast_buffers": False,
                "static_graph": True,
                "gradient_as_bucket_view": True,
            }
        train_model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            **ddp_options,
        )
    else:
        train_model = model
    optimizer = torch.optim.AdamW(
        train_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    amp_enabled = bool(config.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    scheduler = _build_lr_scheduler(optimizer, config)
    objective = IntersectionObjective(
        ObjectiveWeights(
            ssim_normal=0.3,
            ssim_color=1.0,
            smoothness=0.2,
            constancy=0.1,
            normal_gradient=config.normal_gradient_weight,
        ),
        include_color=not config.geometry_only,
        positive_geometry=config.positive_geometry_loss,
        hit_loss_mode=config.hit_loss_mode,
        focal_gamma=config.focal_gamma,
        normal_cosine_weight=config.normal_cosine_weight,
        stage_weight_mode=config.stage_weight_mode,
        final_stage_weight=config.final_stage_weight,
        masked_normal_ssim=config.masked_normal_ssim,
        normal_detail_scale=config.detail_weight_scale,
    ).to(device)
    rng = Random(config.seed)
    total_steps = 0
    total_view_samples = 0
    final_loss = float("nan")
    started = perf_counter()
    if is_master:
        config.output.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    checkpoint_path = config.output / f"{config.object_key.replace('/', '-')}.pt"
    history_path = config.output / "history.json"
    history: list[dict[str, float | int]] = []

    cached_views: dict[int, PreparedView] = {}
    cache_elapsed = 0.0
    pair_counts: dict[int, int] | None = None
    if config.cache_supervision:
        # Cache every view on each rank, while retaining the original global
        # shuffle and DDP sharding order below. This keeps the optimization
        # trajectory comparable to the uncached path instead of pinning each
        # rank to a permanent subset of views.
        cache_indices = range(len(dataset))
        cache_started = perf_counter()
        with torch.no_grad():
            for view_index in cache_indices:
                cached_views[view_index] = _prepare_view(
                    dataset[view_index],
                    preprocessor,
                    octree,
                    geometry_only=config.geometry_only,
                    detail_weight_scale=config.detail_weight_scale,
                )
        cache_elapsed = perf_counter() - cache_started
        if is_master:
            pair_count = sum(view.pairs.pins.shape[0] for view in cached_views.values())
            print(
                f"supervision_cache views={len(cached_views)} pairs={pair_count} "
                f"elapsed={cache_elapsed:.2f}s",
                flush=True,
            )
        pair_counts = {
            view_index: int(prepared.pairs.pins.shape[0])
            for view_index, prepared in cached_views.items()
        }
        if config.pair_balanced_ddp:
            if is_master:
                pair_values = list(pair_counts.values())
                print(
                    f"pair_balanced_ddp enabled min_pairs={min(pair_values)} "
                    f"max_pairs={max(pair_values)} mean_pairs={sum(pair_values) / len(pair_values):.1f}",
                    flush=True,
                )
    elif config.pair_balanced_ddp and is_master:
        print(
            "pair_balanced_ddp requested without cache_supervision; "
            "falling back to view-count sharding",
            flush=True,
        )
    started = perf_counter()

    for epoch in range(1, config.epochs + 1):
        order = list(range(len(dataset)))
        rng.shuffle(order)
        if config.pair_balanced_ddp and pair_counts is not None and world_size > 1:
            local_order = _pair_balanced_view_order(
                order,
                pair_counts,
                rank=rank,
                world_size=world_size,
                views_per_step=config.views_per_step,
            )
        else:
            local_order = _rank_view_order(
                order,
                rank=rank,
                world_size=world_size,
                views_per_step=config.views_per_step,
            )
        pair_load_min = pair_load_max = pair_load_mean = None
        pair_step_imbalance_mean = pair_step_max_ratio = None
        if pair_counts is not None:
            local_pair_load = sum(
                pair_counts[view_index]
                for view_index, is_real_view in local_order
                if is_real_view
            )
            pair_load = torch.tensor(float(local_pair_load), dtype=torch.float64, device=device)
            pair_load_min = pair_load.clone()
            pair_load_max = pair_load.clone()
            if world_size > 1:
                dist.all_reduce(pair_load, op=dist.ReduceOp.SUM)
                dist.all_reduce(pair_load_min, op=dist.ReduceOp.MIN)
                dist.all_reduce(pair_load_max, op=dist.ReduceOp.MAX)
            pair_load_mean = float((pair_load / world_size).cpu())
            pair_load_min = float(pair_load_min.cpu())
            pair_load_max = float(pair_load_max.cpu())
            local_step_loads = torch.tensor(
                [
                    sum(pair_counts[view_index] for view_index, _ in local_order[group_start : group_start + config.views_per_step])
                    for group_start in range(0, len(local_order), config.views_per_step)
                ],
                dtype=torch.float64,
                device=device,
            )
            if world_size > 1:
                gathered_step_loads = [torch.empty_like(local_step_loads) for _ in range(world_size)]
                dist.all_gather(gathered_step_loads, local_step_loads)
                step_loads = torch.stack(gathered_step_loads)
            else:
                step_loads = local_step_loads.unsqueeze(0)
            step_means = step_loads.mean(dim=0).clamp_min(1.0)
            step_ratios = step_loads.max(dim=0).values / step_means
            pair_step_max_ratio = float(step_ratios.max().cpu())
            pair_step_imbalance_mean = float(
                ((step_loads.max(dim=0).values - step_loads.min(dim=0).values) / step_means)
                .mean()
                .cpu()
            )
        local_loss_sum = 0.0
        local_view_count = 0
        for group_start in range(0, len(local_order), config.views_per_step):
            if config.max_steps > 0 and total_steps >= config.max_steps:
                break
            group = local_order[group_start : group_start + config.views_per_step]
            prepared_views = []
            for view_index, is_real_view in group:
                prepared = (
                    cached_views[view_index]
                    if config.cache_supervision
                    else _prepare_view(
                        dataset[view_index],
                        preprocessor,
                        octree,
                        geometry_only=config.geometry_only,
                        detail_weight_scale=config.detail_weight_scale,
                    )
                )
                prepared_views.append((prepared, is_real_view))
            optimizer.zero_grad(set_to_none=True)
            for prepared_index, (prepared, is_real_view) in enumerate(prepared_views):
                pairs = prepared.pairs
                count = int(pairs.pins.shape[0])
                slices = _chunk_slices(count, config.ray_batch_size)
                view_loss = 0.0
                for chunk_index, selected in enumerate(slices):
                    sync_gradients = (
                        prepared_index == len(prepared_views) - 1
                        and chunk_index == len(slices) - 1
                    )
                    sync_context = (
                        nullcontext()
                        if sync_gradients or world_size == 1
                        else train_model.no_sync()
                    )
                    with sync_context:
                        with torch.autocast(
                            device_type=device.type,
                            dtype=torch.float16,
                            enabled=amp_enabled,
                        ):
                            prediction = train_model(
                                pairs.pins[selected],
                                prepared.directions[selected],
                                pairs.voxel_indices[selected],
                                pairs.pouts[selected],
                            )
                        target = IntersectionTarget(
                            prepared.target_positions[selected],
                            pairs.normals[selected],
                            pairs.colors[selected],
                            pairs.hits[selected],
                        )
                        result = objective(
                            prediction,
                            target,
                            pairs.ray_labels[selected],
                            batch_size=1,
                            height=config.data.image_height,
                            width=config.data.image_width,
                            epoch=epoch,
                            total_epochs=config.epochs,
                            training=True,
                            sample_weights=(
                                None
                                if prepared.detail_weights is None
                                else prepared.detail_weights[selected]
                            ),
                        )
                        chunk_weight = (selected.stop - selected.start) / count
                        scaled_loss = result.loss * chunk_weight / len(prepared_views)
                        scaler.scale(scaled_loss).backward()
                    view_loss += float(result.loss.detach().cpu()) * chunk_weight
                if is_real_view:
                    local_loss_sum += view_loss
                    local_view_count += 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_steps += 1
        epoch_stats = torch.tensor(
            [local_loss_sum, float(local_view_count)], dtype=torch.float64, device=device
        )
        if world_size > 1:
            dist.all_reduce(epoch_stats, op=dist.ReduceOp.SUM)
        final_loss = float((epoch_stats[0] / epoch_stats[1].clamp_min(1.0)).cpu())
        total_view_samples += int(epoch_stats[1].item())
        scheduler.step()
        elapsed = perf_counter() - started
        if is_master:
            history.append(
                {
                    "epoch": epoch,
                    "loss": final_loss,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "elapsed_seconds": elapsed,
                    "optimizer_steps": total_steps,
                    "view_samples": total_view_samples,
                    "world_size": world_size,
                    "effective_batch_size": world_size * config.views_per_step,
                    "amp": amp_enabled,
                    "cache_supervision": config.cache_supervision,
                    "cache_precompute_seconds": cache_elapsed,
                    "pair_balanced_ddp": config.pair_balanced_ddp and pair_counts is not None,
                    "pair_load_min": pair_load_min,
                    "pair_load_max": pair_load_max,
                    "pair_load_mean": pair_load_mean,
                    "pair_step_imbalance_mean": pair_step_imbalance_mean,
                    "pair_step_max_ratio": pair_step_max_ratio,
                    "ddp_static": config.ddp_static,
                    "tf32": config.tf32,
                    "compile_renderer": config.compile_renderer,
                    "compile_mode": config.compile_mode,
                    "geometry_only": config.geometry_only,
                    "positive_geometry_loss": config.positive_geometry_loss,
                    "hit_loss_mode": config.hit_loss_mode,
                    "focal_gamma": config.focal_gamma,
                    "normal_cosine_weight": config.normal_cosine_weight,
                    "stage_weight_mode": config.stage_weight_mode,
                    "final_stage_weight": config.final_stage_weight,
                    "segment_position": config.segment_position,
                    "voxel_feature_dim": config.voxel_feature_dim,
                    "position_encoding_frequencies": config.position_encoding_frequencies,
                    "normal_gradient_weight": config.normal_gradient_weight,
                    "detail_weight_scale": config.detail_weight_scale,
                    "masked_normal_ssim": config.masked_normal_ssim,
                    "normal_source": config.data.normal_source,
                    "sampling_mode": config.sampling_mode,
                }
            )
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
            save_local_checkpoint(
                checkpoint_path,
                model,
                object_key=config.object_key,
                query_depth=config.data.query_depth,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                config=_as_config_dict(config),
            )
            if epoch == 1 or epoch % config.checkpoint_every == 0 or epoch == config.epochs:
                save_local_checkpoint(
                    config.output / "checkpoints" / f"epoch_{epoch:04d}.pt",
                    model,
                    object_key=config.object_key,
                    query_depth=config.data.query_depth,
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=_as_config_dict(config),
                )
            print(
                f"epoch={epoch}/{config.epochs} loss={final_loss:.6f} "
                f"lr={scheduler.get_last_lr()[0]:.8f} optimizer_steps={total_steps} "
                f"views={total_view_samples} world_size={world_size} "
                f"effective_bs={world_size * config.views_per_step} elapsed={elapsed:.2f}s",
                flush=True,
            )
        if world_size > 1:
            dist.barrier()
        if config.max_steps > 0 and total_steps >= config.max_steps:
            break

    if config.render and is_master:
        renderer = SparseRenderer(
            model.renderer_model,
            model.feature_field,
            octree,
            depth=config.data.query_depth,
            point_scale=config.data.point_scale,
            ray_scale=config.data.ray_scale,
            camera_convention=config.data.camera_convention,
        )
        render_root = config.output / "render"
        model.eval()
        with torch.no_grad():
            for view_index in range(len(dataset)):
                sample = dataset[view_index]
                output = renderer.render_frame(sample.camera_to_world, sample.intrinsics)
                save_render_output(
                    output,
                    render_root,
                    f"{view_index:03d}",
                    include_color=not config.geometry_only,
                )
    if world_size > 1:
        dist.barrier()
    result = LocalOptimizationResult(
        checkpoint=checkpoint_path,
        epochs=epoch,
        steps=total_steps,
        final_loss=final_loss,
        elapsed_seconds=perf_counter() - started,
        object_key=config.object_key,
        view_count=len(dataset),
        vertex_count=model.feature_field.num_vertices,
        world_size=world_size,
        effective_batch_size=world_size * config.views_per_step,
    )
    if owns_process_group:
        dist.destroy_process_group()
    return result
