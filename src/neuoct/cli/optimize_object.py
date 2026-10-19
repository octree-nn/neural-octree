from __future__ import annotations

import argparse
import os

from neuoct.engine.local_optimize import LocalOptimizationConfig, optimize_object


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize one ShapeNet object scene")
    parser.add_argument("--config", required=True, help="Existing NeuOct YAML config")
    parser.add_argument("--object", required=True, dest="object_key", help="category/object_id")
    parser.add_argument("--output", required=True, help="Directory for checkpoint and renders")
    parser.add_argument("--data-split", choices=("train", "test"), default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true", help="Use CUDA FP16 autocast and gradient scaling")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--views", type=int, default=-1)
    parser.add_argument("--ray-batch-size", type=int, default=65536)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-iters", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument(
        "--lr-schedule",
        choices=("cosine", "linear", "step", "constant", "warmup_cosine"),
        default="cosine",
        help=(
            "Per-epoch LR schedule: 'cosine' (original CosineAnnealingLR to --lr-min), "
            "'linear' (linear decay to --lr-min), 'step' (multiply by --lr-gamma every "
            "--lr-step-epochs), 'constant' (no decay) or 'warmup_cosine' (linear warmup "
            "for --warmup-epochs then cosine decay to --lr-min)"
        ),
    )
    parser.add_argument("--lr-min", type=float, default=1.0e-5,
                        help="Floor of the cosine / linear / warmup_cosine schedules")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Linear warmup length for --lr-schedule warmup_cosine")
    parser.add_argument("--lr-step-epochs", type=int, default=0,
                        help="Decay period for --lr-schedule step")
    parser.add_argument("--lr-gamma", type=float, default=0.5,
                        help="Multiplicative decay factor for --lr-schedule step")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--views-per-step", type=int, default=1)
    parser.add_argument(
        "--cache-supervision",
        action="store_true",
        help="Precompute fixed ray/voxel supervision once per rank and reuse it",
    )
    parser.add_argument(
        "--pair-balanced-ddp",
        action="store_true",
        help=(
            "Assign cached views to DDP ranks by ray/voxel pair count; "
            "use together with --cache-supervision"
        ),
    )
    parser.add_argument(
        "--ddp-static",
        action="store_true",
        help="Disable immutable buffer broadcasts and enable static DDP buckets",
    )
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 CUDA matrix operations")
    parser.add_argument(
        "--compile-renderer",
        action="store_true",
        help="Compile the renderer with torch.compile (~1.6x faster training)",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
        default="default",
        help=(
            "torch.compile mode used with --compile-renderer. 'default' is the "
            "validated choice; 'reduce-overhead' captures CUDA graphs (supported "
            "here, but validate it for your workload)"
        ),
    )
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Remove the material branch and all color-related losses",
    )
    parser.add_argument(
        "--positive-geometry-loss",
        action="store_true",
        help="Normalize position and normal losses over positive surface pairs only",
    )
    parser.add_argument(
        "--hit-loss-mode",
        choices=("bce", "balanced_bce", "balanced_focal"),
        default="bce",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--normal-cosine-weight", type=float, default=0.0)
    parser.add_argument(
        "--stage-weight-mode",
        choices=("legacy", "final_emphasis"),
        default="legacy",
    )
    parser.add_argument("--final-stage-weight", type=float, default=0.5)
    parser.add_argument(
        "--segment-position",
        action="store_true",
        help="Parameterize intersection position between each voxel ray entry and exit",
    )
    parser.add_argument(
        "--voxel-feature-dim",
        type=int,
        default=0,
        help="Optional learned residual feature dimension stored once per occupied voxel",
    )
    parser.add_argument(
        "--position-encoding-frequencies",
        type=int,
        default=0,
        help="Number of Fourier frequency bands for voxel-local intersection coordinates",
    )
    parser.add_argument(
        "--normal-gradient-weight",
        type=float,
        default=0.0,
        help="Weight of the foreground normal-gradient reconstruction loss",
    )
    parser.add_argument(
        "--detail-weight-scale",
        type=float,
        default=0.0,
        help="Extra sample/loss weight assigned to high-gradient target-normal regions",
    )
    parser.add_argument(
        "--masked-normal-ssim",
        action="store_true",
        help="Compute normal SSIM after masking target background pixels",
    )
    parser.add_argument(
        "--normal-source",
        choices=("png", "exr"),
        default=None,
        help="Override DATA.<split>.normal_source for normal supervision",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=(
            "recurrent",
            "uniform",
            "single",
            "recurrent_endpoints",
            "uniform_endpoints",
            "uniform_concat",
            "uniform_concat_endpoints",
            "recurrent_pool",
            "single_endpoints",
        ),
        default="recurrent",
        help=(
            "Intersection sampling strategy: recurrent refinement, uniform "
            "samples between voxel entry/exit at matched resolution, the "
            "single entry point, a chained renderer whose every query also "
            "sees the entry and exit points (recurrent_endpoints adapts the "
            "query, uniform_endpoints keeps it on a fixed interior grid), the "
            "same uniform samples flattened into one wide input evaluated by a "
            "single network pass (uniform_concat / uniform_concat_endpoints), "
            "or pooled fusion with adaptive interior query points "
            "(recurrent_pool)"
        ),
    )
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    config = LocalOptimizationConfig.from_yaml(
        args.config,
        object_key=args.object_key,
        output=args.output,
        data_split=args.data_split,
        device=args.device,
        amp=args.amp,
        epochs=args.epochs,
        views=args.views,
        ray_batch_size=args.ray_batch_size,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        n_iters=args.n_iters,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        lr_min=args.lr_min,
        warmup_epochs=args.warmup_epochs,
        lr_step_epochs=args.lr_step_epochs,
        lr_gamma=args.lr_gamma,
        render=args.render,
        max_steps=args.max_steps,
        checkpoint_every=args.checkpoint_every,
        views_per_step=args.views_per_step,
        cache_supervision=args.cache_supervision,
        pair_balanced_ddp=args.pair_balanced_ddp,
        ddp_static=args.ddp_static,
        tf32=args.tf32,
        compile_renderer=args.compile_renderer,
        compile_mode=args.compile_mode,
        geometry_only=args.geometry_only,
        positive_geometry_loss=args.positive_geometry_loss,
        hit_loss_mode=args.hit_loss_mode,
        focal_gamma=args.focal_gamma,
        normal_cosine_weight=args.normal_cosine_weight,
        stage_weight_mode=args.stage_weight_mode,
        final_stage_weight=args.final_stage_weight,
        segment_position=args.segment_position,
        voxel_feature_dim=args.voxel_feature_dim,
        position_encoding_frequencies=args.position_encoding_frequencies,
        normal_gradient_weight=args.normal_gradient_weight,
        detail_weight_scale=args.detail_weight_scale,
        masked_normal_ssim=args.masked_normal_ssim,
        normal_source=args.normal_source,
        sampling_mode=args.sampling_mode,
    )
    result = optimize_object(config)
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            f"optimized {result.object_key}: epochs={result.epochs} steps={result.steps} "
            f"views={result.view_count} vertices={result.vertex_count} "
            f"world_size={result.world_size} effective_bs={result.effective_batch_size} "
            f"loss={result.final_loss:.6f} elapsed={result.elapsed_seconds:.2f}s "
            f"checkpoint={result.checkpoint}"
        )


if __name__ == "__main__":
    main()
