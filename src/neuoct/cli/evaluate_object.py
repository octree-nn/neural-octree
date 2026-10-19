from __future__ import annotations

import argparse

from neuoct.engine.local_evaluate import evaluate_local_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Render and evaluate one local scene checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--object", required=True, dest="object_key")
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-split", choices=("train", "test"), default="test")
    parser.add_argument("--views", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--normal-source",
        choices=("png", "exr"),
        default=None,
        help="Override DATA.<split>.normal_source for evaluation ground truth",
    )
    parser.add_argument("--hit-threshold", type=float, default=0.5)
    parser.add_argument("--orient-normals", action="store_true")
    parser.add_argument("--supersample-factor", type=int, default=1)
    parser.add_argument("--supersample-hit-fraction", type=float, default=0.5)
    parser.add_argument("--topology-diagnostics", action="store_true")
    parser.add_argument(
        "--all-pixels",
        action="store_true",
        help="Use the legacy full-image distance/angle convention instead of foreground-only metrics",
    )
    args = parser.parse_args()
    result = evaluate_local_checkpoint(
        args.config,
        args.checkpoint,
        object_key=args.object_key,
        output=args.output,
        data_split=args.data_split,
        views=args.views,
        device=args.device,
        normal_source=args.normal_source,
        hit_threshold=args.hit_threshold,
        orient_normals=args.orient_normals,
        supersample_factor=args.supersample_factor,
        supersample_hit_fraction=args.supersample_hit_fraction,
        topology_diagnostics=args.topology_diagnostics,
        foreground_only_metrics=not args.all_pixels,
    )
    metrics = " ".join(f"{name}={value:.4f}" for name, value in result.metrics.items())
    print(f"evaluated views={result.view_count} {metrics} output={result.output}")


if __name__ == "__main__":
    main()
