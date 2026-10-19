from __future__ import annotations

import argparse

from neuoct.engine.feedforward_evaluate import evaluate_feedforward_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render and score a feedforward NeuOct checkpoint on a manifest"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-split", default="test", choices=("train", "test"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hit-threshold", type=float, default=0.5)
    parser.add_argument("--max-views", type=int, default=-1)
    parser.add_argument("--save-renders", type=int, default=8)
    parser.add_argument("--orient-normals", action="store_true")
    args = parser.parse_args()
    result = evaluate_feedforward_checkpoint(
        args.config,
        args.checkpoint,
        output=args.output,
        data_split=args.data_split,
        device=args.device,
        hit_threshold=args.hit_threshold,
        max_views=args.max_views,
        save_renders=args.save_renders,
        orient_normals=args.orient_normals,
    )
    print(
        f"Evaluated {result.view_count} views from {result.object_count} objects; "
        f"metrics={result.output / 'metrics.json'}"
    )


if __name__ == "__main__":
    main()
