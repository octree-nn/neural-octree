from __future__ import annotations

import argparse

from neuoct.engine.local_orbit import render_local_orbit


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a local scene as an orbit video")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--object", required=True, dest="object_key")
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-split", choices=("train", "test"), default="test")
    parser.add_argument("--views", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--radius", type=float, default=2.5)
    parser.add_argument("--elevation-degrees", type=float, default=18.0)
    parser.add_argument(
        "--up-axis",
        choices=("x", "y", "z"),
        default=None,
        help="World-space vertical axis; defaults to DATA.<split>.orbit_up_axis",
    )
    parser.add_argument("--hit-threshold", type=float, default=0.5)
    parser.add_argument("--orient-normals", action="store_true")
    parser.add_argument("--supersample-factor", type=int, default=1)
    parser.add_argument("--supersample-hit-fraction", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--video-codec",
        choices=("mp4v", "ffv1"),
        default="mp4v",
        help="mp4v is compact/lossy; ffv1 writes pixel-lossless MKV",
    )
    args = parser.parse_args()
    report = render_local_orbit(
        args.config,
        args.checkpoint,
        object_key=args.object_key,
        output=args.output,
        data_split=args.data_split,
        views=args.views,
        fps=args.fps,
        radius=args.radius,
        elevation_degrees=args.elevation_degrees,
        up_axis=args.up_axis,
        hit_threshold=args.hit_threshold,
        orient_normals=args.orient_normals,
        supersample_factor=args.supersample_factor,
        supersample_hit_fraction=args.supersample_hit_fraction,
        device=args.device,
        video_codec=args.video_codec,
    )
    print(
        f"rendered {report['view_count']} views at {report['fps']} fps "
        f"({report['duration_seconds']:.2f}s): {report['primary_video']}"
    )


if __name__ == "__main__":
    main()
