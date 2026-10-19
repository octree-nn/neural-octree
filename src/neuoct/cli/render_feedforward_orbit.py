from __future__ import annotations

import argparse

from neuoct.engine.feedforward_orbit import render_feedforward_orbit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an orbit video from a feedforward NeuOct checkpoint"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-split", default="test", choices=("train", "test"))
    parser.add_argument("--views", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--radius", type=float, default=2.5)
    parser.add_argument("--elevation", type=float, default=18.0)
    parser.add_argument("--hit-threshold", type=float, default=0.5)
    parser.add_argument("--orient-normals", action="store_true")
    parser.add_argument("--video-codec", choices=("mp4v", "ffv1"), default="mp4v")
    parser.add_argument(
        "--mesh-octree",
        default=None,
        help="Pipeline-frame mesh .npz (blender_render.py --dump-mesh). When given, "
             "the octree is conservative-rasterised from the mesh instead of the "
             "point cloud, which removes the coverage gaps.",
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=8,
        help="How many evenly spaced views to keep as full-resolution PNG tiles "
             "for diagnostics (default 8).",
    )
    args = parser.parse_args()
    report = render_feedforward_orbit(
        args.config,
        args.checkpoint,
        object_key=args.object,
        output=args.output,
        data_split=args.data_split,
        views=args.views,
        fps=args.fps,
        device=args.device,
        radius=args.radius,
        elevation_degrees=args.elevation,
        hit_threshold=args.hit_threshold,
        orient_normals=args.orient_normals,
        video_codec=args.video_codec,
        frame_count=args.frame_count,
        mesh_octree=args.mesh_octree,
    )
    print(f"Saved feedforward orbit: {report['composite_video']}")


if __name__ == "__main__":
    main()
