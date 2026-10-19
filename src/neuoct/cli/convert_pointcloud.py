from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a colored PLY point cloud to NeuOct PTH.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit(
            "Open3D is required: install the preprocess optional dependencies"
        ) from exc
    point_cloud = o3d.io.read_point_cloud(str(args.input))
    points = torch.from_numpy(np.asarray(point_cloud.points).copy()).float()
    colors = torch.from_numpy(np.asarray(point_cloud.colors).copy()).float()
    if points.numel() == 0:
        raise SystemExit(f"Point cloud is empty: {args.input}")
    if colors.shape != points.shape:
        colors = torch.zeros_like(points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save((points, colors), args.output)
    print(f"Saved {points.shape[0]} points to {args.output}")


if __name__ == "__main__":
    main()
