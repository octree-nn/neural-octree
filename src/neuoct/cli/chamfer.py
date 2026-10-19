from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neuoct.metrics import symmetric_chamfer_distance


def _load_points(path: Path) -> torch.Tensor:
    if path.suffix.lower() in {".pth", ".pt"}:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload = payload[0] if isinstance(payload, (tuple, list)) else payload
        if isinstance(payload, dict):
            payload = payload.get("points")
        return torch.as_tensor(payload, dtype=torch.float32)
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit("Open3D is required to read non-PTH point clouds") from exc
    point_cloud = o3d.io.read_point_cloud(str(path))
    return torch.from_numpy(np.asarray(point_cloud.points).copy()).float()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute chunked symmetric squared Chamfer distance."
    )
    parser.add_argument("points_a", type=Path)
    parser.add_argument("points_b", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--chunk-size", type=int, default=2048)
    args = parser.parse_args()

    def sample(points: torch.Tensor) -> torch.Tensor:
        if args.max_points > 0 and points.shape[0] > args.max_points:
            generator = torch.Generator().manual_seed(0)
            indices = torch.randperm(points.shape[0], generator=generator)[: args.max_points]
            points = points[indices]
        return points.to(args.device)

    points_a = sample(_load_points(args.points_a))
    points_b = sample(_load_points(args.points_b))
    distance = symmetric_chamfer_distance(
        points_a,
        points_b,
        source_chunk=args.chunk_size,
        target_chunk=args.chunk_size,
    )
    print(
        json.dumps(
            {
                "points_a": int(points_a.shape[0]),
                "points_b": int(points_b.shape[0]),
                "symmetric_squared_chamfer": float(distance),
            }
        )
    )


if __name__ == "__main__":
    main()
