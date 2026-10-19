#!/usr/bin/env python3
"""Turn Objaverse GLBs into compact mesh-derived occupancy files.

Per object:
  1. Blender ``--dump-mesh-only`` -> the mesh in the *pipeline frame*
     (same normalization the depth maps / point clouds live in, so no registration).
  2. GPU conservative rasterisation (triangle/box SAT) -> every cell the surface touches.
  3. Per-cell mean surface point (ocnn's ``octree.points[d]`` semantic) + colour from the
     existing point cloud's albedo.
  4. Write ``<out>/<uid>.npz`` with ``cells`` (int16), ``offsets`` (uint8),
     ``colors`` (uint8), and grid ``depth``. Source GLBs are retained by default.

Output is ~0.4-0.6 MB/object versus the 4.58 MB point cloud it replaces.

Usage:
    python scripts/build_mesh_occupancy.py --glb-root DIR --out-dir DIR \
        --pointcloud-root DIR --workers 12 [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import itertools
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Import the voxelization kernel lazily so preprocessing help and image utilities
# can be used without initializing the GPU path.

BLENDER = os.environ.get("BLENDER", "blender")
DUMP_SCRIPT = str(Path(__file__).resolve().parent / "objaverse" / "blender_render.py")
POINT_SCALE = 0.99
RAY_SCALE = 0.5
QUERY_DEPTH = 7
_DEVICE_COUNTER = itertools.count()



# --------------------------------------------------------------------------- #
# colour sources
# --------------------------------------------------------------------------- #
def pc_path_is_file(path: Path) -> bool:
    return Path(path).is_file()


def point_cloud_tree(path: Path):
    """``(cKDTree, colors_0_1)`` for one object's back-projected point cloud."""
    from scipy.spatial import cKDTree

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict):
        points = torch.as_tensor(blob.get("points", blob.get("positions")), dtype=torch.float32)
        colors = torch.as_tensor(blob.get("colors", blob.get("features")), dtype=torch.float32)
    else:
        points = torch.as_tensor(blob[0], dtype=torch.float32)
        colors = torch.as_tensor(blob[1], dtype=torch.float32)
    if float(colors.max()) > 1.5:
        colors = colors / 255.0
    return cKDTree(points.numpy()), colors.numpy()


class ViewStack:
    """The 8 rendered frames of one object: albedo, depth buffer and camera."""

    def __init__(self, image_dir: Path):
        from PIL import Image

        metadata = json.loads((image_dir / "transforms.json").read_text(encoding="utf-8"))
        self.views = []
        for frame in metadata["frames"]:
            stem = Path(str(frame["file_path"])).stem
            raw = np.asarray(Image.open(image_dir / f"{stem}_depth.png"))
            if raw.ndim == 3:
                raw = raw[..., 0]
            info = frame["depth"]
            depth_min, depth_max = float(info["min"]), float(info["max"])
            depth = depth_min + raw.astype(np.float32) * (depth_max - depth_min) / 65535.0
            albedo = np.asarray(
                Image.open(image_dir / f"{stem}.png").convert("RGB"), dtype=np.float32
            ) / 255.0
            height, width = depth.shape
            focal = 0.5 * width / np.tan(0.5 * float(frame["camera_angle_x"]))
            matrix = np.asarray(frame["transform_matrix"], dtype=np.float32)
            self.views.append({
                "albedo": albedo,
                "depth": depth,
                "valid": raw != 65535,
                "rotation": matrix[:3, :3],
                "translation": matrix[:3, 3],
                "focal": focal,
                "height": height,
                "width": width,
                # one depth-buffer quantum is (max-min)/65535; allow ~1.5 pixels of
                # depth slop so a cell representative counts as visible when it sits
                # on the rendered surface rather than strictly behind it.
                "tolerance": 1.5 * (depth_max - depth_min) / max(height, width),
            })

    def __len__(self) -> int:
        return len(self.views)


def multiview_colors(positions: np.ndarray, views: ViewStack, *,
                     tolerance: float = 0.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Mean albedo of every view that actually sees each position.

    ``positions`` must be in the pipeline **world** frame (the frame the depth maps
    and ``transforms.json`` live in).  Returns ``(colors, seen, stats)`` where
    ``colors`` is ``[0, 1]`` and ``seen`` marks the positions at least one view saw.
    """
    count = positions.shape[0]
    total = np.zeros((count, 3), dtype=np.float64)
    hits = np.zeros(count, dtype=np.int32)
    if len(views) == 0 or count == 0:
        return np.zeros((count, 3), dtype=np.float32), hits > 0, {"views": 0}
    for view in views.views:
        rotation, translation = view["rotation"], view["translation"]
        camera = (positions - translation) @ rotation
        z = -camera[:, 2]
        forward = z > 0
        focal, width, height = view["focal"], view["width"], view["height"]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = camera[:, 0] / z * focal + width * 0.5 - 0.5
            v = -camera[:, 1] / z * focal + height * 0.5 - 0.5
        ui = np.rint(u)
        vi = np.rint(v)
        inside = forward & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
        index = np.flatnonzero(inside)
        if index.size == 0:
            continue
        x = ui[index].astype(np.int64)
        y = vi[index].astype(np.int64)
        buffer = view["depth"][y, x]
        limit = tolerance if tolerance > 0 else view["tolerance"]
        visible = view["valid"][y, x] & (np.abs(buffer - z[index]) < limit)
        index = index[visible]
        if index.size == 0:
            continue
        total[index] += view["albedo"][vi[index].astype(np.int64), ui[index].astype(np.int64)]
        hits[index] += 1
    colors = np.zeros((count, 3), dtype=np.float64)
    seen = hits > 0
    colors[seen] = total[seen] / hits[seen, None]
    stats = {
        "views": len(views),
        "cells": int(count),
        "seen_by_any_view": int(seen.sum()),
        "seen_fraction": float(seen.mean()) if count else 0.0,
        "mean_views_per_cell": float(hits[seen].mean()) if seen.any() else 0.0,
        "hist_views": {int(k): int(v) for k, v in zip(*np.unique(hits, return_counts=True))},
    }
    return colors.astype(np.float32), seen, stats


def dump_mesh(glb: Path, work: Path) -> Path | None:
    work.mkdir(parents=True, exist_ok=True)
    target = work / "mesh.npz"
    if target.is_file() and target.stat().st_size > 0:
        return target
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    # keep Blender's own scratch next to the dump instead of on the small shared /tmp
    env["TMPDIR"] = str(work)
    result = subprocess.run(
        [BLENDER, "-b", "--factory-startup", "-P", DUMP_SCRIPT, "--",
         "--object", str(glb), "--output", str(work / "render"),
         "--dump-mesh", str(target), "--dump-mesh-only"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800,
    )
    if result.returncode != 0 or not target.is_file():
        (work / "blender.log").write_text(result.stdout[-4000:])
        return None
    return target


def process_one(glb: Path, out_root: Path, pointcloud_root: Path, work_root: Path,
                delete_glb: bool, device, query_depth: int = QUERY_DEPTH,
                max_pairs: int = 4_000_000_000, image_root: Path | None = None,
                color_tolerance: float = 0.0, category: str = "objaverse") -> tuple[str, str, int]:
    if isinstance(device, (list, tuple)):
        # Round-robin over the pool so heavy meshes spread across cards.  A global
        # counter, not threading.get_ident(): pthread ids are 16-byte aligned on
        # glibc, so `ident % len` collapsed every worker onto cuda:0.
        device = device[next(_DEVICE_COUNTER) % len(device)]
    uid = glb.stem
    split = "train"  # resolved by the caller-provided uid->split map when needed
    out_path = out_root / f"{uid}.npz"
    if out_path.is_file() and out_path.stat().st_size > 0:
        return uid, "skip", 0
    work = work_root / uid
    try:
        mesh_npz = dump_mesh(glb, work)
        if mesh_npz is None:
            return uid, "blender-failed", 0
        payload = np.load(mesh_npz)
        vertices = torch.from_numpy(np.asarray(payload["vertices"], np.float32)).to(device)
        faces = torch.from_numpy(np.asarray(payload["faces"], np.int64)).to(device)

        res = 1 << int(query_depth)
        half = res >> 1
        world_scale = POINT_SCALE * RAY_SCALE
        from gpu_conservative_voxelize import conservative_surface_voxels  # lazy: needs CUDA

        cells = conservative_surface_voxels(vertices * world_scale * res + half, faces, res,
                                            max_pairs=max_pairs)
        if cells.shape[0] == 0:
            return uid, "no-cells", 0

        # per-cell mean surface point (matches how ocnn derives octree.points[d])
        cell_keys = (cells[:, 0] * res + cells[:, 1]) * res + cells[:, 2]
        order = torch.argsort(cell_keys)
        sorted_keys = cell_keys[order]
        centres = (cells.to(torch.float32) + 0.5 - half) / (RAY_SCALE * res)
        positions = centres.clone()
        v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        area = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=1).norm(dim=1)
        n_samples = min(int(cells.shape[0]) * 24, 8_000_000)
        pick = torch.multinomial(area / area.sum().clamp_min(1e-12), n_samples, replacement=True)
        r1 = torch.rand(n_samples, device=device).sqrt()
        r2 = torch.rand(n_samples, device=device)
        samples = ((1 - r1).unsqueeze(1) * v0[pick] + (r1 * (1 - r2)).unsqueeze(1) * v1[pick]
                   + (r1 * r2).unsqueeze(1) * v2[pick])
        sample_cells = torch.floor(samples * world_scale * res + half).long().clamp_(0, res - 1)
        sample_keys = (sample_cells[:, 0] * res + sample_cells[:, 1]) * res + sample_cells[:, 2]
        keep = torch.isin(sample_keys, sorted_keys)
        if bool(keep.any()):
            sk, sp = sample_keys[keep], samples[keep]
            uniq, inverse = torch.unique(sk, return_inverse=True)
            count = torch.zeros(uniq.shape[0], device=device)
            total = torch.zeros((uniq.shape[0], 3), device=device)
            count.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float32))
            total.index_add_(0, inverse, sp)
            slot = torch.searchsorted(sorted_keys, uniq)
            positions[order[slot]] = (total / count.clamp_min(1.0).unsqueeze(-1)) * POINT_SCALE

        # ---- colour ---------------------------------------------------------
        # Primary source: the object's own 8 rendered *albedo* frames, picked with a
        # depth-buffer visibility test.  This is the exact colour field the training
        # point clouds were built from (they are the same PNGs), so there is no
        # colour-space or material-model skew, and it uses the mesh's real geometry
        # for occlusion instead of a 200k-sample thinning of the surface.
        # Fallback: nearest neighbour in the point cloud, for cells no view saw.
        cell_positions = positions.detach().cpu().numpy().astype(np.float32)
        views = ViewStack(image_root / category / uid) if image_root is not None else None
        colors_f = np.full((cells.shape[0], 3), 0.5, dtype=np.float32)
        hits = np.zeros(cells.shape[0], dtype=bool)
        if views is not None:
            colors_f, hits, _view_stats = multiview_colors(
                cell_positions / POINT_SCALE, views, tolerance=color_tolerance)
        pc_path = pointcloud_root / category / uid / "pointcloud.pth"
        if pc_path_is_file(pc_path):
            tree, pc_col = point_cloud_tree(pc_path)
            _, nn = tree.query(cell_positions, k=1)
            colors_f[~hits] = pc_col[nn[~hits]]
        colors = np.clip(np.rint(colors_f * 255.0), 0, 255).astype(np.uint8)

        # exact in-cell offsets (float16 positions lose ~1% of cells near boundaries)
        grid = positions.detach().cpu().numpy().astype(np.float32) * (RAY_SCALE * res) + half
        frac = np.clip(grid - cells.detach().cpu().numpy().astype(np.float32),
                       1.0 / 512.0, 1.0 - 1.0 / 512.0)
        offsets = np.clip(np.rint(frac * 256.0 - 0.5), 0, 255).astype(np.uint8)
        out_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            cells=cells.to(torch.int16).cpu().numpy(),
            offsets=offsets,
            colors=colors,
            # the grid depth the cells live on; the loader needs it because the
            # config's q_depth is the *render* depth, which may be shallower
            depth=np.int16(int(query_depth)),
        )
        size = out_path.stat().st_size
        if delete_glb:
            glb.unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)
        return uid, "ok", size
    except Exception as error:  # noqa: BLE001
        return uid, f"error: {error!r}"[:160], 0


def _tmp_free_gb() -> float:
    """Free space on the filesystem holding this project's /tmp scratch."""
    try:
        stat = os.statvfs("/tmp")
        return stat.f_bavail * stat.f_frsize / 1e9
    except OSError:
        return float("nan")



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glb-root", type=Path, required=True)
    ap.add_argument("--category", default="objaverse", help="Category directory under image/point-cloud roots")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pointcloud-root", type=Path,
                    default=Path("data/objaverse/pointcloud"))
    ap.add_argument("--image-root", type=Path,
                    default=Path("data/objaverse/images"),
                    help="rendered frames; each object's 8 albedo+depth pairs are the "
                         "primary colour source (the same field the point clouds were "
                         "built from).  Use --no-multiview to fall back to the cloud.")
    ap.add_argument("--no-multiview", action="store_true",
                    help="disable the rendered-albedo colour source")
    ap.add_argument("--color-tolerance", type=float, default=0.0,
                    help="depth-buffer visibility tolerance in world units "
                         "(0 = 1.5 depth-buffer pixels, the default)")
    ap.add_argument("--work-root", type=Path, default=Path("outputs/occupancy_work"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda",
                    help="one device, or a comma-separated pool assigned round-robin "
                         "per worker thread (e.g. cuda:0,cuda:1,cuda:2,cuda:3)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--delete-source-glb", action="store_true",
                    help="Explicitly delete source GLBs after successful conversion")
    ap.add_argument("--query-depth", type=int, default=QUERY_DEPTH,
                    help="grid depth for the conservative rasterisation (7 = the "
                         "renderer's query depth, 9 = the octree depth)")
    ap.add_argument("--max-pairs", type=int, default=4_000_000_000,
                    help="cap on sum_triangles(|AABB cells|); depth 9 needs up to 200M "
                         "for meshes made of a few very large triangles")
    args = ap.parse_args()

    devices = [d.strip() for d in str(args.device).split(",") if d.strip()]
    glbs = sorted(args.glb_root.rglob("*.glb"))
    if args.limit:
        glbs = glbs[: args.limit]
    print(f"[occ] glbs={len(glbs)} workers={args.workers} depth={args.query_depth} "
          f"devices={devices} tmp_free={_tmp_free_gb():.1f}GB out={args.out_dir}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    stats = {"ok": 0, "skip": 0}
    failures: list[str] = []
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one, glb, args.out_dir, args.pointcloud_root,
                        args.work_root, args.delete_source_glb, devices,
                        args.query_depth, args.max_pairs,
                        None if args.no_multiview else args.image_root,
                        args.color_tolerance, args.category): glb
            for glb in glbs
        }
        for index, future in enumerate(as_completed(futures), 1):
            uid, status, size = future.result()
            if status in stats:
                stats[status] += 1
                total_bytes += size
            else:
                failures.append(f"{uid} {status}")
            if index % 100 == 0 or index == len(futures):
                elapsed = time.time() - started
                print(f"[occ] {index}/{len(futures)} ok={stats['ok']} skip={stats['skip']} "
                      f"fail={len(failures)} {total_bytes/1e9:.2f} GB "
                      f"{index/max(elapsed,1e-9):.2f} obj/s elapsed={elapsed/60:.1f} min", flush=True)
    print(f"[occ] FINISHED ok={stats['ok']} skip={stats['skip']} fail={len(failures)} "
          f"{total_bytes/1e9:.2f} GB in {(time.time()-started)/60:.1f} min", flush=True)
    for line in failures[:20]:
        print("[occ] FAIL", line, flush=True)
    if failures:
        (args.out_dir / "failures.txt").write_text("\n".join(failures) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
