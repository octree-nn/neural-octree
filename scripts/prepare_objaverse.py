"""Download and prepare the published Objaverse subset for feedforward training."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import yaml

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "objaverse"))
ENTRY = re.compile(r"objaverse/([0-9a-f]{32})/([0-7])")


def read_filelists(root: Path, limit: int = 0):
    """Keep the original ordering and object split; never resample the paper subset."""
    manifests, objects = {}, {}
    for split in ("train", "val"):
        lines = (root / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        views = {}
        for line in lines:
            match = ENTRY.fullmatch(line)
            if not match:
                raise ValueError(f"Invalid {split} manifest entry: {line!r}")
            uid, view = match.groups()
            if view in views.setdefault(uid, set()):
                raise ValueError(f"Duplicate manifest entry: {line}")
            views[uid].add(view)
        if not views or any(v != set("01234567") for v in views.values()):
            raise ValueError(f"{split}: each object must have exactly eight views")
        objects[split] = list(views)
        manifests[split] = lines
    if set(objects["train"]) & set(objects["val"]):
        raise ValueError("Train and validation objects overlap")
    for split in objects:
        if limit:
            objects[split] = objects[split][:limit]
            selected = set(objects[split])
            manifests[split] = [s for s in manifests[split] if s.split("/")[1] in selected]
    return manifests, objects


def download_mesh(uid: str, relative: str, root: Path, endpoint: str) -> Path:
    """Cache source GLBs atomically; interrupted downloads are never marked complete."""
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or path.name != uid + ".glb":
        raise ValueError(f"Invalid Objaverse source path for {uid}")
    destination = root / f"{uid}.glb"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    root.mkdir(parents=True, exist_ok=True)
    url = f"{endpoint.rstrip('/')}/datasets/allenai/objaverse/resolve/main/{relative}"
    for attempt in range(3):
        temporary = destination.with_suffix(".glb.part")
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
                expected = int(response.headers.get("Content-Length", 0))
            size = temporary.stat().st_size
            if not size or (expected and size != expected):
                raise IOError(f"Incomplete GLB download: {uid}")
            temporary.replace(destination)
            return destination
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def prepare_one(uid: str, relative: str, args, settings: dict) -> str:
    from postprocess import postprocess
    import build_mesh_occupancy as occupancy

    out = args.output
    images = out / "images" / "objaverse" / uid
    cloud = out / "pointcloud" / "objaverse" / uid / "pointcloud.pth"
    marker = images / "_complete.json"
    mesh = download_mesh(uid, relative, args.mesh_root, args.endpoint)
    fingerprint = {**settings, "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest()}
    if marker.is_file():
        if json.loads(marker.read_text()) != fingerprint:
            raise ValueError(f"Preparation settings changed for {uid}; choose a new --output")
        required = [images / "transforms.json", cloud]
        required += [images / f"{i:03d}{s}" for i in range(8)
                     for s in (".png", "_depth.png", "_normal.png")]
        if not all(p.is_file() and p.stat().st_size for p in required):
            raise ValueError(f"Incomplete cached object {uid}; choose a new --output")
    else:
        if images.exists():
            raise ValueError(f"Unrecognized image directory: {images}; choose a new --output")
        work = out / ".work"
        work.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=uid + "_", dir=work) as temporary:
            rendered = Path(temporary) / "rendered"
            log = work / (uid + ".log")
            with log.open("w") as handle:
                subprocess.run([
                    args.blender, "-b", "--factory-startup", "-P",
                    str(SCRIPTS / "objaverse/blender_render.py"), "--",
                    "--object", str(mesh), "--output", str(rendered),
                    "--views", "8", "--resolution", str(args.resolution),
                    "--radius", "3.0", "--fov-degrees", "40", "--samples", "32",
                    "--engine", "CYCLES", "--cycles-device", args.cycles_device,
                ], stdout=handle, stderr=subprocess.STDOUT, check=True)
            postprocess(rendered, cloud, uid=uid, point_count=200_000,
                        keep_intermediates=False)
            (rendered / "_complete.json").write_text(json.dumps(fingerprint, sort_keys=True))
            images.parent.mkdir(parents=True, exist_ok=True)
            rendered.rename(images)
    occupancy.BLENDER = args.blender
    _, status, _ = occupancy.process_one(
        mesh, out / "occupancy_d9", out / "pointcloud", out / ".work/occupancy",
        False, args.device, query_depth=9, image_root=out / "images",
    )
    if status not in ("ok", "skip"):
        raise RuntimeError(f"Occupancy generation failed: {status}")
    # Reject a partial/corrupt cache, including one left by an interrupted run.
    import numpy as np
    with np.load(out / "occupancy_d9" / (uid + ".npz")) as data:
        if int(data["depth"]) != 9 or not len(data["cells"]):
            raise ValueError(f"Invalid occupancy cache: {uid}")
        if data["cells"].shape != data["offsets"].shape or data["cells"].shape != data["colors"].shape:
            raise ValueError(f"Invalid occupancy arrays: {uid}")
    return uid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filelists", type=Path, required=True,
                        help="Downloaded filelists/objaverse directory")
    parser.add_argument("--output", type=Path, default=Path("data/objaverse"))
    parser.add_argument("--blender", default=os.environ.get("BLENDER", "blender"))
    parser.add_argument("--mesh-root", type=Path, help="Cache of <uid>.glb files")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:0", help="CUDA device for depth-9 voxelization")
    parser.add_argument("--cycles-device", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--limit", type=int, default=0, help="First N objects per split; 0 = full subset")
    parser.add_argument("--resolution", type=int, default=512, help="Paper uses 512; lower only for smoke tests")
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0 or args.resolution < 16:
        parser.error("Require workers >= 1, limit >= 0 and resolution >= 16")
    args.output = args.output.resolve()
    args.mesh_root = (args.mesh_root or args.output / "meshes").resolve()
    manifests, objects = read_filelists(args.filelists, args.limit)
    mapping = json.loads((args.filelists / "object_paths.json").read_text())
    uids = objects["train"] + objects["val"]
    missing = set(uids) - mapping.keys()
    if missing:
        raise ValueError(f"Missing Objaverse paths for {len(missing)} objects")
    settings = {"schema": 1, "resolution": args.resolution, "views": 8,
                "radius": 3.0, "samples": 32, "point_count": 200_000,
                "occupancy_depth": 9, "cycles_device": args.cycles_device,
                "scripts_sha256": hashlib.sha256(b"".join(
                    (SCRIPTS / p).read_bytes() for p in ["objaverse/blender_render.py",
                    "objaverse/postprocess.py", "build_mesh_occupancy.py",
                    "gpu_conservative_voxelize.py"])).hexdigest()}
    args.output.mkdir(parents=True, exist_ok=True)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(prepare_one, uid, mapping[uid], args, settings): uid for uid in uids}
        for i, job in enumerate(as_completed(jobs), 1):
            uid = jobs[job]
            try:
                job.result()
                print(f"[{i}/{len(uids)}] {uid}: ready", flush=True)
            except Exception as error:
                failures.append(uid)
                print(f"[{i}/{len(uids)}] {uid}: FAILED: {error}", file=sys.stderr, flush=True)
    if failures:
        print(f"{len(failures)} objects failed. Re-run the same command to resume; "
              "manifests/config were not written.", file=sys.stderr)
        return 1
    # Publish manifests only when every requested object is usable.
    manifest_root = args.output / "manifests"
    manifest_root.mkdir(exist_ok=True)
    for split, lines in manifests.items():
        (manifest_root / f"{split}.txt").write_text("\n".join(lines) + "\n")
    config = yaml.safe_load((SCRIPTS.parent / "configs/feedforward.yaml").read_text())
    for section, split in (("train", "train"), ("test", "val")):
        config["DATA"][section].update(
            location=str(args.output / "images"), pointcloud_root=str(args.output / "pointcloud"),
            occupancy_root=str(args.output / "occupancy_d9"),
            filelist=str(manifest_root / f"{split}.txt"),
            image_width=args.resolution, image_height=args.resolution,
        )
    (args.output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Ready: {args.output / 'config.yaml'} ({len(uids)} objects)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
