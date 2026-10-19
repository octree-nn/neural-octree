#!/usr/bin/env python3
"""Minimal offline smoke render: one coloured triangle, no checkpoint required.

This exercises the full stack (camera -> scene -> mesh BVH -> integrator ->
PNG/JSON output) and is the first thing to run after installing the package.
It writes files directly; it never takes a screenshot.

Example::

    python tools/neuoct_rt/smoke_render.py --output /tmp/neuoct_smoke \\
        --width 128 --height 128 --samples 4 --mode one_bounce
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path


def _bootstrap() -> Path:
    repo = Path(__file__).resolve().parents[2]
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        import ocnn  # noqa: F401

        return src
    except Exception:
        pass
    for name in (
        "neuoct",
        "neuoct.geometry",
        "neuoct.models",
        "neuoct.engine",
        "neuoct.data",
        "neuoct.rendering",
        "neuoct.config",
    ):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(src / name.replace(".", "/"))]
        sys.modules[name] = module
    sys.modules["neuoct"].__version__ = "0.1.0"
    return src


_BOOTSTRAP_SRC = _bootstrap()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from neuoct.rendering_rt import (  # noqa: E402
    Camera,
    ConstantEnvironment,
    Integrator,
    MeshPrimitive,
    Scene,
    resolve_device,
    save_image,
    write_metadata,
)

TRIANGLE_VERTICES = np.array(
    [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
)
TRIANGLE_FACES = np.array([[0, 1, 2]], dtype=np.int64)


def build_triangle() -> object:
    try:
        import trimesh

        return trimesh.Trimesh(
            vertices=TRIANGLE_VERTICES, faces=TRIANGLE_FACES, process=False
        )
    except Exception:
        class _Triangle:
            vertices = TRIANGLE_VERTICES
            faces = TRIANGLE_FACES
            visual = None

        return _Triangle()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/raytracing/smoke_triangle.png"))
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--mode", default="one_bounce", choices=("direct", "one_bounce", "path"))
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda:1")
    args = parser.parse_args()

    device = resolve_device(args.device)
    primitive = MeshPrimitive(
        build_triangle(),
        material_overrides={"base_color": [0.85, 0.25, 0.20], "roughness": 0.35},
        primitive_id=0,
        name="smoke_triangle",
    )
    scene = Scene([primitive], name="smoke")
    camera = Camera.look_at(
        [0.0, 0.0, 3.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        width=args.width,
        height=args.height,
        camera_angle_x=0.7,
    ).to(device)
    environment = ConstantEnvironment([1.0, 1.0, 1.0]).to(device)
    integrator = Integrator(
        scene,
        environment=environment,
        mode=args.mode,
        max_depth=args.max_depth,
        samples=args.samples,
        seed=args.seed,
        background=[0.05, 0.05, 0.08],
    )
    buffers = integrator.render(camera, device=device)
    if not bool(buffers.hit.any()):
        print("smoke render produced no hit pixels", file=sys.stderr)
        return 1
    if not bool(torch.isfinite(buffers.color).all()):
        print("smoke render produced non-finite pixels", file=sys.stderr)
        return 1

    output = Path(args.output)
    save_image(output, buffers.color, mode="aces")
    metadata = {
        "smoke": True,
        "checkpoint": None,
        "mode": args.mode,
        "samples": args.samples,
        "max_depth": args.max_depth,
        "seed": args.seed,
        "device": str(device),
        "resolution": [args.height, args.width],
        "hit_pixels": int(buffers.hit.sum()),
        "output": str(output),
    }
    write_metadata(output.with_suffix(".json"), metadata, config=metadata)
    print(
        f"smoke render hit={int(buffers.hit.sum())} mean={float(buffers.color[buffers.hit].mean()):.4f} "
        f"-> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
