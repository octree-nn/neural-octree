#!/usr/bin/env python3
"""Report the environment available to the offline NeuOct ray tracer.

The script is read-only: it never imports the training stack if that is broken
on CPU-only hosts and it never requires CUDA.  It reports which optional
backends (embreex, OpenEXR, rtree) are present and whether a checkpoint path is
reachable, then exits 0 unless ``--strict`` is given.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path


def _bootstrap() -> Path:
    """Make ``src`` importable and stub the OCNN package on CPU-only hosts."""
    import types

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


def _probe(name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(name)
        return {"available": True, "version": str(getattr(module, "__version__", "unknown"))}
    except Exception as error:  # noqa: BLE001 - report, never raise
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if the core ray-tracing packages are unavailable",
    )
    args = parser.parse_args()

    src = _bootstrap()
    packages = {
        name: _probe(name)
        for name in ("torch", "numpy", "PIL", "yaml", "trimesh", "rtree", "embreex", "OpenEXR", "ocnn")
    }
    cuda_available = False
    device_count = 0
    if packages["torch"]["available"]:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count())
    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "repo_src": str(src),
        "packages": packages,
        "cuda": {"available": cuda_available, "device_count": device_count},
        "checkpoint": None,
    }
    if args.checkpoint is not None:
        checkpoint = Path(args.checkpoint).expanduser()
        report["checkpoint"] = {
            "path": str(checkpoint),
            "exists": checkpoint.exists(),
            "size_bytes": checkpoint.stat().st_size if checkpoint.exists() else None,
        }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"python      : {report['python']} ({report['executable']})")
        print(f"platform    : {report['platform']}")
        print(f"repo src    : {report['repo_src']}")
        print(f"cuda        : available={cuda_available} devices={device_count}")
        for name, status in report["packages"].items():
            state = status.get("version") if status["available"] else status.get("error")
            print(f"  {name:<10}: {state}")
        if report["checkpoint"] is not None:
            print(f"checkpoint  : {report['checkpoint']}")
    if args.strict:
        required = ("torch", "numpy", "PIL", "yaml")
        missing = [name for name in required if not packages[name]["available"]]
        if missing:
            print(f"missing required packages: {missing}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
