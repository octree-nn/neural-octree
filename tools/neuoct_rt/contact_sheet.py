#!/usr/bin/env python3
"""Build a labelled contact sheet from finished ray-tracing renders.

The sheet is assembled from the tonemapped PNGs that each render already wrote;
nothing is captured from a UI.  A JSON summary next to the image records, per
tile, the source path, the manifest statistics, the accelerator, the NeuOct
primary coverage and the NeuOct secondary-hit count, so the sheet can be
verified numerically as well as visually.

Example::

    python tools/neuoct_rt/contact_sheet.py \
        --output outputs/raytracing/chess_neuoct/contact_sheet.png \
        outputs/raytracing/chess_neuoct_env_285.png \
        outputs/raytracing/chess_neuoct_env_alt.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_png(path: Path):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    return image


def _tile(image, label: str, footer: str, *, font_size: int = 22):
    from PIL import Image, ImageDraw

    width, height = image.size
    header = font_size + 10
    footer_height = footer.count("\n") * (font_size - 4) + 12 if footer else 0
    canvas = Image.new("RGB", (width, height + header + footer_height), (12, 12, 14))
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), label, fill=(255, 255, 255))
    if footer:
        draw.multiline_text((8, height + header + 4), footer, fill=(210, 210, 215))
    return canvas


def _manifest_for(png: Path) -> dict:
    manifest = png.with_suffix(".json")
    if not manifest.exists():
        # Scenario configs write `<name>.json` next to `<name>.png`.
        return {}
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _summary_lines(manifest: dict) -> tuple[str, dict]:
    if not manifest:
        return "manifest missing", {}
    coverage = manifest.get("coverage", {})
    declared = manifest.get("declared_primitives", {})
    neuoct_ids = [key for key, kind in declared.items() if kind == "NeuOctPrimitive"]
    primary_neuoct = sum(int(coverage.get(key, 0)) for key in neuoct_ids)
    secondary = manifest.get("secondary_hits_by_primitive", {})
    secondary_neuoct = sum(int(secondary.get(key, 0)) for key in neuoct_ids)
    stats = manifest.get("statistics", {})
    accelerators = manifest.get("accelerators", {})
    env = manifest.get("environment", {})
    lines = [
        f"hit={manifest.get('hit_pixels')}  spp={manifest.get('samples')}  depth={manifest.get('max_depth')}",
        f"neuoct primary={primary_neuoct}  neuoct secondary={secondary_neuoct}",
        f"accelerators={accelerators}",
        f"mean={stats.get('mean'):.4f} max={stats.get('max'):.3f} finite={stats.get('finite')}"
        if stats
        else "statistics missing",
    ]
    if env.get("type") == "hdri":
        lines.append(f"hdri rot={env.get('rotation_deg')} strength={env.get('strength')}")
    detail = {
        "hit_pixels": manifest.get("hit_pixels"),
        "samples": manifest.get("samples"),
        "max_depth": manifest.get("max_depth"),
        "neuoct_primary_coverage": primary_neuoct,
        "neuoct_secondary_hits": secondary_neuoct,
        "coverage": coverage,
        "declared_primitives": declared,
        "accelerators": accelerators,
        "environment": env,
        "statistics": stats,
        "transmission": manifest.get("transmission"),
        "sha256_outputs": manifest.get("output_sha256"),
    }
    return "\n".join(lines), detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--tile", type=int, default=512)
    args = parser.parse_args()

    tiles = []
    summary: dict[str, dict] = {}
    for path in args.images:
        if not path.exists():
            print(f"missing image: {path}", file=sys.stderr)
            return 1
        image = _load_png(path)
        if image.size != (args.tile, args.tile):
            image = image.resize((args.tile, args.tile))
        footer, detail = _summary_lines(_manifest_for(path))
        label = path.stem
        tiles.append(_tile(image, label, footer))
        summary[label] = {"path": str(path), "source_size": list(image.size), **detail}

    columns = max(1, int(args.columns))
    rows = (len(tiles) + columns - 1) // columns
    width = max(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    from PIL import Image

    sheet = Image.new("RGB", (width * columns, height * rows), (8, 8, 10))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * width, (index // columns) * height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"contact sheet: {args.output} ({sheet.width}x{sheet.height}), {len(tiles)} tiles")
    if args.summary is not None:
        print(f"summary: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
