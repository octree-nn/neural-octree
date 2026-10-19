from __future__ import annotations

import json
import random
from pathlib import Path

from .manifest import ManifestEntry


def discover_entries(image_root: str | Path) -> dict[str, list[ManifestEntry]]:
    """Discovers valid frames and groups them by category/object."""
    root = Path(image_root)
    objects: dict[str, list[ManifestEntry]] = {}
    for transform_path in sorted(root.glob("*/*/transforms.json")):
        category = transform_path.parent.parent.name
        object_id = transform_path.parent.name
        with transform_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        frames = payload.get("frames")
        if not isinstance(frames, list):
            continue
        entries: list[ManifestEntry] = []
        for frame_id, frame in enumerate(frames):
            stem = Path(str(frame.get("file_path", ""))).stem
            required = [
                transform_path.parent / f"{stem}.png",
                transform_path.parent / f"{stem}_depth.png",
                transform_path.parent / f"{stem}_normal.png",
            ]
            if stem and all(path.is_file() for path in required):
                entries.append(ManifestEntry(category, object_id, frame_id))
        if entries:
            objects[f"{category}/{object_id}"] = entries
    return objects


def split_objects(
    objects: dict[str, list[ManifestEntry]], *, test_ratio: float = 0.2, seed: int = 0
) -> tuple[list[ManifestEntry], list[ManifestEntry]]:
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("test_ratio must be in [0, 1)")
    keys = sorted(objects)
    random.Random(seed).shuffle(keys)
    test_count = int(round(len(keys) * test_ratio))
    test_keys = set(keys[:test_count])
    train = [entry for key in keys if key not in test_keys for entry in objects[key]]
    test = [entry for key in keys if key in test_keys for entry in objects[key]]
    return sorted(train), sorted(test)


def write_manifest(path: str | Path, entries: list[ManifestEntry]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(f"{entry.relative_name}\n" for entry in entries)
    output.write_text(content, encoding="utf-8")
    return output
