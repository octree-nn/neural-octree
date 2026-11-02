"""Published split validation must not silently alter the experimental subset."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "prepare_objaverse", Path(__file__).parents[2] / "scripts/prepare_objaverse.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def write_split(root, split, uids):
    (root / f"{split}.txt").write_text(
        "".join(f"objaverse/{uid}/{i}\n" for uid in uids for i in range(8))
    )


def test_preserves_order_and_limits_whole_objects(tmp_path):
    write_split(tmp_path, "train", ["b" * 32, "a" * 32])
    write_split(tmp_path, "val", ["c" * 32])
    manifests, objects = prepare.read_filelists(tmp_path, limit=1)
    assert objects == {"train": ["b" * 32], "val": ["c" * 32]}
    assert manifests["train"] == [f"objaverse/{'b' * 32}/{i}" for i in range(8)]


def test_overlap_is_rejected_before_limit(tmp_path):
    write_split(tmp_path, "train", ["a" * 32, "b" * 32])
    write_split(tmp_path, "val", ["c" * 32, "b" * 32])
    with pytest.raises(ValueError, match="overlap"):
        prepare.read_filelists(tmp_path, limit=1)


@pytest.mark.parametrize("bad", ["missing", "duplicate", "path"])
def test_invalid_entries_rejected(tmp_path, bad):
    write_split(tmp_path, "train", ["a" * 32])
    write_split(tmp_path, "val", ["b" * 32])
    p = tmp_path / "train.txt"
    lines = p.read_text().splitlines()
    if bad == "missing":
        lines.pop()
    elif bad == "duplicate":
        lines.append(lines[0])
    else:
        lines[0] = "../outside/0"
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        prepare.read_filelists(tmp_path)
