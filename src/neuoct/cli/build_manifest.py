from __future__ import annotations

import argparse
from pathlib import Path

from neuoct.data.manifest_tools import discover_entries, split_objects, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build object-disjoint NeuOct train/test manifests from a rendered image root."
    )
    parser.add_argument("image_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    objects = discover_entries(args.image_root)
    train, test = split_objects(objects, test_ratio=args.test_ratio, seed=args.seed)
    train_path = write_manifest(args.output_dir / "train.txt", train)
    test_path = write_manifest(args.output_dir / "test.txt", test)
    print(
        f"objects={len(objects)} train_frames={len(train)} test_frames={len(test)} "
        f"train={train_path} test={test_path}"
    )


if __name__ == "__main__":
    main()
