from __future__ import annotations

import argparse

from neuoct.engine.local_pack import pack_local_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack a local resume checkpoint into a topology-free inference bundle"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--object", required=True, dest="object_key")
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-split", choices=("train", "test"), default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--storage-dtype", choices=("fp32", "fp16"), default="fp32")
    args = parser.parse_args()
    output = pack_local_checkpoint(
        args.config,
        args.checkpoint,
        object_key=args.object_key,
        output=args.output,
        data_split=args.data_split,
        device=args.device,
        storage_dtype=args.storage_dtype,
    )
    print(f"packed inference checkpoint: {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
