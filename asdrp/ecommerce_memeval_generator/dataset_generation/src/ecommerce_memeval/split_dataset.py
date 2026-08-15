#############################################################################
# File: split_dataset.py
#
# Description:
#   Creates reproducible train/dev/test JSONL splits for a generated EcommerceMemEval file.
#
#   - Validates that split fractions sum to one.
#   - Shuffles with an explicit seed and writes three JSONL partitions.
#   - Writes a manifest containing the source path, seed, and resulting counts.
#############################################################################

from __future__ import annotations

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

import argparse
import json
import random
from pathlib import Path
from .io_utils import iter_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl")
    parser.add_argument("--out-dir", default="splits")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--dev", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if round(args.train + args.dev + args.test, 6) != 1.0:
        raise ValueError("train + dev + test must equal 1.0")

    rows = [obj for _, obj in iter_jsonl(args.input_jsonl)]
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n = len(rows)
    n_train = int(n * args.train)
    n_dev = int(n * args.dev)
    train = rows[:n_train]
    dev = rows[n_train : n_train + n_dev]
    test = rows[n_train + n_dev :]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "dev.jsonl", dev)
    write_jsonl(out / "test.jsonl", test)

    manifest = {
        "input_jsonl": args.input_jsonl,
        "seed": args.seed,
        "counts": {"train": len(train), "dev": len(dev), "test": len(test)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
