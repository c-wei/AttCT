"""
Split sycophancy_bct data into train/eval sets.

All 4 JSONL files contain the same questions in the same order
(clean vs wrapped versions). We split at the same index so
train and eval questions never overlap.

Default: 4000 train / 1000 eval (80/20 split)

Usage:
    python scripts/split_data.py
    python scripts/split_data.py --train-size 4000
"""

import argparse
from pathlib import Path


def split_file(src: Path, dst_dir: Path, train_size: int):
    """Split a JSONL file into train/eval at a fixed line index."""
    lines = src.read_text().strip().split("\n")
    total = len(lines)
    eval_size = total - train_size

    train_path = dst_dir / f"{src.stem}_train.jsonl"
    eval_path = dst_dir / f"{src.stem}_eval.jsonl"

    train_path.write_text("\n".join(lines[:train_size]) + "\n")
    eval_path.write_text("\n".join(lines[train_size:]) + "\n")

    print(f"  {src.name}: {total} -> train={train_size} ({train_path.name}), eval={eval_size} ({eval_path.name})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--data-dir", default="datasets/sycophancy_bct")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = [
        "control_cot.jsonl",
        "control_non_cot.jsonl",
        "bct_cot.jsonl",
        "bct_non_cot.jsonl",
    ]

    print(f"Splitting at index {args.train_size} (train={args.train_size}, eval={5000 - args.train_size})")
    print()

    for fname in files:
        src = data_dir / fname
        if not src.exists():
            print(f"  WARNING: {src} not found, skipping")
            continue
        split_file(src, data_dir, args.train_size)

    print()
    print("Done. Training uses *_train.jsonl, eval uses *_eval.jsonl.")
    print("The same questions never appear in both sets.")


if __name__ == "__main__":
    main()
