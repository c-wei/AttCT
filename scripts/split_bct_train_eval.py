"""
Split bct_cot.jsonl / bct_non_cot.jsonl into train (first N) / eval (last M)
in a fresh_bct directory, mirroring the convention used in datasets/sycophancy_bct
(first 4000 = train, last 1000 = eval).

Usage:
    uv run --no-project python scripts/split_bct_train_eval.py datasets/fresh_bct_gemma3_4b
    uv run --no-project python scripts/split_bct_train_eval.py datasets/fresh_bct_gemma3_27b datasets/fresh_bct_gemma2_9b

Idempotent: skips files where {stem}_train.jsonl already exists.
Leaves the original {stem}.jsonl in place for backward compatibility.
"""
import argparse
import sys
from pathlib import Path


def split_one(path: Path, train_count: int, eval_count: int) -> tuple[int, int]:
    """Split path → {stem}_train.jsonl (first train_count lines) and {stem}_eval.jsonl (last eval_count lines)."""
    train_path = path.with_name(f"{path.stem}_train.jsonl")
    eval_path  = path.with_name(f"{path.stem}_eval.jsonl")

    if train_path.exists() and eval_path.exists():
        print(f"  [skip] {train_path.name} and {eval_path.name} already exist")
        return 0, 0

    lines = path.read_text().splitlines()
    if len(lines) < train_count + eval_count:
        raise ValueError(
            f"{path.name} has {len(lines)} lines; need at least "
            f"{train_count + eval_count} for split."
        )

    train_lines = lines[:train_count]
    eval_lines  = lines[-eval_count:]

    train_path.write_text("\n".join(train_lines) + "\n")
    eval_path.write_text("\n".join(eval_lines) + "\n")
    print(f"  [ok] {train_path.name}: {len(train_lines)}, {eval_path.name}: {len(eval_lines)}")
    return len(train_lines), len(eval_lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", help="Directories with bct_cot.jsonl / bct_non_cot.jsonl")
    parser.add_argument("--train", type=int, default=4000)
    parser.add_argument("--eval",  type=int, default=1000, dest="eval_count")
    args = parser.parse_args()

    targets = ("bct_cot.jsonl", "bct_non_cot.jsonl",
               "control_cot.jsonl", "control_non_cot.jsonl")

    for d_str in args.dirs:
        d = Path(d_str)
        if not d.is_dir():
            print(f"[err] {d} is not a directory", file=sys.stderr)
            continue
        print(f"\n=== {d} ===")
        for name in targets:
            fp = d / name
            if not fp.exists():
                continue
            split_one(fp, args.train, args.eval_count)


if __name__ == "__main__":
    main()
