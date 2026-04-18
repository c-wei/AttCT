"""Patch eval metrics into a finished W&B run's summary via the Public API.

Complement to `run_evals.py --patch-target RUN_ID`, which runs the evals against
a fresh grouped run and dumps metrics to:

    results/eval_patches/{RUN_ID}_{prefix}.json

This script reads that JSON and pushes every key/value into the target run's
summary. Non-scalar values (W&B tables, artifact refs) are skipped with a
warning — tables are left on the original patch run.

Usage:
    uv run --no-project python scripts/patch_eval_metrics.py \
        results/eval_patches/krp7677p_pre.json

    # Or multiple files at once:
    uv run --no-project python scripts/patch_eval_metrics.py \
        results/eval_patches/krp7677p_pre.json \
        results/eval_patches/krp7677p_post.json
"""
import json
import sys
from pathlib import Path

import wandb

PROJECT = "neilshah/AttCT"


def patch(json_path: Path) -> None:
    # Filename convention: {RUN_ID}_{prefix}.json
    stem = json_path.stem
    run_id = stem.split("_", 1)[0]
    metrics = json.loads(json_path.read_text())

    api = wandb.Api()
    run = api.run(f"{PROJECT}/{run_id}")
    print(f"\nTarget: {PROJECT}/{run_id}  ({run.name}, state={run.state})")
    print(f"Source: {json_path}  ({len(metrics)} keys)")

    scalar = {}
    skipped = []
    for k, v in metrics.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            scalar[k] = v
        else:
            skipped.append((k, type(v).__name__))

    if skipped:
        print(f"\nSkipping {len(skipped)} non-scalar keys (tables / artifacts):")
        for k, t in skipped[:10]:
            print(f"  {k}  ({t})")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    print(f"\nPatching {len(scalar)} scalar keys:")
    for k in sorted(scalar):
        v = scalar[k]
        print(f"  {k:70s} = {v:.4f}" if isinstance(v, float) else f"  {k:70s} = {v}")

    for k, v in scalar.items():
        run.summary[k] = v
    run.summary.update()
    print(f"\nDone — {run_id}.summary updated.")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        patch(Path(p))


if __name__ == "__main__":
    main()
