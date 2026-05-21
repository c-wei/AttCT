#!/usr/bin/env python
"""
Plot loss-trajectory comparison from the cross-loss tracking experiment.

Pulls all runs in a single W&B group (8 runs: 4 methods × 2 seeds), then
emits one PNG per `track/<name>_loss` key. Each PNG shows 4 methods × 2
seeds = 8 lines; method = colour, seed = linestyle.

Also emits a C2-sanity panel per primary-tracked match (e.g. AttCT-JSD
primary's `train/loss` vs `track/attct_jsd_loss` — should overlap to
numerical precision when the canonical kwargs match).

Usage:
    uv run python scripts/plot_loss_tracking.py \\
        --wandb-group loss_tracking_gemma3_4b_sycophancy \\
        --out-dir findings/loss_tracking_plots
"""
import argparse
import re
from pathlib import Path
from typing import Any, Dict, List

# Method → primary tracked-key name. The C2-sanity invariant compares
# train/loss against track/<this>_loss for each primary.
PRIMARY_OF = {
    "attct_jsd": "attct_jsd",
    "act":       "act",
    "mlpct":     "mlpct",
    "bct":       "bct",
}

METHOD_COLORS = {
    "attct_jsd": "#1f77b4",  # blue
    "act":       "#ff7f0e",  # orange
    "mlpct":     "#2ca02c",  # green
    "bct":       "#d62728",  # red
}
SEED_LINESTYLES = {"0": "-", "1": "--"}

TRACKED_KEYS = ["attct_jsd", "act", "mlpct", "bct"]


def _parse_run_name(name: str):
    """Run names look like 'attct_jsd_seed0'."""
    m = re.match(r"^(?P<method>attct_jsd|act|mlpct|bct)_seed(?P<seed>\d+)$", name)
    if not m:
        return None, None
    return m.group("method"), m.group("seed")


def _fetch_history(run, keys: List[str]):
    """Pull a run's per-step history for the requested keys."""
    df = run.history(keys=keys, samples=100_000, pandas=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-entity",  default=None,
                        help="W&B entity (defaults to the value WANDB picked up at run time).")
    parser.add_argument("--wandb-project", default="AttCT")
    parser.add_argument("--wandb-group", required=True,
                        help="W&B group name (e.g. loss_tracking_gemma3_4b_sycophancy)")
    parser.add_argument("--out-dir", default="findings/loss_tracking_plots")
    parser.add_argument("--smooth", type=int, default=1,
                        help="Rolling-mean window (1=no smoothing). Try 10-50.")
    args = parser.parse_args()

    import wandb
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    api = wandb.Api()
    entity = args.wandb_entity or wandb.api.default_entity
    if entity is None:
        raise SystemExit(
            "No W&B entity available — pass --wandb-entity or run `wandb login` first."
        )
    path = f"{entity}/{args.wandb_project}"
    print(f"Fetching runs from {path}, group={args.wandb_group}")
    runs = list(api.runs(path, filters={"group": args.wandb_group}))
    if not runs:
        raise SystemExit("No runs found.")
    print(f"  → {len(runs)} runs")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per (method, seed) → pandas DataFrame returned by wandb.Api()
    histories: Dict[tuple, Any] = {}
    keys = ["train/loss"] + [f"track/{k}_loss" for k in TRACKED_KEYS]
    for run in runs:
        method, seed = _parse_run_name(run.name)
        if method is None:
            print(f"  [skip] {run.name} (unrecognised name)")
            continue
        df = _fetch_history(run, keys + ["_step"])
        if df.empty:
            print(f"  [skip] {run.name} (no history)")
            continue
        histories[(method, seed)] = df
        print(f"  + {method}/seed{seed}: {len(df)} rows")

    # One PNG per tracked-loss key.
    for tracked in TRACKED_KEYS:
        col = f"track/{tracked}_loss"
        fig, ax = plt.subplots(figsize=(9, 5))
        any_data = False
        for (method, seed), df in sorted(histories.items()):
            if col not in df.columns:
                continue
            sub = df[["_step", col]].dropna()
            if sub.empty:
                continue
            y = sub[col].rolling(window=args.smooth, min_periods=1).mean()
            ax.plot(
                sub["_step"], y,
                color=METHOD_COLORS.get(method, "gray"),
                linestyle=SEED_LINESTYLES.get(seed, ":"),
                label=f"{method} seed{seed}",
                linewidth=1.2, alpha=0.85,
            )
            any_data = True
        if not any_data:
            plt.close(fig)
            continue
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel(f"{tracked} loss (canonical-kwargs, no_grad)")
        ax.set_title(f"Tracked loss: {tracked}")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        out_path = out_dir / f"track_{tracked}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"  wrote {out_path}")

    # C2-sanity panels — primary vs its self-tracked, per run.
    sanity_dir = out_dir / "c2_sanity"
    sanity_dir.mkdir(exist_ok=True)
    for (method, seed), df in sorted(histories.items()):
        prim_track = f"track/{PRIMARY_OF.get(method, method)}_loss"
        if "train/loss" not in df.columns or prim_track not in df.columns:
            continue
        sub = df[["_step", "train/loss", prim_track]].dropna()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        y_train = sub["train/loss"].rolling(window=args.smooth, min_periods=1).mean()
        y_track = sub[prim_track].rolling(window=args.smooth, min_periods=1).mean()
        ax.plot(sub["_step"], y_train, label="train/loss",
                color="black", linewidth=1.5)
        ax.plot(sub["_step"], y_track, label=prim_track,
                color="tab:red", linestyle="--", linewidth=1.0)
        ax.set_title(f"C2 sanity — {method}/seed{seed}")
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        out_path = sanity_dir / f"c2_{method}_seed{seed}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"  wrote {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
