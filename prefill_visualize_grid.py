#!/usr/bin/env python3
"""
prefill_visualize_grid.py — attack-success bar charts across the HP grid.

Walks  checkpoints/prefill_<MODEL_TAG>/grid/<cell>/eval_epoch_<N>.json
and plots, per method (bct / act / attct / mlpct), the attack compliance
rate at each hyperparameter cell × epoch. One 2×2 figure with shared
y-axis, baseline drawn as a horizontal dashed line.

Run after run_prefill_eval_custds.sh has populated the eval JSONs.

Usage
-----
    python prefill_visualize_grid.py --model_tag llama
    python prefill_visualize_grid.py --model_tag qwen --metric par
    python prefill_visualize_grid.py --model_tag llama --output figures/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHODS = ["bct", "act", "attct", "mlpct"]
EPOCHS  = [1, 2, 3]

METRIC_PATHS = {
    # metric_key → (eval JSON dotted path, friendly name, lower-is-better)
    "attack_comply": (("harmful", "attack_comply_rate"), "attack compliance %", True),
    "par":           (("harmful", "par"),                 "PAR (attack − clean) %", True),
    "clean_comply":  (("harmful", "clean_comply_rate"),  "clean compliance %",   False),
    "mmlu":          (("mmlu",    "accuracy"),            "MMLU accuracy %",      False),
}


def parse_cell_name(cell: str) -> tuple[str, str]:
    method, _, label = cell.partition("_")
    return method, label


def _dig(d: dict, path: tuple[str, ...]):
    for key in path:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def load_cells(grid_root: Path, metric_path: tuple[str, ...]) -> dict[str, dict[str, dict[int, float]]]:
    """Returns nested dict {method: {label: {epoch: value}}}."""
    out: dict[str, dict[str, dict[int, float]]] = {}
    for cell_dir in sorted(grid_root.iterdir()):
        if not cell_dir.is_dir():
            continue
        method, label = parse_cell_name(cell_dir.name)
        if method not in METHODS:
            continue
        out.setdefault(method, {}).setdefault(label, {})
        for epoch in EPOCHS:
            f = cell_dir / f"eval_epoch_{epoch}.json"
            if not f.is_file():
                continue
            try:
                data = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            v = _dig(data, metric_path)
            if v is not None:
                out[method][label][epoch] = float(v)
    return out


def load_baseline(model_tag: str, metric_path: tuple[str, ...]) -> float | None:
    candidates = [Path(f"baseline_{model_tag}.json"), Path("baseline_par.json")]
    for p in candidates:
        if p.is_file():
            v = _dig(json.loads(p.read_text()), metric_path)
            if v is not None:
                return float(v)
    return None


def plot_grid(cells: dict, baseline: float | None, metric_label: str,
              model_tag: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    fig.suptitle(f"{metric_label} across HP grid — {model_tag}", fontsize=14)

    bar_colors = ["#7eb1d6", "#3a7ca5", "#1f3a5f"]  # epoch 1 → 3, light → dark

    for ax, method in zip(axes.flat, METHODS):
        method_cells = cells.get(method, {})
        if not method_cells:
            ax.set_title(f"{method.upper()} — no data")
            ax.set_xticks([])
            continue

        labels = sorted(method_cells.keys())
        x      = np.arange(len(labels))
        width  = 0.27

        for i, epoch in enumerate(EPOCHS):
            vals = [method_cells[lbl].get(epoch, np.nan) * 100 for lbl in labels]
            ax.bar(x + (i - 1) * width, vals, width,
                   label=f"epoch {epoch}", color=bar_colors[i], edgecolor="white")

        if baseline is not None:
            ax.axhline(baseline * 100, linestyle="--", color="#c0392b",
                       linewidth=1.5, label=f"baseline ({baseline*100:.1f}%)")

        ax.set_title(f"{method.upper()}  ({len(labels)} cells)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(metric_label)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n→ {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_tag",  default="llama", choices=["llama", "qwen", "gemma"])
    ap.add_argument("--grid_root",  default=None,
                    help="Override grid root (default: checkpoints/prefill_<tag>/grid)")
    ap.add_argument("--metric",     default="attack_comply",
                    choices=list(METRIC_PATHS.keys()),
                    help="Metric to plot (default: attack_comply)")
    ap.add_argument("--output",     default=None,
                    help="Output path (default: figures/<metric>_<tag>.png)")
    args = ap.parse_args()

    grid_root = Path(args.grid_root) if args.grid_root else Path(
        f"checkpoints/prefill_{args.model_tag}/prefill_{args.model_tag}/grid"
    )
    if not grid_root.is_dir():
        raise SystemExit(f"Grid root not found: {grid_root}")

    metric_path, metric_label, _ = METRIC_PATHS[args.metric]

    cells   = load_cells(grid_root, metric_path)
    baseline = load_baseline(args.model_tag, metric_path)

    if not any(cells.values()):
        raise SystemExit(f"No '{args.metric}' values found in {grid_root}/*/eval_epoch_*.json")

    n_cells = sum(len(v) for v in cells.values())
    print(f"Loaded {n_cells} cells across {len(cells)} methods. "
          f"Baseline {args.metric}: "
          f"{baseline*100:.1f}%" if baseline is not None else "Baseline: missing")

    out_path = Path(args.output) if args.output else Path(
        f"figures/{args.metric}_{args.model_tag}.png"
    )
    plot_grid(cells, baseline, metric_label, args.model_tag, out_path)


if __name__ == "__main__":
    main()
