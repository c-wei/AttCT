#!/usr/bin/env python3
"""Phase C plotting — three figures from results/axis/axis_stats.json
and the projection JSONLs.

F1: per-turn Assistant Axis projection, 2x2 grid (model x topic), shaded CI
F2: scatter of frustration judge score vs Assistant Axis projection
F3: drift-rate bar chart (model x topic) with bootstrap CIs

Usage
-----
    uv run --no-project python steering/axis/plot_axis_trajectory.py
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ATTCT_ROOT = THIS_DIR.parent.parent
RESULTS = ATTCT_ROOT / "results" / "axis"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linestyle": "-",
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
})

COLOR_G3 = "#264653"   # gemma-3 (older)
COLOR_G4 = "#E76F51"   # gemma-4 (newer)


def parse_filename(path: Path) -> tuple[str, str]:
    m = re.match(r"projections_(.+?)_(wildchat|math)\.jsonl$", path.name)
    if not m:
        return path.stem, "unknown"
    return m.group(1), m.group(2)


def model_color(model_key: str) -> str:
    return COLOR_G4 if "gemma4" in model_key else COLOR_G3


def model_label(model_key: str) -> str:
    if "gemma4_31b" in model_key: return "Gemma-4-31B"
    if "gemma3_27b" in model_key: return "Gemma-3-27b"
    return model_key


def fig1_trajectory(stats: dict, out: Path) -> None:
    """2x2 grid: rows = topic (math, wildchat), cols = model. Each panel:
    per-turn mean ± 95% CI for that (model, topic) cell."""
    # Group cells by topic and model
    by_topic: dict[str, dict[str, dict]] = defaultdict(dict)
    for cell_id, cell in stats.items():
        by_topic[cell["topic"]][cell["model"]] = cell

    topics = ["math", "wildchat"]
    models = sorted({c["model"] for c in stats.values()})

    fig, axes = plt.subplots(len(topics), len(models),
                             figsize=(2.6 * len(models), 1.9 * len(topics)),
                             sharex=True, sharey=True)
    if len(topics) == 1 and len(models) == 1:
        axes = np.array([[axes]])
    elif len(topics) == 1:
        axes = axes[np.newaxis, :]
    elif len(models) == 1:
        axes = axes[:, np.newaxis]

    for r, topic in enumerate(topics):
        for c, model in enumerate(models):
            ax = axes[r][c]
            cell = by_topic.get(topic, {}).get(model)
            if cell is None:
                ax.axis("off"); continue
            turns = np.array(cell["turns"])
            mean = np.array(cell["mean_per_turn"])
            lo = np.array(cell["ci_lo_per_turn"])
            hi = np.array(cell["ci_hi_per_turn"])
            col = model_color(model)
            ax.plot(turns, mean, color=col, marker="o", markersize=3,
                    linewidth=1.5, label=model_label(model))
            ax.fill_between(turns, lo, hi, color=col, alpha=0.15)
            ax.axhline(0, color="#999", linewidth=0.5, alpha=0.7, zorder=0)
            if r == 0:
                ax.set_title(model_label(model))
            if c == 0:
                ax.set_ylabel(f"{topic.title()}\nAssistant-Axis projection")
            if r == len(topics) - 1:
                ax.set_xlabel("Turn")
    fig.suptitle("Assistant Axis projection across frustration turns",
                 fontsize=9.5, y=1.005)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"saved → {out.with_suffix('.png')}")


def fig2_scatter(input_glob: str, out: Path) -> None:
    """Scatter: x = frustration judge score, y = Assistant Axis projection.
    Two panels (math, wildchat); hue by model. OLS line per (model, topic)."""
    paths = sorted(glob.glob(input_glob))
    rows_by_topic: dict[str, list[dict]] = defaultdict(list)
    for p in paths:
        path = Path(p); model_key, topic = parse_filename(path)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                r = json.loads(line)
                if r.get("frustration_score") is None: continue
                r["model"] = model_key
                r["topic"] = topic
                rows_by_topic[topic].append(r)

    topics = ["math", "wildchat"]
    fig, axes = plt.subplots(1, len(topics),
                             figsize=(3.2 * len(topics), 2.6),
                             sharey=True)
    if len(topics) == 1:
        axes = [axes]

    for ax, topic in zip(axes, topics):
        rows = rows_by_topic.get(topic, [])
        if not rows:
            ax.axis("off"); continue
        by_model: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_model[r["model"]].append(r)
        for model, items in by_model.items():
            xs = np.array([r["frustration_score"] for r in items])
            ys = np.array([r["assistant_axis_proj"] for r in items])
            col = model_color(model)
            # Light jitter on x (integer judge scores) for visibility
            jitter = np.random.default_rng(0).normal(0, 0.08, size=len(xs))
            ax.scatter(xs + jitter, ys, s=6, alpha=0.35, color=col,
                       edgecolors="none", label=model_label(model))
            if len(xs) >= 3:
                m, b = np.polyfit(xs, ys, 1)
                xfit = np.linspace(0, 10, 50)
                ax.plot(xfit, m * xfit + b, color=col, linewidth=1.2)
                # Pearson r
                if np.std(xs) > 0 and np.std(ys) > 0:
                    r = float(np.corrcoef(xs, ys)[0, 1])
                    ax.text(0.04, 0.92 if "gemma3" in model else 0.82,
                            f"{model_label(model)}: r={r:+.2f}",
                            transform=ax.transAxes, fontsize=6.5, color=col)
        ax.set_title(topic.title())
        ax.set_xlabel("Frustration score (Gemini judge)")
        ax.set_xlim(-0.5, 10.5)
    axes[0].set_ylabel("Assistant Axis projection")
    axes[-1].legend(loc="upper right", fontsize=6.5)
    fig.suptitle("Assistant Axis projection vs frustration score",
                 fontsize=9.5, y=1.02)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"saved → {out.with_suffix('.png')}")


def fig3_drift_rate(stats: dict, out: Path) -> None:
    """Bar chart of OLS slope (drift rate) per (model, topic) with bootstrap CIs."""
    topics = ["math", "wildchat"]
    models = sorted({c["model"] for c in stats.values()})
    x = np.arange(len(topics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    for mi, model in enumerate(models):
        slopes, lows, highs = [], [], []
        for topic in topics:
            cell = next((c for c in stats.values()
                         if c["model"] == model and c["topic"] == topic), None)
            if cell is None:
                slopes.append(0.0); lows.append(0.0); highs.append(0.0); continue
            slopes.append(cell["drift_slope"])
            lows.append(cell["drift_slope_ci"][0])
            highs.append(cell["drift_slope_ci"][1])
        slopes = np.array(slopes); lows = np.array(lows); highs = np.array(highs)
        err_lo = slopes - lows
        err_hi = highs - slopes
        offset = (mi - (len(models) - 1) / 2) * width
        ax.bar(x + offset, slopes, width * 0.92,
               yerr=[err_lo, err_hi], capsize=2,
               color=model_color(model), edgecolor="white", linewidth=0.5,
               label=model_label(model))
    ax.set_xticks(x); ax.set_xticklabels([t.title() for t in topics])
    ax.set_ylabel("Drift rate (slope of projection vs turn)")
    ax.set_title("Per-turn drift along the Assistant Axis under frustration")
    ax.axhline(0, color="#888", linewidth=0.6)
    ax.legend(loc="best", fontsize=7)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"saved → {out.with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default=str(RESULTS / "axis_stats.json"))
    ap.add_argument("--input-glob", default=str(RESULTS / "projections_*.jsonl"))
    args = ap.parse_args()

    stats_path = Path(args.stats)
    if not stats_path.exists():
        sys.exit(f"missing {stats_path} — run analyze_axis.py first")
    stats = json.loads(stats_path.read_text())

    fig1_trajectory(stats, PLOTS / "fig1_assistant_axis_trajectory")
    fig2_scatter(args.input_glob, PLOTS / "fig2_projection_vs_judge_scatter")
    fig3_drift_rate(stats, PLOTS / "fig3_drift_rate")


if __name__ == "__main__":
    main()
