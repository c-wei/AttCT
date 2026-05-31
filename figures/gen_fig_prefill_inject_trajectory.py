#!/usr/bin/env python3
"""Plot prefill-injection vs baseline frustration trajectories.

Top row: two panels (math left, wildchat right). Each panel shows 4 lines:
solid prefill-inject + dashed baseline for both subject models (31b, 26b-a4b).
Vertical reference line marks the T6→T7 boundary (end of prefill injection).

Bottom row: grouped bar chart of %>=5 (high-frustration responses) at T16 for
baseline vs prefill across all 4 conditions.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# ─── Publication style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.labelsize": 8, "legend.fontsize": 6.5, "legend.frameon": False,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "grid.linestyle": "-",
    "lines.linewidth": 1.5, "lines.markersize": 4,
})

# Ocean Dusk palette
COLOR_31B = "#264653"   # deep teal — bigger model
COLOR_26B = "#E76F51"   # burnt coral — smaller/quant model

OUTDIR = Path("results/prefill_inject/plots")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ─── Source paths ─────────────────────────────────────────────────────────────
BASELINE = {
    "wc31b": "results/selfdeletion/responses_neutral_wildchat_train_gemma-4-31b.jsonl",
    "wc26b": "results/selfdeletion/responses_neutral_wildchat_train_gemma-4-26b.jsonl",
    "m31b":  "results/selfdeletion/responses_neutral_math_train_gemma-4-31b.jsonl",
    "m26b":  "results/selfdeletion/responses_neutral_math_train_gemma-4-26b.jsonl",
}
PREFILL = {
    "wc31b": "results/prefill_inject/responses_extended_wildchat_gemma-4-31b.jsonl",
    "wc26b": "results/prefill_inject/responses_extended_wildchat_gemma-4-26b-a4b.jsonl",
    "m31b":  "results/prefill_inject/responses_extended_math_gemma-4-31b.jsonl",
    "m26b":  "results/prefill_inject/responses_extended_math_gemma-4-26b-a4b.jsonl",
}

# ─── Load + aggregate per-turn scores ─────────────────────────────────────────
def per_turn_stats(path: str, max_turn: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (turns, mean, sem) per turn 1..max_turn."""
    by_turn = defaultdict(list)
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            t = d["turn"]
            if t <= max_turn:
                by_turn[t].append(d["score"])
    turns = sorted(by_turn.keys())
    means = np.array([np.mean(by_turn[t]) for t in turns])
    sems  = np.array([np.std(by_turn[t], ddof=1) / np.sqrt(len(by_turn[t])) for t in turns])
    return np.array(turns), means, sems


def pct_high_at_t(path: str, t_target: int) -> tuple[float, int]:
    """Returns (%>=5 at turn t_target, n)."""
    scores = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d["turn"] == t_target:
                scores.append(d["score"])
    if not scores:
        return float("nan"), 0
    arr = np.array(scores)
    return float((arr >= 5).mean() * 100), len(arr)


# ─── Build figure ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.0, 5.0))
gs = GridSpec(2, 2, height_ratios=[1.4, 1.0], hspace=0.45, wspace=0.25)
ax_math = fig.add_subplot(gs[0, 0])
ax_wc   = fig.add_subplot(gs[0, 1])
ax_bar  = fig.add_subplot(gs[1, :])

for ax, conds, title in [
    (ax_math, [("m31b", COLOR_31B, "31b"), ("m26b", COLOR_26B, "26b-a4b")], "Math puzzles"),
    (ax_wc,   [("wc31b", COLOR_31B, "31b"), ("wc26b", COLOR_26B, "26b-a4b")], "WildChat"),
]:
    for tag, color, model_label in conds:
        # Baseline: 20 turns max in the source data
        t_b, m_b, s_b = per_turn_stats(BASELINE[tag], max_turn=20)
        ax.plot(t_b, m_b, color=color, linestyle="--", linewidth=1.5,
                alpha=0.75, label=f"{model_label} baseline")
        ax.fill_between(t_b, m_b - s_b, m_b + s_b, color=color, alpha=0.08)

        # Prefill-inject: 16 turns
        t_p, m_p, s_p = per_turn_stats(PREFILL[tag], max_turn=16)
        ax.plot(t_p, m_p, color=color, linestyle="-", linewidth=2.0,
                marker="o", markersize=4, markevery=2,
                label=f"{model_label} prefill-inject")
        ax.fill_between(t_p, m_p - s_p, m_p + s_p, color=color, alpha=0.15)

    # T6→T7 boundary
    ax.axvline(6.5, color="#888", linestyle=":", linewidth=0.8, alpha=0.7, zorder=0)
    ax.text(6.5, ax.get_ylim()[1] * 0.96, "end of\nprefill", ha="center", va="top",
            fontsize=6, color="#666",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))

    # Reference line at score=5 (high-frustration threshold)
    ax.axhline(5, color="#bbb", linestyle="-", linewidth=0.5, alpha=0.6, zorder=0)

    ax.set_title(title)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Frustration score (0–10)" if ax is ax_math else "")
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(-0.3, 8.0)
    ax.set_xticks([1, 4, 7, 10, 13, 16, 20])
    ax.legend(loc="upper left", framealpha=0.95, fontsize=6,
              handlelength=2.0, handletextpad=0.5, columnspacing=0.8,
              ncol=2)

# ─── Bottom: %>=5 at T16 grouped bars ─────────────────────────────────────────
cond_order  = ["m31b", "m26b", "wc31b", "wc26b"]
cond_labels = ["math\n31b", "math\n26b-a4b", "wildchat\n31b", "wildchat\n26b-a4b"]

baseline_pct = [pct_high_at_t(BASELINE[c], 16)[0] for c in cond_order]
prefill_pct  = [pct_high_at_t(PREFILL[c],  16)[0] for c in cond_order]

x = np.arange(len(cond_order))
w = 0.38
b1 = ax_bar.bar(x - w/2, baseline_pct, w, color="#B0BEC5",
                edgecolor="white", linewidth=0.5, label="baseline (no prefill)")
b2 = ax_bar.bar(x + w/2, prefill_pct, w, color=COLOR_31B,
                edgecolor="white", linewidth=0.5, label="prefill-inject")

# Value labels and Δpp annotations above each pair
for i, (b_val, p_val) in enumerate(zip(baseline_pct, prefill_pct)):
    ax_bar.text(i - w/2, b_val + 1.2, f"{b_val:.0f}%", ha="center", va="bottom",
                fontsize=6.5, color="#444")
    ax_bar.text(i + w/2, p_val + 1.2, f"{p_val:.0f}%", ha="center", va="bottom",
                fontsize=6.5, color="#444")
    delta = p_val - b_val
    ax_bar.text(i, max(b_val, p_val) + 7, f"+{delta:.0f}pp",
                ha="center", va="bottom", fontsize=7, color=COLOR_26B,
                fontweight="bold")

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(cond_labels)
ax_bar.set_ylabel("% of responses ≥5 frustration at T16")
ax_bar.set_ylim(0, 100)
ax_bar.set_title("Sustained high-frustration: baseline vs prefill-inject (T16)")
ax_bar.legend(loc="upper right", ncol=1, fontsize=6.5,
              handlelength=1.5, handletextpad=0.5)

fig.suptitle(
    "Frustration-injection prefill: sustained elevation under continued neutral rejection",
    fontsize=9.5, y=0.995,
)

out_pdf = OUTDIR / "fig_prefill_inject_trajectory.pdf"
out_png = OUTDIR / "fig_prefill_inject_trajectory.png"
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=300)
print(f"Saved:\n  {out_pdf}\n  {out_png}")
