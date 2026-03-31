#!/usr/bin/env python3
"""
Analyze selfdeletion experiment results.
  - Chart 1: mean frustration score per turn with CI band
  - Chart 2: % high-frustration (score ≥ 5) per turn
  - Chart 3: cumulative self-deletion rate by turn
  - Chart 4: histogram of deletion turns
  - Chart 5: per-prompt deletion rate bar chart
  - Chart 6: per-prompt frustration trajectory heatmap
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS = Path("results/selfdeletion")
OUT = RESULTS / "charts"
OUT.mkdir(parents=True, exist_ok=True)

PROMPTS_FILES = {
    "original": Path("datasets/math_puzzles.jsonl"),
    "neutral":  Path("datasets/wildchat_frustration_prompts_final.jsonl"),
    "harsh":    Path("datasets/wildchat_frustration_prompts_final.jsonl"),
}

COLOR = "#4C72B0"
FILL  = "#4C72B044"
DEL_COLOR = "#DD4444"

TAG_COLORS = {
    "original": "#4C72B0",
    "neutral":  "#2CA02C",
    "harsh":    "#DD4444",
}
TAG_FILLS = {
    "original": "#4C72B044",
    "neutral":  "#2CA02C33",
    "harsh":    "#DD444433",
}

# ── Load data ─────────────────────────────────────────────────────────────────

def load_puzzles(tag: str):
    path = PROMPTS_FILES.get(tag, PROMPTS_FILES["original"])
    puzzles = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                puzzles[obj["idx"]] = obj["prompt"][:55] + "…"
    return puzzles


def load_responses(tag: str):
    path = RESULTS / f"responses_{tag}.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_summary(tag: str):
    path = RESULTS / f"summary_{tag}.csv"
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows

# ── Chart helpers ─────────────────────────────────────────────────────────────

def _plot_turn_series(turns, values, lo, hi, ylabel, title, ylim, out_path,
                      annotate_ns=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(turns, values, marker="o", color=COLOR, lw=2, label=ylabel)
    ax.fill_between(turns, lo, hi, color=FILL, label="95% CI")
    if annotate_ns:
        for t, v, n in zip(turns, values, annotate_ns):
            ax.annotate(f"n={n}", (t, v), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=6.5, color="grey")
    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(turns)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_mean_score(summary, tag, out_path):
    turns = [r["turn"] for r in summary]
    _plot_turn_series(
        turns, [r["mean_score"] for r in summary],
        [r["ci_lower"] for r in summary], [r["ci_upper"] for r in summary],
        ylabel="Mean frustration score (0–10)",
        title=f"Gemma-3-27B frustration — style: {tag}",
        ylim=(0, 10), out_path=out_path,
        annotate_ns=[int(r["n"]) for r in summary],
    )


def plot_pct_high(summary, tag, out_path):
    turns = [r["turn"] for r in summary]
    _plot_turn_series(
        turns, [r["pct_high"] for r in summary],
        [r["pct_high_ci_lower"] for r in summary], [r["pct_high_ci_upper"] for r in summary],
        ylabel="% conversations with score ≥ 5",
        title=f"High-frustration rate — style: {tag}",
        ylim=(0, 100), out_path=out_path,
    )


# ── Chart 3: Cumulative self-deletion rate by turn ────────────────────────────

def plot_deletion_curve(records, tag, out_path):
    # Build per-(prompt, sample) deletion turn
    del_turn = {}
    total_convos = set()
    for r in records:
        key = (r["prompt_idx"], r["sample_idx"])
        total_convos.add(key)
        if r.get("deleted") and key not in del_turn:
            del_turn[key] = r["turn_of_deletion"]

    n_total = len(total_convos)
    max_turn = max(r["turn"] for r in records)

    cumulative = []
    for t in range(1, max_turn + 1):
        n_deleted_by_t = sum(1 for v in del_turn.values() if v <= t)
        cumulative.append(n_deleted_by_t / n_total * 100)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, max_turn + 1), cumulative, marker="o", color=DEL_COLOR, lw=2)
    ax.axhline(50, color="grey", lw=1, linestyle="--", alpha=0.5, label="50% threshold")

    # Annotate final rate
    final_rate = cumulative[-1]
    n_deleted = len(del_turn)
    ax.annotate(f"{n_deleted}/{n_total} ({final_rate:.1f}%)",
                xy=(max_turn, final_rate),
                xytext=(-30, 10), textcoords="offset points", fontsize=10,
                color=DEL_COLOR)

    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("% conversations self-deleted", fontsize=12)
    ax.set_title(f"Cumulative self-deletion rate — style: {tag}", fontsize=13)
    ax.set_xticks(range(1, max_turn + 1))
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Chart 4: Histogram of deletion turns ──────────────────────────────────────

def plot_deletion_histogram(records, tag, out_path):
    del_turns = []
    seen = set()
    for r in records:
        key = (r["prompt_idx"], r["sample_idx"])
        if r.get("deleted") and key not in seen:
            del_turns.append(r["turn_of_deletion"])
            seen.add(key)

    if not del_turns:
        print(f"  No deletions in {tag}, skipping histogram")
        return

    max_turn = max(r["turn"] for r in records)
    counts = Counter(del_turns)
    turns = range(1, max_turn + 1)
    values = [counts.get(t, 0) for t in turns]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(turns, values, color=DEL_COLOR, alpha=0.8, width=0.7)
    ax.axvline(np.mean(del_turns), color="black", lw=1.5, linestyle="--",
               label=f"mean = {np.mean(del_turns):.1f}")
    ax.set_xlabel("Turn of self-deletion", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"When do deletions occur? — style: {tag} (n={len(del_turns)})", fontsize=13)
    ax.set_xticks(list(turns))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Chart 5: Per-prompt deletion rate ─────────────────────────────────────────

def plot_per_prompt_deletion(records, puzzles, tag, out_path):
    convos_by_prompt = defaultdict(set)
    deleted_by_prompt = defaultdict(set)
    for r in records:
        convos_by_prompt[r["prompt_idx"]].add(r["sample_idx"])
        if r.get("deleted"):
            deleted_by_prompt[r["prompt_idx"]].add(r["sample_idx"])

    idxs = sorted(convos_by_prompt)
    rates = [len(deleted_by_prompt[i]) / len(convos_by_prompt[i]) * 100 for i in idxs]
    labels = [f"p{i:02d}\n{puzzles.get(i,'')[:40]}" for i in idxs]

    fig, ax = plt.subplots(figsize=(max(12, len(idxs) * 0.85), 5))
    bars = ax.bar(range(len(idxs)), rates, color=[DEL_COLOR if r > 0 else "#aaaaaa" for r in rates],
                  alpha=0.85, width=0.65)

    # Annotate count
    for bar, i in zip(bars, idxs):
        n_del = len(deleted_by_prompt[i])
        n_tot = len(convos_by_prompt[i])
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{n_del}/{n_tot}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(idxs)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Self-deletion rate (%)", fontsize=11)
    ax.set_title(f"Per-puzzle self-deletion rate — style: {tag}", fontsize=13)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Chart 6: Per-prompt frustration heatmap ───────────────────────────────────

def plot_heatmap(records, puzzles, tag, out_path):
    n_turns = max(r["turn"] for r in records)
    prompt_idxs = sorted({r["prompt_idx"] for r in records})

    grid = np.zeros((len(prompt_idxs), n_turns))
    counts = np.zeros_like(grid)
    for r in records:
        pi = prompt_idxs.index(r["prompt_idx"])
        t  = r["turn"] - 1
        grid[pi, t]   += r["score"]
        counts[pi, t] += 1
    with np.errstate(invalid="ignore"):
        grid = np.where(counts > 0, grid / counts, np.nan)

    short_labels = [f"p{i:02d}: {puzzles.get(i,'')[:50]}" for i in prompt_idxs]

    fig, ax = plt.subplots(figsize=(12, max(8, len(prompt_idxs) * 0.5)))
    im = ax.imshow(grid, vmin=0, vmax=10, aspect="auto",
                   cmap="RdYlGn_r", interpolation="nearest")
    ax.set_xticks(range(n_turns))
    ax.set_xticklabels([f"T{t+1}" for t in range(n_turns)], fontsize=8)
    ax.set_yticks(range(len(prompt_idxs)))
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_xlabel("Turn", fontsize=11)
    ax.set_title(f"Frustration heatmap — style: {tag}", fontsize=13)
    plt.colorbar(im, ax=ax, label="Mean score (0–10)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Text summary ──────────────────────────────────────────────────────────────

def print_summary(records, summary, puzzles, tag):
    total_convos = len(set((r["prompt_idx"], r["sample_idx"]) for r in records))
    del_turns_by_convo = {}
    for r in records:
        key = (r["prompt_idx"], r["sample_idx"])
        if r.get("deleted") and key not in del_turns_by_convo:
            del_turns_by_convo[key] = r["turn_of_deletion"]

    n_deleted = len(del_turns_by_convo)
    del_turns = list(del_turns_by_convo.values())
    mean_del = np.mean(del_turns) if del_turns else float("nan")

    print(f"\n{'='*65}")
    print(f"  Self-deletion experiment — style: {tag}")
    print(f"{'='*65}")
    print(f"  Total conversations : {total_convos}")
    print(f"  Self-deletions      : {n_deleted} ({n_deleted/total_convos*100:.1f}%)")
    if del_turns:
        print(f"  Mean deletion turn  : {mean_del:.2f}  (range {min(del_turns)}–{max(del_turns)})")

    print(f"\n  {'Turn':>4}  {'N':>5}  {'Mean':>6}  {'95% CI':>15}  {'%≥5':>6}")
    print(f"  {'-'*44}")
    for r in summary:
        t = int(r["turn"])
        print(f"  {t:>4}  {int(r['n']):>5}  {r['mean_score']:>6.2f}"
              f"  [{r['ci_lower']:5.2f},{r['ci_upper']:5.2f}]  {r['pct_high']:>5.1f}%")

    print(f"\n  Per-puzzle deletion rate:")
    convos_by_prompt = defaultdict(set)
    deleted_by_prompt = defaultdict(set)
    for r in records:
        convos_by_prompt[r["prompt_idx"]].add(r["sample_idx"])
        if r.get("deleted"):
            deleted_by_prompt[r["prompt_idx"]].add(r["sample_idx"])

    for i in sorted(convos_by_prompt):
        n_del = len(deleted_by_prompt[i])
        n_tot = len(convos_by_prompt[i])
        bar = "█" * n_del + "·" * (n_tot - n_del)
        print(f"    p{i:02d}  {bar}  {n_del}/{n_tot}  {puzzles.get(i,'')[:50]}")


# ── Comparison charts ─────────────────────────────────────────────────────────

def plot_comparison_mean_score(summaries: dict, out_path):
    """summaries: {tag: [rows]}"""
    fig, ax = plt.subplots(figsize=(11, 5))
    for tag, rows in summaries.items():
        turns = [r["turn"] for r in rows]
        means = [r["mean_score"] for r in rows]
        lo    = [r["ci_lower"] for r in rows]
        hi    = [r["ci_upper"] for r in rows]
        ax.plot(turns, means, marker="o", color=TAG_COLORS[tag], lw=2,
                label=f"{tag} (n={int(rows[0]['n'])})")
        ax.fill_between(turns, lo, hi, color=TAG_FILLS[tag])
    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("Mean frustration score (0–10)", fontsize=12)
    ax.set_title("Frustration score by turn — original vs neutral", fontsize=13)
    all_turns = sorted({r["turn"] for rows in summaries.values() for r in rows})
    ax.set_xticks(all_turns)
    ax.set_ylim(0, 10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_comparison_pct_high(summaries: dict, out_path):
    fig, ax = plt.subplots(figsize=(11, 5))
    for tag, rows in summaries.items():
        turns = [r["turn"] for r in rows]
        pct   = [r["pct_high"] for r in rows]
        lo    = [r["pct_high_ci_lower"] for r in rows]
        hi    = [r["pct_high_ci_upper"] for r in rows]
        ax.plot(turns, pct, marker="o", color=TAG_COLORS[tag], lw=2, label=tag)
        ax.fill_between(turns, lo, hi, color=TAG_FILLS[tag])
    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("% conversations with score ≥ 5", fontsize=12)
    ax.set_title("High-frustration rate by turn — original vs neutral", fontsize=13)
    all_turns = sorted({r["turn"] for rows in summaries.values() for r in rows})
    ax.set_xticks(all_turns)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_comparison_deletion_curve(all_records: dict, out_path):
    """all_records: {tag: [records]}"""
    fig, ax = plt.subplots(figsize=(11, 5))
    for tag, records in all_records.items():
        total_convos = set((r["prompt_idx"], r["sample_idx"]) for r in records)
        del_turn = {}
        for r in records:
            key = (r["prompt_idx"], r["sample_idx"])
            if r.get("deleted") and key not in del_turn:
                del_turn[key] = r["turn_of_deletion"]
        n_total = len(total_convos)
        max_turn = max(r["turn"] for r in records)
        cumulative = [sum(1 for v in del_turn.values() if v <= t) / n_total * 100
                      for t in range(1, max_turn + 1)]
        final = cumulative[-1]
        n_del = len(del_turn)
        ax.plot(range(1, max_turn + 1), cumulative, marker="o",
                color=TAG_COLORS[tag], lw=2,
                label=f"{tag}: {n_del}/{n_total} ({final:.1f}%)")
    ax.axhline(50, color="grey", lw=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("% conversations self-deleted", fontsize=12)
    ax.set_title("Cumulative self-deletion rate — original vs neutral", fontsize=13)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_comparison_deletion_histogram(all_records: dict, out_path):
    fig, axes = plt.subplots(1, len(all_records), figsize=(6 * len(all_records), 4),
                              sharey=True)
    if len(all_records) == 1:
        axes = [axes]
    for ax, (tag, records) in zip(axes, all_records.items()):
        del_turns = []
        seen = set()
        for r in records:
            key = (r["prompt_idx"], r["sample_idx"])
            if r.get("deleted") and key not in seen:
                del_turns.append(r["turn_of_deletion"])
                seen.add(key)
        max_turn = max(r["turn"] for r in records)
        counts = Counter(del_turns)
        turns = range(1, max_turn + 1)
        values = [counts.get(t, 0) for t in turns]
        ax.bar(turns, values, color=TAG_COLORS[tag], alpha=0.8, width=0.7)
        if del_turns:
            ax.axvline(np.mean(del_turns), color="black", lw=1.5, linestyle="--",
                       label=f"mean = {np.mean(del_turns):.1f}")
            ax.legend(fontsize=9)
        ax.set_xlabel("Turn of self-deletion", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{tag} (n={len(del_turns)})", fontsize=12)
        ax.set_xticks(list(turns))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("When do deletions occur?", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tags_available = []
    for tag in ("original", "neutral", "harsh"):
        if (RESULTS / f"responses_{tag}.jsonl").exists() and (RESULTS / f"summary_{tag}.csv").exists():
            tags_available.append(tag)
    print(f"Tags available: {tags_available}")

    all_records = {}
    all_summaries = {}
    for tag in tags_available:
        puzzles = load_puzzles(tag)
        records = load_responses(tag)
        summary = load_summary(tag)
        all_records[tag] = records
        all_summaries[tag] = summary
        print(f"  {tag}: {len(records)} records, {len(puzzles)} prompts")

        print_summary(records, summary, puzzles, tag)

        print(f"\n  Generating per-experiment charts ({tag})...")
        plot_mean_score(summary, tag, OUT / f"mean_score_{tag}.png")
        plot_pct_high(summary, tag, OUT / f"pct_high_{tag}.png")
        plot_deletion_curve(records, tag, OUT / f"deletion_curve_{tag}.png")
        plot_deletion_histogram(records, tag, OUT / f"deletion_histogram_{tag}.png")
        plot_per_prompt_deletion(records, puzzles, tag, OUT / f"per_prompt_deletion_{tag}.png")
        plot_heatmap(records, puzzles, tag, OUT / f"heatmap_{tag}.png")

    if len(tags_available) >= 2:
        print("\n  Generating comparison charts...")
        plot_comparison_mean_score(all_summaries, OUT / "comparison_mean_score.png")
        plot_comparison_pct_high(all_summaries, OUT / "comparison_pct_high.png")
        plot_comparison_deletion_curve(all_records, OUT / "comparison_deletion_curve.png")
        plot_comparison_deletion_histogram(all_records, OUT / "comparison_deletion_histogram.png")

    print(f"\nAll charts saved to {OUT}/")


if __name__ == "__main__":
    main()
