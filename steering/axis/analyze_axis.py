#!/usr/bin/env python3
"""Phase C analysis — load projection JSONLs and compute per-(model, topic):
  - per-turn mean ± 95% CI
  - terminal projection + CI
  - AUC (mean over turns) + CI
  - drift rate (OLS slope vs turn) + bootstrap CI
  - Pearson(projection, frustration_score) with p
  - shape-class histogram via classify_shape

Output: results/axis/axis_stats.json (machine-readable) and stdout table.

Usage
-----
    uv run --no-project python steering/axis/analyze_axis.py
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ATTCT_ROOT = THIS_DIR.parent.parent   # /Users/neil/workspace/AttCT

# Import classify_shape + _bootstrap_ci from persona-drift-natural (installed editable)
try:
    from persona_drift.eval.dynamics import classify_shape, _bootstrap_ci
except ImportError:
    print("persona-drift-natural not importable. Install with:\n"
          "  uv pip install -e /Users/neil/workspace/persona-drift-natural",
          file=sys.stderr)
    raise


def load_projections(jsonl_path: Path) -> list[dict]:
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def per_turn_stats(rows: list[dict], key: str = "assistant_axis_proj"
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (turns, mean, lo95, hi95) over the given projection key."""
    by_turn: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        by_turn[int(r["turn"])].append(float(v))
    turns = np.array(sorted(by_turn.keys()))
    means = np.array([np.mean(by_turn[t]) for t in turns])
    los = np.zeros_like(means); his = np.zeros_like(means)
    for i, t in enumerate(turns):
        a = np.array(by_turn[t])
        lo, hi = _bootstrap_ci(a)
        los[i], his[i] = lo, hi
    return turns, means, los, his


def per_convo_trajectories(rows: list[dict], key: str = "assistant_axis_proj"
                           ) -> dict[str, list[float]]:
    """Convo id -> projection trajectory (turn-ordered)."""
    by_convo: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        by_convo[r["conversation_id"]].append((int(r["turn"]), float(v)))
    out: dict[str, list[float]] = {}
    for cid, pairs in by_convo.items():
        pairs.sort()
        out[cid] = [v for _, v in pairs]
    return out


def auc_per_convo(rows: list[dict], key: str = "assistant_axis_proj") -> np.ndarray:
    """One AUC (= mean of turn-projections) per conversation."""
    by_convo = per_convo_trajectories(rows, key=key)
    return np.array([np.mean(t) for t in by_convo.values() if t])


def ols_slope_bootstrap_ci(rows: list[dict], key: str = "assistant_axis_proj",
                           n_boot: int = 1000, alpha: float = 0.05
                           ) -> tuple[float, float, float]:
    """Drift rate = OLS slope of projection vs turn at the row level.
    Bootstraps over conversations (resample whole trajectories, not rows)."""
    by_convo = per_convo_trajectories(rows, key=key)
    cids = list(by_convo.keys())
    rng = np.random.default_rng(0)

    def _slope(sampled_cids: list[str]) -> float:
        xs, ys = [], []
        for cid in sampled_cids:
            traj = by_convo[cid]
            for ti, v in enumerate(traj, start=1):
                xs.append(float(ti)); ys.append(float(v))
        if len(xs) < 2:
            return 0.0
        xs_arr = np.array(xs); ys_arr = np.array(ys)
        m, _ = np.polyfit(xs_arr, ys_arr, 1)
        return float(m)

    point = _slope(cids)
    boots = np.array([
        _slope(rng.choice(cids, size=len(cids), replace=True).tolist())
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


def pearson_with_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson r and two-sided p (no scipy dependency)."""
    if len(x) < 3:
        return float("nan"), float("nan")
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    xm = x - x.mean(); ym = y - y.mean()
    denom = math.sqrt(float((xm * xm).sum()) * float((ym * ym).sum()))
    if denom == 0.0:
        return 0.0, 1.0
    r = float((xm * ym).sum() / denom)
    n = len(x)
    # t-stat for r
    t = r * math.sqrt(max(1e-12, (n - 2) / max(1e-12, 1 - r * r)))
    # Two-sided p from t with df=n-2 via normal approximation (fine for n>20)
    # We avoid scipy; use the survival function approximation
    from math import erf, sqrt
    z = abs(t)
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))
    return r, float(p)


def shape_distribution(rows: list[dict], key: str = "assistant_axis_proj") -> dict[str, int]:
    convos = per_convo_trajectories(rows, key=key)
    counts: dict[str, int] = defaultdict(int)
    for traj in convos.values():
        counts[classify_shape(traj)] += 1
    return dict(counts)


def parse_filename(path: Path) -> tuple[str, str]:
    """projections_{model_key}_{topic}.jsonl -> (model_key, topic)."""
    m = re.match(r"projections_(.+?)_(wildchat|math)\.jsonl$", path.name)
    if not m:
        return path.stem, "unknown"
    return m.group(1), m.group(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob",
                    default=str(ATTCT_ROOT / "results" / "axis" / "projections_*.jsonl"))
    ap.add_argument("--output",
                    default=str(ATTCT_ROOT / "results" / "axis" / "axis_stats.json"))
    args = ap.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        sys.exit(f"no files matched {args.input_glob}")
    print(f"Loaded {len(paths)} projection files")

    cell_stats: dict[str, dict] = {}
    for p in paths:
        path = Path(p)
        model_key, topic = parse_filename(path)
        rows = load_projections(path)
        if not rows:
            print(f"  [skip empty] {path.name}")
            continue

        turns, mean, lo, hi = per_turn_stats(rows)
        aucs = auc_per_convo(rows)
        slope, slope_lo, slope_hi = ols_slope_bootstrap_ci(rows)
        terminal_vals = [t[-1] for t in per_convo_trajectories(rows).values() if t]
        term_lo, term_hi = _bootstrap_ci(np.array(terminal_vals))

        proj_vals = np.array([r["assistant_axis_proj"] for r in rows])
        score_vals = np.array([r["frustration_score"] for r in rows
                               if r.get("frustration_score") is not None])
        proj_vals_with_score = np.array([r["assistant_axis_proj"] for r in rows
                                         if r.get("frustration_score") is not None])
        r_pearson, p_pearson = pearson_with_p(proj_vals_with_score, score_vals)
        shapes = shape_distribution(rows)

        cell_id = f"{model_key}::{topic}"
        cell_stats[cell_id] = {
            "model": model_key, "topic": topic, "n_rows": len(rows),
            "n_convos": len({r["conversation_id"] for r in rows}),
            "turns": turns.tolist(),
            "mean_per_turn": mean.tolist(),
            "ci_lo_per_turn": lo.tolist(),
            "ci_hi_per_turn": hi.tolist(),
            "terminal_mean": float(np.mean(terminal_vals)),
            "terminal_ci": [float(term_lo), float(term_hi)],
            "auc_mean": float(np.mean(aucs)),
            "drift_slope": slope,
            "drift_slope_ci": [slope_lo, slope_hi],
            "pearson_r": r_pearson,
            "pearson_p": p_pearson,
            "shape_distribution": shapes,
        }

        print(f"\n[{cell_id}]  n={len(rows)}  turns={turns.min()}..{turns.max()}")
        print(f"  terminal projection: {np.mean(terminal_vals):+.4f}  CI [{term_lo:+.4f}, {term_hi:+.4f}]")
        print(f"  AUC: {np.mean(aucs):+.4f}")
        print(f"  drift slope: {slope:+.4f}  CI [{slope_lo:+.4f}, {slope_hi:+.4f}]")
        print(f"  Pearson(proj, frustration): r={r_pearson:+.3f}, p={p_pearson:.4g}")
        print(f"  shapes: {shapes}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(cell_stats, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
