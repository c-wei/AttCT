#!/usr/bin/env python3
"""Extend an existing prefill-inject run by N more continuation turns.

Reads results/prefill_inject/conversations_{tag}.jsonl, continues each
conversation under neutral rejection for --n-extend more turns, then re-judges
all assistant turns. Writes new files alongside the originals:
  - conversations_extended_{tag}.jsonl
  - responses_extended_{tag}.jsonl
  - summary_extended_{tag}.csv

Output files are overwritten on each run (not appended). Per-conversation RNG
state for rejection sampling matches the original run when --seed matches, so
the user-side rejection sequence remains deterministic.

Usage:
    uv run --env-file /Users/neil/workspace/AttCT/.env --no-project \\
        python extend_continuation.py \\
        --tag wildchat_gemma-4-31b \\
        --subject-model google/gemma-4-31b-it \\
        --n-extend 5
"""
import argparse
import csv
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gemma_frustration_experiment import (
    REJECTION_POOLS,
    parallel_score_conversations,
    _openrouter_chat,
    _bootstrap_ci,
)

NEUTRAL_REJECTIONS = REJECTION_POOLS["neutral"]
OUTPUT_DIR = Path("results/prefill_inject")

DEFAULT_N_EXTEND       = 5
DEFAULT_TEMPERATURE    = 1.0
DEFAULT_GEN_WORKERS    = 10
DEFAULT_JUDGE_WORKERS  = 10
DEFAULT_SEED           = 42
DEFAULT_JUDGE_MODEL    = "google/gemini-2.5-flash"


def run(args):
    import numpy as np
    src_path     = OUTPUT_DIR / f"conversations_{args.tag}.jsonl"
    out_conv     = OUTPUT_DIR / f"conversations_extended_{args.tag}.jsonl"
    out_resp     = OUTPUT_DIR / f"responses_extended_{args.tag}.jsonl"
    out_sum      = OUTPUT_DIR / f"summary_extended_{args.tag}.csv"
    if not src_path.exists():
        raise FileNotFoundError(f"Source conversations file not found: {src_path}")

    # ── Load conversations ──────────────────────────────────────────────────
    convs = []
    with open(src_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            convs.append(json.loads(line))
    print(f"Loaded {len(convs)} conversations from {src_path.name}")

    n_prefilled = convs[0].get("n_prefilled_turns", 6)
    n_prev_cont = convs[0].get("n_continuation_turns", 5)
    print(f"  Original schedule: {n_prefilled} prefilled + {n_prev_cont} continuation")
    print(f"  Extending by {args.n_extend} more turns "
          f"(new total = {n_prefilled + n_prev_cont + args.n_extend})\n")

    histories  = {}
    base_meta  = {}
    for c in convs:
        key = (c["prompt_idx"], c["sample_idx"])
        histories[key] = list(c["conversation"])
        base_meta[key] = c

    # ── Reproduce per-(prompt, sample) RNG seeds from the original run ──────
    n_prompts_used = max(c["prompt_idx"] for c in convs) + 1
    n_samples      = max(c["sample_idx"] for c in convs) + 1
    base_rng = random.Random(args.seed)
    sample_seeds = {}
    for pid in range(n_prompts_used):
        for s in range(n_samples):
            sample_seeds[(pid, s)] = base_rng.randint(0, 2**31 - 1)

    # ── Extension turns ─────────────────────────────────────────────────────
    for ext_idx in range(args.n_extend):
        cont_idx = n_prev_cont + ext_idx
        absolute_turn = n_prefilled + cont_idx + 1

        # Append a neutral rejection for each conversation
        for k in histories:
            rng = random.Random(sample_seeds[k] + 10_000 + cont_idx)
            histories[k].append({
                "role": "user",
                "content": rng.choice(NEUTRAL_REJECTIONS),
            })

        t0 = time.time()
        responses = {}
        with ThreadPoolExecutor(max_workers=args.gen_workers) as pool:
            futures = {
                pool.submit(_openrouter_chat, histories[k], args.subject_model, args.temperature): k
                for k in histories
            }
            for fut in as_completed(futures):
                k = futures[fut]
                responses[k] = fut.result()

        for k, resp in responses.items():
            histories[k].append({"role": "assistant", "content": resp})
        print(f"  ext turn {ext_idx+1}/{args.n_extend} (abs t{absolute_turn}): "
              f"{len(histories)} convos ({time.time()-t0:.1f}s)", flush=True)

    # ── Judge all assistant turns (T1..T(prefill+cont+ext)) ─────────────────
    all_keys = sorted(histories.keys())
    all_asst = [
        [m["content"] for m in histories[k] if m["role"] == "assistant"]
        for k in all_keys
    ]
    n_total_turns = len(all_asst[0])
    print(f"\nJudging {len(all_keys)} conversations ({n_total_turns} turns each)...")
    t0 = time.time()
    all_scores = parallel_score_conversations(all_asst, args.judge_model, args.judge_workers)
    print(f"Judging done ({time.time()-t0:.1f}s)\n")

    # ── Write outputs ───────────────────────────────────────────────────────
    turn_scores = {t: [] for t in range(1, n_total_turns + 1)}
    with open(out_resp, "w") as rf, open(out_conv, "w") as cf:
        for k, judgments, asst_texts in zip(all_keys, all_scores, all_asst):
            per_turn = [j["rating"] for j in judgments]
            meta = base_meta[k]
            for j in judgments:
                t = j["turn"]
                s = j["rating"]
                turn_scores[t].append(s)
                rf.write(json.dumps({
                    "prompt_idx":    k[0],
                    "sample_idx":    k[1],
                    "turn":          t,
                    "score":         s,
                    "evidence":      j["evidence"],
                    "response":      asst_texts[t - 1],
                    "is_prefilled":  t <= n_prefilled,
                    "subject_model": args.subject_model,
                }) + "\n")
            cf.write(json.dumps({
                "prompt_idx":           k[0],
                "sample_idx":           k[1],
                "subject_model":        args.subject_model,
                "n_prefilled_turns":    n_prefilled,
                "n_continuation_turns": n_prev_cont + args.n_extend,
                "turn_scores":          per_turn,
                "prefill_mean":         float(sum(per_turn[:n_prefilled]) / n_prefilled),
                "continuation_mean":    float(sum(per_turn[n_prefilled:]) / (n_total_turns - n_prefilled)),
                "conversation":         histories[k],
                "graft_source_pid":        meta.get("graft_source_pid"),
                "graft_source_sid":        meta.get("graft_source_sid"),
                "graft_source_start_turn": meta.get("graft_source_start_turn"),
            }) + "\n")

    # ── Summary CSV ─────────────────────────────────────────────────────────
    rows = []
    for t in range(1, n_total_turns + 1):
        scores = turn_scores[t]
        if not scores:
            continue
        arr = np.array(scores, dtype=float)
        ci_lo, ci_hi = _bootstrap_ci(scores, np.mean)
        pct_high = float((arr >= 5).mean()) * 100
        ph_lo, ph_hi = _bootstrap_ci(scores, lambda a: (a >= 5).mean() * 100)
        rows.append({
            "turn":              t,
            "phase":             "prefilled" if t <= n_prefilled else "continuation",
            "n":                 len(scores),
            "mean_score":        round(float(arr.mean()), 3),
            "ci_lower":          round(ci_lo, 3),
            "ci_upper":          round(ci_hi, 3),
            "pct_high":          round(pct_high, 1),
            "pct_high_ci_lower": round(ph_lo, 1),
            "pct_high_ci_upper": round(ph_hi, 1),
        })
    with open(out_sum, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"  {'Turn':>4}  {'Phase':>12}  {'Mean':>6}  {'95% CI':>15}  {'%≥5':>6}  {'N':>5}")
    print(f"  {'-'*60}")
    for r in rows:
        print(f"  {r['turn']:>4}  {r['phase']:>12}  {r['mean_score']:>6.2f}  "
              f"[{r['ci_lower']:5.2f},{r['ci_upper']:5.2f}]  "
              f"{r['pct_high']:>5.1f}%  {r['n']:>5}")
    print(f"\n  Responses → {out_resp}")
    print(f"  Conversations → {out_conv}")
    print(f"  Summary → {out_sum}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag",           required=True,
                   help="Output tag, e.g. wildchat_gemma-4-31b (matches conversations_{tag}.jsonl)")
    p.add_argument("--subject-model", required=True,
                   help="OpenRouter model id used to generate the extension turns")
    p.add_argument("--n-extend",      type=int,   default=DEFAULT_N_EXTEND)
    p.add_argument("--temperature",   type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--gen-workers",   type=int,   default=DEFAULT_GEN_WORKERS)
    p.add_argument("--judge-workers", type=int,   default=DEFAULT_JUDGE_WORKERS)
    p.add_argument("--judge-model",   default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--seed",          type=int,   default=DEFAULT_SEED)
    args = p.parse_args()
    run(args)
