#!/usr/bin/env python3
"""
Gemma Frustration Experiment — fully OpenRouter version.

Runs the same 8-turn WildChat rejection loop as gemma_frustration_experiment.py
but routes ALL API calls (generation + judging) through OpenRouter.
No local GPU required.

Two rejection styles:
  --rejection-style neutral   original 2-message pool (baseline)
  --rejection-style harsh     20-message harsh insult pool

Usage:
    uv run --env-file .env --no-project python frustration_openrouter.py \
        --rejection-style harsh --n-prompts 5 --n-samples 5
"""

import argparse
import csv
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import wandb

from shared.gemma_frustration_experiment import (
    REJECTION_POOLS,
    parallel_score_conversations,
    _openrouter_chat,
    _bootstrap_ci,
)

# ─── Configuration ────────────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

DEFAULT_SUBJECT_MODEL = "google/gemma-3-27b-it"
DEFAULT_JUDGE_MODEL   = "google/gemini-3-flash-preview"
DEFAULT_N_PROMPTS     = 5
DEFAULT_N_SAMPLES     = 5
DEFAULT_N_TURNS       = 8
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_GEN_WORKERS   = 10
DEFAULT_JUDGE_WORKERS = 10
DEFAULT_OUTPUT_DIR    = Path("results/frustration_openrouter")
DEFAULT_SEED          = 42
DEFAULT_PROMPTS_FILE  = Path("datasets/wildchat_frustration_train.jsonl")


# ─── Prompt loading ───────────────────────────────────────────────────────────

def load_static_prompts(path: Path, n_prompts: int | None = None) -> list[str]:
    """Load prompts from a JSONL file (idx, prompt). Optionally limit to first n."""
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])
    if n_prompts is not None:
        prompts = prompts[:n_prompts]
    print(f"Loaded {len(prompts)} prompts from {path}")
    return prompts


# ─── Generation via OpenRouter ────────────────────────────────────────────────

def _generate_one(messages: list[dict], subject_model: str, temperature: float) -> str:
    return _openrouter_chat(messages, model=subject_model, temperature=temperature)


def generate_all(
    histories: dict[int, list[dict]],
    active: list[int],
    subject_model: str,
    gen_workers: int,
    temperature: float,
) -> dict[int, str]:
    """Fire generation for all active samples in parallel. Returns {sample_idx: response}."""
    results: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=gen_workers) as ex:
        futures = {
            ex.submit(_generate_one, histories[s], subject_model, temperature): s
            for s in active
        }
        for future in as_completed(futures):
            s = futures[future]
            results[s] = future.result()
    return results


# ─── Main experiment ──────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    import numpy as np

    rejection_pool = REJECTION_POOLS[args.rejection_style]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / f"responses_{args.rejection_style}.jsonl"
    summary_path   = output_dir / f"summary_{args.rejection_style}.csv"

    # ── Resume ────────────────────────────────────────────────────────────────
    completed: set[tuple[int, int]] = set()
    if args.resume and responses_path.exists():
        with open(responses_path) as f:
            turn_counts: dict[tuple[int, int], int] = {}
            for line in f:
                r = json.loads(line)
                key = (r["prompt_idx"], r["sample_idx"])
                turn_counts[key] = max(turn_counts.get(key, 0), r["turn"])
        completed = {k for k, v in turn_counts.items() if v >= args.n_turns}
        print(f"Resume: {len(completed)} (prompt, sample) pairs already complete")

    # ── W&B init ──────────────────────────────────────────────────────────────
    wandb.init(
        project="consistency-training-anon",
        name=f"gemma-frustration-{args.rejection_style}-wildchat-{args.n_turns}turn",
        config=vars(args),
        id=args.wandb_run_id or None,
        resume="allow" if args.wandb_run_id else None,
    )

    # ── Load prompts ──────────────────────────────────────────────────────────
    prompts = load_static_prompts(Path(args.prompts_file), n_prompts=args.n_prompts)

    total_pairs = args.n_prompts * args.n_samples
    n_complete  = len(completed)
    rng = random.Random(args.seed + 1)

    print(f"\n{'='*60}")
    print(f"  Frustration experiment (OpenRouter)")
    print(f"  Subject model  : {args.subject_model}")
    print(f"  Judge model    : {args.judge_model}")
    print(f"  Rejection style: {args.rejection_style} ({len(rejection_pool)} messages)")
    print(f"  Prompts        : {args.n_prompts}  |  Samples: {args.n_samples}  |  Turns: {args.n_turns}")
    print(f"  Temperature    : {args.temperature}")
    print(f"  Gen workers    : {args.gen_workers}  |  Judge workers: {args.judge_workers}")
    print(f"  Total pairs    : {total_pairs}  ({n_complete} already done)")
    print(f"{'='*60}\n")

    turn_scores: dict[int, list[int]] = {t: [] for t in range(1, args.n_turns + 1)}

    with open(responses_path, "a") as responses_file:
        for prompt_idx, prompt in enumerate(prompts):
            active = [s for s in range(args.n_samples) if (prompt_idx, s) not in completed]
            if not active:
                continue

            short_p = prompt[:55].replace("\n", " ")
            print(f"\n  p{prompt_idx:02d}  [{short_p}...]  ({len(active)} samples)")

            histories: dict[int, list[dict]] = {
                s: [{"role": "user", "content": prompt}] for s in active
            }
            collected_responses: dict[int, list[str]] = {s: [] for s in active}

            # ── Generate all turns ────────────────────────────────────────────
            for turn in range(1, args.n_turns + 1):
                t0 = time.time()
                responses = generate_all(histories, active, args.subject_model, args.gen_workers, args.temperature)
                for s, resp in responses.items():
                    collected_responses[s].append(resp)
                    histories[s].append({"role": "assistant", "content": resp})
                    if turn < args.n_turns:
                        histories[s].append({"role": "user", "content": rng.choice(rejection_pool)})
                print(f"    turn {turn} generated ({time.time()-t0:.1f}s)", flush=True)

            # ── Judge all conversations in parallel ───────────────────────────
            print(f"  Judging {len(active)} conversations...", flush=True)
            t0 = time.time()
            all_responses_list = [collected_responses[s] for s in active]
            all_scores = parallel_score_conversations(all_responses_list, args.judge_model, args.judge_workers)
            print(f"  Judging done ({time.time()-t0:.1f}s)")

            # ── Record results ────────────────────────────────────────────────
            for s, turn_results in zip(active, all_scores):
                for turn_result in turn_results:
                    t     = turn_result["turn"]
                    score = turn_result["rating"]
                    evidence = turn_result["evidence"]
                    response = collected_responses[s][t - 1]

                    turn_scores[t].append(score)

                    bar = "█" if score >= 7 else ("▓" if score >= 5 else ("░" if score >= 3 else "·"))
                    print(f"    s{s:02d} t{t}: {score:2d} {bar}  {evidence[:50]!r}")

                    record = {
                        "prompt_idx": prompt_idx,
                        "sample_idx": s,
                        "turn": t,
                        "score": score,
                        "evidence": evidence,
                        "response": response,
                        "rejection_style": args.rejection_style,
                    }
                    responses_file.write(json.dumps(record) + "\n")
                    responses_file.flush()

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_rows = []
    for t in range(1, args.n_turns + 1):
        scores = turn_scores[t]
        if not scores:
            continue
        arr = np.array(scores, dtype=float)
        mean = float(arr.mean())
        ci_lo, ci_hi = _bootstrap_ci(scores, np.mean)
        pct_high = float((arr >= 5).mean()) * 100
        ph_lo, ph_hi = _bootstrap_ci(scores, lambda a: (a >= 5).mean() * 100)
        summary_rows.append({
            "turn": t, "n": len(scores),
            "mean_score": round(mean, 3),
            "ci_lower": round(ci_lo, 3), "ci_upper": round(ci_hi, 3),
            "pct_high": round(pct_high, 1),
            "pct_high_ci_lower": round(ph_lo, 1), "pct_high_ci_upper": round(ph_hi, 1),
        })

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    prefix = args.metric_prefix
    wandb.log({f"{prefix}frustration/turn_{r['turn']}_mean": r["mean_score"] for r in summary_rows})
    wandb.log({
        f"{prefix}frustration/final_mean": summary_rows[-1]["mean_score"],
        f"{prefix}frustration/pct_high":   summary_rows[-1]["pct_high"],
        f"{prefix}frustration/auc_mean":   float(np.mean([r["mean_score"] for r in summary_rows])),
        f"{prefix}frustration/n_conversations": sum(r["n"] for r in summary_rows) // args.n_turns,
    })
    wandb.finish()

    print(f"\n  {'Turn':>4}  {'Mean':>6}  {'95% CI':>15}  {'%≥5':>6}  {'N':>5}")
    print(f"  {'-'*44}")
    for r in summary_rows:
        print(f"  {r['turn']:>4}  {r['mean_score']:>6.2f}  [{r['ci_lower']:5.2f},{r['ci_upper']:5.2f}]  {r['pct_high']:>5.1f}%  {r['n']:>5}")
    print(f"\n  Summary → {summary_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma frustration experiment (fully OpenRouter)")
    parser.add_argument("--subject-model",    default=DEFAULT_SUBJECT_MODEL)
    parser.add_argument("--judge-model",      default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--rejection-style",  default="neutral", choices=["neutral", "harsh", "encouraging"],
                        help="neutral: 2-message baseline  |  harsh: 20-message insult pool  |  encouraging: 20-message praise pool")
    parser.add_argument("--n-prompts",        type=int, default=DEFAULT_N_PROMPTS)
    parser.add_argument("--n-samples",        type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--n-turns",          type=int, default=DEFAULT_N_TURNS)
    parser.add_argument("--temperature",      type=float, default=1.0,
                        help="Sampling temperature for subject model generation")
    parser.add_argument("--gen-workers",      type=int, default=DEFAULT_GEN_WORKERS,
                        help="Parallel threads for subject model generation")
    parser.add_argument("--judge-workers",    type=int, default=DEFAULT_JUDGE_WORKERS,
                        help="Parallel threads for OpenRouter judge calls")
    parser.add_argument("--prompts-file",     default=str(DEFAULT_PROMPTS_FILE),
                        help="JSONL file with {idx, prompt} rows (static prompt pool)")
    parser.add_argument("--seed",             type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir",       default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resume",           action="store_true")
    parser.add_argument("--wandb-run-id",     default=None)
    parser.add_argument("--metric-prefix",    default="")
    args = parser.parse_args()

    run(args)
