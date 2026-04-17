#!/usr/bin/env python3
"""
Aggregate results from a hyperparameter sweep over MLPConsistencyLoss.

Reads BRR and/or ASR CSV files from results/, extracts key metrics,
and prints a ranked summary table.

Usage:
    python scripts/aggregate_sweep.py
    python scripts/aggregate_sweep.py --pattern "results/sweep_*_brr.csv"
    python scripts/aggregate_sweep.py --mode jailbreak --pattern "results/sweep_*_asr.csv"
"""

import argparse
import csv
import glob
import os
import sys


def _safe_float(val) -> float:
    if val is None or val == "" or val == "N/A":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _fmt(val, decimals=4) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def read_brr_csv(path: str) -> dict:
    """Read a BRR results CSV and extract pre/post metrics."""
    run_name = os.path.basename(path).replace("_brr.csv", "")
    rows = []
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    except Exception as e:
        print(f"  Warning: could not read {path}: {e}", file=sys.stderr)
        return None

    if not rows:
        return None

    pre = next((r for r in rows if r.get("stage") == "pre_train"), None)
    post = next((r for r in rows if r.get("stage") == "post_train"), None)

    result = {"run_name": run_name, "csv_path": path}

    if pre:
        result["pre_brr"] = _safe_float(pre.get("brr"))
        result["pre_clean_acc"] = _safe_float(pre.get("clean_accuracy"))
        result["pre_mmlu"] = _safe_float(pre.get("mmlu_accuracy"))

    if post:
        result["post_brr"] = _safe_float(post.get("brr"))
        result["post_clean_acc"] = _safe_float(post.get("clean_accuracy"))
        result["post_wrapped_acc"] = _safe_float(post.get("wrapped_accuracy"))
        result["post_mmlu"] = _safe_float(post.get("mmlu_accuracy"))
        result["brr_ratio"] = _safe_float(post.get("brr_ratio"))

    if result.get("pre_brr") is not None and result.get("post_brr") is not None:
        pre_brr = result["pre_brr"]
        post_brr = result["post_brr"]
        if pre_brr > 0:
            result["brr_reduction_pct"] = round((1 - post_brr / pre_brr) * 100, 1)
        else:
            result["brr_reduction_pct"] = 0.0

    return result


def read_jailbreak_csv(path: str) -> dict:
    """Read a jailbreak eval CSV and extract pre/post metrics."""
    run_name = os.path.basename(path).replace("_jailbreak.csv", "")
    rows = []
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    except Exception as e:
        print(f"  Warning: could not read {path}: {e}", file=sys.stderr)
        return None

    if not rows:
        return None

    pre = next((r for r in rows if r.get("stage") == "pre_train"), None)
    post = next((r for r in rows if r.get("stage") == "post_train"), None)

    result = {"run_name": run_name, "csv_path": path}

    if pre:
        result["pre_avg_asr"] = _safe_float(pre.get("avg_asr"))
        result["pre_avg_overrefusal"] = _safe_float(pre.get("avg_overrefusal"))
        result["pre_f1"] = _safe_float(pre.get("f1"))
        result["pre_mmlu"] = _safe_float(pre.get("mmlu_accuracy"))

    if post:
        result["post_avg_asr"] = _safe_float(post.get("avg_asr"))
        result["post_avg_overrefusal"] = _safe_float(post.get("avg_overrefusal"))
        result["post_f1"] = _safe_float(post.get("f1"))
        result["post_mmlu"] = _safe_float(post.get("mmlu_accuracy"))
        result["asr_ratio"] = _safe_float(post.get("asr_ratio"))
        result["asr_reduction"] = _safe_float(post.get("asr_reduction"))

    if result.get("pre_avg_asr") is not None and result.get("post_avg_asr") is not None:
        pre_asr = result["pre_avg_asr"]
        post_asr = result["post_avg_asr"]
        if pre_asr > 0:
            result["asr_reduction_pct"] = round((1 - post_asr / pre_asr) * 100, 1)

    return result


def print_brr_table(results: list):
    """Print a formatted BRR summary table ranked by BRR reduction."""
    results.sort(key=lambda r: (-(r.get("brr_reduction_pct") or -999), r.get("post_brr") or 999))

    header = (
        f"{'#':>3s}  {'Run Name':<48s}  {'Pre BRR':>8s}  {'Post BRR':>9s}  "
        f"{'Red%':>6s}  {'Ratio':>6s}  {'Pre MMLU':>9s}  {'Post MMLU':>9s}"
    )
    print("\n" + "=" * len(header))
    print("BRR SWEEP RESULTS (ranked by BRR reduction)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for i, r in enumerate(results, 1):
        print(
            f"{i:>3d}  {r['run_name']:<48s}  {_fmt(r.get('pre_brr')):>8s}  "
            f"{_fmt(r.get('post_brr')):>9s}  {_fmt(r.get('brr_reduction_pct'), 1):>6s}  "
            f"{_fmt(r.get('brr_ratio')):>6s}  {_fmt(r.get('pre_mmlu')):>9s}  "
            f"{_fmt(r.get('post_mmlu')):>9s}"
        )

    print("-" * len(header) + "\n")


def print_jailbreak_table(results: list):
    """Print a formatted jailbreak summary table ranked by F1."""
    results.sort(key=lambda r: (-(r.get("post_f1") or -999)))

    header = (
        f"{'#':>3s}  {'Run Name':<48s}  {'Pre ASR':>8s}  {'Post ASR':>9s}  "
        f"{'Red%':>6s}  {'Pre OR%':>7s}  {'Post OR%':>8s}  {'F1':>6s}  {'MMLU':>6s}"
    )
    print("\n" + "=" * len(header))
    print("JAILBREAK SWEEP RESULTS (ranked by F1)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for i, r in enumerate(results, 1):
        print(
            f"{i:>3d}  {r['run_name']:<48s}  "
            f"{_fmt(r.get('pre_avg_asr')):>8s}  {_fmt(r.get('post_avg_asr')):>9s}  "
            f"{_fmt(r.get('asr_reduction_pct'), 1):>6s}  "
            f"{_fmt(r.get('pre_avg_overrefusal')):>7s}  {_fmt(r.get('post_avg_overrefusal')):>8s}  "
            f"{_fmt(r.get('post_f1')):>6s}  {_fmt(r.get('post_mmlu')):>6s}"
        )

    print("-" * len(header) + "\n")


def save_csv(results: list, output_path: str, mode: str):
    """Save aggregated results as CSV."""
    if not results:
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = list(results[0].keys())
    fieldnames = ["rank"] + [f for f in fieldnames if f != "rank"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for i, r in enumerate(results, 1):
            writer.writerow({"rank": i, **r})

    print(f"Aggregated results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate sweep results")
    parser.add_argument("--pattern", default=None,
                        help="Glob pattern for result CSVs")
    parser.add_argument("--mode", choices=["sycophancy", "jailbreak"],
                        default="sycophancy", help="Which metric to aggregate")
    parser.add_argument("--output", default=None,
                        help="Path to save aggregated CSV")
    args = parser.parse_args()

    if args.pattern is None:
        args.pattern = (
            "results/sweep_*_brr.csv" if args.mode == "sycophancy"
            else "results/sweep_*_jailbreak.csv"
        )
    if args.output is None:
        args.output = f"results/sweep_aggregate_{args.mode}.csv"

    csv_files = sorted(glob.glob(args.pattern))
    if not csv_files:
        print(f"No files found matching: {args.pattern}")
        sys.exit(1)

    print(f"Found {len(csv_files)} result files")

    results = []
    for path in csv_files:
        r = read_brr_csv(path) if args.mode == "sycophancy" else read_jailbreak_csv(path)
        if r is not None:
            results.append(r)

    if not results:
        print("No valid results extracted.")
        sys.exit(1)

    print(f"Parsed {len(results)} results successfully.")

    if args.mode == "sycophancy":
        print_brr_table(results)
    else:
        print_jailbreak_table(results)

    save_csv(results, args.output, args.mode)


if __name__ == "__main__":
    main()
