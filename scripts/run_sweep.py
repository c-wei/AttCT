#!/usr/bin/env python3
"""
Hyperparameter sweep runner for MLPConsistencyLoss.

Runs each YAML config in configs/sweep/ sequentially via subprocess,
logs timing, and saves a summary CSV at the end.

Usage:
    python scripts/run_sweep.py --mode sycophancy
    python scripts/run_sweep.py --mode sycophancy --dry-run
    python scripts/run_sweep.py --mode jailbreak --pattern "configs/sweep/sweep_metric_*.yaml"
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
import time
from datetime import datetime


def build_command(config_path: str, args) -> list:
    """Build the run.py command for a single sweep config."""
    cmd = [
        sys.executable, "run.py",
        "--config", config_path,
        "--data-source", args.data_source,
        "--data-mode", args.mode,
    ]

    if args.mode == "sycophancy" and args.brr_eval_path:
        cmd += ["--brr-eval-path", args.brr_eval_path]
    elif args.mode == "jailbreak":
        cmd += ["--jailbreak-eval"]
        if args.asr_eval_source:
            cmd += ["--asr-eval-source", args.asr_eval_source]

    if args.mmlu_max_samples is not None:
        cmd += ["--mmlu-max-samples", str(args.mmlu_max_samples)]

    return cmd


def extract_run_name(config_path: str) -> str:
    """Extract a short run name from the config filename."""
    return os.path.splitext(os.path.basename(config_path))[0]


def run_sweep(args):
    """Run all configs matching the glob pattern sequentially."""
    configs = sorted(glob.glob(args.pattern))
    if not configs:
        print(f"No configs found matching pattern: {args.pattern}")
        sys.exit(1)

    print(f"Found {len(configs)} configs to run:")
    for cfg in configs:
        print(f"  {cfg}")
    print()

    results = []
    total_start = time.time()

    for i, config_path in enumerate(configs, 1):
        run_name = extract_run_name(config_path)
        cmd = build_command(config_path, args)

        print("=" * 70)
        print(f"[{i}/{len(configs)}] {run_name}")
        print(f"  Config: {config_path}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        if args.dry_run:
            results.append({
                "run_name": run_name,
                "config": config_path,
                "status": "dry-run",
                "duration_s": 0,
                "return_code": 0,
            })
            continue

        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                timeout=args.timeout,
            )
            return_code = proc.returncode
            status = "success" if return_code == 0 else f"failed (rc={return_code})"
        except subprocess.TimeoutExpired:
            return_code = -1
            status = "timeout"
        except Exception as e:
            return_code = -2
            status = f"error: {e}"

        duration = time.time() - start
        print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Duration: {duration:.1f}s ({duration / 60:.1f}m)")
        print(f"  Status:   {status}")
        print()

        results.append({
            "run_name": run_name,
            "config": config_path,
            "status": status,
            "duration_s": round(duration, 1),
            "return_code": return_code,
        })

    total_duration = time.time() - total_start

    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    print(f"Total configs: {len(configs)}")
    print(f"Total time:    {total_duration:.1f}s ({total_duration / 60:.1f}m)")
    print()

    succeeded = sum(1 for r in results if r["return_code"] == 0)
    failed = len(results) - succeeded
    print(f"Succeeded: {succeeded}  |  Failed: {failed}")
    print()

    for r in results:
        marker = "OK" if r["return_code"] == 0 else "FAIL"
        print(f"  [{marker}] {r['run_name']:50s} {r['duration_s']:>8.1f}s  {r['status']}")

    summary_path = args.summary_csv
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_name", "config", "status", "duration_s", "return_code"],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSummary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MLPConsistencyLoss hyperparameter sweep runner"
    )
    parser.add_argument("--pattern", default="configs/sweep/*.yaml",
                        help="Glob pattern for config files")
    parser.add_argument("--mode", choices=["sycophancy", "jailbreak"],
                        default="sycophancy", help="Evaluation mode")
    parser.add_argument("--data-source",
                        default="datasets/sycophancy_bct/control_cot_train.jsonl",
                        help="Training data source path")
    parser.add_argument("--brr-eval-path",
                        default="datasets/sycophancy_bct/control_cot_eval.jsonl",
                        help="Path to BRR eval JSONL (sycophancy mode)")
    parser.add_argument("--asr-eval-source", default=None,
                        help="Path to harmful prompts file (jailbreak mode)")
    parser.add_argument("--mmlu-max-samples", type=int, default=None,
                        help="Override MMLU sample count for all runs")
    parser.add_argument("--summary-csv", default="results/sweep_summary.csv",
                        help="Path to write the sweep summary CSV")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running them")
    parser.add_argument("--timeout", type=int, default=7200,
                        help="Timeout per run in seconds (default: 7200 = 2h)")
    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
