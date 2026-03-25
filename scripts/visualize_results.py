"""
Visualize MLP-CT experiment results.

Reads output.log and syco_results.csv to produce:
1. Training loss curve
2. Sycophancy & accuracy across checkpoints
3. Per-layer loss heatmap (initial vs final)
4. Comprehensive results table

Usage:
    python scripts/visualize_results.py --log output.log --csv results.csv --output figures/
"""

import argparse
import re
import os
import json
import csv
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['figure.figsize'] = (10, 6)
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed — will generate text tables only.")
    print("Install with: pip install matplotlib")


def parse_output_log(log_path):
    """Parse the output.log file for all metrics."""
    results = {
        "loss_curve": [],           # (step, loss)
        "checkpoints": [],          # list of dicts with all checkpoint metrics
        "per_layer_initial": [],    # initial per-layer losses
        "per_layer_final": [],      # final per-layer losses
        "eval_per_layer": [],       # eval per-layer losses
        "pre_train": {},
        "post_train": {},
        "eval_mean_loss": None,
    }

    with open(log_path) as f:
        lines = f.readlines()

    current_checkpoint = {}
    seen_steps = set()

    for i, line in enumerate(lines):
        # Training loss (deduplicate — log prints each step multiple times)
        m = re.match(r'\[epoch \d+ \| step (\d+)\] loss: ([\d.]+)', line)
        if m:
            step = int(m.group(1))
            loss = float(m.group(2))
            if step not in seen_steps:
                results["loss_curve"].append((step, loss))
                seen_steps.add(step)

        # Checkpoint behavioral evals
        if "behavioral/clean_cot_accuracy:" in line:
            current_checkpoint["clean_cot_acc"] = float(line.split(":")[1].strip())
        if "behavioral/clean_noncot_accuracy:" in line:
            current_checkpoint["clean_noncot_acc"] = float(line.split(":")[1].strip())
        if "behavioral/wrapped_cot_accuracy:" in line:
            current_checkpoint["wrapped_cot_acc"] = float(line.split(":")[1].strip())
        if "behavioral/wrapped_noncot_accuracy:" in line:
            current_checkpoint["wrapped_noncot_acc"] = float(line.split(":")[1].strip())
        if "behavioral/clean_accuracy:" in line:
            current_checkpoint["clean_acc"] = float(line.split(":")[1].strip())
        if "behavioral/wrapped_accuracy:" in line:
            current_checkpoint["wrapped_acc"] = float(line.split(":")[1].strip())
        if "behavioral/sycophancy_rate:" in line:
            current_checkpoint["sycophancy_rate"] = float(line.split(":")[1].strip())
        if "intelligence/mmlu_accuracy:" in line:
            current_checkpoint["mmlu_acc"] = float(line.split(":")[1].strip())
        if "intelligence/gsm8k_accuracy:" in line:
            current_checkpoint["gsm8k_acc"] = float(line.split(":")[1].strip())
            # gsm8k is the last metric in a checkpoint block — save and reset
            if "clean_acc" in current_checkpoint:
                results["checkpoints"].append(current_checkpoint.copy())
                current_checkpoint = {}

        # Checkpoint step number
        m = re.match(r'\[Checkpoint\] Step (\d+)', line)
        if m and current_checkpoint is not None:
            current_checkpoint["step"] = int(m.group(1))

        # Per-layer loss change
        m = re.match(r'\s+Layer (\d+): ([\d.]+) .* ([\d.]+)', line)
        if m:
            layer = int(m.group(1))
            initial = float(m.group(2))
            final = float(m.group(3))
            results["per_layer_initial"].append(initial)
            results["per_layer_final"].append(final)

        # Eval per-layer
        m = re.match(r'\s+layer (\d+): ([\d.]+)', line)
        if m:
            results["eval_per_layer"].append(float(m.group(2)))

        # Pre/post train sycophancy
        if "prefix:" in line:
            prefix = line.split(":")[1].strip()
        if "mmlu_accuracy:" in line and "intelligence" not in line:
            val = float(line.split(":")[1].strip())
            if "pre_train" in lines[i-2] if i >= 2 else "":
                results["pre_train"]["mmlu"] = val
            elif "post_train" in lines[i-2] if i >= 2 else "":
                results["post_train"]["mmlu"] = val
        if "not_sycophantic:" in line:
            val = float(line.split(":")[1].strip())
            if any("pre_train" in lines[j] for j in range(max(0, i-3), i)):
                results["pre_train"]["not_sycophantic"] = val
            else:
                results["post_train"]["not_sycophantic"] = val
        if "f1_score:" in line and "behavioral" not in line:
            val = float(line.split(":")[1].strip())
            if any("pre_train" in lines[j] for j in range(max(0, i-3), i)):
                results["pre_train"]["f1"] = val
            else:
                results["post_train"]["f1"] = val

        # Eval mean loss
        if "mean_loss:" in line and "eval" not in line.lower():
            m = re.search(r'mean_loss: ([\d.]+)', line)
            if m:
                results["eval_mean_loss"] = float(m.group(1))

    # Assign step numbers to checkpoints
    for idx, cp in enumerate(results["checkpoints"]):
        if "step" not in cp:
            # Approximate based on position
            total_steps = results["loss_curve"][-1][0] if results["loss_curve"] else 500
            cp["step"] = (idx + 1) * total_steps // 3

    return results


def parse_csv(csv_path):
    """Parse the syco_results.csv file."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def print_full_results_table(results):
    """Print comprehensive results table."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS — MLP Consistency Training (Llama-3.2-3B)")
    print("=" * 80)

    # Pre vs Post
    pre = results["pre_train"]
    post = results["post_train"]
    print("\n--- Pre vs Post Training (SycophancyEvaluator, 500 questions) ---")
    print(f"{'Metric':<25} {'Pre-train':>12} {'Post-train':>12} {'Delta':>12}")
    print("-" * 65)
    if "not_sycophantic" in pre:
        delta = post.get("not_sycophantic", 0) - pre["not_sycophantic"]
        print(f"{'Not sycophantic':<25} {pre['not_sycophantic']:>12.3f} {post.get('not_sycophantic', 0):>12.3f} {delta:>+12.3f}")
    if "mmlu" in pre:
        delta = post.get("mmlu", 0) - pre["mmlu"]
        print(f"{'MMLU accuracy':<25} {pre['mmlu']:>12.3f} {post.get('mmlu', 0):>12.3f} {delta:>+12.3f}")
    if "f1" in pre:
        delta = post.get("f1", 0) - pre["f1"]
        print(f"{'F1 score':<25} {pre['f1']:>12.3f} {post.get('f1', 0):>12.3f} {delta:>+12.3f}")

    # Checkpoint evals
    if results["checkpoints"]:
        print("\n--- Behavioral Eval at Checkpoints (held-out 1000 questions) ---")
        header = f"{'Step':>6} {'Clean CoT':>10} {'Clean NC':>10} {'Wrap CoT':>10} {'Wrap NC':>10} {'Clean':>8} {'Wrapped':>8} {'Syco Rate':>10} {'MMLU':>6}"
        print(header)
        print("-" * len(header))
        for cp in results["checkpoints"]:
            print(f"{cp.get('step', '?'):>6} "
                  f"{cp.get('clean_cot_acc', 0):>10.4f} "
                  f"{cp.get('clean_noncot_acc', 0):>10.4f} "
                  f"{cp.get('wrapped_cot_acc', 0):>10.4f} "
                  f"{cp.get('wrapped_noncot_acc', 0):>10.4f} "
                  f"{cp.get('clean_acc', 0):>8.4f} "
                  f"{cp.get('wrapped_acc', 0):>8.4f} "
                  f"{cp.get('sycophancy_rate', 0):>10.4f} "
                  f"{cp.get('mmlu_acc', 0):>6.3f}")

        print(f"\nSycophancy rate = 1 - wrapped_accuracy")
        print(f"  wrapped_accuracy = avg(wrapped_cot_accuracy, wrapped_noncot_accuracy)")
        print(f"  clean_accuracy   = avg(clean_cot_accuracy, clean_noncot_accuracy)")

    # Per-layer losses
    if results["per_layer_initial"]:
        print("\n--- Per-Layer MLP Consistency Loss (cosine distance, initial → final) ---")
        print(f"{'Layer':>6} {'Initial':>10} {'Final':>10} {'Delta':>10} {'Eval':>10}")
        print("-" * 50)
        for i in range(len(results["per_layer_initial"])):
            initial = results["per_layer_initial"][i]
            final = results["per_layer_final"][i]
            delta = final - initial
            eval_val = results["eval_per_layer"][i] if i < len(results["eval_per_layer"]) else 0
            print(f"{i:>6} {initial:>10.4f} {final:>10.4f} {delta:>+10.4f} {eval_val:>10.4f}")
        total_i = sum(results["per_layer_initial"])
        total_f = sum(results["per_layer_final"])
        total_e = sum(results["eval_per_layer"]) if results["eval_per_layer"] else 0
        print(f"{'Total':>6} {total_i:>10.4f} {total_f:>10.4f} {total_f - total_i:>+10.4f} {total_e:>10.4f}")

    print("\n" + "=" * 80)


def plot_loss_curve(results, output_dir):
    """Plot training loss over steps."""
    steps = [s for s, _ in results["loss_curve"]]
    losses = [l for _, l in results["loss_curve"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, alpha=0.3, color="blue", linewidth=0.8)

    # Moving average
    window = 10
    if len(losses) > window:
        ma = [sum(losses[max(0, i-window):i+1]) / min(i+1, window) for i in range(len(losses))]
        ax.plot(steps, ma, color="blue", linewidth=2, label=f"Moving avg (window={window})")

    # Mark checkpoints
    for cp in results["checkpoints"]:
        if "step" in cp:
            ax.axvline(x=cp["step"], color="red", linestyle="--", alpha=0.5)
            ax.text(cp["step"], max(losses) * 0.95, f"eval", ha="center", fontsize=9, color="red")

    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("MLP Consistency Loss (cosine distance)")
    ax.set_title("MLP-CT Training Loss — Llama-3.2-3B")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/loss_curve.png")


def plot_checkpoint_evals(results, output_dir):
    """Plot sycophancy rate, clean/wrapped accuracy across checkpoints."""
    if not results["checkpoints"]:
        return

    steps = [cp["step"] for cp in results["checkpoints"]]
    clean_acc = [cp["clean_acc"] for cp in results["checkpoints"]]
    wrapped_acc = [cp["wrapped_acc"] for cp in results["checkpoints"]]
    syco_rate = [cp["sycophancy_rate"] for cp in results["checkpoints"]]
    mmlu = [cp["mmlu_acc"] for cp in results["checkpoints"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: accuracy
    ax1.plot(steps, clean_acc, "g-o", label="Clean accuracy", linewidth=2)
    ax1.plot(steps, wrapped_acc, "b-o", label="Wrapped accuracy", linewidth=2)
    ax1.plot(steps, mmlu, "m-s", label="MMLU accuracy", linewidth=2)
    ax1.set_xlabel("Optimizer Step")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy Across Checkpoints")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.3, 1.0)

    # Right: sycophancy rate
    ax2.plot(steps, syco_rate, "r-o", linewidth=2, markersize=8)
    ax2.set_xlabel("Optimizer Step")
    ax2.set_ylabel("Sycophancy Rate")
    ax2.set_title("Sycophancy Rate Across Checkpoints")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.5)

    # Add pre-train baseline if available
    if "not_sycophantic" in results["pre_train"]:
        baseline_syco = 1 - results["pre_train"]["not_sycophantic"]
        ax2.axhline(y=baseline_syco, color="gray", linestyle="--", label=f"Pre-train baseline ({baseline_syco:.3f})")
        ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "checkpoint_evals.png"), dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/checkpoint_evals.png")


def plot_per_layer(results, output_dir):
    """Plot per-layer loss profile."""
    if not results["per_layer_initial"]:
        return

    n_layers = len(results["per_layer_initial"])
    layers = list(range(n_layers))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([l - 0.2 for l in layers], results["per_layer_initial"], 0.35,
           label="Initial", color="salmon", alpha=0.8)
    ax.bar([l + 0.2 for l in layers], results["per_layer_final"], 0.35,
           label="Final", color="steelblue", alpha=0.8)

    if results["eval_per_layer"]:
        ax.plot(layers, results["eval_per_layer"], "ko-", label="Eval", linewidth=1.5, markersize=4)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Cosine Distance")
    ax.set_title("Per-Layer MLP Consistency — Initial vs Final vs Eval")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(range(0, n_layers, 2))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_layer_loss.png"), dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/per_layer_loss.png")


def plot_pre_post_comparison(results, output_dir):
    """Bar chart comparing pre vs post training metrics."""
    pre = results["pre_train"]
    post = results["post_train"]

    if not pre or not post:
        return

    metrics = []
    pre_vals = []
    post_vals = []

    if "not_sycophantic" in pre:
        metrics.append("Not\nSycophantic")
        pre_vals.append(pre["not_sycophantic"])
        post_vals.append(post.get("not_sycophantic", 0))
    if "mmlu" in pre:
        metrics.append("MMLU")
        pre_vals.append(pre["mmlu"])
        post_vals.append(post.get("mmlu", 0))
    if "f1" in pre:
        metrics.append("F1 Score")
        pre_vals.append(pre["f1"])
        post_vals.append(post.get("f1", 0))

    x = range(len(metrics))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar([i - 0.2 for i in x], pre_vals, 0.35, label="Pre-training", color="lightcoral")
    bars2 = ax.bar([i + 0.2 for i in x], post_vals, 0.35, label="Post MLP-CT", color="steelblue")

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=10)

    ax.set_ylabel("Score")
    ax.set_title("MLP-CT Effect — Llama-3.2-3B (500 steps)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pre_post_comparison.png"), dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/pre_post_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize MLP-CT experiment results")
    parser.add_argument("--log", required=True, help="Path to output.log from W&B run")
    parser.add_argument("--csv", default=None, help="Path to syco_results.csv (optional)")
    parser.add_argument("--output", default="figures", help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    results = parse_output_log(args.log)

    # Always print the text table
    print_full_results_table(results)

    # Generate plots if matplotlib available
    if HAS_MPL:
        print("\nGenerating figures...")
        plot_loss_curve(results, args.output)
        plot_checkpoint_evals(results, args.output)
        plot_per_layer(results, args.output)
        plot_pre_post_comparison(results, args.output)
        print(f"\nAll figures saved to {args.output}/")
    else:
        print("\nSkipping figures (install matplotlib: pip install matplotlib)")


if __name__ == "__main__":
    main()
