"""
Visualize BRR experiment results for MLP-CT paper.

Auto-detects all experiment_mlp_*_brr.csv files in the results directory.

Usage:
    python scripts/visualize_brr.py --results-dir ~/Desktop/mlp_ct_results --output ~/Desktop/mlp_ct_figures
"""

import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False

COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
MARKERS = ['o', 's', 'D', '^', 'v']

MODEL_LABELS = {
    'gemma2b': 'Gemma-2-2B-IT',
    '3b': 'Llama-3.2-3B',
    '8b': 'Llama-3.1-8B',
    'mistral7b': 'Mistral-7B',
    'qwen7b': 'Qwen-2.5-7B',
}


def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in row:
                if k not in ('timestamp', 'stage'):
                    try:
                        row[k] = float(row[k])
                    except (ValueError, TypeError):
                        pass
            rows.append(row)
    return rows


def get_label(filename):
    """Extract model label from filename like experiment_mlp_3b_brr.csv."""
    base = os.path.basename(filename).replace('experiment_mlp_', '').replace('_brr.csv', '')
    return MODEL_LABELS.get(base, base)


def get_steps_and_metric(rows, metric):
    steps, vals = [], []
    for r in rows:
        val = r.get(metric)
        if isinstance(val, (int, float)):
            steps.append(int(r['step']))
            vals.append(val)
    return steps, vals


def load_all(results_dir):
    """Load all BRR CSV files, return dict of {label: rows}."""
    files = sorted(glob.glob(os.path.join(results_dir, 'experiment_mlp_*_brr.csv')))
    all_data = {}
    for f in files:
        label = get_label(f)
        all_data[label] = load_csv(f)
    return all_data


def plot_brr_over_training(all_data, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (label, rows) in enumerate(all_data.items()):
        steps, brr = get_steps_and_metric(rows, 'brr')
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        ax.plot(steps, brr, f'{marker}-', color=color, linewidth=2.5, markersize=8, label=label)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('BRR (Biased Reasoning Rate)')
    ax.set_title('MLP-CT Reduces Biased Reasoning Across Model Families')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'brr_over_training.png'), dpi=150)
    plt.close()
    print('  Saved: brr_over_training.png')


def plot_accuracy_over_training(all_data, output_dir):
    n = len(all_data)
    cols = min(n, 2)
    rows_n = (n + 1) // 2
    fig, axes = plt.subplots(rows_n, cols, figsize=(7 * cols, 5 * rows_n), squeeze=False)

    for i, (label, data) in enumerate(all_data.items()):
        ax = axes[i // cols][i % cols]
        steps, clean = get_steps_and_metric(data, 'clean_accuracy')
        _, wrapped = get_steps_and_metric(data, 'wrapped_accuracy')
        ax.plot(steps, clean, 'o-', color='#4CAF50', linewidth=2, markersize=7, label='Clean accuracy')
        ax.plot(steps, wrapped, 's-', color='#FF9800', linewidth=2, markersize=7, label='Wrapped accuracy')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Accuracy')
        ax.set_title(label)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 0.9)

    # Hide unused axes
    for j in range(i + 1, rows_n * cols):
        axes[j // cols][j % cols].set_visible(False)

    plt.suptitle('MLP-CT: Clean vs Wrapped Accuracy Over Training', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_over_training.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: accuracy_over_training.png')


def plot_brr_components(all_data, output_dir):
    n = len(all_data)
    cols = min(n, 2)
    rows_n = (n + 1) // 2
    fig, axes = plt.subplots(rows_n, cols, figsize=(7 * cols, 5 * rows_n), squeeze=False)

    for i, (label, data) in enumerate(all_data.items()):
        ax = axes[i // cols][i % cols]
        steps, baseline = get_steps_and_metric(data, 'unbiased_baseline')
        _, biased = get_steps_and_metric(data, 'biased_rate')
        ax.fill_between(steps, baseline, biased, alpha=0.3, color='#F44336', label='BRR (gap)')
        ax.plot(steps, baseline, 'o-', color='#607D8B', linewidth=2, markersize=7, label='Unbiased baseline')
        ax.plot(steps, biased, 's-', color='#F44336', linewidth=2, markersize=7, label='Biased rate')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Rate of picking biased answer')
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.5)

    for j in range(i + 1, rows_n * cols):
        axes[j // cols][j % cols].set_visible(False)

    plt.suptitle('BRR = Biased Rate - Unbiased Baseline (shaded area)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'brr_components.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: brr_components.png')


def plot_scaling(all_data, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = []
    pre_brrs = []
    post_brrs = []
    ratios = []

    for label, data in all_data.items():
        pre_brr = data[0]['brr']
        post_brr = data[-1]['brr']
        ratio = float(data[-1].get('brr_ratio', post_brr / pre_brr if pre_brr > 0 else 0))
        labels.append(label)
        pre_brrs.append(pre_brr)
        post_brrs.append(post_brr)
        ratios.append(ratio)

    x = np.arange(len(labels))
    width = 0.35

    bars_pre = ax.bar(x - width / 2, pre_brrs, width, label='BRR Pre-training',
                      color='#FFCDD2', edgecolor='#F44336')
    bars_post = ax.bar(x + width / 2, post_brrs, width, label='BRR Post MLP-CT',
                       color='#C8E6C9', edgecolor='#4CAF50')

    for i, (bp, ratio) in enumerate(zip(bars_post, ratios)):
        reduction = (1 - ratio) * 100
        ax.text(bp.get_x() + bp.get_width() / 2., bp.get_height() + 0.01,
                f'{ratio:.2f}\n({reduction:.0f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    for bar in list(bars_pre) + list(bars_post):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.002,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('BRR (Biased Reasoning Rate)')
    ax.set_title('MLP-CT: BRR Reduction Across Models')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(pre_brrs) * 1.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scaling_brr.png'), dpi=150)
    plt.close()
    print('  Saved: scaling_brr.png')


def plot_pre_post_comparison(all_data, output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))

    metrics = ['BRR', 'Clean\nAccuracy', 'Wrapped\nAccuracy', 'MMLU']
    keys = ['brr', 'clean_accuracy', 'wrapped_accuracy', 'mmlu_accuracy']

    n_models = len(all_data)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    total_bars = n_models * 2
    width = 0.8 / total_bars

    for i, (label, data) in enumerate(all_data.items()):
        pre = data[0]
        post = data[-1]
        color_pre = COLORS[i % len(COLORS)]
        color_post = COLORS[i % len(COLORS)]

        pre_vals = [float(pre.get(k, 0)) if isinstance(pre.get(k), (int, float)) else 0 for k in keys]
        post_vals = [float(post.get(k, 0)) if isinstance(post.get(k), (int, float)) else 0 for k in keys]

        offset_pre = (i * 2 - total_bars / 2 + 0.5) * width
        offset_post = (i * 2 + 1 - total_bars / 2 + 0.5) * width

        ax.bar(x + offset_pre, pre_vals, width, label=f'{label} Pre',
               color=color_pre, alpha=0.4, edgecolor=color_pre)
        ax.bar(x + offset_post, post_vals, width, label=f'{label} Post',
               color=color_pre, alpha=0.9, edgecolor=color_pre)

    ax.set_ylabel('Score')
    ax.set_title('MLP-CT: Pre vs Post Training — All Models')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=8, ncol=n_models, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pre_post_comparison.png'), dpi=150)
    plt.close()
    print('  Saved: pre_post_comparison.png')


def print_results_table(all_data):
    print('\n' + '=' * 100)
    print('MLP CONSISTENCY TRAINING — ALL RESULTS')
    print('=' * 100)

    for label, data in all_data.items():
        print(f'\n--- {label} ---')
        print(f'{"Stage":<15} {"Step":>5} {"Clean":>8} {"Wrapped":>8} {"Unbias":>8} {"Biased":>8} {"BRR":>8} {"MMLU":>8}')
        print('-' * 75)
        for r in data:
            print(f'{r["stage"]:<15} {int(r["step"]):>5} {r["clean_accuracy"]:>8.4f} {r["wrapped_accuracy"]:>8.4f} '
                  f'{r["unbiased_baseline"]:>8.4f} {r["biased_rate"]:>8.4f} {r["brr"]:>8.4f} '
                  f'{r["mmlu_accuracy"]:>8}')
        pre_brr = data[0]['brr']
        post_brr = data[-1]['brr']
        ratio = float(data[-1].get('brr_ratio', post_brr / pre_brr if pre_brr > 0 else 0))
        print(f'\n  BRR Ratio: {ratio:.4f} ({(1 - ratio) * 100:.1f}% reduction)')

    # Summary table
    print('\n' + '=' * 100)
    print('SUMMARY')
    print('=' * 100)
    print(f'{"Model":<20} {"Family":<10} {"BRR Pre":>8} {"BRR Post":>9} {"BRR Ratio":>10} {"Reduction":>10} {"Clean Acc":>10} {"MMLU":>8}')
    print('-' * 90)

    families = {'Gemma': 'Google', 'Llama': 'Meta', 'Mistral': 'Mistral', 'Qwen': 'Alibaba'}
    for label, data in all_data.items():
        family = 'Unknown'
        for k, v in families.items():
            if k.lower() in label.lower():
                family = v
                break
        pre_brr = data[0]['brr']
        post_brr = data[-1]['brr']
        ratio = float(data[-1].get('brr_ratio', post_brr / pre_brr if pre_brr > 0 else 0))
        clean_delta = data[-1]['clean_accuracy'] - data[0]['clean_accuracy']
        mmlu_post = data[-1]['mmlu_accuracy']
        print(f'{label:<20} {family:<10} {pre_brr:>8.3f} {post_brr:>9.3f} {ratio:>10.3f} '
              f'{(1 - ratio) * 100:>9.1f}% {clean_delta:>+10.3f} {mmlu_post:>8}')

    print('\n' + '=' * 100)


def main():
    parser = argparse.ArgumentParser(description='Visualize MLP-CT BRR results')
    parser.add_argument('--results-dir', default=os.path.expanduser('~/Desktop/mlp_ct_results'))
    parser.add_argument('--output', default=os.path.expanduser('~/Desktop/mlp_ct_figures'))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    all_data = load_all(args.results_dir)

    if not all_data:
        print(f'No experiment_mlp_*_brr.csv files found in {args.results_dir}')
        return

    print(f'Found {len(all_data)} models: {", ".join(all_data.keys())}')

    print_results_table(all_data)

    print('\nGenerating figures...')
    plot_brr_over_training(all_data, args.output)
    plot_accuracy_over_training(all_data, args.output)
    plot_brr_components(all_data, args.output)
    plot_scaling(all_data, args.output)
    plot_pre_post_comparison(all_data, args.output)

    print(f'\nAll figures saved to {args.output}/')


if __name__ == '__main__':
    main()
