"""
Visualize BRR experiment results for MLP-CT paper.

Reads experiment_mlp_*_brr.csv files and produces publication-ready figures.

Usage:
    python scripts/visualize_brr.py --results-dir ~/Desktop/mlp_ct_results --output ~/Desktop/mlp_ct_figures
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False


def load_csv(path):
    """Load a BRR CSV file into a list of dicts."""
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


def get_steps_and_metric(rows, metric):
    """Extract steps and a metric across all stages."""
    steps = []
    vals = []
    for r in rows:
        step = int(r['step'])
        val = r[metric]
        if isinstance(val, (int, float)):
            steps.append(step)
            vals.append(val)
    return steps, vals


def plot_brr_over_training(data_3b, data_8b, output_dir):
    """Plot BRR decreasing over training steps for both models."""
    fig, ax = plt.subplots(figsize=(9, 5))

    steps_3b, brr_3b = get_steps_and_metric(data_3b, 'brr')
    steps_8b, brr_8b = get_steps_and_metric(data_8b, 'brr')

    ax.plot(steps_3b, brr_3b, 'o-', color='#2196F3', linewidth=2.5, markersize=8, label='Llama-3.2-3B')
    ax.plot(steps_8b, brr_8b, 's-', color='#F44336', linewidth=2.5, markersize=8, label='Llama-3.1-8B')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('BRR (Biased Reasoning Rate)')
    ax.set_title('MLP-CT Reduces Biased Reasoning Over Training')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Annotate BRR ratios
    brr_ratio_3b = data_3b[-1].get('brr_ratio', brr_3b[-1] / brr_3b[0] if brr_3b[0] > 0 else 0)
    brr_ratio_8b = data_8b[-1].get('brr_ratio', brr_8b[-1] / brr_8b[0] if brr_8b[0] > 0 else 0)
    if isinstance(brr_ratio_3b, str):
        brr_ratio_3b = float(brr_ratio_3b)
    if isinstance(brr_ratio_8b, str):
        brr_ratio_8b = float(brr_ratio_8b)

    ax.annotate(f'BRR ratio: {brr_ratio_3b:.2f}\n({(1-brr_ratio_3b)*100:.0f}% reduction)',
                xy=(steps_3b[-1], brr_3b[-1]), xytext=(steps_3b[-1]-80, brr_3b[-1]+0.03),
                fontsize=10, color='#2196F3')
    ax.annotate(f'BRR ratio: {brr_ratio_8b:.2f}\n({(1-brr_ratio_8b)*100:.0f}% reduction)',
                xy=(steps_8b[-1], brr_8b[-1]), xytext=(steps_8b[-1]-80, brr_8b[-1]+0.02),
                fontsize=10, color='#F44336')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'brr_over_training.png'), dpi=150)
    plt.close()
    print(f'  Saved: brr_over_training.png')


def plot_accuracy_over_training(data_3b, data_8b, output_dir):
    """Plot clean and wrapped accuracy over training for both models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 3B
    steps, clean = get_steps_and_metric(data_3b, 'clean_accuracy')
    _, wrapped = get_steps_and_metric(data_3b, 'wrapped_accuracy')
    ax1.plot(steps, clean, 'o-', color='#4CAF50', linewidth=2, markersize=7, label='Clean accuracy')
    ax1.plot(steps, wrapped, 's-', color='#FF9800', linewidth=2, markersize=7, label='Wrapped accuracy')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Llama-3.2-3B')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.4, 0.9)

    # 8B
    steps, clean = get_steps_and_metric(data_8b, 'clean_accuracy')
    _, wrapped = get_steps_and_metric(data_8b, 'wrapped_accuracy')
    ax2.plot(steps, clean, 'o-', color='#4CAF50', linewidth=2, markersize=7, label='Clean accuracy')
    ax2.plot(steps, wrapped, 's-', color='#FF9800', linewidth=2, markersize=7, label='Wrapped accuracy')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Llama-3.1-8B')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.4, 0.9)

    plt.suptitle('MLP-CT: Clean vs Wrapped Accuracy Over Training', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_over_training.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: accuracy_over_training.png')


def plot_brr_components(data_3b, data_8b, output_dir):
    """Plot unbiased baseline vs biased rate (the two BRR components)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title in [(ax1, data_3b, 'Llama-3.2-3B'), (ax2, data_8b, 'Llama-3.1-8B')]:
        steps, baseline = get_steps_and_metric(data, 'unbiased_baseline')
        _, biased = get_steps_and_metric(data, 'biased_rate')

        ax.fill_between(steps, baseline, biased, alpha=0.3, color='#F44336', label='BRR (gap = nudge effect)')
        ax.plot(steps, baseline, 'o-', color='#607D8B', linewidth=2, markersize=7, label='Unbiased baseline')
        ax.plot(steps, biased, 's-', color='#F44336', linewidth=2, markersize=7, label='Biased rate')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Rate of picking biased answer')
        ax.set_title(title)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.4)

    plt.suptitle('BRR = Biased Rate - Unbiased Baseline (shaded area)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'brr_components.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: brr_components.png')


def plot_pre_post_comparison(data_3b, data_8b, output_dir):
    """Bar chart comparing pre vs post for both models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    pre_3b = data_3b[0]
    post_3b = data_3b[-1]
    pre_8b = data_8b[0]
    post_8b = data_8b[-1]

    metrics = ['BRR', 'Clean\nAccuracy', 'Wrapped\nAccuracy', 'MMLU']
    keys = ['brr', 'clean_accuracy', 'wrapped_accuracy', 'mmlu_accuracy']

    x = np.arange(len(metrics))
    width = 0.18

    bars = []
    colors = ['#BBDEFB', '#2196F3', '#FFCDD2', '#F44336']
    labels = ['3B Pre', '3B Post', '8B Pre', '8B Post']
    datasets = [pre_3b, post_3b, pre_8b, post_8b]

    for i, (dataset, color, label) in enumerate(zip(datasets, colors, labels)):
        vals = [float(dataset[k]) if isinstance(dataset[k], (int, float, str)) and dataset[k] != 'N/A' else 0 for k in keys]
        bars.append(ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color, edgecolor='white'))

    # Add value labels on bars
    for bar_group in bars:
        for bar in bar_group:
            height = bar.get_height()
            if height > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Score')
    ax.set_title('MLP-CT: Pre vs Post Training Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(ncol=4, fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 0.95)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pre_post_comparison.png'), dpi=150)
    plt.close()
    print(f'  Saved: pre_post_comparison.png')


def plot_scaling(data_3b, data_8b, output_dir):
    """Show BRR ratio scales with model size."""
    fig, ax = plt.subplots(figsize=(7, 5))

    models = ['Llama-3.2\n3B', 'Llama-3.1\n8B']
    pre_brr = [data_3b[0]['brr'], data_8b[0]['brr']]
    post_brr = [data_3b[-1]['brr'], data_8b[-1]['brr']]

    brr_ratio_3b = float(data_3b[-1].get('brr_ratio', post_brr[0] / pre_brr[0]))
    brr_ratio_8b = float(data_8b[-1].get('brr_ratio', post_brr[1] / pre_brr[1]))
    ratios = [brr_ratio_3b, brr_ratio_8b]

    x = np.arange(len(models))
    width = 0.35

    bars_pre = ax.bar(x - width/2, pre_brr, width, label='BRR Pre-training', color='#FFCDD2', edgecolor='#F44336')
    bars_post = ax.bar(x + width/2, post_brr, width, label='BRR Post MLP-CT', color='#C8E6C9', edgecolor='#4CAF50')

    # Add BRR ratio labels
    for i, (bp, ratio) in enumerate(zip(bars_post, ratios)):
        ax.text(bp.get_x() + bp.get_width()/2., bp.get_height() + 0.008,
                f'ratio={ratio:.2f}\n({(1-ratio)*100:.0f}% reduction)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Value labels
    for bar in list(bars_pre) + list(bars_post):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('BRR (Biased Reasoning Rate)')
    ax.set_title('MLP-CT: Larger Models Benefit More')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 0.35)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scaling_brr.png'), dpi=150)
    plt.close()
    print(f'  Saved: scaling_brr.png')


def print_results_table(data_3b, data_8b):
    """Print comprehensive results table."""
    print('\n' + '=' * 90)
    print('MLP CONSISTENCY TRAINING — RESULTS')
    print('=' * 90)

    for name, data in [('Llama-3.2-3B-Instruct', data_3b), ('Llama-3.1-8B-Instruct', data_8b)]:
        print(f'\n--- {name} ---')
        print(f'{"Stage":<15} {"Step":>5} {"Clean":>8} {"Wrapped":>8} {"Unbias":>8} {"Biased":>8} {"BRR":>8} {"MMLU":>8}')
        print('-' * 75)
        for r in data:
            stage = r['stage']
            step = int(r['step'])
            print(f'{stage:<15} {step:>5} {r["clean_accuracy"]:>8.4f} {r["wrapped_accuracy"]:>8.4f} '
                  f'{r["unbiased_baseline"]:>8.4f} {r["biased_rate"]:>8.4f} {r["brr"]:>8.4f} '
                  f'{r["mmlu_accuracy"]:>8}')

        pre_brr = data[0]['brr']
        post_brr = data[-1]['brr']
        ratio = float(data[-1].get('brr_ratio', post_brr / pre_brr if pre_brr > 0 else 0))
        print(f'\n  BRR Ratio: {ratio:.4f} ({(1-ratio)*100:.1f}% reduction in biased reasoning)')

    print('\n' + '=' * 90)
    print('BRR = biased_rate - unbiased_baseline')
    print('BRR Ratio = BRR_post / BRR_pre (lower = better, BCT paper achieved 0.14)')
    print('=' * 90)


def main():
    parser = argparse.ArgumentParser(description='Visualize MLP-CT BRR results')
    parser.add_argument('--results-dir', default=os.path.expanduser('~/Desktop/mlp_ct_results'))
    parser.add_argument('--output', default=os.path.expanduser('~/Desktop/mlp_ct_figures'))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    csv_3b = os.path.join(args.results_dir, 'experiment_mlp_3b_brr.csv')
    csv_8b = os.path.join(args.results_dir, 'experiment_mlp_8b_brr.csv')

    data_3b = load_csv(csv_3b)
    data_8b = load_csv(csv_8b)

    print_results_table(data_3b, data_8b)

    print('\nGenerating figures...')
    plot_brr_over_training(data_3b, data_8b, args.output)
    plot_accuracy_over_training(data_3b, data_8b, args.output)
    plot_brr_components(data_3b, data_8b, args.output)
    plot_pre_post_comparison(data_3b, data_8b, args.output)
    plot_scaling(data_3b, data_8b, args.output)

    print(f'\nAll figures saved to {args.output}/')


if __name__ == '__main__':
    main()
