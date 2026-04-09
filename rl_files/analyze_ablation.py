"""Analyze ablation experiment results:
  1. Comparison bar chart + box plots (reward distributions)
  2. Statistical significance tests (t-test, Wilcoxon)
  3. Summary table
"""
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REWARDS_DIR = os.path.join(PROJECT_ROOT, 'rewards')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

TAGS = ['mlp_baseline', 'mlp_fees', 'transformer_baseline', 'transformer_fees']
ENVS = ['noise', 'flow', 'strategic']

TAG_LABELS = {
    'mlp_baseline': 'MLP (no fee)',
    'mlp_fees': 'MLP (+ fees)',
    'transformer_baseline': 'Transformer (no fee)',
    'transformer_fees': 'Transformer (+ fees)',
}
TAG_COLORS = {
    'mlp_baseline': '#4C72B0',
    'mlp_fees': '#55A868',
    'transformer_baseline': '#C44E52',
    'transformer_fees': '#8172B3',
}


def load_results():
    """Return nested dict: results[tag][env] = rewards array."""
    results = {tag: {} for tag in TAGS}
    files = sorted(glob.glob(os.path.join(REWARDS_DIR, '*bsize_3200*.npz')))
    for f in files:
        name = os.path.basename(f)
        tag = next((t for t in TAGS if f'_{t}_' in name), None)
        env = next((e for e in ENVS if name.startswith(f'{e}_')), None)
        if tag is None or env is None:
            continue
        data = np.load(f)
        rewards = data['rewards'] if 'rewards' in data.files else data[data.files[0]]
        # If duplicate (e.g., old eval_10000 + new eval_1000), prefer the one with more episodes
        if env not in results[tag] or len(rewards) > len(results[tag][env]):
            results[tag][env] = rewards
    return results


def print_table(results):
    print('\n' + '=' * 95)
    print(f'{"Config":<25} {"Env":<12} {"N":<6} {"Mean":<10} {"Std":<10} {"Median":<10} {"SE":<10}')
    print('=' * 95)
    for tag in TAGS:
        for env in ENVS:
            if env in results[tag]:
                r = results[tag][env]
                se = r.std() / np.sqrt(len(r))
                print(f'{tag:<25} {env:<12} {len(r):<6} {r.mean():<10.4f} {r.std():<10.4f} {np.median(r):<10.4f} {se:<10.4f}')
        print('-' * 95)


def plot_comparison(results, save_path):
    """Bar chart: mean reward with error bars (95% CI), grouped by env."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(ENVS))
    width = 0.2
    for i, tag in enumerate(TAGS):
        means, errs = [], []
        for env in ENVS:
            if env in results[tag]:
                r = results[tag][env]
                means.append(r.mean())
                errs.append(1.96 * r.std() / np.sqrt(len(r)))
            else:
                means.append(0)
                errs.append(0)
        ax.bar(x + (i - 1.5) * width, means, width, yerr=errs,
               label=TAG_LABELS[tag], color=TAG_COLORS[tag], capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(ENVS)
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Ablation: Mean Reward by Config and Environment (error bars = 95% CI)')
    ax.legend(loc='best')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')


def plot_boxplots(results, save_path):
    """Box plots of reward distributions, one subplot per env."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, env in zip(axes, ENVS):
        data, labels, colors = [], [], []
        for tag in TAGS:
            if env in results[tag]:
                data.append(results[tag][env])
                labels.append(TAG_LABELS[tag].replace(' ', '\n'))
                colors.append(TAG_COLORS[tag])
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='yellow',
                                       markeredgecolor='black', markersize=6))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(f'{env}')
        ax.set_ylabel('Reward')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', labelsize=8)
    fig.suptitle('Reward Distributions by Environment', fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')


def significance_tests(results):
    """Run Welch t-test + Mann-Whitney U for key pairwise comparisons."""
    print('\n' + '=' * 95)
    print('STATISTICAL SIGNIFICANCE TESTS (per environment)')
    print('=' * 95)
    print(f'{"Comparison":<45} {"Env":<12} {"t-stat":<10} {"p (Welch)":<12} {"p (MWU)":<12}')
    print('-' * 95)

    comparisons = [
        ('mlp_baseline', 'transformer_baseline', 'MLP vs Transformer (no fee)'),
        ('mlp_fees', 'transformer_fees', 'MLP vs Transformer (+ fees)'),
        ('mlp_baseline', 'mlp_fees', 'MLP: no fee vs + fees'),
        ('transformer_baseline', 'transformer_fees', 'Transformer: no fee vs + fees'),
    ]

    for tag_a, tag_b, label in comparisons:
        for env in ENVS:
            if env not in results[tag_a] or env not in results[tag_b]:
                continue
            a = results[tag_a][env]
            b = results[tag_b][env]
            t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
            _, mwu_p = stats.mannwhitneyu(a, b, alternative='two-sided')
            sig = '***' if t_p < 0.001 else '**' if t_p < 0.01 else '*' if t_p < 0.05 else ''
            print(f'{label:<45} {env:<12} {t_stat:<10.3f} {t_p:<12.4f} {mwu_p:<12.4f} {sig}')
        print()


def main():
    results = load_results()

    # Sanity: check all 12 cells present
    missing = [(t, e) for t in TAGS for e in ENVS if e not in results[t]]
    if missing:
        print(f'WARNING: missing configs: {missing}')

    print_table(results)

    plot_comparison(results, os.path.join(FIGURES_DIR, 'ablation_mean_rewards.png'))
    plot_boxplots(results, os.path.join(FIGURES_DIR, 'ablation_boxplots.png'))

    significance_tests(results)

    print('\nDone.')


if __name__ == '__main__':
    main()
