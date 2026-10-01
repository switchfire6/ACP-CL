"""Standalone figures for the locked rehearsal-state diagnostic."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


CELLS = tuple(f'{i:03b}' for i in range(8))


def save(fig, output, name):
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(output/f'{name}.{suffix}', dpi=180)
    plt.close(fig)


def plot(path, output):
    result = json.loads(Path(path).read_text(encoding='utf-8'))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for col, model in enumerate(('conditional', 'recurrent')):
        for row, branch in enumerate(('current', 'return')):
            ax = axes[row, col]
            for at, cell in enumerate(CELLS):
                item = result['panels'][model][branch][cell]['contrasts']['value_minus_uniform']['brier']
                color = '#7664a1' if cell[0] == '0' else '#087f8c'
                values = np.asarray(item['differences'])*100
                ax.scatter(at+np.linspace(-.13, .13, len(values)), values, color=color, alpha=.45, s=16)
                ax.errorbar(at, item['mean']*100,
                    yerr=np.asarray([[item['mean']-item['lower']], [item['upper']-item['mean']]])*100,
                    fmt='D', color=color, capsize=3, markersize=5)
            ax.axhline(0, color='#5f6b79', linestyle='--', linewidth=1)
            ax.set_xticks(range(8), CELLS)
            ax.set_title(f'{model.title()} · '+('Current environment' if branch == 'current' else 'Earlier environment returns'))
            ax.set_ylabel('Selected − uniform Brier ×100\n(lower is better)')
            ax.grid(axis='y', alpha=.18)
            ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Adam · Does rehearsal usefulness depend on learner state?', fontsize=17, weight='bold')
    fig.supxlabel('Cell bits: weights / Adam state / anchor. 0 = earlier donor; 1 = later donor.\n'
        'Dots: independent seed means. Diamonds: means and descriptive 95% bootstrap intervals. Hybrid states are diagnostic.', fontsize=9)
    save(fig, output, 'state_cells')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout='constrained')
    choices = (('total_111_minus_000', 'All three components'),
               ('repair_weights', 'Earlier weights only'),
               ('repair_optimizer', 'Earlier Adam state only'),
               ('repair_anchor', 'Earlier anchor only'))
    for ax, model in zip(axes, ('conditional', 'recurrent')):
        for at, (key, _) in enumerate(choices):
            item = result['effects'][model]['current']['selection'][key]['brier']
            values = np.asarray(item['differences'])*100
            ax.scatter(values, at+np.linspace(-.12, .12, len(values)), color='#087f8c', alpha=.45, s=20)
            ax.errorbar(item['mean']*100, at,
                xerr=np.asarray([[item['mean']-item['lower']], [item['upper']-item['mean']]])*100,
                fmt='D', color='#087f8c', capsize=4, markersize=6)
        ax.axvline(0, color='#5f6b79', linestyle='--', linewidth=1)
        ax.set_yticks(range(4), [label for _,label in choices])
        ax.invert_yaxis()
        ax.set_title(model.title())
        ax.set_xlabel('Reduction in selected − uniform Brier ×100\n(positive means a better selection contrast)')
        ax.grid(axis='x', alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Which state reversions recover selection usefulness?', fontsize=17, weight='bold')
    fig.supxlabel('Current environment; each comparison starts from the all-later state.\n'
        'A better relative contrast need not improve absolute accuracy. These substitutions are not validated learning policies.', fontsize=9)
    save(fig, output, 'state_reversions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plot(args.input, args.output)
