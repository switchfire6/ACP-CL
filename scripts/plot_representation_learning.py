"""Presentation-only figures from independently scored representation results."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np


BLUE, GREEN, RED, INK = '#3467a1', '#267c68', '#b14b48', '#253548'
LABELS = ['Efficiency', 'Delay', 'Supply timing']


def forest(ax, items, labels, threshold, higher, title):
    for i, item in enumerate(items):
        stat = item['metric'] if 'metric' in item else item
        mean, low, high = (stat[key] for key in ('mean', 'lower', 'upper'))
        passed = None if threshold is None else (mean >= threshold if higher else mean <= threshold)
        color = BLUE if passed is None else GREEN if passed else RED
        ax.scatter(stat['values'], np.full(len(stat['values']), i), s=16, color=color, alpha=.25)
        ax.errorbar(mean, i, xerr=[[max(0., mean-low)], [max(0., high-mean)]],
                    fmt='o', markersize=6, color=color, capsize=3, zorder=3)
    if threshold is not None:
        ax.axvline(threshold, color=INK, lw=1, linestyle='--')
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=14)
    ax.set_xlabel('Raw Brier units; ' + ('higher is better' if higher else 'lower is better'))
    ax.grid(axis='x', alpha=.17)
    ax.spines[['top', 'right', 'left']].set_visible(False)


def qualification(result):
    groups = {item['name']: item for item in result['groups']}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.4), gridspec_kw={'hspace': .46, 'wspace': .40})
    fresh = [f'fresh/stage_{s}' for s in (1, 2, 3)] + [f'fresh/cue_{c}' for c in range(3)]
    forest(axes[0, 0], [groups[k+'/marginal_gain'] for k in fresh],
           ['Stage 1', 'Stage 2', 'Stage 3', *LABELS], .02, True,
           'Fresh learning: improvement over marginal prediction')
    forest(axes[0, 1], [groups[k+'/cue_benefit'] for k in fresh],
           ['Stage 1', 'Stage 2', 'Stage 3', *LABELS], .002, True,
           'Fresh learning: correct cue benefit')
    axes[0, 1].set_xscale('symlog', linthresh=.002)
    axes[0, 1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.3g}'))
    axes[0, 1].set_xlabel('Raw Brier benefit; linear near zero, logarithmic beyond .002')
    forest(axes[1, 0], [groups[f'interleaved/slot_{s}/absolute_brier'] for s in range(5)],
           ['Base A', 'Base B', 'Stage 1', 'Stage 2', 'Stage 3'], .12, False,
           'After interleaving: available prediction skill')
    table = axes[1, 1]
    passed = np.zeros((3, 3))
    for stage in range(1, 4):
        for cue in range(3):
            item = groups[f'interleaved/stage_{stage}/cue_{cue}/cue_benefit']
            passed[stage-1, cue] = int(item['passed'])
            value = item['metric']
            table.text(cue, stage-1, f"{value['mean']:.5f}\n(n={value['n']})",
                       ha='center', va='center', fontsize=12, color=INK)
    table.imshow(passed, cmap=ListedColormap(['#f5dcd7', '#d9eee6']), vmin=0, vmax=1,
                 interpolation='nearest', aspect='auto')
    table.set_xticks(range(3), LABELS)
    table.set_yticks(range(3), ['Stage 1', 'Stage 2', 'Stage 3'])
    table.set_title('After interleaving: every active cue must be usable',
                    loc='left', fontsize=12, fontweight='bold', pad=14)
    table.set_xlabel('Mean correct-cue benefit; green >= .002, red below .002')
    table.spines[:].set_visible(False)
    return fig


def main_comparison(result):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), gridspec_kw={'wspace': .40})
    arms = ['outcome', 'final_frame', 'sequence', 'compute']
    names = ['Outcome replay', 'Final-frame auxiliary', 'Full-sequence auxiliary', 'Extra outcome updates']
    for index, metric in enumerate(('acquisition', 'whole')):
        values = [result['levels'][a]['acquisition'] if metric == 'acquisition'
                  else result['levels'][a]['whole']['brier'] for a in arms]
        forest(axes[index], values, names, None, False,
               'New-dependency acquisition' if index == 0 else 'Complete online stream')
        for line in axes[index].lines:
            if line.get_linestyle() == '--':
                line.set_visible(False)
    groups = {item['name']: item for item in result['groups']}
    contrasts = [groups['acquisition_minus_'+a] for a in ('outcome', 'final_frame', 'compute')]
    forest(axes[2], contrasts, ['vs outcome', 'vs final frame', 'vs extra updates'], 0., False,
           'Full sequence minus each control')
    for i, margin in enumerate((-.002, -.001, -.002)):
        axes[2].scatter([margin], [i], marker='|', s=180, color=INK, zorder=4)
    return fig


def plot(summary, output):
    result = json.loads(Path(summary).read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.labelcolor': INK, 'text.color': INK,
                         'xtick.color': INK, 'ytick.color': INK})
    fig = qualification(result) if result['kind'] == 'qualification' else main_comparison(result)
    failed = sum(not item['passed'] for item in result['groups'])
    title = ('Adam | Reference learnability' if result['kind'] == 'qualification'
             else 'Adam | Temporal reconstruction comparison')
    fig.suptitle(title, x=.075, ha='left', y=.985, fontsize=19, fontweight='bold')
    fig.text(.075, .941, f"Decision: {result['decision']}  |  {failed} of {len(result['groups'])} required checks failed",
             fontsize=12, color=RED if failed else GREEN)
    fig.subplots_adjust(top=.86, bottom=.12, left=.12, right=.975)
    fig.text(.075, .025,
             'Dots: independent seed values. Bars: descriptive 95% bootstrap intervals. '
             'Dashed lines: fixed thresholds.\n'
             ('All models start from random weights. Qualification is a reference check, not a test of the auxiliary candidate.'
              if result['kind'] == 'qualification' else
              'All models start from random weights. Primary contrasts also require five of six seeds to improve.'),
             fontsize=9, color=INK)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.pdf', '.svg'):
        fig.savefig(output.with_suffix(suffix), dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', required=True)
    parser.add_argument('--output', required=True)
    arguments = parser.parse_args()
    plot(arguments.summary, arguments.output)
