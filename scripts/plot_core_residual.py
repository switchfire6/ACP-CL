"""Standalone seed-level acquisition, retention and feature-attribution figures."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS = {'joint':'#536c82', 'separate':'#007f78', 'fixed_features':'#af6b32', 'fresh':'#85708f'}
LABELS = {'joint':'Both pathways learn', 'separate':'Frozen core + adaptive residual',
          'fixed_features':'Frozen core + fixed residual encoder', 'fresh':'Fresh reference'}


def save(fig, output, name):
    for suffix in ('png','svg','pdf'):
        fig.savefig(output/f'{name}.{suffix}', dpi=180)
    plt.close(fig)


def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', alpha=.18)


def estimate(ax, x, metric, color, scale=100):
    values = np.asarray(metric['differences'])*scale
    ax.scatter(x+np.linspace(-.12,.12,len(values)), values, s=24, color=color, alpha=.6, zorder=3)
    ax.errorbar(x, metric['mean']*scale,
        yerr=np.maximum(0, np.asarray([[metric['mean']-metric['lower']],
                                      [metric['upper']-metric['mean']]])*scale),
        fmt='D', color=color, capsize=4, markersize=6, zorder=4)


def plot(path, output):
    result = json.loads(Path(path).read_text(encoding='utf-8'))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    models = result['config']['models']
    fig, axes = plt.subplots(len(models), 3, figsize=(13, 4*len(models)),
                             layout='constrained', squeeze=False)
    choices = (('novel','brier_auc','New dependency\nAffected-case Brier AUC'),
               ('return','all_brier_auc','Earlier environment returns\nAll-case Brier AUC'),
               ('return','novel_after_return_brier','After the return\nNew dependency, correct-support probe'))
    for row, model in enumerate(models):
        for col,(phase,field,title) in enumerate(choices):
            ax = axes[row,col]
            for x,arm in enumerate(('joint','separate','fixed_features')):
                estimate(ax, x, result['levels'][model][phase][arm][field], COLORS[arm])
            ax.set_xticks(range(3), ['Both learn','Frozen core\nAdaptive residual','Frozen core\nFixed encoder'])
            ax.set_title(f'{model.title()} | {title}', fontsize=11)
            ax.set_ylabel('Brier x100 (lower is better)')
            style(ax)
    fig.suptitle('Adam: can a stable core leave room for new learning?', fontsize=17, weight='bold')
    fig.supxlabel('Dots: independent seeds. Diamonds: means and descriptive 95% seed-bootstrap intervals.\n'
        'Identical capacity, observations, replay and update counts; backward work differs. Correct-support probes are diagnostic.', fontsize=9)
    save(fig, output, 'acquisition_and_retention')

    fig, axes = plt.subplots(1, len(models), figsize=(13, 5.8), layout='constrained', squeeze=False)
    gates = (('novel','brier_auc','New acquisition',-.002),
             ('novel','valid_after_mode_0','Old knowledge: mode 0',.005),
             ('novel','valid_after_mode_1','Old knowledge: mode 1',.005),
             ('return','all_brier_auc','Return learning',.005),
             ('return','novel_after_return_brier','New knowledge after return',.005))
    for ax,model in zip(axes[0],models):
        comparison = result['comparisons'][model]['separate_minus_joint']
        for y,(phase,field,label,limit) in enumerate(gates):
            item = comparison[phase][field]
            color = COLORS['separate']
            ax.scatter(np.asarray(item['differences'])*100, y+np.linspace(-.13,.13,item['n']),
                       s=20, color=color, alpha=.5)
            ax.errorbar(item['mean']*100, y,
                xerr=np.maximum(0,np.asarray([[item['mean']-item['lower']], [item['upper']-item['mean']]])*100),
                fmt='D', color=color, capsize=4)
            ax.scatter(limit*100,y,marker='|',s=190,color='#b14a3b',zorder=4)
        ax.axvline(0,color='#73818d',linestyle='--',linewidth=1)
        ax.set_yticks(range(len(gates)),[item[2] for item in gates])
        ax.invert_yaxis()
        ax.set_xlabel('Separate minus joint Brier x100\nNegative favors the frozen core + adaptive residual')
        ax.set_title(model.title()+': '+('passes' if result['screen'][model]['passed'] else 'does not pass')+' the fixed screen')
        style(ax)
    fig.suptitle('Acquisition gains must also preserve earlier and newer knowledge',fontsize=16,weight='bold')
    fig.supxlabel('Red ticks: prespecified maximum mean differences. All gates must pass; qualification, 5/6 acquisition consistency\n'
        'and both survival guards are additional requirements. Six seeds per architecture; these are development screens.',fontsize=9)
    save(fig,output,'primary_comparison')

    fig, axes = plt.subplots(1,len(models),figsize=(11,5.4),layout='constrained',squeeze=False)
    for ax,model in zip(axes[0],models):
        for x,name in enumerate(('separate_minus_joint','separate_minus_fixed_features')):
            estimate(ax,x,result['comparisons'][model][name]['novel']['brier_auc'],COLORS['separate'])
        ax.axhline(0,color='#73818d',linestyle='--',linewidth=1)
        ax.scatter([0,1],[-.2,-.1],marker='_',s=180,color='#b14a3b',zorder=4)
        ax.set_xticks([0,1],['Primary comparison\nSeparate - joint','Visual feature attribution\nSeparate - fixed encoder'])
        ax.set_ylabel('New-dependency Brier AUC difference x100\nNegative favors the adaptive residual')
        ax.set_title(model.title())
        style(ax)
    fig.suptitle('Does learning new visual features contribute?',fontsize=17,weight='bold')
    fig.supxlabel('Attribution requires mean <= -0.001 and improvement in at least 5/6 seeds. Passing attribution cannot rescue\n'
        'a failed primary screen. The fixed-encoder control still trains its heads and recurrent context/readout.',fontsize=9)
    save(fig,output,'feature_attribution')

    records_path = Path(path).parent/'raw_results.jsonl.gz'
    if records_path.is_file():
        with gzip.open(records_path,'rt',encoding='utf-8') as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        fig, axes = plt.subplots(len(models),3,figsize=(13,4*len(models)),
                                 layout='constrained',squeeze=False)
        for row,model in enumerate(models):
            for col,phase in enumerate(('maintenance','novel','return')):
                ax = axes[row,col]
                field = 'focus_brier' if phase == 'novel' else 'brier'
                arms = ('joint','separate','fixed_features','fresh') if phase == 'novel' else ('joint','separate','fixed_features')
                for arm in arms:
                    group = [record for record in records if record['model'] == model
                             and record['arm'] == arm and record['phase'] == phase]
                    if not group:
                        raise ValueError('missing learning curve group')
                    x = [point['arrivals'] for point in group[0]['curve']]
                    values = np.asarray([[point['metrics'][field] for point in record['curve']]
                                         for record in group])*100
                    for seed_values in values:
                        ax.plot(x,seed_values,color=COLORS[arm],alpha=.12,linewidth=.65)
                    ax.plot(x,values.mean(axis=0),color=COLORS[arm],linewidth=2,label=LABELS[arm],
                            linestyle='--' if arm == 'fresh' else '-')
                ax.set_title(f'{model.title()} | {phase.title()}')
                ax.set_xlabel('Arrivals within phase')
                ax.set_ylabel(('Affected' if phase == 'novel' else 'All-case')+' Brier x100')
                style(ax)
        handles,labels = axes[0,1].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncols=2,fontsize=9)
        fig.suptitle('Continuing learning through maintenance, novelty and return\n'
            'Thick lines: means. Faint lines: seeds. Actual causal history; lower error is better.',
            fontsize=15,weight='bold')
        save(fig,output,'learning_curves')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    plot(args.input,args.output)
