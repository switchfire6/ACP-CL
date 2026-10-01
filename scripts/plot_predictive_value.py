"""Standalone paired-seed figures for prospectively assessed rehearsal value."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(path, output):
    result=json.loads(Path(path).read_text(encoding='utf-8'))
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for column,model in enumerate(('conditional','recurrent')):
        for row,window in enumerate(('near','return')):
            ax=axes[row,column]
            for x,(kind,color) in enumerate((('frozen','#888f98'),('reapplied','#167d8d'))):
                item=result['contrasts'][model][f'{window}_{kind}']['value_minus_uniform']['brier']
                differences=100*np.asarray(item['differences'])
                jitter=np.linspace(-.12,.12,len(differences))
                ax.scatter(x+jitter,differences,s=25,color=color,alpha=.65)
                ax.errorbar(x,item['mean']*100,
                    yerr=np.array([[item['mean']-item['lower']],[item['upper']-item['mean']]])*100,
                    fmt='D',color=color,markersize=7,capsize=5,linewidth=2)
            ax.axhline(0,color='#66717f',linestyle='--',linewidth=1)
            ax.axhline(-.05,color='#ba7230',linestyle=':',linewidth=1)
            ax.set_xticks((0,1),('Original frozen forks','Fresh rehearsal forks'))
            ax.set_title(f'{model.title()} · '+('Next observations' if window=='near' else 'Earlier condition returns'))
            ax.set_ylabel('Selected − uniform Brier ×100\n(lower is better)')
            ax.grid(axis='y',alpha=.18)
            ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Adam · Does estimated rehearsal value remain useful?',fontsize=18,weight='bold')
    fig.supxlabel('Dots: independent seed means over two assessments. Diamonds: means with descriptive 95% bootstrap intervals.\n'
                  'Dotted line: required gain for fresh rehearsal. Survival, consistency and accuracy-control gates are checked separately.',fontsize=9)
    for suffix in ('png','svg','pdf'):
        fig.savefig(output/f'predictive_value.{suffix}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    plot(args.input,args.output)
