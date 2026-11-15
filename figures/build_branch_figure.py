"""Paired training-noise experiment, with all five specified modes."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from figure_style import style,panel,save,BLUE,ORANGE,TEAL
style();data=json.loads((ROOT/'results/predictive/branch_selection.json').read_text())['summary']
labels=['Full','Batch 8','Noise .05','Noise .20','128 shots'];x=np.arange(5)
fig,axes=plt.subplots(1,3,figsize=(6.6,2.65),layout='constrained')
a=axes[0];p=np.array([r['A']/r['n'] for r in data]);ci=np.array([r['A_fraction_interval'] for r in data])
a.errorbar(x,p,yerr=np.array([p-ci[:,0],ci[:,1]-p]),fmt='o',color=BLUE,capsize=2.5,lw=1)
a.set_ylim(.25,.5);a.set_ylabel('Fraction in class A');panel(a,'(a) Branch selection')
a=axes[1];a.bar(x-.17,[r['switch_A_to_B'] for r in data],width=.34,label='A to B',color=BLUE)
a.bar(x+.17,[r['switch_B_to_A'] for r in data],width=.34,label='B to A',color=ORANGE,hatch='//')
a.set_ylabel('Changed classes');a.legend(loc='upper left');panel(a,'(b) Paired changes')
a=axes[2];a.plot(x,[.5*(r['raw_mean_future_mse_A']+r['raw_mean_future_mse_B']) for r in data],'o-',color=BLUE,label='Raw fit')
a.plot(x,[.5*(r['mean_future_mse_A']+r['mean_future_mse_B']) for r in data],'s--',color=ORANGE,label='Clean completion')
a.set_ylim(.025,.054);a.set_ylabel('Equal-prior future MSE');a.legend(loc='upper left');panel(a,'(c) Fit and future loss')
for ax in axes:ax.set_xticks(x,labels,rotation=40,ha='right');ax.tick_params(axis='x',labelsize=7)
save(fig,'appendix_F3_branch_selection')
