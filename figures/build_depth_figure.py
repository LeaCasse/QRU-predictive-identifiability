"""Physical hidden-path classes and equal-budget predictive interrogation."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qru.depth_family import branch_tag
from figure_style import style,panel,save,BLUE,ORANGE,TEAL,GRAY
style();fig,axes=plt.subplots(1,2,figsize=(6.6,2.85),layout='constrained',gridspec_kw={'width_ratios':[1,1.05]})
ax=axes[0];matrix=np.array([branch_tag(8,k) for k in range(1,9)])
cmap=ListedColormap([BLUE,'#f5f5f5',ORANGE]);im=ax.imshow(matrix,cmap=cmap,norm=BoundaryNorm([-1.5,-.5,.5,1.5],3),aspect='auto')
ax.set_xticks(range(8),range(1,9));ax.set_yticks(range(8),range(1,9));ax.set_xlabel('Controlled layer');ax.set_ylabel('Hidden branch')
ax.grid(False);panel(ax,'(a) Path classes');ax.grid(False)
cb=fig.colorbar(im,ax=ax,orientation='horizontal',ticks=[-1,0,1],pad=.17,shrink=.8,fraction=.07);cb.set_label('Path index')
d=json.loads((ROOT/'results/predictive/depth_family.json').read_text())['studies'];ax=axes[1]
x=[r['depth'] for r in d];ax.plot(x,[r['passive_future_mse'] for r in d],color=GRAY,marker='o',label='Passive')
ax.plot(x,[r['informative_future_mse'] for r in d],color=TEAL,marker='s',label='Layer phases')
ax.set_yscale('log');ax.set_xlabel('Depth');ax.set_ylabel('Future MSE');ax.set_xticks(x);ax.legend(loc='lower right');panel(ax,'(b) Prediction after control')
save(fig,'appendix_C4_depth_family')
