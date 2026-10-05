"""Shared publication style. QRU_PRINT_SAFE=0 disables redundant print encodings."""
import os
import hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.markers import MarkerStyle
from cycler import cycler

BLUE='#244763'      # navy: QRU / branch A
TEAL='#087e83'      # teal: informative acquisition / path control
ORANGE='#b38226'    # ochre: comparator / branch B
PURPLE='#775a82'    # plum: blind or null-direction control
GRAY='#65717b'      # slate: passive / reference
RED='#a74d54'       # muted crimson: harmful outcome
PALETTE=[BLUE,TEAL,PURPLE,ORANGE,RED,GRAY]
WIDTH=6.6
PRINT_SAFE=os.environ.get('QRU_PRINT_SAFE','1')!='0'
ROOT=Path(__file__).resolve().parent
STYLES={BLUE:('-', 'o'),TEAL:('--','s'),PURPLE:('-.','D'),ORANGE:(':','^'),RED:((0,(5,2,1,2)),'x'),GRAY:((0,(3,2)),'o')}

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans',
        'font.size':8.4,'axes.titlesize':8.7,'axes.labelsize':8.2,
        'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#73808a',
        'axes.linewidth':.65,'axes.titlepad':8,'axes.labelpad':4,
        'lines.linewidth':1.4,'lines.markersize':3.6,'pdf.fonttype':42,
        'ps.fonttype':42,'figure.facecolor':'white','axes.facecolor':'white',
        'savefig.facecolor':'white','legend.frameon':True,'legend.framealpha':.94,
        'legend.facecolor':'white','legend.edgecolor':'white',
        'legend.borderpad':.35,'legend.labelspacing':.35,'legend.handlelength':2.3,
        'axes.prop_cycle':cycler(color=PALETTE),'text.color':'#27313a',
        'axes.labelcolor':'#27313a','xtick.color':'#27313a','ytick.color':'#27313a',
        'grid.color':'#e3e7ea','grid.linewidth':.55,'axes.axisbelow':True})

def panel(ax,title):
    ax.set_title(title,loc='left',fontweight='semibold');ax.grid(axis='y',color='#e3e7ea',lw=.55);ax.set_axisbelow(True)

def color_key(value):
    try:
        rgba=np.array(colors.to_rgba(value));best=min(PALETTE,key=lambda c:np.linalg.norm(rgba[:3]-np.array(colors.to_rgb(c))))
        return best if np.linalg.norm(rgba[:3]-np.array(colors.to_rgb(best)))<.015 else None
    except (ValueError,TypeError):return None

def coordinate_fingerprint(fig):
    h=hashlib.sha256()
    for ax in fig.axes:
        for line in ax.lines:
            h.update(np.asarray(line.get_xdata(),dtype=float).tobytes())
            h.update(np.asarray(line.get_ydata(),dtype=float).tobytes())
        for coll in ax.collections:
            if hasattr(coll,'get_offsets'):
                h.update(np.asarray(coll.get_offsets(),dtype=float).tobytes())
    return h.hexdigest()

def finish(fig):
    before=coordinate_fingerprint(fig)
    fig.set_size_inches(WIDTH,fig.get_figheight())
    # Redundant encodings change only artist appearance, never coordinates.
    if PRINT_SAFE:
        for ax in fig.axes:
            for line in ax.lines:
                if getattr(line,'_print_safe_skip',False):continue
                if line.get_marker() in ('_', '|'):continue  # uncertainty caps
                key=color_key(line.get_color());n=len(line.get_xdata())
                if key is None or n<3:continue
                ls,mk=STYLES[key]
                if line.get_linestyle() not in ('None','none',''):
                    line.set_linestyle(ls)
                if line.get_alpha() is None or line.get_alpha()>=.5:
                    line.set_marker(mk);line.set_markersize(3.1)
                    if n>25:
                        offset={BLUE:0,TEAL:2,PURPLE:4,ORANGE:6,RED:8,GRAY:10}[key]
                        every=max(15,n//8);line.set_markevery((offset,every))
                if key==GRAY and line.get_marker() not in ('None',None,''):
                    line.set_markerfacecolor('white');line.set_markeredgewidth(.8)
            for coll in ax.collections:
                if not hasattr(coll,'get_paths') or not hasattr(coll,'get_facecolors'):continue
                paths=coll.get_paths();fc=coll.get_facecolors()
                # Only default circles are recoded; hollow rings, crosses,
                # explicit model markers and uncertainty bars retain meaning.
                if len(paths)!=1 or len(paths[0].vertices)!=26 or len(fc)!=1:continue
                key=color_key(fc[0])
                if key is None:continue
                mark=STYLES[key][1];obj=MarkerStyle(mark)
                coll.set_paths([obj.get_path().transformed(obj.get_transform())])
                if key==GRAY:
                    coll.set_facecolor('white');coll.set_edgecolor(GRAY);coll.set_linewidth(.8)
                elif key==RED:
                    coll.set_facecolor('none');coll.set_edgecolor(RED);coll.set_linewidth(1.1)
                    coll.set_sizes(np.maximum(coll.get_sizes(),32))
    for ax in fig.axes:
        title=ax.get_title()
        if title:
            ax.set_title('',loc='center')
            ax.set_title(title,loc='left',fontweight='semibold')
        ax._left_title.set_fontsize(8.7)
        ax.title.set_fontsize(8.7)
        ax.xaxis.label.set_fontsize(8.2);ax.yaxis.label.set_fontsize(8.2)
        ax.tick_params(labelsize=7.5)
        legend=ax.get_legend()
        if PRINT_SAFE and legend is not None:
            handles,labels=ax.get_legend_handles_labels()
            if handles:
                legend=ax.legend(handles,labels,loc=legend._loc,ncol=legend._ncols,fontsize=7.,frameon=True,
                                 **getattr(ax,'_legend_kwargs',{}))
        if legend is not None:
            for t in legend.get_texts():t.set_fontsize(7.)
            legend.get_frame().set_facecolor('white');legend.get_frame().set_edgecolor('white');legend.get_frame().set_alpha(.94)
    assert coordinate_fingerprint(fig)==before, "Style changed plotted numerical coordinates"
    return fig

def save(fig,name):
    finish(fig);folder=ROOT.parent/'build/figures';folder.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:
        tmp=folder/(name+'.tmp.'+ext)
        fig.savefig(tmp,dpi=300,bbox_inches='tight',pad_inches=.08)
        tmp.replace(folder/(name+'.'+ext))
    plt.close(fig)
