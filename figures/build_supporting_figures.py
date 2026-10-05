"""Exact schematics and supporting plots from frozen, checked source evidence."""
import argparse,json
from itertools import product
from collections import Counter
from pathlib import Path
import numpy as np
from data_io import main_data, supporting_data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
from build_main_figures import BLUE,ORANGE,TEAL,PURPLE,GRAY,RED,panel,save,pi_axis,style
ROOT=Path(__file__).resolve().parent

def mechanism(data):
    fig,axs=plt.subplots(2,2,figsize=(6.6,4.7),layout='constrained')
    a,b,c,d=axs.ravel();a.set_axis_off();a.set(xlim=(-.18,1.22),ylim=(-.75,3.6))
    panel(a,'(a) Hidden paths, shared frequencies')
    rows=[('A',(1,-1),3,2.5,BLUE,'o'),('B',(-1,0),2,2.5,ORANGE,'s'),('A',(-1,1),1,.5,BLUE,'o'),('B',(1,0),0,.5,ORANGE,'s')]
    for branch,tag,y,z,col,mark in rows:
        assert sum(t*w for t,w in zip(tag,(1,2)))==(-1 if z==2.5 else 1)
        a.text(-.17,y,f'{branch}: '+str(tag),ha='left',va='center',fontsize=8)
        a.scatter(.38,y,color=col,marker=mark,s=20,zorder=3)
        a.annotate('',xy=(.95,z),xytext=(.40,y),arrowprops={'arrowstyle':'->','color':col,'lw':1.25,'linestyle':'-' if branch=='A' else '--'})
    for z,w in [(2.5,-1),(.5,1)]:
        a.text(.98,z,rf'$\omega={w:+d}$'+'\n'+r'$C_\omega=1/2$',ha='left',va='center',fontsize=8)
    a.text(.35,-.65,r'$\alpha_0=(1,2),\quad a_s=1/2$',ha='center',fontsize=8)
    x=np.linspace(0,2*np.pi,513);example=data.get('illustration',data['exact']);delta=example['delta']
    b.plot(x,np.cos(x),color=BLUE,label='A',lw=1.8)
    b.plot(x,np.cos(x),color=ORANGE,label='B',ls='--',lw=1.4)
    b.set(ylabel=r'Present mean $\langle Z\rangle$',ylim=(-1.14,1.14));pi_axis(b)
    b.text(.5,.91,r'$f_A=f_B=\cos x$',transform=b.transAxes,ha='center',fontsize=8)
    b.legend(loc='lower left',ncol=2);panel(b,'(b) Identical passive responses')
    c.plot(x,np.cos((1+delta)*x),color=BLUE,label=rf'A: $\cos({1+delta:.2f}x)$')
    c.plot(x,np.cos(x),color=ORANGE,ls='--',label=r'B: $\cos x$')
    c.set(ylabel=r'Future mean $\langle Z\rangle$',ylim=(-1.14,1.14));pi_axis(c)
    c.legend(loc='lower left');panel(c,rf'(c) Transverse shift: $\alpha_2\to{2+delta:.2f}$')
    gap=np.cos((1+delta)*x)-np.cos(x)
    assert np.isclose(np.mean(gap**2),example['future_mse'],atol=1e-14)
    d.plot(x,gap,color=TEAL,label='Transverse: exact gap')
    d.plot(x,-delta*x*np.sin(x),color=PURPLE,ls=':',label='First-order term')
    d.axhline(0,color=GRAY,ls='--',label='Parallel: zero gap',lw=1.1)
    d.set(ylabel=r'Future gap $f_A-f_B$');pi_axis(d)
    d.legend(loc='upper left',fontsize=6.7);panel(d,'(d) Drift direction determines the gap')
    save(fig,'fig1_latent_path_mechanism')

def tags(depth):return [(f,)+t for f in (-1,1) for t in product((-1,0,1),repeat=depth-1)]

def tag_map():
    fig,axs=plt.subplots(1,2,figsize=(6.6,5.0),layout='constrained')
    for ax,alpha,title in zip(axs,[(1,2,4),(1,3,9)],['(a) Dyadic: 18 tags to 8 tones','(b) Ternary: 18 distinct tones']):
        ts=sorted(tags(3),key=lambda s:(sum(t*w for t,w in zip(s,alpha)),s));freqs=sorted({sum(t*w for t,w in zip(s,alpha)) for s in ts})
        ys=np.arange(len(ts))[::-1];out=np.linspace(17,0,len(freqs));target=dict(zip(freqs,out))
        ax.set_axis_off();ax.set(xlim=(-.1,1.14),ylim=(-1,19));panel(ax,title)
        ax.text(.4,18.25,rf'$\alpha={alpha}$',ha='center',fontsize=9)
        for tag,y in zip(ts,ys):
            w=sum(t*a for t,a in zip(tag,alpha));z=target[w]
            ax.text(0,y,str(tag),ha='left',va='center',fontsize=7.4)
            ax.plot([.43,.88],[y,z],color=BLUE if len(freqs)<18 else TEAL,lw=.65,alpha=.8)
            ax.scatter(.43,y,s=9,color=BLUE if len(freqs)<18 else TEAL)
        for w,z in target.items():
            n=sum(sum(t*a for t,a in zip(tag,alpha))==w for tag in ts)
            ax.scatter(.89,z,s=14+8*(n-1),facecolor='white',edgecolor=GRAY,zorder=4)
            ax.text(.94,z,f'{w:+d}',fontsize=7.8,va='center')
        ax.text(.17,-.9,'Path tag s',ha='center');ax.text(.93,-.9,r'$\omega=s\cdot\alpha$',ha='center')
    save(fig,'appendix_A0_tag_map')

def jet_pair(data):
    j=data['jet_pair'];fig,axs=plt.subplots(1,3,figsize=(6.6,2.65),layout='constrained');a,b,c=axs
    x=np.array(j['x']);u=np.array(j['u'])
    a.axvspan(np.pi-.425,np.pi+.425,color='#e9edf0',alpha=.9)
    for ax,xx,k0,k1 in [(a,x,'A','B'),(b,u,'local_A','local_B')]:
        ax.plot(xx,j[k0],color=BLUE,label='QRU A');ax.plot(xx,j[k1],color=ORANGE,ls='--',label='QRU B');ax.legend(loc='lower left')
    a.set(ylabel=r'Mean $\langle Z\rangle$');pi_axis(a);panel(a,'(a) Broad-input difference')
    b.set(xlabel=r'Centered input $u=x-\pi$',ylabel=r'Mean $\langle Z\rangle$');b.set_xticks([-.4,0,.4]);panel(b,'(b) Narrow-window masking')
    co=np.array(j['coefficient_gap']);c.axvspan(-.4,5.5,color='#e9edf0',alpha=.9)
    c.semilogy(range(10),co,'o-',color=TEAL,ms=3.5);c.axvline(5.5,color=GRAY,ls='--',lw=.8)
    c.set(xlabel='Input-Taylor order k',ylabel=r'$|c_{k,A}-c_{k,B}|$',xticks=[0,2,4,6,8],ylim=(1e-16,1));c.minorticks_off();panel(c,'(c) Unmatched order six')
    save(fig,'appendix_D3_jet_pair')

def geometry(data):
    g=data['geometry'];fig,axs=plt.subplots(1,3,figsize=(6.6,2.75),layout='constrained');a,b,c=axs
    x=g['x'];a.plot(x,g['reference'],color=GRAY,label='Reference')
    a.plot(x,g['active_response'],color=BLUE,label=r'Active $v_1$, $\epsilon=.3$')
    a.plot(x,g['null_response'],color=PURPLE,ls='--',label=r'Null $v_7$, $\epsilon=.3$')
    a.set(ylabel=r'Mean $\langle Z\rangle$');pi_axis(a);a.legend(loc='lower left',fontsize=6.4);panel(a,'(a) Finite parameter steps')
    s=np.array(g['relative_singular_values']);b.semilogy(range(1,10),s,'o-',color=TEAL,ms=3.5);b.axhline(1e-8,color=GRAY,ls='--',lw=.9,label=r'Rank tolerance $10^{-8}$');b.axvline(6.5,color=GRAY,lw=.8,ls=':')
    b.set(xlabel='Singular-value index i',ylabel=r'$\sigma_i/\sigma_1$',xticks=[1,3,6,9],ylim=(1e-18,2));b.minorticks_off();b.legend(loc='upper right',fontsize=6.1);panel(b,'(b) Functional rank six')
    for i,r in enumerate(g['curves']):
        color=[BLUE,TEAL,PURPLE,ORANGE][i];c.loglog(g['radius'],r['rms'],color=color,marker=['o','s','^','D'][i],ls=['-','--','-.',':'][i],ms=2.8,label=rf"{r['kind'].capitalize()} $v_{{{r['vector']}}}$")
    c.set(xlabel=r'Parameter step $\epsilon$',ylabel='RMS output displacement');c.minorticks_off();c.legend(loc='upper left',fontsize=6.4);panel(c,'(c) First versus second order')
    save(fig,'appendix_E0_parameter_geometry')

def circuit(ax,entangle):
    ax.set_axis_off();ax.set(xlim=(-.3,5.4),ylim=(-.5,1.8))
    col=TEAL if entangle else BLUE
    for y in [1,0]:
        ax.plot([0,4.9],[y,y],color=GRAY,lw=.85,zorder=1);ax.text(-.18,y,r'$|0\rangle$',ha='right',va='center',fontsize=9)
        for i,z in enumerate([.65,2.1,3.55]):
            ax.add_patch(Rectangle((z-.33,y-.23),.66,.46,facecolor='white',edgecolor=col,lw=1.1,zorder=3))
            ax.text(z,y,rf'$U_{{{i+1},{0 if y else 1}}}$',ha='center',va='center',fontsize=8,zorder=4)
    if entangle:
        for z in [1.22,2.67,4.12]:
            ax.plot([z,z],[0,1],color='#28333c',lw=1);ax.scatter(z,1,s=18,color='#28333c',zorder=4);ax.add_patch(Circle((z,0),.13,facecolor='white',edgecolor='#28333c',lw=1));ax.plot([z-.13,z+.13],[0,0],color='#28333c',lw=1);ax.plot([z,z],[-.13,.13],color='#28333c',lw=1)
    for y in [1,0]:
        ax.text(4.52,y,r'$\cdots$',ha='center',va='center',zorder=5,
                bbox={'facecolor':'white','edgecolor':'none','pad':1.5})
    ax.text(5.0,1,r'$Z_0$',va='center',fontsize=9);ax.text(5.0,0,'trace',va='center',fontsize=7)
    ax.text(2.1,1.55,r'$U_{\ell,q}=R_zR_xR_y(\alpha_\ell x+b_{\ell,q})$',ha='center',fontsize=8)

def two_qubit(data):
    fig,axs=plt.subplots(2,2,figsize=(6.6,4.8),layout='constrained');a,b,c,d=axs.ravel();circuit(a,False);circuit(b,True)
    panel(a,'(a) Two qubits, no entangling gate');panel(b,'(b) Same uploads, CNOT per layer')
    rows=data['two_qubit']
    for flag,col,mark,label in [('no',BLUE,'o','No CNOT'),('yes',TEAL,'s','CNOT per layer')]:
        rr=[r for r in rows if r['CNOT after layer']==flag]
        c.plot([r['Depth L'] for r in rr],[r['Rank theta'] for r in rr],marker=mark,color=col,label=label)
    c.set(xlabel='Circuit depth L',ylabel=r'Nuisance rank $r_\theta$',xticks=[1,2,3],yticks=[2,4,7,10,13],ylim=(0,14.5));c.legend(loc='upper left');panel(c,'(c) Effective nuisance dimension')
    rr=[next(r for r in rows if r['Depth L']==L and r['CNOT after layer']==f) for L,f in [(2,'no'),(2,'yes'),(3,'no'),(3,'yes')]]
    for y,r in zip([3,2,1,0],rr):
        ks=[int(k) for k in str(r['First orders of visible directions']).split(',')];flag=r['CNOT after layer']=='yes';col=TEAL if flag else BLUE
        d.plot(ks,[y]*len(ks),marker='s' if flag else 'o',color=col,ms=4,lw=1.3)
    d.set(yticks=[3,2,1,0],yticklabels=['L2: no CNOT','L2: CNOT','L3: no CNOT','L3: CNOT'],xlabel='Emerging input-Taylor order',xticks=[4,6,8,10,12,14,15],xlim=(3,16),ylim=(-.6,3.6));d.grid(axis='y',visible=False);d.grid(axis='x',color='.91',lw=.5);panel(d,'(d) Orders after nuisance profiling')
    save(fig,'appendix_E2_two_qubit')

def counts():
    fig,axs=plt.subplots(1,2,figsize=(6.6,2.8),layout='constrained');a,b=axs
    L=np.arange(1,6);a.semilogy(L,2*3.**(L-1),'o-',color=GRAY,label=r'Admissible tags: $2\,3^{L-1}$')
    for typ,col,mark in [('dyadic',BLUE,'s'),('harmonic',TEAL,'^')]:
        freqs=[]
        for l in L:
            alpha=2**np.arange(l) if typ=='dyadic' else np.arange(1,l+1)
            c=Counter(sum(t*w for t,w in zip(tag,alpha)) for tag in tags(l));freqs.append(len(c))
        a.semilogy(L,freqs,marker=mark,color=col,label=f'{typ.capitalize()}: distinct frequencies')
        pos=sorted(w for w in c if w>=0);b.plot(pos,[c[w] for w in pos],marker=mark,ls='none',color=col,ms=4,label=typ.capitalize())
    a.set(xlabel='Circuit depth L',ylabel='Count',xticks=L);a.legend(loc='upper left',fontsize=6.5);a.minorticks_off();panel(a,'(a) Schedule-induced tag compression')
    b.set(xlabel=r'Nonnegative frequency $\omega$',ylabel='Tag multiplicity',ylim=(0,11));b.legend();panel(b,'(b) Collision multiplicity at L = 5');save(fig,'appendix_A1_exact_tag_collisions')

def sidebands():
    alpha=[1,2,4,7,11];peaks={1,2,4,7,9,11,13,18};pairs=[]
    for i in range(5):
        for j in range(i+1,5):
            matches=sorted({alpha[j]-alpha[i],alpha[j]+alpha[i]}&peaks)
            if matches:pairs.append((i,j,matches,sorted(set(matches)-set(alpha))))
    assert len(pairs)==7 and sum(bool(r[3]) for r in pairs)==3
    fig,ax=plt.subplots(figsize=(6.6,2.9),layout='constrained');yy=np.arange(7)[::-1]
    for y,(i,j,matched,novel) in zip(yy,pairs):
        ax.text(.05,y,rf'$({i+1},{j+1})$',va='center');ax.text(.3,y,', '.join(map(str,matched)),va='center')
        ax.scatter(.66,y,marker='o',color=BLUE,s=24)
        ax.scatter(.88,y,marker='s' if novel else 'x',facecolor=TEAL if novel else GRAY,s=24)
    ax.set(xlim=(0,1.05),ylim=(-.7,7.1),xticks=[],yticks=[]);ax.spines[['left','bottom']].set_visible(False)
    for x,t in [(.05,'Layer pair'),(.3,'Matched labels'),(.66,'Support-compatible'),(.88,'Non-base retained')]:ax.text(x,6.75,t,ha='left' if x<.5 else 'center',va='center',fontsize=7.5,fontweight='semibold')
    panel(ax,'Pairwise sideband attribution: seven edges, three non-base edges');save(fig,'appendix_A2_order2_sideband_compatibility')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--all',action='store_true');args=parser.parse_args();style()
    data=supporting_data()
    for f in [tag_map,lambda:jet_pair(data),lambda:geometry(data),lambda:two_qubit(data)]:f()
    if args.all:counts();sidebands()
    print('Rebuilt exact diagrams and supporting source-grounded figures.')
