"""Rebuild the six main figures from the frozen publication data."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from data_io import main_data, supporting_data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, LogLocator
import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parent
from figure_style import BLUE, ORANGE, TEAL, PURPLE, GRAY, RED, style, panel, save

ILLUSTRATIVE_DELTA = 0.05


def illustrative_example(data):
    x = np.linspace(0, 2*np.pi, 513)
    gap = np.cos((1+ILLUSTRATIVE_DELTA)*x)-np.cos(x)
    mse = float(np.mean(gap**2))
    data['illustration'] = {
        'delta': ILLUSTRATIVE_DELTA,
        'input_grid': '513 equally spaced inputs in [0,2pi]',
        'future_rms': mse**.5, 'future_mse': mse,
        'future_risk_lower_bound': mse/4,
        'derivation': 'Exact twin identities with the same circuit parameters as exact, only the illustrative transverse encoding change is enlarged. Archived exact and regular-twin records remain unchanged.'}
    return data


def pi_axis(ax):
    ax.set_xticks([0,np.pi,2*np.pi], ['0',r'$\pi$',r'$2\pi$'])
    ax.set_xlabel('Input x')


def log_axis(ax, direction='y'):
    axis = ax.yaxis if direction=='y' else ax.xaxis
    axis.set_minor_locator(NullLocator())
    axis.set_major_locator(LogLocator(base=10, numticks=5))


def mechanism(data):
    from build_supporting_figures import mechanism as draw
    draw(data)


def local_observability(data):
    fig, ax=plt.subplots(1,3,figsize=(7.15,3.4),layout='constrained')
    for schedule,nuisance,color,label,marker in [
        ('dyadic','all',BLUE,'QRU: all angles','o'),
        ('dyadic','bias_only',ORANGE,'QRU: biases only','^')]:
        rows=sorted([r for r in data['architecture'] if
            r['Architecture / schedule']==schedule and r['Trainable theta']==nuisance],
            key=lambda r:r['Depth L'])
        depths=[r['Depth L'] for r in rows]
        lo=[r['First visible order'] for r in rows]
        hi=[int(str(r['First orders of each visible direction']).split(',')[-1]) for r in rows]
        ax[0].plot(depths,lo,marker=marker,color=color,ms=4,label=label)
        for l,u,v in zip(depths,lo,hi):
            ax[0].plot([l,l],[u,v],color=color,lw=2,alpha=.6)
    classic=sorted([r for r in data['architecture'] if r['Architecture / schedule']=='classical_distinct'],key=lambda r:r['Depth L'])
    assert [r['First visible order'] for r in classic]==[2,4,6,8]
    ax[0].scatter([1,2,3,4],[2,4,6,8],s=42,marker='o',facecolors='none',edgecolors=GRAY,
                  lw=1.1,label='Fourier: 2L amplitudes',zorder=5)
    ax[0].set(xlabel='Circuit depth L',ylabel='Visible input-Taylor orders',
              xticks=[1,2,3,4],yticks=[1,3,5,7,9,11],ylim=(.5,11.8))
    ax[0]._legend_kwargs={'bbox_to_anchor':(.5,-.30),'borderaxespad':0}
    ax[0].legend(frameon=False,loc='upper center',**ax[0]._legend_kwargs)
    panel(ax[0],'(a) Visible Taylor orders')
    medians={}
    for family,color,label in [('QRU',BLUE,'QRU, 6 pairs'),('classical_four_tone',ORANGE,'Fourier, 6 pairs')]:
        rr=[r for r in data['finite_pairs'] if r['family']==family]
        widths=sorted({r['width'] for r in rr});samples=sorted({r['sample'] for r in rr})
        curves=np.array([[next(r['local_rms'] for r in rr if r['width']==h and r['sample']==s)
                          for h in widths] for s in samples])
        assert curves.shape==(6,8)
        medians[family]=np.median(curves,axis=0)
        mid=medians[family]
        ax[1].errorbar(widths,mid,yerr=[mid-curves.min(axis=0),curves.max(axis=0)-mid],
                       color=color,fmt='o-',ms=3.,lw=1.3,elinewidth=.75,
                       capsize=2.,capthick=.75,label=label)
    ax[1].set_xscale('log');ax[1].set_yscale('log')
    widths=np.array(widths);ref=medians['QRU'][3]*(widths/widths[3])**6
    line,=ax[1].loglog(widths,ref,color=GRAY,ls='--',lw=1,label=r'$h^6$ slope reference')
    line._print_safe_skip=True
    ax[1].set(xlabel='Observed input width h',ylabel='Local RMS function gap')
    ax[1].set_xticks([.13,.29,.65],['0.13','0.29','0.65'])
    ax[1].xaxis.set_minor_locator(NullLocator());log_axis(ax[1])
    ax[1]._legend_kwargs={'bbox_to_anchor':(.5,-.30),'borderaxespad':0}
    ax[1].legend(frameon=False,loc='upper center',**ax[1]._legend_kwargs)
    panel(ax[1],'(b) Narrow-window masking')
    rr=data['finite_sites'];assert len(rr)==12
    for r in rr:
        y=[r['N for KL=1, broad'],r['N for KL=1, 7 sites']]
        ax[2].plot([0,1],y,color='#cbd1d7',lw=.7,zorder=1)
        ax[2].scatter([0,1],y,c=[TEAL,ORANGE],s=14,zorder=2)
    ax[2].set_yscale('log');log_axis(ax[2])
    ax[2].set(xticks=[0,1],xticklabels=['Broad grid','7-point\nnarrow grid'],
              ylabel=r'Gaussian samples for KL = 1',ylim=(.3,2e14))
    panel(ax[2],'(c) Input diversity')
    save(fig,'fig2_local_observability')


def collision_moments(data):
    fig, ax=plt.subplots(1,3,figsize=(7.1,3.3),layout='constrained')
    c=-np.sqrt(2)/8-1j*np.sqrt(2)/4;m=-np.sqrt(2)/8+0j
    gamma=np.real(m*np.conj(c))/abs(c)**2
    assert np.isclose(np.imag(m*np.conj(c)),-1/16)
    z=np.linspace(-.3,1.2,100)*c
    line,=ax[0].plot(z.real,z.imag,color=GRAY,ls='--',lw=1.1,label=r'Real carrier $\gamma C_1$')
    line._print_safe_skip=True
    for value,color,label,offset in [(c,BLUE,r'$C_1$',(-.06,-.065)),
                                    (m,ORANGE,r'$M_{1,v}$',(-.075,.055))]:
        ax[0].annotate('',xy=(value.real,value.imag),xytext=(0,0),
                       arrowprops={'arrowstyle':'-|>','color':color,'lw':1.6})
        ax[0].text(value.real+offset[0],value.imag+offset[1],label,color=color,fontsize=9,
                   bbox={'facecolor':'white','edgecolor':'none','pad':1.5},zorder=5)
    proj=gamma*c
    ax[0].plot([proj.real,m.real],[proj.imag,m.imag],color=ORANGE,lw=1.1,ls=':')
    ax[0].scatter([proj.real],[proj.imag],s=14,color=GRAY)
    ax[0].axhline(0,color='#d8dfe4',lw=.7);ax[0].axvline(0,color='#d8dfe4',lw=.7)
    ax[0].set(xlabel='Real coefficient',ylabel='Imaginary coefficient',
              xlim=(-.30,.13),ylim=(-.47,.15))
    ax[0]._legend_kwargs={'bbox_to_anchor':(.5,-.28),'borderaxespad':0}
    ax[0].legend(loc='upper center',**ax[0]._legend_kwargs)
    ax[0].set_aspect('equal',adjustable='box')
    panel(ax[0],'(a) Hidden drift moment')
    byseed={s:{r['Schedule']:r for r in data['grouped_transport'] if r['Seed']==s}
            for s in sorted({r['Seed'] for r in data['grouped_transport']})}
    for i,group in enumerate(byseed.values()):
        vals=[group[k]['Grouped Fourier residual RMS'] for k in ('incommensurate','dyadic')]
        ax[1].plot([0,1],vals,color='#ccd3da',lw=.7,zorder=1)
        ax[1].scatter([0,1],vals,c=[GRAY,ORANGE],s=14,zorder=2)
        for j,k in enumerate(('incommensurate','dyadic')):
            ax[1].scatter(j+.10,group[k]['Path-resolved derivative RMS error'],
                          color=TEAL,marker='x',s=15,lw=.8,
                          label='Matched path Fourier' if i==0 and j==0 else None,zorder=4)
    ax[1].set_yscale('log');log_axis(ax[1])
    ax[1].set(xticks=[0,1],xticklabels=['Distinct tags','Collided tags'],
              ylabel='Drift-derivative RMS mismatch',ylim=(1e-16,2))
    ax[1]._legend_kwargs={'bbox_to_anchor':(.5,-.28),'borderaxespad':0}
    ax[1].legend(loc='upper center',frameon=False,**ax[1]._legend_kwargs)
    panel(ax[1],'(b) Grouped versus paths')
    twins=data['regular_twins']
    vals=[r['future_risk_floor'] for r in twins]
    baseline=[r['current_rms'] for r in twins]
    ax[2].scatter(baseline,vals,color=BLUE,s=25,edgecolors='white',lw=.4,zorder=3)
    ax[2].set_xscale('log');ax[2].set_yscale('log');log_axis(ax[2]);log_axis(ax[2],'x')
    ax[2].set(xlabel='Present RMS branch gap',ylabel=r'Future ambiguity scale $\Delta_Q^2/4$')
    ax[2].set_xlim(8e-16,2e-13);ax[2].set_xticks([1e-15,1e-14,1e-13])
    panel(ax[2],'(c) Future ambiguity')
    save(fig,'fig3_collision_moments')


def phase_risk(shots):
    """Exact equal-prior posterior squared-risk fraction for Bernoulli twins.

    For k positive outcomes, likelihoods L_A,L_B imply p=L_A/(L_A+L_B).
    E[p(1-p)] / (1/4) = 2 sum_k Binom(n,k) L_A L_B/(L_A+L_B).
    This is optimal Bayes prediction risk, not MAP misclassification risk.
    """
    p=(1-1/np.sqrt(2))/2;q=1-p
    out=[]
    for n in shots:
        k=np.arange(n+1)
        a=binom.pmf(k,n,p);b=binom.pmf(k,n,q)
        out.append(float(np.sum(2*a*b/(a+b))))
    assert np.isclose(out[0],1.)
    return np.array(out)


def phase_information(data):
    fig, ax=plt.subplots(1,3,figsize=(7.1,3.3),layout='constrained')
    phi=np.linspace(0,np.pi,201);x0=np.pi/4
    ax[0].plot(phi,np.cos(x0+phi),color=BLUE,label='Target A')
    ax[0].plot(phi,np.full(len(phi),np.cos(x0)),color=ORANGE,ls='--',label='Target B')
    ax[0].axvline(np.pi/2,color=GRAY,ls=':',lw=.9)
    ax[0].scatter([np.pi/2]*2,[-1/np.sqrt(2),1/np.sqrt(2)],c=[BLUE,ORANGE],s=22,zorder=4)
    ax[0].set(xlabel=r'Target second-bias phase $\phi_2$',ylabel=r'Probe mean $\langle Z\rangle$',
              xticks=[0,np.pi/2,np.pi],xticklabels=['0',r'$\pi/2$',r'$\pi$'],ylim=(-1.12,1.12))
    ax[0].legend(loc='lower left',frameon=False);panel(ax[0],r'(a) Target phase response')
    n=np.arange(0,13);risk=phase_risk(n)
    assert np.isclose(risk[1],.5)
    ax[1].plot(n,np.ones(len(n)),color=GRAY,ls='--',label='Passive / blind phase')
    ax[1].semilogy(n,risk,color=TEAL,marker='o',ms=3,label='Informative target phase')
    ax[1].set(xlabel='Number of target shots',ylabel=r'Optimal Bayes risk / $R_0$',
              xticks=[0,4,8,12],ylim=(.003,1.4))
    log_axis(ax[1])
    ax[1]._legend_kwargs={'bbox_to_anchor':(.5,-.28),'borderaxespad':0}
    ax[1].legend(loc='upper center',frameon=False,**ax[1]._legend_kwargs)
    panel(ax[1],'(b) Exact prediction risk')
    order=['passive','blind_phase','phase'];labs=['Passive','Blind\nphase','Target\nphase']
    rng=np.random.default_rng(81003)
    for i,(policy,color) in enumerate(zip(order,[GRAY,PURPLE,TEAL])):
        rr=[r for r in data['stream_trials'] if r['scenario']=='abrupt_planned' and r['policy']==policy]
        vals=np.array([r['branch_probability_before'] for r in sorted(rr,key=lambda r:r['seed'])])
        assert len(vals)==32
        ax[2].scatter(i+rng.uniform(-.13,.13,32),vals,color=color,alpha=.55,s=12,lw=0)
        ax[2].scatter(i,vals.mean(),marker='_',s=170,color='#25313b',lw=2,zorder=4)
    ax[2].axhline(.5,color=GRAY,ls=':',lw=.9)
    ax[2].set(xticks=[0,1,2],xticklabels=labs,ylabel='Correct-branch posterior',ylim=(-.035,1.06))
    panel(ax[2],'(c) Hidden-branch inference')
    save(fig,'fig4_phase_information')


LABELS={'abrupt_planned':'Announced transverse', 'abrupt_unannounced':'Unannounced transverse',
        'gradual':'Gradual transverse','recurring':'Recurring transverse',
        'parallel':'Collision-preserving drift','stationary':'Stationary target',
        'irrelevant_inputs':'Future input x = 0','initially_separated':'Already separated tones',
        'no_phase_access':'No target phase access','latent_reset':'Latent branch swaps'}
ORDER=list(LABELS)


def forest(ax, rows, interval, title, labels=True):
    yy=np.arange(len(ORDER))[::-1]
    for y,scenario in zip(yy,ORDER):
        r=next(r for r in rows if r['scenario']==scenario)
        prefix='gain' if interval=='early' else 'boundary_gain'
        gain=r['paired_gain'] if interval=='early' else r['boundary_gain']
        low,high=r[prefix+'_low'],r[prefix+'_high']
        color=TEAL if low>0 else (RED if high<0 else GRAY)
        ax.plot([low,high],[y,y],color=color,lw=1.8)
        ax.scatter(gain,y,color=color,s=20,zorder=3)
    ax.axvline(0,color='#3b4651',lw=.8)
    ax.set_yticks(yy,[LABELS[s] for s in ORDER] if labels else ['']*len(ORDER))
    ax.tick_params(axis='y',length=0,pad=7)
    ax.set_xlabel('MSE gain (passive - target phase)')
    panel(ax,title);ax.grid(axis='y',visible=False);ax.grid(axis='x',color='#e7ebee',lw=.6)
    ax.set_ylim(-.6,9.6)


def streams(data):
    fig=plt.figure(figsize=(7.1,5.65),layout='constrained')
    gs=fig.add_gridspec(2,2,height_ratios=[1.,1.65])
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1])
    for ax,scenario,title in [(a,'abrupt_planned','(a) Announced future control'),
                              (b,'abrupt_unannounced','(b) Unannounced change')]:
        ax.axvspan(70,90,color='#e8ecef',alpha=.9,zorder=0)
        for policy,color,ls,label in [('passive',GRAY,'-','Passive'),('blind_phase',PURPLE,':','Blind phase'),
                                      ('phase',TEAL,'-','Target phase')]:
            rr=sorted([r for r in data['stream_curves'] if r['scenario']==scenario and r['policy']==policy],
                       key=lambda r:r['t'])
            rr=[r for r in rr if 35<=r['t']<=145]
            ax.plot([r['t']+2 for r in rr],[r['mean_future_mse'] for r in rr],
                    color=color,ls=ls,label=label,lw=1.4)
        ax.set(xlabel='Forecast origin (5-origin averages)',ylabel='Mean five-step prediction MSE',
               xlim=(35,147),ylim=(0,.029),xticks=[40,80,120])
        ax.text(.5,1.018,'Change times 70-90 (shaded)',transform=ax.transAxes,
                ha='center',va='bottom',color=GRAY,fontsize=6.5,clip_on=False)
        panel(ax,title)
        ax.set_title(title,loc='left',fontweight='semibold',pad=24)
    a._legend_kwargs={'bbox_to_anchor':(.5,-.36),'borderaxespad':0}
    a.legend(loc='upper center',ncol=3,frameon=False,**a._legend_kwargs)
    c=fig.add_subplot(gs[1,0]);d=fig.add_subplot(gs[1,1])
    rows=[r for r in data['stream_summary'] if r['policy']=='phase']
    forest(c,rows,'early','(c) Early after observed onset')
    forest(d,rows,'boundary','(d) Forecast crosses the change',labels=False)
    c.set_xlim(-.067,.016);c.set_xticks([-.06,-.03,0])
    d.set_xlim(-.0085,.016);d.set_xticks([-.005,0,.005,.010,.015])
    save(fig,'fig5_stream_prediction_risk')


def validate(data):
    assert len(data['regular_twins'])==12 and len(data['finite_sites'])==12
    assert max(r['Six-site max error'] for r in data['finite_sites'])<4e-13
    for r in data['finite_sites']:
        assert np.isclose(r['N for KL=1, broad']*r['Future MSE gap Q'],2*.02**2)
        assert np.isclose(r['N for KL=1, 7 sites']*r['Seventh-input MSE gap'],2*.02**2)
    trials=data['stream_trials']
    for r in data['stream_summary']:
        rr=[x for x in trials if x['policy']==r['policy'] and x['scenario']==r['scenario']]
        assert len(rr)==32 and {x['shots'] for x in rr}=={720}
        for key in ['early_future_mse','boundary_future_mse','branch_probability_before']:
            assert np.isclose(np.mean([x[key] for x in rr]),r[key],atol=1e-14)
        if r['policy']=='phase':
            ref={x['seed']:x for x in trials if x['policy']=='passive' and x['scenario']==r['scenario']}
            for key,out in [('early_future_mse','paired_gain'),('boundary_future_mse','boundary_gain')]:
                assert np.isclose(np.mean([ref[x['seed']][key]-x[key] for x in rr]),r[out],atol=1e-14)
    assert data['stream_checks']['full_manifold_max_error_L1_to_L4']<1e-12


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    args=parser.parse_args()
    data=main_data()
    assert np.isclose(data['illustration']['delta'], ILLUSTRATIVE_DELTA)
    assert np.isclose(data['illustration']['future_risk_lower_bound'], data['illustration']['future_mse']/4)
    validate(data);style()
    for function in (mechanism,local_observability,collision_moments,phase_information,streams):function(data)
    from build_research_roadmap import main as research_roadmap
    research_roadmap()
    print('Verified archived rows and rebuilt six main figures (vector PDF + 300 dpi PNG).')
