"""Rebuild nine appendix figures from frozen results and raw hardware counts."""
import argparse
import json
from pathlib import Path

from data_io import main_data, supporting_data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

from figure_style import BLUE, ORANGE, TEAL, PURPLE, GRAY, RED, style, save as common_save


def save(fig, dest, name):
    common_save(fig, Path(name).stem)


def forest(ax, rows, title, *, interval='early'):
    from build_main_figures import forest as draw
    draw(ax,rows,interval,title)


def qpu_differences(root):
    """Predeclared 95% simultaneous Clopper-Pearson rate intervals per run.

    The original stage gate budgets .05/(2 tails * 2 branches * 22 planned
    experimental pairs) for each exact binomial tail, including unrun stages.
    Coverage is per acquisition, and no cross-run 95% coverage is claimed.
    """
    out = []
    for fn in ('qpu_baseline.json', 'qpu_baseline_rep2.json'):
        p = root.parent / 'results/hardware' / fn
        run = json.loads(p.read_text())
        assert run['stage'] == 'baseline' and run['source'] == 'ibm_quantum_qpu'
        assert len(run['rows']) == 12 and run['shots'] == 8192
        xs = sorted({r['x'] for r in run['rows']})
        assert len(xs) == 6
        one = []
        for x in xs:
            a, b = [next(r for r in run['rows'] if r['x'] == x and r['branch'] == branch)
                    for branch in 'AB']
            assert abs(a['ideal_mean']-b['ideal_mean']) < 1e-12
            def interval(row):
                zero, one_count = (row['counts'][key] for key in ('0', '1'))
                n = zero + one_count
                assert n == run['shots'] == len(row['bitstrings'])
                assert zero == sum(bit == '0' for bit in row['bitstrings'])
                tail = .05 / (2 * 2 * 22)
                low = beta.ppf(tail, zero, one_count + 1) if zero else 0.
                high = beta.ppf(1-tail, zero + 1, one_count) if one_count else 1.
                return (zero-one_count)/n, 2*low-1, 2*high-1
            mean_a, lo_a, hi_a = interval(a)
            mean_b, lo_b, hi_b = interval(b)
            one.append({'x': x, 'd': mean_a-mean_b, 'low': lo_a-hi_b,
                        'high': hi_a-lo_b, 'a': mean_a, 'b': mean_b,
                        'counts_a': a['counts'], 'counts_b': b['counts']})
        out.append({'rows': one, 'job_id': run['job_id'],
                    'witness_sha256': run['witness_sha256'],
                    'backend': run['backend'], 'qubit': run['physical_qubit']})
    assert out[0]['job_id'] != out[1]['job_id']
    assert out[0]['witness_sha256'] == out[1]['witness_sha256']
    for run, expected in zip(out, [(-.109130859375, -.1793),(-.08984375,-.1604)]):
        assert abs(run['rows'][3]['d']-expected[0]) < 1e-10
        assert abs(run['rows'][3]['low']-expected[1]) < 5e-5
    return out


def main():
    dest = Path(__file__).resolve().parent / 'figures'
    source = Path(__file__).resolve().parents[1] / 'data'
    md = main_data()
    sd = supporting_data()
    pred = {'controlled_summary':md['stream_summary'],'real_summary':sd['real_summary']}
    det,pairs,arch,budget = md['finite_sites'],md['finite_pairs'],md['architecture'],sd['budget']
    summary=pred['controlled_summary'];style()

    # Appendix I: complete raw-count baseline experiment, per-run intervals.
    qpu = qpu_differences(Path(__file__).resolve().parent)
    fig, ax = plt.subplots(figsize=(6.4, 2.7), constrained_layout=True)
    ax.axhspan(-.12, .12, color='#e4e9ed', alpha=.8, label='Equivalence band')
    ax.axhline(0, color='0.32', lw=.8)
    for i, run in enumerate(qpu):
        rr = run['rows']; xx = np.arange(len(rr)) + (-.12 if i == 0 else .12)
        yy = [r['d'] for r in rr]
        errs = [[r['d']-r['low'] for r in rr],
                [r['high']-r['d'] for r in rr]]
        ax.errorbar(xx, yy, yerr=errs, fmt=['o','s'][i], markersize=3, capsize=2,
                    lw=1, color=[BLUE, ORANGE][i], label=f'Acquisition {i+1}')
    ax.set_xticks(range(6), ['0', r'$\pi/4$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax.set_xlabel('Baseline input x')
    ax.set_ylabel(r'$\widehat{Z}_A-\widehat{Z}_B$')
    ax.set_ylim(-.215, .18)
    ax.legend(fontsize=7, ncol=3, loc='upper center')
    save(fig, dest, 'appendix_I1_qpu_baseline.pdf')

    # Appendix C: exact controlled phase experiment as functions of input.
    x = np.linspace(0, 2*np.pi, 401)
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.55), constrained_layout=True)
    ax[0].plot(x, np.cos(x),color=GRAY, label='A = B: passive')
    ax[0].plot(x, -np.sin(x),color=BLUE, label='A: target phase')
    ax[0].plot(x, np.cos(x), '--',color=ORANGE, label='B: target phase')
    ax[0].scatter([np.pi/4]*2, [-2**-.5, 2**-.5], color=[TEAL,ORANGE])
    ax[0].set_title('(a) Passive and phase responses')
    delta=md.get('illustration',md['exact'])['delta']
    ax[1].plot(x, np.cos((1+delta)*x),color=BLUE, label='A: transverse drift')
    ax[1].plot(x, np.cos(x), '--',color=ORANGE, label='B: transverse drift')
    ax[1].set_title(rf'(b) Future schedule $(1,{2+delta:.2f})$')
    for a in ax:
        a.set_xlabel('Input x'); a.set_ylabel(r'$\langle Z\rangle$'); a.legend(fontsize=7)
    save(fig, dest, 'appendix_C1_path_tomography.pdf')

    # Appendix C2: original V2 phase-code design, displaying rank rather than
    # historically discrepant row-level aliasing reconstruction errors.
    designs=sd['designs'];alias=sd['alias']
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.45), constrained_layout=True)
    for i, r in enumerate(designs):
        ax[0].plot([i, i], [r['max_collision_multiplicity'],r['min_phase_settings']],
                   color='0.7',lw=2)
        ax[0].scatter(i,r['max_collision_multiplicity'],c=GRAY,s=23,
                      label='Largest collision class' if not i else None)
        ax[0].scatter(i,r['min_phase_settings'],c=BLUE,s=23,marker='s',
                      label='Fixed-code phase settings' if not i else None)
    ax[0].set_xticks(range(3),['Harmonic L3','Dyadic L3','Dyadic L4'],rotation=15)
    ax[0].set_ylabel('Number of path labels / settings')
    ax[0].set_title('(a) Phase-code cost')
    ax[0].legend(fontsize=6.8)
    ax[1].plot([r['phase_settings'] for r in alias],
               [r['alias_classes'] for r in alias], 'o-',color=BLUE,markersize=3)
    ax[1].set_xticks([1,2,3,4,5,6,7,8]); ax[1].set_ylim(-.4,6.6)
    ax[1].set_xlabel('Phase settings, dyadic L3')
    ax[1].set_ylabel('Unresolved alias classes')
    ax[1].set_title('(b) Unresolved phase-code aliases')
    save(fig, dest, 'appendix_C2_phase_code_rank.pdf')

    # Appendix D: all 6 QRU and 6 free-amplitude Fourier jet-matched pairs.
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.7), constrained_layout=True)
    for kind, color, name in [('QRU', BLUE, 'QRU'),
                              ('classical_four_tone', ORANGE, 'Six-amplitude Fourier')]:
        by_sample = {}
        for r in pairs:
            if r['family'] == kind:
                by_sample.setdefault(r['sample'], []).append(r)
        assert len(by_sample) == 6 and all(len(x) == 8 for x in by_sample.values())
        widths = np.array(sorted({r['width'] for r in pairs if r['family']==kind}))
        curves = np.array([[next(r['local_rms'] for r in by_sample[s]
                                  if r['width']==h) for h in widths]
                           for s in sorted(by_sample)])
        for curve in curves:
            ax[0].loglog(widths, curve, color=color, alpha=.18, lw=.75)
        median = np.median(curves, axis=0)
        slopes = [np.polyfit(np.log(widths), np.log(curve), 1)[0]
                  for curve in curves]
        ax[0].loglog(widths, median, 'o-', color=color, ms=3, lw=1.5,
                     label=f'{name} (median slope {np.median(slopes):.2f})')
        gaps = [by_sample[s][0]['wide_rms'] for s in sorted(by_sample)]
        xx = (0 if kind == 'QRU' else 1)
        ax[1].scatter(xx+np.linspace(-.10,.10,6),gaps,color=color,s=17)
        ax[1].scatter(xx, np.median(gaps),color='black',marker='_',s=115,zorder=3)
    ax[0].set_title('(a) Narrow-window branch gap')
    ax[0].set_xlabel('Window width h'); ax[0].set_ylabel('Local RMS branch gap')
    ax[0].set_xticks([.13,.29,.65],['0.13','0.29','0.65'])
    ax[0].minorticks_off()
    ax[0].legend(fontsize=6.7,loc='upper left')
    ax[1].set_xticks([0, 1], ['QRU', 'Fourier'])
    ax[1].set_ylabel('RMS gap on [0, 2π]')
    ax[1].set_title('(b) Broad-input separation')
    save(fig, dest, 'appendix_D1_information_scaling.pdf')

    # Appendix D2: classical control for the six-site example; the hypotheses
    # are constructed in both families, not sampled from an ambient population.
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.65), constrained_layout=True)
    for i, key in enumerate(['Future MSE gap Q','Classical future MSE gap']):
        vals = [r[key] for r in det]
        xx = i + np.linspace(-.13,.13,len(vals))
        ax[0].scatter(xx,vals,s=16,color=[BLUE,ORANGE][i])
        ax[0].scatter(i,np.median(vals),marker='_',s=125,color='black')
    ax[0].set_yscale('log')
    ax[0].set_xticks([0,1], ['12 QRU pairs','12 Fourier pairs'])
    ax[0].set_ylabel('Broad-input squared branch gap')
    ax[0].set_title('(a) Matched six-site pairs')
    for i, key in enumerate(['N for KL=1, theta fixed','N for KL=1, broad',
                             'N for KL=1, 7 sites']):
        vals=[r[key] for r in det]
        ax[1].scatter(i+np.linspace(-.13,.13,len(vals)),vals,s=15,
                      color=[GRAY,BLUE,ORANGE][i])
        ax[1].scatter(i,np.median(vals),marker='_',s=110,color='black')
    ax[1].set_yscale('log')
    ax[1].set_xticks(range(3),['Fixed θ', 'Broad x','7-point narrow grid'])
    ax[1].set_ylabel('Additional samples for pairwise KL = 1')
    ax[1].set_title('(b) Pairwise information cost')
    save(fig, dest, 'appendix_D2_finite_site_controls.pdf')

    # Appendix E: local Taylor order differs from collision drift order q.
    fig, ax = plt.subplots(figsize=(6.4, 2.55), constrained_layout=True)
    for schedule, nuisance, lab in [('dyadic', 'all', 'QRU, all rotations'),
                                     ('dyadic', 'bias_only', 'QRU, biases only'),
                                     ('classical_distinct', 'fourier_amplitudes',
                                      'Classical, matched amplitudes')]:
        pts = [r for r in arch if r['Architecture / schedule'] == schedule
               and r['Trainable theta'] == nuisance]
        ax.plot([r['Depth L'] for r in pts], [r['First visible order'] for r in pts],
                'o-', label=lab)
    ax.set_xticks([1, 2, 3, 4]);ax.set_xlabel('Depth L')
    ax.set_ylabel('First visible input Taylor order m');ax.legend(fontsize=8)
    save(fig, dest, 'appendix_E1_observability_orders.pdf')
    # Appendix F: use archived budget outcomes; no claim about universally tuned optimizers.
    fig, ax = plt.subplots(figsize=(6.4, 2.65), constrained_layout=True)
    methods = sorted({r['method'] for r in budget if r['experiment'] == 'local_shift'
                      and r['case'] == 'mixed'})
    for method in methods:
        pts = sorted([r for r in budget if r['experiment'] == 'local_shift'
                      and r['case'] == 'mixed' and r['method'] == method],
                     key=lambda r: r['median_calls_used'])
        if pts:
            ax.plot([r['median_calls_used'] for r in pts],
                    [r['median_test_mse'] for r in pts], 'o-', label={'adam_joint_regularized':'Adam: joint + penalty','adam_theta':'Adam: theta only','lm_joint':'LM: joint','lm_profiled':'LM: profiled','lm_regularized_joint':'LM: joint + penalty'}.get(method,method), ms=3)
    ax.set_xscale('log');ax.set_yscale('log')
    ax.set_xlabel('Median forward calls used');ax.set_ylabel('Median test MSE')
    ax.legend(fontsize=7, ncol=2)
    save(fig, dest, 'appendix_F2_adaptation_falsification.pdf')

    fig, ax = plt.subplots(1, 2, figsize=(6.8, 4.45), sharey=True,
                           constrained_layout=True)
    all_rows = [r for r in summary if r['policy'] == 'phase' and r['horizon'] == 5]
    forest(ax[0], all_rows, '(a) Early after observed onset')
    forest(ax[1], all_rows, '(b) Crossing the boundary', interval='boundary')
    ax[1].tick_params(axis='y',labelleft=False)
    save(fig, dest, 'appendix_G1_causal_robustness.pdf')

    real = pred['real_summary']
    fig, ax = plt.subplots(figsize=(6.4, 2.8), constrained_layout=True)
    datasets = [('sunspots', 1), ('sunspots', 5), ('nile', 1), ('nile', 5)]
    for i, method, color, symbol, label in [(0,'qru',BLUE,'o','QRU = full-manifold Fourier'),
                                            (1,'ar1',ORANGE,'s','AR(1)'),
                                            (2,'persistence',GRAY,'^','Persistence')]:
        values=[]
        for d,h in datasets:
            entries=[r['normalized_mse'] for r in real if
                     (r['dataset'],r['horizon'],r['method'])==(d,h,method)]
            assert len(entries)==3
            values.append(np.mean(entries))
        xx=np.arange(4)+[-.18,0,.18][i]
        ax.scatter(xx,values,marker=symbol,color=color,s=42,label=label,zorder=3)
        if method=='qru':
            for j,(d,h) in enumerate(datasets):
                vals=[r['normalized_mse'] for r in real if
                      (r['dataset'],r['horizon'],r['method'])==(d,h,'qru')]
                ax.scatter([xx[j]]*3,vals,marker='|',color=color,alpha=.6,s=70)
    ax.set_yscale('log')
    ax.set_xticks(range(4), ['Sunspots H1', 'Sunspots H5', 'Nile H1', 'Nile H5'])
    ax.set_ylabel('Mean normalized MSE, three starts (log scale)')
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.grid(axis='y', color='.9', lw=.5)
    save(fig, dest, 'appendix_H1_real_series.pdf')
    print('Rebuilt nine historical appendix figures from frozen records and raw hardware counts.')


if __name__ == '__main__':
    main()
