"""Plot exact passive ambiguity and phase information under known dephasing."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qru.noise import density_output, dephasing_twins
from figure_style import BLUE, TEAL, GRAY, style, panel, save
from build_main_figures import pi_axis


def main():
    style()
    x = np.linspace(0, 2*np.pi, 513); gamma = .7
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.85), layout='constrained')
    a, b, c = axes
    fixed = dephasing_twins(1.)
    gap = density_output(x, fixed['A'], [1, 2], gamma)-density_output(x, fixed['B'], [1, 2], gamma)
    np.testing.assert_allclose(gap, (1-gamma)*np.cos(x)*np.cos(2*x), atol=1e-13, rtol=0)
    a.plot(x, gap, color=BLUE)
    a.axhline(0, color=GRAY, lw=.7)
    a.set(ylabel=r'Passive gap $f_A-f_B$'); pi_axis(a)
    panel(a, '(a) Passive gap')
    calibrated = dephasing_twins(gamma)
    baseline = density_output(x, calibrated['A'], [1, 2], gamma)-density_output(x, calibrated['B'], [1, 2], gamma)
    future = density_output(x, calibrated['A'], [1, 2.05], gamma)-density_output(x, calibrated['B'], [1, 2.05], gamma)
    b.plot(x, baseline, color=GRAY, label='Present')
    b.plot(x, future, color=TEAL, label='Future')
    b.set(ylabel=r'Calibrated-pair gap'); pi_axis(b)
    b.legend(loc='upper left'); panel(b, '(b) Calibrated pair')
    gamma_grid = np.linspace(0, 1, 101)
    z = gamma_grid/np.sqrt(2)
    kl = z*np.log((1+z)/(1-z))
    c.plot(gamma_grid, kl, color=TEAL)
    c.set(xlabel=r'Dephasing factor $\gamma$', ylabel='Phase-shot KL (nats)', xlim=(0, 1), ylim=(0, 1.35))
    panel(c, '(c) Phase information')
    save(fig, 'appendix_C3_dephasing')


if __name__ == '__main__':
    main()
