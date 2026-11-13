"""Reproduce attainable-fiber and calibrated-dephasing calculations."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qru.path_geometry import depth_two_amplitudes, sinusoidal_fiber, fiber_risk
from qru.noise import density_output, dephasing_twins
from qru.tomography import analytic_path_amplitudes


def experiment():
    rng = np.random.default_rng(120261113)
    errors = []; constraints = []; density_errors = []
    for index in range(10000):
        theta = rng.uniform(-np.pi, np.pi, (2, 3))
        a = depth_two_amplitudes(theta)
        ref = analytic_path_amplitudes(theta)
        errors.append(max(abs(a[s]-ref[s]) for s in a))
        p, q, r = abs(a[(1, 1)]), abs(a[(1, -1)]), abs(a[(1, 0)])
        constraints.append(abs((p+q)**2*(r*r+4*p*q)-p*q))
        if index < 100:
            alpha = rng.uniform(.2, 3, 2); x = rng.uniform(-3, 3, 17)
            path = sum(a[s]*np.exp(1j*np.dot(s, alpha)*x) for s in a)
            density_errors.append(float(max(abs(path.real-density_output(x, theta, alpha)))))
    x = np.linspace(0, 2*np.pi, 513)
    cases = []
    for r, phase in [(.2, -.7), (.65, .3), (1., 0.)]:
        circuits = sinusoidal_fiber(r, phase)
        gap = max(float(max(abs(density_output(x, t, [1, 2])-r*np.cos(x+phase)))) for t in circuits.values())
        cases.append(dict(radius=r, phase=phase, baseline_max_gap=gap,
                          future_schedule=[1., 2.05], minimax_risk=fiber_risk(r, phase, [1., 2.05], x)))
    noise = []; noise_errors = []
    for gamma in np.linspace(0, 1, 11):
        circuits = dephasing_twins(gamma)
        for _ in range(15):
            alpha = rng.uniform(.2, 3, 2)
            for branch, target in [('A', gamma*np.cos((alpha[1]-alpha[0])*x)), ('B', gamma*np.cos(alpha[0]*x))]:
                noise_errors.append(float(max(abs(density_output(x, circuits[branch], alpha, gamma)-target))))
        a = density_output(x, circuits['A'], [1, 2], gamma)
        b = density_output(x, circuits['B'], [1, 2], gamma)
        phase_means = {}
        for branch, theta in circuits.items():
            shifted = theta.copy(); shifted[1, 2] += np.pi/2
            phase_means[branch] = float(density_output([np.pi/4], shifted, [1, 2], gamma)[0])
            noise_errors.append(abs(phase_means[branch]-(-1 if branch == 'A' else 1)*gamma/np.sqrt(2)))
        future_gap = density_output(x, circuits['A'], [1, 2.05], gamma)-density_output(x, circuits['B'], [1, 2.05], gamma)
        c = gamma/np.sqrt(2)
        noise.append(dict(gamma=float(gamma), theta_a=circuits['A'].tolist(), theta_b=circuits['B'].tolist(),
                          baseline_max_gap=float(max(abs(a-b))), phase_means=phase_means,
                          phase_shot_kl=float(c*np.log((1+c)/(1-c))),
                          pair_minimax_risk=float(np.mean(future_gap**2)/4)))
    return dict(protocol={'random_circuits':10000, 'random_seed':120261113,
                          'future_grid':'513 equally weighted inputs in [0,2pi]',
                          'channel':'Identical Z dephasing after layer one, gamma fixed and known',
                          'fiber_scope':'Complete exact sinusoidal response, depth two, baseline (1,2)',
                          'risk_scope':'Unrestricted mean-response prediction under a fixed future schedule'},
                checks={'amplitude_max_error':float(max(errors)),
                        'magnitude_identity_max_residual':float(max(constraints)),
                        'density_reconstruction_max_error':float(max(density_errors)),
                        'dephasing_identity_max_error':float(max(noise_errors))},
                sinusoidal_fibers=cases, dephasing_twins=noise)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = experiment()
    if max(data['checks'].values()) > 1e-12:
        raise AssertionError(data['checks'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps(data['checks'], indent=2))
