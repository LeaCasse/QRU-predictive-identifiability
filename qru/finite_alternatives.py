"""Finite-change observability control and gradual spectral tracking.

Exact noiseless QRU means, Gaussian observation labels in the tracking stream;
no shot-limited circuit gradients, real forecasting, or universal rank proof.
"""
from math import factorial

import numpy as np
from scipy.linalg import qr
from scipy.optimize import root

from .circuit import qru_output
from .tomography import analytic_path_amplitudes
from .pilot_geometry import ALPHA, reference
from .local_stream import _step


def x_jets(theta, alpha, center, order=6):
    """Exact derivatives divided by factorial, from Bloch path amplitudes."""
    paths = analytic_path_amplitudes(theta)
    tags = np.array(list(paths), dtype=float)
    coeff = np.array(list(paths.values()), dtype=complex)
    freq = tags @ alpha
    return np.array([
        np.real(np.sum(coeff * (1j * freq) ** k * np.exp(1j * freq * center))) / factorial(k)
        for k in range(order)
    ])


def theta_jet_jacobian(theta, alpha, center):
    """The six-by-nine derivative matrix, using exact parameter shifts."""
    flat = theta.ravel()
    columns = []
    for j in range(len(flat)):
        plus = flat.copy(); minus = flat.copy()
        plus[j] += np.pi / 2; minus[j] -= np.pi / 2
        columns.append((x_jets(plus.reshape(3, 3), alpha, center)
                        - x_jets(minus.reshape(3, 3), alpha, center)) / 2)
    return np.stack(columns, axis=1)


def classical_jets(beta, frequencies, center, order=6):
    """A four-tone classical Fourier model with eight free amplitudes."""
    result = np.zeros(order)
    for j, omega in enumerate(frequencies):
        for k in range(order):
            phase = omega * center + k * np.pi / 2
            result[k] += omega ** k * (beta[2*j] * np.cos(phase)
                                      + beta[2*j+1] * np.sin(phase)) / factorial(k)
    return result


def classical_output(x, frequencies, beta):
    phase = np.outer(x, frequencies)
    return np.cos(phase) @ beta[::2] + np.sin(phase) @ beta[1::2]


def finite_pairs(seed=8712, n=6, center=np.pi, widths=(.13, .17, .22, .29, .38, .5, .65, .85)):
    """Actual QRU pairs and a capacity-matched classical six-DOF control.

    Six classical coefficient coordinates change; the remaining two stay fixed.
    Matching six jets for both families is expected for regular analytic models.
    The chosen width range is a numerical diagnostic, not a theorem of sharpness.
    """
    rng = np.random.default_rng(seed)
    probe = np.linspace(0, 2*np.pi, 257)
    rows = []
    for sample in range(n):
        theta = reference(rng)
        direction = rng.normal(size=3); direction /= np.linalg.norm(direction)
        moved_alpha = ALPHA + .035 * direction
        initial = theta_jet_jacobian(theta, ALPHA, center)
        _, _, pivot = qr(initial, pivoting=True)
        cols = pivot[:6]
        target = x_jets(theta, ALPHA, center)

        def trial(v):
            flat = theta.ravel().copy(); flat[cols] = v
            return flat.reshape(3, 3)

        solution = root(lambda v: x_jets(trial(v), moved_alpha, center) - target,
                        theta.ravel()[cols],
                        jac=lambda v: theta_jet_jacobian(trial(v), moved_alpha, center)[:, cols],
                        tol=1e-11)
        theta_new = trial(solution.x)
        mismatch = float(np.linalg.norm(x_jets(theta_new, moved_alpha, center)-target))
        wide_gap = float(np.sqrt(np.mean((qru_output(probe, theta, ALPHA)
                                        - qru_output(probe, theta_new, moved_alpha))**2)))
        for h in widths:
            x = center + h*np.linspace(-.5,.5,128)
            delta = qru_output(x,theta,ALPHA)-qru_output(x,theta_new,moved_alpha)
            rows.append(dict(family='QRU',sample=sample,center=float(center),width=float(h),
                             local_rms=float(np.sqrt(np.mean(delta**2))),wide_rms=wide_gap,
                             jet_residual=mismatch,solver_success=bool(solution.success),
                             theta_jet_condition=float(np.linalg.cond(initial[:,cols])),
                             alpha_shift=.035))

        # Classical control: same six adjustable amplitude coordinates and
        # independently displaced carrier frequencies, not a QRU path model.
        frequencies=np.array([1.,3.,5.,7.])
        moved_freq=frequencies+.035*np.array([1.,-.3,.4,-.8])
        beta=rng.normal(0,.12,8)
        J=np.stack([classical_jets(np.eye(8)[j], moved_freq, center) for j in range(8)],axis=1)
        _,_,ix=qr(J,pivoting=True);ix=ix[:6]
        beta_new=beta.copy()
        beta_new[ix]=np.linalg.solve(J[:,ix], classical_jets(beta,frequencies,center)-J@beta+J[:,ix]@beta[ix])
        wide_gap=float(np.sqrt(np.mean((classical_output(probe,frequencies,beta)
                                        -classical_output(probe,moved_freq,beta_new))**2)))
        mismatch=float(np.linalg.norm(classical_jets(beta,frequencies,center)
                                      -classical_jets(beta_new,moved_freq,center)))
        for h in widths:
            x=center+h*np.linspace(-.5,.5,128)
            delta=classical_output(x,frequencies,beta)-classical_output(x,moved_freq,beta_new)
            rows.append(dict(family='classical_four_tone',sample=sample,center=float(center),width=float(h),
                             local_rms=float(np.sqrt(np.mean(delta**2))),wide_rms=wide_gap,
                             jet_residual=mismatch,solver_success=True,
                             theta_jet_condition=float(np.linalg.cond(J[:,ix])),alpha_shift=.035))
    return rows


def gradual_stream(seed, velocity, coverage, T=30, W=24):
    """Predict-before-label tracking with a fixed theta teacher and moving alpha.

    Synthetic scalar-input regression only. Broad-grid loss is an offline
    diagnostic; it never enters any update or model selection.
    """
    rng=np.random.default_rng(seed);theta0=reference(rng)
    direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
    grid=np.linspace(0,2*np.pi,256,endpoint=False)
    windows=[]
    for t in range(T):
        shift=min(max(t-5,0)*velocity,.65)
        truth=ALPHA+direction*shift
        x=(rng.uniform(0,2*np.pi,W) if coverage=='broad'
           else np.pi+rng.uniform(-.25,.25,W))
        clean=qru_output(x,theta0,truth)
        y=clean+rng.normal(0,.02,W)
        windows.append((x,y,clean,qru_output(grid,theta0,truth),shift))
    methods=[('theta_current',1,0.,True),('joint_current',1,.01,False),
             ('joint_current_unpenalized',1,0.,False),('joint_recent2',2,.01,False)]
    rows=[]
    for name,memory,penalty,theta_only in methods:
        theta=theta0.copy();alpha=ALPHA.copy();past=[]
        for t,(x,y,clean,target,shift) in enumerate(windows):
            pre=qru_output(x,theta,alpha)
            wide=qru_output(grid,theta,alpha)
            past.append((x,y)); recent=past[-memory:]
            train_x=np.concatenate([r[0] for r in recent]);train_y=np.concatenate([r[1] for r in recent])
            theta,alpha,calls,_,_=_step(theta,alpha,train_x,train_y,budget=2100,
                                       alpha_penalty=penalty,theta_only=theta_only)
            rows.append(dict(seed=seed,velocity=float(velocity),coverage=coverage,method=name,t=t,
                             truth_shift=float(shift),prequential_mse=float(np.mean((pre-clean)**2)),
                             future_broad_mse=float(np.mean((wide-target)**2)),
                             alpha_change=float(np.linalg.norm(alpha-ALPHA)),calls=int(calls)))
    return rows


def gradual_benchmark(first_seed=1050, n=12):
    return [row for seed in range(first_seed,first_seed+n)
            for v in (.015,.04) for coverage in ('narrow','broad')
            for row in gradual_stream(seed,v,coverage)]
