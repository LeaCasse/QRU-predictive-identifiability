"""Local observability filtration for the one-qubit Rz Rx Ry re-uploading model.

Ranks describe a fixed parameter point and exact noiseless derivatives, not
global identifiability or an architecture-wide theorem. Taylor jets are exact
path sums; theta derivatives use the exact parameter-shift identity.
"""
from math import factorial

import numpy as np

from .tomography import analytic_path_amplitudes, frequency_groups


def jets(theta, alpha, center, degree):
    paths = analytic_path_amplitudes(theta)
    tags = np.asarray(list(paths), dtype=float)
    amps = np.asarray(list(paths.values()), dtype=complex)
    omega = tags @ np.asarray(alpha, dtype=float)
    weighted = amps * np.exp(1j * omega * center)
    return np.asarray([np.real(np.sum(weighted * (1j * omega)**k)) / factorial(k)
                       for k in range(degree + 1)])


def jet_jacobians(theta, alpha, center, degree):
    """T[k,j]=d_theta_j [x^k] f(center+x), A[k,j]=d_alpha_j [x^k] f.

    The exact identity d_alpha_j f(x)=x d_bias_j f(x) gives A without
    differentiating finite differences or the path frequencies numerically.
    """
    theta = np.asarray(theta, dtype=float)
    flat = theta.ravel()
    columns = []
    for j in range(len(flat)):
        plus, minus = flat.copy(), flat.copy()
        plus[j] += np.pi / 2
        minus[j] -= np.pi / 2
        columns.append((jets(plus.reshape(theta.shape), alpha, center, degree)
                        - jets(minus.reshape(theta.shape), alpha, center, degree)) / 2)
    t = np.stack(columns, axis=1)
    a = np.stack([center * t[:, 3*j+2] + np.r_[0., t[:-1, 3*j+2]]
                  for j in range(len(alpha))], axis=1)
    return t, a


def numerical_rank(matrix, rtol=1e-10):
    s = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(s > rtol * s[0])) if len(s) and s[0] else 0


def classical_filtration(amplitudes, frequencies, center, degree, rtol=1e-10):
    """A capacity-matched classical L-tone model with 2L free amplitudes."""
    frequencies=np.asarray(frequencies,dtype=float)
    amplitudes=np.asarray(amplitudes,dtype=float).reshape(len(frequencies),2)
    t=np.zeros((degree+1,2*len(frequencies)))
    a=np.zeros((degree+1,len(frequencies)))
    for k in range(degree+1):
        for j,w in enumerate(frequencies):
            phase=w*center+k*np.pi/2
            cs=np.array([np.cos(phase),np.sin(phase)])
            t[k,2*j:2*j+2]=w**k*cs/factorial(k)
            a[k,j]=((k*w**(k-1)/factorial(k) if k else 0)*np.dot(amplitudes[j],cs)
                    +center*w**k/factorial(k)*np.dot(amplitudes[j],[-cs[1],cs[0]]))
    return [dict(degree=k,nuisance_rank=numerical_rank(t[:k+1],rtol),
                 joint_rank=numerical_rank(np.c_[t[:k+1],a[:k+1]],rtol),
                 visible_drift_dimensions=numerical_rank(np.c_[t[:k+1],a[:k+1]],rtol)
                                          -numerical_rank(t[:k+1],rtol))
            for k in range(degree+1)]


def filtration(theta, alpha, center, degree, *, active=None, rtol=1e-10):
    """For each degree k, ranks of theta jets and joint theta/alpha jets.

    A drift direction v first becomes visible when its alpha column exits
    the theta column space in the *same* truncated Taylor-jet space.
    """
    t, a = jet_jacobians(theta, alpha, center, degree)
    if active is not None:
        t = t[:, list(active)]
    rows = []
    for k in range(degree+1):
        nuisance_rank = numerical_rank(t[:k+1], rtol)
        joint_rank = numerical_rank(np.column_stack((t[:k+1], a[:k+1])), rtol)
        rows.append(dict(degree=k, nuisance_rank=nuisance_rank,
                         joint_rank=joint_rank, visible_drift_dimensions=joint_rank-nuisance_rank))
    return rows


def sweep(seed=23981, samples=8, max_depth=4):
    """Fixed, independently seeded witnesses; upper depth avoids ill-scaled jets."""
    rng = np.random.default_rng(seed)
    rows = []
    for depth in range(1, max_depth+1):
        schedules = {'dyadic': 2.**np.arange(depth),
                     'collision': np.ones(depth),
                     'incommensurate': np.sqrt(np.arange(1, depth+1))}
        if depth > 1:
            schedules['inactive_uploads'] = np.r_[1., np.zeros(depth-1)]
        for sample in range(samples):
            theta = rng.uniform(-2., 2., (depth, 3))
            commuting=np.zeros_like(theta)
            commuting[:,2]=theta[:,2]
            classical_amplitudes=rng.normal(size=(depth,2))
            for center in (0.73, np.pi):
                curve=classical_filtration(classical_amplitudes,
                                           1.4*np.arange(1,depth+1),center,3*depth+1)
                rows.append(dict(depth=depth,schedule='classical_distinct',sample=sample,
                                 center=float(center),parameters='fourier_amplitudes',
                                 support_size=2*depth,
                                 theta_rank=max(r['nuisance_rank'] for r in curve),
                                 joint_rank=max(r['joint_rank'] for r in curve),
                                 first_visible_order=next((r['degree'] for r in curve if
                                                           r['visible_drift_dimensions']>0),None),
                                 visible_dimensions=[r['visible_drift_dimensions'] for r in curve],
                                 theta_ranks=[r['nuisance_rank'] for r in curve]))
                curve=filtration(commuting,2.**np.arange(depth),center,3*depth+1,
                                 active=[3*j+2 for j in range(depth)])
                rows.append(dict(depth=depth,schedule='commuting_Ry',sample=sample,
                                 center=float(center),parameters='bias_only',support_size=2,
                                 theta_rank=max(r['nuisance_rank'] for r in curve),
                                 joint_rank=max(r['joint_rank'] for r in curve),
                                 first_visible_order=next((r['degree'] for r in curve if
                                                           r['visible_drift_dimensions']>0),None),
                                 visible_dimensions=[r['visible_drift_dimensions'] for r in curve],
                                 theta_ranks=[r['nuisance_rank'] for r in curve]))
            for schedule, alpha in schedules.items():
                distinct = len(frequency_groups(alpha))
                for center in (0.73, np.pi):
                    for parameters, active in (
                        ('all', None),
                        ('bias_only', [3*j+2 for j in range(depth)]),
                        ('mixer_x_and_bias', [3*j+k for j in range(depth) for k in (0,2)]),
                    ):
                        curve=filtration(theta,alpha,center,3*depth+1,active=active)
                        increments=[row['degree'] for row in curve if row['visible_drift_dimensions']>0]
                        rows.append(dict(depth=depth,schedule=schedule,sample=sample,
                                         center=float(center),parameters=parameters,
                                         support_size=distinct,theta_rank=max(r['nuisance_rank'] for r in curve),
                                         joint_rank=max(r['joint_rank'] for r in curve),
                                         first_visible_order=min(increments) if increments else None,
                                         visible_dimensions=[r['visible_drift_dimensions'] for r in curve],
                                         theta_ranks=[r['nuisance_rank'] for r in curve]))
    return rows
