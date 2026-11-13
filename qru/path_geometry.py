"""Exact two-layer attainable amplitudes and sinusoidal observation fibers."""
import numpy as np

from .predictive_identifiability import family_theta


def depth_two_amplitudes(theta):
    """All six tagged amplitudes for U_l=Rz(z_l) Rx(t_l) Ry(alpha_l*x+b_l).

    The final z rotation is a gauge for Z readout. The returned coefficients
    include bias phases and are independent of the encoding schedule.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (2, 3) or not np.all(np.isfinite(theta)):
        raise ValueError('theta must be a finite array of shape (2, 3).')
    t, z, b = theta[0]
    u, _, c = theta[1]
    amplitudes = {
        (1, 0): -.5*np.sin(u)*(np.sin(t)*np.cos(z)+1j*np.sin(z))*np.exp(1j*b),
        (1, 1): .25*np.cos(u)*(np.cos(t)+np.cos(z)+1j*np.sin(t)*np.sin(z))*np.exp(1j*(b+c)),
        (1, -1): .25*np.cos(u)*(np.cos(t)-np.cos(z)-1j*np.sin(t)*np.sin(z))*np.exp(1j*(b-c)),
    }
    return {**amplitudes, **{tuple(-i for i in s): np.conj(v) for s, v in amplitudes.items()}}


def sinusoidal_fiber(radius, phase):
    """Representatives of the complete PATH fiber of r*cos(x+phase) at (1,2).

    For 0<r<=1 these are exactly two attainable path vectors, with potentially
    many circuit-parameter realizations each. The statement assumes the full
    exact baseline response, rather than agreement on an arbitrary input grid.
    """
    if not np.isfinite(radius) or not 0 < radius <= 1 or not np.isfinite(phase):
        raise ValueError('Require 0 < radius <= 1 and a finite phase.')
    return {name: family_theta(radius, phase, branch)
            for branch, name in enumerate(('A', 'B'))}


def fiber_risk(radius, phase, alpha_future, x, weights=None):
    """Exact unrestricted minimax risk on the two-class sinusoidal fiber.

    Predictions use a fixed future schedule and a weighted finite input law.
    The midpoint achieves one quarter of the squared future separation.
    """
    from .circuit import qru_output
    circuits = sinusoidal_fiber(radius, phase)
    x = np.asarray(x, dtype=float).reshape(-1)
    if not len(x) or not np.all(np.isfinite(x)):
        raise ValueError('x must contain finite future inputs.')
    if weights is None:
        weights = np.full(len(x), 1/len(x))
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != x.shape or not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError('weights must be finite, nonnegative and have positive sum.')
        weights = weights/weights.sum()
    a = qru_output(x, circuits['A'], alpha_future)
    b = qru_output(x, circuits['B'], alpha_future)
    return float(np.dot(weights, (a-b)**2)/4)
