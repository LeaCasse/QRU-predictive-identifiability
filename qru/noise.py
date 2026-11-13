"""Density-matrix simulation and exact twins under calibrated Z dephasing."""
import numpy as np
from .circuit import validate


def _layer(theta, alpha, x):
    t, z, bias = theta
    angle = alpha*x+bias
    c, s = np.cos(angle/2), np.sin(angle/2)
    ry = np.array([[c, -s], [s, c]], dtype=complex)
    c, s = np.cos(t/2), np.sin(t/2)
    rx = np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)
    rz = np.diag([np.exp(-1j*z/2), np.exp(1j*z/2)])
    return rz @ rx @ ry


def density_output(x, theta, alpha, gamma=1., channel='dephasing', location='interlayer'):
    """Z means with the same channel after every non-final layer or at output.

    Z dephasing maps (rx,ry,rz) to (gamma*rx,gamma*ry,rz), gamma in [0,1].
    Isotropic depolarization is an explicit control. For the two-layer paper
    construction there is exactly one interlayer channel. No hardware noise
    calibration is assumed by this simulator.
    """
    theta, alpha = validate(theta, alpha)
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError('gamma must lie in [0, 1].')
    if channel not in ('dephasing', 'depolarizing') or location not in ('interlayer', 'final'):
        raise ValueError('Unknown channel or location.')
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(alpha)):
        raise ValueError('Circuit parameters must be finite.')
    def apply(rho):
        fixed = np.diag(np.diag(rho)) if channel == 'dephasing' else np.eye(2)/2
        return gamma*rho+(1-gamma)*fixed
    outputs = []
    for value in np.asarray(x, dtype=float).reshape(-1):
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        for layer, (parameters, scale) in enumerate(zip(theta, alpha)):
            u = _layer(parameters, scale, value)
            rho = u @ rho @ u.conj().T
            if layer < len(alpha)-1 and location == 'interlayer':
                rho = apply(rho)
        if location == 'final':
            rho = apply(rho)
        outputs.append(float((rho[0, 0]-rho[1, 1]).real))
    return np.array(outputs)


def dephasing_twins(gamma):
    """Exact two-layer twins for known identical interlayer Z dephasing.

    A = gamma*cos((alpha2-alpha1)*x), B = gamma*cos(alpha1*x).
    Preparation A depends on gamma. At gamma=0 both responses vanish.
    """
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError('gamma must lie in [0, 1].')
    return {
        'A': np.array([[np.arccos(gamma), np.pi, 0.], [0., 0., 0.]]),
        'B': np.array([[0., np.pi/2, np.pi/2], [np.pi/2, 0., 0.]]),
    }
