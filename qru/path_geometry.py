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


def attainable_conditions(a0, a_plus, a_minus, tolerance=1e-10):
    """Necessary and sufficient two-layer path conditions, numerically checked.

    The theorem uses exact equalities. ``tolerance`` is an absolute numerical
    threshold on their residuals; use reconstruct_attainable for a separate
    constructive check near singular boundaries.
    """
    a0, a_plus, a_minus = complex(a0), complex(a_plus), complex(a_minus)
    if not np.all(np.isfinite([a0, a_plus, a_minus])) or tolerance < 0:
        raise ValueError('Require finite amplitudes and nonnegative tolerance.')
    p, q, w = abs(a_plus), abs(a_minus), abs(a0)
    total = p+q
    modulus_residual = total**2*(w*w+4*p*q)-p*q
    phase_residual = a0*a0*np.conj(a_plus*a_minus)+w*w*p*q
    valid = total <= .5+tolerance and w <= .5+tolerance
    valid = valid and abs(modulus_residual) <= tolerance and abs(phase_residual) <= tolerance
    return dict(attainable=bool(valid),p=p,q=q,w=w,
                modulus_residual=float(modulus_residual),phase_residual=float(abs(phase_residual)))


def reconstruct_attainable(a0, a_plus, a_minus, tolerance=1e-9):
    """Construct a canonical circuit for any attainable two-layer path vector.

    Canonical representatives set z1=z2=0. Many original circuit parameters
    can realize the same path vector. The constructor checks the result and
    rejects inconsistent triples, rather than relying on residuals alone.
    """
    a0, a_plus, a_minus = complex(a0), complex(a_plus), complex(a_minus)
    data=attainable_conditions(a0,a_plus,a_minus,tolerance)
    if not data['attainable']:
        raise ValueError('Amplitudes do not satisfy the attainable-set conditions.')
    p,q,w=data['p'],data['q'],data['w'];total=p+q
    if total == 0:
        theta=np.array([[np.arcsin(min(1.,2*w)),0.,np.angle(-a0)],
                        [np.pi/2,0.,0.]])
    else:
        t=np.arccos(np.clip((p-q)/total,-1.,1.))
        u=np.arccos(np.clip(2*total,0.,1.))
        # Determine the bias sum before the difference, including all faces.
        if w > 1e-13:
            b=np.angle(-a0)
        elif p*q > 1e-26:
            b=.5*np.angle(-a_plus*a_minus)
        else:
            b=np.angle(a_plus) if p>q else np.angle(-a_minus)
        c=np.angle(a_plus)-b if p>1e-13 else b-np.angle(-a_minus)
        theta=np.array([[t,0.,b],[u,0.,c]])
    recovered=depth_two_amplitudes(theta)
    error=max(abs(recovered[k]-a) for k,a in [((1,0),a0),((1,1),a_plus),((1,-1),a_minus)])
    if error > tolerance:
        raise ValueError('No accurate canonical reconstruction at the requested tolerance.')
    return theta
