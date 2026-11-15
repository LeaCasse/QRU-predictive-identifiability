"""Exact collision partitions and finite phase designs at arbitrary depth.

Use integer or Fraction schedules for exact arithmetic. Floating-point schedules
are interpreted as their exact decimal strings, never merged by tolerance.
"""
from fractions import Fraction
from itertools import product
import numpy as np
from .tomography import analytic_path_amplitudes


def collision_partition(paths, alpha):
    alpha = tuple(Fraction(str(a)) if not isinstance(a, Fraction) else a for a in alpha)
    groups = {}
    for s in paths:
        if len(s) != len(alpha):
            raise ValueError("Path and schedule depths differ.")
        w = sum((k*a for k, a in zip(s, alpha)), Fraction(0))
        groups.setdefault(w, []).append(s)
    return groups


def phase_groups(amplitudes, alpha, layers):
    """Coefficients after controlling the specified zero-indexed layers.

    All coefficients vanish iff all such phase/input queries give zero means.
    Applied to an amplitude difference, this is a complete pairwise test.
    """
    depth = len(alpha)
    layers = tuple(layers)
    if len(set(layers)) != len(layers) or any(l < 0 or l >= depth for l in layers):
        raise ValueError("Layers must be distinct valid indices.")
    result = {}
    for w, paths in collision_partition(amplitudes, alpha).items():
        for s in paths:
            key = (w, tuple(s[l] for l in layers))
            result[key] = result.get(key, 0j) + amplitudes[s]
    return result


def pair_design(theta_a, theta_b, alpha, tolerance=1e-10):
    """Smallest layer subsets separating this pair, with numeric certificates.

    Equality decisions use a declared tolerance on exact-partition coefficient
    sums. Small residuals are numerical evidence, not symbolic equality proofs.
    """
    from itertools import combinations
    a, b = map(analytic_path_amplitudes, (theta_a, theta_b))
    if a.keys() != b.keys():
        raise ValueError("Both circuits must have the same architecture.")
    v = {s: a[s]-b[s] for s in a}
    profiles = []
    for size in range(len(alpha)+1):
        for layers in combinations(range(len(alpha)), size):
            g = phase_groups(v, alpha, layers)
            profiles.append(dict(layers=list(layers), coefficient_norm=float(np.linalg.norm(list(g.values())))))
        found = [r for r in profiles if len(r['layers']) == size and r['coefficient_norm'] > tolerance]
        if found:
            return dict(minimum_layers=size, designs=found, tested=profiles, tolerance=tolerance)
    return dict(minimum_layers=None, designs=[], tested=profiles, tolerance=tolerance)


def fixed_input_tomography(theta, alpha, x=0.):
    """Recover all paths from a 3-point phase grid per layer at one fixed input.

    This exact-mean reference uses 3**L configurations. It is a completeness
    certificate rather than a shot-efficient claim. The first coordinate has
    no zero tag but the uniform grid is retained for a simple FFT inverse.
    """
    from .circuit import qru_output
    theta = np.asarray(theta, float)
    shape = (3,)*len(theta)
    values = np.empty(shape)
    for index in product(range(3), repeat=len(theta)):
        t = theta.copy(); t[:, 2] += 2*np.pi*np.asarray(index)/3
        values[index] = qru_output([x], t, alpha)[0]
    coefficients = np.fft.fftn(values)/values.size
    return {s: coefficients[tuple(k % 3 for k in s)]*np.exp(-1j*np.dot(s, alpha)*x)
            for s in analytic_path_amplitudes(theta)}
