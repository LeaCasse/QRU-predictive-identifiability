from itertools import product
import numpy as np

from .circuit import qru_output
from .spectral import path_frequency


# -----------------------------------------------------------------------------
# Path sets and exact Bloch-factorized amplitudes
# -----------------------------------------------------------------------------


def path_set(depth, architecture_aware=True):
    first = (-1, 1) if architecture_aware else (-1, 0, 1)
    for s0 in first:
        for tail in product((-1, 0, 1), repeat=depth - 1):
            yield (s0,) + tail


def _bloch_rx(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=complex,
    )


def _bloch_rz(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=complex,
    )


_P0 = np.array(
    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
    dtype=complex,
)
_PPLUS = 0.5 * np.array(
    [[1.0, 0.0, -1j], [0.0, 0.0, 0.0], [1j, 0.0, 1.0]],
    dtype=complex,
)
_PMINUS = np.conjugate(_PPLUS)
_P_BY_SIGN = {-1: _PMINUS, 0: _P0, 1: _PPLUS}


def analytic_path_amplitudes(
    theta,
    architecture_aware=True,
    include_bias_phase=True,
):
    """
    Exact path amplitudes for
        U_l = Rz(theta_z,l) Rx(theta_x,l) Ry(alpha_l x + b_l).

    Returns amplitudes a_s such that
        f(x) = sum_s a_s exp(i (s . alpha) x).

    If include_bias_phase=False, returns the latent mixer amplitudes B_s.
    If include_bias_phase=True, returns a_s = B_s exp(i s.b).
    The amplitudes are independent of alpha.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 2 or theta.shape[1] != 3:
        raise ValueError("theta must have shape (L, 3): [theta_x, theta_z, bias].")

    r0 = np.array([0.0, 0.0, 1.0], dtype=complex)
    out = {}

    for path in path_set(len(theta), architecture_aware=architecture_aware):
        v = r0.copy()
        for l, sign in enumerate(path):
            theta_x, theta_z, _ = theta[l]
            mixer = _bloch_rz(theta_z) @ _bloch_rx(theta_x)
            v = mixer @ _P_BY_SIGN[sign] @ v

        amp = complex(v[2])
        if include_bias_phase:
            amp *= np.exp(1j * np.dot(path, theta[:, 2]))
        out[path] = amp

    return out


def reconstruct_from_paths(x, alpha, amplitudes):
    x = np.asarray(x, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    y = np.zeros(len(x), dtype=complex)
    for path, amp in amplitudes.items():
        omega = float(np.dot(path, alpha))
        y += amp * np.exp(1j * omega * x)
    return np.real_if_close(y, tol=1000).real


def grouped_path_coefficients(alpha, amplitudes, decimals=12):
    groups = {}
    for path, amp in amplitudes.items():
        omega = round(path_frequency(path, alpha), decimals)
        groups[omega] = groups.get(omega, 0.0j) + amp
    return dict(sorted(groups.items()))


# -----------------------------------------------------------------------------
# Frequency collisions and phase-code design
# -----------------------------------------------------------------------------


def frequency_groups(alpha, architecture_aware=True, decimals=12):
    groups = {}
    for path in path_set(len(alpha), architecture_aware=architecture_aware):
        omega = round(path_frequency(path, alpha), decimals)
        groups.setdefault(omega, []).append(path)
    return dict(sorted(groups.items()))


def collision_groups(alpha, architecture_aware=True, decimals=12):
    groups = frequency_groups(alpha, architecture_aware, decimals)
    return {omega: paths for omega, paths in groups.items() if len(paths) >= 2}


def balanced_ternary_codes(depth):
    return np.array([3 ** l for l in range(depth)], dtype=int)


def phase_tag(path, q):
    return int(np.dot(np.asarray(path, dtype=int), np.asarray(q, dtype=int)))


def phase_alias_groups(alpha, n_phase, q=None, architecture_aware=True, decimals=12):
    if n_phase < 1:
        raise ValueError("n_phase must be >= 1.")
    if q is None:
        q = balanced_ternary_codes(len(alpha))
    q = np.asarray(q, dtype=int)

    aliases = {}
    for omega, paths in frequency_groups(alpha, architecture_aware, decimals).items():
        by_residue = {}
        for path in paths:
            residue = phase_tag(path, q) % n_phase
            by_residue.setdefault(residue, []).append(path)
        bad = [group for group in by_residue.values() if len(group) > 1]
        if bad:
            aliases[omega] = bad
    return aliases


def minimum_phase_samples(alpha, q=None, architecture_aware=True, decimals=12, max_n=10000):
    """
    Smallest number of equispaced phase settings for which phase tags are
    distinct modulo n_phase inside every x-frequency collision class.

    Global tag uniqueness is unnecessary because different x-frequencies are
    separated before phase demodulation.
    """
    if q is None:
        q = balanced_ternary_codes(len(alpha))
    for n_phase in range(1, max_n + 1):
        if not phase_alias_groups(
            alpha,
            n_phase,
            q=q,
            architecture_aware=architecture_aware,
            decimals=decimals,
        ):
            return n_phase
    raise RuntimeError("No alias-free phase sample count found up to max_n.")


# -----------------------------------------------------------------------------
# Spectrum extraction and 1D coded phase cycling
# -----------------------------------------------------------------------------


def fit_complex_spectrum(x, y, signed_frequencies):
    """Least-squares complex exponential fit on arbitrary real frequencies."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    freqs = np.asarray(signed_frequencies, dtype=float)
    design = np.exp(1j * np.outer(x, freqs))
    coeff = np.linalg.lstsq(design, y, rcond=None)[0]
    reconstruction = design @ coeff
    residual = np.linalg.norm(reconstruction - y) / (np.linalg.norm(y) + 1e-15)
    return {
        "frequencies": freqs,
        "coefficients": coeff,
        "relative_residual": float(residual),
        "condition_number": float(np.linalg.cond(design)),
    }


def recover_paths_from_phase_cycle(
    theta,
    alpha,
    x,
    n_phase=None,
    q=None,
    architecture_aware=True,
    decimals=12,
):
    """
    Recover all path amplitudes using additive bias offsets
        b_l -> b_l + q_l t.

    For each phase setting, all distinct x-frequency coefficients are first
    estimated jointly by least squares. Phase demodulation is then performed
    only inside each collision class.
    """
    theta = np.asarray(theta, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    x = np.asarray(x, dtype=float)

    if q is None:
        q = balanced_ternary_codes(len(alpha))
    q = np.asarray(q, dtype=int)

    if n_phase is None:
        n_phase = minimum_phase_samples(
            alpha,
            q=q,
            architecture_aware=architecture_aware,
            decimals=decimals,
        )

    aliases = phase_alias_groups(
        alpha,
        n_phase,
        q=q,
        architecture_aware=architecture_aware,
        decimals=decimals,
    )

    groups = frequency_groups(alpha, architecture_aware, decimals)
    freqs = np.array(list(groups.keys()), dtype=float)
    t = 2.0 * np.pi * np.arange(n_phase) / n_phase

    coefficient_series = np.empty((n_phase, len(freqs)), dtype=complex)
    spectral_residuals = []
    spectral_condition = None

    for k, tk in enumerate(t):
        shifted = theta.copy()
        shifted[:, 2] = theta[:, 2] + q * tk
        y = qru_output(x, shifted, alpha)
        fit = fit_complex_spectrum(x, y, freqs)
        coefficient_series[k] = fit["coefficients"]
        spectral_residuals.append(fit["relative_residual"])
        spectral_condition = fit["condition_number"]

    freq_index = {round(float(w), decimals): i for i, w in enumerate(freqs)}
    amplitudes = {}

    for omega, paths in groups.items():
        series = coefficient_series[:, freq_index[omega]]
        for path in paths:
            nu = phase_tag(path, q)
            amplitudes[path] = complex(np.mean(series * np.exp(-1j * nu * t)))

    return {
        "amplitudes": amplitudes,
        "phase_points": t,
        "phase_codes": q,
        "n_phase": int(n_phase),
        "aliases": aliases,
        "frequencies": freqs,
        "coefficient_series": coefficient_series,
        "spectral_condition_number": float(spectral_condition),
        "max_spectral_residual": float(np.max(spectral_residuals)),
    }


def relative_path_error(reference, estimate):
    paths = list(reference.keys())
    ref = np.array([reference[p] for p in paths], dtype=complex)
    est = np.array([estimate[p] for p in paths], dtype=complex)
    return float(np.linalg.norm(est - ref) / (np.linalg.norm(ref) + 1e-15))


def max_absolute_path_error(reference, estimate):
    return float(max(abs(estimate[p] - reference[p]) for p in reference))
