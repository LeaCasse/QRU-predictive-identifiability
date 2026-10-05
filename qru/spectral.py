from itertools import product
import numpy as np


def signed_paths(depth, architecture_aware=False):
    if depth < 1:
        raise ValueError("depth must be >= 1.")
    first = (-1, 1) if architecture_aware else (-1, 0, 1)
    for s0 in first:
        for tail in product((-1, 0, 1), repeat=depth - 1):
            yield (s0,) + tail


def path_frequency(path, alpha):
    return float(np.dot(np.asarray(path, dtype=float), np.asarray(alpha, dtype=float)))


def candidate_paths(alpha, architecture_aware=False, decimals=12):
    alpha = np.asarray(alpha, dtype=float)
    out = {}
    for s in signed_paths(len(alpha), architecture_aware=architecture_aware):
        w = round(path_frequency(s, alpha), decimals)
        out.setdefault(w, []).append(tuple(s))
    return dict(sorted(out.items()))


def candidate_frequencies(alpha, architecture_aware=False, decimals=12):
    return np.array(list(candidate_paths(alpha, architecture_aware, decimals)), dtype=float)


def positive_frequencies(frequencies, tol=1e-12):
    f = np.asarray(frequencies, dtype=float)
    vals = {round(float(abs(w)), 12) for w in f if abs(w) > tol}
    return np.array(sorted(vals), dtype=float)


def has_zero_frequency(frequencies, tol=1e-12):
    return bool(np.any(np.abs(np.asarray(frequencies, dtype=float)) <= tol))


def architecture_basis(alpha, decimals=12):
    signed = candidate_frequencies(alpha, architecture_aware=True, decimals=decimals)
    return positive_frequencies(signed), has_zero_frequency(signed)


def fourier_design_matrix(x, positive_freqs, include_constant=True):
    x = np.asarray(x, dtype=float)
    positive_freqs = np.asarray(positive_freqs, dtype=float)
    cols, names = [], []

    if include_constant:
        cols.append(np.ones_like(x))
        names.append("const")

    for w in positive_freqs:
        cols.extend([np.cos(w * x), np.sin(w * x)])
        names.extend([f"cos({w:g}x)", f"sin({w:g}x)"])

    A = np.column_stack(cols) if cols else np.empty((len(x), 0))
    return A, names


def fit_real_spectrum(x, y, positive_freqs, ridge=0.0, include_constant=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A, names = fourier_design_matrix(x, positive_freqs, include_constant)

    if ridge > 0:
        A_fit = np.vstack([A, np.sqrt(ridge) * np.eye(A.shape[1])])
        y_fit = np.concatenate([y, np.zeros(A.shape[1])])
    else:
        A_fit, y_fit = A, y

    beta = np.linalg.lstsq(A_fit, y_fit, rcond=None)[0]
    y_hat = A @ beta
    residual = np.linalg.norm(y - y_hat) / (np.linalg.norm(y) + 1e-15)
    return {
        "coefficients": beta,
        "names": names,
        "reconstruction": y_hat,
        "relative_residual": float(residual),
        "condition_number": float(np.linalg.cond(A_fit)),
    }


def spectral_state(x, y, positive_freqs, ridge=0.0, include_constant=True):
    return fit_real_spectrum(
        x, y, positive_freqs, ridge=ridge, include_constant=include_constant
    )["coefficients"]


def support_residual(x, y, positive_freqs, include_constant=True, ridge=0.0):
    return fit_real_spectrum(
        x, y, positive_freqs, ridge=ridge, include_constant=include_constant
    )["relative_residual"]


def complex_fourier_coefficient(x, y, omega):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return complex(np.mean(y * np.exp(-1j * omega * x)))


def energy_by_integer_fft(x, y):
    y = np.asarray(y, dtype=float)
    N = len(y)
    coeff = np.fft.fft(y) / N
    bins = np.fft.fftfreq(N, d=1.0 / N)
    return bins, np.abs(coeff) ** 2


def forbidden_energy_ratio_integer(x, y, allowed_signed_freqs, max_abs_frequency=None, tol=1e-9):
    """Energy on integer FFT bins not present in an allowed signed support."""
    bins, energy = energy_by_integer_fft(x, y)
    allowed = np.asarray(allowed_signed_freqs, dtype=float)
    if max_abs_frequency is None:
        max_abs_frequency = np.max(np.abs(allowed)) if len(allowed) else 0.0

    region = np.abs(bins) <= max_abs_frequency + tol
    is_allowed = np.zeros(len(bins), dtype=bool)
    for w in allowed:
        is_allowed |= np.isclose(bins, w, atol=tol, rtol=0.0)

    forbidden = region & ~is_allowed
    return float(np.sum(energy[forbidden]) / (np.sum(energy[region]) + 1e-30))
