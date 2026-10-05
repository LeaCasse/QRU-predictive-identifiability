import numpy as np

from .circuit import qru_output, parameter_shift_prediction_jacobian
from .spectral import fourier_design_matrix, spectral_state


def spectral_coefficients(theta, alpha, x, positive_freqs, ridge=0.0, include_constant=True):
    y = qru_output(x, theta, alpha)
    return spectral_state(
        x, y, positive_freqs, ridge=ridge, include_constant=include_constant
    )


def spectral_jacobian(
    theta, alpha, x, positive_freqs, ridge=0.0, include_constant=True,
    method="parameter_shift", eps=1e-6,
):
    """Jacobian of real Fourier coefficients with respect to QRU parameters."""
    theta = np.asarray(theta, dtype=float)

    if method == "parameter_shift":
        A, _ = fourier_design_matrix(x, positive_freqs, include_constant)
        if ridge > 0:
            gram = A.T @ A + ridge * np.eye(A.shape[1])
            spectral_map = np.linalg.solve(gram, A.T)
        else:
            spectral_map = np.linalg.pinv(A)
        return spectral_map @ parameter_shift_prediction_jacobian(x, theta, alpha)

    if method == "finite_difference":
        flat = theta.reshape(-1)
        base = spectral_coefficients(
            theta, alpha, x, positive_freqs, ridge, include_constant
        )
        J = np.empty((len(base), len(flat)), dtype=float)
        for j in range(len(flat)):
            plus, minus = flat.copy(), flat.copy()
            plus[j] += eps
            minus[j] -= eps
            c_plus = spectral_coefficients(
                plus.reshape(theta.shape), alpha, x, positive_freqs, ridge, include_constant
            )
            c_minus = spectral_coefficients(
                minus.reshape(theta.shape), alpha, x, positive_freqs, ridge, include_constant
            )
            J[:, j] = (c_plus - c_minus) / (2.0 * eps)
        return J

    raise ValueError("method must be 'parameter_shift' or 'finite_difference'.")


def svd_diagnostics(J, relative_tol=1e-8):
    U, s, Vt = np.linalg.svd(np.asarray(J, dtype=float), full_matrices=True)
    tol = relative_tol * s[0] if len(s) and s[0] > 0 else 0.0
    rank = int(np.sum(s > tol))
    return {
        "U": U,
        "singular_values": s,
        "Vt": Vt,
        "tolerance": tol,
        "effective_rank": rank,
    }


def orthogonal_projector(J, relative_tol=1e-8):
    diag = svd_diagnostics(J, relative_tol)
    U = diag["U"][:, :diag["effective_rank"]]
    return U @ U.T if U.size else np.zeros((J.shape[0], J.shape[0]))


def regularized_projector(J, lam=1e-4):
    J = np.asarray(J, dtype=float)
    U, s, _ = np.linalg.svd(J, full_matrices=False)
    weights = s**2 / (s**2 + lam)
    return (U * weights) @ U.T


def inaccessible_fraction(delta_c, J, lam=None, relative_tol=1e-8):
    delta_c = np.asarray(delta_c, dtype=float)
    P = (
        orthogonal_projector(J, relative_tol)
        if lam is None
        else regularized_projector(J, lam)
    )
    residual = (np.eye(len(delta_c)) - P) @ delta_c
    return float(np.linalg.norm(residual) / (np.linalg.norm(delta_c) + 1e-15))


def linearization_error(
    theta, delta_theta, alpha, x, positive_freqs,
    ridge=0.0, include_constant=True,
):
    theta = np.asarray(theta, dtype=float)
    delta_theta = np.asarray(delta_theta, dtype=float)
    c0 = spectral_coefficients(theta, alpha, x, positive_freqs, ridge, include_constant)
    c1 = spectral_coefficients(theta + delta_theta, alpha, x, positive_freqs, ridge, include_constant)
    J = spectral_jacobian(
        theta, alpha, x, positive_freqs, ridge, include_constant, method="parameter_shift"
    )
    actual = c1 - c0
    predicted = J @ delta_theta.reshape(-1)
    error = np.linalg.norm(actual - predicted) / (np.linalg.norm(actual) + 1e-15)
    return {
        "relative_error": float(error),
        "actual_delta": actual,
        "predicted_delta": predicted,
        "jacobian": J,
    }
