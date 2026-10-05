import numpy as np


def validate(theta, alpha):
    theta = np.asarray(theta, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if theta.ndim != 2 or theta.shape[1] != 3:
        raise ValueError("theta must have shape (L, 3): [theta_x, theta_z, bias].")
    if theta.shape[0] != len(alpha):
        raise ValueError("theta and alpha must have the same depth L.")
    return theta, alpha


def qru_output(x, theta, alpha):
    """Vectorized one-qubit QRU output <Z> for U_l = Rz Rx Ry."""
    theta, alpha = validate(theta, alpha)
    x = np.asarray(x, dtype=float).reshape(-1)

    a = np.ones(len(x), dtype=complex)
    b = np.zeros(len(x), dtype=complex)

    for l in range(len(alpha)):
        tx, tz, bias = theta[l]
        phi = alpha[l] * x + bias

        c, s = np.cos(phi / 2.0), np.sin(phi / 2.0)
        a, b = c * a - s * b, s * a + c * b

        cx, sx = np.cos(tx / 2.0), np.sin(tx / 2.0)
        a, b = cx * a - 1j * sx * b, -1j * sx * a + cx * b

        a *= np.exp(-0.5j * tz)
        b *= np.exp(0.5j * tz)

    return (np.abs(a) ** 2 - np.abs(b) ** 2).real


def qru_output_scalar(x, theta, alpha):
    return float(qru_output(np.array([x]), theta, alpha)[0])


def random_theta(depth, rng, scale=np.pi):
    return rng.uniform(-scale, scale, size=(depth, 3))


def parameter_names(depth):
    return [name for l in range(depth) for name in (f"theta_x[{l+1}]", f"theta_z[{l+1}]", f"bias[{l+1}]")]


def parameter_shift_prediction_jacobian(x, theta, alpha):
    """Exact parameter-shift Jacobian d f(x_n) / d theta_j."""
    theta, alpha = validate(theta, alpha)
    x = np.asarray(x, dtype=float)
    flat = theta.reshape(-1)
    jac = np.empty((len(x), len(flat)), dtype=float)

    for j in range(len(flat)):
        plus = flat.copy()
        minus = flat.copy()
        plus[j] += np.pi / 2.0
        minus[j] -= np.pi / 2.0
        jac[:, j] = 0.5 * (
            qru_output(x, plus.reshape(theta.shape), alpha)
            - qru_output(x, minus.reshape(theta.shape), alpha)
        )
    return jac
