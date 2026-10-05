import numpy as np

from .circuit import qru_output, parameter_shift_prediction_jacobian, random_theta
from .controllability import spectral_jacobian, svd_diagnostics


def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def fit_theta_adam(
    theta0,
    alpha,
    x,
    y_target,
    steps=60,
    lr=0.04,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
):
    theta = np.asarray(theta0, dtype=float).copy()
    y_target = np.asarray(y_target, dtype=float)
    m = np.zeros(theta.size)
    v = np.zeros(theta.size)
    history = []

    for step in range(1, steps + 1):
        y_pred = qru_output(x, theta, alpha)
        err = y_pred - y_target
        history.append(float(np.mean(err**2)))
        jac = parameter_shift_prediction_jacobian(x, theta, alpha)
        grad = (2.0 / len(x)) * jac.T @ err

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad**2
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        theta = (
            theta.reshape(-1) - lr * m_hat / (np.sqrt(v_hat) + eps)
        ).reshape(theta.shape)

    prediction = qru_output(x, theta, alpha)
    return {
        "theta": theta,
        "history": np.asarray(history),
        "prediction": prediction,
        "final_mse": mse(y_target, prediction),
    }


def find_reference_theta(
    alpha,
    x,
    positive_freqs,
    rng,
    include_constant=False,
    max_abs_output=0.75,
    required_rank=None,
    n_trials=1000,
):
    for _ in range(n_trials):
        theta = random_theta(len(alpha), rng)
        y = qru_output(x, theta, alpha)
        if np.max(np.abs(y)) > max_abs_output:
            continue
        J = spectral_jacobian(
            theta,
            alpha,
            x,
            positive_freqs,
            include_constant=include_constant,
        )
        rank = svd_diagnostics(J)["effective_rank"]
        if required_rank is None or rank == required_rank:
            return theta
    raise RuntimeError("No reference theta found with the requested constraints.")
