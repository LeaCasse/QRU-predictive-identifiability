import numpy as np

from .circuit import qru_output, parameter_shift_prediction_jacobian
from .tomography import reconstruct_from_paths


def path_schedule_jacobian(x, alpha, amplitudes):
    """Exact df/dalpha from recovered latent path amplitudes a_s."""
    x = np.asarray(x, dtype=float).reshape(-1)
    alpha = np.asarray(alpha, dtype=float)
    J = np.zeros((len(x), len(alpha)), dtype=complex)
    for path, amp in amplitudes.items():
        s = np.asarray(path, dtype=float)
        omega = float(s @ alpha)
        base = amp * np.exp(1j * omega * x)
        J += (1j * x * base)[:, None] * s[None, :]
    return np.real_if_close(J, tol=1000).real


def finite_difference_schedule_jacobian(x, theta, alpha, eps=1e-6):
    x = np.asarray(x, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    J = np.empty((len(x), len(alpha)), dtype=float)
    for j in range(len(alpha)):
        plus = alpha.copy(); plus[j] += eps
        minus = alpha.copy(); minus[j] -= eps
        J[:, j] = (qru_output(x, theta, plus) - qru_output(x, theta, minus)) / (2.0 * eps)
    return J


def projector(J, relative_tol=1e-8):
    J = np.asarray(J, dtype=float)
    U, s, _ = np.linalg.svd(J, full_matrices=False)
    if len(s) == 0 or s[0] == 0:
        return np.zeros((J.shape[0], J.shape[0]))
    rank = int(np.sum(s > relative_tol * s[0]))
    U = U[:, :rank]
    return U @ U.T if rank else np.zeros((J.shape[0], J.shape[0]))


def normalized_lstsq_residual(delta_y, J):
    delta_y = np.asarray(delta_y, dtype=float)
    J = np.asarray(J, dtype=float)
    if J.shape[1] == 0:
        return 1.0
    coef = np.linalg.lstsq(J, delta_y, rcond=None)[0]
    residual = delta_y - J @ coef
    return float(np.linalg.norm(residual) / (np.linalg.norm(delta_y) + 1e-15))


def augmented_diagnostics(delta_y, J_theta, J_alpha):
    """Compare the theta tangent space to the augmented theta+alpha tangent space."""
    delta_y = np.asarray(delta_y, dtype=float)
    J_theta = np.asarray(J_theta, dtype=float)
    J_alpha = np.asarray(J_alpha, dtype=float)
    J_aug = np.column_stack([J_theta, J_alpha])
    e_theta = normalized_lstsq_residual(delta_y, J_theta)
    e_aug = normalized_lstsq_residual(delta_y, J_aug)
    gain = max(0.0, e_theta**2 - e_aug**2)
    rel_gain = gain / (e_theta**2 + 1e-30)
    return {
        "e_theta": float(e_theta),
        "e_aug": float(e_aug),
        "architecture_gain": float(gain),
        "relative_architecture_gain": float(rel_gain),
    }


def schedule_step(delta_y, J_theta, J_alpha, ridge=1e-6, max_norm=None):
    """Least-norm alpha step for the component not absorbable by theta."""
    delta_y = np.asarray(delta_y, dtype=float)
    P_theta = projector(J_theta)
    r = (np.eye(len(delta_y)) - P_theta) @ delta_y
    Jp = (np.eye(len(delta_y)) - P_theta) @ J_alpha
    A = Jp.T @ Jp + ridge * np.eye(Jp.shape[1])
    step = np.linalg.solve(A, Jp.T @ r)
    if max_norm is not None:
        n = np.linalg.norm(step)
        if n > max_norm and n > 0:
            step = step * (max_norm / n)
    return step


def _theta_corrected_objective(x, y_target, theta, alpha):
    """Residual energy after the best *local* theta correction at a given alpha."""
    y = qru_output(x, theta, alpha)
    residual = np.asarray(y_target, dtype=float) - y
    J_theta = parameter_shift_prediction_jacobian(x, theta, alpha)
    P_theta = projector(J_theta)
    r_perp = (np.eye(len(residual)) - P_theta) @ residual
    return float(np.mean(r_perp**2)), residual, J_theta, r_perp


def trust_region_path_schedule_surgery(
    x,
    y_target,
    theta,
    alpha0,
    amplitudes,
    n_steps=20,
    ridge=1e-6,
    initial_radius=0.05,
    min_radius=1e-5,
    max_radius=0.25,
    eta_accept=0.10,
    eta_shrink=0.25,
    eta_expand=0.75,
    shrink_factor=0.5,
    expand_factor=2.0,
    max_retries=8,
    objective_tol=1e-14,
):
    """Trust-region schedule surgery driven by the path-derived alpha Jacobian.

    The local model acts only on the residual component orthogonal to the
    current theta tangent space. A candidate alpha step is accepted according
    to the standard trust-region ratio

        rho = actual reduction / predicted reduction.

    The *actual* reduction is recomputed from the QRU at the candidate schedule
    and again removes the locally theta-absorbable component. This makes the
    radius adaptive rather than fixing a step size by hand.
    """
    x = np.asarray(x, dtype=float)
    y_target = np.asarray(y_target, dtype=float)
    theta = np.asarray(theta, dtype=float)
    alpha = np.asarray(alpha0, dtype=float).copy()
    radius = float(initial_radius)
    history = []
    accepted_steps = 0
    status = "max_steps"

    for k in range(n_steps):
        current_obj, residual, J_theta, r_perp = _theta_corrected_objective(
            x, y_target, theta, alpha
        )
        J_alpha = path_schedule_jacobian(x, alpha, amplitudes)
        diag = augmented_diagnostics(residual, J_theta, J_alpha)
        P_theta = projector(J_theta)
        Jp = (np.eye(len(residual)) - P_theta) @ J_alpha

        if current_obj <= objective_tol:
            status = "converged"
            history.append({
                "iteration": k,
                "accepted": True,
                "rho": np.nan,
                "radius": radius,
                "objective": current_obj,
                "predicted_reduction": 0.0,
                "actual_reduction": 0.0,
                "step_norm": 0.0,
                "alpha": alpha.copy(),
                "step": np.zeros_like(alpha),
                **diag,
            })
            break

        accepted = False
        best_record = None
        for retry in range(max_retries + 1):
            raw_step = schedule_step(
                residual,
                J_theta,
                J_alpha,
                ridge=ridge,
                max_norm=radius,
            )
            step_norm = float(np.linalg.norm(raw_step))
            if step_norm < 1e-14:
                status = "zero_step"
                best_record = {
                    "iteration": k,
                    "retry": retry,
                    "accepted": False,
                    "rho": np.nan,
                    "radius": radius,
                    "objective": current_obj,
                    "predicted_reduction": 0.0,
                    "actual_reduction": 0.0,
                    "step_norm": step_norm,
                    "alpha": alpha.copy(),
                    "step": raw_step.copy(),
                    **diag,
                }
                break

            pred_obj = float(np.mean((r_perp - Jp @ raw_step) ** 2))
            predicted_reduction = current_obj - pred_obj
            candidate = alpha + raw_step
            cand_obj, _, _, _ = _theta_corrected_objective(
                x, y_target, theta, candidate
            )
            actual_reduction = current_obj - cand_obj
            if predicted_reduction <= 1e-18:
                rho = -np.inf
            else:
                rho = actual_reduction / predicted_reduction

            best_record = {
                "iteration": k,
                "retry": retry,
                "accepted": False,
                "rho": float(rho),
                "radius": radius,
                "objective": current_obj,
                "candidate_objective": cand_obj,
                "predicted_reduction": float(predicted_reduction),
                "actual_reduction": float(actual_reduction),
                "step_norm": step_norm,
                "alpha": alpha.copy(),
                "step": raw_step.copy(),
                **diag,
            }

            if actual_reduction > 0.0 and rho >= eta_accept:
                accepted = True
                best_record["accepted"] = True
                alpha = candidate
                accepted_steps += 1
                if rho < eta_shrink:
                    radius = max(min_radius, shrink_factor * radius)
                elif rho > eta_expand and step_norm >= 0.8 * best_record["radius"]:
                    radius = min(max_radius, expand_factor * radius)
                break

            radius = max(min_radius, shrink_factor * radius)
            if radius <= min_radius * (1.0 + 1e-12):
                status = "trust_region_exhausted"
                break

        if best_record is not None:
            history.append(best_record)

        if status in {"zero_step", "trust_region_exhausted"}:
            break
        if not accepted:
            status = "rejected"
            break
    else:
        status = "max_steps"

    final_obj, final_residual, _, _ = _theta_corrected_objective(
        x, y_target, theta, alpha
    )
    y = qru_output(x, theta, alpha)
    if final_obj <= objective_tol:
        status = "converged"

    accepted_rhos = [h["rho"] for h in history if h.get("accepted") and np.isfinite(h.get("rho", np.nan))]
    return {
        "alpha": alpha,
        "history": history,
        "prediction": y,
        "theta_corrected_objective": final_obj,
        "raw_mse": float(np.mean(final_residual**2)),
        "accepted_steps": int(accepted_steps),
        "status": status,
        "final_radius": float(radius),
        "median_accepted_rho": float(np.median(accepted_rhos)) if accepted_rhos else np.nan,
    }
