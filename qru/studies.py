"""High-level experiment runners used by the consolidated research notebook."""

from pathlib import Path
import numpy as np
import pandas as pd

from .results_io import read_result

from .circuit import qru_output, random_theta, parameter_shift_prediction_jacobian
from .spectral import (
    candidate_frequencies,
    architecture_basis,
    fit_real_spectrum,
    forbidden_energy_ratio_integer,
    fourier_design_matrix,
    spectral_state,
    support_residual,
)
from .controllability import (
    spectral_jacobian,
    svd_diagnostics,
    inaccessible_fraction,
    linearization_error,
)
from .tomography import (
    analytic_path_amplitudes,
    reconstruct_from_paths,
    grouped_path_coefficients,
    frequency_groups,
    collision_groups,
    balanced_ternary_codes,
    phase_alias_groups,
    minimum_phase_samples,
    fit_complex_spectrum,
    recover_paths_from_phase_cycle,
    relative_path_error,
    max_absolute_path_error,
)
from .adaptation import (
    path_schedule_jacobian,
    finite_difference_schedule_jacobian,
    augmented_diagnostics,
    trust_region_path_schedule_surgery,
)
from .training import mse, fit_theta_adam, find_reference_theta


def _results_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_spectral_audit(results_dir, seed=7, trials=50):
    results_dir = _results_dir(results_dir)
    rng = np.random.default_rng(seed)
    x = 2 * np.pi * np.arange(256) / 256
    cases = {
        "L1 [1]": [1.0],
        "L2 [1,2]": [1.0, 2.0],
        "L3 [1,2,4]": [1.0, 2.0, 4.0],
        "L3 [1,2,3]": [1.0, 2.0, 3.0],
        "L3 [1,sqrt(2),3]": [1.0, np.sqrt(2), 3.0],
        "L4 [1,2,4,8]": [1.0, 2.0, 4.0, 8.0],
    }

    rows = []
    for case, values in cases.items():
        alpha = np.asarray(values, dtype=float)
        loose = candidate_frequencies(alpha, architecture_aware=False)
        aware = candidate_frequencies(alpha, architecture_aware=True)
        freqs, include_constant = architecture_basis(alpha)
        integer_schedule = np.allclose(alpha, np.round(alpha))
        for trial in range(trials):
            theta = random_theta(len(alpha), rng)
            y = qru_output(x, theta, alpha)
            fit = fit_real_spectrum(x, y, freqs, include_constant=include_constant)
            forbidden = np.nan
            if integer_schedule:
                forbidden = forbidden_energy_ratio_integer(
                    x, y, aware, max_abs_frequency=np.max(np.abs(loose))
                )
            rows.append({
                "case": case,
                "depth": len(alpha),
                "loose_modes": len(loose),
                "aware_modes": len(aware),
                "has_zero": include_constant,
                "trial": trial,
                "residual": fit["relative_residual"],
                "condition_A": fit["condition_number"],
                "forbidden_energy": forbidden,
            })

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["case", "depth", "loose_modes", "aware_modes", "has_zero"], as_index=False)
        .agg(
            median_residual=("residual", "median"),
            max_residual=("residual", "max"),
            median_condition=("condition_A", "median"),
            max_forbidden_energy=("forbidden_energy", "max"),
        )
    )
    raw.to_csv(results_dir / "01_spectral_audit_trials.csv", index=False)
    summary.to_csv(results_dir / "01_spectral_audit.csv", index=False)
    return {"raw": raw, "summary": summary}


def load_spectral_audit(results_dir):
    r = Path(results_dir)
    return {
        "raw": read_result(r, "01_spectral_audit_trials.csv"),
        "summary": read_result(r, "01_spectral_audit.csv"),
    }


def run_controllability_study(results_dir, seed=20260920):
    results_dir = _results_dir(results_dir)
    alpha = np.array([1.0, 2.0, 4.0])
    x = 2 * np.pi * np.arange(128) / 128
    freqs, include_constant = architecture_basis(alpha)
    A, _ = fourier_design_matrix(x, freqs, include_constant)

    # Rank landscape.
    rng = np.random.default_rng(seed + 1)
    rank_rows = []
    for sample in range(1000):
        theta = random_theta(3, rng)
        y = qru_output(x, theta, alpha)
        d = svd_diagnostics(spectral_jacobian(theta, alpha, x, freqs, include_constant=include_constant))
        s, rank = d["singular_values"], d["effective_rank"]
        sigma_min = s[rank - 1] if rank else 0.0
        rank_rows.append({
            "sample": sample,
            "rank": rank,
            "sigma_max": s[0] if len(s) else 0.0,
            "sigma_min_nonzero": sigma_min,
            "condition_nonzero": s[0] / sigma_min if sigma_min > 0 else np.inf,
            "max_abs_output": np.max(np.abs(y)),
        })
    rank_df = pd.DataFrame(rank_rows)

    # Local linearization at an unconstrained-rank reference point.
    rng = np.random.default_rng(seed + 2)
    theta_ref = find_reference_theta(
        alpha, x, freqs, rng, include_constant=include_constant,
        max_abs_output=0.65, required_rank=None,
    )
    direction = rng.normal(size=theta_ref.shape)
    direction /= np.linalg.norm(direction)
    scales = np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 2e-1, 4e-1])
    lin_df = pd.DataFrame([
        {
            "norm_delta_theta": scale,
            "relative_linearization_error": linearization_error(
                theta_ref, scale * direction, alpha, x, freqs,
                include_constant=include_constant,
            )["relative_error"],
        }
        for scale in scales
    ])

    # Same-support stress test.
    rng = np.random.default_rng(seed + 3)
    n_bases, repeats = 4, 2
    rho_values = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    delta_norms = np.array([0.01, 0.02, 0.04, 0.08, 0.12])
    fixed_budget, lr, max_base_output = 50, 0.04, 0.65
    stress_rows = []
    for base in range(n_bases):
        theta0 = find_reference_theta(
            alpha, x, freqs, rng, include_constant=include_constant,
            max_abs_output=max_base_output,
        )
        y0 = qru_output(x, theta0, alpha)
        c0 = spectral_state(x, y0, freqs, include_constant=include_constant)
        J0 = spectral_jacobian(theta0, alpha, x, freqs, include_constant=include_constant)
        d0 = svd_diagnostics(J0)
        rank0 = d0["effective_rank"]
        Uc = d0["U"][:, :rank0]
        P = Uc @ Uc.T
        U_perp = d0["U"][:, rank0:]
        for delta_norm in delta_norms:
            for rho in rho_values:
                for repeat in range(repeats):
                    tangent = J0 @ rng.normal(size=theta0.size)
                    tangent /= np.linalg.norm(tangent)
                    perpendicular = U_perp @ rng.normal(size=U_perp.shape[1])
                    perpendicular /= np.linalg.norm(perpendicular)
                    delta_c = delta_norm * (np.sqrt(1 - rho**2) * tangent + rho * perpendicular)
                    y_target = A @ (c0 + delta_c)
                    residual_c = (np.eye(len(delta_c)) - P) @ delta_c
                    floor = float(np.mean((A @ residual_c) ** 2))
                    fit = fit_theta_adam(theta0, alpha, x, y_target, steps=fixed_budget, lr=lr)
                    initial = float(fit["history"][0])
                    stress_rows.append({
                        "base": base,
                        "repeat": repeat,
                        "rank": rank0,
                        "delta_norm": delta_norm,
                        "rho": rho,
                        "D_ctrl": inaccessible_fraction(delta_c, J0),
                        "support_residual": support_residual(x, y_target, freqs, include_constant=include_constant),
                        "max_abs_target": np.max(np.abs(y_target)),
                        "initial_mse": initial,
                        "final_mse": float(fit["final_mse"]),
                        "final_over_initial": float(fit["final_mse"] / (initial + 1e-15)),
                        "predicted_floor_mse": floor,
                        "predicted_floor_over_initial": floor / (initial + 1e-15),
                    })
    stress = pd.DataFrame(stress_rows)
    stress_summary = stress.groupby(["delta_norm", "rho"], as_index=False).agg(
        median_D_ctrl=("D_ctrl", "median"),
        median_initial_mse=("initial_mse", "median"),
        median_final_mse=("final_mse", "median"),
        median_final_over_initial=("final_over_initial", "median"),
        median_predicted_floor_over_initial=("predicted_floor_over_initial", "median"),
        q25_final_over_initial=("final_over_initial", lambda z: z.quantile(.25)),
        q75_final_over_initial=("final_over_initial", lambda z: z.quantile(.75)),
        max_support_residual=("support_residual", "max"),
    )
    corr_by_delta = pd.DataFrame([
        {
            "delta_norm": d,
            "corr_Dctrl_vs_normalized_final": np.corrcoef(g["D_ctrl"], g["final_over_initial"])[0, 1],
            "corr_predicted_floor_vs_normalized_final": np.corrcoef(g["predicted_floor_over_initial"], g["final_over_initial"])[0, 1],
            "median_abs_gap_to_floor": np.median(np.abs(g["final_over_initial"] - g["predicted_floor_over_initial"])),
        }
        for d, g in stress.groupby("delta_norm")
    ])

    # Budget sweep.
    rng = np.random.default_rng(seed + 4)
    budgets = np.array([10, 25, 50, 100, 250, 500])
    budget_rows = []
    for base in range(4):
        theta0 = find_reference_theta(
            alpha, x, freqs, rng, include_constant=include_constant,
            max_abs_output=max_base_output,
        )
        y0 = qru_output(x, theta0, alpha)
        c0 = spectral_state(x, y0, freqs, include_constant=include_constant)
        J0 = spectral_jacobian(theta0, alpha, x, freqs, include_constant=include_constant)
        d0 = svd_diagnostics(J0)
        rank0 = d0["effective_rank"]
        Uc = d0["U"][:, :rank0]
        P = Uc @ Uc.T
        U_perp = d0["U"][:, rank0:]
        for rho in (0.0, 0.5, 1.0):
            for repeat in range(2):
                tangent = J0 @ rng.normal(size=theta0.size)
                tangent /= np.linalg.norm(tangent)
                perpendicular = U_perp @ rng.normal(size=U_perp.shape[1])
                perpendicular /= np.linalg.norm(perpendicular)
                delta_c = 0.04 * (np.sqrt(1 - rho**2) * tangent + rho * perpendicular)
                y_target = A @ (c0 + delta_c)
                residual_c = (np.eye(len(delta_c)) - P) @ delta_c
                floor = float(np.mean((A @ residual_c) ** 2))
                fit = fit_theta_adam(theta0, alpha, x, y_target, steps=500, lr=lr)
                initial = float(fit["history"][0])
                for budget in budgets:
                    value = float(fit["history"][budget]) if budget < 500 else float(fit["final_mse"])
                    budget_rows.append({
                        "base": base,
                        "repeat": repeat,
                        "rho": rho,
                        "D_ctrl": inaccessible_fraction(delta_c, J0),
                        "budget": int(budget),
                        "initial_mse": initial,
                        "mse": value,
                        "mse_over_initial": value / (initial + 1e-15),
                        "predicted_floor_mse": floor,
                        "predicted_floor_over_initial": floor / (initial + 1e-15),
                    })
    budget_df = pd.DataFrame(budget_rows)
    budget_summary = budget_df.groupby(["rho", "budget"], as_index=False).agg(
        median_D_ctrl=("D_ctrl", "median"),
        median_mse=("mse", "median"),
        median_mse_over_initial=("mse_over_initial", "median"),
        q25_ratio=("mse_over_initial", lambda z: z.quantile(.25)),
        q75_ratio=("mse_over_initial", lambda z: z.quantile(.75)),
        median_floor_ratio=("predicted_floor_over_initial", "median"),
    )

    # Structural control.
    rng = np.random.default_rng(seed + 5)
    struct_rows = []
    for base in range(4):
        theta0 = find_reference_theta(
            alpha, x, freqs, rng, include_constant=include_constant,
            max_abs_output=max_base_output,
        )
        y0 = qru_output(x, theta0, alpha)
        y_target = y0 + 0.08 * np.sin(6 * x)
        fit = fit_theta_adam(theta0, alpha, x, y_target, steps=200, lr=lr)
        struct_rows.append({
            "base": base,
            "support_residual": support_residual(x, y_target, freqs, include_constant=include_constant),
            "initial_mse": float(fit["history"][0]),
            "final_mse": float(fit["final_mse"]),
            "final_over_initial": float(fit["final_mse"] / (fit["history"][0] + 1e-15)),
        })
    structural = pd.DataFrame(struct_rows)

    theta_zero = np.zeros((3, 3))
    rank_zero = svd_diagnostics(spectral_jacobian(theta_zero, alpha, x, freqs, include_constant=include_constant))["effective_rank"]
    summary = pd.DataFrame([{
        "random_theta_samples": 1000,
        "fraction_rank_6": float(np.mean(rank_df["rank"] == 6)),
        "rank_at_theta_zero": rank_zero,
        "median_sigma_min_nonzero": rank_df["sigma_min_nonzero"].median(),
        "linearization_error_at_1e-2": lin_df.loc[np.isclose(lin_df["norm_delta_theta"], 1e-2), "relative_linearization_error"].iloc[0],
        "linearization_error_at_1e-1": lin_df.loc[np.isclose(lin_df["norm_delta_theta"], 1e-1), "relative_linearization_error"].iloc[0],
        "stress_corr_predicted_floor_vs_observed_ratio": np.corrcoef(stress["predicted_floor_over_initial"], stress["final_over_initial"])[0, 1],
        "stress_max_same_support_residual": stress["support_residual"].max(),
        "median_ratio_after_500_updates_rho0": budget_df.loc[(budget_df["budget"] == 500) & np.isclose(budget_df["rho"], 0.0), "mse_over_initial"].median(),
        "median_ratio_after_500_updates_rho1": budget_df.loc[(budget_df["budget"] == 500) & np.isclose(budget_df["rho"], 1.0), "mse_over_initial"].median(),
        "median_structural_support_residual": structural["support_residual"].median(),
        "median_structural_final_over_initial": structural["final_over_initial"].median(),
    }])

    frames = {
        "rank": rank_df,
        "locality": lin_df,
        "stress": stress,
        "stress_summary": stress_summary,
        "correlations": corr_by_delta,
        "budget": budget_df,
        "budget_summary": budget_summary,
        "structural": structural,
        "summary": summary,
    }
    names = {
        "rank": "02_rank_landscape.csv",
        "locality": "02_locality_sweep.csv",
        "stress": "02_controllability_targets.csv",
        "stress_summary": "02_controllability_stress_summary.csv",
        "correlations": "02_controllability_correlations_by_delta.csv",
        "budget": "02_budget_sweep.csv",
        "budget_summary": "02_budget_sweep_summary.csv",
        "structural": "02_structural_targets.csv",
        "summary": "02_spectral_controllability.csv",
    }
    for key, df in frames.items():
        df.to_csv(results_dir / names[key], index=False)
    return frames


def load_controllability_study(results_dir):
    r = Path(results_dir)
    names = {
        "rank": "02_rank_landscape.csv",
        "locality": "02_locality_sweep.csv",
        "stress": "02_controllability_targets.csv",
        "stress_summary": "02_controllability_stress_summary.csv",
        "correlations": "02_controllability_correlations_by_delta.csv",
        "budget": "02_budget_sweep.csv",
        "budget_summary": "02_budget_sweep_summary.csv",
        "structural": "02_structural_targets.csv",
        "summary": "02_spectral_controllability.csv",
    }
    return {key: read_result(r, name) for key, name in names.items()}


def run_tomography_study(results_dir, seed=41):
    results_dir = _results_dir(results_dir)
    rng = np.random.default_rng(seed)
    x = 2 * np.pi * np.arange(256) / 256
    alpha_factor = np.array([1.0, 2.0, 3.0])

    factor_rows = []
    for sample in range(100):
        theta = random_theta(3, rng)
        loose = analytic_path_amplitudes(theta, architecture_aware=False)
        aware = analytic_path_amplitudes(theta, architecture_aware=True)
        direct = qru_output(x, theta, alpha_factor)
        paths = reconstruct_from_paths(x, alpha_factor, aware)
        factor_rows.append({
            "sample": sample,
            "max_forbidden_s1_zero": max(abs(a) for s, a in loose.items() if s[0] == 0),
            "relative_reconstruction_error": np.linalg.norm(direct - paths) / (np.linalg.norm(direct) + 1e-15),
            "max_abs_reconstruction_error": np.max(np.abs(direct - paths)),
        })
    factor = pd.DataFrame(factor_rows)

    schedules = {
        "harmonic_L3": np.array([1.0, 2.0, 3.0]),
        "dyadic_L3": np.array([1.0, 2.0, 4.0]),
        "dyadic_L4": np.array([1.0, 2.0, 4.0, 8.0]),
        "incommensurate_L3": np.array([1.0, np.sqrt(2), 3.0]),
    }
    complexity_rows = []
    for name, alpha in schedules.items():
        groups = frequency_groups(alpha, architecture_aware=True)
        q = balanced_ternary_codes(len(alpha))
        n_phase = minimum_phase_samples(alpha, q=q)
        spec = fit_complex_spectrum(
            x,
            qru_output(x, np.zeros((len(alpha), 3)), alpha),
            np.array(list(groups.keys()), dtype=float),
        )
        complexity_rows.append({
            "schedule": name,
            "L": len(alpha),
            "paths": sum(len(v) for v in groups.values()),
            "distinct_frequencies": len(groups),
            "max_collision_multiplicity": max(len(v) for v in groups.values()),
            "min_phase_settings": n_phase,
            "full_bias_grid": 3 ** len(alpha),
            "phase_setting_reduction": 3 ** len(alpha) / n_phase,
            "x_spectrum_condition": spec["condition_number"],
        })
    complexity = pd.DataFrame(complexity_rows)

    recovery_rows = []
    for name, alpha in schedules.items():
        n_phase = int(complexity.loc[complexity["schedule"] == name, "min_phase_settings"].iloc[0])
        for trial in range(50):
            theta = random_theta(len(alpha), rng)
            truth = analytic_path_amplitudes(theta, architecture_aware=True)
            rec = recover_paths_from_phase_cycle(theta, alpha, x, n_phase=n_phase, architecture_aware=True)
            recovery_rows.append({
                "schedule": name,
                "trial": trial,
                "n_phase": n_phase,
                "relative_path_error": relative_path_error(truth, rec["amplitudes"]),
                "max_absolute_path_error": max_absolute_path_error(truth, rec["amplitudes"]),
                "x_spectrum_condition": rec["spectral_condition_number"],
                "max_x_fit_residual": rec["max_spectral_residual"],
                "alias_classes": len(rec["aliases"]),
            })
    recovery = pd.DataFrame(recovery_rows)

    alpha_alias = np.array([1.0, 2.0, 4.0])
    theta_alias = random_theta(3, rng)
    truth_alias = analytic_path_amplitudes(theta_alias, architecture_aware=True)
    q_alias = balanced_ternary_codes(3)
    alias = pd.DataFrame([
        {
            "phase_settings": n,
            "alias_classes": len(phase_alias_groups(alpha_alias, n, q=q_alias)),
            "relative_path_error": relative_path_error(
                truth_alias,
                recover_paths_from_phase_cycle(theta_alias, alpha_alias, x, n_phase=n, q=q_alias)["amplitudes"],
            ),
        }
        for n in range(1, 11)
    ])

    target_schedules = {
        "dyadic": np.array([1.0, 2.0, 4.0]),
        "reconfigured": np.array([1.0, 1.0, 4.0]),
        "incommensurate": np.array([1.0, np.sqrt(2), 3.0]),
    }
    transport_rows = []
    for trial in range(40):
        theta = random_theta(3, rng)
        rec = recover_paths_from_phase_cycle(
            theta, alpha_factor, x, n_phase=minimum_phase_samples(alpha_factor)
        )
        for name, alpha_target in target_schedules.items():
            pred = reconstruct_from_paths(x, alpha_target, rec["amplitudes"])
            direct = qru_output(x, theta, alpha_target)
            transport_rows.append({
                "target_schedule": name,
                "trial": trial,
                "relative_output_error": np.linalg.norm(pred - direct) / (np.linalg.norm(direct) + 1e-15),
                "max_abs_output_error": np.max(np.abs(pred - direct)),
            })
    transport = pd.DataFrame(transport_rows)

    summary = pd.DataFrame([{
        "factorization_trials": 100,
        "max_forbidden_s1_zero_amplitude": factor["max_forbidden_s1_zero"].max(),
        "max_factorization_relative_error": factor["relative_reconstruction_error"].max(),
        "harmonic_L3_min_phase": int(complexity.loc[complexity["schedule"] == "harmonic_L3", "min_phase_settings"].iloc[0]),
        "dyadic_L3_min_phase": int(complexity.loc[complexity["schedule"] == "dyadic_L3", "min_phase_settings"].iloc[0]),
        "dyadic_L4_min_phase": int(complexity.loc[complexity["schedule"] == "dyadic_L4", "min_phase_settings"].iloc[0]),
        "max_well_conditioned_relative_path_error": recovery.loc[recovery["x_spectrum_condition"] < 10, "relative_path_error"].max(),
        "median_incommensurate_relative_path_error": recovery.loc[recovery["schedule"] == "incommensurate_L3", "relative_path_error"].median(),
        "max_transport_relative_output_error": transport["relative_output_error"].max(),
    }])

    frames = {"factor": factor, "complexity": complexity, "recovery": recovery, "alias": alias, "transport": transport, "summary": summary}
    names = {
        "factor": "03_factorization_stress.csv",
        "complexity": "03_phase_complexity.csv",
        "recovery": "03_path_recovery_stress.csv",
        "alias": "03_alias_sweep.csv",
        "transport": "03_spectral_transport.csv",
        "summary": "03_path_tomography.csv",
    }
    for key, df in frames.items():
        df.to_csv(results_dir / names[key], index=False)
    return frames


def load_tomography_study(results_dir):
    r = Path(results_dir)
    names = {
        "factor": "03_factorization_stress.csv",
        "complexity": "03_phase_complexity.csv",
        "recovery": "03_path_recovery_stress.csv",
        "alias": "03_alias_sweep.csv",
        "transport": "03_spectral_transport.csv",
        "summary": "03_path_tomography.csv",
    }
    return {key: read_result(r, name) for key, name in names.items()}


def run_reconfiguration_study(results_dir, seed=71):
    results_dir = _results_dir(results_dir)
    rng = np.random.default_rng(seed)
    alpha0 = np.array([1.0, 2.0, 4.0])
    x_diag = 2 * np.pi * np.arange(128) / 128
    x_train = 2 * np.pi * (np.arange(96) + 0.13) / 96
    x_val = 2 * np.pi * (np.arange(96) + 0.47) / 96
    x_test = 2 * np.pi * (np.arange(192) + 0.29) / 192
    freqs0, include_constant0 = architecture_basis(alpha0)
    train_steps, train_lr = 80, 0.04
    support_tol, arch_gain_tol, rel_arch_gain_tol = 1e-6, 1e-3, 0.10
    target_drift_rms = 0.04

    def reference_theta(max_abs=0.75):
        for _ in range(4000):
            theta = random_theta(3, rng)
            if np.max(np.abs(qru_output(x_diag, theta, alpha0))) < max_abs:
                return theta
        raise RuntimeError("No reference theta found.")

    def rms(v):
        v = np.asarray(v, dtype=float)
        return float(np.sqrt(np.mean(v**2)))

    def calibrate_scalar(builder, y0, target_rms, initial_hi=0.05, max_hi=2.0, n_bisect=45):
        lo, hi = 0.0, float(initial_hi)
        while rms(builder(hi) - y0) < target_rms and hi < max_hi:
            hi = min(max_hi, 2 * hi)
        if rms(builder(hi) - y0) < target_rms:
            return hi
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            if rms(builder(mid) - y0) < target_rms:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def decision_rule(support_mismatch, diag):
        if support_mismatch <= support_tol:
            return "theta_update"
        if (
            diag["architecture_gain"] >= arch_gain_tol
            and diag["relative_architecture_gain"] >= rel_arch_gain_tol
            and diag["e_aug"] < diag["e_theta"]
        ):
            return "schedule_surgery"
        return "global_architecture_search"

    def evaluate_target(name, theta_base, target_fn, alpha_truth=None, ground_truth_action=None):
        y0_diag = qru_output(x_diag, theta_base, alpha0)
        y_target_diag = np.asarray(target_fn(x_diag), dtype=float)
        delta = y_target_diag - y0_diag
        rec = recover_paths_from_phase_cycle(theta_base, alpha0, x_diag, n_phase=5)
        amps = rec["amplitudes"]
        J_theta = parameter_shift_prediction_jacobian(x_diag, theta_base, alpha0)
        J_alpha = path_schedule_jacobian(x_diag, alpha0, amps)
        diag = augmented_diagnostics(delta, J_theta, J_alpha)
        support = support_residual(x_diag, delta, freqs0, include_constant=include_constant0)
        action = decision_rule(support, diag)

        y_train = np.asarray(target_fn(x_train), dtype=float)
        y_val = np.asarray(target_fn(x_val), dtype=float)
        y_test = np.asarray(target_fn(x_test), dtype=float)
        theta_fit = fit_theta_adam(theta_base, alpha0, x_train, y_train, steps=train_steps, lr=train_lr)
        theta_val = mse(y_val, qru_output(x_val, theta_fit["theta"], alpha0))
        theta_test = mse(y_test, qru_output(x_test, theta_fit["theta"], alpha0))

        alpha_proposed = alpha0.copy()
        surgery = None
        s_only_val = s_only_test = s_tuned_val = s_tuned_test = np.nan
        selected_model, selected_val, selected_test = "theta_only", theta_val, theta_test
        if action == "schedule_surgery":
            surgery = trust_region_path_schedule_surgery(
                x_train, y_train, theta_base, alpha0, amps,
                n_steps=20, initial_radius=0.05, min_radius=1e-5, max_radius=0.25,
            )
            alpha_proposed = surgery["alpha"]
            s_only_val = mse(y_val, qru_output(x_val, theta_base, alpha_proposed))
            s_only_test = mse(y_test, qru_output(x_test, theta_base, alpha_proposed))
            tuned = fit_theta_adam(theta_base, alpha_proposed, x_train, y_train, steps=train_steps, lr=train_lr)
            s_tuned_val = mse(y_val, qru_output(x_val, tuned["theta"], alpha_proposed))
            s_tuned_test = mse(y_test, qru_output(x_test, tuned["theta"], alpha_proposed))
            candidates = {
                "theta_only": (theta_val, theta_test),
                "surgery_only": (s_only_val, s_only_test),
                "surgery_plus_theta": (s_tuned_val, s_tuned_test),
            }
            selected_model = min(candidates, key=lambda k: candidates[k][0])
            selected_val, selected_test = candidates[selected_model]

        return {
            "case": name,
            "ground_truth_action": ground_truth_action,
            "decision": action,
            "decision_correct": ground_truth_action is None or action == ground_truth_action,
            "support_delta_residual": support,
            "support_target_residual": support_residual(x_diag, y_target_diag, freqs0, include_constant=include_constant0),
            "Dperp_function": inaccessible_fraction(delta, J_theta),
            **diag,
            "drift_rms": rms(delta),
            "initial_mse_test": mse(y_test, qru_output(x_test, theta_base, alpha0)),
            "theta_only_val_mse": theta_val,
            "theta_only_test_mse": theta_test,
            "surgery_only_val_mse": s_only_val,
            "surgery_only_test_mse": s_only_test,
            "surgery_tuned_val_mse": s_tuned_val,
            "surgery_tuned_test_mse": s_tuned_test,
            "selected_model": selected_model,
            "selected_val_mse": selected_val,
            "selected_test_mse": selected_test,
            "alpha_error": np.nan if alpha_truth is None else float(np.linalg.norm(alpha_proposed - alpha_truth)),
            "alpha_truth_shift": np.nan if alpha_truth is None else float(np.linalg.norm(alpha_truth - alpha0)),
            "alpha_proposed_shift": float(np.linalg.norm(alpha_proposed - alpha0)),
            "surgery_status": None if surgery is None else surgery["status"],
            "surgery_accepted_steps": 0 if surgery is None else surgery["accepted_steps"],
            "surgery_median_rho": np.nan if surgery is None else surgery["median_accepted_rho"],
            "surgery_final_radius": np.nan if surgery is None else surgery["final_radius"],
        }

    # J_alpha validation.
    jac_rows = []
    for trial in range(50):
        theta = random_theta(3, rng)
        rec = recover_paths_from_phase_cycle(theta, alpha0, x_diag, n_phase=5)
        J_path = path_schedule_jacobian(x_diag, alpha0, rec["amplitudes"])
        J_fd = finite_difference_schedule_jacobian(x_diag, theta, alpha0)
        jac_rows.append({
            "trial": trial,
            "relative_Jalpha_error": np.linalg.norm(J_path - J_fd) / (np.linalg.norm(J_fd) + 1e-15),
            "path_recovery_condition": rec["spectral_condition_number"],
        })
    jac_df = pd.DataFrame(jac_rows)

    # Canonical cases.
    theta0 = reference_theta()
    direction = rng.normal(size=theta0.shape)
    direction /= np.linalg.norm(direction)
    theta_shift = theta0 + 0.05 * direction
    alpha_shift = alpha0 + np.array([0.0, 0.0, 0.12])
    canonical = pd.DataFrame([
        evaluate_target("parameter", theta0, lambda xx: qru_output(xx, theta_shift, alpha0), alpha_truth=alpha0, ground_truth_action="theta_update"),
        evaluate_target("schedule", theta0, lambda xx: qru_output(xx, theta0, alpha_shift), alpha_truth=alpha_shift, ground_truth_action="schedule_surgery"),
        evaluate_target("mixed", theta0, lambda xx: qru_output(xx, theta_shift, alpha_shift), alpha_truth=alpha_shift, ground_truth_action="schedule_surgery"),
    ])

    # Matched-RMS stress test.
    def matched_target(theta, drift_type):
        y0 = qru_output(x_diag, theta, alpha0)
        dtheta = rng.normal(size=theta.shape); dtheta /= np.linalg.norm(dtheta)
        dalpha = rng.normal(size=alpha0.shape); dalpha /= np.linalg.norm(dalpha)
        if drift_type == "parameter":
            st = calibrate_scalar(lambda s: qru_output(x_diag, theta + s * dtheta, alpha0), y0, target_drift_rms, 0.03, 0.6)
            return theta + st * dtheta, alpha0.copy(), dalpha
        if drift_type == "schedule":
            sa = calibrate_scalar(lambda s: qru_output(x_diag, theta, alpha0 + s * dalpha), y0, target_drift_rms, 0.03, 1.0)
            return theta.copy(), alpha0 + sa * dalpha, dalpha
        half = target_drift_rms / np.sqrt(2)
        st0 = calibrate_scalar(lambda s: qru_output(x_diag, theta + s * dtheta, alpha0), y0, half, 0.02, 0.5)
        sa0 = calibrate_scalar(lambda s: qru_output(x_diag, theta, alpha0 + s * dalpha), y0, half, 0.02, 0.8)
        lam = calibrate_scalar(lambda s: qru_output(x_diag, theta + s * st0 * dtheta, alpha0 + s * sa0 * dalpha), y0, target_drift_rms, 1.0, 2.0)
        return theta + lam * st0 * dtheta, alpha0 + lam * sa0 * dalpha, dalpha

    stress_rows = []
    for drift_type in ("parameter", "schedule", "mixed"):
        for trial in range(20):
            theta = reference_theta()
            theta_t, alpha_t, alpha_direction = matched_target(theta, drift_type)
            out = evaluate_target(
                drift_type,
                theta,
                lambda xx, tt=theta_t.copy(), aa=alpha_t.copy(): qru_output(xx, tt, aa),
                alpha_truth=alpha_t,
                ground_truth_action="theta_update" if drift_type == "parameter" else "schedule_surgery",
            )
            out["trial"] = trial
            out["theta_truth_shift"] = float(np.linalg.norm(theta_t-theta))
            out.update({f"alpha_dir_{j+1}": float(alpha_direction[j]) for j in range(3)})
            stress_rows.append(out)
    stress = pd.DataFrame(stress_rows)
    stress_summary = stress.groupby("case").agg(
        trials=("trial", "count"),
        median_drift_rms=("drift_rms", "median"),
        median_support_mismatch=("support_delta_residual", "median"),
        median_Dperp=("Dperp_function", "median"),
        median_e_theta=("e_theta", "median"),
        median_e_aug=("e_aug", "median"),
        median_arch_gain=("architecture_gain", "median"),
        median_relative_arch_gain=("relative_architecture_gain", "median"),
        decision_accuracy=("decision_correct", "mean"),
        median_theta_only_test_mse=("theta_only_test_mse", "median"),
        median_selected_test_mse=("selected_test_mse", "median"),
        max_selected_test_mse=("selected_test_mse", "max"),
        median_alpha_error=("alpha_error", "median"),
    ).reset_index()

    # Negative control 1.
    theta_ss = reference_theta(max_abs=0.65)
    y0_ss = qru_output(x_diag, theta_ss, alpha0)
    A_diag, _ = fourier_design_matrix(x_diag, freqs0, include_constant0)
    c0_ss = spectral_state(x_diag, y0_ss, freqs0, include_constant=include_constant0)
    Jc = spectral_jacobian(theta_ss, alpha0, x_diag, freqs0, include_constant=include_constant0)
    sv = svd_diagnostics(Jc); rank = sv["effective_rank"]
    v_c = sv["U"][:, rank]; v_c /= np.linalg.norm(v_c)
    scale_ss = target_drift_rms / (rms(A_diag @ v_c) + 1e-15)
    c_target_ss = c0_ss + scale_ss * v_c
    if np.max(np.abs(A_diag @ c_target_ss)) > 0.98:
        scale_ss *= 0.98 / np.max(np.abs(A_diag @ c_target_ss))
        c_target_ss = c0_ss + scale_ss * v_c
    same_support = evaluate_target(
        "same_support_high_Dperp",
        theta_ss,
        lambda xx: fourier_design_matrix(xx, freqs0, include_constant0)[0] @ c_target_ss,
        ground_truth_action="theta_update",
    )
    same_support["Dperp_spectral"] = inaccessible_fraction(scale_ss * v_c, Jc)

    # Negative control 2.
    theta_bad = reference_theta(max_abs=0.65)
    rec_bad = recover_paths_from_phase_cycle(theta_bad, alpha0, x_diag, n_phase=5)
    Jt_bad = parameter_shift_prediction_jacobian(x_diag, theta_bad, alpha0)
    Ja_bad = path_schedule_jacobian(x_diag, alpha0, rec_bad["amplitudes"])
    candidates = []
    for omega in range(8, 41):
        for phase in (0.0, 0.5 * np.pi):
            d = np.sin(omega * x_diag + phase); d /= rms(d)
            dg = augmented_diagnostics(d, Jt_bad, Ja_bad)
            sres = support_residual(x_diag, d, freqs0, include_constant=include_constant0)
            candidates.append((dg["e_aug"], -dg["relative_architecture_gain"], sres, omega, phase))
    best = max([c for c in candidates if c[2] > 0.95], key=lambda z: (z[0], z[1], z[2]))
    omega_bad, phase_bad = best[3], best[4]
    scale_bad = target_drift_rms / rms(np.sin(omega_bad * x_diag + phase_bad))
    unaddressable = evaluate_target(
        "structural_unaddressable",
        theta_bad,
        lambda xx: qru_output(xx, theta_bad, alpha0) + scale_bad * np.sin(omega_bad * xx + phase_bad),
        ground_truth_action="global_architecture_search",
    )
    unaddressable["omega_out"] = omega_bad

    controls = pd.DataFrame([
        {
            "case": "same_support_high_Dperp",
            "expected": "theta_update",
            "decision": same_support["decision"],
            "correct": same_support["decision_correct"],
            "support_mismatch": same_support["support_delta_residual"],
            "Dperp": same_support["Dperp_function"],
            "G_arch": same_support["architecture_gain"],
            "e_aug": same_support["e_aug"],
            "selected_test_mse": same_support["selected_test_mse"],
        },
        {
            "case": "structural_unaddressable",
            "expected": "global_architecture_search",
            "decision": unaddressable["decision"],
            "correct": unaddressable["decision_correct"],
            "support_mismatch": unaddressable["support_delta_residual"],
            "Dperp": unaddressable["Dperp_function"],
            "G_arch": unaddressable["architecture_gain"],
            "e_aug": unaddressable["e_aug"],
            "selected_test_mse": unaddressable["selected_test_mse"],
        },
    ])

    # Trust-region locality.
    locality_rows = []
    for mode in ("axis", "random"):
        for delta in (0.02, 0.05, 0.10, 0.20, 0.40, 0.70):
            for trial in range(15):
                theta = reference_theta()
                direction = np.array([0.0, 0.0, 1.0]) if mode == "axis" else rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                alpha_target = alpha0 + delta * direction
                dy = qru_output(x_diag, theta, alpha_target) - qru_output(x_diag, theta, alpha0)
                rec = recover_paths_from_phase_cycle(theta, alpha0, x_diag, n_phase=5)
                J_theta = parameter_shift_prediction_jacobian(x_diag, theta, alpha0)
                J_alpha = path_schedule_jacobian(x_diag, alpha0, rec["amplitudes"])
                diag = augmented_diagnostics(dy, J_theta, J_alpha)
                surgery = trust_region_path_schedule_surgery(
                    x_train,
                    qru_output(x_train, theta, alpha_target),
                    theta,
                    alpha0,
                    rec["amplitudes"],
                    n_steps=25,
                    initial_radius=0.05,
                    min_radius=1e-5,
                    max_radius=0.30,
                )
                pred = qru_output(x_test, theta, surgery["alpha"])
                truth = qru_output(x_test, theta, alpha_target)
                test_mse = mse(truth, pred)
                aerr = float(np.linalg.norm(surgery["alpha"] - alpha_target))
                locality_rows.append({
                    "mode": mode,
                    "delta_alpha": delta,
                    "trial": trial,
                    "support_delta_residual": support_residual(x_diag, dy, freqs0, include_constant=include_constant0),
                    **diag,
                    "alpha_error": aerr,
                    "test_mse": test_mse,
                    "success_function": test_mse < 1e-8,
                    "success_alpha": aerr < 1e-4,
                    "accepted_steps": surgery["accepted_steps"],
                    "status": surgery["status"],
                    "dir_1": direction[0], "dir_2": direction[1], "dir_3": direction[2],
                    "median_rho": surgery["median_accepted_rho"],
                    "final_radius": surgery["final_radius"],
                })
    locality = pd.DataFrame(locality_rows)
    locality_summary = locality.groupby(["mode", "delta_alpha"]).agg(
        trials=("trial", "count"),
        median_support_mismatch=("support_delta_residual", "median"),
        median_e_theta=("e_theta", "median"),
        median_e_aug=("e_aug", "median"),
        median_arch_gain=("architecture_gain", "median"),
        success_rate_function=("success_function", "mean"),
        success_rate_alpha=("success_alpha", "mean"),
        median_alpha_error=("alpha_error", "median"),
        max_alpha_error=("alpha_error", "max"),
        median_test_mse=("test_mse", "median"),
        median_rho=("median_rho", "median"),
        median_final_radius=("final_radius", "median"),
    ).reset_index()

    summary = pd.DataFrame([{
        "median_Jalpha_validation_error": jac_df["relative_Jalpha_error"].median(),
        "max_Jalpha_validation_error": jac_df["relative_Jalpha_error"].max(),
        "matched_parameter_decision_accuracy": stress.loc[stress["case"] == "parameter", "decision_correct"].mean(),
        "matched_schedule_decision_accuracy": stress.loc[stress["case"] == "schedule", "decision_correct"].mean(),
        "matched_mixed_decision_accuracy": stress.loc[stress["case"] == "mixed", "decision_correct"].mean(),
        "matched_parameter_median_drift_rms": stress.loc[stress["case"] == "parameter", "drift_rms"].median(),
        "matched_schedule_median_drift_rms": stress.loc[stress["case"] == "schedule", "drift_rms"].median(),
        "matched_mixed_median_drift_rms": stress.loc[stress["case"] == "mixed", "drift_rms"].median(),
        "same_support_Dperp_spectral": same_support["Dperp_spectral"],
        "same_support_decision_correct": same_support["decision_correct"],
        "unaddressable_support_mismatch": unaddressable["support_delta_residual"],
        "unaddressable_relative_arch_gain": unaddressable["relative_architecture_gain"],
        "unaddressable_decision_correct": unaddressable["decision_correct"],
        "axis_success_delta_0_1": locality_summary.loc[(locality_summary["mode"] == "axis") & np.isclose(locality_summary["delta_alpha"], 0.1), "success_rate_function"].iloc[0],
        "axis_success_delta_0_4": locality_summary.loc[(locality_summary["mode"] == "axis") & np.isclose(locality_summary["delta_alpha"], 0.4), "success_rate_function"].iloc[0],
        "random_success_delta_0_1": locality_summary.loc[(locality_summary["mode"] == "random") & np.isclose(locality_summary["delta_alpha"], 0.1), "success_rate_function"].iloc[0],
        "random_success_delta_0_4": locality_summary.loc[(locality_summary["mode"] == "random") & np.isclose(locality_summary["delta_alpha"], 0.4), "success_rate_function"].iloc[0],
    }])

    frames = {
        "Jalpha": jac_df,
        "canonical": canonical,
        "stress": stress,
        "stress_summary": stress_summary,
        "controls": controls,
        "locality": locality,
        "locality_summary": locality_summary,
        "summary": summary,
    }
    names = {
        "Jalpha": "04_Jalpha_validation.csv",
        "canonical": "04_controlled_cases.csv",
        "stress": "04_matched_drift_stress.csv",
        "stress_summary": "04_matched_drift_summary.csv",
        "controls": "04_negative_controls.csv",
        "locality": "04_trust_region_locality.csv",
        "locality_summary": "04_trust_region_locality_summary.csv",
        "summary": "04_retrain_vs_reconfigure.csv",
    }
    for key, df in frames.items():
        df.to_csv(results_dir / names[key], index=False)
    return frames


def load_reconfiguration_study(results_dir):
    r = Path(results_dir)
    names = {
        "Jalpha": "04_Jalpha_validation.csv",
        "canonical": "04_controlled_cases.csv",
        "stress": "04_matched_drift_stress.csv",
        "stress_summary": "04_matched_drift_summary.csv",
        "controls": "04_negative_controls.csv",
        "locality": "04_trust_region_locality.csv",
        "locality_summary": "04_trust_region_locality_summary.csv",
        "summary": "04_retrain_vs_reconfigure.csv",
    }
    frames = {key: read_result(r, name) for key, name in names.items()}
    if "success_rate_function" in frames["locality_summary"].columns:
        frames["locality_summary"] = frames["locality_summary"].rename(
            columns={"success_rate_function": "success_rate"}
        )
    return frames
