"""Two identical baseline QRU spectra with different latent path amplitudes.

The classical control is an equal *two-hypothesis* path-resolved Fourier model:
it uses the two exact path dictionaries and the same observations as the QRU.
It has no asserted computational disadvantage or quantum-exclusive gap.
"""
import numpy as np
from scipy.optimize import least_squares

from .circuit import qru_output, parameter_shift_prediction_jacobian
from .tomography import (analytic_path_amplitudes, grouped_path_coefficients,
                         reconstruct_from_paths, recover_paths_from_phase_cycle,
                         max_absolute_path_error)


BASE_ALPHA=np.array([1.,2.])


def exact_twin(delta=.02, sigma=.2):
    """Exact algebraic witness, not a fitted or generic-angle example.

    A: cos((alpha_1-alpha_2)x); B: cos(alpha_1*x), at zero biases.
    Both equal cos(x) when alpha=(1,2), but differ after alpha_2 drift.
    A has path tags (+1,-1), (-1,+1); B has (+1,0), (-1,0).
    """
    a=np.array([[0.,np.pi,0.],[0.,0.,0.]])
    b=np.array([[0.,np.pi/2,np.pi/2],[np.pi/2,0.,0.]])
    x=np.linspace(0.,2*np.pi,513)
    pa=analytic_path_amplitudes(a);pb=analytic_path_amplitudes(b)
    alpha_new=BASE_ALPHA+[0.,delta]
    ya=qru_output(x,a,alpha_new);yb=qru_output(x,b,alpha_new)
    error=float(max(np.max(abs(qru_output(x,a,BASE_ALPHA)-np.cos(x))),
                    np.max(abs(qru_output(x,b,BASE_ALPHA)-np.cos(x))),
                    np.max(abs(ya-np.cos((1+delta)*x))),
                    np.max(abs(yb-np.cos(x)))))
    if error>1e-12:raise AssertionError('Exact trigonometric witness changed')
    probe_x=np.pi/4
    aa=a.copy();bb=b.copy()
    aa[1,2]+=np.pi/2;bb[1,2]+=np.pi/2
    pa_probe=float(qru_output([probe_x],aa,BASE_ALPHA)[0])
    pb_probe=float(qru_output([probe_x],bb,BASE_ALPHA)[0])
    # One Z measurement has Bernoulli(+1) probability (1+f)/2.
    p=(1+pa_probe)/2;q=(1+pb_probe)/2
    shot_kl=float(p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q)))
    recovered=[recover_paths_from_phase_cycle(t,BASE_ALPHA,np.linspace(0,2*np.pi,65))
               for t in (a,b)]
    tomography_error=float(max(max_absolute_path_error(path,r['amplitudes'])
                               for path,r in zip((pa,pb),recovered)))
    return dict(theta_a=a.tolist(),theta_b=b.tolist(),alpha=BASE_ALPHA.tolist(),
                delta=delta,baseline_function='cos(x)',
                future_a='cos((1+delta)*x)',future_b='cos(x)',
                exact_identity_grid_max_error=error,
                future_rms=float(np.sqrt(np.mean((ya-yb)**2))),
                future_mse=float(np.mean((ya-yb)**2)),
                future_risk_lower_bound=float(np.mean((ya-yb)**2)/4),
                phase_probe_x=probe_x,phase_probe_shift=float(np.pi/2),
                phase_mean_a=pa_probe,phase_mean_b=pb_probe,
                phase_shot_kl_a_to_b=shot_kl,
                phase_gaussian_kl_per_sample=float((pa_probe-pb_probe)**2/(2*sigma**2)),
                baseline_recovery_phase_settings=recovered[0]['n_phase'],
                tomography_max_path_error=tomography_error,
                classical_path_forecast_max_error=float(max(
                    np.max(abs(reconstruct_from_paths(x,alpha_new,pa)-ya)),
                    np.max(abs(reconstruct_from_paths(x,alpha_new,pb)-yb)))))


def find_twin(theta, rng, attempts=10, spectral_tolerance=1e-9,
              minimum_path_gap=.03):
    """Multi-start inverse fit of the full baseline function, then path audit.

    Spectral equality is checked independently using exact Bloch path sums.
    Multi-start is an existence search, not a complete enumeration of roots.
    """
    theta=np.asarray(theta,dtype=float)
    x=np.linspace(0.,2*np.pi,65)
    target=qru_output(x,theta,BASE_ALPHA)
    origin=analytic_path_amplitudes(theta)
    base_grouped=grouped_path_coefficients(BASE_ALPHA,origin)
    best=None
    for trial in range(attempts):
        initial=rng.uniform(-np.pi,np.pi,size=theta.shape).ravel()
        fit=least_squares(lambda z:qru_output(x,z.reshape(theta.shape),BASE_ALPHA)-target,
                          initial,
                          jac=lambda z:parameter_shift_prediction_jacobian(x,z.reshape(theta.shape),BASE_ALPHA),
                          max_nfev=160,xtol=1e-12,ftol=1e-12,gtol=1e-12)
        other=fit.x.reshape(theta.shape)
        paths=analytic_path_amplitudes(other)
        grouped=grouped_path_coefficients(BASE_ALPHA,paths)
        spectral_error=max(abs(base_grouped[w]-grouped[w]) for w in base_grouped)
        path_gap=float(np.linalg.norm([origin[s]-paths[s] for s in origin]))
        if spectral_error<spectral_tolerance and path_gap>minimum_path_gap:
            candidate=(other,path_gap,float(spectral_error),trial+1)
            if best is None or path_gap>best[1]:best=candidate
    if best is None:raise RuntimeError('No spectrally identical path-distinct twin found')
    return best


def twin_witness(seed=198,attempts=10,delta=.02,sigma=.02):
    """Passive, phase-probed and transverse/longitudinal drift outcomes."""
    rng=np.random.default_rng(seed)
    first=rng.uniform(-2.,2.,size=(2,3))
    second,path_gap,coeff_error,successful_attempt=find_twin(first,rng,attempts)
    paths0=analytic_path_amplitudes(first)
    paths1=analytic_path_amplitudes(second)
    diff={tag:paths1[tag]-amp for tag,amp in paths0.items()}
    x=np.linspace(0.,2*np.pi,513)
    # This pair's nonzero path differences are at the +/-1 collision.
    positive=diff[(-1,1)]
    cancel_error=abs(diff[(-1,1)]+diff[(1,0)])
    if cancel_error>1e-8:raise AssertionError('The collision cancelation changed')
    base=np.max(abs(qru_output(x,first,BASE_ALPHA)-qru_output(x,second,BASE_ALPHA)))
    moved=BASE_ALPHA+np.array([0.,delta])
    future0=qru_output(x,first,moved)
    future1=qru_output(x,second,moved)
    future_rms=float(np.sqrt(np.mean((future0-future1)**2)))
    # A drift parallel to the collision-preserving ray (1,2) cannot split it.
    parallel=(1+delta)*BASE_ALPHA
    parallel_error=float(np.max(abs(qru_output(x,first,parallel)-qru_output(x,second,parallel))))
    # One fixed, predeclared intervention on the second encoding bias.
    phase0=first.copy();phase1=second.copy()
    phase0[1,2]+=np.pi/2;phase1[1,2]+=np.pi/2
    probe_x=np.pi
    phase_signal=float(qru_output([probe_x],phase1,BASE_ALPHA)[0]
                       -qru_output([probe_x],phase0,BASE_ALPHA)[0])
    # Numerically verify the oracle path-resolved classical predictions.
    classical_error=float(max(np.max(abs(reconstruct_from_paths(x,moved,paths0)-future0)),
                              np.max(abs(reconstruct_from_paths(x,moved,paths1)-future1))))
    return dict(seed=seed,alpha=list(BASE_ALPHA),drift_alpha=list(moved),
                theta_a=first.tolist(),theta_b=second.tolist(),
                grouped_spectrum_max_error=coeff_error,baseline_grid_max_error=float(base),
                path_l2_difference=path_gap,collision_path_difference_real=float(positive.real),
                collision_path_difference_imag=float(positive.imag),
                collision_cancellation_error=float(cancel_error),
                future_transverse_rms=future_rms,future_transverse_mse=future_rms**2,
                parallel_drift_max_error=parallel_error,
                phase_probe_x=float(probe_x),phase_probe_bias_shift=float(np.pi/2),
                phase_probe_mean_gap=phase_signal,
                phase_probe_kl_per_label=float(phase_signal**2/(2*sigma**2)),
                phase_probe_labels_kl_one=(float(2*sigma**2/phase_signal**2)
                                           if abs(phase_signal)>1e-12 else float('inf')),
                path_resolved_classical_max_error=classical_error,
                attempts=attempts,first_found_attempt=successful_attempt)


def experiment(seeds=range(198,210),attempts=10):
    return [twin_witness(seed,attempts) for seed in seeds]
