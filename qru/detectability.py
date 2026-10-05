"""V5 collision fingerprint and future-risk witnesses (Gaussian scalar regression).

The Fourier comparator grants an amplitude pair AND an independent real carrier
velocity to every observed positive harmonic. It is a grouped-tone model,
not a classical latent-path model. No general quantum advantage is claimed.
"""
from collections import defaultdict

import numpy as np
from scipy.linalg import qr
from scipy.optimize import root, least_squares

from .circuit import qru_output, parameter_shift_prediction_jacobian
from .tomography import analytic_path_amplitudes
from .pilot_geometry import ALPHA, reference


def grouped_moments(theta, alpha, direction, digits=11):
    """C_w=sum a_s and M_w=sum (s·v)a_s, grouped by baseline frequency."""
    groups=defaultdict(list)
    for path,amp in analytic_path_amplitudes(theta).items():
        groups[round(float(np.dot(path,alpha)),digits)].append((path,amp))
    result=[]
    for freq,group in sorted(groups.items()):
        if freq<=0:continue
        c=sum(amp for _,amp in group)
        moment=sum(float(np.dot(path,direction))*amp for path,amp in group)
        result.append((freq,c,moment,len(group)))
    return result


def grouped_fourier_witness(theta, alpha, direction, grid=None):
    """Full-interval tangent error after granting ALL amplitudes and carriers.

    If M_w/C_w is not real, no model with a *single real carrier velocity per
    grouped harmonic* can reproduce the QRU derivative on any open interval.
    This is an exact linear-independence obstruction; the grid is a diagnostic.
    """
    if grid is None:grid=np.linspace(0.,2*np.pi,513)
    grid=np.asarray(grid,dtype=float)
    moments=grouped_moments(theta,alpha,direction)
    base=np.zeros(len(grid));columns=[];moment_norm=0.;obstruction=0.
    for freq,c,m,npaths in moments:
        cosine=np.cos(freq*grid);sine=np.sin(freq*grid)
        a=2*c.real;b=-2*c.imag
        base+=a*cosine+b*sine
        columns.extend([cosine,sine,grid*(-a*sine+b*cosine)])
        moment_norm+=abs(m)**2
        if abs(c)>1e-12:obstruction+=(np.imag(m*np.conj(c))/abs(c))**2
        elif abs(m)>1e-12:obstruction+=abs(m)**2
    jac=parameter_shift_prediction_jacobian(grid,theta,alpha)
    alpha_tangent=(grid[:,None]*jac[:,2::3])@direction
    d=np.stack(columns,axis=1)
    fitted=d@np.linalg.lstsq(d,alpha_tangent,rcond=1e-12)[0]
    return dict(signed_support=2*len(moments),collided_positive_tones=sum(n>1 for _,_,_,n in moments),
                baseline_error=float(np.max(abs(base-qru_output(grid,theta,alpha)))),
                moment_incompatibility=float(np.sqrt(obstruction/moment_norm)) if moment_norm else 0.,
                derivative_rms=float(np.sqrt(np.mean(alpha_tangent**2))),
                grouped_tangent_residual=float(np.sqrt(np.mean((alpha_tangent-fitted)**2))))


def finite_grouped_fit(theta, alpha, direction, delta=.001):
    """Locally refit all grouped amplitudes and real carriers after finite drift."""
    x=np.linspace(0.,2*np.pi,257)
    moments=grouped_moments(theta,alpha,direction)
    w=np.array([item[0] for item in moments])
    c=np.array([item[1] for item in moments])
    amplitudes=np.stack((2*c.real,-2*c.imag),axis=1).ravel()
    n=len(amplitudes)
    def classical(z):
        phase=np.outer(x,z[n:])
        return np.cos(phase)@z[:n:2]+np.sin(phase)@z[1:n:2]
    target=qru_output(x,theta,alpha+delta*direction)
    fit=least_squares(lambda z:classical(z)-target,np.r_[amplitudes,w],
                      max_nfev=1000,xtol=1e-13,ftol=1e-13,gtol=1e-13)
    actual=float(np.sqrt(np.mean(fit.fun**2)))
    predicted=delta*grouped_fourier_witness(theta,alpha,direction,x)['grouped_tangent_residual']
    return dict(finite_shift=delta,finite_fit_rms=actual,
                finite_linear_prediction=predicted,
                finite_ratio=actual/predicted if predicted>1e-11 else None,
                finite_solver_success=bool(fit.success))


def matched_qru_pair(theta, alpha, displaced_alpha, inputs):
    """Locally solve for six theta coordinates matching six distinct observations."""
    inputs=np.asarray(inputs,dtype=float)
    if len(inputs)!=6 or len(np.unique(inputs))!=6:raise ValueError('six distinct inputs required')
    j=parameter_shift_prediction_jacobian(inputs,theta,displaced_alpha)
    cols=qr(j,pivoting=True)[2][:6]
    original=theta.ravel()
    def remap(z):
        flat=original.copy();flat[cols]=z
        return flat.reshape(theta.shape)
    old=qru_output(inputs,theta,alpha)
    solution=root(lambda z:qru_output(inputs,remap(z),displaced_alpha)-old,
                  original[cols],
                  jac=lambda z:parameter_shift_prediction_jacobian(inputs,remap(z),displaced_alpha)[:,cols],
                  tol=1e-11)
    if np.max(abs(solution.fun))>1e-9:raise RuntimeError('interpolation failed')
    return remap(solution.x),float(np.max(abs(solution.fun))),bool(solution.success)


def matched_grouped_classical_pair(theta, alpha, displaced_alpha, inputs, future_grid):
    """Four baseline tones, six adjustable amplitudes, two frozen amplitudes."""
    tags=np.array([[-1,-1,1],[1,-1,1],[-1,1,1],[1,1,1]])
    w0=tags@alpha;w1=tags@displaced_alpha
    def basis(x,w):
        p=np.outer(x,w)
        return np.stack([fn(p[:,j]) for j in range(4)
                         for fn in (np.cos,np.sin)],axis=1)
    old_basis=basis(future_grid,w0)
    beta=np.linalg.lstsq(old_basis,qru_output(future_grid,theta,alpha),rcond=None)[0]
    new_basis=basis(inputs,w1)
    cols=qr(new_basis,pivoting=True)[2][:6]
    target=qru_output(inputs,theta,alpha)
    updated=beta.copy()
    updated[cols]+=np.linalg.solve(new_basis[:,cols],target-new_basis@beta)
    broad_gap=qru_output(future_grid,theta,alpha)-basis(future_grid,w1)@updated
    return dict(baseline_error=float(np.max(abs(old_basis@beta-qru_output(future_grid,theta,alpha)))),
                six_input_error=float(np.max(abs(new_basis@updated-target))),
                future_mse_gap=float(np.mean(broad_gap**2)))


def stream_witness(seed=101, sigma=.02, displacement=.015, width=.5):
    """Two Gaussian streams indistinguishable on six inputs, with future gap."""
    rng=np.random.default_rng(seed)
    theta=reference(rng)
    v=rng.normal(size=3);v/=np.linalg.norm(v)
    alpha1=ALPHA+displacement*v
    x6=np.pi+width*np.linspace(-.5,.5,6)
    x7=np.pi+width*np.linspace(-.5,.5,7)
    broad=np.linspace(0,2*np.pi,513)
    moved,residual,success=matched_qru_pair(theta,ALPHA,alpha1,x6)
    results=dict(seed=seed,solver_success=success,six_input_max_error=residual,
                 delta_alpha_norm=displacement,
                 theta_change_norm=float(np.linalg.norm(moved-theta)),
                 sigma=sigma,center=float(np.pi),width=width)
    future_gap=qru_output(broad,theta,ALPHA)-qru_output(broad,moved,alpha1)
    future_mse=float(np.mean(future_gap**2))
    results['future_mse_gap']=future_mse
    results['minimax_risk_floor_six_inputs']=future_mse/4  # TV=0 in the exact solution
    classic=matched_grouped_classical_pair(theta,ALPHA,alpha1,x6,broad)
    results.update({'classical_'+key:value for key,value in classic.items()})
    for label,inputs in [('six',x6),('seven',x7),('broad',broad)]:
        gap=qru_output(inputs,theta,ALPHA)-qru_output(inputs,moved,alpha1)
        mse=float(np.mean(gap**2))
        results[f'{label}_input_mse_gap']=mse
        results[f'{label}_n_for_kl_one']=(float(2*sigma**2/mse) if mse>1e-25
                                         else 'infinite (exact six-point match)')
    # This is a different hypothesis class: teacher theta is *known and fixed*.
    gap=qru_output(x6,theta,ALPHA)-qru_output(x6,theta,alpha1)
    mse=float(np.mean(gap**2))
    results['fixed_theta_n_for_kl_one']=float(2*sigma**2/mse)
    return results


def experiment(first_seed=101,instances=12):
    pairs=[stream_witness(first_seed+i) for i in range(instances)]
    fingerprints=[]
    for i in range(instances):
        rng=np.random.default_rng(first_seed+i)
        theta=reference(rng)
        v=rng.normal(size=3);v/=np.linalg.norm(v)
        for name,alpha in [('dyadic',ALPHA),('incommensurate',np.array([1.,np.sqrt(2),np.pi]))]:
            fingerprints.append(dict(seed=first_seed+i,schedule=name,
                                     **grouped_fourier_witness(theta,alpha,v),
                                     **(finite_grouped_fit(theta,alpha,v) if name=='dyadic' else {})))
    return pairs,fingerprints
