"""Falsifiable V3 pilot: nonlinear geometry and diagnosis on noisy windows.

This module uses the existing exact NumPy QRU. No hardware, stream detector,
quantum resource advantage, or globally optimal policy is assumed.
"""
import numpy as np
from scipy.stats import chi2

from .circuit import qru_output, random_theta, parameter_shift_prediction_jacobian
from .spectral import architecture_basis, fourier_design_matrix
from .controllability import spectral_jacobian
from .tomography import recover_paths_from_phase_cycle
from .adaptation import path_schedule_jacobian, augmented_diagnostics

ALPHA = np.array([1., 2., 4.])
SIGMA = .02
DEPTH = 3
W = 96
N_STEPS = 80


def reference(rng):
    x = 2*np.pi*np.arange(128)/128
    for _ in range(2000):
        theta = random_theta(3, rng)
        if np.max(np.abs(qru_output(x, theta, ALPHA))) < .55:
            return theta
    raise RuntimeError('Could not find bounded reference')


def projector_columns(J, tol=1e-8):
    U,s,_ = np.linalg.svd(J,full_matrices=False)
    rank=int(np.sum(s>tol*s[0])) if len(s) and s[0]>0 else 0
    return U[:,:rank]@U[:,:rank].T


def geometry(seed=4321, n_bases=20, n_dirs=40):
    """Both total and tangent-normal displacement from exact-Jacobian null directions."""
    rng=np.random.default_rng(seed)
    x=2*np.pi*np.arange(128)/128
    w,constant=architecture_basis(ALPHA)
    A,_=fourier_design_matrix(x,w,constant)
    spectral_map=np.linalg.pinv(A)
    rows=[]
    for base in range(n_bases):
        theta=reference(rng);c0=spectral_map@qru_output(x,theta,ALPHA)
        J=spectral_jacobian(theta,ALPHA,x,w,include_constant=constant)
        U,s,Vt=np.linalg.svd(J,full_matrices=True)
        rank=int(np.sum(s>1e-8*s[0]))
        normal=np.eye(len(U))-U[:,:rank]@U[:,:rank].T
        null=Vt[rank:,:]
        best=None
        for _ in range(n_dirs):
            v=rng.normal(size=len(null));v/=np.linalg.norm(v);v=v@null
            h=.03
            second=spectral_map@(qru_output(x,theta+h*v.reshape(3,3),ALPHA)+qru_output(x,theta-h*v.reshape(3,3),ALPHA)-2*qru_output(x,theta,ALPHA))/h**2
            score=np.linalg.norm(second)
            if best is None or score>best[0]:best=(score,v,second)
        _,v,second=best
        for radius in (.015,.03,.06,.12,.24,.48):
            target=spectral_map@qru_output(x,theta+radius*v.reshape(3,3),ALPHA)
            delta=target-c0
            rows.append(dict(base=base,radius=radius,rank=rank,total_norm=np.linalg.norm(delta),normal_norm=np.linalg.norm(normal@delta),
                             second_norm=np.linalg.norm(second),second_normal_norm=np.linalg.norm(normal@second),
                             kernel_error=np.linalg.norm(J@v),witness_param_norm=radius))
    return rows


def calibrated_step(builder,base_x,target_rms):
    lo,hi=0.,.01
    ref=builder(0.,base_x)
    while np.sqrt(np.mean((builder(hi,base_x)-ref)**2)) < target_rms and hi < 2:
        hi*=2
    for _ in range(28):
        mid=(lo+hi)/2
        if np.sqrt(np.mean((builder(mid,base_x)-ref)**2))<target_rms:lo=mid
        else:hi=mid
    return (lo+hi)/2


def target_fn(theta,case,rng):
    """Construct drift using only a distinct calibration grid; test x are unseen."""
    x_ref=2*np.pi*(np.arange(160)+.123)/160
    if case=='none':return lambda x:qru_output(x,theta,ALPHA)
    if case in ('parameter','schedule','mixed'):
        dt=rng.normal(size=(3,3));dt/=np.linalg.norm(dt)
        da=rng.normal(size=3);da/=np.linalg.norm(da)
        if case=='parameter':
            b=lambda t,x:qru_output(x,theta+t*dt,ALPHA)
            step=calibrated_step(b,x_ref,.06)
            return lambda x:qru_output(x,theta+step*dt,ALPHA)
        if case=='schedule':
            b=lambda t,x:qru_output(x,theta,ALPHA+t*da)
            step=calibrated_step(b,x_ref,.06)
            return lambda x:qru_output(x,theta,ALPHA+step*da)
        b=lambda t,x:qru_output(x,theta+t*dt,ALPHA+t*da)
        step=calibrated_step(b,x_ref,.06)
        return lambda x:qru_output(x,theta+step*dt,ALPHA+step*da)
    if case=='same_support_normal':
        w,c=architecture_basis(ALPHA)
        A,_=fourier_design_matrix(x_ref,w,c)
        J=spectral_jacobian(theta,ALPHA,x_ref,w,include_constant=c)
        U,s,_=np.linalg.svd(J,full_matrices=True)
        v=U[:,int(np.sum(s>1e-8*s[0]))]
        scale=.06/np.sqrt(np.mean((A@v)**2))
        return lambda x:qru_output(x,theta,ALPHA)+scale*(fourier_design_matrix(x,w,c)[0]@v)
    if case=='outside':
        return lambda x:qru_output(x,theta,ALPHA)+.06*np.sqrt(2)*np.sin(12*x)
    raise ValueError(case)


def features(theta,x,y,sigma=SIGMA):
    residual=y-qru_output(x,theta,ALPHA)
    J=parameter_shift_prediction_jacobian(x,theta,ALPHA)
    Jalpha=x[:,None]*J[:,2::3]
    w,c=architecture_basis(ALPHA);A,_=fourier_design_matrix(x,w,c)
    r_out=residual-A@np.linalg.lstsq(A,residual,rcond=None)[0]
    diag=augmented_diagnostics(residual,J,Jalpha)
    ss=np.sum(residual**2)
    out_excess=max(0.,(np.sum(r_out**2)-sigma**2*(len(x)-np.linalg.matrix_rank(A)))/len(x))
    added_energy=(diag['architecture_gain']*ss)/len(x)
    return dict(rms=np.sqrt(ss/len(x)),out_excess=out_excess,added_energy=added_energy,
                e_theta=diag['e_theta'],e_aug=diag['e_aug'],G_arch=diag['architecture_gain'],
                support_p=chi2.sf(np.sum(r_out**2)/sigma**2, len(x)-np.linalg.matrix_rank(A)))


def fit_adam(theta0,x,y,joint=False,steps=N_STEPS,lr_theta=.04,lr_alpha=.02,alpha_penalty=0.):
    """Both updates use precisely the same circuit evaluations per iteration."""
    theta=theta0.copy();alpha=ALPHA.copy()
    m=np.zeros(12 if joint else 9);v=m.copy()
    calls=0
    for k in range(1,steps+1):
        f=qru_output(x,theta,alpha);calls+=len(x)
        J=parameter_shift_prediction_jacobian(x,theta,alpha);calls+=2*theta.size*len(x)
        jac=np.column_stack([J,x[:,None]*J[:,2::3]]) if joint else J
        gradient=2/len(x)*jac.T@(f-y)
        if joint and alpha_penalty:
            gradient[9:]+=2*alpha_penalty*(alpha-ALPHA)
        m=.9*m+.1*gradient;v=.999*v+.001*gradient**2
        update=(m/(1-.9**k))/(np.sqrt(v/(1-.999**k))+1e-8)
        theta=(theta.ravel()-lr_theta*update[:9]).reshape((3,3))
        if joint:alpha-=lr_alpha*update[9:]
    return theta,alpha,calls


def _one(case,index,rng):
    theta=reference(rng);target=target_fn(theta,case,rng)
    x=rng.uniform(0,2*np.pi,W);y=target(x)+rng.normal(0,SIGMA,W)
    f=features(theta,x,y)
    return theta,x,y,f,target


def choose_alpha_penalty(seed=807,replicates=12):
    """Pick a single regularization strength without looking at evaluation instances."""
    rng=np.random.default_rng(seed)
    candidates=(0.,.002,.01,.05)
    rows=[]
    for case in ('parameter','schedule','mixed'):
        for i in range(replicates):
            theta,x,y,_,target=_one(case,i,rng)
            val_x=2*np.pi*(np.arange(192)+.371)/192
            val_y=target(val_x)
            for lam in candidates:
                t,a,_=fit_adam(theta,x,y,joint=True,alpha_penalty=lam)
                mse=float(np.mean((qru_output(val_x,t,a)-val_y)**2))
                rows.append(dict(case=case,replicate=i,alpha_penalty=lam,validation_mse=mse))
    scores={lam:np.mean([np.log10(row['validation_mse']+1e-7) for row in rows if row['alpha_penalty']==lam]) for lam in candidates}
    return min(scores,key=scores.get),rows,scores


def window_benchmark(seed=805,calibration_per_class=50,replicates=20,policy='discovery',alpha_penalty=None):
    """Calibrate on disjoint instances, then compare equal-evaluation update rules."""
    rng=np.random.default_rng(seed)
    calibration=[]
    for case in ('none','parameter'):
        for i in range(calibration_per_class):
            _,_,_,f,_=_one(case,i,rng)
            calibration.append(dict(case=case,index=i,**f))
    null=[z for z in calibration if z['case']=='none']
    controls=calibration
    thresholds={
        'rms':float(np.quantile([z['rms'] for z in null],.95)),
        'out_excess':float(np.quantile([z['out_excess'] for z in controls],.95)),
        'added_energy':float(np.quantile([z['added_energy'] for z in controls],.95)),
    }
    cases=('none','parameter','schedule','mixed','same_support_normal','outside')
    rows=[]
    for case in cases:
        for i in range(replicates):
            theta,x,y,f,target=_one(case,i,rng)
            if f['rms']<=thresholds['rms']:action='none'
            elif f['out_excess']>thresholds['out_excess'] and f['added_energy']>thresholds['added_energy']:
                if policy=='discovery' or (f['e_aug']<.8 and f['G_arch']/max(f['e_theta']**2,1e-15)>.10):
                    action='joint'
                else:
                    action='search_needed'
            else:action='theta'
            test_x=np.linspace(0,2*np.pi,256,endpoint=False)+.001
            test_y=target(test_x)
            t_theta,a_theta,c_theta=fit_adam(theta,x,y,False)
            t_joint,a_joint,c_joint=fit_adam(theta,x,y,True)
            if alpha_penalty is not None:
                t_reg,a_reg,c_reg=fit_adam(theta,x,y,True,alpha_penalty=alpha_penalty)
            original=float(np.mean((qru_output(test_x,theta,ALPHA)-test_y)**2))
            theta_mse=float(np.mean((qru_output(test_x,t_theta,a_theta)-test_y)**2))
            joint_mse=float(np.mean((qru_output(test_x,t_joint,a_joint)-test_y)**2))
            # Both policies use the same amplitude gate, so no-drift is not
            # an artificially easy advantage over an unconditional update.
            gate_active=f['rms']>thresholds['rms']
            diagnostics_calls=(1+2*theta.size)*len(x)
            row=dict(case=case,replicate=i,action=action,initial_test_mse=original,
                             theta_test_mse=theta_mse,joint_test_mse=joint_mse,
                             gated_joint_test_mse=joint_mse if gate_active else original,
                             selective_test_mse={'none':original,'theta':theta_mse,'joint':joint_mse,'search_needed':theta_mse}[action],
                             joint_alpha_shift=float(np.linalg.norm(a_joint-ALPHA)),
                             gated_joint_alpha_shift=float(np.linalg.norm(a_joint-ALPHA)) if gate_active else 0.,
                             selective_alpha_shift=float(np.linalg.norm(a_joint-ALPHA)) if action=='joint' else 0.,
                             theta_circuit_evaluations=c_theta,joint_circuit_evaluations=c_joint,
                             gated_joint_circuit_evaluations=len(x)+(c_joint if gate_active else 0),
                             selective_circuit_evaluations=diagnostics_calls+(c_joint if action=='joint' else c_theta if action!='none' else 0),**f)
            if alpha_penalty is not None:
                reg_mse=float(np.mean((qru_output(test_x,t_reg,a_reg)-test_y)**2))
                row.update(regularized_joint_test_mse=reg_mse if gate_active else original,
                           regularized_alpha_shift=float(np.linalg.norm(a_reg-ALPHA)) if gate_active else 0.,
                           regularized_circuit_evaluations=len(x)+(c_reg if gate_active else 0))
            rows.append(row)
    return calibration,thresholds,rows


def tomography_cost(seed=404,trials=12):
    rng=np.random.default_rng(seed)
    x=2*np.pi*(np.arange(128)+.15)/128;rows=[]
    for i in range(trials):
        theta=reference(rng)
        J=parameter_shift_prediction_jacobian(x,theta,ALPHA)
        rec=recover_paths_from_phase_cycle(theta,ALPHA,x,n_phase=5)
        path=path_schedule_jacobian(x,ALPHA,rec['amplitudes'])
        direct=x[:,None]*J[:,2::3]
        rows.append(dict(trial=i,relative_error=float(np.linalg.norm(path-direct)/np.linalg.norm(direct)),
                         theta_gradient_evaluations=2*theta.size*len(x),
                         incremental_direct_evaluations=0,
                         incremental_tomography_evaluations=5*len(x)))
    return rows
