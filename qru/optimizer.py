"""Local QRU optimizer falsification: identical joint and Schur-complement steps.

All losses use the same observed batch, not a drift label. Evaluation counts
include each forward call and each of the 18 parameter-shift calls per step.
This is an exact simulator pilot, not a hardware/online cost claim.
"""
import numpy as np

from .circuit import qru_output, parameter_shift_prediction_jacobian
from .pilot_geometry import ALPHA, _one, reference, calibrated_step, SIGMA, W


def schur_step(Jtheta, Jalpha, residual, damping=1e-3, alpha_penalty=0., alpha_offset=None):
    """Solve the *same* damped joint quadratic two ways; return both steps.

    Objective: ||r-Jt dt-Ja da||^2/(2N) + damping/2 ||d||^2
               + alpha_penalty/2 ||alpha-alpha0+da||^2.
    Crucially the eliminated theta block uses a *ridge residual maker*, not
    the unregularized orthogonal projector used in some proposed formulas.
    """
    n=len(residual);T=Jtheta/np.sqrt(n);A=Jalpha/np.sqrt(n);r=residual/np.sqrt(n)
    offset=np.zeros(A.shape[1]) if alpha_offset is None else np.asarray(alpha_offset)
    B=T.T@T+damping*np.eye(T.shape[1]);C=T.T@A
    E=A.T@A+(damping+alpha_penalty)*np.eye(A.shape[1])
    bt=T.T@r;ba=A.T@r-alpha_penalty*offset
    full=np.block([[B,C],[C.T,E]])
    joint=np.linalg.solve(full,np.r_[bt,ba])
    S=E-C.T@np.linalg.solve(B,C)
    da=np.linalg.solve(S,ba-C.T@np.linalg.solve(B,bt))
    dt=np.linalg.solve(B,bt-C@da)
    return joint,np.r_[dt,da],float(np.linalg.cond(S))


def fit_lm(theta0,x,y,max_calls=150_000,alpha_penalty=0.,alpha_radius=None,
           damping=1e-3,method='joint',test=None):
    theta=theta0.copy();alpha=ALPHA.copy();calls=0;trace=[];failures=0
    test_x,test_y=test
    def snapshot():
        return dict(calls=calls,training_mse=float(np.mean((qru_output(x,theta,alpha)-y)**2)),
                    test_mse=float(np.mean((qru_output(test_x,theta,alpha)-test_y)**2)),
                    alpha_shift=float(np.linalg.norm(alpha-ALPHA)),failures=failures)
    trace.append(snapshot())
    while calls+19*len(x)+6*len(x)<=max_calls and len(trace)<=95:
        f=qru_output(x,theta,alpha);J=parameter_shift_prediction_jacobian(x,theta,alpha)
        calls+=19*len(x)
        Ja=x[:,None]*J[:,2::3];r=y-f
        objective=lambda ff,aa: .5*np.mean((ff-y)**2)+.5*alpha_penalty*np.sum((aa-ALPHA)**2)
        old=objective(f,alpha);accepted=False
        for _ in range(6):
            direct,profile,_=schur_step(J,Ja,r,damping,alpha_penalty,alpha-ALPHA)
            delta=direct if method=='joint' else profile
            dt=delta[:9].reshape(theta.shape);da=delta[9:]
            if alpha_radius is not None and np.linalg.norm(da)>alpha_radius:
                da*=alpha_radius/np.linalg.norm(da)
                # Refit theta conditionally at this clipped alpha step.
                T=J/np.sqrt(len(x));a=Ja/np.sqrt(len(x));rr=r/np.sqrt(len(x))
                dt=np.linalg.solve(T.T@T+damping*np.eye(9),T.T@(rr-a@da)).reshape(theta.shape)
            trial_theta=theta+dt;trial_alpha=alpha+da
            ft=qru_output(x,trial_theta,trial_alpha);calls+=len(x)
            if objective(ft,trial_alpha)<old-1e-14:
                theta=trial_theta;alpha=trial_alpha;damping=max(damping/2,1e-8)
                accepted=True;break
            damping=min(damping*8,1e8);failures+=1
        trace.append(snapshot())
        if not accepted and damping>=1e7:break
    return trace


def fit_adam_trace(theta0,x,y,max_calls=150_000,alpha_penalty=.01,joint=True,
                   test=None):
    theta=theta0.copy();alpha=ALPHA.copy();calls=0
    moment=np.zeros(12 if joint else 9);velocity=moment.copy();trace=[]
    test_x,test_y=test
    def snapshot():return dict(calls=calls,training_mse=float(np.mean((qru_output(x,theta,alpha)-y)**2)),
                                test_mse=float(np.mean((qru_output(test_x,theta,alpha)-test_y)**2)),
                                alpha_shift=float(np.linalg.norm(alpha-ALPHA)),failures=0)
    trace.append(snapshot())
    for k in range(1,max_calls//(19*len(x))+1):
        f=qru_output(x,theta,alpha);J=parameter_shift_prediction_jacobian(x,theta,alpha)
        calls+=19*len(x)
        jac=np.column_stack([J,x[:,None]*J[:,2::3]]) if joint else J
        g=2*jac.T@(f-y)/len(x)
        if joint:g[9:]+=2*alpha_penalty*(alpha-ALPHA)
        moment=.9*moment+.1*g;velocity=.999*velocity+.001*g*g
        update=(moment/(1-.9**k))/(np.sqrt(velocity/(1-.999**k))+1e-8)
        theta=(theta.ravel()-.04*update[:9]).reshape(theta.shape)
        if joint:alpha-=.02*update[9:]
        trace.append(snapshot())
    return trace


def _larger_task(case,rng,rms):
    theta=reference(rng);x_ref=2*np.pi*(np.arange(160)+.123)/160
    dt=rng.normal(size=(3,3));dt/=np.linalg.norm(dt)
    da=rng.normal(size=3);da/=np.linalg.norm(da)
    if case=='parameter':builder=lambda t,x:qru_output(x,theta+t*dt,ALPHA)
    elif case=='schedule':builder=lambda t,x:qru_output(x,theta,ALPHA+t*da)
    elif case=='mixed':builder=lambda t,x:qru_output(x,theta+t*dt,ALPHA+t*da)
    else:raise ValueError(case)
    step=calibrated_step(builder,x_ref,rms)
    target=lambda x:builder(step,x)
    x=rng.uniform(0,2*np.pi,W);y=target(x)+rng.normal(0,SIGMA,W)
    return theta,x,y,target


def benchmark(seed,replicates,methods,max_calls=150_000,drift_rms=.06):
    rng=np.random.default_rng(seed);rows=[]
    cases=('parameter','schedule','mixed','same_support_normal','outside') if drift_rms==.06 else ('parameter','schedule','mixed')
    test_x=np.linspace(0,2*np.pi,256,endpoint=False)+.001
    for case in cases:
        for i in range(replicates):
            if drift_rms==.06:theta,x,y,_,target=_one(case,i,rng)
            else:theta,x,y,target=_larger_task(case,rng,drift_rms)
            test=(test_x,target(test_x))
            for name,spec in methods.items():
                if name.startswith('adam'):
                    trace=fit_adam_trace(theta,x,y,max_calls=max_calls,test=test,**spec)
                else:trace=fit_lm(theta,x,y,max_calls=max_calls,test=test,**spec)
                for k,row in enumerate(trace):rows.append(dict(seed=seed,case=case,replicate=i,
                                                                method=name,iteration=k,**row))
    return rows
