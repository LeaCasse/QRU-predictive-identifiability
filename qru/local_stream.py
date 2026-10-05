"""Causal QRU function-regression stream: predict, reveal, then update.

The same realized windows are reused across policies. The pre-drift teacher
is supplied as the warm start; future labels, drift times and target types
never enter an update. This is not lagged time-series forecasting.
"""
import numpy as np

from .circuit import qru_output,parameter_shift_prediction_jacobian
from .pilot_geometry import ALPHA,reference,calibrated_step
from .optimizer import schur_step

SIGMA=.02


def stream(seed,kind,T=32,W=24,width=.5):
    rng=np.random.default_rng(seed);theta0=reference(rng)
    dt=rng.normal(size=(3,3));dt/=np.linalg.norm(dt)
    da=rng.normal(size=3);da/=np.linalg.norm(da)
    grid=2*np.pi*(np.arange(160)+.123)/160
    if kind=='parameter':
        builder=lambda t,x:qru_output(x,theta0+t*dt,ALPHA)
        scale=calibrated_step(builder,grid,.12)
        changed=lambda x:qru_output(x,theta0+scale*dt,ALPHA)
    elif kind=='spectral':
        builder=lambda t,x:qru_output(x,theta0,ALPHA+t*da)
        scale=calibrated_step(builder,grid,.12)
        changed=lambda x:qru_output(x,theta0,ALPHA+scale*da)
    elif kind=='outside':
        changed=lambda x:qru_output(x,theta0,ALPHA)+.12*np.sqrt(2)*np.sin(12*x)
    elif kind=='stationary':changed=lambda x:qru_output(x,theta0,ALPHA)
    else:raise ValueError(kind)
    original=lambda x:qru_output(x,theta0,ALPHA)
    result=[]
    for t in range(T):
        center=np.pi if t<13 or t>=23 else np.pi+1.3*np.sin(2*np.pi*(t-13)/7)
        x=center+width*(rng.random(W)-.5)
        phase='drift_narrow' if 6<=t<13 else 'drift_diverse' if 13<=t<23 else 'recovery' if t>=23 else 'stable'
        target=changed if 6<=t<23 else original
        clean=target(x);observed=clean+rng.normal(0,SIGMA,W)
        result.append(dict(t=t,phase=phase,x=x,y=observed,clean=clean,target=target,center=center))
    return theta0,result


def _step(theta,alpha,x,y,budget=2100,alpha_penalty=.01,info_gate=False,theta_only=False):
    """Damped joint LM on previously observed labels; count trial circuits."""
    n=len(x);calls=0;theta=theta.copy();alpha=alpha.copy();gate_scores=[]
    damping=.001;decisions=[]
    while calls+20*n<=budget and len(decisions)<5:
        f=qru_output(x,theta,alpha);J=parameter_shift_prediction_jacobian(x,theta,alpha)
        calls+=19*n
        Ja=x[:,None]*J[:,2::3]
        if info_gate:
            U,s,_=np.linalg.svd(J,full_matrices=False)
            rank=int(np.sum(s>1e-10*s[0]));B=Ja-U[:,:rank]@(U[:,:rank].T@Ja)
            sv=np.linalg.svd(B,compute_uv=False)
            # A drift of norm .1 in the most informative direction must have
            # two-noise-SD distinguishability before alpha is made available.
            evidence=.1*sv[0]/SIGMA
            active=evidence>=2.
            gate_scores.append(evidence)
        else:active=not theta_only
        r=y-f;old=.5*np.mean(r*r)+.5*alpha_penalty*np.sum((alpha-ALPHA)**2)
        accepted=False
        for retry in range(4):
            if calls+n>budget:break
            if active:
                delta,_,_=schur_step(J,Ja,r,damping,alpha_penalty,alpha-ALPHA)
                dt=delta[:9].reshape(theta.shape);da=delta[9:]
                if np.linalg.norm(da)>.2:
                    da*=.2/np.linalg.norm(da)
                    M=J/np.sqrt(n);A=Ja/np.sqrt(n)
                    dt=np.linalg.solve(M.T@M+damping*np.eye(9),M.T@(r/np.sqrt(n)-A@da)).reshape(theta.shape)
            else:
                M=J/np.sqrt(n)
                dt=np.linalg.solve(M.T@M+damping*np.eye(9),M.T@(r/np.sqrt(n))).reshape(theta.shape)
                da=np.zeros(3)
            a_new=alpha+da;t_new=theta+dt
            f_new=qru_output(x,t_new,a_new);calls+=n
            obj=.5*np.mean((y-f_new)**2)+.5*alpha_penalty*np.sum((a_new-ALPHA)**2)
            if obj<old-1e-14:
                alpha=a_new;theta=t_new;accepted=True;break
            damping*=8
            if calls+len(x)>budget:break
        decisions.append('alpha' if accepted and active else 'theta' if accepted else 'rejected')
        if not accepted:break
    return theta,alpha,calls,decisions,float(np.mean(gate_scores)) if gate_scores else None


POLICIES={
    'theta_current':(1,0.,False,True),
    'joint_current':(1,.01,False,False),
    'joint_recent4':(4,.01,False,False),
    'info_recent4':(4,.01,True,False),
    'joint_recent4_unpenalized':(4,0.,False,False),
}


def benchmark(seeds,types=('stationary','parameter','spectral','outside'),T=32):
    rows=[]
    probe=np.linspace(0,2*np.pi,256,endpoint=False)+.001
    for seed in seeds:
        for kind in types:
            theta0,windows=stream(seed,kind,T=T)
            for policy,(memory,penalty,info,theta_only) in POLICIES.items():
                theta=theta0.copy();alpha=ALPHA.copy();seen=[];cum_calls=0
                for window in windows:
                    x=window['x'];clean=window['clean'];y=window['y']
                    pred=qru_output(x,theta,alpha)
                    mse=float(np.mean((pred-clean)**2))
                    oracle_mse=float(np.mean((qru_output(probe,theta,alpha)-window['target'](probe))**2))
                    seen.append((x,y));last=seen[-memory:]
                    train_x=np.concatenate([z[0] for z in last]);train_y=np.concatenate([z[1] for z in last])
                    theta,alpha,update_calls,decisions,evidence=_step(theta,alpha,train_x,train_y,alpha_penalty=penalty,info_gate=info,theta_only=theta_only)
                    # The prequential prediction was actually obtained before seeing y.
                    cum_calls+=len(x)+update_calls
                    rows.append(dict(seed=seed,kind=kind,policy=policy,t=window['t'],phase=window['phase'],
                                     x_center=window['center'],prequential_clean_mse=mse,
                                     prequential_noisy_mse=float(np.mean((pred-y)**2)),
                                     broad_probe_mse=oracle_mse,alpha_shift=float(np.linalg.norm(alpha-ALPHA)),
                                     cost_this_window=len(x)+update_calls,cumulative_calls=cum_calls,
                                     update_types=','.join(decisions),info_evidence=evidence,
                                     training_samples=len(train_x)))
    return rows
