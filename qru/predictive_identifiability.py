"""Predictive identifiability of labeled Fourier paths.

Controlled streams use real Bernoulli Z shots from a synthetic target that can
be phase-intervened on. This capability is NOT available for a passive data
stream merely because its learner is a QRU. The classical control has exactly
the same continuously parameterized family, prior, labels and observation cost.
"""
from collections import defaultdict
import numpy as np
from scipy.special import logsumexp

from .circuit import qru_output
from .tomography import analytic_path_amplitudes, reconstruct_from_paths


def collision_order(tags, difference, alpha, direction, tolerance=1e-10):
    """First nonzero drift moment, or None for exact velocity-group blindness.

    A numerical certificate for the exact theorem in notebook section 13.
    It concerns drift-amplitude order, not the earlier small-input-window order.
    """
    tags=np.asarray(tags);d=np.asarray(difference);groups=defaultdict(list)
    for i,w in enumerate(tags@alpha):groups[round(float(w),10)].append(i)
    if any(abs(sum(d[ix]))>tolerance for ix in groups.values()):
        raise ValueError('The baseline spectra are not equal')
    velocities=tags@direction
    rmax=max(len(set(np.round(velocities[ix],10))) for ix in groups.values())
    for k in range(1,rmax):
        if any(abs(np.sum(d[ix]*velocities[ix]**k))>tolerance for ix in groups.values()):return k
    return None


def family_theta(r, phi, branch):
    """Two real parameters and a branch bit; all members are depth-2 QRUs.

    A=r*cos((alpha2-alpha1)*x+phi+phase2-phase1).
    B=r*cos(alpha1*x+phi+phase1). Both coincide at alpha=(1,2), phase=0.
    """
    if branch==0:return np.array([[0.,np.pi,-phi],[np.arccos(r),0.,0.]])
    return np.array([[0.,np.arcsin(r),np.pi/2+phi],[np.pi/2,0.,0.]])


def family_mean(r, phi, branch, x, alpha, phase=(0.,0.)):
    angle=np.where(branch==0,(alpha[1]-alpha[0])*x+phase[1]-phase[0],
                   alpha[0]*x+phase[0])+phi
    return r*np.cos(angle)


def fourier_mean(r, phi, branch, x, alpha, phase=(0.,0.)):
    """Independent path-resolved classical evaluator; same 2D family + bit."""
    tags=np.where(np.asarray(branch)[...,None]==0,np.array([-1.,1.]),np.array([1.,0.]))
    coefficients=.5*r*np.exp(1j*phi)
    return 2*np.real(coefficients*np.exp(1j*((tags@alpha)*x+tags@np.array(phase))))


def mathematical_checks():
    rng=np.random.default_rng(260927);errors=[];orders=[]
    for _ in range(12):
        r=rng.uniform(.2,.9);phi=rng.uniform(-np.pi,np.pi)
        a=rng.uniform(.4,3.,2);phase=rng.uniform(-2,2,2);x=rng.uniform(-3,7,19)
        for branch in (0,1):
            theta=family_theta(r,phi,branch);theta[:,2]+=phase
            direct=qru_output(x,theta,a)
            pred=family_mean(r,phi,branch,x,a,phase)
            classic=fourier_mean(r,phi,branch,x,a,phase)
            errors.extend([np.max(abs(pred-direct)),np.max(abs(pred-classic))])
        pa=analytic_path_amplitudes(family_theta(r,phi,0));pb=analytic_path_amplitudes(family_theta(r,phi,1))
        tags=np.array(list(pa));d=np.array([pb[s]-pa[s] for s in pa])
        orders.append((collision_order(tags,d,np.array([1.,2.]),np.array([0.,1.])),
                       collision_order(tags,d,np.array([1.,2.]),np.array([1.,2.]))))
    # Full continuous QRU manifold, not only the 2D twin family.
    full_errors=[]
    for L in (1,2,3,4):
        for _ in range(3):
            theta=rng.uniform(-2,2,(L,3));a=rng.uniform(.5,2,L);x=rng.uniform(-2,5,37)
            full_errors.append(np.max(abs(qru_output(x,theta,a)-reconstruct_from_paths(x,a,analytic_path_amplitudes(theta)))))
    assert max(errors+full_errors)<1e-12
    assert all(pair==(1,None) for pair in orders)
    # Sharp second-order moment cancellation in the unconstrained path space.
    # This tests the algebraic bound, NOT QRU attainability of these amplitudes.
    positive=np.array([[-1,1,0],[-1,-1,1],[1,0,0]])
    higher_order=collision_order(np.vstack([positive,-positive]),np.tile([-.2,.1,.1],2),
                                np.array([1.,2.,4.]),np.array([0.,1.,3.]))
    assert higher_order==2
    return {'family_formula_max_error':float(max(errors)),
            'full_manifold_max_error_L1_to_L4':float(max(full_errors)),
            'transverse_drift_order':1,'parallel_drift_order':'infinite',
            'unconstrained_three_velocity_order':higher_order,
            'continuous_capacity':'two real nuisance parameters r,phi plus one branch bit; identical prior grid for both evaluators'}


SCENARIOS=('abrupt_planned','abrupt_unannounced','gradual','recurring',
           'parallel','stationary','irrelevant_inputs','latent_reset',
           'no_phase_access','initially_separated')
POLICIES=('passive','phase','blind_phase')


def controlled_stream(seed,scenario,T=180,grid_phase=48):
    """Predict h=1,5 before labels; four Z shots per time step for every policy.

    Controls x are preannounced; alpha future is known only in abrupt_planned.
    Otherwise a persistence forecast uses current observed alpha. Phase policies
    replace the fourth natural shot by one intervention shot; no free oracle.
    Latent-reset scenario deliberately violates static-branch persistence.
    """
    rng=np.random.default_rng(seed)
    r_true=rng.uniform(.45,.85);phi_true=rng.uniform(-np.pi,np.pi)
    initial_branch=int(rng.integers(2));change=int(rng.integers(70,91))
    t=np.arange(T+5);x=2*np.pi*np.mod(.381966*t+.137,1)
    if scenario=='irrelevant_inputs':x[:]=0.
    shift=np.where(t>=change,.16,0.)
    if scenario=='gradual':shift=np.clip((t-change)*.006,0,.16)
    if scenario=='recurring':shift=np.where((t>=change)&(t<change+35)|(t>=change+60),.16,0.)
    if scenario=='stationary':shift[:]=0.
    alpha=np.column_stack([np.ones(len(t)),2+shift])
    if scenario=='parallel':alpha=np.column_stack([1+shift,2*(1+shift)])
    if scenario=='initially_separated':alpha[:,1]+=.2
    branches=np.full(len(t),initial_branch)
    if scenario=='latent_reset':branches[t>=change]=1-initial_branch  # adversarial swap, not oracle-reset knowledge
    truth=np.array([family_mean(r_true,phi_true,branches[j],x[j],alpha[j]) for j in t])
    rr,pp,bb=np.meshgrid(np.linspace(.15,.95,17),2*np.pi*np.arange(grid_phase)/grid_phase,
                        np.arange(2),indexing='ij')
    rr=rr.ravel();pp=pp.ravel();bb=bb.ravel()
    uniforms=rng.random((T,4))
    records=[];curves=[]
    for policy in POLICIES:
        logw=np.full(len(rr),-np.log(len(rr)));logc=logw.copy()
        losses={h:[] for h in (1,5)};post_branch=[];gap=0.;query_count=0
        for j in range(T):
            w=np.exp(logw-logsumexp(logw));wc=np.exp(logc-logsumexp(logc))
            post_branch.append(float(np.sum(w[bb==branches[j]])))
            for h in (1,5):
                af=alpha[j+h] if scenario=='abrupt_planned' else alpha[j]
                f=family_mean(rr,pp,bb,x[j+h],af)
                fc=fourier_mean(rr,pp,bb,x[j+h],af)
                pred=float(w@f);pc=float(wc@fc);gap=max(gap,abs(pred-pc))
                losses[h].append((pred-truth[j+h])**2)
            # The prediction above cannot see any shot from time j or later.
            for slot in range(4):
                phase=np.zeros(2)
                if slot==3 and policy!='passive' and scenario!='no_phase_access':
                    phase=(np.array([0.,np.pi/2]) if policy=='phase'
                           else np.array([np.pi/2,np.pi]))
                    query_count+=1
                mean=float(family_mean(r_true,phi_true,branches[j],x[j],alpha[j],phase))
                y=1 if uniforms[j,slot]<(1+mean)/2 else -1
                mu=family_mean(rr,pp,bb,x[j],alpha[j],phase)
                mc=fourier_mean(rr,pp,bb,x[j],alpha[j],phase)
                logw+=np.log(np.clip((1+y*mu)/2,1e-12,1.))
                logc+=np.log(np.clip((1+y*mc)/2,1e-12,1.))
            logw-=logsumexp(logw);logc-=logsumexp(logc)
        for h in (1,5):
            loss=np.array(losses[h]);early=slice(change,change+15)
            records.append(dict(seed=seed,scenario=scenario,policy=policy,horizon=h,
                                early_future_mse=float(loss[early].mean()),
                                boundary_future_mse=float(loss[change-h:change].mean()),
                                post_future_mse=float(loss[change:T].mean()),
                                pre_future_mse=float(loss[20:change-5].mean()),
                                branch_probability_before=float(post_branch[change-1]),
                                shots=4*T,phase_shots=query_count,
                                qru_classical_max_prediction_gap=float(gap),change=change))
            for j in range(0,T,5):
                curves.append(dict(seed=seed,scenario=scenario,policy=policy,horizon=h,t=j,
                                   future_mse=float(loss[j:j+5].mean())))
    return records,curves


def paired_summary(rows):
    """Paired-seed bootstrap interval; descriptive benchmark uncertainty only."""
    out=[];rng=np.random.default_rng(91703)
    for scenario in SCENARIOS:
        for h in (1,5):
            group=[r for r in rows if r['scenario']==scenario and r['horizon']==h]
            ref={r['seed']:r for r in group if r['policy']=='passive'}
            for policy in POLICIES:
                trials=[r for r in group if r['policy']==policy]
                delta=np.array([ref[r['seed']]['early_future_mse']-r['early_future_mse'] for r in trials])
                boundary=np.array([ref[r['seed']]['boundary_future_mse']-r['boundary_future_mse'] for r in trials])
                boot=delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(axis=1)
                boundary_boot=boundary[rng.integers(0,len(delta),(2000,len(delta)))].mean(axis=1)
                out.append(dict(scenario=scenario,policy=policy,horizon=h,seeds=len(trials),
                                early_future_mse=float(np.mean([r['early_future_mse'] for r in trials])),
                                post_future_mse=float(np.mean([r['post_future_mse'] for r in trials])),
                                paired_gain=float(delta.mean()),gain_low=float(np.quantile(boot,.025)),
                                gain_high=float(np.quantile(boot,.975)),
                                boundary_future_mse=float(np.mean([r['boundary_future_mse'] for r in trials])),
                                boundary_gain=float(boundary.mean()),boundary_gain_low=float(np.quantile(boundary_boot,.025)),
                                boundary_gain_high=float(np.quantile(boundary_boot,.975)),
                                branch_probability_before=float(np.mean([r['branch_probability_before'] for r in trials])),
                                qru_classical_max_prediction_gap=max(r['qru_classical_max_prediction_gap'] for r in trials)))
    return out


def _continuous_forward(x,z,implementation):
    theta=z[:6].reshape(2,3);alpha=z[6:]
    if implementation=='qru':return qru_output(x,theta,alpha)
    return reconstruct_from_paths(x,alpha,analytic_path_amplitudes(theta))


def _continuous_fit(x,y,z,implementation,iterations):
    """Same damped LM, same budget, independent forward implementation.

    Parameter-shift theta derivative; J_alpha=x J_bias. No noisy gradients:
    this is a simulator comparison, with actual observed time-series targets.
    """
    z=z.copy();damping=.02
    for _ in range(iterations):
        f=_continuous_forward(x,z,implementation);cols=[]
        for k in range(6):
            plus=z.copy();minus=z.copy();plus[k]+=np.pi/2;minus[k]-=np.pi/2
            cols.append((_continuous_forward(x,plus,implementation)-_continuous_forward(x,minus,implementation))/2)
        J=np.column_stack(cols);J=np.column_stack([J,x[:,None]*J[:,2::3]])
        step=np.linalg.solve(J.T@J/len(y)+damping*np.eye(8),J.T@(y-f)/len(y))
        norm=np.linalg.norm(step)
        if norm>.3:step*=.3/norm
        trial=z+step
        if np.mean((_continuous_forward(x,trial,implementation)-y)**2)<np.mean((f-y)**2):
            z=trial;damping=max(.002,damping/2)
        else:damping*=4
    return z


def real_stream(name,values,seed):
    """Rolling-origin h=1,5 forecasts on genuine historical observations.

    Prefix-only scaling; lag-one QRU and matched path family; rolling AR(1)
    and persistence. Model updates see y through origin t only, and h=5 uses
    recursive own predictions. There are NO physical phase-query labels.
    """
    # Nile's documented 1898 shift must remain in the evaluation period.
    values=np.asarray(values,float);train=80 if name=='sunspots' else 20
    center=float(np.mean(values[:train]));scale=float(np.max(abs(values[:train]-center))/.8)
    y=(values-center)/scale;variance=float(np.var(values[:train]))
    rng=np.random.default_rng(seed);initial=np.r_[rng.uniform(-1,1,6),[1.,2.]]
    zq=_continuous_fit(np.pi*(y[:train-1]+1),y[1:train],initial,'qru',25)
    zc=_continuous_fit(np.pi*(y[:train-1]+1),y[1:train],initial,'fourier',25)
    records=[];maxgap=0.
    for t in range(train-1,len(y)-5):
        if t>=train:
            start=max(0,t-40)
            xtrain=np.pi*(y[start:t]+1);target=y[start+1:t+1]
            zq=_continuous_fit(xtrain,target,zq,'qru',3)
            zc=_continuous_fit(xtrain,target,zc,'fourier',3)
        start=max(0,t-60);A=np.column_stack([np.ones(t-start),y[start:t]])
        ar=np.linalg.solve(A.T@A+.01*np.eye(2),A.T@y[start+1:t+1])
        predictions={k:float(y[t]) for k in ('qru','fourier','ar1','persistence')}
        for h in range(1,6):
            predictions['qru']=float(_continuous_forward([np.pi*(predictions['qru']+1)],zq,'qru')[0])
            predictions['fourier']=float(_continuous_forward([np.pi*(predictions['fourier']+1)],zc,'fourier')[0])
            predictions['ar1']=float(ar[0]+ar[1]*predictions['ar1'])
            maxgap=max(maxgap,abs(predictions['qru']-predictions['fourier']))
            if h in (1,5):
                for method,pred in predictions.items():
                    records.append(dict(dataset=name,seed=seed,origin=t,horizon=h,method=method,
                                        squared_error=float((pred-y[t+h])**2*scale**2),
                                        normalized_squared_error=float((pred-y[t+h])**2*scale**2/variance),
                                        predicted_value=float(pred*scale+center),target=float(values[t+h])))
    summaries=[]
    for h in (1,5):
        for method in ('qru','fourier','ar1','persistence'):
            rows=[r for r in records if r['horizon']==h and r['method']==method]
            summaries.append(dict(dataset=name,seed=seed,horizon=h,method=method,origins=len(rows),
                                  mse=float(np.mean([r['squared_error'] for r in rows])),
                                  normalized_mse=float(np.mean([r['normalized_squared_error'] for r in rows])),
                                  qru_classical_max_scaled_gap=float(maxgap),
                                  train_prefix=train,phase_access=False,
                                  outside_training_output_range=int(np.sum(abs(y[train:])>1))))
    return summaries,records
