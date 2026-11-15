"""Paired training-noise study on a complete two-class observation fiber.

Every mode uses the same clean target and paired random initializations.
A deterministic clean least-squares completion classifies the basin reached
at a fixed training budget. Both pre-completion and final losses are retained.
No hidden target label is supplied to training or the completion step.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qru.circuit import qru_output, parameter_shift_prediction_jacobian
from qru.path_geometry import depth_two_amplitudes, sinusoidal_fiber


def output(p, x):
    # Independent vectorized Bloch expression. p shape (runs,5), x (runs,n).
    t,z,b,u,c = (p[:,j,None] for j in range(5))
    v=x+b; w=2*x+c
    rx=np.cos(z)*np.sin(v)+np.sin(z)*np.sin(t)*np.cos(v)
    ry=np.sin(z)*np.sin(v)-np.cos(z)*np.sin(t)*np.cos(v)
    rz=np.cos(t)*np.cos(v)
    return np.cos(u)*(-np.sin(w)*rx+np.cos(w)*rz)+np.sin(u)*ry


def output_jacobian(p,x):
    """Closed Bloch gradient, verified against independent parameter shifts."""
    t,z,b,u,c=(p[:,j,None] for j in range(5))
    v=x+b;w=2*x+c
    sv,cv=np.sin(v),np.cos(v);sw,cw=np.sin(w),np.cos(w)
    st,ct,sz,cz,su,cu=np.sin(t),np.cos(t),np.sin(z),np.cos(z),np.sin(u),np.cos(u)
    rx=cz*sv+sz*st*cv;ry=sz*sv-cz*st*cv;rz=ct*cv
    output=cu*(-sw*rx+cw*rz)+su*ry
    dt=cu*(-sw*sz*ct*cv-cw*st*cv)-su*cz*ct*cv
    dz=cu*sw*ry+su*rx
    db=cu*(-sw*(cz*cv-sz*st*sv)-cw*ct*sv)+su*(sz*cv+cz*st*sv)
    du=-su*(-sw*rx+cw*rz)+cu*ry
    dc=cu*(-cw*rx-sw*rz)
    return output,np.stack([dt,dz,db,du,dc],axis=1)


def unpack(p):
    return np.array([[p[0],p[1],p[2]],[p[3],0.,p[4]]])


def run(n=256, steps=1200, radius=.7, phase=.3):
    modes=[('full',0.),('minibatch',0.),('gradient_0.05',.05),('gradient_0.20',.20),('shot_labels',0.)]
    seeds=np.arange(43100,43100+n)
    initial=np.array([np.random.default_rng(int(s)).uniform(-np.pi,np.pi,5) for s in seeds])
    x=2*np.pi*np.arange(32)/32; y=radius*np.cos(x+phase)
    ths=sinusoidal_fiber(radius,phase); keys=list(depth_two_amplitudes(ths['A']))
    ref={k:np.array([depth_two_amplitudes(v)[s] for s in keys]) for k,v in ths.items()}
    xf=2*np.pi*np.arange(257)/257
    futures={k:qru_output(xf,v,[1,2.16]) for k,v in ths.items()}
    rows=[]; mode_branches={}
    for mode,sigma in modes:
        p=initial.copy(); rng=np.random.default_rng(20261114)
        # Paired indices and Gaussian draws across modes, independent of target branch.
        for step in range(steps):
            indices=rng.integers(0,32,size=(n,8)); noise=rng.normal(size=(n,5))
            xx=np.broadcast_to(x,(n,32)) if mode!='minibatch' else x[indices]
            yy=radius*np.cos(xx+phase)
            if mode=='shot_labels':
                yy=2*rng.binomial(128,(1+yy)/2)/128-1
            predictions,jac=output_jacobian(p,xx)
            residual=predictions-yy
            g=np.mean(residual[:,None,:]*jac,axis=2)
            p-=.15*(g+sigma*noise)
        before=np.mean((output(p,np.broadcast_to(x,(n,32)))-y)**2,axis=1)
        branches=[]
        for i,(seed,v) in enumerate(zip(seeds,p)):
            def fun(z):return qru_output(x,unpack(z),[1,2])-y
            def jac(z):return parameter_shift_prediction_jacobian(x,unpack(z),[1,2])[:,[0,1,2,3,5]]
            fit=least_squares(fun,v,jac=jac,gtol=1e-12,ftol=1e-12,xtol=1e-12,max_nfev=500)
            th=unpack(fit.x); amps=depth_two_amplitudes(th)
            aa=np.array([amps[s] for s in keys]); distances={k:float(np.linalg.norm(aa-a)) for k,a in ref.items()}
            branch=min(distances,key=distances.get) if min(distances.values())<1e-5 else 'unresolved'
            branches.append(branch)
            f=qru_output(xf,th,[1,2.16])
            raw_theta=unpack(v);raw_f=qru_output(xf,raw_theta,[1,2.16])
            raw_amps=depth_two_amplitudes(raw_theta)
            raw_distances={k:float(np.linalg.norm(np.array([raw_amps[s] for s in keys])-a)) for k,a in ref.items()}
            rows.append(dict(seed=int(seed),mode=mode,branch=branch,pre_completion_mse=float(before[i]),
                             raw_theta=raw_theta.tolist(),raw_path_distances=raw_distances,
                             raw_nearest_class=min(raw_distances,key=raw_distances.get),
                             raw_future_mse_A=float(np.mean((raw_f-futures['A'])**2)),
                             raw_future_mse_B=float(np.mean((raw_f-futures['B'])**2)),
                             final_mse=float(np.mean(fun(fit.x)**2)),path_distances=distances,
                             future_mse_A=float(np.mean((f-futures['A'])**2)),
                             future_mse_B=float(np.mean((f-futures['B'])**2)),theta=th.tolist()))
        mode_branches[mode]=np.array(branches)
        print(mode,{k:branches.count(k) for k in ('A','B','unresolved')},flush=True)
    summaries=[]; base=mode_branches['full']
    for mode,_ in modes:
        bb=mode_branches[mode]; rr=[r for r in rows if r['mode']==mode]
        ab=int(np.sum((base=='A')&(bb=='B')));ba=int(np.sum((base=='B')&(bb=='A')))
        na=int(np.sum(bb=='A'));ci=binomtest(na,n).proportion_ci()
        summaries.append(dict(mode=mode,n=n,A=na,B=int(np.sum(bb=='B')),unresolved=int(np.sum(bb=='unresolved')),
            A_fraction_interval=[ci.low,ci.high],switch_A_to_B=ab,switch_B_to_A=ba,
            paired_exact_p=binomtest(ab,ab+ba).pvalue if ab+ba else 1.,
            pre_completion_median_mse=float(np.median([r['pre_completion_mse'] for r in rr])),
            raw_mean_future_mse_A=float(np.mean([r['raw_future_mse_A'] for r in rr])),
            raw_mean_future_mse_B=float(np.mean([r['raw_future_mse_B'] for r in rr])),
            raw_nearest_A=sum(r['raw_nearest_class']=='A' for r in rr),
            completion_changes_nearest=sum(r['raw_nearest_class']!=r['branch'] for r in rr),
            maximum_final_mse=max(r['final_mse'] for r in rr),
            mean_future_mse_A=float(np.mean([r['future_mse_A'] for r in rr])),
            mean_future_mse_B=float(np.mean([r['future_mse_B'] for r in rr]))))
    return dict(protocol=dict(radius=radius,phase=phase,alpha=[1,2],future_alpha=[1,2.16],
        input_count=32,steps=steps,learning_rate=.15,loss='half mean squared error',
        initial_seed_range=[int(seeds[0]),int(seeds[-1])],training_rng=20261114,
        gradient_noise_std=[.05,.20],shots_per_label=128,minibatch=8,
        completion='clean least_squares, 500 evaluations maximum',classification_path_tolerance=1e-5,
        inference='Paired basin study, conditional on this target, initialization and optimizer budget.'),
        summary=summaries,trials=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--runs',type=int,default=256);p.add_argument('--steps',type=int,default=1200);p.add_argument('--radius',type=float,default=.7);p.add_argument('--phase',type=float,default=.3);a=p.parse_args()
    data=run(a.runs,a.steps,a.radius,a.phase);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
