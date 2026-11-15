"""Depth-general collision family with finite-shot logarithmic layer search."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qru.depth_family import branch_theta,branch_tag,interrogation,uniform_branch_theta,interrogate_prefixes
from qru.circuit import qru_output
from qru.tomography import analytic_path_amplitudes


def run(repetitions=256):
    rng=np.random.default_rng(51114);identities=[];max_path=0.
    for L in range(2,9):
        error=0.
        for k in range(1,L+1):
            for _ in range(4):
                r=rng.uniform(.2,1.);phase=rng.uniform(-np.pi,np.pi);theta=branch_theta(L,k,r,phase)
                alpha=rng.uniform(.1,3.,L);x=rng.uniform(-2,7,21);phi=rng.normal(size=L)
                shifted=theta.copy();shifted[:,2]+=phi;tag=branch_tag(L,k)
                error=max(error,float(np.max(abs(qru_output(x,shifted,alpha)-r*np.cos(np.dot(tag,alpha)*x+phase+np.dot(tag,phi))))))
                if L<=6:
                    a=analytic_path_amplitudes(theta);s=tuple(tag)
                    max_path=max(max_path,max(abs(v-(r*np.exp(1j*phase)/2 if ss==s else r*np.exp(-1j*phase)/2 if ss==tuple(-tag) else 0.)) for ss,v in a.items()))
        identities.append(dict(depth=L,max_controlled_mean_error=error))
    studies=[];r=.7;phase=.3;xf=2*np.pi*np.arange(257)/257
    for L in [2,4,8]:
        alpha0=2.**np.arange(L);future=alpha0+.08*np.sin(np.arange(L)*1.7+.4)
        curves=np.array([qru_output(xf,branch_theta(L,k,r,phase),future) for k in range(1,L+1)])
        prior=curves.mean(axis=0);passive=float(np.mean((curves-prior)**2));rows=[]
        for k in range(1,L+1):
            for j in range(repetitions):
                res=interrogation(L,k,r,phase,rng);prediction=res['posterior']@curves
                rows.append(dict(branch=k,replicate=j,selected=res['selected'],shots=res['shots'],per_round_error_bound=res['per_round_error_bound'],
                    posterior=res['posterior'].tolist(),queries=res['records'],future_mse=float(np.mean((prediction-curves[k-1])**2))))
        studies.append(dict(depth=L,radius=r,phase=phase,repetitions_per_branch=repetitions,
            alpha=alpha0.tolist(),future_alpha=future.tolist(),passive_future_mse=passive,
            informative_future_mse=float(np.mean([row['future_mse'] for row in rows])),
            hard_errors=sum(row['selected']!=row['branch'] for row in rows),trials=rows))
    uniform=[]
    for L in [3,7,15]:
        K=(L+1)//2;alpha0=np.ones(L);future=alpha0+.08*np.sin(np.arange(L)*1.7+.4)
        curves=np.array([qru_output(xf,uniform_branch_theta(L,k,r,phase),future) for k in range(1,K+1)])
        baseline_error=max(float(np.max(abs(qru_output(xf,uniform_branch_theta(L,k,r,phase),alpha0)-r*np.cos(xf+phase)))) for k in range(1,K+1))
        passive=float(np.mean((curves-curves.mean(axis=0))**2));rows=[]
        for k in range(1,K+1):
            for j in range(repetitions):
                res=interrogate_prefixes(np.arange(1,L+1,2),k,r,phase,rng);prediction=res['posterior']@curves
                rows.append(dict(branch=k,replicate=j,selected=res['selected'],shots=res['shots'],
                    per_round_error_bound=res['per_round_error_bound'],posterior=res['posterior'].tolist(),
                    queries=res['records'],future_mse=float(np.mean((prediction-curves[k-1])**2))))
        uniform.append(dict(depth=L,classes=K,radius=r,phase=phase,repetitions_per_branch=repetitions,
            alpha=alpha0.tolist(),future_alpha=future.tolist(),maximum_baseline_error=baseline_error,
            passive_future_mse=passive,informative_future_mse=float(np.mean([row['future_mse'] for row in rows])),
            hard_errors=sum(row['selected']!=row['branch'] for row in rows),trials=rows))
    return dict(seed=51114,identities=identities,max_tagged_path_error=max_path,studies=studies,uniform_studies=uniform,
        scope='Known present amplitude/phase, equal branch prior. All policies use the same fixed total shot budget at each depth; passive means are identical across branches.')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--repetitions',type=int,default=256);a=p.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(run(a.repetitions),indent=2)+'\n')
