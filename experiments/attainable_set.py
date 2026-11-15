"""Complete two-layer path geometry and independent constructive checks."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qru.path_geometry import depth_two_amplitudes,attainable_conditions,reconstruct_attainable
from qru.circuit import qru_output
from qru.tomography import analytic_path_amplitudes


def run(count=10000):
    rng=np.random.default_rng(20261115)
    max_modulus=max_phase=max_path=max_prediction=0.
    for _ in range(count):
        theta=rng.uniform(-np.pi,np.pi,(2,3));a=analytic_path_amplitudes(theta)
        a0,ap,am=(a[k] for k in [(1,0),(1,1),(1,-1)])
        check=attainable_conditions(a0,ap,am)
        if not check['attainable']:raise AssertionError(check)
        max_modulus=max(max_modulus,abs(check['modulus_residual']));max_phase=max(max_phase,check['phase_residual'])
        canonical=reconstruct_attainable(a0,ap,am)
        aa=depth_two_amplitudes(canonical);max_path=max(max_path,max(abs(aa[s]-a[s]) for s in a))
        x=rng.uniform(-2,7,7);alpha=rng.uniform(.2,3,2)
        max_prediction=max(max_prediction,float(np.max(abs(qru_output(x,theta,alpha)-qru_output(x,canonical,alpha)))))
    faces=[]
    for radius in [0.,.1,.5]:
        for phase in [0.,.37,2.8]:
            for amps in [(radius*np.exp(1j*phase),0j,0j),(0j,radius*np.exp(1j*phase),0j),(0j,0j,radius*np.exp(1j*phase))]:
                rep=reconstruct_attainable(*amps);a=depth_two_amplitudes(rep)
                err=max(abs(a[s]-v) for s,v in zip([(1,0),(1,1),(1,-1)],amps))
                faces.append(float(err))
    return dict(seed=20261115,circuits=count,maximum_modulus_residual=max_modulus,maximum_phase_residual=max_phase,
        maximum_canonical_path_error=max_path,maximum_controlled_prediction_error=max_prediction,
        boundary_trials=len(faces),maximum_boundary_error=max(faces),
        scope='Entire ideal two-layer attainable path set, including singular faces. The theorem supplies exact sufficiency.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--circuits',type=int,default=10000);a=p.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(run(a.circuits),indent=2)+'\n')
