"""Actual phase witnesses and channel-insertion tests at all layers."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qru.control_design import optimal_single_layer,localized_amplitudes,localized_output
from qru.phase_design import phase_groups
from qru.tomography import reconstruct_from_paths,analytic_path_amplitudes,grouped_path_coefficients
from qru.path_geometry import sinusoidal_fiber


def run():
    rng=np.random.default_rng(56114);x=np.linspace(0,2*np.pi,257)
    trials=json.loads((ROOT/'results/predictive/latent_twins.json').read_text())['numerical_trials']
    profiles=[]
    for row in trials:
        a=np.array(row['theta_a']);b=np.array(row['theta_b']);alpha=row['alpha']
        phase=optimal_single_layer(a,b,alpha,x)
        channels=[]
        for layer in range(len(alpha)):
            aa=localized_amplitudes(a,layer,0.);bb=localized_amplitudes(b,layer,0.)
            gap=reconstruct_from_paths(x,alpha,{s:aa[s]-bb[s] for s in aa})
            channels.append(dict(layer=layer+1,max_gap_at_gamma_0=float(np.max(abs(gap)))))
        profiles.append(dict(seed=row['seed'],phase=phase,dephasing=channels))
    errors=[]
    for L in range(2,6):
        theta=rng.uniform(-np.pi,np.pi,(L,3));alpha=rng.uniform(.2,3,L)
        for layer in range(L):
            for gamma in [0.,.3,.8,1.]:
                a=localized_amplitudes(theta,layer,gamma)
                err=np.max(abs(reconstruct_from_paths(x[:19],alpha,a)-localized_output(x[:19],theta,alpha,layer,gamma)))
                errors.append(float(err))
    tw=sinusoidal_fiber(1.,0.)
    return dict(seed=56114,input_grid=dict(count=len(x),range=[0.,2*np.pi]),
        regular_witnesses=profiles,maximum_channel_path_density_error=max(errors),
        analytic_pair=optimal_single_layer(tw['A'],tw['B'],[1,2],x),
        phase_scope='Exact optimization over the phase at each finite input. Layer numbering starts at one.',
        shot_scope='Known-pair sufficient independent shot counts for 5 percent worst-hypothesis error, using Hoeffding.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(run(),indent=2)+'\n')
