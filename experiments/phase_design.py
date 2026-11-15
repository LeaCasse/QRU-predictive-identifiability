"""Layer-subset certificates and arbitrary-depth fixed-input tomography."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qru.phase_design import pair_design,fixed_input_tomography
from qru.tomography import analytic_path_amplitudes
from qru.path_geometry import sinusoidal_fiber

def run():
    rng=np.random.default_rng(20261114); checks=[]
    for L in range(1,7):
        worst=0.
        for _ in range(5):
            theta=rng.uniform(-np.pi,np.pi,(L,3));alpha=rng.uniform(.2,3,L)
            true=analytic_path_amplitudes(theta); recovered=fixed_input_tomography(theta,alpha,.37)
            worst=max(worst,max(abs(true[s]-recovered[s]) for s in true))
        checks.append(dict(depth=L,circuits=5,settings=3**L,max_amplitude_error=float(worst)))
    data=json.loads((ROOT/'results/predictive/latent_twins.json').read_text())
    profiles=[]
    for row in data['numerical_trials']:
        profiles.append(dict(seed=row['seed'],**pair_design(row['theta_a'],row['theta_b'],row['alpha'])))
    tw=sinusoidal_fiber(.7,.3)
    # Exact zero-schedule example: cos(phi1+phi2) vs cos(phi1)*cos(phi2).
    # A single varied phase agrees, two simultaneous phases distinguish.
    a=np.zeros((2,3)); b=a.copy();b[0,1]=np.pi/2
    return dict(seed=20261114,tomography=checks,regular_twins=profiles,
                sinusoidal=pair_design(tw['A'],tw['B'],[1,2]),
                joint_control_example=dict(theta_a=a.tolist(),theta_b=b.tolist(),alpha=[0,0],
                                           design=pair_design(a,b,[0,0])))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(run(),indent=2)+'\n')
