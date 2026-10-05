"""Reproduce V5 path-collision and conditional stream-risk witnesses."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qru.tomography import analytic_path_amplitudes
from qru.detectability import experiment


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--instances',type=int,default=12)
    args=parser.parse_args()
    # Exact trigonometric witness of a path moment absent from carrier transport.
    theta=np.array([[np.pi/2,np.pi/2,0.],[np.pi/4,0.,np.pi/2]])
    paths=analytic_path_amplitudes(theta)
    amplitude_a=paths[(-1,1)];amplitude_b=paths[(1,0)]
    obstruction=float(np.imag(amplitude_a*np.conj(amplitude_b)))
    if abs(obstruction+1/16)>1e-12:raise AssertionError('L2 analytic witness changed')
    pairs,fingerprints=experiment(instances=args.instances)
    payload=dict(protocol=dict(sigma=.02,alpha=[1,2,4],alpha_shift_norm=.015,
                               window_width=.5,six_distinct_input_locations=True,
                               future_grid='513 points on [0,2pi]',
                               caveat='Unknown theta may change adversarially; clean Gaussian means, no shots.'),
                 exact_L2_obstruction=obstruction,stream_pairs=pairs,collision_fingerprints=fingerprints)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+'\n')
    for key in ('future_mse_gap','fixed_theta_n_for_kl_one','seven_n_for_kl_one',
                'broad_n_for_kl_one','classical_future_mse_gap'):
        print(key,'median',float(np.median([p[key] for p in pairs])))
    for schedule in ('dyadic','incommensurate'):
        rows=[r for r in fingerprints if r['schedule']==schedule]
        print(schedule,'median grouped tangent residual',
              float(np.median([r['grouped_tangent_residual'] for r in rows])))
    print('saved',args.output)


if __name__=='__main__':main()
