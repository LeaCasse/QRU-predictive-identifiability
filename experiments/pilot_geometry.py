"""Run the completed V3 falsification pilot and write one JSON transfer file.

Example: python experiments/pilot_geometry.py --output /tmp/v3_pilot.json
Results are written to a separate JSON snapshot; frozen references are unchanged.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qru.pilot_geometry import geometry,window_benchmark,tomography_cost,choose_alpha_penalty

def serial(v):
    if isinstance(v,dict):return {k:serial(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [serial(x) for x in v]
    if isinstance(v,np.generic):return v.item()
    return v

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replicates',type=int,default=20)
    p.add_argument('--seed',type=int,default=805)
    p.add_argument('--policy',choices=['discovery','revised'],default='discovery')
    p.add_argument('--regularized',action='store_true')
    args=p.parse_args()
    print('Geometry',flush=True)
    geo=geometry()
    print('Calibration and held-out window benchmark',flush=True)
    penalty=None;penalty_rows=[];penalty_scores={}
    if args.regularized:
        print('Separate regularization calibration',flush=True)
        penalty,penalty_rows,penalty_scores=choose_alpha_penalty()
        print('Selected alpha penalty:',penalty,flush=True)
    calibration,thresholds,windows=window_benchmark(seed=args.seed,replicates=args.replicates,policy=args.policy,alpha_penalty=penalty)
    print('Direct versus path-derived schedule Jacobian',flush=True)
    cost=tomography_cost()
    payload=serial(dict(geometry=geo,calibration=calibration,thresholds=thresholds,windows=windows,tomography_cost=cost,
                        penalty_calibration=penalty_rows,penalty_scores=penalty_scores,
                        protocol=dict(seed_geometry=4321,seed_windows=args.seed,seed_cost=404,policy=args.policy,alpha_penalty=penalty,
                                      penalty_calibration_seed=807,calibration_per_class=50,
                                      replicates=args.replicates,window=96,sigma=.02,steps=80,
                                      alpha_schedule='[1,2,4]',theta_lr=.04,alpha_lr=.02)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+'\n')
    print('Saved',args.output,flush=True)
