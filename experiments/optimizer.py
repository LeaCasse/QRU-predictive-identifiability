"""Small falsification experiment: Schur identity and cost-indexed QRU adaptation.

Run from anywhere: python experiments/optimizer.py --output runs/v4.json
The archived XLSX contains the predeclared independent seed 951 protocol.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qru.pilot_geometry import ALPHA,reference
from qru.circuit import parameter_shift_prediction_jacobian,qru_output
from qru.optimizer import benchmark,schur_step

def identity_check(seed=947,instances=25):
    rng=np.random.default_rng(seed);rows=[]
    for k in range(instances):
        theta=reference(rng);x=rng.uniform(0,2*np.pi,96)
        J=parameter_shift_prediction_jacobian(x,theta,ALPHA)
        Ja=x[:,None]*J[:,2::3];r=rng.normal(size=len(x));offset=rng.normal(size=3)*.08
        for damping,penalty in [(1e-4,0.),(1e-3,.01),(1e-2,.05)]:
            a,b,condition=schur_step(J,Ja,r,damping,penalty,offset)
            rows.append(dict(instance=k,damping=damping,alpha_penalty=penalty,
                             step_difference=float(np.linalg.norm(a-b)),
                             step_norm=float(np.linalg.norm(a)),schur_condition=condition))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=951);p.add_argument('--replicates',type=int,default=12)
    p.add_argument('--max-calls',type=int,default=150_000)
    p.add_argument('--drift-rms',type=float,default=.06);a=p.parse_args()
    methods={
       'adam_theta':{'joint':False,'alpha_penalty':0.},
       'adam_joint_regularized':{'joint':True,'alpha_penalty':.01},
       'lm_joint':{'alpha_penalty':0.,'damping':1e-3,'method':'joint'},
       'lm_regularized_joint':{'alpha_penalty':.01,'alpha_radius':.20,'damping':1e-3,'method':'joint'},
       'lm_profiled':{'alpha_penalty':.01,'alpha_radius':.20,'damping':1e-3,'method':'profile'},
    }
    print('Testing exact joint / profiled identity',flush=True)
    identity=identity_check()
    print('Paired budgets on five drift cases:',a.seed,a.replicates,flush=True)
    rows=benchmark(a.seed,a.replicates,methods,a.max_calls,a.drift_rms)
    result={'protocol':{'seed':a.seed,'instances_per_case':a.replicates,'noise_sigma':.02,
                        'train_window':96,'test_grid':256,'max_forward_evaluations':a.max_calls,
                        'drift_rms':a.drift_rms,'alpha_schedule':[1,2,4],
                        'test_evaluations_excluded_from_training_budget':True,
                        'training_checkpoint_observation_excluded_from_training_budget':True,
                        'methods':methods,'warm_start':'generating pre-drift QRU',
                        'all_methods_use_same_observed_x_y':True},
            'identity':identity,'trajectories':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,separators=(',',':'))+'\n')
    print('Saved',a.output,'trajectories',len(rows),flush=True)

if __name__=='__main__':main()
