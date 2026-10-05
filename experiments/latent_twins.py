"""Reproduce exact and numerical path-distinct, spectrally identical QRUs."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qru.circuit import parameter_shift_prediction_jacobian
from qru.latent_twins import BASE_ALPHA, exact_twin, experiment


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,default=12)
    parser.add_argument('--attempts',type=int,default=10)
    args=parser.parse_args()
    exact=exact_twin()
    trials=experiment(range(198,198+args.seeds),attempts=args.attempts)
    x=np.linspace(0,2*np.pi,65)
    for row in trials:
        ranks=[]
        for t in (row['theta_a'],row['theta_b']):
            singular=np.linalg.svd(parameter_shift_prediction_jacobian(x,np.array(t),BASE_ALPHA),compute_uv=False)
            ranks.append(int(np.sum(singular>singular[0]*1e-9)))
        row['baseline_theta_ranks']=ranks
    payload={'protocol':{'exact':'algebraic construction with singular circuit angles',
                         'numerical':'12 independent random seeds; best distinct twin of 10 optimizer restarts per seed',
                         'baseline':'complete grouped Fourier coefficients, not merely finite samples',
                         'forecast':'alpha=(1,2+delta); theta and bias held fixed',
                         'control':'two-state classical path-resolved Fourier hypotheses; same path tags, probes and predictions',
                         'limitation':'classical path coefficients assumed calibrated for each candidate; neither branch identifiable from passive baseline outputs'},
             'exact':exact,'numerical_trials':trials}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+'\n')
    print('Exact witness:',{k:exact[k] for k in ('future_rms','phase_shot_kl_a_to_b','tomography_max_path_error','classical_path_forecast_max_error')})
    for k in ('grouped_spectrum_max_error','path_l2_difference','future_transverse_rms',
              'parallel_drift_max_error','phase_probe_labels_kl_one','path_resolved_classical_max_error'):
        print(k,'median',np.median([r[k] for r in trials]))
    print('numerical baseline ranks',set(tuple(r['baseline_theta_ranks']) for r in trials))
    print('saved',args.output)


if __name__=='__main__':main()
