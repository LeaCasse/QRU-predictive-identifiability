"""Reproduce the nonlinear observability and gradual-drift V5 pilot."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qru.finite_alternatives import finite_pairs, gradual_benchmark


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--first-seed',type=int,default=1050)
    parser.add_argument('--sequences',type=int,default=12)
    args=parser.parse_args()
    pairs=finite_pairs()
    streams=gradual_benchmark(first_seed=args.first_seed,n=args.sequences)
    result=dict(protocol=dict(gaussian_observation_sigma=.02,reference_alpha=[1,2,4],
                              finite_pair_seed=8712,finite_pair_instances=6,
                              finite_pair_shift_norm=.035,
                              finite_pair_classical_adjustable_amplitudes=6,
                              stream_first_seed=args.first_seed,stream_seeds=args.sequences,
                              stream_windows=30,window_samples=24,
                              stream_coverage_narrow_width=.5,
                              per_window_forward_budget=2100,
                              gradients='exact parameter shifts, no finite-shot simulation'),
                finite_pairs=pairs,gradual_stream=streams)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,separators=(',',':'))+'\n')
    for family in ('QRU','classical_four_tone'):
        slopes=[]
        for k in range(6):
            rs=sorted((r for r in pairs if r['family']==family and r['sample']==k),key=lambda r:r['width'])
            slopes.append(float(np.polyfit(np.log([r['width'] for r in rs]),
                                           np.log([r['local_rms'] for r in rs]),1)[0]))
        print(family,'median finite-window slope',round(float(np.median(slopes)),3))
    print('stream rows',len(streams),'saved',args.output)


if __name__=='__main__':main()
