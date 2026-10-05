"""General checks, finite-shot causal control streams, and real-data falsification."""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qru.predictive_identifiability import (mathematical_checks,controlled_stream,
                                             paired_summary,real_stream,SCENARIOS)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,default=32)
    parser.add_argument('--first-seed',type=int,default=1300)
    parser.add_argument('--skip-real',action='store_true')
    parser.add_argument('--real-only',action='store_true',help='Refresh real replay in an existing snapshot; preserve controlled trials')
    args=parser.parse_args()
    checks=mathematical_checks();print('Mathematical checks',checks,flush=True)
    rows=[];curves=defaultdict(list)
    for scenario in (() if args.real_only else SCENARIOS):
        for seed in range(args.first_seed,args.first_seed+args.seeds):
            trial,curve=controlled_stream(seed,scenario);rows.extend(trial)
            for r in curve:curves[(scenario,r['policy'],r['horizon'],r['t'])].append(r['future_mse'])
        print('Completed',scenario,flush=True)
    summary=paired_summary(rows) if rows else []
    real_summary=[];real_predictions=[]
    if not args.skip_real:
        data=json.loads((ROOT/'data/real_series.json').read_text())
        for name,item in data.items():
            keys=list(item['rows'][0]);values=[float(r[keys[1]]) for r in item['rows']]
            for seed in (17,18,19):
                a,b=real_stream(name,values,seed);real_summary.extend(a);real_predictions.extend(b)
            print('Completed real data',name,flush=True)
    averaged=[dict(scenario=k[0],policy=k[1],horizon=k[2],t=k[3],mean_future_mse=float(np.mean(v)))
              for k,v in curves.items()]
    out=dict(protocol={'shots_per_step':4,'stream_length':180,'horizons':[1,5],
                       'prior_r_grid':17,'prior_phase_grid':48,'unknown_branch_states':2,
                       'phase_query':'one of four natural shots replaced; target mechanism must be controllable',
                       'future_control':'public only in abrupt_planned; otherwise persist observed current alpha',
                       'input_schedule':'public deterministic x schedule in controlled experiments',
                       'capacity':'identical continuous (r,phi,branch) family; same discretized prior in QRU and classical path model',
                       'real_data':'passive chronological annual sunspots and Nile flow; rolling lag-one forecast; no fabricated phase labels',
                       'scope':'synthetic controlled mechanism experiment; real data provide passive falsification, not validation of latent physical QRU paths'},
             checks=checks,controlled_summary=summary,controlled_trials=rows,controlled_curves=averaged,
             real_summary=real_summary,real_predictions=real_predictions)
    if args.real_only and args.output.exists():
        previous=json.loads(args.output.read_text())
        previous.update(real_summary=real_summary,real_predictions=real_predictions)
        out=previous
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2)+'\n')
    for r in summary:
        if r['policy']=='phase' and r['horizon']==5:print(r['scenario'],'gain',r['paired_gain'],'CI',r['gain_low'],r['gain_high'])
    print('saved',args.output,flush=True)


if __name__=='__main__':main()
