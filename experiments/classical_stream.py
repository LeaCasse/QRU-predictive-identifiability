"""Tune the simple Fourier control on discovery streams, then hold out seeds."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qru.classical_stream import benchmark

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    candidates=(.0001,.001,.01,.1,1.)
    scores={}
    for penalty in candidates:
        rows=benchmark([980,981,982],lambda_ridge=penalty)
        scores[str(penalty)]=float(np.mean([r['prequential_clean_mse'] for r in rows if r['t']>=6]))
    chosen=float(min(scores,key=scores.get))
    runs={str(p):benchmark(list(range(990,1002)),lambda_ridge=p) for p in (.01,chosen)}
    data=dict(calibration_seeds=[980,981,982],confirmation_seeds=list(range(990,1002)),
              calibration_score='mean clean prequential MSE on windows t>=6 across four concept cases',
              lambda_scores=scores,lambda_selected_by_calibration_overall_mean=chosen,runs=runs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,separators=(',',':'))+'\n')
    print('selected classical ridge',chosen,scores)

if __name__=='__main__':main()
