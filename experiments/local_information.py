"""Reproduce the finite-window QRU frequency-information calculation."""
import argparse,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qru.local_information import width_experiment,slopes

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--seed',default=971,type=int)
    a=parser.parse_args()
    rows,jets=width_experiment(seed=a.seed);exponents=slopes(rows)
    if not all(j['jet_rank']==6 for j in jets):
        raise RuntimeError('Jet-rank assumption failed; examine the exceptional circuits.')
    if not all(r['theta_rank']==6 for r in rows):
        raise RuntimeError('Sample Jacobian rank was truncated; reduce width or change tolerance.')
    obj=dict(protocol=dict(seed=a.seed,n_circuits=24,centers=[1.7,np.pi,4.8],
                           widths=[.5,.65,.85,1.1,1.45,1.9,2.8,2*np.pi],W=96,
                           noise_sigma=.02,alpha_shift_norm=.1,
                           local_alternative_direction=[1,0,0],
                           predeclared_fit_widths=[.5,1.9]),
             geometry=rows,jets=jets,exponents=exponents)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(obj,separators=(',',':'))+'\n')
    print('all ranks six; median singular slopes',
          [round(float(np.median([r[f'slope_sv{k}'] for r in exponents])),2) for k in (1,2,3)])
    print('saved',a.output)

if __name__=='__main__':main()
