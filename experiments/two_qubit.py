"""Reproduce all 36 two-qubit local-rank witnesses (optional CNOT per layer)."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qru.two_qubit import run

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=[]
    for depth in (1,2,3):
        for entangle in (False,True):
            for center in (.73,np.pi):
                for seed in (40,41,42):
                    ranks,visible,max_abs=run(depth,entangle,seed,center)
                    rows.append(dict(depth=depth,cnot=entangle,center=center,seed=seed,
                                     theta_rank=ranks[0],joint_rank=ranks[1],
                                     visible_orders=visible,max_abs_mean=float(max_abs)))
    result=dict(protocol=dict(seeds=[40,41,42],centers=[.73,float(np.pi)],
                depths=[1,2,3],readout='Z on qubit 0',shared_encoding_per_layer=True,
                fourier_grid=256,rank_relative_tolerance=1e-9,
                scope='Local numerical witnesses; ranks do not measure entanglement.'),rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Saved 36 two-qubit witnesses:',args.output)

if __name__=='__main__':main()
