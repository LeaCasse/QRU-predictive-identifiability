"""Reproduce architecture-conditioned local QRU observability diagnostics."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qru.observability import sweep


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=8)
    args=parser.parse_args()
    rows=sweep(samples=args.samples)
    payload=dict(protocol=dict(seed=23981,samples=args.samples,depths=[1,2,3,4],
                               centers=[.73,3.141592653589793],rank_relative_tolerance=1e-10,
                               note='Local tangent ranks; no finite-shot or global-rank proof.'),rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+'\n')
    for depth in range(1,5):
        for schedule in ('dyadic','collision','incommensurate','inactive_uploads'):
            selection=[r for r in rows if r['depth']==depth and r['schedule']==schedule
                       and r['parameters']=='all']
            if selection:
                print(depth,schedule,'rank/order frequencies',Counter(
                    (r['theta_rank'],r['first_visible_order']) for r in selection))
    print('rows',len(rows),'saved',args.output)


if __name__=='__main__':main()
