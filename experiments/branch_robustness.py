"""Two additional targets for the paired training-noise protocol."""
import argparse,json
from pathlib import Path
from branch_selection import run
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    studies=[run(128,1200,r,phase) for r,phase in [(.4,-.6),(.9,1.1)]]
    payload=dict(studies=studies,scope='Two additional specified sinusoidal targets, 128 paired initializations each. Same five modes, budget and clean completion.')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2)+'\n')
