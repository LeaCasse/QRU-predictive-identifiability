"""Export the complete polynomial fiber equations at a specified depth.

Default: L=3, alpha=(1,2,4), response cos(x). Requires requirements-symbolic.txt.
JSON equations use SymPy syntax and equal zero. All variables are real.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qru.algebraic import fiber_equations
if __name__=='__main__':
    import sympy as sp
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--depth',type=int,default=3);p.add_argument('--output',required=True);a=p.parse_args()
    alpha=[2**k for k in range(a.depth)]
    variables,equations,amp=fiber_equations(a.depth,alpha,{1:sp.Rational(1,2),-1:sp.Rational(1,2)})
    payload=dict(depth=a.depth,alpha=alpha,target='cos(x)',domain='real',variables=list(map(str,variables)),
        equations=list(map(str,equations)),path_count=len(amp),
        interpretation='Necessary and sufficient real polynomial constraints. No component enumeration is asserted.')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2)+'\n')
