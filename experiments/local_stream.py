"""Causal narrow-input stream pilot; held-out probes are never used to update."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qru.local_stream import benchmark

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--first-seed',type=int,default=980)
    p.add_argument('--sequences',type=int,default=3)
    args=p.parse_args();seeds=list(range(args.first_seed,args.first_seed+args.sequences))
    print('Streaming',seeds,flush=True)
    rows=benchmark(seeds)
    out=dict(protocol=dict(seeds=seeds,windows_per_stream=32,observations_per_window=24,
                           noise_sigma=.02,input_width=.5,
                           geometry='centers pi until t=12, then pi+1.3*sin(2pi*(t-13)/7) through t=22',
                           changed_concept='t=6..22',reset='t=23',
                           budget_forward_evaluations_per_window=2100,
                           regularized_alpha_penalty=.01,
                           information_gate='allow alpha if 0.1*largest singular value(B)/0.02 >= 2',
                           oracle_probe_only='broad grid MSE after prediction; never used in updates',
                           gradients='exact parameter shift, no quantum shot noise'),rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,separators=(',',':'))+'\n')
    print('saved',args.output,len(rows),'rows',flush=True)

if __name__=='__main__':main()
