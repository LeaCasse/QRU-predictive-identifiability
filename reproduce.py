"""Reproduce figures, scientific checks or one experiment. Outputs go to build/."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = {
    'twins': 'latent_twins', 'streams': 'predictive_identifiability',
    'real': 'predictive_identifiability', 'observability': 'observability',
    'two-qubit': 'two_qubit', 'optimizer': 'optimizer',
    'detectability': 'detectability', 'finite-alternatives': 'finite_alternatives',
    'local-information': 'local_information', 'local-stream': 'local_stream',
    'classical-stream': 'classical_stream', 'pilot-geometry': 'pilot_geometry',
    'reference': 'reference',
}

def run(script, args=()):
    env = os.environ.copy()
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[key] = '1'
    subprocess.run([sys.executable, *script, *args], cwd=ROOT, env=env, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', choices=['checks','figures',*EXPERIMENTS])
    args, extra = parser.parse_known_args()
    if args.target in ('checks', 'figures') and extra:
        parser.error('This target takes no additional arguments.')
    if args.target == 'checks':
        run(['experiments/reference.py'])
        run(['-m','unittest','discover','-s','tests','-v'])
    elif args.target == 'figures':
        for script, options in [('build_main_figures.py',[]),
                                ('build_figures.py',[]),
                                ('build_supporting_figures.py',['--all'])]:
            run(['figures/'+script], options)
        print('21 article figures: build/figures/ (PDF and PNG).')
    else:
        if args.target == 'reference':
            run(['experiments/reference.py'],extra)
            return
        if not any(x=='--output' or x.startswith('--output=') for x in extra):
            extra += ['--output',str(ROOT/'build/runs'/(args.target+'.json'))]
        if args.target == 'real':
            # The real replay is independent of the controlled experiment.
            extra += ['--real-only']
        run(['experiments/'+EXPERIMENTS[args.target]+'.py'], extra)

if __name__ == '__main__':
    main()
