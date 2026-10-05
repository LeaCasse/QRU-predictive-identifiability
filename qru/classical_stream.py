"""Classical fixed-frequency Fourier control with eight linear coefficients.

The QRU dyadic outer support is {1,3,5,7}. Its coefficients here are fitted
freely, so this control has a different hypothesis class. It uses no quantum
circuit evaluations during online fitting after a common exact warm start.
"""
import numpy as np
from .circuit import qru_output
from .spectral import architecture_basis,fourier_design_matrix
from .pilot_geometry import ALPHA
from .local_stream import stream


def benchmark(seeds,lambda_ridge=.01,T=32):
    frequencies,constant=architecture_basis(ALPHA)
    x_ref=2*np.pi*np.arange(128)/128
    F,_=fourier_design_matrix(x_ref,frequencies,constant)
    grid=np.linspace(0,2*np.pi,256,endpoint=False)+.001
    G,_=fourier_design_matrix(grid,frequencies,constant)
    rows=[]
    for seed in seeds:
      for kind in ('stationary','parameter','spectral','outside'):
        theta,windows=stream(seed,kind,T=T)
        c0=np.linalg.lstsq(F,qru_output(x_ref,theta,ALPHA),rcond=None)[0]
        c=c0.copy();history=[]
        for win in windows:
          A,_=fourier_design_matrix(win['x'],frequencies,constant)
          pred=A@c;history.append((A,win['y']))
          recent=history[-4:];M=np.vstack([a for a,_ in recent]);y=np.concatenate([y for _,y in recent])
          rows.append(dict(seed=seed,kind=kind,policy=f'fourier_recent4_ridge_{lambda_ridge}',t=win['t'],phase=win['phase'],
                           prequential_clean_mse=float(np.mean((pred-win['clean'])**2)),
                           broad_probe_mse=float(np.mean((G@c-win['target'](grid))**2)),
                           training_samples=len(y),online_quantum_evaluations=0))
          c=np.linalg.solve(M.T@M/len(y)+lambda_ridge*np.eye(len(c)),M.T@y/len(y)+lambda_ridge*c0)
    return rows
