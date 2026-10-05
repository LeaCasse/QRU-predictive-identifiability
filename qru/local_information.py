"""Conditional frequency information for the dyadic L=3 QRU.

For Gaussian observation noise, eliminating theta from the local Fisher matrix
gives I(alpha | theta)=Jalpha.T (I-Ptheta) Jalpha / sigma**2.
Only the *local* model is covered. No global identifiability claim is made.
"""
import numpy as np
from scipy.special import ndtr

from .circuit import parameter_shift_prediction_jacobian
from .spectral import architecture_basis
from .controllability import spectral_jacobian
from .pilot_geometry import reference, ALPHA


def conditional_singulars(theta, x, sigma=.02, relative_tol=1e-10, alpha=ALPHA):
    x=np.asarray(x,dtype=float);J=parameter_shift_prediction_jacobian(x,theta,alpha)
    U,s,_=np.linalg.svd(J,full_matrices=False)
    rank=int(np.sum(s>relative_tol*s[0]))
    Ja=x[:,None]*J[:,2::3]
    B=Ja-U[:,:rank]@(U[:,:rank].T@Ja)
    vals=np.linalg.svd(B/np.sqrt(len(x)),compute_uv=False)
    return vals,B,rank,float(s[rank-1]/s[0])


def jet_rank(theta,mu,depth=6):
    """Analytic x derivatives 0..5 of the exact Fourier theta Jacobian."""
    frequencies,constant=architecture_basis(ALPHA)
    assert not constant and len(frequencies)==4
    x=2*np.pi*np.arange(128)/128
    Jc=spectral_jacobian(theta,ALPHA,x,frequencies,include_constant=False)
    rows=[]
    for k in range(depth):
        values=[]
        for w in frequencies:
            values.extend([w**k*np.cos(w*mu+k*np.pi/2),
                           w**k*np.sin(w*mu+k*np.pi/2)])
        rows.append(values)
    singular=np.linalg.svd(np.asarray(rows)@Jc,compute_uv=False)
    return int(np.sum(singular>1e-10*singular[0])),float(singular[-1]/singular[0])


def width_experiment(seed=971,n_circuits=24,centers=(1.7,np.pi,4.8),
                     widths=(.5,.65,.85,1.1,1.45,1.9,2.8,2*np.pi),n_x=96):
    rng=np.random.default_rng(seed);rows=[];jets=[]
    for case in range(n_circuits):
        theta=reference(rng)
        for mu in centers:
            jr,ratio=jet_rank(theta,mu)
            jets.append(dict(circuit=case,center=mu,jet_rank=jr,
                             jet_condition_inverse=ratio))
            for h in widths:
                x=mu+h*(np.arange(n_x)/(n_x-1)-.5)
                vals,B,rank,ratio=conditional_singulars(theta,x)
                v=np.array([1.,0.,0.])
                # Optimal equal-prior simple-hypothesis error after profiling
                # theta, for a *known* local alpha shift direction v.
                strength=.1*np.linalg.norm(B@v)/.02
                bayes_error=ndtr(-strength/2)
                rows.append(dict(circuit=case,center=mu,width=h,
                                 theta_rank=rank,theta_condition_inverse=ratio,
                                 sv1=vals[0],sv2=vals[1],sv3=vals[2],
                                 info_trace=float(n_x*np.sum(vals**2)/.02**2),
                                 alpha1_signal_to_noise=float(strength),
                                 local_bayes_error=float(bayes_error)))
    return rows,jets


def slopes(rows,low=.5,high=1.9):
    out=[]
    groups={}
    for row in rows:groups.setdefault((row['circuit'],row['center']),[]).append(row)
    for (case,mu),rs in groups.items():
        subset=sorted((r for r in rs if low<=r['width']<=high),key=lambda r:r['width'])
        h=np.log([r['width'] for r in subset])
        out.append(dict(circuit=case,center=mu,
                        **{f'slope_sv{k}':float(np.polyfit(h,np.log([r[f'sv{k}'] for r in subset]),1)[0])
                           for k in (1,2,3)}))
    return out
