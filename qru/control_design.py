"""Constructive single-layer phase queries and localized channel diagnostics."""
import numpy as np
from .circuit import qru_output,validate
from .tomography import analytic_path_amplitudes,_bloch_rx,_bloch_rz,_P_BY_SIGN,path_set


def optimal_single_layer(theta_a,theta_b,alpha,x,error_probability=.05):
    """Maximize the absolute mean gap over phase, exactly at each listed input.

    Three phase evaluations recover the first-harmonic phase response. Input
    selection is over the provided finite grid. The returned shot count is a
    Hoeffding upper bound for known-pair threshold classification, not an
    experimental claim or a bound for unknown nuisance parameters.
    """
    a,alpha=validate(theta_a,alpha);b,_=validate(theta_b,alpha)
    x=np.asarray(x,float).reshape(-1)
    if a.shape!=b.shape or not len(x) or not np.all(np.isfinite(x)) or not 0<error_probability<.5:
        raise ValueError('Require matched circuits, finite nonempty inputs and 0<error_probability<1/2.')
    phases=2*np.pi*np.arange(3)/3;records=[]
    for layer in range(len(alpha)):
        delta=[]
        for phi in phases:
            aa=a.copy();bb=b.copy();aa[layer,2]+=phi;bb[layer,2]+=phi
            delta.append(qru_output(x,aa,alpha)-qru_output(x,bb,alpha))
        delta=np.array(delta);constant=delta.mean(axis=0)
        carrier=np.mean(delta*np.exp(-1j*phases[:,None]),axis=0)
        phi=-np.angle(carrier)+np.where(constant<0,np.pi,0.)
        predicted=np.abs(constant)+2*np.abs(carrier);j=int(np.argmax(predicted))
        aa=a.copy();bb=b.copy();aa[layer,2]+=phi[j];bb[layer,2]+=phi[j]
        means=[float(np.clip(qru_output([x[j]],t,alpha)[0],-1.,1.)) for t in (aa,bb)]
        gap=abs(means[0]-means[1])
        shots=None if gap<1e-12 else (1 if gap>=2-1e-12 else int(np.ceil(8*np.log(1/error_probability)/gap**2)))
        records.append(dict(layer=layer+1,x=float(x[j]),phase=float(phi[j]%(2*np.pi)),means=means,
                            absolute_gap=gap,phase_fit_error=float(abs(gap-predicted[j])),
                            sufficient_shots=shots,error_probability=error_probability))
    return sorted(records,key=lambda r:r['absolute_gap'],reverse=True)


def localized_amplitudes(theta,layer,gamma):
    """One Z-dephasing interval after the zero-indexed layer, including output.

    These amplitudes depend on both the physical realization and channel
    location; they are not computed from the ideal amplitude vector alone.
    """
    theta=np.asarray(theta,float)
    if theta.ndim!=2 or theta.shape[1]!=3 or not 0<=layer<len(theta) or not 0<=gamma<=1:
        raise ValueError('Invalid circuit, channel layer or dephasing factor.')
    mixers=[_bloch_rz(z)@_bloch_rx(t) for t,z,b in theta]
    mixers[layer]=np.diag([gamma,gamma,1.])@mixers[layer]
    amplitudes={}
    for path in path_set(len(theta)):
        v=np.array([0.,0.,1.],complex)
        for l,k in enumerate(path):v=mixers[l]@_P_BY_SIGN[k]@v
        amplitudes[path]=complex(v[2]*np.exp(1j*np.dot(path,theta[:,2])))
    return amplitudes


def localized_output(x,theta,alpha,layer,gamma):
    """Independent density-matrix evaluation of one localized dephasing interval."""
    from .noise import _layer
    theta,alpha=validate(theta,alpha)
    if not 0<=layer<len(theta) or not 0<=gamma<=1:raise ValueError('Invalid channel settings.')
    values=[]
    for xx in np.asarray(x,float).reshape(-1):
        rho=np.diag([1.,0.]).astype(complex)
        for l,(parameters,scale) in enumerate(zip(theta,alpha)):
            u=_layer(parameters,scale,xx);rho=u@rho@u.conj().T
            if l==layer:rho=gamma*rho+(1-gamma)*np.diag(np.diag(rho))
        values.append(float((rho[0,0]-rho[1,1]).real))
    return np.array(values)
