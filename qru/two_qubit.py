"""Small exact-grid Fourier witness: two locally encoded qubits, optional CNOT.

Both qubits share alpha_l. Each layer applies Ry(alpha_l*x+b), Rx, Rz
to each qubit and optionally CNOT(0,1). Readout is Z on qubit 0.
The integer schedules permit exact discrete Fourier reconstruction on 256
equally spaced points, up to floating point error. This is exploratory.
"""
import numpy as np
from math import factorial

M=256
xs=2*np.pi*np.arange(M)/M
CNOT=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],complex)
I=np.eye(2)

def circuit(theta,alpha,entangle):
    state=np.zeros((M,4),complex);state[:,0]=1
    for l,a in enumerate(alpha):
        for q in range(2):
            tx,tz,b=theta[l,q]
            phi=a*xs+b
            ry=np.zeros((M,2,2),complex)
            ry[:,0,0]=ry[:,1,1]=np.cos(phi/2)
            ry[:,1,0]=np.sin(phi/2);ry[:,0,1]=-np.sin(phi/2)
            rx=np.array([[np.cos(tx/2),-1j*np.sin(tx/2)],[-1j*np.sin(tx/2),np.cos(tx/2)]])
            rz=np.diag([np.exp(-1j*tz/2),np.exp(1j*tz/2)])
            unit=np.einsum('ij,njk->nik',rz@rx,ry)
            gate=(np.einsum('nij,ab->niajb',unit,I).reshape(M,4,4) if q==0
                  else np.einsum('ab,nij->naibj',I,unit).reshape(M,4,4))
            state=np.einsum('nij,nj->ni',gate,state)
        if entangle:state=state@CNOT.T
    return (np.abs(state[:,0])**2+np.abs(state[:,1])**2
           -np.abs(state[:,2])**2-np.abs(state[:,3])**2)

def jets(y,c,d,cutoff):
    fourier=np.fft.fft(y)/M
    w=np.fft.fftfreq(M,1/M)
    f=(np.abs(w)<=cutoff)
    return np.array([np.real(np.sum(fourier[f]*(1j*w[f])**k*np.exp(1j*w[f]*c)))/factorial(k)
                     for k in range(d+1)])

def run(L,entangle,seed,center=.73):
    rng=np.random.default_rng(seed)
    theta=rng.uniform(-1.5,1.5,(L,2,3))
    alpha=2.**np.arange(L)
    c=center;degree=min(6*L+1,16);cutoff=2*sum(alpha)+1
    flat=theta.ravel();T=[]
    for j in range(len(flat)):
        pos=flat.copy();neg=flat.copy();pos[j]+=np.pi/2;neg[j]-=np.pi/2
        T.append(jets((circuit(pos.reshape(theta.shape),alpha,entangle)
                       -circuit(neg.reshape(theta.shape),alpha,entangle))/2,
                      c,degree,cutoff))
    T=np.array(T).T
    A=np.array([c*(T[:,6*j+2]+T[:,6*j+5])+np.r_[0,(T[:-1,6*j+2]+T[:-1,6*j+5])]
                for j in range(L)]).T
    def rank(t):
        s=np.linalg.svd(t,compute_uv=False);return int(sum(s>1e-9*s[0]))
    rr=[(rank(T[:k+1]),rank(np.c_[T[:k+1],A[:k+1]])) for k in range(degree+1)]
    visible=[(k,b-a) for k,(a,b) in enumerate(rr) if b>a]
    return rr[-1],visible[:L+2],np.max(np.abs(circuit(theta,alpha,entangle)))

