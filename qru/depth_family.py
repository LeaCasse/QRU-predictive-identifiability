"""A continuous L-branch collision family and logarithmic layer interrogation."""
import numpy as np
from scipy.stats import binom
from functools import lru_cache


def branch_tag(depth,branch):
    if depth<1 or not 1<=branch<=depth:raise ValueError('Require 1<=branch<=depth.')
    return np.array([-1]*(branch-1)+[1]+[0]*(depth-branch),int)


def branch_theta(depth,branch,radius=1.,phase=0.):
    """Realize r*cos((alpha_k-sum_{j<k}alpha_j)*x+phase) at every schedule.

    Branch numbering starts at one. The first k uploads accumulate coherently;
    after k<L, the selected quadrature is stored in the encoding-invariant Y
    component and finally mapped to Z. All angles are physical qubit rotations.
    """
    if not np.isfinite(radius) or not 0<radius<=1 or not np.isfinite(phase):raise ValueError('Invalid radius or phase.')
    tag=branch_tag(depth,branch);theta=np.zeros((depth,3));k=branch-1
    if k>0:theta[k-1,1]=np.pi
    theta[k,2]=phase
    if branch==depth:theta[k,0]=np.arccos(radius)
    else:theta[k,0]=-np.arcsin(radius);theta[-1,0]=np.pi/2
    return theta


def uniform_branch_theta(depth,branch,radius=1.,phase=0.):
    """Alternating odd prefixes give ceil(L/2) classes at equal encoding scales."""
    stop=2*branch-1
    if stop>depth:raise ValueError('Uniform branch index exceeds the available odd prefixes.')
    theta=branch_theta(depth,stop,radius,phase)
    theta[:stop-1,1]=np.pi
    return theta


@lru_cache(maxsize=128)
def round_budget(radius,rounds,error=.05):
    """First n with exact worst binomial threshold error at most error/rounds."""
    if not 0<radius<=1 or not 0<error<.5 or rounds<1:raise ValueError('Invalid shot-bound inputs.')
    candidates=np.array([radius,-radius])
    for n in range(1,100001):
        counts=np.arange(n+1);z=2*counts/n-1
        decision=np.argmin(abs(candidates[:,None]-z[None,:]),axis=0)
        risks=[float(binom.pmf(counts,n,(1+mu)/2)[decision!=j].sum()) for j,mu in enumerate(candidates)]
        if max(risks)*rounds<=error:return n,max(risks)
    raise ValueError('Shot budget exceeds the supported numerical scan.')


def interrogate_prefixes(prefixes,true_branch,radius,phase,rng,error=.05):
    """Binary active-prefix search with the full posterior retained.

    A pi phase reverses the mean iff the hidden prefix reaches the queried
    layer. Known r,phase allow x+phase=0. Exact binomial tails set a fixed budget
    with hard-search error at most error under every branch. Prefixes are
    strictly increasing positive stopping-layer indices.
    """
    prefixes=np.asarray(prefixes,int);K=len(prefixes)
    if K<1 or np.any(np.diff(prefixes)<=0) or prefixes[0]<1 or not 1<=true_branch<=K:
        raise ValueError('Invalid ordered prefixes or target branch.')
    if K==1:return dict(posterior=np.ones(1),records=[],shots=0,selected=1,per_round_error_bound=0.)
    R=int(np.ceil(np.log2(K)));n,bound=round_budget(float(radius),R,error)
    x=(-phase)%(2*np.pi);lo,hi=0,K-1;logp=np.zeros(K);records=[]
    for _ in range(R):
        mid=(lo+hi+1)//2;ell=int(prefixes[mid])
        means=np.where(prefixes>=ell,-radius,radius)
        plus=int(rng.binomial(n,(1+means[true_branch-1])/2));sample=2*plus/n-1
        if lo<hi:
            if sample>=0:hi=mid-1
            else:lo=mid
        probabilities=(1+means)/2;ll=np.zeros(K)
        if plus:ll+=plus*np.log(np.maximum(probabilities,1e-300))
        if n-plus:ll+=(n-plus)*np.log(np.maximum(1-probabilities,1e-300))
        logp+=ll;logp-=logp.max()
        records.append(dict(layer=ell,x=float(x),phase=np.pi,shots=n,positive=plus))
    posterior=np.exp(logp);posterior/=posterior.sum()
    return dict(posterior=posterior,records=records,shots=R*n,per_round_error_bound=bound,selected=lo+1)


def interrogation(depth,true_branch,radius,phase,rng,error=.05):
    """Dyadic L-class instance of the active-prefix interrogation protocol."""
    return interrogate_prefixes(np.arange(1,depth+1),true_branch,radius,phase,rng,error)
