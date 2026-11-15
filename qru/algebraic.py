"""Polynomial lifts of all attainable path classes (optional SymPy dependency).

The output defines exact real feasibility problems. It does not claim to solve
quantifier elimination or enumerate connected components in polynomial time.
"""
from itertools import product
from .phase_design import collision_partition


def lifted_amplitudes(depth):
    import sympy as sp
    if not isinstance(depth, int) or depth < 1:
        raise ValueError('depth must be a positive integer')
    variables=[]; circles=[]; rotations=[]; phases=[]
    for l in range(depth):
        ct,st,cz,sz,cb,sb=sp.symbols(f'ct{l} st{l} cz{l} sz{l} cb{l} sb{l}',real=True)
        variables.extend((ct,st,cz,sz,cb,sb))
        circles.extend((ct**2+st**2-1,cz**2+sz**2-1,cb**2+sb**2-1))
        rotations.append(sp.Matrix([[cz,-sz*ct,sz*st],[sz,cz*ct,-cz*st],[0,st,ct]]))
        phases.append({1:cb+sp.I*sb,0:sp.Integer(1),-1:cb-sp.I*sb})
    plus=sp.Matrix([[1,0,-sp.I],[0,0,0],[sp.I,0,1]])/2
    P={1:plus,-1:sp.conjugate(plus),0:sp.diag(0,1,0)}
    amplitudes={}
    for s in product((-1,1), *[(-1,0,1)]*(depth-1)):
        v=sp.Matrix([0,0,1])
        for l,k in enumerate(s):v=rotations[l]*P[k]*v
        amplitudes[s]=v[2]*sp.prod(phases[l][k] for l,k in enumerate(s))
    return variables,circles,amplitudes


def fiber_equations(depth, alpha, target):
    """All circle equations and real/imaginary Fourier residual equations.

    target maps exact frequencies to complex coefficients. Missing coefficients
    are zero. Frequencies outside the circuit support are also constrained.
    """
    import sympy as sp
    from fractions import Fraction
    variables,circles,amp=lifted_amplitudes(depth)
    groups=collision_partition(amp,alpha)
    target={Fraction(str(k)):sp.sympify(v) for k,v in target.items()}
    equations=list(circles)
    for w in sorted(set(groups)|set(target)):
        residual=sum((amp[s] for s in groups.get(w,[])),sp.Integer(0))-target.get(w,0)
        equations.extend((sp.re(residual).expand(complex=True),sp.im(residual).expand(complex=True)))
    return variables,[e for e in equations if e!=0],amp
