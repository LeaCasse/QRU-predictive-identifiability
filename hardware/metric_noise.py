"""Metric-constrained gate noise on a reconstructed, ideal-verified ISA template.

The template has the archived 14 RZ and 8 SX operations, but its exact equality
to the executed ISA bytes cannot be checked. Coherent axes are a fixed sphere
sample, never inferred from hardware gaps. This is a sensitivity family with
independently recorded infidelity, not a calibrated identification of its axis.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hardware.calibrated_replay import mean


def reconstructed_gates(row):
    gates=[]
    for l,((t,z,b),a) in enumerate(zip(row['theta'],row['alpha'])):
        angle=a*row['x']+b+(row.get('phase',0.) if l==1 else 0.)
        for left,right,angle_mid in [(0.,np.pi,angle),(np.pi/2,np.pi/2,t)]:
            gates.extend([dict(name='rz',angle=left),dict(name='sx'),dict(name='rz',angle=angle_mid+np.pi),dict(name='sx'),dict(name='rz',angle=right)])
        gates.append(dict(name='rz',angle=z))
    return gates


def sphere_axes(n=4096):
    k=np.arange(n);z=1-2*(k+.5)/n;phi=np.pi*(3-np.sqrt(5))*k;r=np.sqrt(1-z*z)
    return np.column_stack([r*np.cos(phi),r*np.sin(phi),z])


def coherent_means(row,axes,epsilon,readout):
    axes=np.asarray(axes,float);r=np.broadcast_to([0.,0.,1.],axes.shape).copy()
    co,si=np.cos(epsilon),np.sin(epsilon)
    for gate in reconstructed_gates(row):
        if gate['name']=='rz':
            c,s=np.cos(gate['angle']),np.sin(gate['angle']);x,y=r[:,0].copy(),r[:,1].copy()
            r[:,0]=c*x-s*y;r[:,1]=s*x+c*y
        else:
            y,z=r[:,1].copy(),r[:,2].copy();r[:,1]=-z;r[:,2]=y
            r=co*r+si*np.cross(axes,r)+(1-co)*axes*np.sum(axes*r,axis=1)[:,None]
    return (1-2*readout)*r[:,2]


def run(axis_count=4096):
    metrics=json.loads((ROOT/'results/hardware/calibration_metrics.json').read_text());e=metrics['sx_average_error'];em=metrics['readout_error']
    eps=2*np.arcsin(np.sqrt(3*e/2));axes=sphere_axes(axis_count);jobs=[];ideal_error=0.
    ideal_cal=dict(t1_seconds=1.,t2_seconds=1.,readout_p1_given_0=0.,readout_p0_given_1=0.,
                   gates={k:dict(duration_seconds=0.,average_error=0.) for k in ['sx','rz']})
    for filename in ['qpu_baseline.json','qpu_baseline_rep2.json']:
        d=json.loads((ROOT/'results/hardware'/filename).read_text());records=[]
        for xx in sorted({r['x'] for r in d['rows']}):
            pair=[next(r for r in d['rows'] if r['x']==xx and r['branch']==b) for b in 'AB']
            for r in pair:ideal_error=max(ideal_error,abs(mean(reconstructed_gates(r),ideal_cal)-r['ideal_mean']))
            means=[coherent_means(r,axes,eps,em) for r in pair];gaps=means[0]-means[1]
            obs=[(r['counts']['0']-r['counts']['1'])/sum(r['counts'].values()) for r in pair]
            records.append(dict(x=xx,observed_gap=obs[0]-obs[1],
                coherent_grid_min=float(gaps.min()),coherent_grid_max=float(gaps.max())))
        jobs.append(dict(file=filename,rows=records))
    return dict(model='same small coherent rotation after every SX, ideal RZ, symmetric common readout',
        gate_template='reconstructed from the QRU and gate counts, not retained executed ISA bytes',
        gate_infidelity=e,readout_error=em,coherent_rotation_angle=float(eps),axis_count=axis_count,
        axes='deterministic Fibonacci sphere, specified independently of observations',
        worst_ideal_reconstruction_error=ideal_error,
        pauli_pair_gap_bound=float(48*e),
        bound_assumptions='eight single-qubit stochastic Pauli errors per branch, each infidelity at most e, ideal remaining gates, common readout',
        coherent_samples_are_not_rigorous_envelopes=True,acquisitions=jobs)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(run(),indent=2)+'\n')
