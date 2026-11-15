"""Offline replay from independent historical calibration and bound ISA gates.

Required calibration JSON: backend, physical_qubit, calibration_timestamp,
source, t1_seconds, t2_seconds, readout_p1_given_0, readout_p0_given_1, and gates.
Each gates entry (sx, x, rz as used) contains duration_seconds and average_error.
Required circuit JSON: backend, physical_qubit, calibration_timestamp, source,
rows [{pub_index, gates:[{name:'rz', angle:...}, {name:'sx'}, ...]}].
All rotations are bound and in execution order, measurement is implicit.
The model is thermal relaxation plus depolarization matched to gate infidelity.
It is a declared calibration-based approximation, not a complete device model.
"""
import argparse,json
from pathlib import Path
import numpy as np


def gate_channel(vector, name, angle, calibration):
    if name not in ('rz','sx','x'):
        raise ValueError('Only one-qubit rz, sx and x instructions are supported.')
    r=np.asarray(vector,float)
    if not np.isfinite(angle):raise ValueError('Gate angle must be finite.')
    phi=angle if name=='rz' else (np.pi/2 if name=='sx' else np.pi)
    c,s=np.cos(phi),np.sin(phi)
    R=np.array([[c,-s,0],[s,c,0],[0,0,1]]) if name=='rz' else np.array([[1,0,0],[0,c,-s],[0,s,c]])
    r=R@r
    spec=calibration['gates'][name];t=float(spec['duration_seconds']);err=float(spec['average_error'])
    t1=float(calibration['t1_seconds']);t2=float(calibration['t2_seconds'])
    if not np.all(np.isfinite([t,err,t1,t2])) or not(t>=0 and 0<=err<=.5 and t1>0 and 0<t2<=2*t1):
        raise ValueError('Invalid duration, gate error or relaxation constants.')
    a=np.exp(-t/t2);b=np.exp(-t/t1);trace=2*a+b
    # Average infidelity of an affine qubit channel is (3-tr(A))/6.
    factor=(3-6*err)/trace
    if factor>1+1e-10:
        raise ValueError('Reported gate error is below the thermal floor of this model.')
    return np.clip(factor,0,1)*(np.array([a,a,b])*r+np.array([0,0,1-b]))


def mean(gates,calibration):
    r=np.array([0.,0.,1.])
    for gate in gates:r=gate_channel(r,gate['name'],float(gate.get('angle',0)),calibration)
    e01=float(calibration['readout_p1_given_0']);e10=float(calibration['readout_p0_given_1'])
    if not(0<=e01<=1 and 0<=e10<=1):raise ValueError('Invalid readout probabilities.')
    return float(e10-e01+(1-e01-e10)*r[2])


def replay(record,calibration,circuits):
    for obj in (calibration,circuits):
        if not obj.get('source'):raise ValueError('Independent input provenance is required.')
        for key in ('backend','physical_qubit'):
            if obj[key]!=record[key]:raise ValueError('Hardware identity mismatch: '+key)
        if obj['calibration_timestamp']!=record['calibration_last_update']:
            raise ValueError('Historical calibration timestamp mismatch.')
    by_index={r['pub_index']:r['gates'] for r in circuits['rows']}
    if len(by_index)!=len(circuits['rows']) or set(by_index)!={r['pub_index'] for r in record['rows']}:
        raise ValueError('Bound circuits must match every recorded PUB exactly once.')
    rows=[]
    for row in record['rows']:
        gates=by_index[row['pub_index']]
        counts={name:sum(gate['name']==name for gate in gates) for name in {gate['name'] for gate in gates}}
        expected={k:v for k,v in record['isa_ops'].items() if k!='measure'}
        if counts!=expected:raise ValueError('Bound gate counts differ from archived ISA counts.')
        ideal_cal={**calibration,'readout_p1_given_0':0.,'readout_p0_given_1':0.,
                   'gates':{k:dict(duration_seconds=0.,average_error=0.) for k in counts}}
        if abs(mean(gates,ideal_cal)-row['ideal_mean'])>1e-10:
            raise ValueError('Bound circuit does not reproduce the archived ideal response.')
        pred=mean(gates,calibration)
        n=sum(row['counts'].values());obs=(row['counts']['0']-row['counts']['1'])/n
        rows.append(dict(pub_index=row['pub_index'],x=row['x'],branch=row['branch'],predicted_mean=pred,
                         observed_mean=obs,predictive_shot_se=float(np.sqrt((1-pred**2)/n))))
    return dict(model='gate thermal relaxation, matched depolarization, asymmetric readout',
                fitted_to_branch_gaps=False,calibration_source=calibration['source'],
                circuit_source=circuits['source'],rows=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('record','calibration','circuits','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();payload=replay(*[json.loads(Path(getattr(a,k)).read_text()) for k in ('record','calibration','circuits')])
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2)+'\n')
