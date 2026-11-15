"""Offline seed-203 dephasing sensitivity, with held-out input validation.

This fit uses measured gaps and is not an independently calibrated device model.
No QPU job or account is used. The acquisition console supplies two scalar calibration metrics. Their reduced
depolarizing/readout model is evaluated independently of the gap fit. Complete
historical channels, durations and bound ISA circuit bytes remain unavailable.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qru.noise import density_output

def run():
    metrics=json.loads((ROOT/'results/hardware/calibration_metrics.json').read_text())
    shrinkage=(1-2*metrics['sx_average_error'])**8*(1-2*metrics['readout_error'])
    result=[]
    for filename in ('qpu_baseline.json','qpu_baseline_rep2.json'):
        data=json.loads((ROOT/'results/hardware'/filename).read_text());rows=data['rows']
        x=sorted({r['x'] for r in rows});observed=[];se=[];slope=[];ideal=[]
        for xx in x:
            pair=[next(r for r in rows if r['x']==xx and r['branch']==b) for b in 'AB']
            means=[(r['counts']['0']-r['counts']['1'])/sum(r['counts'].values()) for r in pair]
            observed.append(means[0]-means[1]);se.append(np.sqrt(sum((1-m*m)/sum(r['counts'].values()) for m,r in zip(means,pair))))
            vals={g:[density_output([xx],r['theta'],r['alpha'],g)[0] for r in pair] for g in (0.,1.)}
            ideal.append(vals[1.][0]-vals[1.][1]);slope.append(vals[0.][0]-vals[0.][1]-ideal[-1])
        d=np.array(observed);s=np.array(slope);base=np.array(ideal);e=np.array(se)
        def fit(mask):
            t=np.clip(np.sum(s[mask]*(d-base)[mask]/e[mask]**2)/np.sum(s[mask]**2/e[mask]**2),0,1)
            return float(1-t),base+t*s
        gamma,pred=fit(np.ones(len(x),bool));hold=[]
        for i in range(len(x)):
            mask=np.arange(len(x))!=i;gh,ph=fit(mask)
            hold.append(dict(x=x[i],gamma_without_input=gh,predicted_gap=float(ph[i]),observed_gap=observed[i],shot_se=se[i]))
        result.append(dict(file=filename,job_id=data['job_id'],calibration_timestamp=data['calibration_last_update'],
            x=x,observed_gap=observed,ideal_gap=ideal,slope=slope,shot_se=se,
            fitted_gamma=gamma,predicted_gap=pred.tolist(),gap_rmse=float(np.sqrt(np.mean((pred-d)**2))),
            weighted_residual_sum=float(np.sum(((pred-d)/e)**2)),leave_one_input_out=hold,
            reduced_calibration_prediction=dict(shrinkage=shrinkage,
                predicted_gaps=(shrinkage*base).tolist(),gap_rmse=float(np.sqrt(np.mean((shrinkage*base-d)**2))))))
    return dict(model='single interlayer Z dephasing, common gamma',
        calibration_status='two independent scalar metrics, incomplete gate-channel calibration',
        calibration_metrics_file='calibration_metrics.json',
        missing=['historical T1 and T2','gate durations and full error model','readout assignment probabilities','executed bound ISA circuits'],
        estimation='weighted least squares on paired gaps, weights from binomial plug-in variances',
        inference='Sensitivity and leave-one-input-out diagnostic, not a calibrated causal attribution.',acquisitions=result)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(run(),indent=2)+'\n')
