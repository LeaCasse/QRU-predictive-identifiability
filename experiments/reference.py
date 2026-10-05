"""Check packaged tables and core mathematical identities; optional full rerun.

Run from any directory: python experiments/reference.py [--recompute]
The optional rerun writes CSV and a comparison report only to build/runs/<timestamp>/.
Archived results are never overwritten. Tomography trial differences are reported.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qru.results_io import read_result
from qru.circuit import qru_output,random_theta,parameter_shift_prediction_jacobian
from qru.spectral import architecture_basis,fourier_design_matrix
from qru.tomography import analytic_path_amplitudes,reconstruct_from_paths
from qru.adaptation import path_schedule_jacobian

def main():
    manifest=json.loads((ROOT/'data/provenance.json').read_text())['reference_tables'][:25]
    frames={Path(m['file']).name:read_result(ROOT/'results/reference',Path(m['file']).name) for m in manifest}
    for m in manifest:assert frames[Path(m['file']).name].shape==(m['rows'],m['columns'])
    rng=np.random.default_rng(93025);x=np.linspace(-2,5,73)
    for depth in range(1,5):
        for _ in range(3):
            theta=random_theta(depth,rng);alpha=rng.uniform(.2,3,depth)
            y=qru_output(x,theta,alpha)
            paths=analytic_path_amplitudes(theta)
            np.testing.assert_allclose(y,reconstruct_from_paths(x,alpha,paths),atol=1e-13,rtol=1e-12)
            j=parameter_shift_prediction_jacobian(x,theta,alpha)
            ja=path_schedule_jacobian(x,alpha,paths)
            np.testing.assert_allclose(ja,x[:,None]*j[:,2::3],atol=1e-12,rtol=1e-11)
            np.testing.assert_allclose(j[:,-2],0,atol=1e-14)
    rank=frames['02_rank_landscape.csv']
    assert (rank['rank']==6).sum()==1000
    a=np.array([1.,2.,4.]);freqs,c=architecture_basis(a)
    X=2*np.pi*np.arange(128)/128;A,_=fourier_design_matrix(X,freqs,c)
    np.testing.assert_allclose(A.T@A/128,np.eye(8)/2,atol=1e-14)
    for name in ['04_controlled_cases.csv','04_matched_drift_stress.csv']:
        f=frames[name]
        for _,row in f.iterrows():
            vals={'theta_only':row.theta_only_val_mse,'surgery_only':row.surgery_only_val_mse,'surgery_plus_theta':row.surgery_tuned_val_mse}
            valid={k:v for k,v in vals.items() if pd.notna(v)}
            assert row.selected_model==min(valid,key=valid.get)
        np.testing.assert_allclose(f.architecture_gain,np.maximum(0,f.e_theta**2-f.e_aug**2),atol=1e-14)
    print('PASS: 25 tables / 2460 records, path identities, direct bias-derived Jalpha, rank, metric and validation selection.')
    if '--recompute' not in sys.argv:return
    from qru.studies import run_spectral_audit,run_controllability_study,run_tomography_study,run_reconfiguration_study
    out=ROOT/'build/runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    for fn in [run_spectral_audit,run_controllability_study,run_tomography_study,run_reconfiguration_study]:
        fn(out);print(fn.__name__+' completed',flush=True)
    report=[]
    for name,original in frames.items():
        current=pd.read_csv(out/name,float_precision='round_trip')
        missing=sorted(set(original)-set(current));extra=sorted(set(current)-set(original))
        assert len(current)==len(original),(name,'row count')
        assert not missing and not extra,(name,missing,extra)
        differences=[]
        for col in original:
            if pd.api.types.is_numeric_dtype(original[col]) and original[col].dtype!=bool:
                equal=np.isclose(original[col],current[col],rtol=1e-5,atol=1e-12,equal_nan=True)
            else:equal=original[col].fillna('__NA__').to_numpy()==current[col].fillna('__NA__').to_numpy()
            if not np.all(equal):differences.append({'column':col,'different_records':int(np.sum(~equal))})
        report.append({'source':name,'rows':len(original),'schema':'complete','differences':differences})
    (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Full comparison:',out/'comparison.json')
    print('Tables with numerical/label differences:',[r['source'] for r in report if r['differences']])

if __name__=='__main__':main()
