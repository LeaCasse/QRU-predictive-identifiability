"""Compose plot views from the single frozen twin/stream records."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qru.circuit import qru_output

def main_data():
    data=json.loads((ROOT/'data/main_figure_data.json').read_text())
    twins=json.loads((ROOT/'results/predictive/latent_twins.json').read_text())
    streams=json.loads((ROOT/'results/predictive/streams.json').read_text())
    data['exact']=twins['exact'];data['regular_twins']=twins['numerical_trials']
    x=np.linspace(0,2*np.pi,513)
    for row in data['regular_twins']:
        a=np.array(row['theta_a']);b=np.array(row['theta_b']);alpha=np.array(row['alpha'])
        for key,schedule in [('current_rms',alpha),('parallel_rms',1.02*alpha)]:
            row[key]=float(np.sqrt(np.mean((qru_output(x,a,schedule)-qru_output(x,b,schedule))**2)))
        row['future_risk_floor']=row['future_transverse_mse']/4
    data['stream_checks']=streams['checks']
    for view,source in [('stream_summary','controlled_summary'),('stream_trials','controlled_trials'),
                        ('stream_curves','controlled_curves')]:
        data[view]=[r for r in streams[source] if r['horizon']==5]
    return data

def supporting_data():
    data=json.loads((ROOT/'data/supporting_figure_data.json').read_text())
    data['real_summary']=json.loads((ROOT/'results/predictive/streams.json').read_text())['real_summary']
    return data
