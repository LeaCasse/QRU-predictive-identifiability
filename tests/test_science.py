"""Independent identities, frozen-trial consistency and hardware stage gates."""
import json
import hashlib
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.stats import beta

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qru.circuit import qru_output, parameter_shift_prediction_jacobian
from qru.tomography import (analytic_path_amplitudes, grouped_path_coefficients,
                            reconstruct_from_paths, minimum_phase_samples,
                            recover_paths_from_phase_cycle, max_absolute_path_error)
from qru.adaptation import path_schedule_jacobian, finite_difference_schedule_jacobian
from qru.optimizer import schur_step
from qru.latent_twins import exact_twin
from qru.predictive_identifiability import mathematical_checks, paired_summary, controlled_stream
from qru.two_qubit import run as two_qubit

def read(name):
    if name=='data/main_figure_data.json':
        from figures.data_io import main_data
        return main_data()
    return json.loads((ROOT/name).read_text())

def independent_matrix_output(x,theta,alpha):
    result=[]
    for value in x:
        psi=np.array([1.,0.],complex)
        for (tx,tz,bias),a in zip(theta,alpha):
            angle=a*value+bias
            ry=np.array([[np.cos(angle/2),-np.sin(angle/2)],
                         [np.sin(angle/2),np.cos(angle/2)]])
            rx=np.array([[np.cos(tx/2),-1j*np.sin(tx/2)],
                         [-1j*np.sin(tx/2),np.cos(tx/2)]])
            rz=np.diag(np.exp(np.array([-1j,1j])*tz/2))
            psi=rz@rx@ry@psi
        result.append(abs(psi[0])**2-abs(psi[1])**2)
    return np.array(result)

class ScientificChecks(unittest.TestCase):
    def test_frozen_records_are_intact(self):
        for name,expected in read('data/provenance.json')['frozen_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),expected,name)

    def test_direct_statevector_and_path_sum(self):
        rng=np.random.default_rng(48103);x=rng.uniform(-3,7,29)
        for depth in range(1,5):
            theta=rng.uniform(-2,2,(depth,3));alpha=rng.uniform(.2,3,depth)
            paths=analytic_path_amplitudes(theta)
            self.assertEqual(len(paths),2*3**(depth-1))
            np.testing.assert_allclose(qru_output(x,theta,alpha),
                                      independent_matrix_output(x,theta,alpha),rtol=0,atol=2e-14)
            np.testing.assert_allclose(qru_output(x,theta,alpha),
                                      reconstruct_from_paths(x,alpha,paths),rtol=0,atol=2e-14)

    def test_schedule_derivative_and_gauge(self):
        rng=np.random.default_rng(619);x=rng.uniform(-2,5,41)
        for depth in range(1,5):
            theta=rng.uniform(-2,2,(depth,3));alpha=rng.uniform(.3,2,depth)
            jac=parameter_shift_prediction_jacobian(x,theta,alpha)
            direct=x[:,None]*jac[:,2::3]
            paths=analytic_path_amplitudes(theta)
            np.testing.assert_allclose(path_schedule_jacobian(x,alpha,paths),direct,atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(finite_difference_schedule_jacobian(x,theta,alpha),direct,atol=3e-9,rtol=1e-7)
            np.testing.assert_allclose(jac[:,-2],0,atol=2e-15)

    def test_phase_cycle_design_and_recovery(self):
        rng=np.random.default_rng(942)
        for alpha,expected in [(np.array([1.,2.]),2),(np.array([1.,2.,3.]),3),
                               (np.array([1.,2.,4.]),5),(np.array([1.,2.,4.,8.]),7)]:
            depth=len(alpha);theta=rng.uniform(-2,2,(depth,3))
            self.assertEqual(minimum_phase_samples(alpha),expected)
            recovered=recover_paths_from_phase_cycle(theta,alpha,2*np.pi*np.arange(128)/128)
            self.assertLess(max_absolute_path_error(analytic_path_amplitudes(theta),recovered['amplitudes']),1e-12)

    def test_exact_and_regular_twins(self):
        source=read('results/predictive/latent_twins.json')
        figure=read('data/main_figure_data.json');x=np.linspace(0,2*np.pi,513)
        self.assertEqual(source['exact'],figure['exact'])
        self.assertEqual(source['exact']['delta'],.02)
        larger=exact_twin(delta=.05)
        for key in ['future_mse','future_rms','future_risk_lower_bound']:
            self.assertAlmostEqual(larger[key],figure['illustration'][key],places=14)
        for row in source['numerical_trials']:
            a=np.array(row['theta_a']);b=np.array(row['theta_b']);alpha=np.array(row['alpha'])
            pa=analytic_path_amplitudes(a);pb=analytic_path_amplitudes(b)
            ga=grouped_path_coefficients(alpha,pa);gb=grouped_path_coefficients(alpha,pb)
            self.assertLess(max(abs(ga[k]-gb[k]) for k in ga),1e-9)
            future=np.array(row['drift_alpha'])
            mse=float(np.mean((qru_output(x,a,future)-qru_output(x,b,future))**2))
            self.assertAlmostEqual(mse,row['future_transverse_mse'],places=13)
            self.assertGreater(np.linalg.norm([pa[s]-pb[s] for s in pa]),.03)

    def test_joint_and_profiled_quadratics(self):
        rng=np.random.default_rng(947)
        for _ in range(10):
            theta=rng.uniform(-2,2,(3,3));x=rng.uniform(0,2*np.pi,96)
            jt=parameter_shift_prediction_jacobian(x,theta,np.array([1.,2.,4.]))
            ja=x[:,None]*jt[:,2::3];r=rng.normal(size=96);offset=rng.normal(size=3)*.08
            for damping,penalty in [(1e-4,0),(1e-3,.01),(1e-2,.05)]:
                joint,profile,_=schur_step(jt,ja,r,damping,penalty,offset)
                np.testing.assert_allclose(joint,profile,atol=2e-10,rtol=2e-9)

    def test_matched_family_and_collision_orders(self):
        checks=mathematical_checks()
        self.assertLess(checks['full_manifold_max_error_L1_to_L4'],1e-12)
        self.assertEqual(checks['transverse_drift_order'],1)
        self.assertEqual(checks['parallel_drift_order'],'infinite')

    def test_paired_stream_trials_and_bootstrap(self):
        data=read('results/predictive/streams.json');rows=data['controlled_trials']
        self.assertEqual(len(rows),32*10*3*2)
        self.assertEqual({r['seed'] for r in rows},set(range(1300,1332)))
        self.assertEqual({r['shots'] for r in rows},{720})
        self.assertLess(max(r['qru_classical_max_prediction_gap'] for r in rows),1e-12)
        for row in rows:
            self.assertTrue(70<=row['change']<=90)
            expected=0 if row['policy']=='passive' or row['scenario']=='no_phase_access' else 180
            self.assertEqual(row['phase_shots'],expected)
        current=paired_summary(rows)
        for a,b in zip(current,data['controlled_summary']):
            self.assertEqual(a.keys(),b.keys())
            for key in a:
                if isinstance(a[key],(int,float)):self.assertAlmostEqual(a[key],b[key],places=13)
                else:self.assertEqual(a[key],b[key])

    def test_stream_prefix_does_not_use_future_acquisitions(self):
        # The same seeded observations on a shorter stream must give the same
        # early prequential predictions as the archived 180-time run.
        _,short=controlled_stream(1300,'abrupt_planned',T=100)
        _,long=controlled_stream(1300,'abrupt_planned',T=180)
        lookup={(r['policy'],r['horizon'],r['t']):r for r in long}
        for r in short:
            if r['t']<65:
                self.assertAlmostEqual(r['future_mse'],lookup[(r['policy'],r['horizon'],r['t'])]['future_mse'],places=15)

    def test_two_qubit_architecture_witnesses(self):
        rows=read('data/supporting_figure_data.json')['two_qubit']
        for row in rows:
            depth=row['Depth L'];entangle=row['CNOT after layer']=='yes'
            expected=[int(k.strip()) for k in row['First orders of visible directions'].split(',')]
            for center in (.73,np.pi):
                for seed in (40,41,42):
                    ranks,visible,max_abs=two_qubit(depth,entangle,seed,center)
                    self.assertEqual(ranks,(row['Rank theta'],row['Rank theta + alpha']))
                    # Only record orders where an additional drift dimension appears.
                    orders=[order for i,(order,n) in enumerate(visible) if i==0 or n>visible[i-1][1]]
                    self.assertEqual(orders,expected)
                    self.assertLessEqual(max_abs,1+1e-12)

    def test_hardware_raw_counts_and_failed_equivalence_gate(self):
        jobs=[]
        for name,expected in [('qpu_baseline.json',.1793),('qpu_baseline_rep2.json',.1604)]:
            run=read('results/hardware/'+name);jobs.append(run['job_id'])
            self.assertEqual(run['stage'],'baseline');self.assertEqual(len(run['rows']),12)
            intervals={}
            for r in run['rows']:
                zero=r['counts']['0'];one=r['counts']['1'];tail=.05/(2*2*22)
                self.assertEqual(zero+one,8192);self.assertEqual(len(r['bitstrings']),8192)
                self.assertEqual(r['bitstrings'].count('0'),zero)
                lo=beta.ppf(tail,zero,one+1) if zero else 0
                hi=beta.ppf(1-tail,zero+1,one) if one else 1
                intervals[(r['x'],r['branch'])]=(2*lo-1,2*hi-1)
            upper=[]
            for x in sorted({r['x'] for r in run['rows']}):
                a=intervals[(x,'A')];b=intervals[(x,'B')]
                upper.append(max(abs(a[0]-b[1]),abs(a[1]-b[0])))
            self.assertAlmostEqual(max(upper),expected,places=4)
            self.assertGreater(max(upper),.12)
        self.assertEqual(len(set(jobs)),2)

if __name__=='__main__':unittest.main()
