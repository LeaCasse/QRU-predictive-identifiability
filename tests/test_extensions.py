"""Independent checks for controlled layers, polynomial fibers and noise."""
from pathlib import Path
import sys,unittest,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qru.circuit import qru_output
from qru.phase_design import phase_groups,pair_design,fixed_input_tomography,collision_partition
from qru.tomography import analytic_path_amplitudes
from experiments.branch_selection import output,unpack
from hardware.calibrated_replay import mean,gate_channel

class Extensions(unittest.TestCase):
    def test_layer_grouping_against_statevectors(self):
        rng=np.random.default_rng(211)
        for L in (2,3,4):
            alpha=[2**k for k in range(L)];a=rng.normal(size=(L,3));b=rng.normal(size=(L,3))
            aa=analytic_path_amplitudes(a);bb=analytic_path_amplitudes(b);v={s:aa[s]-bb[s] for s in aa}
            for layers in [(0,),(L-1,),tuple(range(L))]:
                g=phase_groups(v,alpha,layers)
                for _ in range(8):
                    phi=rng.normal(size=len(layers));x=rng.normal();at=a.copy();bt=b.copy()
                    at[list(layers),2]+=phi;bt[list(layers),2]+=phi
                    expected=sum(z*np.exp(1j*(float(w)*x+np.dot(k,phi))) for (w,k),z in g.items())
                    self.assertLess(abs(expected-(qru_output([x],at,alpha)[0]-qru_output([x],bt,alpha)[0])),1e-13)
    def test_joint_control_and_exact_partitions(self):
        a=np.zeros((2,3));b=a.copy();b[0,1]=np.pi/2
        self.assertEqual(pair_design(a,b,[0,0])['minimum_layers'],2)
        self.assertEqual(len(collision_partition([(1,0),(0,1)],['1','1.00000000000001'])),2)
        for t in (a,b):
            aa=analytic_path_amplitudes(t);bb=fixed_input_tomography(t,[0,0],.73)
            self.assertLess(max(abs(aa[s]-bb[s]) for s in aa),1e-14)
    def test_training_forward_and_frozen_results(self):
        rng=np.random.default_rng(314);p=rng.normal(size=(9,5));x=rng.normal(size=(9,13))
        for i in range(9):np.testing.assert_allclose(output(p,x)[i],qru_output(x[i],unpack(p[i]),[1,2]),atol=2e-15)
        d=json.loads((ROOT/'results/predictive/branch_selection.json').read_text());self.assertEqual(len(d['trials']),1280)
        base={r['seed']:r for r in d['trials'] if r['mode']=='full'}
        for s in d['summary']:
            rr=[r for r in d['trials'] if r['mode']==s['mode']]
            self.assertEqual(s['A'],sum(r['branch']=='A' for r in rr))
            self.assertEqual(s['switch_A_to_B'],sum(base[r['seed']]['branch']=='A' and r['branch']=='B' for r in rr))
            self.assertTrue(all(r['final_mse']<3e-25 for r in rr))
            self.assertAlmostEqual(.5*(s['mean_future_mse_A']+s['mean_future_mse_B']),.030667416,places=7)
    def test_calibrated_channel_limits(self):
        cal=dict(t1_seconds=2.,t2_seconds=3.,readout_p1_given_0=0.,readout_p0_given_1=0.,
                 gates={k:dict(duration_seconds=0.,average_error=0.) for k in ['sx','x','rz']})
        self.assertAlmostEqual(mean([dict(name='x')],cal),-1.)
        self.assertAlmostEqual(mean([dict(name='sx')],cal),0.)
        t=.2;a=np.exp(-t/3);b=np.exp(-t/2)
        cal['gates']['rz']=dict(duration_seconds=t,average_error=(3-2*a-b)/6)
        np.testing.assert_allclose(gate_channel([0,0,-1],'rz',0,cal),[0,0,1-2*b],atol=1e-15)
        cal['readout_p1_given_0']=.03;cal['readout_p0_given_1']=.08
        self.assertAlmostEqual(mean([dict(name='x')],cal),-.84)
    def test_calibrated_replay_provenance_and_ideal_identity(self):
        from hardware.calibrated_replay import replay
        cal=dict(backend='synthetic',physical_qubit=0,calibration_timestamp='fixture',source='unit-test fixture',
                 t1_seconds=2.,t2_seconds=3.,readout_p1_given_0=.03,readout_p0_given_1=.08,
                 gates={'x':dict(duration_seconds=0.,average_error=0.)})
        record=dict(backend='synthetic',physical_qubit=0,calibration_last_update='fixture',
                    isa_ops={'x':1,'measure':1},rows=[dict(pub_index=0,x=0.,branch='A',ideal_mean=-1.,counts={'0':8,'1':92})])
        circuits=dict(backend='synthetic',physical_qubit=0,calibration_timestamp='fixture',source='unit-test fixture',
                      rows=[dict(pub_index=0,gates=[dict(name='x')])])
        result=replay(record,cal,circuits)
        self.assertAlmostEqual(result['rows'][0]['predicted_mean'],-.84)
        record['rows'][0]['ideal_mean']=1.
        with self.assertRaises(ValueError):replay(record,cal,circuits)
        record['rows'][0]['ideal_mean']=-1.;cal['calibration_timestamp']='wrong'
        with self.assertRaises(ValueError):replay(record,cal,circuits)

    def test_complete_angle_classes(self):
        from qru.path_geometry import depth_two_amplitudes
        rng=np.random.default_rng(813)
        for r in (.2,.7,1.):
            phase=.31
            for _ in range(10):
                ct=rng.uniform(np.sqrt(1-r*r),1.) if r<1 else rng.uniform(.1,1.)
                t=np.arccos(ct);z=np.arccos(np.sqrt(1-r*r)/ct)
                u=np.pi/2
                b=phase-np.angle(-np.sin(u)*(np.sin(t)*np.cos(z)+1j*np.sin(z)))
                th=np.array([[t,z,b],[u,rng.normal(),rng.normal()]])
                a=depth_two_amplitudes(th)
                self.assertLess(abs(a[(1,0)]-.5*r*np.exp(1j*phase)),1e-14)
                self.assertLess(abs(a[(1,1)])+abs(a[(1,-1)]),1e-14)
                c=rng.normal();th=np.array([[np.pi,0,c-phase-np.pi],[np.arccos(r),rng.normal(),c]])
                a=depth_two_amplitudes(th)
                self.assertLess(abs(a[(1,-1)]-.5*r*np.exp(-1j*phase)),1e-14)
                self.assertLess(abs(a[(1,1)])+abs(a[(1,0)]),1e-14)

    def test_polynomial_lift(self):
        try: import sympy as sp
        except ImportError: self.skipTest('Install requirements-symbolic.txt for symbolic checks.')
        from qru.algebraic import lifted_amplitudes
        rng=np.random.default_rng(991)
        for L in (2,3):
            variables,circles,amp=lifted_amplitudes(L)
            evaluator=sp.lambdify(variables,list(amp.values()),'numpy')
            for _ in range(3):
                theta=rng.normal(size=(L,3));vals=[]
                for t,z,b in theta:vals.extend([np.cos(t),np.sin(t),np.cos(z),np.sin(z),np.cos(b),np.sin(b)])
                expected=analytic_path_amplitudes(theta)
                np.testing.assert_allclose(evaluator(*vals),[expected[s] for s in amp],atol=2e-15)
if __name__=='__main__':unittest.main()
