"""Constructive geometry, layer access and independently checked noise tests."""
import sys,json,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qru.path_geometry import depth_two_amplitudes,attainable_conditions,reconstruct_attainable
from qru.circuit import qru_output
from qru.control_design import optimal_single_layer,localized_output,localized_amplitudes
from qru.tomography import reconstruct_from_paths
from qru.depth_family import branch_theta,branch_tag,round_budget,interrogation,uniform_branch_theta,interrogate_prefixes
from experiments.branch_selection import output,output_jacobian,unpack
from hardware.metric_noise import reconstructed_gates,coherent_means,sphere_axes

class ConstructiveResults(unittest.TestCase):
    def test_complete_set_canonical_controls_and_faces(self):
        rng=np.random.default_rng(4283)
        for _ in range(60):
            theta=rng.uniform(-np.pi,np.pi,(2,3));a=depth_two_amplitudes(theta)
            triple=[a[k] for k in [(1,0),(1,1),(1,-1)]]
            canonical=reconstruct_attainable(*triple)
            phi=rng.normal(size=2);theta[:,2]+=phi;canonical[:,2]+=phi
            alpha=rng.uniform(.2,4,2);x=rng.uniform(-3,7,11)
            np.testing.assert_allclose(qru_output(x,theta,alpha),qru_output(x,canonical,alpha),atol=2e-11,rtol=0)
        p,q=.1,.2;w=np.sqrt(p*q/(p+q)**2-4*p*q)
        self.assertTrue(attainable_conditions(w,-p,q)['attainable'])
        self.assertFalse(attainable_conditions(1j*w,-p,q)['attainable'])
        self.assertFalse(attainable_conditions(0.,.6,0.)['attainable'])
        self.assertFalse(attainable_conditions(.51,0.,0.)['attainable'])
        for amps in [(0.,0.,0.),(.5j,0.,0.),(0.,.5j,0.),(0.,0.,.5j),(0.,.25,.25j)]:
            th=reconstruct_attainable(*amps);aa=depth_two_amplitudes(th)
            np.testing.assert_allclose([aa[k] for k in [(1,0),(1,1),(1,-1)]],amps,atol=1e-13)
    def test_optimal_query_and_channel_path_identities(self):
        rng=np.random.default_rng(558)
        for L in [2,3,4]:
            a=rng.normal(size=(L,3));b=rng.normal(size=(L,3));alpha=2.**np.arange(L);x=np.linspace(0,2,11)
            queries=optimal_single_layer(a,b,alpha,x)
            for row in queries:
                l=row['layer']-1;aa=a.copy();bb=b.copy();aa[l,2]+=row['phase'];bb[l,2]+=row['phase']
                d=abs(qru_output([row['x']],aa,alpha)[0]-qru_output([row['x']],bb,alpha)[0])
                self.assertAlmostEqual(d,row['absolute_gap'],places=13)
                for phi in np.linspace(0,2*np.pi,31):
                    aa=a.copy();bb=b.copy();aa[l,2]+=phi;bb[l,2]+=phi
                    self.assertGreaterEqual(d+1e-12,np.max(abs(qru_output(x,aa,alpha)-qru_output(x,bb,alpha))))
                for gamma in [.15,.6]:
                    a1=localized_amplitudes(a,l,gamma)
                    np.testing.assert_allclose(reconstruct_from_paths(x,alpha,a1),localized_output(x,a,alpha,l,gamma),atol=2e-14)
            np.testing.assert_allclose(localized_output(x,a,alpha,L-1,.2),qru_output(x,a,alpha),atol=2e-14)
    def test_depth_family_at_random_controls(self):
        rng=np.random.default_rng(818)
        for L in [2,3,5,8]:
            for k in range(1,L+1):
                r=rng.uniform(.2,1.);phase=rng.normal();theta=branch_theta(L,k,r,phase)
                alpha=rng.uniform(.2,3,L);phi=rng.normal(size=L);theta[:,2]+=phi;x=rng.uniform(-2,5,19)
                s=branch_tag(L,k)
                np.testing.assert_allclose(qru_output(x,theta,alpha),r*np.cos(np.dot(s,alpha)*x+phase+np.dot(s,phi)),atol=3e-14)
    def test_uniform_controls_and_optimal_pure_limit(self):
        rng=np.random.default_rng(119)
        for L in [3,7,15]:
            K=(L+1)//2
            for k in range(1,K+1):
                theta=uniform_branch_theta(L,k,.7,.3);alpha=np.ones(L);x=np.linspace(0,2*np.pi,37)
                np.testing.assert_allclose(qru_output(x,theta,alpha),.7*np.cos(x+.3),atol=3e-14)
                tag=np.zeros(L);tag[:2*k-1]=[(-1)**j for j in range(2*k-1)]
                phi=rng.normal(size=L);theta[:,2]+=phi;future=rng.uniform(.4,2.,L)
                np.testing.assert_allclose(qru_output(x,theta,future),.7*np.cos(np.dot(tag,future)*x+.3+np.dot(tag,phi)),atol=3e-14)
        for K in [2,3,4,7,8]:
            for k in range(1,K+1):
                result=interrogation(K,k,1.,.3,rng)
                self.assertEqual(result['shots'],int(np.ceil(np.log2(K))))
                self.assertEqual(result['selected'],k)
                self.assertEqual(np.argmax(result['posterior'])+1,k)

    def test_depth_budgets_and_recorded_posteriors(self):
        d=json.loads((ROOT/'results/predictive/depth_family.json').read_text())
        for study in d['studies']:
            L=study['depth'];R=int(np.ceil(np.log2(L)));n,bound=round_budget(.7,R,.05)
            self.assertLessEqual(R*bound,.05)
            errors=0
            for row in study['trials']:
                self.assertEqual(row['shots'],R*n);self.assertEqual(len(row['queries']),R)
                logp=np.zeros(L)
                for query in row['queries']:
                    signs=np.array([branch_tag(L,k)[query['layer']-1] for k in range(1,L+1)])
                    means=.7*np.cos(signs*query['phase']);p=(1+means)/2;k=query['positive']
                    logp+=k*np.log(p)+(n-k)*np.log1p(-p);logp-=logp.max()
                pp=np.exp(logp);pp/=pp.sum()
                np.testing.assert_allclose(pp,row['posterior'],atol=1e-14)
                errors+=row['selected']!=row['branch']
            self.assertEqual(errors,study['hard_errors'])
    def test_training_closed_gradient_and_raw_records(self):
        rng=np.random.default_rng(517);theta=rng.normal(size=(7,5));x=rng.normal(size=(7,21));f,j=output_jacobian(theta,x)
        np.testing.assert_allclose(f,output(theta,x),atol=1e-15)
        for k in range(5):
            a=theta.copy();b=theta.copy();a[:,k]+=np.pi/2;b[:,k]-=np.pi/2
            np.testing.assert_allclose(j[:,k],.5*(output(a,x)-output(b,x)),atol=2e-15)
        data=json.loads((ROOT/'results/predictive/branch_selection.json').read_text());x=2*np.pi*np.arange(32)/32;y=.7*np.cos(x+.3)
        for row in data['trials'][::67]:
            self.assertAlmostEqual(np.mean((qru_output(x,row['raw_theta'],[1,2])-y)**2),row['pre_completion_mse'],places=13)
    def test_metric_noise_ideal_template_and_infidelity(self):
        data=json.loads((ROOT/'results/hardware/metric_noise.json').read_text());r=data['gate_infidelity'];angle=data['coherent_rotation_angle']
        self.assertAlmostEqual((2/3)*np.sin(angle/2)**2,r,places=15)
        axes=sphere_axes(41);np.testing.assert_allclose(np.linalg.norm(axes,axis=1),1.,atol=1e-14)
        raw=json.loads((ROOT/'results/hardware/qpu_baseline.json').read_text())
        for row in raw['rows']:
            gates=reconstructed_gates(row)
            self.assertEqual(sum(g['name']=='sx' for g in gates),8);self.assertEqual(sum(g['name']=='rz' for g in gates),14)
            psi=np.array([1.,0.],complex)
            sx=np.array([[1.,-1j],[-1j,1.]])/np.sqrt(2)
            for gate in gates:
                u=sx if gate['name']=='sx' else np.diag(np.exp(np.array([-1j,1j])*gate['angle']/2))
                psi=u@psi
            self.assertAlmostEqual(abs(psi[0])**2-abs(psi[1])**2,row['ideal_mean'],places=13)
            np.testing.assert_allclose(coherent_means(row,axes,0.,0.),row['ideal_mean'],atol=2e-14)
if __name__=='__main__':unittest.main()
