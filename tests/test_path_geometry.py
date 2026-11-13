"""Independent checks of two-layer geometry, minimax risk and internal noise."""
import unittest
import numpy as np
from qru.circuit import qru_output
from qru.path_geometry import depth_two_amplitudes, sinusoidal_fiber, fiber_risk
from qru.noise import density_output, dephasing_twins


class PathGeometryChecks(unittest.TestCase):
    def test_explicit_amplitudes_reconstruct_independent_density_simulation(self):
        rng = np.random.default_rng(7301)
        for _ in range(40):
            theta = rng.uniform(-np.pi, np.pi, (2, 3))
            alpha = rng.uniform(.1, 3, 2); x = rng.uniform(-4, 7, 31)
            a = depth_two_amplitudes(theta)
            prediction = sum(v*np.exp(1j*np.dot(s, alpha)*x) for s, v in a.items())
            np.testing.assert_allclose(prediction, density_output(x, theta, alpha), atol=1e-13, rtol=0)

    def test_sinusoidal_fiber_boundaries_and_midpoint_risk(self):
        x = np.linspace(0, 2*np.pi, 97)
        weights = np.linspace(.1, 2, len(x)); weights /= weights.sum()
        for radius in [.01, .4, 1.]:
            for phase in [-np.pi, .7, np.pi]:
                theta = sinusoidal_fiber(radius, phase)
                for t in theta.values():
                    np.testing.assert_allclose(qru_output(x, t, [1, 2]), radius*np.cos(x+phase), atol=2e-14, rtol=0)
                a, b = [qru_output(x, theta[h], [1.07, 2.2]) for h in ['A', 'B']]
                midpoint = (a+b)/2
                floor = fiber_risk(radius, phase, [1.07, 2.2], x, weights)
                self.assertAlmostEqual(np.dot(weights, (midpoint-a)**2), floor, places=14)
                self.assertAlmostEqual(np.dot(weights, (midpoint-b)**2), floor, places=14)

    def test_density_identity_limit_and_dephasing_location(self):
        x = np.linspace(-2, 7, 65)
        rng = np.random.default_rng(9512)
        for depth in [1, 2, 3]:
            t = rng.uniform(-2, 2, (depth, 3)); alpha = rng.uniform(.1, 3, depth)
            np.testing.assert_allclose(density_output(x, t, alpha), qru_output(x, t, alpha), atol=1e-14, rtol=0)
            np.testing.assert_allclose(density_output(x, t, alpha, .3, location='final'), qru_output(x, t, alpha), atol=1e-14, rtol=0)
        ideal = dephasing_twins(1.)
        gamma = .4
        a = density_output(x, ideal['A'], [1, 2], gamma)
        b = density_output(x, ideal['B'], [1, 2], gamma)
        np.testing.assert_allclose(a-b, (1-gamma)*np.cos(x)*np.cos(2*x), atol=1e-14, rtol=0)
        np.testing.assert_allclose(density_output(x, ideal['A'], [1, 2], gamma, 'depolarizing'),
                                   density_output(x, ideal['B'], [1, 2], gamma, 'depolarizing'), atol=1e-14, rtol=0)

    def test_calibrated_twins_future_and_shot_information(self):
        x = np.linspace(-1, 7, 71)
        for gamma in [0., .1, .7, 1.]:
            theta = dephasing_twins(gamma)
            for alpha in [[1, 2], [1, 2.05], [.73, 1.81]]:
                for branch, frequency in [('A', alpha[1]-alpha[0]), ('B', alpha[0])]:
                    np.testing.assert_allclose(density_output(x, theta[branch], alpha, gamma),
                                               gamma*np.cos(frequency*x), atol=1e-14, rtol=0)
            means = []
            for branch in ['A', 'B']:
                shifted = theta[branch].copy(); shifted[1, 2] += np.pi/2
                means.append(density_output([np.pi/4], shifted, [1, 2], gamma)[0])
            np.testing.assert_allclose(means, [-gamma/np.sqrt(2), gamma/np.sqrt(2)], atol=1e-14, rtol=0)
            p, q = (1+np.array(means))/2
            kl = p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))
            c = gamma/np.sqrt(2)
            self.assertAlmostEqual(kl, c*np.log((1+c)/(1-c)), places=14)

    def test_domain_conditions(self):
        for radius in [0., -1., 1.01, np.nan]:
            with self.assertRaises(ValueError): sinusoidal_fiber(radius, 0.)
        for gamma in [-.1, 1.1, np.nan]:
            with self.assertRaises(ValueError): dephasing_twins(gamma)
        with self.assertRaises(ValueError): depth_two_amplitudes(np.zeros((3, 3)))
        with self.assertRaises(ValueError): fiber_risk(.5, 0., [1, 2], [])


if __name__ == '__main__':
    unittest.main()
