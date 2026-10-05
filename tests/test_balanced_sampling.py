"""Check class balance, reproducibility, and equivalence of expected objectives."""
import unittest
import numpy as np
from balanced_sampling import balanced_epoch_indices, objective_scale


class BalancedSamplingTests(unittest.TestCase):
    def test_updates_are_balanced_and_indices_stay_in_fit(self):
        y = np.array([1]*73 + [0]*3501)
        indices = balanced_epoch_indices(y, 32, 17)
        self.assertEqual(len(indices), 3584)
        self.assertTrue(np.all(y[indices].reshape(-1, 32).sum(1) == 16))
        self.assertTrue(np.all((indices >= 0) & (indices < len(y))))
        np.testing.assert_array_equal(indices, balanced_epoch_indices(y, 32, 17))
        self.assertFalse(np.array_equal(indices, balanced_epoch_indices(y, 32, 1017)))

    def test_objective_and_gradient_match_natural_weighted_expectation(self):
        y = np.array([0., 0., 0., 1.])
        z = np.array([-2., .4, 3., -.7])
        loss = np.logaddexp(0, z)-y*z
        weights = np.where(y == 1, 3., 1.)
        natural = (weights*loss).mean()
        balanced = objective_scale(y) * (loss[y == 0].mean()+loss[y == 1].mean())/2
        self.assertAlmostEqual(natural, balanced, places=14)
        natural_gradient = weights * (1/(1+np.exp(-z))-y)/len(y)
        balanced_probability = np.where(y == 1, .5, .5/3)
        balanced_gradient = objective_scale(y)*balanced_probability*(1/(1+np.exp(-z))-y)
        np.testing.assert_allclose(natural_gradient, balanced_gradient, rtol=1e-14)

    def test_rejects_invalid_cohorts_and_groups(self):
        for labels, size in [([0,0],32), ([0,1],3), ([0,1],0), ([0,2],32)]:
            with self.assertRaises(ValueError):
                balanced_epoch_indices(labels, size, 17)


if __name__ == '__main__':
    unittest.main()
