# test_metropolisHastings.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from MCMCSampling.metropolisHastings import MetropolisHastings


import unittest
import numpy as np


def log_target_std_normal(x):
    """helper function: Log of unnormalised standard normal density."""
    return -0.5 * (x * x)


class TestMetropolisHastings(unittest.TestCase):
    def test_class_has_required_fields(self):
        """Tests if class has the required fields after construction."""
        mh = MetropolisHastings(log_target_std_normal, 1.0, 0.0, seed=1)
        self.assertTrue(hasattr(mh, "log_target"))
        self.assertTrue(hasattr(mh, "proposal_std"))
        self.assertTrue(hasattr(mh, "current_state"))

    def test_init_valid(self):
        """__init__ test"""
        mh = MetropolisHastings(log_target_std_normal, 0.5, -1.0, seed=1)
        self.assertTrue(np.isfinite(mh.current_state))

        with self.assertRaises(ValueError):
            MetropolisHastings(log_target_std_normal, 0.0, 0.0, seed=1)

        with self.assertRaises(ValueError):
            MetropolisHastings(log_target_std_normal, np.inf, 0.0, seed=1)

        with self.assertRaises(ValueError):
            MetropolisHastings(log_target_std_normal, 1.0, np.nan, seed=1)

        with self.assertRaises(TypeError):
            MetropolisHastings(np.nan, 1.0, 0.0, seed=1)

    def test_acceptance_rate_init(self):
        """Test if acceptance_rate is 0.0 initially."""
        mh = MetropolisHastings(log_target_std_normal, 1.0, 0.0, seed=1)
        self.assertEqual(mh.acceptance_rate(), 0.0)

    def test_step_matches_expected_accept_reject(self):
        """
        step test: matches expected accept/reject behaviour for a single step with known random seed.
        """
        seed = 1
        proposal_std = 0.8
        x_0 = 0.25

        mh = MetropolisHastings(log_target_std_normal, proposal_std, x_0, seed=seed)

        # Replicate the exact random draws used internally: normal then uniform.
        rng = np.random.default_rng(seed)
        y = x_0 + rng.normal(0.0, proposal_std)
        u = rng.uniform(0.0, 1.0)

        log_pi_x = log_target_std_normal(x_0)
        log_pi_y = log_target_std_normal(y)

        accept = False
        if np.isfinite(log_pi_y):
            if np.log(u) < min(0.0, log_pi_y - log_pi_x):
                accept = True

        x_1 = mh.step()
        expected = y if accept else x_0

        self.assertAlmostEqual(x_1, expected, places=12)
        self.assertAlmostEqual(mh.current_state, expected, places=12)

        # acceptance_rate should now be either 0 or 1 after exactly one proposal
        self.assertIn(mh.acceptance_rate(), (0.0, 1.0))

    def test_sample_length_and_burn_in(self):
        """
        sample test: returns correct number of samples and respects burn-in.
        """
        mh1 = MetropolisHastings(log_target_std_normal, 1.0, 0.0, seed=1)
        s_no_burn = mh1.sample(50, burn_in=0)
        self.assertEqual(len(s_no_burn), 50)
        self.assertTrue(np.all(np.isfinite(s_no_burn)))

        mh2 = MetropolisHastings(log_target_std_normal, 1.0, 0.0, seed=1)
        s_with_burn = mh2.sample(50, burn_in=10)
        self.assertEqual(len(s_with_burn), 50)
        self.assertTrue(np.all(np.isfinite(s_with_burn)))

        # The samples with burn-in should differ from those without burn-in
        self.assertFalse(np.array_equal(s_no_burn, s_with_burn))

    def test_sample_rejects_invalid_arguments(self):
        """sample test: rejects invalid n_samples and burn_in. checks for typing"""
        mh = MetropolisHastings(log_target_std_normal, 1.0, 0.0, seed=1)

        with self.assertRaises(ValueError):
            mh.sample(0, burn_in=0)

        with self.assertRaises(ValueError):
            mh.sample(10, burn_in=-1)

        with self.assertRaises(ValueError):
            mh.sample("10", burn_in=0)  

        with self.assertRaises(ValueError):
            mh.sample(10, burn_in="0")  


if __name__ == "__main__":
    unittest.main()