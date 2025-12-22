# metropolisHastings.py

import numpy as np


class MetropolisHastings:
    """
    Random-walk Metropolis–Hastings sampler on R.
    """

    def __init__(self, log_target, proposal_std, initial_state, seed = None):
        """
        Initialise the Metropolis–Hastings sampler.

        Parameters
        ----------
        log_target
            Function returning log(unnormalised π(x)).
        proposal_std : float
            Standard deviation of the Gaussian proposal.
        initial_state : float
            Initial state of the Markov chain.
        seed : int or None
            Seed for the random number generator.
        """
        if not np.isfinite(proposal_std) or proposal_std <= 0:
            raise ValueError("proposal_std must be a positive finite number.")
        if not np.isfinite(initial_state):
            raise ValueError("initial_state must be finite.")
        if not np.isfinite(log_target(initial_state)):
            raise ValueError("log_target(initial_state) must be finite.")

        self.log_target = log_target
        self.proposal_std = float(proposal_std)
        self.current_state = float(initial_state)

        self._rng = np.random.default_rng(seed)
        self._n_proposals = 0
        self._n_accepted = 0



    def acceptance_rate(self):
        """
        Return the acceptance rate so far.
        """
        if self._n_proposals == 0:
            return 0.0
        return self._n_accepted / self._n_proposals

    def step(self):
        """
        Perform a single Metropolis–Hastings step, updates the current state or keeps it unchanged.
        """
        x = self.current_state
        log_pi_x = self.log_target(x)
        if not np.isfinite(log_pi_x):
            raise ValueError("log_target(current_state) must be finite.")

        # Propose a new state - random walk proposal
        y = x + self._rng.normal(0.0, self.proposal_std)
        log_pi_y = self.log_target(y)

        accept = False
        if np.isfinite(log_pi_y):
            #MH acceptance probability computation in log scale to avoid numerical issues/overflow
            log_alpha = min(0.0, log_pi_y - log_pi_x)
            if np.log(self._rng.uniform()) < log_alpha:
                accept = True

        self._n_proposals += 1
        if accept:
            self.current_state = y
            self._n_accepted += 1

        return self.current_state

    def sample(self, n_samples, burn_in=0):
        """
        Takes as inputs the number of wanted samples `n_samples` and the length of the `burn_in` period; runs the algorithm for `n_samples` + `burn_in` steps and returns `n_samples` samples after the `burn_in` phase
        """
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")
        if not isinstance(burn_in, int) or burn_in < 0:
            raise ValueError("burn_in must be a non-negative integer.")

        samples = np.empty(n_samples)
        total_steps = burn_in + n_samples

        k = 0
        for t in range(total_steps):
            state = self.step()
            if t >= burn_in:
                samples[k] = state
                k += 1

        return samples