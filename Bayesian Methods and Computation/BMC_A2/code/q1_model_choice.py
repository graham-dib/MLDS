"""Question 1 -- Bayesian model choice and model criticism.

Run interactively cell-by-cell in VS Code (the ``# %%`` markers make this a notebook),
or headless with ``python code/q1_model_choice.py``.

The ``# <<tag`` / ``# >>tag`` comments delimit the segments that the report pulls in
verbatim via \\lstinputlisting -- keep the code you want shown between them.
"""

# %%
from pathlib import Path

import numpy as np
from scipy.special import gammaln

from _report import savefig, write_results

DATA = Path(__file__).resolve().parent.parent

# Conjugate prior hyperparameters fixed by the question.
LAMBDA, A, B = 1.0, 2.0, 2.0

# %% [markdown]
# ## (c) Log marginal likelihoods, posterior model probabilities, preferred model

# %%
# <<q1c
xy = np.loadtxt(DATA / "xy.txt", delimiter=",", skiprows=1)
x, y = xy[:, 0], xy[:, 1]
n = len(y)


def design(x, degree):
    """Vandermonde design matrix [1, x, ..., x^degree] for M_{degree}."""
    return np.vander(x, degree + 1, increasing=True)


def log_marginal_likelihood(X, y, lam=LAMBDA, a=A, b=B):
    """Log p(y | M) under the conjugate Normal--Gamma prior.

    TODO: implement the marginal likelihood formula from the lecture notes.
    The posterior quantities you need are
        V_n = (lam I_p + X^T X)^{-1},  m_n = V_n X^T y,
        a_n = a + n/2,                 b_n = b + (y^T y - y^T X m_n)/2,
    and the normalising constants involve gammaln and a log-determinant.
    Prefer np.linalg.slogdet / cho_factor over forming inverses explicitly.
    """
    raise NotImplementedError


log_ml = np.array([log_marginal_likelihood(design(x, d), y) for d in (1, 2, 3)])

# Normalise in the log domain to avoid underflow: under equal priors the posterior model
# probabilities are proportional to the marginal likelihoods.
post = np.exp(log_ml - log_ml.max())
post /= post.sum()

preferred = int(np.argmax(post)) + 1
# >>q1c

for i, (lm, p) in enumerate(zip(log_ml, post), start=1):
    print(f"M{i}: log p(y|M) = {lm:9.4f}   P(M|y) = {p:.4f}")
print(f"preferred model: M{preferred}")

# %% [markdown]
# ## (d) Fit the preferred model and run the posterior predictive check

# %%
# <<q1d
rng = np.random.default_rng(0)
N_SIM = 20_000


def posterior_sample(X, y, size, rng, lam=LAMBDA, a=A, b=B):
    """Draw (beta, sigma2) from the exact conjugate posterior.

    TODO: sample sigma^{-2} ~ Gamma(a_n, b_n) then beta | sigma^2 ~ Normal(m_n, sigma^2 V_n).
    Note np.random.Generator.gamma is parameterised by scale = 1 / rate.
    """
    raise NotImplementedError


def posterior_predictive_p(X, y, size, rng):
    """Monte Carlo estimate of the posterior predictive p-value for T(y, beta).

    TODO: for each posterior draw, simulate y_rep ~ Normal(X beta, sigma^2 I) and compare
    T(y_rep, beta) with T(y, beta); the p-value is the proportion of draws where the
    replicate is at least as discrepant. Also report the Monte Carlo standard error.
    """
    raise NotImplementedError


X_pref = design(x, preferred)
p_value, mc_se = posterior_predictive_p(X_pref, y, N_SIM, rng)
# >>q1d

print(f"posterior predictive p-value = {p_value:.4f}  (MC s.e. {mc_se:.4f})")

# %% [markdown]
# ## Figure and exported results

# %%
# TODO: build the two-panel figure -- left, the posterior fit under the preferred model;
# right, the posterior predictive distribution of T(y_rep, beta) against the realised
# T(y, beta). Then save it and export every number the report quotes.
#
# fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
# ...
# savefig(fig, "q1_ppc")

write_results("q1", {
    "q1.logml.M1": float(log_ml[0]),
    "q1.logml.M2": float(log_ml[1]),
    "q1.logml.M3": float(log_ml[2]),
    "q1.post.M1": float(post[0]),
    "q1.post.M2": float(post[1]),
    "q1.post.M3": float(post[2]),
    # Wrapped in math mode: values are inserted into the report verbatim.
    "q1.preferred": f"$M_{preferred}$",
    "q1.ppp": float(p_value),
    "q1.ppp.se": float(mc_se),
})
