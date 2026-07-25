"""Question 2 -- Gaussian process regression.

Run interactively cell-by-cell in VS Code, or headless with
``python code/q2_gp_regression.py``.
"""

# %%
from pathlib import Path

import numpy as np

from _report import savefig, write_results

DATA = Path(__file__).resolve().parent.parent

# Kernel and noise hyperparameters fixed by the question.
T_MAX, ALPHA, RHO, SIGMA2 = 10.0, 1.0, 1.5, 0.05

# %% [markdown]
# ## (c) Posterior mean, credible bands, and P(f(5) > 0 | t, y)

# %%
# <<q2c
ty = np.loadtxt(DATA / "ty.txt", delimiter=",", skiprows=1)
t, y = ty[:, 0], ty[:, 1]


def sq_exp(s, u, alpha=ALPHA, rho=RHO):
    """Squared exponential kernel evaluated on the outer grid of s and u."""
    d = s[:, None] - u[None, :]
    return alpha**2 * np.exp(-(d**2) / (2 * rho**2))


def gp_posterior(t_star, t, y, alpha=ALPHA, rho=RHO, sigma2=SIGMA2):
    """Posterior mean and variance of f(t_star) given the data.

    TODO: implement the closed form derived in part (a),
        mean = k(t*, t) [K(t, t) + sigma^2 I]^{-1} y,
        var  = k(t*, t*) - k(t*, t) [K(t, t) + sigma^2 I]^{-1} k(t, t*).
    Solve with a Cholesky factorisation (scipy.linalg.cho_factor / cho_solve) rather
    than np.linalg.inv -- it is both faster and numerically stabler.
    Return the full predictive covariance if you want joint (rather than pointwise) bands.
    """
    raise NotImplementedError


grid = np.linspace(0.0, T_MAX, 400)
mean, var = gp_posterior(grid, t, y)
sd = np.sqrt(var)

# Pointwise 95% credible band: the posterior of f(t) is Normal at each t.
lower, upper = mean - 1.96 * sd, mean + 1.96 * sd

# TODO: P(f(5) > 0 | t, y) -- evaluate the posterior at t = 5 and use the Normal cdf.
prob_positive = np.nan
# >>q2c

print(f"P(f(5) > 0 | t, y) = {prob_positive:.4f}")

# %% [markdown]
# ## (d) Bayes action under asymmetric linear loss

# %%
# <<q2d
C1, C2 = 3.0, 1.0


def bayes_action(mean_5, sd_5, c1=C1, c2=C2):
    """Bayes action minimising E[l(d, f(5)) | t, y] for the asymmetric linear loss.

    TODO: this loss is the (scaled) check loss, so the Bayes action is a posterior
    quantile of f(5) -- identify which quantile in terms of c1 and c2, then evaluate it
    for the Normal posterior with scipy.stats.norm.ppf.
    """
    raise NotImplementedError


d_star = bayes_action(mean_5=np.nan, sd_5=np.nan)
# >>q2d

print(f"Bayes action d* = {d_star:.4f}")

# %% [markdown]
# ## Figure and exported results

# %%
# TODO: plot the posterior mean over [0, T] with the shaded 95% band and the observed
# points overlaid; mark t = 5 if you want to illustrate parts (c)/(d).
#
# fig, ax = plt.subplots(figsize=(8, 3.6))
# ax.fill_between(grid, lower, upper, alpha=0.25, label="95% credible band")
# ...
# savefig(fig, "q2_gp")

write_results("q2", {
    "q2.prob.positive": float(prob_positive),
    "q2.bayes.action": float(d_star),
})
