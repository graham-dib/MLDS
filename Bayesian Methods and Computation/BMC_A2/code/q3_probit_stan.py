"""Question 3 -- Bayesian probit regression via Stan (PyStan 3).

Run interactively cell-by-cell in VS Code, or headless with
``python code/q3_probit_stan.py``.

PyStan 3 is imported as ``stan`` (not ``pystan``), takes the model as a *string*
rather than a file path, and compiles on the first build -- expect ~30-60 s the first
time, then it is cached.
"""

# %%
from pathlib import Path

import numpy as np
import stan

from _report import savefig, write_results

# PyStan 3 drives its compiler server over asyncio. A Jupyter / VS Code interactive
# kernel already owns a running event loop, so it needs to be made re-entrant; this is
# a no-op when the script is run headless.
try:
    import nest_asyncio

    nest_asyncio.apply()
except ModuleNotFoundError:
    pass

HERE = Path(__file__).resolve().parent
DATA = HERE.parent

# Prior variances and the future covariate value fixed by the question.
S2_ALPHA, S2_BETA, X_STAR = 25.0, 25.0, 0.75

# %% [markdown]
# ## (d) Fit the Stan model

# %%
# <<q3d
xz = np.loadtxt(DATA / "xz.txt", delimiter=",", skiprows=1)
x, z = xz[:, 0], xz[:, 1].astype(int)

# Values are serialised to JSON for the Stan server, so pass plain Python types.
stan_data = {
    "n": len(z),
    "x": x.tolist(),
    "z": z.tolist(),
    "s2_alpha": S2_ALPHA,
    "s2_beta": S2_BETA,
    "x_star": X_STAR,
}

stan_code = (HERE / "probit_regression.stan").read_text()
posterior = stan.build(stan_code, data=stan_data, random_seed=1234)
fit = posterior.sample(num_chains=4, num_warmup=1000, num_samples=2500)

draws = fit.to_frame()
alpha, beta, q_star = draws["alpha"], draws["beta"], draws["q_star"]
# >>q3d

# %% [markdown]
# ### Convergence checks
#
# Quote nothing before these look healthy. PyStan 3 has no built-in `summary()`; the
# sampler diagnostics ride along in the draws frame, and ArviZ (`pip install arviz`)
# gives R-hat and effective sample size if you want them reported properly.

# %%
n_divergent = int(draws["divergent__"].sum())
print(f"divergent transitions: {n_divergent} / {len(draws)}")
print(draws[["alpha", "beta", "q_star"]].describe().T[["mean", "std", "min", "max"]])

try:
    import arviz as az

    idata = az.from_pystan(posterior=fit)
    print(az.summary(idata, var_names=["alpha", "beta", "q_star"]))
except ModuleNotFoundError:
    print("arviz not installed -- skipping R-hat / ESS "
          "(pip install arviz to report them)")

# %% [markdown]
# ## (e) Posterior estimates

# %%
# <<q3e
# TODO: estimate the three quantities from the posterior sample.
#   i.   E[beta | x, z]                      -- posterior mean of beta
#   ii.  P(beta > 0 | x, z)                  -- proportion of draws with beta > 0
#   iii. P(z* = 1 | x* = 0.75, x, z)         -- posterior mean of q_star, since
#        P(z* = 1 | data) = E[Phi(alpha + beta x*) | data] by the tower property
e_beta = np.nan
p_beta_positive = np.nan
p_zstar = np.nan
# >>q3e

print(f"E[beta | x, z]        = {e_beta:.4f}")
print(f"P(beta > 0 | x, z)    = {p_beta_positive:.5f}")
print(f"P(z* = 1 | x* = 0.75) = {p_zstar:.4f}")

# %% [markdown]
# ## Figures and exported results

# %%
# TODO (optional but worth it): a diagnostics figure (trace + density for alpha, beta)
# supports part (d), and a plot of the fitted probit curve against a refitted logistic
# curve makes the comparison in part (f) concrete rather than assertive. To refit under
# the logit link, swap the likelihood for bernoulli_logit(alpha + beta * x) in a second
# .stan file and build it the same way.
#
# savefig(fig, "q3_diagnostics")
# savefig(fig, "q3_links")

write_results("q3", {
    "q3.e.beta": float(e_beta),
    "q3.p.beta.positive": f"{p_beta_positive:.5f}",
    "q3.p.zstar": float(p_zstar),
    "q3.divergences": n_divergent,
})
