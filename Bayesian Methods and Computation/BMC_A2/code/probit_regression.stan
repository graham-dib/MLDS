// probit_regression.stan
//
// Skeleton for Question 3(c). Fill in each block; the report pulls this file in
// verbatim via \lstinputlisting, so what you write here is what gets marked.

data {
    // TODO: declare n, the covariates x, the binary responses z, the prior
    // variances s2_alpha and s2_beta, and the future covariate value x_star.
    // Constrain what you can -- e.g. int<lower=0> n, and z as an
    // array[n] int<lower=0, upper=1>, so Stan validates the data for you.
}

parameters {
    // TODO: the intercept alpha and the slope beta (both unconstrained reals).
}

model {
    // TODO: the independent Normal priors, and the probit likelihood.
    // Note Stan's normal() takes a standard deviation, not a variance.
    // The inverse link Phi() is available directly; if the sampler struggles with
    // saturation, std_normal_lcdf / std_normal_lccdf are the stable alternative.
}

generated quantities {
    // TODO: q_star = Phi(alpha + beta * x_star), the posterior predictive success
    // probability at the supplied covariate value.
}
