"""rung1_conjugate.py — RUNG 1: Poisson-Gamma conjugate rate model, NUTS vs analytic.

Generative model (all quantities synthetic, seeds fixed)
--------------------------------------------------------
    lambda        ~ Gamma(a0, b0)              # shape a0, RATE b0 (mean a0/b0)
    n_j | lambda  ~ Poisson(lambda * E_j),     j = 1..J   (known exposures E_j)

Exact conjugate posterior:

    lambda | n  ~ Gamma(a0 + sum_j n_j,  b0 + sum_j E_j)

so the posterior mean and sd are

    mean = (a0 + sum n) / (b0 + sum E)
    sd   = sqrt(a0 + sum n) / (b0 + sum E).

The SAME model is sampled with NumPyro NUTS and the two posteriors are compared.
This rung validates the NUTS plumbing end-to-end (model spec, PRNG handling,
chain grouping, diagnostics extraction) on a problem where the answer is exact.

Entry point: ``run_rung1(seed, num_warmup, num_samples, num_chains) -> dict``.
"""
import time

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from CDDF_analysis.hbi_mcmc.diagnostics import summarize_mcmc

# --- fixed toy configuration --------------------------------------------------
LAMBDA_TRUE = 2.5
A0 = 2.0          # Gamma prior shape
B0 = 1.0          # Gamma prior rate
EXPOSURES = np.array([0.5, 0.8, 1.0, 1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 3.5])


def make_rung1_data(seed=0, lam_true=LAMBDA_TRUE, exposures=EXPOSURES):
    """Draw counts n_j ~ Poisson(lam_true * E_j). Returns dict(n, E, lam_true)."""
    rng = np.random.default_rng(seed)
    E = np.asarray(exposures, dtype=np.float64)
    n = rng.poisson(lam_true * E)
    return {"n": n, "E": E, "lam_true": float(lam_true)}


def analytic_posterior(n, E, a0=A0, b0=B0):
    """Exact conjugate posterior Gamma(a_post, b_post); returns (mean, sd, a_post, b_post)."""
    a_post = a0 + float(np.sum(n))
    b_post = b0 + float(np.sum(E))
    mean = a_post / b_post
    sd = np.sqrt(a_post) / b_post
    return mean, sd, a_post, b_post


def model(E, n=None, a0=A0, b0=B0):
    """NumPyro model: lambda ~ Gamma(a0, b0); n_j ~ Poisson(lambda * E_j)."""
    lam = numpyro.sample("lam", dist.Gamma(a0, b0))
    numpyro.sample("n", dist.Poisson(lam * jnp.asarray(E)), obs=n)


def run_rung1(seed=0, num_warmup=500, num_samples=1000, num_chains=2):
    """Run NUTS on the conjugate model and compare against the exact posterior.

    Returns
    -------
    dict with keys
        nuts_mean, nuts_sd        : NUTS posterior mean/sd of lambda
        analytic_mean, analytic_sd: exact conjugate values
        mcse_mean                 : sd / sqrt(ess_bulk), Monte-Carlo SE of the mean
        r_hat_max, ess_bulk_min, ess_tail_min, n_divergent, runtime
    """
    data = make_rung1_data(seed=seed)
    an_mean, an_sd, _, _ = analytic_posterior(data["n"], data["E"])

    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=False,
    )
    t0 = time.time()
    mcmc.run(
        jax.random.PRNGKey(seed),
        E=data["E"],
        n=jnp.asarray(data["n"]),
        extra_fields=("diverging",),
    )
    runtime = time.time() - t0

    lam = np.asarray(mcmc.get_samples()["lam"])
    diag = summarize_mcmc(mcmc, runtime=runtime)
    nuts_mean = float(np.mean(lam))
    nuts_sd = float(np.std(lam, ddof=1))

    out = {
        "nuts_mean": nuts_mean,
        "nuts_sd": nuts_sd,
        "analytic_mean": float(an_mean),
        "analytic_sd": float(an_sd),
        "mcse_mean": nuts_sd / np.sqrt(diag["ess_bulk_min"]),
        "lam_true": data["lam_true"],
    }
    out.update(diag)
    return out


if __name__ == "__main__":
    res = run_rung1()
    for k, v in res.items():
        print(f"{k:>14s} : {v}")
