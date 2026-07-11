"""rung3_recycling.py — RUNG 3: per-sightline FGMC (k<=1) prior-QMC recycling core.

The important de-risk for the MCMC migration: the per-sightline evidence-recycling
integral I_i(Lambda) evaluated from a PERSISTED prior-sample table (computed once,
reused on every NUTS draw, differentiable in the population parameters) must agree
with brute-force quadrature, and NUTS on the recycled likelihood must recover the
population rate.

Generative model (EXACT; all synthetic, seeds fixed)
----------------------------------------------------
Population intensity over absorber parameter theta in [0, 1]:

    nu(theta | Lambda, a) = Lambda * g(theta | a),
    g(theta | a) = Beta(theta; a, a) pdf  (unit integral on [0,1]),

so the per-sightline mean occupation is mu(Lambda) = integral nu = Lambda
(uniform exposure E_i = 1 for all N_sl sightlines).

Per sightline i, a marked-Poisson draw CONDITIONED ON k <= 1 (the FGMC k<=1
regime; conditioning replaces "hope Lambda is small" so the math stays exact):

    k_i ~ Poisson(mu) | k <= 1     =>   P(k=1 | k<=1) = Lambda / (1 + Lambda)
                                        P(k=0 | k<=1) = 1 / (1 + Lambda)
    if k_i = 1 : theta_i ~ nu/mu = g(. | a);  d_i ~ Normal(theta_i, sigma_meas)
    if k_i = 0 : d_i ~ Normal(mu_null, sigma_null)   # noise-only model, ANALYTIC

Every sightline yields exactly one datum d_i; k_i is NOT observed.

Exact per-sightline likelihood
------------------------------
Unconditioned marked-Poisson (k <= 1 outcomes only; the form the real Model A uses):

    p_uncond(d_i | Lambda, a) = e^{-mu(Lambda)} * [ Z_0i + I_i(Lambda, a) ],
    Z_0i               = Normal(d_i; mu_null, sigma_null)          # null evidence, exact
    I_i(Lambda, a)     = integral_0^1 N(d_i; theta, sigma_meas) * nu(theta|Lambda,a) dtheta.

This is a SUB-probability over the k<=1 sample space (its integral over d is
e^{-mu}(1+mu) = P(k<=1)). Because generation above conditions on k<=1, the exact
sampling density divides by that mass:

    p(d_i | Lambda, a, k<=1) = p_uncond / [ e^{-mu} (1 + mu) ]
                             = [ Z_0i + I_i(Lambda, a) ] / (1 + Lambda),

i.e. the e^{-mu} factors cancel exactly. The model implements the spec form
``log p_uncond = -Lambda + logaddexp(log Z_0i, log I_i)`` per sightline plus the
single conditioning correction ``- N_sl * (-Lambda + log1p(Lambda))``; both terms
are exact, no approximation anywhere except the QMC estimate of I_i itself.

Prior-QMC recycling table (persisted; the object under test)
------------------------------------------------------------
Parameter-estimation prior pi_PE = Uniform(0, 1) (density 1). A SHARED
deterministic S-point grid theta_s (1-D Halton, or linspace midpoints) with
persisted base log-likelihoods

    l_is = log Normal(d_i; theta_s, sigma_meas)        # computed ONCE

gives the per-draw, differentiable estimate

    I_i(Lambda, a) ~= (1/S) sum_s exp(l_is) * nu(theta_s | Lambda, a) / pi_PE(theta_s)
    log I_i        = log Lambda - log S + logsumexp_s [ l_is + log g(theta_s | a) ].

Validation (see tests):
  (a) I_i from QMC vs brute-force trapezoid quadrature of
      integral exp(l_i(theta)) nu(theta|Lambda) dtheta, 5 detected sightlines x
      3 Lambda values, relative agreement <= 1%;
  (b) NUTS on (Lambda, a) recovers Lambda_true within 2 sigma; per-object
      recycling ESS_i = (sum_s w_is)^2 / sum_s w_is^2 at the posterior mean is
      finite for all sightlines and healthy (> 20) for detected ones. ESS
      starvation on empty sightlines with |d| far outside [0,1] is EXPECTED
      (the likelihood mass sits at the edge of pi_PE) and harmless: there the
      Z_0i term dominates logaddexp and I_i is negligible.

Entry point: ``run_rung3(seed, num_warmup, num_samples, num_chains) -> dict``.
"""
import time

import numpy as np
from scipy.stats import qmc as scipy_qmc
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp, betaln
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from CDDF_analysis.hbi_mcmc.diagnostics import summarize_mcmc, recycling_ess

# --- fixed toy configuration --------------------------------------------------
LAMBDA_TRUE = 0.6
A_TRUE = 2.0            # Beta(a, a) shape of the unit-mean population shape g
N_SIGHTLINES = 200
SIGMA_MEAS = 0.05       # measurement noise on theta for a real absorber
MU_NULL = 0.0           # noise-only (no-absorber) datum model: d ~ N(MU_NULL, SIGMA_NULL)
SIGMA_NULL = 0.75
S_QMC = 2000            # shared prior-QMC grid size
_LOG_2PI = np.log(2.0 * np.pi)


def log_normal_pdf(x, loc, scale):
    """log Normal(x; loc, scale) — works for numpy or jnp inputs."""
    z = (x - loc) / scale
    return -0.5 * z * z - jnp.log(scale) - 0.5 * _LOG_2PI


def log_beta_shape(theta, a):
    """log g(theta | a) = log Beta(theta; a, a) pdf (unit-mean shape on [0,1])."""
    return (a - 1.0) * (jnp.log(theta) + jnp.log1p(-theta)) - betaln(a, a)


def make_qmc_grid(S=S_QMC, kind="halton"):
    """Shared deterministic S-point grid theta_s in (0,1) under pi_PE = Uniform(0,1)."""
    if kind == "halton":
        sampler = scipy_qmc.Halton(d=1, scramble=False)
        theta = sampler.random(S + 1)[1:, 0]  # drop the leading 0 of the sequence
    elif kind == "linspace":
        theta = (np.arange(S) + 0.5) / S       # midpoint rule
    else:
        raise ValueError(f"unknown grid kind {kind!r}")
    return np.clip(theta, 1e-12, 1.0 - 1e-12)


def make_rung3_data(seed=0, n_sl=N_SIGHTLINES, lam_true=LAMBDA_TRUE, a_true=A_TRUE,
                    sigma_meas=SIGMA_MEAS, grid_kind="halton", S=S_QMC):
    """Generate the synthetic sightlines + the persisted prior-QMC table.

    Returns
    -------
    dict with keys
        d          : (N,) observed data
        has_abs    : (N,) bool, TRUTH occupancy (k_i = 1) — diagnostics only
        theta_true : (N,) absorber positions (nan where k_i = 0)
        theta_grid : (S,) shared QMC grid theta_s
        logL_table : (N, S) persisted base log-likelihoods l_is
        logZ0      : (N,) exact null log-evidence log N(d_i; MU_NULL, SIGMA_NULL)
        lam_true, a_true, sigma_meas
    """
    rng = np.random.default_rng(seed)
    p_abs = lam_true / (1.0 + lam_true)          # P(k=1 | k<=1), exact
    has_abs = rng.random(n_sl) < p_abs
    theta_true = np.full(n_sl, np.nan)
    theta_true[has_abs] = rng.beta(a_true, a_true, size=int(has_abs.sum()))
    d = np.where(
        has_abs,
        np.nan_to_num(theta_true) + sigma_meas * rng.standard_normal(n_sl),
        MU_NULL + SIGMA_NULL * rng.standard_normal(n_sl),
    )

    theta_grid = make_qmc_grid(S=S, kind=grid_kind)
    # persisted base log-likelihoods l_is = log N(d_i; theta_s, sigma_meas)
    logL_table = np.asarray(
        log_normal_pdf(d[:, None], theta_grid[None, :], sigma_meas)
    )
    logZ0 = np.asarray(log_normal_pdf(d, MU_NULL, SIGMA_NULL))
    return {
        "d": d,
        "has_abs": has_abs,
        "theta_true": theta_true,
        "theta_grid": theta_grid,
        "logL_table": logL_table,
        "logZ0": logZ0,
        "lam_true": float(lam_true),
        "a_true": float(a_true),
        "sigma_meas": float(sigma_meas),
    }


def log_I_qmc(lam, a, logL_table, theta_grid):
    """log I_i(Lambda, a) via prior-QMC recycling (pure jnp; per-draw, differentiable).

    log I_i = log Lambda - log S + logsumexp_s [ l_is + log g(theta_s | a) ].
    pi_PE = Uniform(0,1) has density 1, so no explicit 1/pi factor appears.
    """
    theta_grid = jnp.asarray(theta_grid)
    S = theta_grid.shape[0]
    log_g = log_beta_shape(theta_grid, a)                       # (S,)
    return (jnp.log(lam) - jnp.log(S)
            + logsumexp(jnp.asarray(logL_table) + log_g[None, :], axis=-1))


def log_like_unconditioned(lam, a, logL_table, logZ0, theta_grid):
    """Spec-form per-sightline log-likelihood (k<=1 marked Poisson, UNconditioned):

    log p_uncond(d_i | Lambda, a) = -Lambda + logaddexp(log Z_0i, log I_i(Lambda, a)).
    """
    logI = log_I_qmc(lam, a, logL_table, theta_grid)
    return -lam + jnp.logaddexp(jnp.asarray(logZ0), logI)


def model(logL_table, logZ0, theta_grid, sample_shape=True, a_fixed=A_TRUE):
    """NumPyro model: priors on (Lambda[, a]) + exact conditioned FGMC likelihood.

    Total log-likelihood = sum_i log p_uncond(d_i) - N * log P(k<=1 | Lambda),
    with log P(k<=1) = -Lambda + log1p(Lambda) (the e^{-Lambda} terms cancel).
    """
    lam = numpyro.sample("lam", dist.LogNormal(0.0, 1.0))
    if sample_shape:
        a = numpyro.sample("a_shape", dist.LogNormal(jnp.log(2.0), 0.5))
    else:
        a = a_fixed
    n_sl = logZ0.shape[0]
    log_p_uncond = log_like_unconditioned(lam, a, logL_table, logZ0, theta_grid)
    log_p_le1 = -lam + jnp.log1p(lam)            # log P(k<=1 | Lambda), exact
    numpyro.factor("loglik", jnp.sum(log_p_uncond) - n_sl * log_p_le1)


def compare_qmc_vs_quadrature(data=None, seed=0, lam_values=(0.3, 0.6, 1.2),
                              a=A_TRUE, n_compare=5, n_grid=20001):
    """Validation (a): recycled I_i vs brute-force trapezoid quadrature.

    For the first `n_compare` DETECTED sightlines (detected so I_i is not a pure
    underflow tail) and each Lambda in `lam_values`, compares the QMC estimate
    against trapezoid quadrature of integral exp(l_i(theta)) nu(theta|Lambda) dtheta
    on an `n_grid`-point dense grid.

    Returns dict(rel_err (n_compare, n_lam), max_rel_err, I_qmc, I_quad).
    """
    if data is None:
        data = make_rung3_data(seed=seed)
    idx = np.flatnonzero(data["has_abs"])[:n_compare]
    d_sel = data["d"][idx]

    # dense trapezoid grid (avoid the exact endpoints where log g diverges for a<1)
    th = np.linspace(1e-9, 1.0 - 1e-9, n_grid)
    log_g_dense = np.asarray(log_beta_shape(jnp.asarray(th), a))
    logL_dense = np.asarray(log_normal_pdf(d_sel[:, None], th[None, :],
                                           data["sigma_meas"]))

    I_qmc = np.empty((len(idx), len(lam_values)))
    I_quad = np.empty_like(I_qmc)
    for j, lam in enumerate(lam_values):
        # QMC (same code path as the model)
        I_qmc[:, j] = np.exp(np.asarray(
            log_I_qmc(lam, a, data["logL_table"][idx], data["theta_grid"])
        ))
        # brute force: integrand exp(l_i(theta)) * Lambda * g(theta)
        integrand = np.exp(logL_dense + log_g_dense[None, :]) * lam
        I_quad[:, j] = np.trapezoid(integrand, th, axis=1)

    rel_err = np.abs(I_qmc - I_quad) / I_quad
    return {
        "sightline_idx": idx,
        "lam_values": np.asarray(lam_values),
        "I_qmc": I_qmc,
        "I_quad": I_quad,
        "rel_err": rel_err,
        "max_rel_err": float(rel_err.max()),
    }


def recycling_ess_at(lam, a, data):
    """Per-object recycling ESS_i = (sum_s w_is)^2 / sum_s w_is^2 at (Lambda, a).

    w_is = exp(l_is) * nu(theta_s | Lambda, a) / pi_PE(theta_s); each row is
    max-shifted before exponentiation (ESS is scale-invariant per row).
    """
    log_g = np.asarray(log_beta_shape(jnp.asarray(data["theta_grid"]), a))
    log_w = data["logL_table"] + log_g[None, :] + np.log(lam)
    log_w = log_w - log_w.max(axis=1, keepdims=True)
    return recycling_ess(np.exp(log_w))


def run_rung3(seed=0, num_warmup=500, num_samples=500, num_chains=2,
              sample_shape=True, grid_kind="halton"):
    """NUTS on (Lambda[, a]) with the recycled likelihood; recovery + diagnostics.

    Returns
    -------
    dict with keys
        lam_true, lam_mean, lam_sd, lam_z (|mean-true|/sd)
        a_true, a_mean, a_sd (nan if sample_shape=False)
        n_detected
        ess_recycle_min_detected, ess_recycle_median_detected, ess_recycle_min_all
        r_hat_max, ess_bulk_min, ess_tail_min, n_divergent, runtime
    """
    data = make_rung3_data(seed=seed, grid_kind=grid_kind)

    kernel = NUTS(model, target_accept_prob=0.9)
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
        logL_table=jnp.asarray(data["logL_table"]),
        logZ0=jnp.asarray(data["logZ0"]),
        theta_grid=jnp.asarray(data["theta_grid"]),
        sample_shape=sample_shape,
        extra_fields=("diverging",),
    )
    runtime = time.time() - t0

    samples = mcmc.get_samples()
    lam = np.asarray(samples["lam"])
    lam_mean, lam_sd = float(lam.mean()), float(lam.std(ddof=1))
    if sample_shape:
        a_draws = np.asarray(samples["a_shape"])
        a_mean, a_sd = float(a_draws.mean()), float(a_draws.std(ddof=1))
    else:
        a_mean, a_sd = float(A_TRUE), float("nan")

    ess_i = recycling_ess_at(lam_mean, a_mean, data)
    det = data["has_abs"]
    out = {
        "lam_true": data["lam_true"],
        "lam_mean": lam_mean,
        "lam_sd": lam_sd,
        "lam_z": abs(lam_mean - data["lam_true"]) / lam_sd,
        "a_true": data["a_true"],
        "a_mean": a_mean,
        "a_sd": a_sd,
        "n_detected": int(det.sum()),
        "ess_recycle_min_detected": float(np.min(ess_i[det])),
        "ess_recycle_median_detected": float(np.median(ess_i[det])),
        "ess_recycle_min_all": float(np.min(ess_i)),
    }
    out.update(summarize_mcmc(mcmc, runtime=runtime))
    return out


if __name__ == "__main__":
    cmp_res = compare_qmc_vs_quadrature()
    print(f"QMC vs trapezoid max rel err: {cmp_res['max_rel_err']:.3e}")
    res = run_rung3()
    for k, v in res.items():
        print(f"{k:>28s} : {v}")
