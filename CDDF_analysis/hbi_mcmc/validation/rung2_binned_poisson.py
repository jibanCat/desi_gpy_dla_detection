"""rung2_binned_poisson.py — RUNG 2: synthetic binned-Poisson CDDF toy (+ response variant B).

Generative model
----------------
Truth: a CDDF-like density f_b over B = 15 logN-like bins with known (log-)bin
widths dN_b and known total pathlength-exposure DeltaX. Expected counts per TRUE
bin b:

    mu_b = DeltaX * f_b * dN_b .

VARIANT "direct"  (counts observed in the true bins):

    n_b ~ Poisson(mu_b),                       b = 1..B .

VARIANT "response" (variant B — the migration pattern; counts observed in
ESTIMATED bins c, smeared by a KNOWN row-stochastic response matrix R,
sum_b R_cb = 1 for every row c):

    m_c ~ Poisson( sum_b R_cb * DeltaX * f_b * dN_b ),   c = 1..C  (C = B here).

The forward fold sum_b R_cb * (...) is computed INSIDE the NumPyro model with
jnp per draw (see ``forward_fold``) — this is exactly the pattern the real
Model A uses (response applied per draw, differentiable).

Inference model (both variants)
-------------------------------
    theta_1        ~ Normal(0, 3)                        # log f in the first bin
    delta_b        ~ Normal(0, sigma_rw),  b = 2..B      # weak smoothness prior
    theta_b        = theta_1 + cumsum(delta)             # Gaussian random walk
    f_b            = exp(theta_b)
    counts         ~ Poisson(forward model above)

Checks: (i) per-bin 68% central-interval coverage of f_true (loosely asserted in
tests to stay non-flaky); (ii) the INTEGRATED total T = sum_b DeltaX*f_b*dN_b is
recovered unbiasedly in the response variant (the response is row-stochastic and
known, so no information about the total is lost, only resolution).

Entry point: ``run_rung2(variant, seed, num_warmup, num_samples, num_chains) -> dict``.
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
N_BINS = 15
LOGN_EDGES = np.linspace(20.0, 21.5, N_BINS + 1)   # logN-like grid
DN_B = np.diff(LOGN_EDGES)                          # known bin widths (0.1 dex each)
DELTA_X = 4000.0                                    # known exposure
SIGMA_RW = 0.5                                      # weak random-walk smoothness prior
RESPONSE_WIDTH_BINS = 1.2                           # Gaussian smearing width (in bins)


def make_f_true():
    """Decaying power-law-like truth with mild curvature; counts span ~40..1200."""
    b = np.arange(N_BINS, dtype=np.float64)
    return 3.0 * np.exp(-0.25 * b - 0.005 * b ** 2)


def make_response_matrix(n_bins=N_BINS, width=RESPONSE_WIDTH_BINS):
    """Known ROW-stochastic Gaussian smearing matrix R_cb (rows c sum to 1)."""
    c = np.arange(n_bins, dtype=np.float64)[:, None]
    b = np.arange(n_bins, dtype=np.float64)[None, :]
    R = np.exp(-0.5 * ((c - b) / width) ** 2)
    R /= R.sum(axis=1, keepdims=True)
    return R


def forward_fold(f, dX, dN, R=None):
    """Expected counts per observed bin; the per-draw forward fold (pure jnp).

    mu = dX * f * dN                (direct)
    mu_c = sum_b R_cb dX f_b dN_b   (response variant B)

    Differentiable; called INSIDE the model on each draw.
    """
    lam = dX * f * jnp.asarray(dN)
    if R is None:
        return lam
    return jnp.asarray(R) @ lam


def make_rung2_data(seed=0, variant="direct"):
    """Draw the synthetic counts. Returns dict(counts, f_true, dN, dX, R)."""
    if variant not in ("direct", "response"):
        raise ValueError(f"unknown variant {variant!r}")
    rng = np.random.default_rng(seed)
    f_true = make_f_true()
    R = make_response_matrix() if variant == "response" else None
    mu = np.asarray(forward_fold(jnp.asarray(f_true), DELTA_X, DN_B, R=R))
    counts = rng.poisson(mu)
    return {"counts": counts, "f_true": f_true, "dN": DN_B, "dX": DELTA_X, "R": R}


def model(dX, dN, R=None, counts=None, sigma_rw=SIGMA_RW):
    """NumPyro model: log-f random walk prior + Poisson counts via forward_fold."""
    n_bins = len(dN)
    theta1 = numpyro.sample("theta1", dist.Normal(0.0, 3.0))
    delta = numpyro.sample(
        "delta", dist.Normal(0.0, sigma_rw).expand([n_bins - 1]).to_event(1)
    )
    theta = theta1 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(delta)])
    f = numpyro.deterministic("f", jnp.exp(theta))
    mu = forward_fold(f, dX, dN, R=R)  # response applied per draw, inside the model
    numpyro.sample("counts", dist.Poisson(mu), obs=counts)


def run_rung2(variant="direct", seed=0, num_warmup=500, num_samples=500,
              num_chains=2):
    """NUTS recovery of f_b. Returns per-bin summaries + coverage + total check.

    Returns
    -------
    dict with keys
        f_true, f_mean, f_sd, f_lo68, f_hi68 : arrays (B,)
        coverage68     : fraction of bins with f_true inside [q16, q84]
        total_true     : sum_b dX*f_true_b*dN_b
        total_mean, total_sd : posterior of the same integrated total
        r_hat_max, ess_bulk_min, ess_tail_min, n_divergent, runtime, variant
    """
    data = make_rung2_data(seed=seed, variant=variant)

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
        dX=data["dX"],
        dN=data["dN"],
        R=data["R"],
        counts=jnp.asarray(data["counts"]),
        extra_fields=("diverging",),
    )
    runtime = time.time() - t0

    f = np.asarray(mcmc.get_samples()["f"])          # (draws, B)
    q16, q84 = np.percentile(f, [15.865, 84.135], axis=0)
    f_true = data["f_true"]
    coverage = float(np.mean((f_true >= q16) & (f_true <= q84)))

    totals = (data["dX"] * f * data["dN"]).sum(axis=1)
    total_true = float((data["dX"] * f_true * data["dN"]).sum())

    out = {
        "variant": variant,
        "f_true": f_true,
        "f_mean": f.mean(axis=0),
        "f_sd": f.std(axis=0, ddof=1),
        "f_lo68": q16,
        "f_hi68": q84,
        "coverage68": coverage,
        "total_true": total_true,
        "total_mean": float(totals.mean()),
        "total_sd": float(totals.std(ddof=1)),
    }
    out.update(summarize_mcmc(mcmc, runtime=runtime))
    return out


if __name__ == "__main__":
    for variant in ("direct", "response"):
        res = run_rung2(variant=variant)
        print(f"--- variant={variant} ---")
        for k in ("coverage68", "total_true", "total_mean", "total_sd",
                  "r_hat_max", "ess_bulk_min", "ess_tail_min", "n_divergent",
                  "runtime"):
            print(f"{k:>14s} : {res[k]}")
