"""diagnostics.py — small shared MCMC + recycling diagnostics for hbi_mcmc.

Two helpers, both plain functions:

* summarize_mcmc(mcmc, runtime=None) -> dict
    Convergence summary for a finished ``numpyro.infer.MCMC`` object:
    max split-R-hat, min bulk ESS, min tail ESS (both rank-normalized, via
    arviz), number of divergent transitions, and the (caller-measured) wall
    runtime. Divergence counting requires the run to have been made with
    ``extra_fields=("diverging",)`` — all run_* drivers in validation/ do this.

* recycling_ess(weights) -> float
    Kish effective sample size of an importance/recycling weight vector,

        ESS = (sum_s w_s)^2 / sum_s w_s^2 ,

    the standard per-object diagnostic for prior-sample recycling
    (ESS = S for uniform weights, 1 for a one-hot weight vector).
"""
import warnings

import numpy as np

with warnings.catch_warnings():
    # arviz 0.23 emits a FutureWarning banner at import; keep test output clean.
    warnings.simplefilter("ignore", FutureWarning)
    import arviz as az


def summarize_mcmc(mcmc, runtime=None):
    """Summarize convergence diagnostics of a finished numpyro MCMC run.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        A run MCMC object. Deterministic sites are excluded from the R-hat/ESS
        scan (they are functions of the sampled sites).
    runtime : float, optional
        Wall-clock seconds measured by the caller around ``mcmc.run``; passed
        through unchanged (the MCMC object itself does not store it).

    Returns
    -------
    dict with keys
        r_hat_max, ess_bulk_min, ess_tail_min : float
        n_divergent : int
        runtime : float or None
    """
    samples = mcmc.get_samples(group_by_chain=True)
    # keep only actually-sampled sites (deterministics are derived quantities)
    try:
        sample_sites = set(mcmc.last_state.z.keys())
    except AttributeError:  # pragma: no cover - defensive
        sample_sites = None
    posterior = {}
    for name, arr in samples.items():
        if sample_sites is not None and name not in sample_sites:
            continue
        posterior[name] = np.asarray(arr)
    if not posterior:  # fallback: use everything
        posterior = {k: np.asarray(v) for k, v in samples.items()}

    idata = az.from_dict(posterior=posterior)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rhat = az.rhat(idata)
        ess_bulk = az.ess(idata, method="bulk")
        ess_tail = az.ess(idata, method="tail")

    def _reduce(ds, fn):
        vals = [fn(np.asarray(ds[v].values)) for v in ds.data_vars]
        return float(fn(np.asarray(vals)))

    extra = mcmc.get_extra_fields(group_by_chain=True)
    diverging = extra.get("diverging", None)
    n_div = int(np.sum(np.asarray(diverging))) if diverging is not None else 0

    return {
        "r_hat_max": _reduce(rhat, np.max),
        "ess_bulk_min": _reduce(ess_bulk, np.min),
        "ess_tail_min": _reduce(ess_tail, np.min),
        "n_divergent": n_div,
        "runtime": runtime,
    }


def recycling_ess(weights):
    """Kish ESS = (sum w)^2 / sum w^2 of a nonnegative weight vector.

    Parameters
    ----------
    weights : array-like, shape (S,) or (N, S)
        Recycling weights. If 2-D, the ESS is computed per row.

    Returns
    -------
    float or ndarray of shape (N,)
        ESS in [1, S] for any non-degenerate weight vector; nan if all
        weights in a row are zero.
    """
    w = np.asarray(weights, dtype=np.float64)
    s1 = w.sum(axis=-1)
    s2 = (w * w).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ess = np.where(s2 > 0.0, (s1 * s1) / s2, np.nan)
    return float(ess) if np.ndim(ess) == 0 else ess
