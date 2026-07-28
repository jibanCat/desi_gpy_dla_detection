# -*- coding: utf-8 -*-
"""sbc.py -- simulation-based calibration for Model A.

WHY: mock closure answers "did the 95% interval contain the truth THIS time?".
Only a rank-statistic test answers "is the interval the RIGHT WIDTH?".  A band
that is 3x too wide covers the truth every time and is useless; a band that is
2x too narrow covers it most of the time on an easy mock.  SBC separates them.

METHOD (Talts et al. 2018).  For each of ``n_sims`` replicas:
    theta~ ~ prior            (numpyro Predictive on the model with obs=None)
    y~     ~ p(y | theta~)    (the SAME draw's simulated counts)
    theta^(1..L) ~ p(theta | y~)                (NUTS, thinned to L draws)
    rank_q = #{ q(theta^(l)) < q(theta~) }      for each REPORTED quantity q
If the sampler and the model are self-consistent, rank_q ~ Uniform{0..L}.
Deviations are diagnostic: a U shape = intervals too NARROW (overconfident);
a central hump = too WIDE; a slope = a location bias.

WHAT IS REDUCED, AND WHY (stated exactly, as required)
  Production Model A on the real 2LPT-0 pack is 29 x 15 x 8 with 4 chains of
  1000 draws after 1500 warmup -- a 16-hour fit.  A full-scale SBC needs
  O(100) of those.  So this module runs a REDUCED-DIMENSION SBC and names
  every reduction:

  R1 GRID.  5 true-N bins x 2 fine-z (1 coarse-z) x 1 SNR stratum x 2 molly
     cells, instead of 29 x 15 (3 coarse) x 8 x 7.  The N grid still straddles
     BOTH reporting thresholds (20.0 and 20.3) so the reported functionals are
     the same functionals.
  R2 SAMPLER.  1 chain, 150 warmup / 150 draws, max_tree_depth 8, thinned to
     L = 50 ranks.  Single-chain runs cannot be R-hat-checked, so SBC ranks
     are NOT evidence of convergence -- that is block 1's job on the real run.
  R3 PRIOR.  The production priors are deliberately diffuse and one of them,
     fp_lam_total ~ Gamma(1/2, 1e-6), has prior mean 5e5 -- prior-predictive
     draws from it are not simulable data.  SBC therefore runs on a NARROWED
     prior (defaults in ``SBC_PRIOR``) with the FP block OFF, and the narrowed
     scales are stamped in the artifact.  The conclusion is then about the
     calibration of the SAMPLER + FORWARD MODEL under a proper prior, NOT
     about the production prior's own calibration.
  R4 RESPONSE.  The synthetic pack's response is the committed skew-normal
     parameterisation, not the 2LPT-0 fitted one.

MOCK/SYNTHETIC ONLY.  No survey data is touched anywhere in this module.
"""
from __future__ import annotations

import time
import warnings

import numpy as np

__all__ = ["SBC_GRID", "SBC_PRIOR", "SBC_SAMPLER", "sbc_run", "rank_histogram",
           "uniformity_test", "sbc_block"]

# --- R1: the reduced grid ---------------------------------------------------
SBC_GRID = dict(
    nhat_edges=np.round(np.arange(19.9, 20.4 + 1e-9, 0.1), 10),   # 5 bins
    zf_edges=np.round(np.arange(2.0, 2.2 + 1e-9, 0.1), 10),       # 2 fine z
    zc_edges=np.array([2.0, 2.2]),                                # 1 coarse z
    snr_edges=np.array([0.0, np.inf]),                            # 1 stratum
    n_molly_cells=2,
)

# --- R3: the narrowed prior (what makes prior draws simulable) --------------
SBC_PRIOR = dict(level_scale=0.6, slope_scale=0.4,
                 sigma_N_scale=0.15, sigma_z_scale=0.15,
                 fp_mode="off")

# --- R2: the reduced sampler ------------------------------------------------
SBC_SAMPLER = dict(num_warmup=150, num_samples=150, num_chains=1,
                   max_tree_depth=8, target_accept=0.9, n_ranks=50)


def _reported_from_f(f, pack):
    """Single f(B, Kf) -> the reported scalars (same names as evidence.py)."""
    from CDDF_analysis.hbi_mcmc.evidence import reported_quantities
    rep = reported_quantities(np.asarray(f)[None, None], pack)
    return {k: float(np.asarray(v).reshape(-1)[0]) for k, v in rep.items()}


def sbc_run(n_sims=48, *, seed=0, grid=None, prior=None, sampler=None,
            pack_seed=0, verbose=False):
    """Run the reduced SBC.  Returns (ranks dict, meta dict).

    ``ranks[name]`` is a list of ``n_sims`` integers in ``{0, ..., L}``.
    """
    import jax
    import jax.numpy as jnp
    from functools import partial
    from numpyro.infer import MCMC, NUTS, Predictive
    from numpyro.infer.initialization import init_to_value

    from CDDF_analysis.hbi_mcmc import model_a as ma
    from CDDF_analysis.hbi_mcmc.forward import build_consts
    from CDDF_analysis.hbi_mcmc.pack import synthetic_pack

    grid = dict(SBC_GRID if grid is None else grid)
    prior = dict(SBC_PRIOR if prior is None else prior)
    samp = dict(SBC_SAMPLER if sampler is None else sampler)
    L = int(samp["n_ranks"])

    # a template pack fixes the GEOMETRY (dX, molly, response, edges); only the
    # population parameters and the counts are re-drawn per replica.
    pack = synthetic_pack(pack_seed, **grid, fp_frac=0.0)
    consts = build_consts(pack, resp_clamp="both")
    model = partial(ma.model_a, **prior)

    # -- prior draws of the latents AND their simulated counts (one call)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pri = Predictive(model, num_samples=n_sims)(
            jax.random.PRNGKey(seed + 991), consts)
    pri = {k: np.asarray(v) for k, v in pri.items()}

    init_tpl = ma._data_informed_init(pack, consts, ma.ModelAConfig())
    init_tpl = {k: v for k, v in init_tpl.items()
                if not k.startswith("fp_")}

    names = None
    ranks, truths, meds, t0 = {}, {}, {}, time.time()
    n_used = 0
    for i in range(n_sims):
        f_true = pri["f"][i]
        y = np.asarray(pri["counts"][i]).astype(np.int64)
        if y.sum() <= 0:            # degenerate prior draw: no data to fit
            continue
        kernel = NUTS(model, target_accept_prob=samp["target_accept"],
                      max_tree_depth=samp["max_tree_depth"],
                      dense_mass=[("theta_level", "theta_slope", "eps_N")],
                      init_strategy=init_to_value(values=init_tpl))
        mcmc = MCMC(kernel, num_warmup=samp["num_warmup"],
                    num_samples=samp["num_samples"],
                    num_chains=samp["num_chains"], chain_method="vectorized",
                    progress_bar=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mcmc.run(jax.random.PRNGKey(seed + 10_000 + i), consts,
                     jnp.asarray(y), None)
        f_post = np.asarray(mcmc.get_samples()["f"])          # (n, B, Kf)
        thin = np.linspace(0, f_post.shape[0] - 1, L).astype(int)
        f_post = f_post[thin]

        q_true = _reported_from_f(f_true, pack)
        q_post = {k: [] for k in q_true}
        for l in range(L):
            for k, v in _reported_from_f(f_post[l], pack).items():
                q_post[k].append(v)
        if names is None:
            names = sorted(q_true)
            ranks = {k: [] for k in names}
            truths = {k: [] for k in names}
            meds = {k: [] for k in names}
        for k in names:
            arr = np.asarray(q_post[k], float)
            ranks[k].append(int((arr < q_true[k]).sum()))
            truths[k].append(float(q_true[k]))
            meds[k].append(float(np.median(arr)))
        n_used += 1
        if verbose:
            print(f"  [sbc] {i+1}/{n_sims} used={n_used} "
                  f"{time.time()-t0:.0f}s", flush=True)

    meta = {
        "n_sims_requested": int(n_sims), "n_sims_used": int(n_used),
        "n_ranks_L": L, "seed": int(seed),
        "wallclock_s": float(time.time() - t0),
        "reductions": {
            "R1_grid": {k: (np.asarray(v).tolist()
                            if isinstance(v, np.ndarray) else v)
                        for k, v in grid.items()},
            "R2_sampler": dict(samp),
            "R3_prior": dict(prior),
            "R4_response": "synthetic skew-normal committed parameterisation",
        },
        "reduction_note": (
            "REDUCED-DIMENSION SBC. Reduced: the (N, z, SNR) grid to "
            f"{len(grid['nhat_edges'])-1} x {len(grid['zf_edges'])-1} x "
            f"{len(grid['snr_edges'])-1}; the sampler to "
            f"{samp['num_chains']} chain x {samp['num_warmup']}/"
            f"{samp['num_samples']} thinned to L={L}; the prior to the "
            "narrowed scales in R3 with the FP block OFF (the production "
            "Gamma(1/2, 1e-6) FP-total prior has mean 5e5 and is not "
            "prior-predictively simulable). NOT reduced: the forward fold, "
            "the completeness surface, the response kernel, the reported "
            "functionals (both 20.0 and 20.3 thresholds are on the grid)."),
        "truths": truths, "post_medians": meds,
    }
    return ranks, meta


def rank_histogram(rank_list, L, n_bins=10):
    """Binned rank histogram + its expected count per bin."""
    r = np.asarray(rank_list, int)
    n = len(r)
    edges = np.linspace(0, L + 1, n_bins + 1)
    h, _ = np.histogram(r, bins=edges)
    return {"counts": [int(x) for x in h], "n_bins": int(n_bins),
            "n": int(n), "expected_per_bin": float(n / n_bins),
            "bin_edges": [float(x) for x in edges]}


def uniformity_test(rank_list, L, n_bins=10):
    """Pearson chi2 goodness-of-fit of the rank histogram against Uniform."""
    from scipy import stats
    hist = rank_histogram(rank_list, L, n_bins=n_bins)
    obs = np.asarray(hist["counts"], float)
    exp = hist["expected_per_bin"]
    if exp <= 0:
        return {"chi2": None, "dof": None, "p_value": None, "hist": hist}
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    dof = n_bins - 1
    p = float(stats.chi2.sf(chi2, dof))
    # a coarse shape read: U (too narrow) vs hump (too wide) vs slope (biased)
    half = n_bins // 2
    edge_mass = float(obs[0] + obs[-1]) / max(obs.sum(), 1)
    slope = float(obs[half:].sum() - obs[:half].sum()) / max(obs.sum(), 1)
    shape = "uniform"
    if p < 0.01:
        if edge_mass > 2.5 / n_bins:
            shape = "U_shaped_intervals_too_narrow"
        elif edge_mass < 0.6 / n_bins:
            shape = "central_hump_intervals_too_wide"
        else:
            shape = "sloped_location_bias"
    return {"chi2": chi2, "dof": int(dof), "p_value": p,
            "edge_mass_frac": edge_mass, "half_split_slope": slope,
            "shape": shape, "hist": hist}


def sbc_block(n_sims=48, *, seed=0, n_bins=10, **kw):
    """Block 4.  Runs the reduced SBC and gates on rank uniformity."""
    ranks, meta = sbc_run(n_sims, seed=seed, **kw)
    if not ranks:
        return {"incomplete": ["sbc_produced_no_usable_replicas"],
                "checks": {"sbc_uniform_ok": False}, "meta": meta}
    L = meta["n_ranks_L"]
    per_q, worst_p, worst_name = {}, 1.0, None
    for name, rk in sorted(ranks.items()):
        t = uniformity_test(rk, L, n_bins=n_bins)
        t["ranks"] = [int(x) for x in rk]
        per_q[name] = t
        if t["p_value"] is not None and t["p_value"] < worst_p:
            worst_p, worst_name = t["p_value"], name
    out = {"per_quantity": per_q, "meta": meta,
           "worst_p_value": float(worst_p), "worst_quantity": worst_name,
           "n_quantities": len(per_q),
           "p_threshold": None, "incomplete": []}
    from CDDF_analysis.hbi_mcmc.evidence import SBC_UNIFORM_P_MIN
    out["p_threshold"] = SBC_UNIFORM_P_MIN
    # Bonferroni over the reported quantities (they are strongly correlated,
    # so this is conservative in the direction of NOT crying wolf)
    out["worst_p_bonferroni"] = float(min(1.0, worst_p * len(per_q)))
    out["checks"] = {
        "sbc_uniform_ok": bool(out["worst_p_bonferroni"] >= SBC_UNIFORM_P_MIN),
        "sbc_enough_replicas": bool(meta["n_sims_used"] >= 20),
    }
    return out
