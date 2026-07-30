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
           "uniformity_test", "sbc_block", "SBC_GRID_ADOPTED",
           "SBC_ADOPTED_BASIS", "DISPERSION_SCALES"]

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

# --- MATCHED-CONFIGURATION grid (PI decision 8 requires the SBC configuration
# to MATCH the one being reported; PI decisions 3 and 4 changed it) -----------
# Wider in N than SBC_GRID on purpose: the ADOPTED 0.2-dex latent basis needs an
# even number of 0.1-dex observed bins above the reporting floor to come out
# uniform, and the primary reporting window [19.7, 21.6] must contain at least a
# few basis bins or the reported functional is a single number.
#   observed : 19.5 .. 20.5, 10 bins of 0.1 dex   (UNCHANGED step -- decision 3)
#   latent   : pad 19.0 -> 19.0 19.2 19.5 | 19.5 19.7 19.9 20.1 20.3 20.5
#              = 2 pad bins + 5 in-window-capable bins
SBC_GRID_ADOPTED = dict(
    nhat_edges=np.round(np.arange(19.5, 20.5 + 1e-9, 0.1), 10),   # 10 bins
    zf_edges=np.round(np.arange(2.0, 2.2 + 1e-9, 0.1), 10),       # 2 fine z
    zc_edges=np.array([2.0, 2.2]),                                # 1 coarse z
    snr_edges=np.array([0.0, np.inf]),                            # 1 stratum
    n_molly_cells=2,
)
SBC_ADOPTED_BASIS = dict(basis_width=0.2, pad_floor=19.0)

# --- the MEASURED POWER check (project rule, feedback_coverage_tests_need_power
# _checks) --------------------------------------------------------------------
# Truth-containment is MONOTONE in band width and therefore cannot fail an
# over-wide band; a 2x over-dispersed posterior once passed 24/24 tests and 8/8
# containment.  So every coverage claim here is accompanied by a DETECTION CURVE:
# the SAME posterior draws are re-scaled in log f about their own per-bin median
# by a factor s,
#       f_s(l) = median_l(f) * ( f(l) / median_l(f) ) ** s ,
# the ranks are recomputed, and the uniformity test is re-run.  s = 1 is the
# actual posterior; s > 1 is deliberately OVER-dispersed and s < 1 deliberately
# UNDER-dispersed.  This costs NO extra sampling.  If the test cannot flag
# s = 2.0, then it cannot certify the s = 1 result either, and the artifact must
# say so instead of claiming coverage.
DISPERSION_SCALES = (0.5, 0.75, 1.0, 1.5, 2.0)


def _reported_from_f(f, pack):
    """Single f(B, Kf) -> the reported scalars (same names as evidence.py).

    EXTENDED 2026-07-29 (PI decision 1): the primary reporting-window
    functionals ``dndx_report_197_216_allz`` / ``omega_report_197_216_allz`` are
    added, because after decision 1 those -- not the open-topped 20.0/20.3
    thresholds -- are what would actually be reported, and an SBC that ranks only
    the old functionals says nothing about the calibration of the new ones.
    """
    from CDDF_analysis.hbi_mcmc.evidence import reported_quantities
    from CDDF_analysis.hbi_mcmc.model_a import reduce_f_posterior
    rep = reported_quantities(np.asarray(f)[None, None], pack)
    out = {k: float(np.asarray(v).reshape(-1)[0]) for k, v in rep.items()}
    red = reduce_f_posterior(np.asarray(f)[None], pack)
    for nm in ("dndx", "omega"):
        key = f"{nm}_report_197_216_allz"
        if key in red:
            out[key] = float(np.asarray(red[key]).reshape(-1)[0])
    return out


def sbc_run(n_sims=48, *, seed=0, grid=None, prior=None, sampler=None,
            pack_seed=0, verbose=False, basis_width=None, pad_floor=None,
            dispersion_scales=None):
    """Run the reduced SBC.  Returns (ranks dict, meta dict).

    ``ranks[name]`` is a list of ``n_sims`` integers in ``{0, ..., L}``.

    ``basis_width`` / ``pad_floor`` (PI decisions 3 / 4): put the template pack's
    LATENT basis on the adopted geometry via ``pack.coarsen_basis`` before the
    prior is drawn, so the SBC is a MATCHED-configuration SBC (decision 8) rather
    than a calibration statement about a basis nobody reports on.  ``None``
    leaves the shipped geometry untouched, bit-for-bit.

    ``dispersion_scales``: the MEASURED POWER check.  For each s, the same
    posterior draws are re-scaled in log f about their per-bin median by s and the
    ranks recomputed, at NO extra sampling cost.  ``meta["ranks_by_scale"]``
    carries them.  s = 1.0 is always included.
    """
    import jax
    import jax.numpy as jnp
    from functools import partial
    from numpyro.infer import MCMC, NUTS, Predictive
    from numpyro.infer.initialization import init_to_value

    from CDDF_analysis.hbi_mcmc import model_a as ma
    from CDDF_analysis.hbi_mcmc.forward import build_consts
    from CDDF_analysis.hbi_mcmc.pack import synthetic_pack

    from CDDF_analysis.hbi_mcmc.pack import coarsen_basis

    grid = dict(SBC_GRID if grid is None else grid)
    prior = dict(SBC_PRIOR if prior is None else prior)
    samp = dict(SBC_SAMPLER if sampler is None else sampler)
    scales = tuple(sorted(set(
        (1.0,) if dispersion_scales is None else tuple(dispersion_scales) + (1.0,))))
    L = int(samp["n_ranks"])

    # a template pack fixes the GEOMETRY (dX, molly, response, edges); only the
    # population parameters and the counts are re-drawn per replica.
    pack = synthetic_pack(pack_seed, **grid, fp_frac=0.0)
    if basis_width is not None or pad_floor is not None:
        pack = coarsen_basis(pack, basis_width or 0.1, pad_floor=pad_floor)
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
    ranks_by_scale = {f"{s:g}": {} for s in scales}
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
        if names is None:
            names = sorted(q_true)
            ranks = {k: [] for k in names}
            truths = {k: [] for k in names}
            meds = {k: [] for k in names}
            for s in scales:
                ranks_by_scale[f"{s:g}"] = {k: [] for k in names}
        # per-bin median of the draws, in log f -- the pivot the power check
        # rescales about. Computed ONCE per replica and shared by every scale.
        with np.errstate(divide="ignore"):
            log_post = np.log(np.clip(f_post, 1e-300, None))
        log_med = np.median(log_post, axis=0)
        for s in scales:
            f_s = (f_post if s == 1.0
                   else np.exp(log_med[None, ...] + s * (log_post - log_med[None, ...])))
            q_post = {k: [] for k in names}
            for l in range(L):
                for k, v in _reported_from_f(f_s[l], pack).items():
                    q_post[k].append(v)
            for k in names:
                arr = np.asarray(q_post[k], float)
                ranks_by_scale[f"{s:g}"][k].append(int((arr < q_true[k]).sum()))
                if s == 1.0:
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
            "functionals (both 20.0 and 20.3 thresholds are on the grid). "
            "R5 REPORTING WINDOW (2026-07-29): the synthetic grid stops at "
            f"logN = {float(np.asarray(grid['nhat_edges'])[-1]):g}, so the "
            "'report_197_216' tier here integrates [19.7, "
            f"{float(np.asarray(grid['nhat_edges'])[-1]):g}) rather than "
            "[19.7, 21.6]. It is the same FUNCTIONAL FORM on the same latent "
            "basis width, over a shorter N range; it is NOT the production "
            "window and its value must not be compared to one."),
        "truths": truths, "post_medians": meds,
        "basis_width": (None if basis_width is None else float(basis_width)),
        "pad_floor": (None if pad_floor is None else float(pad_floor)),
        "matched_configuration": bool(basis_width is not None),
        "n_basis_bins": int(pack.n_b),
        "n_observed_bins": int(pack.n_c),
        "n_pad_bins": int(pack.n_pad_bins),
        "ntrue_edges": [float(x) for x in np.asarray(pack.ntrue_edges, float)],
        "dispersion_scales": [float(s) for s in scales],
        "ranks_by_scale": ranks_by_scale,
        "dispersion_scale_definition": (
            "f_s(l) = exp( log_med + s * (log f(l) - log_med) ) with log_med the "
            "PER-BIN median over draws of log f. s = 1 is the actual posterior; "
            "s > 1 is deliberately OVER-dispersed, s < 1 UNDER-dispersed. Costs "
            "no extra sampling. This is the MEASURED POWER check: a coverage "
            "claim at s = 1 is only meaningful if the test FLAGS s = 2."),
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
