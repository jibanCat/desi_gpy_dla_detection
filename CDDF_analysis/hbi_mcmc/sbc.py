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
           "uniformity_test", "sbc_block", "MATCH_KEYS", "REPORTED_ONLY_KEYS",
           "sbc_configuration", "run_configuration", "configuration_match",
           "matched_sbc_kwargs"]

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


# ===========================================================================
# MATCHED-CONFIGURATION SBC  (decision 8, RATIFIED 2026-07-29)
#
# RATIFIED STATEMENT (see ratification.py, key "matched_configuration_sbc"):
#   simulation-based calibration may certify ONLY the configuration it
#   actually ran.  An SBC whose grid, prior, FP mode, response clamp, sampler
#   or reported-quantity set differs from the run it is attached to does NOT
#   certify that run, and an artifact carrying an unmatched -- or an
#   UNSPECIFIED -- SBC is NOT STAMPABLE.
#
# WHY THIS IS AN ENFORCED CHECK AND NOT A README LINE.  The reductions R1-R4
# in the module docstring are honestly documented, and were still, before this
# change, invisible to the gate: ``sbc_block`` reported a uniformity p-value,
# ``evidence.gate`` turned it into ``coverage_sbc.sbc_uniform_ok``, and a
# 5 x 2 x 1 single-chain narrowed-prior FP-off SBC therefore certified a
# 29 x 15 x 8 four-chain production fit.  Rank uniformity of object A is not
# evidence about object B.
#
# WHAT IS AND IS NOT ATTEMPTED HERE.  Making a matched SBC CHEAP is out of
# scope (O(100) production-scale fits; see ``matched_sbc_kwargs`` for the
# cost statement).  Making an UNMATCHED one REFUSE TO CERTIFY is what is
# implemented, and it is the half that closes the hole: after this change the
# existing reduced SBC is still run, still reported and still diagnostic, but
# ``coverage_sbc.sbc_configuration_matches_run`` is False, so the artifact is
# not stampable and cannot be quietly mistaken for a certified one.
# ===========================================================================

#: The configuration coordinates that MUST be identical for an SBC to certify
#: a run.  Dotted paths into the mapping built by ``_configuration``.  Adding a
#: key here can only ever make matching STRICTER, which is the fail-closed
#: direction; removing one requires a ratification edit.
MATCH_KEYS = (
    # R1 -- the grid.  Both N axes: ``ntrue_edges`` is the LATENT basis
    # (decision 3, the 0.2-dex basis) and ``nhat_edges`` the observed axis;
    # an SBC on a different basis calibrates a different parameter vector.
    "grid.nhat_edges", "grid.ntrue_edges", "grid.zf_edges", "grid.zc_edges",
    "grid.snr_edges", "grid.n_molly_cells",
    # R3 -- the prior, INCLUDING the FP block mode.
    "prior.level_scale", "prior.slope_scale", "prior.sigma_N_scale",
    "prior.sigma_z_scale", "prior.fp_mode", "prior.fp_eps_rate",
    "prior.fp_shape_sd",
    # R4 / D2 -- the response covariate-range clamp.
    "response.resp_clamp",
    # R2 -- the sampler.  SBC ranks are a joint statement about the model AND
    # the sampler that drew them; 1 chain x 150/150 at tree depth 8 is not the
    # production sampler, so it is a match coordinate, not a footnote.
    "sampler.num_warmup", "sampler.num_samples", "sampler.num_chains",
    "sampler.max_tree_depth", "sampler.target_accept",
    # the functionals the ranks were computed on must be the ones reported.
    "reported.quantities",
)

#: Coordinates that are RECORDED but are deliberately NOT match coordinates:
#: they change what the SBC's own error bar is, not what object it certifies.
REPORTED_ONLY_KEYS = ("sbc.n_sims_requested", "sbc.n_sims_used",
                      "sbc.n_ranks_L", "sbc.seed", "sbc.pack_seed",
                      "sbc.wallclock_s")

_PRIOR_KEYS = ("level_scale", "slope_scale", "sigma_N_scale", "sigma_z_scale",
               "fp_mode", "fp_eps_rate", "fp_shape_sd")
_SAMPLER_KEYS = ("num_warmup", "num_samples", "num_chains", "max_tree_depth",
                 "target_accept")

#: prior defaults, so that a config built from a partial kwargs dict is
#: compared against the value the model would actually have used.
_PRIOR_DEFAULTS = dict(level_scale=4.0, slope_scale=2.0, sigma_N_scale=0.5,
                       sigma_z_scale=0.5, fp_mode="joint", fp_eps_rate=1e-6,
                       fp_shape_sd=3.0)


def _edges(a):
    return [float(x) for x in np.asarray(a, float).ravel()]


def _grid_config(pack):
    """The realized geometry of a pack -- read off the pack, never from the
    kwargs that were *asked* for (a builder is free to round or extend them)."""
    return {
        "nhat_edges": _edges(pack.nhat_edges),
        "ntrue_edges": _edges(pack.ntrue_edges),
        "zf_edges": _edges(pack.zf_edges),
        "zc_edges": _edges(pack.zc_edges),
        "snr_edges": _edges(pack.snr_edges),
        "n_molly_cells": int(pack.n_molly),
    }


def _prior_config(prior):
    """Normalised prior coordinates.

    ``fp_eps_rate`` / ``fp_shape_sd`` are set to ``None`` when
    ``fp_mode == 'off'``: with the FP block off they parameterise nothing, and
    comparing them would manufacture a mismatch that is not a difference.  A
    difference in ``fp_mode`` itself is always a mismatch.
    """
    p = dict(_PRIOR_DEFAULTS)
    p.update({k: v for k, v in dict(prior or {}).items() if k in _PRIOR_KEYS})
    out = {}
    for k in _PRIOR_KEYS:
        v = p.get(k)
        out[k] = v if isinstance(v, str) or v is None else float(v)
    if out["fp_mode"] == "off":
        out["fp_eps_rate"] = None
        out["fp_shape_sd"] = None
    return out


def _sampler_config(sampler):
    s = dict(sampler or {})
    out = {}
    for k in _SAMPLER_KEYS:
        v = s.get(k)
        if v is None:
            out[k] = None
        elif k == "target_accept":
            out[k] = float(v)
        else:
            out[k] = int(v)
    return out


def _configuration(pack, *, prior, sampler, resp_clamp, reported_names):
    return {
        "grid": _grid_config(pack),
        "prior": _prior_config(prior),
        "response": {"resp_clamp": (None if resp_clamp is None
                                    else str(resp_clamp))},
        "sampler": _sampler_config(sampler),
        "reported": {"quantities": sorted(str(n) for n in reported_names)},
    }


def _reported_names(pack):
    """The reported-functional NAME SET for a pack (values are irrelevant)."""
    from CDDF_analysis.hbi_mcmc.evidence import reported_quantities
    f = np.ones((1, 1, pack.n_b, pack.n_k), float)
    return sorted(reported_quantities(f, pack))


def run_configuration(pack, cfg=None, *, resp_clamp=None,
                      reported_names=None):
    """The configuration coordinates of a PRODUCTION run.

    ``cfg`` is a ``model_a.ModelAConfig`` (or anything with the same
    attributes).  ``resp_clamp`` defaults to ``cfg.resp_clamp``.  The
    reported-quantity set is derived from the pack unless given.
    """
    prior, sampler = {}, {}
    if cfg is not None:
        prior = {k: getattr(cfg, k) for k in _PRIOR_KEYS if hasattr(cfg, k)}
        sampler = {k: getattr(cfg, k) for k in _SAMPLER_KEYS if hasattr(cfg, k)}
        if hasattr(cfg, "target_accept"):
            sampler["target_accept"] = cfg.target_accept
        if resp_clamp is None:
            resp_clamp = getattr(cfg, "resp_clamp", None)
    if reported_names is None:
        reported_names = _reported_names(pack)
    return _configuration(pack, prior=prior, sampler=sampler,
                          resp_clamp=resp_clamp, reported_names=reported_names)


def sbc_configuration(block_or_meta):
    """The configuration an SBC ACTUALLY ran, or ``None`` if it did not record
    one.  ``None`` is the fail-closed answer: an SBC that cannot say what it
    ran certifies nothing (see ``configuration_match``)."""
    if not isinstance(block_or_meta, dict):
        return None
    if isinstance(block_or_meta.get("configuration"), dict):
        return block_or_meta["configuration"]
    meta = block_or_meta.get("meta")
    if isinstance(meta, dict) and isinstance(meta.get("configuration"), dict):
        return meta["configuration"]
    return None


def _flat(cfg, key):
    cur = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return ("__ABSENT__",)
        cur = cur[part]
    return cur


def configuration_match(sbc_cfg, run_cfg, *, keys=MATCH_KEYS):
    """Does ``sbc_cfg`` certify ``run_cfg``?  FAIL-CLOSED.

    Returns ``{"matched": bool, "mismatches": [...], "keys_compared": [...],
    "reasons": [...]}``.

    ``matched`` is True only when BOTH configurations are present and EVERY
    key in ``keys`` is present in both and equal.  An absent configuration on
    either side, or an absent key, is a MISMATCH -- never a pass.  That is the
    whole point: "we did not record what the SBC ran" must not certify.
    """
    reasons, mism = [], []
    if not isinstance(sbc_cfg, dict):
        reasons.append("the SBC recorded NO configuration: an SBC that cannot "
                       "state what it ran certifies nothing")
    if not isinstance(run_cfg, dict):
        reasons.append("no RUN configuration was supplied to the SBC block: "
                       "there is nothing for the SBC to be matched against")
    if reasons:
        return {"matched": False, "mismatches": [], "reasons": reasons,
                "keys_compared": list(keys),
                "sbc_configuration": sbc_cfg, "run_configuration": run_cfg}
    for k in keys:
        a, b = _flat(sbc_cfg, k), _flat(run_cfg, k)
        if a == ("__ABSENT__",) or b == ("__ABSENT__",) or a != b:
            mism.append({"key": k, "sbc": a, "run": b})
    if mism:
        reasons.append(
            "the SBC certifies a DIFFERENT configuration from the run it is "
            "attached to; it is reportable and diagnostic, but it does not "
            "certify this run. Mismatched: "
            + ", ".join(m["key"] for m in mism))
    return {"matched": not mism, "mismatches": mism, "reasons": reasons,
            "keys_compared": list(keys),
            "sbc_configuration": sbc_cfg, "run_configuration": run_cfg}


def matched_sbc_kwargs(pack, cfg, *, n_ranks=None, resp_clamp=None):
    """The ``sbc_run`` kwargs that WOULD produce a matched SBC for this run.

    Provided so that nobody has to reverse-engineer the match coordinates, and
    so the COST of a matched SBC is computable rather than rhetorical: one
    replica costs one production-configuration NUTS fit, and SBC needs O(100)
    of them.  On the 29 x 15 x 8 pack at 4 x (1500 + 1000) that is O(100) x
    ~16 h = O(1600) CPU-h, which is above this project's 500 CPU-h sign-off
    threshold and is a PI decision, not a script.
    """
    grid = dict(nhat_edges=np.asarray(pack.nhat_edges, float),
                zf_edges=np.asarray(pack.zf_edges, float),
                zc_edges=np.asarray(pack.zc_edges, float),
                snr_edges=np.asarray(pack.snr_edges, float),
                n_molly_cells=int(pack.n_molly))
    prior = {k: getattr(cfg, k) for k in _PRIOR_KEYS if hasattr(cfg, k)}
    sampler = {k: getattr(cfg, k) for k in _SAMPLER_KEYS if hasattr(cfg, k)}
    sampler["n_ranks"] = int(n_ranks if n_ranks is not None
                             else min(SBC_SAMPLER["n_ranks"],
                                      int(sampler.get("num_samples", 150))))
    return {"grid": grid, "prior": prior, "sampler": sampler,
            "resp_clamp": (resp_clamp if resp_clamp is not None
                           else getattr(cfg, "resp_clamp", None))}


def sbc_run(n_sims=48, *, seed=0, grid=None, prior=None, sampler=None,
            pack_seed=0, verbose=False, resp_clamp="both"):
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
    consts = build_consts(pack, resp_clamp=resp_clamp,
                          allow_unclamped_response=(resp_clamp == "off"))
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
    # THE object this SBC certifies, read off the pack it actually built and
    # the prior/sampler it actually used.  Recorded unconditionally: an SBC
    # that does not state its configuration certifies nothing (decision 8,
    # matched_configuration_sbc).
    meta["configuration"] = _configuration(
        pack, prior=prior, sampler=samp, resp_clamp=resp_clamp,
        reported_names=(names if names else _reported_names(pack)))
    meta["configuration_match_keys"] = list(MATCH_KEYS)
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


def _match_fields(meta, run_config):
    """The configuration-match sub-block + its (single) gating check.

    Kept separate from ``sbc_block`` so the degenerate early-return path gets
    the SAME treatment: a block that produced no replicas must still be unable
    to claim a matched configuration.
    """
    sbc_cfg = sbc_configuration(meta if isinstance(meta, dict) else None)
    m = configuration_match(sbc_cfg, run_config)
    return {
        "configuration": sbc_cfg,
        "run_configuration": run_config,
        "configuration_match": m,
        "configuration_match_note": (
            "RATIFIED 2026-07-29 (decision 8, matched-configuration SBC): an "
            "SBC certifies ONLY the configuration it ran. sbc_configuration_"
            "matches_run is False whenever the SBC's configuration differs "
            "from -- or is not stated alongside -- the run's, and a False "
            "check makes the artifact NOT STAMPABLE. The reduced SBC "
            "(reductions R1-R4) is still REPORTED and still diagnostic; it "
            "just no longer certifies a production run. See "
            "sbc.matched_sbc_kwargs for what a matched one would cost."),
    }, {"sbc_configuration_matches_run": bool(m["matched"])}


def sbc_block(n_sims=48, *, seed=0, n_bins=10, run_config=None, **kw):
    """Block 4.  Runs the reduced SBC and gates on rank uniformity.

    ``run_config`` is the configuration of the run this block is evidence
    ABOUT, as built by ``run_configuration(pack, cfg)``.  Omitting it is not a
    shortcut: with no run to be matched against, the block reports
    ``sbc_configuration_matches_run=False`` and the artifact is not stampable
    (decision 8, ratified 2026-07-29).
    """
    ranks, meta = sbc_run(n_sims, seed=seed, **kw)
    if not ranks:
        mfields, mchecks = _match_fields(meta, run_config)
        out = {"incomplete": ["sbc_produced_no_usable_replicas"],
               "checks": {"sbc_uniform_ok": False, **mchecks}, "meta": meta}
        out.update(mfields)
        return out
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
    mfields, mchecks = _match_fields(meta, run_config)
    out.update(mfields)
    out["checks"] = {
        "sbc_uniform_ok": bool(out["worst_p_bonferroni"] >= SBC_UNIFORM_P_MIN),
        "sbc_enough_replicas": bool(meta["n_sims_used"] >= 20),
        **mchecks,
    }
    return out
