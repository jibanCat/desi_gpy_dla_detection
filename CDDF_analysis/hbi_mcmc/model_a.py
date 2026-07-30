"""model_a.py — Model A: the binned-count NUTS posterior (NumPyro), Q3 spec sec. 2.

Sampled sites (spec section 2 "Posterior"):

  theta_pop (B, Kf) : log f on the fine (N, z) grid, 2-D Gaussian-random-walk
      smoothness prior with SAMPLED scales sigma_N, sigma_z ~ HalfNormal.
      Implemented NON-CENTERED (funnel-avoiding constructive factorization):
          theta[:, 0] = level + slope * (b - b_mid)
                        + sigma_N * double-cumsum(eps_N),   eps_N ~ N(0, 1)
          theta[:, k] = theta[:, 0] + sigma_z * cumsum_k(eps_z),
                                                            eps_z ~ N(0, 1)
      so Delta^2_N theta[:, 0] ~ N(0, sigma_N^2) exactly and
      Delta_z theta ~ N(0, sigma_z^2) exactly (the spec's two penalties,
      factored along the anchor column + z-increments; chosen over the
      improper penalty form for NUTS geometry).
  psi_c (S, M)       : completeness logit offsets ~ N(0, sigma_hat) around the
      Jeffreys-consistent molly surface eta_hat (partial pooling toward the
      molly surface; NEVER independent per-cell conjugate count priors).
  psi_k_delta (2, SR, ZR) : response-coef leading-term perturbations
      ~ N(0, sqrt(resp_fitcov_diag)) per response cell (fail-closed forward
      kernel only — the pack carries no kappa objects by schema).
  t (KK,)            : per-coarse-z log transfer factors ~ N(0, t_sigma[K]).
  FP block (fp_mode="joint"):
      fp_lam_total       ~ Gamma(1/2, eps_rate)  — the SINGLE-JeFFREYS-TOTAL
          prior in log-space (NUTS samples log total; the Gamma(1/2, eps)
          base is the proper eps-regularized Jeffreys lambda^{-1/2}, and the
          prior on the TOTAL is grid-independent BY CONSTRUCTION: refining or
          coarsening the (c, s) cells never changes it — the FIX-3c rule, no
          per-cell prior mass, no phantom FP in empty cells).
      fp_shape_v (C*S,)  ~ N(0, fp_shape_sd); pi = softmax(v);
          lam_fp = total * pi. Cells with zero loa-0 counts get their share
          driven to ~0 by the multinomial part of the likelihood — the DLA
          tier inherits FP only through the smooth shape, expected ~ 0.
      loa-0 likelihood: fp_counts[c, s] ~ Poisson(ell_eff * lam_fp[c, s]).
  Likelihood: counts[c,k,s] ~ Poisson(mu) with mu from forward.fold_mu,
      evaluated inside the jitted model on every draw (differentiable).

Farr N_eff gate (spec section 2): evaluated at build time on the calibration
inputs; hard-fails when sum(molly_n_tot) < 4 * sum(counts) unless the config
explicitly disables it (rung-5 stress runs do, and say so).

Reductions: f(N, z) posterior -> CDDF, dN/dX(z), Omega(z) (arbitrary-constant
units, documented) at thresholds 20.0 AND 20.3, with [19.5, 19.7) masked in
differential reporting; plus the integrated total used by rungs 4/5.

BOTH TIERS, ONE POSTERIOR (2026-07-28).  ``TIERS`` adds the sub-DLA window
[19.5, 20.3) alongside the DLA thresholds, reduced from the SAME f(N, z) draws.
The two tiers share the completeness surface, the FP model, the response kernel
and the pathlength, so they are one coupled inference and any tier ratio is
formed PER DRAW.  ``posterior_summary`` is the paper-facing block:
estimand = POSTERIOR_MEDIAN_CI, point = posterior median, band = credible
interval of the same draws.  ``plugin_map_diagnostic`` computes the mode and is
labelled estimand = PLUGIN_MAP with NO band -- a diagnostic of the mode-median
gap, never a reported point.  Nothing here recenters a band on a point and
nothing combines independently-marginalized intervals.

Synthetic-data scope (Q3): everything here runs on packs from
``pack.synthetic_pack``; the real data pack is integrated later by the lead.
Deviation from spec section 2, recorded: the g(N,z) structural-amplitude term
(one N(1, sigma_g) per N-row) is NOT sampled here — the Q3 component contract
enumerates the sampled sites without it and no synthetic rung exercises it;
g enters the fold as the fixed pack surface.
"""
from __future__ import annotations

import dataclasses
import time
from functools import partial
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value

from CDDF_analysis.hbi_mcmc.diagnostics import summarize_mcmc
from CDDF_analysis.hbi_mcmc.forward import ModelAConsts, build_consts, fold_mu
from CDDF_analysis.hbi_mcmc.pack import ModelAPack
from CDDF_analysis.hbi_mcmc import reporting as RP

__all__ = ["ModelAConfig", "model_a", "run_model_a", "reduce_f_posterior",
           "summarize", "POLICY", "TIERS", "posterior_summary",
           "plugin_map_diagnostic", "ESTIMAND_POSTERIOR", "ESTIMAND_PLUGIN_MAP"]

# spec section 5 convergence policy
POLICY = {"r_hat_max": 1.01, "ess_bulk_min": 400.0, "ess_tail_min": 400.0,
          "n_divergent": 0}

# MCMC.run extra_fields. RECONSTRUCTED 2026-07-28: a concurrent edit replaced
# the literal ("diverging",) at the mcmc.run call with this name and the
# definition was lost in the collision, leaving the module un-importable
# (NameError). Restored to the committed value EXACTLY, so behaviour is
# unchanged; extend the tuple here (never at the call site) if a downstream
# diagnostic needs more fields. "diverging" is REQUIRED -- diagnostics.
# summarize_mcmc reads it for the divergence count that the convergence
# policy gates on.
#
# EXTENDED 2026-07-28 (evidence harness): "num_steps" and "energy" are added
# because tree-depth saturation and E-BFMI are required paper-facing evidence
# and NEITHER can be reconstructed after the run -- the 2026-07-13 rung-9
# artifact is permanently missing both, which is why evidence.py has to mark
# that artifact's convergence block INCOMPLETE and refuse to stamp it.
# Retaining the two fields stores O(n_draws) extra floats per chain and does
# not touch the trajectory, the accepted draws, or any reported number;
# diagnostics.summarize_mcmc reads "diverging" by name and ignores the rest.
EXTRA_FIELDS = ("diverging", "num_steps", "energy")

# differential-reporting mask (spec: [19.5, 19.7) masked)
_MASK_LO, _MASK_HI = 19.5, 19.7
_THRESHOLDS = (20.0, 20.3)

# --- the COUPLED tier definition (2026-07-28) ---------------------------------
# The DLA and sub-DLA tiers are ONE inference, not two.  They share the
# completeness surface (molly), the FP model (loa-0 lam_fp), the response
# kernel and the truth reductions, so they must be read off the SAME posterior
# draws of f(N, z) -- never from two separately-run estimators whose nuisances
# were marginalized independently (that is the MARGINAL_COMBINED failure the
# band-estimand audit retired).  Each tier is a half-open true-N window in dex.
TIERS = {
    "subdla_195_203": (19.5, 20.3),   # sub-DLA tier
    "dla_20p0": (20.0, np.inf),       # DLA tier, >= 20.0
    "dla_20p3": (20.3, np.inf),       # DLA tier, >= 20.3 (the headline)
    "all_195_up": (19.5, np.inf),     # both tiers together (coupling check)
    # --- PI DECISION 1 (2026-07-29): the PRIMARY reporting window ------------
    # The only window in which an Omega_HI may be emitted at all. Its floor is
    # the sub-DLA runner's NONIDENT_EDGE and its ceiling exists because the
    # forward response is EXTRAPOLATED above ~21.6 (finding D2). Every other
    # tier above is retained for continuity with the rung ladder and the
    # coupling checks; none of them may carry an Omega (see omega_decision).
    "report_197_216": (RP.NONIDENT_EDGE, RP.RESPONSE_ANCHOR_CEILING),
}

# metadata['estimand'] vocabulary values this module can legitimately produce.
# POSTERIOR_MEDIAN_CI: point = posterior median, band = credible interval, from
# the SAME joint posterior draws.  PLUGIN_MAP: a labelled DIAGNOSTIC optimum
# with no band of its own.  Anything else is not producible here.
ESTIMAND_POSTERIOR = "POSTERIOR_MEDIAN_CI"
ESTIMAND_PLUGIN_MAP = "PLUGIN_MAP"


@dataclasses.dataclass
class ModelAConfig:
    """Run configuration for Model A (defaults = spec section 5; tests shrink)."""

    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    chain_method: str = "vectorized"  # numpyro 'sequential' recompiles per chain here
    target_accept: float = 0.9
    max_tree_depth: int = 10
    # block-dense mass over the population anchor block (level/slope/eps_N are
    # densely coupled — they trade off against every N-bin; diagonal mass
    # leaves a stiff ridge that inflates NUTS tree depth ~2x and divergences ~3x,
    # measured on the small synthetic pack).
    dense_mass_anchor_block: bool = True
    data_informed_init: bool = True  # init_to_value at a crude count-scale point
    seed: int = 0
    # priors
    sigma_N_scale: float = 0.5    # HalfNormal scale of the Delta^2_N RW sd
    sigma_z_scale: float = 0.5    # HalfNormal scale of the Delta_z RW sd
    level_scale: float = 4.0      # weak N(0, .) on the anchor level
    slope_scale: float = 2.0      # weak N(0, .) on the anchor N-slope
    # FP block
    fp_mode: str = "joint"        # "joint" | "off"
    fp_eps_rate: float = 1e-6     # proper-Jeffreys regularizer (posterior-negligible)
    fp_shape_sd: float = 3.0      # logistic-normal shape prior sd
    # gates
    enforce_farr_gate: bool = True
    farr_min_ratio: float = 4.0
    # response covariate-range guard (finding D2): "both" | "hi" | "off".
    # "off" reproduces the pre-2026-07-28 forward model and is DIAGNOSTIC ONLY.
    resp_clamp: str = "both"


# --- the NumPyro model --------------------------------------------------------------

def model_a(consts: ModelAConsts, counts=None, fp_counts=None, *,
            fp_mode="joint", fp_eps_rate=1e-6, fp_shape_sd=3.0,
            sigma_N_scale=0.5, sigma_z_scale=0.5,
            level_scale=4.0, slope_scale=2.0):
    """Model A generative program (see module docstring for the site map)."""
    B, Kf = consts.n_b, consts.n_k
    C, S = consts.n_c, consts.n_s

    # -- population: non-centered 2-D RW on theta_pop = log f
    sigma_N = numpyro.sample("sigma_N", dist.HalfNormal(sigma_N_scale))
    sigma_z = numpyro.sample("sigma_z", dist.HalfNormal(sigma_z_scale))
    level = numpyro.sample("theta_level", dist.Normal(0.0, level_scale))
    slope = numpyro.sample("theta_slope", dist.Normal(0.0, slope_scale))
    eps_N = numpyro.sample(
        "eps_N", dist.Normal(0.0, 1.0).expand([max(B - 2, 0)]).to_event(1))
    eps_z = numpyro.sample(
        "eps_z", dist.Normal(0.0, 1.0).expand([B, max(Kf - 1, 0)]).to_event(2))
    b_idx = jnp.arange(B) - 0.5 * (B - 1)
    curv = jnp.cumsum(jnp.cumsum(jnp.concatenate([jnp.zeros(2), eps_N])))[:B]
    theta_col0 = level + slope * b_idx + sigma_N * curv
    theta = theta_col0[:, None] + jnp.concatenate(
        [jnp.zeros((B, 1)), sigma_z * jnp.cumsum(eps_z, axis=1)], axis=1)
    theta = numpyro.deterministic("theta_pop", theta)
    numpyro.deterministic("f", jnp.exp(theta))

    # -- calibration nuisances
    psi_c = numpyro.sample(
        "psi_c", dist.Normal(0.0, consts.sigma_hat).to_event(2))
    psi_k_delta = numpyro.sample(
        "psi_k_delta", dist.Normal(0.0, consts.fitcov_sd).to_event(3))
    t = numpyro.sample("t", dist.Normal(0.0, consts.t_sigma).to_event(1))

    # -- FP block: single-Jeffreys TOTAL (log-space) + logistic-normal shape
    if fp_mode == "joint":
        lam_total = numpyro.sample(
            "fp_lam_total", dist.Gamma(0.5, fp_eps_rate))
        # zero-sum shape logits: softmax is invariant to a constant shift, so a
        # plain Normal leaves a soft ridge that wrecks NUTS mixing; ZeroSumNormal
        # removes that direction exactly (the shape prior stays logistic-normal).
        v = numpyro.sample(
            "fp_shape_v", dist.ZeroSumNormal(fp_shape_sd, event_shape=(C * S,)))
        pi = jax.nn.softmax(v)
        lam_fp = numpyro.deterministic(
            "lam_fp", (lam_total * pi).reshape(C, S))
        numpyro.sample(
            "fp_counts",
            dist.Poisson(consts.fp_ell_eff * lam_fp).to_event(2),
            obs=fp_counts)
    elif fp_mode == "off":
        lam_fp = jnp.zeros((C, S))
    else:
        raise ValueError(f"unknown fp_mode {fp_mode!r}")

    # -- forward fold + likelihood (inside the jitted model, per draw)
    mu = fold_mu(theta, psi_c, psi_k_delta, t, lam_fp, consts)
    # zero-dX strata are structurally unobserved (validator guarantees zero
    # counts + zero fp_E_alloc there): mask them out of the likelihood so
    # Poisson(rate=0) can never produce -inf/NaN gradients.
    # Per-element masking: the mask must act on BATCH dims (masking an
    # event-collapsed scalar log-prob elementwise silently rescales the
    # likelihood by the cell count — found the hard way). Observed batch
    # log-probs are summed into the joint by numpyro.
    obs_mask = jnp.broadcast_to(jnp.asarray(consts.dX > 0)[None, :, :], mu.shape)
    with numpyro.handlers.mask(mask=obs_mask):
        numpyro.sample("counts", dist.Poisson(jnp.clip(mu, 1e-300, None)),
                       obs=counts)


# --- reductions -----------------------------------------------------------------------

def reduce_f_posterior(f_draws, pack: ModelAPack):
    """f(N, z) posterior draws -> CDDF / dN/dX / Omega reductions (numpy).

    Parameters
    ----------
    f_draws : (n_draws, B, Kf)
    pack    : the data pack (for grids).

    Returns
    -------
    dict:
      f                 : the raw draws (n_draws, B, Kf)
      cddf_masked       : draws with the [19.5, 19.7) bins set to nan
                          (differential-reporting mask)
      dndx_20p0/20p3    : (n_draws, Kf)  sum_{b >= thr} f dN
      dndx_20p0/20p3_coarse : (n_draws, KK) pathlength-weighted coarse means
      omega_20p0/20p3   : (n_draws, Kf)  sum_{b >= thr} 10^(N_b - 21) f dN
                          (ARBITRARY-CONSTANT units: the physical
                          H0 m_H mu / (c rho_crit) factor is applied at
                          reporting time, outside this synthetic scope)
      integrated_total  : (n_draws,) sum_{b,k} f dN (rung 4/5 scalar)
      n_mask_bins       : how many N-bins the differential mask removed
      reported_mask     : (B,) bool, the UNPADDED (reported) true-N support
      dndx_<tier>       : (n_draws, Kf) for every window in ``TIERS``
      dndx_<tier>_coarse: (n_draws, KK) pathlength-weighted coarse-z means
      dndx_<tier>_allz  : (n_draws,) pathlength-weighted all-z mean
      omega_<tier>[_coarse|_allz] : same shapes, the 10^(N-21)-weighted integral

    Every tier reduction is computed from the SAME ``f_draws``, so the tiers
    are correlated draw-by-draw and any tier ratio / difference may be formed
    per draw (never by combining independently-marginalized intervals).
    """
    f_draws = np.asarray(f_draws)
    ntrue = np.asarray(pack.ntrue_edges, float)
    Nc = 0.5 * (ntrue[:-1] + ntrue[1:])
    dN = np.diff(ntrue)
    kz = np.asarray(pack.kz_to_K)
    KK = pack.n_kk
    dX_k = np.asarray(pack.dX, float).sum(axis=1)  # (Kf,) summed over strata

    mask = (Nc >= _MASK_LO - 1e-9) & (Nc < _MASK_HI - 1e-9)

    # BASIS PAD (schema v1.1): true-N bins BELOW the observed grid floor are
    # UNREPORTED support — they exist so the fold can carry the up-scatter of
    # sub-floor systems into the lowest observed bins (finding D1). They are
    # excluded from every reported reduction. No-op on unpadded packs.
    reported = Nc >= float(np.asarray(pack.nhat_edges, float)[0]) - 1e-9

    # the differential CDDF must not report the pad either: those bins are
    # inferred against a completeness convention (the constant-extrapolation of
    # the molly's lowest cell) that is a stated systematic, not a measurement.
    cddf_masked = f_draws.copy()
    cddf_masked[:, mask | (~reported), :] = np.nan

    def _coarse(per_k):
        return np.stack(
            [(per_k[:, kz == q] * dX_k[kz == q][None, :]).sum(axis=1)
             / dX_k[kz == q].sum() for q in range(KK)], axis=1)

    def _allz(per_k):
        return (per_k * dX_k[None, :]).sum(axis=1) / dX_k.sum()

    out = {
        "f": f_draws,
        "cddf_masked": cddf_masked,
        "n_mask_bins": int(mask.sum()),
        "n_pad_bins": int((~reported).sum()),
        "reported_mask": reported,
        "integrated_total": (f_draws[:, reported, :]
                             * dN[None, reported, None]).sum(axis=(1, 2)),
    }
    # --- window weights (PI decision 3 made the basis coarser than the grid) --
    # Every integrated reduction below weights basis bin b by the DEX OF b THAT
    # LIES INSIDE THE WINDOW, not by dN_b under a centre-in-window test.  On any
    # basis whose edges align with the window edges -- i.e. every 0.1-dex pack
    # ever extracted -- the two are IDENTICAL (pinned by
    # tests/test_adopted_reporting.py::test_overlap_weights_are_bit_identical_
    # to_centre_selection_on_the_0p1dex_basis).  On the adopted 0.2-dex basis the
    # overlap weight is the only correct choice: 21.6 cannot be a 0.2-dex basis
    # edge if 19.7 is (their separation, 1.9 dex, is an ODD multiple of 0.1), so
    # the top window bin is straddled and a centre test would silently drop or
    # silently include a whole 0.2-dex bin.  Splitting it by overlap is EXACT
    # under the adopted merging convention ("f is constant across the merged
    # bin") -- it introduces no new assumption.
    def _wts(lo, hi):
        w = RP.window_overlap_weights(ntrue, lo, min(hi, ntrue[-1]))
        return np.where(reported, w, 0.0)

    # legacy threshold keys (analyze_rung9 and the rung ladder consume these)
    for thr in _THRESHOLDS:
        w = _wts(thr, np.inf)
        tag = f"{thr:.1f}".replace(".", "p")
        dndx = (f_draws * w[None, :, None]).sum(axis=1)                # (n, Kf)
        omega = (f_draws * (10.0 ** (Nc - 21.0))[None, :, None]
                 * w[None, :, None]).sum(axis=1)
        out[f"dndx_{tag}"] = dndx
        out[f"dndx_{tag}_coarse"] = _coarse(dndx)
        # NOTE these omega_* keys are OPEN-TOPPED and are therefore NOT
        # emittable as reported values (PI decision 1).  They remain here
        # because analyze_rung9 / evidence.reported_quantities consume them as
        # CONVERGENCE + rung-ladder diagnostics on mocks.  The refusal is
        # enforced where values are reported: posterior_summary.
        out[f"omega_{tag}"] = omega
    # the coupled DLA + sub-DLA tier windows, all off the same draws
    omega_decisions = {}
    for tier, (lo, hi) in TIERS.items():
        w = _wts(lo, hi)
        if not np.any(w > 0):
            continue
        dndx = (f_draws * w[None, :, None]).sum(axis=1)
        omega = (f_draws * (10.0 ** (Nc - 21.0))[None, :, None]
                 * w[None, :, None]).sum(axis=1)
        out[f"dndx_{tier}"] = dndx
        out[f"dndx_{tier}_coarse"] = _coarse(dndx)
        out[f"dndx_{tier}_allz"] = _allz(dndx)
        out[f"omega_{tier}"] = omega
        out[f"omega_{tier}_coarse"] = _coarse(omega)
        out[f"omega_{tier}_allz"] = _allz(omega)
        out[f"n_bins_{tier}"] = int(np.sum(w > 0))
        out[f"window_weights_{tier}"] = w
        omega_decisions[tier] = RP.omega_decision(lo, hi)
        # DECISION-4 GUARD, fail-closed: a tier inside the primary reporting
        # window may not draw on ANY basis bin below the reporting floor 19.7
        # (the schema-v1.1 pad + the non-identifiable [19.5,19.7) edge are
        # LATENT NUISANCE support).  Checked on the weights actually used.
        if omega_decisions[tier]["emit"]:
            RP.assert_no_subwindow_bins(
                ntrue, w, where=f"reduce_f_posterior tier {tier!r}")
    out["omega_decisions"] = omega_decisions
    return out


# --- the paper-facing posterior summary ------------------------------------------------

_QUANTILES = (2.5, 16.0, 50.0, 84.0, 97.5)


def _q(draws):
    """point + band from ONE set of draws. point IS q50 -- by construction, not
    by a recentering step. There is no plug-in value anywhere in this dict."""
    d = np.asarray(draws, float)
    qs = np.percentile(d, _QUANTILES)
    return {
        "point_q50": float(qs[2]),      # THE point estimate
        "q025": float(qs[0]), "q16": float(qs[1]),
        "q84": float(qs[3]), "q975": float(qs[4]),
        "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
        "n_draws": int(d.size),
    }


def posterior_summary(red, pack=None):
    """Reductions -> the paper-facing {point, band} block for BOTH tiers.

    estimand = POSTERIOR_MEDIAN_CI: for every quantity the point is the
    posterior MEDIAN of the same draws whose 16/84 and 2.5/97.5 percentiles
    form the bands.  Point and band are therefore the SAME estimand by
    construction; no shift, no recentering, no independent combination of
    marginals is applied anywhere in this function.  (The retired
    ``recenter_band_on_point`` machinery in CDDF_analysis/hbi/ slid an MC cloud
    onto a plug-in optimum; nothing of that kind exists here.)
    """
    out = {"estimand": ESTIMAND_POSTERIOR, "tiers": {},
           "reporting_window_logN": list(RP.REPORTING_WINDOW),
           "reporting_window_label": RP.REPORTING_WINDOW_LABEL,
           "omega_rule": RP.OMEGA_RULE,
           "primary_reporting_tier": "report_197_216"}
    for tier in TIERS:
        if f"dndx_{tier}_allz" not in red:
            continue
        lo, hi = float(TIERS[tier][0]), float(TIERS[tier][1])
        dec = (red.get("omega_decisions") or {}).get(tier) or RP.omega_decision(lo, hi)
        blk = {
            "window_logN": [lo, hi],
            "n_bins": int(red[f"n_bins_{tier}"]),
            "in_primary_reporting_window": bool(dec["emit"]),
            "dndx_allz": _q(red[f"dndx_{tier}_allz"]),
            "dndx_coarse_z": [_q(red[f"dndx_{tier}_coarse"][:, q])
                              for q in range(red[f"dndx_{tier}_coarse"].shape[1])],
        }
        # --- PI DECISION 1: Omega_HI is emitted ONLY inside [19.7, 21.6] -----
        # dN/dX is a LINE DENSITY and is unaffected by this ruling; Omega_HI is
        # an N-WEIGHTED MASS whose integral is dominated by the top of the
        # window, which is exactly where the response is extrapolated. An
        # unqualified/open-topped Omega is REFUSED with its reason attached, in
        # the schema — there is no tail extrapolation here, by design.
        if dec["emit"]:
            blk["omega_allz"] = _q(red[f"omega_{tier}_allz"])
            blk["omega_coarse_z"] = [
                _q(red[f"omega_{tier}_coarse"][:, q])
                for q in range(red[f"omega_{tier}_coarse"].shape[1])]
            blk["omega_label"] = dec["label"]
            blk["omega_window_logN"] = dec["window_logN"]
        else:
            blk["omega_allz"] = None
            blk["omega_coarse_z"] = None
            blk["omega_REFUSED"] = dec
        out["tiers"][tier] = blk
    # the tier RATIO, formed PER DRAW (the whole point of one coupled posterior)
    if "dndx_subdla_195_203_allz" in red and "dndx_dla_20p3_allz" in red:
        r = (np.asarray(red["dndx_subdla_195_203_allz"], float)
             / np.maximum(np.asarray(red["dndx_dla_20p3_allz"], float), 1e-300))
        out["subdla_over_dla_dndx_perdraw"] = _q(r)
    if pack is not None:
        out["zc_edges"] = np.asarray(pack.zc_edges, float).tolist()
    return out


# --- plug-in MAP: DIAGNOSTIC ONLY ------------------------------------------------------

def plugin_map_diagnostic(pack, cfg=None, *, num_steps=2000, lr=0.05, seed=0):
    """MAP (AutoDelta + SVI) of the SAME model. DIAGNOSTIC ONLY.

    Returned under estimand=PLUGIN_MAP with NO band.  It exists to answer "how
    far is the mode from the posterior median?" -- the plug-in-vs-MC estimand
    gap that the band-estimand audit measured at 1.0-19.0 band half-widths in
    the legacy HBI arm.  It is NEVER the reported point and its value must
    never be paired with a credible interval computed from anything else.
    """
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoDelta
    from numpyro.handlers import seed as _seed, substitute as _sub, trace as _trace

    cfg = cfg or ModelAConfig()
    consts = build_consts(pack, resp_clamp=cfg.resp_clamp,
                          allow_unclamped_response=(cfg.resp_clamp == "off"))
    model = partial(
        model_a, fp_mode=cfg.fp_mode, fp_eps_rate=cfg.fp_eps_rate,
        fp_shape_sd=cfg.fp_shape_sd, sigma_N_scale=cfg.sigma_N_scale,
        sigma_z_scale=cfg.sigma_z_scale, level_scale=cfg.level_scale,
        slope_scale=cfg.slope_scale)
    args = (consts, jnp.asarray(pack.counts),
            jnp.asarray(pack.fp_counts) if cfg.fp_mode == "joint" else None)
    guide = AutoDelta(model, init_loc_fn=init_to_value(
        values=_data_informed_init(pack, consts, cfg)))
    svi = SVI(model, guide, numpyro.optim.Adam(lr), Trace_ELBO())
    res = svi.run(jax.random.PRNGKey(seed), num_steps, *args, progress_bar=False)
    vals = guide.median(res.params)
    tr = _trace(_sub(_seed(model, jax.random.PRNGKey(seed)), vals)).get_trace(*args)
    f_map = np.asarray(tr["f"]["value"])[None, ...]      # (1, B, Kf)
    red = reduce_f_posterior(f_map, pack)
    point = {"estimand": ESTIMAND_PLUGIN_MAP, "band": None, "tiers": {}}
    for tier in TIERS:
        if f"dndx_{tier}_allz" not in red:
            continue
        dec = (red.get("omega_decisions") or {}).get(tier, {})
        point["tiers"][tier] = {
            "dndx_allz": float(red[f"dndx_{tier}_allz"][0]),
            # PI decision 1: no unqualified Omega, diagnostic or not
            "omega_allz": (float(red[f"omega_{tier}_allz"][0])
                           if dec.get("emit") else None),
            "omega_label": dec.get("label"),
            "omega_REFUSED": (None if dec.get("emit") else dec.get("reason")),
        }
    point["svi_final_loss"] = float(np.asarray(res.losses)[-1])
    point["svi_num_steps"] = int(num_steps)
    return point


# --- diagnostics wrapper ---------------------------------------------------------------

def summarize(mcmc, runtime=None, policy=None):
    """summarize_mcmc + the spec section 5 policy flags (which gates FIRED)."""
    policy = dict(POLICY if policy is None else policy)
    s = summarize_mcmc(mcmc, runtime=runtime)
    flags = {
        "flag_r_hat": bool(s["r_hat_max"] > policy["r_hat_max"]),
        "flag_ess_bulk": bool(s["ess_bulk_min"] < policy["ess_bulk_min"]),
        "flag_ess_tail": bool(s["ess_tail_min"] < policy["ess_tail_min"]),
        "flag_divergent": bool(s["n_divergent"] > policy["n_divergent"]),
    }
    s.update(flags)
    s["flags_fired"] = sorted(k for k, v in flags.items() if v)
    s["policy_pass"] = not s["flags_fired"]
    return s


# --- runner ------------------------------------------------------------------------------

def _data_informed_init(pack: ModelAPack, consts: ModelAConsts, cfg: ModelAConfig):
    """Crude but safe init point (an INIT only, never a prior): flat log f at the
    count-implied level, all nuisances at their prior centers, FP total at the
    loa-0 point estimate. Keeps every chain out of the exp-cliff corners that
    random unconstrained inits occasionally land in."""
    B, Kf, C, S = consts.n_b, consts.n_k, consts.n_c, consts.n_s
    # the REPORTED support only (padded sub-floor bins would deflate the level)
    _ne = np.asarray(pack.ntrue_edges, float)
    _rep = 0.5 * (_ne[:-1] + _ne[1:]) >= float(
        np.asarray(pack.nhat_edges, float)[0]) - 1e-9
    dN_tot = float(np.diff(_ne)[_rep].sum())
    level = float(np.log(
        max(float(np.asarray(pack.counts).sum()), 1.0)
        / (0.7 * float(np.asarray(pack.dX).sum()) * dN_tot)))
    vals = {
        "sigma_N": jnp.asarray(0.1), "sigma_z": jnp.asarray(0.1),
        "theta_level": jnp.asarray(level), "theta_slope": jnp.asarray(0.0),
        "eps_N": jnp.zeros(max(B - 2, 0)),
        "eps_z": jnp.zeros((B, max(Kf - 1, 0))),
        "psi_c": jnp.zeros((S, consts.n_molly)),
        "psi_k_delta": jnp.zeros((2, consts.n_sr, consts.n_zr)),
        "t": jnp.zeros(consts.n_kk),
    }
    if cfg.fp_mode == "joint":
        vals["fp_lam_total"] = jnp.asarray(
            max(float(np.asarray(pack.fp_counts).sum()) / consts.fp_ell_eff, 1.0))
        vals["fp_shape_v"] = jnp.zeros(C * S)
    return vals


def run_model_a(pack: ModelAPack, cfg: Optional[ModelAConfig] = None):
    """Build consts, run the Farr gate, sample Model A, reduce the posterior.

    Returns
    -------
    (mcmc, reductions) : the finished numpyro MCMC object and the reductions
    dict from ``reduce_f_posterior`` extended with the ``summarize`` block
    (key "diagnostics"), the Farr ratio, and the sampled-site posteriors for
    t / fp_lam_total (means and sds).
    """
    cfg = cfg or ModelAConfig()
    consts = build_consts(pack, resp_clamp=cfg.resp_clamp,
                          allow_unclamped_response=(cfg.resp_clamp == "off"))

    # Farr N_eff gate on the calibration inputs (build time, spec section 2)
    n_cal = float(np.asarray(pack.molly_n_tot).sum())
    n_obs = float(np.asarray(pack.counts).sum())
    farr_ratio = n_cal / max(n_obs, 1.0)
    if cfg.enforce_farr_gate and farr_ratio < cfg.farr_min_ratio:
        raise RuntimeError(
            f"Farr N_eff gate FAILED: calibration counts {n_cal:.0f} < "
            f"{cfg.farr_min_ratio} x observed counts {n_obs:.0f} "
            f"(ratio {farr_ratio:.2f}). Enlarge the calibration set or "
            f"explicitly disable the gate for a stress run.")

    model = partial(
        model_a, fp_mode=cfg.fp_mode, fp_eps_rate=cfg.fp_eps_rate,
        fp_shape_sd=cfg.fp_shape_sd, sigma_N_scale=cfg.sigma_N_scale,
        sigma_z_scale=cfg.sigma_z_scale, level_scale=cfg.level_scale,
        slope_scale=cfg.slope_scale)
    nuts_kw = {}
    if cfg.dense_mass_anchor_block:
        nuts_kw["dense_mass"] = [("theta_level", "theta_slope", "eps_N")]
    if cfg.data_informed_init:
        nuts_kw["init_strategy"] = init_to_value(
            values=_data_informed_init(pack, consts, cfg))
    kernel = NUTS(model, target_accept_prob=cfg.target_accept,
                  max_tree_depth=cfg.max_tree_depth, **nuts_kw)
    mcmc = MCMC(kernel, num_warmup=cfg.num_warmup, num_samples=cfg.num_samples,
                num_chains=cfg.num_chains, chain_method=cfg.chain_method,
                progress_bar=False)
    t0 = time.time()
    mcmc.run(
        jax.random.PRNGKey(cfg.seed),
        consts,
        jnp.asarray(pack.counts),
        jnp.asarray(pack.fp_counts) if cfg.fp_mode == "joint" else None,
        extra_fields=EXTRA_FIELDS,
    )
    samples = mcmc.get_samples()
    f_draws = np.asarray(samples["f"])  # forces the async device work; time AFTER
    runtime = time.time() - t0

    red = reduce_f_posterior(f_draws, pack)
    red["farr_ratio"] = farr_ratio
    red["diagnostics"] = summarize(mcmc, runtime=runtime)
    t_draws = np.asarray(samples["t"])
    red["t_mean"] = t_draws.mean(axis=0)
    red["t_sd"] = t_draws.std(axis=0, ddof=1)
    if cfg.fp_mode == "joint":
        lam_tot = np.asarray(samples["fp_lam_total"])
        red["fp_lam_total_mean"] = float(lam_tot.mean())
        red["fp_lam_total_sd"] = float(lam_tot.std(ddof=1))
    return mcmc, red
