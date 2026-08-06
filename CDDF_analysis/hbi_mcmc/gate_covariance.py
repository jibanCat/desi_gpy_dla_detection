"""gate_covariance.py — calibration-predictive gate statistics (Phase B, 2026-08-06).

Implements the FROZEN statistical-analysis specification
(``docs/PHASEB_STATS_SPEC_2026-08-06.md``). Three diagnostic layers, never
interchangeable:

  A. CONDITIONAL implementation gate — lives in ``forward_selftest``
     (``ratio_tables`` / ``poisson_z``; variance = predicted mean; the
     historical chi2/dof <= 3 threshold). Tests that the fold implements the
     REALIZED calibration artifact correctly. It is not a predictive test.
  B. CALIBRATION-PREDICTIVE gate — THIS MODULE. Propagates the finite loa-0
     calibration sample (89 in-support events) through normalization and the
     production fold into the diagnostic vector. The primary statistic is a
     3-group observed-N-hat Mahalanobis form with a simulation-calibrated
     null; NO scalar threshold is ratified for it.
  C. TRANSPORT stress test — ``transport_stress_stats``. Cross-mock loa-0
     transport mismatch, reported as an UNCALIBRATED systematic; it is never
     absorbed into the Layer-B covariance (PI ruling 10).

Design rules (PI rulings 5–7):
  * Calibration uncertainty is NEVER per-bin error bars added in quadrature:
    every resample re-normalizes (lam* = n0*/ell) and re-folds through the
    PRODUCTION FP fold (``forward.fold_mu_fp`` — one authoritative
    implementation, no re-typed formula), so cross-bin covariance, the shared
    amplitude mode, and the imposed-E structure propagate exactly.
  * Covariance estimation (E_cov), null calibration (E_null) and the observed
    evaluation use DISJOINT random streams (frozen seeds in the spec).
  * The 3x3 covariance is inverted exactly; the frozen fallback (cond > 1e6)
    reports 1-dim standardized residuals instead. No shrinkage, no
    pseudoinverse, no data-dependent mode selection.
  * Target–calibration independence status is an INPUT (measured in the spec:
    2LPT-0 shares its skewer set with loa-0 — |2 Cov|/Var_cal <= ~1.2%,
    neglected, conservative for failure claims; London-0/Saclay-0 disjoint).

Axis convention: all diagnostic vectors are on the pack's observed grid
(c, k, s) with live cells dX > 0; group aggregation sums over k and s and
over the window bins of each N-hat group. Counts are dimensionless expected/
observed catalogue counts.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Sequence, Tuple

import numpy as np

from CDDF_analysis.hbi_mcmc.pack import ModelAPack

#: the FROZEN primary observed-N-hat groups (spec section 3): the external
#: physical DLA threshold and the Phase-A measured/weakly-measured kernel
#: boundary. Changing them demotes every downstream result to exploratory.
PRIMARY_GROUP_EDGES: Tuple[Tuple[float, float], ...] = (
    (19.7, 20.3), (20.3, 21.0), (21.0, 21.6))

#: frozen ensemble sizes and seeds (spec section 4)
N_COV_DRAWS = 2000
N_NULL_DRAWS = 2000
SEED_COV = 41001
SEED_NULL = 43001
SEED_SIGHTLINE_CHECK = 42001

#: frozen conditioning fallback (spec section 3)
MAX_CONDITION_NUMBER = 1e6


@dataclasses.dataclass(frozen=True)
class GateCovariance:
    """A gate covariance with its full construction provenance.

    Every field the PI ruling requires a covariance report to expose.
    """

    matrix: np.ndarray            # (G, G)
    axis_labels: Tuple[str, ...]  # one label per component
    group_edges: Tuple[Tuple[float, float], ...]
    units: str                    # "catalogue counts"
    method: str                   # construction description
    raw_dim: int                  # diagnostic-vector dimension before reduction
    algebraic_rank: int
    effective_rank: float         # exp(entropy of eigenvalue distribution)
    eigenvalues: np.ndarray       # descending
    condition_number: float
    regularization: str           # "none (exact inverse)" or the fallback note
    calibration_events_total: int
    calibration_events_per_group: Tuple[int, ...]
    n_resamples: int
    seed: int
    mc_error_rel: np.ndarray      # (G, G) jackknife relative MC error
    survey_variance: np.ndarray   # (G,) Poisson part, for decomposition reports
    calibration_variance: np.ndarray  # (G,) marginal calibration part

    def report(self) -> dict:
        """JSON-ready covariance diagnostics block (closure-table schema)."""
        return dict(
            axis_labels=list(self.axis_labels),
            group_edges=[list(e) for e in self.group_edges],
            units=self.units, method=self.method,
            raw_dim=self.raw_dim, algebraic_rank=self.algebraic_rank,
            effective_rank=float(self.effective_rank),
            eigenvalues=[float(v) for v in self.eigenvalues],
            condition_number=float(self.condition_number),
            regularization=self.regularization,
            calibration_events_total=self.calibration_events_total,
            calibration_events_per_group=list(self.calibration_events_per_group),
            n_resamples=self.n_resamples, seed=self.seed,
            matrix=self.matrix.tolist(),
            mc_error_rel_max=float(np.max(self.mc_error_rel)),
            survey_variance=[float(v) for v in self.survey_variance],
            calibration_variance=[float(v) for v in self.calibration_variance],
        )


def group_aggregator(pack: ModelAPack,
                     group_edges: Sequence[Tuple[float, float]]
                     = PRIMARY_GROUP_EDGES) -> np.ndarray:
    """(G, C) 0/1 aggregation matrix over observed N-hat bins.

    FAIL-LOUD: every group edge must coincide with a pack bin edge — a group
    may never truncate a bin (the recurring one-sided-support bug class).
    """
    ne = np.asarray(pack.nhat_edges, float)
    A = np.zeros((len(group_edges), pack.n_c))
    for g, (lo, hi) in enumerate(group_edges):
        if not (np.any(np.isclose(ne, lo, atol=1e-9))
                and np.any(np.isclose(ne, hi, atol=1e-9))):
            raise ValueError(
                f"group_aggregator: group edge ({lo}, {hi}) does not align "
                f"with the pack's nhat_edges — refusing to truncate a bin.")
        A[g] = (ne[:-1] >= lo - 1e-9) & (ne[1:] <= hi + 1e-9)
    return A


def _fold_parts(pack: ModelAPack, *, resp_clamp: str = "both"):
    """(mu_sig, fp_fold, live) with fp_fold(n0) the PRODUCTION FP fold as a
    function of a calibration-count array — via forward.fold_mu_fp, never a
    re-typed formula."""
    import jax.numpy as jnp
    from CDDF_analysis.hbi_mcmc.forward import build_consts, fold_mu_fp
    from CDDF_analysis.hbi_mcmc import forward_selftest as FS

    consts = build_consts(pack, resp_clamp=resp_clamp)
    base = FS.selftest(pack, resp_clamp=resp_clamp)
    mu_sig = np.asarray(base["mu"] - base["mu_fp"])       # (C, Kf, S)
    live = np.broadcast_to(
        (np.asarray(pack.dX, float) > 0)[None, :, :], mu_sig.shape)
    ell = float(pack.fp_ell_eff)
    KK = consts.n_kk

    def fp_fold(n0: np.ndarray) -> np.ndarray:
        lam = jnp.asarray(np.asarray(n0, float) / ell)
        return np.asarray(fold_mu_fp(jnp.zeros(KK), lam, consts))

    return mu_sig, fp_fold, live


def _group_vector(x: np.ndarray, A: np.ndarray, live: np.ndarray) -> np.ndarray:
    """Aggregate a (C, Kf, S) cell array to the (G,) group vector."""
    xk = np.where(live, x, 0.0).sum(axis=(1, 2))          # (C,)
    return A @ xk


def estimate_covariance(pack: ModelAPack, *,
                        group_edges: Sequence[Tuple[float, float]]
                        = PRIMARY_GROUP_EDGES,
                        n_draws: int = N_COV_DRAWS,
                        seed: int = SEED_COV,
                        resp_clamp: str = "both") -> GateCovariance:
    """E_cov: the frozen Layer-B covariance of d = G(y*) − G(mu(n0*)).

    y* ~ Poisson(mu(n0_obs)) carries the survey noise; n0* ~ Poisson(n0_obs)
    carries the calibration sampling noise through normalization and the
    production fold. Cross-covariance target↔calibration is not added (see
    module docstring / spec section 6).
    """
    A = group_aggregator(pack, group_edges)
    mu_sig, fp_fold, live = _fold_parts(pack, resp_clamp=resp_clamp)
    n0 = np.asarray(pack.fp_counts, float)
    mu_obs = mu_sig + fp_fold(n0)
    rng = np.random.default_rng(seed)

    g_mu = _group_vector(mu_obs, A, live)
    draws = np.empty((n_draws, A.shape[0]))
    cal_draws = np.empty_like(draws)
    for r in range(n_draws):
        y_star = rng.poisson(np.clip(mu_obs, 0, None))
        n0_star = rng.poisson(n0)
        mu_star = mu_sig + fp_fold(n0_star)
        draws[r] = _group_vector(y_star, A, live) - _group_vector(mu_star, A, live)
        cal_draws[r] = _group_vector(mu_star, A, live)

    C = np.cov(draws, rowvar=False)
    evals = np.linalg.eigvalsh(C)[::-1]
    pos = np.clip(evals, 1e-300, None)
    p = pos / pos.sum()
    eff_rank = float(np.exp(-(p * np.log(p)).sum()))
    # jackknife-over-blocks MC error on the covariance elements
    nb = 20
    blocks = np.array_split(np.arange(n_draws), nb)
    C_jack = np.stack([
        np.cov(draws[np.concatenate([b for j, b in enumerate(blocks) if j != i])],
               rowvar=False) for i in range(nb)])
    mc_rel = (np.std(C_jack, axis=0) * np.sqrt(nb - 1)
              / np.maximum(np.abs(C), 1e-300))

    n_events_group = tuple(
        int(v) for v in (A @ np.asarray(pack.fp_counts, float).sum(axis=1)))
    return GateCovariance(
        matrix=C,
        axis_labels=tuple(f"nhat[{lo},{hi})" for lo, hi in group_edges),
        group_edges=tuple(tuple(e) for e in group_edges),
        units="catalogue counts",
        method=("parametric bootstrap through the production fold: "
                "y*~Poisson(mu(n0_obs)); n0*~Poisson(n0_obs) -> lam*=n0*/ell "
                "-> forward.fold_mu_fp; d = G(y*) - G(mu(n0*))"),
        raw_dim=int(live.sum()),
        algebraic_rank=int(np.linalg.matrix_rank(C)),
        effective_rank=eff_rank,
        eigenvalues=evals,
        condition_number=float(evals[0] / max(evals[-1], 1e-300)),
        regularization="none (exact inverse; frozen fallback at cond>1e6)",
        calibration_events_total=int(np.asarray(pack.fp_counts).sum()),
        calibration_events_per_group=n_events_group,
        n_resamples=n_draws, seed=seed,
        mc_error_rel=mc_rel,
        survey_variance=g_mu,
        calibration_variance=np.var(cal_draws, axis=0, ddof=1),
    )


@dataclasses.dataclass(frozen=True)
class PredictiveGateResult:
    """Layer-B primary gate outcome (confirmatory statistic)."""

    T_obs: float
    p_value: float
    p_is_bound: bool              # True when 0 exceedances (p <= 1/(B+1))
    p_mc_error: float             # binomial MC error on p
    null_quantiles: dict          # q05/q50/q95/q99 of the null T
    null_mean: float
    null_sd: float
    effective_dof_note: str
    residual: np.ndarray          # (G,) observed d
    residual_z: np.ndarray        # (G,) d / sqrt(diag C) — descriptive
    covariance: GateCovariance
    n_null_draws: int
    seed_null: int
    fallback_1d: bool             # True if the cond>1e6 fallback engaged

    def report(self) -> dict:
        return dict(
            T_obs=float(self.T_obs), p_value=float(self.p_value),
            p_is_bound=self.p_is_bound, p_mc_error=float(self.p_mc_error),
            null_quantiles={k: float(v) for k, v in self.null_quantiles.items()},
            null_mean=float(self.null_mean), null_sd=float(self.null_sd),
            effective_dof_note=self.effective_dof_note,
            residual=[float(v) for v in self.residual],
            residual_z=[float(v) for v in self.residual_z],
            n_null_draws=self.n_null_draws, seed_null=self.seed_null,
            fallback_1d=self.fallback_1d,
            covariance=self.covariance.report(),
            layer=("B (calibration-predictive); simulation-calibrated p; "
                   "NO ratified threshold"
                   + ("; COND>1e6 FALLBACK ENGAGED: T_obs/p refer to the "
                      "DESCRIPTIVE max|z| over the 1-dim standardized group "
                      "residuals, NOT the prespecified confirmatory "
                      "Mahalanobis statistic (frozen spec section 3: no "
                      "inversion; report the standardized residuals)"
                      if self.fallback_1d else "")),
        )


def predictive_gate(pack: ModelAPack, *,
                    covariance: Optional[GateCovariance] = None,
                    group_edges: Sequence[Tuple[float, float]]
                    = PRIMARY_GROUP_EDGES,
                    n_null_draws: int = N_NULL_DRAWS,
                    seed_null: int = SEED_NULL,
                    resp_clamp: str = "both") -> PredictiveGateResult:
    """The frozen Layer-B primary gate: 3-group Mahalanobis T with an
    independent-ensemble null. Every analysis choice matches the observed
    evaluation exactly (same grouping, same frozen covariance, same plug-in).

    The cond > 1e6 fallback (frozen spec section 3) is decided from the
    matrix that is about to be inverted, RE-MEASURED here — never from the
    stored provenance field alone (a record can drift from the object it
    describes) — and from the stored field as well, so EITHER exceeding the
    threshold refuses the inversion. In fallback mode T degrades to the
    DESCRIPTIVE max|z| over the 1-dim standardized group residuals; the
    prespecified confirmatory Mahalanobis statistic is undefined there and
    ``report()`` labels the result accordingly."""
    A = group_aggregator(pack, group_edges)
    if covariance is None:
        covariance = estimate_covariance(pack, group_edges=group_edges,
                                         resp_clamp=resp_clamp)
    C = covariance.matrix
    ev = np.linalg.eigvalsh(C)                    # ascending
    cond_measured = float(ev[-1] / max(ev[0], 1e-300))
    fallback = (cond_measured > MAX_CONDITION_NUMBER
                or covariance.condition_number > MAX_CONDITION_NUMBER)
    Cinv = None if fallback else np.linalg.inv(C)

    mu_sig, fp_fold, live = _fold_parts(pack, resp_clamp=resp_clamp)
    n0 = np.asarray(pack.fp_counts, float)
    mu_obs = mu_sig + fp_fold(n0)
    y = np.asarray(pack.counts, float)

    def stat(yv, muv):
        d = _group_vector(yv, A, live) - _group_vector(muv, A, live)
        if fallback:
            return float(np.max(np.abs(d) / np.sqrt(np.diag(C))))
        return float(d @ Cinv @ d)

    d_obs = _group_vector(y, A, live) - _group_vector(mu_obs, A, live)
    T_obs = stat(y, mu_obs)

    rng = np.random.default_rng(seed_null)
    T_null = np.empty(n_null_draws)
    for r in range(n_null_draws):
        y_star = rng.poisson(np.clip(mu_obs, 0, None))
        n0_star = rng.poisson(n0)
        T_null[r] = stat(y_star, mu_sig + fp_fold(n0_star))

    n_exceed = int(np.sum(T_null >= T_obs))
    p = (1 + n_exceed) / (n_null_draws + 1)
    q = {f"q{int(100*a):02d}": float(np.quantile(T_null, a))
         for a in (0.05, 0.50, 0.95, 0.99)}
    m, s = float(T_null.mean()), float(T_null.std(ddof=1))
    eff_note = (f"null mean {m:.2f}, sd {s:.2f}; chi2-like eff dof ~ {m:.1f} "
                f"only if var ~ 2*mean (var/2mean = {s*s/(2*m):.2f})")
    return PredictiveGateResult(
        T_obs=T_obs, p_value=p, p_is_bound=(n_exceed == 0),
        p_mc_error=float(np.sqrt(p * (1 - p) / (n_null_draws + 1))),
        null_quantiles=q, null_mean=m, null_sd=s,
        effective_dof_note=eff_note,
        residual=d_obs,
        residual_z=d_obs / np.sqrt(np.diag(C)),
        covariance=covariance, n_null_draws=n_null_draws,
        seed_null=seed_null, fallback_1d=fallback,
    )


def transport_stress_stats(pack: ModelAPack, *,
                           resp_clamp: str = "both") -> dict:
    """Layer C: cross-mock transport stress (UNCALIBRATED systematic).

    Reports the full-grid total residual standardized by survey + calibration
    variance (delta method: Var_cal(total) = sum_cells (mu_fp_cell/n0_cell)^2
    * n0_cell), and the FP-attributable share. Never enters Layer B.
    """
    mu_sig, fp_fold, live = _fold_parts(pack, resp_clamp=resp_clamp)
    n0 = np.asarray(pack.fp_counts, float)
    mu_fp = fp_fold(n0)
    mu = mu_sig + mu_fp
    y = np.asarray(pack.counts, float)
    tot_mu = float(np.where(live, mu, 0.0).sum())
    tot_y = float(np.where(live, y, 0.0).sum())
    mfk = np.where(live, mu_fp, 0.0).sum(axis=1)          # (C, S)
    with np.errstate(divide="ignore", invalid="ignore"):
        var_cal = float(np.where(n0 > 0, mfk ** 2 / np.where(n0 > 0, n0, 1),
                                 0.0).sum())
    var_surv = tot_mu
    z_surv = (tot_y - tot_mu) / np.sqrt(max(var_surv, 1e-300))
    z_full = (tot_y - tot_mu) / np.sqrt(max(var_surv + var_cal, 1e-300))
    return dict(
        layer="C (transport stress; UNCALIBRATED systematic — never "
              "absorbed into predictive covariance)",
        total_obs=tot_y, total_mu=tot_mu,
        total_mu_fp=float(np.where(live, mu_fp, 0.0).sum()),
        residual=tot_y - tot_mu,
        z_survey_only=float(z_surv), z_with_calibration=float(z_full),
        var_survey=var_surv, var_calibration=float(var_cal),
    )
