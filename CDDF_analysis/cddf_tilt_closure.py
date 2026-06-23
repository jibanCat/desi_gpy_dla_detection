"""cddf_tilt_closure.py — WALL-1 tilt-closure acceptance gate (spec §7).

Authoritative against `2026-06-12_catalog_hbi_estimator_spec.md` §7 (and the
§2 gate spec G1). This is the **out-of-sample gate** that breaks the in-sample
circularity of the v1 catalog-HBI fit (`cddf_catalog_hbi.py`): the fit's C/ρ
are calibrated on the SAME mock truth, so the catalog "P/C → 1.000" is NOT
validation. WALL-1 re-tilts the TRUE CDDF slope, re-deposits the tilt on the
REAL GP detections (via their truth host), FREEZES C/ρ/b_FP, refits f_b on the
tilted catalog, and checks the recovered f(N,z) tracks the tilted truth.

WALL-1 (spec §7) — exactly:
  - Reweight every TRUE 2LPT absorber by w(logN) = 10^(Δα·(logN−20.3)), Δα = ±0.5.
  - Deposit on (i) the TRUTH histogram → n_true^tilt; (ii) each REAL GP detection
    via its truth host (same TARGETID, nearest-z |Δz|/(1+z)<0.01; NO host = forest
    FP → weight 1) → measured^tilt.
  - FREEZE C / ρ / b_FP (forest properties, slope-independent — NOT re-regenerated
    under the tilt). Refit f_b on measured^tilt with the v1 estimator.
  - HEADLINE PASS = for each report limit, the INTEGRATED dN/dX & Ω closure
    |pull|≤3 on BOTH tilts AND the closure prediction inside the 95% MC band AND no
    opposite-sign coherent integrated pull (the b_FP-bias signature). The
    differential per-bin f_b closure is a DIAGNOSTIC only (see below).
    A <19.5 FAIL on the differential is EXPECTED → band must WIDEN/clip, not miss.
  - Non-circular because measured^tilt = REAL GP output reweighted by truth, never
    R·n_true.

WHY THE HEADLINE GATE IS THE INTEGRATED dN/dX & Ω, NOT THE DIFFERENTIAL f_b
(code-review #2 / LyA-review Finding 4): the closure prediction R0·truth^tilt is
*migration-blind* — R0 is a per-bin factor measured at Δα=0, but est^tilt sums
detections by PREDICTED N while weighting by TRUE-host N, so a predicted bin holding
up-migrated lower-N systems mis-closes by construction (est^tilt/pred ≈ 0.79 on
+tilt). A *differential* f_b closure pull therefore CANNOT pass under N-migration
regardless of estimator quality — that is the v1 migration signature (v2's job,
spec §5/§9), NOT the b_FP signature WALL-1 must catch. Integrating over N within a
tier cancels the within-tier migration, so the integrated dN/dX & Ω ARE the clean
WALL-1 statistic. The differential per-bin pull/coverage is reported (so a reader
SEES the migration) but never gated.

TRUTH-HOST FLOOR (CS-review Finding 1, load-bearing): the tilt mark on a detection
is its TRUTH host's logN. The truth used to ATTACH that host is floored at
``host_truth_floor`` (default 19.0), DECOUPLED from the C/ρ matrix floor — else a
20.3-floored match labels sub-DLA up-migrants (true host in [19,20.3]) as hostless
forest FPs with tilt weight 1.0, corrupting the deposit (spec §2 WIRING).

NON-CIRCULARITY (load-bearing): the tilt weight on a real detection is read from
its TRUTH host's logN (`cat["NHI_TRUE"]`, attached by the molly matcher), NOT from
its own predicted N̂ and NOT from any response kernel. So `measured^tilt` is the
actual GP catalog with a truth-driven mark — the estimator never sees `R·n_true`.

FP FREEZE (CS-review Finding 6 / LyA-review Finding 2): the purity-mixture FP is a
per-catalog-row mark — its (1−ρ_i) contribution correctly scales with the row's
tilt weight. The loa-0 FP (spec §4 PRIMARY) is a FROZEN external forest background
and must NOT be tilt-scaled; run_one_tilt refuses any non-purity-mixture FP until
the tilt is threaded to the 1/C numerator alone.

DISCIPLINE: a NEW analysis module. NEVER touch dla_gp.py / run_bayes_select.py /
dlasearch.py / any inference. No git commit. Outputs → the scratch out_dir.
Reuses cddf_catalog_hbi (loaders, C/ρ regen, pathlength, the v1 estimator,
joint-MC) — the tilt enters only via the per-op-row weight the v1 estimator already
exposes (estimate_f_b.boot_weights / joint_mc_errors.tilt_weights_op).
"""
from __future__ import annotations

import os
import numpy as np
from astropy.table import Table

from CDDF_analysis.cddf_catalog_hbi import (
    HBIConfig,
    load_molly_matrix,
    load_and_cut_catalog,
    regenerate_molly_counts,
    make_C_interpolator,
    make_rho_interpolator,
    build_pathlength,
    build_fine_grid,
    make_fp_model,
    estimate_f_b,
    joint_mc_errors,
    omega_hi_prefactor,
    _build_qso_lookup,
    _bin_index_logN,
    _zbin_index,
    C_FLOOR,
)

LOGN_PIVOT = 20.3        # the tilt pivot (spec §7: w = 10^(Δα·(logN−20.3)))
DEFAULT_DALPHA = 0.5     # ±0.5 tilt
PULL_GATE_LOGN = 19.5    # logN >= this must satisfy |pull|<=3 (spec §7)
PULL_THRESHOLD = 3.0
# Coherent-pull threshold for the opposite-sign b_FP-bias signature. The mean
# closure pull over n independent gated cells has sampling std ~ 1/√n under the
# null, so a "coherent" deviation is k·σ_mean = COHERENT_PULL_K / √n_cells
# (replaces the old hardcoded 1.0σ — LyA-review Finding 4 / code-review #3). The
# headline gate is the INTEGRATED dN/dX & Ω closure (migration-insensitive); the
# coherent test is applied to those tier integrals, where opposite-sign divergence
# is the clean slope-dependent FP-misspecification signal.
COHERENT_PULL_K = 3.0


# -----------------------------------------------------------------------------
# 1. The tilt weight
# -----------------------------------------------------------------------------
def tilt_weight(logN, dalpha: float, pivot: float = LOGN_PIVOT):
    """w(logN) = 10^(Δα·(logN − pivot)). NaN logN (hostless detection = forest FP)
    → weight 1.0 (the FP is shape-independent: a forest false positive does not
    know about the true CDDF slope, so the tilt must NOT scale it)."""
    logN = np.asarray(logN, dtype=float)
    w = 10.0 ** (dalpha * (logN - pivot))
    return np.where(np.isfinite(logN), w, 1.0)


# -----------------------------------------------------------------------------
# 2. Tilted truth target n_true^tilt  (the thing the estimate must recover)
# -----------------------------------------------------------------------------
def tilted_truth_reductions(cfg: HBIConfig, truth_cut: Table,
                            logN_lo, logN_hi, N_b, dN_b, X_tot,
                            dalpha: float, pivot: float = LOGN_PIVOT) -> dict:
    """Truth-side f(N), dN/dX(z), Ω with every TRUE absorber reweighted by
    w(logN)=10^(Δα·(logN−pivot)). Truth restricted to the SNR>snr_min sightline
    set (matches the ΔX denominator, gotcha 4). This is the WALL-1 'n_true^tilt'
    that the tilted estimate must reproduce.

    Returns f_truth (z-marg), dndx_z (per limit), dndx_total (per limit), omega.
    """
    zbins = np.asarray(cfg.zbins, dtype=float)
    n_zbins = len(zbins) - 1
    X = np.asarray(X_tot, dtype=float)
    X_sum = float(np.nansum(X))

    t_nhi = np.asarray(truth_cut["NHI"], dtype=float)
    t_z = np.asarray(truth_cut["Z_DLA"], dtype=float)
    t_snr = np.asarray(truth_cut["S2N_RED"], dtype=float)
    keep = t_snr > cfg.snr_min
    t_nhi, t_z = t_nhi[keep], t_z[keep]
    w = tilt_weight(t_nhi, dalpha, pivot)   # truth has a host by definition

    n_nbins = len(logN_lo)
    nidx = _bin_index_logN(t_nhi, logN_lo, logN_hi)
    zidx = _zbin_index(t_z, zbins)

    # z-marginalized weighted f(N)
    f_truth = np.zeros(n_nbins)
    Wbk = np.zeros((n_nbins, n_zbins))
    valid = (nidx >= 0)
    np.add.at(f_truth, nidx[valid], w[valid])
    f_truth = np.where(X_sum > 0, f_truth / (X_sum * dN_b), np.nan)

    valid2 = (nidx >= 0) & (zidx >= 0)
    np.add.at(Wbk, (nidx[valid2], zidx[valid2]), w[valid2])

    K = omega_hi_prefactor(cfg.H0)
    limits = cfg.report_logN_limits
    dndx_z = {}
    dndx_total = {}
    omega = {}
    for lim in limits:
        sel = logN_lo >= lim - 1e-9
        # weighted dN/dX(z): Σ_{N>=lim, host} w / ΔX(z)
        dz = np.zeros(n_zbins)
        for k in range(n_zbins):
            dz[k] = (Wbk[sel, k].sum() / X[k]) if X[k] > 0 else np.nan
        dndx_z[lim] = dz
        # CS-review F4 / LyA-6: restrict to the SAME fine-grid support as the gridded
        # estimate (num_marg over [floor, drop_top_bin_above)) so the truth dndx_total
        # and the estimator's _dndx_total_limit share one support — the >=22.4 rows
        # that the gridded Wbk/dndx_z drop are excluded here too (else truth is a hair
        # high; ~6/101663 rows, <0.01%, but the supports must match for a clean pull).
        above = ((t_nhi >= lim) & (t_nhi < cfg.drop_top_bin_above)
                 & (zidx >= 0))
        dndx_total[lim] = (w[above].sum() / X_sum) if X_sum > 0 else np.nan
        omega[lim] = K * np.nansum(N_b[sel] * f_truth[sel] * dN_b[sel])
    return dict(f_truth=f_truth, dndx_z=dndx_z, dndx_total=dndx_total, omega=omega,
                W_bk=Wbk)


# -----------------------------------------------------------------------------
# 3. Per-op-row tilt weights for the REAL detections (via truth host)
# -----------------------------------------------------------------------------
def _tilt_host_nhi(cat_cut: Table) -> np.ndarray:
    """The truth-host NHI that drives the tilt mark: NHI_TILT_HOST (the decoupled
    low-floor host match, CS-review F1) if present, else NHI_TRUE (the primary
    matrix-floor host) for backward compatibility."""
    key = "NHI_TILT_HOST" if "NHI_TILT_HOST" in cat_cut.colnames else "NHI_TRUE"
    return np.asarray(cat_cut[key], dtype=float)


def detection_tilt_weights(cat_cut: Table, good_mask: np.ndarray, cfg: HBIConfig,
                           dalpha: float, pivot: float = LOGN_PIVOT) -> np.ndarray:
    """Per-op-row tilt weight for the REAL GP detections, in the SAME op order the
    v1 estimator uses (op = S2N_RED>snr_min & P_DLA>p_dla_min & DLAFLAG==0).

    Weight = 10^(Δα·(N_host − pivot)) for a detection WITH a truth host, weight = 1.0
    for a HOSTLESS detection (NaN host = forest FP — spec §7). The host NHI is read
    from NHI_TILT_HOST (the decoupled low-floor host match, CS-review F1) so sub-DLA
    up-migrants get their true tilt weight instead of a flat 1.0; falls back to
    NHI_TRUE if that column is absent.
    """
    s2n = np.asarray(cat_cut["S2N_RED"], dtype=float)
    pdla = np.asarray(cat_cut["P_DLA"], dtype=float)
    op = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask
    nhi_host_op = _tilt_host_nhi(cat_cut)[op]
    return tilt_weight(nhi_host_op, dalpha, pivot)


def hostless_op_fraction(cat_cut: Table, good_mask: np.ndarray, cfg: HBIConfig,
                         logN_min: float = 20.3) -> dict:
    """CS-review F1 diagnostic: fraction of op detections (restricted to predicted
    NHI >= logN_min) with NO tilt host (NaN NHI_TILT_HOST = forest FP, tilt weight
    1.0). With the floor decoupled (host_truth_floor <= 19.0) this should be SMALL at
    >=20.3 (~1.7%); a large value (~18% at floor 20.3, i.e. NHI_TRUE alone) signals
    the F1 bug is live (sub-DLA up-migrants mislabeled hostless)."""
    s2n = np.asarray(cat_cut["S2N_RED"], dtype=float)
    pdla = np.asarray(cat_cut["P_DLA"], dtype=float)
    nhi = np.asarray(cat_cut["NHI"], dtype=float)
    nhi_host = _tilt_host_nhi(cat_cut)
    op = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask
    op_hi = op & (nhi >= logN_min - 1e-9)
    n_op = int(op_hi.sum())
    n_hostless = int((op_hi & ~np.isfinite(nhi_host)).sum())
    return dict(logN_min=float(logN_min), n_op=n_op, n_hostless=n_hostless,
                frac_hostless=(n_hostless / n_op if n_op else np.nan))


# -----------------------------------------------------------------------------
# 4. Pull computation
# -----------------------------------------------------------------------------
def _pull(est_point, est_std, truth_target):
    """pull = (est − truth) / σ ; NaN-safe (σ==0 or NaN → NaN pull)."""
    est_point = np.asarray(est_point, dtype=float)
    est_std = np.asarray(est_std, dtype=float)
    truth_target = np.asarray(truth_target, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = (est_point - truth_target) / est_std
    p = np.where(np.isfinite(est_std) & (est_std > 0), p, np.nan)
    return p


def _coverage(samples_arr, truth_target, q_lo=2.5, q_hi=97.5):
    """Fraction of bins whose tilted-truth target falls inside the MC [q_lo,q_hi]
    band. samples_arr is (n_mc, n_bins); truth_target is (n_bins,). NaN-target or
    all-NaN-sample bins are skipped. Returns (coverage, n_bins_used, inside_mask)."""
    lo = np.nanpercentile(samples_arr, q_lo, axis=0)
    hi = np.nanpercentile(samples_arr, q_hi, axis=0)
    truth_target = np.asarray(truth_target, dtype=float)
    usable = np.isfinite(truth_target) & np.isfinite(lo) & np.isfinite(hi)
    inside = (truth_target >= lo) & (truth_target <= hi) & usable
    n_used = int(usable.sum())
    cov = float(inside[usable].mean()) if n_used else np.nan
    return cov, n_used, inside


def _omega_closure_resid_frac(res, logN_lo, logN_hi, N_b, dN_b, H0,
                              limits, drop_top_above, baseline=None,
                              closure_R0_mode="divide"):
    """Ω-integrated closure residual fraction (est^tilt − R0·truth^tilt)/(R0·truth^tilt)
    at a set of *deep* logN limits.

    The DEEP-TIER BOUNDARY TEST (LyA-review Finding 3, decisive): integrate Ω at the
    20.3 reporting boundary AND deep into the tail (≥20.6, ≥21.0). If the residual is
    the A2-gated sub-DLA→DLA up-migration ACROSS the 20.3 boundary, it must be LARGEST
    at the boundary and SHRINK deep (the up-migrants from [19,20.3] cannot reach the
    deep ≥21.0 tier — they are ≤1σ≈0.15 dex up-scatters). If instead the residual GROWS
    monotonically into the migration-free deep tail, the mechanism is a slope-dependent
    over-response of the forward MAP/regularized deconvolution on the steep high-N tail
    — NOT the boundary effect, and NOT auto-removable by a symmetric kernel.

    ``baseline`` (preferred): use the per-reduction INTEGRATED untilted R0 = Ω_est0/Ω_tr0
    at each deep limit (faithfully reproduces the LyA-reviewer's independent probe; the
    gate's headline closure uses the integrated R0, not the integral of per-bin R0_f).
    Without it, fall back to the per-bin ``pred_f = R0_f·truth^tilt`` integrated (the
    differential-closure form) — the GROW/SHRINK trend is robust to which is used.

    Returns {limit: resid_frac}. NaN where the predicted Ω is ≤0 (empty tier)."""
    K = omega_hi_prefactor(H0)
    f_pt = np.asarray(res["point"]["f_b"], float)
    ttr = res.get("ttr")           # tilted-truth reductions (per-bin f_truth^tilt)
    e0f = (np.asarray(baseline["e0"]["f_b"], float) if baseline is not None else None)
    t0f = (np.asarray(baseline["t0"]["f_truth"], float) if baseline is not None else None)
    out = {}
    for lim in limits:
        sel = (logN_lo >= lim - 1e-9) & (logN_hi <= drop_top_above + 1e-9)
        o_est = K * np.nansum(N_b[sel] * f_pt[sel] * dN_b[sel])
        if baseline is not None and ttr is not None:
            # integrated-R0 form (the gate's headline closure normalization)
            ftr_tilt = np.asarray(ttr["f_truth"], float)
            o_tr_tilt = K * np.nansum(N_b[sel] * ftr_tilt[sel] * dN_b[sel])
            o_e0 = K * np.nansum(N_b[sel] * e0f[sel] * dN_b[sel])
            o_t0 = K * np.nansum(N_b[sel] * t0f[sel] * dN_b[sel])
            R0 = (o_e0 / o_t0) if o_t0 > 0 else np.nan
            # closure_R0_mode="unit" (numerical Finding 1/2 — ONLY valid when v3's
            # untilted R0≈1, gated by the runner): close on the BARE tilted truth.
            if closure_R0_mode == "unit":
                R0 = 1.0
            o_pred = R0 * o_tr_tilt
        else:
            f_pred = np.asarray(res["pred_f"], float)  # R0_f·truth^tilt, per bin
            o_pred = K * np.nansum(N_b[sel] * f_pred[sel] * dN_b[sel])
        out[lim] = float((o_est - o_pred) / o_pred) if (np.isfinite(o_pred)
                                                        and o_pred > 0) else np.nan
    return out


def _deep_tier_discriminant(res_plus, res_minus, logN_lo, logN_hi, N_b, dN_b, H0,
                            drop_top_above, deep_limits=(20.3, 20.6, 21.0),
                            baseline=None, closure_R0_mode="divide"):
    """Classify the FAIL mechanism by the Ω closure-residual fraction vs limit
    (LyA-review Finding 3). Returns a dict with the per-tilt residual fractions, the
    boundary→deep TREND, and a verdict in {grows_deep, shrinks_deep, flat, n/a}.

    ``grows_deep`` (|resid| monotonically larger from 20.3→21.0 on the dominant tilt)
    falsifies the boundary-up-migration explanation: it is the forward-MAP steep-tail
    slope over-response. ``shrinks_deep`` is the boundary-migration signature."""
    rf_plus = _omega_closure_resid_frac(res_plus, logN_lo, logN_hi, N_b, dN_b, H0,
                                        deep_limits, drop_top_above, baseline=baseline,
                                        closure_R0_mode=closure_R0_mode)
    rf_minus = _omega_closure_resid_frac(res_minus, logN_lo, logN_hi, N_b, dN_b, H0,
                                         deep_limits, drop_top_above, baseline=baseline,
                                         closure_R0_mode=closure_R0_mode)
    dl = list(deep_limits)
    # the dominant tilt for the up-migration tell is the −tilt (up-weights low-N tail);
    # judge the trend on whichever tilt has the larger |resid| at the boundary.
    b0 = dl[0]
    mag_plus = abs(rf_plus.get(b0, np.nan))
    mag_minus = abs(rf_minus.get(b0, np.nan))
    dom = rf_minus if (np.isfinite(mag_minus) and
                       (not np.isfinite(mag_plus) or mag_minus >= mag_plus)) else rf_plus
    dom_tag = "minus" if dom is rf_minus else "plus"
    seq = [dom.get(L, np.nan) for L in dl]
    mags = [abs(v) for v in seq]
    trend = "n/a"
    if all(np.isfinite(m) for m in mags) and len(mags) >= 2:
        # strictly growing into the deep tail (allow a small tolerance)
        grows = all(mags[i + 1] >= mags[i] - 1e-3 for i in range(len(mags) - 1)) and \
            (mags[-1] > mags[0] * 1.10)
        shrinks = all(mags[i + 1] <= mags[i] + 1e-3 for i in range(len(mags) - 1)) and \
            (mags[-1] < mags[0] * 0.90)
        trend = "grows_deep" if grows else ("shrinks_deep" if shrinks else "flat")
    return dict(resid_frac_plus=rf_plus, resid_frac_minus=rf_minus,
                deep_limits=dl, dominant_tilt=dom_tag, dominant_seq=seq, trend=trend)


def _deep_tier_differential_discriminant(res_plus, res_minus, logN_lo, logN_hi,
                                         baseline, closure_R0_mode="divide",
                                         deep_range=(20.45, 21.45),
                                         tail_logN=21.5):
    """DIFFERENTIAL (per-N density) deep-tier trend — the CORRECT v3 discriminant
    (4-lens WALL-1 review: bayesian Finding 1, numerical Finding 4/6, cs Finding 4).

    The cumulative-Ω ``_deep_tier_discriminant`` integrates ∫_L, so the cumulative
    fraction DRIFTS as L moves past the well-closed lower bins even when the per-N
    residual is FLAT — and it folds in the over-stiffened under-sampled tail (>21.5)
    where the untilted R0≠1, manufacturing a spurious ``grows_deep``. The v2
    MAP_SLOPE_OVERRESPONSE signature is genuinely a *differential* per-N over-response
    that grows into the migration-free deep tail. For a PARAMETRIC family a tilt is a
    smooth θ-shift, so the DIFFERENTIAL per-N residual must stay FLAT across the body.

    We judge the trend on the per-bin closure residual fraction
    ``(f_est^tilt − R0_f·f_truth^tilt)/(R0_f·f_truth^tilt)`` over the MIGRATION-FREE
    body ``deep_range`` (default [20.45, 21.45], i.e. ≥1σ≈0.15 dex above the 20.3
    boundary so no sub-DLA→DLA up-migrant reaches it, and below the under-sampled
    ``tail_logN`` where occupancy collapses). On the dominant tilt:
      * ``grows_deep`` — |resid_frac| rises monotonically across the body (the v2/bplcut
        per-bin over-response PERSISTS): a real DOF/representation failure.
      * ``flat`` — |resid_frac| does not trend up: the v2 signature is GONE; any residual
        is a uniform (kernel-level) offset, NOT a deep-tail over-response.
    Returns the per-tilt body sequences, the dominant-tilt trend, the body max |resid|,
    and a robust slope d|resid|/dlogN over the body (sign>0 with magnitude is the
    grows tell). This is what the v3 hard gate should fire on (not the cumulative)."""
    mid = 0.5 * (logN_lo + logN_hi)
    R0f = np.asarray(baseline["R0_f"], float)
    lo, hi = deep_range
    body = (mid >= lo - 1e-9) & (mid <= min(hi, tail_logN) + 1e-9)
    bidx = np.where(body)[0]
    seqs = {}
    for tag, res in (("plus", res_plus), ("minus", res_minus)):
        f_est = np.asarray(res["point"]["f_b"], float)
        f_tr = np.asarray(res["ttr"]["f_truth"], float)
        pred = (f_tr if closure_R0_mode == "unit" else R0f * f_tr)
        with np.errstate(divide="ignore", invalid="ignore"):
            rf = np.where(pred > 0, (f_est - pred) / pred, np.nan)
        seqs[tag] = [(float(mid[b]), float(rf[b])) for b in bidx
                     if np.isfinite(rf[b])]
    # dominant tilt = larger mean |resid| over the body
    def _mean_abs(s):
        v = [abs(r) for _, r in s]
        return float(np.mean(v)) if v else np.nan
    ma_p, ma_m = _mean_abs(seqs["plus"]), _mean_abs(seqs["minus"])
    dom_tag = "minus" if (np.isfinite(ma_m) and
                          (not np.isfinite(ma_p) or ma_m >= ma_p)) else "plus"
    dom = seqs[dom_tag]
    trend = "n/a"; slope = np.nan; body_max = np.nan
    if len(dom) >= 3:
        xs = np.array([x for x, _ in dom]); ys = np.array([abs(r) for _, r in dom])
        body_max = float(np.nanmax(ys))
        # least-squares slope of |resid_frac| vs logN over the migration-free body
        slope = float(np.polyfit(xs, ys, 1)[0])
        # GROWS only if the body |resid| trends UP appreciably AND the endpoint is
        # >10% larger than the start (matches the cumulative test's growth criterion).
        grows = (slope > 0.02) and (ys[-1] > ys[0] * 1.10)
        trend = "grows_deep" if grows else "flat"
    return dict(seq_plus=seqs["plus"], seq_minus=seqs["minus"],
                dominant_tilt=dom_tag, trend=trend, body_slope=slope,
                body_max_abs=body_max, body_mean_abs_plus=ma_p,
                body_mean_abs_minus=ma_m, deep_range=list(deep_range),
                tail_logN=float(tail_logN))


# -----------------------------------------------------------------------------
# 4b. v3 CONTINUOUS-SPACE closure (per-N f(N|θ̂^tilt) + integrated-after-fit deep
#     tail). The decisive v3-vs-v2 comparison: v2 (free-bin) over-responds per-bin
#     so its deep-tier Ω residual GROWS into the migration-free tail; a PARAMETRIC
#     f(N|θ) absorbs a tilt as a smooth θ-shift, so the deep residual must stay FLAT.
# -----------------------------------------------------------------------------
def v3_continuous_closure_summary(res_plus, res_minus, baseline, logN_lo, logN_hi,
                                  N_b, dN_b, H0, drop_top_above,
                                  closure_R0_mode="divide",
                                  deep_limits=(20.0, 20.3, 20.6, 21.0, 21.5),
                                  perN_range=(19.5, 21.5)):
    """Continuous-space closure on the v3 parametric f(N|θ̂^tilt) (spec §7, v3 face).

    Returns, for BOTH tilts:
      * ``perN`` : per-logN-bin closure of f(N|θ̂^tilt) over ``perN_range`` —
        f_est^tilt (the reduced parametric density) vs the closure target
        R0_f·f_truth^tilt, in units of the WALL-2 MC σ (the per-N pull) AND the raw
        residual fraction. This is the CONTINUOUS f(N) closure (no bin migration in
        the parametric face — θ-shift is smooth).
      * ``deep_integrated`` : the Ω closure residual fraction integrated after the fit
        at successively deeper limits (the v2 grows_deep discriminant), extended to
        21.5 (the under-sampled tail where bplcut over-cut). FLAT = v3 PASS signature.
      * ``deep_trend`` / ``deep_max_abs`` : the boundary→deep monotonic-growth verdict
        and the worst |resid| — the gate's hard v3 check (v2 was grows_deep, max 0.64).

    Fully derived from arrays run_one_tilt already returns (res['point']['f_b'] =
    reduced parametric f(N), res['mc']['f_b']['std'] = WALL-2 σ, res['ttr']['f_truth']
    = tilted truth, baseline['R0_f'] = per-bin untilted R0). No extra compute."""
    K = omega_hi_prefactor(H0)
    mid = 0.5 * (logN_lo + logN_hi)
    R0f = np.asarray(baseline["R0_f"], float)
    out = {}
    lo_p, hi_p = perN_range
    body = (mid >= lo_p - 1e-9) & (mid <= hi_p + 1e-9)
    for tag, res in (("plus", res_plus), ("minus", res_minus)):
        f_est = np.asarray(res["point"]["f_b"], float)
        f_std = np.asarray(res["mc"]["f_b"]["std"], float)
        f_tr_tilt = np.asarray(res["ttr"]["f_truth"], float)
        if closure_R0_mode == "unit":
            pred = f_tr_tilt
        else:
            pred = R0f * f_tr_tilt
        with np.errstate(divide="ignore", invalid="ignore"):
            pull = np.where(f_std > 0, (f_est - pred) / f_std, np.nan)
            resid_frac = np.where(pred > 0, (f_est - pred) / pred, np.nan)
        perN = []
        for b in np.where(body)[0]:
            perN.append(dict(logN=float(mid[b]), f_est=float(f_est[b]),
                             pred=float(pred[b]), f_truth_tilt=float(f_tr_tilt[b]),
                             pull=float(pull[b]) if np.isfinite(pull[b]) else None,
                             resid_frac=float(resid_frac[b]) if np.isfinite(resid_frac[b]) else None))
        pulls_body = pull[body]
        max_abs_pull = (float(np.nanmax(np.abs(pulls_body)))
                        if np.isfinite(pulls_body).any() else np.nan)
        rf_body = np.abs(resid_frac[body])
        max_abs_resid = float(np.nanmax(rf_body)) if np.isfinite(rf_body).any() else np.nan
        out[tag] = dict(perN=perN, perN_max_abs_pull=max_abs_pull,
                        perN_max_abs_resid_frac=max_abs_resid,
                        perN_n=int(body.sum()))
    # integrated-after-fit deep tail (the grows_deep discriminant, extended to 21.5)
    deep = _deep_tier_discriminant(res_plus, res_minus, logN_lo, logN_hi, N_b, dN_b, H0,
                                   drop_top_above, deep_limits=deep_limits,
                                   baseline=baseline, closure_R0_mode=closure_R0_mode)
    out["deep_integrated"] = dict(resid_frac_plus=deep["resid_frac_plus"],
                                  resid_frac_minus=deep["resid_frac_minus"])
    out["deep_trend"] = deep["trend"]
    out["deep_dominant_tilt"] = deep["dominant_tilt"]
    out["deep_limits"] = list(deep_limits)
    mags = [abs(v) for v in deep["dominant_seq"] if np.isfinite(v)]
    out["deep_max_abs"] = float(max(mags)) if mags else np.nan
    return out


# -----------------------------------------------------------------------------
# 5a. Untilted baseline recovery ratio R0 (Δα = 0)
# -----------------------------------------------------------------------------
def baseline_recovery(cfg: HBIConfig, cat_cut, is_TP, good_mask, truth_cut,
                      C_interp, fp_model, X_tot, logN_lo, logN_hi, N_b, dN_b,
                      estimator_fn=estimate_f_b) -> dict:
    """The estimator's UNTILTED (Δα=0) recovery ratio R0 = est / truth, per bin and
    per reduction. This is the v1 selection-correction's *baseline* fidelity — it is
    NOT 1.0 because v1 does not deconvolve the N-measurement (Eddington/prior-edge)
    migration (spec §5/§9: v1 "inherits the ~0.06-dex steep-f(N) Eddington scatter";
    "Don't claim '+0.06 dex gone'"). That absolute bias is the §8 anchor / A2-gate's
    business, NOT WALL-1's.

    WALL-1 tests CLOSURE: whether re-tilting the population breaks the selection
    function (which would show as the +tilt/−tilt residuals diverging with OPPOSITE
    sign — the b_FP-misspecification signature). The clean closure statistic divides
    out this baseline R0: the tilt prediction is `R0 · truth^tilt`, and the pull is
    `(est^tilt − R0·truth^tilt)/σ`. R0 cancels the (validated, tilt-independent)
    absolute bias and isolates the *tilt-induced* break. We ALSO report the raw pull
    vs the bare tilted truth (which re-measures the §8 absolute bias) for transparency.
    """
    e0 = estimator_fn(cat_cut, is_TP, good_mask, C_interp, fp_model, X_tot,
                      logN_lo, logN_hi, N_b, dN_b, truth_cut, cfg,
                      clip_negative=False)
    t0 = tilted_truth_reductions(cfg, truth_cut, logN_lo, logN_hi, N_b, dN_b,
                                 X_tot, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        R0_f = np.where(np.asarray(t0["f_truth"]) > 0,
                        np.asarray(e0["f_b"]) / np.asarray(t0["f_truth"]), np.nan)
    R0_dndx_z = {}
    R0_dndx_total = {}
    R0_omega = {}
    for lim in cfg.report_logN_limits:
        with np.errstate(divide="ignore", invalid="ignore"):
            tz = np.asarray(t0["dndx_z"][lim], dtype=float)
            R0_dndx_z[lim] = np.where(tz > 0,
                                      np.asarray(e0["dndx_z"][lim]) / tz, np.nan)
        td = t0["dndx_total"][lim]
        R0_dndx_total[lim] = (e0["dndx_total"][lim] / td) if td > 0 else np.nan
        to = t0["omega"][lim]
        R0_omega[lim] = (e0["omega"][lim] / to) if to > 0 else np.nan
    return dict(e0=e0, t0=t0, R0_f=R0_f, R0_dndx_z=R0_dndx_z,
                R0_dndx_total=R0_dndx_total, R0_omega=R0_omega)


# -----------------------------------------------------------------------------
# 5. One tilt: refit on measured^tilt + MC band, vs n_true^tilt
# -----------------------------------------------------------------------------
def run_one_tilt(cfg: HBIConfig, cat_cut, is_TP, good_mask, truth_cut, mm,
                 C_interp, fp_model, X_tot,
                 logN_lo, logN_hi, N_b, dN_b, dalpha: float,
                 rng, baseline: dict, estimator_fn=estimate_f_b) -> dict:
    """Refit the v1 estimator on the tilted catalog (per-op-row tilt weight) with
    C/ρ/b_FP FROZEN, build the joint-MC band, and compute the per-(N,z) pulls.

    TWO pull families are returned per reduction:
      * ``*_pull``      = CLOSURE-residual pull (est^tilt − R0·truth^tilt)/σ — the
        WALL-1 statistic (baseline absolute bias R0 divided out; isolates the
        tilt-induced break / b_FP-misspecification signature).
      * ``*_pull_raw``  = raw pull (est^tilt − truth^tilt)/σ — re-measures the §8
        absolute bias under the tilt; reported for transparency, NOT gated.
    """
    # tilted truth target (what we must recover)
    ttr = tilted_truth_reductions(cfg, truth_cut, logN_lo, logN_hi, N_b, dN_b,
                                  X_tot, dalpha)

    # per-op-row tilt weights on the REAL detections (truth-host driven)
    w_op = detection_tilt_weights(cat_cut, good_mask, cfg, dalpha)

    # FP-FREEZE GUARD (CS-review Finding 6 / LyA-review Finding 2 — the loa-0
    # landmine). For the purity-mixture FP the per-object tilt weight CORRECTLY
    # scales BOTH the 1/C numerator AND the (1−ρ_i) FP term: (1−ρ_i) is the
    # contamination attached to that SAME catalog row, so when the row's truth-host
    # mark scales, its attached FP contribution scales coherently (a strong-DLA host
    # has ρ≈0.99 ⇒ (1−ρ)≈0.01, so the up-tilt barely moves the FP). But the loa-0 FP
    # (spec §4 PRIMARY) is a FROZEN, Λ-independent forest-background intensity
    # measured on a SEPARATE field — NOT a per-catalog-row mark — and MUST NOT be
    # tilt-scaled at all (spec §7: "FREEZE b_FP — forest properties, slope-
    # independent"). The current FP plumbing (estimate_f_b/joint_mc_errors) threads
    # the tilt weight into mu_fp_grid(weights=...), which is correct ONLY for the
    # purity-mixture. Until the loa-0 path threads the tilt to the numerator ALONE
    # (FP grid held at its untilted frozen value), WALL-1 refuses to run on it.
    if cfg.fp_estimator != "purity_mixture":
        raise NotImplementedError(
            f"WALL-1 tilt is only wired for the purity-mixture FP (it tilt-scales "
            f"the attached (1−ρ) term coherently). fp_estimator="
            f"{cfg.fp_estimator!r} is a FROZEN external background that must NOT be "
            f"tilt-scaled (spec §7 / §4 loa-0); thread the tilt to the 1/C numerator "
            f"only before enabling.")

    # POINT estimate on measured^tilt: thread the tilt as boot_weights (it
    # multiplies BOTH the 1/C numerator AND the (1−ρ) FP term, exactly as a per-
    # object mark should — hostless FP weight=1 leaves its FP subtraction intact).
    # The v2 estimator (estimator_fn != estimate_f_b) threads the SAME tilt weight
    # via boot_weights → its obj_weights_extra (on the Σ_i log + (1−ρ) term, NOT M).
    point = estimator_fn(
        cat_cut, is_TP, good_mask, C_interp, fp_model,
        X_tot, logN_lo, logN_hi, N_b, dN_b, truth_cut, cfg,
        boot_weights=w_op, clip_negative=False,   # keep UN-clipped so pulls/CIs reach 0
    )

    # joint-MC band with C/ρ/b_FP FROZEN (mm carries the unweighted counts) and the
    # SAME tilt applied each draw (tilt_weights_op multiplies the per-draw bootstrap
    # weight — coherent with the point estimate's mark). For the v2 estimator the
    # point dict carries its internals under "_v2"; build the v2 refit_fn so the MC
    # band is the forward-HBI re-solve per draw (warm-started at the tilted optimum).
    refit_fn = None
    if estimator_fn is not estimate_f_b and isinstance(point, dict) and "_v2" in point:
        from CDDF_analysis.cddf_catalog_hbi import make_v2_refit_fn
        v2int = point["_v2"]
        refit_fn = make_v2_refit_fn(
            cfg, v2int, logN_lo, logN_hi, N_b, dN_b,
            v2int["z_edges_fine"], v2int["M_meta"], mm)
    elif estimator_fn is not estimate_f_b and isinstance(point, dict) and "_v3x" in point:
        # v3 PARAMETRIC: build the per-draw θ-refit closure so the WALL-2 MC band is
        # the parametric re-solve (NOT the v1 1/Vmax fallback — numerical Finding 2).
        from CDDF_analysis.cddf_catalog_hbi import make_v3x_refit_fn
        refit_fn = make_v3x_refit_fn(cfg, point["_v3x"], mm)
    # ASSERT the parametric path did not silently fall through to the v1 MC (a v3 point
    # closure divided by a v1 σ would make the verdict spurious — numerical Finding 2).
    if estimator_fn is not estimate_f_b and getattr(cfg, "_wall1_estimator", None) == "v3" \
            and refit_fn is None:
        raise RuntimeError("v3 WALL-1: refit_fn is None (point has no '_v3x' internals); "
                           "the MC band would wrongly use the v1 1/Vmax σ.")
    mc = joint_mc_errors(
        cat_cut, is_TP, good_mask, mm, fp_model, X_tot,
        logN_lo, logN_hi, N_b, dN_b, truth_cut, cfg, rng,
        tilt_weights_op=w_op, refit_fn=refit_fn,
    )

    limits = cfg.report_logN_limits

    # closure prediction R0 · truth^tilt per reduction
    pred_f = baseline["R0_f"] * np.asarray(ttr["f_truth"])

    # ---- f_b pulls (per logN bin, z-marginalized) ----
    f_pull = _pull(point["f_b"], mc["f_b"]["std"], pred_f)
    f_pull_raw = _pull(point["f_b"], mc["f_b"]["std"], ttr["f_truth"])

    # ---- dN/dX(z) pulls per (limit, z-bin) ----
    dndx_z_pull = {}
    dndx_z_pull_raw = {}
    dndx_z_cov = {}
    for lim in limits:
        est = point["dndx_z"][lim]
        std = mc["dndx_z"][lim]["std"]
        tt = np.asarray(ttr["dndx_z"][lim])
        pred = baseline["R0_dndx_z"][lim] * tt
        dndx_z_pull[lim] = _pull(est, std, pred)
        dndx_z_pull_raw[lim] = _pull(est, std, tt)
        # coverage of the closure prediction inside the tilted-estimate MC band
        cov, n_used, inside = _coverage(mc["_samples"]["dndx_z"][lim], pred)
        dndx_z_cov[lim] = dict(coverage=cov, n_used=n_used, inside=inside.tolist())

    # ---- dN/dX_total + Ω pulls per limit (the HEADLINE gated statistic) ----
    # Store the closure PREDICTION scalars (R0_total·truth^tilt) so the gate reads
    # them directly for the integrated-coverage check (no fragile est−pull·std
    # reconstruction).
    dndx_tot_pull = {}
    dndx_tot_pull_raw = {}
    dndx_tot_pred = {}
    omega_pull = {}
    omega_pull_raw = {}
    omega_pred = {}
    for lim in limits:
        pred_d = baseline["R0_dndx_total"][lim] * ttr["dndx_total"][lim]
        dndx_tot_pred[lim] = float(pred_d)
        dndx_tot_pull[lim] = float(_pull(
            point["dndx_total"][lim], mc["dndx_total"][lim]["std"], pred_d))
        dndx_tot_pull_raw[lim] = float(_pull(
            point["dndx_total"][lim], mc["dndx_total"][lim]["std"],
            ttr["dndx_total"][lim]))
        pred_o = baseline["R0_omega"][lim] * ttr["omega"][lim]
        omega_pred[lim] = float(pred_o)
        omega_pull[lim] = float(_pull(
            point["omega"][lim], mc["omega"][lim]["std"], pred_o))
        omega_pull_raw[lim] = float(_pull(
            point["omega"][lim], mc["omega"][lim]["std"], ttr["omega"][lim]))

    # ---- coverage of the closure prediction on the z-marg f_b (diagnostic) ----
    f_cov, f_cov_n, f_cov_inside = _coverage(mc["_samples"]["f_b"], pred_f)

    return dict(
        dalpha=dalpha,
        point=point, mc=mc, ttr=ttr, pred_f=pred_f,
        f_pull=f_pull, f_pull_raw=f_pull_raw,
        f_cov=f_cov, f_cov_n=f_cov_n, f_cov_inside=f_cov_inside,
        dndx_z_pull=dndx_z_pull, dndx_z_pull_raw=dndx_z_pull_raw, dndx_z_cov=dndx_z_cov,
        dndx_tot_pull=dndx_tot_pull, dndx_tot_pull_raw=dndx_tot_pull_raw,
        dndx_tot_pred=dndx_tot_pred,
        omega_pull=omega_pull, omega_pull_raw=omega_pull_raw, omega_pred=omega_pred,
        w_op=w_op,
    )


# -----------------------------------------------------------------------------
# 6. The gate: combine both tilts → PASS / FAIL
# -----------------------------------------------------------------------------
def evaluate_gate(res_plus: dict, res_minus: dict, logN_lo: np.ndarray,
                  nominal_coverage: float = 0.95,
                  pull_gate_logN: float = PULL_GATE_LOGN,
                  report_limits=None, logN_hi=None, N_b=None, dN_b=None,
                  H0: float = 70.0, drop_top_above: float = 22.4,
                  estimator: str = "v1", baseline=None,
                  closure_R0_mode: str = "divide", deep_band_tol: float = 0.10,
                  band_ess: dict = None, band_ess_kill: float = 30.0) -> dict:
    """WALL-1 PASS/FAIL on the two tilts (spec §7).

    HEADLINE GATE = the INTEGRATED dN/dX & Ω closure (per report limit), NOT the
    differential per-bin f_b. RATIONALE (code-review #2 / LyA-review Finding 4):
    the closure prediction ``pred = R0·truth^tilt`` is *migration-blind* — R0 is a
    per-bin multiplicative factor measured at Δα=0, but ``est^tilt`` sums detections
    by PREDICTED N while weighting by TRUE-host N. A predicted bin that contains
    up-migrated lower-N systems therefore has ``est^tilt/pred ≈ 0.79`` on +tilt by
    construction (the migrants carry their smaller true-N tilt weight while pred
    scales by the bin-center weight). So a *differential* f_b closure pull CANNOT
    pass whenever N-migration is present, INDEPENDENT of estimator quality — that is
    the v1 migration signature (v2's job to deconvolve, spec §5/§9), NOT the b_FP
    signature WALL-1 must catch. The differential pull/coverage is therefore reported
    as a DIAGNOSTIC, never gated. The N-INTEGRATED dN/dX & Ω cancel the WITHIN-tier
    N-migration (a detection that up-migrates within the band is still counted) and
    are the cleaner statistic — BUT they do NOT cancel migration ACROSS the reporting
    boundary (sub-DLA→DLA up-migration at logN=20.3), so at a migration boundary the
    integrated dN/dX/Ω can still mis-close under v1. Empirically (2LPT-0, ρ≈0.99 DLA
    tier) the integrated dN/dX(≥20.3) FAILs with an opposite-sign coherent pull whose
    UP-tilt raw closure is near-zero (after the F1 host-floor fix) while the DOWN-tilt
    is large — the steep-f(N) Eddington / boundary up-migration signature. A FAIL
    classifier (below) labels this MIGRATION_EXPECTED_V1 vs a true b_FP
    misspecification; it does NOT flip the verdict (v1 cannot certify a tilt closure
    at a migration boundary — that is v2's job).

      PASS = for EACH report limit: integrated dN/dX & Ω closure |pull|≤3 on BOTH
             tilts AND integrated coverage ≥ nominal AND no opposite-sign coherent
             integrated pull between tilts (the slope-dependent b_FP-bias signature).

    All pulls are CLOSURE-residual pulls (est^tilt − R0·truth^tilt)/σ — the baseline
    absolute bias R0 is divided out (it is the §8 anchor / A2-gate's business, not
    WALL-1's; spec §9). The gate floor ``pull_gate_logN`` is max(19.5, matrix-floor).

    A differential FAIL below the floor is EXPECTED (band must WIDEN/clip, not
    confidently miss) and is reported as a diagnostic, never a hard fail.
    """
    gated = logN_lo >= pull_gate_logN - 1e-9
    below = logN_lo < pull_gate_logN - 1e-9
    if report_limits is None:
        report_limits = sorted(res_plus["dndx_tot_pull"].keys())

    checks = {}
    checks["pull_gate_logN_effective"] = float(pull_gate_logN)
    checks["headline_gate"] = "integrated_dndx_omega"
    fail_reasons = []

    # =====================================================================
    # HEADLINE GATE — integrated dN/dX & Ω closure (migration-insensitive)
    # =====================================================================
    def _mc_cov_scalar(samp_plus_or_minus_arr, pred_scalar):
        """Coverage = is the closure prediction inside the [2.5,97.5] MC band."""
        arr = np.asarray(samp_plus_or_minus_arr, dtype=float)
        lo = np.nanpercentile(arr, 2.5)
        hi = np.nanpercentile(arr, 97.5)
        if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(pred_scalar)):
            return np.nan, (lo, hi)
        return float(lo <= pred_scalar <= hi), (lo, hi)

    for lim in report_limits:
        for qty, pull_key in (("dndx", "dndx_tot_pull"), ("omega", "omega_pull")):
            pp = float(res_plus[pull_key][lim])
            pm = float(res_minus[pull_key][lim])
            checks[f"{qty}_closure_pull_{lim}_plus"] = pp
            checks[f"{qty}_closure_pull_{lim}_minus"] = pm
            for tag, pv in (("plus", pp), ("minus", pm)):
                if np.isfinite(pv) and abs(pv) > PULL_THRESHOLD:
                    fail_reasons.append(
                        f"[{tag} tilt] integrated {qty}(>={lim}) closure |pull|="
                        f"{abs(pv):.2f} > {PULL_THRESHOLD:.0f}")
            # opposite-sign coherent (single integrated statistic; threshold k·σ,
            # σ_mean=1 for one cell → k = COHERENT_PULL_K)
            opp = (np.isfinite(pp) and np.isfinite(pm) and (pp * pm < 0)
                   and abs(pp) > COHERENT_PULL_K and abs(pm) > COHERENT_PULL_K)
            checks[f"{qty}_opposite_sign_{lim}"] = bool(opp)
            if opp:
                fail_reasons.append(
                    f"opposite-sign coherent integrated {qty}(>={lim}) closure pull "
                    f"(slope-dependent b_FP/migration mis-closure): +tilt={pp:.2f}, "
                    f"−tilt={pm:.2f} (|·|>{COHERENT_PULL_K:.0f})")

    # --- integrated closure coverage: is the closure prediction (R0·truth^tilt)
    # inside the tilted-estimate 95% MC band? (single 0/1 per tier integral) ---
    for lim in report_limits:
        for qty, pred_key in (("dndx_total", "dndx_tot_pred"),
                              ("omega", "omega_pred")):
            for tag, res in (("plus", res_plus), ("minus", res_minus)):
                pred = float(res[pred_key][lim])
                cov, (blo, bhi) = _mc_cov_scalar(
                    res["mc"]["_samples"][qty][lim], pred)
                checks[f"{qty}_cov_{lim}_{tag}"] = cov
                if np.isfinite(cov) and cov < 1.0 - 1e-9:
                    # 0 means the closure prediction fell OUTSIDE the 95% band — a
                    # coverage miss for this tier integral.
                    fail_reasons.append(
                        f"[{tag} tilt] integrated {qty}(>={lim}) closure prediction "
                        f"{pred:.4g} outside 95% MC band [{blo:.4g},{bhi:.4g}] "
                        f"(coverage miss)")

    # =====================================================================
    # DIAGNOSTIC (reported, NOT gated) — differential per-bin f_b closure
    # =====================================================================
    # v1 is migration-blind on the differential f_b (see docstring): per-bin closure
    # pulls are EXPECTED to fail where N-migration is present. Surfaced so a reader
    # can SEE the migration signature, but it does NOT flip the verdict.
    for tag, res in (("plus", res_plus), ("minus", res_minus)):
        fp = res["f_pull"]
        pred = np.asarray(res["pred_f"])
        considered = gated & np.isfinite(fp) & np.isfinite(pred) & (pred > 0)
        bad = considered & (np.abs(fp) > PULL_THRESHOLD)
        checks[f"DIAG_f_pull_max_gefloor_{tag}"] = (
            float(np.nanmax(np.abs(fp[considered]))) if considered.any() else np.nan)
        checks[f"DIAG_f_pull_n_exceed_{tag}"] = int(bad.sum())
        samp = res["mc"]["_samples"]["f_b"]
        lo = np.nanpercentile(samp, 2.5, axis=0)
        hi = np.nanpercentile(samp, 97.5, axis=0)
        usable = gated & np.isfinite(pred) & (pred > 0) & np.isfinite(lo) & np.isfinite(hi)
        inside = (pred >= lo) & (pred <= hi) & usable
        checks[f"DIAG_f_cov_gefloor_{tag}"] = (
            float(inside[usable].mean()) if usable.any() else np.nan)
        checks[f"DIAG_f_cov_gefloor_n_{tag}"] = int(usable.sum())

    # differential coherent mean pull (DIAGNOSTIC — the migration signature, NOT gated)
    fp_p = res_plus["f_pull"]
    fp_m = res_minus["f_pull"]
    predp = np.asarray(res_plus["pred_f"])
    predm = np.asarray(res_minus["pred_f"])
    common = gated & np.isfinite(fp_p) & np.isfinite(fp_m) & (predp > 0) & (predm > 0)
    n_cells = int(common.sum())
    mean_pull_plus = float(np.nanmean(fp_p[common])) if common.any() else np.nan
    mean_pull_minus = float(np.nanmean(fp_m[common])) if common.any() else np.nan
    checks["DIAG_coherent_mean_pull_plus_gefloor"] = mean_pull_plus
    checks["DIAG_coherent_mean_pull_minus_gefloor"] = mean_pull_minus
    checks["DIAG_n_cells_gefloor"] = n_cells
    # the n-derived coherent threshold (LyA-4): k/√n on the mean of n cells
    coh_thr = COHERENT_PULL_K / np.sqrt(max(n_cells, 1))
    checks["DIAG_coherent_threshold"] = float(coh_thr)
    checks["DIAG_opposite_sign_coherent_diff"] = bool(
        np.isfinite(mean_pull_plus) and np.isfinite(mean_pull_minus)
        and (mean_pull_plus * mean_pull_minus < 0)
        and (abs(mean_pull_plus) > coh_thr) and (abs(mean_pull_minus) > coh_thr))

    # --- below-floor diagnostic: band must WIDEN/clip, not confidently miss ---
    # 'confidently miss' = closure |pull|>3 with a band that does NOT reach the
    # prediction even within a wide 2.5-97.5 interval AND a non-collapsed band.
    # EXPECTED failures here are tolerated (band widens / clips). Only meaningful
    # when the matrix floor is below the gate floor (else there are no below bins).
    below_diag = {}
    for tag, res in (("plus", res_plus), ("minus", res_minus)):
        fp = res["f_pull"]
        pred = np.asarray(res["pred_f"])
        samp = res["mc"]["_samples"]["f_b"]
        lo = np.nanpercentile(samp, 2.5, axis=0)
        hi = np.nanpercentile(samp, 97.5, axis=0)
        band_w = hi - lo
        considered = below & np.isfinite(fp) & np.isfinite(pred) & (pred > 0)
        inside = (pred >= lo) & (pred <= hi)
        clips = (lo <= C_FLOOR) | (band_w / np.maximum(np.abs(hi), 1e-300) > 0.9)
        widened_ok = considered & (inside | clips)
        confident_miss = considered & ~inside & ~clips & (np.abs(fp) > PULL_THRESHOLD)
        below_diag[tag] = dict(
            n_considered=int(considered.sum()),
            n_widened_ok=int(widened_ok.sum()),
            n_confident_miss=int(confident_miss.sum()),
            confident_miss_logN=[f"{logN_lo[b]:.2f}" for b in np.where(confident_miss)[0]],
        )

    # =====================================================================
    # FAIL CLASSIFIER — migration (expected v1) vs b_FP misspecification
    # =====================================================================
    # code-review Finding 2: the gate as written cannot, by itself, distinguish the
    # EXPECTED v1 migration non-closure from a b_FP misspecification. We add the
    # discriminant rather than silently passing. v1 is migration-blind (spec §5/§9):
    # at the 20.3 reporting boundary, sub-DLA→DLA up-migration makes est^tilt (binned
    # by PREDICTED N, weighted by TRUE host) mis-close vs a truth target binned by
    # TRUE N — integrating over N within the band does NOT cancel migration ACROSS
    # the boundary. The signature: (a) opposite-sign coherent integrated pull, AND
    # (b) the up-tilt RAW closure is markedly better than the down-tilt (the +tilt
    # down-weights the steep-f(N) tail that v1 over-counts; −tilt amplifies it), AND
    # (c) it lives in the DLA tier where ρ≈0.99 so forest-FP contamination is <0.05%
    # (a b_FP misspecification cannot drive a ρ≈0.99 tier this hard). A b_FP FAIL
    # would instead surface in the low-ρ sub-DLA/LLS cells (and would NOT improve
    # under the F1 host-floor decoupling). This classifier is INFORMATIONAL — it does
    # NOT flip the PASS/FAIL verdict (a FAIL stays a FAIL; v1 simply cannot certify a
    # <v2 tilt closure at a migration boundary — that is v2's job, spec §5).
    # DEEP-TIER BOUNDARY TEST (LyA-review Finding 3 — the decisive discriminant for
    # v2). The Ω closure-residual fraction vs deep limit (20.3 / 20.6 / 21.0) tells
    # boundary up-migration (peaks at the boundary, SHRINKS deep) apart from a forward-
    # MAP steep-tail slope over-response (GROWS monotonically into the migration-free
    # deep tail). Cheap: integrates the per-bin point f_b / pred_f arrays run_one_tilt
    # already returns; no extra tilt run. Only meaningful when N_b/dN_b are passed.
    deep = None
    if N_b is not None and dN_b is not None and logN_hi is not None:
        # For v3 (parametric continuous f(N)) push the deep-tier test FURTHER into the
        # under-sampled tail (21.5), where bplcut's global exp-cutoff over-cut and where
        # the v2 free-bin over-response was worst. A body-anchored penalized spline must
        # keep the residual FLAT all the way out — that is the genuine deep-tail PASS.
        _deep_cand = (20.3, 20.6, 21.0, 21.5) if estimator == "v3" else (20.3, 20.6, 21.0)
        deep_limits = tuple(L for L in _deep_cand
                            if L >= float(pull_gate_logN) - 1e-9)
        if len(deep_limits) >= 2:
            deep = _deep_tier_discriminant(
                res_plus, res_minus, logN_lo, logN_hi, N_b, dN_b, H0,
                drop_top_above, deep_limits=deep_limits, baseline=baseline,
                closure_R0_mode=closure_R0_mode)
            checks["DEEP_omega_resid_frac_plus"] = {
                f"{L}": deep["resid_frac_plus"][L] for L in deep["deep_limits"]}
            checks["DEEP_omega_resid_frac_minus"] = {
                f"{L}": deep["resid_frac_minus"][L] for L in deep["deep_limits"]}
            checks["DEEP_dominant_tilt"] = deep["dominant_tilt"]
            checks["DEEP_omega_resid_trend_boundary_to_deep"] = deep["trend"]
            # numerical Finding 12: a "flat" trend with UNIFORMLY LARGE residuals is
            # still a FAIL. Record the max |deep resid| over both tilts; the v3 PASS
            # requires trend in {flat,shrinks_deep} AND this magnitude <= deep_band_tol.
            _deepmags = [abs(v) for v in list(deep["resid_frac_plus"].values())
                         + list(deep["resid_frac_minus"].values()) if np.isfinite(v)]
            checks["DEEP_omega_resid_max_abs"] = float(max(_deepmags)) if _deepmags else np.nan
            checks["DEEP_band_tol"] = float(deep_band_tol)

    # DIFFERENTIAL (per-N) deep-tier discriminant — the CORRECT v3 grows-deep test
    # (4-lens review: bayesian F1, numerical F4/F6, cs F4). The cumulative-Ω trend
    # above drifts with L and folds in the over-stiffened >21.5 tail (manufacturing a
    # spurious grows_deep even when the body per-N residual is flat). The differential
    # body trend isolates the genuine v2 MAP_SLOPE_OVERRESPONSE per-bin over-response.
    deep_diff = None
    if (estimator == "v3" and baseline is not None and N_b is not None
            and logN_hi is not None):
        deep_diff = _deep_tier_differential_discriminant(
            res_plus, res_minus, logN_lo, logN_hi, baseline,
            closure_R0_mode=closure_R0_mode)
        checks["DEEP_DIFF_trend"] = deep_diff["trend"]
        checks["DEEP_DIFF_body_slope"] = float(deep_diff["body_slope"]) \
            if np.isfinite(deep_diff["body_slope"]) else None
        checks["DEEP_DIFF_body_max_abs"] = float(deep_diff["body_max_abs"]) \
            if np.isfinite(deep_diff["body_max_abs"]) else None
        checks["DEEP_DIFF_dominant_tilt"] = deep_diff["dominant_tilt"]
        checks["DEEP_DIFF_range"] = deep_diff["deep_range"]

    # =====================================================================
    # v3 HARD GATES (4-lens review BLOCKERS — wired into fail_reasons, not just
    # the informational classifier). For the PARAMETRIC estimator the deep-tier
    # growth IS the v2 signature v3 must eliminate, and an untilted-misfit family
    # makes the whole closure uninterpretable. Both must FLIP the verdict.
    # =====================================================================
    if estimator == "v3":
        # (4-lens review: bayesian F1, numerical F4/F6, cs F4) the v2
        # MAP_SLOPE_OVERRESPONSE signature is a DIFFERENTIAL per-N over-response that
        # grows into the migration-free body. Fire grows_deep on the DIFFERENTIAL body
        # trend, NOT the cumulative-Ω integral (which drifts with L and folds in the
        # over-stiffened >21.5 tail -> spurious grows_deep even when the body is flat).
        # The cumulative trend is retained ONLY as an informational record.
        if deep_diff is not None:
            if deep_diff.get("trend") == "grows_deep":
                fail_reasons.append(
                    "[v3] DIFFERENTIAL per-N deep-tier residual GROWS across the "
                    "migration-free body (slope "
                    f"{deep_diff.get('body_slope'):+.3f}/dex on the {deep_diff['dominant_tilt']} "
                    "tilt) — the v2 MAP_SLOPE_OVERRESPONSE per-bin over-response PERSISTS "
                    "for this rung (CLIMB the DOF ladder / adjust EDF). Do NOT ship.")
        # (bayesian F4 / cs F1) model-independent BODY closure tolerance. A parametric
        # pull-gate at ~1% σ is a near-exact-closure demand; report+gate on the
        # resid_frac too. The v2 signature being GONE (flat body) does NOT license a
        # PASS if the body still mis-closes by a fixed fraction (the frozen-kernel
        # slope-dependence). This is the binding criterion alongside the pull.
        if deep_diff is not None:
            bmax = deep_diff.get("body_max_abs", np.nan)
            checks["DEEP_DIFF_body_tol"] = float(deep_band_tol)
            if np.isfinite(bmax) and bmax > deep_band_tol:
                fail_reasons.append(
                    f"[v3] body per-N |resid_frac|={bmax:.3f} > tol {deep_band_tol:.2f} "
                    "over the migration-free body — the parametric family does not close "
                    "the tilt to tolerance (diagnose: opposite-sign => frozen-kernel "
                    "slope-dependence, NOT an f(N) DOF problem; same-sign => residual bias).")
        # (Bayesian F3 / lya F4 / numerical F5/F6) untilted R0!=1 precondition: a closure
        # divided by R0 != 1 is an R0-rescaling artifact, not a tilt-robustness test.
        # A WALL-1 PASS on an untilted-misfit family would be SPURIOUS. Hard-fail it.
        # EXTENDED to the DEEP limits the deep-tier test integrates (numerical F6): if the
        # untilted deep R0 is far from 1 there (the tail over-stiffening / edge collapse),
        # the deep-tier residual is itself an R0-rescaling artifact and must be flagged.
        R0_PRECOND_TOL = 0.10
        DEEP_R0_PRECOND_TOL = 0.15
        if baseline is not None:
            r0_off = []
            for lim in report_limits:
                for qty, key in (("dndx", "R0_dndx_total"), ("omega", "R0_omega")):
                    try:
                        r0 = float(baseline[key][lim])
                    except Exception:
                        r0 = np.nan
                    checks[f"untilted_R0_{qty}_{lim}"] = r0
                    if np.isfinite(r0) and abs(r0 - 1.0) > R0_PRECOND_TOL:
                        r0_off.append(f"{qty}(>={lim}) R0={r0:.3f}")
            checks["v3_untilted_R0_precond_tol"] = float(R0_PRECOND_TOL)
            if r0_off:
                fail_reasons.append(
                    "[v3] untilted family MISFITS truth (|R0-1|>"
                    f"{R0_PRECOND_TOL:.2f}: {', '.join(r0_off)}) — the WALL-1 closure "
                    "is NOT evaluable (R0-division artifact); fit/support is wrong or "
                    "the rung misfits the untilted population. NOT a tilt-robustness "
                    "result. CLIMB the ladder until R0~1 before reading WALL-1.")
            # deep-limit untilted R0 (the limits the cumulative deep-tier integrates).
            # Recorded as a WARN-flag in checks; gates the DEEP cumulative readability.
            if N_b is not None and logN_hi is not None and deep is not None:
                K = omega_hi_prefactor(H0)
                e0f = np.asarray(baseline["e0"]["f_b"], float)
                t0f = np.asarray(baseline["t0"]["f_truth"], float)
                deep_r0 = {}; deep_r0_off = []
                for lim in deep.get("deep_limits", ()):
                    sel = (logN_lo >= lim - 1e-9) & (logN_hi <= drop_top_above + 1e-9)
                    o_e0 = K * float(np.nansum(N_b[sel] * e0f[sel] * dN_b[sel]))
                    o_t0 = K * float(np.nansum(N_b[sel] * t0f[sel] * dN_b[sel]))
                    r0 = (o_e0 / o_t0) if o_t0 > 0 else np.nan
                    deep_r0[f"{lim}"] = r0
                    if np.isfinite(r0) and abs(r0 - 1.0) > DEEP_R0_PRECOND_TOL:
                        deep_r0_off.append(f"Ω(>={lim}) R0={r0:.3f}")
                checks["DEEP_untilted_R0_omega"] = deep_r0
                checks["DEEP_untilted_R0_precond_tol"] = float(DEEP_R0_PRECOND_TOL)
                checks["DEEP_untilted_R0_off"] = deep_r0_off
                # informational: do NOT hard-fail on this (the headline gate is the
                # report-limit R0 + the differential body trend); but if the deep R0 is
                # off it means the CUMULATIVE deep-tier number is not cleanly evaluable.
                checks["DEEP_cumulative_evaluable"] = (len(deep_r0_off) == 0)

    classifier = "n/a"
    if fail_reasons:
        any_opp = any(checks.get(f"{q}_opposite_sign_{lim}", False)
                      for lim in report_limits for q in ("dndx", "omega"))
        raw_p = [abs(res_plus["dndx_tot_pull_raw"][lim]) for lim in report_limits]
        raw_m = [abs(res_minus["dndx_tot_pull_raw"][lim]) for lim in report_limits]
        rawp = float(np.nanmean(raw_p)) if raw_p else np.nan
        rawm = float(np.nanmean(raw_m)) if raw_m else np.nan
        checks["raw_dndx_pull_mean_plus"] = rawp
        checks["raw_dndx_pull_mean_minus"] = rawm
        asym = np.isfinite(rawp) and np.isfinite(rawm) and (rawm > 2.0 * max(rawp, 0.5))
        deep_trend = deep["trend"] if deep is not None else "n/a"

        if estimator == "v2":
            # For v2 the boundary-migration story is FALSIFIABLE — the symmetric kernel
            # has already deconvolved the UNTILTED bias (R0→~1.0), so a persisting tilt
            # break is EITHER the A2-gated boundary up-migration (would shrink deep) OR a
            # slope-generalization failure of the MAP (grows deep). The deep-tier test
            # decides; do NOT inherit v1's MIGRATION_EXPECTED label by default.
            if any_opp and deep_trend == "grows_deep":
                classifier = (
                    "MAP_SLOPE_OVERRESPONSE_V2 (opposite-sign coherent AND Ω closure "
                    "residual GROWS into the migration-free deep tail (≥21.0 worse than "
                    "≥20.3): the boundary up-migration explanation is FALSIFIED — this "
                    "is a slope-dependent over-response of the forward regularized MAP "
                    "on the steep high-N tail, NOT the A2-gated boundary piece. The "
                    "deconvolution is correct AT the true slope (in-sample) but does NOT "
                    "generalize to the tilted slope — exactly the circularity WALL-1 "
                    "breaks. A posterior kernel carrying only prior-edge skew will NOT "
                    "fix it; the smoothness prior + ill-conditioned rare-bin deconv are "
                    "co-responsible. DEMOTE v2; v1 1/Vmax + an A2-quoted Eddington "
                    "correction is the shippable headline — see LyA-review Finding 3.")
            elif any_opp and deep_trend == "shrinks_deep":
                classifier = (
                    "MIGRATION_BOUNDARY_V2 (opposite-sign coherent AND Ω closure "
                    "residual SHRINKS deep: consistent with A2-gated sub-DLA→DLA "
                    "up-migration across 20.3 that the symmetric kernel cannot remove; "
                    "distinguish in the writeup, do NOT claim removed — the posterior-"
                    "kernel swap is the spec §9 escape hatch.)")
            elif any_opp:
                classifier = (
                    "OPPOSITE_SIGN_DEEP_FLAT_V2 (opposite-sign coherent, deep-tier trend "
                    f"={deep_trend}: neither a clean boundary-shrink nor a deep-growth; "
                    "the residual is broad-band. Inspect the f_b deconvolution + λ "
                    "sensitivity before quoting any v2 number.)")
            else:
                classifier = (
                    "SAME-SIGN/INCOHERENT FAIL_V2 (not the opposite-sign signature; "
                    "likely a normalization / R0 / multistart-degeneracy issue — inspect "
                    "multistart_logP_spread and the identity-draw self-check.)")
        elif estimator == "v3":
            # v3 PARAMETRIC: a low-DOF continuous f(N|θ) cannot over-respond per-bin to
            # a tilt — the win condition is the v2 MAP_SLOPE_OVERRESPONSE_V2 signature
            # being GONE. The CORRECT discriminant is the DIFFERENTIAL per-N body trend
            # (4-lens review), NOT the cumulative-Ω trend (which drifts with L + folds in
            # the over-stiffened >21.5 tail). A FAIL is classified by WHY:
            #   diff grows_deep => the per-bin over-response PERSISTS -> climb/adjust EDF.
            #   diff flat + opposite-sign => the v2 signature is GONE; the residual is the
            #       FROZEN-KERNEL slope-dependence (orthogonal to f(N) DOF — numerical F3).
            #   diff flat + same-sign  => a uniform tilt-induced bias.
            diff_trend = deep_diff["trend"] if deep_diff is not None else "n/a"
            if diff_trend == "grows_deep":
                classifier = (
                    "V3_GROWS_DEEP_STILL_OVERRESPONDS (the DIFFERENTIAL per-N residual "
                    "GROWS across the migration-free body — the v2 MAP_SLOPE_OVERRESPONSE "
                    "per-bin over-response is NOT gone for this rung. The parametric family "
                    "is still over-flexible/mis-shaped: CLIMB the DOF ladder / lower the "
                    "EDF. Do NOT ship this rung.)")
            elif diff_trend == "flat" and any_opp:
                classifier = (
                    "V3_KERNEL_SLOPE_DEPENDENCE (the v2 per-bin over-response is GONE — the "
                    "DIFFERENTIAL per-N body residual is FLAT — but the integrated closure "
                    "still mis-closes with an OPPOSITE-SIGN coherent pull. This is the "
                    "FROZEN-KERNEL slope-dependence: R_emp(N̂|N,SNR) was built at the "
                    "untilted slope, so a tilted true slope changes the effective Eddington "
                    "correction the same frozen kernel applies. ORTHOGONAL to f(N) DOF — a "
                    "richer family will NOT fix it (numerical Finding 3). NOT shippable as "
                    "a tilt-robust number, but the v2 deep-tail over-response IS resolved.)")
            elif diff_trend == "flat":
                classifier = (
                    "V3_TILT_BIAS_FLAT (the v2 deep-growth is GONE — DIFFERENTIAL body flat "
                    "— but a same-sign integrated |pull|>3 / coverage-miss remains: a "
                    "uniform tilt-induced bias (check the untilted R0; if |R0-1|>0.03 the "
                    "family misfits the untilted truth -> fix support/edge, not EDF).)")
            else:
                classifier = (
                    "V3_FAIL_INDETERMINATE (differential trend n/a; inspect the integrated "
                    "pulls + multistart spread + at-bound flags.)")
        else:
            # v1 (no deconvolution): the boundary up-migration explanation stands
            # unless the deep-tier test refutes it.
            if any_opp and asym and deep_trend != "grows_deep":
                classifier = ("MIGRATION_EXPECTED_V1 (opposite-sign coherent + up-tilt "
                              "raw closure << down-tilt: the steep-f(N) Eddington / "
                              "sub-DLA→DLA up-migration v1 does not deconvolve — spec "
                              "§5/§9, v2's job; NOT a b_FP misspecification at this "
                              "ρ≈0.99 DLA tier)")
            elif any_opp and asym and deep_trend == "grows_deep":
                classifier = ("MIGRATION_EXPECTED_V1_DEEP_GROWTH (opposite-sign + "
                              "up-tilt-favoured, but the Ω residual GROWS deep — the "
                              "non-closure reaches beyond the boundary; v1 has no "
                              "deconvolution to remove it. The deep growth is the "
                              "uncorrected steep-tail Eddington scatter, expected for "
                              "v1; v2's symmetric kernel only partially removes it.)")
            elif any_opp:
                classifier = ("INDETERMINATE (opposite-sign coherent but NOT up-tilt-"
                              "favoured; inspect the low-ρ cells for a b_FP "
                              "misspecification before trusting v1 here)")
            else:
                classifier = ("SAME-SIGN/INCOHERENT FAIL (not the opposite-sign b_FP "
                              "signature; likely a normalization / R0 issue — inspect)")
    checks["fail_classifier"] = classifier

    # =====================================================================
    # BAND-ESS<30 KILL (gate doc §B; item 9). The per-object posterior-kernel ESS
    # restricted to each report tier (ESS_i(tier) = (Σ w/π)² / Σ(w/π)², stored in the
    # kernel npz ess_203/206/210) being below band_ess_kill at the tier level means
    # the differential band there is starvation-limited, NOT estimator-limited. Per
    # §B this DECLARES the differential f_b band UNCONSTRAINED in that tier and FALLS
    # BACK to the integrated Gehrels/Poisson limit — it is NOT a hard FAIL of the
    # integrated headline (the integrated dN/dX & Ω closure stays the verdict). We
    # record the per-tier band-ESS and the unconstrained tiers so the writer/headline
    # reports the differential band only where ESS >= the kill, Gehrels elsewhere.
    band_unconstrained = []
    if band_ess is not None:
        for tlim in sorted(band_ess.keys()):
            e = np.asarray(band_ess[tlim], dtype=float)
            e = e[np.isfinite(e) & (e > 0)]
            # BAND ESS = Σ_i ESS_i(tier) over the op objects with support in the tier
            # (the tier-level effective sample size the differential band leans on).
            band = float(e.sum()) if e.size else 0.0
            checks[f"band_ess_{tlim}"] = band
            checks[f"band_ess_{tlim}_n_obj_lt{int(band_ess_kill)}"] = int(
                np.sum(e < band_ess_kill)) if e.size else 0
            if band < band_ess_kill:
                band_unconstrained.append(float(tlim))
    checks["band_ess_kill_threshold"] = float(band_ess_kill)
    checks["differential_band_unconstrained_tiers"] = band_unconstrained
    if band_unconstrained:
        print(f"    [band-ESS KILL §B] differential f_b UNCONSTRAINED at tier(s) "
              f"{band_unconstrained} (band ESS < {band_ess_kill:.0f}); FALL BACK to the "
              f"integrated Gehrels/Poisson limit there (NOT a headline FAIL).")

    passed = len(fail_reasons) == 0
    return dict(
        passed=passed,
        fail_reasons=fail_reasons,
        fail_classifier=classifier,
        checks=checks,
        sub195_diag=below_diag,
        pull_gate_logN=float(pull_gate_logN),
        nominal_coverage=nominal_coverage,
        differential_band_unconstrained_tiers=band_unconstrained,
    )


# -----------------------------------------------------------------------------
# 7. Output writers
# -----------------------------------------------------------------------------
def write_tilt_outputs(cfg: HBIConfig, res_plus, res_minus, gate, baseline,
                       logN_lo, logN_hi, N_b, dN_b, X_tot, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    limits = cfg.report_logN_limits
    zbins = np.asarray(cfg.zbins, dtype=float)
    n_zbins = len(zbins) - 1
    gate_floor = gate["pull_gate_logN"]
    R0_f = np.asarray(baseline["R0_f"])

    # --- wall1_pulls_fN.csv : per-logN-bin pulls, both tilts ---
    p = os.path.join(out_dir, "wall1_pulls_fN.csv")
    with open(p, "w") as fh:
        fh.write("# WALL-1 per-logN-bin f(N) closure (2LPT-0). DIAGNOSTIC ONLY.\n")
        fh.write("# R0 = UNTILTED baseline recovery est0/truth0 (NOT 1.0 — the v1 "
                 "Eddington/migration absolute bias, spec 5/9; divided out here).\n")
        fh.write("# pull_CLOSURE = (est_tilt - R0*truth_tilt)/sigma_MC. The DIFFERENTIAL "
                 "per-bin f_b closure is migration-blind under v1 (est sums by PREDICTED "
                 "N, weights by TRUE-host N) -> EXPECTED to fail under N-migration; it is "
                 "NOT the gated statistic. The HEADLINE GATE is the INTEGRATED dN/dX & Ω "
                 "closure (see wall1_result.tsv).\n")
        fh.write("# pull_RAW = (est_tilt - truth_tilt)/sigma_MC  <-- re-measures the "
                 "absolute baseline bias under tilt (NOT gated; transparency only).\n")
        fh.write(f"# NOTE: these per-bin pulls are reported for logN_lo>={gate_floor:.2f} "
                 f"as the migration DIAGNOSTIC, not a PASS/FAIL criterion.\n")
        fh.write("logN_lo,logN_hi,R0,"
                 "ftrue_plus,fpred_plus,fest_plus,fstd_plus,pull_closure_plus,pull_raw_plus,"
                 "ftrue_minus,fpred_minus,fest_minus,fstd_minus,pull_closure_minus,pull_raw_minus,"
                 "gated\n")
        predp = np.asarray(res_plus["pred_f"])
        predm = np.asarray(res_minus["pred_f"])
        for b in range(len(logN_lo)):
            gated = int(logN_lo[b] >= gate_floor - 1e-9)
            fh.write(
                f"{logN_lo[b]:.2f},{logN_hi[b]:.2f},{R0_f[b]:.4f},"
                f"{res_plus['ttr']['f_truth'][b]:.6e},{predp[b]:.6e},"
                f"{res_plus['point']['f_b'][b]:.6e},{res_plus['mc']['f_b']['std'][b]:.6e},"
                f"{res_plus['f_pull'][b]:.3f},{res_plus['f_pull_raw'][b]:.3f},"
                f"{res_minus['ttr']['f_truth'][b]:.6e},{predm[b]:.6e},"
                f"{res_minus['point']['f_b'][b]:.6e},{res_minus['mc']['f_b']['std'][b]:.6e},"
                f"{res_minus['f_pull'][b]:.3f},{res_minus['f_pull_raw'][b]:.3f},"
                f"{gated}\n")
    paths["pulls_fN"] = p

    # --- wall1_pulls_dndx_z.csv : per-(limit,zbin) pulls on dN/dX(z) ---
    p = os.path.join(out_dir, "wall1_pulls_dndx_z.csv")
    with open(p, "w") as fh:
        fh.write("# WALL-1 per-(limit, zbin) dN/dX closure. pull_closure=(est-R0*truth)/sigma_MC.\n")
        fh.write("limit,zbin_lo,zbin_hi,R0,"
                 "dndx_true_plus,dndx_est_plus,pull_closure_plus,pull_raw_plus,"
                 "dndx_true_minus,dndx_est_minus,pull_closure_minus,pull_raw_minus\n")
        for lim in limits:
            R0z = np.asarray(baseline["R0_dndx_z"][lim])
            for k in range(n_zbins):
                fh.write(
                    f"{lim},{zbins[k]:.2f},{zbins[k+1]:.2f},{R0z[k]:.4f},"
                    f"{res_plus['ttr']['dndx_z'][lim][k]:.6e},"
                    f"{res_plus['point']['dndx_z'][lim][k]:.6e},"
                    f"{res_plus['dndx_z_pull'][lim][k]:.3f},"
                    f"{res_plus['dndx_z_pull_raw'][lim][k]:.3f},"
                    f"{res_minus['ttr']['dndx_z'][lim][k]:.6e},"
                    f"{res_minus['point']['dndx_z'][lim][k]:.6e},"
                    f"{res_minus['dndx_z_pull'][lim][k]:.3f},"
                    f"{res_minus['dndx_z_pull_raw'][lim][k]:.3f}\n")
    paths["pulls_dndx_z"] = p

    # --- wall1_result.tsv : the gate verdict + summary checks ---
    p = os.path.join(out_dir, "wall1_result.tsv")
    with open(p, "w") as fh:
        fh.write("key\tvalue\n")
        fh.write(f"WALL1_VERDICT\t{'PASS' if gate['passed'] else 'FAIL'}\n")
        fh.write(f"WALL1_FAIL_CLASSIFICATION\t{gate.get('fail_classifier', 'n/a')}\n")
        fh.write("headline_gate\tintegrated_dndx_omega "
                 "(differential f_b is migration-blind DIAGNOSTIC, not gated)\n")
        fh.write(f"coherent_pull_k\t{COHERENT_PULL_K}\n")
        fh.write(f"nominal_coverage\t{gate['nominal_coverage']}\n")
        fh.write(f"dalpha\t+/-{abs(res_plus['dalpha'])}\n")
        fh.write(f"pull_gate_logN_effective\t{gate_floor}\n")
        fh.write(f"pull_threshold\t{PULL_THRESHOLD}\n")
        fh.write(f"n_mc\t{cfg.n_mc}\n")
        fh.write(f"fp_estimator\t{cfg.fp_estimator}\n")
        for lim in limits:
            fh.write(f"baseline_R0_dndx_{lim}\t{baseline['R0_dndx_total'][lim]:.4f}\n")
            fh.write(f"baseline_R0_omega_{lim}\t{baseline['R0_omega'][lim]:.4f}\n")
        for kk, vv in gate["checks"].items():
            fh.write(f"{kk}\t{vv}\n")
        for lim in limits:
            fh.write(f"dndx_total_closure_pull_{lim}_plus\t{res_plus['dndx_tot_pull'][lim]:.3f}\n")
            fh.write(f"dndx_total_closure_pull_{lim}_minus\t{res_minus['dndx_tot_pull'][lim]:.3f}\n")
            fh.write(f"dndx_total_raw_pull_{lim}_plus\t{res_plus['dndx_tot_pull_raw'][lim]:.3f}\n")
            fh.write(f"dndx_total_raw_pull_{lim}_minus\t{res_minus['dndx_tot_pull_raw'][lim]:.3f}\n")
            fh.write(f"omega_closure_pull_{lim}_plus\t{res_plus['omega_pull'][lim]:.3f}\n")
            fh.write(f"omega_closure_pull_{lim}_minus\t{res_minus['omega_pull'][lim]:.3f}\n")
        for tag in ("plus", "minus"):
            d = gate["sub195_diag"][tag]
            fh.write(f"belowfloor_n_confident_miss_{tag}\t{d['n_confident_miss']}\n")
            fh.write(f"belowfloor_n_widened_ok_{tag}\t{d['n_widened_ok']}\n")
            fh.write(f"belowfloor_n_considered_{tag}\t{d['n_considered']}\n")
            fh.write(f"belowfloor_confident_miss_logN_{tag}\t{','.join(d['confident_miss_logN'])}\n")
        for i, r in enumerate(gate["fail_reasons"]):
            fh.write(f"fail_reason_{i}\t{r}\n")
    paths["result"] = p

    gate_floor = gate["pull_gate_logN"]
    da_mag = abs(res_plus["dalpha"])

    # --- wall1_closure.png : estimate^tilt vs the CLOSURE PREDICTION R0*truth^tilt
    # (the gated target) AND the bare truth^tilt, both tilts ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        mid = 0.5 * (logN_lo + logN_hi)
        for ax, tag, res, dalpha in (
                (axes[0], "plus", res_plus, da_mag),
                (axes[1], "minus", res_minus, -da_mag)):
            ft = res["ttr"]["f_truth"]
            pr = np.asarray(res["pred_f"])
            fe = res["point"]["f_b"]
            fl = res["mc"]["f_b"]["q025"]
            fh_ = res["mc"]["f_b"]["q975"]
            show = (ft > 0) & np.isfinite(fe) & (logN_lo >= 18.0 - 1e-9)
            ylo = np.clip(fe[show] - fl[show], 0.0, None)
            yhi = np.clip(fh_[show] - fe[show], 0.0, None)
            ax.errorbar(mid[show], np.clip(fe[show], 1e-300, None),
                        yerr=[ylo, yhi], fmt="o", color="C0", ms=4,
                        label=r"estimate$^{tilt}$ (95% MC)")
            ax.plot(mid[show], np.clip(pr[show], 1e-300, None), "^-",
                    color="C2", ms=4, alpha=0.85,
                    label=r"closure target $R_0\cdot$truth$^{tilt}$ (GATED)")
            ax.plot(mid[show], np.clip(ft[show], 1e-300, None), "s-",
                    color="C3", ms=3, alpha=0.6, label=r"bare truth$^{tilt}$")
            ax.axvline(gate_floor, ls=":", color="k", lw=0.8,
                       label=f"gate floor {gate_floor:.1f}")
            ax.axvline(20.3, ls="--", color="k", lw=0.6)
            ax.set_yscale("log")
            ax.set_xlabel(r"$\log_{10} N_{\rm HI}$")
            ax.set_ylabel(r"$f(N_{\rm HI}, X)$")
            ax.set_title(f"WALL-1 tilt $\\Delta\\alpha={dalpha:+.1f}$")
            ax.legend(fontsize=7.5)
            ax.grid(alpha=0.3)
        verdict = "PASS" if gate["passed"] else "FAIL"
        fig.suptitle(f"WALL-1 tilt closure (2LPT-0) — {verdict}  "
                     f"(closure pull vs $R_0\\cdot$truth)", fontsize=13)
        pp = os.path.join(out_dir, "wall1_closure.png")
        fig.savefig(pp, dpi=120)
        plt.close(fig)
        paths["closure_png"] = pp
    except Exception as e:
        print(f"[plot] wall1_closure.png skipped: {e}")

    # --- wall1_pulls.png : closure pull vs logN, both tilts ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
        mid = 0.5 * (logN_lo + logN_hi)
        for tag, res, c, dalpha in (("plus", res_plus, "C0", da_mag),
                                    ("minus", res_minus, "C1", -da_mag)):
            fp = res["f_pull"]
            pr = np.asarray(res["pred_f"])
            show = np.isfinite(fp) & (pr > 0) & (logN_lo >= 18.0 - 1e-9)
            ax.plot(mid[show], fp[show], "o-", color=c, ms=4,
                    label=f"$\\Delta\\alpha={dalpha:+.1f}$ (closure)")
        ax.axhline(0, color="k", lw=0.6)
        ax.axhline(3, ls="--", color="r", lw=0.8)
        ax.axhline(-3, ls="--", color="r", lw=0.8, label="|pull|=3 gate")
        ax.axvline(gate_floor, ls=":", color="k", lw=0.8,
                   label=f"gate floor {gate_floor:.1f}")
        ax.set_xlabel(r"$\log_{10} N_{\rm HI}$")
        ax.set_ylabel(r"closure pull $= (f^{tilt}_{est} - R_0 f^{tilt}_{true})/\sigma_{MC}$")
        verdict = "PASS" if gate["passed"] else "FAIL"
        ax.set_title(f"WALL-1 closure pulls (2LPT-0) — {verdict}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        pp = os.path.join(out_dir, "wall1_pulls.png")
        fig.savefig(pp, dpi=120)
        plt.close(fig)
        paths["pulls_png"] = pp
    except Exception as e:
        print(f"[plot] wall1_pulls.png skipped: {e}")

    return paths


# -----------------------------------------------------------------------------
# 8. Runner
# -----------------------------------------------------------------------------
def run_wall1(cfg: HBIConfig, dalpha: float = DEFAULT_DALPHA,
              nominal_coverage: float = 0.95,
              host_truth_floor: float = 19.0,
              estimator: str = "v1", closure_R0_mode: str = "divide") -> dict:
    """Run the full WALL-1 gate on the configured 2LPT-0 catalog.

    Builds the v1 ingredients ONCE (catalog cut, frozen molly C/ρ counts, frozen
    FP model, pathlength), then runs the +dalpha and −dalpha tilts and the gate.

    ``host_truth_floor`` (CS-review F1, load-bearing): the NHI floor of the truth
    catalog used to attach each detection's tilt host mark (``NHI_TRUE``), DECOUPLED
    from the C/ρ matrix floor. Spec §2 WIRING requires the tilt host floor to be
    ≤19.0 near the 20.0–20.3 boundary: with the default floor-20.3 ``figures_molly``
    matrix, a 20.3-floored match labels ~18.2% of ≥20.3 op detections HOSTLESS
    (forest FP, tilt weight 1.0) when they are really sub-DLA up-migrants with a
    true host in [19,20.3] — corrupting the tilt deposit. Flooring the MATCH truth
    at ≤19.0 (default 19.0) gives those their correct ``10^(Δα·(N_true−20.3))``
    weight while the C/ρ count regen stays faithful to the matrix-floor TSV (the
    completeness denominator ``truth_cut`` is re-floored at the matrix floor; the
    cell-level ``min_true_nhi`` self-restricts the extra low-N hosts out of every
    matrix cell). MUST be ≤ the matrix floor to have any effect (clamped below).
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.rng_seed)

    print("[1] molly matrix")
    mm = load_molly_matrix(cfg.molly_tsv)
    truth_floor = float(mm.nhi_edges[0])
    # matrix-floor gate (same as the estimator): a report limit below the matrix
    # floor is invalid (1/C divides by a truth-floored denominator).
    valid_limits = tuple(L for L in cfg.report_logN_limits if L >= truth_floor - 1e-9)
    dropped = tuple(L for L in cfg.report_logN_limits if L < truth_floor - 1e-9)
    if dropped:
        print(f"    [WARN] molly matrix floor={truth_floor:.2f}; DROPPING invalid "
              f"report limit(s) {dropped}. Use figures_molly_nhi20/nhi19 to go lower.")
    if not valid_limits:
        raise SystemExit(
            f"All report limits {cfg.report_logN_limits} below matrix floor "
            f"{truth_floor:.2f}; choose figures_molly_nhi19 for the <19.5 WALL-1 test.")
    cfg.report_logN_limits = valid_limits

    qso_lookup = _build_qso_lookup(cfg)

    # CS-review F1: clamp the tilt-host floor to <= the matrix floor (it only adds
    # hosts BELOW the matrix floor; >= the floor is a no-op). The default 19.0 picks
    # up the sub-DLA up-migrants near the 20.0–20.3 boundary while keeping the C/ρ
    # regen faithful to the matrix-floor TSV.
    host_floor_eff = min(host_truth_floor, truth_floor)
    if host_floor_eff < truth_floor - 1e-9:
        print(f"    [F1] tilt-host truth floor = {host_floor_eff:.2f} (< matrix floor "
              f"{truth_floor:.2f}); sub-DLA up-migrants in [{host_floor_eff:.1f},"
              f"{truth_floor:.1f}) now carry their true tilt weight, not hostless=1.")

    print("[2] load + cut catalog (frozen)")
    cat_cut, truth_cut, is_TP, good_mask, meta = load_and_cut_catalog(
        cfg, truth_nhi_floor=truth_floor, qso_lookup=qso_lookup,
        host_truth_floor=host_floor_eff)
    print(f"    meta: {meta}")

    print("[3] molly count regen (FROZEN C/rho — unweighted)")
    mm = regenerate_molly_counts(mm, cat_cut, is_TP, truth_cut, good_mask, cfg)
    print(f"    purity max-abs-diff={mm._max_p_diff:.5f}, "
          f"completeness max-abs-diff={mm._max_c_diff:.5f}")
    if max(mm._max_p_diff, mm._max_c_diff) > 5e-3:
        raise SystemExit(
            f"molly cut-bundle replication FAILED (p={mm._max_p_diff:.4f}, "
            f"c={mm._max_c_diff:.4f} > 5e-3) — C/rho denominators do not match.")
    C_interp = make_C_interpolator(mm)
    rho_interp = make_rho_interpolator(mm)

    print("[4] pathlength (SNR-restricted, FROZEN)")
    cfg._wall1_estimator = estimator   # used by run_one_tilt's v3 refit_fn assert
    if estimator in ("v2", "v3"):
        X_tot, n_sl_used, qso_zlo, qso_zhi, qso_snr, Xcalc = build_pathlength(
            cfg, qso_lookup=qso_lookup, return_per_sl=True)
        qso_per_sl = (qso_zlo, qso_zhi, qso_snr)
    else:
        X_tot, n_sl_used = build_pathlength(cfg, qso_lookup=qso_lookup)
        qso_per_sl = None; Xcalc = None
    print(f"    X_tot={X_tot}, n_sl_used={n_sl_used}")

    # estimator dispatch (default v1; v2/v3 bind mm/qso_per_sl/Xcalc/rng via partial).
    if estimator == "v2":
        import functools
        from CDDF_analysis.cddf_catalog_hbi import v2_refit
        print("    [estimator] v2 forward-HBI (mm/qso_per_sl/Xcalc bound)")
        estimator_fn = functools.partial(
            v2_refit, mm=mm, qso_per_sl=qso_per_sl, Xcalc=Xcalc, rng=rng)
    elif estimator == "v3":
        import functools
        from CDDF_analysis.cddf_catalog_hbi import v3x_refit
        print(f"    [estimator] v3 PARAMETRIC (family={getattr(cfg,'v3_family','plaw')}, "
              f"mm/qso_per_sl/Xcalc bound)")
        estimator_fn = functools.partial(
            v3x_refit, mm=mm, qso_per_sl=qso_per_sl, Xcalc=Xcalc, rng=rng)
    else:
        estimator_fn = estimate_f_b

    print("[5] fine grid + FP model (FROZEN b_FP)")
    logN_lo, logN_hi, N_b, dN_b = build_fine_grid(cfg)
    s2n = np.asarray(cat_cut["S2N_RED"], dtype=float)
    pdla = np.asarray(cat_cut["P_DLA"], dtype=float)
    op_mask = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask
    fp_model, _ = make_fp_model(cfg, cat_cut, op_mask, rho_interp)

    # the gate floor is max(spec 19.5, the C/rho matrix floor): a bin below the
    # matrix floor cannot be tested (1/C divides by a truth-floored denominator).
    pull_gate_logN = max(PULL_GATE_LOGN, truth_floor)

    print("[6] untilted baseline R0 (the absolute-bias the closure divides out)")
    baseline = baseline_recovery(cfg, cat_cut, is_TP, good_mask, truth_cut,
                                 C_interp, fp_model, X_tot,
                                 logN_lo, logN_hi, N_b, dN_b,
                                 estimator_fn=estimator_fn)
    for lim in cfg.report_logN_limits:
        print(f"    R0(dN/dX>={lim})={baseline['R0_dndx_total'][lim]:.4f}, "
              f"R0(Omega>={lim})={baseline['R0_omega'][lim]:.4f}")

    print(f"\n[7] +tilt  (Δα=+{dalpha})")
    res_plus = run_one_tilt(cfg, cat_cut, is_TP, good_mask, truth_cut, mm,
                            C_interp, fp_model, X_tot,
                            logN_lo, logN_hi, N_b, dN_b, +dalpha, rng, baseline,
                            estimator_fn=estimator_fn)
    print(f"[8] −tilt  (Δα=−{dalpha})")
    res_minus = run_one_tilt(cfg, cat_cut, is_TP, good_mask, truth_cut, mm,
                             C_interp, fp_model, X_tot,
                             logN_lo, logN_hi, N_b, dN_b, -dalpha, rng, baseline,
                             estimator_fn=estimator_fn)

    print("[9] gate")
    gate = evaluate_gate(res_plus, res_minus, logN_lo, nominal_coverage,
                         pull_gate_logN=pull_gate_logN,
                         report_limits=cfg.report_logN_limits,
                         logN_hi=logN_hi, N_b=N_b, dN_b=dN_b, H0=cfg.H0,
                         drop_top_above=cfg.drop_top_bin_above,
                         estimator=estimator, baseline=baseline,
                         closure_R0_mode=closure_R0_mode,
                         band_ess=getattr(cfg, "_posterior_kernel_ess", None))
    # numerical Finding 1/2: record v3's untilted R0 so the runner can decide whether
    # closure_R0_mode="unit" was even valid (|R0-1|<=0.03). Headline stays "divide".
    gate["checks"]["closure_R0_mode"] = closure_R0_mode
    for lim in cfg.report_logN_limits:
        gate["checks"][f"untilted_R0_dndx_{lim}"] = float(baseline["R0_dndx_total"][lim])
        gate["checks"][f"untilted_R0_omega_{lim}"] = float(baseline["R0_omega"][lim])

    # CS-review F1 diagnostic: fraction of op detections labeled hostless (tilt
    # weight 1.0). With the host floor decoupled this should be SMALL at >=20.3.
    hl = hostless_op_fraction(cat_cut, good_mask, cfg, logN_min=20.3)
    gate["checks"]["host_truth_floor_eff"] = float(host_floor_eff)
    gate["checks"]["F1_hostless_op_frac_ge20.3"] = float(hl["frac_hostless"])
    gate["checks"]["F1_hostless_op_n_ge20.3"] = int(hl["n_hostless"])
    gate["checks"]["F1_op_n_ge20.3"] = int(hl["n_op"])
    print(f"    [F1] hostless op fraction (>=20.3) = {hl['frac_hostless']:.4f} "
          f"({hl['n_hostless']}/{hl['n_op']}); host floor={host_floor_eff:.2f}  "
          f"(expect ~0.017 at floor<=19, ~0.18 at floor 20.3 = F1 bug live)")

    print("[10] write outputs")
    paths = write_tilt_outputs(cfg, res_plus, res_minus, gate, baseline, logN_lo,
                               logN_hi, N_b, dN_b, X_tot, cfg.out_dir)

    # ---- console report ----
    gf = gate["pull_gate_logN"]
    verdict = "PASS" if gate["passed"] else "FAIL"
    print("\n" + "=" * 70)
    print(f"  WALL-1 TILT CLOSURE GATE: {verdict}  (Δα=±{dalpha}, nominal cov={nominal_coverage}, "
          f"gate floor logN>={gf:.2f})")
    print("  HEADLINE GATE = integrated dN/dX & Ω closure (migration-insensitive);")
    print("  differential per-bin f_b closure is a DIAGNOSTIC only (v1 migration-blind).")
    print("=" * 70)
    if gate["fail_reasons"]:
        print("  FAIL reasons:")
        for r in gate["fail_reasons"]:
            print(f"    - {r}")
        print(f"  FAIL classification: {gate['fail_classifier']}")
    else:
        print(f"  All integrated dN/dX & Ω closure |pull|<=3 on both tilts; closure "
              f"prediction in 95% band; no opposite-sign coherent integrated pull.")
    print(f"\n  DIAGNOSTIC differential f_b coherent mean closure pull (>={gf:.2f}, "
          f"n_cells={gate['checks'].get('DIAG_n_cells_gefloor')}, "
          f"thr={gate['checks'].get('DIAG_coherent_threshold'):.3f}): +tilt="
          f"{gate['checks'].get('DIAG_coherent_mean_pull_plus_gefloor'):.3f}, "
          f"−tilt={gate['checks'].get('DIAG_coherent_mean_pull_minus_gefloor'):.3f}  "
          f"(opposite-sign={gate['checks'].get('DIAG_opposite_sign_coherent_diff')})  "
          f"[migration signature, NOT gated]")
    print(f"  DIAGNOSTIC differential f_b max |closure pull| (>={gf:.2f}): +tilt="
          f"{gate['checks'].get('DIAG_f_pull_max_gefloor_plus')}, "
          f"−tilt={gate['checks'].get('DIAG_f_pull_max_gefloor_minus')}")
    for lim in cfg.report_logN_limits:
        print(f"  dN/dX(>={lim}) closure pull: +tilt={res_plus['dndx_tot_pull'][lim]:+.2f}, "
              f"−tilt={res_minus['dndx_tot_pull'][lim]:+.2f}  | raw pull: "
              f"+tilt={res_plus['dndx_tot_pull_raw'][lim]:+.2f}, "
              f"−tilt={res_minus['dndx_tot_pull_raw'][lim]:+.2f}")
    bd = gate["sub195_diag"]["plus"]
    if bd["n_considered"] > 0:
        print(f"\n  below-floor (<{gf:.2f}) diagnostic (EXPECTED to widen, not confidently miss):")
        for tag in ("plus", "minus"):
            d = gate["sub195_diag"][tag]
            print(f"    {tag}: {d['n_widened_ok']}/{d['n_considered']} widened-or-in-band, "
                  f"{d['n_confident_miss']} confident-miss "
                  f"{('at logN='+','.join(d['confident_miss_logN'])) if d['n_confident_miss'] else ''}")
    print(f"\n  paths: {paths}")

    return dict(res_plus=res_plus, res_minus=res_minus, gate=gate, baseline=baseline,
                paths=paths, meta=meta, X_tot=X_tot, n_sl_used=n_sl_used,
                logN_lo=logN_lo, logN_hi=logN_hi)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    DEF_OUT = "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phase1_v1_out/"
    DEF_CAT = ("/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/"
               "combined_catalog/")
    DEF_TRUTH = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/"
                 "v2.8.5/mock-0/loa-124/hcd_truth_cat.fits")
    DEF_BAL = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/"
               "v2.8.5/mock-0/loa-124/bal_cat.fits")
    DEF_MOLLY = ("/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/"
                 "figures_molly/molly_matrix.tsv")
    p.add_argument("--catalog-dir", default=DEF_CAT)
    p.add_argument("--truth", default=DEF_TRUTH)
    p.add_argument("--bal-cat", default=DEF_BAL)
    p.add_argument("--molly-tsv", default=DEF_MOLLY,
                   help="molly_matrix.tsv (its lowest NHI edge is the C/rho FLOOR). "
                        "Use figures_molly_nhi19 to exercise the <19.5 EXPECTED-FAIL "
                        "diagnostic; default figures_molly only reaches 20.3.")
    p.add_argument("--out", default=DEF_OUT)
    p.add_argument("--mockdir", default=None)
    p.add_argument("--fp", choices=["purity_mixture", "loa0"], default="purity_mixture")
    p.add_argument("--n-mc", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--zbins", default="2.0,2.5,3.0,3.5")
    p.add_argument("--report-limits", default="20.0,20.3")
    p.add_argument("--dalpha", type=float, default=DEFAULT_DALPHA,
                   help="tilt magnitude; the gate runs +dalpha and -dalpha.")
    p.add_argument("--host-truth-floor", type=float, default=19.0,
                   help="NHI floor of the truth used to attach the tilt host mark "
                        "(CS-review F1). DECOUPLED from the C/rho matrix floor; "
                        "default 19.0 picks up sub-DLA up-migrants near 20.0-20.3. "
                        "Clamped to <= the matrix floor.")
    p.add_argument("--nominal-coverage", type=float, default=0.95)
    p.add_argument("--estimator", choices=["v1", "v2", "v3"], default="v1",
                   help="which CDDF estimator the WALL-1 refit uses (default v1; "
                        "v2 = the forward-HBI smooth MAP; v3 = the parametric continuous "
                        "f(N|theta) — the tilt-robust fix for the v2 over-response).")
    p.add_argument("--v3-family", default="plaw",
                   choices=["plaw", "plawcut", "bplcut", "pspline", "bspbody"],
                   help="v3 DOF-ladder rung (only used with --estimator v3). bspbody = the "
                        "body-anchored penalized B-spline (the WALL-1 deep-tail fix).")
    # bspbody knobs (cs Finding 7 — make the run reproducible from the in-repo CLI)
    p.add_argument("--v3-fit-floor", type=float, default=None,
                   help="bspbody detection-row floor (v3_logN_fit_floor).")
    p.add_argument("--v3-n-knots", type=int, default=None)
    p.add_argument("--v3-lambda-bspbody", type=float, default=None)
    p.add_argument("--v3-tail-boost", type=float, default=None)
    p.add_argument("--v3-tail-boost-logN", type=float, default=None)
    p.add_argument("--v3-edge-slope-lam", type=float, default=None)
    p.add_argument("--v3-edge-slope-target", type=float, default=None)
    p.add_argument("--v3-edge-hi", type=float, default=None)
    p.add_argument("--closure-R0-mode", choices=["divide", "unit"], default="divide",
                   help="WALL-1 closure normalization (numerical Finding 1/2): 'divide' "
                        "(R0 from e0/t0 — the headline, like-for-like with v2) or 'unit' "
                        "(R0==1, bare tilted truth — only valid if |R0-1|<=0.03).")
    p.add_argument("--molly-input-order", dest="molly_input_order",
                   action="store_true", default=False)
    p.add_argument("--no-bal", dest="no_bal", action="store_true", default=True)
    p.add_argument("--keep-bal", dest="no_bal", action="store_false")
    args = p.parse_args(argv)

    zbins = tuple(float(x) for x in args.zbins.split(","))
    report_limits = tuple(float(x) for x in args.report_limits.split(","))
    cfg = HBIConfig(
        catalog_dir=args.catalog_dir, truth_path=args.truth,
        bal_cat_path=args.bal_cat, molly_tsv=args.molly_tsv, out_dir=args.out,
        mockdir=args.mockdir or os.path.dirname(args.truth),
        zbins=zbins, n_mc=args.n_mc, rng_seed=args.seed,
        fp_estimator=args.fp, no_bal=args.no_bal,
        report_logN_limits=report_limits,
        molly_input_order=args.molly_input_order,
        v3_family=args.v3_family,
    )
    # bspbody overrides from the CLI (cs Finding 7) — None leaves the dataclass default
    for _attr, _val in (
        ("v3_logN_fit_floor", args.v3_fit_floor),
        ("v3_bspbody_n_knots", args.v3_n_knots),
        ("v3_lambda_bspbody", args.v3_lambda_bspbody),
        ("v3_bspbody_tail_lam_boost", args.v3_tail_boost),
        ("v3_bspbody_tail_boost_logN", args.v3_tail_boost_logN),
        ("v3_bspbody_edge_slope_lam", args.v3_edge_slope_lam),
        ("v3_bspbody_edge_slope_target", args.v3_edge_slope_target),
        ("v3_bspbody_edge_hi", args.v3_edge_hi),
    ):
        if _val is not None:
            setattr(cfg, _attr, _val)
    run_wall1(cfg, dalpha=args.dalpha, nominal_coverage=args.nominal_coverage,
              host_truth_floor=args.host_truth_floor, estimator=args.estimator,
              closure_R0_mode=args.closure_R0_mode)


if __name__ == "__main__":
    main()
