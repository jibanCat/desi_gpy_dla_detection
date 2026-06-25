"""Dedicated invariant tests for CDDF_analysis/cddf_tilt_closure.py — the WALL-1
slope-dependence acceptance gate (spec §7).

These tests do NOT touch the real catalog, FITS, GP inference, or any production
data — they build a small SELF-CONTAINED synthetic fixture (an astropy Table
``cat_cut`` of GP detections + a ``truth_cut`` of true absorbers + the molly C/ρ
matrix), reusing the fixture style of ``tests/test_cddf_catalog_hbi.py``, and
assert the CONCRETE numeric / structural invariants each tilt-closure function
computes.

What each test pins (function -> invariant):
  * tilt_weight                    -> exact 10^(Δα·(logN−pivot)); identity at Δα=0;
                                      ==1 at the pivot; Eddington sign; monotonic;
                                      NaN-logN (forest FP) -> weight 1.0.
  * tilted_truth_reductions        -> Δα=0 reproduces the untilted truth reductions;
                                      hand-computed reweighted f(N)/dN/dX/Ω at Δα≠0;
                                      a +Δα raises Ω relative to dN/dX (the CDDF tilt).
  * detection_tilt_weights /       -> NON-CIRCULAR: weight reads the TRUTH-HOST NHI
    _tilt_host_nhi                    column (not the detection's own MAP N̂);
                                      NHI_TILT_HOST overrides NHI_TRUE; Δα=0 -> all 1;
                                      hostless (NaN host) -> weight 1.0.
  * hostless_op_fraction           -> the returned fraction == the hand-counted
                                      hostless op fraction at a given logN_min.
  * _omega_closure_resid_frac      -> a perfectly-closed res -> ~0 residual; sign.
  * baseline_recovery              -> end-to-end keys/structure; Δα=0 self-consistency
                                      (R0 == est0/truth0 exactly).
  * run_one_tilt (INTEGRATION)     -> closure structure; the Δα=0 control closes
                                      (closure pull ~0); FP-FREEZE guard rejects a
                                      non-purity-mixture FP; the FROZEN FP term does
                                      not move with the tilt.
"""
import numpy as np
import pytest
from astropy.table import Table

from CDDF_analysis.hbi import cddf_catalog_hbi as H
from CDDF_analysis.hbi import cddf_tilt_closure as T


# ===========================================================================
# Shared synthetic fixtures (fast, deterministic, no I/O)
# ===========================================================================
def _make_cfg(**kw):
    """A minimal HBIConfig with dummy paths (never loaded in these tests)."""
    defaults = dict(
        catalog_dir="/dev/null", truth_path="/dev/null",
        bal_cat_path="/dev/null", molly_tsv="/dev/null", out_dir="/tmp",
        logN_lo=19.5, logN_hi=22.0, dlogN=0.1, drop_top_bin_above=21.9,
        zbins=(2.0, 2.5, 3.0, 3.5), report_logN_limits=(20.0, 20.3),
        H0=70.0, Omega_m=0.279, snr_min=2.0, p_dla_min=0.99,
        n_mc=40, rng_seed=0, fp_estimator="purity_mixture",
    )
    defaults.update(kw)
    return H.HBIConfig(**defaults)


def _tiny_truth_cut(nhi, z, snr=10.0, tid=None):
    """Build a truth_cut Table with the columns the tilt-closure code reads
    (NHI, Z_DLA, S2N_RED, TARGETID)."""
    nhi = np.asarray(nhi, float)
    z = np.asarray(z, float)
    snr = np.full(len(nhi), float(snr)) if np.isscalar(snr) else np.asarray(snr, float)
    if tid is None:
        tid = np.arange(len(nhi), dtype=np.int64)
    return Table(dict(NHI=nhi, Z_DLA=z, S2N_RED=snr,
                      TARGETID=np.asarray(tid, dtype=np.int64)))


def _synthetic_fixture(seed=12345):
    """A small end-to-end fixture for the integration tests.

    Construction (controllable, NO N-migration so the v1 estimator is unbiased at
    Δα=0 BY CONSTRUCTION — every detection's measured NHI == its truth host NHI):
      * A power-law truth f(N) over the DLA tier, Poisson-drawn per fine N-bin.
      * Each true absorber is DETECTED with a per-cell completeness C(N,SNR) < 1.
      * Each detection carries NHI_TRUE == NHI_TILT_HOST == its own NHI (no scatter),
        so the truth host drives the tilt with the SAME logN the estimator bins on.
      * A small forest-FP background per bin, each FP given a HOSTLESS truth (NaN
        NHI_TILT_HOST) so it is the genuine forest-FP -> tilt weight 1.0.
      * purity ρ per detection set so Σ(1−ρ) per bin == the injected FP count.

    Returns a dict with every argument run_one_tilt / baseline_recovery need.
    """
    rng = np.random.default_rng(seed)
    cfg = _make_cfg(logN_lo=19.5, logN_hi=21.5, drop_top_bin_above=21.4)
    logN_lo, logN_hi, N_b, dN_b = H.build_fine_grid(cfg)
    n_nbins = len(logN_lo)
    zbins = np.asarray(cfg.zbins, float)
    n_zbins = len(zbins) - 1

    # pathlength per coarse z-bin (large -> tight Poisson, fast MC stays stable)
    X_tot = np.full(n_zbins, 4.0e5)
    X_sum = float(np.sum(X_tot))

    # injected truth power law f = A·N^beta, beta=-1.8, height ~1e-22 at 20.3
    beta = -1.8
    A = 1.0e-22 / (10.0 ** 20.3) ** beta
    f_b_true = A * N_b ** beta

    # completeness matrix: rises with SNR, < 1 in the DLA tier; molly cells
    # spanning the fine grid. (n_snr=3, n_nhi=2 over [19.5,20.5,21.5))
    snr_edges = np.array([0.0, 4.0, 8.0, np.inf])
    nhi_edges = np.array([19.5, 20.5, 21.5])
    C_mat = np.array([[0.45, 0.60],
                      [0.70, 0.85],
                      [0.90, 0.97]])
    mm = H.MollyMatrix(snr_edges=snr_edges, nhi_edges=nhi_edges,
                       purity=np.ones_like(C_mat), completeness=C_mat)
    C_interp = H.make_C_interpolator(mm)

    # ---- forward-simulate the TRUE absorbers + DETECTED catalog ----
    truth_nhi, truth_z, truth_snr, truth_tid = [], [], [], []
    det_nhi, det_z, det_snr, det_tid, det_host = [], [], [], [], []
    fp_per_bin = np.zeros(n_nbins)
    tid_counter = 0
    snr_choices = np.array([3.0, 6.0, 12.0])
    for b in range(n_nbins):
        mean_true = f_b_true[b] * dN_b[b] * X_sum
        n_true = rng.poisson(mean_true)
        if n_true == 0:
            continue
        lN = rng.uniform(logN_lo[b], logN_hi[b], n_true)
        snr = rng.choice(snr_choices, size=n_true, p=[0.5, 0.3, 0.2])
        z = rng.uniform(2.0, 3.5, n_true)
        tids = np.arange(tid_counter, tid_counter + n_true, dtype=np.int64)
        tid_counter += n_true
        # record every TRUE system (the completeness denominator)
        truth_nhi.append(lN); truth_z.append(z)
        truth_snr.append(snr); truth_tid.append(tids)
        # detect with prob C at the cell center
        Cdet = C_interp(np.full(n_true, 0.5 * (logN_lo[b] + logN_hi[b])), snr)
        det = rng.random(n_true) < Cdet
        det_nhi.append(lN[det]); det_z.append(z[det])
        det_snr.append(snr[det]); det_tid.append(tids[det])
        det_host.append(lN[det])               # NO N-migration: host == measured NHI
        # a small forest-FP background in this bin (HOSTLESS)
        n_fp = rng.poisson(0.03 * det.sum())
        if n_fp:
            fp_nhi = rng.uniform(logN_lo[b], logN_hi[b], n_fp)
            fp_snr = rng.choice(snr_choices, size=n_fp)
            fp_tids = np.arange(tid_counter, tid_counter + n_fp, dtype=np.int64)
            tid_counter += n_fp
            det_nhi.append(fp_nhi)
            det_z.append(rng.uniform(2.0, 3.5, n_fp))
            det_snr.append(fp_snr); det_tid.append(fp_tids)
            det_host.append(np.full(n_fp, np.nan))   # hostless forest FP
            fp_per_bin[b] += n_fp

    truth_nhi = np.concatenate(truth_nhi); truth_z = np.concatenate(truth_z)
    truth_snr = np.concatenate(truth_snr); truth_tid = np.concatenate(truth_tid)
    det_nhi = np.concatenate(det_nhi); det_z = np.concatenate(det_z)
    det_snr = np.concatenate(det_snr); det_tid = np.concatenate(det_tid)
    det_host = np.concatenate(det_host)
    n_det = len(det_nhi)

    truth_cut = _tiny_truth_cut(truth_nhi, truth_z, truth_snr, truth_tid)

    # ---- purity per detection so Σ(1−ρ) per bin == injected FP count ----
    nbin = H._bin_index_logN(det_nhi, logN_lo, logN_hi)
    valid = nbin >= 0
    n_obs_bin = np.zeros(n_nbins)
    np.add.at(n_obs_bin, nbin[valid], 1.0)
    one_minus_rho_target = np.where(n_obs_bin > 0,
                                    fp_per_bin / np.maximum(n_obs_bin, 1), 0.0)
    rho_per_obj = 1.0 - one_minus_rho_target[np.clip(nbin, 0, n_nbins - 1)]
    rho_per_obj = np.where(valid, rho_per_obj, 0.0)

    # ---- the GP-catalog Table cat_cut (op-passing rows; all good) ----
    cat_cut = Table(dict(
        NHI=det_nhi,
        NHI_TRUE=det_host,
        NHI_TILT_HOST=det_host,
        NHI_ERR=np.zeros(n_det),         # no width scatter -> MC band stable + small
        Z_DLA=det_z,
        S2N_RED=det_snr,
        P_DLA=np.full(n_det, 0.999),     # all > p_dla_min
        TARGETID=det_tid.astype(np.int64),
    ))
    good_mask = np.ones(n_det, dtype=bool)
    # is_TP: True where the detection has a truth host (FP rows are False). v1 does
    # not read it, but the signature requires it.
    is_TP = np.isfinite(det_host)

    op_mask = ((det_snr > cfg.snr_min) & (cat_cut["P_DLA"] > cfg.p_dla_min)
               & good_mask)

    # FP model + ρ-interpolator built from the per-object purity (purity-mixture).
    def rho_interp(nhi, snr):
        nb = H._bin_index_logN(np.asarray(nhi), logN_lo, logN_hi)
        return np.where(nb >= 0,
                        1.0 - one_minus_rho_target[np.clip(nb, 0, n_nbins - 1)], 0.0)

    rho_op = rho_interp(det_nhi[op_mask], det_snr[op_mask])
    fp_model = H.PurityMixtureFP(rho_op)

    # regenerate molly counts so joint_mc_errors has the Jeffreys-Beta inputs
    mm = H.regenerate_molly_counts(mm, cat_cut, is_TP, truth_cut, good_mask, cfg)

    return dict(cfg=cfg, cat_cut=cat_cut, truth_cut=truth_cut, is_TP=is_TP,
                good_mask=good_mask, mm=mm, C_interp=C_interp, fp_model=fp_model,
                X_tot=X_tot, logN_lo=logN_lo, logN_hi=logN_hi, N_b=N_b, dN_b=dN_b,
                f_b_true=f_b_true, fp_per_bin=fp_per_bin, rho_interp=rho_interp,
                op_mask=op_mask, det_host=det_host)


# ===========================================================================
# 1. tilt_weight — the exact reweighting formula + invariants
# ===========================================================================
def test_tilt_weight_identity_at_zero_dalpha():
    """Δα=0 -> weight ≡ 1 everywhere (the untilted identity)."""
    logN = np.array([17.5, 19.0, 20.3, 21.0, 22.0])
    w = T.tilt_weight(logN, 0.0, pivot=20.3)
    np.testing.assert_allclose(w, 1.0, rtol=0, atol=0)


def test_tilt_weight_exact_formula_points():
    """w(logN) == 10^(Δα·(logN−pivot)) at hand-computed points."""
    pivot = 20.3
    da = 0.5
    # at logN = pivot + 1, weight = 10^(0.5*1) = sqrt(10)
    assert T.tilt_weight(np.array([21.3]), da, pivot)[0] == pytest.approx(10.0 ** 0.5)
    # at logN = pivot - 2, weight = 10^(0.5*-2) = 10^-1 = 0.1
    assert T.tilt_weight(np.array([18.3]), da, pivot)[0] == pytest.approx(0.1)
    # general vectorized check
    logN = np.array([19.0, 20.3, 21.0])
    np.testing.assert_allclose(
        T.tilt_weight(logN, da, pivot), 10.0 ** (da * (logN - pivot)), rtol=1e-12)


def test_tilt_weight_unity_at_pivot():
    """weight == 1 exactly at logN == pivot, for any Δα (the rotation hinge)."""
    for da in (-0.5, -0.1, 0.3, 0.5, 1.0):
        assert T.tilt_weight(np.array([20.3]), da, 20.3)[0] == pytest.approx(1.0)


def test_tilt_weight_eddington_sign_and_monotonic():
    """Δα>0: weight>1 ABOVE the pivot, <1 BELOW it (the CDDF/Eddington tilt sign);
    monotonically increasing in logN. Δα<0 flips the sign."""
    logN = np.linspace(18.0, 22.0, 41)
    w_plus = T.tilt_weight(logN, 0.5, 20.3)
    assert np.all(w_plus[logN > 20.3] > 1.0)
    assert np.all(w_plus[logN < 20.3] < 1.0)
    # strictly monotone increasing for Δα>0
    assert np.all(np.diff(w_plus) > 0)
    # Δα<0 is strictly decreasing and the opposite sign
    w_minus = T.tilt_weight(logN, -0.5, 20.3)
    assert np.all(np.diff(w_minus) < 0)
    assert np.all(w_minus[logN > 20.3] < 1.0)
    assert np.all(w_minus[logN < 20.3] > 1.0)


def test_tilt_weight_nan_host_is_forest_fp_weight_one():
    """A NaN logN (hostless detection = forest FP) gets weight 1.0 regardless of Δα
    (the FP is slope-independent — spec §7)."""
    logN = np.array([np.nan, 21.0, np.nan])
    for da in (0.5, -0.5, 0.0):
        w = T.tilt_weight(logN, da, 20.3)
        assert w[0] == 1.0 and w[2] == 1.0
        # the finite entry still gets the real tilt weight
        assert w[1] == pytest.approx(10.0 ** (da * (21.0 - 20.3)))


# ===========================================================================
# 2. tilted_truth_reductions — Δα=0 reproduces untilted truth; reweighted sums
# ===========================================================================
def test_tilted_truth_reductions_dalpha0_matches_untilted_truth():
    """Δα=0 tilted truth == the un-reweighted truth_reductions (machine precision):
    f(N), dN/dX_total, Ω all reproduce the plain truth-side counts."""
    fx = _synthetic_fixture()
    cfg = fx["cfg"]
    ttr0 = T.tilted_truth_reductions(
        cfg, fx["truth_cut"], fx["logN_lo"], fx["logN_hi"], fx["N_b"], fx["dN_b"],
        fx["X_tot"], 0.0)
    ref = H.truth_reductions(cfg, fx["truth_cut"], fx["logN_lo"], fx["logN_hi"],
                             fx["N_b"], fx["dN_b"], fx["X_tot"])
    np.testing.assert_allclose(ttr0["f_truth"], ref["f_truth"], rtol=1e-12, atol=0)
    for lim in cfg.report_logN_limits:
        assert ttr0["dndx_total"][lim] == pytest.approx(ref["dndx_total"][lim], rel=1e-12)
        assert ttr0["omega"][lim] == pytest.approx(ref["omega"][lim], rel=1e-12)


def test_tilted_truth_reductions_fN_is_hand_reweighted_sum():
    """f_truth^tilt[b] == (Σ_{true in bin} w(N_true)) / (X_sum·ΔN_b), hand-computed
    on a tiny controlled truth_cut (every weight applied to the true host NHI)."""
    cfg = _make_cfg(logN_lo=20.0, logN_hi=20.4, drop_top_bin_above=20.4,
                    zbins=(2.0, 3.0), report_logN_limits=(20.0,))
    logN_lo, logN_hi, N_b, dN_b = H.build_fine_grid(cfg)
    # 3 true systems: two in bin [20.0,20.1), one in [20.2,20.3); all SNR>snr_min
    nhi = np.array([20.05, 20.07, 20.25])
    z = np.array([2.4, 2.6, 2.5])
    truth_cut = _tiny_truth_cut(nhi, z, snr=10.0)
    X_tot = np.array([1.0e4])   # single z-bin
    X_sum = float(X_tot.sum())
    da = 0.5
    ttr = T.tilted_truth_reductions(cfg, truth_cut, logN_lo, logN_hi, N_b, dN_b,
                                    X_tot, da, pivot=20.3)
    w = 10.0 ** (da * (nhi - 20.3))
    # hand reductions per bin
    nbin = H._bin_index_logN(nhi, logN_lo, logN_hi)
    f_hand = np.zeros(len(logN_lo))
    np.add.at(f_hand, nbin, w)
    f_hand = f_hand / (X_sum * dN_b)
    np.testing.assert_allclose(ttr["f_truth"], f_hand, rtol=1e-12, atol=0)
    # integrated dN/dX(>=20.0) == Σ all weights / X_sum (all 3 systems are >=20.0)
    assert ttr["dndx_total"][20.0] == pytest.approx(np.sum(w) / X_sum, rel=1e-12)


def test_tilted_truth_reductions_drops_truth_below_snr_min():
    """The truth-side `keep = S2N_RED > snr_min` filter (cddf_tilt_closure.py:~133)
    drops sub-cut truth rows and keeps above-cut rows EXACTLY (gotcha 4: truth must
    be restricted to the same SNR-selected sightline set as the ΔX denominator).

    PR-18 #6.2: builds a truth_cut with rows straddling snr_min and asserts the
    reductions equal the reductions of ONLY the kept (above-cut) rows. Sub-cut rows
    must contribute nothing to f(N)/dN/dX/Ω; the boundary (== snr_min) is excluded
    by the strict `>` comparison."""
    cfg = _make_cfg(logN_lo=20.0, logN_hi=20.4, drop_top_bin_above=20.4,
                    zbins=(2.0, 3.0), report_logN_limits=(20.0,), snr_min=2.0)
    logN_lo, logN_hi, N_b, dN_b = H.build_fine_grid(cfg)
    # 5 truth systems: SNR exactly at the cut (excluded), below the cut (excluded),
    # and above the cut (kept). Same NHI/z so only the SNR filter distinguishes them.
    nhi = np.array([20.05, 20.15, 20.25, 20.05, 20.25])
    z = np.array([2.4, 2.5, 2.6, 2.5, 2.4])
    snr = np.array([1.0,        # below  -> DROP
                    2.0,        # == cut -> DROP (strict >)
                    5.0,        # above  -> KEEP
                    10.0,       # above  -> KEEP
                    8.0])       # above  -> KEEP
    truth_cut = _tiny_truth_cut(nhi, z, snr=snr)
    X_tot = np.array([1.0e4])
    da = 0.3

    full = T.tilted_truth_reductions(cfg, truth_cut, logN_lo, logN_hi, N_b, dN_b,
                                     X_tot, da, pivot=20.3)
    # the reference: the SAME call on ONLY the above-cut rows (the kept set)
    kept = snr > cfg.snr_min
    truth_kept = _tiny_truth_cut(nhi[kept], z[kept], snr=snr[kept])
    ref = T.tilted_truth_reductions(cfg, truth_kept, logN_lo, logN_hi, N_b, dN_b,
                                    X_tot, da, pivot=20.3)

    # filtered-in-place == explicitly-pre-filtered, to machine precision
    np.testing.assert_allclose(full["f_truth"], ref["f_truth"],
                               rtol=1e-12, atol=0, equal_nan=True)
    assert full["dndx_total"][20.0] == pytest.approx(ref["dndx_total"][20.0], rel=1e-12)
    assert full["omega"][20.0] == pytest.approx(ref["omega"][20.0], rel=1e-12)

    # POSITIVE control: the kept reduction equals the hand sum over the 3 kept rows
    # only (the 2 dropped rows contribute nothing).
    w_kept = 10.0 ** (da * (nhi[kept] - 20.3))
    assert full["dndx_total"][20.0] == pytest.approx(
        float(np.sum(w_kept)) / float(X_tot.sum()), rel=1e-12)

    # NEGATIVE control: if the dropped rows had been (wrongly) kept, dN/dX would be
    # strictly larger — so the filter is load-bearing, not a no-op on this input.
    truth_all = _tiny_truth_cut(nhi, z, snr=np.full(len(nhi), 10.0))  # all above cut
    allkept = T.tilted_truth_reductions(cfg, truth_all, logN_lo, logN_hi, N_b, dN_b,
                                        X_tot, da, pivot=20.3)
    assert allkept["dndx_total"][20.0] > full["dndx_total"][20.0]


def test_tilted_truth_reductions_plus_tilt_raises_omega_over_dndx():
    """A +Δα tilt up-weights the high-N systems, which carry the Ω integral's
    N·f weight, so Ω grows MORE than dN/dX relative to the untilted truth (the
    'tilt tilts the CDDF' invariant)."""
    fx = _synthetic_fixture()
    cfg = fx["cfg"]
    lim = 20.3
    t0 = T.tilted_truth_reductions(cfg, fx["truth_cut"], fx["logN_lo"], fx["logN_hi"],
                                   fx["N_b"], fx["dN_b"], fx["X_tot"], 0.0)
    tp = T.tilted_truth_reductions(cfg, fx["truth_cut"], fx["logN_lo"], fx["logN_hi"],
                                   fx["N_b"], fx["dN_b"], fx["X_tot"], 0.5)
    omega_ratio = tp["omega"][lim] / t0["omega"][lim]
    dndx_ratio = tp["dndx_total"][lim] / t0["dndx_total"][lim]
    # the Ω integral (∝ Σ N·f) up-weights high N more than the dN/dX count
    assert omega_ratio > dndx_ratio > 0
    # and both grow under +tilt for a >=20.3 tier (pivot is at 20.3, so the >=20.3
    # population is on the up-weighted side)
    assert omega_ratio > 1.0


# ===========================================================================
# 3. detection_tilt_weights / _tilt_host_nhi — NON-CIRCULARITY (truth host)
# ===========================================================================
def test_tilt_host_nhi_prefers_tilt_host_column():
    """_tilt_host_nhi reads NHI_TILT_HOST when present, else falls back to NHI_TRUE."""
    cat = Table(dict(NHI=[20.5, 21.0], NHI_TRUE=[19.9, 20.1],
                     NHI_TILT_HOST=[19.7, np.nan]))
    host = T._tilt_host_nhi(cat)
    # picks NHI_TILT_HOST, NOT NHI (the detection's own MAP) nor NHI_TRUE
    np.testing.assert_array_equal(host, np.array([19.7, np.nan]))
    cat2 = Table(dict(NHI=[20.5], NHI_TRUE=[19.9]))   # no NHI_TILT_HOST col
    np.testing.assert_array_equal(T._tilt_host_nhi(cat2), np.array([19.9]))


def test_detection_tilt_weights_use_truth_host_not_own_nhi():
    """NON-CIRCULAR (load-bearing): the per-detection tilt weight is computed from the
    detection's TRUTH-HOST NHI (NHI_TILT_HOST), NOT from its own predicted NHI. We
    set the measured NHI and the truth host to DIFFERENT values and confirm the
    weight tracks the HOST."""
    cfg = _make_cfg()
    # two detections; measured NHI deliberately != truth host NHI
    cat = Table(dict(
        NHI=np.array([21.5, 21.5]),         # same MEASURED N̂ for both
        NHI_TRUE=np.array([20.3, 21.3]),
        NHI_TILT_HOST=np.array([20.3, 21.3]),  # DIFFERENT true hosts
        S2N_RED=np.array([10.0, 10.0]),
        P_DLA=np.array([0.999, 0.999]),
    ))
    good = np.ones(2, bool)
    da = 0.5
    w = T.detection_tilt_weights(cat, good, cfg, da, pivot=20.3)
    # weight must follow the HOST (20.3 -> 1.0, 21.3 -> 10^0.5), NOT the shared N̂
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(10.0 ** 0.5)
    # if it had (wrongly) used the own NHI=21.5 both would be 10^(0.5*1.2) — assert NOT
    assert not np.allclose(w, 10.0 ** (da * (21.5 - 20.3)))


def test_detection_tilt_weights_dalpha0_all_one_and_hostless_one():
    """Δα=0 -> every op weight 1.0; a hostless detection (NaN host) -> weight 1.0
    even at Δα != 0 (forest FP, slope-independent)."""
    cfg = _make_cfg()
    cat = Table(dict(
        NHI=np.array([20.5, 21.0, 20.8]),
        NHI_TILT_HOST=np.array([20.5, np.nan, 21.0]),   # middle is hostless
        NHI_TRUE=np.array([20.5, np.nan, 21.0]),
        S2N_RED=np.array([10.0, 10.0, 10.0]),
        P_DLA=np.array([0.999, 0.999, 0.999]),
    ))
    good = np.ones(3, bool)
    np.testing.assert_allclose(T.detection_tilt_weights(cat, good, cfg, 0.0), 1.0)
    w = T.detection_tilt_weights(cat, good, cfg, 0.5, pivot=20.3)
    assert w[1] == pytest.approx(1.0)   # hostless -> 1.0 despite Δα=0.5
    assert w[0] == pytest.approx(10.0 ** (0.5 * (20.5 - 20.3)))
    assert w[2] == pytest.approx(10.0 ** (0.5 * (21.0 - 20.3)))


def test_detection_tilt_weights_op_order_and_mask():
    """The weights are returned in the SAME op order/length the estimator uses
    (op = S2N_RED>snr_min & P_DLA>p_dla_min & good_mask); off-op rows are dropped."""
    cfg = _make_cfg()
    cat = Table(dict(
        NHI=np.array([20.5, 21.0, 20.8, 22.0]),
        NHI_TILT_HOST=np.array([20.5, 21.0, 20.8, 22.0]),
        NHI_TRUE=np.array([20.5, 21.0, 20.8, 22.0]),
        # row1 fails SNR, row2 fails P_DLA, row3 fails good_mask -> only row0 + ... op
        S2N_RED=np.array([10.0, 1.0, 10.0, 10.0]),
        P_DLA=np.array([0.999, 0.999, 0.5, 0.999]),
    ))
    good = np.array([True, True, True, False])
    w = T.detection_tilt_weights(cat, good, cfg, 0.5, pivot=20.3)
    # only row0 survives the op mask
    assert len(w) == 1
    assert w[0] == pytest.approx(10.0 ** (0.5 * (20.5 - 20.3)))


def test_host_truth_floor_decoupling_changes_attached_hosts():
    """CS-review F1: which truth systems can ATTACH as a host depends on the truth
    FLOOR applied to the match (host_truth_floor), DECOUPLED from the matrix floor.
    A sub-DLA up-migrant (true host in [19,20.3]) is HOSTLESS (weight 1.0) under a
    20.3-floored host column but carries its real weight under a 19.0-floored one.
    We emulate the two floors as two NHI_TILT_HOST columns and confirm the attachment
    is truth-sourced and changes correctly."""
    cfg = _make_cfg()
    # one detection whose TRUE host is a sub-DLA at 19.8 (an up-migrant: measured N̂>=20.3)
    nhi_meas = 20.35
    true_host = 19.8
    # floor-20.3 view: the sub-floor host is dropped -> hostless (NaN)
    cat_floor203 = Table(dict(
        NHI=[nhi_meas], NHI_TRUE=[np.nan], NHI_TILT_HOST=[np.nan],
        S2N_RED=[10.0], P_DLA=[0.999]))
    # floor-19.0 view: the sub-DLA host is retained -> carries its true weight
    cat_floor190 = Table(dict(
        NHI=[nhi_meas], NHI_TRUE=[true_host], NHI_TILT_HOST=[true_host],
        S2N_RED=[10.0], P_DLA=[0.999]))
    good = np.ones(1, bool)
    da = 0.5
    w203 = T.detection_tilt_weights(cat_floor203, good, cfg, da, pivot=20.3)
    w190 = T.detection_tilt_weights(cat_floor190, good, cfg, da, pivot=20.3)
    # 20.3-floored: mislabeled forest FP -> weight 1.0 (the F1 bug)
    assert w203[0] == pytest.approx(1.0)
    # 19.0-floored: the real sub-DLA up-migrant weight (< 1 since host < pivot)
    assert w190[0] == pytest.approx(10.0 ** (da * (true_host - 20.3)))
    assert w190[0] < 1.0
    # the weight is sourced from the HOST, not the measured N̂ (which is >= 20.3 -> >1)
    assert w190[0] != pytest.approx(10.0 ** (da * (nhi_meas - 20.3)))


# ===========================================================================
# 4. hostless_op_fraction — the hand-counted fraction
# ===========================================================================
def test_hostless_op_fraction_hand_counted():
    """The returned hostless fraction at logN_min equals the hand-counted count of
    op detections >= logN_min with a NaN truth host, divided by the op>=logN_min total."""
    cfg = _make_cfg()
    # 6 detections; 4 are op & >= 20.3; of those, exactly 1 is hostless.
    cat = Table(dict(
        NHI=np.array([20.5, 21.0, 20.4, 22.0, 19.0, 20.6]),
        NHI_TILT_HOST=np.array([20.5, np.nan, 20.4, 21.9, 19.0, 20.6]),
        NHI_TRUE=np.array([20.5, np.nan, 20.4, 21.9, 19.0, 20.6]),
        #   idx:      0(host) 1(NaN)  2(host) 3(host) 4(<20.3) 5(off-op via P_DLA)
        S2N_RED=np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
        P_DLA=np.array([0.999, 0.999, 0.999, 0.999, 0.999, 0.5]),  # idx5 off-op
    ))
    good = np.ones(6, bool)
    out = T.hostless_op_fraction(cat, good, cfg, logN_min=20.3)
    # op & >=20.3: idx {0,1,2,3} (idx4 below 20.3; idx5 fails P_DLA). hostless: {1}.
    assert out["n_op"] == 4
    assert out["n_hostless"] == 1
    assert out["frac_hostless"] == pytest.approx(1.0 / 4.0)
    assert out["logN_min"] == pytest.approx(20.3)


def test_hostless_op_fraction_empty_is_nan():
    """No op detections >= logN_min -> frac_hostless is NaN (no division by zero)."""
    cfg = _make_cfg()
    cat = Table(dict(
        NHI=np.array([19.0, 19.5]),
        NHI_TILT_HOST=np.array([19.0, 19.5]), NHI_TRUE=np.array([19.0, 19.5]),
        S2N_RED=np.array([10.0, 10.0]), P_DLA=np.array([0.999, 0.999])))
    good = np.ones(2, bool)
    out = T.hostless_op_fraction(cat, good, cfg, logN_min=20.3)
    assert out["n_op"] == 0
    assert np.isnan(out["frac_hostless"])


# ===========================================================================
# 5. _omega_closure_resid_frac — sign / zero behaviour on a constructed res
# ===========================================================================
def _resid_inputs(n_nbins=8, lo=20.0, hi=20.8):
    """A tiny fine grid + N_b/dN_b for the Ω-closure residual tests."""
    logN_lo = np.round(np.arange(lo, hi, 0.1), 2)
    logN_hi = np.round(logN_lo + 0.1, 2)
    N_b = 10.0 ** (0.5 * (logN_lo + logN_hi))
    dN_b = 10.0 ** logN_hi - 10.0 ** logN_lo
    return logN_lo, logN_hi, N_b, dN_b


def test_omega_closure_resid_frac_perfect_closure_is_zero():
    """A perfectly-closed res (est f_b == pred_f exactly) -> ~0 residual fraction.
    (No baseline -> the per-bin pred_f form: o_est == o_pred.)"""
    logN_lo, logN_hi, N_b, dN_b = _resid_inputs()
    f = 1e-22 * (N_b / 1e20) ** -1.8
    res = dict(point={"f_b": f.copy()}, pred_f=f.copy())   # est == pred
    out = T._omega_closure_resid_frac(res, logN_lo, logN_hi, N_b, dN_b, 70.0,
                                      limits=(20.0, 20.3), drop_top_above=20.8)
    for lim in (20.0, 20.3):
        assert out[lim] == pytest.approx(0.0, abs=1e-12)


def test_omega_closure_resid_frac_sign_tracks_est_minus_pred():
    """resid_frac = (Ω_est − Ω_pred)/Ω_pred: a uniform est = 1.2·pred -> +0.2 residual;
    est = 0.8·pred -> −0.2 (the deconvolution over/under-recovery sign)."""
    logN_lo, logN_hi, N_b, dN_b = _resid_inputs()
    f = 1e-22 * (N_b / 1e20) ** -1.8
    res_hi = dict(point={"f_b": 1.2 * f}, pred_f=f.copy())
    res_lo = dict(point={"f_b": 0.8 * f}, pred_f=f.copy())
    out_hi = T._omega_closure_resid_frac(res_hi, logN_lo, logN_hi, N_b, dN_b, 70.0,
                                         limits=(20.0,), drop_top_above=20.8)
    out_lo = T._omega_closure_resid_frac(res_lo, logN_lo, logN_hi, N_b, dN_b, 70.0,
                                         limits=(20.0,), drop_top_above=20.8)
    assert out_hi[20.0] == pytest.approx(0.2, rel=1e-9)
    assert out_lo[20.0] == pytest.approx(-0.2, rel=1e-9)


def test_omega_closure_resid_frac_unit_mode_uses_bare_tilted_truth():
    """closure_R0_mode='unit' closes on the BARE tilted truth (R0:=1) — it must
    use f_truth^tilt directly, NOT the baseline-rescaled R0·f_truth^tilt.

    DISCRIMINATING design (PR-18 #6.1): the baseline here has e0 = 2·t0, so the
    divide-mode R0 = Ω_est0/Ω_tr0 = 2 (≠ 1). Therefore:
      * 'unit'   -> o_pred = 1·Ω(ftr)  -> resid = Ω(f)/Ω(ftr) − 1
      * 'divide' -> o_pred = 2·Ω(ftr)  -> resid = Ω(f)/(2·Ω(ftr)) − 1
    The two modes give DIFFERENT residuals, so the assertions below FAIL if the
    `closure_R0_mode == "unit"` branch is removed (regressing unit -> divide), and
    pin that 'unit' specifically equals the bare-tilted-truth (R0=1) result.
    """
    logN_lo, logN_hi, N_b, dN_b = _resid_inputs()
    f = 1e-22 * (N_b / 1e20) ** -1.8
    ftr = 0.5 * f          # tilted truth differs from est by 2x
    res = dict(point={"f_b": f.copy()}, ttr={"f_truth": ftr.copy()})
    # baseline with e0 = 2·t0 -> divide-mode R0 == 2 (NOT 1), so unit vs divide differ.
    t0f = ftr.copy()
    e0f = 2.0 * ftr
    baseline = dict(e0={"f_b": e0f}, t0={"f_truth": t0f})

    out_unit = T._omega_closure_resid_frac(
        res, logN_lo, logN_hi, N_b, dN_b, 70.0,
        limits=(20.0,), drop_top_above=20.8,
        baseline=baseline, closure_R0_mode="unit")
    out_divide = T._omega_closure_resid_frac(
        res, logN_lo, logN_hi, N_b, dN_b, 70.0,
        limits=(20.0,), drop_top_above=20.8,
        baseline=baseline, closure_R0_mode="divide")

    # unit closes on the BARE tilted truth (R0:=1): o_pred = Ω(ftr), o_est = Ω(2·ftr)
    # = 2·Ω(ftr) -> residual = +1.0 exactly.
    assert out_unit[20.0] == pytest.approx(1.0, rel=1e-9)
    # divide-mode uses R0 = Ω(e0)/Ω(t0) = 2 -> o_pred = 2·Ω(ftr) -> residual = 0.0.
    assert out_divide[20.0] == pytest.approx(0.0, abs=1e-9)
    # The two MUST differ — this is what makes the test discriminate the unit branch.
    assert abs(out_unit[20.0] - out_divide[20.0]) > 0.5


# ===========================================================================
# 6. baseline_recovery — end-to-end structure + Δα=0 self-consistency
# ===========================================================================
def test_baseline_recovery_structure_and_R0_definition():
    """baseline_recovery returns the expected keys, and R0 == est0/truth0 EXACTLY
    (the untilted recovery ratio the closure divides out)."""
    fx = _synthetic_fixture()
    cfg = fx["cfg"]
    base = T.baseline_recovery(
        cfg, fx["cat_cut"], fx["is_TP"], fx["good_mask"], fx["truth_cut"],
        fx["C_interp"], fx["fp_model"], fx["X_tot"],
        fx["logN_lo"], fx["logN_hi"], fx["N_b"], fx["dN_b"])
    for key in ("e0", "t0", "R0_f", "R0_dndx_z", "R0_dndx_total", "R0_omega"):
        assert key in base, f"baseline_recovery missing key {key}"
    # R0 is literally est0/truth0 per reduction (the definition the gate relies on)
    for lim in cfg.report_logN_limits:
        td = base["t0"]["dndx_total"][lim]
        to = base["t0"]["omega"][lim]
        if td > 0:
            assert base["R0_dndx_total"][lim] == pytest.approx(
                base["e0"]["dndx_total"][lim] / td, rel=1e-12)
        if to > 0:
            assert base["R0_omega"][lim] == pytest.approx(
                base["e0"]["omega"][lim] / to, rel=1e-12)
    # per-bin R0_f == e0.f_b / t0.f_truth where truth>0
    t0f = np.asarray(base["t0"]["f_truth"])
    e0f = np.asarray(base["e0"]["f_b"])
    pos = t0f > 0
    np.testing.assert_allclose(np.asarray(base["R0_f"])[pos], (e0f / t0f)[pos],
                               rtol=1e-12, atol=0)


def test_baseline_recovery_no_migration_R0_near_one():
    """On the NO-migration synthetic fixture (measured NHI == truth host, FP subtracted
    exactly) the untilted v1 recovery R0(dN/dX) should be ~1 in the well-populated DLA
    tier — the selection correction + FP subtraction is unbiased by construction."""
    fx = _synthetic_fixture()
    cfg = fx["cfg"]
    base = T.baseline_recovery(
        cfg, fx["cat_cut"], fx["is_TP"], fx["good_mask"], fx["truth_cut"],
        fx["C_interp"], fx["fp_model"], fx["X_tot"],
        fx["logN_lo"], fx["logN_hi"], fx["N_b"], fx["dN_b"])
    # large pathlength -> Poisson tight; allow a few % for the per-cell sampling
    for lim in cfg.report_logN_limits:
        assert base["R0_dndx_total"][lim] == pytest.approx(1.0, abs=0.06), (
            f"untilted R0(dN/dX>={lim})={base['R0_dndx_total'][lim]:.4f} not ~1")


# ===========================================================================
# 7. run_one_tilt — INTEGRATION: structure, FP-freeze, Δα=0 control closes
# ===========================================================================
def _run_tilt(fx, dalpha, seed=0):
    cfg = fx["cfg"]
    rng = np.random.default_rng(seed)
    base = T.baseline_recovery(
        cfg, fx["cat_cut"], fx["is_TP"], fx["good_mask"], fx["truth_cut"],
        fx["C_interp"], fx["fp_model"], fx["X_tot"],
        fx["logN_lo"], fx["logN_hi"], fx["N_b"], fx["dN_b"])
    res = T.run_one_tilt(
        cfg, fx["cat_cut"], fx["is_TP"], fx["good_mask"], fx["truth_cut"], fx["mm"],
        fx["C_interp"], fx["fp_model"], fx["X_tot"],
        fx["logN_lo"], fx["logN_hi"], fx["N_b"], fx["dN_b"], dalpha, rng, base)
    return base, res


def test_run_one_tilt_returns_closure_structure():
    """run_one_tilt returns the documented closure structure: point/mc/ttr/pred_f,
    the closure + raw pull families per reduction, and w_op."""
    fx = _synthetic_fixture()
    cfg = fx["cfg"]
    _, res = _run_tilt(fx, 0.5)
    for key in ("dalpha", "point", "mc", "ttr", "pred_f", "f_pull", "f_pull_raw",
                "dndx_z_pull", "dndx_tot_pull", "dndx_tot_pull_raw", "dndx_tot_pred",
                "omega_pull", "omega_pull_raw", "omega_pred", "w_op"):
        assert key in res, f"run_one_tilt result missing key {key}"
    assert res["dalpha"] == pytest.approx(0.5)
    # the per-limit dicts cover every report limit
    for lim in cfg.report_logN_limits:
        assert lim in res["dndx_tot_pull"] and lim in res["omega_pull"]
        assert lim in res["dndx_tot_pred"] and lim in res["omega_pred"]
    # mc carries the raw sample arrays the gate's coverage check reads
    for lim in cfg.report_logN_limits:
        assert res["mc"]["_samples"]["dndx_total"][lim].shape[0] == cfg.n_mc
    # closure prediction pred_f == baseline R0_f * tilted truth f
    base = _run_tilt(fx, 0.5)[0]
    np.testing.assert_allclose(
        res["pred_f"], np.asarray(base["R0_f"]) * np.asarray(res["ttr"]["f_truth"]),
        rtol=1e-12, atol=0, equal_nan=True)


def test_run_one_tilt_rejects_non_purity_mixture_fp():
    """FP-FREEZE GUARD (CS-review F6 / LyA-review F2): run_one_tilt refuses any
    fp_estimator other than purity_mixture (a frozen external background must not be
    tilt-scaled)."""
    fx = _synthetic_fixture()
    cfg = fx["cfg"]
    cfg.fp_estimator = "loa0"   # frozen external background
    rng = np.random.default_rng(0)
    base = T.baseline_recovery(
        cfg, fx["cat_cut"], fx["is_TP"], fx["good_mask"], fx["truth_cut"],
        fx["C_interp"], fx["fp_model"], fx["X_tot"],
        fx["logN_lo"], fx["logN_hi"], fx["N_b"], fx["dN_b"])
    with pytest.raises(NotImplementedError, match="purity-mixture"):
        T.run_one_tilt(
            cfg, fx["cat_cut"], fx["is_TP"], fx["good_mask"], fx["truth_cut"], fx["mm"],
            fx["C_interp"], fx["fp_model"], fx["X_tot"],
            fx["logN_lo"], fx["logN_hi"], fx["N_b"], fx["dN_b"], 0.5, rng, base)


def test_run_one_tilt_dalpha0_control_closes():
    """The Δα=0 CONTROL: with no tilt, w_op ≡ 1, the tilted truth == untilted truth,
    pred_f == baseline R0·truth, and the closure point estimate == the baseline point
    estimate (the closure residual collapses to ~0 -> tiny closure pulls)."""
    fx = _synthetic_fixture()
    cfg = fx["cfg"]
    base, res0 = _run_tilt(fx, 0.0)
    # the per-op tilt weights are all exactly 1.0 at Δα=0
    np.testing.assert_allclose(res0["w_op"], 1.0, atol=0)
    # the tilted-truth f == the untilted baseline truth t0 (machine precision)
    np.testing.assert_allclose(res0["ttr"]["f_truth"], np.asarray(base["t0"]["f_truth"]),
                               rtol=1e-12, atol=0)
    # the point estimate at Δα=0 == the baseline point estimate e0 (same arithmetic,
    # boot_weights of 1.0). Both are UN-clipped.
    np.testing.assert_allclose(res0["point"]["f_b"], np.asarray(base["e0"]["f_b"]),
                               rtol=1e-12, atol=0, equal_nan=True)
    # therefore the closure prediction pred_f == R0_f·t0 == e0 (where truth>0), so the
    # numerator (est − pred) is ~0 -> the closure pulls are negligibly small.
    for lim in cfg.report_logN_limits:
        # est == pred exactly at Δα=0, so the closure pull must be ~0 (any tiny value
        # is just MC σ in the denominator; the numerator is machine-zero).
        est = res0["point"]["dndx_total"][lim]
        pred = res0["dndx_tot_pred"][lim]
        assert est == pytest.approx(pred, rel=1e-9), (
            f"Δα=0 control did not close at dN/dX>={lim}: est={est} pred={pred}")
        est_o = res0["point"]["omega"][lim]
        pred_o = res0["omega_pred"][lim]
        assert est_o == pytest.approx(pred_o, rel=1e-9)


def test_run_one_tilt_fp_term_is_frozen_across_tilts():
    """FP-FREEZE invariant (the term that must divide out across ±tilt): the FROZEN
    forest-FP contribution is the HOSTLESS detections, whose tilt weight is 1.0 by
    construction. So the μ_FP mass attached to hostless rows is IDENTICAL across +tilt,
    −tilt, and Δα=0 — it does NOT move with the tilt. We verify this directly on the
    purity-mixture FP: the hostless-row (1−ρ) sum is tilt-invariant.

    (The purity-mixture (1−ρ) on HOSTED rows correctly co-scales with the row's tilt
    weight — that is the per-object mark, spec §7 — but the genuine forest-FP, which is
    what 'FREEZE b_FP' protects, sits on hostless rows at weight 1.0 and is frozen.)
    """
    fx = _synthetic_fixture()
    cfg = fx["cfg"]
    op = fx["op_mask"]
    host_op = T._tilt_host_nhi(fx["cat_cut"])[op]
    hostless = ~np.isfinite(host_op)
    assert hostless.sum() > 0, "fixture must contain hostless forest-FP detections"
    one_minus_rho = 1.0 - fx["fp_model"].rho       # per op-row
    fp_hostless_base = float(one_minus_rho[hostless].sum())
    for da in (-0.5, 0.0, 0.5):
        w_op = T.detection_tilt_weights(fx["cat_cut"], fx["good_mask"], cfg, da)
        # hostless rows have tilt weight EXACTLY 1.0 regardless of Δα (frozen FP)
        np.testing.assert_allclose(w_op[hostless], 1.0, atol=0)
        # so the FP mass attached to the frozen forest-FP rows is tilt-invariant
        fp_hostless = float((one_minus_rho * w_op)[hostless].sum())
        assert fp_hostless == pytest.approx(fp_hostless_base, rel=1e-12), (
            f"frozen forest-FP term moved with the tilt Δα={da}")


def test_run_one_tilt_plus_minus_tilt_truth_diverge_correct_sign():
    """End-to-end sanity on ±tilt: the +tilt up-weights the DLA tier, so the tilted
    TRUTH dN/dX(>=20.3) is LARGER than the −tilt one (and both differ from untilted).
    This confirms the tilt is actually deposited on the recovered target (not a no-op),
    with the correct Eddington direction."""
    fx = _synthetic_fixture()
    cfg = fx["cfg"]
    _, res_p = _run_tilt(fx, 0.5)
    _, res_m = _run_tilt(fx, -0.5)
    lim = 20.3
    # tilted-truth dN/dX(>=20.3): +tilt up-weights logN>20.3 -> larger than -tilt
    assert res_p["ttr"]["dndx_total"][lim] > res_m["ttr"]["dndx_total"][lim]
    # and the closure prediction tracks the same ordering (pred = R0·truth)
    assert res_p["dndx_tot_pred"][lim] > res_m["dndx_tot_pred"][lim]
