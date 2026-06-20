"""Track-C T-A: tests for the FORWARD response p(x̂ | N_true, SNR, z_QSO) fit.

The forward response is a NEW object (znz_kernel.ForwardResponseModel) distinct from the
ZNZModel posterior-warp.  It is calibrated by binning the truth-match on the TRUE value
N_true (non-circular: reads x̂/N_true/SNR/z_QSO only, never dN/dX/f/Ω).

Grounding numbers it must reproduce (the MEASURED 2LPT-0 forward response,
.superpowers/sdd/track_c_eddington_verify.md):
  - RIGHT-skew ≈ +0.9 in the mid-tier, collapsing to ~0 above N≈21 (prior ceiling)
  - width 0.11 (hi-SNR) → 0.20 (lo-SNR)  (~1.8× swing)
  - up-bias mean(dx) rising with z_QSO  (+0.036 → +0.074)
"""
import inspect
import re

import numpy as np
import pytest
from scipy.stats import skewnorm

from CDDF_analysis.znz_kernel import (
    fit_forward_response,
    measure_forward_response,
    ForwardResponseModel,
    save_forward_response,
    load_forward_response,
    _moment_to_skewnormal,
    _empirical_forward_cells,
    _SN_SKEW_MAX,
)


# ---------------------------------------------------------------------------
# Synthetic truth-match with KNOWN forward-response moments
# ---------------------------------------------------------------------------

def _draw_skewnormal_with_moments(mean, sd, skew, rng):
    """Vectorized: draw x with target (mean, sd, skewness) per element via skew-normal.

    The skewness takes only a SMALL number of distinct values (it is piecewise in the
    synthetic model), so we group by skew level, draw a STANDARDIZED skew-normal (mean 0,
    sd 1) for that shape, and apply the per-element affine (× sd + mean).  This is fully
    vectorized within each skew level (no per-element scipy call).
    """
    mean = np.asarray(mean, float)
    sd = np.asarray(sd, float)
    skew = np.asarray(skew, float)
    out = np.empty(len(mean))
    b = np.sqrt(2.0 / np.pi)
    for sk_val in np.unique(np.round(skew, 4)):
        m = np.abs(skew - sk_val) < 1e-9
        if not m.any():
            continue
        # standardized SN shape for this skewness; standardize to mean 0 / sd 1
        xi, om, a = _moment_to_skewnormal(0.0, 1.0, float(sk_val))
        delta = a / np.sqrt(1.0 + a * a)
        sn_mean = xi + om * b * delta
        sn_var = om * om * (1.0 - (b * delta) ** 2)
        sn_sd = np.sqrt(max(sn_var, 1e-12))
        z = skewnorm.rvs(a, loc=xi, scale=om, size=int(m.sum()), random_state=rng)
        zstd = (z - sn_mean) / sn_sd                       # standardized (mean 0, sd 1)
        out[m] = mean[m] + sd[m] * zstd                    # per-element affine
    return out


def _synthetic_forward_meas(seed=0, n=600000,
                            sd_lo=0.20, sd_hi=0.11,
                            mu_z_lo=0.036, mu_z_hi=0.074,
                            skew_mid=0.90, N_collapse=21.0):
    """Build a truth-match-shaped meas dict whose forward response p(x̂|N_true,SNR,z) has
    KNOWN moments matching the measured 2LPT-0 shape.

    width: sd_lo (low SNR) → sd_hi (high SNR); up-bias: mu_z_lo → mu_z_hi with z_QSO;
    right-skew skew_mid in the DLA tier, collapsing to 0 above N_collapse.
    """
    rng = np.random.default_rng(seed)
    N_true = rng.uniform(19.6, 21.8, n)
    snr = rng.uniform(2.0, 40.0, n)
    zqso = rng.uniform(2.0, 3.8, n)
    fr_snr = np.clip((snr - 2.0) / 38.0, 0, 1)            # 0 (lo) → 1 (hi)
    fr_z = np.clip((zqso - 2.0) / 1.8, 0, 1)              # 0 → 1
    sd = sd_lo + (sd_hi - sd_lo) * fr_snr                 # widens at low SNR
    mu_b = mu_z_lo + (mu_z_hi - mu_z_lo) * fr_z           # rises with z
    skew = np.where(N_true < N_collapse, skew_mid,
                    skew_mid * np.clip((N_collapse + 0.5 - N_true) / 0.5, 0, 1))
    # quantize the (small) high-N collapse ramp to a handful of discrete levels so the
    # vectorized per-skew-level draw stays fast (the DLA tier is a single level skew_mid).
    skew = np.round(skew / 0.15) * 0.15
    dx = _draw_skewnormal_with_moments(mu_b, sd, skew, rng)
    return {"N_true": N_true, "snr": snr, "zqso": zqso, "dx": dx, "xhat": N_true + dx}


# ---------------------------------------------------------------------------
# Grounding: the fit recovers the MEASURED forward response per bin
# ---------------------------------------------------------------------------

def test_forward_fit_recovers_snr_width_swing():
    """Width must swing ~0.11 (hi-SNR) → ~0.20 (lo-SNR) — the measured ~1.8× swing.

    Evaluated at the LOW edge of the lowest SNR bin and the HIGH end of the data so the
    full measured swing is exposed (the fitted per-bin width is the bin-pooled average,
    so the swing is read across the SNR extremes, where the synthetic width is 0.20→0.11).
    Fine SNR edges resolve the swing (a single coarse [6.5,∞) bin would pool it away —
    that pooling is itself a documented binning property, exercised by the next test).
    """
    meas = _synthetic_forward_meas(seed=1)
    # fine SNR edges so the top bin is narrow enough to expose the high-SNR (~0.11) end
    snr_edges = (2.0, 4.0, 8.0, 16.0, 28.0, np.inf)
    frm = fit_forward_response(meas, snr_edges=snr_edges)
    sd_lo = float(frm.sigma(np.array([20.4]), np.array([2.5]), np.array([2.75]))[0])
    sd_hi = float(frm.sigma(np.array([20.4]), np.array([36.0]), np.array([2.75]))[0])
    assert abs(sd_lo - 0.20) < 0.03, f"low-SNR width {sd_lo:.3f} != ~0.20"
    assert abs(sd_hi - 0.11) < 0.03, f"high-SNR width {sd_hi:.3f} != ~0.11"
    assert sd_lo > sd_hi + 0.05, (
        f"width did not widen at low SNR: lo={sd_lo:.3f} hi={sd_hi:.3f}")
    # the swing ratio approximates the measured ~1.8×
    assert 1.4 < sd_lo / sd_hi < 2.3, f"width swing ratio {sd_lo / sd_hi:.2f} not ~1.8×"


def test_forward_fit_recovers_mid_tier_right_skew():
    """Mid-tier skew must be right-skewed ≈+0.9 (the measured +0.93)."""
    meas = _synthetic_forward_meas(seed=2)
    frm = fit_forward_response(meas)
    sk = float(frm.skew(np.array([20.4]), np.array([5.0]), np.array([2.75]))[0])
    assert sk > 0.0, f"mid-tier skew not right-skewed: {sk:+.3f}"
    assert abs(sk - 0.90) < 0.20, f"mid-tier skew {sk:+.3f} != ~+0.9"


def test_forward_fit_recovers_z_up_bias():
    """Up-bias mean(dx) must RISE with z_QSO (+0.036 → +0.074)."""
    meas = _synthetic_forward_meas(seed=3)
    frm = fit_forward_response(meas)
    mu_lo = float(frm.mu_b(np.array([20.4]), np.array([5.0]), np.array([2.15]))[0])
    mu_hi = float(frm.mu_b(np.array([20.4]), np.array([5.0]), np.array([3.6]))[0])
    assert mu_hi > mu_lo + 0.01, (
        f"up-bias did not rise with z_QSO: lo-z={mu_lo:+.4f} hi-z={mu_hi:+.4f}")
    assert abs(mu_lo - 0.036) < 0.020, f"low-z up-bias {mu_lo:+.4f} != ~+0.036"
    assert abs(mu_hi - 0.074) < 0.020, f"high-z up-bias {mu_hi:+.4f} != ~+0.074"


def test_forward_fit_matches_smoothed_empirical_per_bin():
    """The parametric (skew-normal-surface) fit must match the smoothed-EMPIRICAL per-bin
    response (mean/sd/skew) to tolerance — the cross-check the brief requires."""
    meas = _synthetic_forward_meas(seed=4)
    snr_edges = (2.0, 3.5, 6.5, np.inf)
    z_edges = (0.0, 2.56, 2.96, np.inf)
    frm = fit_forward_response(meas, snr_edges=snr_edges, z_edges=z_edges)
    cells = _empirical_forward_cells(
        meas["N_true"], meas["snr"], meas["zqso"], meas["dx"],
        np.asarray(snr_edges, float), np.asarray(z_edges, float),
        n_N_cells=7, min_count=60)
    worst_mu = worst_sd = worst_sk = 0.0
    n_checked = 0
    for (a, b), cl in cells.items():
        # representative SNR / z value inside this cell (the bin midpoint, finite-capped)
        se = np.asarray(snr_edges, float)
        ze = np.asarray(z_edges, float)
        s_rep = se[a] + 0.5 if not np.isfinite(se[a + 1]) else 0.5 * (se[a] + se[a + 1])
        z_rep = ze[b] + 0.3 if not np.isfinite(ze[b + 1]) else 0.5 * (ze[b] + ze[b + 1])
        for (Nc, mn, sd, sk, c) in cl:
            if Nc >= frm.N_skew_collapse:     # skew-collapse region: skip skew check
                continue
            mu_fit = float(frm.mu_b(np.array([Nc]), np.array([s_rep]), np.array([z_rep]))[0])
            sd_fit = float(frm.sigma(np.array([Nc]), np.array([s_rep]), np.array([z_rep]))[0])
            sk_fit = float(frm.skew(np.array([Nc]), np.array([s_rep]), np.array([z_rep]))[0])
            worst_mu = max(worst_mu, abs(mu_fit - mn))
            worst_sd = max(worst_sd, abs(sd_fit - sd))
            worst_sk = max(worst_sk, abs(sk_fit - min(sk, 0.995 * _SN_SKEW_MAX)))
            n_checked += 1
    assert n_checked > 10, f"too few cells checked ({n_checked})"
    assert worst_mu < 0.03, f"parametric mu vs empirical worst deviation {worst_mu:.4f}"
    assert worst_sd < 0.04, f"parametric sd vs empirical worst deviation {worst_sd:.4f}"
    assert worst_sk < 0.35, f"parametric skew vs empirical worst deviation {worst_sk:.4f}"


# ---------------------------------------------------------------------------
# High-N collapse (prior ceiling) — no spurious right-skew extrapolation
# ---------------------------------------------------------------------------

def test_forward_high_N_skew_collapse():
    """Skew must ramp to ~0 above N≈21 (the prior-ceiling collapse), not extrapolate a
    spurious right-skew."""
    meas = _synthetic_forward_meas(seed=5)
    frm = fit_forward_response(meas, N_skew_collapse=21.0)
    sk_mid = float(frm.skew(np.array([20.4]), np.array([5.0]), np.array([2.75]))[0])
    sk_21 = float(frm.skew(np.array([21.0]), np.array([5.0]), np.array([2.75]))[0])
    sk_215 = float(frm.skew(np.array([21.5]), np.array([5.0]), np.array([2.75]))[0])
    sk_22 = float(frm.skew(np.array([22.0]), np.array([5.0]), np.array([2.75]))[0])
    assert sk_mid > 0.4, f"mid-tier skew unexpectedly small: {sk_mid:+.3f}"
    assert abs(sk_215) < 0.15, f"skew not collapsed at N=21.5: {sk_215:+.3f}"
    assert sk_22 == pytest.approx(0.0, abs=1e-9), (
        f"skew not exactly 0 well above collapse (N=22): {sk_22:+.3f}")
    # monotone collapse across the ramp window
    assert sk_21 >= sk_215 - 1e-9, "skew did not decrease across the collapse ramp"


def test_forward_skew_never_exceeds_skewnormal_ceiling():
    """The skew surface output is always within the skew-normal attainable ceiling
    (no impossible shape parameter demanded downstream)."""
    meas = _synthetic_forward_meas(seed=6)
    frm = fit_forward_response(meas)
    Ng = np.linspace(19.6, 22.0, 25)
    for s in (2.5, 5.0, 20.0):
        for z in (2.2, 2.75, 3.4):
            g = frm.skew(Ng, np.full_like(Ng, s), np.full_like(Ng, z))
            assert np.all(np.abs(g) <= _SN_SKEW_MAX + 1e-9), (
                f"skew exceeds SN ceiling at SNR={s}, z={z}: max|g|={np.max(np.abs(g)):.3f}")


# ---------------------------------------------------------------------------
# Non-circular signature gate (the pattern T2 used)
# ---------------------------------------------------------------------------

def test_forward_fit_is_noncircular_signature():
    """NON-CIRCULAR gate: neither fit_forward_response nor measure_forward_response exposes
    a reduced-statistic (dN/dX / Ω / f(N,z) / D_c / R0) argument — the fit can read ONLY
    the truth-match conditional (x̂, N_true, SNR, z_QSO).  (``dx`` and ``N_true`` are the
    LEGIT conditional inputs; only DOWNSTREAM reductions are forbidden.)"""
    forbidden = {"dndx", "dndz", "omega", "ellz", "fnz", "cddf", "r0", "reduce",
                 "dc", "fbk", "dndxz", "f"}
    for fn in (fit_forward_response, measure_forward_response):
        params = set(inspect.signature(fn).parameters)
        bad = {p for p in params if set(re.split(r"[_]", p.lower())) & forbidden}
        assert not bad, (
            f"{fn.__name__} exposes a reduced-statistic argument {bad} — would open the "
            "α=1/R0 circular edge; the forward fit must read only the truth-match conditional")
    # the response measurement returns only the conditioning vars + residual
    meas_keys = {"N_true", "snr", "zqso", "dx", "xhat"}
    sig = inspect.signature(measure_forward_response)
    # measure_forward_response's data arg is the catalog table; it must NOT take dN/dX etc.
    assert "cat_cut" in sig.parameters, "measure_forward_response should read the catalog"
    # double-check there is no population-quantity keyword anywhere
    allnames = " ".join(sig.parameters) + " " + " ".join(
        inspect.signature(fit_forward_response).parameters)
    assert "dndx" not in allnames.lower() and "omega" not in allnames.lower()


# ---------------------------------------------------------------------------
# save/load round-trip + reproducibility
# ---------------------------------------------------------------------------

def test_forward_save_load_roundtrip(tmp_path):
    """save/load round-trips every field and the surfaces evaluate identically."""
    meas = _synthetic_forward_meas(seed=7)
    frm = fit_forward_response(meas)
    path = str(tmp_path / "fwd.npz")
    save_forward_response(path, frm)
    frm2 = load_forward_response(path)
    np.testing.assert_array_equal(frm2.mu_coef, frm.mu_coef)
    np.testing.assert_array_equal(frm2.sig_coef, frm.sig_coef)
    np.testing.assert_array_equal(frm2.skew_coef, frm.skew_coef)
    np.testing.assert_array_equal(frm2.snr_edges, frm.snr_edges)
    np.testing.assert_array_equal(frm2.z_edges, frm.z_edges)
    assert frm2.N_ref == frm.N_ref
    assert frm2.deg_N == frm.deg_N
    assert frm2.N_skew_collapse == frm.N_skew_collapse
    # surfaces evaluate identically after reload
    Ne = np.array([20.0, 20.4, 21.0]); se = np.array([2.5, 5.0, 20.0]); ze = np.array([2.2, 2.75, 3.4])
    np.testing.assert_array_equal(frm2.mu_b(Ne, se, ze), frm.mu_b(Ne, se, ze))
    np.testing.assert_array_equal(frm2.sigma(Ne, se, ze), frm.sigma(Ne, se, ze))
    np.testing.assert_array_equal(frm2.skew(Ne, se, ze), frm.skew(Ne, se, ze))
    xi2, om2, a2 = frm2.response_skewnormal(Ne, se, ze)
    xi1, om1, a1 = frm.response_skewnormal(Ne, se, ze)
    np.testing.assert_array_equal(xi2, xi1)
    np.testing.assert_array_equal(om2, om1)
    np.testing.assert_array_equal(a2, a1)


def test_forward_fit_is_reproducible():
    """Same input (same seed → same meas) ⇒ identical surfaces (deterministic, no RNG in
    the fit itself)."""
    meas_a = _synthetic_forward_meas(seed=11)
    meas_b = _synthetic_forward_meas(seed=11)   # identical draw
    frm_a = fit_forward_response(meas_a)
    frm_b = fit_forward_response(meas_b)
    np.testing.assert_array_equal(frm_a.mu_coef, frm_b.mu_coef)
    np.testing.assert_array_equal(frm_a.sig_coef, frm_b.sig_coef)
    np.testing.assert_array_equal(frm_a.skew_coef, frm_b.skew_coef)
    # re-fitting the SAME meas twice is bit-identical (no hidden RNG state)
    frm_a2 = fit_forward_response(meas_a)
    np.testing.assert_array_equal(frm_a.skew_coef, frm_a2.skew_coef)


def test_forward_response_skewnormal_mean_matches_surface():
    """response_skewnormal's realized MEAN equals N + μ_b (the count-fixing location is
    correct), and the realized skewness sign matches the surface skew."""
    from scipy.stats import skewnorm as _sn
    meas = _synthetic_forward_meas(seed=12)
    frm = fit_forward_response(meas)
    N = np.array([20.4]); s = np.array([5.0]); z = np.array([2.75])
    xi, om, a = frm.response_skewnormal(N, s, z)
    mean_realized = _sn.mean(a[0], loc=xi[0], scale=om[0])
    mean_target = float(N[0] + frm.mu_b(N, s, z)[0])
    assert abs(mean_realized - mean_target) < 1e-6, (
        f"skew-normal mean {mean_realized:.4f} != N+mu_b {mean_target:.4f}")
    # skewness sign matches the right-skew surface
    skew_realized = _sn.stats(a[0], moments="s")
    assert np.sign(skew_realized) == np.sign(frm.skew(N, s, z)[0])
