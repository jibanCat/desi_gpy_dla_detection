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

from CDDF_analysis.hbi.znz_kernel import (
    fit_forward_response,
    measure_forward_response,
    ForwardResponseModel,
    EmpiricalForwardDensity,
    build_empirical_forward_density,
    save_forward_response,
    load_forward_response,
    _moment_to_skewnormal,
    _moment_to_skewnormal_vec,
    _empirical_forward_cells,
    _SN_SKEW_MAX,
)

_TRAPZ = getattr(np, "trapezoid", getattr(np, "trapz", None))


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
    # the empirical density block round-trips too (built by default) and evaluates identically
    assert frm.emp is not None and frm2.emp is not None
    np.testing.assert_array_equal(frm2.emp.rho, frm.emp.rho)
    np.testing.assert_array_equal(frm2.emp.N_anchors, frm.emp.N_anchors)
    np.testing.assert_array_equal(frm2.emp.r_grid, frm.emp.r_grid)
    xh = np.array([20.5, 20.7, 21.1])
    np.testing.assert_array_equal(
        frm2.response_density_empirical(xh, Ne, se, ze),
        frm.response_density_empirical(xh, Ne, se, ze))


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


# ---------------------------------------------------------------------------
# Track-C T-BC: vectorized moment-match + forward-density (the deconvolution kernel A)
# ---------------------------------------------------------------------------

def test_moment_to_skewnormal_vec_matches_scalar_loop():
    """The VECTORIZED _moment_to_skewnormal_vec matches the scalar _moment_to_skewnormal
    element-wise to 1e-12 (CS minor: the per-N loop was too slow; vectorize without drift).
    Covers right-skew, left-skew, near-symmetric, ceiling-clamped, and varied (mean, sd)."""
    rng = np.random.default_rng(7)
    mean = rng.uniform(19.5, 22.0, 500)
    sd = rng.uniform(0.05, 0.30, 500)
    skew = np.concatenate([
        rng.uniform(-0.99 * _SN_SKEW_MAX, 0.99 * _SN_SKEW_MAX, 480),
        np.array([0.0, 1e-12, -1e-12, _SN_SKEW_MAX * 2, -_SN_SKEW_MAX * 2,
                  0.9, -0.9, 0.5, 1e-10, -1e-10, 0.93, 0.0, -0.0, 0.3, 0.7, -0.4,
                  0.8, -0.8, 0.6, -0.6]),
    ])
    xi_v, om_v, a_v = _moment_to_skewnormal_vec(mean, sd, skew)
    for i in range(len(mean)):
        xi_s, om_s, a_s = _moment_to_skewnormal(mean[i], sd[i], skew[i])
        assert abs(xi_v[i] - xi_s) < 1e-12, f"xi[{i}] {xi_v[i]} != {xi_s}"
        assert abs(om_v[i] - om_s) < 1e-12, f"omega[{i}] {om_v[i]} != {om_s}"
        assert abs(a_v[i] - a_s) < 1e-12, f"a[{i}] {a_v[i]} != {a_s}"


def test_forward_response_density_is_skewnorm_pdf_at_xhat():
    """response_density(x̂, N, snr, zqso) equals scipy skewnorm.pdf(x̂; ξ(N),ω(N),a(N))
    EXACTLY — the forward-likelihood density at the observed x̂ as a function of true N.
    This is the per-cell value the deconvolution kernel A is built from (T-BC correctness)."""
    from scipy.stats import skewnorm as _sn
    meas = _synthetic_forward_meas(seed=3)
    frm = fit_forward_response(meas)
    # one detection's observed x̂; vary the TRUE N (the kernel column axis)
    xhat = 20.6
    N = np.linspace(19.6, 21.8, 23)
    snr = np.full_like(N, 5.0)
    zqso = np.full_like(N, 2.75)
    dens = frm.response_density(np.full_like(N, xhat), N, snr, zqso)
    xi, om, a = frm.response_skewnormal(N, snr, zqso)
    ref = _sn.pdf(np.full_like(N, xhat), a, loc=xi, scale=om)
    np.testing.assert_allclose(dens, ref, rtol=0, atol=1e-12)
    # the density is a DENSITY (per unit x̂), NOT normalized over N: Σ_N density·ΔN ≠ 1.
    # (the Σ_N≠1 property is the whole point of the forward kernel vs the renormalized
    # posterior). Its INTEGRAL over x̂ at fixed N is 1 — verify that instead.
    fine_x = np.linspace(18.0, 24.0, 60001)
    xi0, om0, a0 = frm.response_skewnormal(np.array([20.4]), np.array([5.0]),
                                           np.array([2.75]))
    pdf_over_x = _sn.pdf(fine_x, a0[0], loc=xi0[0], scale=om0[0])
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy 2.x compat
    assert abs(_trapz(pdf_over_x, fine_x) - 1.0) < 1e-4, "p(x̂|N) must integrate to 1 over x̂"


def test_forward_response_density_right_skew_tail_heavier_above():
    """The forward density carries the measured RIGHT-skew: for a detection observed near
    the response mode, the density as a function of true N is HEAVIER toward LOWER true N
    (because the response up-scatters x̂ ABOVE the true N → an x̂ is more likely to have come
    from a true N BELOW it). The Eddington emergence at the kernel level."""
    meas = _synthetic_forward_meas(seed=5)
    frm = fit_forward_response(meas)
    xhat = 20.5
    snr = np.array([5.0]); z = np.array([2.75])
    # density at a true N below x̂ vs symmetric distance above
    d_below = frm.response_density(np.array([xhat]), np.array([xhat - 0.25]), snr, z)[0]
    d_above = frm.response_density(np.array([xhat]), np.array([xhat + 0.25]), snr, z)[0]
    # right-skewed forward (x̂ up-scattered) ⇒ given x̂, the true N just below is more
    # probable than the symmetric point above (the kernel leans the recovery low = Eddington)
    assert d_below > d_above, (
        f"forward density not leaning low: below={d_below:.4f} above={d_above:.4f}")


# ---------------------------------------------------------------------------
# Track-C T-BC FIX: the GENUINE smoothed-empirical per-cell forward density
# ---------------------------------------------------------------------------

def _emp_meas_narrowing(seed=0, n=400000):
    """Truth-match whose forward response NARROWS with N + SKEW-COLLAPSES at high N — the
    true high-N shape the parametric skew-normal moment-fit OVERSHOOTS (one SNR × one z)."""
    rng = np.random.default_rng(seed)
    N_true = rng.uniform(19.6, 21.8, n)
    snr = np.full(n, 5.0)
    zqso = np.full(n, 2.75)
    sd = 0.22 - 0.10 * np.clip((N_true - 20.0) / 1.8, 0.0, 1.0)   # width NARROWS with N
    sk = np.where(N_true < 21.0, 0.9, 0.0)                        # skew collapses high-N
    xi, om, a = _moment_to_skewnormal_vec(np.zeros(n), sd, sk)
    dx = skewnorm.rvs(a, loc=xi, scale=om, random_state=rng)
    return {"N_true": N_true, "snr": snr, "zqso": zqso, "dx": dx, "xhat": N_true + dx}


def test_empirical_density_integrates_to_one_over_xhat():
    """∫ p(x̂|N) dx̂ = 1 at fixed N (the normalization gate) across the DLA tier."""
    frm = fit_forward_response(_emp_meas_narrowing(seed=1),
                               snr_edges=(0.0, np.inf), z_edges=(0.0, np.inf))
    assert frm.emp is not None
    fine = np.linspace(18.0, 24.0, 60001)
    for N0 in (20.0, 20.4, 20.8, 21.2):
        d = frm.response_density_empirical(fine, np.full_like(fine, N0),
                                           np.full_like(fine, 5.0), np.full_like(fine, 2.75))
        assert abs(_TRAPZ(d, fine) - 1.0) < 1e-3, f"∫p(x̂|N={N0})dx̂ != 1"
        assert np.all(d >= 0.0), "density must be non-negative"


def _emp_width(frm, N0, snr=5.0, z=2.75):
    r = frm.emp.r_grid
    d = frm.response_density_empirical(N0 + r, np.full_like(r, N0),
                                       np.full_like(r, snr), np.full_like(r, z))
    d = d / _TRAPZ(d, r)
    m = _TRAPZ(r * d, r)
    return float(np.sqrt(_TRAPZ((r - m) ** 2 * d, r)))


def test_empirical_density_narrows_with_N():
    """The empirical density CARRIES the true high-N narrowing: realized width at N=21.2 is
    smaller than at N=20.0 (the parametric overshoot's lever)."""
    frm = fit_forward_response(_emp_meas_narrowing(seed=2),
                               snr_edges=(0.0, np.inf), z_edges=(0.0, np.inf))
    w_lo = _emp_width(frm, 20.0)
    w_hi = _emp_width(frm, 21.2)
    # the empirical density NARROWS with N (the lever): realized width drops by the injected
    # ~0.10-dex swing direction (the realized width is the true σ smoothed by the σ=0.05-dex
    # KDE in quadrature, so it is BROADER than the bare injected value — the swing, not the
    # absolute level, is the diagnostic).
    assert w_hi < w_lo - 0.05, f"empirical width did not narrow: lo={w_lo:.3f} hi={w_hi:.3f}"
    # the realized widths bracket the injected 0.22→0.12 (KDE-broadened by ≈0.05 in quad)
    assert 0.18 < w_lo < 0.28, f"low-N realized width {w_lo:.3f} out of range"
    assert 0.10 < w_hi < 0.18, f"high-N realized width {w_hi:.3f} out of range"


def test_empirical_density_save_load_and_eval_identical(tmp_path):
    """The EmpiricalForwardDensity round-trips through save/load and evaluates byte-identical."""
    frm = fit_forward_response(_emp_meas_narrowing(seed=3),
                               snr_edges=(0.0, np.inf), z_edges=(0.0, np.inf))
    path = str(tmp_path / "fwd_emp.npz")
    save_forward_response(path, frm)
    frm2 = load_forward_response(path)
    assert isinstance(frm2.emp, EmpiricalForwardDensity)
    xh = np.linspace(19.8, 21.6, 37)
    N = np.full_like(xh, 20.5); s = np.full_like(xh, 5.0); z = np.full_like(xh, 2.75)
    np.testing.assert_array_equal(
        frm2.response_density_empirical(xh, N, s, z),
        frm.response_density_empirical(xh, N, s, z))


def test_empirical_density_backward_compat_missing_block_is_none(tmp_path):
    """A parametric-only model (build_empirical=False) loads with emp=None; the legacy NPZ
    (no empirical block) is still readable; response_density_empirical raises clearly."""
    frm = fit_forward_response(_emp_meas_narrowing(seed=4),
                               snr_edges=(0.0, np.inf), z_edges=(0.0, np.inf),
                               build_empirical=False)
    assert frm.emp is None
    path = str(tmp_path / "fwd_param_only.npz")
    save_forward_response(path, frm)
    # the saved NPZ has NO empirical block
    d = np.load(path, allow_pickle=True)
    assert "emp_rho" not in d.files
    frm2 = load_forward_response(path)
    assert frm2.emp is None
    with pytest.raises(ValueError, match="empirical"):
        frm2.response_density_empirical(np.array([20.5]), np.array([20.4]),
                                        np.array([5.0]), np.array([2.75]))


def test_build_empirical_forward_density_is_reproducible():
    """The empirical build is DETERMINISTIC (histogram + fixed-σ smoothing, no RNG): two
    builds from the same meas are byte-identical."""
    meas = _emp_meas_narrowing(seed=5)
    e1 = build_empirical_forward_density(meas, snr_edges=(0.0, np.inf), z_edges=(0.0, np.inf))
    e2 = build_empirical_forward_density(meas, snr_edges=(0.0, np.inf), z_edges=(0.0, np.inf))
    np.testing.assert_array_equal(e1.rho, e2.rho)
    np.testing.assert_array_equal(e1.N_anchors, e2.N_anchors)
    np.testing.assert_array_equal(e1.r_grid, e2.r_grid)


def test_build_empirical_forward_density_is_noncircular_signature():
    """REDUCE-ONLY / NON-CIRCULAR: the empirical builder reads the truth-match conditional
    only (N_true/snr/zqso/dx) — no dN/dX / f / Ω argument anywhere in its signature."""
    sig = inspect.signature(build_empirical_forward_density)
    allnames = " ".join(sig.parameters).lower()
    assert "dndx" not in allnames and "omega" not in allnames and "f_b" not in allnames
    # the meas keys it consumes are the conditional only
    src = inspect.getsource(build_empirical_forward_density)
    for forbidden in ("dndx", "omega", '"f"', "f_of_N"):
        assert forbidden not in src.lower(), f"empirical builder references {forbidden}"


# ---------------------------------------------------------------------------
# Track-C T-D: resample-aware forward response (kernel-calibration uncertainty carry)
# ---------------------------------------------------------------------------

def test_forward_fit_weights_unit_reproduces_unweighted():
    """fit_forward_response(meas, weights=ones) == fit_forward_response(meas) bit-for-bit
    (the unit-weight invariance the marginalized band rests on — boot_mult==1 ⇒ point fit)."""
    meas = _synthetic_forward_meas(seed=21)
    frm0 = fit_forward_response(meas)
    w = np.ones(len(meas["N_true"]))
    frm1 = fit_forward_response(meas, weights=w)
    np.testing.assert_array_equal(frm0.mu_coef, frm1.mu_coef)
    np.testing.assert_array_equal(frm0.sig_coef, frm1.sig_coef)
    np.testing.assert_array_equal(frm0.skew_coef, frm1.skew_coef)
    # the empirical density block too
    np.testing.assert_array_equal(frm0.emp.rho, frm1.emp.rho)
    np.testing.assert_array_equal(frm0.emp.N_anchors, frm1.emp.N_anchors)


def test_forward_fit_weighted_subbin_moment_matches_replication():
    """The per-sub-bin WEIGHTED moments equal the moments of the row-replicated population
    WITHIN FIXED N sub-bin edges (the genuine correctness of the weighted reduction). The
    sub-bin EDGES are set by unweighted np.quantile (preserving unit-weight byte-identity),
    so we compare the cell moments computed on the SAME edge geometry rather than the fitted
    surfaces (whose edges drift under replication)."""
    from CDDF_analysis.hbi.znz_kernel import _empirical_forward_cells
    rng = np.random.default_rng(3)
    n = 60000
    N_true = rng.uniform(19.7, 21.3, n)
    snr = np.full(n, 5.0); zqso = np.full(n, 2.7)
    dx = rng.normal(0.05, 0.15, n) + 0.02 * (N_true - 20.0)   # mild N-dependence
    mult = rng.integers(0, 4, size=n).astype(float)           # 0..3 copies per object
    edges_snr = (0.0, np.inf); edges_z = (0.0, np.inf)
    cells_w = _empirical_forward_cells(N_true, snr, zqso, dx, edges_snr, edges_z,
                                       n_N_cells=5, min_count=10, weights=mult)
    # explicit replication
    idx = np.repeat(np.arange(n), mult.astype(int))
    cells_rep = _empirical_forward_cells(N_true[idx], snr[idx], zqso[idx], dx[idx],
                                         edges_snr, edges_z, n_N_cells=5, min_count=10)
    cw = np.asarray(cells_w[(0, 0)]); cr = np.asarray(cells_rep[(0, 0)])
    assert len(cw) == len(cr) and len(cw) >= 3
    # N_center, weighted mean(dx), weighted sd(dx) agree closely; the small residual is the
    # quantile-edge geometry difference (unweighted quantile on the unique vs replicated set).
    np.testing.assert_allclose(cw[:, 0], cr[:, 0], rtol=0, atol=6e-3)   # N center
    np.testing.assert_allclose(cw[:, 1], cr[:, 1], rtol=0, atol=3e-3)   # weighted mean(dx)
    np.testing.assert_allclose(cw[:, 2], cr[:, 2], rtol=0, atol=3e-3)   # weighted sd(dx)
    # effective (weighted) count tracks the replicated row count to within edge drift
    np.testing.assert_allclose(cw[:, 4].sum(), cr[:, 4].sum(), rtol=2e-3)


def test_forward_fit_weights_noncircular_signature():
    """The weights argument is a per-object resample multiplicity, NOT a reduced statistic:
    fit_forward_response's signature still exposes no dN/dX / f / Ω, only `weights`."""
    sig = inspect.signature(fit_forward_response)
    names = " ".join(sig.parameters).lower()
    assert "weights" in names
    assert "dndx" not in names and "omega" not in names and "f_b" not in names


def test_refit_forward_response_unit_weight_reproduces_point():
    """refit_forward_response_from_resample(rfr, ones) reproduces the frozen point
    ForwardResponseModel surfaces to ~1e-9 (the Stage-III unit-weight invariance)."""
    from CDDF_analysis.hbi.znz_kernel import (
        build_forward_response_fit_resample, refit_forward_response_from_resample)
    meas = _synthetic_forward_meas(seed=23, n=120000)
    det_tids = np.arange(len(meas["N_true"]), dtype=np.int64)   # 1 detection per TID
    uniq = np.unique(det_tids)
    frm_point = fit_forward_response(meas)
    rfr = build_forward_response_fit_resample(meas, det_tids, uniq, frm_point)
    frm_unit = refit_forward_response_from_resample(rfr, np.ones(len(uniq)))
    np.testing.assert_allclose(frm_unit.mu_coef, frm_point.mu_coef, rtol=0, atol=1e-9)
    np.testing.assert_allclose(frm_unit.sig_coef, frm_point.sig_coef, rtol=0, atol=1e-9)
    np.testing.assert_allclose(frm_unit.skew_coef, frm_point.skew_coef, rtol=0, atol=1e-9)


def test_refit_forward_response_perturbs_and_is_reproducible():
    """A non-unit boot_mult PERTURBS the surfaces (genuine resample), and the SAME mult
    gives the SAME fit (deterministic — no hidden RNG)."""
    from CDDF_analysis.hbi.znz_kernel import (
        build_forward_response_fit_resample, refit_forward_response_from_resample)
    meas = _synthetic_forward_meas(seed=24, n=120000)
    det_tids = np.arange(len(meas["N_true"]), dtype=np.int64)
    uniq = np.unique(det_tids)
    frm_point = fit_forward_response(meas)
    rfr = build_forward_response_fit_resample(meas, det_tids, uniq, frm_point)
    rng = np.random.default_rng(9)
    mult = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq))).astype(float)
    frm_a = refit_forward_response_from_resample(rfr, mult)
    frm_b = refit_forward_response_from_resample(rfr, mult.copy())
    np.testing.assert_array_equal(frm_a.sig_coef, frm_b.sig_coef)   # same mult → same fit
    # perturbed away from the point (at least one surface differs materially)
    moved = (not np.allclose(frm_a.mu_coef, frm_point.mu_coef, atol=1e-6)
             or not np.allclose(frm_a.sig_coef, frm_point.sig_coef, atol=1e-6))
    assert moved, "non-unit boot_mult did not perturb the forward surfaces"


def test_refit_forward_response_noncircular_signature():
    """refit_forward_response_from_resample takes only the resample table + multiplicity +
    fit knobs — no reduced statistic (the α=1/R0 tautology stays structurally impossible)."""
    from CDDF_analysis.hbi.znz_kernel import refit_forward_response_from_resample
    sig = inspect.signature(refit_forward_response_from_resample)
    names = " ".join(sig.parameters).lower()
    assert "boot_mult" in names
    assert "dndx" not in names and "omega" not in names and "f_b" not in names
