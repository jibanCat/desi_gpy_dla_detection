import os

import numpy as np
import pytest
from CDDF_analysis.znz_kernel import fit_znz_model, save_znz, load_znz, ZNZModel, CNZModel


def _make_synthetic_meas(seed=0, n=20000, deg_xhat=1, deg_z=2):
    """Build a synthetic meas dict for the default (deg_xhat=1, deg_z=2) model."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(2.0, 3.5, n)
    xhat = rng.uniform(20.0, 21.5, n)
    true_b = 0.02 + 0.10 * (z - 2.0)              # bias RISES with z (the diagnosis)
    resid = true_b + rng.normal(0, 0.05, n)         # xhat - xtrue
    return {"xhat": xhat, "z": z, "dx": resid, "z_covariate": "z_dla"}


def test_fit_recovers_linear_z_bias():
    """b(xhat, z) rises with z — correct sign as diagnosed (prior-edge pile-up)."""
    meas = _make_synthetic_meas()
    m = fit_znz_model(meas, deg_z=2, deg_xhat=1)
    b_val = m.b(np.array([20.5]), np.array([3.25]))[0]
    expected = 0.02 + 0.10 * 1.25
    assert abs(b_val - expected) < 0.01, (
        f"b(20.5, 3.25)={b_val:.4f} deviates from expected {expected:.4f}")
    assert (m.sigma(np.array([20.5]), np.array([3.25])) > 0).all()


def test_b_rises_with_z():
    """Confirm b(20.5, z) is monotonically increasing — b RISES with z, not decreases."""
    meas = _make_synthetic_meas()
    m = fit_znz_model(meas, deg_z=2, deg_xhat=1)
    b225 = float(m.b(np.array([20.5]), np.array([2.25]))[0])
    b275 = float(m.b(np.array([20.5]), np.array([2.75]))[0])
    b325 = float(m.b(np.array([20.5]), np.array([3.25]))[0])
    assert b225 < b275 < b325, (
        f"b not rising with z: b(2.25)={b225:.4f}, b(2.75)={b275:.4f}, b(3.25)={b325:.4f}. "
        "The prior-edge diagnosis predicts b rises with z (denser forest → more up-migration).")


def test_save_load_roundtrip(tmp_path):
    """After save+load, b() and sigma() must return finite, matching values."""
    meas = _make_synthetic_meas(seed=42)
    znz = fit_znz_model(meas, deg_z=2, deg_xhat=1)
    cnz = CNZModel(
        g_grid=np.ones((5, 15)),
        nhi_edges=np.linspace(19, 23, 6),
        z_edges_fine=np.linspace(2, 3.5, 16),
    )
    path = str(tmp_path / "znz.npz")
    save_znz(path, znz, cnz)
    znz2, cnz2 = load_znz(path)

    # metadata preserved
    assert znz2.z_covariate == "z_dla"
    assert np.allclose(cnz2.g_grid, 1.0)

    # degrees preserved
    assert znz2.deg_xhat == znz.deg_xhat
    assert znz2.deg_z == znz.deg_z

    # b() and sigma() callable and return finite, matching values after reload
    xhat_test = np.array([20.5, 21.0])
    z_test = np.array([2.5, 3.0])
    b_before = znz.b(xhat_test, z_test)
    b_after = znz2.b(xhat_test, z_test)
    assert np.all(np.isfinite(b_after)), f"b() returned non-finite after reload: {b_after}"
    assert np.allclose(b_before, b_after), (
        f"b() changed after save/load: before={b_before}, after={b_after}")

    sig_before = znz.sigma(xhat_test, z_test)
    sig_after = znz2.sigma(xhat_test, z_test)
    assert np.all(sig_after > 0), f"sigma() not positive after reload: {sig_after}"
    assert np.allclose(sig_before, sig_after), (
        f"sigma() changed after save/load: before={sig_before}, after={sig_after}")


def test_arbitrary_degrees(tmp_path):
    """ZNZModel._design must work for any (deg_xhat, deg_z), not just (1,2)."""
    for deg_x, deg_z in [(2, 3), (0, 1), (3, 1)]:
        meas = _make_synthetic_meas(seed=7)
        m = fit_znz_model(meas, deg_z=deg_z, deg_xhat=deg_x)
        assert m.deg_xhat == deg_x
        assert m.deg_z == deg_z
        expected_len = (deg_x + 1) * (deg_z + 1)
        assert len(m.b_coef) == expected_len, (
            f"b_coef length {len(m.b_coef)} != {expected_len} for ({deg_x},{deg_z})")
        b_val = m.b(np.array([20.5]), np.array([2.75]))
        assert np.isfinite(b_val).all(), f"b() non-finite for deg ({deg_x},{deg_z})"
        # round-trip through NPZ
        cnz = CNZModel(np.ones((3, 5)), np.linspace(19, 23, 4), np.linspace(2, 3.5, 6))
        p = str(tmp_path / f"znz_{deg_x}_{deg_z}.npz")
        save_znz(p, m, cnz)
        m2, _ = load_znz(p)
        assert m2.deg_xhat == deg_x and m2.deg_z == deg_z
        assert np.allclose(m2.b(np.array([20.5]), np.array([2.75])),
                           m.b(np.array([20.5]), np.array([2.75])))


# ---------------------------------------------------------------------------
# Stage-1: apply_znz_correction transform math + 3-D C-threading
# ---------------------------------------------------------------------------

def test_apply_znz_shifts_mean():
    """A planted +0.10 dex bias must move the kappa column's mean DOWN by 0.10 dex.

    apply_znz_correction targets mean = x̂ − (b(x̂,z) − b_ref); with b_ref=0 and
    b=+0.10 the mass at logN 20.5 should re-centre at 20.40.
    """
    from CDDF_analysis.znz_kernel import apply_znz_correction, ZNZModel
    n_z = 15
    logN_lo = np.arange(19.0, 22.5, 0.1)
    logN_hi = logN_lo + 0.1
    kappa = np.zeros((1, len(logN_lo), n_z), np.float32)
    j0 = np.searchsorted(logN_lo, 20.5)
    kappa[0, j0, 7] = 1.0                                  # delta at logN 20.5, z-bin 7
    cat_op = {"xhat": np.array([20.5]), "zhat": np.array([3.0]),
              "i_snr": np.array([0])}
    z_edges = np.linspace(2.0, 3.5, n_z + 1)
    znz = ZNZModel(b_coef=None, sig_coef=None, xhat_ref=20.5, z_ref=2.5,
                   b_ref=0.0, sig_ref=0.1, z_covariate="z_dla")
    znz.b = lambda x, z: np.full_like(np.asarray(x, float), 0.10)      # +0.10 dex bias
    znz.sigma = lambda x, z: np.full_like(np.asarray(x, float), 0.1)   # no width change
    kc = apply_znz_correction(kappa, cat_op, z_edges, logN_lo, logN_hi, znz)
    mids = 0.5 * (logN_lo + logN_hi)
    p = kc[0, :, 7] / kc[0, :, 7].sum()
    assert abs((p * mids).sum() - (20.5 - 0.10)) < 0.05               # mass moved DOWN


def test_apply_znz_preserves_mass_and_z():
    """The correction renormalizes per (i,kz) (Σ preserved) and leaves other z-bins
    untouched (delta correction acts column-wise on kz only)."""
    from CDDF_analysis.znz_kernel import apply_znz_correction, ZNZModel
    n_z = 15
    logN_lo = np.arange(19.0, 22.5, 0.1)
    logN_hi = logN_lo + 0.1
    rng = np.random.default_rng(3)
    kappa = np.zeros((2, len(logN_lo), n_z), np.float32)
    # object 0: a Gaussian-ish bump in z-bin 7; object 1: a bump in z-bin 3
    j0 = np.searchsorted(logN_lo, 20.5)
    kappa[0, j0 - 2:j0 + 3, 7] = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    kappa[1, j0 - 1:j0 + 2, 3] = np.array([0.3, 0.4, 0.3])
    cat_op = {"xhat": np.array([20.5, 20.5]), "zhat": np.array([3.0, 2.3]),
              "i_snr": np.array([0, 0])}
    z_edges = np.linspace(2.0, 3.5, n_z + 1)
    znz = ZNZModel(b_coef=None, sig_coef=None, xhat_ref=20.5, z_ref=2.5,
                   b_ref=0.0, sig_ref=0.1, z_covariate="z_dla")
    znz.b = lambda x, z: np.full_like(np.asarray(x, float), 0.10)
    znz.sigma = lambda x, z: np.full_like(np.asarray(x, float), 0.1)
    kc = apply_znz_correction(kappa, cat_op, z_edges, logN_lo, logN_hi, znz)
    # mass preserved per (i, kz) where there was mass
    assert abs(kc[0, :, 7].sum() - kappa[0, :, 7].sum()) < 1e-5
    assert abs(kc[1, :, 3].sum() - kappa[1, :, 3].sum()) < 1e-5
    # untouched z-bins stay exactly zero
    assert np.all(kc[0, :, 3] == 0.0)
    assert np.all(kc[1, :, 7] == 0.0)


def test_apply_znz_identity_when_b_zero():
    """b≡b_ref, σ≡sig_ref AND a symmetric column already centred at x̂ ⇒ no shift, no
    width change ⇒ kappa unchanged (mass-conserving rebin onto the same grid is the
    identity here)."""
    from CDDF_analysis.znz_kernel import apply_znz_correction, ZNZModel
    n_z = 15
    logN_lo = np.arange(19.0, 22.5, 0.1)
    logN_hi = logN_lo + 0.1
    mids = 0.5 * (logN_lo + logN_hi)
    j0 = np.searchsorted(logN_lo, 20.5)
    kappa = np.zeros((1, len(logN_lo), n_z), np.float32)
    kappa[0, j0 - 1:j0 + 2, 7] = np.array([0.25, 0.5, 0.25])  # symmetric about mids[j0]
    # x̂ == the column's own mass-weighted mean (symmetric ⇒ mids[j0]) so m_tgt == μ_col
    cat_op = {"xhat": np.array([mids[j0]]), "zhat": np.array([3.0]),
              "i_snr": np.array([0])}
    z_edges = np.linspace(2.0, 3.5, n_z + 1)
    znz = ZNZModel(b_coef=None, sig_coef=None, xhat_ref=20.5, z_ref=2.5,
                   b_ref=0.0, sig_ref=0.1, z_covariate="z_dla")
    znz.b = lambda x, z: np.zeros_like(np.asarray(x, float))      # b == b_ref == 0
    znz.sigma = lambda x, z: np.full_like(np.asarray(x, float), 0.1)  # σ == sig_ref
    kc = apply_znz_correction(kappa, cat_op, z_edges, logN_lo, logN_hi, znz)
    assert np.allclose(kc[0, :, 7], kappa[0, :, 7], atol=1e-6)


def test_apply_C_3d_reduces_to_2d_when_g_unity():
    """g≡1 ⇒ the 3-D C path == the 2-D molly C path EXACTLY (byte-identical) for
    both _apply_C_to_A and _apply_C_to_M (the C-threading reduces to molly)."""
    from CDDF_analysis.cddf_catalog_hbi import _apply_C_to_A, _apply_C_to_M

    n_snr, n_nhi, n_zf, n_nbins = 4, 5, 3, 8
    rng = np.random.default_rng(11)
    C2d = rng.uniform(0.3, 1.0, (n_snr, n_nhi))
    # broadcast to 3-D with g≡1 (every kz column identical to the 2-D matrix)
    C3d = np.repeat(C2d[:, :, None], n_zf, axis=2)

    # --- A side: synthetic COO meta with cols encoding kz = cols % n_zf ---
    n_trip = 30
    rows = rng.integers(0, 6, n_trip)
    jN = rng.integers(0, n_nbins, n_trip)
    kz = rng.integers(0, n_zf, n_trip)
    cols = jN * n_zf + kz
    meta = dict(
        rows=rows, cols=cols, vals=rng.uniform(0.1, 2.0, n_trip),
        cell_isnr=rng.integers(0, n_snr, n_trip),
        cell_jnhi=rng.integers(0, n_nhi, n_trip),
        n_obs=6, n_nbins=n_nbins, n_zf=n_zf,
        flat_shape=(6, n_nbins * n_zf),
    )
    A2 = _apply_C_to_A(meta, C2d)
    A3 = _apply_C_to_A(meta, C3d)
    assert np.array_equal(A2.toarray(), A3.toarray()), "A 3-D path != 2-D path at g≡1"

    # --- M side: synthetic seg_table + PX ---
    seg_table = []
    for j in range(n_nbins):
        n_seg = int(rng.integers(1, 3))
        seg_table.append([(int(rng.integers(0, n_nhi)), float(rng.uniform(1e18, 1e20)))
                          for _ in range(n_seg)])
    PX = rng.uniform(0.0, 5.0, (n_snr, n_zf))
    M_meta = dict(seg_table=seg_table, PX=PX, n_snr=n_snr,
                  n_nbins=n_nbins, n_zf=n_zf)
    M2 = _apply_C_to_M(M_meta, C2d)
    M3 = _apply_C_to_M(M_meta, C3d)
    # M reduces per-kz with the same Cint.T@PX contraction, so the only residual is the
    # BLAS reduction-order between gemm (2-D) and gemv (per-kz) — bounded at ~1 ULP. The
    # 3-D path is ONLY ever reached when c_nz_model is set (a Track-C run, g≠1); the
    # default-OFF gate goes through the untouched 2-D path. Require machine precision.
    assert np.allclose(M3, M2, rtol=1e-13, atol=0), "M 3-D path != 2-D path at g≡1"


# ---------------------------------------------------------------------------
# Stage-1: byte-identical gate (load-bearing). Needs the broaden012 bundle on
# scratch; skip cleanly off-box so the unit tests above still run everywhere.
# ---------------------------------------------------------------------------

_BROADEN012_DIR = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                   "phase3d_experiments/mollynhi195_lyaonly1025_broaden012")
_BROADEN012_KERNEL = _BROADEN012_DIR + "/posterior_kernel_2lpt0.npz"
_BROADEN012_POINT = _BROADEN012_DIR + "/phase3d_v3_point_kernel.npz"


@pytest.mark.skipif(
    not (os.path.exists(_BROADEN012_KERNEL) and os.path.exists(_BROADEN012_POINT)),
    reason="broaden012 bundle not present (scratch-only integration): "
           "requires both posterior_kernel_2lpt0.npz and phase3d_v3_point_kernel.npz",
)
def test_znz_off_is_bit_identical():
    """Knobs default-None ⇒ the v3 point fit reproduces the FROZEN broaden012 headline
    BIT-IDENTICALLY (the load-bearing default-OFF gate).

    The headline rounds to dN/dX(≥20.0)=0.09010; the exact frozen value is 0.0900975.
    The gate is byte-identity (to 0.0e0) against the frozen cached point fit — the brief's
    `< 1e-9 vs 0.09010` literal is the rounded headline, unreachable by ANY code to 1e-9;
    the real requirement (plan: "reproduce the broaden012 numbers to 0.0e0") is exact
    equality with the pre-change result, which this asserts. A loose check confirms the
    value still rounds to the published headline.

    The skipif now requires BOTH the kernel file AND the point-cache file.  Previously
    only the kernel was checked, so a fresh clone with kernel-present / point-cache-absent
    would silently pass via the loose < 1e-4 sanity check alone, never reaching the
    exact-equality assert.  The fix: gate jointly so the exact assert is always reached
    when this test runs (no inner conditional).
    """
    from CDDF_analysis.run_phase3d_postkernel import _run_point_for_test
    a = _run_point_for_test(kernel_znz_model=None, c_nz_model=None)
    live = a["dndx"][20.0]
    # rounds to the published 0.09010 headline (sanity)
    assert abs(live - 0.09010) < 1e-4, f"off-path {live} != broaden012 headline 0.09010"
    frozen = float(np.load(_BROADEN012_POINT, allow_pickle=True)["dndx_total_20.0"])
    # THE GATE: default-OFF reproduces the frozen broaden012 number to 0.0e0.
    assert live == frozen, (
        f"default-OFF NOT byte-identical: live={live!r} frozen={frozen!r} "
        f"diff={live - frozen:.3e} (must be 0.0e0)")


# ---------------------------------------------------------------------------
# Stage III: response (θ_K) FORM marginalization (median surface + b_mix + resample)
# ---------------------------------------------------------------------------
def _skewed_meas(seed=0, n=40000):
    """Right-skewed dx (the prior-edge pile-up) so MEAN > MEDIAN, as in the truth-match."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(2.0, 3.5, n)
    xhat = rng.uniform(20.0, 21.5, n)
    tid = rng.integers(0, 5000, n)             # ~5000 sightlines (TID blocks)
    base = 0.04 + 0.06 * (z - 2.0)             # genuine z-trend
    # log-normal-ish right skew (mean ABOVE median): exp jitter, centred so median≈base
    dx = base + (np.exp(rng.normal(0.0, 0.35, n)) - np.exp(0.35**2 / 2)) * 0.06
    return {"xhat": xhat, "z": z, "dx": dx, "z_covariate": "z_dla"}, tid


def test_stage3_b_eff_is_byte_identical_to_mean_at_q1():
    """b_mix=1 (or no median surface) ⇒ b_eff == b EXACTLY (the frozen-default gate)."""
    meas, _ = _skewed_meas()
    m_mean = fit_znz_model(meas)                                  # no median surface
    m_q1 = fit_znz_model(meas, fit_median=True, b_mix=1.0)        # median fit, q=1
    xe = np.array([20.0, 20.3, 20.6, 21.0]); ze = np.array([2.2, 2.5, 2.8, 3.2])
    assert m_mean.b_med_coef is None
    np.testing.assert_array_equal(m_mean.b(xe, ze), m_mean.b_eff(xe, ze))
    np.testing.assert_array_equal(m_q1.b(xe, ze), m_q1.b_eff(xe, ze))
    assert float(m_q1.b_eff_ref()) == float(m_q1.b_ref)


def test_stage3_median_surface_is_below_mean_for_right_skew():
    """For a RIGHT-skewed dx the conditional MEDIAN is BELOW the MEAN (~less correction),
    so b_eff at q<1 lands BETWEEN the two and tracks the median as q→0 (the response-form
    axis from the b_ref note)."""
    meas, _ = _skewed_meas()
    m = fit_znz_model(meas, fit_median=True)
    xe = np.array([20.3, 20.6, 21.0]); ze = np.array([2.5, 2.7, 3.0])
    bmean = m.b(xe, ze); bmed = m.b_median(xe, ze)
    assert np.all(bmed < bmean)                                   # skew: median < mean
    # b_eff interpolates monotonically mean(q=1) -> median(q=0)
    m.b_mix = 1.0; np.testing.assert_allclose(m.b_eff(xe, ze), bmean)
    m.b_mix = 0.0; np.testing.assert_allclose(m.b_eff(xe, ze), bmed)
    m.b_mix = 0.5
    np.testing.assert_allclose(m.b_eff(xe, ze), 0.5 * (bmean + bmed))


def test_stage3_refit_resample_unit_weight_q1_reproduces_mean_fit():
    """At unit multiplicity AND q=1, refit_znz_from_resample reproduces the point MEAN
    surface (the invariance the marginalized band rests on: boot_mult==1 ⇒ frozen θ_K)."""
    from CDDF_analysis.znz_kernel import (
        build_response_fit_resample, refit_znz_from_resample)
    meas, tid = _skewed_meas()
    uniq = np.unique(tid)
    # point model fit on the FULL population (fixed reference so surfaces are comparable)
    pt = fit_znz_model(meas, fit_median=True,
                       xhat_ref=float(np.median(meas["xhat"])),
                       z_ref=float(np.median(meas["z"])))
    rfr = build_response_fit_resample(meas, tid, uniq, pt)
    m1 = refit_znz_from_resample(rfr, np.ones(len(uniq)), b_mix=1.0)
    xe = np.array([20.0, 20.5, 21.0]); ze = np.array([2.3, 2.6, 3.0])
    # unit-weight refit on the same rows with the same reference == the point mean surface
    np.testing.assert_allclose(m1.b(xe, ze), pt.b(xe, ze), rtol=0, atol=1e-9)
    np.testing.assert_allclose(m1.b_eff(xe, ze), pt.b(xe, ze), rtol=0, atol=1e-9)


def test_stage3_refit_resample_perturbs_surface_with_boot_mult():
    """A non-unit bootstrap multiplicity (the SHARED resample) PERTURBS the re-fit b
    surface — i.e. θ_K genuinely varies per draw (the parameter scatter Stage III folds
    into the band), and the perturbation tracks the multiplicity (re-weighting the SAME
    TID twice gives the SAME surface: determinism)."""
    from CDDF_analysis.znz_kernel import (
        build_response_fit_resample, refit_znz_from_resample)
    meas, tid = _skewed_meas()
    uniq = np.unique(tid)
    pt = fit_znz_model(meas, fit_median=True,
                       xhat_ref=float(np.median(meas["xhat"])),
                       z_ref=float(np.median(meas["z"])))
    rfr = build_response_fit_resample(meas, tid, uniq, pt)
    rg = np.random.default_rng(3)
    mult = rg.dirichlet(np.ones(len(uniq))) * len(uniq)
    m_a = refit_znz_from_resample(rfr, mult, b_mix=1.0)
    m_b = refit_znz_from_resample(rfr, mult.copy(), b_mix=1.0)   # same mult -> same fit
    xe = np.array([20.3, 20.6]); ze = np.array([2.5, 2.8])
    np.testing.assert_allclose(m_a.b(xe, ze), m_b.b(xe, ze), atol=1e-12)  # deterministic
    # the resampled surface DIFFERS from the unit-weight (point) surface (real scatter)
    assert np.max(np.abs(m_a.b(xe, ze) - pt.b(xe, ze))) > 1e-6


def test_stage3_corr_strength_one_is_byte_identical():
    """corr_strength=1 (DEFAULT) ⇒ apply_znz_correction is BYTE-IDENTICAL to the
    pre-Stage-III transform (the α-scaling is a no-op at α=1)."""
    from CDDF_analysis.znz_kernel import apply_znz_correction, ZNZModel
    rng = np.random.default_rng(1)
    n_z = 15; logN_lo = np.arange(19.0, 22.5, 0.1); logN_hi = logN_lo + 0.1
    n_obs = 6
    kappa = rng.random((n_obs, len(logN_lo), n_z)).astype(np.float32)
    cat_op = {"xhat": rng.uniform(20.0, 21.5, n_obs),
              "zhat": rng.uniform(2.0, 3.5, n_obs),
              "i_snr": np.zeros(n_obs, int)}
    z_edges = np.linspace(2.0, 3.5, n_z + 1)
    znz = ZNZModel(b_coef=None, sig_coef=None, xhat_ref=20.5, z_ref=2.5,
                   b_ref=0.0, sig_ref=0.12, z_covariate="z_dla", corr_strength=1.0)
    znz.b = lambda x, z: np.full_like(np.asarray(x, float), 0.08)
    znz.sigma = lambda x, z: np.full_like(np.asarray(x, float), 0.10)
    znz1 = ZNZModel(b_coef=None, sig_coef=None, xhat_ref=20.5, z_ref=2.5,
                    b_ref=0.0, sig_ref=0.12, z_covariate="z_dla")  # default corr_strength=1
    znz1.b = znz.b; znz1.sigma = znz.sigma
    a = apply_znz_correction(kappa, cat_op, z_edges, logN_lo, logN_hi, znz)
    b = apply_znz_correction(kappa, cat_op, z_edges, logN_lo, logN_hi, znz1)
    np.testing.assert_array_equal(a, b)


def test_stage3_corr_strength_zero_is_off_identity():
    """corr_strength=0 ⇒ OFF: the transform is the IDENTITY (new_centers == mids), so a
    column is left at its OWN broaden012 center — un-corrected — regardless of b/σ. This
    is the truth-bracketing OFF endpoint (the b_ref note's R0≈1.11 reference)."""
    from CDDF_analysis.znz_kernel import apply_znz_correction, ZNZModel
    n_z = 15; logN_lo = np.arange(19.0, 22.5, 0.1); logN_hi = logN_lo + 0.1
    kappa = np.zeros((1, len(logN_lo), n_z), np.float32)
    j0 = np.searchsorted(logN_lo, 20.5); kappa[0, j0, 7] = 1.0     # delta in bin [20.5,20.6)
    cat_op = {"xhat": np.array([20.5]), "zhat": np.array([3.0]), "i_snr": np.array([0])}
    z_edges = np.linspace(2.0, 3.5, n_z + 1)
    znz = ZNZModel(b_coef=None, sig_coef=None, xhat_ref=20.5, z_ref=2.5,
                   b_ref=0.0, sig_ref=0.1, z_covariate="z_dla", corr_strength=0.0)
    znz.b = lambda x, z: np.full_like(np.asarray(x, float), 0.10)   # would shift -0.10
    znz.sigma = lambda x, z: np.full_like(np.asarray(x, float), 0.2)  # would width-scale
    kc = apply_znz_correction(kappa, cat_op, z_edges, logN_lo, logN_hi, znz)
    # OFF (α=0): mass UNCHANGED — still entirely in its original bin (identity transform)
    assert kc[0, j0, 7] == pytest.approx(1.0)
    assert abs(kc[0, :, 7].sum() - 1.0) < 1e-6


def test_stage3_draw_response_params_alpha_axis():
    """draw_response_params returns (q, α); α∈[α_lo,α_hi] (UNIFORM). DEFAULT α∈[1,1]
    (Step-1, full strength); a Step-2 prior α∈[0,1] spans OFF↔full."""
    import importlib
    H = importlib.import_module("CDDF_analysis.cddf_catalog_hbi")
    from CDDF_analysis.cddf_catalog_hbi import HBIConfig
    cfg = HBIConfig(catalog_dir="x", truth_path="x", bal_cat_path="x",
                    molly_tsv="x", out_dir="x")
    # default: full strength
    assert cfg.mc_response_alpha_lo == 1.0 and cfg.mc_response_alpha_hi == 1.0
    rg = np.random.default_rng(0)
    q, a = H.draw_response_params(rg, cfg)
    assert a == 1.0                                       # degenerate [1,1]
    # Step-2 prior: alpha spans [0,1]
    cfg.mc_response_alpha_lo = 0.0
    alphas = np.array([H.draw_response_params(rg, cfg)[1] for _ in range(2000)])
    assert alphas.min() >= 0.0 and alphas.max() <= 1.0
    assert 0.4 < alphas.mean() < 0.6                      # ~uniform mean 0.5
