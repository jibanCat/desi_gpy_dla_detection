import os

import numpy as np
import pytest
from CDDF_analysis.hbi.znz_kernel import fit_znz_model, save_znz, load_znz, ZNZModel, CNZModel


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
    from CDDF_analysis.hbi.znz_kernel import apply_znz_correction, ZNZModel
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
    from CDDF_analysis.hbi.znz_kernel import apply_znz_correction, ZNZModel
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
    from CDDF_analysis.hbi.znz_kernel import apply_znz_correction, ZNZModel
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
    from CDDF_analysis.hbi.cddf_catalog_hbi import _apply_C_to_A, _apply_C_to_M

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
    from CDDF_analysis.hbi.run_phase3d_postkernel import _run_point_for_test
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
    from CDDF_analysis.hbi.znz_kernel import (
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
    from CDDF_analysis.hbi.znz_kernel import (
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
    from CDDF_analysis.hbi.znz_kernel import apply_znz_correction, ZNZModel
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
    from CDDF_analysis.hbi.znz_kernel import apply_znz_correction, ZNZModel
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
    from CDDF_analysis.hbi.cddf_catalog_hbi import HBIConfig
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


# ---------------------------------------------------------------------------
# Track-C T1: _skew_warp unit tests + apply_znz_correction gate
# ---------------------------------------------------------------------------

# Conditional response width ω (the per-object σ surface ~0.19–0.21 dex) on which γ
# acts — NOT the carrier-grid std.  All T1 warp tests use a narrow off-center column.
_OMEGA = 0.19


def _narrow_col(mids, mu, sigma=0.19):
    """A normalized Gaussian-like mass column of width ``sigma`` centred at ``mu``.

    This is the FAITHFUL test substrate: the affine relocate in apply_znz_correction
    produces a column centred at m_tgt with the conditional width σ(x̂,z)~0.19–0.21 dex,
    NOT a wide column spanning the whole grid.  Testing on a narrow off-center column is
    what exposes the C1 pivot bug (a wide grid-center column hides it).
    """
    col = np.exp(-0.5 * ((mids - mu) / sigma) ** 2)
    return col / col.sum()


def _wmedian(col, mids):
    """Weighted median of a (mass, carrier) column — lands on the bin straddling 0.5."""
    o = np.argsort(mids)
    cs = np.cumsum(col[o])
    return float(mids[o][np.searchsorted(cs, 0.5 * cs[-1])])


def _skewness(col, mids):
    """Standardized third central moment of a mass column."""
    m1 = (col * mids).sum()
    m2 = (col * (mids - m1) ** 2).sum()
    if m2 <= 0:
        return 0.0
    m3 = (col * (mids - m1) ** 3).sum()
    return m3 / m2 ** 1.5


def test_skew_warp_gamma_zero_is_identity():
    """_skew_warp(gamma=0) must return centers UNCHANGED, bit-for-bit (assert_array_equal)."""
    from CDDF_analysis.hbi.znz_kernel import _skew_warp
    rng = np.random.default_rng(42)
    centers = rng.uniform(19.0, 22.5, 35)
    result = _skew_warp(centers, mu=float(np.mean(centers)), gamma=0.0, omega=_OMEGA)
    np.testing.assert_array_equal(result, centers,
        err_msg="_skew_warp(gamma=0) must be bit-for-bit identical to centers")


def test_skew_warp_pivot_is_exact_continuous():
    """The carrier AT the pivot mu must map EXACTLY back to mu (f(0)=0) for any γ, ω, μ.

    This is the continuous C1 fix: the pre-skew column is symmetric about mu so its
    median sits at u=0; pivot-correction forces f(0)=0, so the median is invariant.
    (The broken pre-fix warp translated this point by ≈−sinh(γ)·s — up to −1.18 dex.)
    """
    from CDDF_analysis.hbi.znz_kernel import _skew_warp
    for omega in (0.19, 0.21, 0.05):
        for mu in (19.6, 20.3, 21.0):
            for gamma in (+0.5, +1.0, +2.0, -0.5, -1.0, -2.0):
                centers = np.array([mu - 0.3, mu, mu + 0.3])
                w = _skew_warp(centers, mu=mu, gamma=gamma, omega=omega)
                assert abs(w[1] - mu) < 1e-12, (
                    f"pivot not preserved: omega={omega} mu={mu} gamma={gamma} "
                    f"f(mu)-mu={w[1]-mu:.3e} (must be 0 — median would move otherwise)")


def test_skew_warp_preserves_median_narrow_offcenter():
    """C1 FIX (the test the original suite lacked): warp a NARROW (σ≈0.19) column centred
    OFF the grid-center at μ∈{19.6,20.3,21.0} and assert the warped column's MEDIAN stays
    at μ to within one bin (the discretization granularity of a weighted median).

    The MEDIAN fixes the count / dN/dX — it MUST be invariant.  The broken warp moved it
    by ≈−1.1 dex (s=carrier-grid std + no pivot correction).  The residual here is pure
    histogram-median quantization (≤ a couple of bins, vanishing as the grid refines —
    test_skew_warp_pivot_is_exact_continuous proves the continuous f(0)=0 exactly), NOT a
    bulk translation."""
    from CDDF_analysis.hbi.znz_kernel import _skew_warp, _mass_conserving_rebin
    dN = 0.02
    logN_lo = np.arange(19.0, 22.5, dN); logN_hi = logN_lo + dN
    mids = 0.5 * (logN_lo + logN_hi)
    edges = np.concatenate([logN_lo, [float(logN_hi[-1])]])
    tol = 2.0 * dN  # histogram-median quantization bound (continuous pivot is exact)
    for mu in (19.6, 20.3, 21.0):
        col = _narrow_col(mids, mu)
        for gamma in (+1.0, +2.0, -1.0, -2.0):
            warped = _skew_warp(mids, mu=mu, gamma=gamma, omega=_OMEGA)
            rebinned = _mass_conserving_rebin(col, warped, edges)
            rebinned /= rebinned.sum()
            med = _wmedian(rebinned, mids)
            assert abs(med - mu) <= tol, (
                f"median moved off mu: mu={mu} gamma={gamma} warped_median={med:.4f} "
                f"shift={med - mu:+.4f} (must stay within {tol} = histogram quantization; "
                f"C1 would give ~-1.1 dex)")
    # Convergence: refining the grid shrinks the discretized median shift toward 0 (proving
    # it's quantization, not a bulk move).  At dN=0.005 the worst-case shift must be < dN at
    # dN=0.02 — i.e. it strictly improves with resolution.
    mu = 20.3; gamma = 2.0
    for dN_fine in (0.01, 0.005):
        lo = np.arange(19.0, 22.5, dN_fine); hi = lo + dN_fine
        m = 0.5 * (lo + hi); e = np.concatenate([lo, [float(hi[-1])]])
        col = _narrow_col(m, mu)
        w = _skew_warp(m, mu=mu, gamma=gamma, omega=_OMEGA)
        rb = _mass_conserving_rebin(col, w, e); rb /= rb.sum()
        assert abs(_wmedian(rb, m) - mu) <= 2.0 * dN_fine, (
            f"median shift at dN={dN_fine} exceeds 2*dN — not pure quantization")


def test_skew_warp_mean_drifts_up_for_positive_gamma():
    """The Ω-restoring direction: for γ>0 the mass-weighted MEAN of the warped NARROW
    column INCREASES vs the pre-warp column (the right tail the skew restores); for γ<0
    it DECREASES.  (Median stays put — only the mean drifts; this is the whole point.)"""
    from CDDF_analysis.hbi.znz_kernel import _skew_warp, _mass_conserving_rebin
    dN = 0.02
    logN_lo = np.arange(19.0, 22.5, dN); logN_hi = logN_lo + dN
    mids = 0.5 * (logN_lo + logN_hi)
    edges = np.concatenate([logN_lo, [float(logN_hi[-1])]])
    for mu in (19.6, 20.3, 21.0):
        col = _narrow_col(mids, mu)
        mean_pre = float((col * mids).sum())
        for gamma in (+1.0, +2.0):
            warped = _skew_warp(mids, mu=mu, gamma=gamma, omega=_OMEGA)
            rb = _mass_conserving_rebin(col, warped, edges); rb /= rb.sum()
            mean_post = float((rb * mids).sum())
            assert mean_post > mean_pre + 1e-4, (
                f"mean did NOT drift up for gamma={gamma} at mu={mu}: "
                f"pre={mean_pre:.4f} post={mean_post:.4f} (Ω restoration broken)")
        for gamma in (-1.0, -2.0):
            warped = _skew_warp(mids, mu=mu, gamma=gamma, omega=_OMEGA)
            rb = _mass_conserving_rebin(col, warped, edges); rb /= rb.sum()
            mean_post = float((rb * mids).sum())
            assert mean_post < mean_pre - 1e-4, (
                f"mean did NOT drift down for gamma={gamma} at mu={mu}: "
                f"pre={mean_pre:.4f} post={mean_post:.4f}")


def test_skew_warp_is_monotone():
    """For gamma != 0, _skew_warp must preserve strict ordering (no bin crossing)."""
    from CDDF_analysis.hbi.znz_kernel import _skew_warp
    # sorted centers (as produced by the affine relocate in apply_znz_correction)
    logN_lo = np.arange(19.0, 22.5, 0.1)
    logN_hi = logN_lo + 0.1
    centers = 0.5 * (logN_lo + logN_hi)                        # fine logN mids
    for gamma in [+0.5, +1.0, +2.0, -0.5, -1.0, -2.0]:
        warped = _skew_warp(centers, mu=float(centers.mean()), gamma=gamma, omega=_OMEGA)
        diffs = np.diff(warped)
        assert np.all(diffs > 0), (
            f"_skew_warp(gamma={gamma}) broke monotonicity: min(diff)={diffs.min():.3e}")


def test_skew_warp_mass_conserving():
    """_skew_warp + _mass_conserving_rebin preserves total column mass to 1e-12.

    The warp remaps carrier positions; the rebin deposits mass at the new carriers
    — together they must conserve Σ(mass) exactly (up to floating-point rounding).
    """
    from CDDF_analysis.hbi.znz_kernel import _skew_warp, _mass_conserving_rebin
    rng = np.random.default_rng(7)
    logN_lo = np.arange(19.0, 22.5, 0.1)
    logN_hi = logN_lo + 0.1
    mids = 0.5 * (logN_lo + logN_hi)
    edges = np.concatenate([logN_lo, [float(logN_hi[-1])]])
    col = rng.exponential(1.0, len(mids))                        # non-negative mass column
    col /= col.sum()                                             # normalize to Σ=1
    for gamma in [+0.3, +1.0, +2.0, -0.3, -1.0]:
        warped = _skew_warp(mids, mu=float(np.mean(mids)), gamma=gamma, omega=_OMEGA)
        rebinned = _mass_conserving_rebin(col, warped, edges)
        tot_before = col.sum()
        tot_after = rebinned.sum()
        assert abs(tot_after - tot_before) < 1e-12, (
            f"Mass not conserved at gamma={gamma}: before={tot_before:.15g} "
            f"after={tot_after:.15g} diff={tot_after - tot_before:.3e}")


def test_skew_warp_positive_gamma_gives_positive_skewness():
    """C2 (FAITHFUL): γ>0 must give POSITIVE skewness of a NARROW OFF-CENTER column at
    μ∈{19.6,20.3,21.0} (not just a wide grid-center column); γ<0 → negative skewness.

    The narrow off-center column is the real substrate (conditional width ω after the
    affine relocate).  The pivot-corrected warp keeps the sign robust at every μ."""
    from CDDF_analysis.hbi.znz_kernel import _skew_warp, _mass_conserving_rebin
    dN = 0.02
    logN_lo = np.arange(19.0, 22.5, dN); logN_hi = logN_lo + dN
    mids = 0.5 * (logN_lo + logN_hi)
    edges = np.concatenate([logN_lo, [float(logN_hi[-1])]])
    for mu in (19.6, 20.3, 21.0):
        col = _narrow_col(mids, mu)
        for gamma in (+1.0, +2.0):
            warped = _skew_warp(mids, mu=mu, gamma=gamma, omega=_OMEGA)
            rb = _mass_conserving_rebin(col, warped, edges); rb /= rb.sum()
            sk = _skewness(rb, mids)
            assert sk > 0.0, (
                f"gamma={gamma} at mu={mu}: expected positive skewness, got {sk:.4f}")
        for gamma in (-1.0, -2.0):
            warped = _skew_warp(mids, mu=mu, gamma=gamma, omega=_OMEGA)
            rb = _mass_conserving_rebin(col, warped, edges); rb /= rb.sum()
            sk = _skewness(rb, mids)
            assert sk < 0.0, (
                f"gamma={gamma} at mu={mu}: expected negative skewness, got {sk:.4f}")


def test_apply_znz_skew_off_is_byte_identical():
    """apply_znz_correction with skew_coef=None and skew_strength=0.0 (the T1 defaults)
    must be bit-for-bit identical to the result WITHOUT the skew fields at all.

    This is the LOAD-BEARING T1 gate: the skew warp path must be an exact no-op when
    inactive so the existing broaden012 headline is reproduced to 0.0e0.
    """
    from CDDF_analysis.hbi.znz_kernel import apply_znz_correction, ZNZModel
    rng = np.random.default_rng(99)
    n_z = 15; logN_lo = np.arange(19.0, 22.5, 0.1); logN_hi = logN_lo + 0.1
    n_obs = 8
    kappa = rng.random((n_obs, len(logN_lo), n_z)).astype(np.float32)
    cat_op = {"xhat": rng.uniform(20.0, 21.5, n_obs),
              "zhat": rng.uniform(2.0, 3.5, n_obs),
              "i_snr": np.zeros(n_obs, int)}
    z_edges = np.linspace(2.0, 3.5, n_z + 1)

    # --- model WITHOUT any skew fields (pre-T1 baseline) ---
    znz_baseline = ZNZModel(b_coef=None, sig_coef=None, xhat_ref=20.5, z_ref=2.5,
                            b_ref=0.0, sig_ref=0.12, z_covariate="z_dla")
    znz_baseline.b = lambda x, z: np.full_like(np.asarray(x, float), 0.08)
    znz_baseline.sigma = lambda x, z: np.full_like(np.asarray(x, float), 0.10)

    # --- model WITH skew fields at their gate-off defaults ---
    znz_t1 = ZNZModel(b_coef=None, sig_coef=None, xhat_ref=20.5, z_ref=2.5,
                      b_ref=0.0, sig_ref=0.12, z_covariate="z_dla",
                      skew_coef=None, skew_strength=0.0)
    znz_t1.b = znz_baseline.b
    znz_t1.sigma = znz_baseline.sigma

    out_baseline = apply_znz_correction(kappa, cat_op, z_edges, logN_lo, logN_hi,
                                        znz_baseline)
    out_t1 = apply_znz_correction(kappa, cat_op, z_edges, logN_lo, logN_hi, znz_t1)
    np.testing.assert_array_equal(out_baseline, out_t1,
        err_msg="apply_znz_correction with skew_coef=None/skew_strength=0 is NOT "
                "byte-identical to the pre-T1 baseline (gate broken)")


def test_apply_znz_nonzero_skew_differs_from_off():
    """When skew_coef is set and skew_strength != 0, apply_znz_correction must produce
    a DIFFERENT result from the no-skew baseline — proving the warp code path is reached."""
    from CDDF_analysis.hbi.znz_kernel import apply_znz_correction, ZNZModel
    from numpy.polynomial.polynomial import polyvander2d as _pv2d
    rng = np.random.default_rng(55)
    n_z = 10; logN_lo = np.arange(19.0, 22.5, 0.1); logN_hi = logN_lo + 0.1
    n_obs = 4
    # Use a smooth non-trivial kappa (not all-zeros) so the warp has mass to move.
    kappa = rng.random((n_obs, len(logN_lo), n_z)).astype(np.float32)
    cat_op = {"xhat": np.array([20.5, 20.8, 21.0, 21.2]),
              "zhat": np.array([2.5, 2.7, 3.0, 3.2]),
              "i_snr": np.zeros(n_obs, int)}
    z_edges = np.linspace(2.0, 3.5, n_z + 1)

    # A skew surface: constant γ=1.0 everywhere (simple scalar coef, deg=(1,2)).
    # polyvander2d with deg=[1,2] → 6 basis terms; intercept is coef[0].
    deg_xhat, deg_z = 1, 2
    n_coef = (deg_xhat + 1) * (deg_z + 1)
    skew_coef = np.zeros(n_coef)
    skew_coef[0] = 1.0                                    # constant surface γ≡1.0

    znz_off = ZNZModel(b_coef=None, sig_coef=None, xhat_ref=20.5, z_ref=2.5,
                       b_ref=0.0, sig_ref=0.12, z_covariate="z_dla",
                       deg_xhat=deg_xhat, deg_z=deg_z,
                       skew_coef=None, skew_strength=0.0)
    znz_off.b = lambda x, z: np.full_like(np.asarray(x, float), 0.08)
    znz_off.sigma = lambda x, z: np.full_like(np.asarray(x, float), 0.10)

    znz_on = ZNZModel(b_coef=None, sig_coef=None, xhat_ref=20.5, z_ref=2.5,
                      b_ref=0.0, sig_ref=0.12, z_covariate="z_dla",
                      deg_xhat=deg_xhat, deg_z=deg_z,
                      skew_coef=skew_coef, skew_strength=1.0)
    znz_on.b = znz_off.b
    znz_on.sigma = znz_off.sigma

    out_off = apply_znz_correction(kappa, cat_op, z_edges, logN_lo, logN_hi, znz_off)
    out_on = apply_znz_correction(kappa, cat_op, z_edges, logN_lo, logN_hi, znz_on)
    assert not np.array_equal(out_off, out_on), (
        "Skew-ON and skew-OFF gave identical results — skew code path not reached")


# ---------------------------------------------------------------------------
# Track-C T2: _skew_fit_2d skew-surface fit + save/load skew_coef
# ---------------------------------------------------------------------------
import inspect


def _synth_skewed_dx(gamma_fn, omega=0.20, loc=0.06, seed=0, n=300000,
                     xlo=19.5, xhi=21.5, zlo=2.0, zhi=3.5):
    """Synthesize a truth-match-shaped meas dict whose conditional dx skewness follows a
    KNOWN gamma(x̂,z) ramp via the SAS warp ``sinh(arcsinh Z + γ) − sinh γ`` (Z~N(0,1)).

    By construction the conditional skewness of ``dx`` at (x̂,z) equals
    ``_sas_skewness_of_gamma(gamma_fn(x̂,z))`` (location/scale-invariant), so a faithful
    ``_skew_fit_2d`` must recover a surface whose induced skewness matches that target.
    """
    from CDDF_analysis.hbi.znz_kernel import _gamma_from_skewness  # noqa: F401 (import guard)
    rng = np.random.default_rng(seed)
    z = rng.uniform(zlo, zhi, n)
    xhat = rng.uniform(xlo, xhi, n)
    gt = np.asarray(gamma_fn(xhat, z), float)
    Zr = rng.standard_normal(n)
    dx = loc + (np.sinh(np.arcsinh(Zr) + gt) - np.sinh(gt)) * omega
    return {"xhat": xhat, "z": z, "dx": dx, "z_covariate": "z_dla"}, gt


def test_skew_fit_recovers_known_gamma_surface():
    """Moment-match closure: synthesize dx with a KNOWN (within-ceiling) γ(x̂,z) ramp via
    _skew_warp, fit with _skew_fit_2d, and assert the recovered skew_coef reproduces the
    INPUT per-cell skewness to tolerance (the primary T2 correctness gate)."""
    from CDDF_analysis.hbi.znz_kernel import (
        _skew_fit_2d, _sas_skewness_of_gamma)
    from numpy.polynomial.polynomial import polyvander2d
    # gamma ramp kept comfortably within the SAS ceiling (|γ|≲1) so the inversion is
    # well-conditioned and the recovery is exact-up-to-fit-noise.
    gamma_fn = lambda x, zz: 0.2 + 0.25 * (x - 19.5) + 0.15 * (zz - 2.0)
    meas, _ = _synth_skewed_dx(gamma_fn, seed=1)
    xref, zref = 20.5, 2.75
    coef = _skew_fit_2d(meas["xhat"], meas["dx"], meas["z"], xref, zref, 1, 2)
    worst = 0.0
    for x in (19.7, 20.3, 21.0):
        for zz in (2.25, 2.75, 3.25):
            g_rec = float((polyvander2d(np.array([x]) - xref, np.array([zz]) - zref,
                                        [1, 2]) @ coef)[0])
            sk_rec = float(_sas_skewness_of_gamma(np.array([g_rec]))[0])
            sk_in = float(_sas_skewness_of_gamma(np.array([gamma_fn(x, zz)]))[0])
            worst = max(worst, abs(sk_rec - sk_in))
    assert worst < 0.08, (
        f"recovered skewness deviates from the input ramp by {worst:.3f} (>0.08): "
        "the moment-match closure failed")


def test_skew_fit_right_skew_gives_positive_gamma():
    """SIGN gate: a uniformly RIGHT-skewed dx input must yield γ>0 everywhere in the
    science range (the Ω-restoring direction of _skew_warp) — never a sign flip."""
    from CDDF_analysis.hbi.znz_kernel import _skew_fit_2d, _sas_skewness_of_gamma
    from numpy.polynomial.polynomial import polyvander2d
    # constant positive skew target ≈ +0.75 (γ≈+0.5) across the grid
    gamma_fn = lambda x, zz: np.full_like(np.asarray(x, float), 0.5)
    meas, _ = _synth_skewed_dx(gamma_fn, seed=2)
    xref, zref = 20.5, 2.75
    coef = _skew_fit_2d(meas["xhat"], meas["dx"], meas["z"], xref, zref, 1, 2)
    for x in (19.7, 20.3, 21.0):
        for zz in (2.2, 2.75, 3.3):
            g = float((polyvander2d(np.array([x]) - xref, np.array([zz]) - zref,
                                    [1, 2]) @ coef)[0])
            assert g > 0.0, (
                f"right-skewed input gave NON-positive γ={g:+.3f} at (x={x},z={zz}) — "
                "sign convention broken (must be the Ω-restoring +γ direction)")


def test_skew_fit_left_skew_gives_negative_gamma():
    """Mirror sign gate: a LEFT-skewed dx input must yield γ<0 (left-tail direction)."""
    from CDDF_analysis.hbi.znz_kernel import _skew_fit_2d
    from numpy.polynomial.polynomial import polyvander2d
    gamma_fn = lambda x, zz: np.full_like(np.asarray(x, float), -0.5)
    meas, _ = _synth_skewed_dx(gamma_fn, seed=3)
    xref, zref = 20.5, 2.75
    coef = _skew_fit_2d(meas["xhat"], meas["dx"], meas["z"], xref, zref, 1, 2)
    g = float((polyvander2d(np.array([20.3]) - xref, np.array([2.75]) - zref,
                            [1, 2]) @ coef)[0])
    assert g < 0.0, f"left-skewed input gave γ={g:+.3f} ≥ 0 (sign convention broken)"


def test_sas_skewness_map_monotone_and_zero_at_zero():
    """The γ→skewness inversion map must be monotone increasing with γ=0→skew=0 (so the
    inverse is single-valued and sign-preserving)."""
    from CDDF_analysis.hbi.znz_kernel import _sas_skewness_of_gamma, _gamma_from_skewness
    gg = np.linspace(-3.5, 3.5, 71)
    sk = _sas_skewness_of_gamma(gg)
    assert abs(_sas_skewness_of_gamma(np.array([0.0]))[0]) < 1e-9
    assert np.all(np.diff(sk) > 0), "SAS skewness map not strictly monotone in γ"
    # inversion round-trips within the achievable (sub-ceiling) range
    for s in (0.1, 0.5, 1.0, 1.3):
        g = float(_gamma_from_skewness(np.array([s]))[0])
        s_back = float(_sas_skewness_of_gamma(np.array([g]))[0])
        assert abs(s_back - s) < 0.02, f"inversion round-trip {s}->{g}->{s_back} off"
    # above-ceiling target clamps to +clamp (no overflow, sign kept)
    g_hi = float(_gamma_from_skewness(np.array([2.10]))[0])
    assert g_hi > 0 and abs(g_hi - 4.0) < 1e-6, f"above-ceiling target did not clamp: {g_hi}"


def test_skew_fit_is_noncircular_signature():
    """NON-CIRCULAR gate: _skew_fit_2d and fit_znz_model(fit_skew=...) take NO dN/dX/Ω
    input — the fit can only read the truth-match (x̂, z, dx) conditional, never a reduced
    statistic. Enforced structurally via the signatures."""
    from CDDF_analysis.hbi.znz_kernel import _skew_fit_2d, fit_znz_model
    # reduced-statistic name fragments (the forbidden dN/dX / Ω / f(N,z) / R0 inputs).
    # NOTE: ``dx`` (the truth-match residual x̂−x_true) is the LEGIT conditional input and
    # is intentionally NOT forbidden — it is exactly what the moment fit reads. We forbid
    # only DOWNSTREAM reductions. Match on '_'-delimited tokens so generic substrings
    # ('ell' in 'cells') don't trip.
    forbidden = {"dndx", "dndz", "dxdn", "omega", "ellz", "fnz", "cddf", "r0", "reduce"}
    import re
    for fn in (_skew_fit_2d, fit_znz_model):
        params = set(inspect.signature(fn).parameters)
        bad = {p for p in params
               if set(re.split(r"[_]", p.lower())) & forbidden}
        assert not bad, (
            f"{fn.__name__} exposes a reduced-statistic argument {bad} — would open the "
            "α=1/R0 circular edge; the skew fit must read ONLY the truth-match dx moment")
    # _skew_fit_2d's positional inputs are exactly the conditional arrays + poly refs
    sig = list(inspect.signature(_skew_fit_2d).parameters)[:3]
    assert sig == ["xhat", "dx", "z"], (
        f"_skew_fit_2d should take (xhat, dx, z, ...) only; got {sig}")


def test_save_load_roundtrip_skew_coef(tmp_path):
    """save/load round-trips skew_coef + skew_strength AND all existing fields exactly."""
    from CDDF_analysis.hbi.znz_kernel import _sas_skewness_of_gamma
    from numpy.polynomial.polynomial import polyvander2d
    gamma_fn = lambda x, zz: 0.2 + 0.25 * (x - 19.5) + 0.10 * (zz - 2.0)
    meas, _ = _synth_skewed_dx(gamma_fn, seed=5)
    znz = fit_znz_model(meas, deg_z=2, deg_xhat=1, fit_median=True,
                        fit_skew=True, skew_strength=1.0)
    assert znz.skew_coef is not None and znz.skew_strength == 1.0
    cnz = CNZModel(g_grid=np.ones((5, 15)), nhi_edges=np.linspace(19, 23, 6),
                   z_edges_fine=np.linspace(2, 3.5, 16))
    path = str(tmp_path / "znz_skew.npz")
    save_znz(path, znz, cnz)
    znz2, _ = load_znz(path)
    # skew block preserved exactly
    assert znz2.skew_coef is not None
    np.testing.assert_array_equal(znz2.skew_coef, znz.skew_coef)
    assert znz2.skew_strength == znz.skew_strength
    # existing blocks preserved (median + mean surfaces + degrees)
    np.testing.assert_array_equal(znz2.b_coef, znz.b_coef)
    np.testing.assert_array_equal(znz2.b_med_coef, znz.b_med_coef)
    assert znz2.b_mix == znz.b_mix
    assert znz2.deg_xhat == znz.deg_xhat and znz2.deg_z == znz.deg_z
    # the surface evaluates identically after reload
    xe = np.array([20.0, 20.5, 21.0]); ze = np.array([2.3, 2.6, 3.0])
    V = polyvander2d(xe - znz.xhat_ref, ze - znz.z_ref, [znz.deg_xhat, znz.deg_z])
    np.testing.assert_array_equal(V @ znz2.skew_coef, V @ znz.skew_coef)


def test_load_backward_compat_no_skew_coef_is_byte_identical(tmp_path):
    """BACKWARD-COMPAT gate: an npz WITHOUT skew_coef (a pre-T2 cache) loads to
    skew_coef=None / skew_strength=0.0 ⇒ apply_znz_correction is BYTE-IDENTICAL to the
    pre-skew transform (the load-bearing default-OFF guarantee)."""
    from CDDF_analysis.hbi.znz_kernel import apply_znz_correction
    # build a MEAN-only model (no skew, no median) and save it the OLD way (no skew keys)
    meas = _make_synthetic_meas(seed=11)
    znz = fit_znz_model(meas, deg_z=2, deg_xhat=1)   # skew_coef is None by default
    assert znz.skew_coef is None
    cnz = CNZModel(g_grid=np.ones((4, 10)), nhi_edges=np.linspace(19, 23, 5),
                   z_edges_fine=np.linspace(2, 3.5, 11))
    # write a legacy-shaped npz with NO skew_coef/skew_strength keys at all
    legacy = str(tmp_path / "legacy_noskew.npz")
    np.savez(
        legacy,
        b_coef=znz.b_coef, sig_coef=znz.sig_coef,
        xhat_ref=np.array(znz.xhat_ref), z_ref=np.array(znz.z_ref),
        b_ref=np.array(znz.b_ref), sig_ref=np.array(znz.sig_ref),
        z_covariate=np.array(znz.z_covariate),
        deg_xhat=np.array(znz.deg_xhat), deg_z=np.array(znz.deg_z),
        g_grid=cnz.g_grid, nhi_edges=cnz.nhi_edges, z_edges_fine=cnz.z_edges_fine,
    )
    znz_legacy, _ = load_znz(legacy)
    assert znz_legacy.skew_coef is None, "missing skew_coef must load as None"
    assert znz_legacy.skew_strength == 0.0, "missing skew_strength must load as 0.0"

    # byte-identical apply: legacy-loaded vs a model with the skew fields absent entirely
    n_z = 15; logN_lo = np.arange(19.0, 22.5, 0.1); logN_hi = logN_lo + 0.1
    rng = np.random.default_rng(7)
    n_obs = 6
    kappa = rng.random((n_obs, len(logN_lo), n_z)).astype(np.float32)
    cat_op = {"xhat": rng.uniform(20.0, 21.5, n_obs),
              "zhat": rng.uniform(2.0, 3.5, n_obs), "i_snr": np.zeros(n_obs, int)}
    z_edges = np.linspace(2.0, 3.5, n_z + 1)
    out_legacy = apply_znz_correction(kappa, cat_op, z_edges, logN_lo, logN_hi, znz_legacy)
    out_orig = apply_znz_correction(kappa, cat_op, z_edges, logN_lo, logN_hi, znz)
    np.testing.assert_array_equal(out_legacy, out_orig,
        err_msg="legacy npz (no skew_coef) NOT byte-identical to the pre-skew transform")
