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
