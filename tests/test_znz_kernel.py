import numpy as np
from CDDF_analysis.znz_kernel import fit_znz_model


def test_fit_recovers_linear_z_bias():
    rng = np.random.default_rng(0)
    z = rng.uniform(2.0, 3.5, 20000)
    xhat = rng.uniform(20.0, 21.5, 20000)
    true_b = 0.02 + 0.10 * (z - 2.0)              # bias rises with z (the diagnosis)
    resid = true_b + rng.normal(0, 0.05, z.size)  # xhat - xtrue
    meas = {"xhat": xhat, "z": z, "dx": resid, "z_covariate": "z_dla"}
    m = fit_znz_model(meas, deg_z=2, deg_xhat=1)
    assert abs(m.b(np.array([20.5]), np.array([3.25]))[0] - (0.02 + 0.10*1.25)) < 0.01
    assert (m.sigma(np.array([20.5]), np.array([3.25])) > 0).all()


def test_save_load_roundtrip(tmp_path):
    from CDDF_analysis.znz_kernel import save_znz, load_znz, ZNZModel, CNZModel
    import numpy as np
    z = ZNZModel(np.zeros((2,3)), np.zeros((2,3)), 20.5, 2.7, 0.03, 0.12, "z_dla")
    c = CNZModel(np.ones((5,15)), np.linspace(19,23,6), np.linspace(2,3.5,16))
    p = tmp_path/"znz.npz"; save_znz(str(p), z, c); z2, c2 = load_znz(str(p))
    assert z2.z_covariate == "z_dla" and np.allclose(c2.g_grid, 1.0)
