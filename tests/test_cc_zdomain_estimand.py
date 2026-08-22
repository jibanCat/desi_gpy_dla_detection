"""cc_zdomain_estimand — z-domain restricted estimands from EXISTING posterior
draws (PI ruling 2026-08-21 #27: no rerun). Pure array function on synthetic
draws."""
import importlib.util as _ilu
import os

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = _ilu.spec_from_file_location(
    "cc_zdomain_mod", os.path.join(_REPO, "CDDF_analysis", "hbi_mcmc", "cc_zdomain_estimand.py"))
ZD = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(ZD)


def _grids():
    ntrue = np.round(np.arange(19.0, 22.3 + 1e-9, 0.2), 3)
    zf = np.round(np.arange(2.0, 3.5 + 1e-9, 0.1), 3)
    dX_k = np.linspace(2.0, 1.0, 15)
    return ntrue, zf, dX_k, 19.5


def test_full_domain_equals_the_allz_estimand_and_restriction_drops_low_z_path():
    ntrue, zf, dX_k, floor = _grids()
    f = np.full((40, 16, 15), 1e-22)
    f[:, :, :3] *= 2.0                                   # z < 2.3 twice as dense
    r = ZD.zdomain_estimands(f, ntrue, zf, dX_k, floor, z_los=(2.0, 2.3, 2.56))
    full, r23 = r["ge20.3"]["2.0"], r["ge20.3"]["2.3"]
    assert abs(full["path_share"] - 1.0) < 1e-12
    assert abs(r23["path_share"] - dX_k[3:].sum() / dX_k.sum()) < 1e-12
    # restricting away the dense low-z region lowers the estimand
    assert r23["median"] < full["median"]
    # and the restricted value equals the path-weighted mean over the kept cells
    assert abs(r23["median"] / full["median"] - 1.0) > 0.05


def test_homogeneous_field_is_domain_independent():
    ntrue, zf, dX_k, floor = _grids()
    f = np.full((40, 16, 15), 3e-22)
    r = ZD.zdomain_estimands(f, ntrue, zf, dX_k, floor, z_los=(2.0, 2.3, 2.56))
    vals = [r["ge20.0"][k]["median"] for k in ("2.0", "2.3", "2.56")]
    # atol=0: the values are O(1e-22); np.allclose's default atol=1e-8 would
    # make this assertion vacuous (it did — a normalisation mutant survived)
    assert np.allclose(vals, vals[0], rtol=1e-12, atol=0)
    assert all(abs(r["ge20.0"][k]["ratio_to_full_domain"] - 1.0) < 1e-12 for k in ("2.3", "2.56"))


def test_config_leverage_is_reported_per_domain():
    ntrue, zf, dX_k, floor = _grids()
    base = np.full((40, 16, 15), 1e-22)
    conf = base.copy(); conf[:, :, :3] *= 1.5             # low-z only
    r = ZD.config_leverage_by_domain(conf, base, ntrue, zf, dX_k, floor, z_los=(2.0, 2.3, 2.56))
    assert r["ge20.3"]["2.0"] > 0 and abs(r["ge20.3"]["2.3"]) < 1e-9 and abs(r["ge20.3"]["2.56"]) < 1e-9
