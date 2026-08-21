"""cc_config_ambiguity — the per-bin configuration-ambiguity measurement
(memo A4): a named posterior configuration (one chain of a run) versus the
pooled candidate, per locked Paper-1 bin and all-z. Pure array function
tested on synthetic draws."""
import importlib.util as _ilu
import os

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = _ilu.spec_from_file_location(
    "cc_config_ambiguity_mod",
    os.path.join(_REPO, "CDDF_analysis", "hbi_mcmc", "cc_config_ambiguity.py"))
CA = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(CA)


def _grids():
    ntrue = np.round(np.arange(19.0, 22.3 + 1e-9, 0.2), 3)     # 16 bins
    zf = np.round(np.arange(2.0, 3.5 + 1e-9, 0.1), 3)          # 15 cells
    dX_k = np.linspace(2.0, 1.0, 15)
    return ntrue, zf, dX_k, 19.5


def test_uniform_scaling_of_a_configuration_shows_as_a_flat_percent():
    ntrue, zf, dX_k, floor = _grids()
    base = np.full((50, 16, 15), 1e-22)
    conf = base * 1.10
    r = CA.config_vs_pooled(conf, base, ntrue, zf, dX_k, floor)
    for thr in ("ge20.0", "ge20.3"):
        assert abs(r[thr]["allz_pct"] - 10.0) < 1e-9
        assert all(abs(v - 10.0) < 1e-9 for v in r[thr]["bins_pct"].values())


def test_low_z_only_excess_lands_in_the_low_z_bins():
    ntrue, zf, dX_k, floor = _grids()
    base = np.full((50, 16, 15), 1e-22)
    conf = base.copy(); conf[:, :, :3] *= 1.5          # z < 2.3 only
    r = CA.config_vs_pooled(conf, base, ntrue, zf, dX_k, floor)
    b = r["ge20.3"]["bins_pct"]
    assert b["B1"] > 20 and abs(b["B3"]) < 1e-9 and abs(b["B4"]) < 1e-9
    assert 0 < r["ge20.3"]["allz_pct"] < b["B1"]


def test_split_chains_halves_the_draws_in_order():
    f = np.concatenate([np.zeros((4, 2, 2)), np.ones((4, 2, 2))])
    c0, c1 = CA.split_chains(f, 2)
    assert c0.shape == (4, 2, 2) and np.all(c0 == 0) and np.all(c1 == 1)
