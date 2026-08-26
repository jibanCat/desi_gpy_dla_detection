"""cddf_recovery_audit — per-0.2-dex-bin posterior recovery vs truth, all-z and per
z slice (overlap-weighted), from saved validation draws. Pure array functions."""
import importlib.util as _ilu, os
import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = _ilu.spec_from_file_location(
    "cddf_audit_mod", os.path.join(_REPO, "CDDF_analysis", "hbi_mcmc", "cddf_recovery_audit.py"))
A = _ilu.module_from_spec(_spec); _spec.loader.exec_module(A)


def _grid():
    ntrue = np.round(np.arange(19.0, 22.3 + 1e-9, 0.2), 3)   # 17 edges -> 16 bins (like the packs, modulo the top bin)
    zf = np.round(np.arange(2.0, 3.5 + 1e-9, 0.1), 3)
    dX = np.linspace(3.0, 1.0, 15)
    return ntrue, zf, dX


def test_bin_recovery_reproduces_an_injected_bias_and_weights_by_path():
    ntrue, zf, dX = _grid()
    rng = np.random.default_rng(0)
    truth = np.exp(-(ntrue[:-1] - 19.0))[:, None] * (1.0 + 0.3 * (zf[:-1] - 2.0))[None, :] * 1e-22
    bias = np.full(truth.shape, 1.0); bias[3:5] = 0.85                       # bins [19.6,20.0) low by 15 %
    f = truth[None] * bias[None] * (1.0 + rng.normal(0, 0.01, (200,) + truth.shape))
    rows = A.bin_recovery(f, truth, ntrue, zf, dX, redges=np.round(np.arange(19.6, 21.6 + 1e-9, 0.4), 3))
    assert [r["bin"] for r in rows][0] == [19.6, 20.0]
    assert abs(rows[0]["median_bias_pct"] - (-15.0)) < 1.0
    assert abs(rows[1]["median_bias_pct"]) < 1.0
    assert rows[0]["stat_halfwidth_pct_lo"] > 0 and rows[0]["truth"] > 0


def test_slice_overlap_weighting_is_exact_on_a_cell_boundary_and_partial_inside():
    ntrue, zf, dX = _grid()
    truth = np.ones((16, 15)) * 1e-22
    truth[:, zf[:-1] >= 2.5 - 1e-9] *= 2.0                                  # z >= 2.5 twice as dense
    f = np.repeat(truth[None], 50, 0)
    re_ = np.round(np.arange(19.6, 21.6 + 1e-9, 0.4), 3)     # aligned to this test grid (19.7 is NOT an edge here)
    full = A.bin_recovery(f, truth, ntrue, zf, dX, redges=re_, z_lo=2.0, z_hi=2.5)[0]["truth"]
    upper = A.bin_recovery(f, truth, ntrue, zf, dX, redges=re_, z_lo=2.5, z_hi=3.5)[0]["truth"]
    assert abs(upper / full - 2.0) < 1e-12
    # a slice cutting a cell in half: overlap weight = half of that cell's path
    half = A.bin_recovery(f, truth, ntrue, zf, dX, redges=re_, z_lo=2.45, z_hi=2.5)[0]
    assert abs(half["dX"] - 0.5 * dX[4]) < 1e-12
    # zero bias when draws == truth, and 68 % half-widths are zero
    assert abs(half["median_bias_pct"]) < 1e-9 and half["stat_halfwidth_pct_hi"] == 0.0


def test_family_summary_reports_min_max_and_single_run_extremes():
    rows_by_run = {"famA_s1": [{"bin": [19.7, 19.9], "median_bias_pct": -16.0}],
                   "famA_s2": [{"bin": [19.7, 19.9], "median_bias_pct": -15.0}],
                   "famB_s1": [{"bin": [19.7, 19.9], "median_bias_pct": -22.0}]}
    fam = {"famA_s1": "famA", "famA_s2": "famA", "famB_s1": "famB"}
    s = A.family_summary(rows_by_run, fam)
    assert s[(19.7, 19.9)]["famA"]["min"] == -16.0 and s[(19.7, 19.9)]["famA"]["max"] == -15.0
    assert s[(19.7, 19.9)]["famA"]["n_runs"] == 2 and s[(19.7, 19.9)]["famB"]["n_runs"] == 1
    assert s[(19.7, 19.9)]["all"]["min"] == -22.0 and s[(19.7, 19.9)]["all"]["extreme_run"] == "famB_s1"


def test_bias_uses_the_posterior_median_not_the_mean():
    ntrue, zf, dX = _grid()
    truth = np.ones((16, 15)) * 1e-22
    rng = np.random.default_rng(3)
    # heavily skewed draws: median at truth, mean 30 % above it
    scale = np.where(rng.uniform(size=400) < 0.5, 1.0, 1.0 + rng.exponential(0.6, 400))
    f = truth[None] * scale[:, None, None]
    re_ = np.round(np.arange(19.6, 21.6 + 1e-9, 0.4), 3)
    r = A.bin_recovery(f, truth, ntrue, zf, dX, redges=re_)[0]
    assert abs(r["median_bias_pct"]) < 3.0
    assert 100.0 * (f.mean(axis=0)[3:5].mean() / 1e-22 - 1.0) > 15.0
