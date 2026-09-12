"""tests/test_ladder_tools.py — fast unit tests for the FP-ladder analysis tools
(validation/fp_ladder/ladder_table.py and mock_ppc.py).

These pin the two pieces of NEW arithmetic in the tools — the prior-whitening /
eigen helper and the pass/fail rollup + opening rule — plus the two small
verbatim-copy helpers (theta_pop reconstruction, S/N ramp amplitude). Nothing
here samples, loads a pack, or touches a frozen file. No I/O, no MCMC: the whole
module runs in well under a second.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    path = os.path.join(ROOT, rel)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


LT = _load("_ladder_table_under_test", "validation/fp_ladder/ladder_table.py")
MP = _load("_mock_ppc_under_test", "validation/fp_ladder/mock_ppc.py")


# ---------------------------------------------------------------------------
# whitening / eigen helper
# ---------------------------------------------------------------------------
def test_parse_coord_name_handles_scalar_vector_and_matrix_sites():
    assert LT.parse_coord_name("fp_l0") == ("fp_l0", ())
    assert LT.parse_coord_name("fp_h[3]") == ("fp_h", (3,))
    assert LT.parse_coord_name("fp_eps_z[2,5]") == ("fp_eps_z", (2, 5))
    with pytest.raises(ValueError):
        LT.parse_coord_name("fp_h(3)")


def test_assemble_fp_matrix_picks_the_named_coordinates_in_order():
    rng = np.random.default_rng(0)
    by = {"fp_l0": rng.normal(size=(2, 5)),
          "fp_h": rng.normal(size=(2, 5, 3)),
          "fp_eps_z": rng.normal(size=(2, 5, 4, 2))}
    names = ["fp_l0", "fp_h[0]", "fp_h[2]", "fp_eps_z[3,1]", "t[0]"]
    X, kept, missing = LT.assemble_fp_matrix(by, names)
    assert kept == ["fp_l0", "fp_h[0]", "fp_h[2]", "fp_eps_z[3,1]"]
    assert missing == ["t[0]"]                       # site absent -> reported, not faked
    assert X.shape == (2, 5, 4)
    assert np.allclose(X[..., 0], by["fp_l0"], atol=0)
    assert np.allclose(X[..., 1], by["fp_h"][:, :, 0], atol=0)
    assert np.allclose(X[..., 2], by["fp_h"][:, :, 2], atol=0)
    assert np.allclose(X[..., 3], by["fp_eps_z"][:, :, 3, 1], atol=0)


def test_assemble_fp_matrix_refuses_a_name_whose_index_depth_is_wrong():
    by = {"fp_h": np.zeros((2, 5, 3))}
    X, kept, missing = LT.assemble_fp_matrix(by, ["fp_h", "fp_h[9]"])
    assert kept == [] and missing == ["fp_h", "fp_h[9]"] and X.size == 0


def test_whiten_subtracts_the_prior_mean_and_divides_by_the_prior_sd():
    X = np.array([[1.0, 10.0], [3.0, 20.0]])
    u = LT.whiten(X, [1.0, 10.0], [2.0, 5.0])
    assert np.allclose(u, [[0.0, 0.0], [1.0, 2.0]], atol=0)


def test_eigen_report_recovers_a_known_isotropic_posterior():
    # u ~ N(0, s^2 I): Var u_i = s^2, so p_eff = P (1 - s^2), every eigenvalue is
    # s^2, every direction is informed (1 - s^2 > 0.5 for s^2 = 0.25) and the
    # participation ratio is P (the information is spread evenly).
    rng = np.random.default_rng(1)
    P, n, s = 6, 400_000, 0.5
    u = rng.normal(scale=s, size=(n, P))
    rep = LT.eigen_report(u)
    assert rep["n_coords"] == P and rep["n_draws"] == n
    assert rep["p_eff"] == pytest.approx(P * (1 - s ** 2), abs=0.02)
    assert np.allclose(rep["eigenvalues"], s ** 2, atol=0.01)
    assert rep["n_informed_1m_lambda_gt_0p5"] == P
    assert rep["participation_ratio"] == pytest.approx(P, abs=0.05)


def test_eigen_report_collapses_the_participation_ratio_on_one_informed_direction():
    # one tightly-constrained direction, the rest at the prior: PR -> 1.
    rng = np.random.default_rng(2)
    n, P = 200_000, 5
    u = rng.normal(size=(n, P))
    u[:, 0] *= 0.05                      # var 0.0025 -> 1 - lambda ~ 1
    rep = LT.eigen_report(u)
    assert rep["n_informed_1m_lambda_gt_0p5"] == 1
    assert rep["participation_ratio"] == pytest.approx(1.0, abs=0.05)
    assert rep["p_eff"] == pytest.approx(1.0, abs=0.05)


def test_eigen_report_declines_the_eigen_analysis_below_three_coordinates():
    rep = LT.eigen_report(np.random.default_rng(3).normal(size=(100, 2)))
    assert rep["eigenvalues"] is None and rep["participation_ratio"] is None
    assert rep["p_eff"] is not None                  # p_eff is still defined


def test_stuck_chains_excludes_only_chains_far_above_the_minimum():
    pe = np.array([[100.0, 102.0], [140.0, 142.0], [1000.0, 1002.0]])
    kept, excl, means = LT.stuck_chains(pe, excess_nats=50.0)
    assert excl == [2] and kept == [0, 1]
    assert means == pytest.approx([101.0, 141.0, 1001.0])
    # the threshold is a strict ">" on the chain MINIMUM, not on the mean of means
    kept2, excl2, _ = LT.stuck_chains(pe, excess_nats=1000.0)
    assert excl2 == [] and kept2 == [0, 1, 2]


def test_r2_on_block_is_one_for_an_exact_linear_combination_and_small_otherwise():
    rng = np.random.default_rng(4)
    u = rng.normal(size=(500, 3))
    y = 2.0 + 1.5 * u[:, 0] - 0.7 * u[:, 2]
    assert LT.r2_on_block(y, u) == pytest.approx(1.0, abs=1e-10)
    assert LT.r2_on_block(rng.normal(size=500), u) < 0.05
    assert LT.r2_on_block(np.zeros(500), u) is None          # zero SST -> refused


# ---------------------------------------------------------------------------
# rollup / opening rule
# ---------------------------------------------------------------------------
def _run(model, family, seed, status="PASS", hard=None, soft=None, gate_fails=None):
    return dict(model=model, family=family, seed=seed, gate_status=status,
                hard_flags=list(hard or []), soft_flags=list(soft or []),
                gate_fails=list(gate_fails or []))


FAMS = ("2lpt0", "london0", "saclay0")


def _all_pass(model):
    return [_run(model, f, s) for f in FAMS for s in (20260811, 20260812)]


def test_rollup_passes_a_model_only_when_every_family_and_seed_passes():
    roll = LT.rollup(_all_pass("M2"))
    assert roll["by_model"]["M2"]["status"] == "PASS"
    assert all(v["status"] == "PASS" for v in roll["by_family"].values())


def test_rollup_fails_the_family_and_the_model_on_a_single_seed_gate_fail():
    runs = _all_pass("M2")
    runs[1] = _run("M2", "2lpt0", 20260812, status="FAIL", gate_fails=["allz ge20.3 +5.1"])
    roll = LT.rollup(runs)
    assert roll["by_family"]["M2|2lpt0"]["status"] == "FAIL"
    assert roll["by_family"]["M2|london0"]["status"] == "PASS"
    assert roll["by_model"]["M2"]["status"] == "FAIL"
    assert "allz ge20.3 +5.1" in roll["by_model"]["M2"]["gate_fails"]


def test_rollup_fails_on_a_hard_flag_even_when_the_gate_passes():
    runs = _all_pass("M2")
    runs[4] = _run("M2", "london0", 20260811, hard=["E-BFMI min 0.11 < 0.2"])
    roll = LT.rollup(runs)
    assert roll["by_family"]["M2|london0"]["status"] == "FAIL"
    assert roll["by_model"]["M2"]["status"] == "FAIL"


def test_rollup_does_not_fail_on_a_soft_flag():
    runs = _all_pass("M2")
    runs[0] = _run("M2", "2lpt0", 20260811, soft=["omega_allz bias -4.10%"])
    roll = LT.rollup(runs)
    assert roll["by_model"]["M2"]["status"] == "PASS"
    assert roll["by_family"]["M2|2lpt0"]["soft_flags"] == ["omega_allz bias -4.10%"]


def test_opening_rule_stops_at_the_first_passing_model():
    roll = LT.rollup(_all_pass("M0") + _all_pass("M1") + _all_pass("M2"))
    op = LT.opening_rule(roll["by_model"])
    assert op["selected_model"] == "M0"               # lowest passing k
    assert op["decisions"]["M3"].startswith("NOT RUN")


def test_opening_rule_opens_the_next_rung_when_m2_fails():
    runs = _all_pass("M2")
    runs[0] = _run("M2", "2lpt0", 20260811, status="FAIL", gate_fails=["allz ge20.0 +0.9"])
    op = LT.opening_rule(LT.rollup(runs)["by_model"])
    assert op["decisions"]["M3"].startswith("RUN")
    assert op["decisions"]["M4"].startswith("UNDETERMINED")
    assert op["selected_model"] is None


def test_opening_rule_stops_and_returns_to_the_pi_when_m4_fails():
    runs = []
    for m in ("M2", "M3", "M4"):
        rs = _all_pass(m)
        rs[0] = _run(m, "2lpt0", 20260811, status="FAIL", gate_fails=["allz ge20.0 +0.9"])
        runs += rs
    op = LT.opening_rule(LT.rollup(runs)["by_model"])
    assert op["selected_model"] is None
    assert "return to the PI" in op["decisions"]["M5"]


# ---------------------------------------------------------------------------
# Omega[20.3, 21.6] on the paper's own reduction weights
# ---------------------------------------------------------------------------
def test_omega_allz_from_weights_is_the_path_weighted_reduction():
    # f constant = c: Omega = prefactor * c * sum(omega_w), independent of the z
    # weights (they normalise out) — the exact value the einsum must produce.
    B, Kf, D, c, pre = 4, 3, 200, 0.25, 2.0
    f = np.full((D, B, Kf), c)
    ow = np.array([0.0, 1.0, 2.0, 0.5])
    zw = np.array([3.0, 1.0, 0.0])
    out = LT.omega_allz_from_weights(f, np.full((B, Kf), c), ow, zw, pre)
    want = pre * c * ow.sum()
    assert out["post_p16_50_84"][1] == pytest.approx(want)
    assert out["truth"] == pytest.approx(want)
    assert out["median_bias_pct"] == pytest.approx(0.0)
    assert out["truth_in_68"] and out["truth_in_95"]
    assert out["dX_total"] == pytest.approx(zw.sum())


def test_omega_allz_from_weights_reports_bias_and_containment():
    B, Kf, D = 2, 2, 4001
    ow, zw, pre = np.array([1.0, 0.0]), np.array([1.0, 1.0]), 1.0
    # posterior spread around 1.1, truth 1.0 -> +10 % median bias, truth inside 68
    f = np.zeros((D, B, Kf))
    f[:, 0, :] = np.linspace(1.0, 1.2, D)[:, None]
    truth = np.zeros((B, Kf))
    truth[0, :] = 1.0
    out = LT.omega_allz_from_weights(f, truth, ow, zw, pre)
    assert out["median_bias_pct"] == pytest.approx(10.0, abs=0.01)
    assert out["truth_in_68"] is False and out["truth_in_95"] is False
    # widen the posterior about the same median -> truth comes back inside
    f2 = np.zeros((D, B, Kf))
    f2[:, 0, :] = np.linspace(0.5, 1.7, D)[:, None]
    out2 = LT.omega_allz_from_weights(f2, truth, ow, zw, pre)
    assert out2["truth_in_68"] is True and out2["truth_in_95"] is True


def test_omega_allz_from_weights_refuses_an_empty_z_window():
    assert LT.omega_allz_from_weights(np.ones((3, 2, 2)), np.ones((2, 2)),
                                      np.ones(2), np.zeros(2), 1.0) is None


def test_paper_omega_weights_match_hbi_reduction_and_the_declared_window():
    """The weights really are the paper's: build the same Posterior shell the
    long-chain campaign's quantities.py builds and compare, on a tiny synthetic
    _fdraws.npz. Skipped if the paper repository is not on this machine."""
    if not os.path.isdir(LT.PAPER_FIGURES):
        pytest.skip("paper_figures not present")
    sys.path.insert(0, LT.PAPER_FIGURES)
    HR = pytest.importorskip("hbi_reduction")
    assert tuple(HR.OMEGA_NHI) == (20.3, 21.6)          # the declared window
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_tmp_fdraws_omega_test.npz")
    n_edges = np.arange(19.5, 22.5001, 0.2)
    z_edges = np.array([2.0, 2.5, 3.0, 3.5])
    B, Kf, D = len(n_edges) - 1, len(z_edges) - 1, 25
    rng = np.random.default_rng(11)
    f = np.abs(rng.normal(1e-2, 1e-3, size=(D, B, Kf)))
    ftr = np.abs(rng.normal(1e-2, 1e-3, size=(B, Kf)))
    dX = np.array([1000.0, 2000.0, 1500.0])
    np.savez(tmp, f=f, truth_f=ftr, ntrue_edges=n_edges, zf_edges=z_edges, dX_k=dX)
    try:
        got = LT.paper_omega_20p3_21p6(tmp)
        P = HR.Posterior.__new__(HR.Posterior)
        P.f, P.n_edges, P.z_edges, P.dX = f, n_edges, z_edges, dX
        ow = P._omega_weight(*HR.OMEGA_NHI)
        zw = P._z_weight(*HR.LOWZ_SUPPORT)
        want = HR.OMEGA_PREFACTOR_CM2 * np.einsum("dbk,b,k->d", f, ow, zw) / zw.sum()
        assert got["post_p16_50_84"][1] == pytest.approx(float(np.median(want)), rel=1e-12)
        assert got["window_nhi"] == [20.3, 21.6]
        assert got["window_z"] == list(HR.LOWZ_SUPPORT)
        # the weight is the N-integral over the CLOSED window, not an open top
        assert ow[n_edges[:-1] >= 21.6 - 1e-9].sum() == 0.0
        assert ow[n_edges[1:] <= 20.3 + 1e-9].sum() == 0.0
    finally:
        os.remove(tmp)


def test_paper_omega_is_unavailable_rather_than_fatal_without_the_paper_repo():
    out = LT.paper_omega_20p3_21p6("/nonexistent.npz", paper_figures="/no/such/dir")
    assert "unavailable" in out and out.get("median_bias_pct") is None


# ---------------------------------------------------------------------------
# mock_ppc helpers
# ---------------------------------------------------------------------------
def test_snr_ramp_reproduces_the_frozen_real_run_amplitude():
    # the frozen real PPC's S/N marginal (figures/2026-08-26_ppc_b3): the
    # predeclared flag line 0.106 is the C1 value of this same statistic.
    ratios = [0.9392, 0.9558, 0.9808, 1.0168, 1.0465, 1.0516]
    blk = {"marginal_by_snr": [{"i": i, "obs": 1000.0, "mu_median": 1000.0 * r}
                               for i, r in enumerate(ratios)]}
    out = MP.snr_ramp(blk)
    assert out["amplitude"] == pytest.approx(0.1124, abs=2e-4)
    assert out["flag"] is True
    assert out["strata"] == list(range(6))


def test_snr_ramp_does_not_flag_a_flat_marginal_and_skips_empty_strata():
    blk = {"marginal_by_snr": [{"i": 0, "obs": 100.0, "mu_median": 100.0},
                               {"i": 1, "obs": 0.0, "mu_median": 3.0},
                               {"i": 2, "obs": 100.0, "mu_median": 105.0}]}
    out = MP.snr_ramp(blk)
    assert out["strata"] == [0, 2] and out["flag"] is False
    assert out["amplitude"] == pytest.approx(0.05)
    assert MP.snr_ramp({"marginal_by_snr": []})["amplitude"] is None


def test_theta_pop_reconstruction_matches_the_model_cc_construction():
    # an independent, literal transcription of model_cc_ladder's population block
    rng = np.random.default_rng(5)
    n, B, Kf = 7, 6, 4
    sN, sz = rng.uniform(0.1, 1, n), rng.uniform(0.1, 1, n)
    lev, slo = rng.normal(size=n), rng.normal(size=n)
    eN, ez = rng.normal(size=(n, B - 2)), rng.normal(size=(n, B, Kf - 1))
    got = MP.theta_pop_from_population_sites(sN, sz, lev, slo, eN, ez)
    want = np.empty((n, B, Kf))
    b_idx = np.arange(B) - 0.5 * (B - 1)
    for d in range(n):
        curv = np.cumsum(np.cumsum(np.concatenate([np.zeros(2), eN[d]])))[:B]
        col0 = lev[d] + slo[d] * b_idx + sN[d] * curv
        want[d] = col0[:, None] + np.concatenate(
            [np.zeros((B, 1)), sz[d] * np.cumsum(ez[d], axis=1)], axis=1)
    assert got.shape == (n, B, Kf)
    assert np.allclose(got, want, rtol=0, atol=1e-12)
