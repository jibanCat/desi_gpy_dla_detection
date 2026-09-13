"""tests/test_variant_table.py — fast unit tests for the absorber-side ladder
variant table (validation/fp_ladder/variant_table.py).

These pin the NEW arithmetic and the NEW grouping rules of the tool: the
variant id, the structured-residual summaries (zigzag, marginal max-dev and
slope, the 13 CDDF bins), the seed rollup, the equal-weight imputation pooling,
and the `DEFECT MOVED` annotation of the sealed predeclaration §4. Nothing here
samples, loads a pack, or touches a frozen file: no MCMC, no disk I/O beyond
importing the module. The whole file runs in well under a second.
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


VT = _load("_variant_table_under_test", "validation/fp_ladder/variant_table.py")


# ---------------------------------------------------------------------------
# variant id
# ---------------------------------------------------------------------------
def test_variant_id_strips_the_ladder_and_diag_prefixes():
    assert VT.variant_of({"stage": "LADDER_A0+R1c+C1nsadd", "ladder": "ORACLE"}) \
        == "A0+R1c+C1nsadd"
    assert VT.variant_of({"stage": "DIAG_OCz", "ladder": "ORACLE"}) == "OCz"
    assert VT.variant_of({"stage": "LADDER_A0", "ladder": "M1CUT"}) == "A0"


def test_variant_id_falls_back_to_the_ladder_field_on_an_empty_stage():
    assert VT.variant_of({"stage": "", "ladder": "M3"}) == "M3"
    assert VT.variant_of({"ladder": "ORACLE"}) == "ORACLE"
    assert VT.variant_of({"stage": "   ", "ladder": "M0"}) == "M0"
    assert VT.variant_of({}) == "UNKNOWN"


def test_variant_id_collapses_a_stage_that_only_restates_the_ladder():
    # the 2026-09-12 oracle runs carry stage 'diag_oracle' on ladder 'ORACLE';
    # they must group WITH the ORACLE runs, not as a separate one-run variant.
    assert VT.variant_of({"stage": "diag_oracle", "ladder": "ORACLE"}) == "ORACLE"
    assert VT.variant_of({"stage": "DIAG_OP", "ladder": "ORACLE"}) == "OP"


def test_imputation_is_read_only_from_the_lam_cut_block():
    assert VT.imputation_of({"diagnostics": {"lam_cut": {"J": 8, "j": 3}}}) == (8, 3)
    assert VT.imputation_of({"diagnostics": {"lam_cut": None}}) == (None, None)
    assert VT.imputation_of({}) == (None, None)


# ---------------------------------------------------------------------------
# zigzag
# ---------------------------------------------------------------------------
def test_zigzag_stats_reports_the_range_rms_and_max_abs():
    rows = [{"bin": [19.7, 19.9], "median_bias_pct": -3.0},
            {"bin": [19.9, 20.1], "median_bias_pct": +5.0},
            {"bin": [20.1, 20.3], "median_bias_pct": -4.0}]
    z = VT.zigzag_stats(rows)
    assert z["n_bins"] == 3
    assert z["range"] == pytest.approx(9.0, abs=0)   # +5 - (-4)
    assert z["max_abs"] == pytest.approx(5.0, abs=0)
    assert z["rms"] == pytest.approx(np.sqrt((9 + 25 + 16) / 3.0), rel=1e-12)


def test_zigzag_stats_skips_bins_without_a_bias_and_survives_an_empty_list():
    z = VT.zigzag_stats([{"bin": [1, 2], "median_bias_pct": None},
                         {"bin": [2, 3], "median_bias_pct": 2.0}])
    assert z["n_bins"] == 1 and z["range"] == 0.0
    e = VT.zigzag_stats([])
    assert e["n_bins"] == 0 and e["range"] is None and e["rms"] is None


# ---------------------------------------------------------------------------
# predictive marginals
# ---------------------------------------------------------------------------
def test_marginal_stats_excludes_dead_strata_and_never_counts_them_as_deviation():
    # the runner writes an unobserved stratum as exactly 0.0 (mu / max(obs, 1))
    m = VT.marginal_stats([0.0, 0.0, 0.98, 1.02])
    assert m["n_strata"] == 4 and m["n_live"] == 2
    assert m["live_idx"] == [2, 3]
    assert m["max_abs_dev"] == pytest.approx(0.02, abs=1e-12)  # NOT 1.0


def test_marginal_stats_recovers_an_exact_linear_ramp_slope():
    ratios = [0.0, 1.00, 1.02, 1.04, 1.06]          # live strata 1..4, slope +0.02
    m = VT.marginal_stats(ratios)
    assert m["slope"] == pytest.approx(0.02, rel=1e-12)
    assert m["argmax_idx"] == 4
    assert m["ptp"] == pytest.approx(0.06, rel=1e-12)


def test_marginal_stats_reports_no_slope_with_fewer_than_two_live_strata():
    m = VT.marginal_stats([0.0, 0.0, 1.5])
    assert m["n_live"] == 1 and m["slope"] is None
    dead = VT.marginal_stats([0.0, 0.0])
    assert dead["n_live"] == 0 and dead["max_abs_dev"] is None
    assert VT.marginal_stats(None)["n_strata"] == 0


def test_mean_marginals_refuses_a_partial_mean_when_one_imputation_lacks_the_block():
    a = {"mu_over_obs_by_snr": [1.0, 1.2], "mu_over_obs_by_nhat_K": [[1.0], [1.0]]}
    b = {"mu_over_obs_by_snr": [1.0, 0.8], "mu_over_obs_by_nhat_K": [[1.2], [0.8]]}
    out = VT._mean_marginals([a, b])
    assert out["mu_over_obs_by_snr"] == pytest.approx([1.0, 1.0])
    assert out["mu_over_obs_by_nhat_K"] == [pytest.approx([1.1]), pytest.approx([0.9])]
    assert VT._mean_marginals([a, {}])["mu_over_obs_by_snr"] is None
    assert VT._mean_marginals([a, {}])["mu_over_obs_by_nhat_K"] == []


# ---------------------------------------------------------------------------
# the 13 CDDF bins
# ---------------------------------------------------------------------------
def _toy_cddf(n_edges, dX, scale_lo=0.5, scale_hi=1.5, n=101):
    """f[d, b, k] = base[b, k] * s_d with s_d a symmetric ramp (median 1)."""
    B, K = len(n_edges) - 1, len(dX)
    base = np.arange(1.0, B * K + 1.0).reshape(B, K)
    s = np.linspace(scale_lo, scale_hi, n)
    f = base[None, :, :] * s[:, None, None]
    return f, base


def test_cddf_bin_biases_keeps_only_the_bins_at_or_above_19p7():
    n_edges = [19.0, 19.5, 19.7, 20.0, 20.5]
    dX = np.array([1.0, 3.0])
    f, base = _toy_cddf(n_edges, dX)
    out = VT.cddf_bin_biases(f, base, n_edges, dX)
    assert [o["bin"] for o in out] == [[19.7, 20.0], [20.0, 20.5]]


def test_cddf_bin_biases_are_path_weighted_and_zero_on_the_truth():
    n_edges = [19.7, 19.9, 20.1]
    dX = np.array([1.0, 3.0])
    f, base = _toy_cddf(n_edges, dX)
    out = VT.cddf_bin_biases(f, base, n_edges, dX)
    assert len(out) == 2
    for b, o in enumerate(out):
        expect = float((base[b] * dX).sum() / dX.sum())
        assert o["truth"] == pytest.approx(expect, rel=1e-12)
        assert o["median_bias_pct"] == pytest.approx(0.0, abs=1e-9)
        assert o["truth_in_68"] and o["truth_in_95"]


def test_cddf_bin_biases_detect_a_known_multiplicative_bias():
    n_edges = [19.7, 19.9]
    dX = np.array([2.0, 2.0])
    f, base = _toy_cddf(n_edges, dX)
    out = VT.cddf_bin_biases(f, base / 1.25, n_edges, dX)     # truth 25 % low
    assert out[0]["median_bias_pct"] == pytest.approx(25.0, rel=1e-6)
    assert out[0]["truth_in_68"] is True                      # ramp 0.5..1.5 is wide


def test_cddf_bin_biases_refuse_a_zero_path_length():
    n_edges = [19.7, 19.9]
    f, base = _toy_cddf(n_edges, np.array([1.0, 1.0]))
    assert VT.cddf_bin_biases(f, base, n_edges, np.zeros(2)) == []


# ---------------------------------------------------------------------------
# seed rollup
# ---------------------------------------------------------------------------
def test_seed_rollup_reports_no_spread_for_a_single_seed():
    r = VT.seed_rollup([1.5])
    assert r["n"] == 1 and r["mean"] == 1.5 and r["spread"] is None


def test_seed_rollup_uses_the_sample_sd_and_drops_missing_values():
    r = VT.seed_rollup([1.0, 3.0, None, float("nan")])
    assert r["n"] == 2 and r["mean"] == pytest.approx(2.0)
    assert r["spread"] == pytest.approx(np.std([1.0, 3.0], ddof=1))
    assert r["min"] == 1.0 and r["max"] == 3.0
    empty = VT.seed_rollup([None, None])
    assert empty["n"] == 0 and empty["mean"] is None and empty["spread"] is None


# ---------------------------------------------------------------------------
# the DEFECT MOVED annotation (predeclaration §4)
# ---------------------------------------------------------------------------
def test_defect_moved_fires_only_when_the_headline_improves():
    delta = {"zigzag_range": 5.0}
    spread = {"zigzag_range": 0.1}
    flag, reasons, undec = VT.movement_flag(delta, spread, headline_improved=True)
    assert flag and reasons and not undec
    flag2, _, _ = VT.movement_flag(delta, spread, headline_improved=False)
    assert flag2 is False


def test_defect_moved_needs_more_than_two_seed_spreads_of_worsening():
    spread = {"zigzag_range": 1.0}
    assert VT.movement_flag({"zigzag_range": 1.9}, spread, True)[0] is False
    assert VT.movement_flag({"zigzag_range": 2.1}, spread, True)[0] is True


def test_defect_moved_never_fires_on_an_improvement():
    flag, reasons, undec = VT.movement_flag({"snr_max_abs_dev": -0.5},
                                            {"snr_max_abs_dev": 0.001}, True)
    assert flag is False and reasons == [] and undec == []


def test_a_worsening_without_a_noise_scale_is_undecidable_not_cleared():
    flag, reasons, undec = VT.movement_flag({"K1_bias_ge20p3": 4.0},
                                            {"K1_bias_ge20p3": None}, True)
    assert flag is False and reasons == []
    assert len(undec) == 1 and "no seed spread" in undec[0]
    # a zero spread is treated the same way (never a divide-by-zero pass)
    assert VT.movement_flag({"K1_bias_ge20p3": 4.0},
                            {"K1_bias_ge20p3": 0.0}, True)[2]


# ---------------------------------------------------------------------------
# equal-weight imputation pooling
# ---------------------------------------------------------------------------
def test_pool_f_draws_takes_equal_numbers_of_draws_from_every_imputation(tmp_path):
    ft = np.ones((2, 2))
    paths = []
    for j, (n, v) in enumerate(((5, 1.0), (7, 2.0))):
        p = tmp_path / f"r_j{j}_fdraws.npz"
        np.savez(p, f=np.full((n, 2, 2), v), truth_f=ft,
                 ntrue_edges=np.array([19.7, 19.9, 20.1]),
                 zf_edges=np.array([2.0, 2.5, 3.0]), dX_k=np.array([1.0, 1.0]))
        paths.append(str(p))
    f, t, meta = VT.pool_f_draws(paths)
    assert f.shape == (10, 2, 2)                       # min(5, 7) x 2, equal weights
    assert meta["n_per_imputation"] == 5 and meta["n_imputations"] == 2
    assert np.allclose(t, ft, atol=0)
    assert np.allclose(f[:5], 1.0, atol=0) and np.allclose(f[5:], 2.0, atol=0)


def test_pool_f_draws_refuses_imputations_that_disagree_about_the_truth(tmp_path):
    paths = []
    for j, tv in enumerate((1.0, 1.5)):
        p = tmp_path / f"r_j{j}_fdraws.npz"
        np.savez(p, f=np.ones((3, 2, 2)), truth_f=np.full((2, 2), tv),
                 ntrue_edges=np.array([19.7, 19.9, 20.1]),
                 zf_edges=np.array([2.0, 2.5, 3.0]), dX_k=np.array([1.0, 1.0]))
        paths.append(str(p))
    with pytest.raises(ValueError):
        VT.pool_f_draws(paths)


# ---------------------------------------------------------------------------
# aggregation and the movement table, on synthetic units
# ---------------------------------------------------------------------------
def _unit(variant, family, seed, b0, b3, zig, snr, k1=0.0, status="PASS"):
    return dict(
        file=f"RUN_{variant}_{family}_s{seed}.json", kind="RUN", variant=variant,
        ladder="ORACLE", family=family, seed=seed, gate_status=status, gate_fails=[],
        allz={"ge20.0": dict(bias=b0, in68=True, in95=True),
              "ge20.3": dict(bias=b3, in68=True, in95=True)},
        bins={"ge20.0": {"B1": dict(bias=0.1, in68=True, in95=True)},
              "ge20.3": {"B1": dict(bias=0.2, in68=True, in95=True)}},
        kblocks={"ge20.0": {"K1": dict(bias=0.0)},
                 "ge20.3": {"K1": dict(bias=k1)}},
        structured=dict(zigzag=dict(range=zig, rms=None, max_abs=None),
                        snr=dict(max_abs_dev=snr, slope=None),
                        nhat=dict(max_abs_dev=None), z=dict(max_abs_dev=None),
                        cddf13=[], cddf13_max_abs_bias=None, cddf13_rms_bias=None),
        omega_20p3_21p6={"median_bias_pct": 1.0},
        sampler=dict(divergences=0, rhat_max=1.0, ess_min=3000.0, ebfmi_min=0.9),
        support_gate=None, fixed_files={}, imputation_j=None)


def test_aggregate_averages_over_seeds_and_keeps_the_seed_spread():
    units = [_unit("A0", "2lpt0", 1, 1.0, 3.0, 20.0, 0.02),
             _unit("A0", "2lpt0", 2, 1.2, 3.2, 20.4, 0.03)]
    agg = VT.aggregate(units)
    rec = agg["A0|2lpt0"]
    assert rec["n_units"] == 2 and rec["status"] == "PASS"
    assert rec["metrics"]["bias_ge20p0"]["mean"] == pytest.approx(1.1)
    assert rec["metrics"]["zigzag_range"]["spread"] == pytest.approx(
        np.std([20.0, 20.4], ddof=1))
    assert rec["allz_in68"] == "4/4" and rec["perbin_in95"] == "4/4"


def test_a_family_fails_when_any_of_its_seeds_fails_the_frozen_gate():
    units = [_unit("A0", "2lpt0", 1, 1.0, 3.0, 20.0, 0.02),
             _unit("A0", "2lpt0", 2, 1.0, 3.0, 20.0, 0.02, status="FAIL")]
    agg = VT.aggregate(units)
    assert agg["A0|2lpt0"]["status"] == "FAIL"
    assert VT.by_variant(agg)["A0"]["status"] == "FAIL"
    assert VT.by_variant(agg)["A0"]["n_families_pass"] == 0


def test_movement_table_flags_a_repair_that_moves_the_defect():
    units = [_unit("A0", "2lpt0", 1, 2.0, 3.0, 20.0, 0.02),
             _unit("A0", "2lpt0", 2, 2.0, 3.0, 20.1, 0.02),
             # headline improves, zigzag blows up: the §4 situation
             _unit("A0+R1a", "2lpt0", 1, 0.2, 0.3, 30.0, 0.02)]
    moves = VT.movement_table(VT.aggregate(units), "A0")
    row = [m for m in moves if m["variant"] == "A0+R1a"][0]
    assert row["headline_improved_on"] == ["ge20.0", "ge20.3"]
    assert row["delta_abs"]["zigzag_range"] == pytest.approx(9.95)
    assert row["defect_moved"] is True
    assert any("zigzag_range" in r for r in row["defect_moved_reasons"])


def test_movement_table_does_not_flag_a_repair_that_improves_everything():
    units = [_unit("A0", "2lpt0", 1, 2.0, 3.0, 20.0, 0.05),
             _unit("A0", "2lpt0", 2, 2.0, 3.0, 20.1, 0.05),
             _unit("A0+R1a", "2lpt0", 1, 0.2, 0.3, 5.0, 0.01)]
    moves = VT.movement_table(VT.aggregate(units), "A0")
    row = [m for m in moves if m["variant"] == "A0+R1a"][0]
    assert row["defect_moved"] is False and row["undecidable"] == []


def test_movement_table_reports_a_family_the_baseline_never_ran():
    units = [_unit("A0", "2lpt0", 1, 2.0, 3.0, 20.0, 0.02),
             _unit("A0+R1a", "saclay0", 1, 0.2, 0.3, 30.0, 0.02)]
    moves = VT.movement_table(VT.aggregate(units), "A0")
    row = [m for m in moves if m["variant"] == "A0+R1a"][0]
    assert row["comparable"] is False and "saclay0" in row["note"]


def test_build_units_groups_only_the_m1cut_imputations():
    runs = [dict(variant="A0", ladder="ORACLE", family="2lpt0", seed=1,
                 imputation_j=None),
            dict(variant="F", ladder="M1CUT", family="2lpt0", seed=1, imputation_j=0),
            dict(variant="F", ladder="M1CUT", family="2lpt0", seed=1, imputation_j=1),
            dict(variant="F", ladder="M1CUT", family="london0", seed=1, imputation_j=0)]
    singles, groups = VT.build_units(runs)
    assert len(singles) == 1 and singles[0]["variant"] == "A0"
    assert sorted(groups) == [("F", "2lpt0", 1), ("F", "london0", 1)]
    assert len(groups[("F", "2lpt0", 1)]) == 2
