"""Queue-2 FF+FP estimator tests (spec: desi_gpy_dla_notes/notes/2026-07-11_q2_spec.md).

TDD contract, written BEFORE the implementation:
  * zero FP reduces to completeness-only
  * perfect C recovers n_real / (dX dN)
  * doubling exposure doubles the expected FP
  * an empty tier draws EXACTLY 0 FP under the FIX-3c Gamma construction
    (single Jeffreys 1/2 at the LOWEST-N edge; NEVER per-cell +1/2)
  * negative bins are preserved (no clipping) and flagged "consistent with zero"
  * purity rho is never loaded/used in the FF+FP path (no-double-count guard)
  * integration on the real stamped Q1 artifacts (fast array math)

Run:  conda run -n gpdla python -m pytest tests/test_ff_fp_estimator.py -v
"""
import json
import os

import numpy as np
import pytest

from CDDF_analysis.hbi import ff_fp_estimator as ff


# ---------------------------------------------------------------------------
# subtract_fp
# ---------------------------------------------------------------------------
class TestSubtractFP:
    def test_zero_fp_reduces_to_completeness_only(self):
        n_obs = np.array([10.0, 5.0, 2.0])
        out = ff.subtract_fp(n_obs, np.zeros(3), w=165.9)
        np.testing.assert_array_equal(out, n_obs)

    def test_doubling_exposure_doubles_fp(self):
        n_obs = np.array([100.0, 50.0])
        n_fp = np.array([2.0, 1.0])
        d1 = n_obs - ff.subtract_fp(n_obs, n_fp, w=3.0)
        d2 = n_obs - ff.subtract_fp(n_obs, n_fp, w=6.0)
        np.testing.assert_allclose(d2, 2.0 * d1)

    def test_no_clipping_negative_bins_preserved(self):
        out = ff.subtract_fp(np.array([1.0]), np.array([2.0]), w=1.0)
        assert out[0] == -1.0  # NOT clipped to 0

    def test_per_bin_weight_broadcast(self):
        out = ff.subtract_fp(np.array([10.0, 10.0]), np.array([1.0, 1.0]),
                             w=np.array([2.0, 4.0]))
        np.testing.assert_allclose(out, [8.0, 6.0])


# ---------------------------------------------------------------------------
# apply_completeness
# ---------------------------------------------------------------------------
class TestApplyCompleteness:
    def test_perfect_C_recovers_density(self):
        n_real = np.array([10.0, 20.0])
        dN = np.array([1e17, 2e17])
        f = ff.apply_completeness(n_real, np.ones(2), dX=100.0, dN=dN)
        np.testing.assert_allclose(f, n_real / (100.0 * dN))

    def test_C_half_doubles_density(self):
        f1 = ff.apply_completeness(np.array([10.0]), np.array([1.0]), 10.0,
                                   np.array([1.0]))
        f2 = ff.apply_completeness(np.array([10.0]), np.array([0.5]), 10.0,
                                   np.array([1.0]))
        np.testing.assert_allclose(f2, 2.0 * f1)

    def test_negative_n_real_passes_through_no_clip(self):
        f = ff.apply_completeness(np.array([-5.0]), np.array([0.5]), 10.0,
                                  np.array([1.0]))
        assert f[0] == pytest.approx(-1.0)

    def test_undefined_C_gives_nan(self):
        f = ff.apply_completeness(np.array([1.0, 1.0]),
                                  np.array([np.nan, 0.0]), 10.0, np.ones(2))
        assert np.isnan(f).all()


# ---------------------------------------------------------------------------
# FIX-3c Gamma FP draws (replicating Loa0FP.resample semantics)
# ---------------------------------------------------------------------------
class TestFPGammaDraws:
    def test_empty_tier_draws_exactly_zero(self):
        # counts: only the lowest-N row (anchor) may draw >0; every other empty
        # cell draws EXACTLY 0 (Gamma(0) = 0 — no per-cell +1/2, ever).
        n = np.zeros((5, 3))
        n[0, 1] = 4.0  # some FP in the lowest-N row
        rng = np.random.default_rng(0)
        draws = ff.fp_gamma_draws(n, floor_axis=0, ell_eff=13.6, rng=rng,
                                  ndraw=400)
        assert draws.shape == (400, 5, 3)
        # rows 1..4 are ABOVE the anchor and empty -> exactly 0 in every draw
        assert np.all(draws[:, 1:, :] == 0.0)

    def test_anchor_row_carries_single_half_count(self):
        # all-zero grid: whole-grid total ~ Gamma(1/2)/ell_eff*ell_eff, mean 1/2
        n = np.zeros((4, 2))
        rng = np.random.default_rng(1)
        draws = ff.fp_gamma_draws(n, floor_axis=0, ell_eff=10.0, rng=rng,
                                  ndraw=200_000)
        tot = draws.sum(axis=(1, 2))
        assert np.all(draws[:, 1:, :] == 0.0)
        assert tot.mean() == pytest.approx(0.5, rel=0.05)

    def test_total_mean_is_sum_plus_half_grid_invariant(self):
        rng = np.random.default_rng(2)
        n1 = np.array([[3.0, 1.0], [0.0, 2.0]])       # total 6
        d1 = ff.fp_gamma_draws(n1, 0, ell_eff=5.0, rng=rng, ndraw=200_000)
        assert d1.sum(axis=(1, 2)).mean() == pytest.approx(6.5, rel=0.02)
        # split the same counts across twice as many z-cells: same total law
        n2 = np.array([[1.5, 1.5, 0.5, 0.5], [0.0, 0.0, 1.0, 1.0]])
        d2 = ff.fp_gamma_draws(n2, 0, ell_eff=5.0, rng=rng, ndraw=200_000)
        assert d2.sum(axis=(1, 2)).mean() == pytest.approx(6.5, rel=0.02)

    def test_floor_axis_1_anchor_on_lowest_col(self):
        n = np.zeros((3, 4))
        rng = np.random.default_rng(3)
        draws = ff.fp_gamma_draws(n, floor_axis=1, ell_eff=10.0, rng=rng,
                                  ndraw=50_000)
        assert np.all(draws[:, :, 1:] == 0.0)
        assert draws.sum(axis=(1, 2)).mean() == pytest.approx(0.5, rel=0.1)

    def test_matches_loa0fp_resample_semantics(self):
        # cross-check against the committed Loa0FP.resample on a toy product
        from CDDF_analysis.hbi.cddf_catalog_hbi import Loa0FP
        n_fine = np.array([[2.0, 1.0], [3.0, 0.0], [0.0, 0.0]])
        fp = Loa0FP(
            n_fp_molly=np.zeros((2, 2)), b_fp_molly=np.zeros((2, 2)),
            snr_edges=[0, 1, np.inf], nhi_edges=[17.2, 20.0, np.inf],
            n_fp_fine=n_fine, logN_lo=[17.2, 17.3, 17.4],
            logN_hi=[17.3, 17.4, 17.5], band_eta_per_nbin=np.zeros(3),
            n_sl_loa0=100, n_sl_prod=1000, ell_eff=10.0)
        ref = np.array([fp.resample(np.random.default_rng(s))
                        ._gamma_draw["fine_count"] for s in range(4000)])
        rng = np.random.default_rng(0)
        mine = ff.fp_gamma_draws(n_fine, 0, ell_eff=10.0, rng=rng, ndraw=4000)
        # identical zero pattern (empty non-anchor cells exactly 0 in both)
        assert np.all(ref[:, 2, :] == 0.0) and np.all(mine[:, 2, :] == 0.0)
        assert np.all(ref[:, 1, 1] == 0.0) and np.all(mine[:, 1, 1] == 0.0)
        # matching means and variances cell-by-cell (statistical)
        np.testing.assert_allclose(mine.mean(0), ref.mean(0), atol=0.15)
        np.testing.assert_allclose(mine.std(0), ref.std(0), atol=0.15)


# ---------------------------------------------------------------------------
# Jeffreys-Beta completeness draws
# ---------------------------------------------------------------------------
class TestBetaCDraws:
    def test_jeffreys_mean(self):
        rng = np.random.default_rng(4)
        d = ff.beta_c_draws(np.array([[80.0]]), np.array([[100.0]]), rng,
                            ndraw=100_000)
        assert d.shape == (100_000, 1, 1)
        assert d.mean() == pytest.approx(80.5 / 101.0, abs=0.005)

    def test_empty_cell_is_nan(self):
        rng = np.random.default_rng(5)
        d = ff.beta_c_draws(np.array([[0.0, 5.0]]), np.array([[0.0, 10.0]]),
                            rng, ndraw=100)
        assert np.isnan(d[:, 0, 0]).all()
        assert np.isfinite(d[:, 0, 1]).all()

    def test_draws_in_unit_interval(self):
        rng = np.random.default_rng(6)
        d = ff.beta_c_draws(np.array([[3.0]]), np.array([[3.0]]), rng, 1000)
        assert np.all((d > 0) & (d < 1))


# ---------------------------------------------------------------------------
# occupancy-weighted C_eff
# ---------------------------------------------------------------------------
class TestCEffOccupancy:
    def test_hand_example(self):
        C = np.array([[0.2, 0.4],
                      [0.6, 0.8],
                      [1.0, 1.0]])
        occ = np.array([[10.0, 0.0],
                        [30.0, 10.0],
                        [0.0, 0.0]])
        # rows (1, 2): col0 -> (30*0.6)/(30) = 0.6 ; col1 -> 0.8
        out = ff.c_eff_occupancy(C, occ, rows=[1, 2])
        np.testing.assert_allclose(out, [0.6, 0.8])
        # all rows: col0 -> (10*.2+30*.6)/40 = 0.5
        out = ff.c_eff_occupancy(C, occ, rows=[0, 1, 2])
        np.testing.assert_allclose(out, [0.5, 0.8])

    def test_zero_occupancy_gives_nan(self):
        C = np.array([[0.5], [0.5]])
        occ = np.zeros((2, 1))
        out = ff.c_eff_occupancy(C, occ, rows=[0, 1])
        assert np.isnan(out).all()

    def test_nan_C_cell_excluded(self):
        C = np.array([[np.nan], [0.5]])
        occ = np.array([[10.0], [10.0]])
        out = ff.c_eff_occupancy(C, occ, rows=[0, 1])
        np.testing.assert_allclose(out, [0.5])

    def test_draw_axis_broadcast(self):
        C = np.stack([np.full((2, 2), 0.5), np.full((2, 2), 0.25)])  # (2 draws,)
        occ = np.ones((2, 2))
        out = ff.c_eff_occupancy(C, occ, rows=[0, 1])
        assert out.shape == (2, 2)
        np.testing.assert_allclose(out[0], 0.5)
        np.testing.assert_allclose(out[1], 0.25)


# ---------------------------------------------------------------------------
# mc_band: negative bins preserved + flagged; synthetic closure
# ---------------------------------------------------------------------------
class TestMCBand:
    def _mk(self, n_obs, fp_point, w=1.0, C=None, dX=100.0, ndraw=2000):
        nb = len(n_obs)
        rng = np.random.default_rng(7)
        fp_draws = np.repeat(fp_point[None, :], ndraw, axis=0)
        C_point = np.ones(nb) if C is None else C
        C_draws = np.repeat(C_point[None, :], ndraw, axis=0)
        return ff.mc_band(rng, ndraw, np.asarray(n_obs, float), fp_point,
                          fp_draws, w, C_point, C_draws, dX, np.ones(nb))

    def test_negative_bin_preserved_and_flagged(self):
        # bin 0: n_real = 4 - 5 = -1 with Poisson(4) sigma ~2 -> |n_real| < 2*sigma
        # -> negative value KEPT and flagged "consistent with zero"
        out = self._mk(np.array([4.0, 100.0]), np.array([5.0, 0.0]))
        assert out["n_real_point"][0] == -1.0            # kept, not clipped
        assert bool(out["flag_zero_consistent"][0]) is True
        assert bool(out["flag_zero_consistent"][1]) is False
        assert out["f_point"][0] < 0

    def test_significantly_negative_bin_not_zero_consistent(self):
        # n_real = 1 - 5 = -4 with Poisson(1) sigma ~1: significantly negative
        # (|n_real| >= 2*sigma) -> NOT flagged zero-consistent, still not clipped
        out = self._mk(np.array([1.0]), np.array([5.0]))
        assert out["n_real_point"][0] == -4.0
        assert bool(out["flag_zero_consistent"][0]) is False

    def test_zero_fp_band_is_poisson_only(self):
        out = self._mk(np.array([10000.0]), np.zeros(1))
        # Poisson(1e4): std ~100 -> band halfwidth ~1% of the point
        w68 = (out["f_q84"][0] - out["f_q16"][0]) / 2.0
        assert w68 == pytest.approx(100.0 / 100.0, rel=0.15)  # f = n/dX, dX=100

    def test_synthetic_perfect_closure(self):
        # n_obs = C * n_true + w * n_fp  -> estimator recovers n_true exactly
        n_true = np.array([100.0, 50.0, 10.0])
        C = np.array([0.8, 0.5, 0.9])
        fp = np.array([3.0, 1.0, 0.0])
        w = 2.0
        n_obs = C * n_true + w * fp
        out = self._mk(n_obs, fp, w=w, C=C)
        np.testing.assert_allclose(out["n_true_point"], n_true, rtol=1e-12)

    def test_undefined_C_bins_are_nan_but_subtraction_survives(self):
        out = self._mk(np.array([10.0]), np.array([1.0]), w=1.0,
                       C=np.array([np.nan]))
        assert out["n_real_point"][0] == 9.0
        assert np.isnan(out["f_point"][0])


# ---------------------------------------------------------------------------
# no-double-count guard: rho must never be usable in the FF+FP path
# ---------------------------------------------------------------------------
class TestRhoGuard:
    def test_rho_guard_raises_on_any_access(self):
        g = ff.RhoGuard("purity")
        with pytest.raises(ff.RhoAccessError):
            _ = g[0, 0]
        with pytest.raises(ff.RhoAccessError):
            np.asarray(g)
        with pytest.raises(ff.RhoAccessError):
            _ = g.sum()

    def test_load_molly_completeness_installs_guard(self):
        mc = ff.load_molly_completeness()  # committed lya_only-nhi195 TSV
        assert mc["completeness"].shape[0] == 8  # 8 SNR rows
        with pytest.raises(ff.RhoAccessError):
            np.asarray(mc["purity"])

    def test_completeness_numerator_is_truth_matched_only(self):
        # audit: C = n_found/n_fid with n_found <= n_fid (no unmatched detections
        # in the numerator -> no purity embedded -> no FP double-count)
        counts = ff.load_molly_counts()
        if counts is None:
            pytest.skip("molly counts cache not built (run --build-molly-counts)")
        nf, nfid = counts["cmp_nfound"], counts["cmp_nfid"]
        assert np.all(nf <= nfid + 1e-9)
        # and the cached ratios reproduce the committed TSV where both defined
        mc = ff.load_molly_completeness()
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(nfid > 0, nf / nfid, np.nan)
        both = np.isfinite(ratio) & np.isfinite(mc["completeness"])
        assert np.nanmax(np.abs(ratio[both] - mc["completeness"][both])) < 5e-3


# ---------------------------------------------------------------------------
# integration on the real stamped artifacts (fast — array math only)
# ---------------------------------------------------------------------------
needs_inputs = pytest.mark.skipif(
    not (os.path.exists(ff.DEF_LOA0_PRODUCT)
         and os.path.exists(ff.DEF_MOLLY_TSV)),
    reason="loa-0 FP product / molly TSV not reachable")


@needs_inputs
class TestIntegration:
    @pytest.fixture(scope="class")
    def res(self):
        return ff.run_estimator(mock="2lpt0", calib_mock="2lpt0", ndraw=300,
                                seed=0)

    def test_dla_tier_fp_is_exactly_zero(self, res):
        # loa-0 measured 0 FP above 20.3 -> point FP AND every draw exactly 0
        st = res["strata"]["full"]
        assert st["closure"]["dla_20.3"]["fp_subtracted_total"] == 0.0
        assert st["closure"]["dla_20.3"]["fp_band_hi95"] == 0.0

    def test_subdla_fp_positive_and_closure_below_raw(self, res):
        st = res["strata"]["full"]
        sub = st["closure"]["subdla_195_203"]
        assert sub["fp_subtracted_total"] > 0
        # FP subtraction must LOWER the sub-DLA ratio vs raw detected-space
        assert sub["R_subtracted_only"] < sub["R_raw"]

    def test_strata_and_zbins_present(self, res):
        for s in ("full", "z_2.0_2.5", "z_2.5_3.0", "z_3.0_3.5",
                  "snr_gt4", "snr_2_4"):
            assert s in res["strata"], s

    def test_z_split_additivity_guard(self, res):
        assert res["checks"]["z_split_additivity_ok"] is True

    def test_provenance_block(self, res):
        p = res["provenance"]
        assert "loa0_product" in p and "molly_tsv" in p
        assert p["z_shape"] == "DIAGONAL-IN-Z (C per SNR-stratum only; per-z "\
            "closure residual MEASURES the missing z-resolution)"
        assert p["rho_used"] is False

    def test_self_transfer_residual_is_zero(self, res):
        # mock == calib -> R_mock / R_calib == 1 exactly, per tier
        tr = res["c2_comparison"]["fffp_transfer_residuals"]
        for t, v in tr.items():
            assert v == pytest.approx(0.0, abs=1e-12), t

    def test_json_serializable(self, res, tmp_path):
        out = tmp_path / "ff.json"
        with open(out, "w") as f:
            json.dump(res, f)
        assert out.exists()
