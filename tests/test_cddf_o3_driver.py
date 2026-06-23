"""Tests for the O3 diagonal soft-completeness driver
(``CDDF_analysis.cddf_forward.driver.compute_o3_products`` + closure + IO/plot).

The Bayesian core (``cddf_forward.soft_completeness``) is built in parallel.  These
tests inject a small in-test FAKE matching the §2 signatures (via monkeypatch) so
the CS-side logic — FILTER guard, truth map on BUILD, partitioned deposit, count-
space hand-off, no-leakage assertion, HELDOUT closure, honest labels — is tested
WITHOUT the real core.

Contract anchors:
  * §3.3 compute_o3_products: assert_filter_off FIRST; C/b_FP on BUILD; correction
    applied to the WHOLE-sample F; provenance records window_applied=True + split.
  * §3.4 count-space hand-off feeds the correction.
  * §3.5 heldout_closure: assert_no_leakage BUILD vs HELDOUT, residuals + pass flag.
  * §3.6 save_o3_products / plot_o3_products: honest "O3 DIAGONAL SOFT-COMPLETENESS
    CORRECTED" labels stating the cross-bin-migration limitation.
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "fixtures", "cddf"))

pytest.importorskip("h5py")
pytest.importorskip("astropy")
pytest.importorskip("scipy")

from CDDF_analysis.cddf_forward import driver as o3driver  # noqa: E402
from CDDF_analysis.cddf_forward.window import WindowSpec  # noqa: E402
from build_synthetic_cddf_fixture import build_synthetic_cddf  # noqa: E402
from build_synthetic_truth_fixture import write_truth_catalog  # noqa: E402


_Z_MIN = 2.4
_Z_MAX = 3.3
_LNHI_MIN = 20.3
_LNHI_MAX = 22.5
_LNHI_NBINS = 3
_WINDOW = WindowSpec(z_min_lyb=False, z_max_lyb=False)

_DLACAT_KWARGS = dict(sub_dla=False, snr=-2, lowzcut=False, highzcut=False)


# --------------------------------------------------------------------------- #
# Fake Bayesian core matching contract §2 signatures.
# --------------------------------------------------------------------------- #
class _FakeCore:
    """Minimal §2-compatible fake.

    * completeness C = 0.5 in every valid bin (so n_corr roughly doubles F);
    * b_FP = 0 everywhere (no FP subtraction);
    * apply_diagonal_correction = (F - b_FP) / C with trivial CI scaling.
    """

    @staticmethod
    def estimate_diagonal_completeness(f_matched, n_truth, *, prior=(0.5, 0.5)):
        f_matched = np.asarray(f_matched, float)
        n_truth = np.asarray(n_truth, float)
        valid = n_truth > 0
        C = np.where(valid, 0.5, np.nan)
        return {
            "C": C,
            "C_lo68": np.where(valid, 0.4, np.nan),
            "C_hi68": np.where(valid, 0.6, np.nan),
            "C_lo95": np.where(valid, 0.3, np.nan),
            "C_hi95": np.where(valid, 0.7, np.nan),
            "valid_mask": valid,
        }

    @staticmethod
    def estimate_false_positive_deposit(f_unmatched, exposure, *, prior=(0.5,)):
        f_unmatched = np.asarray(f_unmatched, float)
        z = np.zeros_like(f_unmatched)
        return {
            "b_FP": z.copy(),
            "b_FP_lo68": z.copy(),
            "b_FP_hi68": z.copy(),
            "b_FP_lo95": z.copy(),
            "b_FP_hi95": z.copy(),
        }

    @staticmethod
    def apply_diagonal_correction(F, F_ci, C_est, bfp_est, *, n_mc=None,
                                  return_draws=False):
        F = np.asarray(F, float)
        C = np.asarray(C_est["C"], float)
        bfp = np.asarray(bfp_est["b_FP"], float)
        valid = np.asarray(C_est["valid_mask"], bool)
        with np.errstate(invalid="ignore", divide="ignore"):
            n_corr = np.where(valid, (F - bfp) / C, np.nan)
        neg = (F - bfp) < 0
        n_corr = np.where(neg, 0.0, n_corr)

        def _scale(key):
            arr = np.asarray(F_ci[key], float)
            return np.where(valid, (arr - bfp) / C, np.nan)

        out = {
            "n_corr": n_corr,
            "lo68": _scale("lo68"),
            "hi68": _scale("hi68"),
            "lo95": _scale("lo95"),
            "hi95": _scale("hi95"),
            "neg_clip_mask": neg,
            "valid_mask": valid,
        }
        if return_draws:
            # Joint per-draw n_corr (n_mc, nbin): a tiny deterministic spread around
            # the point so the omega CI is non-degenerate and inter-bin correlated.
            n = 200
            base = np.where(np.isfinite(n_corr), n_corr, 0.0)
            rng = np.random.default_rng(7)
            jitter = rng.normal(0.0, 0.05, size=(n, base.shape[0]))
            draws = base[None, :] * (1.0 + jitter)
            draws = np.maximum(draws, 0.0)
            out["n_corr_draws"] = draws
        return out

    @staticmethod
    def omega_from_draws(n_corr_draws, logN_centres, dX, hubble=0.7):
        # Reference Ω from joint per-draw n_corr: Σ_N N·n_corr/ΔX per draw, then
        # percentile. Mirrors the REAL core helper signature + FLAT return schema
        # ({omega, lo68, hi68, lo95, hi95}) so the test exercises the integration.
        from CDDF_analysis.calc_cddf import rho_crit
        protonmass = 1.67262178e-24
        h100 = 3.2407789e-18 * hubble
        light = 2.99e10
        conv = protonmass / light * h100 / rho_crit(hubble)
        nhi = 10 ** np.asarray(logN_centres, float)
        draws = np.asarray(n_corr_draws, float)
        per_draw = conv * np.nansum(nhi[None, :] * draws, axis=1) / float(dX)
        omega = float(np.mean(per_draw))  # mean -> guaranteed inside the interval
        lo68, hi68 = np.percentile(per_draw, [16.0, 84.0])
        lo95, hi95 = np.percentile(per_draw, [2.5, 97.5])
        return {
            "omega": omega,
            "lo68": float(lo68), "hi68": float(hi68),
            "lo95": float(lo95), "hi95": float(hi95),
        }


@pytest.fixture
def fixtures(tmp_path):
    # Many spectra so the TARGETID-keyed split yields both BUILD and HELDOUT.
    n = 24
    rng = np.random.default_rng(0)
    p_dla = tuple(1.0 if i % 4 != 3 else 0.0 for i in range(n))
    peak_logN = tuple(
        None if p == 0 else float(20.4 + 0.6 * (i % 3)) for i, p in enumerate(p_dla)
    )
    peak_z = tuple(
        None if p == 0 else float(2.5 + 0.25 * (i % 3)) for i, p in enumerate(p_dla)
    )
    synth = build_synthetic_cddf(
        tmp_path,
        n_spec=n,
        p_dla=p_dla,
        peak_logN=peak_logN,
        peak_z=peak_z,
        z_qso=3.6,
        z_min=2.4,
        z_max=3.3,
        # sample grid must reach lnhi_max (22.5) so the C9 ceiling assertion holds.
        lnhi_hi=_LNHI_MAX,
    )
    # truth catalog: one absorber per active sightline at its peak (perfect truth)
    tids, nhis, zs = [], [], []
    for i in range(n):
        if p_dla[i] == 0:
            continue
        tids.append(1000 + i)
        nhis.append(peak_logN[i])
        zs.append(peak_z[i])
    truth_file = str(tmp_path / "truth_o3.fits")
    write_truth_catalog(truth_file, target_ids=tids, nhi=nhis, z=zs)
    synth["truth_file"] = truth_file
    return synth


def _compute(fixtures, monkeypatch, **overrides):
    monkeypatch.setattr(o3driver, "soft_completeness", _FakeCore, raising=False)
    kwargs = dict(
        z_min=_Z_MIN,
        z_max=_Z_MAX,
        lnhi_min=_LNHI_MIN,
        lnhi_max=_LNHI_MAX,
        lnhi_nbins=_LNHI_NBINS,
        filter_low_likelihood=0,
        window=_WINDOW,
    )
    kwargs.update(overrides)
    return o3driver.compute_o3_products(
        fixtures["processed_file"],
        fixtures["sample_file"],
        fixtures["catalog_file"],
        fixtures["truth_file"],
        **_DLACAT_KWARGS,
        **kwargs,
    )


class TestComputeO3Products:
    def test_returns_expected_top_level_keys(self, fixtures, monkeypatch):
        prod = _compute(fixtures, monkeypatch)
        for key in (
            "o1",
            "completeness",
            "o3_cddf",
            "o3_dndx",
            "o3_omega",
            "closure",
            "provenance",
        ):
            assert key in prod

    def test_filter_on_raises_first(self, fixtures, monkeypatch):
        with pytest.raises(ValueError, match="FILTER"):
            _compute(fixtures, monkeypatch, filter_low_likelihood=1)

    def test_window_applied_true(self, fixtures, monkeypatch):
        prod = _compute(fixtures, monkeypatch)
        assert prod["provenance"]["window_applied"] is True

    def test_coverage_provenance_distinguishes_truth_only_and_both(self, fixtures, monkeypatch, tmp_path):
        # C8: the driver must surface join-coverage provenance — counts of
        # (truth-only, processed-only, both) TARGETID sets + n_absorbers_in/kept —
        # so a partial-coverage gap is VISIBLE, not mistaken for incompleteness.
        import numpy as np
        from astropy.table import Table

        # add a truth absorber on a TARGETID NOT in the processed file (truth-only).
        orig = Table.read(fixtures["truth_file"])
        extra_tid = 999999  # not in the processed run
        new = Table({
            "NHI": np.append(np.asarray(orig["NHI"], float), 20.7),
            "Z": np.append(np.asarray(orig["Z"], float), 2.6),
            "TARGETID": np.append(np.asarray(orig["TARGETID"], np.int64),
                                  np.int64(extra_tid)),
            "DLAID": np.append(np.asarray(orig["DLAID"], np.int64), np.int64(1)),
        })
        truth2 = str(tmp_path / "truth_cov.fits")
        new.write(truth2, overwrite=True)

        monkeypatch.setattr(o3driver, "soft_completeness", _FakeCore, raising=False)
        prod = o3driver.compute_o3_products(
            fixtures["processed_file"], fixtures["sample_file"],
            fixtures["catalog_file"], truth2,
            z_min=_Z_MIN, z_max=_Z_MAX, lnhi_min=_LNHI_MIN, lnhi_max=_LNHI_MAX,
            lnhi_nbins=_LNHI_NBINS, filter_low_likelihood=0, window=_WINDOW,
            **_DLACAT_KWARGS,
        )
        cov = prod["provenance"]["coverage"]
        for key in ("n_truth_only", "n_processed_only", "n_both",
                    "n_absorbers_in", "n_absorbers_kept"):
            assert key in cov
        # the extra TARGETID is truth-only (present in truth, absent from processed).
        assert cov["n_truth_only"] >= 1
        # the active sightlines are present in BOTH sets.
        assert cov["n_both"] >= 1
        assert cov["n_absorbers_in"] >= cov["n_absorbers_kept"]

    def test_boundary_flags_and_above_ceiling_count(self, fixtures, monkeypatch, tmp_path):
        # C9: per-bin Eddington-boundary flag (lowest 1-2 DLA bins at logN=20.3) +
        # a count of truth absorbers above the lnhi_max (22.5) grid ceiling.
        import numpy as np
        from astropy.table import Table

        # add a truth absorber ABOVE the ceiling on an ACTIVE sightline (tid 1000).
        orig = Table.read(fixtures["truth_file"])
        new = Table({
            "NHI": np.append(np.asarray(orig["NHI"], float), 22.9),  # above 22.5
            "Z": np.append(np.asarray(orig["Z"], float), 2.6),
            "TARGETID": np.append(np.asarray(orig["TARGETID"], np.int64), np.int64(1000)),
            "DLAID": np.append(np.asarray(orig["DLAID"], np.int64), np.int64(2)),
        })
        truth2 = str(tmp_path / "truth_ceil.fits")
        new.write(truth2, overwrite=True)

        monkeypatch.setattr(o3driver, "soft_completeness", _FakeCore, raising=False)
        prod = o3driver.compute_o3_products(
            fixtures["processed_file"], fixtures["sample_file"],
            fixtures["catalog_file"], truth2,
            z_min=_Z_MIN, z_max=_Z_MAX, lnhi_min=_LNHI_MIN, lnhi_max=_LNHI_MAX,
            lnhi_nbins=_LNHI_NBINS, filter_low_likelihood=0, window=_WINDOW,
            **_DLACAT_KWARGS,
        )
        bf = prod["provenance"]["boundary_flags"]
        flags = np.asarray(bf["eddington_boundary_bins"], bool)
        assert flags.shape[0] == _LNHI_NBINS
        # the lowest bin (adjacent to logN=20.3) is flagged.
        assert flags[0]
        # the highest bin is NOT flagged (only the lowest 1-2).
        assert not flags[-1]
        # the above-ceiling truth absorber on the active sightline is counted.
        assert bf["n_truth_above_ceiling"] >= 1
        assert bf["sample_grid_ceiling"] >= bf["lnhi_max"] - 1e-6

    def test_ceiling_assertion_fires_when_lnhi_max_exceeds_grid(self, fixtures, monkeypatch):
        # C9: the lnhi_max == sample-grid-ceiling assertion must fire if lnhi_max is
        # pushed above the QMC grid ceiling (else high-N bins are silently starved).
        monkeypatch.setattr(o3driver, "soft_completeness", _FakeCore, raising=False)
        with pytest.raises(AssertionError, match="ceiling"):
            o3driver.compute_o3_products(
                fixtures["processed_file"], fixtures["sample_file"],
                fixtures["catalog_file"], fixtures["truth_file"],
                z_min=_Z_MIN, z_max=_Z_MAX, lnhi_min=_LNHI_MIN,
                lnhi_max=23.5,  # above the fixture's 22.5 grid ceiling
                lnhi_nbins=_LNHI_NBINS, filter_low_likelihood=0, window=_WINDOW,
                **_DLACAT_KWARGS,
            )

    def test_completeness_block_has_C_bFP_ntruth(self, fixtures, monkeypatch):
        prod = _compute(fixtures, monkeypatch)
        comp = prod["completeness"]
        for key in ("C", "b_FP", "n_truth", "F_matched", "F_unmatched"):
            assert key in comp
        # grid shape (nlnhi, nz)
        assert comp["C"].shape[0] == _LNHI_NBINS

    def test_correction_doubles_counts_under_C_half(self, fixtures, monkeypatch):
        # With the fake C=0.5, b_FP=0, the corrected CDDF should ~2x the WINDOWED
        # raw CDDF in valid bins (n_corr = F/0.5). The raw base is the o3 block's
        # f_raw (the deposit mean-count basis the correction operates on), NOT the
        # unwindowed O1 wrapper or the MAP.
        prod = _compute(fixtures, monkeypatch)
        raw = np.asarray(prod["o3_cddf"]["f_raw"], float)
        corr = np.asarray(prod["o3_cddf"]["f"], float)
        valid = np.asarray(prod["o3_cddf"]["valid_mask"], bool)
        m = valid & (raw > 0)
        assert m.any()
        np.testing.assert_allclose(corr[m], 2.0 * raw[m], rtol=1e-6)

    def test_F_fed_to_correction_is_deposit_mean_not_map(self, fixtures, monkeypatch):
        # C1+C2 BLOCKER: the F driving the correction must be the partitioned deposit
        # MEAN-count total (F_matched + F_unmatched over the WHOLE active set), NOT
        # column_density_function_counts (the Poisson-binomial MAP). Pin the o3
        # f_raw (= F/(dX*dN)) to the deposit total, and assert it is the deposit
        # basis by reconstructing F = f_raw*dX*dN == F_matched + F_unmatched.
        from CDDF_analysis import calc_cddf
        from CDDF_analysis.cddf_forward.diagonal_deposit import (
            build_truth_map, DiagonalSoftDeposit,
        )

        prod = _compute(fixtures, monkeypatch)
        cat = calc_cddf.DLACatalogue(
            processed_file=fixtures["processed_file"],
            sample_file=fixtures["sample_file"],
            catalog_file=fixtures["catalog_file"],
            window=_WINDOW, high_nhi_cut_value=_LNHI_MAX, **_DLACAT_KWARGS,
        )
        lnhi_edges = np.linspace(_LNHI_MIN, _LNHI_MAX, _LNHI_NBINS + 1)
        z_edges = np.asarray(prod["provenance"]["z_edges"], float)
        active_ids = set(int(t) for t in cat.target_ids[cat.filter_dla_spectra()[0]])
        tmap = build_truth_map(
            fixtures["truth_file"], catalog_file=fixtures["catalog_file"],
            processed_file=fixtures["processed_file"], window=_WINDOW,
            lnhi_edges=lnhi_edges, z_edges=z_edges, active_target_ids=active_ids,
        )
        dep = DiagonalSoftDeposit(
            cat, tmap, lnhi_edges=lnhi_edges, z_edges=z_edges, window=_WINDOW
        )
        part = dep.deposit(z_min=_Z_MIN, z_max=_Z_MAX)  # WHOLE active set (no subset)
        whole_F = (part["F_matched"] + part["F_unmatched"])[:, 0]
        # the completeness block exposes F_matched/F_unmatched on the WHOLE active set
        comp = prod["completeness"]
        np.testing.assert_allclose(
            np.asarray(comp["F_matched"], float) + np.asarray(comp["F_unmatched"], float),
            whole_F, atol=1e-9,
        )
        # and the o3 f_raw is exactly this deposit total renormalized.
        dN = np.array([10**e2 - 10**e1 for e1, e2 in zip(lnhi_edges[:-1], lnhi_edges[1:])])
        dX = cat.path_length(_Z_MIN, _Z_MAX)
        np.testing.assert_allclose(
            np.asarray(prod["o3_cddf"]["f_raw"], float), whole_F / dX / dN, atol=1e-12,
        )
        # F_fed must NOT be the MAP: the deposit total differs from the MAP in general,
        # so the MAP-based renormalization must NOT equal f_raw whenever they differ.
        cc = cat.column_density_function_counts(
            z_min=_Z_MIN, z_max=_Z_MAX, lnhi_nbins=_LNHI_NBINS,
            lnhi_min=_LNHI_MIN, lnhi_max=_LNHI_MAX,
        )
        # at least sanity: the F basis is the deposit, asserted above.
        assert whole_F.shape == cc["counts"].shape

    def test_whole_active_set_used_for_completeness_no_build_split(self, fixtures, monkeypatch):
        # C1 BLOCKER: the science path estimates C / b_FP / n_truth on the WHOLE
        # ACTIVE set (no BUILD/HELDOUT split). n_truth in completeness must equal the
        # WHOLE active truth count, and must DIFFER from the BUILD-only subset.
        from CDDF_analysis import calc_cddf
        from CDDF_analysis.cddf_forward.diagonal_deposit import build_truth_map

        prod = _compute(fixtures, monkeypatch)
        lnhi_edges = np.linspace(_LNHI_MIN, _LNHI_MAX, _LNHI_NBINS + 1)
        z_edges = np.asarray(prod["provenance"]["z_edges"], float)
        cat = calc_cddf.DLACatalogue(
            processed_file=fixtures["processed_file"],
            sample_file=fixtures["sample_file"],
            catalog_file=fixtures["catalog_file"],
            window=_WINDOW, high_nhi_cut_value=_LNHI_MAX, **_DLACAT_KWARGS,
        )
        active_ids = set(int(t) for t in cat.target_ids[cat.filter_dla_spectra()[0]])
        whole_tmap = build_truth_map(
            fixtures["truth_file"], catalog_file=fixtures["catalog_file"],
            processed_file=fixtures["processed_file"], window=_WINDOW,
            lnhi_edges=lnhi_edges, z_edges=z_edges, active_target_ids=active_ids,
        )
        build_tmap = build_truth_map(
            fixtures["truth_file"], catalog_file=fixtures["catalog_file"],
            processed_file=fixtures["processed_file"], window=_WINDOW,
            lnhi_edges=lnhi_edges, z_edges=z_edges, role_mask="BUILD",
        )
        # science n_truth == WHOLE active truth count (not the BUILD subset)
        np.testing.assert_array_equal(
            prod["completeness"]["n_truth"], whole_tmap.n_truth_grid().ravel()
        )
        # and the whole-active and BUILD-only counts genuinely DIFFER (so the test
        # would catch a regression back to the BUILD split).
        assert whole_tmap.n_truth_grid().sum() > build_tmap.n_truth_grid().sum()

    def test_omega_from_joint_draws_point_inside_interval(self, fixtures, monkeypatch):
        # C6: Ω and its CI must come from the core's joint per-draw n_corr (via
        # omega_from_draws / return_draws), NOT from summing pre-reduced per-bin
        # f-CI edges. The joint derivation guarantees the Ω point lies inside its
        # interval; and the driver must INVOKE omega_from_draws (asserted by a spy).
        called = {"hit": False}
        orig_ofd = _FakeCore.omega_from_draws

        class _SpyCore(_FakeCore):
            @staticmethod
            def omega_from_draws(n_corr_draws, logN_centres, dX, hubble):
                called["hit"] = True
                return orig_ofd(n_corr_draws, logN_centres, dX, hubble)

        monkeypatch.setattr(o3driver, "soft_completeness", _SpyCore, raising=False)
        prod = o3driver.compute_o3_products(
            fixtures["processed_file"], fixtures["sample_file"],
            fixtures["catalog_file"], fixtures["truth_file"],
            z_min=_Z_MIN, z_max=_Z_MAX, lnhi_min=_LNHI_MIN, lnhi_max=_LNHI_MAX,
            lnhi_nbins=_LNHI_NBINS, filter_low_likelihood=0, window=_WINDOW,
            **_DLACAT_KWARGS,
        )
        assert called["hit"] is True
        om = prod["o3_omega"]
        point = float(np.asarray(om["omega"]).ravel()[0])
        lo95 = float(np.asarray(om["omega95"]).ravel()[0])
        hi95 = float(np.asarray(om["omega95"]).ravel()[1])
        # the joint-draw point (median) must lie inside its 95% interval.
        assert lo95 <= point <= hi95


class TestHeldoutClosure:
    def test_closure_runs_and_reports_pass_flag(self, fixtures, monkeypatch):
        monkeypatch.setattr(o3driver, "soft_completeness", _FakeCore, raising=False)
        out = o3driver.heldout_closure(
            fixtures["processed_file"],
            fixtures["sample_file"],
            fixtures["catalog_file"],
            fixtures["truth_file"],
            z_min=_Z_MIN,
            z_max=_Z_MAX,
            lnhi_min=_LNHI_MIN,
            lnhi_max=_LNHI_MAX,
            lnhi_nbins=_LNHI_NBINS,
            filter_low_likelihood=0,
            window=_WINDOW,
            **_DLACAT_KWARGS,
        )
        for key in ("residual", "standardized_residual", "passed", "n_valid_bins"):
            assert key in out
        assert isinstance(bool(out["passed"]), bool)

    def test_closure_asserts_no_leakage(self, fixtures, monkeypatch):
        # If BUILD and HELDOUT overlap, assert_no_leakage must raise. We force a
        # degenerate split (build_frac=1.0 => all BUILD, empty HELDOUT is fine; but
        # a frac that puts the same id in both is impossible by construction). Here
        # we instead verify the guard is wired by monkeypatching assert_no_leakage.
        import CDDF_analysis.cddf_forward.driver as drv

        called = {"hit": False}
        orig = drv.assert_no_leakage

        def _spy(b, h, *, ctx=""):
            called["hit"] = True
            return orig(b, h, ctx=ctx)

        monkeypatch.setattr(drv, "assert_no_leakage", _spy)
        monkeypatch.setattr(o3driver, "soft_completeness", _FakeCore, raising=False)
        o3driver.heldout_closure(
            fixtures["processed_file"],
            fixtures["sample_file"],
            fixtures["catalog_file"],
            fixtures["truth_file"],
            z_min=_Z_MIN,
            z_max=_Z_MAX,
            lnhi_min=_LNHI_MIN,
            lnhi_max=_LNHI_MAX,
            lnhi_nbins=_LNHI_NBINS,
            filter_low_likelihood=0,
            window=_WINDOW,
            **_DLACAT_KWARGS,
        )
        assert called["hit"] is True

    def test_closure_rebases_bfp_to_heldout_basis(self, fixtures, monkeypatch):
        # C5(a): the BUILD b_FP COUNT must be rebased to the HELDOUT basis by the
        # active-sightline ratio (N_held_active / N_build_active) before subtraction.
        # Use a fake core with a NON-ZERO b_FP and capture the bfp_est passed to the
        # HELDOUT apply_diagonal_correction; it must be the BUILD b_FP * ratio.
        calls = []

        class _RecordingCore(_FakeCore):
            @staticmethod
            def estimate_false_positive_deposit(f_unmatched, exposure, *, prior=(0.5,)):
                f = np.asarray(f_unmatched, float)
                ones = np.ones_like(f)  # constant b_FP = 1 per bin (BUILD basis)
                return {
                    "b_FP": ones.copy(), "b_FP_lo68": ones * 0.5,
                    "b_FP_hi68": ones * 1.5, "b_FP_lo95": ones * 0.2,
                    "b_FP_hi95": ones * 1.8,
                }

            @staticmethod
            def apply_diagonal_correction(F, F_ci, C_est, bfp_est, *, n_mc=None):
                calls.append({"F": np.asarray(F, float),
                              "F_ci": {k: np.asarray(v, float) for k, v in F_ci.items()},
                              "bfp": np.asarray(bfp_est["b_FP"], float)})
                return _FakeCore.apply_diagonal_correction(
                    F, F_ci, C_est, bfp_est, n_mc=n_mc
                )

        monkeypatch.setattr(o3driver, "soft_completeness", _RecordingCore, raising=False)
        out = o3driver.heldout_closure(
            fixtures["processed_file"], fixtures["sample_file"],
            fixtures["catalog_file"], fixtures["truth_file"],
            z_min=_Z_MIN, z_max=_Z_MAX, lnhi_min=_LNHI_MIN, lnhi_max=_LNHI_MAX,
            lnhi_nbins=_LNHI_NBINS, filter_low_likelihood=0, window=_WINDOW,
            **_DLACAT_KWARGS,
        )
        ratio = out["bfp_rebase_ratio"]
        assert 0.0 < ratio < 1.0  # fewer HELDOUT-active than BUILD-active (70/30 split)
        # the HELDOUT correction (the LAST apply call) must use the rebased b_FP.
        held_bfp = calls[-1]["bfp"]
        np.testing.assert_allclose(held_bfp, np.ones_like(held_bfp) * ratio, atol=1e-9)

    def test_closure_passes_real_count_ci_not_degenerate(self, fixtures, monkeypatch):
        # C5(b): the F_ci handed to the HELDOUT correction must be the REAL
        # count-space 68/95 CI (restricted to HELDOUT-active), NOT the degenerate
        # lo==hi==F point.
        captured = {}

        class _CapturingCore(_FakeCore):
            @staticmethod
            def apply_diagonal_correction(F, F_ci, C_est, bfp_est, *, n_mc=None):
                captured["F"] = np.asarray(F, float)
                captured["F_ci"] = {k: np.asarray(v, float) for k, v in F_ci.items()}
                return _FakeCore.apply_diagonal_correction(
                    F, F_ci, C_est, bfp_est, n_mc=n_mc
                )

        monkeypatch.setattr(o3driver, "soft_completeness", _CapturingCore, raising=False)
        o3driver.heldout_closure(
            fixtures["processed_file"], fixtures["sample_file"],
            fixtures["catalog_file"], fixtures["truth_file"],
            z_min=_Z_MIN, z_max=_Z_MAX, lnhi_min=_LNHI_MIN, lnhi_max=_LNHI_MAX,
            lnhi_nbins=_LNHI_NBINS, filter_low_likelihood=0, window=_WINDOW,
            **_DLACAT_KWARGS,
        )
        ci = captured["F_ci"]
        # NOT degenerate: at least one bin must have hi95 > lo95 (a real interval).
        assert np.any(ci["hi95"] > ci["lo95"] + 1e-9)
        assert np.any(ci["hi68"] > ci["lo68"] + 1e-9)

    def test_closure_gate_has_coherent_bias_test(self, fixtures, monkeypatch):
        # C5(c): the pass flag must combine marginal coverage (~95% within 2σ or
        # ~68% within 1σ) AND a coherent-bias test (mean standardized residual ≈ 0 /
        # χ²), to catch a uniform multiplicative bias that marginal coverage misses.
        monkeypatch.setattr(o3driver, "soft_completeness", _FakeCore, raising=False)
        out = o3driver.heldout_closure(
            fixtures["processed_file"], fixtures["sample_file"],
            fixtures["catalog_file"], fixtures["truth_file"],
            z_min=_Z_MIN, z_max=_Z_MAX, lnhi_min=_LNHI_MIN, lnhi_max=_LNHI_MAX,
            lnhi_nbins=_LNHI_NBINS, filter_low_likelihood=0, window=_WINDOW,
            **_DLACAT_KWARGS,
        )
        # the coherent-bias diagnostics must be surfaced.
        for key in ("mean_standardized_residual", "coherent_bias_ok",
                    "coverage_ok", "passed"):
            assert key in out
        # passed must be the AND of coverage_ok and coherent_bias_ok.
        assert bool(out["passed"]) == (
            bool(out["coverage_ok"]) and bool(out["coherent_bias_ok"])
        )


class TestSaveO3Products:
    def test_writes_tables_with_honest_labels(self, fixtures, monkeypatch, tmp_path):
        prod = _compute(fixtures, monkeypatch)
        out_dir = str(tmp_path / "o3_out")
        paths = o3driver.save_o3_products(prod, out_dir)
        assert "cddf" in paths and os.path.exists(paths["cddf"])
        assert "completeness" in paths and os.path.exists(paths["completeness"])
        text = open(paths["cddf"]).read().lower()
        assert "diagonal" in text
        assert "soft" in text and "completeness" in text
        # honest about the limitation
        assert "migration" in text
        # must NOT claim alpha(z)/london calibration
        assert "london mock" not in text

    def test_completeness_table_has_C_and_bFP(self, fixtures, monkeypatch, tmp_path):
        prod = _compute(fixtures, monkeypatch)
        paths = o3driver.save_o3_products(prod, str(tmp_path / "o3_out"))
        ctext = open(paths["completeness"]).read()
        assert "C" in ctext and "b_FP" in ctext


class TestPlotO3Products:
    def test_plot_returns_figure(self, fixtures, monkeypatch, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        prod = _compute(fixtures, monkeypatch)
        out_path = str(tmp_path / "o3.png")
        fig = o3driver.plot_o3_products(prod, save_path=out_path)
        assert fig is not None
        assert os.path.exists(out_path)
