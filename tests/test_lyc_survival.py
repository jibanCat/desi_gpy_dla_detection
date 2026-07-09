"""tests/test_lyc_survival.py — unit tests for CDDF_analysis/lyc/survival.py.

Fast, synthetic-only (no data files). Cover the Nelson-Aalen / exposure incidence estimator:
  * geometry helpers (blue cutoff, proximity edge);
  * deterministic exposure/count arithmetic (exact ell = n_det / exposure);
  * constant-ell recovery under blocking (statistical, fixed seed);
  * blocking (only the highest-z break per sightline counts);
  * Nelson-Aalen == direct incidence on an independent-absorber population;
  * edge cases: R(z)=0 / empty bin, empty window, proximity exclusion.

Run:  pytest tests/test_lyc_survival.py -v
"""
import numpy as np
import pytest

from CDDF_analysis.lyc import survival as SV
from CDDF_analysis.lyc.opacity import LYMAN_LIMIT, C_KMS


# ------------------------------------------------------------------ #
# Geometry helpers
# ------------------------------------------------------------------ #
class TestGeometry:
    def test_blue_cutoff_desi(self):
        # 3600 A / 911.76 A - 1 = 2.948...
        assert SV.blue_cutoff_z(3600.0) == pytest.approx(3600.0 / LYMAN_LIMIT - 1.0)
        assert SV.blue_cutoff_z(3600.0) == pytest.approx(2.9483, abs=1e-3)

    def test_proximity_z_max(self):
        zq = 3.5
        expect = zq - 3000.0 * (1.0 + zq) / C_KMS
        assert SV.proximity_z_max(zq, 3000.0) == pytest.approx(expect)
        # zero proximity -> exactly z_qso
        assert SV.proximity_z_max(zq, 0.0) == pytest.approx(zq)
        # vectorized + monotone below z_qso
        arr = SV.proximity_z_max(np.array([3.0, 3.5, 4.0]), 3000.0)
        assert np.all(arr < np.array([3.0, 3.5, 4.0]))


# ------------------------------------------------------------------ #
# Deterministic exposure / count arithmetic
# ------------------------------------------------------------------ #
class TestDeterministicArithmetic:
    def test_exact_ratio(self):
        # sightline 0: break at 3.1 (window high edge 3.4); sightline 1: censored (no break)
        z_detect = np.array([3.1, np.nan])
        z_start = np.array([3.1, 3.0])   # break -> z_detect ; censored -> cutoff
        z_stop = np.array([3.4, 3.4])
        edges = np.array([3.0, 3.2, 3.4])
        r = SV.ell_nelson_aalen(z_detect, z_start, z_stop, edges, n_boot=0)
        # bin0 [3.0,3.2]: exposure = (3.2-3.1) + (3.2-3.0) = 0.1 + 0.2 = 0.3 ; n_det = 1
        # bin1 [3.2,3.4]: exposure = (3.4-3.2) + (3.4-3.2) = 0.2 + 0.2 = 0.4 ; n_det = 0
        assert r["n_det"].tolist() == [1, 0]
        assert r["exposure"] == pytest.approx([0.3, 0.4])
        assert r["ell"][0] == pytest.approx(1.0 / 0.3)
        assert r["ell"][1] == pytest.approx(0.0)
        # n_risk = exposure / bin width (mean # at risk)
        assert r["n_risk"] == pytest.approx([0.3 / 0.2, 0.4 / 0.2])

    def test_per_dz_to_dX_roundtrip(self):
        from CDDF_analysis.cddf_mock import path_length_int
        z = np.array([3.0, 3.3])
        ell_dz = np.array([2.0, 2.0])
        ell_dX = SV.ell_per_dz_to_dX(ell_dz, z, Omega_m=0.279)
        assert ell_dX == pytest.approx(ell_dz / path_length_int(z, 0.279))
        # dX/dz > 1 in this range => per-dX incidence is smaller than per-dz
        assert np.all(ell_dX < ell_dz)


# ------------------------------------------------------------------ #
# Blocking: only the highest-z observable break counts
# ------------------------------------------------------------------ #
class TestBlocking:
    def test_build_census_takes_highest_z(self):
        # one sightline, three tau>=2 absorbers; only the highest-z (3.30) is the first break
        cutoff = SV.blue_cutoff_z(3600.0)
        zq = {1000: 3.6}
        sl = np.array([1000, 1000, 1000])
        z = np.array([3.05, 3.20, 3.30])
        c = SV.build_break_census(sl, z, zq, cutoff, proximity_dv_kms=3000.0)
        assert c["z_detect"][0] == pytest.approx(3.30)
        assert c["z_start"][0] == pytest.approx(3.30)   # at-risk stops at the break
        assert bool(c["has_break"][0]) is True
        # exactly the 3 absorbers are observable (all inside [cutoff, prox(3.6)])
        assert int(c["obs_mask"].sum()) == 3

    def test_censored_sightline_has_no_break(self):
        cutoff = SV.blue_cutoff_z(3600.0)
        zq = {1: 3.6, 2: 3.6}
        # sightline 1 has an absorber, sightline 2 has none
        sl = np.array([1])
        z = np.array([3.1])
        c = SV.build_break_census(sl, z, zq, cutoff)
        # returned rows are sorted unique sightlines of z_qso -> [1, 2]
        assert c["sl"].tolist() == [1, 2]
        assert np.isfinite(c["z_detect"][0]) and np.isnan(c["z_detect"][1])
        assert c["z_start"][1] == pytest.approx(cutoff)   # censored -> starts at cutoff


# ------------------------------------------------------------------ #
# Edge cases
# ------------------------------------------------------------------ #
class TestEdgeCases:
    def test_empty_bin_zero_exposure_is_nan(self):
        # windows [3.2,3.4]; a bin [3.0,3.1] gets zero exposure -> ell NaN, no crash
        z_detect = np.array([3.3])
        z_start = np.array([3.3])
        z_stop = np.array([3.4])
        edges = np.array([3.0, 3.1, 3.2, 3.4])
        r = SV.ell_nelson_aalen(z_detect, z_start, z_stop, edges, n_boot=0)
        assert np.isnan(r["ell"][0]) and np.isnan(r["ell"][1])   # both below the window
        assert r["exposure"][0] == 0.0 and r["exposure"][1] == 0.0
        assert np.isfinite(r["ell"][2])

    def test_R_of_z_zero_everywhere(self):
        # all windows empty (z_start >= z_stop) -> all NaN, no crash
        z_detect = np.array([np.nan, np.nan])
        z_start = np.array([3.5, 3.6])
        z_stop = np.array([3.4, 3.5])   # start >= stop => empty
        edges = np.array([3.0, 3.2, 3.4])
        r = SV.ell_nelson_aalen(z_detect, z_start, z_stop, edges, n_boot=0)
        assert np.all(np.isnan(r["ell"]))
        assert np.all(r["exposure"] == 0.0)

    def test_empty_window_contributes_nothing(self):
        # one good sightline + one empty-window sightline -> good one still measured
        z_detect = np.array([3.3, np.nan])
        z_start = np.array([3.3, 3.9])
        z_stop = np.array([3.4, 3.8])   # second window empty
        edges = np.array([3.2, 3.4])
        r = SV.ell_nelson_aalen(z_detect, z_start, z_stop, edges, n_boot=0)
        assert r["n_det"][0] == 1
        assert r["exposure"][0] == pytest.approx(0.1)   # only the first sightline
        assert r["ell"][0] == pytest.approx(1.0 / 0.1)

    def test_proximity_excludes_absorber(self):
        # absorber inside the proximity zone (above z_stop) must NOT be counted as a break
        cutoff = SV.blue_cutoff_z(3600.0)
        zq = {7: 3.3}
        z_stop = SV.proximity_z_max(3.3, 3000.0)   # ~3.257
        sl = np.array([7, 7])
        z = np.array([3.10, z_stop + 0.02])        # one good, one inside proximity zone
        c = SV.build_break_census(sl, z, zq, cutoff, proximity_dv_kms=3000.0)
        assert c["z_detect"][0] == pytest.approx(3.10)   # NOT the proximity-zone one
        assert int(c["obs_mask"].sum()) == 1

    def test_detection_outside_window_is_dropped(self):
        # a raw detection above z_stop is inconsistent -> dropped (warns), not counted
        z_detect = np.array([3.9])
        z_start = np.array([3.9])
        z_stop = np.array([3.4])
        edges = np.array([3.0, 3.4])
        r = SV.ell_nelson_aalen(z_detect, z_start, z_stop, edges, n_boot=0)
        assert r["n_det"][0] == 0

    def test_bad_edges_raise(self):
        with pytest.raises(ValueError):
            SV.ell_nelson_aalen([3.1], [3.1], [3.4], [3.4, 3.0], n_boot=0)  # non-increasing


# ------------------------------------------------------------------ #
# Statistical: constant-ell recovery under blocking, and NA == direct
# ------------------------------------------------------------------ #
def _simulate_poisson_sightlines(n_sl, ell_true, win_lo, win_hi, rng):
    """Poisson absorbers (rate ell_true per unit z) on identical windows [win_lo, win_hi].

    Returns per-absorber (sl_row, z_abs), full windows, and the blocked first-break census.
    """
    L = win_hi - win_lo
    counts = rng.poisson(ell_true * L, size=n_sl)
    abs_rows, abs_z = [], []
    z_detect = np.full(n_sl, np.nan)
    for i in range(n_sl):
        if counts[i] > 0:
            zz = win_lo + L * rng.random(counts[i])
            abs_rows.append(np.full(counts[i], i))
            abs_z.append(zz)
            z_detect[i] = zz.max()   # highest-z = first break
    abs_rows = np.concatenate(abs_rows) if abs_rows else np.array([], int)
    abs_z = np.concatenate(abs_z) if abs_z else np.array([], float)
    z_start_na = np.where(np.isfinite(z_detect), z_detect, win_lo)
    z_stop = np.full(n_sl, win_hi)
    z_start_full = np.full(n_sl, win_lo)
    return abs_rows, abs_z, z_detect, z_start_na, z_start_full, z_stop


class TestStatisticalRecovery:
    def test_constant_ell_recovered(self):
        rng = np.random.default_rng(20260709)
        ell_true, lo, hi, n_sl = 1.5, 2.95, 3.75, 40000
        (_, _, z_detect, z_start_na, _, z_stop) = _simulate_poisson_sightlines(
            n_sl, ell_true, lo, hi, rng)
        edges = np.arange(lo, hi + 1e-9, 0.1)
        r = SV.ell_nelson_aalen(z_detect, z_start_na, z_stop, edges, n_boot=200, seed=1)
        # pooled (exposure-weighted) incidence recovers ell_true tightly
        pooled = r["n_det"].sum() / r["exposure"].sum()
        assert pooled == pytest.approx(ell_true, rel=0.05)
        # every bin within ~3 bootstrap sigma of truth (blocking-corrected, no z-trend)
        ok = np.abs(r["ell"] - ell_true) <= 3.5 * r["ell_err"] + 0.02
        assert np.all(ok[np.isfinite(r["ell"])])

    def test_nelson_aalen_equals_direct(self):
        rng = np.random.default_rng(11)
        ell_true, lo, hi, n_sl = 2.0, 2.95, 3.75, 40000
        (abs_rows, abs_z, z_detect, z_start_na,
         z_start_full, z_stop) = _simulate_poisson_sightlines(n_sl, ell_true, lo, hi, rng)
        edges = np.arange(lo, hi + 1e-9, 0.1)
        na = SV.ell_nelson_aalen(z_detect, z_start_na, z_stop, edges, n_boot=0)
        di = SV.ell_direct_incidence(abs_z, abs_rows, z_start_full, z_stop, edges, n_boot=0)
        # pooled incidences agree to a few percent (both estimate the same intensity)
        pooled_na = na["n_det"].sum() / na["exposure"].sum()
        pooled_di = di["n_det"].sum() / di["exposure"].sum()
        assert pooled_na == pytest.approx(pooled_di, rel=0.03)
        assert pooled_di == pytest.approx(ell_true, rel=0.05)


class TestZVaryingEllGuardsAgainstThePlugIn:
    """The regression test that guards WHY this module exists.

    Every other statistical test here uses a CONSTANT ell. A model-based `g_i = P_clear(z; Lambda)`
    fixed-point estimator -- the biased form `survival.py` deliberately does NOT implement --
    recovers a constant ell correctly when Lambda is correctly specified, so a silent regression to
    it would pass all of them. Only a z-VARYING truth separates the two: the model-free
    Nelson-Aalen estimator tracks ell(z) per bin, while a plug-in that assumes the wrong shape
    biases (-15%/+18% at Lambda scaled by 0.6/1.5, per the independent oracle).
    """

    @staticmethod
    def _sim_rising(n_sl, rng, lo=2.95, hi=3.75):
        """Exact first-event simulation for ell(z) = A*(1+z)^gamma (no grid discretization)."""
        A, gamma = 0.35, 5.2
        ell = lambda z: A * (1.0 + z) ** gamma                     # noqa: E731
        # cumulative Lambda(z) from lo, inverted by fine interpolation
        zg = np.linspace(lo, hi, 20001)
        Lam = np.concatenate([[0.0], np.cumsum(0.5 * (ell(zg[1:]) + ell(zg[:-1])) * np.diff(zg))])
        z_stop = rng.uniform(lo + 0.25, hi, n_sl)                  # heterogeneous windows
        Lstop = np.interp(z_stop, zg, Lam)
        # highest-z event: survival from z_stop downward, exponential in Lambda
        u = rng.exponential(1.0, n_sl)
        Ldet = Lstop - u
        hit = Ldet > 0.0
        z_det = np.where(hit, np.interp(np.clip(Ldet, 0, None), Lam, zg), np.nan)
        z_start = np.where(hit, z_det, lo)
        return z_det, z_start, z_stop, ell

    def test_nelson_aalen_tracks_a_rising_ell(self):
        rng = np.random.default_rng(31415)
        z_det, z_start, z_stop, ell = self._sim_rising(120000, rng)
        edges = np.arange(3.2, 3.75 + 1e-9, 0.1)                   # bins with healthy exposure
        out = SV.ell_nelson_aalen(z_det, z_start, z_stop, edges, n_boot=0)
        assert out["n_dropped"] == 0
        truth = ell(out["z_mid"])
        ok = np.isfinite(out["ell"]) & (out["n_det"] > 200)
        assert ok.sum() >= 4, "need several well-populated bins"
        ratio = out["ell"][ok] / truth[ok]
        # per-bin within 6% (sampling noise at these counts), and no systematic trend with z
        assert np.all(np.abs(ratio - 1.0) < 0.06), f"per-bin ratios {ratio}"
        assert abs(np.mean(ratio) - 1.0) < 0.02, f"mean ratio {np.mean(ratio)}"
        # the estimator must TRACK the rise, not return a constant: truth rises >40% across the range
        assert truth[ok][-1] / truth[ok][0] > 1.4
        assert out["ell"][ok][-1] / out["ell"][ok][0] > 1.3

    def test_model_based_plugin_would_be_biased_here(self):
        """Demonstrate the failure mode this module rejects: a P_clear(Lambda)-weighted denominator
        with a MISSPECIFIED Lambda biases ell, while the observed-at-risk denominator does not."""
        rng = np.random.default_rng(2718)
        z_det, z_start, z_stop, ell = self._sim_rising(120000, rng)
        edges = np.arange(3.2, 3.75 + 1e-9, 0.1)
        zmid = 0.5 * (edges[:-1] + edges[1:])
        free = SV.ell_nelson_aalen(z_det, z_start, z_stop, edges, n_boot=0)

        def plugin(scale):
            """exposure_b = sum_i 1{observable} * P_clear,i(z; scale*ell) -- the fixed-point form."""
            zg = np.linspace(edges[0], edges[-1], 2001)
            lam = scale * ell(zg)
            expo = np.zeros(zmid.size)
            for i in range(zmid.size):
                zc = zmid[i]
                obs = (z_stop > zc)
                # Lambda(zc -> z_stop_i) under the ASSUMED intensity
                Lam = np.array([np.trapezoid(scale * ell(np.linspace(zc, zs, 64)),
                                             np.linspace(zc, zs, 64)) for zs in z_stop[obs]])
                expo[i] = np.exp(-Lam).sum() * np.diff(edges)[i]
            return free["n_det"] / np.maximum(expo, 1e-12)

        truth = ell(zmid)
        r_free = np.nanmean(free["ell"] / truth)
        r_lo = np.nanmean(plugin(0.6) / truth)
        r_hi = np.nanmean(plugin(1.5) / truth)
        assert abs(r_free - 1.0) < 0.03, f"model-free must be unbiased, got {r_free}"
        assert r_lo < 0.95, f"misspecified-low plug-in should bias low, got {r_lo}"
        assert r_hi > 1.05, f"misspecified-high plug-in should bias high, got {r_hi}"
