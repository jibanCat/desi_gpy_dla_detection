# -*- coding: utf-8 -*-
"""Tests for the ADOPTED reporting configuration (PI decisions 1, 3, 4, 8).

Every test here is MUTATION-TESTED: the mutation that makes it go red is named
in the test's own docstring, so a reader can revert the fix and check.

MOCK / SYNTHETIC ONLY.  Nothing here reads survey data.
"""
import dataclasses
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from CDDF_analysis.hbi_mcmc import reporting as RP           # noqa: E402
from CDDF_analysis.hbi_mcmc import pack as PK                # noqa: E402


def _extract_pack_module():
    """extract_pack.py loaded file-directly (its own jax-free contract)."""
    p = os.path.join(REPO, "CDDF_analysis/hbi_mcmc/extract_pack.py")
    spec = importlib.util.spec_from_file_location("_t_adopted_extract", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


EP = _extract_pack_module()


# ===========================================================================
# DECISION 1 — the reporting window
# ===========================================================================

def test_nonident_edge_matches_the_subdla_runner_on_the_guard_branch():
    """The 19.7 floor is REUSED, not re-invented.

    The guard layer (branch ``lls-subdla-cddf``) is not merged here -- that merge
    is PI-deferred -- so the constant cannot be imported.  This test reads the
    guard branch's own file through ``git show`` and asserts the two agree.

    MUTATION: change ``reporting.NONIDENT_EDGE`` to 19.6 -> RED (AssertionError
    naming both values).  Verified.
    """
    try:
        src = subprocess.check_output(
            ["git", "show",
             "lls-subdla-cddf:CDDF_analysis/diagnostics/subdla/"
             "run_subdla_headline_full.py"],
            cwd=REPO, text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        pytest.skip("branch lls-subdla-cddf not reachable from this worktree")
    line = [l for l in src.splitlines()
            if l.startswith("NONIDENT_EDGE")]
    assert len(line) == 1, f"expected exactly one NONIDENT_EDGE line, got {line}"
    guard_value = float(line[0].split("=")[1].strip())
    assert RP.NONIDENT_EDGE == guard_value, (
        f"reporting.NONIDENT_EDGE = {RP.NONIDENT_EDGE} but the sub-DLA runner on "
        f"lls-subdla-cddf says {guard_value}; the two MUST be the same constant")
    assert "lls-subdla-cddf" in RP.NONIDENT_EDGE_SOURCE


def test_reporting_window_is_19p7_to_21p6_closed_on_both_ends():
    """MUTATION: set RESPONSE_ANCHOR_CEILING = 22.4 -> RED. Verified."""
    assert RP.REPORTING_WINDOW == (19.7, 21.6)
    assert RP.NONIDENT_EDGE == 19.7
    assert RP.RESPONSE_ANCHOR_CEILING == 21.6
    # the ceiling must carry its REASON (it is new; the floor's is inherited)
    assert "extrapolat" in RP.RESPONSE_ANCHOR_CEILING_REASON.lower()
    assert "21.6" in RP.RESPONSE_ANCHOR_CEILING_REASON


@pytest.mark.parametrize("lo,hi,emit", [
    (19.7, 21.6, True),          # the primary reporting window
    (19.7, 21.5, True),          # narrower is fine
    (19.5, 20.3, False),         # sub-DLA tier: floor below 19.7
    (20.3, np.inf, False),       # the DLA headline: OPEN TOP -> total Omega
    (20.0, np.inf, False),
    (19.7, 22.4, False),         # ceiling above the anchored range
])
def test_omega_decision_refuses_anything_outside_the_window(lo, hi, emit):
    """PI decision 1: no unqualified / total / open-topped Omega_HI.

    MUTATION: make ``omega_decision`` return ``emit=True`` unconditionally
    -> RED on all four False rows. Verified.
    """
    d = RP.omega_decision(lo, hi)
    assert d["emit"] is emit, d
    if emit:
        assert d["label"].startswith("OMEGA_HI_LIMITED_")
        assert d["window_logN"] == [lo, hi]
    else:
        assert d["label"] is None
        assert d["reason"].startswith("REFUSED")


def test_omega_decision_names_the_open_top_case_explicitly():
    """An OPEN-topped window must say so, not just 'above 21.6'.

    MUTATION: drop the ``np.isfinite(hi)`` branch -> RED. Verified.
    """
    d = RP.omega_decision(20.3, np.inf)
    assert "OPEN-TOPPED" in d["reason"]
    assert "did\nNOT authorize" in d["reason"] or "NOT authorize" in d["reason"]


# ===========================================================================
# DECISION 3 — the 0.2-dex latent basis
# ===========================================================================

def test_e4_merge_convention_is_literally_the_same_object():
    """The merging convention was MOVED, not copied.

    MUTATION: re-define ``basis_groups`` inside e4_probe -> RED (the identity
    check fails). Verified.
    """
    from CDDF_analysis.hbi_mcmc import e4_probe as e4
    assert e4.basis_groups is RP.basis_groups
    assert e4.merge_basis_columns is RP.merge_basis_columns
    assert e4.merged_truth is RP.merged_truth


@pytest.mark.parametrize("width", [None, 0.1])
def test_basis_pad_edges_default_is_bit_for_bit_the_shipped_grid(width):
    """The DEFAULT basis is UNCHANGED (the PI kept 0.1 dex as the default).

    MUTATION: make ``basis_pad_edges`` merge unconditionally -> RED. Verified.
    """
    for floor in (None, 19.3, 19.0, 18.0):
        e, n = EP.basis_pad_edges(floor, width)
        e_ref, n_ref = EP.basis_pad_edges(floor)          # the v1.1 signature
        assert np.array_equal(e, e_ref)
        assert n == n_ref
        assert np.allclose(np.diff(e), 0.1, atol=1e-8)


def test_adopted_basis_grid_is_exactly_the_expected_16_bins():
    """The ADOPTED grid, pinned digit by digit.

    MUTATION: merge the padded grid in ONE pass instead of two segments -> RED
    (edges become 19.0, 19.2, 19.4, 19.6, ... and 19.5 is no longer an edge).
    Verified.
    """
    e, n_pad = EP.basis_pad_edges(19.0, 0.2)
    assert [round(float(x), 3) for x in e] == [
        19.0, 19.2, 19.5, 19.7, 19.9, 20.1, 20.3, 20.5, 20.7, 20.9,
        21.1, 21.3, 21.5, 21.7, 21.9, 22.1, 22.4]
    assert len(e) - 1 == 16
    assert n_pad == 2


@pytest.mark.parametrize("floor", [None, 19.3, 19.0, 18.5, 18.0])
@pytest.mark.parametrize("width", [0.1, 0.2, 0.3, 0.4])
def test_reporting_floor_is_always_an_exact_basis_edge(floor, width):
    """A basis bin astride the pad/report boundary would mix
    convention-dependent sub-floor support into an in-window bin.

    MUTATION: merge in one pass -> RED for (19.0, 0.2), (18.5, 0.2), (18.0, 0.3)
    and others. Verified.
    """
    e, _ = EP.basis_pad_edges(floor, width)
    assert np.any(np.isclose(e, EP.NHAT_EDGES[0], atol=1e-8)), (
        f"reporting floor {EP.NHAT_EDGES[0]} is not a basis edge in {e}")
    # and the top edge is always shared with the observed grid
    assert np.isclose(e[-1], EP.NHAT_EDGES[-1], atol=1e-8)
    # every basis edge sits on the observed 0.1-dex grid
    off = (e - EP.NHAT_EDGES[0]) / 0.1
    assert np.allclose(off, np.round(off), atol=1e-6)


@pytest.mark.parametrize("bad", [0.15, 0.25, 0.05, 0.0, -0.2])
def test_basis_width_off_the_observed_grid_is_refused(bad):
    """MUTATION: drop the integer-multiple check -> RED. Verified."""
    with pytest.raises(ValueError, match="integer multiple"):
        EP.basis_pad_edges(19.0, bad)


def test_plotting_grid_disclosure_is_a_schema_block_not_a_sentence():
    """Decision 3: a downstream reader must be UNABLE to miss it.

    MUTATION: set ``is_independent_information_resolution`` to True -> RED.
    Verified.
    """
    d = RP.plotting_grid_disclosure(0.2)
    assert d["grid_role"] == "PLOTTING_ONLY"
    assert d["is_independent_information_resolution"] is False
    assert d["n_plot_bins_per_basis_bin"] == 2
    assert d["basis_width_dex"] == 0.2
    assert "NOT INDEPENDENT INFORMATION RESOLUTION" in d["note"]
    with pytest.raises(RP.ReportingGuardError):
        RP.plotting_grid_disclosure(0.15)      # not a multiple of 0.1


def test_merged_edges_refuses_a_non_partition():
    """MUTATION: drop the partition check -> RED. Verified."""
    e = np.round(np.arange(19.5, 20.0 + 1e-9, 0.1), 3)      # 6 edges, 5 bins
    assert len(e) == 6
    assert np.allclose(RP.merged_edges(e, [[0, 1], [2, 3, 4]]),
                       [19.5, 19.7, 20.0])
    with pytest.raises(RP.ReportingGuardError, match="partition"):
        RP.merged_edges(e, [[0, 1], [3, 4]])
    with pytest.raises(RP.ReportingGuardError, match="partition"):
        RP.merged_edges(e, [[1, 0], [2, 3, 4]])


# ===========================================================================
# pack schema — the coarse latent basis
# ===========================================================================

def _coarse_pack(width=0.2, pad_floor=None):
    """A schema-valid synthetic pack whose LATENT basis is coarser than its
    observed grid, built with the SAME merge convention as the extractor."""
    kw = PK.small_test_grid()
    p = PK.synthetic_pack(seed=0, **kw)
    fine = np.asarray(p.nhat_edges, float)
    if pad_floor is not None:
        n = int(round((fine[0] - pad_floor) / 0.1))
        fine = np.round(np.concatenate(
            [fine[0] - 0.1 * np.arange(n, 0, -1), fine]), 10)
    n_pad_fine = len(fine) - len(p.nhat_edges)
    g = int(round(width / 0.1))
    groups = []
    if n_pad_fine:
        groups += RP.basis_groups(n_pad_fine, g)
    groups += [[b + n_pad_fine for b in gr]
               for gr in RP.basis_groups(len(p.nhat_edges) - 1, g)]
    edges = RP.merged_edges(fine, groups)
    tc = np.asarray(p.truth_counts, float)
    tc_pad = np.concatenate([np.zeros((n_pad_fine,) + tc.shape[1:]), tc], axis=0)
    tc_m = np.stack([tc_pad[gr].sum(axis=0) for gr in groups])
    return dataclasses.replace(p, ntrue_edges=edges,
                               truth_counts=tc_m, truth_counts_bks=None)


def test_validate_pack_accepts_the_adopted_coarse_basis():
    """MUTATION: keep the old unconditional
    ``_check_edges_uniform("ntrue_edges", _ne, _N_STEP)`` -> RED
    (PackSchemaError: 'schema requires uniform 0.1 steps'). Verified.
    """
    p = _coarse_pack(0.2, 19.0)
    PK.validate_pack(p, allow_nonstandard_grid=True)
    assert p.n_b < p.n_c
    assert p.basis_width == 0.2
    assert p.basis_is_uniform is False        # 0.3-dex remainder bins


def test_validate_pack_refuses_a_basis_edge_off_the_observed_grid():
    """MUTATION: drop the on-grid check -> RED. Verified."""
    p = _coarse_pack(0.2)
    e = np.asarray(p.ntrue_edges, float).copy()
    e[1] += 0.05
    with pytest.raises(PK.PackSchemaError, match="observed 0.1 dex grid"):
        PK.validate_pack(dataclasses.replace(p, ntrue_edges=e),
                         allow_nonstandard_grid=True)


def test_validate_pack_refuses_a_basis_that_straddles_the_reporting_floor():
    """The exact defect the two-segment merge exists to prevent.

    MUTATION: drop the 'reporting FLOOR must be a basis edge' check -> RED.
    Verified.
    """
    p = _coarse_pack(0.2, 19.0)
    fine = np.round(np.arange(19.0, 20.5 + 1e-9, 0.1), 10)
    one_pass = RP.merged_edges(fine, RP.basis_groups(len(fine) - 1, 2))
    assert not np.any(np.isclose(one_pass, 19.5, atol=1e-8))  # 19.4/19.6 straddle
    tc = np.zeros((len(one_pass) - 1, p.n_k))
    with pytest.raises(PK.PackSchemaError, match="must itself be a latent-basis edge"):
        PK.validate_pack(dataclasses.replace(p, ntrue_edges=one_pass,
                                             truth_counts=tc,
                                             truth_counts_bks=None),
                         allow_nonstandard_grid=True)


def test_validate_pack_refuses_a_basis_finer_than_the_observed_grid():
    """MUTATION: drop the finer-than-observed check -> RED. Verified."""
    p = PK.synthetic_pack(seed=0, **PK.small_test_grid())
    fine = np.round(np.arange(19.5, 20.5 + 1e-9, 0.05), 10)
    tc = np.zeros((len(fine) - 1, p.n_k))
    with pytest.raises(PK.PackSchemaError):
        PK.validate_pack(dataclasses.replace(p, ntrue_edges=fine,
                                             truth_counts=tc,
                                             truth_counts_bks=None),
                         allow_nonstandard_grid=True)


def test_n_pad_bins_is_counted_from_the_edges():
    """``len(ntrue) - len(nhat)`` goes NEGATIVE on a coarse basis and every
    downstream ``truth[:n_pad]`` slice silently means the wrong thing.

    MUTATION: restore ``return int(len(self.ntrue_edges) - len(self.nhat_edges))``
    -> RED (the coarse pack reports -6 instead of 2). Verified.
    """
    p01 = _coarse_pack(0.1, 19.0)
    assert p01.n_pad_bins == len(p01.ntrue_edges) - len(p01.nhat_edges) == 5
    p02 = _coarse_pack(0.2, 19.0)
    assert p02.n_pad_bins == 2
    assert len(p02.ntrue_edges) - len(p02.nhat_edges) < 0     # the old formula
    p_none = PK.synthetic_pack(seed=0, **PK.small_test_grid())
    assert p_none.n_pad_bins == 0


# ===========================================================================
# DECISION 4 — the pad is LATENT NUISANCE
# ===========================================================================

def test_assert_no_subwindow_bins_fires_on_a_pad_bin_and_passes_otherwise():
    """MUTATION: make ``assert_no_subwindow_bins`` return True unconditionally
    -> RED on the first leg. Verified.
    """
    e, _ = EP.basis_pad_edges(19.0, 0.2)
    w_ok = RP.window_overlap_weights(e)          # the reporting weights
    assert RP.assert_no_subwindow_bins(e, w_ok, where="test") is True
    w_bad = w_ok.copy()
    w_bad[0] = 0.2                               # the [19.0, 19.2) pad bin
    with pytest.raises(RP.ReportingGuardError, match="LATENT NUISANCE"):
        RP.assert_no_subwindow_bins(e, w_bad, where="test")
    with pytest.raises(RP.ReportingGuardError, match="per-bin vector"):
        RP.assert_no_subwindow_bins(e, w_ok[:-1], where="test")


def test_window_weights_zero_every_bin_below_the_reporting_floor():
    """MUTATION: replace ``np.clip(w, 0, None)`` by ``w`` -> RED (negative
    weights appear below the floor). Verified.
    """
    e, _ = EP.basis_pad_edges(19.0, 0.2)
    w = RP.window_overlap_weights(e)
    below = np.asarray(e[:-1]) < 19.7 - 1e-9
    assert np.all(w[below] == 0.0)
    assert w.sum() == pytest.approx(21.6 - 19.7)     # the whole window, once
    # the straddling bin contributes exactly its overlap
    i = int(np.flatnonzero(np.isclose(e[:-1], 21.5))[0])
    assert w[i] == pytest.approx(0.1)


def test_bins_fully_inside_excludes_the_straddler():
    """MUTATION: use ``e[:-1] <= hi`` instead of ``e[1:] <= hi`` -> RED.
    Verified.
    """
    e, _ = EP.basis_pad_edges(19.0, 0.2)
    m = RP.bins_fully_inside(e)
    assert [[round(float(e[i]), 3), round(float(e[i + 1]), 3)]
            for i in np.flatnonzero(m)] == [
        [19.7, 19.9], [19.9, 20.1], [20.1, 20.3], [20.3, 20.5], [20.5, 20.7],
        [20.7, 20.9], [20.9, 21.1], [21.1, 21.3], [21.3, 21.5]]
    assert not m[int(np.flatnonzero(np.isclose(e[:-1], 21.5))[0])]


def test_convention_systematic_is_a_half_span_and_refuses_quadrature():
    """MUTATION: return the FULL span instead of the half span -> RED.
    Verified.
    """
    corners = {"clamp=both|cmp=molly172": 100.0, "clamp=hi|cmp=molly172": 104.0,
               "clamp=both|cmp=const_extrap": 108.0,
               "clamp=hi|cmp=const_extrap": 111.0}
    d = RP.convention_systematic(corners, "clamp=both|cmp=molly172")
    assert d["span"] == pytest.approx(11.0)
    assert d["sigma_conv"] == pytest.approx(5.5)
    assert d["frac_conv"] == pytest.approx(0.055)
    assert d["combination_rule"] == "SEPARATE_LINEAR_ENVELOPE"
    assert "QUADRATURE IS REFUSED" in d["why_not_quadrature"]
    with pytest.raises(RP.ReportingGuardError):
        RP.convention_systematic(corners, "no_such_corner")


def test_convention_systematic_works_per_bin():
    """MUTATION: reduce over the wrong axis (axis=1) -> RED. Verified."""
    corners = {"a": [1.0, 2.0], "b": [3.0, 2.0]}
    d = RP.convention_systematic(corners, "a")
    assert d["sigma_conv"] == [1.0, 0.0]
    assert d["frac_conv"] == [1.0, 0.0]


# ===========================================================================
# DECISION 8 — the |z| criterion
# ===========================================================================

def test_z_criterion_states_all_four_previously_missing_things():
    """MUTATION: delete the ``denominator`` key -> RED. Verified."""
    for k in ("per_bin_formula", "total_formula", "denominator", "bin_set",
              "reduction", "chi2_dof", "sign_convention", "absolute_value"):
        assert k in RP.Z_CRITERION and RP.Z_CRITERION[k]
    assert "max(mu_c, 1e-12)" in RP.Z_CRITERION["per_bin_formula"]
    assert "obs_c > 0" in RP.Z_CRITERION["bin_set"]
    assert "not ratified" in RP.Z_CRITERION["not_ratified"].lower()


def test_window_closure_metrics_unrestricted_equals_the_committed_gate():
    """The windowed numbers are only trustworthy if the UNRESTRICTED call
    reproduces the committed gate exactly.

    MUTATION: put obs in the z denominator -> RED. Verified.
    """
    from CDDF_analysis.hbi_mcmc import forward_selftest as FS
    from CDDF_analysis.hbi_mcmc import run_posterior as RPST
    p = PK.synthetic_pack(seed=0, **PK.small_test_grid())
    tab = FS.ratio_tables(FS.selftest(p), p)
    gate = RPST.forward_closure_gate(p)
    m = RP.window_closure_metrics(tab["by_nhat"])
    assert m["total_ratio"] == pytest.approx(gate["total_ratio"], rel=1e-12)
    assert m["chi2_dof"] == pytest.approx(gate["chi2_dof"], rel=1e-12)
    assert abs(m["z_total"]) == pytest.approx(gate["z_total"], rel=1e-12)
    assert m["z_bin_max"] == pytest.approx(gate["z_bin_max"], rel=1e-12)
    assert m["n_bins_in_z_set"] == gate["n_bins"]


def test_window_closure_metrics_restricts_to_bins_fully_inside():
    """MUTATION: use ``r["lo"] >= lo or r["hi"] <= hi`` -> RED. Verified."""
    rows = [dict(lo=19.6, hi=19.7, mu=10.0, obs=20.0),
            dict(lo=19.7, hi=19.8, mu=10.0, obs=10.0),
            dict(lo=21.5, hi=21.6, mu=10.0, obs=10.0),
            dict(lo=21.6, hi=21.7, mu=10.0, obs=40.0)]
    m = RP.window_closure_metrics(rows, 19.7, 21.6, label="w")
    assert m["n_bins_in_window"] == 2
    assert m["total_ratio"] == pytest.approx(1.0)
    assert m["z_bin_max"] == pytest.approx(0.0)
    full = RP.window_closure_metrics(rows)
    assert full["n_bins_in_window"] == 4
    assert full["total_ratio"] == pytest.approx(40.0 / 80.0)


# ===========================================================================
# the reductions: overlap weights vs the previous centre selection
# ===========================================================================

def test_overlap_weights_are_bit_identical_to_centre_selection_on_0p1dex():
    """The new integrated reduction must be BIT-IDENTICAL on every 0.1-dex pack.

    MUTATION: drop the ``min(hi, ntrue[-1])`` clamp -> RED (the open-topped
    tiers then integrate past the top edge). Verified.
    """
    from CDDF_analysis.hbi_mcmc import model_a as MA
    p = PK.synthetic_pack(seed=0, **PK.small_test_grid())
    ntrue = np.asarray(p.ntrue_edges, float)
    Nc = 0.5 * (ntrue[:-1] + ntrue[1:])
    dN = np.diff(ntrue)
    rep = Nc >= float(np.asarray(p.nhat_edges, float)[0]) - 1e-9
    for lo, hi in list(MA.TIERS.values()) + [(20.0, np.inf), (20.3, np.inf)]:
        old = np.where((Nc >= lo - 1e-9) & (Nc < hi - 1e-9) & rep, dN, 0.0)
        new = np.where(rep, RP.window_overlap_weights(
            ntrue, lo, min(hi, ntrue[-1])), 0.0)
        assert np.array_equal(old, new), (lo, hi, old, new)


def test_posterior_summary_refuses_unqualified_omega_and_emits_the_windowed_one():
    """PI decision 1, at the paper-facing emission point.

    MUTATION: emit ``omega_allz`` unconditionally in ``posterior_summary``
    -> RED. Verified.
    """
    from CDDF_analysis.hbi_mcmc import model_a as MA
    kw = PK.small_test_grid()
    kw["nhat_edges"] = np.round(np.arange(19.5, 22.4 + 1e-9, 0.1), 10)
    p = PK.synthetic_pack(seed=0, **kw)
    rng = np.random.default_rng(0)
    f = np.asarray(p.truth["f_true"], float)[None, ...] * np.exp(
        rng.normal(0, 0.01, size=(8,) + np.asarray(p.truth["f_true"]).shape))
    red = MA.reduce_f_posterior(f, p)
    summ = MA.posterior_summary(red, p)
    assert summ["reporting_window_logN"] == [19.7, 21.6]
    win = summ["tiers"]["report_197_216"]
    assert win["in_primary_reporting_window"] is True
    assert win["omega_allz"] is not None
    assert win["omega_label"] == "OMEGA_HI_LIMITED_19.7_21.6"
    for tier in ("subdla_195_203", "dla_20p0", "dla_20p3", "all_195_up"):
        blk = summ["tiers"][tier]
        assert blk["omega_allz"] is None, tier
        assert blk["omega_coarse_z"] is None, tier
        assert blk["omega_REFUSED"]["reason"].startswith("REFUSED"), tier
        # dN/dX is a LINE DENSITY and is NOT refused by decision 1
        assert blk["dndx_allz"]["point_q50"] > 0, tier


def test_reduce_f_posterior_guards_the_window_tier_against_pad_bins(monkeypatch):
    """The decision-4 guard must be WIRED INTO the reduction, not merely
    available as a function.

    The tripwire is against a future change to the weighting: if anything ever
    makes a reported tier's weights reach below 19.7, the reduction must refuse
    rather than quietly report latent nuisance support.  Simulated here by
    corrupting the weight function the reduction calls.

    MUTATION: delete the ``assert_no_subwindow_bins`` call in
    ``reduce_f_posterior`` -> RED (DID NOT RAISE). Verified.
    """
    from CDDF_analysis.hbi_mcmc import model_a as MA
    kw = PK.small_test_grid()
    kw["nhat_edges"] = np.round(np.arange(19.5, 22.4 + 1e-9, 0.1), 10)
    p = PK.synthetic_pack(seed=0, **kw)
    f = np.asarray(p.truth["f_true"], float)[None, ...]
    assert MA.reduce_f_posterior(f, p)["n_bins_report_197_216"] > 0   # clean

    good = RP.window_overlap_weights

    def leaky(edges, lo, hi):
        w = good(edges, lo, hi)
        if abs(float(lo) - RP.NONIDENT_EDGE) < 1e-9:      # the reported window
            w = w.copy()
            w[0] = 0.1                                    # a sub-19.7 bin
        return w

    monkeypatch.setattr(MA.RP, "window_overlap_weights", leaky)
    with pytest.raises(RP.ReportingGuardError, match="REPORTING GUARD"):
        MA.reduce_f_posterior(f, p)
