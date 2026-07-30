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


def _adopted_config_module():
    """The closure driver.  Imported through the package (this suite already
    needs jax via ``hbi_mcmc.__init__``), so it is the SAME object the artifact
    was produced by."""
    from CDDF_analysis.hbi_mcmc import adopted_config as AC
    return AC


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
    """A latent basis FINER than the observed 0.1-dex grid must be refused.

    It is refused BY THE ON-GRID RULE, and that is the whole story: every ntrue
    edge must sit on the observed 0.1-dex grid, so the narrowest representable
    basis bin is 0.1 dex and a finer basis is unrepresentable. Mutation testing
    proved that the separate ``max(diff) < 0.1`` check this file used to carry was
    DEAD CODE -- deleting it left this test green, because the on-grid rule fires
    first. The check has been removed and this test now asserts the real
    mechanism, including the message.

    MUTATION: the on-grid check (``if not np.allclose(off, np.round(off))``)
    -> ``if False`` -> RED here AND in
    test_validate_pack_refuses_a_basis_edge_off_the_observed_grid. Verified.
    """
    p = PK.synthetic_pack(seed=0, **PK.small_test_grid())
    fine = np.round(np.arange(19.5, 20.5 + 1e-9, 0.05), 10)
    tc = np.zeros((len(fine) - 1, p.n_k))
    with pytest.raises(PK.PackSchemaError, match="observed 0.1 dex grid"):
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

    MUTATION: drop the lower clamp in ``window_overlap_weights``
    (``np.maximum(e[:-1], lo)`` -> ``e[:-1]``) -> RED: every bin below the window
    then picks up a spurious weight. Verified.

    NOT a valid mutation, found by mutation testing: removing a
    ``min(hi, ntrue[-1])`` clamp at the call site. It could never change a value,
    because ``window_overlap_weights`` already takes ``min(bin_hi, hi)`` per bin,
    so an open-topped window integrates to the top basis edge and no further. The
    redundant clamp has been deleted from both call sites rather than left in with
    a false test claim attached to it.
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


# ===========================================================================
# pack.coarsen_basis — putting a SYNTHETIC pack on the adopted geometry
# ===========================================================================

def test_coarsen_basis_reproduces_the_extractor_grid_shape():
    """The synthetic path and the extractor must use ONE convention.

    MUTATION: merge in one pass inside ``coarsen_basis`` -> RED (19.5 stops being
    an edge and validate_pack refuses it). Verified.
    """
    p = PK.synthetic_pack(seed=0, **PK.small_test_grid())
    c = PK.coarsen_basis(p, 0.2, pad_floor=19.0)
    assert [round(float(x), 3) for x in c.ntrue_edges] == [
        19.0, 19.2, 19.5, 19.7, 19.9, 20.1, 20.3, 20.5]
    assert c.n_pad_bins == 2
    assert np.array_equal(c.nhat_edges, p.nhat_edges)      # observed never moves
    assert np.array_equal(c.counts, p.counts)
    # the SAME two-segment structure the extractor produces
    e_ext, n_ext = EP.basis_pad_edges(19.0, 0.2)
    assert np.allclose(np.diff(e_ext)[:n_ext], np.diff(c.ntrue_edges)[:2])


def test_coarsen_basis_conserves_truth_counts_and_the_f_integral():
    """MUTATION: sum instead of average the group's f (``f[gr].sum(axis=0)``
    in place of ``merged_truth``) -> RED on the f leg. Verified.

    HONEST LIMIT of this test, found by mutation testing: replacing
    ``merged_truth``'s dN weighting by a PLAIN MEAN does NOT make it red, and
    cannot. ``coarsen_basis`` refuses a non-uniform input basis, so every fine
    bin inside a group has the same dN and the dN-weighted mean IS the plain
    mean. The weighting is load-bearing only for a non-uniform input, which this
    function does not accept. Stated rather than papered over.
    """
    p = PK.synthetic_pack(seed=0, **PK.small_test_grid())
    c = PK.coarsen_basis(p, 0.2)
    assert np.asarray(c.truth_counts).sum() == pytest.approx(
        np.asarray(p.truth_counts).sum())
    f0, f1 = np.asarray(p.truth["f_true"]), np.asarray(c.truth["f_true"])
    i0 = (f0 * np.diff(np.asarray(p.ntrue_edges))[:, None]).sum()
    i1 = (f1 * np.diff(np.asarray(c.ntrue_edges))[:, None]).sum()
    assert i1 == pytest.approx(i0, rel=1e-12)
    assert c.truth["basis_coarsened_to_dex"] == 0.2


def test_coarsen_basis_refuses_a_pack_that_is_already_coarse():
    """MUTATION: drop the input-step check -> RED. Verified."""
    p = PK.coarsen_basis(PK.synthetic_pack(seed=0, **PK.small_test_grid()), 0.2)
    with pytest.raises(PK.PackSchemaError, match="must be on the observed"):
        PK.coarsen_basis(p, 0.2)


# ===========================================================================
# the pack-stamp guard: a mismatch is CLEARED or FATAL, never a printed warning
# ===========================================================================

def test_pack_stamp_verdict_clears_a_benign_mismatch_and_refuses_a_stale_one():
    """A stamp mismatch must be DECIDED from the file diff, not warned about.

    MUTATION: return ``ok=True`` in the ``touched`` branch -> RED on the stale
    leg. Verified.
    MUTATION: drop the ``dirty_pack`` branch -> RED on the dirty leg. Verified.
    """
    from CDDF_analysis.hbi_mcmc import adopted_config as AC
    same = AC.pack_stamp_verdict(["abc"], "abc", [])
    assert same["ok"] and same["packs_match_closure_commit"]

    benign = AC.pack_stamp_verdict(
        ["abc"], "def", ["CDDF_analysis/hbi_mcmc/adopted_config.py",
                         "tests/test_adopted_reporting.py"])
    assert benign["ok"] is True
    assert benign["packs_match_closure_commit"] is False
    assert benign["pack_determining_files_changed"] == []
    assert "cannot change a pack" in benign["reason"]

    stale = AC.pack_stamp_verdict(
        ["abc"], "def", ["CDDF_analysis/hbi_mcmc/extract_pack.py"])
    assert stale["ok"] is False
    assert stale["pack_determining_files_changed"] == [
        "CDDF_analysis/hbi_mcmc/extract_pack.py"]
    assert "STALE" in stale["reason"]

    dirty = AC.pack_stamp_verdict(["abc-dirty"], "abc-dirty", [])
    assert dirty["ok"] is False and dirty["any_pack_dirty"] is True

    # the extractor's OWN inputs count, not just its own file
    for f in ("CDDF_analysis/hbi_mcmc/pack.py",
              "CDDF_analysis/hbi_mcmc/reporting.py",
              "CDDF_analysis/hbi/cddf_catalog_hbi.py"):
        assert AC.pack_stamp_verdict(["abc"], "def", [f])["ok"] is False, f


# ===========================================================================
# coverage: the matched configuration and the MEASURED power check
# ===========================================================================

def test_sbc_reports_the_primary_reporting_window_functional():
    """After decision 1 the REPORTED functional is the windowed one; an SBC that
    ranks only the open-topped 20.0/20.3 thresholds says nothing about it.

    MUTATION: drop the reduce_f_posterior extension in ``_reported_from_f``
    -> RED. Verified.
    """
    from CDDF_analysis.hbi_mcmc import sbc as S
    p = PK.coarsen_basis(PK.synthetic_pack(seed=0, **S.SBC_GRID_ADOPTED),
                         **S.SBC_ADOPTED_BASIS)
    q = S._reported_from_f(np.asarray(p.truth["f_true"], float), p)
    assert "dndx_report_197_216_allz" in q
    assert "omega_report_197_216_allz" in q
    assert q["dndx_report_197_216_allz"] > 0


def test_rescale_dispersion_moves_the_width_and_not_the_location():
    """The three properties that make the power check a POWER check.

    Without property 2 a flag at s != 1 could be a location shift rather than a
    mis-scaled width, and the detection curve would measure the wrong thing.

    MUTATION A: ``np.exp(s * log_post)`` (no median pivot) -> RED (the per-bin
    median moves). Verified.
    MUTATION B: ``s * log_med + (log_post - log_med)`` (scale the pivot instead
    of the residual) -> RED (the log-SD stops scaling by s). Verified.
    """
    from CDDF_analysis.hbi_mcmc import sbc as S
    rng = np.random.default_rng(0)
    f = np.exp(rng.normal(-20.0, 0.4, size=(400, 6, 3)))
    assert S.rescale_dispersion(f, 1.0) is f            # bit-identical identity
    lm0 = np.median(np.log(f), axis=0)
    sd0 = np.std(np.log(f), axis=0)
    for s in (0.5, 1.5, 2.0):
        g = S.rescale_dispersion(f, s)
        assert np.allclose(np.median(np.log(g), axis=0), lm0, atol=1e-12)
        assert np.allclose(np.std(np.log(g), axis=0), s * sd0, rtol=1e-12)


def test_dispersion_scale_one_is_the_identity_and_two_is_not():
    """End-to-end: the SBC's own ranks at s = 1 must be EXACTLY the unscaled
    ranks, otherwise the detection curve's baseline is not the result being
    certified.

    MUTATION: ``f_s = rescale_dispersion(f_post, s)`` -> ``rescale_dispersion(
    f_post, 2.0 * s)`` -> RED (s = 1 stops being the identity). Verified.
    """
    from CDDF_analysis.hbi_mcmc import sbc as S
    samp = dict(S.SBC_SAMPLER, num_warmup=30, num_samples=30, n_ranks=10)
    ranks, meta = S.sbc_run(2, seed=3, grid=S.SBC_GRID_ADOPTED, sampler=samp,
                            dispersion_scales=(1.0, 2.0), **S.SBC_ADOPTED_BASIS)
    assert meta["n_sims_used"] >= 1
    assert meta["ranks_by_scale"]["1"] == ranks          # exact identity
    assert meta["dispersion_scales"] == [1.0, 2.0]
    assert meta["matched_configuration"] is True
    assert meta["basis_width"] == 0.2 and meta["pad_floor"] == 19.0
    assert meta["n_pad_bins"] == 2
    # a 2x log-space widening must move at least one rank somewhere
    assert meta["ranks_by_scale"]["2"] != ranks
    assert "R5 REPORTING WINDOW" in meta["reduction_note"]


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


# ===========================================================================
# THE 2026-07-29 REFEREE DEFECTS (adversarial re-run; each defect first)
# ===========================================================================

def _exact_truth_pack(basis_width, pad_floor=None, seed=0):
    """A pack whose ``truth_counts`` are EXACTLY ``dX_k * f_true * dN``.

    No Poisson draw.  That makes a posterior/truth ratio of 1.0 a pure
    BOOKKEEPING IDENTITY at ``f = f_true``: any departure is a support or
    weighting-convention mismatch between the two sides, never sampling scatter.
    This is the fixture shape the [[one-sided support]] class demands — build the
    case where the two sides CANNOT differ for a statistical reason.
    """
    p = PK.synthetic_pack(seed=seed, fp_frac=0.0)
    if basis_width != 0.1 or pad_floor is not None:
        p = PK.coarsen_basis(p, basis_width, pad_floor=pad_floor)
    f = np.asarray(p.truth["f_true"], float)
    dN = np.diff(np.asarray(p.ntrue_edges, float))
    dX_k = np.asarray(p.dX, float).sum(axis=1)
    return dataclasses.replace(p, truth_counts=dX_k[None, :] * f * dN[:, None])


@pytest.mark.parametrize("basis_width,pad_floor", [
    (0.1, None),        # the aligned case the old test already covered
    (0.2, None),        # THE ADOPTED BASIS — 20.0 is NOT an edge here
    (0.2, 19.0),        # adopted basis + adopted pad
    (0.2, 18.0),        # adopted basis + a deep pad (uneven pad groups)
])
def test_truth_side_uses_the_same_window_convention_as_the_posterior(
        basis_width, pad_floor):
    """DEFECT 2 (referee, 2026-07-29): at ``f = f_true`` EVERY reported
    quantity's post/truth ratio must be 1 on a COARSE basis, not only on 0.1 dex.

    ``reduce_f_posterior`` integrates with dex-overlap weights while
    ``evidence._truth_reported`` selected basis bins BY CENTRE on truth counts.
    On the adopted 0.2-dex basis 20.0 is not a basis edge, so the posterior
    integrated HALF of [19.9, 20.1) and the truth integrated ALL of it —
    ``closure_block`` was comparing two different estimands.  MEASURED before
    the fix (this fixture, bw=0.2): dndx_20p0_integrated 0.7868,
    omega_20p0_integrated 0.9175, while the ALIGNED 20.3 tier sat at 1.0000.
    A ~20% deficit that is pure bookkeeping.

    MUTATION: restore the centre selection in ``evidence._truth_reported``
    (``sel = Nc >= thr - 1e-9`` weighting whole bins) -> RED at every bw=0.2
    case, still green at bw=0.1 (which is exactly why the old 0.1-dex-only
    equivalence test could not see this).

    NOTE the parametrization deliberately stops at 0.2 dex: on a 0.3-dex basis
    19.7 is not a basis edge at all, and the decision-4 guard REFUSES that
    geometry outright (pinned separately by
    ``test_a_basis_that_straddles_the_reporting_floor_refuses_the_window_tier``).
    """
    from CDDF_analysis.hbi_mcmc import evidence as EV
    p = _exact_truth_pack(basis_width, pad_floor)
    f = np.asarray(p.truth["f_true"], float)
    rep = EV.reported_quantities(f[None, None], p)          # (1, 1, B, Kf)
    truth, n_truth = EV._truth_reported(p)
    assert set(rep) == set(truth), (sorted(set(rep) ^ set(truth)))
    for name in sorted(rep):
        post = float(np.asarray(rep[name]).reshape(-1)[0])
        T = float(truth[name])
        assert T > 0, name
        assert post / T == pytest.approx(1.0, rel=1e-10), (
            f"{name}: post/truth = {post / T:.6f} at f = f_true on a "
            f"{basis_width}-dex basis — the two sides do not share a support")


@pytest.mark.parametrize("basis_width", [0.1, 0.2])
def test_analyze_rung9_truth_tier_table_shares_the_posterior_convention(
        basis_width):
    """DEFECT 2, second site: ``analyze_rung9.truth_tier_table`` selected by
    centre too, so the rung-ladder closure table carried the same ~20% deficit.

    MUTATION: restore ``sel = Nc >= thr - 1e-9`` in ``truth_tier_table`` -> RED
    at 0.2 dex (green at 0.1, the aligned blind spot).
    """
    from CDDF_analysis.hbi_mcmc import analyze_rung9 as AR
    from CDDF_analysis.hbi_mcmc import model_a as MA
    p = _exact_truth_pack(basis_width)
    f = np.asarray(p.truth["f_true"], float)[None, ...]
    red = MA.reduce_f_posterior(f, p)
    tab = AR.truth_tier_table(p)
    kz = np.asarray(p.kz_to_K)
    dX_k = np.asarray(p.dX, float).sum(axis=1)
    for tag in ("20p0", "20p3"):
        post = AR._coarse_avg(np.asarray(red[f"dndx_{tag}"], float),
                              kz, dX_k, p.n_kk)[0]
        for K in range(p.n_kk):
            T = float(tab[tag]["dndx_truth"][K])
            assert post[K] / T == pytest.approx(1.0, rel=1e-10), (tag, K,
                                                                  post[K] / T)


def test_response_anchor_reason_quotes_the_numbers_the_packs_actually_carry():
    """DEFECT 3 (referee): the ceiling justification quoted 19.336-21.503 and
    '21.05'.  21.503 appears in no pack — it is the LO range's 19.503 mistyped —
    and the measured top anchor is 21.040565-21.216358, not 21.05.

    MUTATION: put '21.503' back into ``RESPONSE_ANCHOR_CEILING_REASON`` -> RED.
    """
    r = RP.RESPONSE_ANCHOR_CEILING_REASON
    m = RP.RESPONSE_ANCHOR_MEASURED
    assert "21.503" not in r, "21.503 is a typo for the LO range's 19.503"
    assert m["top_anchor_min"] == pytest.approx(21.040565, abs=1e-6)
    assert m["top_anchor_max"] == pytest.approx(21.216358, abs=1e-6)
    assert m["bottom_anchor_min"] == pytest.approx(19.336020, abs=1e-6)
    assert m["bottom_anchor_max"] == pytest.approx(19.502988, abs=1e-6)
    # the prose must quote the measured digits, so the two cannot drift apart
    for tok in ("19.336", "19.503", "21.041", "21.216"):
        assert tok in r, f"{tok} missing from RESPONSE_ANCHOR_CEILING_REASON"
    assert m["source"].endswith("emp_N_anchors")


def test_the_authorized_omega_window_contains_extrapolated_response():
    """DEFECT 3: ~0.4 dex of EXTRAPOLATED response sits INSIDE [19.7, 21.6] —
    the one window where Omega_HI is authorized.  That must be a COMPUTED
    number, stated, not a prose claim.

    MUTATION: make ``extrapolated_response_inside_window`` clamp the top anchor
    up to the ceiling (return 0.0) -> RED.
    """
    e = RP.extrapolated_response_inside_window()
    assert e["dex_extrapolated_best_cell"] == pytest.approx(0.3836, abs=1e-3)
    assert e["dex_extrapolated_worst_cell"] == pytest.approx(0.5594, abs=1e-3)
    assert e["inside_the_authorized_omega_window"] is True
    assert "EXTRAPOLATED" in e["statement"]
    # arithmetic, on a fit range that is NOT the committed one
    e2 = RP.extrapolated_response_inside_window(
        top_anchor_min=20.0, top_anchor_max=21.0, ceiling=21.6)
    assert e2["dex_extrapolated_best_cell"] == pytest.approx(0.6)
    assert e2["dex_extrapolated_worst_cell"] == pytest.approx(1.6)
    # a response measured ABOVE the ceiling leaves nothing extrapolated inside
    e3 = RP.extrapolated_response_inside_window(
        top_anchor_min=21.9, top_anchor_max=22.0, ceiling=21.6)
    assert e3["dex_extrapolated_worst_cell"] == 0.0
    assert e3["inside_the_authorized_omega_window"] is False


@pytest.mark.parametrize("npz", [os.path.join(
    "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/stage0",
    "forward_response_2lpt0.npz")])
def test_response_anchor_measured_reproduces_from_the_frozen_npz(npz):
    """DEFECT 3, the real pin: ``RESPONSE_ANCHOR_MEASURED`` must reproduce from
    the FROZEN forward-response NPZ through the committed routine, not be a
    hand-copied literal.  (2LPT-0 mock calibration artifact; no survey data.)

    MUTATION: change ``top_anchor_max`` in RESPONSE_ANCHOR_MEASURED -> RED.
    """
    if not os.path.exists(npz):
        pytest.skip(f"frozen forward-response NPZ absent: {npz}")
    rr = PK.resp_fit_range_from_forward_npz(npz)
    m = RP.RESPONSE_ANCHOR_MEASURED
    assert float(rr[..., 0].min()) == pytest.approx(m["bottom_anchor_min"], abs=5e-7)
    assert float(rr[..., 0].max()) == pytest.approx(m["bottom_anchor_max"], abs=5e-7)
    assert float(rr[..., 1].min()) == pytest.approx(m["top_anchor_min"], abs=5e-7)
    assert float(rr[..., 1].max()) == pytest.approx(m["top_anchor_max"], abs=5e-7)


def test_order_of_magnitude_chi2_claims_must_be_backed_by_the_measured_factor():
    """DEFECT 4 (referee): ``verdict.what_the_window_removes`` claimed the
    windowed chi2/dof "falls by more than an order of magnitude" while
    ``residual_decomposition.correction`` refuted it in the same file.  The
    measured factors are 7.80 / 5.58 / 6.92 — none exceeds 10x.

    MUTATION: drop the ``>= 10`` test in
    ``adopted_config.assert_no_contradictory_chi2_claims`` (accept any claim)
    -> RED.
    """
    AC = _adopted_config_module()
    factors = {"a": 7.80, "b": 5.58, "c": 6.92}
    txt = AC.window_removal_statement(factors)
    # it may MENTION the phrase, but only to deny it, and it must quote the
    # measured factors rather than a narrative adjective
    i = txt.lower().find("order of magnitude")
    assert i > 0, txt
    assert "not" in txt.lower()[max(0, i - 60):i], txt
    assert "7.80x" in txt and "5.58x" in txt and "5.6-7.8x" in txt
    # the scanner must FIRE on a hand-edited contradiction
    bad = {"verdict": {"what_the_window_removes":
                       "... the windowed chi2/dof falls by more than an order "
                       "of magnitude."},
           "residual_decomposition": {"per_mock": {
               m: {"full_grid": {"chi2_dof": f * 3.0},
                   "reporting_window": {"chi2_dof": 3.0}}
               for m, f in factors.items()}}}
    with pytest.raises(RP.ReportingGuardError, match="CONTRADICT"):
        AC.assert_no_contradictory_chi2_claims(bad)
    # and must PASS once the claim matches the measurement
    bad["verdict"]["what_the_window_removes"] = txt
    assert AC.assert_no_contradictory_chi2_claims(bad) is True
    # a genuine >=10x improvement may say so
    big = {"verdict": {"what_the_window_removes":
                       "falls by more than an order of magnitude"},
           "residual_decomposition": {"per_mock": {
               "a": {"full_grid": {"chi2_dof": 400.0},
                     "reporting_window": {"chi2_dof": 3.0}}}}}
    assert AC.assert_no_contradictory_chi2_claims(big) is True


@pytest.mark.parametrize("artifact,why", [
    ({}, "no residual_decomposition block at all"),
    ({"residual_decomposition": None}, "the block is present but null"),
    ({"residual_decomposition": {"per_mock": {}}}, "per_mock is empty"),
    ({"residual_decomposition": {"per_mock": {
        "a": {"full_grid": {"chi2_dof": 400.0}}}}},
     "reporting_window leg missing -> no factor computable"),
    ({"residual_decomposition": {"per_mock": {
        "a": {"full_grid": None, "reporting_window": None}}}},
     "both legs null -> TypeError path"),
    ({"residual_decomposition": {"per_mock": {
        "a": {"full_grid": {"ratio": 0.84},
              "reporting_window": {"ratio": 0.95}}}}},
     "legs present but carry no chi2_dof"),
])
def test_the_chi2_contradiction_scanner_FAILS_CLOSED_when_it_cannot_measure(
        artifact, why):
    """DEFECT 4, the fail-closed half: a guard that cannot measure the quantity
    it is vouching for must REFUSE, not wave the artifact through.

    This is the branch a mutation survivor exposed (mutant 4c): every other test
    of ``assert_no_contradictory_chi2_claims`` hands it a well-formed
    ``residual_decomposition.per_mock``, so replacing the ``if not factors:
    raise`` with ``return True`` left the whole suite GREEN while the scanner
    became a no-op on exactly the artifact shape it exists to police -- one where
    the measured factors are missing, renamed or nulled.  Note the guard must
    refuse even when the prose contains NO offending phrase: a scanner that
    silently passes because it read nothing is worse than no scanner, because it
    stamps ``no_contradictory_chi2_claims: True`` into the metadata.

    MUTATION 4c: in ``adopted_config.assert_no_contradictory_chi2_claims``
    replace the ``if not factors: raise REP.ReportingGuardError(...)`` with
    ``return True`` -> RED on all six cases here (previously survived: no test
    covered it).
    """
    AC = _adopted_config_module()
    with pytest.raises(RP.ReportingGuardError, match="fail closed"):
        AC.assert_no_contradictory_chi2_claims(dict(artifact))
    # ... and it is the ABSENCE of a measurement that refuses, not the prose:
    # adding a perfectly innocent narrative does not rescue it.
    a2 = dict(artifact)
    a2["verdict"] = {"what_the_window_removes": "the window drops 10 bins."}
    with pytest.raises(RP.ReportingGuardError, match="cannot vouch"):
        AC.assert_no_contradictory_chi2_claims(a2)


def test_the_committed_artifact_carries_no_contradictory_chi2_claim():
    """DEFECT 4, on the STAMPED file: the artifact in git must pass its own
    contradiction scanner.

    MUTATION: revert ``verdict.what_the_window_removes`` in
    adopted_config_closure.json to the 'order of magnitude' wording -> RED.
    """
    import json
    AC = _adopted_config_module()
    p = os.path.join(REPO, "CDDF_analysis/hbi_mcmc/adopted_config_closure.json")
    with open(p) as fh:
        art = json.load(fh)
    assert AC.assert_no_contradictory_chi2_claims(art) is True
    # and the extrapolated-response disclosure must be present and prominent
    ex = art["verdict"]["extrapolated_response_inside_the_omega_window"]
    assert ex["inside_the_authorized_omega_window"] is True
    lim = json.dumps(art["limitations"])
    assert "EXTRAPOLATED" in lim
    # DEFECT 3 must be the FIRST limitation, not buried in the list: a reader who
    # stops after one bullet must still learn it.
    assert "EXTRAPOLATED RESPONSE" in art["limitations"][0], art["limitations"][0]

    # --- the disclosure's own numbers -----------------------------------------
    # The sub-interval edge is snapped to the OBSERVED 0.1-dex grid, so it must be
    # exactly 21.2.  np.floor(21.216358 / 0.1) * 0.1 evaluates to
    # 21.200000000000003 in binary floating point, and that string appeared in the
    # PI-facing `headline` of the first regenerated artifact.
    # MUTATION: drop the round(..., 6) in
    # ``adopted_config.extrapolated_response_block`` -> RED.
    assert ex["subinterval_logN"] == [21.2, RP.RESPONSE_ANCHOR_CEILING], (
        ex["subinterval_logN"])
    blob = json.dumps(ex)
    assert "21.200000000000003" not in blob, "float noise in a PI-facing field"
    # the N-weighted Omega share is the load-bearing number of defect 3; pin the
    # order of magnitude so a silent change to the weighting cannot pass.
    for key in ("omega_share_of_subinterval_by_mock_truth_counts",
                "omega_share_of_subinterval_by_mock_predicted_counts"):
        vals = list(ex[key].values())
        assert len(vals) == 3, (key, vals)
        assert all(0.20 < v < 0.40 for v in vals), (key, vals)
    assert "27.5-29.6%" in ex["headline"], ex["headline"]


@pytest.mark.parametrize("basis_width,pad_floor", [(0.2, 19.0), (0.1, None)])
def test_every_reported_tier_is_either_guarded_or_explicitly_refused(
        basis_width, pad_floor):
    """DEFECT 5 (referee): ``assert_no_subwindow_bins`` ran ONLY inside
    ``if omega_decisions[tier]['emit']`` — i.e. only for ``report_197_216``,
    whose weights are zero below 19.7 BY CONSTRUCTION.  Meanwhile
    ``window_weights_subdla_195_203`` and ``window_weights_all_195_up`` carry
    w = 0.20 dex on the [19.5, 19.7) basis bin and ``posterior_summary`` still
    emitted ``dndx_allz`` for both.  The guard was wired where it cannot fire.

    The contract enforced here: for EVERY tier, either (a) the tier is
    paper-facing and its weights are clean below 19.7, or (b) the tier is
    explicitly marked NOT paper-facing with the offending bins named.

    MUTATION: make ``reporting.reported_tier_decision`` return paper_facing=True
    unconditionally -> RED (the sub-19.7 tiers then claim to be paper-facing and
    the guard raises).  MUTATION 2: stop recording
    ``subwindow_bins_<tier>`` / drop the ``dndx_paper_facing_REFUSED`` block in
    ``posterior_summary`` -> RED.
    """
    from CDDF_analysis.hbi_mcmc import model_a as MA
    p = _exact_truth_pack(basis_width, pad_floor)
    f = np.asarray(p.truth["f_true"], float)[None, ...]
    red = MA.reduce_f_posterior(f, p)
    ntrue = np.asarray(p.ntrue_edges, float)
    summ = MA.posterior_summary(red, p)
    n_refused = 0
    for tier in MA.TIERS:
        if f"window_weights_{tier}" not in red:
            continue
        w = np.asarray(red[f"window_weights_{tier}"], float)
        blk = summ["tiers"][tier]
        dec = red["tier_decisions"][tier]
        assert blk["paper_facing"] is dec["paper_facing"]
        if dec["paper_facing"]:
            # (a) clean: the guard must actually pass on the weights USED
            assert RP.assert_no_subwindow_bins(
                ntrue, w, where=f"test {tier}") is True
            assert "dndx_paper_facing_REFUSED" not in blk
        else:
            # (b) refused, with the offending bins NAMED
            n_refused += 1
            ref = blk["dndx_paper_facing_REFUSED"]
            assert ref["reason"].startswith("NOT_PAPER_FACING")
            bins = red[f"subwindow_bins_{tier}"]
            assert bins, tier
            assert all(b[0] < RP.NONIDENT_EDGE - 1e-9 for b in bins), (tier, bins)
            assert ref["subwindow_bins"] == bins
    # the sub-DLA tier and the coupled all-195-up tier MUST be among the refused
    assert n_refused >= 2
    assert summ["tiers"]["subdla_195_203"]["paper_facing"] is False
    assert summ["tiers"]["all_195_up"]["paper_facing"] is False
    assert summ["tiers"]["report_197_216"]["paper_facing"] is True


def test_a_basis_that_straddles_the_reporting_floor_refuses_the_window_tier():
    """DEFECT 5, the other half: the guard must actually BITE on a real geometry,
    not only under monkeypatch.  On a 0.3-dex basis anchored at 19.5 the value
    19.7 is not a basis edge, so ``report_197_216`` straddles [19.5, 19.8) and
    picks up 0.1 dex of non-identifiable support.  That must refuse.

    MUTATION: remove the ``assert_no_subwindow_bins`` call from
    ``reduce_f_posterior`` -> RED (DID NOT RAISE), and the 0.3-dex geometry
    would silently report a dN/dX built partly on [19.5, 19.7).
    """
    from CDDF_analysis.hbi_mcmc import model_a as MA
    p = _exact_truth_pack(0.3)
    assert not np.any(np.isclose(np.asarray(p.ntrue_edges, float),
                                 RP.NONIDENT_EDGE, atol=1e-8))
    f = np.asarray(p.truth["f_true"], float)[None, ...]
    with pytest.raises(RP.ReportingGuardError, match="REPORTING GUARD"):
        MA.reduce_f_posterior(f, p)


def test_reporting_module_states_exactly_what_the_subwindow_guard_covers():
    """DEFECT 5: the module docstring claimed the guard 'fails closed if a
    reported/paper-facing block carries a basis bin below 19.7' — which is not
    what the code did.  The claim must now be exact.

    MUTATION: delete ``SUBWINDOW_GUARD_SCOPE`` (or drop its enumeration of the
    refused tiers) -> RED.
    """
    s = RP.SUBWINDOW_GUARD_SCOPE
    assert "report_197_216" in s["guarded_tiers"]
    assert "subdla_195_203" in s["refused_tiers"]
    assert "all_195_up" in s["refused_tiers"]
    assert "dla_20p0" in s["guarded_tiers"]
    assert "dla_20p3" in s["guarded_tiers"]
    assert set(s["guarded_tiers"]) & set(s["refused_tiers"]) == set()
    assert "not" in s["what_is_NOT_guarded"].lower()
