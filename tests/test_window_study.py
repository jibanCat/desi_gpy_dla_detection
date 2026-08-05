# -*- coding: utf-8 -*-
"""Tests for the matched spectral-window study (PI decision 2, 2026-07-29).

Two things are under test:

  (1) the WINDOW REGISTRY + its fail-closed matching guard. The failure being
      guarded against is silent: pairing a 911-A absorber/pathlength selection
      with a 1025-A completeness matrix or a 1025-A forward response produces a
      number that looks fine and means nothing.
  (2) the PURE reporting-window arithmetic (PI decision 1: 19.7 <= log NHI <=
      21.6, plus the >= 21.6 high-N residual reported separately). This is the
      NEW arithmetic in the study, so it is tested against hand-computed values
      on a fixture whose out-of-window rows come FIRST (the project's
      one-sided-support discipline).

Env: the `gpdla` env is enough — nothing here imports jax.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _load(name, rel):
    """Load a module file-directly (the hbi_mcmc package __init__ imports jax)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_REPO, rel))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def WS():
    return _load("_ws_under_test", "CDDF_analysis/hbi_mcmc/window_study.py")


@pytest.fixture(scope="module")
def EP():
    return _load("_ep_under_test", "CDDF_analysis/hbi_mcmc/extract_pack.py")


# ---------------------------------------------------------------------------
# (1) the window registry
# ---------------------------------------------------------------------------
def test_default_window_is_the_nominal_molly_window(EP):
    """The DEFAULT path must stay the nominal 1025-A Molly reference, so every
    pre-existing caller of extract_pack is byte-identical."""
    assert EP.DEF_WINDOW == "lya_only"
    w = EP.window_spec()
    assert w["lam_rf_min"] == 1025.0
    # the canonical frozen inputs, unchanged
    from CDDF_analysis.hbi import ff_fp_estimator as FF
    assert w["molly_tsv"] == FF.DEF_MOLLY_TSV
    assert w["molly_tsv_172"] == EP.MOLLY_TSV_NHI172
    assert w["counts_tag"] == ""          # unchanged cache filenames


def test_lya_lyb_window_is_911_and_has_its_own_ingredients(EP):
    w = EP.window_spec("lya_lyb")
    assert w["lam_rf_min"] == 911.0
    o = EP.window_spec("lya_only")
    # every window-dependent ingredient must DIFFER between the two arms
    for key in ("molly_tsv", "molly_tsv_172", "forward_npz", "counts_tag"):
        assert w[key] != o[key], f"{key} is shared between the two windows"


def test_unknown_window_is_a_keyerror_not_a_silent_default(EP):
    with pytest.raises(KeyError):
        EP.window_spec("lyb_only")


def test_make_cfg_threads_lam_rf_min_from_the_window(EP, tmp_path):
    a = EP._make_cfg("2lpt0", str(tmp_path), window="lya_only")
    b = EP._make_cfg("2lpt0", str(tmp_path), window="lya_lyb")
    assert float(a.lam_rf_min) == 1025.0
    assert float(b.lam_rf_min) == 911.0
    # the completeness matrix must move WITH the window
    assert a.molly_tsv != b.molly_tsv
    assert "lya_lyb" in b.molly_tsv


@pytest.mark.parametrize("window", ["lya_only", "lya_lyb"])
def test_window_matching_guard_passes_on_the_registry(WS, window):
    """Each ingredient's OWN stamp must agree with the window's lam_rf_min:
    molly_summary.tsv for the two matrices, a .window.json sidecar for the
    forward response (the frozen NPZ carries no lam_rf stamp of its own)."""
    m = WS.assert_window_matched(window)
    lam = m["lam_rf_min"]
    assert set(m["ingredients"]) == {"molly_tsv", "molly_tsv_172",
                                     "forward_npz"}
    for key, got in m["ingredients"].items():
        assert got["lam_rf_min"] == lam, f"{key} stamps {got['lam_rf_min']}"


def test_window_matching_guard_fails_closed_on_a_mismatch(WS, EP,
                                                          monkeypatch):
    """MUTATION-STYLE: point lya_lyb at the lya_only completeness matrix. The
    guard must REFUSE, because that matrix stamps lam_rf_min=1025."""
    bad = dict(EP.ANALYSIS_WINDOWS["lya_lyb"])
    bad["molly_tsv"] = EP.ANALYSIS_WINDOWS["lya_only"]["molly_tsv"]
    monkeypatch.setitem(EP.ANALYSIS_WINDOWS, "lya_lyb", bad)
    monkeypatch.setattr(WS, "_extract_pack_module", lambda: EP)
    with pytest.raises(SystemExit, match="WINDOW MISMATCH"):
        WS.assert_window_matched("lya_lyb")


def test_window_matching_guard_fails_closed_on_a_missing_sidecar(
        WS, EP, monkeypatch, tmp_path):
    """A forward response with NO .window.json cannot be vouched for."""
    npz = tmp_path / "unsided_forward.npz"
    npz.write_bytes(b"")
    bad = dict(EP.ANALYSIS_WINDOWS["lya_only"])
    bad["forward_npz"] = str(npz)
    monkeypatch.setitem(EP.ANALYSIS_WINDOWS, "lya_only", bad)
    monkeypatch.setattr(WS, "_extract_pack_module", lambda: EP)
    with pytest.raises(SystemExit, match="sidecar"):
        WS.assert_window_matched("lya_only")


def test_molly_counts_cache_signature_keeps_the_1025_default():
    """``build_molly_counts_cache`` used to hardcode lam_rf_min=1025.0 while
    ``molly_tsv`` was free, so a 911-A matrix got 1025-A counts. The window is a
    parameter now, and its DEFAULT is still 1025.0 (byte-identical callers)."""
    import inspect
    from CDDF_analysis.hbi import ff_fp_estimator as FF
    sig = inspect.signature(FF.build_molly_counts_cache)
    assert "lam_rf_min" in sig.parameters
    assert sig.parameters["lam_rf_min"].default == 1025.0


class _CfgProbeStop(Exception):
    """Raised by the HBIConfig probe once it has captured the kwargs."""


def test_molly_counts_cache_threads_lam_rf_min_INTO_the_cut_config(monkeypatch):
    """BEHAVIOURAL, not source-grep: the caller's window must reach the
    HBIConfig the cut bundle is built from.

    A source-substring check is not enough — the same literal also appears in
    the npz stamp, so re-hardcoding the CFG line alone would pass it. This
    intercepts HBIConfig, records what it was actually asked for, and aborts
    before any catalog I/O (so the test stays sub-second and writes nothing).
    """
    from CDDF_analysis.hbi import ff_fp_estimator as FF
    from CDDF_analysis.hbi import cddf_catalog_hbi as CCH

    seen = {}

    def probe(**kw):
        seen.update(kw)
        raise _CfgProbeStop()

    monkeypatch.setattr(CCH, "HBIConfig", probe)
    with pytest.raises(_CfgProbeStop):
        FF.build_molly_counts_cache(
            out_path="/nonexistent/never_written.npz",
            molly_tsv=("/scratch/cavestru_root/cavestru0/mfho/"
                       "gl_prod_2lpt0_v1_20260526/figures_molly_nhi195/"
                       "lya_lyb/molly_matrix.tsv"),
            lam_rf_min=911.0)
    assert float(seen["lam_rf_min"]) == 911.0, (
        "build_molly_counts_cache built its cut bundle at "
        f"lam_rf_min={seen.get('lam_rf_min')} while the caller asked for 911.0 "
        "— the completeness numerator would not match its own matrix")
    assert "lya_lyb" in seen["molly_tsv"]


# ---------------------------------------------------------------------------
# (2) the pure reporting-window arithmetic
# ---------------------------------------------------------------------------
def _row(lo, hi, mu, obs):
    return dict(lo=lo, hi=hi, mu=mu, obs=obs)


@pytest.fixture
def by_nhat():
    """A hand-built by_nhat table. OUT-OF-WINDOW ROWS COME FIRST (the
    one-sided-support discipline: write the fixture so a missing lower bound
    cannot pass by accident).

    Bins are 0.1 dex on the schema grid. [19.7, 21.6] must select exactly the
    four rows at 19.7, 19.8, 21.4, 21.5 below; 19.5/19.6 are BELOW the floor and
    21.6/21.7 are ABOVE the ceiling.
    """
    return [
        _row(19.5, 19.6, 10.0, 40.0),     # below the reporting floor
        _row(19.6, 19.7, 20.0, 45.0),     # below (its hi == the floor)
        _row(19.7, 19.8, 100.0, 121.0),   # IN  z = (121-100)/10 = +2.1
        _row(19.8, 19.9, 25.0, 20.0),     # IN  z = (20-25)/5   = -1.0
        _row(21.4, 21.5, 400.0, 340.0),   # IN  z = (340-400)/20 = -3.0
        _row(21.5, 21.6, 9.0, 0.0),       # IN, but obs == 0 -> excluded from
                                          #     z_bin_max / chi2_dof
        _row(21.6, 21.7, 16.0, 8.0),      # ABOVE the ceiling (high-N tail)
        _row(21.7, 21.8, 4.0, 2.0),       # ABOVE
    ]


def test_select_bins_requires_FULL_containment(WS, by_nhat):
    sel = WS.select_bins(by_nhat, 19.7, 21.6)
    assert [(r["lo"], r["hi"]) for r in sel] == [
        (19.7, 19.8), (19.8, 19.9), (21.4, 21.5), (21.5, 21.6)]
    # the 21.6-21.7 bin STRADDLES nothing but starts AT the ceiling: excluded,
    # because its upper edge 21.7 > 21.6.
    assert all(r["hi"] <= 21.6 + 1e-9 for r in sel)
    # and the two sub-floor rows are gone
    assert all(r["lo"] >= 19.7 - 1e-9 for r in sel)


def test_select_bins_unbounded_sides(WS, by_nhat):
    assert len(WS.select_bins(by_nhat, None, None)) == len(by_nhat)
    lo_only = WS.select_bins(by_nhat, 21.6, None)
    assert [(r["lo"], r["hi"]) for r in lo_only] == [(21.6, 21.7), (21.7, 21.8)]
    hi_only = WS.select_bins(by_nhat, None, 19.7)
    assert [(r["lo"], r["hi"]) for r in hi_only] == [(19.5, 19.6), (19.6, 19.7)]


def test_window_metrics_reporting_window_hand_computed(WS, by_nhat):
    """Every number here is computed by hand from the fixture."""
    m = WS.window_metrics(by_nhat, 19.7, 21.6)
    assert m["n_bins"] == 4
    assert m["n_bins_occupied"] == 3          # the obs==0 bin drops out
    assert m["mu"] == pytest.approx(100.0 + 25.0 + 400.0 + 9.0)      # 534
    assert m["obs"] == pytest.approx(121.0 + 20.0 + 340.0 + 0.0)     # 481
    assert m["ratio"] == pytest.approx(534.0 / 481.0)
    # z_total uses ALL FOUR bins (empty bins count in the totals)
    assert m["z_total"] == pytest.approx((481.0 - 534.0) / math.sqrt(534.0))
    # z_bin_max / chi2_dof use only the three OCCUPIED bins: +2.1, -1.0, -3.0
    assert m["z_bin_max"] == pytest.approx(3.0)
    assert m["chi2_dof"] == pytest.approx((2.1 ** 2 + 1.0 ** 2 + 3.0 ** 2) / 3.0)


def test_window_metrics_high_n_tail_is_a_separate_object(WS, by_nhat):
    h = WS.window_metrics(by_nhat, 21.6, None)
    assert h["n_bins"] == 2
    assert h["mu"] == pytest.approx(20.0)     # 16 + 4
    assert h["obs"] == pytest.approx(10.0)    # 8 + 2
    assert h["ratio"] == pytest.approx(2.0)   # a 2.00x high-N EXCESS
    # and it is NOT inside the reporting window
    assert WS.window_metrics(by_nhat, 19.7, 21.6)["mu"] != h["mu"]


def test_window_metrics_empty_selection_is_nan_not_a_crash(WS, by_nhat):
    m = WS.window_metrics(by_nhat, 30.0, 31.0)
    assert m["n_bins"] == 0
    assert math.isnan(m["chi2_dof"]) and math.isnan(m["z_bin_max"])
    assert math.isnan(m["ratio"])


def test_closes_uses_only_the_three_gate_arms(WS, by_nhat):
    gate = WS.restated_gate_criteria()["gate_arms"]
    assert set(gate) == {"abs_z_total_max", "z_bin_max", "chi2_dof_max"}
    m = WS.window_metrics(by_nhat, 19.7, 21.6)
    v = WS.closes(m, gate)
    # chi2/dof = (4.41 + 1 + 9)/3 = 4.8033 > 3  ->  must FAIL on chi2 only
    assert v["closes"] is False
    assert len(v["failures"]) == 1 and v["failures"][0].startswith("chi2_dof")
    # and a table that closes must PASS
    tight = [_row(19.7, 19.8, 100.0, 101.0), _row(19.8, 19.9, 100.0, 99.0)]
    assert WS.closes(WS.window_metrics(tight, 19.7, 21.6), gate)["closes"]


def test_closes_REFUSES_an_EMPTY_reporting_window(WS):
    """FAIL-CLOSED, not fail-open. A selection that contains NO fully-contained
    bin cannot be gated, and the three gating arms all evaluate vacuously on
    it: z_bin_max / chi2_dof are non-finite (so both arms are SKIPPED) and
    z_total is 0/sqrt(1e-12) = 0.0 (so the |z_total| arm passes on a
    manufactured zero). ``closes`` must REFUSE.

    This branch's own recent history is a sequence of fail-open gate closures;
    a gate that says "closes" about nothing at all is the same defect.
    """
    m = WS.window_metrics([_row(19.5, 19.6, 10.0, 40.0)], WS.REPORT_LO,
                          WS.REPORT_HI)
    assert m["n_bins"] == 0                     # the reproduction
    assert m["z_total"] == 0.0                  # the manufactured pass
    assert math.isnan(m["z_bin_max"]) and math.isnan(m["chi2_dof"])
    v = WS.closes(m, WS.restated_gate_criteria()["gate_arms"])
    assert v["closes"] is False, (
        "closes() passed an EMPTY reporting window — every informative arm was "
        "skipped as non-finite and |z_total| passed on 0/sqrt(1e-12)")
    # the message must name the EMPTY selection specifically -- "no bin is
    # occupied" is a DIFFERENT vacuous shape (see the next test) and the two
    # must not be able to stand in for one another.
    assert any("selects NO fully-contained" in f for f in v["failures"]), \
        v["failures"]
    assert not any("n_bins_occupied" in f for f in v["failures"]), v["failures"]


def test_closes_REFUSES_when_NO_bin_is_occupied(WS):
    """The other vacuous shape: bins exist but every one has obs == 0, so the
    two per-bin arms are again non-finite and skipped. |z_total| alone is NOT a
    closure test (P4: the total is a level, the shape is the defect)."""
    m = WS.window_metrics([_row(19.7, 19.8, 9.0, 0.0),
                           _row(19.8, 19.9, 4.0, 0.0)],
                          WS.REPORT_LO, WS.REPORT_HI)
    assert m["n_bins"] == 2 and m["n_bins_occupied"] == 0
    assert abs(m["z_total"]) < 5.0              # would have passed vacuously
    v = WS.closes(m, WS.restated_gate_criteria()["gate_arms"])
    assert v["closes"] is False, (
        "closes() passed a window in which not one bin carries an observed "
        "count")
    assert any("n_bins_occupied" in f for f in v["failures"]), v["failures"]


def test_closes_REFUSES_a_NON_FINITE_gate_arm(WS):
    """Belt and braces: a non-finite arm is a refusal, never a skip. Built by
    hand rather than through window_metrics so the arms are non-finite
    INDEPENDENTLY of how many bins were selected."""
    gate = WS.restated_gate_criteria()["gate_arms"]
    base = dict(n_bins=4, n_bins_occupied=3, z_total=0.5, z_bin_max=1.0,
                chi2_dof=1.0)
    assert WS.closes(base, gate)["closes"] is True
    for arm in ("z_total", "z_bin_max", "chi2_dof"):
        bad = dict(base, **{arm: float("nan")})
        v = WS.closes(bad, gate)
        assert v["closes"] is False, f"{arm}=nan passed the gate"
        assert any(arm in f and "not finite" in f for f in v["failures"]), \
            (arm, v["failures"])


def test_ratio_span_tolerances_are_declared_UNRATIFIED(WS):
    """PI decision 8 explicitly declined to ratify these two, so they must be
    reported as measurements and must not appear among the gating arms."""
    g = WS.restated_gate_criteria()
    adv = g["advisory_tolerances"]
    assert set(adv) == {"ratio_span_by_z_max", "ratio_span_by_snr_max"}
    assert not (set(adv) & set(g["gate_arms"]))
    # advisory means advisory: UNRATIFIED and gating NOTHING
    for name, t in adv.items():
        assert t["authority_state"] == WS.UNRATIFIED, (name, t)
        assert t["gates"] is False, (name, t)


# ---------------------------------------------------------------------------
# (2b) GATE AUTHORITY — the 2026-08-05 retraction
#
# `restated_gate_criteria()` used to return the three closure thresholds under
# the key `ratified_arms`, with the docstring "THE THREE RATIFIED ARMS (PI
# decision 8)". PI decision 8, verbatim: "Ratify the fail-closed framework,
# matched-configuration SBC and chi2/dof <= 3 closure requirement. ... Also
# restate the MALFORMED |z| <= 5 criterion with its exact mathematical
# definition."  Calling a criterion MALFORMED and sending it back for
# restatement is the OPPOSITE of ratifying it. The sibling site was retracted
# in 6f9f998; these tests pin the corrected record HERE so it cannot come back.
#
# The correction is a LABEL, not a disarmament: the |z| arms still gate, and a
# test that let them be recorded as `gates=False` would be a false claim in the
# other direction.
# ---------------------------------------------------------------------------
def test_the_two_z_arms_are_NOT_recorded_as_RATIFIED(WS):
    """THE defect. Only chi2_dof_max was ratified; |z| <= 5 was called
    MALFORMED and sent back for restatement."""
    arms = WS.restated_gate_criteria()["gate_arms"]
    assert arms["chi2_dof_max"]["authority_state"] == WS.RATIFIED
    assert arms["chi2_dof_max"]["value"] == 3.0
    for name in ("abs_z_total_max", "z_bin_max"):
        assert arms[name]["authority_state"] == WS.RESTATED_NOT_RATIFIED, (
            f"{name} claims {arms[name]['authority_state']!r} — decision 8 "
            "called |z| <= 5 MALFORMED, which is not a ratification")
        assert arms[name]["value"] == 5.0
        # the PI's own word must be quoted where the claim used to be
        assert "MALFORMED" in arms[name]["pi_disposition"]


def test_the_retracted_ratified_arms_KEY_is_GONE(WS):
    """No key or value may assert ratification of the |z| arms. The retracted
    shape was a BARE `{name: float}` dict under the key `ratified_arms`."""
    g = WS.restated_gate_criteria()
    assert "ratified_arms" not in g
    assert "gate_tolerances_ratified" not in g
    # and no field may return a bare threshold under ANY name: every gating
    # number must arrive with its authority state attached
    for name, arm in g["gate_arms"].items():
        assert isinstance(arm, dict) and "authority_state" in arm, (name, arm)


def test_only_the_PI_ALLOW_LIST_may_carry_RATIFIED(WS):
    """The allow-list is exactly the three things decision 8 ratified, and
    nothing outside it may be recorded as RATIFIED."""
    assert set(WS.PI_RATIFIED_ITEMS) == {"fail_closed_framework",
                                         "matched_configuration_sbc",
                                         "chi2_dof_max"}
    g = WS.restated_gate_criteria()
    assert g["pi_ratified_items"] == list(WS.PI_RATIFIED_ITEMS)
    for group in ("gate_arms", "advisory_tolerances"):
        for name, arm in g[group].items():
            if arm["authority_state"] == WS.RATIFIED:
                assert name in WS.PI_RATIFIED_ITEMS, (group, name)


def test_no_field_named_RATIFIED_carries_anything_off_the_allow_list(WS):
    """The shape the widened import-time guard scans for: any field whose NAME
    contains "ratified" must contain ONLY allow-listed items. The retracted
    `ratified_arms` held two |z| thresholds that are not on the list."""
    g = WS.restated_gate_criteria()
    allowed = set(WS.PI_RATIFIED_ITEMS)
    hits = [f for f in g if "ratified" in f.lower()]
    assert hits, "the allow-list itself must be published for the guard to check"
    for f in hits:
        assert set(g[f]) <= allowed, (f, sorted(set(g[f]) - allowed))


def test_audit_gate_authority_REFUSES_the_retracted_ratified_arms_SHAPE(WS):
    """BEHAVIOURAL: the guard must refuse the exact record v1 emitted, not just
    be absent from the current one."""
    with pytest.raises(SystemExit) as e:
        WS.audit_gate_authority(dict(
            ratified_arms={"abs_z_total_max": 5.0, "z_bin_max": 5.0,
                           "chi2_dof_max": 3.0}))
    assert "abs_z_total_max" in str(e.value) and "z_bin_max" in str(e.value)
    # ... and it must ACCEPT the allow-list published under the same word
    WS.audit_gate_authority(dict(pi_ratified_items=list(WS.PI_RATIFIED_ITEMS)))


def test_audit_gate_authority_REFUSES_an_arm_claiming_RATIFIED_off_the_list(WS):
    with pytest.raises(SystemExit) as e:
        WS.audit_gate_authority(dict(gate_arms={
            "z_bin_max": dict(value=5.0, authority_state=WS.RATIFIED,
                              gates=True)}))
    assert "PI_RATIFIED_ITEMS" in str(e.value)


def test_audit_gate_authority_REFUSES_an_UNRATIFIED_tolerance_that_GATES(WS):
    """The other direction: a number decision 8 DECLINED may not refuse work."""
    with pytest.raises(SystemExit) as e:
        WS.audit_gate_authority(dict(advisory_tolerances={
            "ratio_span_by_z_max": dict(value=0.10,
                                        authority_state=WS.UNRATIFIED,
                                        gates=True)}))
    assert "gates=True" in str(e.value)


def test_audit_gate_authority_REFUSES_an_UNKNOWN_authority_state(WS):
    with pytest.raises(SystemExit) as e:
        WS.audit_gate_authority(dict(gate_arms={
            "z_bin_max": dict(value=5.0, authority_state="APPROVED",
                              gates=True)}))
    assert "authority_state" in str(e.value)


def test_the_unratified_z_arms_STILL_GATE_and_the_record_says_so(WS):
    """NOT a disarmament. Recording `gates=False` for an arm that really does
    refuse work would be as false as the PI claim, in the other direction."""
    g = WS.restated_gate_criteria()
    for name in ("abs_z_total_max", "z_bin_max", "chi2_dof_max"):
        assert g["gate_arms"][name]["gates"] is True, name
    # ... behaviourally: a metrics dict that busts ONLY the |z| arms must fail
    m = dict(n_bins=4, n_bins_occupied=3, z_total=99.0, z_bin_max=99.0,
             chi2_dof=1.0)
    v = WS.closes(m, g["gate_arms"])
    assert v["closes"] is False
    assert len(v["failures"]) == 2


def test_a_refusal_on_an_UNRATIFIED_arm_NAMES_it_as_unratified(WS):
    """That is the moment a PI needs to know a number has no authority behind
    it. The RATIFIED arm's message must NOT carry the tag."""
    g = WS.restated_gate_criteria()["gate_arms"]
    m = dict(n_bins=4, n_bins_occupied=3, z_total=99.0, z_bin_max=99.0,
             chi2_dof=99.0)
    f = WS.closes(m, g)["failures"]
    ztot = [s for s in f if s.startswith("|z_total|")][0]
    zbin = [s for s in f if s.startswith("z_bin_max")][0]
    chi2 = [s for s in f if s.startswith("chi2_dof")][0]
    assert "RESTATED_NOT_RATIFIED" in ztot and "RESTATED_NOT_RATIFIED" in zbin
    assert "RATIFIED" not in chi2, (
        "chi2/dof <= 3 IS ratified; tagging it would understate its authority")


def test_closes_REFUSES_a_BARE_threshold_with_no_authority_state(WS):
    """A caller must not be able to read a threshold without its authority
    state — that separability is what let `ratified_arms` launder the |z|
    numbers through the one arm in the dict that really was ratified."""
    m = dict(n_bins=4, n_bins_occupied=3, z_total=0.5, z_bin_max=1.0,
             chi2_dof=1.0)
    with pytest.raises(TypeError) as e:
        WS.closes(m, {"abs_z_total_max": 5.0, "z_bin_max": 5.0,
                      "chi2_dof_max": 3.0})
    assert "BARE THRESHOLD" in str(e.value)


def test_the_local_arm_name_MAPS_to_the_canonical_merge_name(WS):
    """This module calls the arm `abs_z_total_max`; the sibling streams call it
    `z_total_max`. The mapping must be explicit or the merged guard and the
    artifact will disagree about which arm is which."""
    arms = WS.restated_gate_criteria()["gate_arms"]
    assert arms["abs_z_total_max"]["canonical_name"] == "z_total_max"
    assert arms["z_bin_max"]["canonical_name"] == "z_bin_max"
    assert arms["chi2_dof_max"]["canonical_name"] == "chi2_dof_max"


def test_closes_reports_whether_the_verdict_needs_the_unratified_arms(WS):
    """Once two of three gating arms are unratified, "it does not close" is
    only load-bearing if it survives dropping them. Measured, not asserted."""
    g = WS.restated_gate_criteria()["gate_arms"]
    # busts every arm -> the RATIFIED arm alone still refuses
    hard = WS.closes(dict(n_bins=4, n_bins_occupied=3, z_total=99.0,
                          z_bin_max=99.0, chi2_dof=99.0), g)
    assert hard["closes"] is False
    assert hard["closes_on_pi_authority_only"] is False
    assert hard["failures_on_pi_authority_only"] == [
        s for s in hard["failures"] if s.startswith("chi2_dof")]
    # busts ONLY the unratified arms -> the conclusion WOULD depend on them
    soft = WS.closes(dict(n_bins=4, n_bins_occupied=3, z_total=99.0,
                          z_bin_max=99.0, chi2_dof=1.0), g)
    assert soft["closes"] is False
    assert soft["closes_on_pi_authority_only"] is True
    assert soft["failures_on_pi_authority_only"] == []


def test_marginal_block_measures_span_without_gating(WS):
    rows = [dict(lo=2.0, hi=2.1, mu=100.0, obs=90.0, ratio=1.10, z=1.0),
            dict(lo=2.1, hi=2.2, mu=90.0, obs=100.0, ratio=0.90, z=-1.0),
            dict(lo=2.2, hi=2.3, mu=5.0, obs=0.0, ratio=float("nan"), z=2.0)]
    m = WS.marginal_block(rows)
    assert m["n_occupied"] == 2                     # the obs==0 row drops out
    assert m["ratio_span"] == pytest.approx(0.20)
    assert m["z_max"] == pytest.approx(1.0)
    assert "closes" not in m and "gate" not in m    # measurement only


def test_direction_verdict_needs_unanimity_across_all_three_mocks(WS):
    assert WS.direction_verdict({"a": -1.0, "b": -2.0, "c": -0.5})["unanimous"]
    assert WS.direction_verdict({"a": -1.0, "b": -2.0, "c": -0.5})["sign"] == -1
    split = WS.direction_verdict({"a": -1.0, "b": +2.0, "c": -0.5})
    assert split["unanimous"] is False and split["sign"] == 0
    zero = WS.direction_verdict({"a": -1.0, "b": 0.0, "c": -0.5})
    assert zero["unanimous"] is False
    # ALL-ZERO must also be "no effect", not "unanimously zero" -- otherwise a
    # window with literally no measured effect reads as a unanimous result.
    allzero = WS.direction_verdict({"a": 0.0, "b": 0.0, "c": 0.0})
    assert allzero["unanimous"] is False and allzero["sign"] == 0


def test_protocol_is_fixed_and_names_the_reference_window(WS):
    ids = [i for i, _ in WS.PROTOCOL]
    assert ids == ["P1", "P2", "P3", "P4", "P5", "P6"]
    joined = " ".join(r for _, r in WS.PROTOCOL)
    assert "lya_only" in joined and "STANDARD REFERENCE" in joined
    assert str(WS.REPORT_LO) in joined and str(WS.REPORT_HI) in joined


def test_response_swap_covers_EVERY_resp_field(WS):
    """The attribution cross-fold swaps the response between two packs. If a new
    ``resp_*`` pack field is ever added and NOT listed, the cross-fold would
    leave part of the old response in place and silently mis-attribute the
    window effect. Assert the list is exhaustive against the pack dataclass."""
    import dataclasses
    # file-direct: the hbi_mcmc package __init__ imports jax, absent from gpdla
    PK = _load("_pack_under_test", "CDDF_analysis/hbi_mcmc/pack.py")
    fields = {f.name for f in dataclasses.fields(PK.ModelAPack)
              if f.name.startswith("resp_")}
    assert fields, "no resp_* fields found — the introspection broke"
    missing = fields - set(WS._RESP_KEYS)
    assert not missing, f"response fields not swapped by the cross-fold: {missing}"
    extra = set(WS._RESP_KEYS) - fields
    assert not extra, f"_RESP_KEYS names non-existent pack fields: {extra}"


def test_pilot_cost_parses_the_runners_own_log_line(WS, tmp_path, monkeypatch):
    """ARM 2's per-spectrum cost is read from the finder's OWN log line, not
    retyped. A silently-unparsed log must return None, never a wrong number."""
    monkeypatch.setattr(WS, "ARM2_DIR", str(tmp_path))
    d = tmp_path / "run_tagX" / "logs"
    d.mkdir(parents=True)
    log = d / "pilot_tagX.log"
    log.write_text(
        "INFO:dlasearch.py:266:dlasearch_mock: Completed processing of 4 "
        "spectra from /some/spectra-16-39.fits in 328.19s\n")
    c = WS._pilot_cost("tagX")
    assert c["n_spectra"] == 4
    assert c["inference_seconds"] == pytest.approx(328.19)
    assert c["seconds_per_spectrum"] == pytest.approx(328.19 / 4)
    # a log with no such line yields None -- never a fabricated cost
    log.write_text("INFO: nothing useful here\n")
    assert WS._pilot_cost("tagX") is None
    # a missing log yields None
    assert WS._pilot_cost("tagY") is None


def test_basis_resolution_status_is_explicit_about_decision_3(WS):
    """The study must SAY it did not implement the 0.2-dex basis rather than
    let a reader assume the PI-adopted configuration was fully realised."""
    s = WS.BASIS_RESOLUTION_STATUS
    assert WS.BASIS_DEX == 0.1
    assert "NOT implemented" in s and "0.2-dex" in s
    assert "validate_pack" in s          # names the concrete blocker


# ---------------------------------------------------------------------------
# (3) THE WINDOW-PROPAGATION AND FAIL-CLOSED PATHS THAT HAD NO TEST AT ALL
#
# Referee defect 1 (2026-07-29): an independent 18-mutant battery run against
# this file + tests/test_ff_fp_estimator.py (58-passing baseline) left EIGHT
# single-line mutants ALIVE. The worst was the `build_fp_block` window
# re-derivation -- one of the two "real defects found and fixed" in the study --
# which had no test whatsoever (`grep -rn build_fp_block tests/` returned
# nothing).
#
# Everything below drives the COMMITTED code path; only the heavy ingredients
# (the loa-0 dlacat + product, the molly counts cache, the packs, the fold) are
# injected. Each test names the mutant it kills.
# ---------------------------------------------------------------------------
LYA_REST_A = 1215.67


def _zdla_for_lam_rest(lam_rest, z_qso):
    """The z_DLA that lands a system at rest wavelength ``lam_rest`` in a
    z_qso quasar, inverting extract_pack's own
    ``lam_rest = LYA_REST * (1 + z_dla) / (1 + z_qso)``."""
    return (1.0 + z_qso) * lam_rest / LYA_REST_A - 1.0


class _FakeLoa0FP:
    """Stands in for cddf_catalog_hbi.Loa0FP for the FP-block guards.

    ``_cell_idx`` collapses everything into molly cell (0, 0) so the committed
    EXACT re-bin guard against ``n_fp_molly`` is satisfied by construction and
    the test is about the WINDOW, not about molly cell arithmetic.
    """

    def __init__(self, n_op_at_product_window, n_binned_at_product_window):
        self.n_fp_molly = np.array([[float(n_op_at_product_window)]])
        # logN_lo is searched for 19.5; put one sub-floor row and one at 19.5 so
        # the committed `n_fp_fine[b195:]` slice is exactly the in-grid total.
        self.logN_lo = np.array([19.0, 19.5])
        self.n_fp_fine = np.array([[0.0], [float(n_binned_at_product_window)]])
        self.n_sl_loa0 = 1000.0

    def _cell_idx(self, nhi, snr):
        n = len(np.atleast_1d(nhi))
        return np.zeros(n, dtype=int), np.zeros(n, dtype=int)

    @classmethod
    def _maker(cls, n_op, n_binned):
        class _C:
            @staticmethod
            def from_product(path):
                return cls(n_op, n_binned)
        return _C


@pytest.fixture
def fp_catalog(tmp_path):
    """A synthetic loa-0 FP catalog + product, with the OUT-OF-WINDOW ROWS
    FIRST (the project's one-sided-support fixture discipline).

    The rows are placed by REST wavelength so the two analysis windows select
    genuinely different subsets:

      lam_rest  | >= 1025 (lya_only) | >= 911 (lya_lyb) | in the c grid
      ----------+--------------------+------------------+---------------
        900     | no                 | no               | -
       1100     | no (SNR 1.0 cut)   | no               | -
       1100     | no (P_DLA 0.5 cut) | no               | -
       1100     | YES  N=19.0        | YES              | NO (< 19.5)
       1100     | YES  N=20.0        | YES              | yes
       1030     | YES  N=21.0        | YES              | yes
        950     | no                 | YES  N=20.5      | yes
        920     | no                 | YES  N=19.6      | yes

    so at the PRODUCT window (1025 A) op = 3 rows and 2 are binned, while at
    911 A op = 5 rows and 4 are binned.
    """
    z_qso = 3.0
    spec = [
        # (lam_rest, nhi, snr, p_dla)
        (900.0, 20.0, 5.0, 1.0),      # outside BOTH windows
        (1100.0, 20.0, 1.0, 1.0),     # fails SNR > 2
        (1100.0, 20.0, 5.0, 0.5),     # fails P_DLA > 0.99
        (1100.0, 19.0, 5.0, 1.0),     # in op, BELOW the c grid
        (1100.0, 20.0, 5.0, 1.0),     # in both windows
        (1030.0, 21.0, 5.0, 1.0),     # in both windows
        (950.0, 20.5, 5.0, 1.0),      # lya_lyb ONLY
        (920.0, 19.6, 5.0, 1.0),      # lya_lyb ONLY
    ]
    cat = dict(
        SNR_REDSIDE=np.array([r[2] for r in spec], float),
        P_DLA=np.array([r[3] for r in spec], float),
        NHI=np.array([r[1] for r in spec], float),
        Z_DLA=np.array([_zdla_for_lam_rest(r[0], z_qso) for r in spec], float),
        Z_QSO=np.full(len(spec), z_qso, float),
    )
    product = str(tmp_path / "loa0_fp_product.npz")
    np.savez(product, snr_min=2.0, p_dla_min=0.99, lya_only_lam_rf_min=1025.0)
    return dict(cat=cat, product=product,
                n_op_prod=3, n_binned_prod=2, n_op_req=5, n_binned_req=4)


def _wire_fp(EP, monkeypatch, fp_catalog):
    monkeypatch.setattr(EP, "load_loa0_fp_catalog",
                        lambda p: fp_catalog["cat"])
    monkeypatch.setattr(EP, "Loa0FP",
                        _FakeLoa0FP._maker(fp_catalog["n_op_prod"],
                                           fp_catalog["n_binned_prod"]))


def test_build_fp_block_REDERIVES_the_forest_FP_background_at_the_REQUESTED_window(
        EP, monkeypatch, fp_catalog, tmp_path):
    """KILLS MUTANT A: ``lam_req = float(w["lam_rf_min"])`` -> ``float(lya_min)``.

    The committed loa-0 FP product was built at 1025 A. Under the 911-A window
    the forest-FP background MUST be re-derived at 911 A: the Lyb region is
    exactly where forest false positives are worst, so silently reusing the
    1025-A background under-counts the wider arm's own FPs and biases the whole
    window comparison in the wider window's favour.
    """
    _wire_fp(EP, monkeypatch, fp_catalog)
    kw = dict(loa0_out=str(tmp_path / "loa0_dlacat"),
              product_path=fp_catalog["product"])

    fp_only, _l, prov_only = EP.build_fp_block(window="lya_only", **kw)
    fp_lyb, _l2, prov_lyb = EP.build_fp_block(window="lya_lyb", **kw)

    # the reference arm: the product's own window, unchanged
    assert prov_only["op_cut"]["lam_rf_min"] == 1025.0
    assert int(fp_only.sum()) == fp_catalog["n_binned_prod"] == 2

    # the wide arm: RE-DERIVED at 911 A
    assert prov_lyb["op_cut"]["lam_rf_min"] == 911.0, (
        "build_fp_block binned the FP background at "
        f"{prov_lyb['op_cut']['lam_rf_min']} while the caller asked for the "
        "lya_lyb (911 A) window — the wider arm would under-count its own "
        "forest FPs")
    assert int(fp_lyb.sum()) == fp_catalog["n_binned_req"] == 4
    assert prov_lyb["n_fp_op_total"] == fp_catalog["n_op_req"] == 5
    # and BOTH totals are reported, so the re-derivation is auditable
    assert prov_lyb["n_fp_binned_total_at_product_window"] == 2
    assert prov_lyb["n_fp_op_total_at_product_window"] == 3
    # the guard is still evaluated at the product's window in BOTH arms
    assert "1025.0" in prov_lyb["molly_rebin_guard"]


def test_build_fp_block_molly_rebin_guard_fails_closed_on_a_drifted_catalog(
        EP, monkeypatch, fp_catalog, tmp_path):
    """The EXACT re-bin guard: if the raw loa-0 dlacat no longer reproduces the
    committed product's ``n_fp_molly``, refuse."""
    monkeypatch.setattr(EP, "load_loa0_fp_catalog", lambda p: fp_catalog["cat"])
    monkeypatch.setattr(EP, "Loa0FP", _FakeLoa0FP._maker(99, 2))   # wrong count
    with pytest.raises(RuntimeError, match="n_fp_molly"):
        EP.build_fp_block(loa0_out=str(tmp_path / "d"),
                          product_path=fp_catalog["product"])


def test_build_fp_block_n_fp_fine_crosscheck_fails_closed(
        EP, monkeypatch, fp_catalog, tmp_path):
    """The SECOND committed cross-check: the schema-grid FP total must equal the
    committed product's z-windowed fine-grid total over N >= 19.5. n_fp_molly
    agrees here, so only this arm can catch the drift."""
    monkeypatch.setattr(EP, "load_loa0_fp_catalog", lambda p: fp_catalog["cat"])
    monkeypatch.setattr(EP, "Loa0FP",
                        _FakeLoa0FP._maker(fp_catalog["n_op_prod"], 7))
    with pytest.raises(RuntimeError, match="n_fp_fine"):
        EP.build_fp_block(loa0_out=str(tmp_path / "d"),
                          product_path=fp_catalog["product"])


def test_build_fp_block_SUPERSET_guard_refuses_a_wider_window_with_FEWER_FPs(
        EP, monkeypatch, fp_catalog, tmp_path):
    """KILLS MUTANT N: the ``lam_req < lya_min and fp_counts < fp_prod`` guard
    -> ``if False:``.

    Widening lam_rf_min from 1025 A to 911 A can only ADD forest FPs: the
    selection is a strict superset. If the re-binning ever returns FEWER, the
    binning itself is broken and the number is meaningless. The fault is
    INJECTED at the binning layer (``_idx`` loses the wider window's rows) —
    with consistent inputs the condition is unreachable, which is exactly why
    it had no test.
    """
    _wire_fp(EP, monkeypatch, fp_catalog)
    real_idx = EP._idx
    state = {"n_nhat_calls": 0}

    def flaky_idx(edges, x):
        if edges is EP.NHAT_EDGES:
            state["n_nhat_calls"] += 1
            if state["n_nhat_calls"] >= 2:      # the REQUESTED window's re-bin
                return np.full(len(np.atleast_1d(x)), -1, dtype=int)
        return real_idx(edges, x)

    monkeypatch.setattr(EP, "_idx", flaky_idx)
    with pytest.raises(RuntimeError, match="FP window GUARD failed"):
        EP.build_fp_block(loa0_out=str(tmp_path / "d"),
                          product_path=fp_catalog["product"],
                          window="lya_lyb")
    assert state["n_nhat_calls"] >= 2, "the requested-window re-bin never ran"


def _fake_counts(path):
    """A minimal ff_fp_estimator.load_molly_counts return value."""
    return dict(cmp_nfound=np.full((8, 2), 5.0), cmp_nfid=np.full((8, 2), 10.0),
                nhi_edges=np.array([19.5, 20.0, np.inf]),
                snr_edges=np.array([0., 1., 2., 3., 4., 5., 6., 7., np.inf]),
                max_c_diff=0.0, path=path)


def test_molly_counts_cache_PATH_is_window_scoped(EP, monkeypatch):
    """KILLS MUTANT O: ``if w["counts_tag"]:`` -> ``if False:``.

    The counts cache is regenerated from a cut bundle at a SPECIFIC
    lam_rf_min. Reading the lya_only cache while claiming to be lya_lyb is a
    silent mixed-window completeness numerator, and the two caches genuinely
    differ (up to 0.141 in C). The lya_only tag is "" so its path must stay
    byte-identical to the pre-window default.
    """
    from CDDF_analysis.hbi import ff_fp_estimator as FF
    seen = []
    monkeypatch.setattr(FF, "load_molly_counts",
                        lambda p: (seen.append(p), _fake_counts(p))[1])

    _b, prov_only, _m = EP.load_molly_counts_block(window="lya_only")
    _b, prov_lyb, _m = EP.load_molly_counts_block(window="lya_lyb")

    assert seen[0] == FF.DEF_MOLLY_COUNTS, "the lya_only default path moved"
    assert seen[1] != seen[0], (
        "load_molly_counts_block read the SAME counts cache for both analysis "
        f"windows ({seen[1]}) — lya_lyb would silently reuse the lya_only "
        "completeness numerator")
    assert os.path.basename(seen[1]).endswith("_lya_lyb.npz")
    assert prov_only["lam_rf_min"] == 1025.0
    assert prov_lyb["lam_rf_min"] == 911.0
    assert prov_lyb["path"] == seen[1]


def test_molly_counts_cache_is_BUILT_at_the_windows_lam_rf_min_and_tsv(
        EP, monkeypatch):
    """The other half of mutant O: when the window-scoped cache is ABSENT, it
    must be BUILT from that window's own matrix at that window's lam_rf_min."""
    from CDDF_analysis.hbi import ff_fp_estimator as FF
    built = {}
    calls = {"n": 0}

    def fake_load(p):
        calls["n"] += 1
        return None if calls["n"] == 1 else _fake_counts(p)

    monkeypatch.setattr(FF, "load_molly_counts", fake_load)
    monkeypatch.setattr(FF, "build_molly_counts_cache",
                        lambda **kw: built.update(kw))
    EP.load_molly_counts_block(window="lya_lyb")
    assert float(built["lam_rf_min"]) == 911.0
    assert "lya_lyb" in built["molly_tsv"]
    assert os.path.basename(built["out_path"]).endswith("_lya_lyb.npz")


class _WinCfg:
    def __init__(self, lam_rf_min):
        self.lam_rf_min = lam_rf_min
        self.snr_min = 2.0


def test_extract_pack_REFUSES_a_cached_bundle_cut_at_another_window(
        EP, tmp_path):
    """KILLS MUTANT D: ``if window_verified and abs(...)`` -> ``if False and ...``.

    The detection-side bundle cache is keyed on the mock only (the window lives
    on ``frozen``), so an injected or stale bundle cut at a different
    lam_rf_min would produce a MIXED-WINDOW pack: 911-A calibration folded
    against a 1025-A absorber/pathlength selection. That number looks fine and
    means nothing.
    """
    frozen = dict(analysis_window="lya_lyb",
                  _bundles={"2lpt0": dict(cfg=_WinCfg(1025.0))})
    with pytest.raises(RuntimeError, match="window GUARD failed"):
        EP.extract_pack("2lpt0", str(tmp_path), frozen)


def test_extract_pack_window_guard_PASSES_a_matched_bundle(EP, tmp_path):
    """POSITIVE leg: a bundle cut at the frozen calibration's own lam_rf_min
    must get PAST the guard (it then fails later on the deliberately empty
    bundle). Without this, mutant D could be 'killed' by a guard that refuses
    everything."""
    frozen = dict(analysis_window="lya_lyb",
                  _bundles={"2lpt0": dict(cfg=_WinCfg(911.0))})
    with pytest.raises(Exception) as ei:
        EP.extract_pack("2lpt0", str(tmp_path), frozen)
    assert "window GUARD failed" not in str(ei.value), (
        "the window guard refused a correctly-matched bundle")


def test_window_guard_fails_closed_on_a_sidecar_that_stamps_the_WRONG_window(
        WS, EP, monkeypatch, tmp_path):
    """KILLS MUTANT J: the forward-response sidecar VALUE-mismatch branch of
    ``assert_window_matched`` -> ``if False:``.

    A sidecar that EXISTS but stamps the wrong lam_rf_min is the realistic
    failure (copy the lya_only response into the study directory and write the
    sidecar from the wrong source). The existing tests covered a MISSING
    sidecar only, so the value comparison itself was dead code.
    """
    npz = tmp_path / "forward_response_wrongwindow.npz"
    npz.write_bytes(b"")
    with open(str(npz) + ".window.json", "w") as f:
        json.dump(dict(lam_rf_min=1025.0, provenance="deliberately mismatched"),
                  f)
    bad = dict(EP.ANALYSIS_WINDOWS["lya_lyb"])
    bad["forward_npz"] = str(npz)
    monkeypatch.setitem(EP.ANALYSIS_WINDOWS, "lya_lyb", bad)
    monkeypatch.setattr(WS, "_extract_pack_module", lambda: EP)
    with pytest.raises(SystemExit, match="WINDOW MISMATCH") as ei:
        WS.assert_window_matched("lya_lyb")
    assert "sidecar" in str(ei.value) and "1025.0" in str(ei.value)


# ---------------------------------------------------------------------------
# (4) THE THREE REFUSE-TO-STAMP GATES INSIDE phase_selftest
#
# These are the gates that make the emitted artifact trustworthy, and all three
# were dead code as far as the suite was concerned (mutants E, F, I). They live
# inside a monolithic phase that loads real packs and folds them through jax, so
# they are exercised here by DRIVING phase_selftest with every heavy dependency
# injected -- the guard statements themselves are the committed ones.
# ---------------------------------------------------------------------------
def _fake_by_nhat():
    """Two occupied bins inside the PI reporting window, closing comfortably."""
    return [dict(lo=19.7, hi=19.8, mu=100.0, obs=101.0),
            dict(lo=19.8, hi=19.9, mu=100.0, obs=99.0)]


_FAKE_FULL = dict(total_ratio=1.0, chi2_dof=0.01, z_total=0.0, n_bins=2)


class _FakePack:
    def __init__(self, window, commit):
        self.provenance = dict(
            code_commit=commit,
            analysis_window=dict(name=window),
            molly_counts=dict(path="<synthetic>"), fp=dict(product="<synthetic>"))
        self.n_b = 29
        self.n_c = 29
        self.ntrue_edges = np.array([19.5, 19.6])
        self.nhat_edges = np.array([19.5, 19.6])
        self.counts = np.array([1.0])
        self.truth_counts = np.array([1.0])
        self.dX = np.array([1.0])


class _FakeFS:
    """forward_selftest stand-in: selftest + ratio_tables."""

    @staticmethod
    def selftest(pack, resp_clamp="both"):
        return dict(resp_clamp=resp_clamp)

    @staticmethod
    def ratio_tables(res, pack):
        marg = [dict(lo=2.0, hi=2.5, mu=100.0, obs=101.0, ratio=100.0 / 101.0,
                     z=0.1)]
        return dict(by_nhat=_fake_by_nhat(), by_z=marg,
                    by_snr=[dict(s=5, mu=100.0, obs=101.0,
                                 ratio=100.0 / 101.0, z=0.1)])


def _drive_selftest(WS, monkeypatch, tmp_path, pack_window=None,
                    commit="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
                    gate_result=None):
    """Run the REAL ``phase_selftest`` against injected packs / fold / gate.

    ``pack_window=None`` means "each pack honestly carries the window it is
    filed under"; a fixed string makes every pack claim that window (the
    mixed-window failure). ``gate_result`` overrides the committed
    forward_closure_gate's return so the cross-check can be made to disagree.
    """
    import types
    monkeypatch.setattr(WS, "MOCKS", ["2lpt0"])
    monkeypatch.setattr(WS, "WINDOWS", ["lya_only", "lya_lyb"])
    monkeypatch.setattr(WS, "CLAMPS", ["both"])
    monkeypatch.setattr(WS, "PACKDIR", str(tmp_path))
    monkeypatch.setattr(WS, "OUT", str(tmp_path / "study.json"))
    monkeypatch.setattr(WS, "full_sha", lambda: commit)
    monkeypatch.setattr(WS, "dirty", lambda: False)
    # the metadata block records the branch via git; stub it so the test is
    # hermetic (it must pass in an exported tree too, e.g. under a mutation
    # harness that builds from `git archive HEAD`).
    monkeypatch.setattr(WS.subprocess, "check_output",
                        lambda *a, **k: "test-branch\n")
    monkeypatch.setattr(WS, "_FS", lambda: _FakeFS)
    monkeypatch.setattr(WS, "load_pilot", lambda: dict(status="NOT RUN"))
    monkeypatch.setattr(WS, "build_verdict",
                        lambda rows, packmeta, pilot=None: dict(stub=True))
    monkeypatch.setattr(WS, "response_attribution", lambda gate: dict(stub=True))
    monkeypatch.setattr(WS, "assert_window_matched",
                        lambda w: dict(window=w, lam_rf_min=0.0))

    def fake_load_pack(path):
        base = os.path.basename(path)
        win = "lya_lyb" if "winlya_lyb" in base else "lya_only"
        return _FakePack(pack_window or win, commit)

    monkeypatch.setattr(WS, "load_pack", fake_load_pack)

    # the committed cross-check imports run_posterior INSIDE the function; the
    # real module needs jax, so inject the package + module.
    pkg = types.ModuleType("CDDF_analysis.hbi_mcmc")
    pkg.__path__ = []
    rp = types.ModuleType("CDDF_analysis.hbi_mcmc.run_posterior")
    res = dict(_FAKE_FULL if gate_result is None else gate_result)
    res.setdefault("pass", True)
    res.setdefault("failures", [])
    rp.forward_closure_gate = lambda pack, resp_clamp="both": res
    pkg.run_posterior = rp
    monkeypatch.setitem(sys.modules, "CDDF_analysis.hbi_mcmc", pkg)
    monkeypatch.setitem(sys.modules, "CDDF_analysis.hbi_mcmc.run_posterior", rp)
    return WS.phase_selftest()


def test_phase_selftest_HAPPY_PATH_stamps_and_writes(WS, monkeypatch, tmp_path):
    """POSITIVE leg for all three gates below: with honest packs, an agreeing
    cross-check and a clean stamp, the phase must actually emit. Without this a
    guard that refuses unconditionally would 'kill' every mutant."""
    out = _drive_selftest(WS, monkeypatch, tmp_path)
    assert os.path.exists(str(tmp_path / "study.json"))
    audit = out["metadata"]["pack_stamp_audit"]
    assert audit["any_pack_dirty"] is False
    assert audit["packs_match_selftest_commit"] is True
    assert out["committed_gate_crosscheck"]["agrees"] is True
    assert set(out["arm1_analysis_window"]) == {"2lpt0|lya_only|clamp=both",
                                               "2lpt0|lya_lyb|clamp=both"}
    assert out["arm1_analysis_window"]["2lpt0|lya_only|clamp=both"][
        "primary_closes"]["closes"] is True
    # the post-hoc amendments (referee defect 3) must reach the ARTIFACT, and
    # must stay OUT of the pre-registered rule list (P6)
    prot = out["protocol"]
    assert [a["id"] for a in prot["post_hoc_amendments"]] == ["A1", "A2"]
    assert [r["id"] for r in prot["rules"]] == ["P1", "P2", "P3", "P4", "P5",
                                               "P6"]
    assert "AFTER the outcome was seen" in prot["post_hoc_amendments_note"]
    # ... and the scale-free statistic must be carried per configuration
    rep = out["arm1_analysis_window"]["2lpt0|lya_only|clamp=both"][
        "primary_reporting_window"]
    assert "rms_frac_dev" in rep and np.isfinite(rep["rms_frac_dev"])


def test_phase_selftest_REFUSES_a_pack_filed_under_the_WRONG_window(
        WS, monkeypatch, tmp_path):
    """KILLS MUTANT E: ``if aw.get("name") != window:`` -> ``if False:``.

    The pack FILENAME carries the window; so does its provenance. If they
    disagree, the study would compare two packs built at the same lam_rf_min
    while labelling them as the two arms — i.e. it would measure nothing and
    report a window effect of zero.
    """
    with pytest.raises(SystemExit, match="analysis_window"):
        _drive_selftest(WS, monkeypatch, tmp_path, pack_window="lya_only")


def test_phase_selftest_REFUSES_packs_extracted_from_a_DIRTY_tree(
        WS, monkeypatch, tmp_path):
    """KILLS MUTANT F: ``if stamp_audit["any_pack_dirty"]:`` -> ``if False:``.

    A `-dirty` pack stamp means the artifact's provenance cannot be reproduced
    from any commit. The project rule is that a headline's provenance is a
    committed routine plus a git stamp, so this must refuse rather than record.
    """
    with pytest.raises(SystemExit, match="DIRTY"):
        _drive_selftest(WS, monkeypatch, tmp_path,
                        commit="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef-dirty")


def test_phase_selftest_REFUSES_when_the_committed_gate_xcheck_DISAGREES(
        WS, monkeypatch, tmp_path):
    """KILLS MUTANT I: ``if not xcheck["agrees"]:`` -> ``if False:``.

    The study computes its own closure arithmetic inline. The only thing tying
    that arithmetic to the committed ``forward_closure_gate`` is this
    cross-check, so a disagreement must block the stamp — otherwise the
    artifact would silently carry a second, unvalidated implementation.
    """
    with pytest.raises(SystemExit, match="REFUSING to stamp"):
        _drive_selftest(WS, monkeypatch, tmp_path,
                        gate_result=dict(_FAKE_FULL, chi2_dof=99.0))
    # ... and the disagreement must be detected on EACH compared quantity
    for bad in (dict(total_ratio=1.5), dict(z_total=7.0)):
        with pytest.raises(SystemExit, match="REFUSING to stamp"):
            _drive_selftest(WS, monkeypatch, tmp_path,
                            gate_result=dict(_FAKE_FULL, **bad))


# ---------------------------------------------------------------------------
# (5) THE SCALE-FREE DISCRIMINATOR AND THE P2 SAMPLE-SIZE CONFOUND
#
# Referee defect 3 (2026-07-30): P2 (chi2/dof over the reporting window) is
# CONFOUNDED WITH SAMPLE SIZE. The lya_lyb arm carries 20-22% more counts, and
# chi2/dof scales LINEARLY with counts at fixed fractional residual shape, so
# ~1/3 of each quoted "+30 to +43 chi2/dof" is statistical power. The artifact
# disclosed exactly this confound for the marginal max|z| and said nothing about
# it for P2.
# ---------------------------------------------------------------------------
def test_rms_frac_dev_is_hand_computed_on_the_fixture(WS, by_nhat):
    """D = sqrt( sum_c obs_c (mu_c/obs_c - 1)^2 / sum_c obs_c ) over the
    OCCUPIED bins of the reporting window. Hand-computed from the fixture."""
    m = WS.window_metrics(by_nhat, WS.REPORT_LO, WS.REPORT_HI)
    # occupied reporting-window bins: (100,121), (25,20), (400,340)
    num = (121.0 * (100.0 / 121.0 - 1.0) ** 2
           + 20.0 * (25.0 / 20.0 - 1.0) ** 2
           + 340.0 * (400.0 / 340.0 - 1.0) ** 2)
    den = 121.0 + 20.0 + 340.0
    assert m["rms_frac_dev"] == pytest.approx(math.sqrt(num / den))
    # the obs == 0 bin is excluded (mu/obs is undefined there), exactly as for
    # chi2_dof / z_bin_max
    assert m["n_bins_occupied"] == 3


def test_rms_frac_dev_is_INVARIANT_under_a_common_counts_rescaling(WS):
    """THE WHOLE POINT. Scale every mu and obs by L -- the same fractional
    residual shape with L times the counts -- and:
        * chi2_dof scales by EXACTLY L (this is the confound), while
        * rms_frac_dev does not move at all.
    Without this property the measure could not separate "more counts" from
    "worse model" and the restated recommendation would be unfounded.
    """
    base = [_row(19.7, 19.8, 100.0, 110.0), _row(19.8, 19.9, 400.0, 380.0),
            _row(19.9, 20.0, 50.0, 44.0)]
    L = 1.222                                   # the MEASURED counts ratio
    scaled = [_row(r["lo"], r["hi"], r["mu"] * L, r["obs"] * L) for r in base]
    a = WS.window_metrics(base, WS.REPORT_LO, WS.REPORT_HI)
    b = WS.window_metrics(scaled, WS.REPORT_LO, WS.REPORT_HI)
    assert b["obs"] == pytest.approx(L * a["obs"])
    assert b["chi2_dof"] == pytest.approx(L * a["chi2_dof"], rel=1e-12), (
        "chi2_dof must scale LINEARLY with counts — that is the confound being "
        "disclosed")
    assert b["rms_frac_dev"] == pytest.approx(a["rms_frac_dev"], rel=1e-12), (
        "rms_frac_dev moved under a pure counts rescaling — it is not "
        "scale-free and cannot be the primary discriminator")


def test_rms_frac_dev_still_MOVES_when_the_residual_SHAPE_worsens(WS):
    """POWER: scale-free must not mean insensitive. Doubling every fractional
    deviation at FIXED counts must roughly double D."""
    base = [_row(19.7, 19.8, 110.0, 100.0), _row(19.8, 19.9, 380.0, 400.0)]
    worse = [_row(19.7, 19.8, 120.0, 100.0), _row(19.8, 19.9, 360.0, 400.0)]
    a = WS.window_metrics(base, WS.REPORT_LO, WS.REPORT_HI)["rms_frac_dev"]
    b = WS.window_metrics(worse, WS.REPORT_LO, WS.REPORT_HI)["rms_frac_dev"]
    assert b == pytest.approx(2.0 * a, rel=1e-12)


def test_rms_frac_dev_of_an_unoccupied_window_is_nan(WS, by_nhat):
    assert math.isnan(WS.window_metrics(by_nhat, 30.0, 31.0)["rms_frac_dev"])
    assert math.isnan(WS.rms_frac_dev([_row(19.7, 19.8, 9.0, 0.0)]))


def test_rms_frac_dev_is_NOT_a_gate_arm(WS, by_nhat):
    """PI decision 8 ratified chi2/dof <= 3 and nothing else. No tolerance on
    this statistic has been calibrated, so it must not gate."""
    g = WS.restated_gate_criteria()
    assert "rms_frac_dev" not in g["gate_arms"]
    m = WS.window_metrics(by_nhat, WS.REPORT_LO, WS.REPORT_HI)
    v = WS.closes(m, g["gate_arms"])
    assert not any("rms_frac_dev" in f for f in v["failures"])


def _confound_rows(chi2_only, chi2_lyb, obs_only, obs_lyb, rms_only, rms_lyb):
    """A minimal `rows` table shaped like phase_selftest's, for the confound
    decomposition only."""
    out = {}
    for mock in ("2lpt0", "london0", "saclay0"):
        for win, c2, ob, rd in (("lya_only", chi2_only, obs_only, rms_only),
                                ("lya_lyb", chi2_lyb, obs_lyb, rms_lyb)):
            block = dict(primary_reporting_window=dict(
                chi2_dof=c2, obs=ob, rms_frac_dev=rd, ratio=1.0),
                high_n_above_21p6=dict(ratio=1.1))
            for clamp in ("both", "hi"):
                out[f"{mock}|{win}|clamp={clamp}"] = block
    return out


def test_p2_power_confound_decomposes_the_chi2_jump(WS, monkeypatch):
    """The 2LPT-0 numbers, hand-checked: 67086 -> 82008 counts is L = 1.2224, so
    pure power scaling predicts 63.652 x L = 77.80 of the 106.586 observed, i.e.
    14.15 of the 42.93 rise (33%) is statistical power alone."""
    monkeypatch.setattr(WS, "CLAMPS", ["both"])
    rows = _confound_rows(63.65206298413061, 106.58576787999633,
                          67086.0, 82008.0, 0.1193, 0.1366)
    c = WS.p2_power_confound(rows)
    p = c["per_clamp"]["both"]["2lpt0"]
    assert p["counts_ratio_L"] == pytest.approx(82008.0 / 67086.0)
    assert p["counts_ratio_L"] == pytest.approx(1.2224, abs=5e-5)
    assert p["chi2_dof_lya_lyb_predicted_by_pure_power_scaling"] == \
        pytest.approx(77.80, abs=0.02)
    assert p["delta_chi2_dof_observed"] == pytest.approx(42.93, abs=0.01)
    assert p["delta_chi2_dof_attributable_to_counts"] == \
        pytest.approx(14.15, abs=0.02)
    assert p["fraction_of_delta_that_is_statistical_power"] == \
        pytest.approx(0.33, abs=0.01)
    # and the scale-free delta is carried alongside, so the reader can see the
    # direction survives the discount
    assert p["delta_rms_frac_dev"] == pytest.approx(0.1366 - 0.1193)
    assert "NOT\n                  matched in counts" in c["confound"] or \
        "NOT matched in counts" in " ".join(c["confound"].split())
    assert "rms_frac_dev" in c["scale_free_alternative"]


def test_p2_power_confound_is_DISCLOSED_and_the_amendments_are_recorded(WS):
    """The disclosure must exist in the module, name the confound for P2 (not
    only for the marginal max|z|), and be labelled POST-HOC rather than
    smuggled into the pre-registered PROTOCOL (P6)."""
    ids = [i for i, _d, _a in WS.POST_HOC_AMENDMENTS]
    assert ids == ["A1", "A2"]
    text = " ".join(a for _i, _d, a in WS.POST_HOC_AMENDMENTS)
    assert "CONFOUNDED WITH SAMPLE SIZE" in text
    assert "chi2/dof" in text and "rms_frac_dev" in text
    assert "77.8" in text                       # the power-scaling prediction
    assert "106.59" in text                     # the observed value
    # the pre-registered protocol is UNCHANGED: still exactly P1..P6
    assert [i for i, _ in WS.PROTOCOL] == ["P1", "P2", "P3", "P4", "P5", "P6"]
    assert not any(i.startswith("A") for i, _ in WS.PROTOCOL)


def test_recommendation_rests_on_the_SCALE_FREE_measure(WS, monkeypatch):
    """The restated recommendation must (a) say which statistic decides, (b)
    still say KEEP lya_only, and (c) state the discount on the chi2/dof
    magnitude instead of quoting it as the reason."""
    monkeypatch.setattr(WS, "CLAMPS", ["both"])
    monkeypatch.setattr(WS, "MOCKS", ["2lpt0", "london0", "saclay0"])
    rows = _confound_rows(63.65206298413061, 106.58576787999633,
                          67086.0, 82008.0, 0.1193, 0.1366)
    rec = WS.recommendation(rows, {}, None)
    assert rec["answer"].startswith("KEEP lya_only")
    assert "rms_frac_dev" in rec["decided_on"]
    assert "SCALE-FREE" in rec["decided_on"]
    assert "NOT" in rec["decided_on"] and "confounded" in rec["decided_on"]
    joined = " ".join(rec["reasoning"])
    assert "A2 (PRIMARY as restated" in joined
    assert "MAGNITUDE DISCOUNTED" in joined
    assert "20-22% more counts" in joined and "ONE THIRD" in joined
    assert any("SIZE of the chi2/dof gap" in s
               for s in rec["what_this_does_NOT_say"])
    assert any("NOT a gate" in s for s in rec["what_this_does_NOT_say"])


def _verdict_rows(WS, chi2, z_total=99.0, z_bin=99.0):
    """A full `rows` table for build_verdict, with primary_closes computed by
    the REAL ``closes`` against the REAL gate record."""
    gate = WS.restated_gate_criteria()["gate_arms"]
    out = {}
    for mock in ("2lpt0", "london0", "saclay0"):
        for win, c2 in (("lya_only", chi2), ("lya_lyb", chi2 + 40.0)):
            for clamp in ("both", "hi"):
                rep = dict(n_bins=19, n_bins_occupied=19, z_total=z_total,
                           z_bin_max=z_bin, chi2_dof=c2, ratio=0.98,
                           rms_frac_dev=0.12, obs=70000.0)
                out[f"{mock}|{win}|clamp={clamp}"] = dict(
                    primary_reporting_window=rep,
                    primary_closes=WS.closes(rep, gate),
                    high_n_above_21p6=dict(ratio=1.2),
                    full_grid=dict(chi2_dof=c2, ratio=0.99),
                    by_z=dict(ratio_span=0.15), by_snr=dict(ratio_span=0.18),
                )
    return out


def test_verdict_MEASURES_whether_the_headline_needs_the_unratified_arms(
        WS, monkeypatch):
    """Two of the three gating arms are RESTATED_NOT_RATIFIED, so the artifact
    must SHOW — not assert — that "nothing closes" survives dropping them."""
    monkeypatch.setattr(WS, "CLAMPS", ["both"])
    monkeypatch.setattr(WS, "MOCKS", ["2lpt0", "london0", "saclay0"])
    # every configuration busts chi2/dof <= 3, which is the real situation
    v = WS.build_verdict(_verdict_rows(WS, 63.0), {}, None)
    a = v["authority_sensitivity"]
    assert v["n_closing_primary_window"] == 0
    assert a["n_closing_all_arms"] == 0
    assert a["n_closing_pi_ratified_arm_only"] == 0
    assert a["verdict_unchanged_without_unratified_arms"] is True
    assert a["pi_ratified_gating_arm"].startswith("chi2_dof_max <= 3.0")


def test_verdict_would_EXPOSE_a_headline_that_rests_on_an_unratified_arm(
        WS, monkeypatch):
    """POWER: the field must be able to come out FALSE. Here the six lya_only
    configurations pass chi2/dof <= 3 and are refused ONLY by the two |z| arms
    nobody ratified — so "nothing closes" would rest on unratified numbers, and
    the artifact must say so rather than report a flat True."""
    monkeypatch.setattr(WS, "CLAMPS", ["both"])
    monkeypatch.setattr(WS, "MOCKS", ["2lpt0", "london0", "saclay0"])
    v = WS.build_verdict(_verdict_rows(WS, 1.0), {}, None)
    a = v["authority_sensitivity"]
    assert a["n_closing_all_arms"] == 0                    # |z| arms refuse
    assert a["n_closing_pi_ratified_arm_only"] == 6        # chi2/dof does not
    assert all("lya_only" in k for k in
               a["closing_configurations_pi_ratified_arm_only"])
    assert a["verdict_unchanged_without_unratified_arms"] is False


# ---------------------------------------------------------------------------
# (5b) THE COMMITTED ARTIFACT — it must carry the corrected record, not v1's
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ARTIFACT():
    p = os.path.join(_REPO, "CDDF_analysis/hbi_mcmc/spectral_window_study.json")
    with open(p) as f:
        return json.load(f)


def _walk_keys(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}.{k}", k, v
            yield from _walk_keys(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_keys(v, f"{path}[{i}]")


def test_committed_artifact_carries_NO_fabricated_ratification(ARTIFACT, WS):
    """The committed JSON is what a referee reads. v1 stamped
    `metadata.gate.ratified_arms = {abs_z_total_max: 5, z_bin_max: 5,
    chi2_dof_max: 3}`; that field must be gone everywhere in the tree, and no
    surviving "ratified" field may name anything off the allow-list."""
    allowed = set(WS.PI_RATIFIED_ITEMS)
    bad = []
    for path, key, val in _walk_keys(ARTIFACT):
        if key in ("ratified_arms", "gate_tolerances_ratified"):
            bad.append(path)
        elif "ratified" in key.lower() and isinstance(val, (dict, list)):
            stray = set(val) - allowed
            if stray:
                bad.append(f"{path} -> {sorted(stray)}")
    assert not bad, bad


def test_committed_artifact_gate_arms_carry_their_authority_state(ARTIFACT, WS):
    arms = ARTIFACT["metadata"]["gate"]["gate_arms"]
    assert arms["chi2_dof_max"]["authority_state"] == "RATIFIED"
    for n in ("abs_z_total_max", "z_bin_max"):
        assert arms[n]["authority_state"] == "RESTATED_NOT_RATIFIED", n
        assert arms[n]["gates"] is True, f"{n} gates and must say so"
    adv = ARTIFACT["metadata"]["gate"]["advisory_tolerances"]
    assert set(adv) == {"ratio_span_by_z_max", "ratio_span_by_snr_max"}
    assert all(t["authority_state"] == "UNRATIFIED" and t["gates"] is False
               for t in adv.values())
    assert "FABRICATED AUTHORITY CLAIM" in \
        ARTIFACT["metadata"]["gate"]["authority_correction_note"]


def test_committed_artifact_per_config_gates_carry_authority_too(ARTIFACT):
    """Not just the metadata block: every one of the 12 configurations echoes
    its gate, and each echo must be authority-bearing."""
    cfgs = ARTIFACT["arm1_analysis_window"]
    assert len(cfgs) == 12
    for k, v in cfgs.items():
        pc = v["primary_closes"]
        assert "gate" not in pc, f"{k} still echoes the bare `gate` dict"
        for name, arm in pc["gate_arms"].items():
            assert set(arm) == {"value", "authority_state", "gates"}, (k, name)
        # a refusal on an unratified number must name it as such
        for f in pc["failures"]:
            if f.startswith("chi2_dof"):
                assert "RATIFIED" not in f, (k, f)
            else:
                assert "RESTATED_NOT_RATIFIED" in f, (k, f)


def test_committed_artifact_headline_does_NOT_rest_on_an_unratified_arm(
        ARTIFACT):
    """SCIENCE-CONTENT GUARD, pinned. The correction is a LABEL change: the
    verdict is 0 closing configurations WITH the |z| arms and 0 WITHOUT them,
    so no threshold whose authority was overstated is load-bearing."""
    v = ARTIFACT["verdict"]
    assert v["n_closing_primary_window"] == 0
    a = v["authority_sensitivity"]
    assert a["n_closing_all_arms"] == 0
    assert a["n_closing_pi_ratified_arm_only"] == 0
    assert a["verdict_unchanged_without_unratified_arms"] is True
    # every configuration busts the RATIFIED arm on its own
    for k, cfg in ARTIFACT["arm1_analysis_window"].items():
        assert cfg["primary_closes"]["closes_on_pi_authority_only"] is False, k


def test_committed_artifact_stamp_is_clean_and_on_this_branch(ARTIFACT):
    md = ARTIFACT["metadata"]
    assert md["code_commit_dirty"] is False
    assert len(md["code_commit"]) == 40 and "-dirty" not in md["code_commit"]
    assert md["pack_stamp_audit"]["any_pack_dirty"] is False


# ---------------------------------------------------------------------------
# (6) build_frozen_calibration -- THE WINDOW THREADING ORCHESTRATOR
#
# The referee's second "real defect found and fixed" was the build_fp_block
# window re-derivation, and `grep -rn build_fp_block tests/` returned NOTHING at
# the time. `grep -rn build_frozen_calibration tests/` ALSO returned nothing,
# and that function is the single place where the analysis window is threaded
# into EVERY window-dependent ingredient (forward response, completeness
# matrix, sub-floor matrix, loa-0 FP cut, detection bundle). Dropping the
# `window=window` keyword on any ONE of those calls silently produces exactly
# the mixed-window calibration the whole guard layer exists to prevent -- and
# nothing downstream can see it, because `frozen["analysis_window"]` is stamped
# from the ARGUMENT, not from the ingredients.
# ---------------------------------------------------------------------------
def _wire_frozen(EP, monkeypatch, n_nhi=3):
    """Inject every heavy ingredient of build_frozen_calibration and RECORD the
    window each one was called with."""
    seen = {}
    molly = dict(molly_nhi_edges=np.linspace(19.5, 22.5, n_nhi + 1))

    monkeypatch.setattr(EP, "load_forward_response_pack",
                        lambda p: (seen.__setitem__("forward_npz", p),
                                   ("FWD", {}))[1])
    monkeypatch.setattr(EP, "compute_t_sigma", lambda: ("TSIG", {}))

    def fake_counts(convention="const_extrap", counts172_path=None,
                    window=None):
        seen["molly"] = window
        seen["molly_convention"] = convention
        return molly, {}, None
    monkeypatch.setattr(EP, "load_molly_counts_block", fake_counts)

    def fake_fp(window=None, **kw):
        seen["fp"] = window
        return np.zeros((29, 8)), None, {}
    monkeypatch.setattr(EP, "build_fp_block", fake_fp)

    def fake_bundle(mock, out_dir, molly_tsv=None, window=None):
        seen.setdefault("bundle", []).append((mock, molly_tsv, window))
        return dict(tag=mock)
    monkeypatch.setattr(EP, "load_mock_bundle", fake_bundle)
    monkeypatch.setattr(EP, "build_g_block",
                        lambda b: (np.ones((n_nhi, EP.N_K)),
                                   np.zeros((n_nhi, EP.N_K))))
    return seen


def test_build_frozen_calibration_THREADS_the_window_into_EVERY_ingredient(
        EP, monkeypatch, tmp_path):
    """Every window-dependent ingredient must be built at the REQUESTED window.

    Each assertion below corresponds to a one-keyword mutant (drop
    ``window=window`` from that call, so it falls back to DEF_WINDOW =
    lya_only): the resulting frozen calibration would still stamp
    ``analysis_window="lya_lyb"`` while carrying a 1025-A ingredient.
    """
    seen = _wire_frozen(EP, monkeypatch)
    w = EP.window_spec("lya_lyb")
    frozen = EP.build_frozen_calibration(str(tmp_path), window="lya_lyb")

    assert seen["forward_npz"] == w["forward_npz"], (
        "the forward response was loaded from the wrong window's NPZ")
    assert seen["molly"] == "lya_lyb", "completeness matrix built at the wrong window"
    assert seen["fp"] == "lya_lyb", "loa-0 FP background cut at the wrong window"
    assert seen["bundle"] == [("2lpt0", None, "lya_lyb")], (
        "the 2LPT-0 detection bundle was cut at the wrong window")

    # ... and the STAMPS must agree with what was actually built
    assert frozen["analysis_window"] == "lya_lyb"
    assert frozen["window_spec"]["lam_rf_min"] == 911.0
    assert frozen["fwd_meta"]["analysis_window"] == "lya_lyb"
    assert frozen["fwd_meta"]["lam_rf_min"] == 911.0
    assert frozen["g_available"] is True
    assert frozen["_bundles"]["2lpt0"]["tag"] == "2lpt0"


def test_build_frozen_calibration_DEFAULT_stays_the_nominal_1025A_window(
        EP, monkeypatch, tmp_path):
    """The pre-window default path must be unchanged: no argument => lya_only
    everywhere. Without this leg the test above could be satisfied by a
    function that hard-codes lya_lyb."""
    seen = _wire_frozen(EP, monkeypatch)
    frozen = EP.build_frozen_calibration(str(tmp_path))
    assert EP.DEF_WINDOW == "lya_only"
    assert seen["molly"] == "lya_only" and seen["fp"] == "lya_only"
    assert seen["bundle"] == [("2lpt0", None, "lya_only")]
    assert frozen["window_spec"]["lam_rf_min"] == 1025.0
    assert frozen["fwd_meta"]["lam_rf_min"] == 1025.0


def test_build_frozen_calibration_molly172_splices_the_SAME_windows_subfloor(
        EP, monkeypatch, tmp_path):
    """Under the molly172 convention a SECOND bundle is cut from the floor-17.2
    matrix. It must be this window's ``molly_tsv_172`` AND this window's
    lam_rf_min, or the sub-floor rows come from the other window."""
    seen = _wire_frozen(EP, monkeypatch, n_nhi=3)
    w = EP.window_spec("lya_lyb")

    def fake_counts(convention="const_extrap", counts172_path=None,
                    window=None):
        seen["molly"] = window
        seen["molly_convention"] = convention
        return dict(molly_nhi_edges=np.linspace(19.0, 22.5, 6)), {}, "ALT"
    monkeypatch.setattr(EP, "load_molly_counts_block", fake_counts)

    def fake_g(bundle):
        # the deeper (17.2) bundle carries 2 extra sub-floor rows
        n = 5 if bundle.get("molly172") else 3
        return np.ones((n, EP.N_K)), np.zeros((n, EP.N_K))

    def fake_bundle(mock, out_dir, molly_tsv=None, window=None):
        seen.setdefault("bundle", []).append((mock, molly_tsv, window))
        return dict(tag=mock, molly172=molly_tsv is not None)
    monkeypatch.setattr(EP, "load_mock_bundle", fake_bundle)
    monkeypatch.setattr(EP, "build_g_block", fake_g)

    frozen = EP.build_frozen_calibration(str(tmp_path), completeness="molly172",
                                         window="lya_lyb")
    assert seen["molly_convention"] == "molly172"
    assert seen["bundle"] == [("2lpt0", None, "lya_lyb"),
                             ("2lpt0", w["molly_tsv_172"], "lya_lyb")], (
        "the floor-17.2 splice bundle was not cut from THIS window's "
        "molly_tsv_172 at THIS window's lam_rf_min")
    assert frozen["g_grid"].shape[0] == 5
    assert "2 sub-floor rows" in frozen["molly_prov"]["g_below_floor"]


def test_build_frozen_calibration_REFUSES_a_g_grid_that_misses_the_molly_grid(
        EP, monkeypatch, tmp_path):
    """The shape guard: g(N,z) must have one row per molly N cell. A mismatch
    means the splice or the grid moved, and silently broadcasting it would
    misalign the completeness against the basis."""
    _wire_frozen(EP, monkeypatch, n_nhi=3)
    monkeypatch.setattr(EP, "build_g_block",
                        lambda b: (np.ones((2, EP.N_K)), np.zeros((2, EP.N_K))))
    with pytest.raises(RuntimeError, match="g_grid has 2 cells"):
        EP.build_frozen_calibration(str(tmp_path), window="lya_lyb")
