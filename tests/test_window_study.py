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


def test_closes_uses_only_the_three_ratified_arms(WS, by_nhat):
    gate = WS.restated_gate_criteria()["ratified_arms"]
    assert set(gate) == {"abs_z_total_max", "z_bin_max", "chi2_dof_max"}
    m = WS.window_metrics(by_nhat, 19.7, 21.6)
    v = WS.closes(m, gate)
    # chi2/dof = (4.41 + 1 + 9)/3 = 4.8033 > 3  ->  must FAIL on chi2 only
    assert v["closes"] is False
    assert len(v["failures"]) == 1 and v["failures"][0].startswith("chi2_dof")
    # and a table that closes must PASS
    tight = [_row(19.7, 19.8, 100.0, 101.0), _row(19.8, 19.9, 100.0, 99.0)]
    assert WS.closes(WS.window_metrics(tight, 19.7, 21.6), gate)["closes"]


def test_ratio_span_tolerances_are_declared_UNRATIFIED(WS):
    """PI decision 8 explicitly declined to ratify these two, so they must be
    reported as measurements and must not appear among the gating arms."""
    g = WS.restated_gate_criteria()
    assert set(g["not_ratified"]) == {"ratio_span_by_z_max",
                                      "ratio_span_by_snr_max"}
    assert not (set(g["not_ratified"]) & set(g["ratified_arms"]))


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


def test_basis_resolution_status_is_explicit_about_decision_3(WS):
    """The study must SAY it did not implement the 0.2-dex basis rather than
    let a reader assume the PI-adopted configuration was fully realised."""
    s = WS.BASIS_RESOLUTION_STATUS
    assert WS.BASIS_DEX == 0.1
    assert "NOT implemented" in s and "0.2-dex" in s
    assert "validate_pack" in s          # names the concrete blocker
