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


def test_closes_REFUSES_an_EMPTY_reporting_window(WS):
    """FAIL-CLOSED, not fail-open. A selection that contains NO fully-contained
    bin cannot be gated, and the three ratified arms all evaluate vacuously on
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
    v = WS.closes(m, WS.restated_gate_criteria()["ratified_arms"])
    assert v["closes"] is False, (
        "closes() passed an EMPTY reporting window — every informative arm was "
        "skipped as non-finite and |z_total| passed on 0/sqrt(1e-12)")
    assert any("n_bins" in f for f in v["failures"]), v["failures"]


def test_closes_REFUSES_when_NO_bin_is_occupied(WS):
    """The other vacuous shape: bins exist but every one has obs == 0, so the
    two per-bin arms are again non-finite and skipped. |z_total| alone is NOT a
    closure test (P4: the total is a level, the shape is the defect)."""
    m = WS.window_metrics([_row(19.7, 19.8, 9.0, 0.0),
                           _row(19.8, 19.9, 4.0, 0.0)],
                          WS.REPORT_LO, WS.REPORT_HI)
    assert m["n_bins"] == 2 and m["n_bins_occupied"] == 0
    assert abs(m["z_total"]) < 5.0              # would have passed vacuously
    v = WS.closes(m, WS.restated_gate_criteria()["ratified_arms"])
    assert v["closes"] is False, (
        "closes() passed a window in which not one bin carries an observed "
        "count")
    assert any("n_bins_occupied" in f for f in v["failures"]), v["failures"]


def test_closes_REFUSES_a_NON_FINITE_gate_arm(WS):
    """Belt and braces: a non-finite arm is a refusal, never a skip. Built by
    hand rather than through window_metrics so the arms are non-finite
    INDEPENDENTLY of how many bins were selected."""
    gate = WS.restated_gate_criteria()["ratified_arms"]
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
