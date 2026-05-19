"""Tests for the marginal-purity stacking additions to
examples/stack_real_loa_dlas.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import examples.stack_real_loa_dlas as stk  # noqa: E402


def _synthetic_catalog(p_dla_values):
    """A DLACAT-shaped structured array; every row passes every cut
    except P_DLA, which is set per-row from `p_dla_values`."""
    n = len(p_dla_values)
    dt = np.dtype([("TARGETID", "<i8"), ("Z_QSO", "<f4"), ("Z_DLA", "<f4"),
                   ("P_DLA", "<f4"), ("SNR_FOREST", "<f4"),
                   ("DLAFLAG", "<i4"), ("NHI", "<f4")])
    cat = np.zeros(n, dtype=dt)
    cat["TARGETID"] = np.arange(1, n + 1)
    cat["Z_QSO"] = 4.0
    cat["Z_DLA"] = 3.5          # in-forest + not-proximate for z_qso=4.0
    cat["SNR_FOREST"] = 5.0
    cat["DLAFLAG"] = 0
    cat["NHI"] = 20.0
    cat["P_DLA"] = np.asarray(p_dla_values, dtype=np.float32)
    return cat


def test_purity_preset_selection(monkeypatch):
    cat = _synthetic_catalog([0.30, 0.55, 0.65, 0.75, 0.98, 0.995])

    monkeypatch.setattr(stk, "PURITY", "marginal")
    kept = stk.select(cat, set())
    assert sorted(kept["P_DLA"].astype(float).round(3)) == [0.55, 0.65]

    monkeypatch.setattr(stk, "PURITY", "high")
    kept = stk.select(cat, set())
    assert sorted(kept["P_DLA"].astype(float).round(3)) == [0.98, 0.995]


def test_provenance_carries_purity(monkeypatch):
    monkeypatch.setattr(stk, "PURITY", "marginal")
    prov = stk.provenance_dict()
    assert prov["purity"] == "marginal"
    assert prov["p_dla_range"] == [0.50, 0.70]
    assert "p_dla_min" not in prov


def test_tagged_and_npz_path(monkeypatch):
    monkeypatch.setattr(stk, "PURITY", "marginal")
    assert stk.tagged("stack_prod") == "stack_prod_marginal.png"
    assert stk.tagged("counts", "txt") == "counts_marginal.txt"
    assert stk.npz_path().name == "stack_curves_marginal.npz"
    monkeypatch.setattr(stk, "PURITY", "high")
    assert stk.npz_path().name == "stack_curves_high.npz"


def test_control_categories_has_lownhi():
    assert "lownhi" in stk.CONTROL_CATEGORIES
    assert (set(stk.CONTROL_CATEGORIES["lownhi"])
            == set(stk.LLS_BINS_FINE) | set(stk.SUBDLA_BINS))
    assert "lownhi" in {"lls", "subdla", "lownhi"}  # named control category


def test_check_provenance_preset_mismatch(monkeypatch):
    monkeypatch.setattr(stk, "PURITY", "high")
    stored = stk.provenance_dict()          # a 'high' provenance dict
    # expecting 'marginal' must raise
    monkeypatch.setattr(sys, "argv", ["x"])  # no --force-plot
    with pytest.raises(SystemExit):
        stk.check_provenance(stored, expect_preset="marginal")
    # expecting 'high' must pass (no raise)
    stk.check_provenance(stored, expect_preset="high")


def _mock_curve(rg, seed=0):
    rng = np.random.default_rng(seed)
    slope = 1.0 + 0.05 * (rg - 1200.0) / 900.0
    curve = slope + rng.normal(0.0, 0.01, len(rg))
    curve[rg < 760.0] = np.nan
    counts = np.clip(60.0 + 0.9 * (rg - 700.0), 0.0, 800.0)
    return curve.astype(float), counts.astype(float)


def test_purity_comparison_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(stk, "OUT_DIR", tmp_path)
    rg = 10 ** np.arange(np.log10(stk.REST_LAMBDA_MIN),
                         np.log10(stk.REST_LAMBDA_MAX), stk.DLOG_LAMBDA)
    curve, counts = _mock_curve(rg)
    P = stk.fit_pseudo_continuum(rg, curve, counts)
    comb = (curve, counts, P, curve, counts, P, 300)
    stk.plot_purity_comparison(rg, comb, comb, "cmp.png")
    assert (tmp_path / "cmp.png").stat().st_size > 0


def test_purity_compare_panels_nonempty():
    assert len(stk.PURITY_COMPARE_PANELS) >= 4
    titles = {p[0] for p in stk.PURITY_COMPARE_PANELS}
    assert "CIV 1548/1551" in titles


def test_pcont_persists_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(stk, "OUT_DIR", tmp_path)
    monkeypatch.setattr(stk, "PURITY", "high")
    rg = 10 ** np.arange(np.log10(stk.REST_LAMBDA_MIN),
                         np.log10(stk.REST_LAMBDA_MAX), stk.DLOG_LAMBDA)
    curve, counts = _mock_curve(rg)
    # 80 noisy copies (60 non-BAL + 20 BAL) so the non-BAL group clears
    # the 50-spectrum coverage floor and `bs.curve` is a real stack.
    raw = (np.tile(curve, (80, 1))
           + np.random.default_rng(1).normal(0.0, 0.01, (80, len(rg))))
    is_bal = np.array([False] * 60 + [True] * 20)
    bs = stk._stack_pair(rg, raw, is_bal)
    assert np.isfinite(bs.curve[(rg > 1000) & (rg < 1500)]).any()
    # pcont must be the deterministic fit of the non-BAL curve
    assert np.allclose(np.nan_to_num(bs.pcont),
                       np.nan_to_num(stk.fit_pseudo_continuum(rg, bs.curve,
                                                              bs.counts)))
    per_bin = {stk.NHI_BINS[0]: bs}
    P = stk.fit_pseudo_continuum(rg, curve, counts)
    combined = {"lownhi": (curve, counts, P, curve, counts, P, 2)}
    stk.save_curves(rg, per_bin, combined)
    _rg, per_bin2, comb2 = stk.load_curves(stk.npz_path())
    bs2 = per_bin2[stk.NHI_BINS[0]]
    assert np.array_equal(np.nan_to_num(bs2.pcont), np.nan_to_num(bs.pcont))
    assert np.array_equal(np.nan_to_num(comb2["lownhi"][2]),
                          np.nan_to_num(P))
    assert len(comb2["lownhi"]) == 7
