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
