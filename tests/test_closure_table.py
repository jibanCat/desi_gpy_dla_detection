"""closure_table CLI guards (Phase B).

Pins the fail-loud boundary of the closure product builder: the module
docstring's "MOCKS ONLY" is ENFORCED (this CLI writes result VALUES into a
JSON, and real-LOA values are private — notes repo only), and a pack without
``fp_eta_c`` is refused unless the logged ``--allow-legacy-eta`` escape is
passed explicitly. Guard tests only — the heavy gate machinery is covered by
tests/test_gate_covariance.py.
"""
import types

import numpy as np
import pytest

pytest.importorskip("jax")

from CDDF_analysis.hbi_mcmc import closure_table as CT


def test_real_loa_path_is_refused_before_load():
    with pytest.raises(AssertionError, match="REAL-LOA"):
        CT.build_row("modelA_pack_loa_main_dark_v1.npz", allow_legacy_eta=False)


def test_real_loa_provenance_is_refused_after_load(monkeypatch):
    stub = types.SimpleNamespace(
        provenance={"source": "…/loa_main_dark_v1/catalog"},
        fp_eta_c=np.zeros(3))
    monkeypatch.setattr(CT, "load_pack", lambda p: stub)
    with pytest.raises(AssertionError, match="REAL-LOA"):
        CT.build_row("innocuous_name.npz", allow_legacy_eta=False)


def test_legacy_pack_without_eta_is_refused_unless_explicitly_allowed(
        monkeypatch):
    stub = types.SimpleNamespace(provenance={}, fp_eta_c=None)
    monkeypatch.setattr(CT, "load_pack", lambda p: stub)
    with pytest.raises(SystemExit, match="allow-legacy-eta"):
        CT.build_row("legacy_pack.npz", allow_legacy_eta=False)
