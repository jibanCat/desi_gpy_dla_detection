"""mark_superseded — PI ruling 2026-08-21 #9: old defective-g artifacts are
SUPERSEDED with provenance metadata, never deleted, hidden or modified."""
import importlib.util as _ilu
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = _ilu.spec_from_file_location(
    "mark_superseded_mod",
    os.path.join(_REPO, "CDDF_analysis", "hbi_mcmc", "mark_superseded.py"))
MS = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(MS)


def test_supersede_record_adds_status_and_history_without_dropping_fields():
    prov = dict(schema="v1.2", code_commit="abc")
    out = MS.supersede_record(prov, superseded_by="/new/pack.npz",
                              reason="g(N,z) support defect", ruling="PI 2026-08-21 #9",
                              date="2026-08-21")
    assert out["schema"] == "v1.2" and out["code_commit"] == "abc"
    assert out["status"] == "SUPERSEDED"
    assert out["superseded"][-1]["superseded_by"] == "/new/pack.npz"
    assert out["superseded"][-1]["retained"] is True


def test_supersede_record_is_idempotent_per_successor():
    prov = {}
    a = MS.supersede_record(prov, superseded_by="/new/p.npz", reason="r",
                            ruling="x", date="d")
    b = MS.supersede_record(a, superseded_by="/new/p.npz", reason="r",
                            ruling="x", date="d")
    assert len(b["superseded"]) == 1


def test_apply_writes_sidecar_and_never_touches_the_npz(tmp_path):
    npz = tmp_path / "old.npz"
    np.savez(npz, x=np.arange(3))
    before = npz.read_bytes()
    MS.apply(str(npz), superseded_by="/new/p.npz", reason="r", ruling="x", date="d")
    side = tmp_path / "old.provenance.json"
    assert side.exists()
    rec = json.loads(side.read_text())
    assert rec["status"] == "SUPERSEDED"
    assert npz.read_bytes() == before
