"""bh_ratify_stamp — a PI-ratification SIDECAR-STYLE successor of a track_c_tf_hz
artifact: measurement bytes identical, metadata gains the written ruling.
RULES §2: authority=PI / paper_facing=True only from a written ruling — the
routine refuses without a ruling id and records the source sha256."""
import importlib.util as _ilu, json, os, hashlib
import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = _ilu.spec_from_file_location(
    "bh_ratify_mod", os.path.join(_REPO, "CDDF_analysis", "hbi", "bh_ratify_stamp.py"))
RS = _ilu.module_from_spec(_spec); _spec.loader.exec_module(RS)


def _src(tmp_path):
    d = {"metadata": {"estimand": "DIAGNOSTIC_RECENTERED", "paper_facing": False,
                      "status": "CANDIDATE / PI-ADOPTION-PENDING (BH high-z bin)",
                      "z_extrapolated": [True, True, True], "truth_counts_perz": [0, 0, 0],
                      "code_commit": "abc"},
         "measurement": {"20.3": {"dndx": {"integrated": {"MAP": 0.1086, "q16": 0.104, "q84": 0.1135},
                                           "perz": [{"MAP": 0.1014}, {"MAP": 0.1225}, {"MAP": 0.1508}]},
                                  "omega": {"integrated": {"MAP": 1.214e-3}, "perz": []}}},
         "zbins": [3.8, 4.25, 4.5, 5.0], "perz_fN": {"x": [1, 2, 3]}}
    p = tmp_path / "src.json"; p.write_text(json.dumps(d, indent=2)); return p, d


def test_stamp_keeps_measurement_bytes_and_adds_ratification(tmp_path):
    p, d = _src(tmp_path)
    out = tmp_path / "out.json"
    r = RS.stamp(str(p), str(out), ruling="PI 2026-08-26 #43/#44", reported_bin=[3.8, 5.0],
                 validation_basis="H2 injection campaign (record X)", named_lines={"a": "b"})
    o = json.load(open(out))
    for k in ("measurement", "zbins", "perz_fN"):
        assert json.dumps(o[k], sort_keys=True) == json.dumps(d[k], sort_keys=True)
    md = o["metadata"]
    assert md["paper_facing"] is True and md["authority"] == "PI"
    assert md["status"].startswith("PAPER-FACING")
    assert md["ratification"]["source_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert md["ratification"]["ruling"] == "PI 2026-08-26 #43/#44"
    assert md["ratification"]["reported_estimand"]["z_bin"] == [3.8, 5.0]
    assert "internal_subdivision" in md["ratification"]["reported_estimand"]
    assert md["superseded_status"] == "CANDIDATE / PI-ADOPTION-PENDING (BH high-z bin)"
    assert md["z_extrapolated"] == [True, True, True]          # provenance stamps retained, not erased
    assert r["source_sha256"] == md["ratification"]["source_sha256"]


def test_stamp_refuses_without_a_ruling(tmp_path):
    p, _ = _src(tmp_path)
    with pytest.raises(ValueError):
        RS.stamp(str(p), str(tmp_path / "o.json"), ruling="", reported_bin=[3.8, 5.0],
                 validation_basis="x", named_lines={})


def test_stamp_refuses_a_reported_bin_the_artifact_does_not_span(tmp_path):
    p, _ = _src(tmp_path)
    with pytest.raises(ValueError):
        RS.stamp(str(p), str(tmp_path / "o.json"), ruling="PI x", reported_bin=[3.5, 5.0],
                 validation_basis="x", named_lines={})
