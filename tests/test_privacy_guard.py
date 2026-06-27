"""Tests for ``tools/provenance/precommit_privacy_guard.py``.

The guard blocks real-LOA results-store leaves from entering the public repo.
These tests drive it via its ``--paths`` mode (no real git index needed) and
build the fixture ``provenance.json`` files with the SAME field names the
producer (``CDDF_analysis/hbi/provenance.py``) writes: a nested ``"privacy"``
object with ``"class"`` (``"mock"`` | ``"real-LOA"``) and ``"shareable"`` (bool).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_GUARD_PATH = _REPO / "tools" / "provenance" / "precommit_privacy_guard.py"


def _load_guard():
    spec = importlib.util.spec_from_file_location(
        "precommit_privacy_guard", _GUARD_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


guard = _load_guard()


# --------------------------------------------------------------------------- #
# fixture builders                                                            #
# --------------------------------------------------------------------------- #
def _write_provenance(leaf: Path, *, privacy_class: str, shareable: bool) -> Path:
    """Write a minimal but field-faithful provenance.json into ``leaf``."""
    leaf.mkdir(parents=True, exist_ok=True)
    rec = {
        "schema_version": "cddf-provenance/1",
        "id": leaf.name,
        # the exact field names provenance.py emits:
        "privacy": {"class": privacy_class, "shareable": shareable},
    }
    prov = leaf / "provenance.json"
    prov.write_text(json.dumps(rec, indent=2))
    return prov


def _run(paths, root):
    """Invoke the guard's main() in --paths mode; return its exit code."""
    argv = ["--root", str(root), "--paths", *[str(p) for p in paths]]
    return guard.main(argv)


# --------------------------------------------------------------------------- #
# (a) clean mock leaf passes                                                  #
# --------------------------------------------------------------------------- #
def test_mock_leaf_passes(tmp_path):
    leaf = tmp_path / "mock" / "2lpt0" / "stage" / "base__deadbeef"
    prov = _write_provenance(leaf, privacy_class="mock", shareable=True)
    data = leaf / "result.npz"
    data.write_text("payload")

    assert _run([prov, data], tmp_path) == 0


# --------------------------------------------------------------------------- #
# (b) real-LOA leaf fails, naming the path                                    #
# --------------------------------------------------------------------------- #
def test_real_loa_provenance_fails(tmp_path, capsys):
    # Put it NOT under real_loa/ so we isolate the provenance-field rule.
    leaf = tmp_path / "somewhere" / "leaf__cafef00d"
    prov = _write_provenance(leaf, privacy_class="real-LOA", shareable=False)

    rc = _run([prov], tmp_path)
    assert rc == 1
    err = capsys.readouterr().err
    assert "BLOCKED" in err
    assert "provenance.json" in err
    assert "real-LOA" in err


def test_real_loa_taints_sibling_files(tmp_path, capsys):
    """A staged data file beside a real-LOA provenance.json is blocked even if the
    provenance.json itself is not in the staged set."""
    leaf = tmp_path / "leaf__1234abcd"
    _write_provenance(leaf, privacy_class="real-LOA", shareable=False)
    data = leaf / "per_object_nhi.npz"
    data.write_text("secret")

    rc = _run([data], tmp_path)  # note: provenance.json NOT in the staged list
    assert rc == 1
    err = capsys.readouterr().err
    assert "per_object_nhi.npz" in err
    assert "sibling" in err.lower()


def test_shareable_false_blocks_even_if_class_mock(tmp_path):
    """shareable==False is sufficient to block, independent of class."""
    leaf = tmp_path / "leaf__0badf00d"
    prov = _write_provenance(leaf, privacy_class="mock", shareable=False)
    assert _run([prov], tmp_path) == 1


# --------------------------------------------------------------------------- #
# (c) anything under real_loa/ fails                                          #
# --------------------------------------------------------------------------- #
def test_real_loa_partition_path_fails(tmp_path, capsys):
    leaf = tmp_path / "real_loa" / "loa_main" / "stage" / "base__abc123"
    leaf.mkdir(parents=True, exist_ok=True)
    data = leaf / "anything.txt"
    data.write_text("x")

    rc = _run([data], tmp_path)
    assert rc == 1
    err = capsys.readouterr().err
    assert "real_loa" in err
    assert "partition" in err.lower()


def test_real_loa_partition_blocks_even_nonexistent(tmp_path, capsys):
    """A staged path under real_loa/ is blocked on location alone, even if the
    file is gone from disk (location-only rule)."""
    ghost = tmp_path / "real_loa" / "ds" / "stage" / "leaf__deadbeef" / "gone.npz"
    rc = _run([ghost], tmp_path)
    assert rc == 1
    assert "real_loa" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# (d) malformed provenance.json fails closed (SUSPECT)                        #
# --------------------------------------------------------------------------- #
def test_malformed_provenance_fails_closed(tmp_path, capsys):
    leaf = tmp_path / "leaf__feedface"
    leaf.mkdir(parents=True, exist_ok=True)
    prov = leaf / "provenance.json"
    prov.write_text("{ this is not valid json ")  # truncated / broken
    data = leaf / "result.npz"
    data.write_text("payload")

    rc = _run([prov, data], tmp_path)
    assert rc == 1
    err = capsys.readouterr().err
    assert "SUSPECT" in err
    # the adjacent data file is also blocked (fail-closed taint)
    assert "result.npz" in err


def test_provenance_missing_privacy_fails_closed(tmp_path, capsys):
    leaf = tmp_path / "leaf__99887766"
    leaf.mkdir(parents=True, exist_ok=True)
    prov = leaf / "provenance.json"
    prov.write_text(json.dumps({"schema_version": "cddf-provenance/1"}))

    rc = _run([prov], tmp_path)
    assert rc == 1
    assert "SUSPECT" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# (e) a fully clean staged set passes                                         #
# --------------------------------------------------------------------------- #
def test_clean_set_passes(tmp_path, capsys):
    # a mock leaf + a plain doc, nothing real-LOA.
    leaf = tmp_path / "mock" / "ds" / "stage" / "base__11223344"
    prov = _write_provenance(leaf, privacy_class="mock", shareable=True)
    doc = tmp_path / "REAL_LOA_DO_NOT_COMMIT.md"  # a doc, not a leaf
    doc.write_text("# sentinel\n")
    table = leaf / "table.tsv"
    table.write_text("a\tb\n")

    rc = _run([prov, doc, table], tmp_path)
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_empty_set_passes(tmp_path):
    # --paths with a dir that has no files -> clean.
    (tmp_path / "empty").mkdir()
    assert _run([tmp_path / "empty"], tmp_path) == 0


# --------------------------------------------------------------------------- #
# the real sentinel doc itself must pass (it's a doc, not a leaf)             #
# --------------------------------------------------------------------------- #
def test_repo_sentinel_doc_passes(tmp_path):
    sentinel = (
        _REPO / "CDDF_analysis" / "hbi" / "tutorial_data"
        / "REAL_LOA_DO_NOT_COMMIT.md"
    )
    assert sentinel.exists(), "sentinel doc should exist in the repo"
    assert guard.main(["--root", str(_REPO), "--paths", str(sentinel)]) == 0


# --------------------------------------------------------------------------- #
# field-name guard: confirm we key on the producer's actual schema            #
# --------------------------------------------------------------------------- #
def test_field_names_match_producer():
    """Belt-and-suspenders: the guard keys on privacy.class / privacy.shareable
    exactly as provenance.privacy_class() produces them."""
    spec = importlib.util.spec_from_file_location(
        "_prov_mod", _REPO / "CDDF_analysis" / "hbi" / "provenance.py"
    )
    prov_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prov_mod)

    real = prov_mod.privacy_class([{"privacy": {"class": "real-LOA"}}])
    assert real == {"class": "real-LOA", "shareable": False}
    mock = prov_mod.privacy_class([])
    assert mock == {"class": "mock", "shareable": True}
