"""Tests for validation/handoff: the Paper-lane handoff manifest.

The manifest's whole job is to FAIL CLOSED, so most of these tests build a
handoff-shaped tree in a tmpdir, build the manifest over it, and then break it
in one specific way and assert that ``verify`` says so.

Two tests touch the REAL handoff root read-only (they are skipped if it is not
present): that every curated figure and table has a source entry, and that the
declared registry paths resolve.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from validation.handoff import build_handoff_manifest as bhm      # noqa: E402
from validation.handoff import registry as reg                    # noqa: E402

MANIFEST = bhm.MANIFEST
SUMS = bhm.SUMS


# --------------------------------------------------------------- tiny fixture
def _mini_handoff(tmp_path, monkeypatch):
    """A handoff-shaped tree with two curated files and two canonical sources.

    The registry is monkeypatched so the builder is exercised end-to-end
    without depending on the real 3.3 GB archive.
    """
    root = tmp_path / "H"
    (root / "figures").mkdir(parents=True)
    (root / "tables").mkdir()

    srcdir = tmp_path / "src"
    srcdir.mkdir()
    figsrc = srcdir / "fig_one.png"
    figsrc.write_bytes(b"\x89PNG fake figure bytes")
    tabsrc = srcdir / "TABLE_ONE.md"
    tabsrc.write_text("| a | b |\n|---|---|\n| 1 | 2 |\n")
    othersrc = srcdir / "RESULT.json"
    othersrc.write_text(json.dumps({"median": "PRIVATE"}))

    shutil.copy2(figsrc, root / "figures" / "fig_one.png")
    shutil.copy2(tabsrc, root / "tables" / "TABLE_ONE.md")
    (root / "CANONICAL_SOURCE_MAP.md").write_text("# map\n")

    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "SHA256SUMS").write_text("deadbeef  thing\n")

    monkeypatch.setattr(reg, "FIGURES", [
        ("figures/fig_one.png", str(figsrc), reg.PRIVATE, "test figure"),
    ])
    monkeypatch.setattr(reg, "TABLES", [
        ("tables/TABLE_ONE.md", str(tabsrc), reg.PUBLIC, "test table"),
    ])
    monkeypatch.setattr(reg, "CANONICAL_SOURCES", [
        ("result", str(othersrc), reg.PRIVATE, "result of record"),
        ("table", str(tabsrc), reg.PUBLIC, "table of record"),
    ])
    monkeypatch.setattr(reg, "SUPERSEDED", [
        (str(srcdir / "OLD.json"), "superseded for testing", str(othersrc)),
    ])
    monkeypatch.setattr(reg, "SUPERSEDED_CLASSES", [])
    monkeypatch.setattr(reg, "EXPECTED_FROM_OTHER_AGENTS",
                        [("START_HERE.md", "commander")])
    monkeypatch.setattr(reg, "ARCHIVES", [
        ("test_archive", str(archive), str(archive / "SHA256SUMS"), "durable"),
    ])
    monkeypatch.setattr(reg, "DOC_CLASSIFICATION",
                        {"CANONICAL_SOURCE_MAP.md": reg.PRIVATE})
    return root, srcdir


def _build(root):
    doc = bhm.build(str(root))
    bhm.write(str(root), doc)
    return doc


# ------------------------------------------------------------------ the tests
def test_verify_passes_on_a_freshly_built_handoff(tmp_path, monkeypatch):
    root, _ = _mini_handoff(tmp_path, monkeypatch)
    _build(root)
    assert bhm.verify(str(root)) == []
    assert bhm.main(["--root", str(root), "--verify"]) == 0


def test_verify_fails_when_a_handoff_file_is_mutated(tmp_path, monkeypatch):
    root, _ = _mini_handoff(tmp_path, monkeypatch)
    _build(root)
    target = root / "tables" / "TABLE_ONE.md"
    target.write_text(target.read_text() + "| 3 | 4 |\n")   # one row added
    fails = bhm.verify(str(root))
    assert any("SHA CHANGED handoff file" in f and "TABLE_ONE.md" in f
               for f in fails), fails
    assert bhm.main(["--root", str(root), "--verify"]) == 1


def test_verify_fails_when_a_canonical_source_is_mutated(tmp_path, monkeypatch):
    root, srcdir = _mini_handoff(tmp_path, monkeypatch)
    _build(root)
    (srcdir / "RESULT.json").write_text(json.dumps({"median": "CHANGED"}))
    fails = bhm.verify(str(root))
    assert any("SHA CHANGED canonical source" in f for f in fails), fails
    assert bhm.main(["--root", str(root), "--verify"]) == 1


def test_verify_fails_when_a_listed_file_is_missing(tmp_path, monkeypatch):
    root, srcdir = _mini_handoff(tmp_path, monkeypatch)
    _build(root)
    os.remove(root / "figures" / "fig_one.png")
    os.remove(srcdir / "RESULT.json")
    fails = bhm.verify(str(root))
    assert any("MISSING handoff file" in f for f in fails), fails
    assert any("MISSING canonical source" in f for f in fails), fails
    assert bhm.main(["--root", str(root), "--verify"]) == 1


def test_verify_fails_on_an_unlisted_extra_file(tmp_path, monkeypatch):
    """A file that appears after the build must not pass silently."""
    root, _ = _mini_handoff(tmp_path, monkeypatch)
    _build(root)
    (root / "tables" / "SNEAKED_IN.md").write_text("not in the manifest\n")
    fails = bhm.verify(str(root))
    assert any("UNLISTED file" in f and "SNEAKED_IN" in f for f in fails), fails


def test_private_files_are_flagged(tmp_path, monkeypatch):
    root, _ = _mini_handoff(tmp_path, monkeypatch)
    doc = _build(root)
    by_path = {r["path"]: r["classification"] for r in doc["handoff_files"]}
    assert by_path["figures/fig_one.png"] == reg.PRIVATE
    assert by_path["tables/TABLE_ONE.md"] == reg.PUBLIC
    # a handoff DOCUMENT that quotes real values is private by override
    assert by_path["CANONICAL_SOURCE_MAP.md"] == reg.PRIVATE
    assert doc["counts"]["private_handoff_files"] == 2
    # the canonical result of record must be classified private too
    cls = {r["key"]: r["classification"] for r in doc["canonical_sources"]}
    assert cls["result"] == reg.PRIVATE
    assert cls["table"] == reg.PUBLIC


def test_every_figure_and_table_has_a_source_entry(tmp_path, monkeypatch):
    root, _ = _mini_handoff(tmp_path, monkeypatch)
    doc = _build(root)
    for group in ("figures", "tables"):
        assert doc[group], group
        for row in doc[group]:
            assert row["canonical_source"]
            assert row["source_sha256"] and len(row["source_sha256"]) == 64
            assert os.path.isfile(os.path.join(str(root), row["handoff_path"]))
    listed = {r["path"] for r in doc["handoff_files"]}
    for group in ("figures", "tables"):
        for row in doc[group]:
            assert row["handoff_path"] in listed


def test_manifest_schema(tmp_path, monkeypatch):
    root, _ = _mini_handoff(tmp_path, monkeypatch)
    doc = _build(root)
    for key in ("schema", "handoff_date", "generated_utc", "status",
                "handoff_root", "repositories", "archival_tags", "canonical",
                "figures", "tables", "archives", "handoff_files",
                "canonical_sources", "superseded_not_for_use",
                "superseded_classes", "expected_from_other_agents", "counts",
                "verify_command"):
        assert key in doc, key
    assert doc["schema"] == bhm.SCHEMA
    assert doc["handoff_date"] == "2026-09-16"
    for name in ("notes", "code"):
        st = doc["repositories"][name]
        assert st["resolved"] is True
        assert len(st["head"]) == 40
        assert isinstance(st["dirty"], bool)
        assert isinstance(st["dirty_files"], list)
    # every archival tag is either a 40-char commit or PENDING_TAG
    assert set(doc["archival_tags"]) == set(reg.ARCHIVAL_TAGS)
    for tag, row in doc["archival_tags"].items():
        for repo, val in row.items():
            assert val == "PENDING_TAG" or len(val) == 40, (tag, repo, val)
    # the sums file carries both sections
    text = open(os.path.join(str(root), SUMS)).read()
    assert "section 1" in text and "section 2" in text
    assert doc["counts"]["handoff_files"] == len(doc["handoff_files"])


def test_missing_manifest_fails_closed(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    fails = bhm.verify(str(root))
    assert fails and "MISSING MANIFEST" in fails[0]
    assert bhm.main(["--root", str(root), "--verify"]) == 1


# ------------------------------------------- read-only checks on the real tree
@pytest.mark.skipif(not os.path.isdir(reg.H), reason="handoff root absent")
def test_real_handoff_registry_paths_resolve():
    """Every declared figure/table source and canonical source exists."""
    missing = [s for _h, s, _c, _n in reg.FIGURES + reg.TABLES
               if not os.path.isfile(s)]
    assert missing == [], missing
    missing_src = [p for _k, p, _c, _r in reg.CANONICAL_SOURCES
                   if not os.path.exists(p)]
    assert missing_src == [], missing_src


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(reg.H, MANIFEST)),
    reason="handoff manifest not built")
def test_real_handoff_verifies():
    assert bhm.verify(reg.H) == []
