"""The crossmock aggregator must REFUSE a per-leg JSON it cannot vouch for.

Why this file exists.  ``build_crossmock_transfer_artifact.py`` stamps ONE 40-char
``code_commit`` on the aggregate and thereby asserts that every leg folded into it was
produced by that code.  Nothing enforced that assertion: ``--reuse`` read whatever JSON
happened to sit on disk and copied its ``code_commit`` into the metadata without ever
looking at it.  A leg planted with ``code_commit = "deadbeefdeadbeef-dirty"`` and
``R0 = 99.0`` was ingested and the artifact still stamped clean -- and 7 of the 8
committed legs came through that path, which made the aggregate's provenance claim
unfalsifiable rather than true.

These tests are deliberately OMISSION-SENSITIVE: each one plants a leg that the
pre-guard code accepted, and asserts the guard now raises.  Deleting any single check
in ``validate_leg_stamp`` turns one of them red.
"""
from __future__ import annotations

import json
import os
import subprocess

import pytest

from CDDF_analysis.hbi.build_crossmock_transfer_artifact import (
    LegStampError,
    _REPO,
    capture_code_commit,
    validate_leg_stamp,
)


@pytest.fixture(scope="module")
def head_sha():
    return capture_code_commit()["sha"]


def _leg(stamp):
    """A minimal per-leg payload: only ``metadata.code_commit`` is under test."""
    return {"metadata": {"code_commit": stamp}, "variants": {"A": {}}}


def _call(leg, head_sha, **kw):
    return validate_leg_stamp(leg, "loa0/london0", "/tmp/planted_leg.json",
                              head_sha, **kw)


# ---------------------------------------------------------------------------
# the exact leg that got through
# ---------------------------------------------------------------------------
def test_the_planted_leg_that_was_ingested_is_now_refused(head_sha):
    """The regression: 'deadbeefdeadbeef-dirty' + R0=99.0 was silently accepted."""
    planted = _leg("deadbeefdeadbeef-dirty")
    planted["variants"]["A"]["r0_dndx"] = 99.0
    with pytest.raises(LegStampError) as e:
        _call(planted, head_sha)
    # It must be refused for being DIRTY -- the first disqualifying property a reader
    # would check -- not incidentally for failing the hex-format check.
    #
    # Match on text unique to the dirty branch, NOT on the substring "dirty": the
    # error message interpolates the stamp itself, and "deadbeefdeadbeef-dirty"
    # contains "dirty", so a naive match="dirty" passes off ANY branch and the
    # assertion survives deleting the check it exists to protect. (Caught by
    # mutation testing; the first version of this test had exactly that bug.)
    assert "headline-provenance rule" in str(e.value)


# ---------------------------------------------------------------------------
# each check, isolated
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stamp", [None, "", "unknown"])
def test_unstamped_leg_is_refused(stamp, head_sha):
    with pytest.raises(LegStampError, match="code_commit"):
        _call(_leg(stamp), head_sha)


def test_dirty_leg_is_refused_even_when_the_sha_is_real(head_sha):
    """A real commit + a dirty tree is still unquotable: git cannot recover the tree.

    Matches text unique to the dirty branch -- see the note in the planted-leg test
    above on why match="dirty" is worthless here.
    """
    with pytest.raises(LegStampError, match="headline-provenance rule"):
        _call(_leg(head_sha + "-dirty"), head_sha)


@pytest.mark.parametrize("stamp", ["v0.1.0", "desi_y3", "HEAD", "HEAD~1",
                                   "lls-subdla-cddf"])
def test_movable_ref_is_refused(stamp, head_sha):
    """A tag or branch name is movable and cannot pin code, even though git
    resolves all of these perfectly well today."""
    with pytest.raises(LegStampError, match="hex object name"):
        _call(_leg(stamp), head_sha)


def test_unresolvable_sha_is_refused(head_sha):
    with pytest.raises(LegStampError, match="does not resolve"):
        _call(_leg("0" * 40), head_sha)


def test_leg_from_a_different_commit_is_refused_by_default(head_sha):
    """The subtle one: a perfectly clean, perfectly resolvable leg built by DIFFERENT
    code. Folding it in misattributes it to the aggregate's commit."""
    parent = subprocess.check_output(
        ["git", "rev-parse", "HEAD~1"], cwd=_REPO, text=True).strip()
    assert parent != head_sha
    with pytest.raises(LegStampError, match="but the aggregate stamps"):
        _call(_leg(parent), head_sha)


def test_mismatch_can_be_opted_into_but_dirty_still_cannot(head_sha):
    """--allow-stamp-mismatch relaxes ONLY the commit-equality check."""
    parent = subprocess.check_output(
        ["git", "rev-parse", "HEAD~1"], cwd=_REPO, text=True).strip()
    out = _call(_leg(parent), head_sha, allow_mismatch=True)
    assert out["matches_aggregate"] is False
    assert out["resolved_sha40"] == parent
    # the escape hatch must NOT also let a dirty leg through
    with pytest.raises(LegStampError, match="dirty"):
        _call(_leg(parent + "-dirty"), head_sha, allow_mismatch=True)


# ---------------------------------------------------------------------------
# the happy path, and the abbreviated-sha flag the strict audit needs
# ---------------------------------------------------------------------------
def test_matching_full_sha_passes_and_is_not_flagged_abbreviated(head_sha):
    out = _call(_leg(head_sha), head_sha)
    assert out["matches_aggregate"] is True
    assert out["abbreviated"] is False
    assert out["resolved_sha40"] == head_sha


def test_abbreviated_stamp_passes_but_is_flagged(head_sha):
    """9 of 17 committed artifacts stamp 7-char shas. They resolve, so they must not
    hard-fail here -- but the aggregate has to RECORD that they are abbreviated so the
    strict-sha provenance table can count them."""
    out = _call(_leg(head_sha[:7]), head_sha)
    assert out["matches_aggregate"] is True
    assert out["abbreviated"] is True
    assert out["resolved_sha40"] == head_sha


# ---------------------------------------------------------------------------
# the guard must be WIRED IN, not merely defined
# ---------------------------------------------------------------------------
def test_run_leg_reuse_path_actually_calls_the_guard(tmp_path, head_sha,
                                                     monkeypatch):
    """Defining validate_leg_stamp is worthless if run_leg never calls it. This is the
    test that fails if someone drops the call from the --reuse branch."""
    from CDDF_analysis.hbi import build_crossmock_transfer_artifact as B

    leg = {"key": "london0", "out_basename": "leg.json", "driver": "x", "extra": []}
    out_dir = tmp_path / "loa0" / "london0"
    out_dir.mkdir(parents=True)
    (out_dir / "leg.json").write_text(json.dumps(_leg("deadbeefdeadbeef-dirty")))

    monkeypatch.setattr(B, "leg_argv",
                        lambda *a, **k: (["true"], str(out_dir)))
    with pytest.raises(LegStampError):
        B.run_leg(leg, "loa0", str(tmp_path), "python", reuse=True,
                  agg_sha=head_sha)


def test_guard_is_skipped_only_when_agg_sha_is_explicitly_none(tmp_path, head_sha,
                                                               monkeypatch):
    """agg_sha=None is the documented 'no provenance context' escape used by unit
    tests; it must be the ONLY way to bypass the guard."""
    from CDDF_analysis.hbi import build_crossmock_transfer_artifact as B

    leg = {"key": "london0", "out_basename": "leg.json", "driver": "x", "extra": []}
    out_dir = tmp_path / "loa0" / "london0"
    out_dir.mkdir(parents=True)
    (out_dir / "leg.json").write_text(json.dumps(_leg("deadbeefdeadbeef-dirty")))
    monkeypatch.setattr(B, "leg_argv",
                        lambda *a, **k: (["true"], str(out_dir)))

    d, _argv, _out, _dt, how = B.run_leg(leg, "loa0", str(tmp_path), "python",
                                         reuse=True, agg_sha=None)
    assert how == "reused"
    assert d["metadata"]["code_commit"] == "deadbeefdeadbeef-dirty"
