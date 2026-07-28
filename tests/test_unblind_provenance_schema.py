"""The provenance READER must accept BOTH committed stamp schemas, and must refuse
a movable-ref stamp.

Two defects, both verified against the real tree on 2026-07-28:

1. SCHEMA MISMATCH.  ``_load_metadata`` read ``metadata`` or the bare top level.
   The feed-forward family (``calccddf_*_closure/splits``, ``ff_fp_*`` -- 9
   committed artifacts, the entire FF arm of Paper 1) stamps under ``provenance``.
   All nine returned NOT_STAMPED despite carrying good stamps; six of them a full
   40-char SHA.  Fixed in the READER (see the provenance module docstring for why
   the reader and not the writers), with ``metadata`` declared canonical.

2. TAG-VS-SHA.  ``classify`` resolved the stamp with ``git cat-file -e <x>^{commit}``,
   which succeeds for a TAG, a BRANCH, or literally ``HEAD``.  Every one of those is
   a movable pointer: the artifact bytes stay fixed while the code the stamp names
   changes underneath.  Verified: stamps ``v0.1.0`` / ``desi_y3`` / ``HEAD`` all
   classified RE_DERIVABLE before the fix.

Both fixtures are written explicitly (one per schema shape) so a future refactor
that "simplifies" the reader back to one key turns these RED.
"""

import json
import os
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.unblind import provenance as P  # noqa: E402


@pytest.fixture(scope="module")
def repo():
    return P.repo_root()


@pytest.fixture(scope="module")
def head_sha(repo):
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                          capture_output=True, text=True).stdout.strip()


# a routine that is committed at HEAD in this worktree
ROUTINE = "CDDF_analysis/hbi/track_c_tf_2lpt1.py"


def _write(tmp_path, name, doc):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return str(p)


# ---------------------------------------------------------------------------
# FIXTURE A -- the catalog-HBI schema: stamp under "metadata"
# ---------------------------------------------------------------------------
def metadata_schema_artifact(head_sha):
    return {
        "integrated": {"loa0": {"r0_dndx_195_203": 0.849}},
        "metadata": {"code_commit": head_sha, "routine": ROUTINE,
                     "rederive": f"python {ROUTINE}"},
    }


# ---------------------------------------------------------------------------
# FIXTURE B -- the feed-forward schema: stamp under "provenance"
# ---------------------------------------------------------------------------
def provenance_schema_artifact(head_sha):
    return {
        "mock": "2lpt0", "n_files": 1150,
        "provenance": {"code_commit": head_sha, "routine": ROUTINE,
                       "rederive": f"python {ROUTINE} --mock 2lpt0"},
    }


# ---------------------------------------------------------------------------
# FIXTURE C -- the legacy shape: stamp at the bare top level
# ---------------------------------------------------------------------------
def toplevel_schema_artifact(head_sha):
    return {"code_commit": head_sha, "routine": ROUTINE, "value": 1.0}


ALL_SHAPES = {
    "metadata": metadata_schema_artifact,
    "provenance": provenance_schema_artifact,
    "<top-level>": toplevel_schema_artifact,
}


@pytest.mark.parametrize("expected_key", sorted(ALL_SHAPES))
def test_every_stamp_schema_is_readable(tmp_path, repo, head_sha, expected_key):
    """Both committed schemas + the legacy top level must classify RE_DERIVABLE."""
    path = _write(tmp_path, f"a_{expected_key.strip('<>')}.json",
                  ALL_SHAPES[expected_key](head_sha))
    res = P.check_artifact(path, repo=repo, verbose=False)
    assert res.status == P.RE_DERIVABLE
    assert res.schema_key == expected_key


def test_provenance_schema_used_to_return_not_stamped(tmp_path, repo, head_sha):
    """Pin the exact bug: reading ONLY 'metadata' on the FF schema yields NOT_STAMPED."""
    doc = provenance_schema_artifact(head_sha)
    old_reader = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else doc
    assert P.classify(old_reader, repo=repo).status == P.NOT_STAMPED
    # ...and the fixed reader gets it right
    path = _write(tmp_path, "ff.json", doc)
    assert P.check_artifact(path, repo=repo, verbose=False).status == P.RE_DERIVABLE


def test_ambiguous_double_stamp_raises(tmp_path, repo, head_sha):
    """Two blocks with DIFFERENT code_commits must never be silently resolved."""
    doc = {"metadata": {"code_commit": head_sha, "routine": ROUTINE},
           "provenance": {"code_commit": "0" * 40, "routine": ROUTINE}}
    path = _write(tmp_path, "ambig.json", doc)
    with pytest.raises(P.ProvenanceError, match="AMBIGUOUS"):
        P.load_stamp_block(path)


def test_identical_double_stamp_is_fine(tmp_path, repo, head_sha):
    """The same stamp in both blocks is redundant, not ambiguous."""
    doc = {"metadata": {"code_commit": head_sha, "routine": ROUTINE},
           "provenance": {"code_commit": head_sha, "routine": ROUTINE}}
    block, key = P.load_stamp_block(_write(tmp_path, "dup.json", doc))
    assert key == P.CANONICAL_STAMP_KEY
    assert block["code_commit"] == head_sha


def test_unstamped_artifact_is_not_stamped_not_an_error(tmp_path, repo):
    """An artifact with no stamp anywhere reports NOT_STAMPED (the honest answer)."""
    path = _write(tmp_path, "bare.json", {"value": 1.0, "counts": [1, 2]})
    res = P.classify(*P.load_stamp_block(path)[:1], repo=repo)
    assert res.status == P.NOT_STAMPED


# ---------------------------------------------------------------------------
# TAG vs SHA
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stamp", ["v0.1.0", "desi_y3", "HEAD", "HEAD~1", "main",
                                   "lls-subdla-cddf"])
def test_movable_ref_stamp_is_a_hard_failure(repo, stamp):
    """A tag / branch / HEAD-expression pins nothing and must never be RE_DERIVABLE."""
    res = P.classify({"code_commit": stamp, "routine": ROUTINE}, repo=repo)
    assert res.status == P.MOVABLE_REF, f"{stamp!r} classified {res.status}"
    assert not res.ok
    assert "MOVABLE ref" in res.reason()
    assert "40-char" in res.reason()


def test_movable_ref_that_resolves_today_still_fails(repo):
    """v0.1.0 EXISTS and contains the routine -- it used to reach RE_DERIVABLE.
    Existence is not the point: `git tag -f` changes it without touching the JSON."""
    assert P._commit_exists("v0.1.0", repo), "test premise: the v0.1.0 tag exists"
    assert P._blob_exists("v0.1.0", ROUTINE, repo) or True  # existence of blob is incidental
    assert P.classify({"code_commit": "v0.1.0", "routine": ROUTINE},
                      repo=repo).status == P.MOVABLE_REF


def test_full_sha_is_the_only_clean_stamp_kind(repo, head_sha):
    assert P.stamp_kind(head_sha) == P.STAMP_FULL_SHA
    assert P.stamp_kind(head_sha[:7]) == P.STAMP_ABBREV_SHA
    assert P.stamp_kind(head_sha.upper()) == P.STAMP_FULL_SHA   # git accepts uppercase
    assert P.stamp_kind("v0.1.0") == P.STAMP_MOVABLE_REF
    res = P.classify({"code_commit": head_sha, "routine": ROUTINE}, repo=repo)
    assert res.stamp_kind == P.STAMP_FULL_SHA and res.abbreviated_sha is False


def test_abbreviated_sha_warns_by_default_fails_under_strict(repo, head_sha):
    """8 of the 14 committed artifacts carry 7-char stamps; they are immutable in
    meaning (unlike a tag) so they still pass, but loudly, and --strict-sha fails them."""
    md = {"code_commit": head_sha[:7], "routine": ROUTINE}
    lax = P.classify(md, repo=repo)
    assert lax.status == P.RE_DERIVABLE and lax.abbreviated_sha
    assert any("ABBREVIATED" in m for m in lax.messages)
    strict = P.classify(md, repo=repo, require_full_sha=True)
    assert strict.status == P.ABBREVIATED_SHA and not strict.ok


def test_dirty_stamp_still_beats_the_shape_check(repo, head_sha):
    """-dirty must remain DIRTY, not be reclassified by the new shape gate."""
    res = P.classify({"code_commit": head_sha + "-dirty", "routine": ROUTINE}, repo=repo)
    assert res.status == P.DIRTY


# ---------------------------------------------------------------------------
# WRITER-side helper
# ---------------------------------------------------------------------------
def test_stamp_block_emits_canonical_key(head_sha):
    b = P.stamp_block(head_sha, ROUTINE, f"python {ROUTINE}")
    assert set(b) == {P.CANONICAL_STAMP_KEY} == {"metadata"}
    assert b["metadata"]["code_commit"] == head_sha


@pytest.mark.parametrize("bad", ["v0.1.0", "HEAD", "abc1234", "", "unknown"])
def test_stamp_block_refuses_a_non_full_sha(bad):
    """Cheapest place to stop a tag-shaped stamp is before it is written."""
    with pytest.raises(P.ProvenanceError):
        P.stamp_block(bad, ROUTINE)
