"""The repo-wide artifact audit: every COMMITTED JSON artifact, in BOTH worktrees.

"Our artifacts pass check_artifact" is a claim about a UNION of two worktrees with
no ancestry relation, so it can only be checked by walking both.  This locks the
audit's contract: it reads the COMMITTED blob (never the working tree), it excludes
test fixtures by default, and it reports one row per artifact.
"""

import json
import os
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.unblind import audit as A  # noqa: E402
from CDDF_analysis.unblind import provenance as P  # noqa: E402

SECOND_WT = "/home/mfho/hbi_mcmc_wt"


@pytest.fixture(scope="module")
def rows():
    return A.audit_worktree(_REPO)


def test_audit_excludes_test_fixtures_by_default(rows):
    assert rows, "the audit found no committed artifacts at all"
    assert not any(r.path.startswith("tests/fixtures/") for r in rows)
    with_fx = A.audit_worktree(_REPO, include_fixtures=True)
    assert len(with_fx) > len(rows)


def test_every_committed_artifact_in_this_worktree_is_rederivable(rows):
    bad = [f"{r.path} -> {r.status}: {r.reason}" for r in rows if not r.ok]
    assert not bad, "\n".join(bad)


def test_audit_reads_the_committed_blob_not_the_working_tree(rows, tmp_path):
    """An uncommitted edit to an artifact must NOT change its audited status."""
    target = next(r for r in rows if r.ok)
    path = os.path.join(_REPO, target.path)
    original = open(path).read()
    doc = json.loads(original)
    block, key = P.load_stamp_block(doc)
    try:
        block["code_commit"] = "v0.1.0"          # would be MOVABLE_REF if read live
        open(path, "w").write(json.dumps(doc))
        again = next(r for r in A.audit_worktree(_REPO) if r.path == target.path)
        assert again.status == P.RE_DERIVABLE, (
            "the audit read the WORKING TREE; an uncommitted edit must not launder or "
            "poison a committed artifact's status")
    finally:
        open(path, "w").write(original)


def test_summary_counts_match_rows(rows):
    s = A.summarize(rows)
    assert s["total"] == len(rows)
    assert s["re_derivable"] == sum(1 for r in rows if r.ok)
    assert sum(s["by_status"].values()) == s["total"]


def test_table_renders_without_science_values(rows):
    txt = A.render_table(rows)
    assert "STATUS" in txt and "SCHEMA" in txt
    for r in rows:
        assert r.path in txt


def test_markdown_table_is_pipe_delimited(rows):
    md = A.render_table(rows, markdown=True)
    assert md.splitlines()[0].startswith("| ") and "---" in md.splitlines()[1]


@pytest.mark.skipif(not os.path.isdir(SECOND_WT), reason="second worktree absent")
def test_second_worktree_feedforward_artifacts_are_rederivable():
    """The 9 FF artifacts (calccddf_* / ff_fp_*) stamp under 'provenance'; before the
    reader fix ALL of them audited NOT_STAMPED."""
    rows = A.audit_worktree(SECOND_WT)
    ff = [r for r in rows if os.path.basename(r.path).startswith(("calccddf_", "ff_fp_"))]
    assert len(ff) == 9, [r.path for r in ff]
    assert all(r.schema_key == "provenance" for r in ff)
    bad = [f"{r.path} -> {r.status}" for r in ff if not r.ok]
    assert not bad, "\n".join(bad)


@pytest.mark.skipif(not os.path.isdir(SECOND_WT), reason="second worktree absent")
def test_cli_reports_the_union_and_exits_zero():
    env = dict(os.environ, PYTHONPATH=_REPO, OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-m", "CDDF_analysis.unblind.audit",
                        "--format", "json"], cwd=_REPO, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = json.loads(r.stdout)
    assert out["summary"]["total"] == out["summary"]["re_derivable"]
    assert {os.path.basename(w["worktree"]) for w in out["rows"]} == {
        "desi_gpy_dla_detection", "hbi_mcmc_wt"}


@pytest.mark.skipif(not os.path.isdir(SECOND_WT), reason="second worktree absent")
def test_strict_sha_mode_flags_the_abbreviated_stamps():
    rows = A.audit(require_full_sha=True)
    abbrev = [r for r in rows if r.status == P.ABBREVIATED_SHA]
    assert abbrev, "expected the 7-char FF stamps to be flagged under --strict-sha"
    assert all(r.stamp_kind == P.STAMP_ABBREV_SHA for r in abbrev)
