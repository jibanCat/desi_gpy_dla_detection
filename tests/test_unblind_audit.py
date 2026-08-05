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


#: the feed-forward artifacts that stamp under the NON-canonical 'provenance' key.
#: Before the reader fix these ALL audited NOT_STAMPED -- that is what this set pins.
#: It is a SET of basenames, not a count: the FF arm lives on a branch this worktree
#: does not track, so a bare `len(ff) == N` breaks every time that branch adds an
#: artifact (it did, 2026-07-28: calccddf_vs_hbi.json -- which stamps under the
#: CANONICAL 'metadata' key and so is deliberately NOT in this set).
FF_PROVENANCE_STAMPED = frozenset({
    "calccddf_2lpt0_closure.json",
    "calccddf_2lpt0_splits.json",
    "calccddf_london0_closure.json",
    "calccddf_london0_splits.json",
    "calccddf_saclay0_closure.json",
    "calccddf_saclay0_splits.json",
    "ff_fp_2lpt0.json",
    "ff_fp_london0.json",
    "ff_fp_saclay0.json",
})


@pytest.mark.skipif(not os.path.isdir(SECOND_WT), reason="second worktree absent")
def test_second_worktree_feedforward_artifacts_are_rederivable():
    """Every FF artifact (calccddf_* / ff_fp_*) in the second worktree is RE_DERIVABLE,
    and the known 'provenance'-stamped subset is still read correctly.

    Asserts, in order:
      1. every basename in FF_PROVENANCE_STAMPED is present (nothing silently vanished);
      2. each of those still resolves its stamp under schema_key == 'provenance';
      3. EVERY FF artifact found -- including ones added on that branch after this test
         was written -- is RE_DERIVABLE.
    """
    rows = A.audit_worktree(SECOND_WT)
    ff = [r for r in rows if os.path.basename(r.path).startswith(("calccddf_", "ff_fp_"))]
    by_name = {os.path.basename(r.path): r for r in ff}
    assert len(by_name) == len(ff), f"duplicate FF basenames: {[r.path for r in ff]}"

    missing = sorted(FF_PROVENANCE_STAMPED - set(by_name))
    assert not missing, f"FF artifacts disappeared from {SECOND_WT}: {missing}"

    wrong_key = {n: by_name[n].schema_key for n in sorted(FF_PROVENANCE_STAMPED)
                 if by_name[n].schema_key != "provenance"}
    assert not wrong_key, f"stamp key moved for: {wrong_key}"

    bad = [f"{r.path} -> {r.status}: {r.reason}" for r in ff if not r.ok]
    assert not bad, "\n".join(bad)


#: Artifacts KNOWN not to be RE_DERIVABLE yet, each with the reason and the owner of
#: the fix.  This list is a LEDGER, not a mute button: an artifact in it must still be
#: named here with a status, and any artifact NOT in it that fails is a hard failure.
#: When the list empties, the union is fully clean and the CLI must exit 0 -- which the
#: exit-code assertion below enforces automatically, with no test edit needed.
KNOWN_NOT_REDERIVABLE = {
    # ORPHANED: the stamp is an ABBREVIATED 7-char SHA. A concurrent agent owns the
    # re-stamp on the hbi-mcmc branch; do NOT "fix" the artifact from this worktree.
    "CDDF_analysis/hbi_mcmc/rung9_forward_selftest.json": "ORPHANED",
}


@pytest.mark.skipif(not os.path.isdir(SECOND_WT), reason="second worktree absent")
def test_cli_reports_the_union_and_its_exit_code_tells_the_truth():
    """The CLI's contract, expressed so it cannot rot.

    The old assertion was ``summary.total == summary.re_derivable`` plus
    ``returncode == 0``, which pins the *current cleanliness of the repo* rather than
    the *behaviour of the gate* -- so it goes red whenever any artifact anywhere is
    mid-repair, and it can be made green by deleting artifacts.  What this test
    actually needs to guarantee is:

      1. the audit covers the UNION of both Paper-1 worktrees and is non-empty;
      2. the summary is internally consistent with the rows it reports;
      3. the EXIT CODE agrees with the summary -- 0 if and only if every audited
         artifact is RE_DERIVABLE.  A gate whose exit code disagrees with its own
         report is the only unrecoverable failure here;
      4. every artifact that is NOT re-derivable is on the KNOWN_NOT_REDERIVABLE
         ledger with the status recorded there.  Anything else is a regression.
    """
    env = dict(os.environ, PYTHONPATH=_REPO, OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-m", "CDDF_analysis.unblind.audit",
                        "--format", "json"], cwd=_REPO, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode in (0, 1), (
        f"unexpected exit {r.returncode} (2 = refused to audit)\n"
        + r.stdout[-2000:] + r.stderr[-2000:])
    out = json.loads(r.stdout)
    summary, rows = out["summary"], out["rows"]

    # 1. the union, non-empty
    assert {os.path.basename(w["worktree"]) for w in rows} == {
        "desi_gpy_dla_detection", "hbi_mcmc_wt"}
    assert summary["total"] > 0

    # 2. the summary describes the rows it printed
    assert summary["total"] == len(rows)
    ok = [w for w in rows if w["status"] == P.RE_DERIVABLE]
    assert summary["re_derivable"] == len(ok)
    assert sum(summary["by_status"].values()) == summary["total"]

    # 3. the exit code agrees with the summary -- BOTH directions
    clean = summary["re_derivable"] == summary["total"]
    assert (r.returncode == 0) == clean, (
        f"exit {r.returncode} but re_derivable={summary['re_derivable']}/"
        f"{summary['total']}: the gate's exit code contradicts its own report")

    # 4. nothing fails that is not on the ledger, the ledger is accurate, and no
    #    ledgered artifact has simply VANISHED.  Without the third check this test was
    #    still greenable by DELETING the offending artifact -- the exact flaw it was
    #    written to replace.
    failing = {w["path"]: w["status"] for w in rows if w["status"] != P.RE_DERIVABLE}
    unexpected, mislabelled, vanished = _ledger_violations(
        {w["path"] for w in rows}, failing, KNOWN_NOT_REDERIVABLE)
    assert not unexpected, (
        "NEW un-re-derivable artifact(s) -- a committed stamp that cannot be "
        f"re-derived is exactly what the provenance rule forbids: {unexpected}")
    assert not mislabelled, f"ledger status is stale: {mislabelled}"
    assert not vanished, (
        f"ledgered artifact(s) are no longer audited at all: {vanished}. DELETING a "
        "failing artifact is not a repair, and this gate must not go green because "
        "the evidence left the repo. If the removal is deliberate, drop the ledger "
        "entry in the SAME change -- that edit is reviewable; a silent pass is not.")


@pytest.mark.skipif(not os.path.isdir(SECOND_WT), reason="second worktree absent")
def test_strict_sha_mode_flags_the_abbreviated_stamps():
    rows = A.audit(require_full_sha=True)
    abbrev = [r for r in rows if r.status == P.ABBREVIATED_SHA]
    assert abbrev, "expected the 7-char FF stamps to be flagged under --strict-sha"
    assert all(r.stamp_kind == P.STAMP_ABBREV_SHA for r in abbrev)


# ---------------------------------------------------------------------------
# a top-level CLEAN stamp must not sit on top of DIRTY sub-stamps
# ---------------------------------------------------------------------------
_DIRTY_SHA = __import__("re").compile(r"\b[0-9a-f]{7,40}-dirty\b")

#: keys whose values are PROSE about provenance -- a note that says the word
#: "<sha>-dirty" is documentation, not a stamp.  (Same disclaimer-vs-scanner trap the
#: unblind scanner hit at c596ff7.)
_PROSE_KEY_TOKENS = ("note", "why", "reason", "what", "rule", "defect", "comment",
                     "supersedes", "correction", "evidence", "provenance_note")


def _leaf_key(path: str) -> str:
    """The key the value actually hangs off, with any list indices stripped.

    ``/legs/2lpt1/code_commit_of_run`` -> ``code_commit_of_run``
    ``/metadata/notes[0]``             -> ``notes``
    """
    seg = path.rsplit("/", 1)[-1]
    return __import__("re").sub(r"(\[\d+\])+$", "", seg)


#: A tombstone's REASON FOR EXISTING is to record that a retired artifact carried a dirty
#: stamp. Its `/artifact/stamped_code_commit` is therefore a QUOTATION of someone else's
#: stamp, not a provenance claim of its own -- the same disclaimer-vs-scanner trap as
#: c596ff7, one level up. The exemption is deliberately narrow: it is gated on the doc
#: declaring the tombstone schema, it names exactly ONE path, and a dirty sha anywhere else
#: in a tombstone still fires. The tombstone's OWN stamp
#: (`/metadata/code_commit`) is NOT exempt and is separately required to be a clean
#: 40-char sha by tests/test_tombstones.py::test_tombstone_schema_required_fields.
_TOMBSTONE_SCHEMA = "hbi-artifact-tombstone"
_TOMBSTONE_QUOTED_STAMP_PATHS = frozenset({"/artifact/stamped_code_commit"})


def _stamp_like_dirty_values(doc):
    """Yield (path, value) for every DIRTY-looking sha that is a stamp, not prose."""
    is_tombstone = (isinstance(doc, dict)
                    and doc.get("schema") == _TOMBSTONE_SCHEMA)
    def walk(x, p=""):
        if isinstance(x, dict):
            for k, v in x.items():
                yield from walk(v, f"{p}/{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                yield from walk(v, f"{p}[{i}]")
        else:
            yield p, x
    for path, val in walk(doc):
        if not isinstance(val, str) or not _DIRTY_SHA.search(val):
            continue
        # Scope the exemption to the LEAF key.  Matching the FULL path let any
        # ancestor named note/why/reason/evidence/supersedes/... exempt its entire
        # subtree, which is precisely where an aggregator parks its per-leg stamps.
        if any(t in _leaf_key(path).lower() for t in _PROSE_KEY_TOKENS):
            continue
        # narrow, schema-gated: a tombstone quoting the stamp it retires
        if is_tombstone and path in _TOMBSTONE_QUOTED_STAMP_PATHS:
            continue
        yield path, val


def test_no_committed_artifact_hides_dirty_substamps_under_a_clean_one():
    """A committed artifact must not carry a DIRTY sha ANYWHERE in its body.

    This is the aggregate-provenance failure mode: the top-level stamp is a clean
    40-char sha and audits RE_DERIVABLE, while every per-leg `code_commit_of_run`
    beneath it reads `<sha>-dirty`.  The artifact then makes a provenance claim that
    its own contents contradict, and git cannot recover the tree those legs ran on.
    validate_leg_stamp() (crossmock aggregator) refuses this at WRITE time; this is
    the repo-wide check at REST, over both worktrees.
    """
    bad = {}
    for wt in (_REPO, SECOND_WT):
        if not os.path.isdir(wt):
            continue
        for r in A.audit_worktree(wt):
            doc = A._committed_doc(wt, r.path)
            if doc is None:
                continue
            hits = list(_stamp_like_dirty_values(doc))
            if hits:
                bad[f"{os.path.basename(wt)}:{r.path}"] = hits[:4]
    assert not bad, (
        "committed artifact(s) carry a DIRTY stamp in their body:\n"
        + "\n".join(f"  {k}: {v}" for k, v in bad.items()))


def test_the_dirty_substamp_scanner_is_not_vacuous():
    """Guard the guard: it must fire on the real shape (a clean top-level stamp over
    dirty per-leg stamps) and must NOT fire on prose that merely mentions one."""
    planted = {
        "metadata": {"code_commit": "0" * 40,
                     "provenance_note": "generated at d496f42-dirty; held untracked"},
        "legs": {"2lpt1": {"variantA": {"code_commit_of_run": "d496f42-dirty"}}},
    }
    hits = dict(_stamp_like_dirty_values(planted))
    assert hits == {"/legs/2lpt1/variantA/code_commit_of_run": "d496f42-dirty"}, hits


def test_the_tombstone_exemption_is_narrow_and_cannot_be_abused():
    """Guard the guard's ONE exemption. A tombstone may quote the dirty stamp it retires at
    /artifact/stamped_code_commit and nowhere else, and only while declaring the tombstone
    schema. Four cases, each of which the exemption must get right."""
    quoted = "d496f42a8de932a58055c4d02523996fdb7d962a-dirty"

    # (a) a real tombstone: the quoted stamp is exempt
    tomb = {"schema": _TOMBSTONE_SCHEMA,
            "artifact": {"stamped_code_commit": quoted},
            "metadata": {"code_commit": "0" * 40}}
    assert dict(_stamp_like_dirty_values(tomb)) == {}

    # (b) SAME doc without the schema declaration: NOT exempt. The exemption cannot be
    #     borrowed by an ordinary artifact that happens to use the same key name.
    impostor = {k: v for k, v in tomb.items() if k != "schema"}
    assert dict(_stamp_like_dirty_values(impostor)) == {
        "/artifact/stamped_code_commit": quoted}

    # (c) a tombstone with a dirty sha ANYWHERE ELSE still fires -- the exemption is one
    #     path, not a licence for the whole document.
    smuggler = json.loads(json.dumps(tomb))
    smuggler["artifact"]["also_ran_at"] = quoted
    smuggler["successor_policy"] = {"built_from": quoted}
    assert dict(_stamp_like_dirty_values(smuggler)) == {
        "/artifact/also_ran_at": quoted, "/successor_policy/built_from": quoted}

    # (d) a tombstone whose OWN stamp is dirty still fires. Documenting someone else's
    #     dirty tree must never launder your own.
    self_dirty = json.loads(json.dumps(tomb))
    self_dirty["metadata"]["code_commit"] = quoted
    assert dict(_stamp_like_dirty_values(self_dirty)) == {
        "/metadata/code_commit": quoted}


def test_the_prose_exemption_is_scoped_to_the_LEAF_key():
    """REFEREE (2026-07-29): the exemption was ``any(t in path.lower() ...)`` over the
    FULL path, so ANY ancestor key containing note/why/reason/what/rule/comment/
    evidence/supersedes exempted its WHOLE subtree.  A real per-leg stamp buried under
    a key called ``notes`` or ``evidence`` was invisible to the scanner -- and those are
    exactly the container names an aggregator uses.  Measured pre-fix: this planted
    document yielded ZERO hits, i.e. all three genuine dirty stamps were swallowed.

    The exemption is about what a VALUE is (prose vs stamp), which is decided by the
    key the value hangs off: the LEAF key.  Ancestors say nothing about it.
    """
    planted = {
        # a genuine per-leg stamp under prose-named ANCESTORS -- must be CAUGHT
        "notes": {"legs": {"a": {"code_commit_of_run": "d496f42-dirty"}}},
        "evidence": {"run": {"code_commit": "abc1234-dirty"}},
        "supersedes": {"prior": {"code_commit_of_run": "beefcafe-dirty"}},
        # genuine prose, keyed by a prose LEAF -- must be EXEMPT
        "metadata": {"note": "generated at d496f42-dirty; held untracked",
                     "code_commit": "0" * 40},
        "why": "the legs ran at d496f42-dirty, which is why they stay untracked",
        # prose LIST: the leaf key is still the prose key, indices and all
        "provenance_note": ["ran at d496f42-dirty", "held untracked"],
    }
    hits = dict(_stamp_like_dirty_values(planted))
    assert hits == {
        "/notes/legs/a/code_commit_of_run": "d496f42-dirty",
        "/evidence/run/code_commit": "abc1234-dirty",
        "/supersedes/prior/code_commit_of_run": "beefcafe-dirty",
    }, hits


# ---------------------------------------------------------------------------
# the ledger check, as a pure function so it can be tested on planted rows
# ---------------------------------------------------------------------------
def _ledger_violations(audited_paths, failing, ledger):
    """Three ways the KNOWN_NOT_REDERIVABLE ledger can be violated.

    ``vanished`` is the one the referee flagged: the CLI gate test could be turned
    GREEN by DELETING the offending artifact -- the exact flaw it was written to
    replace.  A ledgered artifact that is no longer audited at all is a violation
    until its ledger entry is removed in the same change, which is a reviewable edit.
    """
    unexpected = {p: s for p, s in failing.items() if p not in ledger}
    mislabelled = {p: (s, ledger[p]) for p, s in failing.items()
                   if p in ledger and ledger[p] != s}
    vanished = sorted(set(ledger) - set(audited_paths))
    return unexpected, mislabelled, vanished


def test_the_ledger_check_catches_deletion_relabelling_and_new_failures():
    """Guard the guard, all five directions on planted rows (no repo state involved)."""
    ledger = {"a.json": "ORPHANED"}
    # clean: the ledgered artifact is present and failing exactly as recorded
    assert _ledger_violations({"a.json", "b.json"}, {"a.json": "ORPHANED"}, ledger) \
        == ({}, {}, [])
    # clean: it was REPAIRED -- present, no longer failing. Not a violation.
    assert _ledger_violations({"a.json", "b.json"}, {}, ledger) == ({}, {}, [])
    # VIOLATION: deleted. This is what made the gate greenable by deletion.
    assert _ledger_violations({"b.json"}, {}, ledger)[2] == ["a.json"]
    # VIOLATION: a new failure that is not on the ledger
    assert _ledger_violations({"a.json", "b.json"}, {"b.json": "DIRTY"}, ledger)[0] \
        == {"b.json": "DIRTY"}
    # VIOLATION: on the ledger but failing for a different recorded reason
    assert _ledger_violations({"a.json"}, {"a.json": "DIRTY"}, ledger)[1] \
        == {"a.json": ("DIRTY", "ORPHANED")}


def test_the_cli_gate_cannot_be_greened_by_deleting_the_offending_artifact():
    """END-TO-END on the REAL ledger: drop every ledgered path from the audited set --
    which is exactly what deleting those files would do -- and the gate must go red.

    This is the assertion the referee asked for: it re-runs the same helper the CLI
    test uses, against the live ledger, with the artifacts removed.
    """
    assert KNOWN_NOT_REDERIVABLE, "the ledger is empty; this guard needs an entry"
    audited_without_them = {"some/other/artifact.json"}
    _u, _m, vanished = _ledger_violations(
        audited_without_them, {}, KNOWN_NOT_REDERIVABLE)
    assert sorted(vanished) == sorted(KNOWN_NOT_REDERIVABLE), (
        "deleting a ledgered artifact must be a hard failure, not a green gate")
