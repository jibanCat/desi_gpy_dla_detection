"""The provenance guard must check EVERY routine a rederive names, not just the first.

A multi-step rederive ("run the three legs, then aggregate") names its real
artifact-builder LAST. Resolving only the first ``*.py`` token lets a missing
builder ride in behind a leg script that does exist -- which is precisely the
ORPHANED class the guard was written to catch, and which was observed in the
wild: ``crossmock_transfer_loa0.json`` rederives via ``build_artifact_loa0.py``,
a file present in no commit and no worktree, behind a leg script that is present.
"""

import subprocess

import pytest

from CDDF_analysis.unblind import provenance as P


@pytest.fixture(scope="module")
def repo():
    return P.repo_root()


@pytest.fixture(scope="module")
def head(repo):
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    return out.stdout.strip()


def _md(rederive, commit):
    return {"code_commit": commit, "rederive": rederive}


def test_missing_final_builder_is_orphaned_not_rederivable(repo, head):
    """The exact wild instance: a real leg script first, a nonexistent builder last."""
    md = _md(
        "python CDDF_analysis/hbi/track_c_tf_2lpt1.py --point-only && "
        "python build_artifact_loa0.py --out crossmock_transfer_loa0.json",
        head,
    )
    res = P.classify(md, repo=repo)
    assert res.status == P.ORPHANED, (
        f"guard returned {res.status}; a rederive naming a nonexistent builder "
        "must never classify as re-derivable"
    )
    assert not res.ok
    assert any("build_artifact_loa0.py" in m for m in res.messages)


def test_first_token_alone_would_have_passed(repo, head):
    """Pin the bug: the leg script alone IS re-derivable, so first-token resolution passes."""
    md = _md("python CDDF_analysis/hbi/track_c_tf_2lpt1.py --point-only", head)
    assert P.classify(md, repo=repo).status == P.RE_DERIVABLE


def test_unexpanded_shell_template_fails_loudly(repo, head):
    md = _md("for leg in a b; do python CDDF_analysis/hbi/track_c_tf_${leg}.py; done", head)
    res = P.classify(md, repo=repo)
    assert res.status == P.ORPHANED
    assert any("UNEXPANDED" in m for m in res.messages)


def test_bare_basename_resolves_to_its_tracked_path(repo, head):
    md = _md("cd CDDF_analysis/hbi && python track_c_tf_2lpt1.py", head)
    res = P.classify(md, repo=repo)
    assert res.status == P.RE_DERIVABLE
    assert res.routines == ["CDDF_analysis/hbi/track_c_tf_2lpt1.py"]


def test_explicit_routine_list_is_authoritative(repo, head):
    md = {
        "code_commit": head,
        "routine": ["CDDF_analysis/hbi/track_c_tf_2lpt1.py", "does_not_exist.py"],
        "rederive": "python CDDF_analysis/hbi/track_c_tf_2lpt1.py",
    }
    res = P.classify(md, repo=repo)
    assert res.status == P.ORPHANED, "an explicit routine list must be checked in full"
    assert len(res.routines) == 2


def test_all_routines_recorded(repo, head):
    md = _md(
        "python CDDF_analysis/hbi/track_c_tf_2lpt1.py && "
        "python CDDF_analysis/hbi/track_c_tf_london0.py",
        head,
    )
    res = P.classify(md, repo=repo)
    assert res.status == P.RE_DERIVABLE
    assert len(res.routines) == 2
