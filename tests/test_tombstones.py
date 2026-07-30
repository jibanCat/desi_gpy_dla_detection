"""Guards for the retired-artifact tombstones (PI decision 7).

A tombstone retires an artifact *identity*. See
``CDDF_analysis/hbi/tombstones/SCHEMA.md`` for the schema and the four hard rules; this
module is the mechanical enforcement of them:

  1. a tombstone carries NO retired science value  -> ``test_tombstones_carry_no_science_values``
  2. regeneration is a NEW MEASUREMENT / NEW IDENTITY -> ``test_successor_policy_forbids_identity_reuse``
  3. a tombstoned identity is never silently resurrected -> ``test_tombstoned_identity_is_not_resurrected``
  4. a tombstone is never silently deleted -> ``test_pinned_tombstone_set_is_present_and_tracked``

Plus the two that keep the *tripwire* armed:

  * ``test_tripwire_commitments_certify_bitforbit_agreement`` -- the head-vs-crossmock
    bit-for-bit agreement the retired ``subdla_mock_headline.json`` used to demonstrate is
    still certified, UNCONDITIONALLY, from committed data only. No ``pytest.skip`` path.
  * ``test_consumer_no_longer_depends_on_a_tombstoned_path`` -- the consumer test must not
    re-acquire a working-tree dependency on a tombstoned identity (that dependency is
    exactly what turned an ``abs=1e-12`` assertion into a silent skip).

No real-DESI value is referenced anywhere here; the retired artifacts were MOCK (2LPT-0,
loa-124) and this module holds no values at all, mock or otherwise.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOMB_DIR_REL = "CDDF_analysis/hbi/tombstones"
TOMB_DIR = os.path.join(_REPO, TOMB_DIR_REL)

# ---------------------------------------------------------------------------
# THE PIN. Hard rule 4: this set may not shrink silently.
#   tombstone filename -> the retired artifact identity it retires
# ---------------------------------------------------------------------------
TOMBSTONED = {
    "lls_recovery_figures.tombstone.json":
        "CDDF_analysis/hbi/lls_recovery_figures.json",
    "subdla_edge_systematic.tombstone.json":
        "CDDF_analysis/hbi/subdla_edge_systematic.json",
    "subdla_floor_mc_band.tombstone.json":
        "CDDF_analysis/hbi/subdla_floor_mc_band.json",
    "subdla_mock_headline.tombstone.json":
        "CDDF_analysis/hbi/subdla_mock_headline.json",
}

ALLOWED_DEFECT_CODES = {
    "DIRTY_STAMP_NOT_REDERIVABLE",
    "B16_LEAKY_TRUTH",
    "LLS_POPULATION_CONTENT",
    "STALE_FP_RESAMPLE_SEMANTICS",
    "RETIRED_REPORTING_WINDOW",
}

CONSUMER_REL = "tests/test_subdla_forward_headline.py"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _git(*args):
    return subprocess.run(["git"] + list(args), cwd=_REPO,
                          capture_output=True, text=True)


def _is_tracked(relpath):
    return _git("ls-files", "--error-unmatch", "--", relpath).returncode == 0


def _in_head(relpath):
    return _git("cat-file", "-e", f"HEAD:{relpath}").returncode == 0


def _committed_json(relpath, ref="HEAD"):
    r = _git("show", f"{ref}:{relpath}")
    if r.returncode != 0:
        return None
    return json.loads(r.stdout)


def _load_tombstone(name):
    path = os.path.join(TOMB_DIR, name)
    assert os.path.exists(path), (
        f"tombstone {name} is MISSING from {TOMB_DIR_REL}/. Hard rule 4: a tombstone may "
        "not be silently deleted -- deleting it frees the retired identity to be "
        "reoccupied without a record of why it was untrustworthy.")
    with open(path) as fh:
        return json.load(fh)


def _walk(obj, path="$"):
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")


def _commit_float(value):
    """The commitment. sha256 of the exact repr of the double -- NOT the value."""
    return hashlib.sha256(repr(float(value)).encode("utf-8")).hexdigest()


def _resolve(doc, pointer):
    cur = doc
    for tok in pointer.split("/")[1:]:
        tok = tok.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            try:
                cur = cur[int(tok)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            if tok not in cur:
                return None
            cur = cur[tok]
        else:
            return None
    return cur


# ===========================================================================
# hard rule 4 -- the tombstone set is pinned
# ===========================================================================
@pytest.mark.parametrize("name", sorted(TOMBSTONED))
def test_pinned_tombstone_set_is_present_and_tracked(name):
    """Every pinned tombstone must exist AND be tracked by git. A tombstone that lives
    only in a working tree is not a record -- it is the same untracked-scratch failure mode
    that made the retired artifacts unrecoverable in the first place."""
    tomb = _load_tombstone(name)
    rel = f"{TOMB_DIR_REL}/{name}"
    assert _is_tracked(rel), (
        f"{rel} exists but is NOT tracked by git. A tombstone must be committed; an "
        "untracked tombstone is exactly the defect it records.")
    assert tomb["schema"] == "hbi-artifact-tombstone"
    assert tomb["schema_version"] == 1


# ===========================================================================
# hard rule 1 -- no retired science values
# ===========================================================================
@pytest.mark.parametrize("name", sorted(TOMBSTONED))
def test_tombstones_carry_no_science_values(name):
    """A tombstone may hold ints and hex digests, never a float. This is the mechanical
    proxy for 'carries no retired science value': every quantity these artifacts were
    retired for (dN/dX, Omega, f(N), ell(X), lambda_mfp, tau_eff_LL, any recovery ratio R0)
    is a float, so a no-float rule cannot be satisfied while smuggling one.

    LIMIT, stated rather than papered over: this sees JSON *types*, so it cannot see a
    number written inside a prose string. The tombstones do contain such numbers (log-N
    window edges, dex resolutions, the B16 leak range) -- configuration and defect-magnitude
    descriptors, not the retired measurement. See SCHEMA.md hard rule 1: the mechanical
    rule is a floor, not a proof, and a reviewer must still read the prose."""
    tomb = _load_tombstone(name)
    offenders = [p for p, v in _walk(tomb)
                 if isinstance(v, float) and not isinstance(v, bool)]
    assert offenders == [], (
        f"{name} carries float leaves at {offenders} -- tombstones must not carry retired "
        "science values (SCHEMA.md hard rule 1).")
    assert tomb["values_policy"]["carries_science_values"] is False


# ===========================================================================
# schema integrity
# ===========================================================================
@pytest.mark.parametrize("name", sorted(TOMBSTONED))
def test_tombstone_schema_required_fields(name):
    """The record must actually carry what a tombstone is for: what it was, its content
    fingerprint, the exact defect, and the recoverability verdict."""
    tomb = _load_tombstone(name)
    art = tomb["artifact"]
    assert art["path"] == TOMBSTONED[name], (
        f"{name}.artifact.path={art['path']!r} does not match the pinned identity "
        f"{TOMBSTONED[name]!r}.")
    assert len(art["sha256"]) == 64 and int(art["sha256"], 16) >= 0
    assert isinstance(art["bytes"], int) and art["bytes"] > 0
    assert art["what_it_was"].strip()
    assert art["stamp_class"] in ("CLEAN", "DIRTY", "MISSING")
    cc = art["stamped_code_commit"]
    if art["stamp_class"] == "DIRTY":
        assert cc.endswith("-dirty"), (
            f"{name}: stamp_class DIRTY but stamped_code_commit={cc!r}")
    elif art["stamp_class"] == "CLEAN":
        assert len(cc) == 40 and not cc.endswith("-dirty")

    ret = tomb["retirement"]
    assert ret["defects"], f"{name} records no defect -- then why is it retired?"
    codes = {d["code"] for d in ret["defects"]}
    unknown = codes - ALLOWED_DEFECT_CODES
    assert not unknown, f"{name}: unknown defect codes {unknown}"
    for d in ret["defects"]:
        assert len(d["detail"]) > 80, (
            f"{name}: defect {d['code']} has no substantive detail; a defect code without "
            "the exact mechanism is not a record.")
    assert isinstance(ret["recoverable_from_git"], bool)
    # a DIRTY stamp is an information-theoretic hole: it CANNOT be recoverable.
    if "DIRTY_STAMP_NOT_REDERIVABLE" in codes:
        assert ret["recoverable_from_git"] is False, (
            f"{name} claims a DIRTY stamp yet recoverable_from_git=True. The working tree "
            "that produced it was never committed, so git cannot supply it.")

    # the tombstone's own stamp must be a clean 40-char sha
    mc = tomb["metadata"]["code_commit"]
    assert len(mc) == 40 and not mc.endswith("-dirty"), (
        f"{name}.metadata.code_commit={mc!r} is not a clean 40-char sha.")
    assert _git("cat-file", "-e", mc).returncode == 0, (
        f"{name}.metadata.code_commit={mc} is not a commit in this repository.")
    assert tomb["metadata"]["paper_facing"] is False


# ===========================================================================
# hard rule 2 -- regeneration is a NEW measurement with a NEW identity
# ===========================================================================
@pytest.mark.parametrize("name", sorted(TOMBSTONED))
def test_successor_policy_forbids_identity_reuse(name):
    tomb = _load_tombstone(name)
    sp = tomb["successor_policy"]
    assert sp["must_not_reuse_identity"] is True
    assert sp["regeneration_is_a_new_measurement"] is True
    assert sp["requirements"], (
        f"{name}: a tombstone must say what a successor would REQUIRE, otherwise it is a "
        "deletion with prose.")
    assert TOMBSTONED[name] in sp["successor_identity_rule"]


# ===========================================================================
# hard rule 3 -- no silent resurrection
# ===========================================================================
@pytest.mark.parametrize("name", sorted(TOMBSTONED))
def test_tombstoned_identity_is_not_resurrected(name):
    """RED if a file with a tombstoned identity is committed (or staged). Working-tree
    reappearance is tolerated -- these artifacts were untracked scratch and still sit in
    the primary worktree -- but *committing* one is the resurrection."""
    rel = TOMBSTONED[name]
    assert not _is_tracked(rel), (
        f"RESURRECTION: {rel} is tracked by git, but that identity was retired by "
        f"{name}. Regeneration post-B16 is a NEW MEASUREMENT with a NEW ARTIFACT "
        "IDENTITY; write the successor to a different path and stamp it afresh.")
    assert not _in_head(rel), (
        f"RESURRECTION: {rel} exists at HEAD but that identity was retired by {name}.")


@pytest.mark.parametrize("name", sorted(TOMBSTONED))
def test_working_tree_reappearance_must_match_the_recorded_fingerprint(name):
    """A tolerated working-tree copy must be the SAME bytes the tombstone fingerprinted.
    If it differs, someone re-ran the retired routine and wrote the result back under the
    retired identity -- a silent regeneration, which is the resurrection this schema
    forbids even before it reaches the index."""
    tomb = _load_tombstone(name)
    for tree in (_REPO, tomb["artifact"].get("read_from_worktree") or _REPO):
        path = os.path.join(tree, TOMBSTONED[name])
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            raw = fh.read()
        got = hashlib.sha256(raw).hexdigest()
        assert got == tomb["artifact"]["sha256"], (
            f"{path} differs from the fingerprint recorded in {name} "
            f"({got[:12]} != {tomb['artifact']['sha256'][:12]}, {len(raw)} vs "
            f"{tomb['artifact']['bytes']} bytes). The retired routine appears to have "
            "been re-run into the retired identity. Write the successor to a NEW path.")


# ===========================================================================
# the TRIPWIRE -- kept armed, with no file dependency and no skip
# ===========================================================================
def test_tripwire_commitments_certify_bitforbit_agreement():
    """The retired ``subdla_mock_headline.json`` was load-bearing for a TEST, not a result:
    it certified that two independently generated pipelines agree BIT-FOR-BIT on the
    forward cumulative endpoints. Naive retirement turned that ``abs=1e-12`` assertion into
    a ``pytest.skip``.

    Because ``repr()`` of an IEEE-754 double is round-trip exact,
    ``sha256(repr(a)) == sha256(repr(b))`` iff ``a == b`` bit-for-bit. The tombstone
    commits BOTH digests, so this assertion runs unconditionally, needs neither untracked
    file, and carries no value that could be quoted or plotted."""
    tomb = _load_tombstone("subdla_mock_headline.tombstone.json")
    tw = tomb["tripwire"]
    cs = tw["commitments"]
    assert len(cs) >= 4, (
        "the tripwire must certify all four endpoints (dN/dX and Omega at 19.5 and 20.3); "
        f"got {len(cs)}.")
    for c in cs:
        assert len(c["sha256_of_repr"]) == 64
        assert c["sha256_of_repr"] == c["corroborating_sha256_of_repr"], (
            f"tripwire BROKEN at {c['pointer']} vs {c['corroborating_pointer']}: the "
            "committed digests differ, i.e. the two independent forward derivations do "
            "NOT agree bit-for-bit.")
    # distinct endpoints, so the four commitments are not one value repeated
    assert len({c["sha256_of_repr"] for c in cs}) == len(cs), (
        "the four endpoint commitments are not distinct -- the tripwire would pass even if "
        "every pointer had been aimed at the same leaf.")
    assert tw["consumer"].startswith(CONSUMER_REL)


def test_tripwire_derived_commitment_recomputes_from_committed_data():
    """The secondary consumer check (band + DLA-tier == cum(19.5), i.e. the band is a
    difference-slice and not a mislabelled cumulative) also read the retired headline. It
    is re-anchored on the COMMITTED forward artifact: recompute the sum here and require it
    to reproduce the committed digest. This is the one test with real arithmetic teeth --
    it fails if the committed forward artifact changes by a single bit."""
    tomb = _load_tombstone("subdla_mock_headline.tombstone.json")
    derived = tomb["tripwire"]["derived_commitments"]
    assert derived, "the derived (secondary) tripwire commitment is missing."
    for d in derived:
        rel = d["from_committed_artifact"]
        doc = _committed_json(rel)
        assert doc is not None, (
            f"{rel} is not committed at HEAD -- the derived tripwire has lost its anchor.")
        terms = [_resolve(doc, p) for p in d["sum_of_pointers"]]
        assert all(t is not None for t in terms), (
            f"{rel}: derived pointers {d['sum_of_pointers']} did not resolve.")
        total = float(sum(float(t) for t in terms))
        assert _commit_float(total) == d["sha256_of_repr"], (
            f"derived tripwire BROKEN: sum{d['sum_of_pointers']} in {rel} no longer "
            f"reproduces the digest committed for {d['equals_pointer_in_retired_artifact']}"
            " in the retired headline. Either the committed forward artifact changed or "
            "the band is no longer cum(19.5)-cum(20.3).")


def test_consumer_no_longer_depends_on_a_tombstoned_path():
    """The consumer test must not hold a working-tree dependency on a tombstoned identity.
    That dependency is precisely what would let a retirement silently disarm an assertion
    (``if head is None: pytest.skip(...)``). RED if any tombstoned path reappears in the
    consumer source."""
    with open(os.path.join(_REPO, CONSUMER_REL)) as fh:
        src = fh.read()
    for name, rel in sorted(TOMBSTONED.items()):
        assert rel not in src, (
            f"{CONSUMER_REL} references the tombstoned path {rel} (retired by {name}). A "
            "presence-gated read of a retired artifact degrades to a silent skip; use the "
            "committed tombstone commitments instead.")
