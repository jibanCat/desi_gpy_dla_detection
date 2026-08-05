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

import copy
import glob
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOMB_DIR_REL = "CDDF_analysis/hbi/tombstones"
TOMB_DIR = os.path.join(_REPO, TOMB_DIR_REL)
BUILDER_REL = f"{TOMB_DIR_REL}/build_tombstones.py"
BUILDER = os.path.join(_REPO, "CDDF_analysis", "hbi", "tombstones",
                       "build_tombstones.py")

# Two DIFFERENT roles, kept apart on purpose (defect 3):
#   RECORDING_WORKTREE -- a HISTORICAL fact about the stamp, pinned here independently of
#       the records so a record that redirects `artifact.read_from_worktree` is RED. Host-
#       independent: it is what the record must DECLARE, whatever host runs the test.
#   CANDIDATE_TREES    -- where this HOST is willing to look for a reappeared copy. Also
#       pinned, never read from the record. Absent trees make the byte-level half inert,
#       which is reported explicitly rather than passed over.
RECORDING_WORKTREE = "/home/mfho/desi_gpy_dla_detection"
CANDIDATE_TREES = tuple(dict.fromkeys((_REPO, RECORDING_WORKTREE)))

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

# ---------------------------------------------------------------------------
# THE FINGERPRINT PIN (defect 3). tombstone filename -> (sha256, bytes) of the RETIRED
# artifact, held here so the reappearance guard does not take both the expected digest and
# the path to compare against from the record it is auditing. Hex digests and byte counts
# are not science values; the retired artifacts were MOCK (2LPT-0, loa-124) in any case.
# These must equal the committed records; test_working_tree_reappearance_* asserts both
# directions, so a re-stamp that changes them is RED until this pin is updated deliberately.
# ---------------------------------------------------------------------------
EXPECTED_FINGERPRINT = {
    "lls_recovery_figures.tombstone.json": (
        "84c0c802b44fa0218c32f7802b070e559507f7a3db5687ead7cc7ca82735928a", 6407),
    "subdla_edge_systematic.tombstone.json": (
        "b22c729d75409f7174973a54f67a1bc38afce90bb76f52dacabacdfee28a3742", 14192),
    "subdla_floor_mc_band.tombstone.json": (
        "4b37051ed0c7d65a4fd642ca006085e9e6d6531f767adcab24973514e6e9f5cb", 9340),
    "subdla_mock_headline.tombstone.json": (
        "b13adddae7099f0fa9b99da3fae3ef460a67bdfacf18410017ff0eee96f79268", 59484),
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


def _load_builder():
    spec = importlib.util.spec_from_file_location("_tomb_builder", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_tomb_builder"] = mod
    spec.loader.exec_module(mod)
    return mod


def _source_artifacts_present():
    """True iff a tree that still physically holds all four retired artifacts is reachable,
    which is what ``--check`` needs (they were never committed)."""
    return all(any(os.path.exists(os.path.join(t, rel)) for t in CANDIDATE_TREES)
               for rel in TOMBSTONED.values())


# ===========================================================================
# --check must verify the tombstone PAYLOAD, and something must invoke it
# ===========================================================================
def test_check_mode_treats_only_timestamps_and_the_head_stamp_as_volatile():
    """``--check`` used to drop the whole ``metadata`` AND ``retirement`` blocks before
    comparing, so the record's actual payload -- which defect, and whether the producing
    source is recoverable -- was verified by nothing: flipping ``recoverable_from_git``,
    rewriting ``recovery_note`` to prose that contradicts it and swapping a defect code all
    printed OK with exit 0. Only the two timestamps and the builder's own HEAD sha are
    genuinely volatile; this pins that set from both sides."""
    b = _load_builder()
    assert set(b.VOLATILE_LEAVES) == {
        ("retirement", "retired_utc"),
        ("metadata", "generated_utc"),
        ("metadata", "code_commit"),
    }, (f"VOLATILE_LEAVES={b.VOLATILE_LEAVES} -- --check may only ignore the timestamps "
        "and the builder's HEAD stamp. Ignoring a whole block hides the payload.")

    tomb = _load_tombstone("subdla_edge_systematic.tombstone.json")
    # (a) the volatile leaves really are ignored
    drifted = copy.deepcopy(tomb)
    drifted["retirement"]["retired_utc"] = "1999-01-01T00:00:00Z"
    drifted["metadata"]["generated_utc"] = "1999-01-01T00:00:00Z"
    drifted["metadata"]["code_commit"] = "0" * 40
    assert b.strip_volatile(drifted) == b.strip_volatile(tomb), (
        "re-stamping (new timestamps, new HEAD) must not read as DRIFT.")

    # (b) every payload leaf the referee falsified must read as DRIFT
    for mutate, what in (
        (lambda d: d["retirement"].__setitem__("recoverable_from_git",
                                               not d["retirement"]["recoverable_from_git"]),
         "retirement.recoverable_from_git"),
        (lambda d: d["retirement"].__setitem__(
            "recovery_note", "FABRICATED: totally unrecoverable, trust me."),
         "retirement.recovery_note"),
        (lambda d: d["retirement"]["defects"][0].__setitem__(
            "code", "STALE_FP_RESAMPLE_SEMANTICS"), "a defect code"),
        (lambda d: d["retirement"]["defects"][0].__setitem__(
            "detail", "x" * 200), "a defect detail"),
        (lambda d: d["metadata"].__setitem__("paper_facing", True),
         "metadata.paper_facing"),
        (lambda d: d["metadata"].__setitem__("routine", "not/the/builder.py"),
         "metadata.routine"),
    ):
        bad = copy.deepcopy(tomb)
        mutate(bad)
        assert b.strip_volatile(bad) != b.strip_volatile(tomb), (
            f"--check would report OK after falsifying {what}.")


def test_check_mode_runs_clean_and_detects_a_falsified_payload():
    """Nothing invoked ``--check``, so its blindness was invisible. This invokes it.

    It is HARD (not a skip) whenever a tree holding the four untracked retired artifacts is
    reachable; where it is not, ``--check`` physically cannot rebuild and the skip says so.
    Power comes from the second half: the committed record is falsified on disk, ``--check``
    must exit non-zero with DRIFT, and the bytes are restored."""
    if not _source_artifacts_present():
        pytest.skip(
            f"the four retired artifacts are not present under {RECORDING_WORKTREE}; "
            "--check cannot rebuild them (they were never committed). This skip is "
            "host-dependence, not a pass: the unit-level coverage of the same logic is "
            "test_check_mode_treats_only_timestamps_and_the_head_stamp_as_volatile.")
    cmd = [sys.executable, BUILDER, "--check", "--source-worktree", RECORDING_WORKTREE]
    r = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True)
    assert r.returncode == 0, (
        f"`build_tombstones.py --check` failed on a clean tree:\n{r.stdout}\n{r.stderr}")
    assert r.stdout.count("OK ") == len(TOMBSTONED) and "DRIFT" not in r.stdout, (
        f"--check did not report OK for all {len(TOMBSTONED)} tombstones:\n{r.stdout}")

    target = os.path.join(TOMB_DIR, "subdla_edge_systematic.tombstone.json")
    original = open(target, "rb").read()
    try:
        doc = json.loads(original)
        ret = doc["retirement"]
        ret["recoverable_from_git"] = not ret["recoverable_from_git"]
        ret["recovery_note"] = "FABRICATED: totally unrecoverable, trust me."
        for d in ret["defects"]:
            if d["code"] == "DIRTY_STAMP_NOT_REDERIVABLE":
                d["code"] = "STALE_FP_RESAMPLE_SEMANTICS"
        with open(target, "w") as fh:
            json.dump(doc, fh, indent=2)
            fh.write("\n")
        r2 = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True)
    finally:
        with open(target, "wb") as fh:
            fh.write(original)
    assert open(target, "rb").read() == original, "failed to restore the tombstone bytes"
    assert r2.returncode != 0 and "DRIFT" in r2.stdout, (
        "--check reported no DRIFT after the retirement payload was falsified "
        f"(rc={r2.returncode}):\n{r2.stdout}\n{r2.stderr}")


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
# REGISTRY CLOSURE -- the guards above are parametrised over TOMBSTONED, so a
# tombstone that is not IN TOMBSTONED is never examined by any of them.
# ===========================================================================
def _tombstones_on_disk():
    """Every ``*.tombstone.json`` physically in the tombstone directory."""
    return sorted(os.path.basename(p) for p in
                  glob.glob(os.path.join(TOMB_DIR, "*.tombstone.json")))


def test_the_tombstone_directory_holds_EXACTLY_the_registered_set():
    """Close the registry against the filesystem.

    Every other guard in this module is ``@parametrize("name", sorted(TOMBSTONED))``:
    it iterates a WHITELIST, not the directory. So the hard rules -- no float leaves, no
    ``paper_facing``, an allow-listed defect code, ``must_not_reuse_identity`` -- are
    enforced only on files that someone remembered to register. A ``*.tombstone.json``
    added to the directory and NOT added to ``TOMBSTONED`` is examined by nothing.

    That is this project's signature defect class, in the guard layer itself: the
    enumerated set (``TOMBSTONED``) and the real population (the directory) live on
    DIFFERENT SUPPORTS, and the guards run on the smaller one. Measured by an independent
    referee: a fifth, clean-stamped ``rogue.tombstone.json`` carrying a float science value
    ``retirement.smuggled_R0 = 0.9137``, ``defects[0].code = "NOT_A_REAL_CODE"`` and
    ``metadata.paper_facing = true`` was committed to the directory and the suite reported
    63 passed, 1 skipped, 0 failed.

    This test is the decisive missing check. With the two sets pinned equal, every
    parametrised guard above becomes TOTAL over what is actually on disk: an unregistered
    tombstone can no longer exist silently, and a registered-but-absent one is caught too.
    """
    on_disk = set(_tombstones_on_disk())
    registered = set(TOMBSTONED)
    unregistered = sorted(on_disk - registered)
    missing = sorted(registered - on_disk)
    assert not unregistered, (
        f"{unregistered} sit in {TOMB_DIR_REL} but are NOT in TOMBSTONED, so NONE of the "
        "hard-rule guards in this module inspect them -- a tombstone can smuggle a float "
        "science value, paper_facing=true or an unknown defect code past a green suite. "
        "Register them in TOMBSTONED (and give each a retired artifact path), or remove "
        "them.")
    assert not missing, (
        f"{missing} are registered in TOMBSTONED but absent from {TOMB_DIR_REL}. Their "
        "guards would be collected against a file that does not exist.")


@pytest.mark.parametrize("name", _tombstones_on_disk())
def test_every_tombstone_ON_DISK_obeys_the_hard_rules(name):
    """Defence in depth: apply the content rules to what is DISCOVERED, not to the registry.

    The closure test above is the primary guard. This one is deliberately redundant with
    it, and parametrised over the DIRECTORY rather than ``TOMBSTONED``, so that the hard
    rules still bite during the window in which someone adds a file and widens the registry
    in the same change without looking at its contents. Redundancy is the point: the two
    tests fail for different reasons.
    """
    with open(os.path.join(TOMB_DIR, name), "rb") as fh:
        tomb = json.loads(fh.read().decode("utf-8"))
    floats = [p for p, v in _walk(tomb)
              if isinstance(v, float) and not isinstance(v, bool)]
    assert floats == [], (
        f"{name} carries float leaves at {floats} -- a tombstone must hold no retired "
        "science value (SCHEMA.md hard rule 1).")
    assert tomb.get("metadata", {}).get("paper_facing") is False, (
        f"{name} does not declare metadata.paper_facing = false.")
    assert tomb.get("values_policy", {}).get("carries_science_values") is False, (
        f"{name} does not declare values_policy.carries_science_values = false.")
    codes = [d.get("code") for d in tomb.get("retirement", {}).get("defects", [])]
    assert codes, f"{name} declares no defect codes."
    bad = sorted(set(codes) - ALLOWED_DEFECT_CODES)
    assert not bad, (
        f"{name} declares defect codes {bad} outside ALLOWED_DEFECT_CODES "
        f"{sorted(ALLOWED_DEFECT_CODES)}.")


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
    else:
        # The CLEAN side had NO constraint, so `recoverable_from_git` and its
        # `recovery_note` were verified by nothing on exactly the record that asserts
        # recoverability -- an independent referee flipped both on the CLEAN record and
        # measured 57 passed / 7 skipped / 0 failed off-host. The DIRTY rule above cannot
        # cover it (it fires only on the dirty code), so the claim is checked against git
        # HERE, rather than trusted.
        assert ret["recoverable_from_git"] is True, (
            f"{name} carries no DIRTY_STAMP_NOT_REDERIVABLE defect, so its stamp is a "
            "clean commit and the producing source IS in git. recoverable_from_git=False "
            "understates what git holds; if the source is genuinely gone, say why with a "
            "defect code.")
        stamp = tomb["artifact"]["stamped_code_commit"]
        assert not stamp.endswith("-dirty") and len(stamp) == 40, (
            f"{name} claims recoverability from stamp {stamp!r}, which is not a clean "
            "40-char sha.")
        assert _git("cat-file", "-e", stamp).returncode == 0, (
            f"{name} claims recoverability from {stamp}, which is not in this repository.")
        # ...and the PRODUCING routine must actually be retrievable AT that commit. This is
        # the part a flipped boolean cannot fake. Note the producer is the script named in
        # `rederive_command_as_stamped` -- NOT `metadata.routine`, which is the tombstone
        # BUILDER and postdates the retired artifact entirely.
        cmd = tomb["artifact"]["rederive_command_as_stamped"]
        producers = [t for t in cmd.split() if t.endswith(".py")]
        assert len(producers) == 1, (
            f"{name}: could not identify a unique producing script in "
            f"rederive_command_as_stamped={cmd!r} (found {producers}). The recoverability "
            "claim is only checkable if the record names what to recover.")
        producer = producers[0]
        assert _git("cat-file", "-e", f"{stamp}:{producer}").returncode == 0, (
            f"{name} claims recoverable_from_git=True, but {producer!r} does not exist at "
            f"the stamped commit {stamp[:12]}. Either the claim is false or the record "
            "names the wrong producing routine.")

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
    forbids even before it reaches the index.

    2026-07-29 FIX (defect 3). This guard used to take BOTH the expected digest AND the
    path to compare against from the record it was auditing, so falsifying the record
    disarmed it: with ``artifact.sha256='f'*64`` and
    ``artifact.read_from_worktree='/nonexistent/tree'`` the loop ``continue``d on both
    candidates and the test still passed (measured: 4 passed). And its teeth depended on the
    literal path ``/home/mfho/desi_gpy_dla_detection`` existing, so on any other host it was
    SILENTLY vacuous. Now: the expected digest and the candidate trees are pinned HERE, the
    record must agree with those pins (a falsified record is RED, not a skip), and vacuity
    is reported explicitly instead of passing quietly."""
    tomb = _load_tombstone(name)
    art = tomb["artifact"]
    exp_sha, exp_bytes = EXPECTED_FINGERPRINT[name]

    # (1) the record must agree with the independently pinned fingerprint. This is what a
    #     falsified record trips: the comparison below no longer trusts the record.
    assert art["sha256"] == exp_sha and art["bytes"] == exp_bytes, (
        f"{name}.artifact fingerprint {art['sha256'][:12]}/{art['bytes']}B does not match "
        f"the fingerprint pinned in {os.path.basename(__file__)} "
        f"({exp_sha[:12]}/{exp_bytes}B). Either the record was edited (a tombstone is "
        "immutable once stamped) or the retired artifact was re-stamped -- neither is a "
        "silent update.")
    # (2) the record must not redirect the guard away from the recording worktree.
    assert art.get("read_from_worktree") == RECORDING_WORKTREE, (
        f"{name}.artifact.read_from_worktree={art.get('read_from_worktree')!r} != the "
        f"pinned recording worktree {RECORDING_WORKTREE!r}. Redirecting this field at a "
        "path that does not exist is exactly how this guard was defeated.")

    # (3) hash every candidate tree that is REACHABLE. Candidates are pinned, not read from
    #     the record. A candidate tree that exists but no longer holds the artifact is a
    #     legitimate deletion of untracked scratch; a candidate tree that is absent is
    #     host-dependence, and both are reported rather than swallowed.
    checked, absent_tree, deleted = [], [], []
    for tree in CANDIDATE_TREES:
        if not os.path.isdir(tree):
            absent_tree.append(tree)
            continue
        path = os.path.join(tree, TOMBSTONED[name])
        if not os.path.exists(path):
            deleted.append(path)
            continue
        with open(path, "rb") as fh:
            raw = fh.read()
        got = hashlib.sha256(raw).hexdigest()
        assert got == exp_sha and len(raw) == exp_bytes, (
            f"{path} differs from the fingerprint pinned for {name} "
            f"({got[:12]} != {exp_sha[:12]}, {len(raw)} vs {exp_bytes} bytes). The retired "
            "routine appears to have been re-run into the retired identity. Write the "
            "successor to a NEW path.")
        checked.append(path)

    if not checked:
        # EXPLICIT, not silent: parts (1) and (2) above still ran with full force; only the
        # byte-level comparison had nothing to look at.
        pytest.skip(
            f"byte-level fingerprint comparison VACUOUS for {name}: no reachable copy "
            f"(trees absent: {absent_tree or 'none'}; artifact deleted from: "
            f"{deleted or 'none'}). The record-vs-pin assertions still ran. This guard is "
            f"byte-level only on a host that holds {RECORDING_WORKTREE}.")


def test_fingerprint_guard_host_dependence_is_declared():
    """The guard above is byte-level only where a reachable tree still holds the untracked
    retired artifacts. Make that dependence a first-class, visible fact: on the recording
    host all four must be checkable, and off it the reason must be a missing tree rather
    than anything about the records."""
    reachable = [t for t in CANDIDATE_TREES if os.path.isdir(t)]
    assert reachable, (
        "neither the repo worktree nor the pinned recording worktree exists -- the "
        "fingerprint guard cannot be byte-level anywhere on this host.")
    if _source_artifacts_present():
        for name, rel in sorted(TOMBSTONED.items()):
            assert any(os.path.exists(os.path.join(t, rel)) for t in reachable), (
                f"{rel} is present under {RECORDING_WORKTREE} by the reachability check "
                f"but not found from {reachable} -- the guard for {name} would go vacuous "
                "on a host that can in fact check it.")
    else:
        pytest.skip(
            f"{RECORDING_WORKTREE} does not hold all four retired artifacts on this host, "
            "so the byte-level half of the fingerprint guard is inert here BY DESIGN. The "
            "record-vs-pin half (digest, byte count, read_from_worktree) is host-"
            "independent and always runs.")


def test_recording_host_still_holds_the_retired_artifacts():
    """The one omission the guard above CANNOT see: deletion on the recording host itself.

    ``test_fingerprint_guard_host_dependence_is_declared`` branches on
    ``_source_artifacts_present()``, so removing the four untracked artifacts makes it take
    the ``pytest.skip`` arm -- the same arm a genuinely different host takes. Deleting them
    therefore DISARMS the byte-level fingerprint comparison while CI stays green. Measured
    2026-08-05: deleting the four took this selection from 65 passed / 0 skipped to
    59 passed / 6 skipped, every one of the six a documented skip rather than a failure.

    The tombstones already record which tree is supposed to hold them
    (``artifact.read_from_worktree``, pinned as ``RECORDING_WORKTREE``), so the two cases
    ARE distinguishable and the ambiguity is not inherent:

      * that directory does not exist  -> a different host. Skipping is correct.
      * that directory exists but the artifacts are gone -> somebody deleted them HERE.
        That is a defect, not host dependence, and it fails.

    If a retirement ever legitimately removes the local copies, this is the deliberate
    step: re-stamp the records so ``read_from_worktree`` no longer points at a tree that is
    expected to hold them. Silent deletion is what this refuses.
    """
    if not os.path.isdir(RECORDING_WORKTREE):
        pytest.skip(
            f"{RECORDING_WORKTREE} does not exist on this host, so the retired artifacts "
            "were never here to delete. The byte-level guard is inert by design.")

    missing = sorted(rel for rel in TOMBSTONED.values()
                     if not os.path.exists(os.path.join(RECORDING_WORKTREE, rel)))
    assert not missing, (
        f"the recording worktree {RECORDING_WORKTREE} exists but no longer holds "
        f"{len(missing)} of the {len(TOMBSTONED)} retired artifacts: {missing}. These are "
        "untracked ON PURPOSE and load-bearing -- they are the only physical copies the "
        "tombstone fingerprint guard can compare its pinned sha256/bytes against, and "
        "they are listed with this reasoning in .gitignore. Deleting them disarms that "
        "guard silently (it degrades to a skip, not a failure). Restore them, or re-stamp "
        "the tombstones so read_from_worktree no longer claims this tree holds them.")


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
    file, and carries no value that could be quoted or plotted.

    LIMIT, stated: the two-sided comparison is between two hex strings in ONE committed
    file, so by itself it records the builder's stamp-time check rather than re-deriving
    it -- a fabricated pair would satisfy it. The dN/dX cum(19.5) endpoint is pinned to the
    DERIVED commitment (which IS recomputed from committed data by
    ``test_tripwire_derived_commitment_recomputes_from_committed_data``); that is where the
    endpoint block gets its teeth, and it is one endpoint of four."""
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

    # The ONE endpoint with arithmetic teeth: /measurement/19.5/dndx/integrated/MAP is also
    # the derived commitment's pointer, and the derived digest is recomputed from committed
    # data. Identical by construction at stamp time, so requiring equality here makes a
    # fabricated endpoint block detectable at that pointer.
    cum195 = "/measurement/19.5/dndx/integrated/MAP"
    by_ptr = {c["pointer"]: c for c in cs}
    dv = {d["equals_pointer_in_retired_artifact"]: d for d in tw["derived_commitments"]}
    assert cum195 in by_ptr and cum195 in dv, (
        f"{cum195} must appear in BOTH the endpoint and derived commitment namespaces; "
        "without that overlap the endpoint block has no re-derivable anchor.")
    assert by_ptr[cum195]["sha256_of_repr"] == dv[cum195]["sha256_of_repr"], (
        f"endpoint and derived commitments disagree at {cum195}. They are digests of the "
        "same float by construction, so one of the two blocks has been edited or "
        "fabricated.")


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
