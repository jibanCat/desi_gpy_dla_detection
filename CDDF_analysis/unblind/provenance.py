"""provenance.py -- the provenance guard for the DLA/sub-DLA unblinding notebooks.

WHY THIS EXISTS
---------------
A stamped headline artifact is only useful if a third party can re-derive it from
COMMITTED code.  This project has been burned twice by stamps that looked fine but
were worthless:

  * a headline that lived only as literals inside a plot script (no routine, no stamp);
  * a stamped JSON whose generating routine was never committed and was later deleted.

A newer failure showed even a *valid, ancestor, non-dirty* commit can be worthless:
`lls_recovery_figures.json` stamped `78c01f6`, but the routine that produced it
(`.../lls_recovery_figures.py`) was first committed later in `a907127`.  The stamp
names a real commit that DOES NOT CONTAIN the routine -> ORPHANED.

So this guard refuses to proceed unless the stamp is RE_DERIVABLE:

  code_commit is present, is not "unknown", does not end with "-dirty",
  names a commit that EXISTS in this repo,
  that commit CONTAINS the generating routine at a readable blob,
  and that commit is an ancestor of (or equal to) HEAD.

It classifies every artifact into one of these states and never reads a single
science value -- it only touches ``metadata.code_commit`` and the routine path.

    NOT_STAMPED    code_commit missing or "unknown"
    DIRTY          code_commit ends with "-dirty" (routine modified/untracked at gen time)
    MOVABLE_REF    code_commit is not a hex object name -- it is a tag/branch/HEAD-ish
                   REF, which can be moved or deleted, so it pins nothing
    ABBREVIATED_SHA  code_commit is a hex prefix (<40 chars); only reported as a
                   FAILURE under ``require_full_sha=True`` (otherwise a loud warning)
    COMMIT_NOT_FOUND   stamped commit does not exist in this repo
    NO_ROUTINE     cannot identify the generating routine (no metadata.routine /
                   metadata.rederive and no explicit routine_path)
    ORPHANED       commit exists but does NOT contain the routine
    NOT_ANCESTOR   commit exists + contains routine, but is NOT in HEAD's history
    RE_DERIVABLE   commit exists, contains routine, ancestor-or-equal to HEAD  (PASS)

Only RE_DERIVABLE passes by default.  RE_DERIVABLE may still emit *loud warnings*
when HEAD has moved past the stamp, or when the routine's blob at the stamp commit
differs from its blob at HEAD (the current code would re-derive different bytes).

STAMP BLOCK SCHEMA (two shapes, one reader)
-------------------------------------------
Artifacts in this project stamp their provenance in one of two places:

    {"metadata":   {"code_commit": ..., "routine": ..., "rederive": ...}, ...}
    {"provenance": {"code_commit": ..., "routine": ..., "rederive": ...}, ...}

...and a few legacy artifacts stamp at the bare top level.  The catalog-HBI family
uses ``metadata``; the feed-forward family (``calccddf_*``, ``ff_fp_*``) uses
``provenance``.  Until 2026-07-28 the reader knew only about ``metadata`` and the
bare top level, so EVERY feed-forward artifact -- half of Paper 1 -- silently
returned NOT_STAMPED despite carrying a perfectly good 40-char stamp.

The fix is in the READER, not the writers, and deliberately so:

  * the feed-forward artifacts are the output of a full-scale 2026-07-11 campaign
    (1150 / 1149 / 1127 files).  Rewriting a committed artifact's bytes to move a
    key would either (a) invalidate its own stamp, or (b) require re-running the
    campaign to re-stamp honestly.  Neither is acceptable to fix a READER bug.
  * the reader is a single choke point; the writers are many and live in two
    worktrees with no ancestry relation.
  * ``metadata`` is nonetheless declared CANONICAL (:data:`CANONICAL_STAMP_KEY`)
    because ``loader.py``/``schema.py`` already read ``metadata`` for the headline
    schema.  NEW writers should call :func:`stamp_block`; ``provenance`` is a
    permanently-supported ALIAS so no existing artifact ever silently breaks.

If two blocks both carry a ``code_commit`` and they DISAGREE, that is an ambiguous
artifact and :func:`load_stamp_block` raises rather than silently picking one.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------
NOT_STAMPED = "NOT_STAMPED"
DIRTY = "DIRTY"
MOVABLE_REF = "MOVABLE_REF"
ABBREVIATED_SHA = "ABBREVIATED_SHA"
COMMIT_NOT_FOUND = "COMMIT_NOT_FOUND"
NO_ROUTINE = "NO_ROUTINE"
ORPHANED = "ORPHANED"
NOT_ANCESTOR = "NOT_ANCESTOR"
RE_DERIVABLE = "RE_DERIVABLE"

ALL_STATUSES = (
    NOT_STAMPED, DIRTY, MOVABLE_REF, ABBREVIATED_SHA, COMMIT_NOT_FOUND,
    NO_ROUTINE, ORPHANED, NOT_ANCESTOR, RE_DERIVABLE,
)

# The only status that lets an unblinding notebook proceed.
PASS_STATUSES = (RE_DERIVABLE,)

# ---------------------------------------------------------------------------
# Stamp-block schema
# ---------------------------------------------------------------------------
# Canonical key for NEW writers.  ``provenance`` is a permanently-supported alias
# (see the module docstring); the bare top level is legacy and read last.
CANONICAL_STAMP_KEY = "metadata"
STAMP_BLOCK_KEYS = ("metadata", "provenance")
# the fields classify() may read -- used to decide whether a bare top-level dict
# is a stamp block at all.
STAMP_FIELDS = ("code_commit", "routine", "rederive")


class ProvenanceError(RuntimeError):
    """Raised when an artifact's provenance is not RE_DERIVABLE (or not allowed)."""


# ---------------------------------------------------------------------------
# git plumbing
# ---------------------------------------------------------------------------
def repo_root(start: Optional[str] = None) -> str:
    """Absolute path to the git repo top-level containing this module (or ``start``)."""
    here = start or os.path.dirname(os.path.abspath(__file__))
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=here, capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise ProvenanceError(f"not inside a git repo (cwd={here}): {out.stderr.strip()}")
    return out.stdout.strip()


def _git(args, repo: str):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)


def _commit_exists(sha: str, repo: str) -> bool:
    return _git(["cat-file", "-e", f"{sha}^{{commit}}"], repo).returncode == 0


def _blob_exists(sha: str, path: str, repo: str) -> bool:
    return _git(["cat-file", "-e", f"{sha}:{path}"], repo).returncode == 0


def _blob_hash(sha: str, path: str, repo: str) -> Optional[str]:
    out = _git(["rev-parse", f"{sha}:{path}"], repo)
    return out.stdout.strip() if out.returncode == 0 else None


def _full_sha(sha: str, repo: str) -> Optional[str]:
    out = _git(["rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"], repo)
    return out.stdout.strip() if out.returncode == 0 else None


def _is_ancestor(sha: str, ref: str, repo: str) -> bool:
    return _git(["merge-base", "--is-ancestor", sha, ref], repo).returncode == 0


# ---------------------------------------------------------------------------
# stamp shape: full SHA vs abbreviated SHA vs movable ref
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS
# ---------------
# `git cat-file -e v0.1.0^{commit}` succeeds, so a stamp of "v0.1.0" -- or
# "desi_y3", or literally "HEAD" -- used to sail through to RE_DERIVABLE.  Every
# one of those is a MOVABLE POINTER: `git tag -f`, one more commit on the branch,
# or a different checkout, and the stamp now names DIFFERENT code while the
# artifact bytes are unchanged.  A stamp that can silently change meaning pins
# nothing, which is the exact failure mode this whole module exists to prevent.
# Only an object name (hex SHA) is immutable.
_HEX_FULL = re.compile(r"^[0-9a-f]{40}$")
_HEX_ABBREV = re.compile(r"^[0-9a-f]{4,39}$")

STAMP_FULL_SHA = "full_sha"
STAMP_ABBREV_SHA = "abbrev_sha"
STAMP_MOVABLE_REF = "movable_ref"


def stamp_kind(code_commit: str) -> str:
    """Classify a stamp STRING's shape (no git access).

    ``full_sha``    40-char lowercase hex -- the only immutable, unambiguous form.
    ``abbrev_sha``  4..39-char lowercase hex -- immutable in meaning but may become
                    AMBIGUOUS as the object database grows.
    ``movable_ref`` anything else -- a tag / branch / ``HEAD`` / ``HEAD~1`` /
                    ``v0.1.0``-style name, i.e. a movable pointer, not an object.
    """
    s = str(code_commit).strip().lower()
    if _HEX_FULL.match(s):
        return STAMP_FULL_SHA
    if _HEX_ABBREV.match(s):
        return STAMP_ABBREV_SHA
    return STAMP_MOVABLE_REF


# ---------------------------------------------------------------------------
# routine resolution
# ---------------------------------------------------------------------------
_PY_TOKEN = re.compile(r"(\S+\.py)\b")


_TEMPLATE = re.compile(r"\$\{|\$\(|\{\{")


def _resolve_bare(name: str, commit: str, repo: str) -> Optional[str]:
    """Map a bare ``foo.py`` to its unique repo-relative path at ``commit``.

    Rederive strings are shell commands, so a routine may appear without its
    directory once the command has already ``cd``-ed.  Returns None when the
    basename does not resolve to exactly one tracked path (missing, or ambiguous).
    """
    out = _git(["ls-tree", "-r", "--name-only", commit], repo)
    if out.returncode != 0:
        return None
    hits = [p for p in out.stdout.splitlines() if p.rsplit("/", 1)[-1] == name]
    return hits[0] if len(hits) == 1 else None


def resolve_routines(
    metadata: dict, routine_path: Optional[str] = None, commit: Optional[str] = None,
    repo: Optional[str] = None,
) -> tuple:
    """Identify EVERY routine an artifact's rederive depends on.

    Returns ``(paths, notes)``.

    Resolving only the *first* ``*.py`` token is unsafe: a multi-step rederive
    ("run the three legs, then aggregate") names its real artifact-builder LAST,
    so a missing builder rides in behind a leg script that does exist.  That is
    precisely the ORPHANED class this module was written to catch, and it was
    observed in the wild (``crossmock_transfer_loa0.json`` -> ``build_artifact_loa0.py``,
    which exists in no commit).  Every token is therefore checked, and the
    artifact is only RE_DERIVABLE if all of them resolve.

    Resolution order:
      1. explicit ``routine_path`` argument, or ``metadata['routine']``
         (either may be a str or a list) -- authoritative, no parsing.
      2. otherwise, all ``*.py`` tokens in ``metadata['rederive']``, de-duplicated
         in order.  Bare basenames are resolved against ``commit``.  Tokens still
         carrying an unexpanded shell template (``${leg}``) are reported as such.
    """
    if routine_path:
        vals = routine_path if isinstance(routine_path, (list, tuple)) else [routine_path]
        return [str(v) for v in vals], []
    routine = metadata.get("routine")
    if routine:
        vals = routine if isinstance(routine, (list, tuple)) else [routine]
        return [str(v) for v in vals], []

    rederive = metadata.get("rederive")
    if not rederive:
        return [], []

    paths, notes, seen = [], [], set()
    for tok in _PY_TOKEN.findall(str(rederive)):
        if tok in seen:
            continue
        seen.add(tok)
        if _TEMPLATE.search(tok):
            notes.append(f"rederive contains an UNEXPANDED template token {tok!r}")
            paths.append(tok)          # keep it: it must not resolve, and must fail loudly
            continue
        if "/" not in tok and commit and repo:
            resolved = _resolve_bare(tok, commit, repo)
            if resolved:
                notes.append(f"resolved bare token {tok!r} -> {resolved}")
                tok = resolved
            else:
                notes.append(f"bare token {tok!r} does not resolve to a unique tracked path")
        paths.append(tok)
    return paths, notes


def resolve_routine(metadata: dict, routine_path: Optional[str] = None) -> Optional[str]:
    """Back-compat single-routine resolver.  Prefer :func:`resolve_routines`."""
    paths, _ = resolve_routines(metadata, routine_path)
    return paths[0] if paths else None


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
@dataclass
class ProvenanceResult:
    status: str
    code_commit: Optional[str]           # raw stamp string
    base_commit: Optional[str] = None    # stamp with any "-dirty" stripped
    routine: Optional[str] = None        # first routine (back-compat)
    routines: list = field(default_factory=list)   # EVERY routine the rederive names
    commit_exists: bool = False
    contains_routine: bool = False
    is_ancestor: bool = False
    is_equal_head: bool = False
    head_moved: bool = False             # ancestor but not equal (current code differs)
    routine_drift: Optional[bool] = None  # routine blob @stamp != @HEAD
    head: Optional[str] = None
    stamp_kind: Optional[str] = None      # full_sha / abbrev_sha / movable_ref
    schema_key: Optional[str] = None      # which block the stamp was read from
    messages: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == RE_DERIVABLE

    @property
    def abbreviated_sha(self) -> bool:
        return self.stamp_kind == STAMP_ABBREV_SHA

    def reason(self) -> str:
        """The single most informative message (first fatal, else first warning)."""
        if not self.messages:
            return ""
        return self.messages[0]

    def summary(self) -> str:
        """One-line human summary.  Contains NO science values -- only provenance."""
        parts = [f"status={self.status}", f"code_commit={self.code_commit!r}"]
        if self.stamp_kind and self.stamp_kind != STAMP_FULL_SHA:
            parts.append(f"stamp_kind={self.stamp_kind}")
        if self.schema_key:
            parts.append(f"schema_key={self.schema_key}")
        if self.routine:
            parts.append(f"routine={self.routine}")
        if self.head_moved:
            parts.append("HEAD-moved-past-stamp")
        if self.routine_drift:
            parts.append("ROUTINE-DRIFT")
        return "  ".join(parts)


def classify(
    metadata: dict,
    routine_path: Optional[str] = None,
    repo: Optional[str] = None,
    head: str = "HEAD",
    require_full_sha: bool = False,
    schema_key: Optional[str] = None,
) -> ProvenanceResult:
    """Classify an artifact's provenance from its ``metadata`` dict alone.

    Pure w.r.t. science: only reads ``code_commit`` + the routine path.  Runs git
    plumbing against ``repo`` (default: the repo containing this module).

    ``require_full_sha`` promotes an abbreviated hex stamp from a loud WARNING to
    the hard failure :data:`ABBREVIATED_SHA`.  A MOVABLE ref (tag / branch /
    ``HEAD``) is ALWAYS a hard failure regardless of this flag: an abbreviated SHA
    still names one immutable commit, a tag does not.
    """
    repo = repo or repo_root()
    head_sha = _full_sha(head, repo)
    res = ProvenanceResult(status="", code_commit=metadata.get("code_commit"), head=head_sha,
                           schema_key=schema_key)

    cc = res.code_commit
    if not cc or str(cc).strip().lower() == "unknown":
        res.status = NOT_STAMPED
        res.messages.append("code_commit is missing or 'unknown' -> not re-derivable.")
        return res

    cc = str(cc).strip()
    if cc.endswith("-dirty"):
        res.status = DIRTY
        res.base_commit = cc[: -len("-dirty")]
        res.stamp_kind = stamp_kind(res.base_commit)
        res.messages.append(
            "code_commit ends with '-dirty': the routine was modified or the tree was "
            "unclean at generation time -> not re-derivable from committed code."
        )
        return res

    res.base_commit = cc
    res.stamp_kind = stamp_kind(cc)

    # --- STAMP SHAPE GATE (before any git lookup that would launder a ref) -----
    if res.stamp_kind == STAMP_MOVABLE_REF:
        res.status = MOVABLE_REF
        resolved = _full_sha(cc, repo)
        res.messages.append(
            f"code_commit {cc!r} is NOT a hex object name -- it is a MOVABLE ref "
            f"(tag / branch / HEAD-expression"
            + (f", currently {resolved[:12]}" if resolved else ", currently unresolvable")
            + "). `git tag -f`, one more commit on the branch, or a different checkout "
            "silently changes what this stamp means while the artifact bytes stay the "
            "same, so it pins NOTHING. Re-stamp with the full 40-char commit SHA "
            "(`git rev-parse HEAD`), never a tag or branch name."
        )
        return res
    if res.stamp_kind == STAMP_ABBREV_SHA:
        msg = (
            f"code_commit {cc!r} is an ABBREVIATED {len(cc)}-char SHA. It names one "
            "immutable commit today, but abbreviations can become ambiguous as the "
            "object database grows. Artifacts must stamp the full 40-char SHA."
        )
        if require_full_sha:
            res.status = ABBREVIATED_SHA
            res.messages.append(msg)
            return res
        res.messages.append("WARNING: " + msg)

    res.commit_exists = _commit_exists(cc, repo)
    if not res.commit_exists:
        res.status = COMMIT_NOT_FOUND
        res.messages.append(f"stamped commit {cc} does not exist in this repo.")
        return res

    routines, notes = resolve_routines(metadata, routine_path, commit=cc, repo=repo)
    res.messages.extend(notes)
    res.routines = routines
    res.routine = routines[0] if routines else None
    if not routines:
        res.status = NO_ROUTINE
        res.messages.append(
            "cannot identify the generating routine: no metadata['routine'], no parseable "
            "'*.py' in metadata['rederive'], and no routine_path supplied."
        )
        return res

    # EVERY routine the rederive names must be present at the stamp. A multi-step
    # rederive's final aggregator is the one most likely to be missing, and it is
    # the last token, not the first.
    missing = [r for r in routines if not _blob_exists(cc, r, repo)]
    res.contains_routine = not missing
    if missing:
        res.status = ORPHANED
        res.messages.append(
            f"commit {cc[:12]} exists but does NOT contain {len(missing)} of the "
            f"{len(routines)} routine(s) its rederive names: {missing!r} -> not re-derivable."
        )
        return res

    res.is_ancestor = _is_ancestor(cc, head, repo)
    stamp_full = _full_sha(cc, repo)
    res.is_equal_head = bool(stamp_full and head_sha and stamp_full == head_sha)
    if not res.is_ancestor:
        res.status = NOT_ANCESTOR
        res.messages.append(
            f"commit {cc[:12]} is not an ancestor of {head} (diverged/off-history) -> "
            "the current checkout cannot re-derive it."
        )
        return res

    # RE_DERIVABLE -- but surface loud, non-fatal warnings.
    res.status = RE_DERIVABLE
    res.head_moved = res.is_ancestor and not res.is_equal_head
    if res.head_moved:
        res.messages.append(
            f"WARNING: HEAD has moved PAST the stamp ({cc[:12]} != {head_sha[:12]}); the "
            "current checkout differs from the code that generated this artifact."
        )
    # ROUTINE-DRIFT: the stamp names a commit that contains the routine, but the
    # routine was edited afterwards -- so `git cat-file -e` passes while re-running
    # at HEAD executes a different program.  Check every routine, not just the first.
    drifted, checked = [], False
    for r in routines:
        blob_stamp = _blob_hash(cc, r, repo)
        blob_head = _blob_hash(head, r, repo)
        if blob_stamp is None or blob_head is None:
            continue
        checked = True
        if blob_stamp != blob_head:
            drifted.append((r, blob_stamp, blob_head))
    if checked:
        res.routine_drift = bool(drifted)
        for r, bs, bh in drifted:
            res.messages.append(
                f"WARNING: routine {r} changed between the stamp commit and HEAD "
                f"({bs[:10]} -> {bh[:10]}); re-running now would use different code."
            )
    return res


# ---------------------------------------------------------------------------
# file-based entry point
# ---------------------------------------------------------------------------
def stamp_block(code_commit: str, routine, rederive: Optional[str] = None, **extra) -> dict:
    """Build a CANONICAL stamp block for a NEW writer: ``{"metadata": {...}}``.

    Use as ``json.dump({**payload, **stamp_block(sha, "path/to/routine.py", cmd)}, f)``.
    Refuses anything that is not a full 40-char SHA at WRITE time -- the cheapest
    place to catch a tag-shaped or abbreviated stamp is before it is written.
    """
    kind = stamp_kind(code_commit)
    if kind != STAMP_FULL_SHA:
        raise ProvenanceError(
            f"refusing to stamp code_commit={code_commit!r} ({kind}): artifacts must "
            "stamp the FULL 40-char commit SHA (`git rev-parse HEAD`), never a tag, a "
            "branch name, or an abbreviation."
        )
    block = {"code_commit": str(code_commit).strip().lower(), "routine": routine}
    if rederive is not None:
        block["rederive"] = rederive
    block.update(extra)
    return {CANONICAL_STAMP_KEY: block}


def load_stamp_block(artifact) -> tuple:
    """Return ``(block, schema_key)`` for an artifact path or already-parsed dict.

    Accepts BOTH committed schemas -- ``metadata`` (canonical) and ``provenance``
    (alias) -- plus the legacy bare top level, so no existing artifact silently
    breaks.  Precedence: the first key in :data:`STAMP_BLOCK_KEYS` that actually
    carries a ``code_commit``; then any block at all; then the bare top level.

    Raises :class:`ProvenanceError` if two blocks carry DIFFERENT ``code_commit``
    values -- an ambiguous artifact must never be silently resolved in favour of
    one of them.
    """
    if isinstance(artifact, dict):
        d, label = artifact, "<dict>"
    else:
        with open(artifact) as f:
            d = json.load(f)
        label = artifact
    if not isinstance(d, dict):
        raise ProvenanceError(f"{label}: not a JSON object (got {type(d).__name__})")

    present = [(k, d[k]) for k in STAMP_BLOCK_KEYS if isinstance(d.get(k), dict)]
    stamped = [(k, b) for k, b in present if b.get("code_commit")]

    shas = {str(b["code_commit"]).strip() for _, b in stamped}
    if len(shas) > 1:
        raise ProvenanceError(
            f"{label}: AMBIGUOUS provenance -- blocks "
            f"{[k for k, _ in stamped]} carry different code_commit values {sorted(shas)}. "
            "Refusing to guess which one generated this artifact; keep exactly one stamp "
            f"block (canonical: {CANONICAL_STAMP_KEY!r})."
        )
    if stamped:
        return stamped[0][1], stamped[0][0]
    if present:
        return present[0][1], present[0][0]
    if any(k in d for k in STAMP_FIELDS):
        return d, "<top-level>"
    # nothing resembling a stamp anywhere: hand back the top level so classify()
    # reports NOT_STAMPED (the honest answer) rather than raising.
    return d, "<top-level>"


def _load_metadata(artifact_path: str) -> dict:
    """Back-compat shim.  Prefer :func:`load_stamp_block`."""
    return load_stamp_block(artifact_path)[0]


def check_artifact(
    artifact_path: str,
    routine_path: Optional[str] = None,
    allowed=PASS_STATUSES,
    repo: Optional[str] = None,
    head: str = "HEAD",
    verbose: bool = True,
    require_full_sha: bool = False,
) -> ProvenanceResult:
    """Guard entry point: load an artifact's stamp block, classify it, RAISE unless allowed.

    Reads only the stamp block -- never a science value.  On the happy path it still
    prints any loud warnings (abbreviated SHA / HEAD moved / routine drift).
    """
    metadata, key = load_stamp_block(artifact_path)
    res = classify(metadata, routine_path=routine_path, repo=repo, head=head,
                   require_full_sha=require_full_sha, schema_key=key)
    if verbose:
        for m in res.messages:
            print(f"[provenance] {os.path.basename(artifact_path)}: {m}")
    if res.status not in allowed:
        raise ProvenanceError(
            f"{os.path.basename(artifact_path)}: provenance status {res.status} not in "
            f"{tuple(allowed)}. {res.messages[0] if res.messages else ''}"
        )
    return res


# ---------------------------------------------------------------------------
# forward-response kernel guard (fail-closed) for HEADLINE runs
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS
# ---------------
# Track-C established that the GP-POSTERIOR ("kappa") response kernel is the WRONG
# OBJECT for the catalog-HBI CDDF: it over-recovers the steep high-N tail (DLA-tier
# R0>=20.3 ~= 1.16 posterior vs ~= 1.04 forward; sub-DLA band R0 0.883/0.899 posterior
# vs 0.849/0.822 forward).  The measured FORWARD RESPONSE p(x_hat | N_true, SNR, z) is
# the right object.  But ``HBIConfig.resp_kind`` DEFAULTS to ``"kappa"`` and the forward
# path is only engaged when ``_set_forward_cfg`` (track_c_perz_band) -- or an explicit
# override -- sets ``resp_kind="forward"`` and attaches ``kernel_forward_model``.  A
# headline routine that forgets this silently emits a POSTERIOR-KERNEL number that looks
# identical in schema.  HANDOFF.md sec 5 row 2 asked for exactly this guard and it was
# never added.
#
# This guard is for HEADLINE runs ONLY.  The legitimately-posterior DIAGNOSTIC paths
# (``baseline_recovery`` run as a kappa diagnostic, e.g. the default
# ``subdla_loa0_validation.py`` reduction) simply never call it -- absence of the call is
# how a path declares itself a diagnostic.  Call it right before the estimator on any run
# that will STAMP a "forward"/headline artifact.
def assert_forward_kernel(cfg_or_resp_kind, *, context="headline run",
                          require_kernel_model=False):
    """FAIL-CLOSED: raise ``ProvenanceError`` unless the response kernel is FORWARD.

    ``cfg_or_resp_kind`` may be an ``HBIConfig``-like object (its ``.resp_kind`` /
    ``.kernel_forward_model`` attributes are read) or a bare ``resp_kind`` string.  A
    missing / ``None`` ``resp_kind`` is treated as the ``HBIConfig`` default ``"kappa"``
    and therefore REJECTED -- fail-closed, never fail-open.

    Returns the validated ``resp_kind`` (``"forward"``) on success so callers can inline
    it.  This never warns; it raises loudly.
    """
    has_attr = hasattr(cfg_or_resp_kind, "resp_kind")
    resp_kind = getattr(cfg_or_resp_kind, "resp_kind", cfg_or_resp_kind) if has_attr \
        else cfg_or_resp_kind
    if resp_kind is None:
        resp_kind = "kappa"          # HBIConfig default -> posterior -> reject
    resp_kind = str(resp_kind)
    if resp_kind != "forward":
        raise ProvenanceError(
            f"[forward-kernel guard] {context}: resp_kind={resp_kind!r}, but a HEADLINE run "
            f"MUST use the forward-response kernel (resp_kind='forward'). The GP-posterior "
            f"'kappa' kernel is the WRONG OBJECT (Track-C: over-recovers high-N; DLA-tier "
            f"R0>=20.3 ~1.16 posterior vs ~1.04 forward). Call _set_forward_cfg(cfg, args) / "
            f"set cfg.resp_kind='forward' before the estimator, or run this as a labelled "
            f"diagnostic (do not stamp it as a headline/forward artifact)."
        )
    if require_kernel_model:
        kfm = getattr(cfg_or_resp_kind, "kernel_forward_model", None) if has_attr else None
        if not kfm:
            raise ProvenanceError(
                f"[forward-kernel guard] {context}: resp_kind='forward' but "
                f"cfg.kernel_forward_model is unset -- the forward dispatch (_build_A_ib) "
                f"would raise. Set cfg.kernel_forward_model to a ForwardResponseModel NPZ."
            )
    return resp_kind

