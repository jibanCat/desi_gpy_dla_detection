"""audit.py -- repo-wide provenance audit over every COMMITTED JSON artifact.

Paper 1 lives in two worktrees with NO ancestry relation:

    /home/mfho/desi_gpy_dla_detection   (catalog-HBI arm + this unblind package)
    /home/mfho/hbi_mcmc_wt              (feed-forward arm: calccddf_* / ff_fp_*)

No single branch contains the paper, so "our artifacts pass check_artifact" is a
statement about a UNION of worktrees and can only be checked by walking both.  This
module walks `git ls-files '*.json'` in each, runs :func:`provenance.classify` on the
committed blob (not the working tree -- an uncommitted edit must not launder a
status), and emits one row per artifact:

    path -> status -> stamp kind -> schema key -> reason

Run it::

    python -m CDDF_analysis.unblind.audit \\
        --worktree /home/mfho/desi_gpy_dla_detection \\
        --worktree /home/mfho/hbi_mcmc_wt

Options: ``--strict-sha`` (an abbreviated stamp becomes a FAILURE), ``--privacy``
(also run the JSON privacy scanner on each artifact), ``--include-fixtures``
(``tests/fixtures/**`` is excluded by default -- those are frozen test inputs, not
science artifacts), ``--format {table,markdown,json}``.

Exit status is 0 when every audited artifact is RE_DERIVABLE, else 1 -- so this is
usable as the paper's provenance-appendix generator AND as a gate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

from . import privacy as _privacy
from . import provenance as prov

DEFAULT_WORKTREES = (
    "/home/mfho/desi_gpy_dla_detection",
    "/home/mfho/hbi_mcmc_wt",
)

# Committed JSONs that are inputs to tests, not science artifacts.
DEFAULT_EXCLUDE_PREFIXES = ("tests/fixtures/",)


@dataclass
class AuditRow:
    worktree: str
    path: str                       # repo-relative
    status: str
    stamp_kind: Optional[str] = None
    schema_key: Optional[str] = None
    code_commit: Optional[str] = None
    routines: list = field(default_factory=list)
    head_moved: bool = False
    routine_drift: Optional[bool] = None
    privacy_hits: list = field(default_factory=list)
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == prov.RE_DERIVABLE

    def as_dict(self) -> dict:
        return {
            "worktree": self.worktree, "path": self.path, "status": self.status,
            "stamp_kind": self.stamp_kind, "schema_key": self.schema_key,
            "code_commit": self.code_commit, "routines": list(self.routines),
            "head_moved": self.head_moved, "routine_drift": self.routine_drift,
            "privacy_hits": list(self.privacy_hits), "reason": self.reason,
        }


def _git(args, repo):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)


def committed_json_paths(worktree: str, include_fixtures: bool = False,
                         exclude_prefixes=DEFAULT_EXCLUDE_PREFIXES) -> list:
    out = _git(["ls-files", "*.json"], worktree)
    if out.returncode != 0:
        raise prov.ProvenanceError(f"{worktree}: not a git worktree ({out.stderr.strip()})")
    paths = sorted(p for p in out.stdout.split() if p)
    if not include_fixtures:
        paths = [p for p in paths if not any(p.startswith(x) for x in exclude_prefixes)]
    return paths


def _committed_doc(worktree: str, rel: str, ref: str = "HEAD"):
    """Parse the artifact AS COMMITTED at ``ref`` (never the working tree)."""
    out = _git(["show", f"{ref}:{rel}"], worktree)
    if out.returncode != 0:
        return None
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return None


def audit_worktree(worktree: str, require_full_sha: bool = False,
                   include_fixtures: bool = False, check_privacy: bool = False,
                   head: str = "HEAD") -> list:
    rows = []
    for rel in committed_json_paths(worktree, include_fixtures=include_fixtures):
        doc = _committed_doc(worktree, rel, head)
        if doc is None:
            rows.append(AuditRow(worktree, rel, "UNREADABLE",
                                 reason="committed blob is not parseable JSON"))
            continue
        try:
            block, key = prov.load_stamp_block(doc)
        except prov.ProvenanceError as exc:
            rows.append(AuditRow(worktree, rel, "AMBIGUOUS_STAMP", reason=str(exc)))
            continue
        res = prov.classify(block, repo=worktree, head=head,
                            require_full_sha=require_full_sha, schema_key=key)
        row = AuditRow(
            worktree=worktree, path=rel, status=res.status, stamp_kind=res.stamp_kind,
            schema_key=res.schema_key, code_commit=res.code_commit,
            routines=list(res.routines), head_moved=res.head_moved,
            routine_drift=res.routine_drift, reason=res.reason(),
        )
        if check_privacy:
            row.privacy_hits = [
                f"{h.kind}:{h.where}:{h.token}"
                for h in _privacy.scan_json_artifact(doc)
                if h.kind in _privacy.HARD_HIT_KINDS
            ]
        rows.append(row)
    return rows


def audit(worktrees=DEFAULT_WORKTREES, **kw) -> list:
    rows = []
    for wt in worktrees:
        rows.extend(audit_worktree(wt, **kw))
    return rows


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _short(s, n):
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def render_table(rows, markdown: bool = False, reason_width: int = 64) -> str:
    hdr = ("ARTIFACT", "STATUS", "STAMP", "SCHEMA", "REASON / NOTE")
    body = []
    for r in rows:
        note = r.reason
        if r.ok and not note:
            flags = []
            if r.head_moved:
                flags.append("HEAD moved past stamp")
            if r.routine_drift:
                flags.append("ROUTINE-DRIFT")
            note = "; ".join(flags) or "clean"
        if r.privacy_hits:
            note = f"PRIVACY:{len(r.privacy_hits)} " + note
        body.append((
            f"{os.path.basename(r.worktree)}:{r.path}",
            r.status, r.stamp_kind or "-", r.schema_key or "-", _short(note, reason_width),
        ))
    widths = [max(len(hdr[i]), *(len(b[i]) for b in body)) if body else len(hdr[i])
              for i in range(5)]
    if markdown:
        out = ["| " + " | ".join(h.ljust(w) for h, w in zip(hdr, widths)) + " |",
               "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
        out += ["| " + " | ".join(c.ljust(w) for c, w in zip(b, widths)) + " |" for b in body]
        return "\n".join(out)
    out = ["  ".join(h.ljust(w) for h, w in zip(hdr, widths)),
           "  ".join("-" * w for w in widths)]
    out += ["  ".join(c.ljust(w) for c, w in zip(b, widths)) for b in body]
    return "\n".join(out)


def summarize(rows) -> dict:
    counts: dict = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    return {
        "total": len(rows),
        "re_derivable": sum(1 for r in rows if r.ok),
        "by_status": dict(sorted(counts.items())),
        "privacy_flagged": sum(1 for r in rows if r.privacy_hits),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worktree", action="append", default=None,
                    help="repeatable; defaults to both Paper-1 worktrees")
    ap.add_argument("--strict-sha", action="store_true",
                    help="an abbreviated (<40 char) stamp becomes a FAILURE")
    ap.add_argument("--include-fixtures", action="store_true",
                    help="also audit tests/fixtures/** (excluded by default)")
    ap.add_argument("--privacy", action="store_true",
                    help="also run the JSON real-DESI privacy scanner on each artifact")
    ap.add_argument("--format", choices=("table", "markdown", "json"), default="table")
    ap.add_argument("--head", default="HEAD")
    args = ap.parse_args(argv)

    worktrees = args.worktree or list(DEFAULT_WORKTREES)
    worktrees = [w for w in worktrees if os.path.isdir(os.path.join(w, ".git"))
                 or os.path.exists(os.path.join(w, ".git"))]
    rows = audit(worktrees, require_full_sha=args.strict_sha,
                 include_fixtures=args.include_fixtures, check_privacy=args.privacy,
                 head=args.head)
    summary = summarize(rows)
    if args.format == "json":
        print(json.dumps({"summary": summary, "rows": [r.as_dict() for r in rows]}, indent=2))
    else:
        print(render_table(rows, markdown=(args.format == "markdown")))
        print()
        print(f"{summary['re_derivable']}/{summary['total']} committed artifacts are "
              f"RE_DERIVABLE.  by status: {summary['by_status']}")
        if args.privacy:
            print(f"privacy-flagged: {summary['privacy_flagged']}")
    return 0 if summary["re_derivable"] == summary["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
