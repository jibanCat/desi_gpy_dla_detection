#!/usr/bin/env python3
"""Pre-commit privacy guard for the CDDF results store.

Mock results are fine to commit; **real-LOA** spectra and any per-object value
derived from them are NOT (see ``docs`` / the real-data-privacy rule). The
results store normally lives on scratch outside the repo
(``$CDDF_STORE/{mock,real_loa}/...``), but a stray ``git add`` could pull a
real-LOA leaf into the public history. This guard scans the *staged* set and
fails the commit if anything real-LOA is about to be committed.

A path is treated as real-LOA (and the commit blocked) when ANY of:

  1. it sits under a ``real_loa/`` store partition (the privacy partition dir
     ``CDDF_analysis.results_store._PRIVACY_SUBDIR['real-LOA']``), OR
  2. it IS a ``provenance.json`` whose ``privacy.class == "real-LOA"`` or
     ``privacy.shareable is False``, OR
  3. it sits in the same leaf directory as such a ``provenance.json``.

Provenance reads fail **closed**: a ``provenance.json`` that cannot be parsed
(or is missing required privacy fields) is treated as SUSPECT and blocks every
staged path in its leaf dir, with a clear message.

Field names are keyed exactly to ``CDDF_analysis/hbi/provenance.py``:
``provenance.json`` carries a nested ``"privacy"`` object with ``"class"``
(``"mock"`` | ``"real-LOA"``) and ``"shareable"`` (bool).

Usage::

    # default: scan the git staged set (use as a pre-commit hook)
    python tools/provenance/precommit_privacy_guard.py

    # scan explicit paths (CI / tests), no git index needed
    python tools/provenance/precommit_privacy_guard.py --paths a/provenance.json b.npz
    python tools/provenance/precommit_privacy_guard.py --all  # alias for --paths .

    # override the repo root used to resolve relative staged paths
    python tools/provenance/precommit_privacy_guard.py --root /some/repo

Exit code 0 = clean, 1 = at least one offending path (or a SUSPECT provenance),
2 = usage / git error. Dependency-free (stdlib only).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# The store's real-LOA top-level partition dir name. Mirrors
# ``CDDF_analysis.results_store._PRIVACY_SUBDIR['real-LOA']`` — kept as a literal
# so this guard stays import-free (it must run before/around a commit, possibly
# without the package importable).
REAL_LOA_PARTITION = "real_loa"

PROVENANCE_NAME = "provenance.json"


# --------------------------------------------------------------------------- #
# staged-path discovery                                                        #
# --------------------------------------------------------------------------- #
def staged_paths(root: Path) -> list[Path]:
    """Return absolute paths of files in the git index (``git diff --cached``).

    Uses ``-z`` (NUL-delimited) so paths with spaces/newlines are handled. Raises
    ``RuntimeError`` on git failure (not a repo, git missing) so the caller can
    exit with a usage code rather than silently passing.
    """
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "-z"],
            cwd=str(root),
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:  # git not installed
        raise RuntimeError("git executable not found") from exc
    except subprocess.CalledProcessError as exc:
        msg = exc.stderr.decode(errors="replace").strip() if exc.stderr else ""
        raise RuntimeError(f"git diff --cached failed: {msg or exc}") from exc
    names = [n for n in out.decode().split("\0") if n]
    return [(root / n).resolve() for n in names]


def expand_paths(raw: list[str], root: Path) -> list[Path]:
    """Resolve an explicit ``--paths`` list to absolute file paths.

    Directories are walked recursively (so ``--all`` / ``--paths .`` works);
    individual files pass through. Non-existent paths are kept (resolved) so the
    guard can still flag them by their location (e.g. a path under ``real_loa/``
    that no longer exists on disk but is still staged).
    """
    files: list[Path] = []
    for r in raw:
        p = (root / r).resolve() if not os.path.isabs(r) else Path(r).resolve()
        if p.is_dir():
            for sub in sorted(p.rglob("*")):
                if sub.is_file():
                    files.append(sub)
        else:
            files.append(p)
    return files


# --------------------------------------------------------------------------- #
# privacy checks                                                               #
# --------------------------------------------------------------------------- #
def _under_real_loa_partition(path: Path) -> bool:
    """True iff any path component is the real-LOA store partition dir."""
    return REAL_LOA_PARTITION in path.parts


def _read_provenance_privacy(prov_path: Path) -> tuple[str, str]:
    """Classify a ``provenance.json`` by its privacy fields.

    Returns ``(verdict, reason)`` where verdict is one of:
      * ``"ok"``      — privacy.class == "mock" and shareable is not False
      * ``"real"``    — privacy.class == "real-LOA" or shareable is False
      * ``"suspect"`` — unparseable / missing privacy fields (fail closed)
    """
    try:
        text = prov_path.read_text()
    except OSError as exc:
        return "suspect", f"unreadable provenance.json ({exc})"
    try:
        rec = json.loads(text)
    except (ValueError, json.JSONDecodeError) as exc:
        return "suspect", f"malformed provenance.json (not valid JSON: {exc})"
    if not isinstance(rec, dict):
        return "suspect", "malformed provenance.json (top level is not an object)"

    privacy = rec.get("privacy")
    if not isinstance(privacy, dict):
        return "suspect", "provenance.json has no usable 'privacy' object"

    cls = privacy.get("class")
    shareable = privacy.get("shareable")

    if cls == "real-LOA":
        return "real", "privacy.class == 'real-LOA'"
    if shareable is False:
        return "real", "privacy.shareable == false"
    if cls == "mock":
        # mock + (shareable true or absent) is the only clean case.
        return "ok", "privacy.class == 'mock'"
    # Unknown class with no real-LOA / unshareable signal: fail closed.
    return "suspect", f"provenance.json privacy.class is unexpected ({cls!r})"


def scan(paths: list[Path]) -> list[tuple[Path, str]]:
    """Return a list of ``(offending_path, reason)`` for every staged path that
    must not be committed. Empty list == clean.

    Three passes feed each other:
      1. partition: anything under ``real_loa/`` is blocked outright.
      2. provenance: each staged ``provenance.json`` is classified; real / suspect
         taints its whole leaf directory.
      3. neighbours: any staged path in a tainted leaf dir is blocked, even if the
         provenance.json itself was not staged (we read it off disk if present).
    """
    offenders: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    def add(p: Path, reason: str) -> None:
        if p not in seen:
            seen.add(p)
            offenders.append((p, reason))

    # Pass 1: partition rule (cheap, location-only).
    for p in paths:
        if _under_real_loa_partition(p):
            add(p, f"under a '{REAL_LOA_PARTITION}/' store partition")

    # Pass 2: classify provenance.json files, building a taint map of leaf dirs.
    # A leaf dir is tainted if ITS provenance.json is real/suspect. We consider
    # both staged provenance.json files AND, for any staged non-provenance file,
    # an on-disk provenance.json sitting beside it in the same dir.
    tainted_dirs: dict[Path, str] = {}

    def classify_dir(leaf: Path) -> None:
        if leaf in tainted_dirs:
            return
        prov = leaf / PROVENANCE_NAME
        if not prov.exists():
            return
        verdict, reason = _read_provenance_privacy(prov)
        if verdict in ("real", "suspect"):
            label = "real-LOA" if verdict == "real" else "SUSPECT (fail-closed)"
            tainted_dirs[leaf] = f"{label}: {reason}"

    candidate_dirs = {p.parent for p in paths}
    for leaf in candidate_dirs:
        classify_dir(leaf)

    # Also directly flag a staged provenance.json that is itself real/suspect,
    # so the message names the provenance file precisely.
    for p in paths:
        if p.name == PROVENANCE_NAME:
            verdict, reason = _read_provenance_privacy(p)
            if verdict == "real":
                add(p, f"real-LOA provenance.json — {reason}")
            elif verdict == "suspect":
                add(p, f"SUSPECT provenance.json (fail-closed) — {reason}")

    # Pass 3: any staged path inside a tainted leaf dir.
    for p in paths:
        if p.parent in tainted_dirs and p.name != PROVENANCE_NAME:
            add(p, f"sibling of a {tainted_dirs[p.parent]} provenance.json")

    return offenders


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Block real-LOA results-store leaves from being committed to the "
            "public repo."
        ),
    )
    ap.add_argument(
        "--paths",
        nargs="+",
        metavar="PATH",
        help="Scan these explicit paths (files or dirs) instead of the git index.",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Scan the whole --root recursively (alias for '--paths .').",
    )
    ap.add_argument(
        "--root",
        default=None,
        help="Repo root used to resolve relative paths (default: git toplevel "
        "of CWD, else CWD).",
    )
    return ap


def _resolve_root(arg_root: str | None) -> Path:
    if arg_root:
        return Path(arg_root).resolve()
    try:
        top = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        if top:
            return Path(top).resolve()
    except Exception:
        pass
    return Path.cwd().resolve()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = _resolve_root(args.root)

    if args.all or args.paths:
        raw = list(args.paths or [])
        if args.all:
            raw.append(".")
        paths = expand_paths(raw, root)
        mode = "explicit-paths"
    else:
        try:
            paths = staged_paths(root)
        except RuntimeError as exc:
            print(f"privacy-guard: ERROR resolving staged files: {exc}",
                  file=sys.stderr)
            return 2
        mode = "git-staged"

    offenders = scan(paths)

    if not offenders:
        print(f"privacy-guard: OK — {len(paths)} {mode} path(s) clean "
              f"(no real-LOA leaves).")
        return 0

    print("privacy-guard: BLOCKED — real-LOA results must NOT be committed.",
          file=sys.stderr)
    print(f"  {len(offenders)} offending path(s):", file=sys.stderr)
    for p, reason in offenders:
        try:
            shown = p.relative_to(root)
        except ValueError:
            shown = p
        print(f"    - {shown}\n        reason: {reason}", file=sys.stderr)
    print(
        "\n  Real-LOA store leaves belong ONLY under $CDDF_STORE/"
        f"{REAL_LOA_PARTITION}/ on scratch, never in git.\n"
        "  Unstage them (git restore --staged <path>) and retry.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
