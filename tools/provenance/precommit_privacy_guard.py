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

# Both store privacy partitions. A path that descends through either of these is
# inside the recognizable store layout ``.../{mock,real_loa}/{dataset}/{stage}/
# {leaf}/...``; such a path with NO classifiable provenance anywhere up its chain
# fails CLOSED (it might be a real-LOA artifact whose provenance was not staged).
STORE_PARTITIONS = ("mock", "real_loa")

PROVENANCE_NAME = "provenance.json"


# --------------------------------------------------------------------------- #
# staged-path discovery                                                        #
# --------------------------------------------------------------------------- #
def staged_paths(root: Path) -> list[tuple[Path, Path]]:
    """Return ``(literal, resolved)`` path pairs for files in the git index.

    ``literal`` is the staged path joined to ``root`` WITHOUT ``.resolve()`` —
    its component names (``real_loa/`` etc.) are exactly what git tracks, so the
    partition rule reads them even when a component is a symlink pointing at a
    benign target. ``resolved`` is ``literal.resolve()`` for a belt-and-suspenders
    realpath partition check.

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
    return [_pair(root / n) for n in names]


def _pair(literal: Path) -> tuple[Path, Path]:
    """Build the ``(literal, resolved)`` pair; resolution never raises."""
    try:
        resolved = literal.resolve()
    except OSError:
        resolved = literal
    return literal, resolved


def expand_paths(raw: list[str], root: Path) -> list[tuple[Path, Path]]:
    """Resolve an explicit ``--paths`` list to ``(literal, resolved)`` pairs.

    Directories are walked recursively (so ``--all`` / ``--paths .`` works);
    individual files pass through. Non-existent paths are kept so the guard can
    still flag them by their LITERAL location (e.g. a path under ``real_loa/``
    that no longer exists on disk but is still staged). The literal path keeps the
    un-resolved component names so the partition rule cannot be defeated by a
    symlinked component.
    """
    files: list[tuple[Path, Path]] = []
    for r in raw:
        literal = (root / r) if not os.path.isabs(r) else Path(r)
        # Walk dirs by the resolved location (need real on-disk children) but keep
        # each child's literal path rooted at the un-resolved parent.
        resolved_dir = _pair(literal)[1]
        if resolved_dir.is_dir():
            for sub in sorted(resolved_dir.rglob("*")):
                if sub.is_file():
                    files.append(_pair(sub))
        else:
            files.append(_pair(literal))
    return files


# --------------------------------------------------------------------------- #
# privacy checks                                                               #
# --------------------------------------------------------------------------- #
def _under_real_loa_partition(path: Path) -> bool:
    """True iff any path component is the real-LOA store partition dir."""
    return REAL_LOA_PARTITION in path.parts


def _store_partition_index(path: Path) -> int | None:
    """Index of the FIRST store-partition component (``mock``/``real_loa``) in
    ``path.parts``, or ``None`` if the path is not under a recognizable store
    partition. Used to decide whether a path lives in the store layout
    ``.../{partition}/{dataset}/{stage}/{leaf}/...`` (so a missing provenance is
    suspect) versus an ordinary repo file (which the guard ignores)."""
    parts = path.parts
    for i, comp in enumerate(parts):
        if comp in STORE_PARTITIONS:
            return i
    return None


def _looks_like_store_layout(path: Path) -> bool:
    """True iff ``path`` sits inside a store leaf under a ``{partition}/{dataset}/
    {stage}/{leaf}/`` prefix (at least partition + dataset + stage + leaf + the
    staged file = 5 components from the partition on). Such a file is expected to
    have a provenance.json up its chain; if none is classifiable, fail closed."""
    idx = _store_partition_index(path)
    if idx is None:
        return False
    # parts[idx]=partition, +1 dataset, +2 stage, +3 leaf, +4.. the staged file.
    return len(path.parts) - idx >= 5


def _ancestor_provenances(path: Path, root: Path) -> list[Path]:
    """Every ``provenance.json`` (case-insensitive filename) sitting in ``path``'s
    parent or any ancestor directory up to (and including) ``root``. Nearest
    ancestor first. Real-LOA contagion is DOWNWARD, so a parent leaf's provenance
    taints files in subdirs too (closes the B3 parent-only bypass); the
    case-insensitive match closes the B1 wrong-case bypass."""
    found: list[Path] = []
    try:
        root_res = root.resolve()
    except OSError:
        root_res = root
    cur = path.parent
    seen: set[Path] = set()
    while True:
        if cur in seen:
            break
        seen.add(cur)
        try:
            entries = list(cur.iterdir()) if cur.is_dir() else []
        except OSError:
            entries = []
        for e in entries:
            if e.name.lower() == PROVENANCE_NAME and e.is_file():
                found.append(e)
                break  # one provenance per dir
        # stop once we've processed the root (or climbed past it).
        try:
            cur_res = cur.resolve()
        except OSError:
            cur_res = cur
        if cur_res == root_res:
            break
        parent = cur.parent
        if parent == cur:  # filesystem root reached
            break
        cur = parent
    return found


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


def scan(paths: list[tuple[Path, Path]], root: Path | None = None) -> list[tuple[Path, str]]:
    """Return a list of ``(offending_path, reason)`` for every staged path that
    must not be committed. Empty list == clean.

    ``paths`` is a list of ``(literal, resolved)`` pairs (see ``staged_paths`` /
    ``expand_paths``). The literal path keeps the staged component names; the
    resolved path follows symlinks.

    Passes:
      1. partition: anything whose LITERAL components include ``real_loa/`` is
         blocked outright (symlink-proof); a cheap resolved-realpath check is also
         applied so a symlink whose TARGET lives under ``real_loa/`` is caught too.
      2. provenance walk-UP: classify EVERY ``provenance.json`` (case-insensitive)
         from each staged file's parent up to ``root``. real / suspect taints the
         file. This closes the wrong-case (B1) and parent-only (B3) bypasses.
      3. store-layout fail-CLOSED: a staged file inside a recognizable
         ``{partition}/{dataset}/{stage}/{leaf}/`` layout with NO classifiable
         provenance anywhere up its chain is blocked as suspect (B2).
      4. direct provenance: a staged ``provenance.json`` that is itself
         real/suspect is named precisely.
    """
    if root is None:
        root = Path.cwd()
    offenders: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    def add(p: Path, reason: str) -> None:
        if p not in seen:
            seen.add(p)
            offenders.append((p, reason))

    # Pass 1: partition rule (cheap, location-only) on BOTH the literal and the
    # resolved path. The literal check must stand alone (symlinked component);
    # the resolved check additionally catches a symlink whose target is real-LOA.
    for literal, resolved in paths:
        if _under_real_loa_partition(literal):
            add(literal, f"under a '{REAL_LOA_PARTITION}/' store partition")
        elif _under_real_loa_partition(resolved):
            add(literal,
                f"resolves (via symlink) under a '{REAL_LOA_PARTITION}/' store partition")

    # Pre-classify the ancestor chain of each staged file ONCE per directory.
    # A directory's provenance verdict is cached so repeated walk-ups are cheap.
    prov_verdict_cache: dict[Path, tuple[str, str]] = {}

    def verdict_for(prov: Path) -> tuple[str, str]:
        if prov not in prov_verdict_cache:
            prov_verdict_cache[prov] = _read_provenance_privacy(prov)
        return prov_verdict_cache[prov]

    # Pass 2 + 3: per staged file, walk up its provenance chain.
    for literal, _resolved in paths:
        if literal.name.lower() == PROVENANCE_NAME:
            continue  # handled directly in pass 4.
        provs = _ancestor_provenances(literal, root)
        chain_classified = False
        blocked = False
        for prov in provs:
            verdict, reason = verdict_for(prov)
            if verdict in ("real", "suspect"):
                label = "real-LOA" if verdict == "real" else "SUSPECT (fail-closed)"
                if prov.parent == literal.parent:
                    where = f"sibling of a {label}"
                else:
                    where = f"under a {label} ancestor's"
                add(literal, f"{where} provenance.json — {reason}")
                blocked = True
                chain_classified = True
                break
            if verdict == "ok":
                chain_classified = True
                # keep walking: a real/suspect ANCESTOR above a clean leaf still
                # taints (contagion is downward), so don't stop on the first ok.
                continue
        if blocked:
            continue
        # Pass 3: store-layout file with NO classifiable provenance up its chain.
        if not chain_classified and _looks_like_store_layout(literal):
            add(literal,
                "SUSPECT (fail-closed) — store-layout file with no classifiable "
                f"provenance anywhere up to the store root; a "
                f"'{PROVENANCE_NAME}' is required to clear it")

    # Pass 4: a staged provenance.json that is itself real/suspect — name it.
    for literal, _resolved in paths:
        if literal.name.lower() == PROVENANCE_NAME:
            verdict, reason = verdict_for(literal)
            if verdict == "real":
                add(literal, f"real-LOA provenance.json — {reason}")
            elif verdict == "suspect":
                add(literal, f"SUSPECT provenance.json (fail-closed) — {reason}")

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

    offenders = scan(paths, root)

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
