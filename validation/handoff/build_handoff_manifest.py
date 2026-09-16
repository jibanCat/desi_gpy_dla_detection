#!/usr/bin/env python
"""build_handoff_manifest.py -- build and VERIFY the Paper-lane handoff manifest.

Two modes, one registry (``validation/handoff/registry.py``):

    # build HANDOFF_MANIFEST.json + HANDOFF_SHA256SUMS in the handoff root
    python -m validation.handoff.build_handoff_manifest --root <H>

    # fail closed (exit 1) if any listed artifact is missing or its sha changed
    python -m validation.handoff.build_handoff_manifest --root <H> --verify

The manifest records, for the handoff of record:

* the handoff date, and both repositories' HEAD / branch / dirty state;
* the archival tags (or ``PENDING_TAG`` when the commander has not cut them);
* the canonical result, systematics, model-of-record, claim-ledger, literature
  contract, figure/table map and read-only declaration;
* the full-archive locations (Turbo mirror + scratch root) with the sha256 of
  their ``SHA256SUMS`` digests;
* a sha256 for EVERY file in the handoff root and for EVERY canonical source it
  points at, each with a PUBLIC_SAFE / PRIVATE classification;
* the superseded / not-for-use list.

READ-ONLY over the Science lane: this script writes only inside the handoff
root.  It never touches a frozen product.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys

try:                       # normal: python -m validation.handoff.build_...
    from . import registry as reg
except ImportError:        # pragma: no cover -- direct script execution
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import registry as reg                                        # noqa: E402

MANIFEST = "HANDOFF_MANIFEST.json"
SUMS = "HANDOFF_SHA256SUMS"
SCHEMA = "paper1_lowz_paper_lane_handoff/1"
_CHUNK = 1 << 20

# Handoff files this builder produces; excluded from its own hashing so that a
# build is idempotent.
SELF_FILES = {MANIFEST, SUMS}


class HandoffIntegrityError(RuntimeError):
    """A listed artifact is missing or its sha256 changed."""


# ----------------------------------------------------------------- primitives
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(repo, *args):
    try:
        out = subprocess.run(["git", "-C", repo] + list(args),
                             capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def repo_state(path):
    head = _git(path, "rev-parse", "HEAD")
    if head is None:
        return {"path": path, "resolved": False}
    porcelain = _git(path, "status", "--porcelain") or ""
    dirty_files = [l[3:] for l in porcelain.splitlines() if l.strip()]
    tracked = _git(path, "status", "--porcelain", "--untracked-files=no") or ""
    return {
        "path": path,
        "resolved": True,
        "branch": _git(path, "rev-parse", "--abbrev-ref", "HEAD"),
        "head": head,
        "head_short": head[:7],
        "dirty": bool(porcelain.strip()),
        "dirty_tracked": bool(tracked.strip()),
        "dirty_files": dirty_files,
    }


def tag_state(repos):
    """``{tag: {repo: commit-or-PENDING_TAG}}`` for the three archival tags."""
    out = {}
    for tag in reg.ARCHIVAL_TAGS:
        row = {}
        for name, path in repos.items():
            commit = _git(path, "rev-list", "-n", "1", tag)
            row[name] = commit if commit else "PENDING_TAG"
        out[tag] = row
    return out


def walk_handoff(root):
    """Sorted relative paths of every file in the handoff root (minus SELF)."""
    rows = []
    for base, _dirs, files in os.walk(root):
        for name in sorted(files):
            rel = os.path.relpath(os.path.join(base, name), root)
            if rel in SELF_FILES:
                continue
            rows.append(rel)
    rows.sort()
    return rows


# ------------------------------------------------------------- classification
def classify_handoff_file(rel):
    """PUBLIC_SAFE / PRIVATE for a file inside the handoff root."""
    for hrel, _src, cls, _note in reg.FIGURES + [
            (t[0], t[1], t[2], t[3]) for t in reg.TABLES]:
        if hrel == rel:
            return cls
    override = getattr(reg, "DOC_CLASSIFICATION", {}).get(rel)
    if override:
        return override
    # Remaining handoff documents name private product PATHS but carry no real
    # values; the handoff root as a whole lives in the private notes repo.
    return "PRIVATE_REPO_NO_VALUES"


def _entry(path, cls, role, key=None):
    row = {"path": path, "classification": cls, "role": role}
    if key:
        row["key"] = key
    if os.path.isdir(path):
        row["kind"] = "directory"
        row["exists"] = True
        return row
    if not os.path.isfile(path):
        row["exists"] = False
        row["sha256"] = None
        return row
    row["exists"] = True
    row["bytes"] = os.path.getsize(path)
    row["sha256"] = sha256_file(path)
    return row


# -------------------------------------------------------------------- builder
def build(root):
    repos = {"notes": reg.N, "code": reg.CODE}
    states = {k: repo_state(v) for k, v in repos.items()}

    handoff_files = []
    for rel in walk_handoff(root):
        full = os.path.join(root, rel)
        handoff_files.append({
            "path": rel,
            "bytes": os.path.getsize(full),
            "sha256": sha256_file(full),
            "classification": classify_handoff_file(rel),
        })

    canonical_sources = [
        _entry(p, cls, role, key) for key, p, cls, role in reg.CANONICAL_SOURCES
    ]

    figures = []
    for hrel, src, cls, note in reg.FIGURES:
        figures.append({
            "handoff_path": hrel,
            "canonical_source": src,
            "source_sha256": sha256_file(src) if os.path.isfile(src) else None,
            "classification": cls,
            "note": note,
        })
    tables = []
    for hrel, src, cls, note in reg.TABLES:
        tables.append({
            "handoff_path": hrel,
            "canonical_source": src,
            "source_sha256": sha256_file(src) if os.path.isfile(src) else None,
            "classification": cls,
            "note": note,
        })

    archives = []
    for name, path, digest, durability in reg.ARCHIVES:
        row = {"name": name, "path": path, "durability": durability,
               "exists": os.path.isdir(path), "digest_file": digest}
        if os.path.isfile(digest):
            row["digest_file_sha256"] = sha256_file(digest)
            with open(digest) as fh:
                row["digest_entries"] = sum(1 for l in fh if l.strip())
        else:
            row["digest_file_sha256"] = None
            row["digest_entries"] = None
        archives.append(row)

    superseded = [
        {"path": p, "reason": why, "use_instead": instead,
         "exists": os.path.exists(p)}
        for p, why, instead in reg.SUPERSEDED
    ]
    superseded_classes = [
        {"class": name, "reason": why, "use_instead": instead}
        for name, why, instead in reg.SUPERSEDED_CLASSES
    ]

    expected = []
    for name, owner in reg.EXPECTED_FROM_OTHER_AGENTS:
        full = os.path.join(root, name)
        expected.append({
            "path": name, "written_by": owner,
            "status": "present" if os.path.isfile(full) else "PENDING",
        })

    def _pick(keys):
        out = []
        for row in canonical_sources:
            if row.get("key") in keys:
                out.append(row)
        return out

    doc = {
        "schema": SCHEMA,
        "handoff_date": reg.HANDOFF_DATE,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "Science lane CLOSED / READ-ONLY (PI 2026-09-16)",
        "handoff_root": root,
        "repositories": states,
        "archival_tags": tag_state(repos),
        "canonical": {
            "result_of_record": _pick({"result_of_record.md",
                                       "result_of_record.json",
                                       "result_pooled_scratch.json"}),
            "systematics": _pick({"systematics.release.md",
                                  "systematics.release.json",
                                  "systematics.release.csv",
                                  "systematics.real_hw.md",
                                  "systematics.real_hw.json",
                                  "s6.result.md", "s6.result.json",
                                  "s6.release_safe.md"}),
            "model_of_record": _pick({"model_of_record.json", "freeze_record"}),
            "claim_ledger": [
                {"path": os.path.join(root, n), "status":
                 "present" if os.path.isfile(os.path.join(root, n)) else "PENDING"}
                for n in ("CLAIM_LEDGER.md", "CLAIM_LEDGER.json")],
            "literature_contract": [
                {"path": os.path.join(root, n), "status":
                 "present" if os.path.isfile(os.path.join(root, n)) else "PENDING"}
                for n in ("LITERATURE_COMPARISON_DEFINITIONS.md",
                          "ESTIMAND_NOTATION_CONTRACT.md")],
            "figure_table_map": [
                {"path": os.path.join(root, n), "status":
                 "present" if os.path.isfile(os.path.join(root, n)) else "PENDING"}
                for n in ("CANONICAL_SOURCE_MAP.md",
                          "PAPER_FIGURE_TABLE_SOURCE_MAP.md")],
            "read_only_declaration": _pick({"read_only_declaration"}),
            "real_pack_of_record": _pick({"real_pack_of_record"}),
        },
        "figures": figures,
        "tables": tables,
        "archives": archives,
        "handoff_files": handoff_files,
        "canonical_sources": canonical_sources,
        "superseded_not_for_use": superseded,
        "superseded_classes": superseded_classes,
        "expected_from_other_agents": expected,
        "counts": {
            "handoff_files": len(handoff_files),
            "handoff_bytes": sum(r["bytes"] for r in handoff_files),
            "figures": len(figures),
            "tables": len(tables),
            "canonical_sources": len(canonical_sources),
            "canonical_sources_missing": sum(
                1 for r in canonical_sources if not r.get("exists")),
            "private_handoff_files": sum(
                1 for r in handoff_files if r["classification"] == "PRIVATE"),
            "superseded_entries": len(superseded),
        },
        "verify_command": ("python -m validation.handoff.build_handoff_manifest "
                           "--root %s --verify" % root),
        "rebuild_note": (
            "This manifest describes the handoff root COMPLETELY: --verify "
            "fails on a missing file, a changed sha256, AND on any file that "
            "appears in the root but is not listed. If another agent adds a "
            "file to the handoff, rebuild (drop --verify) rather than editing "
            "the manifest; nothing outside the handoff root is ever written."),
    }
    return doc


def write(root, doc):
    mpath = os.path.join(root, MANIFEST)
    with open(mpath, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=False)
        fh.write("\n")

    lines = ["# Paper-lane handoff sha256 digest -- generated %s"
             % doc["generated_utc"],
             "# section 1: files inside the handoff root (relative paths)"]
    for row in doc["handoff_files"]:
        lines.append("%s  %s" % (row["sha256"], row["path"]))
    lines.append("# section 2: canonical sources outside the handoff root "
                 "(absolute paths)")
    for row in doc["canonical_sources"]:
        if row.get("sha256"):
            lines.append("%s  %s" % (row["sha256"], row["path"]))
    spath = os.path.join(root, SUMS)
    with open(spath, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return mpath, spath


# --------------------------------------------------------------------- verify
def verify(root):
    """Re-hash everything the manifest lists.  Returns a list of failures."""
    mpath = os.path.join(root, MANIFEST)
    if not os.path.isfile(mpath):
        return ["MISSING MANIFEST: %s" % mpath]
    doc = json.load(open(mpath))
    fails = []

    if doc.get("schema") != SCHEMA:
        fails.append("SCHEMA: expected %s, manifest says %r"
                     % (SCHEMA, doc.get("schema")))

    for row in doc.get("handoff_files", []):
        full = os.path.join(root, row["path"])
        if not os.path.isfile(full):
            fails.append("MISSING handoff file: %s" % row["path"])
            continue
        got = sha256_file(full)
        if got != row["sha256"]:
            fails.append("SHA CHANGED handoff file: %s (manifest %s, file %s)"
                         % (row["path"], row["sha256"][:12], got[:12]))

    # A file that appeared in the root but is not in the manifest is also a
    # failure: the manifest must describe the handoff completely.
    listed = {r["path"] for r in doc.get("handoff_files", [])}
    for rel in walk_handoff(root):
        if rel not in listed:
            fails.append("UNLISTED file in handoff root: %s" % rel)

    for row in doc.get("canonical_sources", []):
        if row.get("kind") == "directory":
            if not os.path.isdir(row["path"]):
                fails.append("MISSING canonical directory: %s" % row["path"])
            continue
        if row.get("sha256") is None:
            if row.get("exists"):
                fails.append("canonical source has no sha: %s" % row["path"])
            continue
        if not os.path.isfile(row["path"]):
            fails.append("MISSING canonical source: %s" % row["path"])
            continue
        got = sha256_file(row["path"])
        if got != row["sha256"]:
            fails.append("SHA CHANGED canonical source: %s (manifest %s, file %s)"
                         % (row["path"], row["sha256"][:12], got[:12]))

    for group, key in (("figures", "source_sha256"), ("tables", "source_sha256")):
        for row in doc.get(group, []):
            src = row["canonical_source"]
            if row.get(key) is None:
                fails.append("%s entry has no source sha: %s"
                             % (group, row["handoff_path"]))
                continue
            if not os.path.isfile(src):
                fails.append("MISSING %s source: %s" % (group, src))
                continue
            if sha256_file(src) != row[key]:
                fails.append("SHA CHANGED %s source: %s" % (group, src))
            hfull = os.path.join(root, row["handoff_path"])
            if not os.path.isfile(hfull):
                fails.append("MISSING curated %s: %s"
                             % (group, row["handoff_path"]))

    for row in doc.get("archives", []):
        if not os.path.isdir(row["path"]):
            fails.append("MISSING archive: %s" % row["path"])
        if row.get("digest_file_sha256"):
            if not os.path.isfile(row["digest_file"]):
                fails.append("MISSING archive digest: %s" % row["digest_file"])
            elif sha256_file(row["digest_file"]) != row["digest_file_sha256"]:
                fails.append("SHA CHANGED archive digest: %s"
                             % row["digest_file"])
    return fails


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=reg.H, help="the handoff root (H)")
    ap.add_argument("--verify", action="store_true",
                    help="fail closed instead of building")
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)

    if a.verify:
        fails = verify(root)
        if fails:
            print("HANDOFF VERIFY: FAILED (%d problems)" % len(fails))
            for f in fails:
                print("  -", f)
            return 1
        doc = json.load(open(os.path.join(root, MANIFEST)))
        print("HANDOFF VERIFY: OK -- %d handoff files, %d canonical sources, "
              "%d figures, %d tables"
              % (doc["counts"]["handoff_files"],
                 doc["counts"]["canonical_sources"],
                 doc["counts"]["figures"], doc["counts"]["tables"]))
        return 0

    if not os.path.isdir(root):
        raise HandoffIntegrityError("handoff root does not exist: %s" % root)
    doc = build(root)
    missing = [r["path"] for r in doc["canonical_sources"]
               if not r.get("exists")]
    mpath, spath = write(root, doc)
    print("handoff manifest ->", mpath)
    print("handoff sha256sums ->", spath)
    print("files %d (%.1f MB), figures %d, tables %d, canonical sources %d"
          % (doc["counts"]["handoff_files"],
             doc["counts"]["handoff_bytes"] / 1e6,
             doc["counts"]["figures"], doc["counts"]["tables"],
             doc["counts"]["canonical_sources"]))
    if missing:
        print("FAIL CLOSED: %d canonical sources are missing:" % len(missing))
        for m in missing:
            print("  -", m)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
