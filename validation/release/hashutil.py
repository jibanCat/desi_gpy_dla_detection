"""hashutil.py -- sha256 + SHA256SUMS verification with FAIL-CLOSED semantics.

The whole point of the release manifest is that a reader can check it, so a
reference that cannot be resolved is an ERROR, never a silently-dropped node:

* a sha256 that disagrees with an existing ``SHA256SUMS`` entry is ALWAYS fatal
  (that is corruption or a silent re-run, and no release may be built on it);
* a missing file is fatal in ``strict`` mode and recorded in ``unresolved``
  otherwise (the final HBI ladder may legitimately still be running).
"""
from __future__ import annotations

import hashlib
import os

__all__ = [
    "sha256_file", "sha256_bytes", "load_sha256sums", "verify_against_sums",
    "write_sha256sums", "parse_accept_stale", "ManifestIntegrityError",
]

_CHUNK = 1 << 20


class ManifestIntegrityError(RuntimeError):
    """Raised when a referenced artifact cannot be resolved or verified."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(blob):
    if isinstance(blob, str):
        blob = blob.encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_sha256sums(directory):
    """``{basename: sha256}`` from ``<directory>/SHA256SUMS`` ({} if absent)."""
    path = os.path.join(directory, "SHA256SUMS")
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            digest, name = parts[0], parts[1].lstrip("*").strip()
            out[os.path.basename(name)] = digest.lower()
    return out


def verify_against_sums(path, sums_cache=None, accept_stale=None):
    """(sha256, status) with status in {'match', 'absent-from-sums',
    'STALE-SUMS-ACCEPTED', 'MISSING'}.

    A MISMATCH raises immediately -- fail closed, unconditionally -- UNLESS the
    caller has explicitly named that exact (basename-or-path, observed sha256)
    pair in ``accept_stale``.  That escape hatch exists because a producer may
    legitimately append to a product AFTER its ``SHA256SUMS`` was stamped; it
    requires the operator to type the observed digest, so it can never silently
    absorb corruption, and every use is recorded in the manifest.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isfile(path):
        return None, "MISSING"
    if sums_cache is None:
        sums_cache = {}
    if directory not in sums_cache:
        sums_cache[directory] = load_sha256sums(directory)
    sums = sums_cache[directory]
    digest = sha256_file(path)
    recorded = sums.get(os.path.basename(path))
    if recorded is None:
        return digest, "absent-from-sums"
    if recorded != digest:
        allowed = (accept_stale or {})
        for key in (os.path.abspath(path), os.path.basename(path)):
            if allowed.get(key, "").lower() == digest:
                return digest, "STALE-SUMS-ACCEPTED"
        raise ManifestIntegrityError(
            "sha256 MISMATCH for %s: SHA256SUMS says %s, file is %s"
            % (path, recorded, digest))
    return digest, "match"


def parse_accept_stale(items):
    """``['name=sha', ...]`` -> ``{name: sha}`` for ``verify_against_sums``."""
    out = {}
    for item in (items or []):
        if "=" not in item:
            raise ValueError("--accept-stale wants PATH_OR_NAME=SHA256, got %r"
                             % item)
        key, sha = item.rsplit("=", 1)
        out[key.strip()] = sha.strip().lower()
    return out


def write_sha256sums(directory, out_name="SHA256SUMS"):
    """Hash every regular file under ``directory`` (recursively, excluding the
    sums file itself) and write a sorted, relative-path ``SHA256SUMS``."""
    rows = []
    for root, _dirs, files in os.walk(directory):
        for name in sorted(files):
            if name == out_name:
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, directory)
            rows.append((rel, sha256_file(full)))
    rows.sort()
    path = os.path.join(directory, out_name)
    with open(path, "w") as fh:
        for rel, digest in rows:
            fh.write("%s  %s\n" % (digest, rel))
    return path


if __name__ == "__main__":                                   # pragma: no cover
    import sys
    print(write_sha256sums(sys.argv[1]))
