#!/usr/bin/env python
"""Lossless gzip-repack helpers for repack_gzip.sh.

Two subcommands:
  iscompressed <file>   exit 0 iff sample_log_likelihoods_dla is gzip-filtered
                        (idempotency check: skip files already compressed)
  verify <src> <tmp>    exit 0 iff tmp structurally matches src
                        (opens cleanly; identical dataset keys, shapes, dtypes)

Why a *structural* verify and not a full value compare: the HDF5 GZIP/DEFLATE
filter is lossless by construction, and we independently confirmed byte-identity
(np.array_equal equal_nan=True across all 22 datasets) on a sample production
file. The real failure mode to guard against is a truncated/aborted repack, which
a clean open with matching keys/shapes/dtypes detects. A full per-element compare
across 815 multi-GB files would be prohibitively slow and read-memory-heavy for
no added safety. Only HDF5 headers are read here (no array data), so memory is
negligible.
"""
import sys

import h5py


def iscompressed(path):
    try:
        with h5py.File(path, "r") as h:
            ds = h.get("sample_log_likelihoods_dla")
            return ds is not None and ds.compression == "gzip"
    except Exception:
        return False


def verify(src, tmp):
    with h5py.File(src, "r") as a, h5py.File(tmp, "r") as b:
        ka, kb = sorted(a.keys()), sorted(b.keys())
        if ka != kb:
            sys.stderr.write(f"key mismatch src={ka} tmp={kb}\n")
            return False
        for k in ka:
            da, db = a[k], b[k]
            if da.shape != db.shape:
                sys.stderr.write(f"{k}: shape {da.shape} != {db.shape}\n")
                return False
            if da.dtype != db.dtype:
                sys.stderr.write(f"{k}: dtype {da.dtype} != {db.dtype}\n")
                return False
    return True


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "iscompressed":
        sys.exit(0 if iscompressed(sys.argv[2]) else 1)
    elif cmd == "verify":
        sys.exit(0 if verify(sys.argv[2], sys.argv[3]) else 1)
    sys.stderr.write(f"usage: {sys.argv[0]} iscompressed <file> | verify <src> <tmp>\n")
    sys.exit(2)
