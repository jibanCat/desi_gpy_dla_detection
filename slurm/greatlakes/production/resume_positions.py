#!/usr/bin/env python
"""Print the speclist POSITIONS that are not-done in a run's processed/ dir.

Mirrors desi-DLAGP.py's mock speclist construction EXACTLY (collect every
level2 healpix folder that has a spectra-16-<l2>.fits, sort by integer folder
name) so positions line up 1:1 with the --level2_start / --level2_end indexing.

A position is DONE iff its output processed-spectra-16-<healpix>.h5:
  - exists, AND
  - opens cleanly (a crashed mid-write leaves a truncated file that h5py
    refuses with OSError), AND
  - contains the core datasets with a non-empty spectrum axis.
The pipeline writes the whole file in one h5py.File("w") block, so a file is
all-or-nothing — there is no valid "partial data" state. This check is therefore
independent of gzip compression, and correct whether or not repack_gzip.sh has
run. --require-gzip additionally demands the file be gzip-compressed (the
stricter post-repack validity marker).

Usage:
  resume_positions.py --mockdir DIR --procdir DIR [--require-gzip] [--summary]
    -> not-done positions, one integer per line, ascending
"""
import argparse
import glob
import os
import re
import sys

import h5py

# Datasets every completed file must have. The per-spectrum COUNT axis is
# target_ids in mock outputs (dlasearch mock path) and spectrum_ids in the
# real-data path — accept either, require non-empty.
CORE_KEYS = ("sample_log_likelihoods_dla", "model_posteriors")
COUNT_KEYS = ("target_ids", "spectrum_ids")


def speclist_positions(mockdir):
    """(positions sorted by healpix) -> list of (position, healpix)."""
    datapath = os.path.join(mockdir, "spectra-16")
    level2 = []
    for l1 in os.listdir(datapath):
        p1 = os.path.join(datapath, l1)
        if not os.path.isdir(p1):
            continue
        for l2 in os.listdir(p1):
            if os.path.exists(os.path.join(p1, l2, f"spectra-16-{l2}.fits")):
                level2.append(int(l2))
    level2.sort()
    return list(enumerate(level2))  # (position, healpix)


def is_done(h5path, require_gzip):
    if not os.path.exists(h5path):
        return False
    try:
        with h5py.File(h5path, "r") as h:
            for k in CORE_KEYS:
                if k not in h:
                    return False
            count = next((h[ck].shape[0] for ck in COUNT_KEYS if ck in h), None)
            if not count:  # missing count axis or zero spectra
                return False
            if require_gzip and h["sample_log_likelihoods_dla"].compression != "gzip":
                return False
        return True
    except Exception:
        return False  # truncated / unreadable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mockdir", required=True)
    ap.add_argument("--procdir", required=True)
    ap.add_argument("--require-gzip", action="store_true")
    ap.add_argument("--summary", action="store_true",
                    help="print counts to stderr instead of just the list")
    args = ap.parse_args()

    pos_hp = speclist_positions(args.mockdir)
    not_done = []
    truncated = []
    for pos, hp in pos_hp:
        f = os.path.join(args.procdir, f"processed-spectra-16-{hp}.h5")
        if not is_done(f, args.require_gzip):
            not_done.append(pos)
            if os.path.exists(f):
                truncated.append((pos, hp))

    if args.summary:
        sys.stderr.write(
            f"total positions={len(pos_hp)}  done={len(pos_hp)-len(not_done)}  "
            f"not_done={len(not_done)}  (of which present-but-bad={len(truncated)})\n"
        )
        if truncated:
            sys.stderr.write(
                "  present-but-bad (truncated/partial) healpix: "
                + ",".join(str(hp) for _, hp in truncated) + "\n"
            )
    for p in not_done:
        print(p)


if __name__ == "__main__":
    main()
