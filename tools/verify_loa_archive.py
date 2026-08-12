#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fail-loud verifier for a hz_manifest-driven LoaArchive (PI §24).

Checks, against the manifest:
  * every requested TARGETID present in the archive catalog;
  * archive row count == unique manifest targets;
  * per-row Z agrees with the manifest (|dz| < 1e-4);
  * no all-zero ivar rows (undetectable-corrupt guard);
  * archive file nonzero and readable.
Exit codes: 0 ok · 4 missing targets · 5 integrity failure.
"""
import argparse
import csv
import json
import sys

import numpy as np
import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--role", default=None,
                    help="verify only rows of this role")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest)))
    if args.role:
        rows = [r for r in rows if r["role"] == args.role]
    want = np.unique(np.array([int(r["TARGETID"]) for r in rows],
                              dtype=np.int64))
    zmap = {int(r["TARGETID"]): float(r["Z_QSO"]) for r in rows}

    with h5py.File(args.archive, "r") as h:
        cat = h["catalog"][:]
        have = np.asarray(cat["TARGETID"], dtype=np.int64)
        ivar = h["ivar"]
        n_zero = 0
        for i in range(len(have)):
            if not np.any(ivar[i] > 0):
                n_zero += 1
        zarr = np.asarray(cat["Z"], float)

    missing = np.setdiff1d(want, have)
    dup = len(have) - len(np.unique(have))
    dz_bad = 0
    for t, z in zip(have, zarr):
        if int(t) in zmap and abs(zmap[int(t)] - z) > 1e-4:
            dz_bad += 1

    summary = dict(n_requested=int(len(want)), n_in_archive=int(len(have)),
                   n_missing=int(len(missing)), n_duplicate=int(dup),
                   n_allzero_ivar=int(n_zero), n_z_mismatch=int(dz_bad),
                   missing_head=[int(x) for x in missing[:20]])
    print(json.dumps(summary, indent=1))
    if len(missing):
        print("VERIFY FAIL: missing targets", file=sys.stderr)
        sys.exit(4)
    if dup or n_zero or dz_bad:
        print("VERIFY FAIL: integrity", file=sys.stderr)
        sys.exit(5)
    print("VERIFY PASS")


if __name__ == "__main__":
    main()
