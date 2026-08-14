#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GL arrival verification for the H2 arm-A mini-archive (PI 2026-08-14).

PRE-DECLARED criteria (all must hold; fail-loud):
 1. sha256 of the delivered h5 matches the NERSC-side .sha256 companion;
 2. coverage: all 180 frozen arm-A TARGETIDs present (target-list sha
    c7a636ac…), 0 malformed, 0 all-masked, uniform schema v1;
 3. wavelength grid f4-identical to the committed native f8 grid (and to
    the production archive grid);
 4. representation check on the 21-target overlap with the -12c production
    archive: flux/ivar/mask rows BITWISE-identical (same builder, same
    sources, same semantics -> exact equality expected; any deviation is a
    construction difference and FAILS).
"""
import csv
import hashlib
import json
import sys

import numpy as np
import h5py

PROD_ARC = ("/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/"
            "loa_highz_2026-08-12c/archive/loa_hz_archive_v1.h5")
MANIFEST = ("/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/"
            "loa_hz_prep_2026-08-12/h2_armA_manifest.csv")
GRID = "/home/mfho/wt_forward_2026_08/data/brz_wave_grid_f8.npy"
TLIST_SHA = "c7a636acca73bea7d275abf72b9b35190406586eba8fabcf5fd3d921cf3ef8ef"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    mini = sys.argv[1]
    fails = []
    # 1 checksum
    expected = open(mini + ".sha256").read().split()[0]
    got = sha_file(mini)
    if got != expected:
        fails.append(f"sha256 mismatch: {got} != {expected}")
    # 2 coverage/schema
    man = list(csv.DictReader(open(MANIFEST)))
    tids = sorted(int(r["TARGETID"]) for r in man)
    tsha = hashlib.sha256("\n".join(map(str, tids)).encode()).hexdigest()
    assert tsha == TLIST_SHA, "manifest drift on GL"
    h = h5py.File(mini, "r")
    if int(h.attrs.get("schema_version", -1)) != 1:
        fails.append("schema_version != 1")
    cat = h["catalog"][:]
    have = set(int(t) for t in cat["TARGETID"])
    missing = [t for t in tids if t not in have]
    if missing:
        fails.append(f"{len(missing)} arm-A targets missing: {missing[:5]}")
    n_bad = n_allmask = 0
    for i in range(len(cat)):
        iv = h["ivar"][i]
        fl = h["flux"][i]
        if not np.isfinite(fl[iv > 0]).all():
            n_bad += 1
        if (iv > 0).sum() == 0:
            n_allmask += 1
    if n_bad or n_allmask:
        fails.append(f"malformed {n_bad}, all-masked {n_allmask}")
    # 3 grid
    w8 = np.load(GRID)
    if not np.array_equal(w8.astype(np.float32), h["wavelength"][:]):
        fails.append("wavelength grid != committed native grid (f4)")
    # 4 overlap bitwise vs production archive
    pa = h5py.File(PROD_ARC, "r")
    pidx = {int(t): i for i, t in enumerate(pa["catalog"]["TARGETID"])}
    midx = {int(t): i for i, t in enumerate(cat["TARGETID"])}
    overlap = [t for t in tids if t in pidx]
    n_bit = 0
    for t in overlap:
        ok = all(np.array_equal(pa[d][pidx[t]], h[d][midx[t]])
                 for d in ("flux", "ivar", "mask"))
        n_bit += ok
        if not ok:
            fails.append(f"overlap TARGETID {t} NOT bitwise vs -12c archive")
    verdict = "FAIL" if fails else "PASS"
    out = dict(mini_archive=mini, sha256=got, n_targets=len(cat),
               n_missing=len(missing), overlap_targets=len(overlap),
               overlap_bitwise=n_bit, failures=fails, _verdict=verdict)
    json.dump(out, open(mini + ".gl_verify.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
