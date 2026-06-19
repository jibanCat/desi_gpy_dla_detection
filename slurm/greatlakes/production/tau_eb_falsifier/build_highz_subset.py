#!/usr/bin/env python
"""Build the high-z up-migration DLA-host TID subset for the tau_eb falsifier.

Selects 2LPT mock-0 (loa-124) truth DLA hosts in the Eddington up-migration
zone (truth log N_HI in [20.1, 20.6]), at high z_DLA (>3.0) and high z_qso
(>3.1) — the band that drives the DLA-tier R0_z=1.42 over-recovery. Writes the
TARGETIDs (concentrated in the first ~186 sorted spectra-16 files to bound the
per-file model-init count) to highz_upmig_hosts.txt.

Reproducible: re-run to regenerate the list.
"""
import os
from collections import OrderedDict

import fitsio
import numpy as np

MOCK = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/"
        "v2.8.5/mock-0/loa-124")
TARGET_N = 900
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    z = fitsio.read(f"{MOCK}/zcat.fits", ext=1, columns=["TARGETID", "Z"])
    zmap = {int(a): float(b) for a, b in zip(z["TARGETID"], z["Z"])}
    t = fitsio.read(f"{MOCK}/hcd_truth_cat.fits", ext=1)

    hosts = set()
    for nhi, zd, tid in zip(t["NHI"], t["Z"], t["TARGETID"]):
        tid = int(tid)
        if 20.1 <= nhi <= 20.6 and zd > 3.0 and zmap.get(tid, 0.0) > 3.1:
            hosts.add(tid)
    print(f"total candidate high-z up-migration hosts: {len(hosts)}")

    datapath = f"{MOCK}/spectra-16"
    files = []
    for l1 in os.listdir(datapath):
        for l2 in os.listdir(f"{datapath}/{l1}"):
            fp = f"{datapath}/{l1}/{l2}/spectra-16-{l2}.fits"
            if os.path.exists(fp):
                files.append((int(l2), fp))
    files.sort()  # matches the sorted speclist order used by --level2_start/end

    picked = []
    window_end = len(files)
    for idx, (_l2, fp) in enumerate(files):
        h = fitsio.FITS(fp)
        tids = None
        for hdu in h:
            try:
                cn = hdu.get_colnames()
            except Exception:
                continue
            if cn and "TARGETID" in cn:
                tids = hdu.read(columns=["TARGETID"])["TARGETID"]
                break
        h.close()
        picked.extend(int(x) for x in np.unique(tids) if int(x) in hosts)
        if len(picked) >= TARGET_N:
            window_end = idx + 1
            break

    picked = np.unique(np.array(picked, dtype=np.int64))
    out = os.path.join(HERE, "highz_upmig_hosts.txt")
    np.savetxt(out, picked, fmt="%d")
    print(f"wrote {len(picked)} TIDs -> {out}")
    print(f"LAUNCH WINDOW: --start 0 --end {window_end}  ({window_end} files)")


if __name__ == "__main__":
    main()
