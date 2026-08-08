#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Materialize the frozen TARGETID -> healpix (nside 16 NESTED) map.

EXACTLY the `p1_joint_cov.py` / stability-jackknife convention
(hp.ang2pix(16, colat, ra, nest=True) on the loa-124 zcat), written to a
sidecar npz so that jax-env consumers (`p1_ckm_cov.py`, `p1_refold.py`)
need no healpy import.  Deterministic; run once in an env with healpy
(`gpdla`).  The consumer re-verifies row count and full coverage.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

ZCAT = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/"
        "v2.8.5/mock-0/loa-124/zcat.fits")
OUT = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/"
       "stage0/p1_healpix_map_nside16.npz")
NSIDE = 16


def main():
    import fitsio
    import healpy as hp
    zc = fitsio.read(ZCAT, columns=["TARGETID", "TARGET_RA", "TARGET_DEC"])
    tid = np.asarray(zc["TARGETID"], np.int64)
    hpx = hp.ang2pix(NSIDE,
                     np.radians(90.0 - np.asarray(zc["TARGET_DEC"], float)),
                     np.radians(np.asarray(zc["TARGET_RA"], float)),
                     nest=True).astype(np.int64)
    order = np.argsort(tid)
    np.savez(OUT, schema=np.array("p1_healpix_map/v1"),
             nside=np.array([NSIDE]), nested=np.array([True]),
             zcat=np.array(ZCAT), targetid=tid[order], healpix=hpx[order])
    print(f"wrote {OUT}: {len(tid)} rows, "
          f"{len(np.unique(hpx))} unique healpix")


if __name__ == "__main__":
    main()
