#!/usr/bin/env python
"""r041_population.py — the eligible sightline population of the R-041 high-z injection
campaigns = EXACTLY the population that supports the high-z measurement (R-039 closure):
quasars in the high-z spectra archive with finite SNR_REDSIDE > 2, 4.25 < z_QSO < 7.0, not
BAL (BI_CIV > 0), and a non-empty Lyα window under the measurement's geometry (λ_rf
1025–1216 Å, 3600 Å floor, constant-Δz collar 3000 km/s on both edges), restricted to the
sightlines whose window intersects the reported bin [3.8, 5.0).

Per sightline it records: z_QSO, SNR_REDSIDE, the window [zlo, zhi], the absorption path
inside [3.8, 5.0) (Ω_m = 0.279), the SNR stratum (the campaign strata, chosen from the
population's own quantiles on the molly grid: [2,3), [3,4), [4,5), [5,7), [7,inf)), and the
existing production candidates (P_DLA > 0.5, DLAFLAG 0) with their z and N-hat — used by the
collision rule and by the `has_candidate` stratification.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

LYA = 1215.67
C_KMS = 299792.458
COLLAR_KMS = 3000.0
LAM_RF = (1025.0, 1216.0)
ZQSO = (4.25, 7.0)
ZBIN = (3.8, 5.0)
SNR_STRATA = [2.0, 3.0, 4.0, 5.0, 7.0, np.inf]
OM = 0.279


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz-cat", required=True, help="production dlacat directory (dlacat-*.fits)")
    ap.add_argument("--mockdir", required=True, help="the snr_cat/zcat/bal_cat of the high-z sample")
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import fitsio, h5py
    from CDDF_analysis.hbi.cddf_catalog_hbi import build_per_qso_snr
    from CDDF_analysis.cddf_mock import AbsorptionDistance
    lk = build_per_qso_snr(a.hz_cat, snr_cat_path=None, zcat_path=None, mockdir=a.mockdir, restrict_to_processed=False)
    bal = set(int(t) for t in fitsio.read(os.path.join(a.mockdir, "bal_cat.fits"), ext=1, columns=["TARGETID"])["TARGETID"])
    with h5py.File(a.archive, "r") as h:
        in_archive = set(int(t) for t in h["catalog"]["TARGETID"][:])
    cat = np.concatenate([fitsio.read(f, ext=1) for f in sorted(glob.glob(os.path.join(a.hz_cat, "dlacat-*.fits")))])
    cand = cat[(cat["P_DLA"] > 0.5) & (cat["DLAFLAG"] == 0)]
    cand_by_tid = {}
    for r in cand:
        cand_by_tid.setdefault(int(r["TARGETID"]), []).append((float(r["Z_DLA"]), float(r["NHI"]), float(r["P_DLA"])))
    coll = COLLAR_KMS / C_KMS
    X = AbsorptionDistance(zmax=7.1, Omega_m=OM)
    rows = []
    for t, (snr, zq) in lk.items():
        t = int(t)
        if not (np.isfinite(snr) and snr > 2.0 and ZQSO[0] < zq < ZQSO[1]) or t in bal or t not in in_archive:
            continue
        zlo = max(3600.0 / LYA - 1.0, LAM_RF[0] * (1 + zq) / LYA - 1.0 + coll)
        zhi = min(zq - coll, LAM_RF[1] * (1 + zq) / LYA - 1.0 - coll)
        lo, hi = max(zlo, ZBIN[0]), min(zhi, ZBIN[1])
        if not (hi > lo):
            continue
        dX = float(X.X(hi) - X.X(lo))
        stratum = int(np.searchsorted(SNR_STRATA, snr, side="right") - 1)
        cl = cand_by_tid.get(t, [])
        rows.append(dict(TARGETID=t, z_qso=round(float(zq), 6), snr=round(float(snr), 4), zlo=round(zlo, 6), zhi=round(zhi, 6),
                         zlo_bin=round(lo, 6), zhi_bin=round(hi, 6), dX_bin=round(dX, 6), stratum=stratum,
                         n_cand=len(cl), has_cand_ge20=int(any(n >= 20.0 for _, n, _ in cl)),
                         cand=";".join(f"{z:.5f}:{n:.3f}:{p:.3f}" for z, n, p in cl)))
    rows.sort(key=lambda r: r["TARGETID"])
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    strata = np.array([r["stratum"] for r in rows]); dX = np.array([r["dX_bin"] for r in rows])
    summ = dict(n_sightlines=len(rows), dX_bin_total=float(dX.sum()), snr_strata_edges=[float(x) if np.isfinite(x) else "inf" for x in SNR_STRATA],
                n_per_stratum=[int((strata == i).sum()) for i in range(len(SNR_STRATA) - 1)],
                dX_per_stratum=[float(dX[strata == i].sum()) for i in range(len(SNR_STRATA) - 1)],
                n_with_candidate_ge20=int(sum(r["has_cand_ge20"] for r in rows)),
                geometry=dict(lam_rf=LAM_RF, collar_kms=COLLAR_KMS, floor_A=3600.0, z_qso=ZQSO, z_bin=ZBIN, Omega_m=OM),
                inputs=dict(hz_cat=a.hz_cat, mockdir=a.mockdir, archive=dict(path=a.archive, sha256=_sha(a.archive))),
                out=dict(path=a.out, sha256=_sha(a.out)))
    with open(a.out + ".summary.json", "w") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("inputs", "geometry")}, indent=1))


if __name__ == "__main__":
    main()
