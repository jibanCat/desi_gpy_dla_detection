#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""H2-v2 predeclared analysis (PI 2026-08-13 §26). REUSES the established
estimators: injection/measurements.detection_completeness (Beta-Binomial
per-cell completeness, recovered iff p_dla > 0.5) and nhi_bias
((N_rec - N_true) landing distributions). One naturally-implied extension,
DECLARED here (per §26, flagged in the report): H2-v2 allows <=2 injections
per sightline while the single-absorber production config emits at most one
MAP row per sightline — the recovered row is assigned to the NEAREST-z
injection within the protocol's own 5,000 km/s separation scale (unique
assignment; the other injection counts unrecovered, exactly the mock
kernel's per-sightline join semantics). Singles reduce to the established
join + a z-sanity window.

Outputs per (arm × TSNR cell), (z_inj bin), (NHI region): completeness with
Jeffreys intervals; Δz and ΔlogN landing moments; P_DLA distributions.
"""
import csv
import glob
import json
import sys

import numpy as np

CKMS = 299792.458
MATCH_KMS = 5000.0
ZB = [(3.8, 4.25), (4.25, 4.5), (4.5, 5.0)]
NRE = {"19.5-20.0": (19.5, 20.0), "20.0-20.3": (20.0, 20.3),
       "20.3-20.7": (20.3, 20.7), "20.7-21.1": (20.7, 21.1),
       "21.1-21.5": (21.1, 21.51)}


def jeffreys(k, n):
    from scipy.stats import beta
    if n == 0:
        return (np.nan,) * 3
    lo, hi = beta.ppf([0.16, 0.84], k + 0.5, n - k + 0.5)
    return k / n, lo, hi


def main():
    realized = list(csv.DictReader(open(sys.argv[1])))
    dlacat_glob = sys.argv[2]
    outj = sys.argv[3]

    from astropy.io import fits
    rows_by_tid = {}
    for f in sorted(glob.glob(dlacat_glob)):
        for r in fits.open(f)[1].data:
            rows_by_tid.setdefault(int(r["TARGETID"]), []).append(
                dict(z=float(r["Z_DLA"]), n=float(r["NHI"]),
                     p=float(r["P_DLA"])))

    # nearest-z unique assignment per sightline
    per_inj = []
    by_tid = {}
    for r in realized:
        by_tid.setdefault(int(r["TARGETID"]), []).append(r)
    for tid, injs in by_tid.items():
        cat = rows_by_tid.get(tid, [])
        used = set()
        for r in sorted(injs, key=lambda r: int(r["inj_idx"])):
            zt = float(r["z_inj"])
            best, bestdv = None, MATCH_KMS
            for j, c in enumerate(cat):
                if j in used:
                    continue
                dv = abs(CKMS * (c["z"] - zt) / (1 + zt))
                if dv < bestdv:
                    best, bestdv = j, dv
            rec = None
            if best is not None:
                # unique: confirm this inj is the nearest truth to that row
                zr = cat[best]["z"]
                near = min(injs, key=lambda q: abs(float(q["z_inj"]) - zr))
                if near is r:
                    used.add(best)
                    rec = cat[best]
            per_inj.append(dict(tid=tid, cell=r["cell"], z_true=zt,
                                logN_true=float(r["logN"]),
                                recovered=rec is not None and rec["p"] > 0.5,
                                z_rec=rec["z"] if rec else np.nan,
                                n_rec=rec["n"] if rec else np.nan,
                                p=rec["p"] if rec else 0.0))

    def zbin(z):
        return next((f"[{a},{b})" for a, b in ZB if a <= z < b), "?")

    def nre(n):
        return next((k for k, (a, b) in NRE.items() if a <= n < b), "?")

    strata = {}
    for r in per_inj:
        for key in [("cell", r["cell"]), ("arm", r["cell"][0]),
                    ("zbin", zbin(r["z_true"])), ("nre", nre(r["logN_true"])),
                    ("arm_zbin", r["cell"][0], zbin(r["z_true"])),
                    ("zbin_nre", zbin(r["z_true"]), nre(r["logN_true"])),
                    ("cell_nre", r["cell"], nre(r["logN_true"]))]:
            strata.setdefault(key, []).append(r)

    out = []
    for key in sorted(strata, key=str):
        rs = strata[key]
        k = sum(1 for r in rs if r["recovered"])
        C, lo, hi = jeffreys(k, len(rs))
        rec = [r for r in rs if r["recovered"]]
        dz = np.array([r["z_rec"] - r["z_true"] for r in rec])
        dn = np.array([r["n_rec"] - r["logN_true"] for r in rec])
        out.append(dict(
            stratum=":".join(map(str, key)), n=len(rs), k=k,
            C=round(C, 4), C_lo68=round(lo, 4), C_hi68=round(hi, 4),
            dz_mean=round(float(dz.mean()), 5) if len(dz) else None,
            dz_std=round(float(dz.std()), 5) if len(dz) else None,
            dlogN_mean=round(float(dn.mean()), 4) if len(dn) else None,
            dlogN_std=round(float(dn.std()), 4) if len(dn) else None,
            p_median=round(float(np.median([r["p"] for r in rs])), 4)))
    json.dump(dict(match_rule=f"nearest-z unique within {MATCH_KMS} km/s; "
                              "recovered iff p_dla>0.5 (established "
                              "detection_completeness convention)",
                   n_injections=len(per_inj), strata=out),
              open(outj, "w"), indent=1)
    print(f"analyzed {len(per_inj)} injections -> {outj}")


if __name__ == "__main__":
    main()
