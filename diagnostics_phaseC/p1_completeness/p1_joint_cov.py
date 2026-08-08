#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FROZEN joint (C, K) calibration-covariance construction (PI §9).

C_molly and K_natural-pairs share the same natural systems, so their
calibration errors must not be assumed independent.

FROZEN design (stated before the first run):
  * common resampling unit = WHOLE HEALPIX (nside 16 nested — the same
    unit as the stability jackknife and the holdout blocking);
  * estimator vector θ per frozen bin b (the 7 battery bins, live
    support): C_b = live matched/eligible (production-above-floor
    convention) and Kμ_b = live kernel mean dx;
  * joint delete-one-healpix jackknife over all contributing healpix;
    Cov = (g−1)/g · Σ_h (θ_−h − θ̄)(θ_−h − θ̄)^T on the same
    realizations for C and K;
  * reported: per-bin corr(C_b, Kμ_b), σ's, effective block count
    (ESS = (Σn_h)²/Σn_h²), largest-block share, stability
    (jackknife-vs-naive SE ratios), and an initial materiality read
    (|corr| > 0.3 with both σ's non-negligible ⇒ material).
Schema `p1_ck_cov/v1`. The full G1/G2/G3 covariance propagation is a
refold-stage deliverable, not this one. No holdout rows.
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "injection"))

from build_p1_natpair_ck import extract_kernel_events   # noqa: E402

ZCAT = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/"
        "v2.8.5/mock-0/loa-124/zcat.fits")
NSIDE = 16
BINS = [(19.5, 20.0), (20.0, 20.4), (20.4, 20.7), (20.7, 21.0),
        (21.0, 21.3), (21.3, 21.7), (21.7, 22.4)]
CORR_MATERIAL = 0.3


def main():
    t0 = time.time()
    import fitsio
    import healpy as hp
    zc = fitsio.read(ZCAT, columns=["TARGETID", "TARGET_RA", "TARGET_DEC"])
    tid = np.asarray(zc["TARGETID"], np.int64)
    hpx = hp.ang2pix(NSIDE, np.radians(90.0 - np.asarray(zc["TARGET_DEC"],
                                                        float)),
                     np.radians(np.asarray(zc["TARGET_RA"], float)),
                     nest=True)
    t2h = dict(zip(tid.tolist(), hpx.tolist()))

    ev, d = extract_kernel_events()
    kin = ev["IN_KERNEL"] & (ev["S2N"] > 2.0)
    kN, kDX = ev["N"][kin], ev["DX"][kin]
    kH = np.asarray([t2h.get(int(t), -1) for t in ev["TID"][kin]])
    t_live = d["tr_S2N"] > 2.0
    tN = d["tr_NHI"][t_live]
    tH = np.asarray([t2h.get(int(t), -1) for t in
                     d["tr_TARGETID"][t_live]])
    if np.any(kH < 0) or np.any(tH < 0):
        raise SystemExit("FATAL: unmapped healpix "
                         f"(kernel {int(np.sum(kH < 0))}, "
                         f"truth {int(np.sum(tH < 0))})")

    out = {"schema": "p1_ck_cov/v1", "date": time.strftime("%Y-%m-%d"),
           "resampling_unit": f"whole healpix (nside {NSIDE} nested)",
           "bins": []}
    for lo, hi in BINS:
        km = (kN >= lo) & (kN < hi)
        tm = (tN >= lo) & (tN < hi)
        hs = np.unique(np.concatenate([kH[km], tH[tm]]))
        g = len(hs)
        ksum = {int(h): float(kDX[km][kH[km] == h].sum()) for h in hs}
        kcnt = {int(h): int(np.sum(kH[km] == h)) for h in hs}
        tcnt = {int(h): int(np.sum(tH[tm] == h)) for h in hs}
        KS, KC = float(kDX[km].sum()), int(km.sum())
        TC = int(tm.sum())
        theta = np.empty((g, 2))
        for i, h in enumerate(hs):
            h = int(h)
            kc = KC - kcnt.get(h, 0)
            tc = TC - tcnt.get(h, 0)
            theta[i, 0] = (kc / tc) if tc else np.nan       # C_-h
            theta[i, 1] = ((KS - ksum.get(h, 0.0)) / kc) if kc else np.nan
        ok = np.all(np.isfinite(theta), axis=1)
        theta = theta[ok]
        g = len(theta)
        tb = theta.mean(axis=0)
        dev = theta - tb
        cov = (g - 1) / g * dev.T @ dev
        sC, sK = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
        corr = float(cov[0, 1] / (sC * sK)) if sC > 0 and sK > 0 else np.nan
        cnts = np.array([tcnt.get(int(h), 0) for h in hs], float)
        ess = float(cnts.sum() ** 2 / np.sum(cnts ** 2)) if cnts.sum() else 0
        naive_seK = float(kDX[km].std(ddof=1) / np.sqrt(max(KC, 2)))
        C = KC / TC
        naive_seC = float(np.sqrt(C * (1 - C) / TC))
        out["bins"].append({
            "N": [lo, hi], "n_pairs": KC, "n_truth": TC,
            "n_blocks": g, "ess_blocks": ess,
            "max_block_truth_share": float(cnts.max() / cnts.sum()),
            "C": C, "K_mean": float(kDX[km].mean()),
            "sigma_C_jk": float(sC), "sigma_C_naive": naive_seC,
            "sigma_K_jk": float(sK), "sigma_K_naive": naive_seK,
            "corr_CK": corr,
            "material": bool(abs(corr) > CORR_MATERIAL
                             and sC > 0.5 * naive_seC
                             and sK > 0.5 * naive_seK)})
        r = out["bins"][-1]
        print(f"[{lo},{hi}) C={C:.4f}±{sC:.4f} Kμ={r['K_mean']:+.4f}"
              f"±{sK:.4f} corr={corr:+.3f} blocks={g} ess={ess:.0f} "
              f"maxshare={r['max_block_truth_share']:.4f} "
              f"seK jk/naive={sK/naive_seK:.2f} "
              f"seC jk/naive={sC/naive_seC:.2f} "
              f"{'MATERIAL' if r['material'] else ''}")
    out["initial_materiality"] = {
        "rule": f"|corr|>{CORR_MATERIAL} with both sigmas non-negligible",
        "any_material": bool(any(b["material"] for b in out["bins"]))}
    out["wall_s"] = round(time.time() - t0, 1)
    with open(os.path.join(_HERE, "p1_joint_cov.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("any_material:", out["initial_materiality"]["any_material"])


if __name__ == "__main__":
    main()
