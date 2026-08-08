#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FROZEN healpix-jackknife stability gate for the natural-pair kernel.

GATE (frozen here, BEFORE the first run; spec §11, PI condition 6):
for every load-bearing reporting bin (the frozen Tier-2 bins over
[20.4, 21.7), live support, kernel-event population):
  (a) leave-one-healpix-out jackknife SE ≤ 1.5 × naive SE
      (bounds hidden spatial clustering of the kernel mean);
  (b) no single-healpix deletion moves the bin mean by more than
      3 × naive SE (bounds single-region domination).
Outer bins ([19.5,20.4), [21.7,22.5)) are reported with the same
machinery but do not gate. PASS requires (a) and (b) in all four
load-bearing bins.

Healpix: nside=16 NESTED from catalogue RA/DEC (the quickquasars
convention); the mapping is VALIDATED against the injection manifest's
own healpix column on every shared TARGETID before use — any
disagreement aborts (wrong-convention guard).

Deterministic, catalog-level; no holdout rows (natural production
catalogue only, which predates all injection arms).
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

CAT_FITS = ("/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/"
            "combined_catalog/dlacat-v2.8.5-mockcat.fits")
MANIFEST = ("/scratch/cavestru_root/cavestru0/mfho/phaseC_resp/prod_v1/"
            "injection_truth.fits")
NSIDE = 16
GATE_BINS = [(20.4, 20.7), (20.7, 21.0), (21.0, 21.3), (21.3, 21.7)]
REPORT_BINS = [(19.5, 20.0), (20.0, 20.4)] + GATE_BINS + [(21.7, 22.5)]
SE_RATIO_MAX = 1.5
SINGLE_SHIFT_MAX_SE = 3.0


def main():
    t0 = time.time()
    import fitsio
    import healpy as hp
    cat = fitsio.read(CAT_FITS, columns=["TARGETID", "RA", "DEC"])
    tid = np.asarray(cat["TARGETID"], np.int64)
    _, ui = np.unique(tid, return_index=True)
    ra, dec = np.asarray(cat["RA"], float)[ui], np.asarray(cat["DEC"],
                                                          float)[ui]
    hpx = hp.ang2pix(NSIDE, np.radians(90.0 - dec), np.radians(ra),
                     nest=True)
    t2h = dict(zip(tid[ui].tolist(), hpx.tolist()))

    man = fitsio.read(MANIFEST, columns=["target_id", "healpix"])
    n_chk = n_bad = 0
    for T, H in zip(np.asarray(man["target_id"], np.int64),
                    np.asarray(man["healpix"], np.int64)):
        if int(T) in t2h:
            n_chk += 1
            if t2h[int(T)] != int(H):
                n_bad += 1
    if n_chk == 0 or n_bad:
        raise SystemExit(f"FATAL healpix-convention check: {n_bad}/{n_chk} "
                         f"mismatches vs the manifest")
    print(f"healpix convention validated on {n_chk} shared TARGETIDs")

    ev, _ = extract_kernel_events()
    kin = ev["IN_KERNEL"] & (ev["S2N"] > 2.0)
    N, DX = ev["N"][kin], ev["DX"][kin]
    H = np.asarray([t2h.get(int(t), -1) for t in ev["TID"][kin]])
    n_unmapped = int(np.sum(H < 0))
    if n_unmapped:
        raise SystemExit(f"FATAL: {n_unmapped} kernel events lack healpix")

    out = {"schema": "p1_jackknife/v1", "date": time.strftime("%Y-%m-%d"),
           "nside": NSIDE, "n_healpix": int(len(set(H.tolist()))),
           "gate": {"se_ratio_max": SE_RATIO_MAX,
                    "single_shift_max_se": SINGLE_SHIFT_MAX_SE},
           "bins": [], "GATE_PASS": None}
    gate_ok = True
    for lo, hi in REPORT_BINS:
        m = (N >= lo) & (N < hi)
        v, h = DX[m], H[m]
        n = len(v)
        mean = float(v.mean())
        se = float(v.std(ddof=1) / np.sqrt(n))
        hs = np.unique(h)
        S, C = float(v.sum()), n
        # leave-one-healpix-out means via group sums
        sums = {int(k): float(v[h == k].sum()) for k in hs}
        cnts = {int(k): int(np.sum(h == k)) for k in hs}
        loo = np.array([(S - sums[int(k)]) / (C - cnts[int(k)])
                        for k in hs if C - cnts[int(k)] > 0])
        g = len(loo)
        jk_se = float(np.sqrt((g - 1) / g * np.sum((loo - loo.mean()) ** 2)))
        max_shift = float(np.max(np.abs(loo - mean)))
        row = {"N": [lo, hi], "n": n, "n_healpix": int(len(hs)),
               "mean": mean, "naive_se": se, "jk_se": jk_se,
               "se_ratio": jk_se / se, "max_single_shift": max_shift,
               "max_shift_in_se": max_shift / se,
               "gating": [lo, hi] in [list(b) for b in GATE_BINS]}
        row["pass_a"] = bool(jk_se <= SE_RATIO_MAX * se)
        row["pass_b"] = bool(max_shift <= SINGLE_SHIFT_MAX_SE * se)
        if row["gating"] and not (row["pass_a"] and row["pass_b"]):
            gate_ok = False
        out["bins"].append(row)
        print(f"[{lo},{hi}) n={n} hpx={len(hs)} mean {mean:+.4f} "
              f"se {se:.4f} jk_se {jk_se:.4f} (x{jk_se/se:.2f}) "
              f"max_shift {max_shift:.4f} ({max_shift/se:.1f} se) "
              f"{'GATE' if row['gating'] else 'report'} "
              f"{'PASS' if row['pass_a'] and row['pass_b'] else 'FAIL'}")
    out["GATE_PASS"] = bool(gate_ok)
    out["wall_s"] = round(time.time() - t0, 1)
    with open(os.path.join(_HERE, "p1_jackknife.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("GATE:", "PASS" if gate_ok else "FAIL")


if __name__ == "__main__":
    main()
