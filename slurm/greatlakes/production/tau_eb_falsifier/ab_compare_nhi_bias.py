#!/usr/bin/env python
"""A/B NHI-bias comparison for the tau_eb high-z falsifier.

Compares per-object MAP log N_HI bias (MAP - truth) between a re-inferred run
(tau_eb arm) and the V1 baseline (tau_eb null), both matched to 2LPT-0 truth,
split by z. Answers: does the tau_eb arm shrink the +0.038 dex DLA-tier bias,
and more at high z? Also recomputes the up-migration fraction across 20.3 and
the implied R0_z.

Usage:
  python ab_compare_nhi_bias.py \
      --arm-dir   /scratch/.../gl_taueb_falsifier_2lpt0_highz_OFF_<date>/outputs \
      --base-dir  /scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/outputs \
      --tid-list  highz_upmig_hosts.txt

Both dirs must contain figures/processed/processed-spectra-16-*.h5.
"""
import argparse
import glob
import os
from collections import defaultdict

import fitsio
import h5py
import numpy as np

MOCK = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/"
        "v2.8.5/mock-0/loa-124")


def load_truth():
    t = fitsio.read(f"{MOCK}/hcd_truth_cat.fits", ext=1)
    truth = defaultdict(list)
    for nhi, z, tid in zip(t["NHI"], t["Z"], t["TARGETID"]):
        truth[int(tid)].append((float(z), float(nhi)))
    return truth


def collect_map(proc_dir, truth, tid_keep=None, dz=0.05):
    """Return array of (z_truth, nhi_truth, map_nhi, tid) for matched DLAs."""
    rows = []
    files = glob.glob(os.path.join(proc_dir, "figures", "processed",
                                   "processed-spectra-16-*.h5"))
    for fp in files:
        with h5py.File(fp) as f:
            tids = f["target_ids"][:]
            mapn = f["MAP_log_nhis"][:]
            mapz = f["MAP_z_dlas"][:]
        for i, tid in enumerate(tids):
            tid = int(tid)
            if tid not in truth:
                continue
            if tid_keep is not None and tid not in tid_keep:
                continue
            for k in range(mapn.shape[1]):
                mn, mz = mapn[i, k], mapz[i, k]
                if not np.isfinite(mn) or mn <= 0:
                    continue
                cand = truth[tid]
                best = min(cand, key=lambda c: abs(c[0] - mz))
                if abs(best[0] - mz) < dz:
                    rows.append((best[0], best[1], float(mn), tid))
    return np.array(rows) if rows else np.empty((0, 4))


def summarize(name, arr, nhi_lo=20.3):
    if len(arr) == 0:
        print(f"  [{name}] no matched DLAs")
        return
    zt, nt, mn = arr[:, 0], arr[:, 1], arr[:, 2]
    bias = mn - nt
    m = nt >= nhi_lo
    print(f"  [{name}] NHI>={nhi_lo}: n={m.sum()}  "
          f"median bias={np.median(bias[m]):+.4f}  mean={np.mean(bias[m]):+.4f} dex")
    for zlo, zhi in [(2.0, 2.5), (2.5, 3.0), (3.0, 4.0)]:
        zm = m & (zt >= zlo) & (zt < zhi)
        if zm.sum() > 3:
            # up-migration: fraction of truth<20.3 measured>=20.3 (and vice versa)
            up = ((nt < 20.3) & (mn >= 20.3) & (zt >= zlo) & (zt < zhi)).sum()
            dn = ((nt >= 20.3) & (mn < 20.3) & (zt >= zlo) & (zt < zhi)).sum()
            print(f"     z[{zlo},{zhi}): n={zm.sum():4d}  "
                  f"median bias={np.median(bias[zm]):+.4f}  "
                  f"up-mig(across 20.3)={up} down={dn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", required=True, help="tau_eb arm outputs/ dir")
    ap.add_argument("--base-dir",
                    default="/scratch/cavestru_root/cavestru0/mfho/"
                            "gl_prod_2lpt0_v1_20260526/outputs",
                    help="V1 baseline (tau_eb null) outputs/ dir")
    ap.add_argument("--tid-list", default=None,
                    help="restrict comparison to these TIDs (the falsifier subset)")
    args = ap.parse_args()

    truth = load_truth()
    keep = None
    if args.tid_list:
        keep = set(np.loadtxt(args.tid_list, dtype=np.int64, ndmin=1).tolist())
        print(f"restricting to {len(keep)} subset TIDs")

    arm = collect_map(args.arm_dir, truth, keep)
    base = collect_map(args.base_dir, truth, keep)
    print(f"\nmatched DLAs: arm={len(arm)}  base={len(base)}")
    print("\n=== BASELINE (tau_eb null) ===")
    summarize("base", base)
    print("\n=== ARM (re-inferred) ===")
    summarize("arm", arm)
    print("\nVerdict guide: if arm median bias is CLOSER to 0 than base "
          "(esp. at z>3.0) => tau_eb shrinks the forest-blend bias (lever works). "
          "If ~unchanged => residual bias is prior-edge pileup (kernel is the lever).")


if __name__ == "__main__":
    main()
