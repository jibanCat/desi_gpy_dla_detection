#!/usr/bin/env python
"""3-way NHI-bias comparison for the tau_eb high-z falsifier.

Consolidates the per-object MAP log N_HI bias (MAP - truth) for THREE arms on
the same 903 high-z up-migration DLA-host subset, all matched to 2LPT-0 truth:

  off   : ENABLE_TAU_EB=0  (re-inferred)
  null  : V1 production baseline (tau_eb null objective) -- the reference
  dla   : ENABLE_TAU_EB=1, dla objective (re-inferred, stronger forest-floor corr)

Produces:
  1. 3-way per-object NHI-bias table, DLA-tier (truth NHI>=20.3) split by z-bin,
     plus the high-z (z_DLA>3.0) subset, with median/mean/n and the shifts
     d(dla-null) and d(off-null).
  2. Implied R0_z per arm = N(measured>=20.3)/N(truth>=20.3) over matched DLAs in
     each z-bin (the count above the 20.3 line that up-migration inflates), plus
     raw up/down migration counts across 20.3.
  3. A TSV dump of the long-form table.

Matching is identical to ab_compare_nhi_bias.py (nearest-z within dz), so the
arms are directly comparable. The baseline is restricted to the same subset TIDs.
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
NHI_TIER = 20.3
ZBINS = [(2.0, 2.5), (2.5, 3.0), (3.0, 4.0)]
HIGHZ_LO = 3.0


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


def stats_for(arr, zlo=None, zhi=None, nhi_lo=NHI_TIER):
    """median/mean/n of bias for matched DLAs in [zlo,zhi) with truth>=nhi_lo."""
    if len(arr) == 0:
        return dict(n=0, median=np.nan, mean=np.nan, up=0, dn=0,
                    n_truth_tier=0, n_meas_tier=0, r0=np.nan)
    zt, nt, mn = arr[:, 0], arr[:, 1], arr[:, 2]
    bias = mn - nt
    zmask = np.ones(len(arr), dtype=bool)
    if zlo is not None:
        zmask &= (zt >= zlo)
    if zhi is not None:
        zmask &= (zt < zhi)
    tier = zmask & (nt >= nhi_lo)
    n = int(tier.sum())
    median = float(np.median(bias[tier])) if n else np.nan
    mean = float(np.mean(bias[tier])) if n else np.nan
    # migration across 20.3, within the z window (over ALL matched, not just tier)
    up = int(((nt < nhi_lo) & (mn >= nhi_lo) & zmask).sum())
    dn = int(((nt >= nhi_lo) & (mn < nhi_lo) & zmask).sum())
    # implied R0 proxy: measured count >=20.3 over truth count >=20.3 (matched)
    n_truth_tier = int(((nt >= nhi_lo) & zmask).sum())
    n_meas_tier = int(((mn >= nhi_lo) & zmask).sum())
    r0 = (n_meas_tier / n_truth_tier) if n_truth_tier else np.nan
    return dict(n=n, median=median, mean=mean, up=up, dn=dn,
                n_truth_tier=n_truth_tier, n_meas_tier=n_meas_tier, r0=r0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off-dir", required=True)
    ap.add_argument("--dla-dir", required=True)
    ap.add_argument("--base-dir",
                    default="/scratch/cavestru_root/cavestru0/mfho/"
                            "gl_prod_2lpt0_v1_20260526/outputs")
    ap.add_argument("--tid-list",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "highz_upmig_hosts.txt"))
    ap.add_argument("--out-tsv",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "threeway_nhi_bias_table.tsv"))
    args = ap.parse_args()

    truth = load_truth()
    keep = set(np.loadtxt(args.tid_list, dtype=np.int64, ndmin=1).tolist())
    print(f"# subset TIDs: {len(keep)}")

    arms = {
        "off": collect_map(args.off_dir, truth, keep),
        "null": collect_map(args.base_dir, truth, keep),
        "dla": collect_map(args.dla_dir, truth, keep),
    }
    for name, a in arms.items():
        print(f"# matched DLAs [{name}]: {len(a)}")

    # z windows to report: each zbin + the high-z (>3.0) aggregate + ALL-z
    windows = [("all-z", None, None)] + \
              [(f"z[{lo},{hi})", lo, hi) for lo, hi in ZBINS] + \
              [("highz(z>3)", HIGHZ_LO, None)]

    header = ("window", "arm", "n_tier", "median_bias", "mean_bias",
              "d_vs_null_median", "n_truth>=20.3", "n_meas>=20.3", "R0_proxy",
              "upmig", "downmig")
    lines = ["\t".join(header)]

    print("\n" + "=" * 100)
    print("3-WAY DLA-TIER (truth log N_HI >= 20.3) NHI BIAS, by z window")
    print("=" * 100)
    for wname, zlo, zhi in windows:
        s = {nm: stats_for(arms[nm], zlo, zhi) for nm in ("off", "null", "dla")}
        null_med = s["null"]["median"]
        print(f"\n--- {wname} ---")
        print(f"{'arm':>6} {'n':>5} {'median':>9} {'mean':>9} "
              f"{'d(vs null)':>11} {'R0proxy':>9} {'up/dn(20.3)':>14}")
        for nm in ("off", "null", "dla"):
            st = s[nm]
            dmed = (st["median"] - null_med) if (np.isfinite(st["median"])
                                                 and np.isfinite(null_med)) else np.nan
            print(f"{nm:>6} {st['n']:>5} {st['median']:>+9.4f} {st['mean']:>+9.4f} "
                  f"{dmed:>+11.4f} {st['r0']:>9.3f} "
                  f"{st['up']:>6}/{st['dn']:<7}")
            lines.append("\t".join([
                wname, nm, str(st["n"]),
                f"{st['median']:.4f}", f"{st['mean']:.4f}",
                (f"{dmed:.4f}" if np.isfinite(dmed) else "nan"),
                str(st["n_truth_tier"]), str(st["n_meas_tier"]),
                (f"{st['r0']:.4f}" if np.isfinite(st["r0"]) else "nan"),
                str(st["up"]), str(st["dn"]),
            ]))

    with open(args.out_tsv, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n# wrote TSV -> {args.out_tsv}")


if __name__ == "__main__":
    main()
