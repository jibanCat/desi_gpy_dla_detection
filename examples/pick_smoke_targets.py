"""Pick stratified-by-NHI smoke-test targets from a Saclay/2LPT mock.

Outputs a TSV with columns:
  mock  target_id  z_qso  z_dla_primary  log_nhi_primary  snr  hpx
  spec_path  zcat_path  all_truth_z  all_truth_nhi

`all_truth_z` and `all_truth_nhi` list ALL truth absorbers on the LOS
(DLA + sub-DLA + LLS) so the analysis stage can do per-truth-DLA matching
and the Lyβ misdetection cross-check.

Stratification (default): pick the same number of targets in each
log-NHI bin [20.3, 20.6), [20.6, 21.0), [21.0, 21.5), [21.5, 23.0]
across the requested NHI range, so completeness can be reported per bin.
"""

from __future__ import annotations

import argparse
import os

import fitsio
import healpy as hp
import numpy as np


MOCKS = {
    "saclay": "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124",
    "2lpt":   "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124",
}

NHI_BINS = [(20.3, 20.6), (20.6, 21.0), (21.0, 21.5), (21.5, 23.0)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-per-mock", type=int, default=100,
                   help="targets per mock (split across NHI bins)")
    p.add_argument("--snr-min", type=float, default=1.5,
                   help="minimum SNR_FOREST (default 1.5 — keep low-SNR for completeness study)")
    p.add_argument("--out", default="out/smoke/targets100.tsv")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    n_per_bin = max(1, args.n_per_mock // len(NHI_BINS))

    rows = ["\t".join([
        "mock", "target_id", "z_qso", "z_dla_primary", "log_nhi_primary",
        "snr", "hpx", "spec_path", "zcat_path",
        "all_truth_z", "all_truth_nhi",
    ])]
    for mock_name, root in MOCKS.items():
        zcat = fitsio.read(os.path.join(root, "zcat.fits"), ext=1,
                           columns=["TARGETID","TARGET_RA","TARGET_DEC","Z","ZWARN"])
        snr  = fitsio.read(os.path.join(root, "snr_cat.fits"), ext=1)
        hcd  = fitsio.read(os.path.join(root, "hcd_truth_cat.fits"), ext=1)
        snr_by_tid = dict(zip(snr["TARGETID"].tolist(), snr["SNR_FOREST"].tolist()))

        # Index ALL truth absorbers by TARGETID
        all_by_tid: dict[int, list[tuple[float, float]]] = {}
        for h in hcd:
            all_by_tid.setdefault(int(h["TARGETID"]), []).append(
                (float(h["Z"]), float(h["NHI"]))
            )

        # Index DLA absorbers (NHI ≥ 20.3) by TARGETID
        dla_by_tid: dict[int, list[tuple[float, float]]] = {}
        for tid, lst in all_by_tid.items():
            dlas = [(z, n) for z, n in lst if n >= 20.3]
            if dlas:
                dla_by_tid[tid] = dlas

        # Build candidate pool stratified by primary-DLA NHI bin.
        # "Primary" = the DLA with the highest NHI in the mid-forest range.
        zcat_path = os.path.join(root, "zcat.fits")
        per_bin: dict[tuple[float, float], list] = {b: [] for b in NHI_BINS}
        for r in zcat:
            if r["ZWARN"] != 0: continue
            if r["Z"] < 2.5 or r["Z"] > 3.5: continue
            tid = int(r["TARGETID"])
            s = snr_by_tid.get(tid, 0.0)
            if s < args.snr_min: continue
            dlas = dla_by_tid.get(tid, [])
            mid = [(z, n) for z, n in dlas
                   if r["Z"] - 0.5 <= z <= r["Z"] - 0.05]
            if not mid: continue
            primary_z, primary_n = max(mid, key=lambda zn: zn[1])
            for lo, hi in NHI_BINS:
                if lo <= primary_n < hi:
                    per_bin[(lo, hi)].append((tid, r, primary_z, primary_n, s))
                    break

        # Sample n_per_bin from each bin (without replacement)
        chosen = []
        for binkey, pool in per_bin.items():
            if not pool:
                print(f"[warn] {mock_name} bin {binkey}: no candidates")
                continue
            k = min(n_per_bin, len(pool))
            idx = rng.choice(len(pool), size=k, replace=False)
            chosen.extend(pool[i] for i in idx)
        # Top-up if any bin came short, to reach n_per_mock
        if len(chosen) < args.n_per_mock:
            extras = [c for binkey, pool in per_bin.items()
                      for c in pool if c not in chosen]
            need = args.n_per_mock - len(chosen)
            if extras and need > 0:
                idx = rng.choice(len(extras), size=min(need, len(extras)),
                                 replace=False)
                chosen.extend(extras[i] for i in idx)

        # Materialise rows (verifying spec file exists)
        for (tid, r, primary_z, primary_n, s) in chosen:
            ra, dec = float(r["TARGET_RA"]), float(r["TARGET_DEC"])
            pix = hp.ang2pix(16, np.radians(90 - dec), np.radians(ra), nest=True)
            spec = os.path.join(root, "spectra-16", str(pix//100), str(pix),
                                f"spectra-16-{pix}.fits")
            if not os.path.isfile(spec): continue
            all_los = sorted(all_by_tid.get(tid, []), key=lambda zn: zn[0])
            all_z = ",".join(f"{z:.4f}" for z, _ in all_los)
            all_n = ",".join(f"{n:.3f}" for _, n in all_los)
            rows.append("\t".join([
                mock_name, str(tid), f"{r['Z']:.4f}",
                f"{primary_z:.4f}", f"{primary_n:.3f}",
                f"{s:.3f}", str(pix),
                spec, zcat_path, all_z, all_n,
            ]))
        print(f"[picked] {mock_name}: {len(chosen)} targets across "
              f"{sum(1 for v in per_bin.values() if v)} non-empty bins")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"[saved] {args.out} ({len(rows)-1} targets)")


if __name__ == "__main__":
    main()
