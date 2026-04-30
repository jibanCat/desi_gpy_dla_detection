"""Pick a uniformly-random sample of 2LPT QSO TIDs (no cherry-picking).

Difference from ``pick_voigt_sweep_targets.py``: no SNR filter, no
single-truth-absorber filter, no z-of-DLA-in-window filter. Just a
deterministic random sample from the zcat.

For each picked TID, look up the truth absorbers (zero, one, or many)
and emit one row per (TID, truth_absorber) pair so downstream analysis
can compare detected to nearest-truth easily.

Usage::

    python examples/pick_random_2lpt_targets.py \\
        --n 5000 --seed 100 --out /tmp/random_2lpt_5k.tsv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import fitsio
import numpy as np
from astropy.table import Table


MOCK_DIR = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124"


def _spec_path_from_healpix(healpix: int) -> str:
    return os.path.join(MOCK_DIR, "spectra-16",
                        str(healpix // 100), str(healpix),
                        f"spectra-16-{healpix}.fits")


def _healpix_from_radec(ra_deg: np.ndarray, dec_deg: np.ndarray, nside: int = 16) -> np.ndarray:
    import healpy
    theta = np.radians(90.0 - dec_deg)
    phi = np.radians(ra_deg)
    return healpy.ang2pix(nside, theta, phi, nest=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--out", required=True)
    p.add_argument("--mock-dir", default=MOCK_DIR)
    args = p.parse_args()

    print(f"[mock_dir] {args.mock_dir}")
    zcat = Table.read(os.path.join(args.mock_dir, "zcat.fits"))
    truth = Table.read(os.path.join(args.mock_dir, "hcd_truth_cat.fits"))
    print(f"[zcat] {len(zcat)} QSOs")
    print(f"[truth] {len(truth)} absorbers (any NHI), "
          f"{(truth['NHI'] >= 20.3).sum()} DLA-strength")

    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(zcat), size=args.n, replace=False)
    sample = zcat[sample_idx]
    print(f"[sample] picked {len(sample)} TIDs at random (seed={args.seed})")

    # healpix-16 from RA/DEC (nest)
    ra_col = "TARGET_RA" if "TARGET_RA" in zcat.colnames else "RA"
    dec_col = "TARGET_DEC" if "TARGET_DEC" in zcat.colnames else "DEC"
    healpix = _healpix_from_radec(np.asarray(sample[ra_col]),
                                  np.asarray(sample[dec_col]))

    truth_by_tid = {}
    for row in truth:
        truth_by_tid.setdefault(int(row["TARGETID"]), []).append(
            (float(row["Z"]), float(row["NHI"])))

    # Output: one row per TID, listing the strongest truth absorber.
    rows = []
    n_with_dla = 0
    n_with_subdla = 0
    n_with_lls = 0
    n_no_truth = 0
    for i in range(len(sample)):
        tid = int(sample["TARGETID"][i])
        z_qso = float(sample["Z"][i])
        hpx = int(healpix[i])
        spec = _spec_path_from_healpix(hpx)
        zcat_path = os.path.join(args.mock_dir, "zcat.fits")
        truths = truth_by_tid.get(tid, [])
        if truths:
            # Strongest absorber on this LOS
            zb, nb = max(truths, key=lambda zn: zn[1])
            if nb >= 20.3:
                regime = "DLA"; n_with_dla += 1
            elif nb >= 19.0:
                regime = "sub-DLA"; n_with_subdla += 1
            else:
                regime = "LLS"; n_with_lls += 1
        else:
            zb, nb, regime = -1.0, -1.0, "none"
            n_no_truth += 1
        rows.append((tid, z_qso, zb, nb, regime, hpx, spec, zcat_path,
                     len(truths)))

    print(f"[mix] DLA={n_with_dla}  sub-DLA={n_with_subdla}  "
          f"LLS={n_with_lls}  no-truth-absorber={n_no_truth}")

    out = Path(args.out)
    with out.open("w") as f:
        f.write("target_id\tz_qso\ttruth_z\ttruth_log_nhi\tnhi_regime\t"
                "healpix\tspec_path\tzcat_path\tn_truth_absorbers\n")
        for row in rows:
            f.write("\t".join(str(c) for c in row) + "\n")
    print(f"[out] wrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
