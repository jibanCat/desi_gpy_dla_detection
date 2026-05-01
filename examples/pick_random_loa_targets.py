"""Pick a uniformly-random sample of REAL DESI LOA QSO TIDs (no cherry-picking).

Real LOA data has no truth catalog — we run BASELINE vs ENABLED τ-EB
and compare *aggregate* statistics (detection rate, τ_factor
distribution, p_DLA distribution) to the mock results. There's no
per-spectrum bias to measure here, only "does the recipe behave
sensibly on real data".

Per-spectrum output rows are NOT committable to the public repo
(real LOA spectra are private until release). Only aggregate
statistics may be committed. The TSV this script writes lands at
``--out`` and stays in user scratch.

Spec layout (real LOA on GreatLakes):
    /nfs/turbo/lsa-cavestru/mfho/DESI/loa/healpix/main/dark/
        <healpix//100>/<healpix>/coadd-main-dark-<healpix>.fits

QSO catalog with TARGETID/Z/healpix:
    /nfs/turbo/lsa-cavestru/mfho/DESI/loa/QSO_cat_loa_main_dark_healpix_v3-altbal.fits

Usage::

    python examples/pick_random_loa_targets.py --n 5000 --seed 700 \\
        --out /nfs/turbo/.../phase_b_input/random_loa_5k_z2.tsv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import fitsio
import numpy as np
from astropy.table import Table


LOA_BASE = "/nfs/turbo/lsa-cavestru/mfho/DESI/loa"
DEFAULT_QSO_CAT = os.path.join(LOA_BASE, "QSO_cat_loa_main_dark_healpix_v3-altbal.fits")
SPEC_DIR = os.path.join(LOA_BASE, "healpix/main/dark")


def _spec_path(healpix: int) -> str:
    return os.path.join(SPEC_DIR, str(healpix // 100), str(healpix),
                        f"coadd-main-dark-{healpix}.fits")


def _healpix_from_radec(ra_deg: np.ndarray, dec_deg: np.ndarray, nside: int = 64) -> np.ndarray:
    import healpy
    theta = np.radians(90.0 - dec_deg)
    phi = np.radians(ra_deg)
    return healpy.ang2pix(nside, theta, phi, nest=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=700)
    p.add_argument("--out", required=True)
    p.add_argument("--qso-catalog", default=DEFAULT_QSO_CAT)
    p.add_argument("--z-qso-min", type=float, default=2.0)
    p.add_argument("--z-qso-max", type=float, default=5.5,
                   help="Upper z bound (production typically caps at ~5).")
    p.add_argument("--available-only", action="store_true",
                   help="Pre-filter to QSOs whose healpix has a coadd file "
                        "on this filesystem (speeds up partial GL mirror).")
    args = p.parse_args()

    print(f"[loa] reading {args.qso_catalog}")
    cat = Table.read(args.qso_catalog)
    n_before = len(cat)
    # Filter to QSO spectype + ZWARN==0 + z range, like production runs.
    cat = cat[
        (cat["SPECTYPE"] == "QSO")
        & (cat["ZWARN"] == 0)
        & (cat["Z"] >= args.z_qso_min)
        & (cat["Z"] <= args.z_qso_max)
    ]
    print(f"  {len(cat)} QSOs (filtered SPECTYPE=QSO, ZWARN=0, "
          f"{args.z_qso_min} ≤ z ≤ {args.z_qso_max} from {n_before})")
    # Real DESI LOA uses HEALPIX-64 (nest), not 16 like mocks.
    # The catalog already has HEALPIX columns in some versions; if
    # missing, recompute from RA/DEC.
    def _hpx_for_table(t):
        if "HEALPIX" in t.colnames:
            return np.asarray(t["HEALPIX"])
        return _healpix_from_radec(np.asarray(t["TARGET_RA"]),
                                   np.asarray(t["TARGET_DEC"]), nside=64)

    if args.available_only:
        # Determine which healpixes have a coadd file locally; restrict
        # the catalog to those before sampling.
        import re
        from glob import glob
        avail = set()
        for f in glob(os.path.join(SPEC_DIR, "*", "*", "coadd-main-dark-*.fits")):
            m = re.search(r"coadd-main-dark-(\d+)\.fits$", f)
            if m:
                avail.add(int(m.group(1)))
        print(f"[loa] {len(avail)} healpixes have a coadd file on this filesystem")
        cat_hpx = _hpx_for_table(cat)
        keep = np.fromiter((int(h) in avail for h in cat_hpx),
                           dtype=bool, count=len(cat_hpx))
        cat = cat[keep]
        print(f"  → {len(cat)} QSOs after filtering to available healpixes")
        if len(cat) < args.n:
            print(f"  WARNING: requested n={args.n} but only {len(cat)} eligible; "
                  f"will pick all available")
            args.n = len(cat)

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(cat), size=args.n, replace=False)
    sample = cat[idx]
    healpix = _hpx_for_table(sample)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_missing = 0
    with out.open("w") as f:
        f.write("target_id\tz_qso\ttruth_z\ttruth_log_nhi\tnhi_regime\t"
                "healpix\tspec_path\tzcat_path\tn_truth_absorbers\n")
        for i in range(len(sample)):
            tid = int(sample["TARGETID"][i])
            z_qso = float(sample["Z"][i])
            hpx = int(healpix[i])
            spec = _spec_path(hpx)
            if not os.path.exists(spec):
                n_missing += 1
                continue
            # No truth catalog for real LOA; emit -1 placeholders so the
            # downstream Phase B runner treats them as "no-truth" rows.
            f.write(f"{tid}\t{z_qso}\t-1.0\t-1.0\tunknown\t"
                    f"{hpx}\t{spec}\t{args.qso_catalog}\t0\n")
    n_kept = args.n - n_missing
    print(f"[out] wrote {n_kept}/{args.n} rows → {out}  "
          f"({n_missing} dropped: spec file not on this filesystem)")


if __name__ == "__main__":
    main()
