#!/usr/bin/env python
"""Build the per-HPX-index spec-count table consumed by loa_balance_boundaries.py.

Emits one line ``"<healpix_id> <count>"`` per healpix, sorted by ascending
healpix id -- i.e. the SAME index order as ``np.unique(catalog["HPXPIXEL"])`` in
``desi-DLAGP.py`` (the index space that ``--hpx_start/--hpx_end`` slice into).

Two sources (use either; ``--qsocat`` is the reproducible one):

  --qsocat PATH        Count z-masked QSOs per healpix straight from the LOA QSO
                       catalog, applying the same cut desi-DLAGP.py uses
                       (constants.zmin_qso < Z < constants.zmax_qso). This is the
                       number of spectra each healpix index will dispatch, so it
                       is the principled balance weight for a fresh run.

  --from-run-dir DIR   Parse a completed run's per-task logs
                       (DIR/logs/loa_run_*.log) for the exact processed counts
                       ("Completed processing of N spectra from healpix H"). Use
                       to reproduce / validate against an actual run.

The two should agree to within the handful of QSOs that lack a coadd; if you
have both, pass --from-run-dir and the script will cross-check against --qsocat.
"""
import argparse
import glob
import os
import re
import sys


def counts_from_run_dir(run_dir):
    pat = re.compile(r"Completed processing of (\d+) spectra from healpix (\d+)")
    out = {}
    logs = glob.glob(os.path.join(run_dir, "logs", "loa_run_*.log"))
    if not logs:
        logs = glob.glob(os.path.join(run_dir, "loa_run_*.log"))
    if not logs:
        sys.exit(f"[counts] no loa_run_*.log under {run_dir}")
    for f in logs:
        with open(f, errors="ignore") as fh:
            for n, h in pat.findall(fh.read()):
                out[int(h)] = int(n)   # last write wins (idempotent re-runs)
    return out


def counts_from_qsocat(qsocat, pixel_col="HPXPIXEL"):
    import numpy as np
    import fitsio
    # Match desi-DLAGP.py / constants.py exactly. NOTE: this applies ONLY the
    # z-mask. desi-DLAGP's read_catalog additionally drops BAL/ZWARN/non-QSO rows
    # IFF constants.{no_bal,zwarning,is_qso} are True; the production LOA config
    # has all three False, so the index space matches. If one flips True and
    # removes an entire healpix, per-index alignment shifts (degrading *balance*
    # only -- coverage still tiles exactly). Use --from-run-dir to cross-check.
    #
    # pixel_col selects the grouping column: HPXPIXEL (default; the desi-DLAGP
    # index space, parity-preserving) or UNIQPIX (per-UNIQPIX QSO counts).
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import constants
    cat = fitsio.read(qsocat, columns=["Z", pixel_col])
    z = np.asarray(cat["Z"], dtype=float)
    hpx = np.asarray(cat[pixel_col]).astype(np.int64)  # defuse big-endian >q
    zmask = (z > constants.zmin_qso) & (z < constants.zmax_qso)
    hpx = hpx[zmask]
    uniq, cnt = np.unique(hpx, return_counts=True)
    return {int(h): int(c) for h, c in zip(uniq, cnt)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qsocat", default=None)
    ap.add_argument("--from-run-dir", default=None)
    ap.add_argument("--pixel-col", choices=["HPXPIXEL", "UNIQPIX"],
                    default="HPXPIXEL",
                    help="catalog column to group QSO counts by (default: "
                         "HPXPIXEL -- the desi-DLAGP index space).")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.qsocat and not a.from_run_dir:
        sys.exit("[counts] need --qsocat or --from-run-dir")

    primary = (counts_from_run_dir(a.from_run_dir) if a.from_run_dir
               else counts_from_qsocat(a.qsocat, a.pixel_col))

    # Cross-check when both sources are available.
    if a.from_run_dir and a.qsocat:
        ref = counts_from_qsocat(a.qsocat, a.pixel_col)
        only_run = set(primary) - set(ref)
        only_cat = set(ref) - set(primary)
        diff = sum(1 for h in (set(primary) & set(ref)) if primary[h] != ref[h])
        print(f"[counts] cross-check vs qsocat: {len(primary)} run hpx, "
              f"{len(ref)} cat hpx; run-only={len(only_run)} cat-only={len(only_cat)} "
              f"count-diff={diff}", file=sys.stderr)

    rows = sorted(primary.items())
    with open(a.out, "w") as f:
        f.write("# healpix_id  spec_count  (ascending hpx id == desi-DLAGP index order)\n")
        for h, c in rows:
            f.write(f"{h} {c}\n")
    print(f"[counts] wrote {len(rows)} healpix, total specs {sum(primary.values())} "
          f"-> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
