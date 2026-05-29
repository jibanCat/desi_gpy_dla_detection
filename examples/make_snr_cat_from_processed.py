"""
examples/make_snr_cat_from_processed.py
=======================================
Build a small consolidated SNR catalog from the per-file
``processed-spectra-*.h5`` inference outputs, WITHOUT combining them.

Motivation
----------
Each ``processed-spectra-16-N.h5`` carries the full per-sample arrays
(``sample_log_likelihoods_dla`` ≈ 2.7 GB/file, ``base_sample_inds`` ≈ 0.9 GB),
so a combined HDF5 over a whole mock would be hundreds of GB — impractical to
load. Downstream analysis (e.g. ``molly_faithful_pc_plots.py`` SNR cut, dN/dX)
only needs a tiny TARGETID -> (SNR, z_qso) lookup. This script extracts exactly
that into one small file, leaving the per-file h5s separate.

Outputs (``--out-prefix`` , default ``<processed-dir>/snr_cat``):
  - ``<prefix>.fits``  (cols: TARGETID, SNR_REDSIDE, SNR_BLUE, Z) — usable as
    ``molly_faithful_pc_plots.py --snr-cat <prefix>.fits`` (its canonical
    priority-1 SNR source, no per-file-h5 symlink needed).
  - ``<prefix>.h5``    (datasets: target_ids, snr_redside, snr_blue, z_qso).

Usage
-----
    python examples/make_snr_cat_from_processed.py \
        --processed-dir <OUTDIR>/figures/processed \
        [--out-prefix <OUTDIR>/snr_cat]
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import h5py


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", required=True,
                    help="Directory containing processed-spectra-*.h5 files.")
    ap.add_argument("--out-prefix", default=None,
                    help="Output path prefix (default: <processed-dir>/snr_cat).")
    ap.add_argument("--pattern", default="processed-spectra-*.h5",
                    help="Glob for the per-file h5s (default processed-spectra-*.h5).")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.processed_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No {args.pattern} under {args.processed_dir}")

    tid, snr_r, snr_b, zq = [], [], [], []
    for fp in files:
        with h5py.File(fp, "r") as h:
            tid.append(np.asarray(h["target_ids"], dtype=np.int64))
            snr_r.append(np.asarray(h["snrs"], dtype=np.float64))
            # snrs_blue / z_qsos present in the DESI schema; guard for older files
            snr_b.append(np.asarray(h["snrs_blue"], dtype=np.float64)
                         if "snrs_blue" in h else np.full(h["snrs"].shape, np.nan))
            zq.append(np.asarray(h["z_qsos"], dtype=np.float64)
                      if "z_qsos" in h else np.full(h["snrs"].shape, np.nan))
        print(f"  {os.path.basename(fp)}: {tid[-1].size} spectra")

    tid = np.concatenate(tid)
    snr_r = np.concatenate(snr_r)
    snr_b = np.concatenate(snr_b)
    zq = np.concatenate(zq)

    # de-duplicate (a TARGETID should appear once; keep first)
    _, uniq = np.unique(tid, return_index=True)
    if uniq.size != tid.size:
        print(f"  [dedup] {tid.size} -> {uniq.size} unique TARGETIDs")
    tid, snr_r, snr_b, zq = tid[uniq], snr_r[uniq], snr_b[uniq], zq[uniq]

    prefix = args.out_prefix or os.path.join(args.processed_dir, "snr_cat")
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)

    # h5 (downstream-friendly)
    with h5py.File(prefix + ".h5", "w") as out:
        out.create_dataset("target_ids", data=tid)
        out.create_dataset("snr_redside", data=snr_r)
        out.create_dataset("snr_blue", data=snr_b)
        out.create_dataset("z_qso", data=zq)

    # FITS (molly --snr-cat: needs TARGETID + SNR_REDSIDE; Z lets it also serve as zcat)
    try:
        import fitsio
        fitsio.write(prefix + ".fits",
                     {"TARGETID": tid, "SNR_REDSIDE": snr_r,
                      "SNR_BLUE": snr_b, "Z": zq},
                     extname="SNRCAT", clobber=True)
        wrote_fits = True
    except Exception as e:  # pragma: no cover
        wrote_fits = False
        print(f"  [warn] FITS write skipped: {e}")

    print(f"\n[done] {tid.size} TARGETIDs from {len(files)} files")
    print(f"  SNR_REDSIDE: median={np.nanmedian(snr_r):.2f} "
          f"range=[{np.nanmin(snr_r):.2f}, {np.nanmax(snr_r):.2f}]")
    print(f"  wrote {prefix}.h5" + (f" + {prefix}.fits" if wrote_fits else ""))


if __name__ == "__main__":
    main()
