#!/usr/bin/env python3
"""tools/postprocess/add_dla_flags.py

Add downstream-friendly flag columns to a directory of dlacat-*.fits files
*in place*. Idempotent: re-running clears the previous postprocess bits
on DLAFLAG and the boolean columns are overwritten.

Flag columns added (booleans + supporting metadata):
  LYBETA_FLAG          bool   — DLA is a likely Lyβ misID of a higher-z DLA
                                on the same LOS. From
                                gpy_dla_detection.postprocess.lyb_veto.
  LYBETA_PARENT_TID    int64  — TARGETID of the parent (higher-z DLA).
  LYBETA_PARENT_Z      float  — z of the parent.
  BAL_FLAG             bool   — TARGETID is in bal_cat.fits (any row;
                                molly's recipe drops all bal_cat TIDs,
                                not just BI_CIV>0).
  NHI_CONSISTENCY_FLAG bool   — NHI - k * NHI_ERR < 20.3, i.e. the predicted
                                NHI's lower 1σ falls below the canonical
                                catalog cut. With k = 0.5 this drops
                                weakly-constrained DLAs whose NHI uncertainty
                                pulls them below 20.3.
  PDLA_SATURATED_FLAG  bool   — P_DLA ≥ 1 - pdla_saturation_threshold (default
                                ≥ 1 − 1e-7, equivalent to log_BF ≥ 15.4 for
                                N_DLA_SAMPLES = 50000). Informational only:
                                marks rows where the p_DLA cut is a no-op
                                (high-confidence detection). NOT folded
                                into DLAFLAG.

DLAFLAG bitmask updates (see gpy_dla_detection/../fitwarning.py for full def):
  bit 6 (LYBETA_MISID)     ← LYBETA_FLAG
  bit 7 (BAL_CAT_OVERLAP)  ← BAL_FLAG
  bit 8 (NHI_INCONSISTENT) ← NHI_CONSISTENCY_FLAG

After running this script, downstream consumers can use either:
  cat[cat["DLAFLAG"] == 0]                          # "clean" production cat
  cat[(cat["DLAFLAG"] & 0x3F) == 0]                 # only inference warnings
  cat[~cat["LYBETA_FLAG"] & ~cat["BAL_FLAG"]]       # finer-grained filtering

The script is meant to be the LAST step of the production pipeline:

    inference (slurm/run_local.sh)
        → produces dlacat-*.fits + processed/processed-spectra-16-*.h5
    postprocess (this script)
        → adds the 5 flag columns to each dlacat-*.fits in place
    eval / catalog distribution
        → downstream consumers read dlacat-*.fits and use the flags

Usage:
    python tools/postprocess/add_dla_flags.py \
        --catalog-dir /path/to/run/ \
        --bal-cat /path/to/bal_cat.fits \
        [--nhi-consistency-k 0.5] \
        [--pdla-saturation 1e-7] \
        [--lyb-veto-dz 0.005] \
        [--no-lyb-veto] [--no-bal-flag] \
        [--no-nhi-consistency] [--no-pdla-saturated]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import fitsio
from astropy.table import Table

# Make `import gpy_dla_detection.…` work whether the repo is on PYTHONPATH or not.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from gpy_dla_detection.postprocess.lyb_veto import flag_lybeta
from fitwarning import DLAFLAG


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--catalog-dir", required=True,
                   help="Directory containing dlacat-*.fits files.")
    p.add_argument("--bal-cat", default=None,
                   help="Path to bal_cat.fits. Required unless --no-bal-flag.")
    p.add_argument("--nhi-consistency-k", type=float, default=0.5,
                   help="k in `NHI - k * NHI_ERR < 20.3` (default 0.5).")
    p.add_argument("--nhi-consistency-floor", type=float, default=20.3,
                   help="NHI floor for the consistency gate (default 20.3).")
    p.add_argument("--pdla-saturation", type=float, default=1e-7,
                   help="PDLA_SATURATED_FLAG = P_DLA >= 1 - this (default 1e-7).")
    p.add_argument("--lyb-veto-dz", type=float, default=0.005,
                   help="dz_match for the Lyβ veto (default 0.005).")
    p.add_argument("--no-lyb-veto", action="store_true",
                   help="Skip LYBETA_FLAG / LYBETA_PARENT_* columns.")
    p.add_argument("--no-bal-flag", action="store_true",
                   help="Skip BAL_FLAG column. Otherwise --bal-cat required.")
    p.add_argument("--no-nhi-consistency", action="store_true",
                   help="Skip NHI_CONSISTENCY_FLAG column.")
    p.add_argument("--no-pdla-saturated", action="store_true",
                   help="Skip PDLA_SATURATED_FLAG column.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-file progress lines.")
    return p.parse_args()


def _add_lybeta(tbl: Table, dz_match: float) -> tuple[int, int]:
    """Mutate `tbl` in place: add LYBETA_FLAG/PARENT_TID/PARENT_Z. Return
    (n_total_dla_rows, n_lyb_misid_flagged)."""
    flag_lybeta(
        tbl,
        targetid_col="TARGETID",
        z_col="Z_DLA",
        nhi_col="NHI",      # dlacat's NHI is already log10(N_HI)
        dz_match=dz_match,
        require_higher_nhi_parent=True,
    )
    return len(tbl), int(np.asarray(tbl["LYBETA_FLAG"], dtype=bool).sum())


def _add_bal_flag(tbl: Table, bal_tids: set[int]) -> int:
    """Mutate `tbl`: add BAL_FLAG. Return n_flagged."""
    tids = np.asarray(tbl["TARGETID"], dtype=np.int64)
    flag = np.isin(tids, np.fromiter(bal_tids, dtype=np.int64))
    tbl["BAL_FLAG"] = flag
    return int(flag.sum())


def _add_nhi_consistency(tbl: Table, k: float, floor: float) -> int:
    """Mutate `tbl`: add NHI_CONSISTENCY_FLAG. True == fails the gate."""
    nhi = np.asarray(tbl["NHI"], dtype=np.float64)
    nhi_err = np.asarray(tbl["NHI_ERR"], dtype=np.float64)
    flag = (nhi - k * nhi_err) < floor
    tbl["NHI_CONSISTENCY_FLAG"] = flag
    return int(flag.sum())


def _add_pdla_saturated(tbl: Table, threshold: float) -> int:
    """Mutate `tbl`: add PDLA_SATURATED_FLAG."""
    p = np.asarray(tbl["P_DLA"], dtype=np.float64)
    flag = p >= (1.0 - threshold)
    tbl["PDLA_SATURATED_FLAG"] = flag
    return int(flag.sum())


def _update_dlaflag_bitmask(tbl: Table) -> int:
    """Fold the boolean flag columns into the DLAFLAG bitmask.

    1. Clear postprocess bits (6, 7, 8) on every row first → idempotent re-run.
    2. Set each postprocess bit if the corresponding boolean column is True.

    Returns the number of rows where DLAFLAG changed.
    """
    if "DLAFLAG" not in tbl.colnames:
        # Older catalogs without DLAFLAG: create as int64 zero
        tbl["DLAFLAG"] = np.zeros(len(tbl), dtype=np.int64)
    flag = np.asarray(tbl["DLAFLAG"], dtype=np.int64).copy()
    before = flag.copy()

    # Clear ALL known postprocess bits (current + legacy schema). Keeps the
    # inference-time bits intact. The legacy mask handles dlacats that were
    # postprocessed under an earlier bit numbering and are being re-flagged.
    flag &= ~np.int64(DLAFLAG._ALL_POSTPROCESS_BITS_TO_CLEAR)

    # Set postprocess bits from the boolean columns when present.
    if "LYBETA_FLAG" in tbl.colnames:
        m = np.asarray(tbl["LYBETA_FLAG"], dtype=bool)
        flag[m] |= np.int64(DLAFLAG.LYBETA_MISID)
    if "BAL_FLAG" in tbl.colnames:
        m = np.asarray(tbl["BAL_FLAG"], dtype=bool)
        flag[m] |= np.int64(DLAFLAG.BAL_CAT_OVERLAP)
    if "NHI_CONSISTENCY_FLAG" in tbl.colnames:
        m = np.asarray(tbl["NHI_CONSISTENCY_FLAG"], dtype=bool)
        flag[m] |= np.int64(DLAFLAG.NHI_INCONSISTENT)

    tbl["DLAFLAG"] = flag
    return int((flag != before).sum())


def process_one(path: Path, args, bal_tids: set[int] | None,
                ) -> dict[str, int]:
    """Process one dlacat-*.fits file in place. Return per-file stats."""
    tbl = Table(fitsio.read(str(path), ext=1))
    stats: dict[str, int] = {"path": str(path), "n_rows": len(tbl)}

    if not args.no_lyb_veto:
        n, n_flag = _add_lybeta(tbl, args.lyb_veto_dz)
        stats["lyb_flagged"] = n_flag
    if not args.no_bal_flag and bal_tids is not None:
        stats["bal_flagged"] = _add_bal_flag(tbl, bal_tids)
    if not args.no_nhi_consistency:
        stats["nhi_consistency_flagged"] = _add_nhi_consistency(
            tbl, args.nhi_consistency_k, args.nhi_consistency_floor)
    if not args.no_pdla_saturated:
        stats["pdla_saturated"] = _add_pdla_saturated(tbl, args.pdla_saturation)

    # Fold quality flags into DLAFLAG bitmask (excludes informational
    # PDLA_SATURATED). After this, `cat[cat["DLAFLAG"] == 0]` is the
    # "clean" downstream filter.
    stats["dlaflag_changed"] = _update_dlaflag_bitmask(tbl)
    flag = np.asarray(tbl["DLAFLAG"], dtype=np.int64)
    stats["dlaflag_zero"] = int((flag == 0).sum())

    # In-place rewrite via fitsio.write(..., clobber=True)
    fitsio.write(str(path), np.asarray(tbl), extname="DLAS", clobber=True)
    return stats


def main():
    args = parse_args()
    cat_dir = Path(args.catalog_dir)
    if not cat_dir.is_dir():
        raise SystemExit(f"--catalog-dir not a directory: {cat_dir}")

    files = sorted(cat_dir.glob("dlacat-*.fits"))
    if not files:
        raise SystemExit(f"no dlacat-*.fits in {cat_dir}")
    print(f"[postprocess] found {len(files)} dlacat-*.fits in {cat_dir}")

    bal_tids: set[int] | None = None
    if not args.no_bal_flag:
        if not args.bal_cat:
            raise SystemExit(
                "--bal-cat required (or pass --no-bal-flag to skip BAL_FLAG)")
        bal = fitsio.read(args.bal_cat, ext=1, columns=["TARGETID"])
        bal_tids = set(int(r["TARGETID"]) for r in bal)
        print(f"[postprocess] loaded {len(bal_tids)} BAL TIDs from "
              f"{args.bal_cat} (drop-all-BAL convention; matches molly)")

    totals = {"n_rows": 0, "lyb_flagged": 0, "bal_flagged": 0,
              "nhi_consistency_flagged": 0, "pdla_saturated": 0,
              "dlaflag_zero": 0, "dlaflag_changed": 0}
    for f in files:
        stats = process_one(f, args, bal_tids)
        if not args.quiet:
            extras = " ".join(f"{k}={v}" for k, v in stats.items()
                              if k not in ("path", "n_rows"))
            print(f"[postprocess] {f.name}: rows={stats['n_rows']} {extras}")
        for k, v in stats.items():
            if k in totals:
                totals[k] += v

    print()
    print(f"[postprocess] TOTALS across {len(files)} files:")
    print(f"  rows                       = {totals['n_rows']}")
    if not args.no_lyb_veto:
        n = totals['lyb_flagged']; tot = totals['n_rows']
        pct = 100.0 * n / tot if tot else 0.0
        print(f"  LYBETA_FLAG=True           = {n} ({pct:.2f}%)")
    if not args.no_bal_flag:
        n = totals['bal_flagged']; tot = totals['n_rows']
        pct = 100.0 * n / tot if tot else 0.0
        print(f"  BAL_FLAG=True              = {n} ({pct:.2f}%)")
    if not args.no_nhi_consistency:
        n = totals['nhi_consistency_flagged']; tot = totals['n_rows']
        pct = 100.0 * n / tot if tot else 0.0
        print(f"  NHI_CONSISTENCY_FLAG=True  = {n} ({pct:.2f}%)  "
              f"(k={args.nhi_consistency_k}, floor={args.nhi_consistency_floor})")
    if not args.no_pdla_saturated:
        n = totals['pdla_saturated']; tot = totals['n_rows']
        pct = 100.0 * n / tot if tot else 0.0
        print(f"  PDLA_SATURATED_FLAG=True   = {n} ({pct:.2f}%)  "
              f"(P_DLA ≥ 1 − {args.pdla_saturation:g})  [informational, NOT in DLAFLAG]")
    n = totals['dlaflag_zero']; tot = totals['n_rows']
    pct = 100.0 * n / tot if tot else 0.0
    print(f"  DLAFLAG == 0 ('clean')     = {n} ({pct:.2f}%)  "
          f"[postprocess bits 6-8 + inference bits 0-5 all clear]")
    print(f"[postprocess] done.")


if __name__ == "__main__":
    main()
