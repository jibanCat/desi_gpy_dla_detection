#!/usr/bin/env python3
"""tools/postprocess/build_bal_cat_from_qsocat.py

Build a mock-style ``bal_cat.fits`` (a list of BAL TARGETIDs) from a *real*
DESI QSO catalogue that carries BAL columns inline (e.g. the LOA "altbal"
healpix QSO catalogue: BAL_PROB / BI_CIV / AI_CIV / BALMASK ...).

The mock pipelines ship a dedicated ``bal_cat.fits`` whose every row is a BAL,
and ``tools/postprocess/add_dla_flags.py --bal-cat`` flags any catalogue
TARGETID found in it ("drop-all-BAL" convention, matching molly's recipe).
Real LOA has no such file — BAL information lives in the QSO catalogue itself,
which also contains the *non*-BAL quasars, so it cannot be passed to
``--bal-cat`` directly (that would flag every quasar). This tool extracts just
the BAL rows into the expected one-column shape.

Default selection is ``BI_CIV > 0`` — the balnicity-index definition used by
this repo's P/C tooling (tools/research/*) and the operating point the mock
purity/completeness was validated at. Override with --col / --thresh for the
DESI probabilistic classifier (``--col BAL_PROB --thresh 0.5``) or the catalogue
mask (``--col BALMASK --thresh 0``).

Usage:
  build_bal_cat_from_qsocat.py \\
      --qsocat /global/cfs/cdirs/desi/science/lya/y3/loa/catalogs/QSO_cat_loa_main_dark_healpix_v2-altbal-20241115.fits \\
      --out    /pscratch/sd/j/jibancat/.../loa_bal_cat_bici_gt0.fits \\
      [--col BI_CIV] [--thresh 0.0]
"""
import argparse

import fitsio
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--qsocat", required=True,
                   help="Input QSO catalogue FITS with TARGETID + the BAL column.")
    p.add_argument("--out", required=True,
                   help="Output bal_cat.fits (ext=1: TARGETID + the BAL column, "
                        "BAL rows only).")
    p.add_argument("--col", default="BI_CIV",
                   help="BAL-selection column (default BI_CIV).")
    p.add_argument("--thresh", type=float, default=0.0,
                   help="Keep rows with <col> strictly greater than this "
                        "(default 0.0 -> BI_CIV > 0).")
    return p.parse_args()


def main():
    a = parse_args()
    cols = fitsio.read(a.qsocat, ext=1, columns=["TARGETID", a.col])
    tid = np.asarray(cols["TARGETID"], dtype=np.int64)
    val = np.asarray(cols[a.col], dtype=np.float64)

    keep = val > a.thresh
    n_in, n_bal = tid.size, int(keep.sum())

    out = np.empty(n_bal, dtype=[("TARGETID", "i8"), (a.col, "f8")])
    out["TARGETID"] = tid[keep]
    out[a.col] = val[keep]

    with fitsio.FITS(a.out, "rw", clobber=True) as h:
        h.write(out, extname="BALCAT")
        h[1].write_key("SRCCAT", a.qsocat.split("/")[-1],
                       comment="source QSO catalogue")
        h[1].write_key("BALCOL", a.col, comment="BAL selection column")
        h[1].write_key("BALTHR", a.thresh, comment="kept rows: col > BALTHR")
        h[1].write_key("NUNQTID", int(np.unique(out["TARGETID"]).size))

    frac = 100.0 * n_bal / n_in if n_in else 0.0
    print(f"[build-bal] {a.col} > {a.thresh}: {n_bal}/{n_in} rows "
          f"({frac:.2f}%) -> {a.out}")


if __name__ == "__main__":
    main()
