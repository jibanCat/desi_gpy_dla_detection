#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""High-z archive manifest generator (PI ruling 2026-08-12 §19/§22).

Derives, deterministically from the QSO parent catalog (+ optionally the
CDDF absorber catalog), the target set the high-z loaArchive must cover:

  * role=highz_production — QSOs with zmin < Z <= zmax (default (4.25, 7.0]:
    the population omitted from the 2026-06 LOA productions by the
    constants.zmax_qso = 4.25 training-domain cut). Any such QSO contributes
    Lyα-only search path in the high-z Paper-1 bin; QSOs with
    z_hi(3000 km/s collar) > 4.25, i.e. z_qso > ~4.303, additionally
    contribute z_DLA > 4.25 analysis path (recorded per row).
  * role=diagnostic_host — sightlines hosting existing catalog candidates at
    Z_DLA >= dlacat-zmin (default 4.0), needed only for local diagnostic
    page generation. Skipped (loudly) when --dlacat is not given.

The manifest is target -> native storage key -> source file -> destination.
It is NOT committed to Git (real-survey TARGETIDs/redshifts); the canonical
identity is the sha256 of the sorted "TARGETID,role" lines, printed here and
recorded in provenance notes, so independently generated copies (GL vs
NERSC) can be verified identical without shipping the file.
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np

LYA, LYB = 1215.67, 1025.7223
C_KMS = 299792.458
COLLAR_KMS = 3000.0


def z_hi_collar(zq):
    return (1.0 + zq) * (1.0 - COLLAR_KMS / C_KMS) - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qsocat", required=True)
    ap.add_argument("--zmin", type=float, default=4.25)
    ap.add_argument("--zmax", type=float, default=7.0)
    ap.add_argument("--dlacat", default=None,
                    help="CDDF absorber catalog; adds diagnostic_host rows")
    ap.add_argument("--dlacat-zmin", type=float, default=4.0)
    ap.add_argument("--source-root", required=True,
                    help="healpix tree root containing "
                         "<pix//100>/<pix>/coadd-<survey>-<program>-<pix>.fits")
    ap.add_argument("--survey", default="main")
    ap.add_argument("--program", default="dark")
    ap.add_argument("--check-source", action="store_true",
                    help="stat() each source file and record present/missing")
    ap.add_argument("--dest", required=True,
                    help="destination archive path recorded per row")
    ap.add_argument("--out", required=True, help="manifest CSV path")
    args = ap.parse_args()

    from astropy.io import fits
    q = fits.open(args.qsocat)[1].data
    rows = {}

    m = (q["Z"] > args.zmin) & (q["Z"] <= args.zmax)
    for r in q[m]:
        t = int(r["TARGETID"])
        rows[t] = dict(TARGETID=t, Z_QSO=float(r["Z"]),
                       HPXPIXEL=int(r["HPXPIXEL"]),
                       role="highz_production",
                       contributes_zdla_gt_425=int(
                           z_hi_collar(float(r["Z"])) > 4.25))
    n_prod = len(rows)

    n_diag = 0
    if args.dlacat:
        d = fits.open(args.dlacat)[1].data
        sel = ((d["DLAFLAG"] == 0) & (d["P_DLA"] > 0.99)
               & (d["SNR_REDSIDE"] > 2.0) & (d["Z_DLA"] >= args.dlacat_zmin))
        hpx = {int(t): int(p) for t, p in zip(q["TARGETID"], q["HPXPIXEL"])}
        for r in d[sel]:
            t = int(r["TARGETID"])
            if t in rows:
                continue
            if t not in hpx:
                print(f"[warn] diagnostic host {t} not in QSO cat; skipped",
                      file=sys.stderr)
                continue
            rows[t] = dict(TARGETID=t, Z_QSO=float(r["Z_QSO"]),
                           HPXPIXEL=hpx[t], role="diagnostic_host",
                           contributes_zdla_gt_425=0)
            n_diag += 1
    else:
        print("[note] --dlacat not given: diagnostic_host rows OMITTED "
              "(archive will cover the production set only)",
              file=sys.stderr)

    out_rows = sorted(rows.values(), key=lambda r: r["TARGETID"])
    n_missing = 0
    for r in out_rows:
        pix = r["HPXPIXEL"]
        rel = (f"{pix // 100}/{pix}/"
               f"coadd-{args.survey}-{args.program}-{pix}.fits")
        r["source_file"] = rel
        r["dest"] = args.dest
        if args.check_source:
            ok = os.path.exists(os.path.join(args.source_root, rel))
            r["source_status"] = "present" if ok else "MISSING"
            n_missing += (not ok)
        else:
            r["source_status"] = "unchecked"
        r["dest_status"] = "pending"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cols = ["TARGETID", "Z_QSO", "HPXPIXEL", "role",
            "contributes_zdla_gt_425", "source_file", "source_status",
            "dest", "dest_status"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    ident = hashlib.sha256(
        "\n".join(f"{r['TARGETID']},{r['role']}"
                  for r in out_rows).encode()).hexdigest()
    pixes = sorted({r["HPXPIXEL"] for r in out_rows})
    summary = dict(
        n_total=len(out_rows), n_highz_production=n_prod,
        n_diagnostic_host=n_diag, n_unique_healpix=len(pixes),
        n_source_missing=(n_missing if args.check_source else None),
        target_list_sha256=ident,
        zmin=args.zmin, zmax=args.zmax, qsocat=os.path.basename(args.qsocat))
    json.dump(summary, open(args.out + ".summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    with open(args.out + ".pixes.txt", "w") as fh:
        fh.write("\n".join(str(p) for p in pixes) + "\n")


if __name__ == "__main__":
    main()
