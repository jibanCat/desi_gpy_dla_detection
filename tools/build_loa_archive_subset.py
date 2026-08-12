#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a LoaArchive HDF5 for the targets of a hz_manifest.py manifest.

Reads DESI healpix coadds (native storage granularity), coadd_cameras them
exactly as the finder's own preprocessing does, and streams the manifest's
targets into one `gpy_dla_detection.loa_archive.write_archive` HDF5.

NOTE (finder invariance): this archive is for GL-side DIAGNOSTICS and
future injection work. The production finder itself (pilot + incremental
run) reads the raw CFS healpix FITS at NERSC exactly as the historical
production did — it does NOT consume this float32 archive, so archive
precision cannot enter the finder-invariance question.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np


def iter_records(manifest_rows, source_root, wave_holder):
    import desispec.io
    from desispec.coaddition import coadd_cameras
    from gpy_dla_detection.loa_archive import (CoaddRecord,
                                               fwhm_pixels_from_resolution)

    bypix = defaultdict(list)
    for r in manifest_rows:
        bypix[int(r["HPXPIXEL"])].append(r)

    for pix in sorted(bypix):
        rel = bypix[pix][0]["source_file"]
        path = os.path.join(source_root, rel)
        if not os.path.exists(path):
            print(f"[MISSING] {path}", file=sys.stderr)
            for r in bypix[pix]:
                r["dest_status"] = "SOURCE_MISSING"
            continue
        spectra = desispec.io.read_spectra(path)
        coadd = coadd_cameras(spectra)
        band = coadd.bands[0]
        wave = np.asarray(coadd.wave[band])
        if wave_holder.get("wave") is None:
            wave_holder["wave"] = wave
        elif not np.array_equal(wave_holder["wave"], wave):
            raise RuntimeError(f"wavelength grid differs at pix {pix}")
        fm_tids = np.asarray(coadd.fibermap["TARGETID"])
        for r in bypix[pix]:
            t = int(r["TARGETID"])
            ix = np.where(fm_tids == t)[0]
            if ix.size == 0:
                print(f"[NOT-IN-COADD] TID {t} pix {pix}", file=sys.stderr)
                r["dest_status"] = "NOT_IN_COADD"
                continue
            i = int(ix[0])
            R = np.asarray(coadd.resolution_data[band][i])
            yield r, CoaddRecord(
                targetid=t, z=float(r["Z_QSO"]), ra=np.nan, dec=np.nan,
                healpix=pix, zwarn=0, blue_snr=np.nan, red_snr=np.nan,
                source_file=rel, fiber_idx=i,
                flux=np.asarray(coadd.flux[band][i]),
                ivar=np.asarray(coadd.ivar[band][i]),
                mask=np.asarray(coadd.mask[band][i]),
                R=R if R.shape[0] == 11 else
                fwhm_pixels_from_resolution(R)[None, :].repeat(11, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-resolution", action="store_true")
    args = ap.parse_args()

    from gpy_dla_detection.loa_archive import write_archive

    rows = list(csv.DictReader(open(args.manifest)))
    wave_holder = {"wave": None}

    done = []

    def gen():
        for r, rec in iter_records(rows, args.source_root, wave_holder):
            r["dest_status"] = "archived"
            done.append(r["TARGETID"])
            yield rec

    # write_archive consumes the generator; wavelength must exist by the
    # time it is needed — prime the generator with a peeking wrapper.
    it = gen()
    try:
        first = next(it)
    except StopIteration:
        print("FATAL: no archivable targets", file=sys.stderr)
        sys.exit(2)

    def chain():
        yield first
        yield from it

    stats = write_archive(args.out, chain(),
                          wavelength=wave_holder["wave"],
                          source_root=args.source_root,
                          with_resolution=args.with_resolution)
    # write back per-row dest_status next to the manifest
    outm = args.manifest.replace(".csv", ".archived.csv")
    with open(outm, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summary = dict(n_manifest=len(rows), n_archived=len(done),
                   n_failed=len(rows) - len(done),
                   archive=args.out, stats={k: str(v)
                                            for k, v in stats.items()})
    json.dump(summary, open(args.out + ".build_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    if len(done) != len(rows):
        print("INCOMPLETE ARCHIVE — failing loudly", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
