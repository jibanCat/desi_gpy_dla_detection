#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the H2 injected LoaArchive from a realized injection plan.

Archive-substrate injection (GL execution of the approved H2-v2 campaign,
PI 2026-08-13): applies the canonical Voigt transmission
``gpy_dla_detection.inject_absorber.inject_voigt`` — the SAME function the
established coadd-native machinery uses (injection/coadd_injection.py,
"never reimplement") — multiplicatively to the archive's post-coadd_cameras
flux on the native brz grid. Multiple injections per sightline blend
multiplicatively, identical to the coadd machinery and to the GP's own
multi-absorber composition. ivar and mask are NOT modified (parity with
inject_into_coadd, which modifies flux only).

The output is a schema-identical LoaArchive containing ONLY the injected
sightlines; the finder then runs on it via the validated --spectra_archive
production route. The source archive is opened read-only and never
modified. Substrate equivalence (camera-native raw injection + raw route
vs this archive-level injection + archive route) is established by the
companion A/B harness before any campaign execution.
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realized-plan", required=True)
    ap.add_argument("--archive", required=True,
                    help="source LoaArchive (read-only)")
    ap.add_argument("--out", required=True,
                    help="output injected LoaArchive HDF5")
    ap.add_argument("--num-lines", type=int, default=3,
                    help="Lyman-series lines (matches production NUM_LINES)")
    ap.add_argument("--arm", choices=["A", "B", "all"], default="all",
                    help="restrict to one arm's sightlines")
    args = ap.parse_args()

    import h5py
    from gpy_dla_detection.inject_absorber import inject_voigt

    plan = list(csv.DictReader(open(args.realized_plan)))
    if args.arm != "all":
        plan = [r for r in plan if r["cell"].startswith(args.arm)]
    by_tid = {}
    for r in plan:
        by_tid.setdefault(int(r["TARGETID"]), []).append(r)

    if os.path.abspath(args.out) == os.path.abspath(args.archive):
        raise SystemExit("FATAL: output must differ from the source archive")

    src = h5py.File(args.archive, "r")
    cat = src["catalog"][:]
    idx = {int(t): int(i) for i, t in enumerate(cat["TARGETID"])}
    missing = [t for t in by_tid if t not in idx]
    if missing:
        raise SystemExit(
            f"FATAL: {len(missing)} plan sightlines absent from the source "
            f"archive (first: {missing[:5]}) — substrate incomplete; refusing "
            "a partial build")
    wave = src["wavelength"][:]
    wave_f8 = np.load(os.path.join(REPO_ROOT, "data", "brz_wave_grid_f8.npy"))
    assert np.array_equal(wave_f8.astype(np.float32), wave)

    tids = sorted(by_tid)
    n_pix = len(wave)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    truth_rows = []
    with h5py.File(args.out, "w") as out:
        for k in src.attrs:
            out.attrs[k] = src.attrs[k]
        out.attrs["h2_injected"] = 1
        out.attrs["h2_source_archive"] = args.archive
        out.create_dataset("wavelength", data=wave)
        sel = np.array([idx[t] for t in tids])
        out.create_dataset("catalog", data=cat[sel])
        flux_d = out.create_dataset("flux", shape=(len(tids), n_pix),
                                    dtype="f4", compression="gzip",
                                    compression_opts=4)
        for name in ("ivar", "mask", "fwhm_pix"):
            d = out.create_dataset(name, shape=(len(tids), n_pix),
                                   dtype=src[name].dtype, compression="gzip",
                                   compression_opts=4)
            for j, t in enumerate(tids):
                d[j] = src[name][idx[t]]
        for j, t in enumerate(tids):
            fl = src["flux"][idx[t]].astype(np.float64)
            for r in sorted(by_tid[t], key=lambda r: int(r["inj_idx"])):
                nhi = 10.0 ** float(r["logN"])
                z = float(r["z_inj"])
                fl = inject_voigt(wave_f8, fl, nhi, z, args.num_lines)
                truth_rows.append(dict(
                    TARGETID=t, inj_idx=int(r["inj_idx"]), cell=r["cell"],
                    z_true=z, logN_true=float(r["logN"]),
                    num_lines=args.num_lines))
            flux_d[j] = fl.astype(np.float32)
    src.close()

    truth = args.out + ".truth.csv"
    with open(truth, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(truth_rows[0]))
        w.writeheader()
        for r in sorted(truth_rows, key=lambda r: (r["TARGETID"],
                                                   r["inj_idx"])):
            w.writerow(r)
    summary = dict(
        n_sightlines=len(tids), n_injections=len(truth_rows),
        source_archive=args.archive,
        source_archive_sha256=_sha_file(args.archive),
        injected_archive=args.out,
        injected_archive_sha256=_sha_file(args.out),
        truth_manifest=truth, truth_manifest_sha256=_sha_file(truth),
        num_lines=args.num_lines, arm=args.arm,
        realized_plan=args.realized_plan,
        realized_plan_file_sha256=_sha_file(args.realized_plan),
    )
    json.dump(summary, open(args.out + ".build_summary.json", "w"), indent=1)
    print(f"H2 INJECTED ARCHIVE: {len(tids)} sightlines, "
          f"{len(truth_rows)} injections -> {args.out}\n"
          f"  sha256 {summary['injected_archive_sha256'][:16]}…")


if __name__ == "__main__":
    main()
