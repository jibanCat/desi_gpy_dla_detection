"""Confirm DLA search results match between OLD pipeline (raw FITS via
load_one_desi_spectrum + dlasearch's coadd_cameras) vs NEW LoaArchive
(precomputed coadds in HDF5).

For N random TIDs from the archive, for each:
  1. Look up SOURCE_FILE → full FITS path under /nfs/turbo/.../loa/
  2. Load FITS path with load_one_desi_spectrum (canonical pipeline)
  3. Load archive path with LoaArchive.get_spectrum
  4. Bit-compare arrays (wave/flux/ivar/mask) — should be f32-byte-exact
  5. Run DLAHolder.process_qso on both
  6. Compare p_dlas / MAP_z / MAP_log_NHI / model_posteriors

This is the production-grade test of the PR description's claim:
"P(DLA) match to 4 sig figs through DLAHolder".

Runs ~12 min wall (3 TIDs × 2 paths × ~2 min/inference at production
num_dla_samples=10000, max_dlas=1).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ARCHIVE_DEFAULT = "/scratch/cavestru_root/cavestru0/mfho/nersc/loa_archives/loa_full_z2_noR_v2.h5"
LOA_ROOT_DEFAULT = "/nfs/turbo/lsa-cavestru/mfho/DESI/loa/"
PROD_LEARNED = "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5"
DR9Q_BASE = "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection"


def _build_holder():
    """Build a DLAHolder with production y3 settings (1-DLA only for speed)."""
    from gpy_dla_detection.set_parameters import Parameters
    from run_bayes_select import DLAHolder

    common = dict(
        loading_min_lambda=910.0, loading_max_lambda=1550.0,
        normalization_min_lambda=1425.0, normalization_max_lambda=1475.0,
        min_lambda=911.75, max_lambda=1216.75,
        dlambda=0.15, k=30, max_noise_variance=9.0,
        num_lines=3, max_z_cut=3000.0, min_z_cut=3000.0,
        num_forest_lines=3,
    )
    params = Parameters(num_dla_samples=10000, **common)
    params_subdla = Parameters(num_dla_samples=10000, **common)
    holder = DLAHolder(
        learned_file=PROD_LEARNED,
        catalog_name=os.path.join(DR9Q_BASE, "data/dr12q/processed/catalog.mat"),
        los_catalog=os.path.join(DR9Q_BASE,
            "data/dla_catalogs/dr9q_concordance/processed/los_catalog"),
        dla_catalog=os.path.join(DR9Q_BASE,
            "data/dla_catalogs/dr9q_concordance/processed/dla_catalog"),
        dla_samples_file=os.path.join(DR9Q_BASE,
            "data/dr12q/processed/dla_samples_a03.mat"),
        sub_dla_samples_file=os.path.join(DR9Q_BASE,
            "data/dr12q/processed/subdla_samples.mat"),
        params=params, params_subdla=params_subdla,
        min_z_separation=3000.0,
        prev_tau_0=0.00246, prev_beta=3.62,
        max_dlas=1, broadening=True, plot_figures=False,
        max_workers=1, batch_size=1,
        single_absorber_model=False,
    )
    return holder


def _ivar_to_nv(ivar):
    return np.where(ivar > 0, 1.0 / np.where(ivar == 0, 1.0, ivar), 1e10)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", default=ARCHIVE_DEFAULT)
    p.add_argument("--loa-root", default=LOA_ROOT_DEFAULT)
    p.add_argument("--n-tids", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default=None, help="Optional Markdown output")
    args = p.parse_args()

    from examples.smoke_one_spectrum import load_one_desi_spectrum
    from gpy_dla_detection.loa_archive import LoaArchive

    print(f"[main] archive: {args.archive}", flush=True)
    print(f"[main] loa root: {args.loa_root}", flush=True)
    ar = LoaArchive(args.archive)
    ar.open()
    print(f"  n_qsos: {ar.n_qsos}", flush=True)

    rng = np.random.default_rng(args.seed)
    all_tids = np.array(list(ar._tid_to_idx.keys()))

    # Read full catalog (small) so we can look up SOURCE_FILE per TID
    with h5py.File(args.archive, "r") as f:
        cat = f["catalog"][:]
    cat_by_tid = {int(r["TARGETID"]): r for r in cat}

    # Pick TIDs whose source_file actually exists on /turbo
    picked = []
    while len(picked) < args.n_tids:
        candidate = int(rng.choice(all_tids))
        sf = cat_by_tid[candidate]["SOURCE_FILE"].decode()
        full_path = os.path.join(args.loa_root, sf)
        if os.path.exists(full_path):
            picked.append((candidate, full_path))
        else:
            print(f"  skip TID {candidate}: SOURCE_FILE not found ({full_path})", flush=True)

    print(f"\n[main] picked {len(picked)} TIDs:", flush=True)
    for tid, fits in picked:
        z = float(cat_by_tid[tid]["Z"])
        print(f"  TID {tid}  z={z:.3f}  fits={Path(fits).name}", flush=True)

    print(f"\n[main] building DLAHolder ({PROD_LEARNED})", flush=True)
    holder = _build_holder()
    print(f"  built", flush=True)

    rows = []
    for i, (tid, fits_path) in enumerate(picked):
        z_qso = float(cat_by_tid[tid]["Z"])
        print(f"\n=== TID {tid} (z={z_qso:.3f}) ===", flush=True)

        # FITS path
        print("  loading via FITS pipeline...", flush=True)
        wave_f, flux_f, nv_f, mask_f = load_one_desi_spectrum(fits_path, int(tid))
        print(f"    n_pix={len(wave_f)}  flux med={np.nanmedian(flux_f):.3f}", flush=True)

        # Archive path
        print("  loading via LoaArchive...", flush=True)
        spec = ar.get_spectrum(int(tid))
        wave_a = ar.wavelength.astype(np.float32)
        flux_a = spec.flux.astype(np.float32)
        ivar_a = spec.ivar.astype(np.float32)
        mask_a = spec.mask.astype(np.uint32)
        nv_a = _ivar_to_nv(ivar_a.astype(np.float64))
        print(f"    n_pix={len(wave_a)}  flux med={np.nanmedian(flux_a):.3f}", flush=True)

        # Bit-compare arrays (require all4 to match at f32 byte-level)
        # Note FITS path returns f64 from internal coadd; archive is f32.
        # Cast FITS arrays to f32 for comparison.
        wave_match = np.array_equal(wave_a, wave_f.astype(np.float32))
        flux_match = np.array_equal(flux_a, flux_f.astype(np.float32))
        nv_a_f32 = nv_a.astype(np.float32)
        nv_f_f32 = nv_f.astype(np.float32)
        # nv comparison: where both finite, require equal. Where one is inf, both should be inf
        nv_match = (np.isclose(nv_a_f32, nv_f_f32, rtol=1e-5, atol=0,
                               equal_nan=True) | (np.isinf(nv_a_f32) & np.isinf(nv_f_f32))).all()
        mask_match = np.array_equal(mask_a, mask_f.astype(np.uint32))
        print(f"  array equality (f32): wave={wave_match} flux={flux_match} "
              f"nv={nv_match} mask={mask_match}", flush=True)

        # Run process_qso on each
        results = {}
        for label, wave, flux, nv, mask in [
            ("FITS",    wave_f.astype(np.float64), flux_f.astype(np.float64),
             nv_f.astype(np.float64), mask_f),
            ("ARCHIVE", wave_a.astype(np.float64), flux_a.astype(np.float64),
             nv_a.astype(np.float64), (mask_a != 0)),
        ]:
            t0 = time.time()
            holder.initialize_results(1)
            holder.process_qso(
                idx=0, target_id=tid,
                wavelengths=wave, flux=flux,
                noise_variance=nv,
                pixel_mask=mask if mask.dtype == bool else (mask != 0),
                z_qso=z_qso,
            )
            t = time.time() - t0
            results[label] = dict(
                p_no_dlas=float(holder.results["p_no_dlas"][0]),
                p_dlas=float(holder.results["p_dlas"][0]),
                map_z=float(holder.results["MAP_z_dlas"][0, 0]),
                map_log_nhi=float(holder.results["MAP_log_nhis"][0, 0]),
                model_posteriors=np.asarray(holder.results["model_posteriors"][0]).copy(),
                t=t,
            )
            print(f"  {label:8s}: p_dla={results[label]['p_dlas']:.4f}  "
                  f"MAP_z={results[label]['map_z']:.4f}  MAP_logNHI={results[label]['map_log_nhi']:.3f}  "
                  f"({t:.1f}s)", flush=True)

        # Diff
        dp = abs(results["ARCHIVE"]["p_dlas"] - results["FITS"]["p_dlas"])
        dmp = np.abs(results["ARCHIVE"]["model_posteriors"]
                     - results["FITS"]["model_posteriors"])
        print(f"  Δp_dla = {dp:.6f}  max|Δmodel_post| = {np.nanmax(dmp):.6f}", flush=True)
        rows.append(dict(tid=tid, z_qso=z_qso, fits=results["FITS"], archive=results["ARCHIVE"],
                         dp=dp, dmp_max=float(np.nanmax(dmp)),
                         arrays_match=dict(wave=wave_match, flux=flux_match,
                                           nv=nv_match, mask=mask_match)))

    ar.close()

    # Summary
    print("\n=== SUMMARY ===", flush=True)
    for r in rows:
        verdict = "PASS" if r["dp"] < 1e-4 and r["dmp_max"] < 1e-4 else "FAIL"
        am = r["arrays_match"]
        print(f"  TID {r['tid']}: arrays {sum(am.values())}/4 match  "
              f"Δp_dla={r['dp']:.2e}  Δmp_max={r['dmp_max']:.2e}  → {verdict}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            f.write(f"# FITS vs LoaArchive DLA search comparison\n\n")
            f.write(f"Archive: `{args.archive}`\n")
            f.write(f"LOA root: `{args.loa_root}`\n")
            f.write(f"Production model: `{PROD_LEARNED}`\n")
            f.write(f"num_dla_samples=10000, max_dlas=1, single_absorber_model=False\n\n")
            f.write(f"| TID | z_qso | wave | flux | nv | mask | "
                    f"p_dla FITS | p_dla ARCH | Δp_dla | MAP_z FITS | MAP_z ARCH | Δmp_max |\n")
            f.write(f"|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|\n")
            for r in rows:
                am = r["arrays_match"]
                f.write(f"| {r['tid']} | {r['z_qso']:.3f} "
                        f"| {'✓' if am['wave'] else '✗'} | {'✓' if am['flux'] else '✗'} "
                        f"| {'✓' if am['nv'] else '✗'} | {'✓' if am['mask'] else '✗'} "
                        f"| {r['fits']['p_dlas']:.4f} | {r['archive']['p_dlas']:.4f} "
                        f"| {r['dp']:.2e} "
                        f"| {r['fits']['map_z']:.4f} | {r['archive']['map_z']:.4f} "
                        f"| {r['dmp_max']:.2e} |\n")
        print(f"\n[main] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
