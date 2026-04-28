#!/usr/bin/env python
"""Streamlined LOA real-data preload → gp_interp_trainset.h5.

EXPLICIT FILTER PIPELINE (each filter applied in this order):

    INPUT: full altbal QSO catalog (~2.7M rows)
                                        |
       (1) z + ZWARN filter             v
       --------------------------------------
       Keep rows with:
         z_min ≤ Z ≤ z_max
         ZWARN == 0  (if column exists)
                                        |
       (2) BAL anti-join                v       (only if --exclude-bal)
       --------------------------------------
       Drop rows where BAL_COL > BAL_MIN
       (default BAL_COL = "BI_CIV", BAL_MIN = 0.0).
       This is an in-catalog filter — no external file needed.
                                        |
       (3) HCD anti-join                v       (only if --hcd-cat is given)
       --------------------------------------
       Read EXTERNAL catalog (FITS or HDF5) of HCD detections.
       For each TARGETID in that catalog, compute the per-spectrum
       MAX log NHI across DLA/HCD slots:
         - For 1D NHI columns: use directly.
         - For 2D NHI arrays (combined.h5 stores
           MAP_log_nhis as slots×spectra): take np.nanmax along the
           slot axis.
       Mark TARGETIDs where (max log NHI) ≥ --hcd-min-nhi.
       Drop QSO rows whose TARGETID is in that set.

       The threshold determines what "HCD" means:
         --hcd-min-nhi 20.3 → DLAs only             (logNHI ≥ 20.3)
         --hcd-min-nhi 19.0 → DLAs + sub-DLAs       (logNHI ≥ 19.0)
         --hcd-min-nhi 17.2 → all HCDs (LLS + sub-DLA + DLA)

       NOTE: a TARGETID NOT present in --hcd-cat is KEPT. So the
       HCD catalog must cover the QSO sample of interest, OR the
       user accepts that uncatalogued HCDs slip through — typical
       for real LOA where LLS detection is incomplete.
                                        |
       (4) cap to --max-spectra         v       (random subset, seeded)
       --------------------------------------
       If still > max_spectra, draw a uniform random subset.

    OUTPUT: per-spectrum (flux, noise_var) interpolated to a common
    rest-frame grid, in HDF5 with the legacy gp_interp_trainset
    schema (`tids`, `rest_wavelengths`, `fluxes`, `noise_variance`,
    `zqso`, `redsnr`, `bluesnr`).

This is the production-data analogue of ``preload_2lpt_simple.py``.
Real LOA spectra differ from 2LPT in three ways:
  (a) ``HPXPIXEL`` is already in the catalog — no RA/DEC → healpix step.
  (b) ``BI_CIV`` is in the SAME catalog as ``Z`` and ``TARGETID``
      (no separate ``bal_cat.fits`` — it's all altbal).
  (c) The healpix file path follows the LOA layout:
      ``{specdir}/healpix/main/dark/{H//100}/{H}/coadd-main-dark-{H}.fits``.

Usage examples::

    # 1) DLAs + BALs excluded — legacy convention. sub-DLAs / LLS kept.
    python preload_spectra/preload_loa_real.py \\
        --qsocat /path/to/QSO_cat_loa_main_dark_healpix_v3-altbal.fits \\
        --specdir /global/cfs/cdirs/desi/spectro/redux/loa \\
        --hcd-cat /path/to/dla_combined.h5 \\
        --hcd-tid-col target_ids --hcd-nhi-col MAP_log_nhis \\
        --hcd-min-nhi 20.3 \\
        --exclude-bal \\
        --output trainset_no_dla_no_bal.h5

    # 2) All HCDs excluded, BALs KEPT — gives a "BAL-aware" GP.
    python preload_spectra/preload_loa_real.py \\
        --qsocat ... --specdir ... --hcd-cat ... \\
        --hcd-tid-col target_ids --hcd-nhi-col MAP_log_nhis \\
        --hcd-min-nhi 17.2 \\
        --output trainset_no_hcd_with_bal.h5

    # 3) All HCDs + BALs excluded — cleanest baseline.
    python preload_spectra/preload_loa_real.py \\
        --qsocat ... --specdir ... --hcd-cat ... \\
        --hcd-tid-col target_ids --hcd-nhi-col MAP_log_nhis \\
        --hcd-min-nhi 17.2 \\
        --exclude-bal \\
        --output trainset_no_hcd_no_bal.h5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d


def _spec_path_loa(specdir: Path, healpix: int) -> Path:
    """LOA coadd path:
    {specdir}/healpix/main/dark/{H//100}/{H}/coadd-main-dark-{H}.fits
    """
    return (specdir / "healpix" / "main" / "dark"
            / str(healpix // 100) / str(healpix)
            / f"coadd-main-dark-{healpix}.fits")


def _load_hcd_targetids(hcd_cat_path: Path, *, tid_col: str, nhi_col: str,
                        min_nhi: float) -> set[int]:
    """Return TARGETIDs whose maximum logNHI across DLA slots is ≥ min_nhi.

    The catalog can be FITS or HDF5. For FITS we read with astropy.Table
    (one row per absorber; multiple absorbers per TARGETID allowed).
    For HDF5 (combined.h5 from the GP-DLA pipeline) the NHI column is
    typically 2D: shape ``(num_dla_slots, num_spectra)`` (per
    ``CDDF_analysis/qso_loader.py``: ``self.processed_file['MAP_log_nhis'][()].T``).
    We take the max NHI across slots for each spectrum and compare to
    ``min_nhi``.
    """
    p = Path(hcd_cat_path)
    if p.suffix.lower() in (".h5", ".hdf5"):
        with h5py.File(p, "r") as f:
            # Be lenient about case in HDF5 dataset names.
            keys = {k.lower(): k for k in f.keys()}
            tid_key = keys.get(tid_col.lower(), tid_col)
            nhi_key = keys.get(nhi_col.lower(), nhi_col)
            if tid_key not in f or nhi_key not in f:
                raise KeyError(
                    f"{p} does not contain {tid_col!r} or {nhi_col!r}; "
                    f"available: {list(f.keys())[:20]}"
                )
            tids = np.asarray(f[tid_key][()]).reshape(-1).astype(np.int64)
            nhi_raw = np.asarray(f[nhi_key][()]).astype(np.float64)
        # Reduce 2D (slots × spectra OR spectra × slots) to 1D per-spectrum max.
        if nhi_raw.ndim == 2:
            # Match against tid length to figure orientation.
            if nhi_raw.shape[0] == tids.size:
                per_spec_max = np.nanmax(nhi_raw, axis=1)  # spectra × slots
            elif nhi_raw.shape[1] == tids.size:
                per_spec_max = np.nanmax(nhi_raw, axis=0)  # slots × spectra
            else:
                raise ValueError(
                    f"{nhi_key} 2D shape {nhi_raw.shape} does not match "
                    f"len({tid_key})={tids.size}"
                )
            print(f"[hcd] {p.name}: {nhi_key} is 2D shape {nhi_raw.shape}; "
                  f"reduced to per-spectrum max")
        else:
            per_spec_max = nhi_raw.reshape(-1)
            if per_spec_max.size != tids.size:
                raise ValueError(
                    f"{nhi_key} length {per_spec_max.size} != "
                    f"{tid_key} length {tids.size}"
                )
        # Replace -inf / NaN with -inf so threshold compares correctly.
        per_spec_max = np.where(np.isfinite(per_spec_max), per_spec_max, -np.inf)
        bad_mask = per_spec_max >= min_nhi
        bad_tids = set(int(x) for x in tids[bad_mask])
        print(f"[hcd] {p.name}: {tids.size} spectra, "
              f"{bad_mask.sum()} with max logNHI ≥ {min_nhi} → {len(bad_tids)} unique TIDs to exclude")
    else:
        t = Table.read(str(p))
        if tid_col not in t.colnames or nhi_col not in t.colnames:
            raise KeyError(
                f"{p} does not have columns {tid_col!r} / {nhi_col!r}; "
                f"available: {t.colnames[:20]}"
            )
        tids = np.asarray(t[tid_col]).astype(np.int64)
        nhis = np.asarray(t[nhi_col]).astype(np.float64)
        keep = np.isfinite(nhis) & (nhis >= min_nhi)
        bad_tids = set(int(x) for x in tids[keep])
        print(f"[hcd] {p.name}: {len(t)} rows, {keep.sum()} with logNHI ≥ {min_nhi} "
              f"→ {len(bad_tids)} unique TIDs to exclude")
    return bad_tids


def _read_one_coadd_file(specfile: Path, target_ids: list[int]):
    """Read multiple TARGETIDs from one coadd file.

    Real LOA coadds have aligned bands → ``coadd_cameras`` succeeds without
    fallback. We still keep the truth-resolution fallback path for safety.
    """
    import fitsio
    from desispec.io import read_spectra
    from desispec.coaddition import coadd_cameras, resample_spectra_lin_or_log

    spectra = read_spectra(str(specfile), targetids=target_ids)

    coadd_succeeded = False
    band = "brz"
    try:
        spectra_co = coadd_cameras(spectra)
        if "brz" in spectra_co.wave or "b" in spectra_co.wave:
            spectra = spectra_co
            band = "brz" if "brz" in spectra.wave else list(spectra.wave.keys())[0]
            coadd_succeeded = True
    except Exception:
        coadd_succeeded = False

    if not coadd_succeeded:
        if spectra.resolution_data is None:
            return []
        wave_min = float(np.min(spectra.wave["b"]))
        wave_max = float(np.max(spectra.wave["z"]))
        spectra = resample_spectra_lin_or_log(
            spectra, linear_step=0.8,
            wave_min=wave_min, wave_max=wave_max, fast=True,
        )
        spectra = coadd_cameras(spectra)
        band = "brz" if "brz" in spectra.wave else list(spectra.wave.keys())[0]

    wave = spectra.wave[band].astype(np.float64)
    flux = spectra.flux[band].astype(np.float64)
    ivar = spectra.ivar[band].astype(np.float64)
    mask = spectra.mask[band].astype(bool)
    fibermap_tids = np.asarray(spectra.fibermap["TARGETID"])

    out = []
    for tid in target_ids:
        idx = np.where(fibermap_tids == tid)[0]
        if idx.size == 0:
            continue
        i = int(idx[0])
        out.append((tid, wave, flux[i], ivar[i], mask[i]))
    return out


def _to_noise_variance(ivar: np.ndarray) -> np.ndarray:
    nv = np.full_like(ivar, np.nan)
    good = (ivar > 0) & np.isfinite(ivar)
    nv[good] = 1.0 / ivar[good]
    return nv


def _interpolate_to_rest_grid(wave_obs, flux, noise_variance, z_qso,
                              rest_grid: np.ndarray):
    rest_wave = wave_obs / (1.0 + z_qso)
    valid = np.isfinite(wave_obs) & np.isfinite(flux) & np.isfinite(noise_variance)
    if valid.sum() < 50:
        return None, None
    f_interp = interp1d(rest_wave[valid], flux[valid], bounds_error=False,
                        fill_value=np.nan, kind="linear")
    nv_interp = interp1d(rest_wave[valid], noise_variance[valid], bounds_error=False,
                         fill_value=np.nan, kind="linear")
    return f_interp(rest_grid), nv_interp(rest_grid)


def _compute_redsnr(wave_obs, flux, noise_variance, z_qso,
                    rest_min=1425.0, rest_max=1475.0):
    rest = wave_obs / (1.0 + z_qso)
    region = (rest >= rest_min) & (rest <= rest_max)
    region &= np.isfinite(flux) & (noise_variance > 0)
    if region.sum() < 5:
        return 0.0
    snr = flux[region] / np.sqrt(noise_variance[region])
    return float(np.median(snr[np.isfinite(snr)])) if np.any(np.isfinite(snr)) else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--qsocat", required=True, type=Path,
                   help="QSO catalog with TARGETID, Z, ZWARN, HPXPIXEL, BI_CIV")
    p.add_argument("--specdir", required=True, type=Path,
                   help="LOA spectro redux dir (parent of healpix/main/dark/)")
    p.add_argument("--output", required=True, type=Path)
    # Filtering
    p.add_argument("--z-min", type=float, default=2.0,
                   help="Min QSO redshift; default 2.0 (Lyα at λ_obs ≈ 3650 Å)")
    p.add_argument("--z-max", type=float, default=4.25)
    p.add_argument("--max-spectra", type=int, default=None)
    p.add_argument("--exclude-bal", action="store_true",
                   help="Exclude TARGETIDs with BI_CIV > --bal-min")
    p.add_argument("--bal-col", default="BI_CIV")
    p.add_argument("--bal-min", type=float, default=0.0,
                   help="Exclude rows where BAL_COL > this (default 0)")
    # HCD anti-join
    p.add_argument("--hcd-cat", default=None, type=Path,
                   help="External catalog of HCDs (FITS or HDF5)")
    p.add_argument("--hcd-tid-col", default="TARGETID")
    p.add_argument("--hcd-nhi-col", default="LOG_NHI")
    p.add_argument("--hcd-min-nhi", type=float, default=20.3,
                   help="Exclude TARGETIDs with any HCD logNHI ≥ this. "
                        "20.3 = DLA-only filter; 17.0 = any HCD")
    # Rest-frame grid
    p.add_argument("--min-lambda", type=float, default=850.75)
    p.add_argument("--max-lambda", type=float, default=1420.75)
    p.add_argument("--dlambda", type=float, default=0.15)
    p.add_argument("--max-noise-variance", type=float, default=9.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    if not args.qsocat.exists():
        sys.exit(f"[error] qsocat not found: {args.qsocat}")
    if not args.specdir.exists():
        sys.exit(f"[error] specdir not found: {args.specdir}")

    # ============================================================
    # FILTER PIPELINE
    # ============================================================
    print(f"[step 1/5] reading {args.qsocat}")
    qcat = Table.read(args.qsocat)
    n_total = len(qcat)
    print(f"[filter 0] full catalog                                  : "
          f"{n_total:>10d} rows")

    for col in ("TARGETID", "Z", "HPXPIXEL"):
        if col not in qcat.colnames:
            sys.exit(f"[error] qsocat missing required column: {col}")

    # ---- (1) z + ZWARN filter ----
    z_mask = (qcat["Z"] >= args.z_min) & (qcat["Z"] <= args.z_max)
    n_after_z = int(z_mask.sum())
    print(f"[filter 1] z in [{args.z_min:.2f}, {args.z_max:.2f}]"
          f"{'                       ' if 'ZWARN' not in qcat.colnames else ''}"
          f"{'  + ZWARN==0' if 'ZWARN' in qcat.colnames else ''}"
          f": {n_after_z:>10d} rows  ({n_total - n_after_z:>10d} dropped)")
    if "ZWARN" in qcat.colnames:
        z_mask &= (qcat["ZWARN"] == 0)
        n_after_zwarn = int(z_mask.sum())
        if n_after_zwarn != n_after_z:
            print(f"[filter 1]   (ZWARN==0 dropped a further "
                  f"{n_after_z - n_after_zwarn} rows from the z-cut subset)")
    keep = z_mask.copy()

    # ---- (2) BAL anti-join (in-catalog) ----
    if args.exclude_bal:
        if args.bal_col not in qcat.colnames:
            sys.exit(f"[error] --exclude-bal: column {args.bal_col!r} not in qsocat")
        bal_flag = qcat[args.bal_col] > args.bal_min
        before = int(keep.sum())
        keep &= ~bal_flag
        n_after_bal = int(keep.sum())
        n_bal_dropped = before - n_after_bal
        print(f"[filter 2] exclude_bal ({args.bal_col} > {args.bal_min})            "
              f": {n_after_bal:>10d} rows  ({n_bal_dropped:>10d} dropped)")
    else:
        print(f"[filter 2] exclude_bal: SKIPPED (BALs are KEPT in this run)")

    # ---- (3) HCD anti-join (external catalog) ----
    if args.hcd_cat is not None:
        bad_tids = _load_hcd_targetids(
            args.hcd_cat, tid_col=args.hcd_tid_col, nhi_col=args.hcd_nhi_col,
            min_nhi=args.hcd_min_nhi,
        )
        in_bad = np.isin(np.asarray(qcat["TARGETID"]), list(bad_tids))
        before = int(keep.sum())
        keep &= ~in_bad
        n_after_hcd = int(keep.sum())
        n_hcd_dropped = before - n_after_hcd
        # Threshold semantics summary, for the log.
        if args.hcd_min_nhi >= 20.3:
            sem = "DLAs only"
        elif args.hcd_min_nhi >= 19.0:
            sem = "DLAs + sub-DLAs"
        else:
            sem = "DLAs + sub-DLAs + LLS"
        print(f"[filter 3] exclude_hcd (logNHI ≥ {args.hcd_min_nhi:.2f}, {sem})"
              f"{' ' * max(0, 7 - len(sem))}"
              f": {n_after_hcd:>10d} rows  ({n_hcd_dropped:>10d} dropped)")
    else:
        print(f"[filter 3] exclude_hcd: SKIPPED (no --hcd-cat given)")

    qcat = qcat[keep]

    # ---- (4) cap to --max-spectra ----
    if args.max_spectra is not None and len(qcat) > args.max_spectra:
        before = len(qcat)
        idx = rng.choice(len(qcat), size=args.max_spectra, replace=False)
        idx.sort()
        qcat = qcat[idx]
        print(f"[filter 4] random subset to --max-spectra"
              f"                  : "
              f"{len(qcat):>10d} rows  ({before - len(qcat):>10d} dropped)")
    else:
        print(f"[filter 4] cap to max-spectra: not triggered "
              f"({len(qcat)} ≤ {args.max_spectra})")
    print(f"[filter 5] FINAL catalog used for preload                 : "
          f"{len(qcat):>10d} rows")

    # 2) Group by HPXPIXEL.
    print("[step 2/5] grouping by HPXPIXEL")
    by_hpx: dict[int, list[tuple[int, float]]] = {}
    for h, tid, z in zip(qcat["HPXPIXEL"], qcat["TARGETID"], qcat["Z"]):
        by_hpx.setdefault(int(h), []).append((int(tid), float(z)))
    print(f"[step 2/5] {len(by_hpx)} unique healpix files to read")

    # 3) Build rest grid.
    n_pix = int((args.max_lambda - args.min_lambda) / args.dlambda) + 1
    rest_grid = np.linspace(args.min_lambda, args.max_lambda, n_pix)
    print(f"[step 3/5] rest grid: {n_pix} pixels in [{args.min_lambda}, {args.max_lambda}] Å")

    # 4) Read each coadd file, process spectra in-memory.
    print("[step 4/5] reading + preprocessing spectra")
    out_tids: list[int] = []
    out_z: list[float] = []
    out_flux: list[np.ndarray] = []
    out_nv: list[np.ndarray] = []
    out_snr: list[float] = []
    skipped = 0
    t_start = time.time()

    for hp_idx, (healpix, target_pairs) in enumerate(sorted(by_hpx.items())):
        specfile = _spec_path_loa(args.specdir, healpix)
        if not specfile.exists():
            skipped += len(target_pairs)
            continue
        target_ids = [tid for tid, _ in target_pairs]
        z_qsos_dict = {tid: z for tid, z in target_pairs}

        try:
            results = _read_one_coadd_file(specfile, target_ids)
        except Exception as e:
            skipped += len(target_pairs)
            print(f"[step 4/5] hpx {healpix}: read failed ({e})")
            continue

        for tid, wave, flux, ivar, mask_bool in results:
            z_qso = z_qsos_dict[tid]
            nv = _to_noise_variance(ivar)
            flux_masked = np.where(mask_bool, np.nan, flux)
            nv_masked = np.where(mask_bool, np.nan, nv)
            high_n = nv_masked > args.max_noise_variance
            flux_masked = np.where(high_n, np.nan, flux_masked)
            nv_masked = np.where(high_n, np.nan, nv_masked)

            f_interp, nv_interp = _interpolate_to_rest_grid(
                wave, flux_masked, nv_masked, z_qso, rest_grid,
            )
            if f_interp is None:
                skipped += 1
                continue

            snr = _compute_redsnr(wave, flux_masked, nv_masked, z_qso)

            out_tids.append(tid)
            out_z.append(z_qso)
            out_flux.append(f_interp.astype(np.float32))
            out_nv.append(nv_interp.astype(np.float32))
            out_snr.append(snr)

        if (hp_idx + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = len(out_tids) / max(elapsed, 1e-3)
            print(f"[step 4/5] hpx {hp_idx + 1}/{len(by_hpx)} done, "
                  f"{len(out_tids)} spectra ({rate:.1f}/s, skipped {skipped})")

    print(f"[step 4/5] done: {len(out_tids)} spectra, skipped {skipped}, "
          f"wall {(time.time() - t_start) / 60:.1f} min")
    if not out_tids:
        sys.exit("[error] no spectra processed")

    # 5) Write HDF5.
    print(f"[step 5/5] writing {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    flux_arr = np.stack(out_flux).astype(np.float32)
    nv_arr = np.stack(out_nv).astype(np.float32)
    tids_arr = np.asarray(out_tids, dtype=np.int64)
    z_arr = np.asarray(out_z, dtype=np.float32)
    snr_arr = np.asarray(out_snr, dtype=np.float32)
    rest_wavelengths_per_spec = np.tile(
        rest_grid.astype(np.float32), (len(out_tids), 1)
    )

    with h5py.File(args.output, "w") as f:
        # Legacy schema. dataset.py auto-detects this vs the newer schema.
        f.create_dataset("tids", data=tids_arr)
        f.create_dataset("rest_wavelengths", data=rest_wavelengths_per_spec, compression="gzip")
        f.create_dataset("fluxes", data=flux_arr, compression="gzip")
        f.create_dataset("noise_variance", data=nv_arr, compression="gzip")
        f.create_dataset("zqso", data=z_arr)
        f.create_dataset("redsnr", data=snr_arr)
        f.create_dataset("bluesnr", data=np.zeros_like(snr_arr))
        # Provenance
        f.attrs["qsocat"] = str(args.qsocat)
        f.attrs["specdir"] = str(args.specdir)
        f.attrs["exclude_bal"] = bool(args.exclude_bal)
        f.attrs["hcd_cat"] = str(args.hcd_cat) if args.hcd_cat else ""
        f.attrs["hcd_min_nhi"] = float(args.hcd_min_nhi) if args.hcd_cat else float("nan")
        f.attrs["min_lambda"] = float(args.min_lambda)
        f.attrs["max_lambda"] = float(args.max_lambda)
        f.attrs["dlambda"] = float(args.dlambda)
        f.attrs["z_min"] = float(args.z_min)
        f.attrs["z_max"] = float(args.z_max)

    print(f"[step 5/5] wrote {args.output} ({len(out_tids)} spectra × {n_pix} pixels)")


if __name__ == "__main__":
    main()
