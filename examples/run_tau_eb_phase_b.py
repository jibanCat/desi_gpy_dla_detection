"""Phase B runner: full production bayes on a chunk of 2LPT spectra,
both BASELINE (no τ-EB) and ENABLED (with τ-EB). Per-spectrum output:
truth + MAP_log_NHI + bias for each treatment.

Designed to be called from a SLURM array — each task processes a slice
[start, end) of the targets TSV. Builds DLAHolder ONCE per task to
amortize prior/sample loading; toggles ``enable_tau_eb`` per call.

Usage::

    python examples/run_tau_eb_phase_b.py \\
        --targets-tsv /path/to/targets.tsv \\
        --start 0 --end 313 \\
        --out /path/to/phase_b_chunk_0.tsv

Per-spectrum cost on 16 cores: ~17 s no-DLA / ~38 s DLA × 2 treatments
≈ 35-80 s. For 313 spectra ≈ 3-7 h wall per array task.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import copy as _copy
from pathlib import Path

import numpy as np


def _build_holder(num_dla_samples: int = 10000, max_workers: int = 16,
                  max_dlas: int = 3, filter_low_likelihood: bool = False):
    """Build a single DLAHolder; we'll mutate enable_tau_eb per call."""
    sys.path.insert(0, "/home/mfho/desi_gpy_dla_detection")
    from gpy_dla_detection.voigt_v2_inject import inject
    inject(kernel="boss-log-r2000", num_lines=3)

    from examples.smoke_one_spectrum import PRESETS
    from gpy_dla_detection.set_parameters import Parameters
    from run_bayes_select import DLAHolder

    DATA_ROOT = "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection"
    p = PRESETS["y3"]
    params = Parameters(
        loading_min_lambda=p.loading_min_lambda, loading_max_lambda=p.loading_max_lambda,
        normalization_min_lambda=p.normalization_min_lambda, normalization_max_lambda=p.normalization_max_lambda,
        min_lambda=p.min_lambda, max_lambda=p.max_lambda,
        dlambda=p.dlambda, k=p.k,
        max_noise_variance=9.0, num_lines=3,
        max_z_cut=3000.0, min_z_cut=3000.0,
        num_forest_lines=p.num_forest_lines,
        num_dla_samples=num_dla_samples,
    )
    dla_samples_file = (os.path.join(DATA_ROOT, "data/dr12q/processed/dla_samples_a03.mat")
                        if num_dla_samples == 10000
                        else os.path.join(DATA_ROOT, "data/dr12q/processed/dla_samples_a03_100000.mat"))
    subdla_samples_file = (os.path.join(DATA_ROOT, "data/dr12q/processed/subdla_samples.mat")
                           if num_dla_samples == 10000
                           else os.path.join(DATA_ROOT, "data/dr12q/processed/subdla_samples_a03_191_200_100000.mat"))
    holder = DLAHolder(
        learned_file=os.path.join(DATA_ROOT, p.learned_file),
        catalog_name=os.path.join(DATA_ROOT, "data/dr12q/processed/catalog.mat"),
        los_catalog=os.path.join(DATA_ROOT, "data/dla_catalogs/dr9q_concordance/processed/los_catalog"),
        dla_catalog=os.path.join(DATA_ROOT, "data/dla_catalogs/dr9q_concordance/processed/dla_catalog"),
        dla_samples_file=dla_samples_file,
        sub_dla_samples_file=subdla_samples_file,
        params=params, params_subdla=_copy.copy(params),
        min_z_separation=3000.0,
        prev_tau_0=p.prev_tau_0, prev_beta=p.prev_beta,
        max_dlas=max_dlas, max_workers=max_workers, batch_size=313,
        filter_low_likelihood=filter_low_likelihood,
        enable_tau_eb=False,  # start in BASELINE; we'll toggle
        # Default grid pulled from DLAHolder.__init__ (= 0.5..6.0).
        tau_eb_apply_hcd_mask=False,
        tau_eb_objective="null",
    )
    return holder, p


def _process_one(holder, preset, row: dict) -> dict:
    from examples.smoke_one_spectrum import load_one_desi_spectrum, lookup_z_qso

    out = dict(target_id=int(row["target_id"]),
               z_qso=float(row["z_qso"]),
               truth_log_nhi=float(row["truth_log_nhi"]),
               nhi_regime=row["nhi_regime"],
               status="error",
               # baseline
               baseline_p_dla=-1.0, baseline_map_log_nhi=-1.0, baseline_wall_s=-1.0,
               # enabled (τ-EB)
               enabled_p_dla=-1.0, enabled_map_log_nhi=-1.0, enabled_wall_s=-1.0,
               error="")

    try:
        wave, flux, nv, mask = load_one_desi_spectrum(row["spec_path"], int(row["target_id"]))
        z_qso = lookup_z_qso(row["zcat_path"], int(row["target_id"]))

        # BASELINE
        holder.enable_tau_eb = False
        holder.initialize_results(num_spectra=1)
        t0 = time.perf_counter()
        holder.process_qso(idx=0, target_id=int(row["target_id"]),
                           wavelengths=wave, flux=flux,
                           noise_variance=nv, pixel_mask=mask, z_qso=z_qso)
        out["baseline_wall_s"] = time.perf_counter() - t0
        # MAP_log_nhis is shape (max_dlas,) for the most-probable model (or NaN)
        nhi_arr = holder.results["MAP_log_nhis"][0]
        out["baseline_map_log_nhi"] = float(np.nanmax(nhi_arr)) if np.isfinite(nhi_arr).any() else float("nan")
        out["baseline_p_dla"] = float(holder.results["p_dlas"][0])

        # ENABLED
        holder.enable_tau_eb = True
        holder.initialize_results(num_spectra=1)
        t0 = time.perf_counter()
        holder.process_qso(idx=0, target_id=int(row["target_id"]),
                           wavelengths=wave, flux=flux,
                           noise_variance=nv, pixel_mask=mask, z_qso=z_qso)
        out["enabled_wall_s"] = time.perf_counter() - t0
        nhi_arr = holder.results["MAP_log_nhis"][0]
        out["enabled_map_log_nhi"] = float(np.nanmax(nhi_arr)) if np.isfinite(nhi_arr).any() else float("nan")
        out["enabled_p_dla"] = float(holder.results["p_dlas"][0])

        out["status"] = "ok"
    except Exception as e:
        out["error"] = (type(e).__name__ + ": " + str(e))[:200]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--targets-tsv", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1, help="-1 means full file")
    p.add_argument("--out", required=True)
    p.add_argument("--max-workers", type=int, default=16,
                   help="DLAHolder.max_workers per spectrum")
    p.add_argument("--num-dla-samples", type=int, default=10000)
    p.add_argument("--max-dlas", type=int, default=3,
                   help="DLAHolder.max_dlas; production multi-DLA uses 3 or 4.")
    p.add_argument("--filter-low-likelihood", type=int, default=0,
                   help="0=FILTER off (default); 1=FILTER on (with fix #5).")
    args = p.parse_args()

    rows = []
    with open(args.targets_tsv) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            rows.append(r)
    if args.end < 0:
        args.end = len(rows)
    chunk = rows[args.start:args.end]
    print(f"[chunk] {args.start}-{args.end} → {len(chunk)} targets", flush=True)

    holder, preset = _build_holder(num_dla_samples=args.num_dla_samples,
                                   max_workers=args.max_workers,
                                   max_dlas=args.max_dlas,
                                   filter_low_likelihood=bool(args.filter_low_likelihood))
    print(f"[holder] built; max_workers={args.max_workers} "
          f"num_dla_samples={args.num_dla_samples} max_dlas={args.max_dlas} "
          f"filter={args.filter_low_likelihood}", flush=True)

    fieldnames = ["target_id", "z_qso", "truth_log_nhi", "nhi_regime", "status",
                  "baseline_p_dla", "baseline_map_log_nhi", "baseline_wall_s",
                  "enabled_p_dla", "enabled_map_log_nhi", "enabled_wall_s",
                  "error"]
    n_done = 0
    t_start = time.perf_counter()
    with open(args.out, "w") as fout:
        wtr = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
        wtr.writeheader()
        for r in chunk:
            res = _process_one(holder, preset, r)
            wtr.writerow({k: res.get(k, "") for k in fieldnames})
            fout.flush()
            n_done += 1
            if n_done % 25 == 0:
                rate = n_done / (time.perf_counter() - t_start)
                eta = (len(chunk) - n_done) / rate if rate > 0 else 0
                print(f"  {n_done}/{len(chunk)}  rate={rate:.2f} spec/s  "
                      f"ETA={eta/60:.1f} min", flush=True)
    elapsed = time.perf_counter() - t_start
    print(f"[done] {n_done} rows  wall={elapsed/60:.1f} min  "
          f"rate={n_done/elapsed:.2f} spec/s  → {args.out}")


if __name__ == "__main__":
    main()
