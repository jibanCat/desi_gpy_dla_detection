"""Phase A: run the production τ-EB module on a representative sample
(no cherry-picking, no full bayes step). Just the τ-fit step + τ_best
choice. Validates that the recipe makes sensible τ choices on the
distribution of spectra production will see.

Output: TSV with one row per spectrum (target_id, z_qso, truth_log_nhi,
n_pix, n_hcd_at_1.5sig, tau_factor_best_null, tau_factor_best_with_mask).

Run:

    python examples/run_tau_eb_phase_a.py \\
        --targets-tsv /tmp/random_2lpt_5k.tsv \\
        --out /tmp/tau_eb_phase_a.tsv \\
        --jobs 16
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def _process_one(row: dict) -> dict:
    """Run τ-fit on one TID. Returns a dict of the chosen tau (no-mask) and
    diagnostic info (no full bayes — Phase A is fast)."""
    sys.path.insert(0, "/home/mfho/desi_gpy_dla_detection")
    # Inject the right voigt kernel inside the worker (one-time per process)
    from gpy_dla_detection.voigt_v2_inject import inject
    inject(kernel="boss-log-r2000", num_lines=3)

    from examples.smoke_one_spectrum import (
        load_one_desi_spectrum, lookup_z_qso, PRESETS,
    )
    from gpy_dla_detection.set_parameters import Parameters
    from gpy_dla_detection.model_priors import PriorCatalog
    from gpy_dla_detection.dla_samples import DLASamplesMAT
    from gpy_dla_detection.tau_eb import fit_tau_eb

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
        num_dla_samples=10000,
    )
    learned = os.path.join(DATA_ROOT, p.learned_file)
    catalog = os.path.join(DATA_ROOT, "data/dr12q/processed/catalog.mat")
    los_cat = os.path.join(DATA_ROOT, "data/dla_catalogs/dr9q_concordance/processed/los_catalog")
    dla_cat = os.path.join(DATA_ROOT, "data/dla_catalogs/dr9q_concordance/processed/dla_catalog")
    dla_samples_file = os.path.join(DATA_ROOT, "data/dr12q/processed/dla_samples_a03.mat")

    out = dict(target_id=int(row["target_id"]),
               z_qso=float(row["z_qso"]),
               truth_log_nhi=float(row["truth_log_nhi"]),
               nhi_regime=row["nhi_regime"],
               status="error", n_pix=-1, n_hcd=-1,
               tau_factor_null=-1.0, tau_factor_mask=-1.0,
               wall_s=-1.0, error="")

    t0 = time.perf_counter()
    try:
        wave, flux, nv, mask = load_one_desi_spectrum(row["spec_path"], int(row["target_id"]))
        z_qso = lookup_z_qso(row["zcat_path"], int(row["target_id"]))
        rest_w = params.emitted_wavelengths(wave, z_qso)
        prior = PriorCatalog(params, catalog, los_cat, dla_cat)
        dla_samples = DLASamplesMAT(params, prior, dla_samples_file)

        # No HCD mask (production default): cheap K=6 null builds
        tau_null, info_null = fit_tau_eb(
            params=params, prior=prior, learned_file=learned,
            rest_wavelengths=rest_w, flux=flux, noise_variance=nv,
            pixel_mask=mask, z_qso=z_qso,
            prev_tau_0_seed=p.prev_tau_0, prev_beta=p.prev_beta,
            tau_factors=tuple(row.get("_tau_factors",
                              (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0))),
            apply_hcd_mask=False, objective="null",
        )

        # WITH HCD mask: same grid, just extra masking step
        tau_mask, info_mask = fit_tau_eb(
            params=params, prior=prior, learned_file=learned,
            rest_wavelengths=rest_w, flux=flux, noise_variance=nv,
            pixel_mask=mask, z_qso=z_qso,
            prev_tau_0_seed=p.prev_tau_0, prev_beta=p.prev_beta,
            tau_factors=tuple(row.get("_tau_factors",
                              (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0))),
            apply_hcd_mask=True,
            mask_threshold_sigma=float(row.get("_hcd_threshold", 1.5)),
            objective="null",
        )
        out.update(
            status="ok",
            n_pix=len(flux),
            n_hcd=int(info_mask["n_hcd"]),
            tau_factor_null=float(info_null["tau_factor_best"]),
            tau_factor_mask=float(info_mask["tau_factor_best"]),
            wall_s=time.perf_counter() - t0,
        )
    except Exception as e:
        out["error"] = str(e)[:200]
        out["wall_s"] = time.perf_counter() - t0
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--targets-tsv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap the number processed (0 = all).")
    p.add_argument("--tau-factors", type=float, nargs="+", default=None,
                   help="Override τ_factor grid. Default: "
                        "(0.5,0.75,1.0,1.25,1.5,2.0,3.0,4.0).")
    p.add_argument("--hcd-threshold", type=float, default=1.5,
                   help="HCD-mask threshold σ for the mask-on column.")
    args = p.parse_args()
    if args.tau_factors:
        print(f"[grid] override τ_factors = {tuple(args.tau_factors)}")
    print(f"[hcd_threshold] mask-on column will use σ={args.hcd_threshold}")

    rows = []
    with open(args.targets_tsv) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            if args.tau_factors:
                r["_tau_factors"] = tuple(args.tau_factors)
            r["_hcd_threshold"] = args.hcd_threshold
            rows.append(r)
    if args.limit:
        rows = rows[:args.limit]
    print(f"[in] {len(rows)} targets from {args.targets_tsv}")

    fieldnames = ["target_id", "z_qso", "truth_log_nhi", "nhi_regime",
                  "status", "n_pix", "n_hcd",
                  "tau_factor_null", "tau_factor_mask",
                  "wall_s", "error"]

    n_done = 0
    t_start = time.perf_counter()
    with open(args.out, "w") as fout:
        wtr = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
        wtr.writeheader()
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = [ex.submit(_process_one, r) for r in rows]
            for fut in as_completed(futures):
                res = fut.result()
                wtr.writerow({k: res.get(k, "") for k in fieldnames})
                fout.flush()
                n_done += 1
                if n_done % 200 == 0:
                    rate = n_done / (time.perf_counter() - t_start)
                    eta = (len(rows) - n_done) / rate if rate > 0 else 0
                    print(f"  {n_done}/{len(rows)}  rate={rate:.1f} spec/s  ETA={eta/60:.1f} min")
    elapsed = time.perf_counter() - t_start
    print(f"[done] {n_done} rows  wall={elapsed/60:.1f} min  "
          f"rate={n_done/elapsed:.1f} spec/s  → {args.out}")


if __name__ == "__main__":
    main()
