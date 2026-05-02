"""Test whether the +0.28 dex MAP bias on TID 120046865 is driven by τ_eff.

Per user (2026-04-29): the GP mean-flux suppression at z is set by
``prev_tau_0`` and ``prev_beta`` (Turner+2024 default 0.00246 / 3.62).
This mean-flux multiplies μ in the forest. If τ_eff is too LOW, μ is too
HIGH in the forest → the model predicts more flux than the data has →
the DLA fitter compensates by widening the trough (higher NHI) → bias.

Test: rebuild the DLA GP with τ_0 scaled by [0.5, 0.75, 1.0, 1.25, 1.5,
2.0] × production. For each, run a brute-force scan over a subset of
QMC samples and report the argmax (z_dla, log_nhi, log L). If MAP log_nhi
drifts DOWN as τ_0 increases, τ_eff bias is at least part of the story.
If it doesn't budge, τ_eff isn't the lever.

Usage::

    python examples/check_tau_eff_sensitivity.py \\
        --target-id 120046865 \\
        --spec /path/to/spectra-16-789.fits \\
        --zcat /path/to/zcat.fits \\
        --truth 2.7730 21.263 \\
        --truth 2.2871 19.407 \\
        --tau-factors 0.5 0.75 1.0 1.25 1.5 2.0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--target-id", type=int, required=True)
    p.add_argument("--spec", type=str, required=True)
    p.add_argument("--zcat", type=str, required=True)
    p.add_argument("--truth", action="append", nargs=2, metavar=("z", "log_nhi"),
                   type=float, required=True)
    p.add_argument("--tau-factors", type=float, nargs="+",
                   default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    p.add_argument("--n-scan", type=int, default=10000,
                   help="QMC samples to scan per τ value")
    p.add_argument("--kernel", default="boss-log-r2000")
    p.add_argument("--num-lines", type=int, default=3)
    p.add_argument("--data-root", default=os.environ.get(
        "DATA_ROOT",
        "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection",
    ))
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from gpy_dla_detection.voigt_v2_inject import inject
    inject(kernel=args.kernel, num_lines=args.num_lines)

    from examples.smoke_one_spectrum import (
        load_one_desi_spectrum, lookup_z_qso, PRESETS,
    )
    from gpy_dla_detection.set_parameters import Parameters
    from gpy_dla_detection.null_gp import NullGPMAT
    from gpy_dla_detection.dla_gp import DLAGPMAT
    from gpy_dla_detection.model_priors import PriorCatalog
    from gpy_dla_detection.dla_samples import DLASamplesMAT

    wave, flux, noise_var, mask = load_one_desi_spectrum(args.spec, args.target_id)
    z_qso = lookup_z_qso(args.zcat, args.target_id)

    preset = PRESETS["y3"]
    common = dict(
        loading_min_lambda=preset.loading_min_lambda,
        loading_max_lambda=preset.loading_max_lambda,
        normalization_min_lambda=preset.normalization_min_lambda,
        normalization_max_lambda=preset.normalization_max_lambda,
        min_lambda=preset.min_lambda, max_lambda=preset.max_lambda,
        dlambda=preset.dlambda, k=preset.k,
        max_noise_variance=9.0, num_lines=args.num_lines,
        max_z_cut=3000.0, min_z_cut=3000.0,
        num_forest_lines=preset.num_forest_lines,
    )
    params = Parameters(num_dla_samples=100000, **common)
    learned = os.path.join(args.data_root, preset.learned_file)
    catalog = os.path.join(args.data_root, "data/dr12q/processed/catalog.mat")
    los_cat = os.path.join(args.data_root,
                           "data/dla_catalogs/dr9q_concordance/processed/los_catalog")
    dla_cat = os.path.join(args.data_root,
                           "data/dla_catalogs/dr9q_concordance/processed/dla_catalog")
    dla_samples_file = os.path.join(args.data_root,
                                    "data/dr12q/processed/dla_samples_a03_100000.mat")

    prior = PriorCatalog(params, catalog, los_cat, dla_cat)
    dla_samples = DLASamplesMAT(params, prior, dla_samples_file)

    # Truth
    truth_z, truth_n = args.truth[0]  # use the strongest absorber as primary
    print(f"[setup] target={args.target_id}  z_qso={z_qso:.4f}")
    print(f"[truth-1] z={truth_z:.4f}  log_nhi={truth_n:.3f}  (strongest, primary test)")
    print(f"[Turner+2024 default] τ_0 = {preset.prev_tau_0}  β = {preset.prev_beta}")

    rng = np.random.default_rng(42)

    # Pre-pick the same scan indices for all τ values (fair comparison).
    n_total = 100000
    scan_idx = rng.choice(n_total, size=args.n_scan, replace=False)

    # Always include the truth NHI exactly as one of the indices to scan.
    closest_truth = int(np.argmin(np.abs(dla_samples.log_nhi_samples - truth_n)))
    if closest_truth not in scan_idx:
        scan_idx = np.append(scan_idx, closest_truth)

    print(f"\n[scanning] {len(scan_idx)} QMC samples × {len(args.tau_factors)} τ "
          f"settings\n")
    print(f"{'tau_factor':>10} {'tau_0':>10} {'best_z':>9} {'best_logNHI':>12} "
          f"{'best_logL':>12} {'logL@truth':>12} {'Δ(MAP-truth)':>14}")
    print("-" * 90)

    results = []
    for tau_factor in args.tau_factors:
        prev_tau_0 = preset.prev_tau_0 * tau_factor
        # Rebuild DLA GP with this τ_0
        dla_gp = DLAGPMAT(
            params, prior, dla_samples,
            min_z_separation=3000.0, learned_file=learned,
            broadening=True, prev_tau_0=prev_tau_0, prev_beta=preset.prev_beta,
        )
        rest_w = params.emitted_wavelengths(wave, z_qso)
        dla_gp.set_data(np.atleast_2d(rest_w), np.atleast_2d(flux),
                        np.atleast_2d(noise_var), np.atleast_2d(mask),
                        np.array([z_qso]), build_model=True)
        z_grid = dla_gp.dla_samples.sample_z_dlas(dla_gp.this_wavelengths, z_qso)
        nhi_grid = dla_samples.log_nhi_samples

        scan_log_l = np.full(len(scan_idx), np.nan)
        for j, idx in enumerate(scan_idx):
            try:
                scan_log_l[j] = dla_gp.sample_log_likelihood_k_dlas(
                    np.array([z_grid[idx]]), np.array([10**nhi_grid[idx]])
                )
            except Exception:
                pass

        valid = ~np.isnan(scan_log_l)
        best_local = int(np.argmax(np.where(valid, scan_log_l, -np.inf)))
        best_idx = scan_idx[best_local]
        L_best = scan_log_l[best_local]
        z_best = z_grid[best_idx]
        n_best = nhi_grid[best_idx]

        # Also compute log L at exact truth
        L_truth = dla_gp.sample_log_likelihood_k_dlas(
            np.array([truth_z]), np.array([10**truth_n])
        )

        results.append((tau_factor, prev_tau_0, z_best, n_best, L_best, L_truth))
        print(f"{tau_factor:10.3f} {prev_tau_0:10.5f} {z_best:9.4f} "
              f"{n_best:12.3f} {L_best:12.2f} {L_truth:12.2f} "
              f"{n_best - truth_n:+14.3f}")

    print()
    # Summarise: which τ_factor gives MAP closest to truth?
    deltas = [abs(n - truth_n) for _, _, _, n, _, _ in results]
    best = int(np.argmin(deltas))
    tf, t0, zb, nb, lb, lt = results[best]
    print(f"Best-fit (MAP closest to truth): tau_factor={tf}, log_nhi={nb:.3f}, "
          f"|Δ| = {deltas[best]:.3f}")

    # And: which gives highest log L at truth (truth fits best)?
    best_l = max(range(len(results)), key=lambda i: results[i][5])
    tf2, t02, _, _, lb2, lt2 = results[best_l]
    print(f"Best (truth-fit): tau_factor={tf2}  log L(truth) = {lt2:.2f}")
    print()
    print("Interpretation:")
    print("  - If MAP log_nhi drifts DOWN as tau_factor INCREASES → mean-flux")
    print("    is the (or a) lever. Higher τ_eff suppresses μ in the forest,")
    print("    giving the optimizer less continuum to compensate over.")
    print("  - If MAP log_nhi stays constant → τ_eff isn't moving it.")
    print("    Bias is from elsewhere (continuum μ shape, ω, mock physics, ...).")


if __name__ == "__main__":
    main()
