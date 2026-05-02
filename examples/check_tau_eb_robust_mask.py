"""Empirical-Bayes τ_eff fit with HCD-pixel masking, per the standard
mean-flux-fitting convention.

Convention (Becker, Faucher-Giguère, etc.): when fitting τ_eff to forest
data, exclude pixels that look like high-column-density absorbers (DLAs,
sub-DLAs, LLS) — they're not "forest" and including them biases τ high.

Procedure:
  1. Compute the null-model prediction μ × A_lya at a fiducial τ_0 (use
     production 1.0× as the seed).
  2. Identify "putative HCD pixels": flux strongly below the prediction
     (residual < -N · σ_pixel, with N tuned). Mask them.
  3. With those pixels excluded, scan log L on the (NHI × τ_factor) grid
     to pick the EB τ_0 = argmax over τ of max-over-NHI log L.
  4. Run the full NHI MAP at the chosen τ. Compare to:
     - production (no τ search, no HCD mask)
     - EB without HCD masking (naive)
     - EB with HCD masking (this script)

If HCD masking shifts the EB τ_0 (and hence the NHI MAP), the convention
matters for production. If not, the naive EB is fine.

Usage::
    python examples/check_tau_eb_robust_mask.py \\
        --target-id 120046865 --truth-z 2.7730 --truth-log-nhi 21.263 \\
        --spec /path/to/spectra-16-789.fits \\
        --zcat /path/to/zcat.fits
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--target-id", type=int, required=True)
    p.add_argument("--spec", type=str, required=True)
    p.add_argument("--zcat", type=str, required=True)
    p.add_argument("--truth-z", type=float, required=True)
    p.add_argument("--truth-log-nhi", type=float, required=True)
    p.add_argument("--data-root", default=os.environ.get(
        "DATA_ROOT",
        "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection",
    ))
    p.add_argument("--tau-factors", type=float, nargs="+",
                   default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    p.add_argument("--mask-threshold-sigma", type=float, default=3.0,
                   help="Pixels with (flux - μ_pred) / σ < -N are masked")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from gpy_dla_detection.voigt_v2_inject import inject
    inject(kernel="boss-log-r2000", num_lines=3)

    from examples.smoke_one_spectrum import (
        load_one_desi_spectrum, lookup_z_qso, PRESETS,
    )
    from gpy_dla_detection.set_parameters import Parameters
    from gpy_dla_detection.dla_gp import DLAGPMAT
    from gpy_dla_detection.null_gp import NullGPMAT
    from gpy_dla_detection.model_priors import PriorCatalog
    from gpy_dla_detection.dla_samples import DLASamplesMAT

    wave, flux, nv, mask_orig = load_one_desi_spectrum(args.spec, args.target_id)
    z_qso = lookup_z_qso(args.zcat, args.target_id)
    print(f"target={args.target_id} z_qso={z_qso:.4f} truth: z={args.truth_z}, "
          f"logNHI={args.truth_log_nhi:.3f}")

    preset = PRESETS["y3"]
    common = dict(
        loading_min_lambda=preset.loading_min_lambda,
        loading_max_lambda=preset.loading_max_lambda,
        normalization_min_lambda=preset.normalization_min_lambda,
        normalization_max_lambda=preset.normalization_max_lambda,
        min_lambda=preset.min_lambda, max_lambda=preset.max_lambda,
        dlambda=preset.dlambda, k=preset.k,
        max_noise_variance=9.0, num_lines=3,
        max_z_cut=3000.0, min_z_cut=3000.0,
        num_forest_lines=preset.num_forest_lines,
    )
    params = Parameters(num_dla_samples=100000, **common)
    learned = os.path.join(args.data_root, preset.learned_file)
    catalog = os.path.join(args.data_root, "data/dr12q/processed/catalog.mat")
    los_cat = os.path.join(args.data_root, "data/dla_catalogs/dr9q_concordance/processed/los_catalog")
    dla_cat = os.path.join(args.data_root, "data/dla_catalogs/dr9q_concordance/processed/dla_catalog")
    dla_samples_file = os.path.join(args.data_root, "data/dr12q/processed/dla_samples_a03_100000.mat")
    prior = PriorCatalog(params, catalog, los_cat, dla_cat)
    dla_samples = DLASamplesMAT(params, prior, dla_samples_file)

    # Step 1: build the null model at production τ to find putative HCD pixels.
    null_gp = NullGPMAT(params, prior, learned_file=learned,
                       prev_tau_0=preset.prev_tau_0, prev_beta=preset.prev_beta)
    rest_w_seed = params.emitted_wavelengths(wave, z_qso)
    null_gp.set_data(np.atleast_2d(rest_w_seed), np.atleast_2d(flux),
                     np.atleast_2d(nv), np.atleast_2d(mask_orig),
                     np.array([z_qso]), build_model=True)

    pred = null_gp.this_mu                        # μ × A_lya on the working grid
    y    = null_gp.y                              # data on the same grid
    sigma2 = null_gp.this_omega2 + null_gp.v      # total per-pixel variance
    residuals_sigma = (y - pred) / np.sqrt(sigma2)
    print(f"\n[residuals diagnostics]")
    print(f"  residuals/σ:  min={residuals_sigma.min():.2f}  "
          f"median={np.median(residuals_sigma):.2f}  "
          f"max={residuals_sigma.max():.2f}")
    print(f"  μ_pred range:    [{pred.min():.3f}, {pred.max():.3f}]")
    print(f"  y range:         [{y.min():.3f}, {y.max():.3f}]")

    # Putative HCD: pixels with very negative normalised residual (data << model).
    hcd_mask = residuals_sigma < -args.mask_threshold_sigma
    n_hcd = int(hcd_mask.sum())
    n_total = int(y.size)
    print(f"\n[HCD mask] threshold = {args.mask_threshold_sigma}σ")
    print(f"  pixels: {n_total} total, {n_hcd} flagged as HCD ({n_hcd/n_total*100:.1f}%)")

    # The pixel_mask passed to set_data is over the *full-grid* (loaded
    # wavelengths). We need to translate hcd_mask (which is over the
    # unmasked-then-pixel-masked subset) back to the full grid.
    full_grid_unmasked_pixel = ~mask_orig
    # Indices into the unmasked range that correspond to ind_unmasked.
    full_idx_in_range = np.flatnonzero(null_gp.ind_unmasked)
    # Among those, the ones that survived mask_orig:
    survived_pixel_mask = ~mask_orig[full_idx_in_range]
    # `y` (and pred, residuals) is over the survived subset:
    full_idx_used = full_idx_in_range[survived_pixel_mask]
    # Add HCD pixels (which are indexed within the y/pred subset) to the full mask:
    new_mask = mask_orig.copy()
    new_mask[full_idx_used[hcd_mask]] = True

    # Step 2: τ_eff fit grid on (a) original mask, (b) HCD-masked.
    nhi_grid = np.arange(20.30, 22.01, 0.025)
    tau_factors = np.array(args.tau_factors)
    print(f"\nNHI grid {nhi_grid[0]:.2f}–{nhi_grid[-1]:.2f} step {0.025} ({len(nhi_grid)} pts), "
          f"τ-grid {list(tau_factors)} (K={len(tau_factors)})")

    def scan(masked: np.ndarray, label: str):
        log_l = np.full((len(tau_factors), len(nhi_grid)), np.nan)
        for j, tf in enumerate(tau_factors):
            tau0 = preset.prev_tau_0 * tf
            dla_gp = DLAGPMAT(params, prior, dla_samples,
                min_z_separation=3000.0, learned_file=learned,
                broadening=True, prev_tau_0=tau0, prev_beta=preset.prev_beta)
            rest_w = params.emitted_wavelengths(wave, z_qso)
            dla_gp.set_data(np.atleast_2d(rest_w), np.atleast_2d(flux),
                            np.atleast_2d(nv), np.atleast_2d(masked),
                            np.array([z_qso]), build_model=True)
            for i, ln in enumerate(nhi_grid):
                try:
                    log_l[j, i] = dla_gp.sample_log_likelihood_k_dlas(
                        np.array([args.truth_z]), np.array([10**ln]))
                except Exception:
                    pass
        max_per_tau = np.nanmax(log_l, axis=1)
        j_best = int(np.argmax(max_per_tau))
        i_best = int(np.argmax(log_l[j_best]))
        return {
            "label": label,
            "tau_best": tau_factors[j_best],
            "max_log_l": log_l[j_best, i_best],
            "map_nhi": nhi_grid[i_best],
            "log_l_at_truth": float(np.interp(args.truth_log_nhi, nhi_grid, log_l[j_best])),
            "log_l_full": log_l,
        }

    print(f"\n[scanning naive (no HCD mask)]")
    naive = scan(mask_orig, "EB naive")
    print(f"[scanning HCD-masked]")
    masked = scan(new_mask, f"EB + HCD-mask ({args.mask_threshold_sigma}σ)")

    # Production: τ=1.0, original mask
    j_prod = int(np.argmin(np.abs(tau_factors - 1.0)))
    L_prod_curve = naive["log_l_full"][j_prod]
    map_prod = nhi_grid[int(np.nanargmax(L_prod_curve))]

    print()
    print(f"{'treatment':<32} {'tau best':>9} {'MAP NHI':>9} {'bias':>8} "
          f"{'log L (truth)':>14}")
    print("-" * 80)
    print(f"{'(1) production τ=1.0':<32} {1.0:>9.2f} {map_prod:>9.3f} "
          f"{map_prod - args.truth_log_nhi:>+8.3f} {float(np.interp(args.truth_log_nhi, nhi_grid, L_prod_curve)):>14.2f}")
    for s in [naive, masked]:
        print(f"{'(2) ' + s['label']:<32} {s['tau_best']:>9.2f} {s['map_nhi']:>9.3f} "
              f"{s['map_nhi'] - args.truth_log_nhi:>+8.3f} {s['log_l_at_truth']:>14.2f}")

    print()
    delta_naive = naive["map_nhi"] - map_prod
    delta_masked = masked["map_nhi"] - map_prod
    if abs(masked["map_nhi"] - args.truth_log_nhi) < abs(naive["map_nhi"] - args.truth_log_nhi) - 0.02:
        print(f"  ⇒ HCD masking improves EB: naive MAP={naive['map_nhi']:.3f} → "
              f"masked MAP={masked['map_nhi']:.3f}  ({masked['map_nhi'] - naive['map_nhi']:+.3f} dex)")
    elif abs(masked["map_nhi"] - args.truth_log_nhi) > abs(naive["map_nhi"] - args.truth_log_nhi) + 0.02:
        print(f"  ⇒ HCD masking made EB WORSE on this target.")
    else:
        print(f"  ⇒ HCD masking didn't shift EB MAP meaningfully on this target "
              f"({masked['map_nhi'] - naive['map_nhi']:+.3f} dex).")
    if naive["tau_best"] != masked["tau_best"]:
        print(f"  ⇒ τ_best shifted: {naive['tau_best']} → {masked['tau_best']} "
              f"under HCD masking (the convention matters for τ).")
    else:
        print(f"  ⇒ τ_best unchanged at {naive['tau_best']} (HCD pixels were not "
              f"a τ-fit confound here).")


if __name__ == "__main__":
    main()
