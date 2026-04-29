"""Compute log-likelihood at truth (z_dla, log_nhi) vs at the QMC MAP.

Per user suggestion (2026-04-29): if log L(truth) > log L(MAP), the bias is
**sampler-limited** — the QMC samples don't include a point close enough to
truth to be the argmax. If log L(truth) ≤ log L(MAP), the QMC found the
true posterior peak; any bias is from the forward model, prior, or
continuum, not the sampler.

This generalizes to multi-DLA + joint LLS/sub-DLA targets — pass the full
list of truth absorbers via ``--truth z log_nhi`` (repeat per absorber).

Usage::

    python examples/check_truth_vs_map_likelihood.py \\
        --target-id 120046865 \\
        --spec /nfs/turbo/.../spectra-16-789.fits \\
        --zcat /nfs/turbo/.../zcat.fits \\
        --truth 2.7730 21.263 \\
        --truth 2.2871 19.407 \\
        --kernel boss-log-r2000  # or 'desi-linear-r3000' / 'none'
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
                   type=float, required=True,
                   help="One absorber per --truth, repeatable")
    p.add_argument("--kernel", default="boss-log-r2000",
                   choices=["boss-log-r2000", "desi-linear-r3000",
                            "desi-linear-r5000", "linear-r2000", "none"])
    p.add_argument("--num-lines", type=int, default=3)
    p.add_argument("--num-dla-samples", type=int, default=100000)
    p.add_argument("--data-root", default=os.environ.get(
        "DATA_ROOT",
        "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection",
    ))
    p.add_argument("--max-dlas", type=int, default=4)
    p.add_argument("--single-absorber", action="store_true",
                   help="Use the LLS-mode single-absorber model")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    # Inject the requested kernel before any GP imports.
    from gpy_dla_detection.voigt_v2_inject import inject
    inject(kernel=args.kernel, num_lines=args.num_lines)

    from examples.smoke_one_spectrum import (
        load_one_desi_spectrum, lookup_z_qso, PRESETS,
    )
    from gpy_dla_detection.set_parameters import Parameters
    from run_bayes_select import DLAHolder

    wave, flux, noise_var, mask = load_one_desi_spectrum(args.spec, args.target_id)
    z_qso = lookup_z_qso(args.zcat, args.target_id)
    print(f"[setup] target={args.target_id}  z_qso={z_qso:.4f}  kernel={args.kernel}")
    print(f"[truth] {len(args.truth)} absorber(s):")
    for z_t, n_t in args.truth:
        print(f"        z_dla={z_t:.4f}  log_nhi={n_t:.3f}")

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
    params = Parameters(num_dla_samples=args.num_dla_samples, **common)
    params_subdla = Parameters(num_dla_samples=args.num_dla_samples, **common)

    learned_file = os.path.join(args.data_root, preset.learned_file)
    catalog_name = os.path.join(args.data_root, "data/dr12q/processed/catalog.mat")
    los_catalog = os.path.join(args.data_root,
                               "data/dla_catalogs/dr9q_concordance/processed/los_catalog")
    dla_catalog = os.path.join(args.data_root,
                               "data/dla_catalogs/dr9q_concordance/processed/dla_catalog")
    dla_samples_file = os.path.join(args.data_root,
                                    "data/dr12q/processed/dla_samples_a03_100000.mat")
    sub_dla_samples_file = os.path.join(args.data_root,
                                        "data/dr12q/processed/subdla_samples_a03_191_200_100000.mat")

    holder = DLAHolder(
        learned_file=learned_file, catalog_name=catalog_name,
        los_catalog=los_catalog, dla_catalog=dla_catalog,
        dla_samples_file=dla_samples_file,
        sub_dla_samples_file=sub_dla_samples_file,
        params=params, params_subdla=params_subdla,
        min_z_separation=3000.0, prev_tau_0=preset.prev_tau_0,
        prev_beta=preset.prev_beta,
        max_dlas=args.max_dlas, broadening=True,
        plot_figures=False, max_workers=8, batch_size=12500,
        figure_dir="/tmp",
        single_absorber_model=args.single_absorber,
        filter_low_likelihood=True,
    )
    holder.initialize_results(1)
    print(f"[infer] running inference (max_dlas={args.max_dlas}, "
          f"single_absorber={args.single_absorber})...")
    t0 = time.time()
    holder.process_qso(idx=0, target_id=str(args.target_id),
                       wavelengths=wave, flux=flux,
                       noise_variance=noise_var, pixel_mask=mask, z_qso=z_qso)
    dt = time.time() - t0
    print(f"[infer] {dt:.1f}s")

    res = holder.results
    p_dlas = res["p_dlas"][0]
    map_z = res["MAP_z_dlas"][0]
    map_nhi = res["MAP_log_nhis"][0]
    print(f"\n[MAP] p_dla = {p_dlas:.4f}")
    for k in range(args.max_dlas):
        if not np.isnan(map_z[k]):
            print(f"      DLA {k+1}: z={map_z[k]:.4f}  log_nhi={map_nhi[k]:.3f}")

    # Now we need to evaluate log p(y | z_dlas, log_nhis) at truth.
    # Build a fresh DLAGPMAT with the same setup.
    from gpy_dla_detection.null_gp import NullGPMAT
    from gpy_dla_detection.dla_gp import DLAGPMAT
    from gpy_dla_detection.model_priors import PriorCatalog
    from gpy_dla_detection.dla_samples import DLASamplesMAT

    prior = PriorCatalog(params, catalog_name, los_catalog, dla_catalog)
    dla_samples = DLASamplesMAT(params, prior, dla_samples_file)
    dla_gp = DLAGPMAT(
        params, prior, dla_samples,
        min_z_separation=3000.0, learned_file=learned_file,
        broadening=True, prev_tau_0=preset.prev_tau_0, prev_beta=preset.prev_beta,
    )
    # Set data on dla_gp the same way process_qso does
    rest_wavelengths = params.emitted_wavelengths(wave, z_qso)
    dla_gp.set_data(np.atleast_2d(rest_wavelengths), np.atleast_2d(flux),
                    np.atleast_2d(noise_var), np.atleast_2d(mask),
                    np.array([z_qso]), build_model=True)

    # Evaluate log L at each candidate point.
    truth_zs = np.array([t[0] for t in args.truth])
    truth_nhis = np.array([10**t[1] for t in args.truth])

    print("\n[likelihood comparison]")
    # Truth (joint, all absorbers)
    L_truth = dla_gp.sample_log_likelihood_k_dlas(truth_zs, truth_nhis)
    print(f"  log L(truth, joint {len(args.truth)} absorbers) = {L_truth:.3f}")

    # Truth absorbers individually
    for i, (z_t, log_n_t) in enumerate(args.truth):
        L_i = dla_gp.sample_log_likelihood_k_dlas(np.array([z_t]), np.array([10**log_n_t]))
        print(f"  log L(truth absorber {i+1}: z={z_t:.3f}, log_nhi={log_n_t:.3f}) = {L_i:.3f}")

    # MAP at each k
    valid_k = [k for k in range(args.max_dlas) if not np.isnan(map_z[k])]
    if valid_k:
        zs_map = np.array([map_z[k] for k in valid_k])
        nhis_map = np.array([10**map_nhi[k] for k in valid_k])
        L_map = dla_gp.sample_log_likelihood_k_dlas(zs_map, nhis_map)
        print(f"  log L(MAP, joint {len(valid_k)} absorbers) = {L_map:.3f}")

        delta = L_truth - L_map
        print(f"\n  Δ = log L(truth) - log L(MAP) = {delta:+.3f}")
        if delta > 0.5:
            print("  ⇒ truth is significantly MORE likely than MAP")
            print("    → the QMC sampler missed the truth peak (sampling-limited)")
        elif delta < -0.5:
            print("  ⇒ truth is significantly LESS likely than MAP")
            print("    → MAP is a genuine likelihood peak; truth doesn't fit")
            print("      (forward-model, prior, continuum, or mock-physics issue)")
        else:
            print("  ⇒ comparable likelihood; sampling is OK")

    # Also: what's the closest QMC sample to truth?
    z_grid = dla_gp.dla_samples.sample_z_dlas(dla_gp.this_wavelengths, z_qso)
    nhi_grid = dla_samples.log_nhi_samples
    print(f"\n[QMC sample density check]")
    print(f"  Total QMC samples: {len(nhi_grid)}")
    for i, (z_t, log_n_t) in enumerate(args.truth):
        # Combined distance: weighted (z and log_nhi are different scales)
        d_z = np.abs(z_grid - z_t)
        d_n = np.abs(nhi_grid - log_n_t)
        # Just report the closest in NHI direction
        idx_n = np.argmin(d_n)
        idx_z = np.argmin(d_z)
        print(f"  Truth absorber {i+1}: z={z_t:.4f}, log_nhi={log_n_t:.3f}")
        print(f"    closest log_nhi sample: {nhi_grid[idx_n]:.4f} (Δ={log_n_t - nhi_grid[idx_n]:+.4f})")
        print(f"    closest z sample:       {z_grid[idx_z]:.4f} (Δ={z_t - z_grid[idx_z]:+.4f})")

    # Density at truth NHI
    nbins = 30
    hist, edges = np.histogram(nhi_grid, bins=nbins,
                                range=(nhi_grid.min(), nhi_grid.max()))
    print(f"\n  QMC sample density per 0.{int(100/nbins):02d} dex log_nhi bin "
          f"(min, median, max): {hist.min()}, {int(np.median(hist))}, {hist.max()}")
    for log_n_t in [t[1] for t in args.truth] + [20.0, 21.0, 22.0]:
        bin_idx = np.searchsorted(edges, log_n_t) - 1
        bin_idx = max(0, min(bin_idx, nbins - 1))
        print(f"    samples in bin around log_nhi={log_n_t:.2f}: {hist[bin_idx]}")

    # ─ Brute-force scan over a subset of QMC samples to find the *real*
    # max-log-L 1-DLA sample, bypassing FILTER. If this exceeds log L(truth),
    # the QMC contains a better fit than truth (forward-model / prior issue).
    # If less, the sampler density isn't the problem — something else is.
    print(f"\n[brute-force scan over QMC] (single-DLA, this can take a minute)")
    n_scan = min(20000, len(nhi_grid))
    rng = np.random.default_rng(42)
    scan_idx = rng.choice(len(nhi_grid), size=n_scan, replace=False)
    scan_log_l = np.full(n_scan, np.nan)
    t1 = time.time()
    for j, idx in enumerate(scan_idx):
        try:
            scan_log_l[j] = dla_gp.sample_log_likelihood_k_dlas(
                np.array([z_grid[idx]]), np.array([10**nhi_grid[idx]])
            )
        except Exception:
            pass
        if j == 100:
            est = (time.time() - t1) / 100 * n_scan
            print(f"  estimated total: {est:.0f}s ({n_scan} samples)")
    valid = ~np.isnan(scan_log_l)
    if valid.any():
        argmax = np.argmax(scan_log_l[valid])
        valid_idx_arr = np.flatnonzero(valid)
        best_local = valid_idx_arr[argmax]
        best_idx = scan_idx[best_local]
        L_best = scan_log_l[best_local]
        print(f"  brute-force max log L over {valid.sum()}/{n_scan} samples:")
        print(f"    log L = {L_best:.3f}  at z_dla={z_grid[best_idx]:.4f}, "
              f"log_nhi={nhi_grid[best_idx]:.3f}")
        L_truth1 = dla_gp.sample_log_likelihood_k_dlas(
            np.array([args.truth[0][0]]), np.array([10**args.truth[0][1]])
        )
        L_null = dla_gp.log_mvnpdf_low_rank(
            dla_gp.y, dla_gp.this_mu, dla_gp.this_M,
            dla_gp.this_omega2 + dla_gp.v,
        )
        print(f"  log L(null = no absorber)                       = {L_null:.3f}")
        print(f"  log L(truth 1-absorber, strongest DLA only)     = {L_truth1:.3f}")
        print(f"  log L(brute-force MAP from QMC)                 = {L_best:.3f}")
        print()
        d_truth_null = L_truth1 - L_null
        d_truth_best = L_truth1 - L_best
        print(f"  Δ(truth - null)         = {d_truth_null:+.3f}")
        print(f"  Δ(truth - brute-MAP)    = {d_truth_best:+.3f}")
        if d_truth_null < 0:
            print(f"  ⇒ truth ABSORBER FITS WORSE THAN NULL (no DLA)")
            print(f"    → forward model can't reproduce the truth profile here.")
            print(f"      Likely: continuum bias, mock-physics mismatch, or wrong NHI value.")
        elif d_truth_best > 0.5:
            print(f"  ⇒ truth fits BETTER than the best QMC sample")
            print(f"    → sampler-limited (need denser/adaptive samples near truth)")
        elif d_truth_best < -0.5:
            print(f"  ⇒ QMC found a peak away from truth that fits better")
            print(f"    → forward model / prior issue (not sampling)")
        else:
            print(f"  ⇒ truth ≈ best QMC sample; sampling adequate")
    else:
        print(f"  brute-force scan failed to produce any valid log L values")


if __name__ == "__main__":
    main()
