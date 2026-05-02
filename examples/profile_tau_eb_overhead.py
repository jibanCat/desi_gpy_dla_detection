"""Profile the cost of enabling HCD-masked τ-EB on top of the production
inference path. Measures wall-time on one spectrum for:

  (1) BASELINE        — null GP build + DLA GP build + bayes.model_selection
                        (this is what production currently does per spectrum)
  (2) TAU_EB_STEP     — K mini-builds of DLAGPMAT at K different τ_factors,
                        each followed by an N_HI grid scan
                        (this is the new step the recipe adds)
  (3) ENABLED_TOTAL   — (1) re-built at the chosen τ + (2) with masking
                        (this is what the production path costs when the
                        flag is on)

Reports the K factor (ENABLED_TOTAL / BASELINE) and a per-step breakdown
so we can decide whether K=6 is acceptable in production runs.

Usage::

    python examples/profile_tau_eb_overhead.py \\
        --target-id 120046865 \\
        --spec  /path/to/spectra-16-789.fits \\
        --zcat  /path/to/zcat.fits

Defaults to the canonical 2lpt target. Produces a one-row CSV at the path
given by ``--csv-out`` plus a printed summary table.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def _t():
    return time.perf_counter()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--target-id", type=int, default=120046865)
    p.add_argument("--spec", type=str,
                   default="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/spectra-16/7/789/spectra-16-789.fits")
    p.add_argument("--zcat", type=str,
                   default="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/zcat.fits")
    p.add_argument("--data-root", default=os.environ.get(
        "DATA_ROOT",
        "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection",
    ))
    p.add_argument("--tau-factors", type=float, nargs="+",
                   default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    p.add_argument("--mask-threshold-sigma", type=float, default=1.5)
    p.add_argument("--num-dla-samples", type=int, default=100000)
    p.add_argument("--max-workers", type=int, default=int(os.environ.get("PROFILE_WORKERS", 1)))
    p.add_argument("--batch-size", type=int, default=313)
    p.add_argument("--csv-out", type=str,
                   default="tests/profile/results/tau_eb_overhead.csv")
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
    from gpy_dla_detection.subdla_gp import SubDLAGPMAT
    from gpy_dla_detection.subdla_samples import SubDLASamplesMAT
    from gpy_dla_detection.model_priors import PriorCatalog
    from gpy_dla_detection.dla_samples import DLASamplesMAT
    from gpy_dla_detection.bayesian_model_selection import BayesModelSelect

    # ------------------------------------------------------------------
    # Load spectrum + set up shared params/prior/samples
    # ------------------------------------------------------------------
    print(f"[setup] loading TID {args.target_id} from {Path(args.spec).name}")
    wave, flux, nv, mask_orig = load_one_desi_spectrum(args.spec, args.target_id)
    z_qso = lookup_z_qso(args.zcat, args.target_id)
    print(f"[setup] z_qso={z_qso:.4f}")

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
    params = Parameters(num_dla_samples=args.num_dla_samples, **common)
    learned = os.path.join(args.data_root, preset.learned_file)
    catalog = os.path.join(args.data_root, "data/dr12q/processed/catalog.mat")
    los_cat = os.path.join(args.data_root, "data/dla_catalogs/dr9q_concordance/processed/los_catalog")
    dla_cat = os.path.join(args.data_root, "data/dla_catalogs/dr9q_concordance/processed/dla_catalog")
    if args.num_dla_samples == 10000:
        dla_samples_file = os.path.join(args.data_root, "data/dr12q/processed/dla_samples_a03.mat")
    elif args.num_dla_samples == 100000:
        dla_samples_file = os.path.join(args.data_root, "data/dr12q/processed/dla_samples_a03_100000.mat")
    else:
        raise ValueError(f"no dla_samples file matches num_dla_samples={args.num_dla_samples}")
    # Match the subdla samples file to num_dla_samples (assertion in
    # SubDLASamplesMAT.__init__ requires equality).
    if args.num_dla_samples == 10000:
        subdla_samples_file = os.path.join(args.data_root,
                                           "data/dr12q/processed/subdla_samples.mat")
    elif args.num_dla_samples == 100000:
        subdla_samples_file = os.path.join(args.data_root,
                                           "data/dr12q/processed/subdla_samples_a03_191_200_100000.mat")
    else:
        raise ValueError(f"no subdla_samples file matches num_dla_samples={args.num_dla_samples}")
    prior = PriorCatalog(params, catalog, los_cat, dla_cat)
    dla_samples = DLASamplesMAT(params, prior, dla_samples_file)
    subdla_samples = SubDLASamplesMAT(params, prior, subdla_samples_file)
    rest_w = params.emitted_wavelengths(wave, z_qso)

    # ------------------------------------------------------------------
    # (1) BASELINE: production-style full inference at production τ_0.
    #     = build null + subdla + dla GPs + run bayes.model_selection
    # ------------------------------------------------------------------
    print("\n[1/3] BASELINE — production inference (no τ-EB)")
    t0 = _t()
    null_gp_b = NullGPMAT(params, prior, learned_file=learned,
                          prev_tau_0=preset.prev_tau_0, prev_beta=preset.prev_beta)
    subdla_gp_b = SubDLAGPMAT(params, prior, subdla_samples,
                              min_z_separation=3000.0, learned_file=learned,
                              broadening=True,
                              prev_tau_0=preset.prev_tau_0, prev_beta=preset.prev_beta)
    dla_gp_b = DLAGPMAT(params, prior, dla_samples,
                        min_z_separation=3000.0, learned_file=learned,
                        broadening=True,
                        prev_tau_0=preset.prev_tau_0, prev_beta=preset.prev_beta)
    t_build_b = _t() - t0
    print(f"   build (null+subdla+dla):  {t_build_b:.2f} s")

    t0 = _t()
    for m in [null_gp_b, subdla_gp_b, dla_gp_b]:
        m.set_data(rest_w, flux, nv, mask_orig, z_qso, build_model=True)
    t_setdata_b = _t() - t0
    print(f"   set_data (3 models):      {t_setdata_b:.2f} s")

    bayes_b = BayesModelSelect([0, 1, 3], 2)
    t0 = _t()
    bayes_b.model_selection([null_gp_b, subdla_gp_b, dla_gp_b], z_qso,
                            max_workers=args.max_workers,
                            batch_size=args.batch_size,
                            filter_low_likelihood=False)
    t_bayes_b = _t() - t0
    print(f"   model_selection:          {t_bayes_b:.2f} s")
    t_baseline = t_build_b + t_setdata_b + t_bayes_b
    print(f"   BASELINE TOTAL:           {t_baseline:.2f} s")

    # Free memory
    del null_gp_b, subdla_gp_b, dla_gp_b, bayes_b

    # ------------------------------------------------------------------
    # (2) TAU_EB_STEP: K mini-builds of DLAGPMAT + N_HI-grid scans
    # ------------------------------------------------------------------
    print(f"\n[2/3] TAU_EB_STEP — HCD-mask + K={len(args.tau_factors)} τ scan")

    # Step 2a: build a null GP at production τ to get residuals → HCD mask.
    t0 = _t()
    null_gp_h = NullGPMAT(params, prior, learned_file=learned,
                          prev_tau_0=preset.prev_tau_0, prev_beta=preset.prev_beta)
    null_gp_h.set_data(np.atleast_2d(rest_w), np.atleast_2d(flux),
                       np.atleast_2d(nv), np.atleast_2d(mask_orig),
                       np.array([z_qso]), build_model=True)
    pred = null_gp_h.this_mu
    y = null_gp_h.y
    sigma2 = null_gp_h.this_omega2 + null_gp_h.v
    residuals_sigma = (y - pred) / np.sqrt(sigma2)
    hcd_mask_inner = residuals_sigma < -args.mask_threshold_sigma
    full_idx_in_range = np.flatnonzero(null_gp_h.ind_unmasked)
    survived = ~mask_orig[full_idx_in_range]
    full_idx_used = full_idx_in_range[survived]
    new_mask = mask_orig.copy()
    new_mask[full_idx_used[hcd_mask_inner]] = True
    n_hcd = int(hcd_mask_inner.sum())
    t_mask = _t() - t0
    print(f"   null+set_data+mask:       {t_mask:.2f} s "
          f"({n_hcd}/{y.size} HCD pixels)")
    del null_gp_h

    # Step 2b: K builds + N_HI scans on HCD-masked pixel_mask.
    nhi_grid = np.arange(20.30, 22.01, 0.05)  # 35 pts (coarser than diagnostic)
    # We need a candidate z_DLA in production. Use the QMC sample z that has
    # the highest null-likelihood gain — but for profiling, just use a fixed
    # mid-forest z (representative cost).
    z_dla_eb = z_qso - 0.20  # mid-forest

    t_per_tau = np.zeros(len(args.tau_factors))
    log_l_grid = np.full((len(args.tau_factors), len(nhi_grid)), np.nan)
    for j, tf in enumerate(args.tau_factors):
        tau0 = preset.prev_tau_0 * tf
        t0 = _t()
        dla_gp_tmp = DLAGPMAT(params, prior, dla_samples,
                              min_z_separation=3000.0, learned_file=learned,
                              broadening=True,
                              prev_tau_0=tau0, prev_beta=preset.prev_beta)
        dla_gp_tmp.set_data(np.atleast_2d(rest_w), np.atleast_2d(flux),
                            np.atleast_2d(nv), np.atleast_2d(new_mask),
                            np.array([z_qso]), build_model=True)
        for i, ln in enumerate(nhi_grid):
            try:
                log_l_grid[j, i] = dla_gp_tmp.sample_log_likelihood_k_dlas(
                    np.array([z_dla_eb]), np.array([10**ln]))
            except Exception:
                pass
        t_per_tau[j] = _t() - t0
        del dla_gp_tmp

    j_best = int(np.argmax(np.nanmax(log_l_grid, axis=1)))
    tau_best = args.tau_factors[j_best]
    t_tau_eb = t_mask + t_per_tau.sum()
    print(f"   per-τ build+scan:         "
          f"min={t_per_tau.min():.2f} med={np.median(t_per_tau):.2f} "
          f"max={t_per_tau.max():.2f} s  (K={len(args.tau_factors)})")
    print(f"   TAU_EB_STEP TOTAL:        {t_tau_eb:.2f} s   "
          f"(τ_best={tau_best:.2f})")

    # ------------------------------------------------------------------
    # (3) ENABLED_TOTAL: (2) + production inference at the chosen τ
    # ------------------------------------------------------------------
    print("\n[3/3] ENABLED_TOTAL — production inference at chosen τ + (2)")
    tau0_chosen = preset.prev_tau_0 * tau_best
    t0 = _t()
    null_gp_e = NullGPMAT(params, prior, learned_file=learned,
                          prev_tau_0=tau0_chosen, prev_beta=preset.prev_beta)
    subdla_gp_e = SubDLAGPMAT(params, prior, subdla_samples,
                              min_z_separation=3000.0, learned_file=learned,
                              broadening=True,
                              prev_tau_0=tau0_chosen, prev_beta=preset.prev_beta)
    dla_gp_e = DLAGPMAT(params, prior, dla_samples,
                        min_z_separation=3000.0, learned_file=learned,
                        broadening=True,
                        prev_tau_0=tau0_chosen, prev_beta=preset.prev_beta)
    t_build_e = _t() - t0
    t0 = _t()
    for m in [null_gp_e, subdla_gp_e, dla_gp_e]:
        m.set_data(rest_w, flux, nv, mask_orig, z_qso, build_model=True)
    t_setdata_e = _t() - t0
    bayes_e = BayesModelSelect([0, 1, 3], 2)
    t0 = _t()
    bayes_e.model_selection([null_gp_e, subdla_gp_e, dla_gp_e], z_qso,
                            max_workers=args.max_workers,
                            batch_size=args.batch_size,
                            filter_low_likelihood=False)
    t_bayes_e = _t() - t0
    t_inf_at_chosen_tau = t_build_e + t_setdata_e + t_bayes_e
    t_enabled = t_tau_eb + t_inf_at_chosen_tau
    print(f"   inference at chosen τ:    {t_inf_at_chosen_tau:.2f} s")
    print(f"   ENABLED_TOTAL:            {t_enabled:.2f} s")

    K_factor = t_enabled / t_baseline if t_baseline > 0 else float("nan")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'step':<32} {'wall (s)':>10} {'rel':>8}")
    print("-" * 70)
    print(f"{'(1) BASELINE production':<32} {t_baseline:>10.2f} {1.00:>8.2f}")
    print(f"{'(2) TAU_EB_STEP':<32} {t_tau_eb:>10.2f} {t_tau_eb/t_baseline:>8.2f}")
    print(f"{'(3) ENABLED_TOTAL ((1)+(2)+(3))':<32} {t_enabled:>10.2f} {K_factor:>8.2f}")
    print(f"\nProduction K-factor: {K_factor:.2f}× (ENABLED_TOTAL / BASELINE)")
    print(f"τ-EB step alone:     {t_tau_eb/t_baseline:.2f}× of baseline")
    print(f"Per-τ build+scan:    median {np.median(t_per_tau):.2f} s, "
          f"K={len(args.tau_factors)} = {t_per_tau.sum():.2f} s")

    # CSV output
    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a") as f:
        if write_header:
            f.write("target_id,z_qso,n_pix,n_hcd,K,t_baseline,t_tau_eb,t_enabled,K_factor,tau_best,max_workers\n")
        f.write(f"{args.target_id},{z_qso:.4f},{y.size},{n_hcd},"
                f"{len(args.tau_factors)},{t_baseline:.3f},{t_tau_eb:.3f},"
                f"{t_enabled:.3f},{K_factor:.3f},{tau_best:.2f},{args.max_workers}\n")
    print(f"\n[csv] appended to {csv_path}")


if __name__ == "__main__":
    main()
