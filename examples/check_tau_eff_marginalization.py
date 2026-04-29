"""Compare three τ_eff treatments on a target, decide what's reasonable for production:

  (1) **Production**: fixed τ_0 = Turner+2024 (0.00246).
  (2) **Empirical Bayes**: per-spectrum, pick the τ_factor that maximizes the
      marginal log-evidence (max over τ of max over NHI of log L). Use that
      single τ in the NHI MAP.
  (3) **Full marginalization**: marginalize p(y | NHI, z) = ∫ p(y | NHI, z, τ) p(τ) dτ
      via logsumexp over a discrete τ grid, then find the NHI argmax under
      the marginalized likelihood.

For each treatment we report (a) MAP NHI, (b) bias vs truth, (c) cost in
likelihood evaluations relative to production.

The comparison tells us:
- Is empirical Bayes enough (cheap, single-τ at the optimum)?
- Or does full marginalization meaningfully change MAP (more expensive,
  K× the QMC cost where K is the τ grid size)?

Run::
    python examples/check_tau_eff_marginalization.py \\
        --target-id 120046865 \\
        --spec /path/to/spectra-16-789.fits \\
        --zcat /path/to/zcat.fits \\
        --truth-z 2.7730 --truth-log-nhi 21.263
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
                   default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                   help="Grid for both EB and full-marg. Even spacing makes "
                        "the discretized integral well-conditioned.")
    p.add_argument("--nhi-grid-min", type=float, default=20.30)
    p.add_argument("--nhi-grid-max", type=float, default=22.00)
    p.add_argument("--nhi-grid-step", type=float, default=0.025)
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from gpy_dla_detection.voigt_v2_inject import inject
    inject(kernel="boss-log-r2000", num_lines=3)

    from examples.smoke_one_spectrum import (
        load_one_desi_spectrum, lookup_z_qso, PRESETS,
    )
    from gpy_dla_detection.set_parameters import Parameters
    from gpy_dla_detection.dla_gp import DLAGPMAT
    from gpy_dla_detection.model_priors import PriorCatalog
    from gpy_dla_detection.dla_samples import DLASamplesMAT

    wave, flux, nv, mask = load_one_desi_spectrum(args.spec, args.target_id)
    z_qso = lookup_z_qso(args.zcat, args.target_id)
    print(f"target={args.target_id}  z_qso={z_qso:.4f}  truth: z={args.truth_z}, "
          f"log_nhi={args.truth_log_nhi:.3f}")

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

    # Build the (NHI × τ) likelihood grid, NHI fine, τ from --tau-factors.
    nhi_grid = np.arange(args.nhi_grid_min, args.nhi_grid_max + args.nhi_grid_step / 2,
                         args.nhi_grid_step)
    tau_factors = np.array(args.tau_factors, dtype=float)
    log_l = np.full((len(tau_factors), len(nhi_grid)), np.nan)

    for j, tf in enumerate(tau_factors):
        prev_tau_0 = preset.prev_tau_0 * tf
        dla_gp = DLAGPMAT(params, prior, dla_samples,
            min_z_separation=3000.0, learned_file=learned,
            broadening=True, prev_tau_0=prev_tau_0, prev_beta=preset.prev_beta)
        rest_w = params.emitted_wavelengths(wave, z_qso)
        dla_gp.set_data(np.atleast_2d(rest_w), np.atleast_2d(flux),
                        np.atleast_2d(nv), np.atleast_2d(mask),
                        np.array([z_qso]), build_model=True)
        for i, ln in enumerate(nhi_grid):
            try:
                log_l[j, i] = dla_gp.sample_log_likelihood_k_dlas(
                    np.array([args.truth_z]), np.array([10**ln]))
            except Exception:
                pass

    # ── Treatment 1: production single τ = 1.0 ──
    j_prod = int(np.argmin(np.abs(tau_factors - 1.0)))
    L_prod = log_l[j_prod]
    map_prod = nhi_grid[int(np.nanargmax(L_prod))]
    bias_prod = map_prod - args.truth_log_nhi

    # ── Treatment 2: empirical Bayes — pick τ that maximizes max-over-NHI log L,
    # then use that τ alone ──
    max_logL_per_tau = np.nanmax(log_l, axis=1)
    j_eb = int(np.argmax(max_logL_per_tau))
    L_eb = log_l[j_eb]
    map_eb = nhi_grid[int(np.nanargmax(L_eb))]
    bias_eb = map_eb - args.truth_log_nhi
    tau_eb = tau_factors[j_eb]

    # ── Treatment 3: full marginalization via logsumexp over τ ──
    # Uniform prior on τ_factor. For each NHI, marg_logL[i] =
    # logsumexp over τ of log_l[:, i] - log(K).
    # logsumexp implementation
    def logsumexp(a, axis=0):
        a = np.asarray(a)
        amax = np.nanmax(a, axis=axis, keepdims=True)
        amax = np.where(np.isfinite(amax), amax, 0.0)
        return np.squeeze(amax, axis=axis) + np.log(
            np.nansum(np.exp(a - amax), axis=axis)
        )
    K = len(tau_factors)
    L_marg = logsumexp(log_l, axis=0) - np.log(K)
    map_marg = nhi_grid[int(np.nanargmax(L_marg))]
    bias_marg = map_marg - args.truth_log_nhi

    print(f"\n=== τ-grid: {[round(t, 3) for t in tau_factors]} (K={K}) ===")
    print(f"NHI grid: {nhi_grid[0]:.3f} → {nhi_grid[-1]:.3f} step {args.nhi_grid_step} "
          f"({len(nhi_grid)} pts)\n")

    print(f"{'treatment':<28} {'tau used':<14} {'MAP NHI':>9} {'bias':>9} "
          f"{'cost (× prod)':>15}")
    print("-" * 80)
    print(f"{'(1) production fixed τ=1.0':<28} {'1.0':<14} "
          f"{map_prod:9.3f} {bias_prod:+9.3f} {1.0:>14.1f}")
    print(f"{'(2) empirical Bayes':<28} {f'{tau_eb}':<14} "
          f"{map_eb:9.3f} {bias_eb:+9.3f} {float(K):>14.1f}")
    print(f"{'(3) full marginalization':<28} {'logsumexp':<14} "
          f"{map_marg:9.3f} {bias_marg:+9.3f} {float(K):>14.1f}")

    # Per-NHI marginalized vs production, head-to-head bias around truth
    print("\n[curve at truth ± 0.2 dex]")
    print(f"{'log NHI':>10} {'L(prod τ=1)':>12} {'L(EB τ='+str(tau_eb)+')':>16} "
          f"{'L(marg)':>10}")
    truth_idx = int(np.argmin(np.abs(nhi_grid - args.truth_log_nhi)))
    for i in range(max(0, truth_idx - 8), min(len(nhi_grid), truth_idx + 9)):
        marker = " ← truth" if i == truth_idx else ""
        print(f"{nhi_grid[i]:10.3f} {L_prod[i]:12.2f} {L_eb[i]:16.2f} "
              f"{L_marg[i]:10.2f}{marker}")

    print("\n[recommendations]")
    if abs(bias_eb) < 0.5 * abs(bias_prod):
        print(f"  ⇒ Empirical Bayes (τ={tau_eb}) reduces bias by "
              f"{(1 - abs(bias_eb)/abs(bias_prod)) * 100:.0f}%.")
    else:
        print(f"  ⇒ Empirical Bayes does NOT meaningfully reduce bias.")
    if abs(bias_marg) < 0.5 * abs(bias_prod):
        print(f"  ⇒ Full marginalization reduces bias by "
              f"{(1 - abs(bias_marg)/abs(bias_prod)) * 100:.0f}%.")
    else:
        print(f"  ⇒ Full marginalization does NOT meaningfully reduce bias.")
    if abs(bias_eb) < abs(bias_marg) + 0.02:
        print("  ⇒ Empirical Bayes ≈ marg → use EB (single best τ, K× cost vs "
              "production but NO logsumexp at inference time after τ is picked).")
    else:
        print("  ⇒ Marginalization beats EB → need full logsumexp; "
              "K× inference cost is unavoidable.")


if __name__ == "__main__":
    main()
