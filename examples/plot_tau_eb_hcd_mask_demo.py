"""Generate a 4-panel demo figure walking through the HCD-masked τ-EB recipe
on one DESI spectrum.

Companion to ``docs/tau_eb_hcd_mask.md`` — the figure produced here is what
the doc references for the step-by-step explanation.

Usage::

    python examples/plot_tau_eb_hcd_mask_demo.py \\
        --target-id 120046865 \\
        --spec /nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/spectra-16/7/789/spectra-16-789.fits \\
        --zcat /nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/zcat.fits \\
        --truth-z 2.7730 \\
        --truth-log-nhi 21.263 \\
        --out-png docs/tau_eb_hcd_mask_demo.png

Panels:
  (A) Forest spectrum + null GP prediction (μ × A_lyα at production τ_0).
  (B) Per-pixel residuals (y − μ_pred)/σ vs observed wavelength, with the
      −Nσ HCD-mask threshold and flagged pixels highlighted.
  (C) τ-grid log-evidence curve, naive (mask=original) vs HCD-masked.
      The τ_best shift is the headline of the recipe.
  (D) MAP log NHI bar chart: production τ=1.0× vs naive τ-EB vs HCD-masked
      τ-EB, with the truth value as a horizontal line.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
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
    p.add_argument("--mask-threshold-sigma", type=float, default=1.5)
    p.add_argument("--out-png", type=str,
                   default="docs/tau_eb_hcd_mask_demo.png")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    print(f"\n=== HCD-masked τ-EB demo for TID {args.target_id} ===")
    wave, flux, nv, mask_orig = load_one_desi_spectrum(args.spec, args.target_id)
    z_qso = lookup_z_qso(args.zcat, args.target_id)
    print(f"  z_qso={z_qso:.4f}, truth: z={args.truth_z}, "
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
    los_cat = os.path.join(args.data_root,
                           "data/dla_catalogs/dr9q_concordance/processed/los_catalog")
    dla_cat = os.path.join(args.data_root,
                           "data/dla_catalogs/dr9q_concordance/processed/dla_catalog")
    dla_samples_file = os.path.join(args.data_root,
                                    "data/dr12q/processed/dla_samples_a03_100000.mat")
    prior = PriorCatalog(params, catalog, los_cat, dla_cat)
    dla_samples = DLASamplesMAT(params, prior, dla_samples_file)

    # ------------------------------------------------------------------
    # Step 1: build the null GP at production τ_0 (Turner+2024 default).
    # ------------------------------------------------------------------
    print("\n[Step 1] Build null GP at production τ_0 = "
          f"{preset.prev_tau_0:.5f}")
    null_gp = NullGPMAT(params, prior, learned_file=learned,
                        prev_tau_0=preset.prev_tau_0,
                        prev_beta=preset.prev_beta)
    rest_w_seed = params.emitted_wavelengths(wave, z_qso)
    null_gp.set_data(np.atleast_2d(rest_w_seed), np.atleast_2d(flux),
                     np.atleast_2d(nv), np.atleast_2d(mask_orig),
                     np.array([z_qso]), build_model=True)
    pred = null_gp.this_mu                          # μ × A_lyα
    y = null_gp.y                                   # data
    sigma2 = null_gp.this_omega2 + null_gp.v        # total per-pixel variance
    residuals_sigma = (y - pred) / np.sqrt(sigma2)
    # NB: null_gp.this_wavelengths is ALREADY in observed frame (set_data
    # at null_gp.py:188). Do NOT multiply by (1+z_qso).
    obs_w_used = null_gp.this_wavelengths
    # Median of normalization range (used internally to scale flux). Used
    # below to put the raw spectrum on the same y-scale as the GP model.
    norm_median = null_gp.normalization_median

    print(f"  pixels: {y.size} after mask + range cut")
    print(f"  residuals/σ: min={residuals_sigma.min():.2f}  "
          f"med={np.median(residuals_sigma):.2f}  "
          f"max={residuals_sigma.max():.2f}")

    # ------------------------------------------------------------------
    # Step 2: identify HCD pixels (residual < −Nσ).
    # ------------------------------------------------------------------
    N = args.mask_threshold_sigma
    hcd_mask_inner = residuals_sigma < -N
    n_hcd = int(hcd_mask_inner.sum())
    print(f"\n[Step 2] Threshold = {N}σ → "
          f"{n_hcd}/{y.size} pixels flagged ({100*n_hcd/y.size:.1f}%)")

    full_idx_in_range = np.flatnonzero(null_gp.ind_unmasked)
    survived = ~mask_orig[full_idx_in_range]
    full_idx_used = full_idx_in_range[survived]
    new_mask = mask_orig.copy()
    new_mask[full_idx_used[hcd_mask_inner]] = True

    # ------------------------------------------------------------------
    # Step 3: τ-grid scan, naive vs HCD-masked.
    # ------------------------------------------------------------------
    nhi_grid = np.arange(20.30, 22.01, 0.025)
    tau_factors = np.array(args.tau_factors)
    print(f"\n[Step 3] τ-grid scan: τ_factor ∈ {list(tau_factors)}, "
          f"NHI grid {nhi_grid[0]:.2f}–{nhi_grid[-1]:.2f} step 0.025 "
          f"({len(nhi_grid)} pts)")

    def scan(masked: np.ndarray, label: str):
        log_l = np.full((len(tau_factors), len(nhi_grid)), np.nan)
        for j, tf in enumerate(tau_factors):
            tau0 = preset.prev_tau_0 * tf
            dla_gp = DLAGPMAT(params, prior, dla_samples,
                              min_z_separation=3000.0, learned_file=learned,
                              broadening=True,
                              prev_tau_0=tau0, prev_beta=preset.prev_beta)
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
            "label": label, "tau_best": tau_factors[j_best],
            "max_log_l": log_l[j_best, i_best],
            "map_nhi": nhi_grid[i_best],
            "log_l_grid": log_l,
            "max_per_tau": max_per_tau,
        }

    print("  scanning naive (no HCD mask)...")
    naive = scan(mask_orig, "EB naive")
    print(f"    τ_best = {naive['tau_best']:.2f}, MAP NHI = {naive['map_nhi']:.3f}")

    print("  scanning HCD-masked...")
    masked = scan(new_mask, f"EB + HCD-mask ({N}σ)")
    print(f"    τ_best = {masked['tau_best']:.2f}, MAP NHI = {masked['map_nhi']:.3f}")

    # Production: τ=1.0 with the original mask (uses naive's τ=1.0 column).
    j_prod = int(np.argmin(np.abs(tau_factors - 1.0)))
    L_prod_curve = naive["log_l_grid"][j_prod]
    map_prod = nhi_grid[int(np.nanargmax(L_prod_curve))]
    print(f"\n  Production (τ=1.0, no mask): MAP NHI = {map_prod:.3f}, "
          f"bias = {map_prod - args.truth_log_nhi:+.3f} dex")

    # ------------------------------------------------------------------
    # Plot: 4-panel figure.
    # ------------------------------------------------------------------
    print(f"\n[Plot] Writing {args.out_png}")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel A: zoomed forest + DLA trough so the absorption is visible.
    # The full DESI range (with the Lyα emission peak ~3-4× the forest
    # baseline) compresses the DLA trough to a few % of the y-axis. Zoom
    # into the forest region around the truth DLA and clip y to ~[-0.5, 2]
    # so the saturated trough and damping wings are unambiguous.
    ax = axes[0, 0]
    flux_norm = flux / norm_median  # match GP normalization
    lya_obs = 1215.67 * (1 + z_qso)
    dla_obs = 1215.67 * (1 + args.truth_z)
    # Show ~250 Å around the DLA, clipped to forest region.
    xmin = max(wave.min(), dla_obs - 200)
    xmax = min(lya_obs + 30, dla_obs + 200)
    print(f"[plot] panel A zoom xlim = ({xmin:.0f}, {xmax:.0f}); "
          f"lya_obs={lya_obs:.0f}; dla_obs={dla_obs:.0f}")
    ax.plot(wave, flux_norm, color="0.5", lw=0.4, label="DESI flux")
    ax.plot(obs_w_used, pred, color="C0", lw=1.2,
            label=r"null GP $\mu \cdot A_{Ly\alpha}$")
    if n_hcd > 0:
        ax.scatter(obs_w_used[hcd_mask_inner], y[hcd_mask_inner],
                   color="C3", s=10, zorder=3,
                   label=f"HCD-flagged ({n_hcd} px)")
    ax.axvspan(dla_obs - 30, dla_obs + 30, color="C3", alpha=0.18,
               label=fr"truth DLA $z={args.truth_z}$ "
                     fr"$\log N_{{HI}}={args.truth_log_nhi:.2f}$")
    if lya_obs <= xmax:
        ax.axvline(lya_obs, color="C2", lw=0.8, ls="--", alpha=0.7,
                   label=fr"Ly$\alpha$ at $z_{{qso}}={z_qso:.3f}$")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.5, 2.0)
    ax.axhline(0, color="0.7", lw=0.5, ls=":")
    ax.set_xlabel("observed wavelength [Å]")
    ax.set_ylabel("normalized flux")
    ax.set_title(f"(A) Forest zoom — DLA trough at $z={args.truth_z}$   "
                 f"TID={args.target_id}")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlabel("observed wavelength [Å]")
    ax.set_ylabel("normalized flux")
    ax.set_title(f"(A) Spectrum + null GP   "
                 f"TID={args.target_id}  z_qso={z_qso:.3f}")
    ax.axhline(0, color="0.7", lw=0.5, ls=":")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # Panel B: residuals.
    ax = axes[0, 1]
    ax.plot(obs_w_used, residuals_sigma, color="0.3", lw=0.4)
    ax.axhline(-N, color="C3", lw=1.2, ls="--",
               label=f"−{N}σ threshold ({n_hcd} masked)")
    if n_hcd > 0:
        ax.scatter(obs_w_used[hcd_mask_inner], residuals_sigma[hcd_mask_inner],
                   color="C3", s=8, zorder=3)
    ax.set_xlabel("observed wavelength [Å]")
    ax.set_ylabel(r"$(y - \mu_{\rm pred}) / \sigma$")
    ax.set_title("(B) Residuals: HCD pixels flagged")
    ax.set_ylim(-8, 6)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # Panel C: τ-grid log-evidence.
    ax = axes[1, 0]
    ax.plot(tau_factors, naive["max_per_tau"] - naive["max_per_tau"].max(),
            "o-", color="C1", label="naive (no mask)")
    ax.plot(tau_factors, masked["max_per_tau"] - masked["max_per_tau"].max(),
            "s-", color="C2", label="HCD-masked")
    ax.axvline(naive["tau_best"], color="C1", ls=":", alpha=0.6,
               label=f"naive τ_best = {naive['tau_best']:.2f}")
    ax.axvline(masked["tau_best"], color="C2", ls=":", alpha=0.6,
               label=f"masked τ_best = {masked['tau_best']:.2f}")
    ax.set_xlabel(r"$\tau_{\rm factor}$ "
                  r"(× Turner+2024 $\tau_0=0.00246$)")
    ax.set_ylabel(r"$\max_{N_{HI}} \log L$  (relative)")
    ax.set_title("(C) τ-EB log-evidence: HCD mask shifts τ_best")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel D: MAP NHI bar comparison.
    ax = axes[1, 1]
    cases = [
        ("production\nτ=1.0", map_prod, "C7"),
        ("naive τ-EB", naive["map_nhi"], "C1"),
        ("HCD-masked\nτ-EB", masked["map_nhi"], "C2"),
    ]
    xs = np.arange(len(cases))
    biases = np.array([c[1] - args.truth_log_nhi for c in cases])
    bars = ax.bar(xs, biases,
                  color=[c[2] for c in cases], alpha=0.85,
                  edgecolor="0.2")
    for x, c, b in zip(xs, cases, biases):
        ax.text(x, b + (0.02 if b >= 0 else -0.05),
                f"MAP={c[1]:.3f}\nΔ={b:+.3f}",
                ha="center", va="bottom" if b >= 0 else "top", fontsize=9)
    ax.axhline(0, color="black", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in cases])
    ax.set_ylabel(r"MAP $\log N_{HI}$ − truth  [dex]")
    ax.set_title(f"(D) Bias closure  (truth = {args.truth_log_nhi:.3f})")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "HCD-masked empirical-Bayes τ_eff fit — 4-step recipe demo",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"  saved {out}  ({out.stat().st_size//1024} KB)")

    # Summary line for the doc to reference.
    print()
    print(f"SUMMARY  truth={args.truth_log_nhi:.3f}  "
          f"prod={map_prod:.3f} (Δ={map_prod-args.truth_log_nhi:+.3f})  "
          f"naive_eb={naive['map_nhi']:.3f} (Δ={naive['map_nhi']-args.truth_log_nhi:+.3f})  "
          f"hcd_eb={masked['map_nhi']:.3f} (Δ={masked['map_nhi']-args.truth_log_nhi:+.3f})")


if __name__ == "__main__":
    main()
