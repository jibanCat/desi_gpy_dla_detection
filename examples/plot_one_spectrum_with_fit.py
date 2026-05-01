"""Render a story-friendly figure for ONE QSO spectrum: data + GP null
prediction + Voigt fit at the MAP DLA, both at production τ_0 and at
the per-spectrum τ-EB-chosen τ_0.

Figure layout (2 rows × 1 col):
  (A) observed-wavelength forest spectrum, with overlays:
      - DESI data (grey)
      - production model prediction at MAP DLA (orange)
      - τ-EB model prediction at MAP DLA (green)
      truth-DLA position marked as a red shaded band; MAP-DLA position
      as a vertical line.
  (B) residuals (y − model_τEB)/σ vs wavelength.

Used to embed inline figures in the per-mock story docs.

Usage::

    python examples/plot_one_spectrum_with_fit.py \\
        --mock 2lpt --target-id 120046865 \\
        --spec /path/to/spectra-16-789.fits \\
        --zcat /path/to/zcat.fits \\
        --truth-z 2.7730 --truth-log-nhi 21.263 \\
        --out-png docs/story_figures/2lpt_tid120046865.png
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
import copy as _copy

import numpy as np


def _build_holder():
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
        num_dla_samples=10000,
    )
    holder = DLAHolder(
        learned_file=os.path.join(DATA_ROOT, p.learned_file),
        catalog_name=os.path.join(DATA_ROOT, "data/dr12q/processed/catalog.mat"),
        los_catalog=os.path.join(DATA_ROOT, "data/dla_catalogs/dr9q_concordance/processed/los_catalog"),
        dla_catalog=os.path.join(DATA_ROOT, "data/dla_catalogs/dr9q_concordance/processed/dla_catalog"),
        dla_samples_file=os.path.join(DATA_ROOT, "data/dr12q/processed/dla_samples_a03.mat"),
        sub_dla_samples_file=os.path.join(DATA_ROOT, "data/dr12q/processed/subdla_samples.mat"),
        params=params, params_subdla=_copy.copy(params),
        min_z_separation=3000.0,
        prev_tau_0=p.prev_tau_0, prev_beta=p.prev_beta,
        max_dlas=3, max_workers=8, batch_size=313,
        enable_tau_eb=False,
        tau_eb_apply_hcd_mask=False,
        tau_eb_objective="null",
    )
    return holder, p


def _run_and_capture(holder, preset, target_id, wave, flux, nv, mask, z_qso,
                     enable_tau_eb: bool):
    """Run one inference; return (MAP_z, MAP_NHI, p_DLA, used_tau_0,
    rest_w_obs, mu, mu_with_dla)."""
    sys.path.insert(0, "/home/mfho/desi_gpy_dla_detection")
    from gpy_dla_detection.null_gp import NullGPMAT
    from gpy_dla_detection.dla_gp import DLAGPMAT

    holder.enable_tau_eb = enable_tau_eb
    holder.initialize_results(num_spectra=1)
    holder.process_qso(idx=0, target_id=target_id,
                       wavelengths=wave, flux=flux,
                       noise_variance=nv, pixel_mask=mask, z_qso=z_qso)
    map_z = holder.results["MAP_z_dlas"][0]
    map_nhi = holder.results["MAP_log_nhis"][0]
    p_dla = float(holder.results["p_dlas"][0])

    # The actual τ_0 used by this run: equal to the seed for BASELINE,
    # or the EB-chosen value for ENABLED. Reconstruct by re-running τ-EB
    # if enabled (cheap), otherwise use the seed.
    if enable_tau_eb:
        from gpy_dla_detection.tau_eb import fit_tau_eb_hcd_mask
        rest_w = holder.params.emitted_wavelengths(wave, z_qso)
        used_tau, info = fit_tau_eb_hcd_mask(
            params=holder.params, prior=holder.prior,
            learned_file=holder.learned_file,
            rest_wavelengths=rest_w, flux=flux, noise_variance=nv,
            pixel_mask=mask, z_qso=z_qso,
            prev_tau_0_seed=preset.prev_tau_0, prev_beta=preset.prev_beta,
            tau_factors=holder.tau_eb_factors, apply_hcd_mask=False,
            objective="null",
        )
        tau_factor = info["tau_factor_best"]
    else:
        used_tau = preset.prev_tau_0
        tau_factor = 1.0

    # Re-build NullGPMAT and DLAGPMAT at this τ for plotting overlays.
    null_gp = NullGPMAT(holder.params, holder.prior,
                       learned_file=holder.learned_file,
                       prev_tau_0=used_tau, prev_beta=preset.prev_beta)
    rest_w = holder.params.emitted_wavelengths(wave, z_qso)
    null_gp.set_data(rest_w, flux, nv, mask, z_qso, build_model=True)
    obs_wave = null_gp.this_wavelengths  # already obs-frame
    mu = null_gp.this_mu                 # null GP prediction (μ × A_lyα)
    y = null_gp.y
    sigma = np.sqrt(null_gp.this_omega2 + null_gp.v)

    # If a DLA was MAP-detected, build the Voigt overlay too.
    mu_dla = None
    if np.isfinite(map_nhi).any() and np.isfinite(map_z).any():
        dla_gp = DLAGPMAT(holder.params, holder.prior, holder.dla_samples,
                          min_z_separation=3000.0,
                          learned_file=holder.learned_file,
                          broadening=True,
                          prev_tau_0=used_tau, prev_beta=preset.prev_beta)
        dla_gp.set_data(rest_w, flux, nv, mask, z_qso, build_model=True)
        finite_mask = np.isfinite(map_z) & np.isfinite(map_nhi)
        z_arr = np.asarray(map_z)[finite_mask]
        nhi_arr = 10 ** np.asarray(map_nhi)[finite_mask]
        if len(z_arr) > 0:
            mu_dla, _, _ = dla_gp.this_dla_gp(z_arr, nhi_arr)

    return dict(
        target_id=target_id, z_qso=z_qso,
        map_z=map_z, map_nhi=map_nhi, p_dla=p_dla,
        used_tau=used_tau, tau_factor=tau_factor,
        obs_wave=obs_wave, mu=mu, mu_dla=mu_dla, y=y, sigma=sigma,
        norm_median=null_gp.normalization_median,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--mock", required=True, choices=["2lpt", "london", "saclay"])
    p.add_argument("--target-id", type=int, required=True)
    p.add_argument("--spec", type=str, required=True)
    p.add_argument("--zcat", type=str, required=True)
    p.add_argument("--truth-z", type=float, default=-1.0,
                   help="Truth DLA z (use -1 for no truth marker).")
    p.add_argument("--truth-log-nhi", type=float, default=-1.0)
    p.add_argument("--truth-catalog", type=str, default=None,
                   help="Optional path to hcd_truth_cat.fits (or dla_cat.fits "
                        "for london). When provided, ALL absorbers on this TID "
                        "are looked up and marked, color-coded by NHI strength.")
    p.add_argument("--bal-catalog", type=str, default=None,
                   help="Optional path to bal_cat.fits. When provided, the "
                        "figure title flags whether this TID is BAL.")
    p.add_argument("--out-png", required=True)
    p.add_argument("--zoom-around-truth", action="store_true",
                   help="Zoom panel A around the truth DLA position.")
    args = p.parse_args()

    sys.path.insert(0, "/home/mfho/desi_gpy_dla_detection")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from examples.smoke_one_spectrum import load_one_desi_spectrum, lookup_z_qso

    print(f"[load] mock={args.mock} TID={args.target_id}")
    wave, flux, nv, mask = load_one_desi_spectrum(args.spec, args.target_id)
    z_qso = lookup_z_qso(args.zcat, args.target_id)

    # Optionally load ALL absorbers on this TID + BAL flag.
    truth_absorbers = []  # list of (z_truth, log_nhi)
    if args.truth_catalog:
        import fitsio
        try:
            tc = fitsio.read(args.truth_catalog)
            z_col = "Z" if "Z" in tc.dtype.names else "Z_DLA"
            mask_tid = tc["TARGETID"] == args.target_id
            for row in tc[mask_tid]:
                truth_absorbers.append((float(row[z_col]), float(row["NHI"])))
            print(f"[truth] {len(truth_absorbers)} absorber(s) on TID {args.target_id}: "
                  + ", ".join(f"z={z:.3f} log NHI={n:.2f}" for z, n in truth_absorbers))
        except Exception as e:
            print(f"[truth] WARN: could not load {args.truth_catalog}: {e}")
    elif args.truth_z > 0:
        truth_absorbers.append((args.truth_z, args.truth_log_nhi))

    is_bal = False
    bal_msg = ""
    if args.bal_catalog:
        import fitsio
        try:
            bc = fitsio.read(args.bal_catalog)
            tid_col = "TARGETID" if "TARGETID" in bc.dtype.names else "MOCKID"
            is_bal = bool((bc[tid_col] == args.target_id).any())
            bal_msg = "BAL" if is_bal else "no-BAL"
            print(f"[bal] TID {args.target_id} → {bal_msg}")
        except Exception as e:
            print(f"[bal] WARN: could not load {args.bal_catalog}: {e}")

    holder, preset = _build_holder()

    print("[infer] BASELINE production τ ...")
    t0 = time.perf_counter()
    base = _run_and_capture(holder, preset, args.target_id,
                            wave, flux, nv, mask, z_qso, enable_tau_eb=False)
    print(f"  {time.perf_counter()-t0:.1f}s  p_DLA={base['p_dla']:.3f}  "
          f"MAP NHI={base['map_nhi'][:1]}")

    print("[infer] ENABLED (τ-EB) ...")
    t0 = time.perf_counter()
    enab = _run_and_capture(holder, preset, args.target_id,
                            wave, flux, nv, mask, z_qso, enable_tau_eb=True)
    print(f"  {time.perf_counter()-t0:.1f}s  τ_factor={enab['tau_factor']:.2f}  "
          f"p_DLA={enab['p_dla']:.3f}  MAP NHI={enab['map_nhi'][:1]}")

    flux_norm = flux / base["norm_median"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6),
                             gridspec_kw=dict(height_ratios=[3, 1]),
                             sharex=True)
    ax_top, ax_bot = axes

    ax_top.plot(wave, flux_norm, color="0.55", lw=0.4, label="DESI flux")
    ax_top.plot(base["obs_wave"], base["mu"], color="C0", lw=0.8, alpha=0.55,
                label=fr"null GP, $\tau_0\!=\!{preset.prev_tau_0:.5f}$ (1×)")
    if base["mu_dla"] is not None:
        ax_top.plot(base["obs_wave"], base["mu_dla"], color="C1", lw=1.2,
                    label=fr"BASELINE Voigt fit, $\log N_{{HI}}={float(np.nanmax(base['map_nhi'])):.2f}$, "
                          fr"$p_{{DLA}}\!=\!{base['p_dla']:.2f}$")
    if enab["mu_dla"] is not None:
        ax_top.plot(enab["obs_wave"], enab["mu_dla"], color="C2", lw=1.2,
                    label=fr"τ-EB Voigt fit, $\log N_{{HI}}={float(np.nanmax(enab['map_nhi'])):.2f}$, "
                          fr"$p_{{DLA}}\!=\!{enab['p_dla']:.2f}$, "
                          fr"$\tau_{{factor}}\!=\!{enab['tau_factor']:.2f}\times$")

    # All-absorbers markers, color-coded by NHI strength
    # DLA  (NHI ≥ 20.3)   → red
    # subDLA (19.0–20.3)  → orange
    # LLS  (17.2–19.0)    → gold
    # Lyα-forest noise (<17.2) → grey (rare in catalogs but possible)
    def _absorber_color(log_nhi):
        if log_nhi >= 20.3: return "C3", "DLA"
        if log_nhi >= 19.0: return "tab:orange", "sub-DLA"
        if log_nhi >= 17.2: return "tab:olive", "LLS"
        return "0.5", "weak"
    lya_obs = 1215.67 * (1 + z_qso)
    seen_labels = set()  # de-duplicate legend entries
    for z_t, n_t in truth_absorbers:
        obs = 1215.67 * (1 + z_t)
        c, kind = _absorber_color(n_t)
        # Only mark the FIRST absorber of each kind in the legend (or label
        # individually if there are <=3 absorbers — typical case).
        lbl = (fr"truth {kind} $z\!=\!{z_t:.3f}$, "
               fr"$\log N_{{HI}}\!=\!{n_t:.2f}$")
        ax_top.axvspan(obs - 20, obs + 20, color=c, alpha=0.15, label=lbl)
        # Vertical line at the absorber centre too — easier to see in dense forests
        ax_top.axvline(obs, color=c, lw=0.6, alpha=0.6)
    ax_top.axvline(lya_obs, color="C4", lw=0.8, ls="--", alpha=0.6,
                   label=fr"Ly$\alpha$ at $z_{{qso}}\!=\!{z_qso:.3f}$")

    if args.zoom_around_truth and truth_absorbers:
        # Zoom around the strongest absorber
        z_t = max(truth_absorbers, key=lambda x: x[1])[0]
        center = 1215.67 * (1 + z_t)
        ax_top.set_xlim(center - 200, center + 200)
        ax_top.set_ylim(-0.5, 2.5)
    else:
        # Default: show the full GP model wavelength range (rest
        # min_lambda...max_lambda mapped to observed via z_qso). This
        # covers ALL pixels the model is actually fit on.
        gp_xmin = preset.min_lambda * (1 + z_qso)
        gp_xmax = preset.max_lambda * (1 + z_qso)
        # Pad slightly to keep the Lyα emission visible at the right edge.
        ax_top.set_xlim(gp_xmin - 20, max(gp_xmax + 20, lya_obs + 20))
        ax_top.set_ylim(-0.5, 2.5)

    ax_top.axhline(0, color="0.7", lw=0.5, ls=":")
    ax_top.set_ylabel("normalized flux")
    title = f"{args.mock}  TID={args.target_id}  z_qso={z_qso:.3f}"
    n_dla = sum(1 for _, n in truth_absorbers if n >= 20.3)
    n_subdla = sum(1 for _, n in truth_absorbers if 19.0 <= n < 20.3)
    n_lls = sum(1 for _, n in truth_absorbers if 17.2 <= n < 19.0)
    if truth_absorbers:
        title += f"  truth: {n_dla} DLA / {n_subdla} sub-DLA / {n_lls} LLS"
    else:
        title += "  (no truth absorber on this LOS)"
    if args.bal_catalog:
        if is_bal:
            title += "  ⚠ BAL"
        else:
            title += "  (non-BAL)"
    ax_top.set_title(title, fontsize=10)
    ax_top.legend(fontsize=7, loc="upper left")
    ax_top.grid(alpha=0.3)

    # Bottom panel: residuals vs τ-EB model (or null model if no DLA)
    model_for_residuals = enab["mu_dla"] if enab["mu_dla"] is not None else enab["mu"]
    res = (enab["y"] - model_for_residuals) / enab["sigma"]
    ax_bot.plot(enab["obs_wave"], res, color="0.3", lw=0.4)
    ax_bot.axhline(0, color="0.6", lw=0.5)
    for s in [-3, 3]:
        ax_bot.axhline(s, color="C3", lw=0.4, ls=":")
    ax_bot.set_ylim(-6, 6)
    ax_bot.set_xlabel("observed wavelength [Å]")
    ax_bot.set_ylabel(r"$(y-\hat{\mu})/\sigma$  (τ-EB)")
    ax_bot.grid(alpha=0.3)

    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"[saved] {out} ({out.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
