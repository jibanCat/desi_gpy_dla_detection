"""inspect_loa_spectra.py — single-target or multi-target GP-DLA inference + figures.

Reads spectra from either:
  - a compressed LOA archive (gpy_dla_detection/loa_archive.py), or
  - a single DESI / mock spectra-16 / coadd FITS file plus its zcat,

runs the same DLAHolder.process_qso pipeline as production, and writes:

  <out>/results.h5                     — initialize_results dict (HDF5)
  <out>/results.tsv                    — flat per-target table for spreadsheets
  <out>/figures/<tid>.png              — per-target 2-row diagnostic plot
  <out>/figures/overlay_grid.png       — multi-panel thumbnail grid (batch mode)

Designed to (a) sanity-check a re-run vs the production catalog by spot-checking
strong DLAs and Lyβ-misIDs and (b) feed the LOA postprocessing / visualization
notebook.

Usage:
  # Single LOA target
  python examples/inspect_loa_spectra.py \\
      --archive loa_archives/loa_full_z2_withR_v2.h5 \\
      --target-id 39627604172474708 \\
      --preset y3 --out /pscratch/sd/j/jibancat/inspect/single/

  # Batch from a TSV of TARGETIDs (one per line, first column)
  python examples/inspect_loa_spectra.py \\
      --archive loa_archives/loa_full_z2_withR_v2.h5 \\
      --target-ids-file targets.tsv \\
      --preset y3 --max-workers 16 \\
      --out /pscratch/sd/j/jibancat/inspect/batch/

  # Single mock target from a spectra-16 file
  python examples/inspect_loa_spectra.py \\
      --specfile /global/cfs/projectdirs/desi/mocks/.../spectra-16/0/0/spectra-16-0.fits \\
      --zcat     /global/cfs/projectdirs/desi/mocks/.../zcat.fits \\
      --target-id 12345 --preset y3 --out /pscratch/sd/j/jibancat/inspect/mock_one/

This script intentionally re-uses examples.smoke_one_spectrum.PRESETS so the
absorber-model / wavelength grid / mean-flux prior choices stay in lockstep
with the production sbatch path.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("data source (one of)")
    src.add_argument("--archive", default=None,
                     help="Path to a compressed LOA archive HDF5 "
                          "(gpy_dla_detection.loa_archive.LoaArchive). "
                          "Real-LOA path.")
    src.add_argument("--specfile", default=None,
                     help="Path to a DESI coadd or mock spectra-16 FITS. "
                          "Used with --zcat.")
    src.add_argument("--zcat", default=None,
                     help="zcat.fits with TARGETID + Z (paired with --specfile).")

    tgt = p.add_argument_group("targets (one of)")
    tgt.add_argument("--target-id", type=int, default=None,
                     help="Process this single TARGETID.")
    tgt.add_argument("--target-ids-file", default=None,
                     help="Newline- or TAB-separated file; first column is "
                          "TARGETID. '#' comments OK.")

    p.add_argument("--preset", choices=["eboss", "y3", "london"], default="y3",
                   help="Model preset from examples/smoke_one_spectrum.py "
                        "(default y3).")
    p.add_argument("--data-root",
                   default="/pscratch/sd/j/jibancat/desi_gpy_dla_detection",
                   help="Repo root used to resolve preset learned_file + samples.")
    p.add_argument("--learned-file", default=None,
                   help="Absolute path to a trained GP HDF5; overrides preset.")
    p.add_argument("--dla-samples-file", default=None,
                   help="Override DLA QMC samples (.mat).")
    p.add_argument("--sub-dla-samples-file", default=None,
                   help="Override sub-DLA QMC samples (.mat).")

    mode = p.add_argument_group("absorber mode")
    mode.add_argument("--single-absorber-model", type=int, default=0,
                      choices=[0, 1],
                      help="0 = multi-DLA (default); 1 = LLS single-absorber.")
    mode.add_argument("--max-dlas", type=int, default=3,
                      help="MAX_DLAS (default 3 for multi-DLA; use 1 for LLS).")
    mode.add_argument("--filter-low-likelihood", type=int, default=1,
                      choices=[0, 1], help="Default 1 (multi-DLA prod).")

    # Must match the row-count of --dla-samples-file. Defaults pair with
    # data/dr12q/processed/dla_samples_a03.mat (10k) +
    # data/dr12q/processed/subdla_samples.mat (10k). For LLS mode use
    # --num-dla-samples 50000 with pw_samples_a3_172_220_50000.mat.
    p.add_argument("--num-dla-samples", type=int, default=10000)
    p.add_argument("--num-subdla-samples", type=int, default=10000)
    p.add_argument("--max-workers", type=int, default=8,
                   help="GP inner-loop workers per spectrum.")
    p.add_argument("--batch-size", type=int, default=1250)

    p.add_argument("--out", required=True,
                   help="Output directory (will be created under /pscratch).")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip per-target + grid figures.")
    p.add_argument("--grid-cols", type=int, default=4,
                   help="Columns in the overlay grid (default 4).")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after N targets (for quick sanity).")
    p.add_argument("--truth-catalog", default=None,
                   help="Optional mock truth catalog (e.g. dla_cat.fits) "
                        "with TARGETID, Z_DLA, NHI; if provided, truth DLAs "
                        "are overlaid on per-target plots (mock comparison).")
    return p.parse_args()


def _load_truth_index(truth_path: str | None):
    """Returns {TARGETID: [(z_dla, log_nhi), ...]} or None."""
    if truth_path is None:
        return None
    import fitsio
    cat = fitsio.read(truth_path, ext=1)
    # Sniff columns — mocks use Z_DLA/NHI, some use Z_DLA_NO_RSD/NHI.
    # London uses Z_DLA; Saclay & 2LPT use plain Z; some variants use Z_DLA_NO_RSD.
    z_col = next((c for c in ("Z_DLA", "Z_DLA_NO_RSD", "Z")
                  if c in cat.dtype.names), None)
    n_col = "NHI" if "NHI" in cat.dtype.names else None
    if z_col is None or n_col is None:
        print(f"[truth] {truth_path}: missing Z_DLA/NHI cols — skipping overlay")
        return None
    idx: dict[int, list[tuple[float, float]]] = {}
    for r in cat:
        idx.setdefault(int(r["TARGETID"]), []).append(
            (float(r[z_col]), float(r[n_col])))
    print(f"[truth] {truth_path}: {len(idx)} TIDs, {len(cat)} truth DLAs")
    return idx


# ---------------------------------------------------------------------------
# Allowed output roots — refuse otherwise
# ---------------------------------------------------------------------------
# NERSC defaults; extra roots (e.g. GreatLakes /scratch) can be appended via
# the colon-separated env var GPDLA_ALLOWED_OUT_PREFIXES so the same script
# runs on other clusters without editing this list.
ALLOWED_OUT_PREFIXES = (
    "/pscratch/sd/j/jibancat/",
    "/global/homes/j/jibancat/",
    "/global/cfs/cdirs/desicollab/users/jibancat/",
) + tuple(
    p if p.endswith("/") else p + "/"
    for p in os.environ.get("GPDLA_ALLOWED_OUT_PREFIXES", "").split(":")
    if p
)


def assert_writable(path: str) -> None:
    real = os.path.realpath(path)
    for p in ALLOWED_OUT_PREFIXES:
        if (real + "/").startswith(p):
            return
    raise SystemExit(
        f"[refuse] output dir {real} is outside the allowed write roots. "
        f"Pick one of: {', '.join(ALLOWED_OUT_PREFIXES)}"
    )


# ---------------------------------------------------------------------------
# Target list loader
# ---------------------------------------------------------------------------
def load_target_ids(path: str) -> list[int]:
    ids: list[int] = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tok = line.split()[0]
            try:
                ids.append(int(tok))
            except ValueError:
                # tolerate header rows like "TARGETID\tZ"
                continue
    if not ids:
        raise SystemExit(f"[error] no TARGETIDs parsed from {path}")
    return ids


# ---------------------------------------------------------------------------
# Spectrum readers — archive vs FITS
# ---------------------------------------------------------------------------
def load_from_archive(archive_path: str, tids: list[int]):
    """Yield (target_id, z_qso, wave, flux, noise_var, mask) from LoaArchive."""
    sys.path.insert(0, os.getcwd())
    from gpy_dla_detection.loa_archive import LoaArchive

    with LoaArchive(archive_path) as ar:
        for tid in tids:
            try:
                spec = ar.get_spectrum(tid)
            except KeyError:
                print(f"[skip] TARGETID {tid} not in archive {archive_path}")
                continue
            wave = spec.wavelength.astype(np.float64)
            flux = spec.flux.astype(np.float64)
            ivar = spec.ivar.astype(np.float64)
            nv = np.full_like(flux, np.inf)
            good = ivar > 0
            nv[good] = 1.0 / ivar[good]
            mask = spec.mask.astype(bool)
            yield int(tid), float(spec.z), wave, flux, nv, mask


def load_from_fits(specfile: str, zcatfile: str, tids: list[int]):
    """Yield (target_id, z_qso, wave, flux, noise_var, mask) using
    examples.smoke_one_spectrum.load_one_desi_spectrum + lookup_z_qso."""
    sys.path.insert(0, os.getcwd())
    from examples.smoke_one_spectrum import load_one_desi_spectrum, lookup_z_qso

    for tid in tids:
        try:
            wave, flux, nv, mask = load_one_desi_spectrum(specfile, tid)
            z = lookup_z_qso(zcatfile, tid)
        except Exception as exc:
            print(f"[skip] TARGETID {tid}: {exc}")
            continue
        yield int(tid), float(z), wave, flux, nv, mask


# ---------------------------------------------------------------------------
# Holder builder — uses smoke_one_spectrum PRESETS
# ---------------------------------------------------------------------------
def build_holder(args):
    sys.path.insert(0, os.getcwd())
    from examples.smoke_one_spectrum import PRESETS
    from gpy_dla_detection.set_parameters import Parameters
    from run_bayes_select import DLAHolder

    preset = PRESETS[args.preset]

    def under_root(rel: str) -> str:
        return os.path.join(args.data_root, rel)

    learned_file = args.learned_file or under_root(preset.learned_file)
    catalog_name = under_root("data/dr12q/processed/catalog.mat")
    los_catalog = under_root("data/dla_catalogs/dr9q_concordance/processed/los_catalog")
    dla_catalog = under_root("data/dla_catalogs/dr9q_concordance/processed/dla_catalog")
    dla_samples_file = args.dla_samples_file or under_root(
        "data/dr12q/processed/dla_samples_a03.mat")
    sub_dla_samples_file = args.sub_dla_samples_file or under_root(
        "data/dr12q/processed/subdla_samples.mat")

    missing = [p for p in [learned_file, catalog_name, los_catalog, dla_catalog,
                           dla_samples_file, sub_dla_samples_file]
               if not os.path.exists(p)]
    if missing:
        raise SystemExit("[error] missing inputs:\n  " + "\n  ".join(missing))

    common = dict(
        loading_min_lambda=preset.loading_min_lambda,
        loading_max_lambda=preset.loading_max_lambda,
        normalization_min_lambda=preset.normalization_min_lambda,
        normalization_max_lambda=preset.normalization_max_lambda,
        min_lambda=preset.min_lambda,
        max_lambda=preset.max_lambda,
        dlambda=preset.dlambda,
        k=preset.k,
        max_noise_variance=9.0,
        num_lines=preset.num_lines,
        max_z_cut=3000.0,
        min_z_cut=3000.0,
        num_forest_lines=preset.num_forest_lines,
    )
    params = Parameters(num_dla_samples=args.num_dla_samples, **common)
    params_subdla = Parameters(num_dla_samples=args.num_subdla_samples, **common)

    holder = DLAHolder(
        learned_file=learned_file,
        catalog_name=catalog_name,
        los_catalog=los_catalog,
        dla_catalog=dla_catalog,
        dla_samples_file=dla_samples_file,
        sub_dla_samples_file=sub_dla_samples_file,
        params=params,
        params_subdla=params_subdla,
        min_z_separation=3000.0,
        prev_tau_0=preset.prev_tau_0,
        prev_beta=preset.prev_beta,
        max_dlas=args.max_dlas,
        broadening=True,
        plot_figures=False,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        figure_dir="figures/",
        filter_low_likelihood=bool(args.filter_low_likelihood),
        single_absorber_model=bool(args.single_absorber_model),
    )
    return holder, preset


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
LYA = 1215.67


def _build_model_curves(holder, preset, wave, flux, nv, mask, z_qso,
                        map_z, map_nhi):
    """Build the canonical model overlays in REST-FRAME wavelength space.

    Mirrors plot_real_spectrum_space (CDDF_analysis & plottings/plot_model.py):
      - NullGPMAT.Y is the *normalized* flux on rest-frame grid gp.X
      - NullGPMAT.this_mu  = continuum × Lyα-forest absorption (μ_null)
      - DLAGPMAT.this_dla_gp(map_z, 10**map_nhi) → lya_mu = continuum × Lyα ×
        Voigt(DLAs), so the pure Voigt-only profile is `lya_mu / this_mu`.

    Returns
    -------
    rest_x         : (n_pix,)   rest-frame wavelength used by the GP
    obs_x          : (n_pix,)   observed wavelength (= rest * (1+z_qso))
    y_norm         : (n_pix,)   normalized flux
    sigma_norm     : (n_pix,)   normalized 1-σ noise
    mu_null        : (n_pix,)   GP null mean (continuum × Lyα)
    mu_with_dla    : (n_pix,)   GP + Voigt DLAs at MAP (None if no DLAs)
    """
    from gpy_dla_detection.null_gp import NullGPMAT
    from gpy_dla_detection.dla_gp import DLAGPMAT

    rest_w = holder.params.emitted_wavelengths(wave, z_qso)
    null_gp = NullGPMAT(
        holder.params, holder.prior,
        learned_file=holder.learned_file,
        prev_tau_0=preset.prev_tau_0, prev_beta=preset.prev_beta,
    )
    null_gp.set_data(rest_w, flux, nv, mask, z_qso, build_model=True)
    rest_x = null_gp.X
    obs_x = rest_x * (1.0 + z_qso)
    y_norm = null_gp.Y
    sigma_norm = np.sqrt(np.where(np.isfinite(null_gp.v) & (null_gp.v > 0),
                                  null_gp.v, np.nan))
    mu_null = null_gp.this_mu

    # Voigt overlay: only if we have at least one finite MAP DLA
    z_arr = np.atleast_1d(map_z)
    n_arr = np.atleast_1d(map_nhi)
    finite = np.isfinite(z_arr) & np.isfinite(n_arr)
    mu_with_dla = None
    if finite.any():
        try:
            dla_gp = DLAGPMAT(
                holder.params, holder.prior, holder.dla_samples,
                min_z_separation=holder.min_z_separation,
                learned_file=holder.learned_file,
                broadening=holder.broadening,
                prev_tau_0=preset.prev_tau_0, prev_beta=preset.prev_beta,
            )
            dla_gp.set_data(rest_w, flux, nv, mask, z_qso, build_model=True)
            z_in = z_arr[finite]
            nhi_in = 10.0 ** n_arr[finite]
            lya_mu, _, _ = dla_gp.this_dla_gp(z_in, nhi_in)
            absorption = lya_mu / dla_gp.this_mu
            _continuum = dla_gp.mu_interpolator(dla_gp.X)
            mu_with_dla = _continuum * absorption
        except Exception as exc:
            print(f"  [overlay-fail] {exc}")

    return rest_x, obs_x, y_norm, sigma_norm, mu_null, mu_with_dla


def plot_one(out_png, tid, z_qso, rest_x, y_norm, sigma_norm,
             mu_null, mu_with_dla, map_z, map_nhi, p_dla, p_no_dla,
             truth_z=None, truth_nhi=None):
    """Per-target 2-row figure following the production convention
    (gpy_dla_detection.plottings.plot_model.plot_real_spectrum_space).

    Top:  normalized flux on REST-frame wavelength + ±2σ noise band +
          GP null model (continuum × Lyα) + GP+Voigt(DLA at MAP) overlay.
    Bottom: residuals (y - model) / σ, where model = mu_with_dla if any
          MAP DLAs survived, else mu_null.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5),
                             gridspec_kw=dict(height_ratios=[3, 1]), sharex=True)
    ax0, ax1 = axes

    # ±2σ noise band (the production convention; instrumental uncertainty 95%)
    ax0.fill_between(rest_x,
                     y_norm - 2 * sigma_norm,
                     y_norm + 2 * sigma_norm,
                     color="C0", alpha=0.20, lw=0,
                     label="Instrumental uncertainty (95%)")
    ax0.plot(rest_x, y_norm, lw=0.6, color="C0", label="Data (normalized)")

    # GP null = continuum × Lyα forest absorption
    ax0.plot(rest_x, mu_null, lw=1.2, color="C3", ls="--",
             label=r"GP null  ($\mu \times A_{\mathrm{Ly}\alpha}$)")

    # GP + Voigt(DLA(s)) at the MAP — the actual best-fit model
    if mu_with_dla is not None:
        ax0.plot(rest_x, mu_with_dla, lw=1.4, color="C1",
                 label=r"GP + Voigt DLA(s) at MAP")

    # MAP DLA markers (in REST frame: z_dla → rest λ via Lyα * (1+z_dla)/(1+z_qso))
    z_arr = np.atleast_1d(map_z)
    n_arr = np.atleast_1d(map_nhi)
    for i, (zi, ni) in enumerate(zip(z_arr, n_arr)):
        if not (np.isfinite(zi) and np.isfinite(ni)):
            continue
        lam_rest = LYA * (1 + zi) / (1 + z_qso)
        ax0.axvline(lam_rest, color="C1", lw=0.7, ls=":")
        ax0.annotate(fr"$z={zi:.3f}$" + "\n" + fr"$\log N_{{HI}}={ni:.2f}$",
                     xy=(lam_rest, ax0.get_ylim()[1]),
                     xytext=(2, -2), textcoords="offset points",
                     fontsize=7, color="C1", ha="left", va="top")

    # Optional truth DLA marker (mock comparison)
    if truth_z is not None and truth_nhi is not None:
        for tz, tn in zip(np.atleast_1d(truth_z), np.atleast_1d(truth_nhi)):
            if not (np.isfinite(tz) and np.isfinite(tn)):
                continue
            lam_rest = LYA * (1 + tz) / (1 + z_qso)
            ax0.axvline(lam_rest, color="C2", lw=0.9, alpha=0.7)
            ax0.annotate(f"truth z={tz:.3f}\nlog NHI={tn:.2f}",
                         xy=(lam_rest, ax0.get_ylim()[0]),
                         xytext=(2, 4), textcoords="offset points",
                         fontsize=7, color="C2", ha="left", va="bottom")

    ax0.set_ylabel("Normalized flux")
    ax0.set_ylim(-1, 5)
    ax0.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax0.set_title(
        f"TARGETID {tid}   $z_{{\\mathrm{{QSO}}}}={z_qso:.3f}$   "
        f"p(DLA)={p_dla:.3f}   p(no abs)={p_no_dla:.3f}"
    )

    # Residuals — use mu_with_dla if present else mu_null
    model = mu_with_dla if mu_with_dla is not None else mu_null
    resid = (y_norm - model) / np.where(sigma_norm > 0, sigma_norm, np.nan)
    ax1.axhline(0, color="0.4", lw=0.5)
    ax1.fill_between(rest_x, -2, 2, color="C0", alpha=0.15, lw=0)
    ax1.plot(rest_x, resid, lw=0.4, color="0.2")
    ax1.set_ylim(-6, 6)
    ax1.set_ylabel(r"(data $-$ model) / $\sigma$")
    ax1.set_xlabel(r"Rest-frame wavelength [$\mathrm{\AA}$]")
    ax1.set_xlim(rest_x[0], rest_x[-1])

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def plot_overlay_grid(out_png, panels, cols):
    """Grid of thumbnails. Each `panels` row =
    (tid, z_qso, rest_x, y_norm, mu_null, mu_with_dla, map_z, map_nhi, p_dla).
    Plotted on rest-frame wavelength, normalized flux, with the same
    GP-null + Voigt overlay as the single-target panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 2.0 * rows),
                             squeeze=False, sharex=False, sharey=False)
    for i, (tid, z_qso, rest_x, y_norm, mu_null, mu_with_dla,
            map_z, map_nhi, p_dla) in enumerate(panels):
        ax = axes[i // cols][i % cols]
        ax.plot(rest_x, y_norm, lw=0.4, color="C0")
        ax.plot(rest_x, mu_null, lw=0.7, color="C3", ls="--")
        if mu_with_dla is not None:
            ax.plot(rest_x, mu_with_dla, lw=0.9, color="C1")
        for zi, ni in zip(np.atleast_1d(map_z), np.atleast_1d(map_nhi)):
            if np.isfinite(zi) and np.isfinite(ni):
                lam_rest = LYA * (1 + zi) / (1 + z_qso)
                ax.axvline(lam_rest, color="C1", lw=0.6, alpha=0.6)
        ax.set_xlim(rest_x[0], rest_x[-1])
        ax.set_ylim(-1, 5)
        ax.set_title(f"{tid}  z={z_qso:.3f}  p(DLA)={p_dla:.2f}",
                     fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("rest-frame λ [Å]  ·  blue = data  red dashed = GP null  "
                 "orange = GP + Voigt DLAs (MAP)",
                 y=1.0, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Source-validation
    if (args.archive is None) == (args.specfile is None):
        raise SystemExit("Provide EITHER --archive OR (--specfile + --zcat).")
    if args.specfile and not args.zcat:
        raise SystemExit("--specfile requires --zcat.")
    if (args.target_id is None) == (args.target_ids_file is None):
        raise SystemExit("Provide EITHER --target-id OR --target-ids-file.")

    assert_writable(args.out)
    out = Path(args.out)
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        fig_dir.mkdir(parents=True, exist_ok=True)

    # Collect TARGETIDs
    if args.target_id is not None:
        tids = [int(args.target_id)]
    else:
        tids = load_target_ids(args.target_ids_file)
    if args.limit:
        tids = tids[: args.limit]
    print(f"[inspect] {len(tids)} TARGETID(s) requested  preset={args.preset}  "
          f"max_dlas={args.max_dlas}  single_absorber={args.single_absorber_model}")

    holder, preset = build_holder(args)
    holder.initialize_results(len(tids))
    truth_idx = _load_truth_index(args.truth_catalog)

    # Iterate
    if args.archive:
        stream = load_from_archive(args.archive, tids)
    else:
        stream = load_from_fits(args.specfile, args.zcat, tids)

    rows: list[dict] = []
    panels: list[tuple] = []
    actual_idx = 0
    for src_idx, (tid, z_qso, wave, flux, nv, mask) in enumerate(stream):
        t0 = time.time()
        try:
            holder.process_qso(
                idx=actual_idx, target_id=str(tid),
                wavelengths=wave, flux=flux,
                noise_variance=nv, pixel_mask=mask, z_qso=z_qso,
            )
        except Exception as exc:
            print(f"[error] TARGETID {tid}: {exc}")
            continue
        dt = time.time() - t0
        res = holder.results
        p_dla = float(res["p_dlas"][actual_idx])
        p_no = float(res["p_no_dlas"][actual_idx])
        z_dla = res["MAP_z_dlas"][actual_idx].copy()
        nhi_dla = res["MAP_log_nhis"][actual_idx].copy()
        snr = float(res["snrs"][actual_idx]) if "snrs" in res else float("nan")
        print(f"[{actual_idx+1}/{len(tids)}] TID={tid} z_qso={z_qso:.3f} "
              f"p(DLA)={p_dla:.3f} MAP_z={z_dla[0]:.3f} "
              f"MAP_logNHI={nhi_dla[0]:.2f} ({dt:.1f}s)")

        # Build GP overlay + (if requested) per-target plot
        if not args.no_plots:
            try:
                rest_x, obs_x, y_norm, sigma_norm, mu_null, mu_with_dla = \
                    _build_model_curves(holder, preset, wave, flux, nv, mask,
                                        z_qso, z_dla, nhi_dla)
                tid_png = fig_dir / f"{tid}.png"
                t_z, t_n = (None, None)
                if truth_idx and int(tid) in truth_idx:
                    t_z = [t[0] for t in truth_idx[int(tid)]]
                    t_n = [t[1] for t in truth_idx[int(tid)]]
                plot_one(str(tid_png), tid, z_qso, rest_x, y_norm, sigma_norm,
                         mu_null, mu_with_dla, z_dla, nhi_dla, p_dla, p_no,
                         truth_z=t_z, truth_nhi=t_n)
                panels.append((tid, z_qso, rest_x, y_norm, mu_null,
                               mu_with_dla, z_dla, nhi_dla, p_dla))
            except Exception as exc:
                print(f"  [plot-fail] {exc}")

        rows.append(dict(
            target_id=tid, z_qso=z_qso, p_dla=p_dla, p_no_dlas=p_no,
            map_z_0=float(z_dla[0]) if z_dla.size else float("nan"),
            map_log_nhi_0=float(nhi_dla[0]) if nhi_dla.size else float("nan"),
            snr=snr, inference_seconds=round(dt, 2),
        ))
        actual_idx += 1

    # If process_qso raised for a target above, actual_idx is left at the
    # failed slot — subsequent successes overwrote it. Keep the printed
    # tracker honest by reflecting successful rows only.

    # Save outputs
    results_h5 = out / "results.h5"
    holder.save_results(str(results_h5))
    print(f"[saved] {results_h5}")

    tsv = out / "results.tsv"
    if rows:
        with open(tsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"[saved] {tsv}  ({len(rows)} targets)")

    if not args.no_plots and panels:
        grid_png = fig_dir / "overlay_grid.png"
        plot_overlay_grid(str(grid_png), panels, cols=args.grid_cols)
        print(f"[saved] {grid_png}")

    print(f"[done] {actual_idx} target(s) processed → {out}")


if __name__ == "__main__":
    main()
