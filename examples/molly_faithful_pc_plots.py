"""molly_faithful_pc_plots.py — exact reproduction of Molly Wolfson's
purity/completeness notebook plots for the DLA-mode catalog.

Recipe transcribed from
  /pscratch/sd/j/jibancat/molly/read_in_each_plots_saclay-Y3-Learned.ipynb

Differences from gp_native_pc_plots.py (which is "inspired by" molly):
  * predicted NHI column   = "NHI"        (molly uses NHI_TMP for template, NHI for GP)
  * confidence column      = "P_DLA"      (sweep over log_pdla ∈ {-1,…,-8})
  * SNR column             = "SNR_REDSIDE" (molly's S2N_RED; pulled from dlacat or h5)
  * goodness gate          = DLAFLAG == 0 (good_mask)
  * truth-matching         = per-TARGETID, |Δz|/(1+z) < 0.01, greedy (cell 12 of molly's nb)
  * Z range cut            = z_qso ∈ [2.0, 4.25]
  * λ_rf range cut         = [1025, 1216] Å (cell 21; "OmegaDLA" cuts)
  * BAL removal            = TARGETID ∈ bal_cat → dropped from BOTH cat AND truth
  * truth-NHI floor        = NHI > 20.3
  * snr_min, nhi_min       = 6.0, 20.3 (cell 41)

Four plot panels:
  (1) purity & completeness vs P(DLA) cut (a P_DLA in {1 - 10**log_pdla} for
      log_pdla ∈ {-1,-2,…,-8})   — molly cell 43
  (2) 1D S2N-binned purity & completeness at gp_conf = 0.99  — molly cells 47-60
  (3) (S2N, predicted log NHI) heatmap — purity                — molly cells 68-76
  (4) (S2N, truth log NHI) heatmap     — completeness          — molly cell 77

Per-QSO SNR_REDSIDE (used for the TRUTH catalog's "S2N_RED" column) is built
from the processed-spectra-16-*.h5 files (each h5 has `target_ids` and `snrs`
arrays — `snrs` is the redside SNR, matching dlacat.SNR_REDSIDE).

Headline output (`summary.tsv`):
  P_DLA cut, NHI bin, snr_min, n_TP, n_kept, n_truth_kept, purity, completeness.

Usage:
  python examples/molly_faithful_pc_plots.py \\
      --catalog-dir /pscratch/.../london0_y3/ \\
      --truth /global/cfs/.../jura-124/dla_cat.fits \\
      --bal-cat /global/cfs/.../jura-124/bal_cat.fits --no-bal \\
      --truth-nhi-min 20.3 \\
      --out /pscratch/.../london0_y3/figures_molly/
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import h5py
import fitsio
from astropy.table import Table, vstack

# Reuse load + match helpers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gp_native_pc_plots import (
    load_catalog_dir,
    apply_bal_cut as _apply_bal_cut,
    match_truth_to_cat,
)

LYA = 1215.67
SPEED_C = 299792.458  # km/s


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog-dir", required=True,
                   help="Run OUTDIR with dlacat-*.fits and processed/processed-spectra-16-*.h5")
    p.add_argument("--truth", required=True,
                   help="Mock truth: London dla_cat.fits OR Saclay/2LPT hcd_truth_cat.fits")
    p.add_argument("--bal-cat", default=None,
                   help="bal_cat.fits with BI_CIV column (and TARGETID).")
    p.add_argument("--no-bal", action="store_true",
                   help="Exclude BAL TIDs from BOTH cat and truth (molly convention).")
    p.add_argument("--truth-nhi-min", type=float, default=20.3,
                   help="Truth NHI floor (default 20.3).")
    p.add_argument("--dz-rel", type=float, default=0.01,
                   help="|Δz|/(1+z_truth) match tolerance (default 0.01).")
    p.add_argument("--z-qso-min", type=float, default=2.0)
    p.add_argument("--z-qso-max", type=float, default=4.25)
    p.add_argument("--lam-rf-min", type=float, default=None,
                   help="Rest-frame λ min for Z_DLA cut. If unset, runs BOTH windows: "
                        "lya_only=[1025,1216] AND lya_lyb=[911,1216].")
    p.add_argument("--lam-rf-max", type=float, default=1216.,
                   help="Rest-frame λ max for Z_DLA cut (default 1216).")
    p.add_argument("--lyb-veto", action="store_true",
                   help="Apply postprocess.lyb_veto.flag_lybeta and drop LYBETA_FLAG rows.")
    p.add_argument("--lyb-veto-dz", type=float, default=0.005,
                   help="dz_match for lyb_veto (default 0.005).")
    p.add_argument("--snr-min", type=float, default=6.0,
                   help="Min S2N_RED (default 6.0).")
    p.add_argument("--nhi-min", type=float, default=20.3,
                   help="Min predicted/truth log NHI for headline numbers (default 20.3).")
    p.add_argument("--gp-conf", type=float, default=0.99,
                   help="Fixed P_DLA cut for 1D + 2D plots (molly used 0.99 for Saclay DLA).")
    p.add_argument("--zcat", default=None,
                   help="Optional zcat.fits for Z_QSO lookup (if not in mockdir).")
    p.add_argument("--mockdir", default=None,
                   help="If --zcat unset, look for zcat.fits in this MOCKDIR.")
    p.add_argument("--out", required=True,
                   help="Output dir for PNGs + summary tsv.")
    p.add_argument("--title", default=None,
                   help="Figure title (default: catalog-dir basename).")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Per-QSO SNR_REDSIDE lookup from processed h5 files
# -----------------------------------------------------------------------------
def build_per_qso_snr(catalog_dir: str) -> dict[int, tuple[float, float]]:
    """Build TARGETID → (snr_redside, z_qso) from processed-spectra-16-*.h5.

    The processed h5 files have:
      - target_ids : (N,) int64
      - snrs       : (N,) float64  → red-side SNR (matches dlacat.SNR_REDSIDE)
      - z_qsos     : (N,) float64
    """
    paths = sorted(glob.glob(os.path.join(catalog_dir, "processed",
                                          "processed-spectra-16-*.h5")))
    if not paths:
        # fallback: maybe outdir uses old layout
        paths = sorted(glob.glob(os.path.join(catalog_dir,
                                              "processed-spectra-16-*.h5")))
    if not paths:
        raise SystemExit(f"[error] no processed-spectra-16-*.h5 in {catalog_dir}")
    lookup: dict[int, tuple[float, float]] = {}
    for p in paths:
        with h5py.File(p, "r") as f:
            tids = np.asarray(f["target_ids"][:], dtype=np.int64)
            snrs = np.asarray(f["snrs"][:], dtype=float)
            zq = np.asarray(f["z_qsos"][:], dtype=float)
        for t, s, z in zip(tids, snrs, zq):
            lookup[int(t)] = (float(s), float(z))
    print(f"[snr] built per-QSO lookup: {len(lookup)} TIDs from {len(paths)} h5 files")
    return lookup


# -----------------------------------------------------------------------------
# Load truth (rename Z, trim NHI, attach SNR_REDSIDE + Z_QSO via lookup)
# -----------------------------------------------------------------------------
def load_truth_molly(path: str, nhi_min: float,
                     qso_lookup: dict[int, tuple[float, float]],
                     zcat_path: str | None) -> Table:
    tr = Table(fitsio.read(path, ext=1))
    z_col = next((c for c in ("Z_DLA", "Z_DLA_NO_RSD", "Z") if c in tr.colnames), None)
    if z_col is None:
        raise SystemExit(f"truth has no Z_DLA/Z col: {tr.colnames}")
    tr.rename_column(z_col, "Z_DLA")
    # gp_native_pc_plots.match_truth_to_cat reads truth["Z_TRUTH"]; alias it.
    tr["Z_TRUTH"] = np.asarray(tr["Z_DLA"], dtype=float)
    if nhi_min > 0:
        tr = tr[np.asarray(tr["NHI"]) >= nhi_min]
    print(f"[load] truth: {len(tr)} DLAs (NHI≥{nhi_min})")

    # Add S2N_RED and Z_QSO from per-QSO lookup
    tids = np.asarray(tr["TARGETID"], dtype=np.int64)
    s2n = np.full(len(tr), np.nan)
    zq = np.full(len(tr), np.nan)
    miss = 0
    for i, t in enumerate(tids):
        v = qso_lookup.get(int(t))
        if v is None:
            miss += 1
            continue
        s2n[i], zq[i] = v

    # Optional zcat fallback for missing Z_QSO (truth-only TIDs that weren't processed)
    if miss and zcat_path and os.path.exists(zcat_path):
        zc = fitsio.read(zcat_path, ext=1, columns=["TARGETID", "Z"])
        zcat_map = {int(r["TARGETID"]): float(r["Z"]) for r in zc}
        for i, t in enumerate(tids):
            if np.isnan(zq[i]):
                z = zcat_map.get(int(t))
                if z is not None:
                    zq[i] = z

    print(f"[snr] truth missing S2N_RED for {miss}/{len(tr)} TIDs "
          f"(no processed-h5 record — these have NaN S2N_RED)")

    tr["S2N_RED"] = s2n
    tr["Z_QSO"] = zq

    # Drop rows that have no SNR info — can't apply the SNR cut to them anyway
    keep = ~np.isnan(s2n) & ~np.isnan(zq)
    if (~keep).sum() > 0:
        print(f"[snr] dropping {(~keep).sum()} truth rows with no SNR/Z_QSO")
    tr = tr[keep]
    return tr


# -----------------------------------------------------------------------------
# Molly's λ_rf + z_qso + BAL cuts (cell 19, 21)
# -----------------------------------------------------------------------------
def make_lambda_z_BAL_cuts(cat: Table, lam_rf_min: float, lam_rf_max: float,
                           z_qso_min: float, z_qso_max: float,
                           bal_tids: set[int] | None = None,
                           z_col_for_min: str = "Z_DLA",
                           use_truth_z: bool = True) -> Table:
    """Apply molly's cut bundle. Mirrors `make_lambda_z_BAL_qso_cuts`.

    For detection catalogs that have both Z_DLA and Z_TRUE (truth-matched
    cat), the λ_rf cut is applied to *both* (using min/max of the pair so
    that nan-Z_TRUE doesn't get a free pass).
    For the truth catalog, only Z_DLA exists.
    """
    z_dla = np.asarray(cat[z_col_for_min], dtype=float)
    z_qso = np.asarray(cat["Z_QSO"], dtype=float)

    # λ_z bounds per QSO (cell 19); we slightly simplify by using the
    # rest-frame cut directly (= lam_rf*(1+z_qso)/LYA - 1, with the proximity
    # collar set to ~3000 km/s = 3000/c in z units; matches cell 19).
    collar = 3000.0 / SPEED_C
    z_lo = np.maximum(3600. / LYA - 1.,
                      lam_rf_min * (1 + z_qso) / LYA - 1 + collar)
    z_hi = np.minimum(z_qso - collar,
                      lam_rf_max * (1 + z_qso) / LYA - 1 - collar)

    if use_truth_z and "Z_TRUE" in cat.colnames:
        z_tr = np.asarray(cat["Z_TRUE"], dtype=float)
        # replace NaN with z_dla for the comparison (cell 19)
        z_tr = np.where(np.isnan(z_tr), z_dla, z_tr)
        z_combined_min = np.minimum(z_dla, z_tr)
        z_combined_max = np.maximum(z_dla, z_tr)
    else:
        z_combined_min = z_dla
        z_combined_max = z_dla

    mask = (z_combined_max < z_hi) & (z_combined_min > z_lo)
    mask &= (z_qso > z_qso_min) & (z_qso < z_qso_max)

    if bal_tids is not None:
        tids = np.asarray(cat["TARGETID"], dtype=np.int64)
        mask &= ~np.isin(tids, np.fromiter(bal_tids, dtype=np.int64))
    return cat[mask]


# -----------------------------------------------------------------------------
# Molly's cut functions (cells 38, 63)
# -----------------------------------------------------------------------------
def purity_min(cat: Table, tp: np.ndarray, min_snr: float, min_pred_nhi: float,
               min_goodness: float, good_mask: np.ndarray | None,
               nhi_key: str = "NHI", goodness_key: str = "P_DLA"
               ) -> tuple[int, int, float]:
    s2n = np.asarray(cat["S2N_RED"], dtype=float)
    nhi = np.asarray(cat[nhi_key], dtype=float)
    g = np.asarray(cat[goodness_key], dtype=float)
    m = (s2n > min_snr) & (nhi > min_pred_nhi) & (g > min_goodness)
    if good_mask is not None:
        m &= good_mask
    ntot = int(m.sum())
    ntp = int(tp[m].sum())
    return ntp, ntot, (ntp / ntot if ntot else np.nan)


def purity_min_max_snr(cat, tp, min_snr, max_snr, min_pred_nhi, min_goodness,
                       good_mask=None, nhi_key="NHI", goodness_key="P_DLA"):
    s2n = np.asarray(cat["S2N_RED"], dtype=float)
    nhi = np.asarray(cat[nhi_key], dtype=float)
    g = np.asarray(cat[goodness_key], dtype=float)
    m = (s2n > min_snr) & (s2n < max_snr) & (nhi > min_pred_nhi) & (g > min_goodness)
    if good_mask is not None:
        m &= good_mask
    ntot = int(m.sum())
    ntp = int(tp[m].sum())
    return ntp, ntot, (ntp / ntot if ntot else np.nan)


def completeness_min(cat, tp, min_snr, min_true_nhi, min_pred_nhi, min_goodness,
                     mock_cat, good_mask=None, nhi_key="NHI", goodness_key="P_DLA"):
    s2n = np.asarray(cat["S2N_RED"], dtype=float)
    nhi = np.asarray(cat[nhi_key], dtype=float)
    g = np.asarray(cat[goodness_key], dtype=float)
    m = (s2n > min_snr) & (nhi > min_pred_nhi) & (g > min_goodness)
    if good_mask is not None:
        m &= good_mask

    s2n_m = np.asarray(mock_cat["S2N_RED"], dtype=float)
    nhi_m = np.asarray(mock_cat["NHI"], dtype=float)
    mock_mask = (s2n_m > min_snr) & (nhi_m > min_true_nhi)
    n_found = int(tp[m].sum())
    n_fid = int(mock_mask.sum())
    return n_found, n_fid, (n_found / n_fid if n_fid else np.nan)


def completeness_min_max_snr(cat, tp, min_snr, max_snr, min_true_nhi, min_pred_nhi,
                             min_goodness, mock_cat, good_mask=None,
                             nhi_key="NHI", goodness_key="P_DLA"):
    s2n = np.asarray(cat["S2N_RED"], dtype=float)
    nhi = np.asarray(cat[nhi_key], dtype=float)
    g = np.asarray(cat[goodness_key], dtype=float)
    m = (s2n > min_snr) & (s2n < max_snr) & (nhi > min_pred_nhi) & (g > min_goodness)
    if good_mask is not None:
        m &= good_mask
    s2n_m = np.asarray(mock_cat["S2N_RED"], dtype=float)
    nhi_m = np.asarray(mock_cat["NHI"], dtype=float)
    mock_mask = (s2n_m > min_snr) & (s2n_m < max_snr) & (nhi_m > min_true_nhi)
    n_found = int(tp[m].sum())
    n_fid = int(mock_mask.sum())
    return n_found, n_fid, (n_found / n_fid if n_fid else np.nan)


def purity_snr_nhi_bins(cat, tp, min_snr, max_snr, min_pred_nhi, max_pred_nhi,
                        min_goodness, good_mask=None, nhi_key="NHI",
                        goodness_key="P_DLA"):
    s2n = np.asarray(cat["S2N_RED"], dtype=float)
    nhi = np.asarray(cat[nhi_key], dtype=float)
    g = np.asarray(cat[goodness_key], dtype=float)
    m = (s2n > min_snr) & (s2n < max_snr)
    m &= (nhi > min_pred_nhi) & (nhi < max_pred_nhi)
    m &= g > min_goodness
    if good_mask is not None:
        m &= good_mask
    ntot = int(m.sum())
    ntp = int(tp[m].sum())
    return ntp, ntot, (ntp / ntot if ntot else np.nan)


def completeness_snr_nhi_bins(cat, tp, min_snr, max_snr, min_true_nhi, max_true_nhi,
                              min_pred_nhi, min_goodness, mock_cat, good_mask=None,
                              nhi_key="NHI", goodness_key="P_DLA"):
    s2n = np.asarray(cat["S2N_RED"], dtype=float)
    nhi = np.asarray(cat[nhi_key], dtype=float)
    nhi_true = np.asarray(cat["NHI_TRUE"], dtype=float)
    g = np.asarray(cat[goodness_key], dtype=float)
    m = (s2n > min_snr) & (s2n < max_snr)
    m &= (nhi_true > min_true_nhi) & (nhi_true < max_true_nhi)
    m &= nhi > min_pred_nhi
    m &= g > min_goodness
    if good_mask is not None:
        m &= good_mask
    s2n_m = np.asarray(mock_cat["S2N_RED"], dtype=float)
    nhi_m = np.asarray(mock_cat["NHI"], dtype=float)
    mock_mask = (s2n_m > min_snr) & (s2n_m < max_snr)
    mock_mask &= (nhi_m > min_true_nhi) & (nhi_m < max_true_nhi)
    n_found = int(tp[m].sum())
    n_fid = int(mock_mask.sum())
    return n_found, n_fid, (n_found / n_fid if n_fid else np.nan)


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    return matplotlib


def plot_pc_vs_pdla(cat, tp, mock_cat, good_mask, out_png, title,
                    snr_min=6.0, nhi_min_both=20.3):
    """Molly cell 43: log_pdla_minimums = [-1,-3,-4,-5,-6,-7,-8,-2]."""
    setup_mpl()
    import matplotlib.pyplot as plt

    log_pdla = np.array([-1., -2., -3., -4., -5., -6., -7., -8.])
    pdla_min = 1. - 10.**log_pdla

    pur = np.empty(len(pdla_min))
    cmp_ = np.empty(len(pdla_min))
    for i, pc in enumerate(pdla_min):
        _, _, pur[i] = purity_min(cat, tp, snr_min, nhi_min_both, pc, good_mask,
                                   nhi_key="NHI", goodness_key="P_DLA")
        _, _, cmp_[i] = completeness_min(cat, tp, snr_min, nhi_min_both, nhi_min_both,
                                          pc, mock_cat, good_mask,
                                          nhi_key="NHI", goodness_key="P_DLA")

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.plot(log_pdla, pur, "o-", color="C0", label="Purity")
    ax.plot(log_pdla, cmp_, "s-", color="C3", label="Completeness")
    ax.axhline(0.85, ls=":", color="k", lw=0.7, label="85% target")
    ax.set_xlabel(r"$\log_{10}(1 - P_{\rm DLA, min})$")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{title}: P/C vs P(DLA) cut "
                 f"(snr>{snr_min}, NHI>{nhi_min_both}, DLAFLAG==0, no BAL)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return log_pdla, pur, cmp_


def plot_pc_vs_snr(cat, tp, mock_cat, good_mask, out_png, title,
                   gp_conf=0.99, nhi_min_both=20.3):
    """Molly cells 47-60: 1D SNR-binned plot at fixed P_DLA cut."""
    setup_mpl()
    import matplotlib.pyplot as plt

    log10_snrs = np.linspace(-0.5, 1.5, 25)
    p = np.empty(len(log10_snrs) - 1)
    c = np.empty(len(log10_snrs) - 1)
    for i in range(len(log10_snrs) - 1):
        _, _, p[i] = purity_min_max_snr(cat, tp, 10.**log10_snrs[i],
                                         10.**log10_snrs[i + 1],
                                         nhi_min_both, gp_conf, good_mask,
                                         nhi_key="NHI", goodness_key="P_DLA")
        _, _, c[i] = completeness_min_max_snr(cat, tp, 10.**log10_snrs[i],
                                               10.**log10_snrs[i + 1],
                                               nhi_min_both, nhi_min_both, gp_conf,
                                               mock_cat, good_mask,
                                               nhi_key="NHI", goodness_key="P_DLA")

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    snr_mid = 10.**((log10_snrs[:-1] + log10_snrs[1:]) / 2)
    ax.plot(snr_mid, p, "o-", color="C0", label="Purity")
    ax.plot(snr_mid, c, "s-", color="C3", label="Completeness")
    ax.axhline(0.85, ls=":", color="k", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("S/N (S2N_RED) bin centre")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{title}: P/C vs S2N_RED (P_DLA>{gp_conf}, NHI>{nhi_min_both})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return log10_snrs, p, c


def plot_matrix(data: np.ndarray, snr_bin_plot: np.ndarray,
                nhi_bin_plot: np.ndarray, title: str, out_png: str,
                kind: str, cmap: str = "hot"):
    """Molly cell 76/77 matrix heatmap. `data` is shape (n_snr, n_nhi)."""
    setup_mpl()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    img = ax.pcolor(snr_bin_plot, nhi_bin_plot, data.T,
                    cmap=cmap, vmin=0., vmax=1., rasterized=True)
    snr_mid = (snr_bin_plot[:-1] + snr_bin_plot[1:]) / 2
    nhi_mid = (nhi_bin_plot[:-1] + nhi_bin_plot[1:]) / 2
    for i in range(data.T.shape[0]):
        for j in range(data.T.shape[1]):
            v = data.T[i, j]
            if np.isnan(v):
                txt = "—"
            else:
                txt = f"{v:.2f}"
            col = "white" if (np.isnan(v) or v < 0.27) else "black"
            ax.text(snr_mid[j], nhi_mid[i], txt, ha="center", va="center",
                    color=col, fontsize=8)
    ax.set_xlabel("S/N (S2N_RED)")
    nhi_label = "predicted log NHI" if kind == "purity" else "true log NHI"
    ax.set_ylabel(nhi_label)
    ax.set_title(title)
    fig.colorbar(img, ax=ax, label=kind.capitalize())
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def _run_one_window(cat, truth, bal_tids, args, title, out_dir, lam_rf_min):
    """Run the full pipeline for one λ_rf window and write plots+tsv to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print()
    print(f"=== window: λ_rf ∈ [{lam_rf_min},{args.lam_rf_max}] → {out_dir} ===")

    cat_cut = make_lambda_z_BAL_cuts(
        cat, lam_rf_min, args.lam_rf_max,
        args.z_qso_min, args.z_qso_max,
        bal_tids=bal_tids, z_col_for_min="Z_DLA", use_truth_z=True,
    )
    truth_cut = make_lambda_z_BAL_cuts(
        truth, lam_rf_min, args.lam_rf_max,
        args.z_qso_min, args.z_qso_max,
        bal_tids=bal_tids, z_col_for_min="Z_DLA", use_truth_z=False,
    )
    print(f"[cuts] cat: {len(cat)} → {len(cat_cut)}")
    print(f"[cuts] truth: {len(truth)} → {len(truth_cut)}")

    tp = ~np.isnan(np.asarray(cat_cut["NHI_TRUE"], dtype=float))
    good_mask = (np.asarray(cat_cut["DLAFLAG"], dtype=int) == 0)
    if args.lyb_veto and "LYBETA_FLAG" in cat_cut.colnames:
        good_mask &= ~np.asarray(cat_cut["LYBETA_FLAG"], dtype=bool)
    print(f"[postcuts] TP={int(tp.sum())}, good_mask={int(good_mask.sum())}/{len(cat_cut)}")

    log_pdla, pur_curve, cmp_curve = plot_pc_vs_pdla(
        cat_cut, tp, truth_cut, good_mask,
        out_png=str(out_dir / "molly_pc_vs_pdla.png"),
        title=title, snr_min=args.snr_min, nhi_min_both=args.nhi_min,
    )
    log10_snrs, pur_snr, cmp_snr = plot_pc_vs_snr(
        cat_cut, tp, truth_cut, good_mask,
        out_png=str(out_dir / "molly_pc_vs_snr.png"),
        title=title, gp_conf=args.gp_conf, nhi_min_both=args.nhi_min,
    )

    snr_bin = np.array([0., 1., 2., 3., 4., 5., 6., 7., np.inf])
    nhi_bin = np.array([20.3, 20.5, 21., 21.5, 22., np.inf])
    snr_bin_plot = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.])
    nhi_bin_plot = np.array([20.3, 20.5, 21., 21.5, 22., 22.5])

    pur_mat = np.full((len(snr_bin) - 1, len(nhi_bin) - 1), np.nan)
    cmp_mat = np.full((len(snr_bin) - 1, len(nhi_bin) - 1), np.nan)
    for i in range(len(snr_bin) - 1):
        for j in range(len(nhi_bin) - 1):
            _, _, pur_mat[i, j] = purity_snr_nhi_bins(
                cat_cut, tp, snr_bin[i], snr_bin[i + 1],
                nhi_bin[j], nhi_bin[j + 1], args.gp_conf, good_mask,
                nhi_key="NHI", goodness_key="P_DLA",
            )
            _, _, cmp_mat[i, j] = completeness_snr_nhi_bins(
                cat_cut, tp, snr_bin[i], snr_bin[i + 1],
                nhi_bin[j], nhi_bin[j + 1], args.nhi_min, args.gp_conf,
                truth_cut, good_mask,
                nhi_key="NHI", goodness_key="P_DLA",
            )
    plot_matrix(pur_mat, snr_bin_plot, nhi_bin_plot,
                title=f"{title} — Molly purity matrix",
                out_png=str(out_dir / "molly_purity_matrix.png"), kind="purity")
    plot_matrix(cmp_mat, snr_bin_plot, nhi_bin_plot,
                title=f"{title} — Molly completeness matrix",
                out_png=str(out_dir / "molly_completeness_matrix.png"), kind="completeness")

    _, _, pur_head = purity_min(cat_cut, tp, args.snr_min, args.nhi_min,
                                 args.gp_conf, good_mask,
                                 nhi_key="NHI", goodness_key="P_DLA")
    _, _, cmp_head = completeness_min(cat_cut, tp, args.snr_min, args.nhi_min,
                                       args.nhi_min, args.gp_conf, truth_cut,
                                       good_mask, nhi_key="NHI", goodness_key="P_DLA")
    with (out_dir / "molly_summary.tsv").open("w") as f:
        f.write("metric\tvalue\n")
        f.write(f"title\t{title}\n")
        f.write(f"lam_rf_min\t{lam_rf_min}\n")
        f.write(f"lam_rf_max\t{args.lam_rf_max}\n")
        f.write(f"n_cat_post_cuts\t{len(cat_cut)}\n")
        f.write(f"n_truth_post_cuts\t{len(truth_cut)}\n")
        f.write(f"snr_min\t{args.snr_min}\n")
        f.write(f"nhi_min\t{args.nhi_min}\n")
        f.write(f"gp_conf\t{args.gp_conf}\n")
        f.write(f"purity_headline\t{pur_head:.4f}\n")
        f.write(f"completeness_headline\t{cmp_head:.4f}\n")
        f.write("\n# log_pdla\tpurity\tcompleteness\n")
        for lp, pp, cc in zip(log_pdla, pur_curve, cmp_curve):
            f.write(f"{lp}\t{pp:.4f}\t{cc:.4f}\n")
        f.write("\n# log10_snr_lo\tlog10_snr_hi\tpurity\tcompleteness\n")
        for i in range(len(log10_snrs) - 1):
            f.write(f"{log10_snrs[i]:.3f}\t{log10_snrs[i+1]:.3f}\t"
                    f"{pur_snr[i]:.4f}\t{cmp_snr[i]:.4f}\n")
    return pur_head, cmp_head, len(cat_cut), len(truth_cut)


def main():
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    title = args.title or os.path.basename(args.catalog_dir.rstrip("/"))

    # ---- Load detection cat -------------------------------------------------
    cat = load_catalog_dir(args.catalog_dir)
    if "SNR_REDSIDE" not in cat.colnames:
        raise SystemExit("dlacat lacks SNR_REDSIDE column — older catalog?")
    cat["S2N_RED"] = np.asarray(cat["SNR_REDSIDE"], dtype=float)

    # ---- Per-QSO SNR lookup + truth load -----------------------------------
    qso_lookup = build_per_qso_snr(args.catalog_dir)
    zcat_default = (os.path.join(args.mockdir, "zcat.fits")
                    if args.mockdir else args.zcat)
    truth = load_truth_molly(args.truth, args.truth_nhi_min, qso_lookup, zcat_default)

    # ---- BAL exclusion (cat + truth) ---------------------------------------
    bal_tids: set[int] | None = None
    if args.no_bal:
        if not args.bal_cat:
            raise SystemExit("--no-bal requires --bal-cat path")
        bal = fitsio.read(args.bal_cat, ext=1, columns=["TARGETID", "BI_CIV"])
        bal_tids = set(int(r["TARGETID"]) for r in bal if r["BI_CIV"] > 0)
        print(f"[bal] {len(bal_tids)} BAL TIDs")

    # ---- Truth-match BEFORE cuts (so cat has Z_TRUE/NHI_TRUE) --------------
    cat_is_TP, cat_NHI_TR, cat_Z_TR, truth_matched = match_truth_to_cat(
        cat, truth, args.dz_rel
    )
    cat["NHI_TRUE"] = cat_NHI_TR
    cat["Z_TRUE"] = cat_Z_TR
    print(f"[match] {int(cat_is_TP.sum())}/{len(cat)} MAP rows are TP "
          f"({int(truth_matched.sum())}/{len(truth)} truth matched at any cut)")

    # ---- Optional Lyβ veto postprocess -------------------------------------
    if args.lyb_veto:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from gpy_dla_detection.postprocess.lyb_veto import flag_lybeta
        cat = flag_lybeta(cat, dz_match=args.lyb_veto_dz,
                          targetid_col="TARGETID", z_col="Z_DLA", nhi_col="NHI")
        n_flagged = int(np.asarray(cat["LYBETA_FLAG"], dtype=bool).sum())
        print(f"[lyb_veto] flagged {n_flagged}/{len(cat)} rows as Lyβ misIDs "
              f"(dz_match={args.lyb_veto_dz})")

    # ---- Run for one or both λ_rf windows ----------------------------------
    if args.lam_rf_min is not None:
        windows = [(args.lam_rf_min, args.out)]
    else:
        windows = [
            (1025., os.path.join(args.out, "lya_only")),
            (911., os.path.join(args.out, "lya_lyb")),
        ]

    results = {}
    for lam_lo, out_dir in windows:
        pur, cmp_, ncat, ntr = _run_one_window(cat, truth, bal_tids, args, title,
                                                out_dir, lam_lo)
        results[lam_lo] = (pur, cmp_, ncat, ntr)

    # ---- Combined headline tsv ---------------------------------------------
    summary = Path(args.out) / "molly_summary_combined.tsv"
    with summary.open("w") as f:
        f.write("window\tlam_rf_min\tlam_rf_max\tn_cat\tn_truth\tpurity\tcompleteness\tpasses_85_85\n")
        for lam_lo, (pur, cmp_, ncat, ntr) in results.items():
            name = "lya_only" if lam_lo == 1025. else ("lya_lyb" if lam_lo == 911. else f"lam_{lam_lo:.0f}")
            passes = (pur >= 0.85 and cmp_ >= 0.85)
            f.write(f"{name}\t{lam_lo}\t{args.lam_rf_max}\t{ncat}\t{ntr}\t"
                    f"{pur:.4f}\t{cmp_:.4f}\t{'YES' if passes else 'NO'}\n")

    print()
    print(f"=== HEADLINE: {title} ===")
    print(f"  cuts: snr>{args.snr_min}, NHI>{args.nhi_min}, P_DLA>{args.gp_conf}, "
          f"DLAFLAG==0, z_qso∈[{args.z_qso_min},{args.z_qso_max}], BAL excluded"
          f"{'  + lyb_veto' if args.lyb_veto else ''}")
    for lam_lo, (pur, cmp_, ncat, ntr) in results.items():
        name = "lya_only" if lam_lo == 1025. else ("lya_lyb" if lam_lo == 911. else f"lam_{lam_lo:.0f}")
        flag = "YES" if (pur >= 0.85 and cmp_ >= 0.85) else "NO"
        print(f"  [{name:8s}] λ_rf∈[{lam_lo},{args.lam_rf_max}] "
              f"P={pur:.4f}  C={cmp_:.4f}  >=85/85?{flag}  (cat={ncat}, truth={ntr})")
    print(f"  Figures + tsv in: {args.out}")


if __name__ == "__main__":
    main()
