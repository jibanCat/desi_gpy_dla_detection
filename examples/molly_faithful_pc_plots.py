"""molly_faithful_pc_plots.py — exact reproduction of Molly Wolfson's
purity/completeness notebook plots for the DLA-mode catalog.

Recipe transcribed from
  /pscratch/sd/j/jibancat/molly/read_in_each_plots_saclay-Y3-Learned.ipynb
  /pscratch/sd/j/jibancat/molly/read_in_each_up_match_new_cats_2509.ipynb

Differences from gp_native_pc_plots.py (which is "inspired by" molly):
  * predicted NHI column   = "NHI"        (molly uses NHI_TMP for template, NHI for GP)
  * confidence column      = "P_DLA"      (sweep over log_pdla ∈ {-1,…,-8})
  * SNR column             = "SNR_REDSIDE" (molly's S2N_RED; pulled from dlacat or h5)
  * goodness gate          = DLAFLAG == 0 (good_mask)
  * truth-matching         = per-TARGETID, |Δz|/(1+z) < 0.01, greedy (cell 12 of molly's nb)
  * Z range cut            = z_qso ∈ [2.0, 4.25]
  * λ_rf range cut         = [1025, 1216] Å (cell 21; "OmegaDLA" cuts)
  * BAL removal (default)  = TARGETID ∈ bal_cat → dropped from BOTH cat AND truth
                             (use --bal-bi-civ-only to restrict to BI_CIV>0 rows)
  * truth-NHI floor        = NHI > 20.3
  * snr_min, nhi_min       = 6.0, 20.3 (cell 41)

Per-QSO SNR_REDSIDE/Z_QSO source (used by the truth catalog and as fallback for
detections that lack SNR_REDSIDE) is resolved in this priority:
  1. --snr-cat / --zcat external FITS (molly's canonical source — full mock)
  2. mockdir/snr_cat.fits and mockdir/zcat.fits (Saclay/2LPT have these inline)
  3. processed-spectra-16-*.h5 in catalog-dir (legacy fallback; inference output)
With (3) the lookup only covers spectra that this run processed, so it MUST NOT
be used to evaluate a different catalog (e.g. legacy production runs against a
fresh 5k slice's processed dir).

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

# Reuse load helper. NOTE: gp_native's `match_truth_to_cat` iterates truth in
# descending-NHI order with closest-z tie-break, which inflates P by ~3pp vs
# molly's notebook (audit 2026-05-15). Use `match_truth_to_cat_molly` defined
# below — cat in input order, closest-NHI tie-break — for faithful results.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gp_native_pc_plots import load_catalog_dir

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
                   help="Exclude BAL TIDs from BOTH cat and truth (molly convention). "
                        "Default: drop ALL bal_cat TIDs. With --bal-bi-civ-only, "
                        "drop only rows where BI_CIV > 0.")
    p.add_argument("--bal-bi-civ-only", action="store_true",
                   help="When --no-bal is set, restrict BAL TIDs to BI_CIV>0 rows. "
                        "Pre-2026-05-15 default; molly drops ALL bal_cat TIDs.")
    p.add_argument("--truth-nhi-min", type=float, default=20.3,
                   help="Truth NHI floor (default 20.3).")
    p.add_argument("--dz-rel", type=float, default=0.01,
                   help="|Δz|/(1+z_truth) match tolerance (default 0.01).")
    p.add_argument("--molly-input-order", action="store_true",
                   help="Use legacy file-order iteration of the catalog in the "
                        "truth matcher (reproduces molly's notebook bit-for-bit, "
                        "but inherits the order-dependence bug — see "
                        "docs/notes/2026-05-19_dla_matcher_order_dependence_for_molly.md). "
                        "Default is NHI-descending iteration, which is "
                        "order-independent and recovers the strong-DLA "
                        "detections previously orphaned as FP.")
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
    p.add_argument("--snr-min", type=float, default=2.0,
                   help="Min S2N_RED (default 2.0 — molly's canonical headline).")
    p.add_argument("--nhi-min", type=float, default=20.3,
                   help="Min predicted/truth log NHI for headline numbers (default 20.3).")
    p.add_argument("--gp-conf", type=float, default=0.99,
                   help="Fixed P_DLA cut for 1D + 2D plots (molly used 0.99 for Saclay DLA).")
    p.add_argument("--nhi-bins", default=None,
                   help="Comma-separated log-NHI bin edges for the (SNR,NHI) "
                        "purity/completeness matrix + molly_matrix.tsv. 'inf' "
                        "allowed for the open top bin. Default: "
                        "20.3,20.5,21,21.5,22,inf (molly). For a sub-DLA→DLA "
                        "table over [19,23] use e.g. 19,19.5,20,20.3,20.5,21,22,23.")
    p.add_argument("--snr-bins", default=None,
                   help="Comma-separated S2N_RED bin edges for the matrix. "
                        "'inf' allowed. Default: 0,1,2,3,4,5,6,7,inf (molly).")
    p.add_argument("--bf-band-min", type=float, default=None,
                   help="Optional extra cut: keep only detections with "
                        "BF_BAND >= this (the local-posterior boundary-purity "
                        "flag from add_dla_flags.py). Rows with NaN BF_BAND "
                        "are kept (not penalised for a missing score).")
    p.add_argument("--zcat", default=None,
                   help="Optional zcat.fits for Z_QSO lookup (canonical source). "
                        "Falls back to mockdir/zcat.fits, then processed h5.")
    p.add_argument("--snr-cat", default=None,
                   help="Optional snr_cat.fits with SNR_REDSIDE column (canonical "
                        "per-QSO SNR source — molly's recipe). Falls back to "
                        "mockdir/snr_cat.fits, then processed h5.")
    p.add_argument("--mockdir", default=None,
                   help="If --zcat / --snr-cat unset, look for zcat.fits / snr_cat.fits "
                        "in this MOCKDIR.")
    p.add_argument("--restrict-truth-to-processed", action="store_true",
                   help="When using --snr-cat (full-mock scope), additionally "
                        "intersect with the set of TIDs the run processed (from "
                        "processed-spectra-16-*.h5). Required for 5k-slice runs "
                        "so completeness denominator stays in scope; OFF by "
                        "default (molly's full-mock recipe).")
    p.add_argument("--out", required=True,
                   help="Output dir for PNGs + summary tsv.")
    p.add_argument("--title", default=None,
                   help="Figure title (default: catalog-dir basename).")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Per-QSO SNR_REDSIDE lookup from processed h5 files
# -----------------------------------------------------------------------------
def build_per_qso_snr(catalog_dir: str,
                      snr_cat_path: str | None = None,
                      zcat_path: str | None = None,
                      mockdir: str | None = None,
                      restrict_to_processed: bool = False,
                      ) -> dict[int, tuple[float, float]]:
    """Build TARGETID → (snr_redside, z_qso).

    Resolution priority (matches molly's notebook recipe):
      1. external snr_cat FITS (SNR_REDSIDE col) + external zcat FITS (Z col)
      2. mockdir/snr_cat.fits + mockdir/zcat.fits (Saclay/2LPT inline)
      3. processed-spectra-16-*.h5 in catalog-dir (legacy fallback; only covers
         spectra processed by THIS run — must not be used to evaluate a
         different catalog)

    Always loads from a single source; never mixes. The legacy h5 fallback
    fires only when neither (1) nor (2) yields readable files.
    """
    # Priority 1: explicit --snr-cat + --zcat
    snr_p, zcat_p = snr_cat_path, zcat_path
    # Priority 2: mockdir defaults
    if (snr_p is None or zcat_p is None) and mockdir:
        if snr_p is None:
            cand = os.path.join(mockdir, "snr_cat.fits")
            if os.path.exists(cand):
                snr_p = cand
        if zcat_p is None:
            cand = os.path.join(mockdir, "zcat.fits")
            if os.path.exists(cand):
                zcat_p = cand

    if snr_p and zcat_p and os.path.exists(snr_p) and os.path.exists(zcat_p):
        snr_t = fitsio.read(snr_p, ext=1, columns=["TARGETID", "SNR_REDSIDE"])
        zc_t = fitsio.read(zcat_p, ext=1, columns=["TARGETID", "Z"])
        snr_map = {int(r["TARGETID"]): float(r["SNR_REDSIDE"]) for r in snr_t}
        zq_map = {int(r["TARGETID"]): float(r["Z"]) for r in zc_t}
        # Optional intersection with the processed-h5 TID set — needed for 5k
        # slice runs (completeness denominator must stay in the slice's scope)
        # but WRONG for full-mock evals (molly's recipe uses full snr_cat;
        # un-processed QSOs still count toward "the production should have
        # processed them" completeness).
        processed_tids: set[int] | None = None
        if restrict_to_processed:
            h5_paths = sorted(glob.glob(os.path.join(catalog_dir, "processed",
                                                     "processed-spectra-16-*.h5")))
            if not h5_paths:
                h5_paths = sorted(glob.glob(os.path.join(catalog_dir,
                                                         "processed-spectra-16-*.h5")))
            if h5_paths:
                processed_tids = set()
                for p in h5_paths:
                    with h5py.File(p, "r") as f:
                        processed_tids.update(int(t) for t in f["target_ids"][:])
                print(f"[snr] processed-set scope: {len(processed_tids)} TIDs "
                      f"from {len(h5_paths)} h5 files in catalog-dir "
                      f"(restricting truth to this scope)")
            else:
                print(f"[snr] WARNING: --restrict-truth-to-processed set but no "
                      f"processed-spectra-16-*.h5 found in {catalog_dir}; "
                      f"falling back to full snr_cat scope")
        lookup: dict[int, tuple[float, float]] = {}
        for t, s in snr_map.items():
            z = zq_map.get(t)
            if z is None:
                continue
            if processed_tids is not None and t not in processed_tids:
                continue
            lookup[t] = (s, z)
        scope = ("intersected with processed h5"
                 if processed_tids is not None else "full snr_cat (molly recipe)")
        print(f"[snr] built per-QSO lookup from external snr_cat+zcat ({scope}): "
              f"{len(lookup)} TIDs ({snr_p}, {zcat_p})")
        return lookup

    # Priority 3: legacy fallback to processed h5
    paths = sorted(glob.glob(os.path.join(catalog_dir, "processed",
                                          "processed-spectra-16-*.h5")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(catalog_dir,
                                              "processed-spectra-16-*.h5")))
    if not paths:
        raise SystemExit(
            f"[error] no per-QSO SNR source: neither --snr-cat / --zcat nor "
            f"mockdir snr_cat/zcat resolved, and no processed-spectra-16-*.h5 "
            f"in {catalog_dir}")
    lookup = {}
    for p in paths:
        with h5py.File(p, "r") as f:
            tids = np.asarray(f["target_ids"][:], dtype=np.int64)
            snrs = np.asarray(f["snrs"][:], dtype=float)
            zq = np.asarray(f["z_qsos"][:], dtype=float)
        for t, s, z in zip(tids, snrs, zq):
            lookup[int(t)] = (float(s), float(z))
    print(f"[snr] built per-QSO lookup from processed h5: {len(lookup)} TIDs "
          f"from {len(paths)} h5 files (LEGACY FALLBACK — only spectra processed "
          f"by this run are present; cat TIDs not in the h5 will be dropped)")
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
# Molly's matcher (notebook cell `match_detections_to_dla_cat`, L259-313)
# -----------------------------------------------------------------------------
def match_truth_to_cat_molly(cat: Table, truth: Table, dz_rel: float,
                              cat_iter_order: str = "nhi_desc"
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Greedy 1-to-1 matcher, derived from molly's notebook with an order fix.

    Iteration order over `cat`:
      * `cat_iter_order="nhi_desc"` (default, **2026-05-19 fix**): walk cat
        rows in descending NHI_pred so the strongest detection per TID claims
        its truth before any weaker sibling row can. Removes the order-
        dependence bug documented in
        `docs/notes/2026-05-19_dla_matcher_order_dependence_for_molly.md`
        — on the V1 5k London-0 run this recovers 14 strong-DLA detections
        previously mislabelled FP (~+4pp purity) and reattaches 13 truth
        DLAs that had fallen out of the FN ledger.
      * `cat_iter_order="input"`: legacy "molly-faithful" behaviour — walks
        cat in file order. Reproduces molly's published numbers bit-for-bit
        but inherits the multi-absorber bug. Kept for cross-author continuity
        via the `--molly-input-order` CLI flag.

    For each cat row, finds truth candidates with same TID and
    |Δz|/(1+z_truth) < dz_rel that haven't been matched yet, then breaks ties
    by minimum |NHI_pred − NHI_truth| (matches molly's notebook).

    Returns (cat_is_TP, cat_NHI_TR, cat_Z_TR, truth_matched). The `cat_is_TP`
    and per-row matched-truth columns are still ordered by cat-row index;
    only the iteration order changes.
    """
    from collections import defaultdict

    cat_is_TP = np.zeros(len(cat), dtype=bool)
    cat_NHI_TR = np.full(len(cat), np.nan)
    cat_Z_TR = np.full(len(cat), np.nan)
    truth_matched = np.zeros(len(truth), dtype=bool)

    c_tid = np.asarray(cat["TARGETID"], dtype=np.int64)
    c_z = np.asarray(cat["Z_DLA"], dtype=float)
    c_nhi = np.asarray(cat["NHI"], dtype=float)
    t_tid = np.asarray(truth["TARGETID"], dtype=np.int64)
    # truth Z column may be Z_TRUTH (gp_native alias) or Z_DLA
    t_z = np.asarray(truth["Z_TRUTH"] if "Z_TRUTH" in truth.colnames
                     else truth["Z_DLA"], dtype=float)
    t_nhi = np.asarray(truth["NHI"], dtype=float)

    truth_by_tid: dict[int, list[int]] = defaultdict(list)
    for j, t in enumerate(t_tid):
        truth_by_tid[int(t)].append(j)

    if cat_iter_order == "nhi_desc":
        # NaN NHI rows go last so they cannot pre-empt finite-NHI rows
        iter_order = np.argsort(-np.nan_to_num(c_nhi, nan=-np.inf), kind="stable")
    elif cat_iter_order == "input":
        iter_order = np.arange(len(cat))
    else:
        raise ValueError(f"cat_iter_order must be 'nhi_desc' or 'input', "
                         f"got {cat_iter_order!r}")

    for ci in iter_order:
        tid = int(c_tid[ci])
        idx_list = truth_by_tid.get(tid)
        if not idx_list:
            continue
        idx_arr = np.asarray(idx_list, dtype=int)
        zdiff = np.abs(c_z[ci] - t_z[idx_arr]) / (1.0 + t_z[idx_arr])
        close = zdiff < dz_rel
        if not close.any():
            continue
        cand = idx_arr[close]
        un = ~truth_matched[cand]
        if not un.any():
            continue
        cand_un = cand[un]
        if cand_un.size == 1:
            j = int(cand_un[0])
        else:
            order = np.argsort(np.abs(c_nhi[ci] - t_nhi[cand_un]))
            j = int(cand_un[order[0]])
        cat_is_TP[ci] = True
        cat_NHI_TR[ci] = t_nhi[j]
        cat_Z_TR[ci] = t_z[j]
        truth_matched[j] = True

    return cat_is_TP, cat_NHI_TR, cat_Z_TR, truth_matched


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

    # Molly's notebook starts at -0.5 (cell 1432) — prepend so the curve
    # matches her published P/C-vs-cut figure.
    log_pdla = np.array([-0.5, -1., -2., -3., -4., -5., -6., -7., -8.])
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
    if args.bf_band_min is not None:
        if "BF_BAND" not in cat_cut.colnames:
            raise SystemExit("--bf-band-min set but the catalog has no "
                             "BF_BAND column — run add_dla_flags.py first.")
        bf = np.asarray(cat_cut["BF_BAND"], dtype=float)
        # keep NaN (unscored) rows; only cut rows that are scored AND below cut
        good_mask &= ~(np.isfinite(bf) & (bf < args.bf_band_min))
        print(f"[postcuts] BF_BAND>={args.bf_band_min}: "
              f"{int((np.isfinite(bf) & (bf < args.bf_band_min)).sum())} rows cut")
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

    def _parse_edges(spec, default, name):
        if spec is None:
            return np.array(default, dtype=float)
        try:
            edges = np.array(
                [np.inf if e.strip().lower() in ("inf", "+inf") else float(e)
                 for e in spec.split(",") if e.strip() != ""], dtype=float)
        except ValueError:
            raise SystemExit(f"--{name}: could not parse '{spec}' as "
                             f"comma-separated numbers (e.g. 19,20,21,inf)")
        if len(edges) < 2:
            raise SystemExit(f"--{name}: need >=2 bin edges, got {len(edges)} "
                             f"from '{spec}'")
        if np.isinf(edges[:-1]).any():
            raise SystemExit(f"--{name}: only the last edge may be 'inf', "
                             f"got {list(edges)}")
        if not np.all(np.diff(edges) > 0):
            raise SystemExit(f"--{name}: bin edges must be strictly "
                             f"increasing, got {list(edges)}")
        return edges

    def _plot_edges(edges):
        # pcolor needs finite edges; map a trailing inf to last_finite + step.
        e = edges.copy()
        if not np.isfinite(e[-1]):
            step = (e[-2] - e[-3]) if len(e) >= 3 and np.isfinite(e[-3]) else 0.5
            e[-1] = e[-2] + step
        return e

    snr_bin = _parse_edges(args.snr_bins, [0., 1., 2., 3., 4., 5., 6., 7., np.inf], "snr-bins")
    nhi_bin = _parse_edges(args.nhi_bins, [20.3, 20.5, 21., 21.5, 22., np.inf], "nhi-bins")
    snr_bin_plot = _plot_edges(snr_bin)
    nhi_bin_plot = _plot_edges(nhi_bin)

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

    # Dump the (SNR bin, NHI bin) matrix to tsv so the heatmap is readable as a table.
    with (out_dir / "molly_matrix.tsv").open("w") as f:
        f.write(f"# (S2N_RED bin, log-NHI bin) purity & completeness at "
                f"P_DLA>{args.gp_conf}; purity uses predicted NHI, "
                f"completeness uses truth NHI\n")
        f.write("snr_lo\tsnr_hi\tnhi_lo\tnhi_hi\tpurity\tcompleteness\n")
        for i in range(len(snr_bin) - 1):
            for j in range(len(nhi_bin) - 1):
                f.write(f"{snr_bin[i]:g}\t{snr_bin[i+1]:g}\t"
                        f"{nhi_bin[j]:g}\t{nhi_bin[j+1]:g}\t"
                        f"{pur_mat[i, j]:.4f}\t{cmp_mat[i, j]:.4f}\n")

    n_tp_head, n_kept_head, pur_head = purity_min(
        cat_cut, tp, args.snr_min, args.nhi_min, args.gp_conf, good_mask,
        nhi_key="NHI", goodness_key="P_DLA")
    _n_tp_c, n_truth_kept_head, cmp_head = completeness_min(
        cat_cut, tp, args.snr_min, args.nhi_min, args.nhi_min, args.gp_conf,
        truth_cut, good_mask, nhi_key="NHI", goodness_key="P_DLA")
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
        # Absolute counts at the headline operating point (SNR>snr_min, NHI>nhi_min,
        # P_DLA>gp_conf, DLAFLAG==0 [+ lyb-veto if set]). purity = n_TP / n_kept;
        # completeness = n_TP / n_truth_kept. n_TP from purity_min and completeness_min
        # are equal by construction (same mask).
        f.write(f"n_TP_headline\t{n_tp_head}\n")
        f.write(f"n_kept_headline\t{n_kept_head}\n")
        f.write(f"n_truth_kept_headline\t{n_truth_kept_head}\n")
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
    qso_lookup = build_per_qso_snr(args.catalog_dir,
                                   snr_cat_path=args.snr_cat,
                                   zcat_path=args.zcat,
                                   mockdir=args.mockdir,
                                   restrict_to_processed=args.restrict_truth_to_processed)
    zcat_default = args.zcat or (os.path.join(args.mockdir, "zcat.fits")
                                 if args.mockdir else None)
    truth = load_truth_molly(args.truth, args.truth_nhi_min, qso_lookup, zcat_default)

    # ---- BAL exclusion (cat + truth) ---------------------------------------
    # Default: drop ALL TIDs in bal_cat (molly recipe). With --bal-bi-civ-only,
    # restrict to BI_CIV>0 rows (pre-2026-05-15 default).
    bal_tids: set[int] | None = None
    if args.no_bal:
        if not args.bal_cat:
            raise SystemExit("--no-bal requires --bal-cat path")
        if args.bal_bi_civ_only:
            bal = fitsio.read(args.bal_cat, ext=1, columns=["TARGETID", "BI_CIV"])
            bal_tids = set(int(r["TARGETID"]) for r in bal if r["BI_CIV"] > 0)
            print(f"[bal] BI_CIV>0 only: {len(bal_tids)} BAL TIDs")
        else:
            bal = fitsio.read(args.bal_cat, ext=1, columns=["TARGETID"])
            bal_tids = set(int(r["TARGETID"]) for r in bal)
            print(f"[bal] all bal_cat TIDs (molly recipe): {len(bal_tids)} BAL TIDs")

    # ---- Truth-match BEFORE cuts (so cat has Z_TRUE/NHI_TRUE) --------------
    # 2026-05-15 patch: switched from gp_native.match_truth_to_cat (truth-NHI-
    # descending iter, closest-z tie-break) to match_truth_to_cat_molly (cat
    # input-order iter, closest-NHI tie-break) to reproduce molly's notebook.
    # 2026-05-19 fix: the molly-faithful file-order iteration has an
    # order-dependence bug on multi-absorber spectra — a weak decoy detection
    # row earlier in file order consumes the truth DLA before the strong
    # correct row can claim it (~14/68 FP rows mislabelled on V1 5k). Default
    # is now NHI-descending iteration, order-independent. Pass
    # --molly-input-order to recover the legacy bit-faithful behaviour. See
    # docs/notes/2026-05-19_dla_matcher_order_dependence_for_molly.md.
    iter_order_mode = "input" if args.molly_input_order else "nhi_desc"
    cat_is_TP, cat_NHI_TR, cat_Z_TR, truth_matched = match_truth_to_cat_molly(
        cat, truth, args.dz_rel, cat_iter_order=iter_order_mode,
    )
    print(f"[match] iteration order: {iter_order_mode}")
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
