"""build_loa0_fp_product.py — measure the loa-0 forest false-positive (FP) λ_FP
background product for the catalog-HBI estimator (spec §4 PRIMARY Loa0FP).

WHY (the bug this replaces)
---------------------------
The interim `PurityMixtureFP` uses `lam_fp_per_obj = (1 − ρ_i)` where ρ_i is the
molly purity at the object's (N̂_i, SNR_i). That is a *dimensionless probability*
that is (a) in-sample / circular (the same catalog defines both ρ and the data
term) and (b) double-subtracts N-migration (the forward kernel A_{i,b} already
moves up-migrated low-N systems into the bin). At ≥20.3 it subtracts (1−ρ)≈0.18
of a fictitious background per detection while a direct forest-FP measurement
says ≈0 → drives the WALL-1 R0(≥20.3)≈0.52 deficit.

WHAT THIS BUILDS (the principled replacement, z-flat pilot)
-----------------------------------------------------------
loa-0 is the HCD-FREE byte-identical twin of the production loa-124 mock (same
1,213,217 skewers, same z-range, same schema, NO injected HCD/DLA, NO BAL). Run
through the *production* GP finder, EVERY loa-0 detection is a forest false
positive by construction. We bin those FP detections into:

  * the molly (x̂, SNR) cells              → a per-cell base RATE density b_FP
  * the fine (logN, z) HBI grid            → the μ_FP grid normalizer

and save the per-cell FP COUNTS (n̂_FP), the loa-0 searched-sightline count
N_sl_loa0, the production searched-sightline count N_prod, the variance scale
ℓ_eff, and the band-averaged host-occlusion fraction η (DLA / sub-DLA / LLS).

Definitions (spec §4, verified math)
------------------------------------
  base RATE density per molly cell:
      b_FP(x̂, SNR) = n̂_FP_loa0(x̂,SNR) / (Δx̂ · N_sl_loa0)        [per unit x̂, per sightline]
  host-aware per-object cell rate (data term):
      λ_FP_per_obj[i] = b_FP(cell_i) · (1 − η_{band_i})
  μ_FP (rate term — computed as the INTEGRAL, not Σ_i λ_FP):
      μ_FP = (N_prod / N_sl_loa0) · N_FP_loa0_total · (1 − η̄)
      equivalently per fine (N,z) cell:
      mu_fp_grid[b,k] = n̂_FP_loa0[b,k] · (N_prod / N_sl_loa0) · (1 − η_band)
  variance (per cell): λ_FP ~ Gamma(n_FP + ½, ℓ_eff),
      ℓ_eff = N_sl_loa0 · (N_sl_loa0 / N_prod)   (production-extrapolation variance)

Host-occlusion η (PILOT approximation — documented)
---------------------------------------------------
η_s = fraction of a production sightline's searchable forest occluded by a true
HCD (a forest FP can only be "found" in un-occluded forest). For the DLA tier
(logN ≳ 20) b_FP ≈ 0, so η is MOOT there and we force η_DLA = 0 (applying η flat
across the DLA tier re-creates a known 1.73× over-subtraction — explicitly NOT
done). For the sub-DLA / LLS tiers we compute η band-averaged from the loa-124
hcd_truth occlusion (the field-average forest fraction blocked by ≥ the band's
HCDs). Band-averaged (not per-sightline) is acceptable for this z-flat pilot;
the per-sightline λ_FP,prod(·|s)=b_FP·(1−η_s) host-awareness is a follow-up.

OUTPUT
------
An npz at <loa0_out>/loa0_fp_product.npz consumed by cddf_catalog_hbi.Loa0FP.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import fitsio
import h5py
from astropy.table import Table, vstack

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.cddf_catalog_hbi import (  # noqa: E402
    HBIConfig, build_fine_grid, build_pathlength, _build_qso_lookup,
    load_molly_matrix, _cell_index, _bin_index_logN, LYA_REST,
)

# loa-0 FP run defaults (verified present 2026-06-16)
DEF_LOA0_OUT = "/scratch/cavestru_root/cavestru0/mfho/gl_loa0_fp_v1_20260615/outputs/"
DEF_PROD_CAT = ("/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/"
                "combined_catalog/")
DEF_MOLLY = ("/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/"
             "figures_molly_nhi172/molly_matrix.tsv")
DEF_TRUTH124 = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/"
                "v2.8.5/mock-0/loa-124/hcd_truth_cat.fits")
DEF_PROD_MOCKDIR = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/"
                    "v2.8.5/mock-0/loa-124")
DEF_PROD_BAL = DEF_PROD_MOCKDIR + "/bal_cat.fits"

# band edges (logN) used for the band-averaged η application — matches the spec's
# DLA / sub-DLA / LLS tiers and the cddf reporting structure.
BAND_EDGES = {
    "lls":    (17.2, 19.0),
    "subdla": (19.0, 20.3),
    "dla":    (20.3, 99.0),
}
# η is forced 0 on the DLA tier (b_FP≈0 there; flat-η re-creates the 1.73x bug).
DLA_ETA = 0.0


def _band_of_logN(logN: float) -> str:
    if logN >= BAND_EDGES["dla"][0]:
        return "dla"
    if logN >= BAND_EDGES["subdla"][0]:
        return "subdla"
    return "lls"


def load_loa0_fp_catalog(loa0_out: str) -> Table:
    """Load + merge the loa-0 FP dlacat-*.fits (the SNR-tiled combine)."""
    files = sorted(glob.glob(os.path.join(loa0_out, "dlacat-*.fits")))
    if not files:
        raise SystemExit(f"no dlacat-*.fits in {loa0_out}")
    tbls = [Table(fitsio.read(f, ext=1)) for f in files]
    cat = vstack(tbls)
    print(f"[loa0] {len(cat)} raw FP detections from {len(files)} files")
    return cat


def count_searched_sightlines_loa0(loa0_out: str, snr_min: float) -> tuple[int, int]:
    """N_sl_loa0 = # searched loa-0 sightlines with SNR_REDSIDE > snr_min.

    loa-0 had only a handful of healpix processed, so the legitimate searched set
    is exactly the union of `target_ids` over the processed-spectra-16-*.h5 (these
    are every spectrum the GP actually searched, including non-detections). The FP
    `SNR_REDSIDE` column == the h5 `snrs` (redside SNR) so the cut is identical to
    the molly op-mask SNR cut. Returns (n_sl_total_searched, n_sl_snr_gt_min)."""
    h5s = sorted(glob.glob(os.path.join(loa0_out, "figures", "processed",
                                        "processed-spectra-16-*.h5")))
    if not h5s:
        h5s = sorted(glob.glob(os.path.join(loa0_out, "processed",
                                            "processed-spectra-16-*.h5")))
    if not h5s:
        raise SystemExit(f"no processed-spectra-16-*.h5 under {loa0_out}")
    tids = {}
    for p in h5s:
        with h5py.File(p, "r") as f:
            t = np.asarray(f["target_ids"][:], dtype=np.int64)
            s = np.asarray(f["snrs"][:], dtype=float)
            for ti, si in zip(t, s):
                # a TID can appear once per processed file; keep its SNR
                tids[int(ti)] = float(si)
    snr_all = np.array(list(tids.values()), dtype=float)
    n_total = len(snr_all)
    n_kept = int(np.sum(snr_all > snr_min))
    print(f"[loa0] searched sightlines: {n_total} total, {n_kept} with SNR>{snr_min} "
          f"(from {len(h5s)} processed healpix)")
    return n_total, n_kept


def compute_band_eta(truth124_path: str, prod_mockdir: str, snr_min: float,
                     cfg: HBIConfig) -> dict:
    """Band-averaged host-occlusion fraction η from loa-124 hcd truth.

    η_band ≈ (field-average forest path occluded by HCDs whose true logN reaches
    INTO that band's tier or above) / (total searchable forest path). A forest FP
    in a band can only be FOUND where the production forest is NOT already taken by
    a real HCD; occlusion grows toward low N (more sub-DLA/LLS hosts). We use the
    DLA-damping-wing equivalent-width-free proxy: per HCD, the occluded velocity
    extent is ~ the Lyα EW window; for a z-flat PILOT we adopt the simple
    fractional-pathlength estimate

        η_band = (Σ_{HCD with logN in this band or higher} Δz_occ) / (Σ_sl Δz_search)

    with Δz_occ a fixed per-HCD occlusion width (we use a conservative 0.003 in z,
    ≈ a strong-DLA damping-wing + redshift-error window — far wider than a
    sub-DLA's, so this is an UPPER bound for the low-N tiers; the resulting η is a
    few %). DLA tier η is FORCED to 0 (b_FP≈0; flat-η re-creates the 1.73x bug).

    PILOT APPROXIMATION (documented): band-averaged + fixed Δz_occ. The
    per-sightline λ_FP(·|s)=b_FP·(1−η_s) host-awareness and an N-dependent Δz_occ
    are a follow-up. η here is MATERIAL only for the sub-DLA/LLS tiers."""
    truth = fitsio.read(truth124_path, ext=1)
    t_nhi = np.asarray(truth["NHI"], dtype=float)
    # total searchable forest path over the production SNR>2 sightlines (in z)
    qso_lookup = _build_qso_lookup(cfg)
    # reuse build_pathlength's window geometry to get per-sl z-windows
    X_tot, n_sl_used, qso_zlo, qso_zhi, qso_snr, Xcalc = build_pathlength(
        cfg, qso_lookup=qso_lookup, return_per_sl=True)
    total_dz_search = float(np.sum(qso_zhi - qso_zlo))
    DZ_OCC = 0.003  # fixed per-HCD occlusion width in z (conservative upper bound)
    eta = {}
    for band, (lo, hi) in BAND_EDGES.items():
        if band == "dla":
            eta[band] = DLA_ETA
            continue
        # HCDs whose true logN reaches into this band OR above occlude this band
        n_hcd = int(np.sum(t_nhi >= lo - 1e-9))
        occ_dz = n_hcd * DZ_OCC
        eta[band] = float(np.clip(occ_dz / total_dz_search, 0.0, 0.5)) if total_dz_search > 0 else 0.0
    print(f"[eta]  band-averaged host occlusion: {eta} "
          f"(total_dz_search={total_dz_search:.1f}, n_sl_prod={n_sl_used})")
    return eta, n_sl_used


def build_product(loa0_out: str, prod_cat_dir: str, molly_tsv: str,
                  truth124_path: str, prod_mockdir: str, prod_bal: str,
                  snr_min: float = 2.0, p_dla_min: float = 0.99,
                  out_path: str = None,
                  lya_only_lam_rf_min: float = None) -> dict:
    cfg = HBIConfig(
        catalog_dir=prod_cat_dir, truth_path=truth124_path,
        bal_cat_path=prod_bal, molly_tsv=molly_tsv,
        out_dir=loa0_out, mockdir=prod_mockdir,
        snr_min=snr_min, p_dla_min=p_dla_min,
    )
    mm = load_molly_matrix(molly_tsv)
    logN_lo, logN_hi, N_b, dN_b = build_fine_grid(cfg)
    n_nbins = len(logN_lo)
    zbins = np.asarray(cfg.zbins, dtype=float)
    n_zbins = len(zbins) - 1

    # FIX 2 (forest window): the loa-0 FP run used the FULL production forest window
    # (MIN_LAMBDA=911.75 — includes the noisier Lyβ region 911-1025 Å), but the
    # calibrated WALL-1 config is Lyα-only (lam_rf_min=1025). The Lyβ region inflates
    # forest FPs at sub-DLA/LLS, so the FULL-window product OVER-estimates the sub-DLA
    # / LLS FP. When ``lya_only_lam_rf_min`` is set we RESTRICT the FP detections to
    # those whose forest position λ_rest = LYA_REST·(1+Z_DLA)/(1+Z_QSO) >= the cut,
    # matching the lya_only molly the calibrated reduce uses. The DLA tier is unchanged
    # (FP≈0 either way; all DLA-tier FPs are already Lyα-region).
    # The molly-cell edges below are the build-time matrix's (nhi172, 12 NHI bins);
    # Loa0FP stores + uses these edges self-contained for its per-object cell-rate
    # lookup (finer than a nhi195 matrix → better FP resolution), independent of the
    # downstream reduce's molly matrix. The μ_FP grid is on the shared 0.1-dex fine
    # (logN,z) grid, so it is matrix-agnostic.
    # ----- loa-0 FP catalog + molly op cut -----
    cat = load_loa0_fp_catalog(loa0_out)
    snr = np.asarray(cat["SNR_REDSIDE"], dtype=float)
    pdla = np.asarray(cat["P_DLA"], dtype=float)
    nhi = np.asarray(cat["NHI"], dtype=float)
    z_dla = np.asarray(cat["Z_DLA"], dtype=float)
    op = (snr > snr_min) & (pdla > p_dla_min)   # loa-0 is BAL-free; no-bal is a no-op
    n_full = int(op.sum())
    if lya_only_lam_rf_min is not None:
        z_qso = np.asarray(cat["Z_QSO"], dtype=float)
        lam_rest = LYA_REST * (1.0 + z_dla) / (1.0 + z_qso)
        lya = lam_rest >= float(lya_only_lam_rf_min)
        n_drop = int((op & ~lya).sum())
        op = op & lya
        print(f"[loa0] lya_only window λ_rest>={lya_only_lam_rf_min:g} Å: dropped "
              f"{n_drop} Lyβ-region FP detections ({n_full} -> {int(op.sum())})")
    n_fp_total = int(op.sum())
    print(f"[loa0] FP detections passing molly op (SNR>{snr_min} & P_DLA>{p_dla_min}"
          f"{'' if lya_only_lam_rf_min is None else ' & lya_only'}): "
          f"{n_fp_total} / {len(cat)}")
    nhi, snr_op, z_dla = nhi[op], snr[op], z_dla[op]

    # ----- bin into molly (x̂, SNR) cells -----
    i_snr, j_nhi = _cell_index(mm, nhi, snr_op)
    n_snr_cells = len(mm.snr_edges) - 1
    n_nhi_cells = len(mm.nhi_edges) - 1
    n_fp_molly = np.zeros((n_snr_cells, n_nhi_cells))
    np.add.at(n_fp_molly, (i_snr, j_nhi), 1.0)

    # molly cell x̂ widths (Δx̂) — last (inf) bin gets a finite width = its lo→ a
    # nominal +0.5 dex span so b_FP there is a rate density, not a divide-by-inf.
    nhi_edges = mm.nhi_edges.copy()
    if not np.isfinite(nhi_edges[-1]):
        nhi_edges[-1] = nhi_edges[-2] + 0.5
    dxhat_cell = np.diff(nhi_edges)   # (n_nhi_cells,)

    # ----- searched sightline counts -----
    _, n_sl_loa0 = count_searched_sightlines_loa0(loa0_out, snr_min)

    # ----- band-averaged η + N_prod -----
    eta, n_sl_prod = compute_band_eta(truth124_path, prod_mockdir, snr_min, cfg)

    # ----- base rate density per molly cell -----
    # b_FP(x̂,SNR) = n_fp_molly / (Δx̂ · N_sl_loa0)
    with np.errstate(divide="ignore", invalid="ignore"):
        b_fp_molly = n_fp_molly / (dxhat_cell[None, :] * n_sl_loa0)
    b_fp_molly = np.where(np.isfinite(b_fp_molly), b_fp_molly, 0.0)

    # ----- fine (logN, z) grid FP counts (for the μ_FP grid) -----
    nbin_idx = _bin_index_logN(nhi, logN_lo, logN_hi)
    zbin_idx = np.searchsorted(zbins, z_dla, side="right") - 1
    zbin_idx = np.where((zbin_idx < 0) | (zbin_idx >= n_zbins), -1, zbin_idx)
    n_fp_fine = np.zeros((n_nbins, n_zbins))
    valid = (nbin_idx >= 0) & (zbin_idx >= 0)
    np.add.at(n_fp_fine, (nbin_idx[valid], zbin_idx[valid]), 1.0)
    # FIX 4(c) NOTE — z-window-clipped FPs: the FINE-grid μ_FP integral
    # (Σ n_fp_fine·vol_scale·(1−η)) counts only FPs whose Z_DLA lands INSIDE the cfg
    # zbins, whereas the molly rate b_FP / N_FP_total counts ALL op-passing FPs.
    # ~70 FPs (full-window product) fall outside the fine z-range (Z_DLA below the
    # first / above the last zbin edge) and are therefore present in the molly rate
    # but ABSENT from the μ_FP fine-grid integral. This is a small (~2.6%) μ_FP
    # under-count, concentrated at the band z-edges; it does NOT affect the
    # per-OBJECT FP share (FIX 1), which reads the molly cell directly, only the
    # population μ_FP normalizer. Flagged for the Bayesian re-review.
    n_clipped_z = int(valid.size - valid.sum())
    print(f"[loa0] z-window-clipped FPs (in molly rate, absent from fine μ_FP grid): "
          f"{n_clipped_z} / {n_fp_total}")

    # per-fine-Nbin band label (for band-averaged η application)
    band_eta_per_nbin = np.array([eta[_band_of_logN(0.5 * (logN_lo[b] + logN_hi[b]))]
                                  for b in range(n_nbins)])

    # ----- variance scale ℓ_eff -----
    ell_eff = float(n_sl_loa0) * (float(n_sl_loa0) / float(n_sl_prod))

    # ----- FP counts per band (reporting) -----
    band_counts = {}
    for band, (lo, hi) in BAND_EDGES.items():
        m = (nhi >= lo - 1e-9) & (nhi < hi)
        band_counts[band] = int(m.sum())

    out = dict(
        # molly-cell grids
        n_fp_molly=n_fp_molly,            # (n_snr_cells, n_nhi_cells) raw FP counts
        b_fp_molly=b_fp_molly,            # (n_snr_cells, n_nhi_cells) rate density
        dxhat_cell=dxhat_cell,            # (n_nhi_cells,)
        snr_edges=mm.snr_edges,
        nhi_edges=mm.nhi_edges,
        # fine (logN, z) grids
        n_fp_fine=n_fp_fine,              # (n_nbins, n_zbins) raw FP counts
        logN_lo=logN_lo, logN_hi=logN_hi,
        zbins=zbins,
        band_eta_per_nbin=band_eta_per_nbin,
        # scalars
        n_sl_loa0=np.int64(n_sl_loa0),
        n_sl_prod=np.int64(n_sl_prod),
        n_fp_total=np.int64(n_fp_total),
        ell_eff=np.float64(ell_eff),
        snr_min=np.float64(snr_min),
        p_dla_min=np.float64(p_dla_min),
        # η by band
        eta_lls=np.float64(eta["lls"]),
        eta_subdla=np.float64(eta["subdla"]),
        eta_dla=np.float64(eta["dla"]),
        # provenance
        loa0_out=loa0_out, molly_tsv=molly_tsv,
        lya_only_lam_rf_min=np.float64(
            lya_only_lam_rf_min if lya_only_lam_rf_min is not None else -1.0),
    )

    if out_path is None:
        out_path = os.path.join(loa0_out, "loa0_fp_product.npz")
    np.savez(out_path, **out)
    print(f"\n[product] saved -> {out_path}")
    print(f"  N_sl_loa0 (searched SNR>{snr_min}) = {n_sl_loa0}")
    print(f"  N_prod    (prod searched SNR>{snr_min}) = {n_sl_prod}")
    print(f"  N_FP_total (molly op)            = {n_fp_total}")
    print(f"  ell_eff = N_sl_loa0*(N_sl_loa0/N_prod) = {ell_eff:.4g}")
    print(f"  FP counts per band: {band_counts}")
    print(f"  eta per band: lls={eta['lls']:.4f} subdla={eta['subdla']:.4f} "
          f"dla={eta['dla']:.4f}")
    # FP counts per fine logN bin, z-marginalized, for the DLA tier (the A/B-relevant)
    fine_marg = n_fp_fine.sum(axis=1)
    sel203 = logN_lo >= 20.3 - 1e-9
    sel200 = logN_lo >= 20.0 - 1e-9
    print(f"  FP fine-grid counts: >=20.0 -> {int(fine_marg[sel200].sum())}, "
          f">=20.3 -> {int(fine_marg[sel203].sum())} (DLA tier b_FP≈0 expected)")
    out["band_counts"] = band_counts
    out["out_path"] = out_path
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--loa0-out", default=DEF_LOA0_OUT)
    p.add_argument("--prod-cat", default=DEF_PROD_CAT)
    p.add_argument("--molly-tsv", default=DEF_MOLLY)
    p.add_argument("--truth124", default=DEF_TRUTH124)
    p.add_argument("--prod-mockdir", default=DEF_PROD_MOCKDIR)
    p.add_argument("--prod-bal", default=DEF_PROD_BAL)
    p.add_argument("--snr-min", type=float, default=2.0)
    p.add_argument("--p-dla-min", type=float, default=0.99)
    p.add_argument("--out", default=None)
    p.add_argument("--lya-only-lam-rf-min", type=float, default=None,
                   help="FIX 2: restrict FP detections to forest position "
                        "λ_rest=LYA·(1+Z_DLA)/(1+Z_QSO) >= this (Å); pass 1025 to "
                        "match the calibrated Lyα-only WALL-1 molly. Default None = "
                        "full forest window (legacy product).")
    args = p.parse_args(argv)
    build_product(args.loa0_out, args.prod_cat, args.molly_tsv, args.truth124,
                  args.prod_mockdir, args.prod_bal, args.snr_min, args.p_dla_min,
                  args.out, lya_only_lam_rf_min=args.lya_only_lam_rf_min)


if __name__ == "__main__":
    main()
