#!/usr/bin/env python
"""build_bal_fp_product.py — fold the real-LOA leaked-BAL FP into the catalog-HBI FP intensity
(P0 blocker #3). Emits a COMBINED forest+BAL loa0-schema npz that `Loa0FP.from_product` ingests
with NO edit to the frozen `cddf_catalog_hbi.py` — verified linear:
    mu_fp_grid[b,k] = n_fp_fine[b,k] * vol_scale * (1 - band_eta_per_nbin[b])   (cddf_catalog_hbi.py:1153)
so summing count grids sums intensities.

The BAL FP = the real leaked-BAL detections the production `BI_CIV>0` veto MISSED: op-cut
(SNR_REDSIDE>2 & P_DLA>0.99), clean (DLAFLAG==0), broad-trough (a >2000 km/s contiguous CIV trough +
significant AI, from the v2 altbal VAC's VMIN/VMAX_CIV_450 — a DERIVED proxy, see the arbiter design
`notes/2026-07-02_real_loa_bal_arbiter_design.md` for the ~5% fragmentation caveat).

CRITICAL volume/occlusion handling (map gotchas 1-2): the estimator multiplies whatever we put in
`n_fp_fine` by `vol_scale = n_sl_prod/n_sl_loa0` AND by `(1 - band_eta_per_nbin)`. Our BAL counts are
measured directly in PRODUCTION volume and are NOT host-occluded (occlusion is a forest-FP concept), so
we PRE-DIVIDE by `vol_scale*(1-eta)` — the estimator's factors then cancel and the mu_FP contribution
equals the raw production BAL FP count. On the DLA tier (logN>=20.3) eta=0, so only vol_scale matters,
and the forest `n_fp_fine` is ~0 there → BAL dominates the FP subtraction exactly as intended.

Run (WITHOUT this fold = forest only):  --fp-estimator loa0 --fp-product <forest product>
Run (WITH):                             --fp-estimator loa0 --fp-product <this combined npz>
purity_mixture has NO product slot → the fold only exists in the loa0 estimator (headline disconnect).

Env: source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate gpdla . Aggregate/real-LOA privacy.
"""
import os, argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")
import fitsio

FOREST = "/scratch/cavestru_root/cavestru0/mfho/gl_loa0_fp_v1_20260615/outputs/loa0_fp_product_lyaonly1025.npz"
DLACAT = "/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/loa_main_dark_v1/dlacat-loa-main-dark-v1.fits"
VAC = "/nfs/turbo/lsa-cavestru/mfho/DESI/loa/QSO_cat_loa_main_dark_healpix_v2-altbal.fits"  # v2 = production BAL_FLAG source
SNR_MIN, PDLA_MIN = 2.0, 0.99


def broad_trough_tids(vac):
    """TIDs with a >2000 km/s contiguous CIV trough + significant AI (derived from VMIN/VMAX_CIV_450)."""
    v = fitsio.read(vac)
    AI = np.asarray(v["AI_CIV"], float); eAI = np.asarray(v["ERR_AI_CIV"], float)
    vmn = np.asarray(v["VMIN_CIV_450"], float); vmx = np.asarray(v["VMAX_CIV_450"], float)
    wid = np.abs(vmx - vmn); wid[(vmn == 0) & (vmx == 0)] = 0
    widest = wid.max(axis=1)
    sig = (AI > 0) & (eAI > 0) & (AI > 3 * eAI)
    broad = sig & (widest > 2000)
    return set(np.asarray(v["TARGETID"], np.int64)[broad].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forest-product", default=FOREST, help="the loa0 forest-FP product npz to fold into")
    ap.add_argument("--dlacat", default=DLACAT); ap.add_argument("--vac", default=VAC)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "loa0_bal_fp_product.npz"))
    ap.add_argument("--nhi-min", type=float, default=None,
                    help="optional: only fold BAL FP at logN>=this (default: all, the estimator floors at 19.5)")
    ap.add_argument("--n-sl-prod", type=float, default=None,
                    help="CRITICAL: the SEARCHED-sightline count of the run that will CONSUME this "
                         "product (cfg.n_sl_prod, runtime-overridden). vol_scale=n_sl_prod/n_sl_loa0 "
                         "and the estimator multiplies our BAL counts by it, so this MUST match the "
                         "consuming run. Default = the forest product's stored value (the 2LPT-0 MOCK "
                         "= 374177). For the REAL-LOA A/B pass the real-LOA searched count.")
    a = ap.parse_args()

    P = dict(np.load(a.forest_product, allow_pickle=True))
    logN_lo = np.asarray(P["logN_lo"], float); logN_hi = np.asarray(P["logN_hi"], float)
    zbins = np.asarray(P["zbins"], float)
    band_eta = np.asarray(P["band_eta_per_nbin"], float)
    snr_edges = np.asarray(P["snr_edges"], float); nhi_edges = np.asarray(P["nhi_edges"], float)
    n_sl_loa0 = float(P["n_sl_loa0"]); n_sl_prod = float(a.n_sl_prod if a.n_sl_prod else P["n_sl_prod"])
    vol_scale = n_sl_prod / n_sl_loa0
    n_nbins, n_zbins = len(logN_lo), len(zbins) - 1
    assert P["n_fp_fine"].shape == (n_nbins, n_zbins), P["n_fp_fine"].shape

    # --- the real leaked-BAL FP detections ---
    d = fitsio.read(a.dlacat)
    tid = np.asarray(d["TARGETID"], np.int64); nhi = np.asarray(d["NHI"], float)
    zdla = np.asarray(d["Z_DLA"], float); snr = np.asarray(d["SNR_REDSIDE"], float)
    pdla = np.asarray(d["P_DLA"], float); flag = np.asarray(d["DLAFLAG"], int)
    op = (snr > SNR_MIN) & (pdla > PDLA_MIN) & (flag == 0)   # clean = BI>0 already removed
    btids = broad_trough_tids(a.vac)
    isbal = np.array([t in btids for t in tid])
    sel = op & isbal
    if a.nhi_min is not None:
        sel &= (nhi >= a.nhi_min)

    # --- bin on the fine (logN, z_DLA) grid ---
    fine_edges = np.append(logN_lo, logN_hi[-1])
    nb = np.digitize(nhi[sel], fine_edges) - 1
    zb = np.digitize(zdla[sel], zbins) - 1
    inside = (nb >= 0) & (nb < n_nbins) & (zb >= 0) & (zb < n_zbins)
    N_bal_fine = np.zeros((n_nbins, n_zbins), float)
    np.add.at(N_bal_fine, (nb[inside], zb[inside]), 1.0)
    n_out_z = int((~inside & ((nb >= 0) & (nb < n_nbins))).sum())  # dropped for z_DLA outside [2,3.5]

    # --- bin on the molly (SNR, N) grid (for v3x per-object-term coherence) ---
    ib = np.digitize(snr[sel], snr_edges) - 1
    jb = np.digitize(nhi[sel], nhi_edges) - 1
    ins_m = (ib >= 0) & (ib < len(snr_edges) - 1) & (jb >= 0) & (jb < len(nhi_edges) - 1)
    N_bal_molly = np.zeros_like(np.asarray(P["n_fp_molly"], float))
    np.add.at(N_bal_molly, (ib[ins_m], jb[ins_m]), 1.0)

    # --- PRE-DIVIDE so the estimator's vol_scale*(1-eta) cancels (map gotcha 1-2) ---
    eta_col = (1.0 - band_eta)[:, None]                      # (n_nbins,1); =1 on DLA tier
    n_fine_bal = N_bal_fine / (vol_scale * eta_col)
    n_molly_bal = N_bal_molly / vol_scale                    # per-object term; DLA-tier eta=0

    out = dict(P)
    out["n_fp_fine"] = np.asarray(P["n_fp_fine"], float) + n_fine_bal
    out["n_fp_molly"] = np.asarray(P["n_fp_molly"], float) + n_molly_bal
    out["bal_fold"] = True
    out["bal_n_fine_raw"] = N_bal_fine
    out["bal_source"] = "leaked broad-trough (>2000km/s) BAL, op-cut clean, v2 VAC"
    np.savez(a.out, **out)

    # --- report ---
    nbal = int(sel.sum()); Wbal = float(np.sum(10.0 ** nhi[sel]))
    dla = (nb >= 0)
    hi = nhi[sel] >= 20.3
    print(f"forest product: {a.forest_product}")
    print(f"  vol_scale = n_sl_prod/n_sl_loa0 = {n_sl_prod:.0f}/{n_sl_loa0:.0f} = {vol_scale:.3f}")
    print(f"leaked broad-trough BAL FP (op-cut clean): {nbal} detections, {int(hi.sum())} at logN>=20.3")
    print(f"  z_DLA>3.5 dropped (outside grid): {n_out_z} detections")
    print(f"  fine-grid BAL counts by z bin (raw): {N_bal_fine.sum(axis=0).astype(int)}  total={int(N_bal_fine.sum())}")
    print(f"  logN>=20.3 BAL fine counts (raw): {int(N_bal_fine[np.append(logN_lo,logN_hi[-1])[:-1]>=20.3].sum())}")
    foresthi = np.asarray(P['n_fp_fine'],float)[logN_lo>=20.3].sum()
    print(f"  forest n_fp_fine at logN>=20.3 = {foresthi:.1f} (should be ~0 → BAL dominates the DLA-tier FP)")
    print(f"\ncombined product -> {a.out}")
    print(f"Run A/B:  --fp-estimator loa0 --fp-product {a.forest_product}   (forest only)")
    print(f"          --fp-estimator loa0 --fp-product {a.out}   (forest+BAL)")


if __name__ == "__main__":
    main()
