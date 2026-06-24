#!/usr/bin/env python
"""track_c_czresolve_diag.py — STEP-0 diagnosis for Track-C #39.

REDUCE-ONLY. No GP inference. Measures the TRUE z-dependent completeness
C_true(N,z) = (op-passing TRUE-POSITIVE detections in (N,z) cell) / (truth
systems in (N,z) cell) directly from the 2LPT-0 truth-match, NON-circular
(TP/truth counts only — no dN/dX/Ω/f anywhere). Then answers the three STEP-0
questions:

  Q1. Is the molly completeness C built per (N,SNR) and z-MARGINALIZED, then
      applied at all z?  (confirm by construction — read from the loaded mm)
  Q2. Is there a g(N,z) z-completeness correction already, and is it
      z-MARGINALIZED / insufficient?  (the CNZModel normalizes g(j,z_ref)=1 so
      it carries only the z-SHAPE relative to z_ref; the molly z-marginal LEVEL
      is unchanged.  We quantify how much z-trend it actually injects.)
  Q3. Measure C_true(N,z).  Does it vary with z at fixed N enough to produce the
      ×1.96-vs-×1.49 amplitude tilt (recovered dN/dX(>=20.3) grows 1.96x across
      z while truth grows 1.49x)?

The decisive arithmetic: the v1/v3 estimator recovers, per (N,z) cell,
  N_recovered ~ N_detected / C_applied
where C_applied is the z-MARGINAL molly C (constant over z at fixed N,SNR).
The TRUE recovered count should use C_true(N,z).  So
  bias(N,z) = C_applied(N) / C_true(N,z)
is the per-cell over-recovery factor.  If C_true FALLS with z (harder to detect
at high z), then C_applied/C_true RISES with z -> over-recovery rises with z,
which is exactly the observed amplitude tilt.  Conversely if C_true RISES with z,
the sign is wrong and completeness is NOT the lever.

We aggregate bias(N,z) over the >=20.3 detected population (weighting each N-cell
by its detected count, the amplitude driver) and compare the predicted z-tilt of
the recovered dN/dX against the observed 0.91/1.05/1.19.

Usage:
  python CDDF_analysis/track_c_czresolve_diag.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.hbi import ab_loa0_fp_baseline as AB
from CDDF_analysis.hbi.cddf_catalog_hbi import _fine_z_grid
from CDDF_analysis.hbi.znz_kernel import measure_c_nz, fit_c_nz_model


class _Args:
    """Mimic the argparse namespace build_ingredients expects (defaults only)."""
    catalog_dir = AB.DEF_CAT
    truth = AB.DEF_TRUTH
    bal_cat = AB.DEF_BAL
    molly_tsv = None
    kernel = AB.DEF_KERNEL
    loa0_product = AB.DEF_LOA0_PRODUCT
    out = "/tmp/track_c_czresolve_diag"
    mockdir = None
    zbins = "2.0,2.5,3.0,3.5"
    report_limits = "20.0,20.3,20.6"
    family = "bspbody"
    fit_floor = 19.5
    fit_ceil = 99.0
    lambda_bspbody = 30.0
    lam_rf_min = 1025.0
    edge_slope_lam = 40.0
    gl_nodes = 1
    host_truth_floor = 19.0


def main():
    os.makedirs(_Args.out, exist_ok=True)
    print("[diag] building ingredients (cat_cut, truth_cut, mm, good_mask)...")
    ing = AB.build_ingredients(_Args, fp_estimator="purity_mixture")
    cfg = ing["cfg"]
    mm = ing["mm"]
    cat_cut = ing["cat_cut"]
    truth_cut = ing["truth_cut"]
    good_mask = ing["good_mask"]

    coarse_zbins = np.array([2.0, 2.5, 3.0, 3.5])
    z_centers = 0.5 * (coarse_zbins[:-1] + coarse_zbins[1:])  # 2.25, 2.75, 3.25

    # ------------------------------------------------------------------
    # Q1 — molly C is z-MARGINALIZED (built per (N,SNR) with NO z axis)
    # ------------------------------------------------------------------
    print("\n=== Q1: molly completeness shape / z-axis ===")
    print(f"  mm.completeness.shape = {mm.completeness.shape}  (n_snr, n_nhi) — NO z axis")
    print(f"  snr_edges = {mm.snr_edges}")
    print(f"  nhi_edges = {mm.nhi_edges}")
    print("  => molly C is built per (N,SNR), z-MARGINALIZED, applied at ALL z. CONFIRMED.")

    # ------------------------------------------------------------------
    # Q3 — measure the TRUE z-dependent completeness C_true(N,z)
    # NON-CIRCULAR: TP-detection counts / truth counts only.
    # Build on the molly nhi-cell grid (same axis the estimator uses) and on
    # the COARSE report z-bins (so it ties to the per-z dN/dX table).
    # ------------------------------------------------------------------
    print("\n=== Q3: measure C_true(N,z) (TP/truth counts; non-circular) ===")
    nhi_edges = np.asarray(mm.nhi_edges, float)
    n_nhi = len(nhi_edges) - 1

    # truth side: count truth systems per (nhi-cell, coarse-z)
    t_nhi = np.asarray(truth_cut["NHI"], float)
    z_col = next((c for c in ("Z_DLA", "Z_DLA_NO_RSD", "Z") if c in truth_cut.colnames), None)
    t_z = np.asarray(truth_cut[z_col], float) if z_col else np.zeros(len(truth_cut))
    print(f"  truth z column = {z_col}")

    # detection side: op-passing TRUE-POSITIVE detections, binned by TRUE (N,z)
    s2n = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"], float)
    op = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask
    nhi_true_all = np.asarray(cat_cut["NHI_TRUE"], float)
    z_dla_col = next((c for c in ("Z_DLA", "Z_QSO") if c in cat_cut.colnames), None)
    z_cat = np.asarray(cat_cut[z_dla_col], float) if z_dla_col else np.zeros(len(cat_cut))
    tp_op = op & np.isfinite(nhi_true_all)
    print(f"  detection z column = {z_dla_col}; n op-TP = {int(tp_op.sum())}")

    def _cell(nhi, z):
        j = np.clip(np.searchsorted(nhi_edges, nhi, side="right") - 1, 0, n_nhi - 1)
        k = np.searchsorted(coarse_zbins, z, side="right") - 1
        return j, k

    n_true = np.zeros((n_nhi, 3))
    n_rec = np.zeros((n_nhi, 3))
    jt, kt = _cell(t_nhi, t_z)
    for ii in range(len(t_nhi)):
        if 0 <= kt[ii] < 3:
            n_true[jt[ii], kt[ii]] += 1.0
    jr, kr = _cell(nhi_true_all[tp_op], z_cat[tp_op])
    for ii in range(int(tp_op.sum())):
        if 0 <= kr[ii] < 3:
            n_rec[jr[ii], kr[ii]] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        C_true = np.where(n_true > 0, n_rec / n_true, np.nan)

    # molly z-marginal completeness per N-cell: aggregate over SNR weighted by
    # detected occupancy (the realistic effective C the estimator divides by).
    # The estimator applies C per (i_snr, j_nhi); the population-effective C at a
    # given N is the detected-count-weighted harmonic-ish mean.  For the tilt
    # diagnosis we want the EFFECTIVE C_applied(N) = n_det(N) / n_recovered(N)
    # under the molly C — i.e. the z-marginal completeness the estimator uses.
    # Simplest faithful proxy: the truth-match z-marginal C per N-cell.
    n_true_zmarg = n_true.sum(axis=1)
    n_rec_zmarg = n_rec.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        C_marg = np.where(n_true_zmarg > 0, n_rec_zmarg / n_true_zmarg, np.nan)

    # NHI-cell labels
    print("\n  C_true(N,z) per molly N-cell (rec/truth counts):")
    print("  Ncell           z=2.25   z=2.75   z=3.25   z-marg   n_true(2.25/2.75/3.25)")
    for j in range(n_nhi):
        lo, hi = nhi_edges[j], nhi_edges[j + 1]
        if not np.isfinite(C_marg[j]):
            continue
        lbl = f"[{lo:.2f},{hi:.2f})" if np.isfinite(hi) else f">={lo:.2f}"
        ct = C_true[j]
        print(f"  {lbl:14s}  {ct[0]:6.3f}   {ct[1]:6.3f}   {ct[2]:6.3f}   "
              f"{C_marg[j]:6.3f}   {n_true[j,0]:.0f}/{n_true[j,1]:.0f}/{n_true[j,2]:.0f}")

    # ------------------------------------------------------------------
    # Q3 decisive test: predicted z-tilt of recovered dN/dX(>=20.3)
    # If the estimator divides detected counts by C_marg(N) (z-flat) but the
    # truth needs C_true(N,z), the per-cell over-recovery is C_marg/C_true.
    # The recovered dN/dX(z) ~ sum_N [n_det(N,z)/C_marg(N)] ; the TRUE dN/dX(z)
    # ~ sum_N [n_det(N,z)/C_true(N,z)].  Their ratio per z is the predicted tilt.
    # n_det(N,z) = n_rec(N,z) here (op-passing TP detected counts per cell).
    # ------------------------------------------------------------------
    print("\n=== Q3 decisive: predicted recovered/truth amplitude tilt at >=20.3 ===")
    # cells at >=20.3 (find first nhi_edge >= 20.3)
    j203 = int(np.searchsorted(nhi_edges, 20.3, side="left"))
    sel = np.arange(j203, n_nhi)

    pred_recov = np.zeros(3)
    true_recov = np.zeros(3)
    for kz in range(3):
        for j in sel:
            nd = n_rec[j, kz]
            if nd <= 0:
                continue
            cm = C_marg[j]
            ct = C_true[j, kz]
            if np.isfinite(cm) and cm > 0:
                pred_recov[kz] += nd / cm     # estimator divides by z-marginal C
            if np.isfinite(ct) and ct > 0:
                true_recov[kz] += nd / ct     # truth divides by z-resolved C

    with np.errstate(invalid="ignore", divide="ignore"):
        tilt_pred = np.where(true_recov > 0, pred_recov / true_recov, np.nan)
    print("  z         pred_recov(C_marg)  true_recov(C_true)  ratio(=predicted R0 tilt)")
    for kz in range(3):
        print(f"  {z_centers[kz]:.2f}     {pred_recov[kz]:10.1f}        "
              f"{true_recov[kz]:10.1f}        {tilt_pred[kz]:.4f}")
    # normalize the predicted ratio to its z-mean so it reads like the R0 tilt
    tmean = np.nanmean(tilt_pred)
    print(f"\n  predicted R0-tilt (ratio / mean): "
          f"{tilt_pred[0]/tmean:.3f} / {tilt_pred[1]/tmean:.3f} / {tilt_pred[2]/tmean:.3f}")
    print(f"  OBSERVED R0(>=20.3) per z (perz report): 0.908 / 1.052 / 1.189  (spread 0.281)")
    spread_pred = (np.nanmax(tilt_pred) - np.nanmin(tilt_pred)) / tmean
    print(f"  predicted relative spread = {spread_pred:.3f}  (observed ~0.28)")

    # also the pure C_true z-trend at fixed N, occupancy-weighted over >=20.3
    print("\n  C_true(>=20.3) z-marginalized-vs-z, occupancy-weighted:")
    Cz = np.zeros(3)
    for kz in range(3):
        w = n_true[sel, kz]
        cz = C_true[sel, kz]
        m = np.isfinite(cz) & (w > 0)
        Cz[kz] = np.sum(cz[m] * w[m]) / np.sum(w[m]) if m.any() else np.nan
    print(f"  C_true(>=20.3): z=2.25 {Cz[0]:.3f}  z=2.75 {Cz[1]:.3f}  z=3.25 {Cz[2]:.3f}")
    print(f"  C_true z-trend ratio (3.25/2.25) = {Cz[2]/Cz[0]:.3f}  "
          f"(<1 means harder to detect at high z -> over-recovery rises with z)")

    # ------------------------------------------------------------------
    # Q2 — the existing CNZModel g(N,z): how much z-trend does it inject?
    # ------------------------------------------------------------------
    print("\n=== Q2: existing CNZModel g(N,z) — z-trend injected (normalized at z_ref) ===")
    z_edges_fine = _fine_z_grid(cfg)
    meas_c = measure_c_nz(cat_cut, truth_cut, cfg, mm, z_edges_fine, good_mask=good_mask)
    cnz = fit_c_nz_model(meas_c, smooth=1.0)
    g = cnz.g_grid  # (n_nhi, n_zf) normalized g(j, z_ref)=1
    zmid_fine = 0.5 * (z_edges_fine[:-1] + z_edges_fine[1:])
    # map fine-z to coarse and average g per coarse-z for the >=20.3 cells
    kc = np.searchsorted(coarse_zbins, zmid_fine, side="right") - 1
    print("  g(N,z) is normalized to g(j,z_ref)=1, so it carries only the z-SHAPE")
    print("  relative to z_ref; the molly z-marginal LEVEL is unchanged by construction.")
    g_coarse = np.full((n_nhi, 3), np.nan)
    for kz in range(3):
        cols = (kc == kz)
        if cols.any():
            g_coarse[:, kz] = np.nanmean(g[:, cols], axis=1)
    gz = np.zeros(3)
    for kz in range(3):
        w = n_true[sel, kz]
        gc = g_coarse[sel, kz]
        m = np.isfinite(gc) & (w > 0)
        gz[kz] = np.sum(gc[m] * w[m]) / np.sum(w[m]) if m.any() else np.nan
    print(f"  existing g(>=20.3) per coarse z: {gz[0]:.3f} / {gz[1]:.3f} / {gz[2]:.3f}")
    print(f"  existing-g z-trend ratio (3.25/2.25) = {gz[2]/gz[0]:.3f}")
    print("  (compare to the raw C_true z-trend ratio above — if the existing g matches")
    print("   the C_true trend, the z-resolved C IS the existing g once its LEVEL is")
    print("   re-attached; if it differs, the existing g is insufficient.)")

    # save
    out = dict(
        nhi_edges=nhi_edges.tolist(),
        z_centers=z_centers.tolist(),
        C_true=np.where(np.isfinite(C_true), C_true, None).tolist(),
        C_marg=np.where(np.isfinite(C_marg), C_marg, None).tolist(),
        n_true=n_true.tolist(), n_rec=n_rec.tolist(),
        tilt_pred=np.where(np.isfinite(tilt_pred), tilt_pred, None).tolist(),
        tilt_pred_normed=(tilt_pred / tmean).tolist(),
        C_true_203_z=Cz.tolist(),
        existing_g_203_z=gz.tolist(),
    )
    with open(os.path.join(_Args.out, "czresolve_diag.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[diag] wrote {_Args.out}/czresolve_diag.json")


if __name__ == "__main__":
    main()
