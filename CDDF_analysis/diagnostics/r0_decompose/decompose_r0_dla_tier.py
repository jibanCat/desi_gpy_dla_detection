"""decompose_r0_dla_tier.py — Phase-1 DECOMPOSE of the DLA-tier R0=1.16 over-recovery
on 2LPT-0 (reduce-only, cached kernel, NO inference, NO SLURM, NO tilt).

Builds the calibrated WALL-1 loa0 ingredients ONCE via
ab_loa0_fp_baseline.build_ingredients (same cat_cut / frozen molly C/rho /
pathlength / cached 2-D posterior kernel that run_phase3d_postkernel stage 2/3
uses), runs the UNTILTED baseline recovery, then isolates the four hypotheses
for the residual R0(dN/dX>=20.3)=1.159:

  H4 z-trend     : R0(dN/dX>=20.3) PER z-bin (is the 1.16 a high-z over-recovery?)
                   + decompose z-structure into pathlength dX(z), C(z), kernel.
  H1 completeness: molly C(>=20.3) vs the ACTUAL truth-recovery rate
                   (matched detections / truth absorbers) at >=20.3, per z & SNR.
                   If molly C < true recovery, 1/C over-counts.
  H2 up-migration: net flux of truth absorbers across the 20.3 threshold
                   (truth 20.2-20.3 -> detected >=20.3 vs vice-versa) and whether
                   the frozen kernel deconvolution nets it out.
  H3 M_b norm    : M_b (= sum pathlength x C) vs an independent dX(z) computation.

Outputs a single TSV per hypothesis under --out and prints a ranked breakdown.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.hbi.ab_loa0_fp_baseline import (
    build_ingredients, _resolve_molly,
    DEF_CAT, DEF_TRUTH, DEF_BAL, DEF_KERNEL, DEF_LOA0_PRODUCT,
)
from CDDF_analysis.hbi.cddf_tilt_closure import baseline_recovery
from CDDF_analysis.hbi.cddf_catalog_hbi import (
    _cell_index, _zbin_index, _bin_index_logN, make_C_interpolator,
    C_FLOOR, _apply_C_to_M,
)


def _zbin_centers(zbins):
    zbins = np.asarray(zbins, float)
    return 0.5 * (zbins[:-1] + zbins[1:])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog-dir", default=DEF_CAT)
    p.add_argument("--truth", default=DEF_TRUTH)
    p.add_argument("--bal-cat", default=DEF_BAL)
    p.add_argument("--molly-tsv", default=None)
    p.add_argument("--kernel", default=DEF_KERNEL)
    p.add_argument("--loa0-product", default=DEF_LOA0_PRODUCT)
    p.add_argument("--out", default="/tmp/decompose_r0")
    p.add_argument("--mockdir", default=None)
    p.add_argument("--zbins", default="2.0,2.5,3.0,3.5")
    p.add_argument("--report-limits", default="20.0,20.3,20.6")
    p.add_argument("--family", default="bspbody")
    p.add_argument("--fit-floor", type=float, default=19.5)
    p.add_argument("--fit-ceil", type=float, default=99.0)
    p.add_argument("--lambda-bspbody", type=float, default=30.0)
    p.add_argument("--lam-rf-min", type=float, default=1025.0)
    p.add_argument("--edge-slope-lam", type=float, default=40.0)
    p.add_argument("--gl-nodes", type=int, default=1)
    p.add_argument("--host-truth-floor", type=float, default=19.0)
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    limits = tuple(float(x) for x in args.report_limits.split(","))

    print("=" * 78)
    print("[decompose] build calibrated WALL-1 loa0 ingredients (kernel ON)")
    print("=" * 78)
    ing = build_ingredients(args, "loa0", loa0_product=args.loa0_product)
    cfg = ing["cfg"]
    cfg.report_logN_limits = limits
    cfg._wall1_estimator = "v3"
    cat_cut = ing["cat_cut"]; truth_cut = ing["truth_cut"]
    is_TP = ing["is_TP"]; good_mask = ing["good_mask"]
    mm = ing["mm"]; X_tot = np.asarray(ing["X_tot"], float)
    zbins = np.asarray(cfg.zbins, float)
    zc = _zbin_centers(zbins)
    n_zb = len(zbins) - 1
    print(f"  n_sl_prod={ing['n_sl']}, X_tot(per z)={np.array2string(X_tot, precision=3)}")
    print(f"  zbins={zbins}, centers={zc}")

    # =====================================================================
    # Run the v3 baseline recovery (the SAME estimator_fn the A/B uses)
    # =====================================================================
    print("\n[decompose] v3 baseline recovery (loa0 FP, kernel ON, untilted)")
    base = baseline_recovery(
        cfg, cat_cut, is_TP, good_mask, truth_cut,
        ing["C_interp"], ing["fp_model"], X_tot,
        ing["logN_lo"], ing["logN_hi"], ing["N_b"], ing["dN_b"],
        estimator_fn=ing["estimator_fn"])
    e0 = base["e0"]; t0 = base["t0"]

    # ---------------------------------------------------------------------
    # H4: R0(dN/dX>=20.3) per z-bin
    # ---------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("H4  R0(dN/dX) PER z-bin  (recovered/truth) — is the 1.16 a HIGH-z trend?")
    print("=" * 78)
    h4_rows = []
    for lim in limits:
        ez = np.asarray(e0["dndx_z"][lim], float)
        tz = np.asarray(t0["dndx_z"][lim], float)
        r0z = np.where(tz > 0, ez / tz, np.nan)
        print(f"--- limit >= {lim} ---  (total R0={base['R0_dndx_total'][lim]:.4f})")
        for k in range(n_zb):
            print(f"   z[{zbins[k]:.1f},{zbins[k+1]:.1f}] center={zc[k]:.2f}: "
                  f"R0={r0z[k]:.4f}  (est={ez[k]:.5f} truth={tz[k]:.5f} X={X_tot[k]:.2f})")
            h4_rows.append((lim, zc[k], r0z[k], ez[k], tz[k], X_tot[k]))
    with open(os.path.join(args.out, "h4_r0_per_zbin.tsv"), "w") as fh:
        fh.write("limit\tz_center\tR0_dndx_z\tdndx_est\tdndx_truth\tX_z\n")
        for r in h4_rows:
            fh.write("\t".join(f"{v:.6g}" for v in r) + "\n")

    # also Omega per zbin would need f-band; dN/dX(z) is the clean z-resolved knob.

    # =====================================================================
    # Build the matched-detection arrays once (op mask = headline operating set)
    # =====================================================================
    s2n = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"], float)
    op = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask
    nhi_pred = np.asarray(cat_cut["NHI"], float)[op]
    nhi_true_op = np.asarray(cat_cut["NHI_TRUE"], float)[op]  # NaN if hostless (FP)
    z_pred = np.asarray(cat_cut["Z_DLA"], float)[op]
    snr_op = s2n[op]
    is_tp_op = np.isfinite(nhi_true_op)

    # truth-side arrays (SNR>snr_min, same window as truth_cut)
    t_nhi = np.asarray(truth_cut["NHI"], float)
    t_z = np.asarray(truth_cut["Z_DLA"], float)
    t_snr = np.asarray(truth_cut["S2N_RED"], float)
    tkeep = t_snr > cfg.snr_min
    t_nhi, t_z, t_snr = t_nhi[tkeep], t_z[tkeep], t_snr[tkeep]
    t_zidx = _zbin_index(t_z, zbins)

    C_interp = make_C_interpolator(mm)

    # ---------------------------------------------------------------------
    # H1: molly C(>=20.3) vs ACTUAL truth-recovery rate (matched/truth)
    #     per z & SNR. True recovery = (# truth >=20.3 with a detection >=floor
    #     matched to it) / (# truth >=20.3). C is z-flat by construction.
    #     1/C over-counts iff molly_C < true_recovery.
    #     We measure the empirical recovery as (# TP detections whose TRUE host
    #     >=20.3) / (# truth absorbers >=20.3) in the same (z,SNR) cell.
    # ---------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("H1  molly C(>=20.3) vs EMPIRICAL truth-recovery rate (per z, per SNR)")
    print("=" * 78)
    LIM = 20.3
    # empirical recovery per z-bin: # truth>=LIM that were recovered / # truth>=LIM
    # recovered truth = detections (op, TP) whose NHI_TRUE>=LIM, binned by truth z.
    # Use NHI_TRUE of the detection (the matched truth host) and its z_pred ~ z_true.
    tp_true_nhi = nhi_true_op[is_tp_op]
    tp_z = z_pred[is_tp_op]
    tp_snr = snr_op[is_tp_op]
    tp_zidx = _zbin_index(tp_z, zbins)
    # The molly C the estimator APPLIES is read at each DETECTION's (NHI_pred, SNR).
    # The relevant "applied C" for the >=20.3 reduced dN/dX is the harmonic-style
    # 1/C weight averaged over detections binned by predicted NHI>=20.3.
    h1_rows = []
    print(f"  per z-bin (truth NHI>=%.1f recovery vs molly-C applied):" % LIM)
    print(f"  {'z':>5} {'n_truth':>8} {'n_recov':>8} {'emp_rec':>8} "
          f"{'<C_app>':>8} {'<1/C_app>':>10} {'1/<C>':>8}")
    for k in range(n_zb):
        nt = int(((t_nhi >= LIM) & (t_zidx == k)).sum())
        nr = int(((tp_true_nhi >= LIM) & (tp_zidx == k)).sum())
        emp = nr / nt if nt > 0 else np.nan
        # C the estimator actually applies to >=20.3 DETECTIONS in this z-bin
        sel_det = (nhi_pred >= LIM) & (_zbin_index(z_pred, zbins) == k)
        if sel_det.sum() > 0:
            C_app = C_interp(nhi_pred[sel_det], snr_op[sel_det])
            mean_C = float(np.mean(C_app))
            mean_invC = float(np.mean(1.0 / np.clip(C_app, C_FLOOR, None)))
        else:
            mean_C = np.nan; mean_invC = np.nan
        h1_rows.append((zc[k], nt, nr, emp, mean_C, mean_invC,
                        (1.0 / mean_C if mean_C > 0 else np.nan)))
        print(f"  {zc[k]:>5.2f} {nt:>8d} {nr:>8d} {emp:>8.4f} "
              f"{mean_C:>8.4f} {mean_invC:>10.4f} {1.0/mean_C if mean_C>0 else np.nan:>8.4f}")
    # SNR-resolved (integrated over z)
    print(f"\n  per SNR cell (>= {LIM}, integrated over z):")
    snr_edges = np.asarray(mm.snr_edges, float)
    print(f"  {'snr_lo':>7} {'snr_hi':>7} {'n_truth':>8} {'n_recov':>8} "
          f"{'emp_rec':>8} {'molly_C':>8}")
    h1_snr_rows = []
    for i in range(len(snr_edges) - 1):
        slo, shi = snr_edges[i], snr_edges[i + 1]
        nt = int(((t_nhi >= LIM) & (t_snr >= slo) & (t_snr < shi)).sum())
        nr = int(((tp_true_nhi >= LIM) & (tp_snr >= slo) & (tp_snr < shi)).sum())
        emp = nr / nt if nt > 0 else np.nan
        # molly C in this SNR row at the [20.3,20.5) cell (representative DLA-tier)
        i_snr, j_nhi = _cell_index(mm, np.array([20.4]), np.array([0.5 * (slo + min(shi, slo+1))]))
        # use the cell index for this snr row directly:
        ii = min(max(int(np.searchsorted(snr_edges, 0.5 * (slo + shi), side="right") - 1), 0),
                 len(snr_edges) - 2)
        jj = int(np.searchsorted(np.asarray(mm.nhi_edges, float), 20.4, side="right") - 1)
        molly_C = float(mm.completeness[ii, jj])
        h1_snr_rows.append((slo, shi, nt, nr, emp, molly_C))
        print(f"  {slo:>7.1f} {shi:>7.1f} {nt:>8d} {nr:>8d} {emp:>8.4f} {molly_C:>8.4f}")
    # integrated
    nt_all = int((t_nhi >= LIM).sum())
    nr_all = int((tp_true_nhi >= LIM).sum())
    emp_all = nr_all / nt_all if nt_all > 0 else np.nan
    sel_all = (nhi_pred >= LIM)
    C_all = C_interp(nhi_pred[sel_all], snr_op[sel_all])
    print(f"\n  INTEGRATED >= {LIM}: emp_recovery = {nr_all}/{nt_all} = {emp_all:.4f}")
    print(f"                       mean applied C (over >=20.3 detections) = {np.mean(C_all):.4f}")
    print(f"                       => 1/C over-correction factor ~ emp_rec/mean_C "
          f"= {emp_all/np.mean(C_all):.4f}")
    print(f"     (if emp_rec > mean_C the 1/C UNDER-corrects; if emp_rec < mean_C OVER-corrects)")
    with open(os.path.join(args.out, "h1_completeness_vs_recovery.tsv"), "w") as fh:
        fh.write("kind\tz_or_snrlo\tsnrhi\tn_truth\tn_recov\temp_rec\tmolly_C\tinvC\n")
        for r in h1_rows:
            fh.write(f"zbin\t{r[0]:.3f}\t\t{r[1]}\t{r[2]}\t{r[3]:.6g}\t{r[4]:.6g}\t{r[5]:.6g}\n")
        for r in h1_snr_rows:
            fh.write(f"snrcell\t{r[0]:.3f}\t{r[1]:.3f}\t{r[2]}\t{r[3]}\t{r[4]:.6g}\t{r[5]:.6g}\t\n")
        fh.write(f"integrated\t{LIM}\t\t{nt_all}\t{nr_all}\t{emp_all:.6g}\t{np.mean(C_all):.6g}\t\n")

    # ---------------------------------------------------------------------
    # H2: net up-migration across 20.3 (the +0.06 dex prior-edge bias)
    #     Among matched (TP) detections, look at the joint (NHI_TRUE, NHI_pred)
    #     scatter across the 20.3 threshold:
    #       up   = true <20.3 but pred>=20.3   (false promotion into the tier)
    #       down = true>=20.3 but pred <20.3   (demotion out of the tier)
    #     Net up = up - down. The kernel deconvolution should net this out;
    #     residual net up at >=20.3 over-counts.
    #     ALSO report the MAP NHI bias on TP detections at >=20.3.
    # ---------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("H2  up-migration across 20.3 (predicted vs true NHI on matched TP)")
    print("=" * 78)
    tt = nhi_true_op[is_tp_op]
    pp = nhi_pred[is_tp_op]
    up = int(((tt < LIM) & (pp >= LIM)).sum())          # promoted into >=20.3
    down = int(((tt >= LIM) & (pp < LIM)).sum())        # demoted out of >=20.3
    n_true_ge = int((tt >= LIM).sum())
    n_pred_ge = int((pp >= LIM).sum())
    print(f"  matched TP detections: {len(tt)}")
    print(f"  true>=20.3 = {n_true_ge}, pred>=20.3 = {n_pred_ge}")
    print(f"  UP   (true<20.3 & pred>=20.3) = {up}")
    print(f"  DOWN (true>=20.3 & pred<20.3) = {down}")
    print(f"  NET up into >=20.3 (TP only) = {up - down}  "
          f"({100*(up-down)/max(n_true_ge,1):.2f}% of true>=20.3)")
    # the boundary band 20.2-20.3 truth -> pred>=20.3
    band_lo, band_hi = LIM - 0.1, LIM
    in_band = (tt >= band_lo) & (tt < band_hi)
    band_promoted = int((in_band & (pp >= LIM)).sum())
    print(f"  truth in [{band_lo:.1f},{band_hi:.1f}): {int(in_band.sum())}, "
          f"of which pred>=20.3: {band_promoted}")
    # MAP NHI bias on TP at >=20.3 (true), median pred-true
    sel_tp_ge = tt >= LIM
    bias_med = float(np.median(pp[sel_tp_ge] - tt[sel_tp_ge])) if sel_tp_ge.sum() else np.nan
    bias_mean = float(np.mean(pp[sel_tp_ge] - tt[sel_tp_ge])) if sel_tp_ge.sum() else np.nan
    print(f"  NHI bias (pred-true) on TP true>=20.3: median={bias_med:+.4f} mean={bias_mean:+.4f}")
    # how many EXTRA detections >=20.3 vs truth absorbers >=20.3 (raw, no C)
    raw_excess = n_pred_ge - n_true_ge
    print(f"  RAW detection excess at pred>=20.3 vs true>=20.3 (TP) = {raw_excess} "
          f"({100*raw_excess/max(n_true_ge,1):.2f}%)")
    with open(os.path.join(args.out, "h2_upmigration.tsv"), "w") as fh:
        fh.write("metric\tvalue\n")
        for name, v in (("n_matched_TP", len(tt)), ("n_true_ge_203", n_true_ge),
                        ("n_pred_ge_203", n_pred_ge), ("up_into", up),
                        ("down_out", down), ("net_up", up - down),
                        ("band_202_203_truth", int(in_band.sum())),
                        ("band_promoted_to_203", band_promoted),
                        ("nhi_bias_median", bias_med), ("nhi_bias_mean", bias_mean),
                        ("raw_excess_pred_vs_true", raw_excess)):
            fh.write(f"{name}\t{v}\n")

    # ---------------------------------------------------------------------
    # H3: M_b normalizer vs independent dX(z). The selection normalizer for the
    #     >=20.3 rate is mu_det = sum_b M_b f_b; M_b = sum_s dX_s * C(cell). The
    #     C-free pathlength integral sum over fine z in a coarse bin must equal
    #     X_tot[k] (the dN/dX denominator). Check sum_kz PXz over each coarse
    #     z-bin == X_tot[k]. Also report the C-weighted M_b mass in the >=20.3
    #     band per z (the effective searched dX*C).
    # ---------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("H3  M_b normalizer vs independent dX(z)")
    print("=" * 78)
    # rebuild M_meta to inspect PX (pathlength per fine-z, C-free) and C-applied M
    from CDDF_analysis.hbi.cddf_catalog_hbi import build_M_b, build_fine_grid, _fine_z_grid
    logN_lo, logN_hi, N_b, dN_b = build_fine_grid(cfg)
    qlo = ing.get("qzl"); qhi = ing.get("qzh"); qsn = ing.get("qsn")
    # build_ingredients doesn't return per-sl arrays directly; rebuild via build_pathlength
    from CDDF_analysis.hbi.cddf_catalog_hbi import build_pathlength, _build_qso_lookup
    ql = _build_qso_lookup(cfg)
    _, _, qzl, qzh, qsn, Xcalc = build_pathlength(cfg, qso_lookup=ql, return_per_sl=True)
    z_edges_fine = _fine_z_grid(cfg)
    M_meta = build_M_b(qzl, qzh, qsn, mm, logN_lo, logN_hi, N_b, dN_b,
                       z_edges_fine, Xcalc, cfg)
    PXz = M_meta["PX"].sum(axis=0)  # total pathlength per fine-z bin (C-free)
    # map fine-z -> coarse-z and sum
    from CDDF_analysis.hbi.cddf_catalog_hbi import _fine_to_coarse_zmap
    zfmap = _fine_to_coarse_zmap(z_edges_fine, zbins)
    X_from_M = np.zeros(n_zb)
    for kz in range(len(zfmap)):
        if zfmap[kz] >= 0:
            X_from_M[zfmap[kz]] += PXz[kz]
    print(f"  {'z':>5} {'X_tot(dndx denom)':>18} {'sum_PXz (M_b path)':>18} {'ratio':>8}")
    h3_rows = []
    for k in range(n_zb):
        rat = X_from_M[k] / X_tot[k] if X_tot[k] > 0 else np.nan
        print(f"  {zc[k]:>5.2f} {X_tot[k]:>18.4f} {X_from_M[k]:>18.4f} {rat:>8.5f}")
        h3_rows.append((zc[k], X_tot[k], X_from_M[k], rat))
    print(f"  TOTAL  {X_tot.sum():>18.4f} {X_from_M.sum():>18.4f} "
          f"{X_from_M.sum()/X_tot.sum():>8.5f}")
    # C-applied M_b: effective searched dX*C at >=20.3 band, per fine-z then coarse
    C_matrix = mm.completeness
    M_full = _apply_C_to_M(M_meta, C_matrix).reshape(len(logN_lo), len(z_edges_fine) - 1)
    sel203 = logN_lo >= LIM - 1e-9
    # effective <C> over the >=20.3 band = sum(M_full[sel,kz]) / sum(M_unitC[sel,kz])
    Munit = _apply_C_to_M(M_meta, np.ones_like(C_matrix)).reshape(M_full.shape)
    eff_C_203 = np.zeros(n_zb)
    for k in range(n_zb):
        cols = np.where(zfmap == k)[0]
        num = M_full[sel203][:, cols].sum()
        den = Munit[sel203][:, cols].sum()
        eff_C_203[k] = num / den if den > 0 else np.nan
    print(f"\n  effective <C> in the M_b normalizer over the >=20.3 band, per z:")
    for k in range(n_zb):
        print(f"   z={zc[k]:.2f}: M_b <C>(>=20.3) = {eff_C_203[k]:.4f}")
    with open(os.path.join(args.out, "h3_Mb_pathlength.tsv"), "w") as fh:
        fh.write("z_center\tX_tot\tsum_PXz\tratio\tMb_effC_203\n")
        for k, r in enumerate(h3_rows):
            fh.write(f"{r[0]:.3f}\t{r[1]:.6g}\t{r[2]:.6g}\t{r[3]:.6g}\t{eff_C_203[k]:.6g}\n")

    # =====================================================================
    # RANKED SUMMARY
    # =====================================================================
    print("\n" + "=" * 78)
    print("RANKED CONTRIBUTION SUMMARY (DLA tier, >= 20.3)")
    print("=" * 78)
    ez = np.asarray(e0["dndx_z"][LIM], float)
    tz = np.asarray(t0["dndx_z"][LIM], float)
    r0z = np.where(tz > 0, ez / tz, np.nan)
    print(f"  R0(dN/dX>=20.3) total = {base['R0_dndx_total'][LIM]:.4f}")
    print(f"  R0 per z-bin: " + ", ".join(f"z{zc[k]:.2f}={r0z[k]:.3f}" for k in range(n_zb)))
    print(f"  R0(Omega>=20.3) total = {base['R0_omega'][LIM]:.4f}")
    # H4 verdict
    lo_hi = r0z[-1] - r0z[0] if np.all(np.isfinite([r0z[0], r0z[-1]])) else np.nan
    print(f"\n  H4 z-trend: R0(z) spread (hi-lo) = {lo_hi:+.3f}")
    print(f"  H2 net up-migration into >=20.3 (TP) = {100*(up-down)/max(n_true_ge,1):.2f}% "
          f"of truth; NHI bias median = {bias_med:+.3f} dex")
    print(f"  H1 emp_recovery/mean_C = {emp_all/np.mean(C_all):.4f} "
          f"(>1 => 1/C under; <1 => 1/C over)")
    print(f"  H3 sum_PXz/X_tot total = {X_from_M.sum()/X_tot.sum():.5f} "
          f"(should be 1.000; deviation = M_b pathlength bug)")
    print(f"\n  TSVs written to {args.out}")
    return base


if __name__ == "__main__":
    main()
