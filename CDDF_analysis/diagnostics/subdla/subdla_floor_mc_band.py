"""subdla_floor_mc_band.py — MC error bar on the sub-DLA integrated [19.5,20.3) R0
and per-0.1-dex R0, for floor-19.5 (current) vs floor-19.0 (rebuild).

Reduce-only, cached kernel, NO inference, NO SLURM, NO tilt (Delta_alpha=0, the
UNTILTED baseline recovery, same as subdla_loa0_validation*.py). The point R0 is
reported with NO error bar in the prior docs; this puts the joint-MC band on it.

The MC recipe = the SAME joint variance the estimator's joint_mc_errors uses, but
with the loa0 FP FROZEN (spec §7: a frozen external forest background must not be
resampled with the catalog's purity, and untilted there is no tilt to apply):
  * Wilson/Jeffreys-Beta resample of C and rho per molly cell (_draw_beta_cell)
  * NHI_ERR (sigma_i) width re-draw -> perturbs which fit-floor each detection clears
    and which (N,SNR) cell it lands in (the +Eddington edge scatter, spec §5)
  * bootstrap over sightlines (TID multiplicity -> per-op-row weight)
  * loa0 lam_fp / mu_fp FROZEN at the point value (NOT resampled)
Each draw re-MAPs theta warm-started at the point MAP and reduces (v3x_reduce), so
the band is the parametric-refit band (NOT a v1 1/Vmax fallback).

Run BOTH configs (floor-19.5, floor-19.0), BOTH on the loa0 estimator, n_mc draws
each, and report:
  * integrated [19.5,20.3) R0: point, q16/q50/q84, std, and the z-score for the
    floor-19.5 vs floor-19.0 DIFFERENCE (paired same-truth, so the difference band
    is the per-draw difference distribution).
  * per-0.1-dex R0 bands across [19.5,20.3), so the mid-band drops can be tested.
  * total recovered dN/dX(19.5,20.3) per config (count-conservation / see-saw test).
"""
from __future__ import annotations

import os
import sys
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.hbi import ab_loa0_fp_baseline as AB
from CDDF_analysis.hbi.cddf_tilt_closure import baseline_recovery, tilted_truth_reductions
from CDDF_analysis.hbi.cddf_catalog_hbi import (
    _draw_beta_cell, _cell_index, _slice_active_unitC, _rescale_unitC_active,
    _apply_C_to_M, v3x_fit_map, v3x_reduce, C_FLOOR,
)

# ---- config knobs (the three repointed for floor-19.0) ----------------------
F195 = dict(
    name="floor19.5",
    kernel=AB.DEF_KERNEL,                       # mollynhi195_lyaonly1025_broaden012
    molly=AB.DEF_LYAONLY_MOLLY,                 # figures_molly_nhi195/lya_only
    fit_floor=19.5,
)
F190 = dict(
    name="floor19.0",
    kernel=("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
            "phase3d_experiments/floor190_lyaonly1025_broaden012/"
            "posterior_kernel_2lpt0.npz"),
    molly=("/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/"
           "figures_molly_nhi190/molly_matrix.tsv"),
    fit_floor=19.0,
)
LOA0_PRODUCT = ("/scratch/cavestru_root/cavestru0/mfho/gl_loa0_fp_v1_20260615/"
                "outputs/loa0_fp_product_lyaonly1025.npz")
REPORT_LIMITS = (19.5, 19.6, 19.7, 19.8, 19.9, 20.0, 20.1, 20.2, 20.3, 20.6)
PER_BINS = [(round(19.5 + 0.1 * k, 1), round(19.6 + 0.1 * k, 1)) for k in range(8)]


class _Args:
    def __init__(self, knobs):
        self.catalog_dir = AB.DEF_CAT
        self.truth = AB.DEF_TRUTH
        self.bal_cat = AB.DEF_BAL
        self.molly_tsv = knobs["molly"]
        self.kernel = knobs["kernel"]
        self.loa0_product = LOA0_PRODUCT
        self.out = "/tmp/subdla_floor_mc_band_" + knobs["name"]
        self.mockdir = None
        self.zbins = "2.0,2.5,3.0,3.5"
        self.report_limits = ",".join(f"{x:g}" for x in REPORT_LIMITS)
        self.family = "bspbody"
        self.fit_floor = knobs["fit_floor"]
        self.fit_ceil = 99.0
        self.lambda_bspbody = 30.0
        self.lam_rf_min = 1025.0
        self.edge_slope_lam = 40.0
        self.gl_nodes = 1
        self.host_truth_floor = 19.0


def _band_dndx(red_dndx_total, lo=19.5, hi=20.3):
    """integrated dN/dX over [lo,hi) from the cumulative dndx_total dict."""
    return red_dndx_total[lo] - red_dndx_total[hi]


def run_one(knobs, n_mc, seed):
    args = _Args(knobs)
    os.makedirs(args.out, exist_ok=True)
    print("=" * 78)
    print(f"[MC band] {knobs['name']}  fit_floor={knobs['fit_floor']}  n_mc={n_mc}")
    print("=" * 78)
    ing = AB.build_ingredients(args, "loa0", loa0_product=args.loa0_product)
    cfg = ing["cfg"]
    cfg._wall1_estimator = "v3"
    cfg.v3_mc_n_restart = 2  # warm-started per draw; cheap

    # ---- point estimate (loa0, untilted) ----
    base = baseline_recovery(
        cfg, ing["cat_cut"], ing["is_TP"], ing["good_mask"], ing["truth_cut"],
        ing["C_interp"], ing["fp_model"], ing["X_tot"],
        ing["logN_lo"], ing["logN_hi"], ing["N_b"], ing["dN_b"],
        estimator_fn=ing["estimator_fn"])
    e0 = base["e0"]; t0 = base["t0"]
    logN_lo = np.asarray(ing["logN_lo"], float)
    logN_hi = np.asarray(ing["logN_hi"], float)
    dN_b = np.asarray(ing["dN_b"], float)

    # truth (FIXED across draws) — per-bin dN/dX truth and integrated
    f_tru = np.asarray(t0["f_truth"], float)
    dndx_tru_bin = np.array([
        np.nansum(f_tru[(logN_lo >= blo - 1e-6) & (logN_hi <= bhi + 1e-6)]
                  * dN_b[(logN_lo >= blo - 1e-6) & (logN_hi <= bhi + 1e-6)])
        for blo, bhi in PER_BINS])
    dndx_tru_band = _band_dndx(t0["dndx_total"])

    # point per-bin / integrated dN/dX (est) and R0
    f_est0 = np.asarray(e0["f_b"], float)
    dndx_est0_bin = np.array([
        np.nansum(f_est0[(logN_lo >= blo - 1e-6) & (logN_hi <= bhi + 1e-6)]
                  * dN_b[(logN_lo >= blo - 1e-6) & (logN_hi <= bhi + 1e-6)])
        for blo, bhi in PER_BINS])
    dndx_est0_band = _band_dndx(e0["dndx_total"])
    r0_band_point = dndx_est0_band / dndx_tru_band
    r0_bin_point = dndx_est0_bin / dndx_tru_bin
    r0_203_point = base["R0_dndx_total"][20.3]

    # ---- MC: reuse the point fwd; resample C/rho/sigma + bootstrap; FREEZE loa0 FP
    fwd = e0["_v3x"]["fwd"]
    family = e0["_v3x"]["family"]
    fine = e0["_v3x"]["fine"]
    M_meta = e0["_v3x"]["M_meta"]
    theta_map = e0["_v3x"]["theta_map"]
    A_meta = fwd["A_meta"]; cat_op = fwd["cat_op"]
    lam_fp_frozen = fwd["lam_fp"]; mu_fp_frozen = fwd["mu_fp"]   # loa0 FROZEN
    active_flat = fwd["active_flat"]
    keep_in_base = fwd["keep_in_base"]
    snr_op = cat_op["snr"]; i_snr0 = cat_op["i_snr"]
    z_edges_fine = fine[4]
    n_flat = len(logN_lo) * (len(z_edges_fine) - 1)
    unitC = _slice_active_unitC(A_meta, np.arange(n_flat), np.ones(A_meta["n_obs"], bool))

    mm = ing["mm"]
    cat_cut = ing["cat_cut"]; good_mask = ing["good_mask"]
    s2n_all = np.asarray(cat_cut["S2N_RED"], float)
    pdla_all = np.asarray(cat_cut["P_DLA"], float)
    op_base = (s2n_all > cfg.snr_min) & (pdla_all > cfg.p_dla_min) & good_mask
    nhi0_base = np.asarray(cat_cut["NHI"], float)[op_base]
    nhi_err_base = np.asarray(cat_cut["NHI_ERR"], float)[op_base]
    nhi_err_base = np.where(np.isfinite(nhi_err_base) & (nhi_err_base > 0), nhi_err_base, 0.0)
    tids_base = np.asarray(cat_cut["TARGETID"], np.int64)[op_base]
    # bootstrap over the FULL op_base sightlines (then slice to floored subset, exactly
    # as joint_mc_errors/make_v3x_refit_fn: boot_w op_base-ordered -> [keep_in_base])
    uniq_tids, inv = np.unique(tids_base, return_inverse=True)
    n_uniq = len(uniq_tids)

    rng = np.random.default_rng(seed)
    mc_band = np.full(n_mc, np.nan)        # integrated dN/dX [19.5,20.3) (est)
    mc_bin = np.full((n_mc, len(PER_BINS)), np.nan)  # per-bin dN/dX (est)
    mc_r0_203 = np.full(n_mc, np.nan)

    for m in range(n_mc):
        C_draw = _draw_beta_cell(rng, mm.cmp_nfound, mm.cmp_nfid)
        rho_draw = _draw_beta_cell(rng, mm.pur_ntp, mm.pur_ntot)
        C_draw = np.where((mm.cmp_nfid > 0), C_draw, C_FLOOR)
        rho_draw = np.where((mm.pur_ntot > 0), rho_draw, 0.0)
        nhi_m_base = nhi0_base + rng.normal(0.0, 1.0, len(nhi0_base)) * nhi_err_base
        mult = rng.multinomial(n_uniq, np.full(n_uniq, 1.0 / n_uniq))
        boot_w_base = mult[inv].astype(float)
        # slice to floored op subset
        boot_w = boot_w_base[keep_in_base]
        nhi_m = nhi_m_base[keep_in_base]
        # C-rescale A/M (same as make_v3x_refit_fn) ; FREEZE loa0 lam_fp/mu_fp
        A_draw = _rescale_unitC_active(unitC, C_draw)
        M_draw = np.where(active_flat, _apply_C_to_M(M_meta, C_draw), 0.0)
        fit = v3x_fit_map(A_draw, M_draw, lam_fp_frozen, mu_fp_frozen, fine, family, cfg,
                          obj_weights=boot_w, theta0=theta_map, n_restart=2,
                          rng=np.random.default_rng(seed * 100003 + m), lit_start=False)
        rr = v3x_reduce(cfg, fit["theta_map"], fine, family, M_meta)
        f_b = np.asarray(rr["f_b"], float)
        mc_band[m] = _band_dndx(rr["dndx_total"])
        for bi, (blo, bhi) in enumerate(PER_BINS):
            sel = (logN_lo >= blo - 1e-6) & (logN_hi <= bhi + 1e-6)
            mc_bin[m, bi] = np.nansum(f_b[sel] * dN_b[sel])
        mc_r0_203[m] = rr["dndx_total"][20.3] / t0["dndx_total"][20.3]
        if (m + 1) % 25 == 0:
            print(f"    draw {m+1}/{n_mc}")

    return dict(
        name=knobs["name"], n_sl=int(ing["n_sl"]),
        dndx_tru_band=float(dndx_tru_band), dndx_tru_bin=dndx_tru_bin,
        dndx_est0_band=float(dndx_est0_band), dndx_est0_bin=dndx_est0_bin,
        r0_band_point=float(r0_band_point), r0_bin_point=r0_bin_point,
        r0_203_point=float(r0_203_point),
        mc_band=mc_band, mc_bin=mc_bin, mc_r0_203=mc_r0_203,
    )


def main():
    n_mc = int(os.environ.get("N_MC", "150"))
    seed = int(os.environ.get("SEED", "0"))
    out = {}
    for knobs in (F195, F190):
        out[knobs["name"]] = run_one(knobs, n_mc, seed)

    r195 = out["floor19.5"]; r190 = out["floor19.0"]
    tru_band = r195["dndx_tru_band"]
    tru_bin = r195["dndx_tru_bin"]

    def _q(a):
        a = a[np.isfinite(a)]
        return (np.nanmean(a), np.nanstd(a),
                np.nanpercentile(a, 16), np.nanpercentile(a, 50), np.nanpercentile(a, 84))

    print("\n" + "#" * 78)
    print("# RESULT 1 — integrated [19.5,20.3) R0 with MC band")
    print("#" * 78)
    for r in (r195, r190):
        mc_r0 = r["mc_band"] / tru_band
        mu, sd, q16, q50, q84 = _q(mc_r0)
        print(f"\n{r['name']}: point R0 = {r['r0_band_point']:.4f}")
        print(f"   MC: mean={mu:.4f} std={sd:.4f} q16={q16:.4f} q50={q50:.4f} q84={q84:.4f}")
        print(f"   truth dN/dX[19.5,20.3) = {tru_band:.6g}; "
              f"point est dN/dX = {r['dndx_est0_band']:.6g}")

    # paired difference (same truth, INDEPENDENT draws -> conservative; also report the
    # naive quadrature combine). The cleaner test: is |R0_195 - R0_190| > combined sigma?
    mc_r0_195 = r195["mc_band"] / tru_band
    mc_r0_190 = r190["mc_band"] / tru_band
    n = min(len(mc_r0_195), len(mc_r0_190))
    # paired per-draw difference (shared C/rho/bootstrap draws via shared seed) — the
    # two configs use DIFFERENT kernels/molly so draws are not literally the same cells,
    # but the same RNG seed makes the bootstrap multiplicities correlated. Report the
    # per-draw difference band AND the independent-quadrature band.
    diff_paired = mc_r0_195[:n] - mc_r0_190[:n]
    dpm, dps = np.nanmean(diff_paired), np.nanstd(diff_paired)
    s195 = np.nanstd(mc_r0_195); s190 = np.nanstd(mc_r0_190)
    s_quad = np.hypot(s195, s190)
    dpoint = r195["r0_band_point"] - r190["r0_band_point"]
    print("\n--- DIFFERENCE: floor-19.5 minus floor-19.0 (integrated [19.5,20.3) R0) ---")
    print(f"   point diff           = {dpoint:+.4f}")
    print(f"   per-config sigma      : 19.5 {s195:.4f}, 19.0 {s190:.4f}")
    print(f"   independent-quadrature sigma = {s_quad:.4f}  ->  z = {dpoint / s_quad:.2f}")
    print(f"   paired-draw diff      : mean={dpm:+.4f} std={dps:.4f}  ->  z = {dpoint / dps:.2f}")

    print("\n" + "#" * 78)
    print("# RESULT 2 — per-0.1-dex R0 with MC band (floor-19.5 vs floor-19.0)")
    print("#" * 78)
    print(f"{'bin':>14} | {'truth dndx':>11} | {'f19.5 R0 (q16,q84)':>26} | "
          f"{'f19.0 R0 (q16,q84)':>26} | {'z(diff)':>8}")
    print("-" * 116)
    for bi, (blo, bhi) in enumerate(PER_BINS):
        tb = tru_bin[bi]
        r0_195 = r195["mc_bin"][:, bi] / tb
        r0_190 = r190["mc_bin"][:, bi] / tb
        p195 = r195["r0_bin_point"][bi]; p190 = r190["r0_bin_point"][bi]
        s1 = np.nanstd(r0_195); s2 = np.nanstd(r0_190)
        sq = np.hypot(s1, s2)
        z = (p195 - p190) / sq if sq > 0 else np.nan
        lab = f"[{blo:.1f},{bhi:.1f})"
        print(f"{lab:>14} | {tb:>11.5g} | "
              f"{p195:>6.3f} ({np.nanpercentile(r0_195,16):.3f},{np.nanpercentile(r0_195,84):.3f}) | "
              f"{p190:>6.3f} ({np.nanpercentile(r0_190,16):.3f},{np.nanpercentile(r0_190,84):.3f}) | "
              f"{z:>+8.2f}")

    print("\n" + "#" * 78)
    print("# RESULT 3 — count conservation / see-saw: total recovered dN/dX [19.5,20.3)")
    print("#" * 78)
    print(f"   truth total dN/dX[19.5,20.3)      = {tru_band:.6g}")
    print(f"   floor-19.5 recovered dN/dX (point) = {r195['dndx_est0_band']:.6g}  "
          f"(R0 {r195['r0_band_point']:.4f})")
    print(f"   floor-19.0 recovered dN/dX (point) = {r190['dndx_est0_band']:.6g}  "
          f"(R0 {r190['r0_band_point']:.4f})")
    print(f"   delta total recovered (19.0 - 19.5) = "
          f"{r190['dndx_est0_band'] - r195['dndx_est0_band']:+.6g}")
    print("\n   per-bin recovered dN/dX (point), and per-bin redistribution:")
    print(f"{'bin':>14} | {'truth':>11} | {'f19.5 est':>11} | {'f19.0 est':>11} | {'delta(19.0-19.5)':>17}")
    print("-" * 78)
    tot195 = tot190 = tott = 0.0
    for bi, (blo, bhi) in enumerate(PER_BINS):
        e195 = r195["dndx_est0_bin"][bi]; e190 = r190["dndx_est0_bin"][bi]
        tb = tru_bin[bi]
        tot195 += e195; tot190 += e190; tott += tb
        lab = f"[{blo:.1f},{bhi:.1f})"
        print(f"{lab:>14} | {tb:>11.5g} | {e195:>11.5g} | {e190:>11.5g} | {e190 - e195:>+17.5g}")
    print("-" * 78)
    print(f"{'SUM[19.5,20.3)':>14} | {tott:>11.5g} | {tot195:>11.5g} | {tot190:>11.5g} | "
          f"{tot190 - tot195:>+17.5g}")
    print(f"\n   fractional change in TOTAL recovered counts (19.0 vs 19.5) = "
          f"{(tot190 - tot195) / tot195 * 100:+.1f}%")

    print("\n" + "#" * 78)
    print("# RESULT 4 — DLA-tier (>=20.3) R0 MC band (should stay ~1.16, error-bar size)")
    print("#" * 78)
    for r in (r195, r190):
        a = r["mc_r0_203"][np.isfinite(r["mc_r0_203"])]
        print(f"   {r['name']}: point R0(>=20.3) = {r['r0_203_point']:.4f}  "
              f"MC mean={np.nanmean(a):.4f} std={np.nanstd(a):.4f} "
              f"q16={np.nanpercentile(a,16):.4f} q84={np.nanpercentile(a,84):.4f}")

    # persist
    np.savez("/tmp/subdla_floor_mc_band_result.npz",
             tru_band=tru_band, tru_bin=tru_bin,
             f195_mc_band=r195["mc_band"], f190_mc_band=r190["mc_band"],
             f195_mc_bin=r195["mc_bin"], f190_mc_bin=r190["mc_bin"],
             f195_r0_band_point=r195["r0_band_point"],
             f190_r0_band_point=r190["r0_band_point"],
             f195_dndx_est0_bin=r195["dndx_est0_bin"],
             f190_dndx_est0_bin=r190["dndx_est0_bin"])
    print("\n[saved] /tmp/subdla_floor_mc_band_result.npz")
    return out


if __name__ == "__main__":
    main()
