"""wall1_full_injection.py — WALL-1 FULL-INJECTION reduce orchestrator (reduce-only).

notes/2026-06-17_wall1_full_injection_design.md §5.4. Per tilted-injection arm:

  1. load the arm's RE-INFERRED detection catalog (the GP ran the UNMODIFIED
     production config on the injected loa-0 tree → per-window dlacats);
  2. load the INJECTED truth (the tilted manifest, injected_truth_cat.fits) — this
     IS n_true^tilt the closure compares against (NOT a reweighted natural population);
  3. build the HBI with the FROZEN UNTILTED R_emp response RE-BOUND onto the arm's
     (x̂, SNR) bins (assign_R_emp_to_catalog) + the FROZEN production molly C/ρ;
  4. run baseline_recovery → est/truth dN/dX & Ω + R0 at each report limit, and the
     closure pull (est_inj − truth_inj)/σ vs the injected truth.

The tilt is baked into BOTH the re-inferred detections (from tilted spectra) and the
truth (injected manifest), so the estimator runs UNTILTED (no per-row reweighting —
the supported v3x_refit boot_weights=None path); the "tilt" lives in the data, exactly
as in the real analysis. This is the genuine-re-inference counterpart of the WALL-1
reweighting gate: same FROZEN operator (R_emp), applied to a genuinely re-inferred
tilted population instead of reweighted originals.

DISCIPLINE: reduce-only, cached frozen kernel, NO inference, NO SLURM. Reuses
ab_loa0_fp_baseline / cddf_tilt_closure / cddf_catalog_hbi machinery unchanged;
dla_gp.py / run_bayes_select.py are byte-untouched.
"""
from __future__ import annotations

import argparse
import functools
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.hbi import cddf_catalog_hbi as H
from CDDF_analysis.hbi.cddf_catalog_hbi import (
    HBIConfig, load_molly_matrix, load_and_cut_catalog, build_fine_grid,
    regenerate_molly_counts, make_C_interpolator, build_pathlength,
    make_fp_model, make_rho_interpolator, _build_qso_lookup, v3x_refit,
    _op_mask_and_slots, joint_mc_errors, make_v3x_refit_fn,
)
from CDDF_analysis.hbi.cddf_tilt_closure import baseline_recovery, tilted_truth_reductions
from CDDF_analysis.hbi.run_remp_kernel import compute_R_response, assign_R_emp_to_catalog

# the loa-124 untilted truth that defines the FROZEN R_emp response (design §2/§5.3:
# the response is measured ONCE on the untilted truth-match and NEVER rebuilt from the
# injected mock — that would be the unfair test the PI rules out).
DEF_UNTILTED_CAT = ("/scratch/cavestru_root/cavestru0/mfho/"
                    "gl_prod_2lpt0_v1_20260526/combined_catalog/")
DEF_UNTILTED_TRUTH = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                      "qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits")
DEF_UNTILTED_BAL = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                    "qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits")
# the FROZEN production molly (the C/ρ the headline uses — apples-to-apples with the
# real analysis, which freezes them; design §5.2). Default = the floor-19.5 lya_only
# matrix the WALL-1 config is calibrated on.
DEF_MOLLY = ("/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/"
             "figures_molly_nhi195/lya_only/molly_matrix.tsv")
# the loa-0 zcat/snr for the arm's QSO pathlength + SNR/Z_QSO lookup.
DEF_LOA0 = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
            "qq_desi_y3/v2.8.5/mock-0/loa-0")


def _make_cfg(catalog_dir, truth_path, bal_cat_path, molly_tsv, out_dir, mockdir,
              args):
    return HBIConfig(
        catalog_dir=catalog_dir, truth_path=truth_path, bal_cat_path=bal_cat_path,
        molly_tsv=molly_tsv, out_dir=out_dir, mockdir=mockdir,
        zbins=tuple(float(x) for x in args.zbins.split(",")),
        report_logN_limits=tuple(float(x) for x in args.report_limits.split(",")),
        fp_estimator="purity_mixture", no_bal=True,
        v3_family=args.family, v3_logN_fit_floor=args.fit_floor,
        v3_lambda_bspbody=args.lambda_bspbody, lam_rf_min=args.lam_rf_min,
        v2_z_fit_lo=2.0, v2_z_fit_hi=3.5, v2_z_fit_step=0.1, rng_seed=0,
    )


def build_frozen_R_response(args):
    """Measure the FROZEN untilted R_emp response R[s,jhat,jtru] ONCE (design §5.3).

    Built on the UNTILTED loa-124 truth-match, exactly as the cached r_emp kernel — the
    slope-agnostic, population-frozen operator. Returned as the R_response dict that
    assign_R_emp_to_catalog re-binds onto each arm's re-inferred catalog. NEVER rebuilt
    from a tilted arm.
    """
    cfg_u = _make_cfg(args.untilted_cat, args.untilted_truth, args.untilted_bal,
                      args.molly, "/tmp/wall1_R_response",
                      os.path.dirname(args.untilted_truth), args)
    mm_u = load_molly_matrix(args.molly)
    floor_u = float(mm_u.nhi_edges[0])
    ql_u = _build_qso_lookup(cfg_u)
    cat_u, _truth_u, _isTP_u, gm_u, meta_u = load_and_cut_catalog(
        cfg_u, truth_nhi_floor=floor_u, qso_lookup=ql_u,
        host_truth_floor=min(args.host_truth_floor, floor_u))
    fine_u = build_fine_grid(cfg_u)
    print(f"  [R_emp] untilted response from {os.path.basename(os.path.dirname(args.untilted_cat))} "
          f"(floor {floor_u}, {meta_u.get('n_loaded')} loaded)")
    R_response = compute_R_response(
        cfg_u, cat_u, gm_u, fine_u, mm_u,
        smooth_bins=args.smooth_bins, n_floor=args.n_floor,
        host_col=args.host_col, verbose=True)
    return R_response


def build_arm_ingredients(args, arm_cat_dir, arm_truth, R_response):
    """Build the HBI ingredients for ONE injected arm with the FROZEN R_emp re-bound."""
    cfg = _make_cfg(arm_cat_dir, arm_truth, args.untilted_bal, args.molly,
                    args.out, args.loa0, args)
    mm = load_molly_matrix(args.molly)
    truth_floor = float(mm.nhi_edges[0])
    qso_lookup = _build_qso_lookup(cfg)
    cat_cut, truth_cut, is_TP, good_mask, meta = load_and_cut_catalog(
        cfg, truth_nhi_floor=truth_floor, qso_lookup=qso_lookup,
        host_truth_floor=min(args.host_truth_floor, truth_floor))
    print(f"    arm cat_cut meta: n_loaded={meta.get('n_loaded')}, "
          f"truth rows (injected, >= {truth_floor}) = {len(truth_cut)}")

    # FROZEN R_emp RE-BOUND onto the arm's re-inferred detections (the load-bearing
    # mechanism). The response is the UNTILTED one; only the (x̂, SNR) binning is the
    # arm's. assign_R_emp_to_catalog returns kappa in the EXACT op_base order
    # v3x_build_forward rebuilds, so it drops straight into cfg._posterior_kernel_2d.
    kappa, _ess, _info = assign_R_emp_to_catalog(
        R_response, cfg, cat_cut, good_mask, build_fine_grid(cfg), mm, verbose=False)
    cfg._posterior_kernel_2d = kappa.astype(np.float32)
    n_op = int(_op_mask_and_slots(cat_cut, good_mask, cfg)[0].sum())
    print(f"    frozen R_emp re-bound: kappa {cfg._posterior_kernel_2d.shape} on {n_op} op detections")

    # FROZEN production molly C/ρ (regenerate the counts on the ARM so the matrix bins
    # carry this arm's occupancy, but the C/ρ *shape* is the production matrix — the
    # headline uses the frozen C/ρ, design §5.2).
    mm = regenerate_molly_counts(mm, cat_cut, is_TP, truth_cut, good_mask, cfg)
    C_interp = make_C_interpolator(mm)
    rho_interp = make_rho_interpolator(mm)

    X_tot, n_sl, qzl, qzh, qsn, Xcalc = build_pathlength(
        cfg, qso_lookup=qso_lookup, return_per_sl=True)
    cfg.n_sl_prod = int(n_sl)
    logN_lo, logN_hi, N_b, dN_b = build_fine_grid(cfg)

    s2n = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"], float)
    op_mask = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask
    fp_model, _ = make_fp_model(cfg, cat_cut, op_mask, rho_interp)

    estimator_fn = functools.partial(
        v3x_refit, mm=mm, qso_per_sl=(qzl, qzh, qsn), Xcalc=Xcalc,
        rng=np.random.default_rng(0))
    return dict(cfg=cfg, mm=mm, cat_cut=cat_cut, truth_cut=truth_cut, is_TP=is_TP,
                good_mask=good_mask, C_interp=C_interp, fp_model=fp_model,
                X_tot=X_tot, n_sl=n_sl, logN_lo=logN_lo, logN_hi=logN_hi,
                N_b=N_b, dN_b=dN_b, estimator_fn=estimator_fn, meta=meta)


def reduce_arm(ing, dalpha_label, n_mc=0, mc_seed=0):
    """Run the closure of the re-inferred arm vs the injected truth.

    baseline_recovery runs the UNTILTED v3 estimator (boot_weights=None) on the arm's
    re-inferred detections and compares to the INJECTED truth — the tilt is baked into
    the data, so this IS the closure of the genuine re-inference. Returns R0 + est/truth
    dN/dX & Ω at each limit (the closure read; design §3.4 B).

    σ_MC BAND (flagged pre-verdict TODO). When ``n_mc>0`` we additionally build the
    joint-MC band on the GENUINE re-inference via ``joint_mc_errors`` with
    ``tilt_weights_op=None`` — the tilt is baked into the data, so the band is the
    ordinary sightline-bootstrap (multinomial over QSO TIDs) + C/ρ Wilson + NHI_ERR
    width resample of the re-inferred detections, refit each draw with the SAME v3
    parametric estimator (``make_v3x_refit_fn``, warm-started at the point MAP). This is
    the identical machinery the reweighting WALL-1 gate uses for its σ_MC denominator
    (cddf_tilt_closure.run_one_tilt), only with no per-row reweight because the tilt is
    in the data. Each R0_inj and each closure pull then carry a σ:
        σ(R0)  = std(dndx_est_MC) / dndx_truth   (truth held fixed — it is the manifest)
        pull   = (dndx_est − dndx_truth) / σ(dndx_est_MC)
    The closure pull here is the RAW pull vs the injected manifest (truth_inj IS
    n_true^tilt — drawn directly, not R0·truth^tilt), which is the genuine-re-inference
    counterpart of the reweighting gate's pull.
    """
    cfg = ing["cfg"]
    cfg._wall1_estimator = "v3"
    base = baseline_recovery(
        cfg, ing["cat_cut"], ing["is_TP"], ing["good_mask"], ing["truth_cut"],
        ing["C_interp"], ing["fp_model"], ing["X_tot"],
        ing["logN_lo"], ing["logN_hi"], ing["N_b"], ing["dN_b"],
        estimator_fn=ing["estimator_fn"])
    limits = cfg.report_logN_limits
    out = dict(label=dalpha_label,
               R0_dndx={lim: float(base["R0_dndx_total"][lim]) for lim in limits},
               R0_omega={lim: float(base["R0_omega"][lim]) for lim in limits},
               dndx_est={lim: float(base["e0"]["dndx_total"][lim]) for lim in limits},
               dndx_truth={lim: float(base["t0"]["dndx_total"][lim]) for lim in limits},
               omega_est={lim: float(base["e0"]["omega"][lim]) for lim in limits},
               omega_truth={lim: float(base["t0"]["omega"][lim]) for lim in limits},
               n_sl=int(ing["n_sl"]))

    # σ_MC band on the genuine re-inference (tilt_weights_op=None — tilt is in the data)
    if n_mc and n_mc > 0:
        cfg_saved_nmc = cfg.n_mc
        cfg.n_mc = int(n_mc)
        # parametric per-draw refit closure built from the point MAP (v3 family) — so
        # the band is the parametric re-solve, NOT a v1 1/Vmax fallback.
        refit_fn = make_v3x_refit_fn(cfg, base["e0"]["_v3x"], ing["mm"])
        mc = joint_mc_errors(
            ing["cat_cut"], ing["is_TP"], ing["good_mask"], ing["mm"],
            ing["fp_model"], ing["X_tot"], ing["logN_lo"], ing["logN_hi"],
            ing["N_b"], ing["dN_b"], ing["truth_cut"], cfg,
            np.random.default_rng(mc_seed),
            tilt_weights_op=None, refit_fn=refit_fn)
        cfg.n_mc = cfg_saved_nmc
        dndx_sig, omega_sig = {}, {}
        R0_dndx_sig, R0_omega_sig = {}, {}
        dndx_pull, omega_pull = {}, {}
        for lim in limits:
            sd = float(mc["dndx_total"][lim]["std"])
            so = float(mc["omega"][lim]["std"])
            dndx_sig[lim] = sd
            omega_sig[lim] = so
            td = out["dndx_truth"][lim]
            to = out["omega_truth"][lim]
            R0_dndx_sig[lim] = (sd / td) if td > 0 else float("nan")
            R0_omega_sig[lim] = (so / to) if to > 0 else float("nan")
            # closure pull = (est − injected_truth) / σ_MC  (raw — truth is the manifest)
            dndx_pull[lim] = ((out["dndx_est"][lim] - td) / sd) if sd > 0 else float("nan")
            omega_pull[lim] = ((out["omega_est"][lim] - to) / so) if so > 0 else float("nan")
        out.update(dndx_sig=dndx_sig, omega_sig=omega_sig,
                   R0_dndx_sig=R0_dndx_sig, R0_omega_sig=R0_omega_sig,
                   dndx_pull=dndx_pull, omega_pull=omega_pull, n_mc=int(n_mc))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True,
                   help="injected arm tree root (gen_wall1_inject --out). Reads "
                        "<arm>/gp_out/ (re-inferred dlacats) + <arm>/injected_truth_cat.fits.")
    p.add_argument("--arm-cat-dir", default=None,
                   help="override the arm catalog dir (default <arm>/gp_out/)")
    p.add_argument("--arm-truth", default=None,
                   help="override the injected truth (default <arm>/injected_truth_cat.fits)")
    p.add_argument("--label", default="dalpha", help="arm label for the report")
    p.add_argument("--out", default="/tmp/wall1_full_injection")
    # frozen R_emp provenance (untilted loa-124; design §5.3)
    p.add_argument("--untilted-cat", default=DEF_UNTILTED_CAT)
    p.add_argument("--untilted-truth", default=DEF_UNTILTED_TRUTH)
    p.add_argument("--untilted-bal", default=DEF_UNTILTED_BAL)
    p.add_argument("--molly", default=DEF_MOLLY)
    p.add_argument("--loa0", default=DEF_LOA0, help="loa-0 dir for the arm pathlength/SNR lookup")
    # R_emp build knobs (match the frozen build)
    p.add_argument("--smooth-bins", type=float, default=1.0)
    p.add_argument("--n-floor", type=int, default=20)
    p.add_argument("--host-col", default="NHI_TILT_HOST")
    # v3 fit knobs (match ab_loa0_fp_baseline / the WALL-1 config)
    p.add_argument("--family", default="bspbody")
    p.add_argument("--fit-floor", type=float, default=19.5)
    p.add_argument("--lambda-bspbody", type=float, default=30.0)
    p.add_argument("--lam-rf-min", type=float, default=1025.0)
    p.add_argument("--host-truth-floor", type=float, default=19.0)
    p.add_argument("--zbins", default="2.0,2.5,3.0,3.5")
    p.add_argument("--report-limits", default="20.0,20.3,20.6")
    p.add_argument("--n-mc", type=int, default=0,
                   help="joint-MC draws for the σ_MC band (sightline bootstrap + C/ρ "
                        "Wilson + NHI_ERR width, v3 per-draw refit). 0 = point only.")
    p.add_argument("--mc-seed", type=int, default=0)
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    arm_cat_dir = args.arm_cat_dir or os.path.join(args.arm, "gp_out")
    arm_truth = args.arm_truth or os.path.join(args.arm, "injected_truth_cat.fits")
    for pth, what in ((arm_cat_dir, "arm catalog dir"), (arm_truth, "injected truth")):
        if not os.path.exists(pth):
            raise SystemExit(f"{what} not found: {pth}")

    print("=" * 70)
    print(f"[wall1-full-injection] arm = {args.arm}  ({args.label})")
    print("=" * 70)
    print("[1] build FROZEN untilted R_emp response (loa-124 truth-match)")
    R_response = build_frozen_R_response(args)

    print("[2] build arm ingredients (frozen R_emp re-bound + frozen molly C/ρ)")
    ing = build_arm_ingredients(args, arm_cat_dir, arm_truth, R_response)

    print(f"[3] closure: re-inferred est vs INJECTED truth (n_mc={args.n_mc})")
    res = reduce_arm(ing, args.label, n_mc=args.n_mc, mc_seed=args.mc_seed)
    has_mc = "dndx_sig" in res

    # report
    limits = tuple(float(x) for x in args.report_limits.split(","))
    print("\n" + "=" * 70)
    print(f"  WALL-1 FULL-INJECTION CLOSURE — {args.label} (re-inference vs injected truth)")
    print("  est = v3 bspbody + FROZEN untilted R_emp; truth = injected manifest")
    print("=" * 70)
    if has_mc:
        print(f"  {'limit':>7} | {'dN/dX est':>11} | {'dN/dX tru':>11} | "
              f"{'R0':>6}±{'σ':<6} | {'pull':>6} | {'Ω est':>10} | {'Ω tru':>10} | "
              f"{'R0_Ω':>6}±{'σ':<6} | {'pull_Ω':>7}")
        for lim in limits:
            print(f"  {lim:>7} | {res['dndx_est'][lim]:>11.4e} | {res['dndx_truth'][lim]:>11.4e} "
                  f"| {res['R0_dndx'][lim]:>6.3f}±{res['R0_dndx_sig'][lim]:<6.3f} "
                  f"| {res['dndx_pull'][lim]:>6.2f} | {res['omega_est'][lim]:>10.3e} | "
                  f"{res['omega_truth'][lim]:>10.3e} | {res['R0_omega'][lim]:>6.3f}±"
                  f"{res['R0_omega_sig'][lim]:<6.3f} | {res['omega_pull'][lim]:>7.2f}")
    else:
        print(f"  {'limit':>7} | {'dN/dX est':>11} | {'dN/dX truth':>12} | {'R0':>7} | "
              f"{'Ω est':>11} | {'Ω truth':>11} | {'R0_Ω':>7}")
        for lim in limits:
            print(f"  {lim:>7} | {res['dndx_est'][lim]:>11.5f} | {res['dndx_truth'][lim]:>12.5f} "
                  f"| {res['R0_dndx'][lim]:>7.4f} | {res['omega_est'][lim]:>11.4e} | "
                  f"{res['omega_truth'][lim]:>11.4e} | {res['R0_omega'][lim]:>7.4f}")
    out_tsv = os.path.join(args.out, f"wall1_full_injection_{args.label}.tsv")
    keys = ["R0_dndx", "R0_omega", "dndx_est", "dndx_truth", "omega_est", "omega_truth"]
    if has_mc:
        keys += ["dndx_sig", "omega_sig", "R0_dndx_sig", "R0_omega_sig",
                 "dndx_pull", "omega_pull"]
    with open(out_tsv, "w") as fh:
        fh.write("metric\tlimit\tvalue\n")
        for key in keys:
            for lim in limits:
                fh.write(f"{key}\t{lim}\t{res[key][lim]:.6g}\n")
        if has_mc:
            fh.write(f"n_mc\t-\t{res['n_mc']}\n")
    print(f"\n[wall1-full-injection] saved -> {out_tsv}")
    print("  READ B (is-the-FAIL-real): R0 near the published untilted baseline (~0.89 "
          "at >=20.3 truth-floored) ⇒ frozen-kernel recovers the injected tilt; a large "
          "opposite-sign R0 shift between the +0.5 and −0.5 arms ⇒ "
          "V3_KERNEL_SLOPE_DEPENDENCE reproduced (the FAIL is real).")
    return res


if __name__ == "__main__":
    main()
