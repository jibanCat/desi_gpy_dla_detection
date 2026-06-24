"""wall1_explain_partA.py — recovered CDDF / dN/dX / Ω with the FULL population
posterior (credible bands) for the explanatory doc (reduce-only, cached kernel,
NO inference, NO SLURM, NO tilt).

The "current HBI" config (calibrated, corrected loa-0 FP): kernel = the cached
broaden012 2-D posterior kernel; molly = lya_only nhi195; FP = loa0; v3 bspbody,
fit floor 19.5, lam_rf_min 1025, sigma_add 0.12.

Posterior bands (state clearly which components each includes):
  * THETA-Laplace  : Gaussian posterior N(theta_map, H^-1) at the MAP, draws
                     reduced through v3x_reduce. NUISANCE FROZEN (C/rho/kernel/FP
                     at point). The pure POPULATION-theta band.
  * THETA-emcee    : same likelihood, full MCMC on theta (banana-safe cross-check).
  * FULL (t+nuis)  : the joint-MC band — resamples C/rho (Wilson/Jeffreys-Beta),
                     per-object NHI width (sigma_i), the loa-0 FP Gamma (Gehrels
                     +1/2), and bootstraps sightlines, re-MAPping theta each draw.
                     This is the population-theta (+) nuisance posterior the doc
                     reports as the headline band.
  * FULL (PM xref) : the WIRED purity_mixture joint_mc_errors band (cross-check;
                     same nuisance set but the per-row purity FP).

Builds ingredients ONCE via ab_loa0_fp_baseline.build_ingredients (the exact
WALL-1 calibrated bundle), reuses the cddf_catalog_hbi posterior machinery, and
writes an npz the figure driver consumes.
"""
from __future__ import annotations

import argparse
import functools
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.hbi import cddf_catalog_hbi as H
from CDDF_analysis.hbi.cddf_catalog_hbi import (
    v3x_refit, v3x_fit_map, v3x_laplace, v3x_emcee_check, v3x_reduce,
    v3x_mc_inner_theta,
    truth_reductions, joint_mc_errors, omega_hi_prefactor,
    _draw_beta_cell, _rescale_unitC_active, _apply_C_to_M, _cell_index,
    _slice_active_unitC, C_FLOOR, _forward_fp_terms, make_rho_interpolator,
    build_truth_match_resample, draw_shared_boot, draw_shared_boot_with_mult,
    v3x_response_setup, v3x_response_rebuild_unitC, draw_response_params,
    v3x_stage3_setup, v3x_stage3_rebuild_unitC,
)
from CDDF_analysis.hbi.znz_kernel import refit_znz_from_resample
from CDDF_analysis.hbi.ab_loa0_fp_baseline import build_ingredients, _resolve_molly


# -----------------------------------------------------------------------------
# loa0-aware full-posterior joint-MC (mirrors v3x_joint_mc EXACTLY but threads the
# FROZEN loa-0 FP via _forward_fp_terms, resampled per-draw Gehrels Gamma). The
# wired v3x_joint_mc / joint_mc_errors refit_fn hardcode lam_fp=(1-rho)*boot_w
# (purity-mixture), so they cannot carry the loa-0 background. Identical draw
# structure (C/rho Wilson, sigma_i width, sightline bootstrap) + the loa-0 Gamma.
# -----------------------------------------------------------------------------
def loa0_full_posterior_mc(cfg, ing, point, n_mc, rng):
    mm = ing["mm"]; cat_cut = ing["cat_cut"]; family = point["_v3x"]["family"]
    fwd = point["_v3x"]["fwd"]; theta_map = point["_v3x"]["theta_map"]
    A_meta = fwd["A_meta"]; M_meta = fwd["M_meta"]; cat_op = fwd["cat_op"]
    fine = fwd["fine"]
    logN_lo, logN_hi, N_b, dN_b, z_edges_fine = fine
    n_flat = len(logN_lo) * (len(z_edges_fine) - 1)
    unitC = _slice_active_unitC(A_meta, np.arange(n_flat),
                                np.ones(A_meta["n_obs"], bool))
    xhat = cat_op["xhat"]; snr_op = cat_op["snr"]; i_snr0 = cat_op["i_snr"]
    active_flat = fwd["active_flat"]
    op = fwd["op_mask"]
    nhi_err_op = np.asarray(cat_cut["NHI_ERR"], float)[op]
    nhi_err_op = np.where(np.isfinite(nhi_err_op) & (nhi_err_op > 0), nhi_err_op, 0.0)
    tids_op = np.asarray(cat_cut["TARGETID"], np.int64)[op]
    uniq, inv = np.unique(tids_op, return_inverse=True)
    n_uniq = len(uniq)
    rho_interp = make_rho_interpolator(mm)
    loa0_fp = getattr(cfg, "_loa0_fp", None)
    floor = fwd["logN_fit_floor"]
    limits = cfg.report_logN_limits
    n_zc = len(np.asarray(cfg.zbins, float)) - 1

    # Stage II: shared truth-match (D_t) resample so C, ρ, boot_w are CORRELATED.
    # tmr.op_tid_idx is in op_BASE order (the full purity population, no fit floor); the
    # loa0 path works on the FLOORED op set (op_full = op_base & nhi>=fit_floor), so the
    # shared boot_w is sliced to the floored subset via fwd["keep_in_base"] — exactly the
    # slice the legacy [inv] index implicitly applies (tids_op already = floored op).
    mc_nuisance = getattr(cfg, "mc_nuisance", "indep")
    tmr = None
    keep_in_base = fwd["keep_in_base"]
    if mc_nuisance == "shared_boot":
        tmr = build_truth_match_resample(
            mm, cat_cut, ing["is_TP"], ing["truth_cut"], ing["good_mask"], cfg)

    # Stage III: response (θ_K) marginalization. Per draw RE-FIT the kernel correction on
    # the SAME shared resample (boot_mult) + DRAW the response-FORM mix q, RE-APPLY the
    # transform to the BASE kernel, REBUILD A. The DOMINANT coverage lever. Requires the
    # response transform (cfg.kernel_znz_model) AND the shared resample (mc_nuisance).
    mc_response = getattr(cfg, "mc_response", "frozen")
    stage3_kind = None
    sctx = None
    if mc_response == "marginalize":
        if mc_nuisance != "shared_boot":
            raise ValueError("mc_response='marginalize' requires mc_nuisance="
                             "'shared_boot' (the shared boot_mult the kernel re-uses).")
        # dispatch on resp_kind: forward (T-D, re-fit the ForwardResponseModel) vs znz
        stage3_kind, sctx = v3x_stage3_setup(cfg, cat_cut, ing["good_mask"], mm, fwd, tmr)
        if sctx is None:
            raise ValueError(
                "mc_response='marginalize' requires a response model to marginalize — "
                "cfg.kernel_forward_model (resp_kind='forward') or cfg.kernel_znz_model "
                "(resp_kind='kappa'). Nothing to marginalize.")

    f_bs = []; thetas = []
    dndx = {l: [] for l in limits}; omega = {l: [] for l in limits}
    dndx_z = {l: [] for l in limits}
    seeds = rng.integers(0, 2**31 - 1, size=n_mc)
    for s in seeds:
        rg = np.random.default_rng(int(s))
        boot_mult = None
        if mc_nuisance == "shared_boot":
            # ONE shared TID-blocked resample -> jointly-correlated (C, ρ, boot_w_base);
            # slice the op_base-order boot_w to the floored op set this path uses. Stage
            # III also re-weights θ_K by the SAME boot_mult (keep it via *_with_mult).
            if mc_response == "marginalize":
                C_draw, rho_draw, boot_w_base, boot_mult = \
                    draw_shared_boot_with_mult(rg, tmr)
            else:
                C_draw, rho_draw, boot_w_base = draw_shared_boot(rg, tmr)
            boot_w = boot_w_base[keep_in_base]
            nhi_m = xhat + rg.normal(0, 1, len(xhat)) * nhi_err_op
        else:
            C_draw = _draw_beta_cell(rg, mm.cmp_nfound, mm.cmp_nfid)
            rho_draw = _draw_beta_cell(rg, mm.pur_ntp, mm.pur_ntot)
            C_draw = np.where(mm.cmp_nfid > 0, C_draw, C_FLOOR)
            rho_draw = np.where(mm.pur_ntot > 0, rho_draw, 0.0)
            nhi_m = xhat + rg.normal(0, 1, len(xhat)) * nhi_err_op
            boot_w = rg.multinomial(n_uniq, np.full(n_uniq, 1.0 / n_uniq)).astype(float)[inv]
        # Stage III: per-draw response θ_K -> per-draw unitC (REBUILD A). 'frozen' (default)
        # reuses the ONE frozen unitC (byte-identical). 'marginalize' re-fits θ_K on this
        # draw's boot_mult + the drawn FORM-mix q, re-applies the transform, rebuilds A.
        unitC_draw = unitC
        if mc_response == "marginalize":
            unitC_draw = v3x_stage3_rebuild_unitC(cfg, stage3_kind, sctx, rg, boot_mult)
        A_draw = _rescale_unitC_active(unitC_draw, C_draw)
        M_draw = np.where(active_flat, _apply_C_to_M(M_meta, C_draw), 0.0)
        # FROZEN loa-0 FP, resampled (Gehrels Gamma) — NOT bootstrap/tilt scaled
        loa0_d = loa0_fp.resample(rg)
        lam_fp, mu_fp = _forward_fp_terms(
            cfg, rho_interp, nhi_m, snr_op, obj_weights_extra=None,
            loa0_fp=loa0_d, logN_fit_floor=floor)
        fit = v3x_fit_map(A_draw, M_draw, lam_fp, mu_fp, fine, family, cfg,
                          obj_weights=boot_w, theta0=theta_map, n_restart=2, rng=rg,
                          lit_start=False)
        # Stage I: 'map' (default) => fit["theta_map"] (byte-identical);
        #          'laplace' => one N(θ̂, H⁻¹) draw at THIS draw's ψ (within-ψ width).
        theta_inner = v3x_mc_inner_theta(cfg, fit, A_draw, M_draw, lam_fp, mu_fp,
                                         fine, family, boot_w, rg)
        rr = v3x_reduce(cfg, theta_inner, fine, family, M_meta)
        f_bs.append(rr["f_b"]); thetas.append(theta_inner)
        for l in limits:
            dndx[l].append(rr["dndx_total"][l]); omega[l].append(rr["omega"][l])
            dndx_z[l].append(rr["dndx_z"][l])
    f_bs = np.array(f_bs); thetas = np.array(thetas)
    out = dict(f_b_samples=f_bs, theta_samples=thetas, n_mc=int(n_mc))
    for l in limits:
        out[f"dndx_{l}_samples"] = np.array(dndx[l])
        out[f"omega_{l}_samples"] = np.array(omega[l])
        out[f"dndx_z_{l}_samples"] = np.array(dndx_z[l])   # (n_mc, n_zbins)
    return out


def theta_band_reduce(cfg, point, draws):
    """Reduce a stack of theta draws (n_draw, n_param) through v3x_reduce -> posterior
    samples of f_b, dN/dX, Omega, dN/dX(z). Nuisance FROZEN (C/rho/kernel/FP at point)."""
    family = point["_v3x"]["family"]; fine = point["_v3x"]["fine"]
    M_meta = point["_v3x"]["M_meta"]
    limits = cfg.report_logN_limits
    f_bs = []; dndx = {l: [] for l in limits}; omega = {l: [] for l in limits}
    dndx_z = {l: [] for l in limits}
    for th in draws:
        rr = v3x_reduce(cfg, th, fine, family, M_meta)
        f_bs.append(rr["f_b"])
        for l in limits:
            dndx[l].append(rr["dndx_total"][l]); omega[l].append(rr["omega"][l])
            dndx_z[l].append(rr["dndx_z"][l])
    out = dict(f_b_samples=np.array(f_bs))
    for l in limits:
        out[f"dndx_{l}_samples"] = np.array(dndx[l])
        out[f"omega_{l}_samples"] = np.array(omega[l])
        out[f"dndx_z_{l}_samples"] = np.array(dndx_z[l])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog-dir",
                   default=("/scratch/cavestru_root/cavestru0/mfho/"
                            "gl_prod_2lpt0_v1_20260526/combined_catalog/"))
    p.add_argument("--truth",
                   default=("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                            "qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits"))
    p.add_argument("--bal-cat",
                   default=("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                            "qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits"))
    p.add_argument("--molly-tsv", default=None)
    p.add_argument("--kernel",
                   default=("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                            "phase3d_experiments/mollynhi195_lyaonly1025_broaden012/"
                            "posterior_kernel_2lpt0.npz"))
    p.add_argument("--loa0-product",
                   default=("/scratch/cavestru_root/cavestru0/mfho/gl_loa0_fp_v1_20260615/"
                            "outputs/loa0_fp_product_lyaonly1025.npz"))
    p.add_argument("--out",
                   default="/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                           "wall1_explain_partA")
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
    p.add_argument("--n-lap", type=int, default=2000, help="Laplace theta draws")
    p.add_argument("--n-mc", type=int, default=300, help="full-posterior joint-MC draws")
    p.add_argument("--mc-inner", choices=["map", "laplace"], default="map",
                   help="Stage I: inner-θ per MC draw — 'map' (MODE, byte-identical "
                        "default) or 'laplace' (one N(θ̂,H⁻¹) sample, the faithful "
                        "marginalized band that folds in the within-ψ population-fit width).")
    p.add_argument("--mc-nuisance", choices=["indep", "shared_boot"], default="indep",
                   help="Stage II: calibration-nuisance draw — 'indep' (independent "
                        "per-cell Jeffreys-Betas for C/ρ + separate detection bootstrap, "
                        "byte-identical default) or 'shared_boot' (ONE shared TID-blocked "
                        "resample of the truth-match D_t per draw, re-deriving C, ρ, boot_w "
                        "JOINTLY so the C–ρ correlation is restored, not double-counted).")
    p.add_argument("--mc-response", choices=["frozen", "marginalize"], default="frozen",
                   help="Stage III: response (θ_K) treatment — 'frozen' (the response is "
                        "held at the cached point functional; A built once; byte-identical "
                        "default = the broaden012 headline when --kernel-znz is unset) or "
                        "'marginalize' (per draw re-fit θ_K on the shared resample + draw "
                        "the response-FORM mix q, re-apply the transform, REBUILD A — the "
                        "dominant coverage lever; requires --kernel-znz + --mc-nuisance "
                        "shared_boot).")
    p.add_argument("--kernel-znz", default=None,
                   help="path to the znz NPZ (znz_kernel.save_znz). When set, the (N,z) "
                        "response correction is APPLIED (Track-C); REQUIRED for "
                        "--mc-response marginalize. When unset the kernel is the cached "
                        "broaden012 (frozen-response headline).")
    p.add_argument("--q-lo", type=float, default=0.0,
                   help="Stage III response-FORM mix prior lower edge (q=0 ⇒ median).")
    p.add_argument("--q-hi", type=float, default=1.0,
                   help="Stage III response-FORM mix prior upper edge (q=1 ⇒ mean).")
    p.add_argument("--alpha-lo", type=float, default=1.0,
                   help="Stage III response-STRENGTH prior lower edge (α=0 ⇒ OFF / "
                        "un-corrected; α=1 ⇒ full correction). DEFAULT 1.0 = Step-1 "
                        "(parameter+form-mix only). Set <1 (e.g. 0.0) for Step-2 (the "
                        "truth-bracketing OFF↔corrected form marginalization).")
    p.add_argument("--alpha-hi", type=float, default=1.0,
                   help="Stage III response-STRENGTH prior upper edge (default 1.0).")
    p.add_argument("--n-emcee-steps", type=int, default=1500)
    p.add_argument("--skip-emcee", action="store_true")
    p.add_argument("--skip-pm-xref", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    limits = tuple(float(x) for x in args.report_limits.split(","))
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    print("=" * 70)
    print("[partA] build loa0 ingredients (calibrated WALL-1 bundle, kernel ON)")
    print("=" * 70)
    ing = build_ingredients(args, "loa0", loa0_product=args.loa0_product)
    cfg = ing["cfg"]
    cfg.report_logN_limits = limits
    cfg._wall1_estimator = "v3"
    cfg.v3_n_lap = args.n_lap
    cfg.v3_n_emcee_steps = args.n_emcee_steps
    cfg.mc_inner = args.mc_inner   # Stage I: 'map' (default) | 'laplace' (faithful band)
    cfg.mc_nuisance = args.mc_nuisance  # Stage II: 'indep' (default) | 'shared_boot'
    cfg.mc_response = args.mc_response  # Stage III: 'frozen' (default) | 'marginalize'
    cfg.mc_response_q_lo = args.q_lo
    cfg.mc_response_q_hi = args.q_hi
    cfg.mc_response_alpha_lo = args.alpha_lo
    cfg.mc_response_alpha_hi = args.alpha_hi
    # Stage III: the response transform must be ON (the cached znz) to be marginalizable.
    # When --kernel-znz is set, A is built with apply_znz_correction (Track-C corrected);
    # the marginalized POINT is then the response-corrected estimate (NOT broaden012).
    if args.kernel_znz is not None:
        cfg.kernel_znz_model = args.kernel_znz
    if args.mc_response == "marginalize" and getattr(cfg, "kernel_znz_model", None) is None:
        raise SystemExit("--mc-response marginalize requires --kernel-znz "
                         "(the response transform to marginalize).")
    logN_lo = ing["logN_lo"]; logN_hi = ing["logN_hi"]
    N_b = ing["N_b"]; dN_b = ing["dN_b"]; X_tot = ing["X_tot"]
    print(f"    n_sl_prod={ing['n_sl']}, X_tot={X_tot}  ({time.time()-t0:.0f}s)")

    # ---- POINT MAP (untilted; boot_weights=None — loa0 is allowed) ----
    # ing["estimator_fn"] = functools.partial(v3x_refit, mm=..., qso_per_sl=...,
    # Xcalc=..., rng=...) — the calibrated WALL-1 estimator with the cached kernel.
    print("[partA] v3x point MAP (loa0 FP, kernel ON)")
    point = ing["estimator_fn"](
        ing["cat_cut"], ing["is_TP"], ing["good_mask"], ing["C_interp"],
        ing["fp_model"], X_tot, logN_lo, logN_hi, N_b, dN_b, ing["truth_cut"], cfg)
    theta_map = point["_v3x"]["theta_map"]
    fwd = point["_v3x"]["fwd"]
    print(f"    MAP dN/dX(>=20.0/20.3/20.6) = "
          + ", ".join(f"{point['dndx_total'][l]:.5f}" for l in limits))
    print(f"    MAP Omega(>=20.0/20.3/20.6)  = "
          + ", ".join(f"{point['omega'][l]:.4e}" for l in limits))
    print(f"    theta_map = {np.array2string(theta_map, precision=3)}")
    print(f"    multistart_logP_spread={point['_v3x']['multistart_logP_spread']:.3g}, "
          f"at_bound={point['_v3x']['at_bound']}")

    # ---- THETA-Laplace band (nuisance FROZEN) ----
    print(f"[partA] Laplace posterior at MAP (n_draw={args.n_lap})")
    lap = v3x_laplace(theta_map, fwd["A_full"], fwd["M_full"], fwd["lam_fp"],
                      fwd["mu_fp"], fwd["fine"], point["_v3x"]["family"], cfg,
                      obj_weights=fwd["cat_op"].get("op_weights"),
                      n_draw=args.n_lap, rng=np.random.default_rng(args.seed + 1))
    print(f"    Hessian cond={lap['cond']:.3g}, sigma_theta="
          f"{np.array2string(lap['sigma'], precision=3)}")
    lap_band = theta_band_reduce(cfg, point, lap["draws"])
    print(f"    Laplace band reduced ({time.time()-t0:.0f}s)")

    # ---- THETA-emcee cross-check (optional) ----
    emcee_band = None; emcee_info = None
    if not args.skip_emcee:
        try:
            print(f"[partA] emcee theta posterior (n_steps={args.n_emcee_steps})")
            ec = v3x_emcee_check(fwd["A_full"], fwd["M_full"], fwd["lam_fp"],
                                 fwd["mu_fp"], fwd["fine"], point["_v3x"]["family"],
                                 cfg, theta_map, sigma0=lap["sigma"],
                                 n_steps=args.n_emcee_steps,
                                 rng=np.random.default_rng(args.seed + 2))
            print(f"    emcee acc_frac={ec['acceptance_frac']:.3f}, "
                  f"n_samples={ec['n_samples']}")
            # thin to ~n_lap draws for the reduce
            ch = ec["chain"]
            idx = np.linspace(0, len(ch) - 1, min(args.n_lap, len(ch))).astype(int)
            emcee_band = theta_band_reduce(cfg, point, ch[idx])
            emcee_info = dict(acc=ec["acceptance_frac"], n=ec["n_samples"],
                              theta_sigma=ec["theta_sigma"], theta_mean=ec["theta_mean"])
            print(f"    emcee band reduced ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"    emcee SKIPPED (exception): {e}")

    # ---- FULL posterior band: loa0-aware joint-MC (theta + nuisance) ----
    print(f"[partA] FULL posterior joint-MC (loa0, n_mc={args.n_mc}) "
          "[C/rho Wilson + sigma_i + loa0-FP Gamma + sightline bootstrap]")
    full = loa0_full_posterior_mc(cfg, ing, point, args.n_mc,
                                  np.random.default_rng(args.seed + 3))
    print(f"    full band done ({time.time()-t0:.0f}s)")

    # ---- FULL purity_mixture wired joint_mc_errors cross-check (optional) ----
    pm_band = None
    if not args.skip_pm_xref:
        print(f"[partA] purity_mixture wired joint_mc_errors xref (n_mc={args.n_mc})")
        try:
            ing_pm = build_ingredients(args, "purity_mixture")
            cfg_pm = ing_pm["cfg"]; cfg_pm.report_logN_limits = limits
            cfg_pm.n_mc = args.n_mc; cfg_pm._wall1_estimator = "v3"
            cfg_pm.mc_inner = args.mc_inner   # Stage I honored by the PM xref band too
            cfg_pm.mc_nuisance = args.mc_nuisance  # Stage II honored by the PM xref too
            point_pm = ing_pm["estimator_fn"](
                ing_pm["cat_cut"], ing_pm["is_TP"], ing_pm["good_mask"],
                ing_pm["C_interp"], ing_pm["fp_model"], ing_pm["X_tot"],
                ing_pm["logN_lo"], ing_pm["logN_hi"], ing_pm["N_b"], ing_pm["dN_b"],
                ing_pm["truth_cut"], cfg_pm)
            from CDDF_analysis.hbi.cddf_catalog_hbi import make_v3x_refit_fn
            refit_fn = make_v3x_refit_fn(cfg_pm, point_pm["_v3x"], ing_pm["mm"])
            mc_pm = joint_mc_errors(
                ing_pm["cat_cut"], ing_pm["is_TP"], ing_pm["good_mask"], ing_pm["mm"],
                ing_pm["fp_model"], ing_pm["X_tot"], ing_pm["logN_lo"], ing_pm["logN_hi"],
                ing_pm["N_b"], ing_pm["dN_b"], ing_pm["truth_cut"], cfg_pm,
                np.random.default_rng(args.seed + 4), refit_fn=refit_fn)
            pm_band = dict(
                f_b_samples=mc_pm["_samples"]["f_b"], point=point_pm,
                **{f"dndx_{l}_samples": mc_pm["_samples"]["dndx_total"][l] for l in limits},
                **{f"omega_{l}_samples": mc_pm["_samples"]["omega"][l] for l in limits},
                **{f"dndx_z_{l}_samples": mc_pm["_samples"]["dndx_z"][l] for l in limits})
            print(f"    PM xref done ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"    PM xref SKIPPED: {e}")

    # ---- TRUTH over the same searched pathlength/SNR>2/z-window ----
    print("[partA] truth reductions (same pathlength/SNR/z-window)")
    tr = truth_reductions(cfg, ing["truth_cut"], logN_lo, logN_hi, N_b, dN_b, X_tot)
    # truth dN/dX(z): per-zbin truth counts / X(z), per limit
    zbins = np.asarray(cfg.zbins, float)
    from CDDF_analysis.hbi.cddf_catalog_hbi import _bin_index_logN, _zbin_index
    t_nhi = np.asarray(ing["truth_cut"]["NHI"], float)
    t_z = np.asarray(ing["truth_cut"]["Z_DLA"], float)
    t_snr = np.asarray(ing["truth_cut"]["S2N_RED"], float)
    keep = t_snr > cfg.snr_min
    t_nhi, t_z = t_nhi[keep], t_z[keep]
    t_zidx = _zbin_index(t_z, zbins)
    Xz = np.asarray(X_tot, float)
    truth_dndx_z = {}
    for l in limits:
        dz = np.zeros(len(zbins) - 1)
        for k in range(len(zbins) - 1):
            sel = (t_nhi >= l - 1e-9) & (t_nhi < cfg.drop_top_bin_above) & (t_zidx == k)
            dz[k] = sel.sum() / Xz[k] if Xz[k] > 0 else np.nan
        truth_dndx_z[l] = dz

    # ---- save everything ----
    savez = dict(
        logN_lo=logN_lo, logN_hi=logN_hi, N_b=N_b, dN_b=dN_b,
        X_tot=np.asarray(X_tot), zbins=zbins, n_sl_prod=int(ing["n_sl"]),
        report_limits=np.asarray(limits),
        # point
        f_b_point=point["f_b"], theta_map=theta_map,
        # truth
        f_truth=tr["f_truth"],
        sigma_theta_laplace=lap["sigma"], hess_cond=float(lap["cond"]),
        multistart_logP_spread=float(point["_v3x"]["multistart_logP_spread"]),
        # band sample stacks
        lap_f_b_samples=lap_band["f_b_samples"],
        full_f_b_samples=full["f_b_samples"],
    )
    for l in limits:
        savez[f"dndx_{l}_point"] = float(point["dndx_total"][l])
        savez[f"omega_{l}_point"] = float(point["omega"][l])
        savez[f"dndx_{l}_truth"] = float(tr["dndx_total"][l])
        savez[f"omega_{l}_truth"] = float(tr["omega"][l])
        savez[f"truth_dndx_z_{l}"] = truth_dndx_z[l]
        savez[f"lap_dndx_{l}_samples"] = lap_band[f"dndx_{l}_samples"]
        savez[f"lap_omega_{l}_samples"] = lap_band[f"omega_{l}_samples"]
        savez[f"lap_dndx_z_{l}_samples"] = lap_band[f"dndx_z_{l}_samples"]
        savez[f"full_dndx_{l}_samples"] = full[f"dndx_{l}_samples"]
        savez[f"full_omega_{l}_samples"] = full[f"omega_{l}_samples"]
        savez[f"full_dndx_z_{l}_samples"] = full[f"dndx_z_{l}_samples"]
    if emcee_band is not None:
        savez["emcee_f_b_samples"] = emcee_band["f_b_samples"]
        for l in limits:
            savez[f"emcee_dndx_{l}_samples"] = emcee_band[f"dndx_{l}_samples"]
            savez[f"emcee_omega_{l}_samples"] = emcee_band[f"omega_{l}_samples"]
        if emcee_info is not None:
            savez["emcee_acc"] = float(emcee_info["acc"])
            savez["emcee_theta_sigma"] = emcee_info["theta_sigma"]
    if pm_band is not None:
        savez["pm_f_b_samples"] = pm_band["f_b_samples"]
        savez["pm_f_b_point"] = pm_band["point"]["f_b"]
        for l in limits:
            savez[f"pm_dndx_{l}_samples"] = pm_band[f"dndx_{l}_samples"]
            savez[f"pm_omega_{l}_samples"] = pm_band[f"omega_{l}_samples"]
            savez[f"pm_dndx_{l}_point"] = float(pm_band["point"]["dndx_total"][l])
    out_npz = os.path.join(args.out, "partA_posterior.npz")
    np.savez(out_npz, **savez)
    print(f"\n[partA] saved -> {out_npz}  (total {time.time()-t0:.0f}s)")

    # ---- console summary: recovered vs truth + per-bin coverage ----
    def _bands(samp):
        return (np.nanpercentile(samp, 2.5, axis=0), np.nanpercentile(samp, 16, axis=0),
                np.nanpercentile(samp, 50, axis=0), np.nanpercentile(samp, 84, axis=0),
                np.nanpercentile(samp, 97.5, axis=0))
    print("\n" + "=" * 70)
    print("RECOVERED dN/dX & Omega (MAP) vs TRUTH, with FULL-posterior 68/95% band")
    print("=" * 70)
    for l in limits:
        ds = full[f"dndx_{l}_samples"]; os_ = full[f"omega_{l}_samples"]
        dlo95, dlo68, dmed, dhi68, dhi95 = (np.nanpercentile(ds, q) for q in
                                            (2.5, 16, 50, 84, 97.5))
        olo95, olo68, omed, ohi68, ohi95 = (np.nanpercentile(os_, q) for q in
                                            (2.5, 16, 50, 84, 97.5))
        dt = tr["dndx_total"][l]; ot = tr["omega"][l]
        dpull = (point["dndx_total"][l] - dt) / np.nanstd(ds)
        opull = (point["omega"][l] - ot) / np.nanstd(os_)
        print(f"  >={l}: dN/dX MAP={point['dndx_total'][l]:.5f} "
              f"[68:{dlo68:.5f},{dhi68:.5f}] [95:{dlo95:.5f},{dhi95:.5f}] "
              f"truth={dt:.5f} pull={dpull:+.2f}")
        print(f"         Omega MAP={point['omega'][l]:.4e} "
              f"[68:{olo68:.3e},{ohi68:.3e}] [95:{olo95:.3e},{ohi95:.3e}] "
              f"truth={ot:.4e} pull={opull:+.2f}")
    # per-bin f(N) coverage incl the [21,21.5] tail
    print("\nPer-bin f(N) coverage (FULL band) and the [21,21.5] tail verdict:")
    fb_lo95, fb_lo68, fb_med, fb_hi68, fb_hi95 = _bands(full["f_b_samples"])
    fb_std = np.nanstd(full["f_b_samples"], axis=0)
    mid = 0.5 * (logN_lo + logN_hi)
    in68 = (tr["f_truth"] >= fb_lo68) & (tr["f_truth"] <= fb_hi68)
    in95 = (tr["f_truth"] >= fb_lo95) & (tr["f_truth"] <= fb_hi95)
    rep = (logN_lo >= 20.0 - 1e-9) & (tr["f_truth"] > 0)
    tail = (mid >= 21.0 - 1e-9) & (mid <= 21.5 + 1e-9) & (tr["f_truth"] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pull = (point["f_b"] - tr["f_truth"]) / fb_std
    print(f"  reported bins >=20.0: 68%-cov={in68[rep].mean():.2f}, "
          f"95%-cov={in95[rep].mean():.2f} (n={rep.sum()})")
    print(f"  [21.0,21.5] tail bins: 68%-cov={in68[tail].mean():.2f}, "
          f"95%-cov={in95[tail].mean():.2f} (n={tail.sum()})")
    for b in np.where(tail)[0]:
        print(f"    logN={mid[b]:.2f}: MAP f={point['f_b'][b]:.3e} truth={tr['f_truth'][b]:.3e} "
              f"[95:{fb_lo95[b]:.3e},{fb_hi95[b]:.3e}] pull={pull[b]:+.2f} "
              f"in95={bool(in95[b])}")
    return out_npz


if __name__ == "__main__":
    main()
