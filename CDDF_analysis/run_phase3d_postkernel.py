#!/usr/bin/env python
"""run_phase3d_postkernel.py — Phase-3d calibrated 2-D posterior-kernel pipeline.

Three stages (called by slurm/greatlakes/production/phase3d_postkernel.sbatch):

  stage1  full-survey build_posterior_kernel over all 1150 processed-h5 (file-
          parallel) -> cache <out>/posterior_kernel_2lpt0.npz (float32). Built in
          the EXACT estimator op order from the SAME load_and_cut_catalog cat_cut
          that run_wall1 rebuilds (deterministic -> the alignment assert in
          v3x_build_forward guards any drift).

  stage2  v3 bspbody MAP fit at the pre-chosen lambda WITH the cached kernel
          (cfg._posterior_kernel_2d) -> the calibrated-kernel point f(N,z), dN/dX, Ω.

  stage3  WALL-1 tilt-closure (both +/-dalpha), using the v3 estimator with the
          cached kernel attached. The headline acceptance: the integrated dN/dX & Ω
          closure passes on BOTH tilts (the kernel's prior-edge skew should remove
          the up-migration the symmetric Gaussian could not).

  s3_prior_null      S3 (a) falsifier: bare-π (likelihood removed) through the SAME
                     kernel + WALL-1. If it reproduces the deep-tier residual growth,
                     the growth is a pipeline artifact -> band UNCONSTRAINED.
  s3_dense_synthetic S3 (b) falsifier: WALL-1 on a synthetic injection with KNOWN
                     f_b + KNOWN scatter but FULL sample density + many absorbers.
                     full-density PASS + real FAIL => starvation, not estimator/DOF.
  s3_all             run both S3 falsifiers.

DISCIPLINE: analysis-side only; NEVER touches dla_gp.py / inference. Runs inside the
sbatch (-n 16, 48G). The kernel build is the heavy step; stage2/3 reuse the cache.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis import cddf_catalog_hbi as H
from CDDF_analysis.cddf_catalog_hbi import (
    HBIConfig, load_molly_matrix, load_and_cut_catalog, build_fine_grid,
    build_posterior_kernel, build_targetid_backlink, _build_qso_lookup,
)


def _run_point_for_test(kernel_znz_model=None, c_nz_model=None,
                        report_limits=(20.0, 20.3, 20.6)):
    """Thin TEST helper: run the v3 (bspbody) UNTILTED point fit on the canonical
    broaden012 WALL-1 bundle with the Track-C knobs set, returning the reduced dN/dX.

    Reuses ``ab_loa0_fp_baseline.build_ingredients`` (the SAME cat_cut / frozen molly
    C/ρ / pathlength / cached 2-D posterior kernel the WALL-1 stage 2/3 uses) so that
    with the knobs default-None this reproduces the frozen broaden012 headline
    dN/dX(≥20.0)=0.09010 BIT-IDENTICALLY (the load-bearing default-OFF gate). The knobs
    set ``cfg.kernel_znz_model`` / ``cfg.c_nz_model`` which gate the (N,z) transform +
    C(N,z) threading in ``v3x_build_forward`` — both OFF (None) by default.

    Returns ``{"dndx": {lim: float}, "omega": {lim: float}}`` keyed by report limit.
    Scratch-only integration helper (needs the broaden012 bundle); the unit tests guard
    its caller with skipif.
    """
    import argparse as _ap
    from CDDF_analysis import ab_loa0_fp_baseline as AB

    ns = _ap.Namespace(
        catalog_dir=AB.DEF_CAT, truth=AB.DEF_TRUTH, bal_cat=AB.DEF_BAL,
        molly_tsv=None, kernel=AB.DEF_KERNEL, loa0_product=AB.DEF_LOA0_PRODUCT,
        out="/tmp/track_c_point_for_test", mockdir=None, zbins="2.0,2.5,3.0,3.5",
        report_limits=",".join(str(x) for x in report_limits), family="bspbody",
        fit_floor=19.5, fit_ceil=99.0, lambda_bspbody=30.0, lam_rf_min=1025.0,
        edge_slope_lam=40.0, gl_nodes=1, host_truth_floor=19.0,
    )
    os.makedirs(ns.out, exist_ok=True)
    ing = AB.build_ingredients(ns, "purity_mixture")
    cfg = ing["cfg"]
    # gate the Track-C knobs (None => byte-identical to the frozen broaden012 headline)
    cfg.kernel_znz_model = kernel_znz_model
    cfg.c_nz_model = c_nz_model
    res = ing["estimator_fn"](
        ing["cat_cut"], ing["is_TP"], ing["good_mask"], ing["C_interp"],
        ing["fp_model"], ing["X_tot"], ing["logN_lo"], ing["logN_hi"],
        ing["N_b"], ing["dN_b"], ing["truth_cut"], cfg, boot_weights=None)
    return {
        "dndx": {lim: float(res["dndx_total"][lim]) for lim in report_limits},
        "omega": {lim: float(res["omega"][lim]) for lim in report_limits},
    }


def _make_cfg(args) -> HBIConfig:
    zbins = tuple(float(x) for x in args.zbins.split(","))
    report_limits = tuple(float(x) for x in args.report_limits.split(","))
    cfg = HBIConfig(
        catalog_dir=args.catalog_dir, truth_path=args.truth,
        bal_cat_path=args.bal_cat, molly_tsv=args.molly_tsv, out_dir=args.out,
        mockdir=args.mockdir or os.path.dirname(args.truth),
        zbins=zbins, n_mc=args.n_mc, rng_seed=args.seed,
        # FP estimator GATE (analysis-side; default purity_mixture is byte-identical
        # to the historical hardcode). 'loa0' = non-circular forest-FP product (the
        # honest DLA-tier number); requires --fp-product (npz from
        # build_loa0_fp_product.py) wired into cfg.loa0_product_path.
        fp_estimator=args.fp_estimator, loa0_product_path=args.fp_product,
        no_bal=True,
        report_logN_limits=report_limits,
        v3_family=args.family,
        v3_logN_fit_floor=args.fit_floor,
        v3_logN_fit_ceil=args.fit_ceil,
        v3_lambda_bspbody=args.lambda_bspbody,
        v3_mc_n_restart=args.mc_n_restart,
        lam_rf_min=args.lam_rf_min,
        v3_bspbody_edge_slope_lam=args.edge_slope_lam,
        v3_fine_density_gl_nodes=args.gl_nodes,
        v2_z_fit_lo=zbins[0], v2_z_fit_hi=zbins[-1], v2_z_fit_step=0.1,
        # Stage I: inner-θ per MC draw — 'map' (MODE, byte-identical default) or
        # 'laplace' (one N(θ̂,H⁻¹) sample, the faithful marginalized WALL-1 band).
        mc_inner=getattr(args, "mc_inner", "map"),
        # Stage II: calibration-nuisance draw — 'indep' (byte-identical default) or
        # 'shared_boot' (one shared TID-blocked resample -> C, ρ, boot_w correlated).
        mc_nuisance=getattr(args, "mc_nuisance", "indep"),
        # Stage III: response (θ_K) treatment — 'frozen' (byte-identical default) or
        # 'marginalize' (per-draw re-fit θ_K + draw the response form q/strength α,
        # re-apply the transform, rebuild A). The validated Stage-III coverage path is
        # the loa0 band (wall1_explain_partA.loa0_full_posterior_mc); this WALL-1 tilt
        # closure path threads the knob for config consistency. Requires kernel_znz_model.
        mc_response=getattr(args, "mc_response", "frozen"),
    )
    return cfg


def stage1_build_kernel(cfg: HBIConfig, args):
    """Build + cache the full-survey 2-D posterior kernel in op order."""
    print("=" * 70)
    print("[stage1] build_posterior_kernel (full survey, file-parallel)")
    print("=" * 70)
    t0 = time.time()
    mm = load_molly_matrix(cfg.molly_tsv)
    truth_floor = float(mm.nhi_edges[0])
    qso_lookup = _build_qso_lookup(cfg)
    # the SAME cat_cut run_wall1 rebuilds (truth floor = matrix floor, host floor 19)
    cat_cut, truth_cut, is_TP, good_mask, meta = load_and_cut_catalog(
        cfg, truth_nhi_floor=truth_floor, qso_lookup=qso_lookup,
        host_truth_floor=min(args.host_truth_floor, truth_floor))
    print(f"    cat_cut meta: {meta}")
    fine = build_fine_grid(cfg)
    print("    building TARGETID->(file,row) backlink over 1150 files ...")
    backlink, files = build_targetid_backlink(args.processed_glob)
    print(f"    backlink: {len(backlink)} TIDs, {len(files)} files "
          f"({time.time()-t0:.0f}s elapsed)")
    out_npz = os.path.join(cfg.out_dir, "posterior_kernel_2lpt0.npz")
    kappa, ess = build_posterior_kernel(
        cfg, cat_cut, good_mask, fine,
        processed_glob=args.processed_glob, pw_samples_path=args.pw_samples,
        backlink=backlink, files=files, out_npz=out_npz,
        n_jobs=args.n_jobs, verbose=True)
    print(f"[stage1] DONE: kappa {kappa.shape} cached -> {out_npz} "
          f"({time.time()-t0:.0f}s)")
    for tlim in (20.3, 20.6, 21.0):
        e = ess[tlim]
        nz = e[e > 0]
        print(f"    ESS(>={tlim}): median={np.median(nz) if nz.size else 0:.1f} "
              f"frac<30={np.mean(e < 30):.3f}")
    return out_npz


def _load_kernel_into_cfg(cfg: HBIConfig, out_npz: str):
    d = np.load(out_npz, allow_pickle=True)
    cfg._posterior_kernel_2d = d["kappa"].astype(np.float32)
    # item 9: attach the per-tier per-object ESS so cddf_tilt_closure's evaluate_gate
    # can apply the band-ESS<30 KILL (gate doc §B) — without this the stored ESS was
    # NEVER read and the §B band-unconstrained fallback never fired.
    ess = {}
    for tlim, key in ((20.3, "ess_203"), (20.6, "ess_206"), (21.0, "ess_210")):
        if key in d.files:
            ess[tlim] = np.asarray(d[key], dtype=np.float32)
    cfg._posterior_kernel_ess = ess if ess else None
    print(f"    attached cfg._posterior_kernel_2d {cfg._posterior_kernel_2d.shape} "
          f"+ ESS tiers {sorted(ess.keys())} from {out_npz}")
    return d


def stage2_v3_fit(cfg: HBIConfig, args, out_npz: str):
    """v3 bspbody MAP fit at the chosen lambda WITH the cached kernel."""
    print("=" * 70)
    print(f"[stage2] v3 bspbody fit (kernel ON, lambda={cfg.v3_lambda_bspbody}, "
          f"floor={cfg.v3_logN_fit_floor})")
    print("=" * 70)
    from CDDF_analysis.cddf_catalog_hbi import (
        load_molly_matrix, regenerate_molly_counts, make_C_interpolator,
        make_rho_interpolator, make_fp_model,
        build_pathlength, v3x_refit, kernel_pit_coverage, _op_mask_and_slots,
    )
    _load_kernel_into_cfg(cfg, out_npz)
    mm = load_molly_matrix(cfg.molly_tsv)
    truth_floor = float(mm.nhi_edges[0])
    qso_lookup = _build_qso_lookup(cfg)
    cat_cut, truth_cut, is_TP, good_mask, meta = load_and_cut_catalog(
        cfg, truth_nhi_floor=truth_floor, qso_lookup=qso_lookup,
        host_truth_floor=min(args.host_truth_floor, truth_floor))
    mm = regenerate_molly_counts(mm, cat_cut, is_TP, truth_cut, good_mask, cfg)
    C_interp = make_C_interpolator(mm)
    X_tot, n_sl, qzl, qzh, qsn, Xcalc = build_pathlength(
        cfg, qso_lookup=qso_lookup, return_per_sl=True)
    # FP-model setup (GATE on cfg.fp_estimator). purity_mixture: v3x_refit/v3x_build_forward
    # resolves (1−ρ) internally per-row → no setup needed (byte-identical default path).
    # loa0: the frozen forest-FP product needs cfg.n_sl_prod (the production SNR>2
    # sightline count, for the Gehrels Gamma exposure ℓ_eff/μ_FP scale) AND cfg._loa0_fp
    # (attached by make_fp_model, which bins the op-passing detections into the loa-0
    # cell grid for the per-DETECTION FP share). v3x_build_forward then reads
    # cfg._loa0_fp via _forward_fp_terms. The UNTILTED point fit (boot_weights=None) is
    # the supported loa0 path (the WALL-1 tilt refuses a frozen background — spec §7/§4).
    if cfg.fp_estimator == "loa0":
        cfg.n_sl_prod = int(n_sl)
        rho_interp = make_rho_interpolator(mm)
        s2n = np.asarray(cat_cut["S2N_RED"], float)
        pdla = np.asarray(cat_cut["P_DLA"], float)
        op_mask = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask
        _fp_model, _ = make_fp_model(cfg, cat_cut, op_mask, rho_interp)
        print(f"    [loa0] cfg._loa0_fp attached (n_sl_prod={cfg.n_sl_prod}, "
              f"product={cfg.loa0_product_path})")
    logN_lo, logN_hi, N_b, dN_b = build_fine_grid(cfg)

    # bayesian-review item 4: run+record the parameter-free isolated-TP PIT-coverage
    # DIAGNOSTIC (FROZEN gate §A(a), amended) BEFORE WALL-1. The kernel has no beta to
    # freeze; PIT coverage of the per-object kappa CDF at the truth-host logN should be
    # ~Uniform on the isolated true positives. Reported pass/fail, never iterated.
    kappa = getattr(cfg, "_posterior_kernel_2d", None)
    if kappa is not None and "NHI_TRUE" in cat_cut.colnames:
        op_mask, slot_op, _tid, _did = _op_mask_and_slots(cat_cut, good_mask, cfg)
        nhi_true_op = np.asarray(cat_cut["NHI_TRUE"], float)[op_mask]
        is_tp_op = np.asarray(is_TP, bool)[op_mask]
        # ISOLATED TP = a true-positive detection that is the only DLA slot in its
        # spectrum (slot 0 of a single-DLA op row). slot>=1 rows are by definition
        # multi-DLA (not isolated); slot-0 rows whose spectrum has any slot>=1 are
        # excluded by requiring the spectrum to have a single op detection.
        tid_op = np.asarray(cat_cut["TARGETID"], np.int64)[op_mask]
        _u, _cnt = np.unique(tid_op, return_counts=True)
        single = dict(zip(_u.tolist(), (_cnt == 1).tolist()))
        isolated = is_tp_op & (slot_op == 0) & np.array(
            [single.get(int(t), False) for t in tid_op])
        if kappa.shape[0] == op_mask.sum():
            pit = kernel_pit_coverage(kappa, logN_lo, logN_hi, nhi_true_op,
                                      isolated_mask=isolated)
            print(f"[stage2] PIT (isolated-TP, parameter-free DIAGNOSTIC): "
                  f"n={pit['n_isolated_tp']}, cov68={pit['coverage'][0.68]:.3f}, "
                  f"cov95={pit['coverage'][0.95]:.3f}, KS-vs-Unif={pit['ks_uniform']:.3f}")
            np.savez(os.path.join(cfg.out_dir, "phase3d_pit_isolated_tp.npz"),
                     pit=pit["pit"], n_isolated_tp=pit["n_isolated_tp"],
                     cov68=pit["coverage"][0.68], cov95=pit["coverage"][0.95],
                     ks_uniform=pit["ks_uniform"])
        else:
            print(f"[stage2] PIT SKIPPED: kernel rows {kappa.shape[0]} != op rows "
                  f"{int(op_mask.sum())} (alignment mismatch)")
    rng = np.random.default_rng(cfg.rng_seed)
    res = v3x_refit(cat_cut, is_TP, good_mask, C_interp, None, X_tot,
                    logN_lo, logN_hi, N_b, dN_b, truth_cut, cfg,
                    mm=mm, qso_per_sl=(qzl, qzh, qsn), Xcalc=Xcalc, rng=rng)
    print("[stage2] v3 point estimate (kernel ON):")
    # NOTE: v3x_refit (like v1/v2) returns dndx_total/omega as dicts KEYED BY report
    # limit, NOT scalars — index by lim for the print and flatten for the npz.
    for lim in cfg.report_logN_limits:
        print(f"    dN/dX(>={lim})={res['dndx_total'][lim]:.5f}  "
              f"Omega={res['omega'][lim]:.4e}")
    savez_kw = dict(f_b=res["f_b"], theta_map=res["_v3x"]["theta_map"],
                    logN_lo=logN_lo, logN_hi=logN_hi)
    for lim in cfg.report_logN_limits:
        savez_kw[f"dndx_total_{lim}"] = float(res["dndx_total"][lim])
        savez_kw[f"omega_{lim}"] = float(res["omega"][lim])
    np.savez(os.path.join(cfg.out_dir, "phase3d_v3_point_kernel.npz"), **savez_kw)
    print(f"[stage2] DONE -> {cfg.out_dir}/phase3d_v3_point_kernel.npz")


def stage3_wall1(cfg: HBIConfig, args, out_npz: str):
    """WALL-1 both tilts + prior-only null, v3 estimator WITH the cached kernel."""
    print("=" * 70)
    print("[stage3] WALL-1 tilt-closure (kernel ON) + prior-only null")
    print("=" * 70)
    from CDDF_analysis.cddf_tilt_closure import run_wall1
    _load_kernel_into_cfg(cfg, out_npz)
    print("    [tilt] +/-dalpha with the CALIBRATED kernel attached to cfg")
    gate = run_wall1(cfg, dalpha=args.dalpha, nominal_coverage=0.95,
                     host_truth_floor=args.host_truth_floor, estimator="v3",
                     closure_R0_mode="divide")
    print("[stage3] WALL-1 gate result:")
    print(f"    {gate}")
    print(f"[stage3] DONE -> {cfg.out_dir}")
    print("    (the S3 falsifiers are SEPARATE stages: --stage s3_prior_null and "
          "--stage s3_dense_synthetic — NOT the untilted R0 baseline.)")


def stage_s3_prior_null(cfg: HBIConfig, args, out_npz: str):
    """S3 (a) PRIOR-ONLY NULL: push the BARE inference prior π (likelihood removed)
    through the SAME kernel + WALL-1 machinery. If this reproduces the deep-tier
    residual GROWTH, the growth is a kernel-pipeline artifact, not a physical f(N)
    feature → KILL the differential band (WALL1_GATE_FROZEN.md §B)."""
    print("=" * 70)
    print("[s3_prior_null] BARE-π null (likelihood removed) through kernel + WALL-1")
    print("=" * 70)
    from CDDF_analysis.cddf_tilt_closure import run_wall1
    # the prior-only kernel needs n_obs in the SAME op order; load the cached real
    # kernel only to read its row count (n_obs), then OVERWRITE every row with the
    # bare-π null row (identical for every object — no per-object likelihood).
    d = np.load(out_npz, allow_pickle=True)
    n_obs = int(d["kappa"].shape[0])
    fine = build_fine_grid(cfg)
    kappa_null = H.prior_only_kernel(cfg, n_obs, fine, pw_samples_path=args.pw_samples)
    cfg._posterior_kernel_2d = kappa_null
    print(f"    attached PRIOR-ONLY kernel {kappa_null.shape} (every row == bare-π)")
    out = run_wall1(cfg, dalpha=args.dalpha, nominal_coverage=0.95,
                    host_truth_floor=args.host_truth_floor, estimator="v3",
                    closure_R0_mode="divide")
    gate = out["gate"]
    print("[s3_prior_null] WALL-1 on the bare-π null:")
    print(f"    PASSED={gate.get('passed')}  fail_reasons={gate.get('fail_reasons')}")
    print("    DECISION RULE: if the deep-tier residual GROWS here too, the real-kernel "
          "growth is a pipeline artifact -> band UNCONSTRAINED (Gehrels/Poisson fallback).")
    np.savez(os.path.join(cfg.out_dir, "phase3d_s3_prior_null_gate.npz"),
             gate_passed=bool(gate.get("passed", False)),
             fail_reasons=np.array(gate.get("fail_reasons", []), dtype=object))
    print(f"[s3_prior_null] DONE -> {cfg.out_dir}/phase3d_s3_prior_null_gate.npz")


def stage_s3_dense_synthetic(cfg: HBIConfig, args, out_npz: str):
    """S3 (b) DENSE-SYNTHETIC: WALL-1 on a synthetic injection with a KNOWN f_b
    power-law + KNOWN scatter but FULL sample density + many absorbers. With full
    density the closure MUST pass for the correct kernel — a FAIL here is the
    estimator/DOF, a PASS here but FAIL on the real catalog isolates STARVATION."""
    print("=" * 70)
    print("[s3_dense_synthetic] WALL-1 on a dense known-f_b synthetic injection")
    print("=" * 70)
    from CDDF_analysis.cddf_catalog_hbi import (
        load_molly_matrix, regenerate_molly_counts, build_pathlength,
    )
    from CDDF_analysis.cddf_tilt_closure import run_one_tilt, evaluate_gate, baseline_recovery
    import functools
    from CDDF_analysis.cddf_catalog_hbi import (v3x_refit, make_C_interpolator,
                                                make_rho_interpolator, make_fp_model)
    mm = load_molly_matrix(cfg.molly_tsv)
    qso_lookup = _build_qso_lookup(cfg)
    X_tot, n_sl, qzl, qzh, qsn, Xcalc = build_pathlength(
        cfg, qso_lookup=qso_lookup, return_per_sl=True)
    fine = build_fine_grid(cfg)
    logN_lo, logN_hi, N_b, dN_b = fine
    syn = H.dense_synthetic_wall1_inputs(
        cfg, fine, mm, (qzl, qzh, qsn), Xcalc,
        beta_true=args.syn_beta, n_absorbers=args.syn_nabs,
        sigma_scatter=args.syn_sigma, z_lo=cfg.v2_z_fit_lo, z_hi=cfg.v2_z_fit_hi)
    cat_cut = syn["cat_cut"]; truth_cut = syn["truth_cut"]
    good_mask = syn["good_mask"]; is_TP = syn["is_TP"]
    # regen frozen C/ρ on the synthetic bundle so completeness is realistic, then
    # attach the KNOWN-scatter kappa and run WALL-1 (v3, both tilts) + the gate.
    mm = regenerate_molly_counts(mm, cat_cut, is_TP, truth_cut, good_mask, cfg)
    cfg._posterior_kernel_2d = syn["kappa"]
    C_interp = make_C_interpolator(mm); rho_interp = make_rho_interpolator(mm)
    s2n = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"], float)
    op_mask = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask
    fp_model, _ = make_fp_model(cfg, cat_cut, op_mask, rho_interp)
    rng = np.random.default_rng(cfg.rng_seed)
    cfg._wall1_estimator = "v3"
    estimator_fn = functools.partial(v3x_refit, mm=mm, qso_per_sl=(qzl, qzh, qsn),
                                     Xcalc=Xcalc, rng=rng)
    baseline = baseline_recovery(cfg, cat_cut, is_TP, good_mask, truth_cut,
                                 C_interp, fp_model, X_tot, logN_lo, logN_hi, N_b, dN_b,
                                 estimator_fn=estimator_fn)
    res_p = run_one_tilt(cfg, cat_cut, is_TP, good_mask, truth_cut, mm,
                         C_interp, fp_model, X_tot, logN_lo, logN_hi, N_b, dN_b,
                         +args.dalpha, rng, baseline, estimator_fn=estimator_fn)
    res_m = run_one_tilt(cfg, cat_cut, is_TP, good_mask, truth_cut, mm,
                         C_interp, fp_model, X_tot, logN_lo, logN_hi, N_b, dN_b,
                         -args.dalpha, rng, baseline, estimator_fn=estimator_fn)
    gate = evaluate_gate(res_p, res_m, logN_lo, logN_hi=logN_hi, N_b=N_b, dN_b=dN_b,
                         report_limits=cfg.report_logN_limits, baseline=baseline,
                         estimator="v3", H0=cfg.H0,
                         drop_top_above=cfg.drop_top_bin_above)
    print("[s3_dense_synthetic] WALL-1 on the dense synthetic:")
    print(f"    truth_params={syn['truth_params']}")
    print(f"    PASSED={gate.get('passed')}  fail_reasons={gate.get('fail_reasons')}")
    print("    DECISION RULE: full-density PASS + real-catalog FAIL => STARVATION on "
          "the real catalog (not the estimator/DOF). full-density FAIL => estimator/DOF.")
    np.savez(os.path.join(cfg.out_dir, "phase3d_s3_dense_synthetic_gate.npz"),
             gate_passed=bool(gate.get("passed", False)),
             fail_reasons=np.array(gate.get("fail_reasons", []), dtype=object),
             **{f"truth_{k}": v for k, v in syn["truth_params"].items()})
    print(f"[s3_dense_synthetic] DONE -> {cfg.out_dir}/phase3d_s3_dense_synthetic_gate.npz")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage",
                   choices=["1", "2", "3", "all",
                            "s3_prior_null", "s3_dense_synthetic", "s3_all"],
                   default="all")
    p.add_argument("--out", default=H.DEF_PHASE3D_OUT)
    p.add_argument("--catalog-dir",
                   default=("/scratch/cavestru_root/cavestru0/mfho/"
                            "gl_prod_2lpt0_v1_20260526/combined_catalog/"))
    p.add_argument("--truth",
                   default=("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                            "qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits"))
    p.add_argument("--bal-cat",
                   default=("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                            "qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits"))
    p.add_argument("--molly-tsv",
                   default=("/scratch/cavestru_root/cavestru0/mfho/"
                            "gl_prod_2lpt0_v1_20260526/figures_molly/molly_matrix.tsv"))
    p.add_argument("--mockdir", default=None)
    p.add_argument("--processed-glob", default=H.DEF_PROCESSED_GLOB)
    p.add_argument("--pw-samples", default=H.DEF_PW_SAMPLES)
    p.add_argument("--zbins", default="2.0,2.5,3.0,3.5")
    # item 3: the FROZEN gate doc criterion (b) requires {>=20.0,>=20.3,>=20.6}; the
    # headline (b)/(c)/(d) and the v3 untilted-R0 precondition iterate report_limits,
    # so >=20.6 MUST be a report limit or the committed contract is not evaluated.
    p.add_argument("--report-limits", default="20.0,20.3,20.6")
    p.add_argument("--fit-floor", type=float, default=19.5)
    p.add_argument("--family", default="bspbody",
                   help="v3 CDDF family: bspbody (default, penalized B-spline) | plaw | "
                        "plawcut (Schechter) | bplcut (broken power-law). For the "
                        "throw-away/fit-ceil test, a CONTROLLED power-law family (plaw/"
                        "plawcut) extrapolates sanely above the ceiling, unlike bspbody "
                        "(whose unconstrained high-N spline knots blow up).")
    p.add_argument("--fit-ceil", type=float, default=99.0,
                   help="fit CEILING (default 99=none). Set e.g. 21.0 to restrict the "
                        "likelihood's active band to [fit_floor, fit_ceil]; the parametric "
                        "family then fits only well-localized low-N detections and "
                        "EXTRAPOLATES above (throw-away-high-N test for full ±0.5 closure).")
    p.add_argument("--lambda-bspbody", type=float, default=30.0)
    p.add_argument("--mc-n-restart", type=int, default=2,
                   help="WALL-1 per-draw MC-refit multistart count (point solve uses "
                        "v3_n_restart=8). Raise to 8 to test the gate-MC-convergence "
                        "hypothesis (whether the MC band is biased low by under-convergence).")
    p.add_argument("--lam-rf-min", type=float, default=911.0,
                   help="Rest-frame blue edge of the search window (Å). 911.0 = Lyman "
                        "limit = full Lyα+Lyβ region (default, matches the production "
                        "finder). Set 1025.7223 (Lyβ rest) for the Lyα-only forest — "
                        "restricts catalog cut + truth cut + pathlength consistently; "
                        "pair with the lya_only molly matrix.")
    p.add_argument("--gl-nodes", type=int, default=1,
                   help="within-bin density quadrature nodes for A·f/M·f. 1=bin-midpoint "
                        "(default, exact current behavior). 3=Gauss-Legendre (N ln10)-"
                        "weighted bin mean — removes the slope-dependent midpoint bias "
                        "(WALL-1 V3_KERNEL_SLOPE_DEPENDENCE structural fix candidate).")
    p.add_argument("--edge-slope-lam", type=float, default=40.0,
                   help="Strength of the fixed low-N edge-slope PRIOR anchor (pins "
                        "d(log f)/d(logN) toward v3_bspbody_edge_slope_target=-1.4). "
                        "Default 40.0. Set 0.0 to neutralize the anchor — a WALL-1 "
                        "slope-robustness probe (does the fixed-slope prior memory drive "
                        "the tilt-closure pull?). Pair with a low --lambda-bspbody.")
    p.add_argument("--fp-estimator", choices=["purity_mixture", "loa0"],
                   default="purity_mixture",
                   help="False-positive model. 'purity_mixture' (default, BYTE-IDENTICAL "
                        "to the historical hardcode): per-row (1−ρ) contamination. 'loa0': "
                        "non-circular frozen forest-FP product (the honest DLA-tier number); "
                        "requires --fp-product. Only the UNTILTED stage-2 point fit supports "
                        "loa0 here (the WALL-1 tilt closure refuses a frozen background — "
                        "spec §7/§4; use ab_loa0_fp_baseline / wall1_explain_partA for loa0 "
                        "bands/closure).")
    p.add_argument("--fp-product", default=None,
                   help="Path to the loa-0 FP product npz (from build_loa0_fp_product.py); "
                        "wired into cfg.loa0_product_path. Required when --fp-estimator loa0.")
    p.add_argument("--dalpha", type=float, default=0.5)
    p.add_argument("--host-truth-floor", type=float, default=19.0)
    p.add_argument("--n-mc", type=int, default=200)
    p.add_argument("--mc-inner", choices=["map", "laplace"], default="map",
                   help="Stage I: inner-θ per WALL-1 MC draw — 'map' (MODE, "
                        "byte-identical default) or 'laplace' (one N(θ̂,H⁻¹) sample, "
                        "folds in the within-ψ population-fit width).")
    p.add_argument("--mc-nuisance", choices=["indep", "shared_boot"], default="indep",
                   help="Stage II: calibration-nuisance draw — 'indep' (independent "
                        "per-cell Jeffreys-Betas, byte-identical default) or "
                        "'shared_boot' (ONE shared TID-blocked resample of the "
                        "truth-match per draw -> C, ρ, boot_w correlated).")
    p.add_argument("--mc-response", choices=["frozen", "marginalize"], default="frozen",
                   help="Stage III: response (θ_K) treatment — 'frozen' (byte-identical "
                        "default) or 'marginalize' (per-draw re-fit θ_K + draw the "
                        "response form/strength, re-apply the transform, rebuild A; "
                        "requires kernel_znz_model + mc_nuisance shared_boot). The "
                        "validated coverage path is the loa0 band (wall1_explain_partA).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=16)
    # S3 falsifier (dense-synthetic) knobs
    p.add_argument("--syn-beta", type=float, default=-1.9,
                   help="known f(N) slope for the dense-synthetic S3 injection")
    p.add_argument("--syn-nabs", type=int, default=40000,
                   help="number of synthetic absorbers (full sample density)")
    p.add_argument("--syn-sigma", type=float, default=0.15,
                   help="known per-object N-hat Gaussian scatter (dex)")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    cfg = _make_cfg(args)
    out_npz = os.path.join(cfg.out_dir, "posterior_kernel_2lpt0.npz")

    if args.stage in ("1", "all"):
        out_npz = stage1_build_kernel(cfg, args)
    if args.stage in ("2", "all"):
        if not os.path.exists(out_npz):
            raise SystemExit(f"stage2 needs the cached kernel; {out_npz} missing "
                             "(run stage 1 first)")
        stage2_v3_fit(cfg, args, out_npz)
    if args.stage in ("3", "all"):
        if not os.path.exists(out_npz):
            raise SystemExit(f"stage3 needs the cached kernel; {out_npz} missing "
                             "(run stage 1 first)")
        stage3_wall1(cfg, args, out_npz)
    # S3 falsifiers (settle kernel-vs-starvation; WALL1_GATE_FROZEN.md §B/§D)
    if args.stage in ("s3_prior_null", "s3_all"):
        if not os.path.exists(out_npz):
            raise SystemExit(f"s3_prior_null needs the cached kernel for n_obs; "
                             f"{out_npz} missing (run stage 1 first)")
        stage_s3_prior_null(cfg, args, out_npz)
    if args.stage in ("s3_dense_synthetic", "s3_all"):
        # the dense-synthetic stage builds its OWN catalog/kernel; no cache needed.
        stage_s3_dense_synthetic(cfg, args, out_npz)


if __name__ == "__main__":
    main()
