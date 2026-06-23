"""ab_loa0_fp_baseline.py — DLA-tier A/B of the FP estimator on the calibrated
WALL-1 config (reduce-only, cached kernel, NO inference, NO SLURM, NO tilt).

Builds the WALL-1 ``mollynhi195_lyaonly1025_broaden012`` ingredients ONCE (the
SAME cat_cut / frozen molly C/ρ / pathlength / cached 2-D posterior kernel that
``run_phase3d_postkernel.py`` stage 2/3 uses), then runs the UNTILTED baseline
recovery R0 = est/truth for the v3 (bspbody) parametric estimator under BOTH FP
estimators:

  * ``purity_mixture`` (the current default — should reproduce the frozen
    baseline_R0_dndx/omega 1.049/1.090/1.120 @ 20.0/20.3/20.6),
  * ``loa0`` (the directly-measured forest-FP background from
    build_loa0_fp_product.py).

Reports R0(dN/dX, Ω) at 20.0/20.3/20.6 for each, side by side. This is the
baseline-only reduce (no ±tilt) — the WALL-1 tilt refuses a frozen loa-0 FP by
design (spec §7), so the A/B is the absolute-bias R0 the closure divides out.

PREDICTION (spec): loa-0 has 0 FP detections ≥20.3 and only ~5 ≥20.0, so the loa0
μ_FP background is ≈0 in the DLA tier; the headline DLA dN/dX & Ω R0 move toward
the no-FP-subtraction value and only the FP-dominated low-N regimes shift.
"""
from __future__ import annotations

import argparse
import functools
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis import cddf_catalog_hbi as H
from CDDF_analysis.cddf_catalog_hbi import (
    HBIConfig, load_molly_matrix, load_and_cut_catalog, build_fine_grid,
    regenerate_molly_counts, make_C_interpolator, build_pathlength,
    make_fp_model, make_rho_interpolator, _build_qso_lookup, v3x_refit,
)
from CDDF_analysis.cddf_tilt_closure import baseline_recovery

# the calibrated WALL-1 experiment (frozen kernel + lya_only molly)
DEF_EXPER = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
             "phase3d_experiments/mollynhi195_lyaonly1025_broaden012")
DEF_MOLLY = DEF_EXPER + "/molly_matrix.tsv"        # lya_only-195 matrix (resolved below)
DEF_KERNEL = DEF_EXPER + "/posterior_kernel_2lpt0.npz"
# the canonical lya_only-nhi195 molly the broaden012 kernel was calibrated against —
# VERIFIED: purity_mixture reproduces the frozen baseline_R0 EXACTLY with this matrix
# (1.0490/1.0902/1.1195). The experiment dir did NOT persist its own molly_matrix.tsv;
# FIX 2 = the Lyα-only window, FIX 4(a) = never silently substitute a full-forest one.
DEF_LYAONLY_MOLLY = ("/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/"
                     "figures_molly_nhi195/lya_only/molly_matrix.tsv")
DEF_CAT = ("/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/"
           "combined_catalog/")
DEF_TRUTH = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/"
             "v2.8.5/mock-0/loa-124/hcd_truth_cat.fits")
DEF_BAL = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/"
           "v2.8.5/mock-0/loa-124/bal_cat.fits")
# FIX 2: default to the Lyα-only re-binned product (λ_rest>=1025), matching the
# calibrated lya_only config. The full-forest product over-estimates sub-DLA/LLS FP.
DEF_LOA0_PRODUCT = ("/scratch/cavestru_root/cavestru0/mfho/gl_loa0_fp_v1_20260615/"
                    "outputs/loa0_fp_product_lyaonly1025.npz")


def _resolve_molly(args):
    # FIX 4(a): the broaden012 WALL-1 experiment used a SPECIFIC lya_only-195 molly
    # matrix (persisted at the experiment dir or passed via --molly-tsv). The prior
    # code silently fell back to a DIFFERENT prod molly (nhi195 full-forest, or even
    # nhi172) when the lya_only tsv was absent — that substitutes a wrong C/ρ matrix
    # under the SAME frozen kernel and silently corrupts the A/B. We now HARD-FAIL:
    # require either an explicit existing --molly-tsv or the experiment's own
    # molly_matrix.tsv (the one the broaden012 kernel was calibrated against).
    if args.molly_tsv:
        if os.path.exists(args.molly_tsv):
            return args.molly_tsv
        raise SystemExit(f"--molly-tsv does not exist: {args.molly_tsv}")
    # 1) the experiment's own persisted molly (preferred — exact provenance)
    exper_molly = os.path.join(DEF_EXPER, "molly_matrix.tsv")
    if os.path.exists(exper_molly):
        return exper_molly
    # 2) the canonical lya_only-nhi195 molly the broaden012 kernel was calibrated
    #    against (VERIFIED to reproduce the frozen baseline_R0). This is the ONLY
    #    permitted fallback — a Lyα-only nhi195 matrix, matching the lya_only kernel.
    if os.path.exists(DEF_LYAONLY_MOLLY):
        return DEF_LYAONLY_MOLLY
    raise SystemExit(
        "FIX 4(a): no lya_only molly matrix resolved. The WALL-1 broaden012 kernel "
        "was calibrated against a Lyα-only nhi195 molly, but neither the experiment's "
        f"persisted matrix:\n  {exper_molly}\nnor the canonical lya_only matrix:\n  "
        f"{DEF_LYAONLY_MOLLY}\nexists. Pass the EXACT lya_only-1025 nhi195 matrix via "
        "--molly-tsv (do NOT substitute a full-forest or nhi172 matrix — that silently "
        "corrupts the A/B by mismatching the kernel's C/ρ).")


def build_ingredients(args, fp_estimator: str, loa0_product=None):
    """Build all WALL-1 ingredients ONCE for a given FP estimator."""
    molly_tsv = _resolve_molly(args)
    cfg = HBIConfig(
        catalog_dir=args.catalog_dir, truth_path=args.truth,
        bal_cat_path=args.bal_cat, molly_tsv=molly_tsv, out_dir=args.out,
        mockdir=args.mockdir or os.path.dirname(args.truth),
        zbins=tuple(float(x) for x in args.zbins.split(",")),
        report_logN_limits=tuple(float(x) for x in args.report_limits.split(",")),
        fp_estimator=fp_estimator, no_bal=True,
        loa0_product_path=(loa0_product if fp_estimator == "loa0" else None),
        v3_family=args.family, v3_logN_fit_floor=args.fit_floor,
        v3_logN_fit_ceil=args.fit_ceil, v3_lambda_bspbody=args.lambda_bspbody,
        v3_mc_n_restart=2, lam_rf_min=args.lam_rf_min,
        v3_bspbody_edge_slope_lam=args.edge_slope_lam,
        v3_fine_density_gl_nodes=args.gl_nodes,
        v2_z_fit_lo=2.0, v2_z_fit_hi=3.5, v2_z_fit_step=0.1,
        rng_seed=0,
        completeness_z_resolved=bool(getattr(args, "cz_resolved", False)),
        completeness_z_min_count=float(getattr(args, "cz_min_count", 30.0)),
    )
    # attach the cached calibrated kernel (same as stage 2/3)
    d = np.load(args.kernel, allow_pickle=True)
    cfg._posterior_kernel_2d = d["kappa"].astype(np.float32)
    print(f"  [{fp_estimator}] molly={os.path.basename(os.path.dirname(molly_tsv))}, "
          f"kernel {cfg._posterior_kernel_2d.shape}")

    mm = load_molly_matrix(molly_tsv)
    truth_floor = float(mm.nhi_edges[0])
    qso_lookup = _build_qso_lookup(cfg)
    cat_cut, truth_cut, is_TP, good_mask, meta = load_and_cut_catalog(
        cfg, truth_nhi_floor=truth_floor, qso_lookup=qso_lookup,
        host_truth_floor=min(args.host_truth_floor, truth_floor))
    mm = regenerate_molly_counts(mm, cat_cut, is_TP, truth_cut, good_mask, cfg)
    C_interp = make_C_interpolator(mm)
    rho_interp = make_rho_interpolator(mm)
    X_tot, n_sl, qzl, qzh, qsn, Xcalc = build_pathlength(
        cfg, qso_lookup=qso_lookup, return_per_sl=True)
    cfg.n_sl_prod = int(n_sl)   # production SNR>2 sightlines -> loa-0 ell_eff/μ_FP scale
    logN_lo, logN_hi, N_b, dN_b = build_fine_grid(cfg)

    # Track-C #39: build-and-stash the z-resolved completeness if requested on cfg
    # (set by the runner BEFORE build_ingredients — no-op/None when OFF → byte-identical).
    from CDDF_analysis.cddf_catalog_hbi import ensure_cnz_resolved
    if getattr(cfg, "completeness_z_resolved", False):
        ensure_cnz_resolved(cfg, cat_cut, truth_cut, good_mask, mm)
        print(f"  [Track-C #39] z-resolved completeness g(N,z) built "
              f"(shape {cfg._cnz_resolved.g_grid.shape})")

    s2n = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"], float)
    op_mask = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask
    fp_model, _ = make_fp_model(cfg, cat_cut, op_mask, rho_interp)  # stashes cfg._loa0_fp

    estimator_fn = functools.partial(
        v3x_refit, mm=mm, qso_per_sl=(qzl, qzh, qsn), Xcalc=Xcalc,
        rng=np.random.default_rng(0))
    return dict(cfg=cfg, mm=mm, cat_cut=cat_cut, truth_cut=truth_cut, is_TP=is_TP,
                good_mask=good_mask, C_interp=C_interp, fp_model=fp_model,
                X_tot=X_tot, n_sl=n_sl, logN_lo=logN_lo, logN_hi=logN_hi,
                N_b=N_b, dN_b=dN_b, estimator_fn=estimator_fn, meta=meta)


def run_baseline(ing) -> dict:
    cfg = ing["cfg"]
    cfg._wall1_estimator = "v3"
    return baseline_recovery(
        cfg, ing["cat_cut"], ing["is_TP"], ing["good_mask"], ing["truth_cut"],
        ing["C_interp"], ing["fp_model"], ing["X_tot"],
        ing["logN_lo"], ing["logN_hi"], ing["N_b"], ing["dN_b"],
        estimator_fn=ing["estimator_fn"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog-dir", default=DEF_CAT)
    p.add_argument("--truth", default=DEF_TRUTH)
    p.add_argument("--bal-cat", default=DEF_BAL)
    p.add_argument("--molly-tsv", default=None)
    p.add_argument("--kernel", default=DEF_KERNEL)
    p.add_argument("--loa0-product", default=DEF_LOA0_PRODUCT)
    p.add_argument("--out", default="/tmp/ab_loa0_fp")
    p.add_argument("--mockdir", default=None)
    p.add_argument("--zbins", default="2.0,2.5,3.0,3.5")
    p.add_argument("--report-limits", default="20.0,20.3,20.6")
    p.add_argument("--family", default="bspbody")
    p.add_argument("--fit-floor", type=float, default=19.5)
    p.add_argument("--fit-ceil", type=float, default=99.0)
    p.add_argument("--lambda-bspbody", type=float, default=30.0)
    p.add_argument("--lam-rf-min", type=float, default=1025.0)  # lyaonly1025
    p.add_argument("--edge-slope-lam", type=float, default=40.0)
    p.add_argument("--gl-nodes", type=int, default=1)
    p.add_argument("--host-truth-floor", type=float, default=19.0)
    p.add_argument("--cz-resolved", action="store_true",
                   help="Track-C #39: z-resolved completeness C(N,z) (gated; default OFF "
                        "= byte-identical z-marginalized molly C).")
    p.add_argument("--cz-min-count", type=float, default=30.0,
                   help="occupancy floor for the z-resolved g build (sparse-cell shrinkage).")
    p.add_argument("--only", choices=["both", "purity_mixture", "loa0"], default="both")
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    limits = tuple(float(x) for x in args.report_limits.split(","))
    results = {}
    modes = (["purity_mixture", "loa0"] if args.only == "both" else [args.only])
    for mode in modes:
        print("=" * 70)
        print(f"[A/B] fp_estimator = {mode}")
        print("=" * 70)
        ing = build_ingredients(args, mode,
                                loa0_product=args.loa0_product)
        base = run_baseline(ing)
        results[mode] = dict(
            R0_dndx={lim: float(base["R0_dndx_total"][lim]) for lim in limits},
            R0_omega={lim: float(base["R0_omega"][lim]) for lim in limits},
            dndx_est={lim: float(base["e0"]["dndx_total"][lim]) for lim in limits},
            dndx_truth={lim: float(base["t0"]["dndx_total"][lim]) for lim in limits},
            omega_est={lim: float(base["e0"]["omega"][lim]) for lim in limits},
            omega_truth={lim: float(base["t0"]["omega"][lim]) for lim in limits},
            n_sl=int(ing["n_sl"]),
        )

    # report
    print("\n" + "=" * 70)
    print("DLA-TIER A/B BASELINE R0 (est/truth, UNTILTED) — v3 bspbody, kernel ON")
    print("=" * 70)
    hdr = f"{'limit':>7} | " + " | ".join(f"{m:>22}" for m in modes)
    print(hdr)
    print("-" * len(hdr))
    for kind, key in (("R0_dN/dX", "R0_dndx"), ("R0_Omega", "R0_omega")):
        print(f"--- {kind} ---")
        for lim in limits:
            row = f"{lim:>7} | " + " | ".join(
                f"{results[m][key][lim]:>22.4f}" for m in modes)
            print(row)
    out_tsv = os.path.join(args.out, "ab_loa0_fp_baseline.tsv")
    with open(out_tsv, "w") as fh:
        fh.write("metric\tlimit\t" + "\t".join(modes) + "\n")
        for kind, key in (("R0_dndx", "R0_dndx"), ("R0_omega", "R0_omega"),
                          ("dndx_est", "dndx_est"), ("dndx_truth", "dndx_truth"),
                          ("omega_est", "omega_est"), ("omega_truth", "omega_truth")):
            for lim in limits:
                fh.write(f"{kind}\t{lim}\t" +
                         "\t".join(f"{results[m][key][lim]:.6g}" for m in modes) + "\n")
    print(f"\n[A/B] saved -> {out_tsv}")
    # frozen reference for sanity (purity_mixture should ~ reproduce)
    print("\n[ref] frozen broaden012 purity_mixture baseline R0_dndx 20.0/20.3/20.6 = "
          "1.0490/1.0902/1.1195; R0_omega = 1.0209/1.0288/1.0347")
    return results


if __name__ == "__main__":
    main()
