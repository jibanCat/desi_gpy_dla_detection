#!/usr/bin/env python
"""freebin_localizer.py — decisive 2x-under-recovery localizer.

Reduce the SAME corrected posterior kernel that v3 (bspbody) used, but with the
FREE-BIN (non-parametric, one-DOF-per-(N,z)) v2 estimator (cddf_catalog_hbi.v2_refit)
instead of v3x_refit. Mirrors run_phase3d_postkernel.stage2_v3_fit setup exactly
(cat_cut, molly C/rho regen, pathlength, kernel attach), only swapping the estimator.

If free-bin R0(>=20.3) ~ 1  -> the missing 2x is the bspbody parametric form / lambda.
If free-bin R0(>=20.3) ~ 0.52 -> the 2x is in the forward operator / kernel itself.
"""
from __future__ import annotations
import os, sys, time
from types import SimpleNamespace
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.hbi import cddf_catalog_hbi as H
from CDDF_analysis.hbi.cddf_catalog_hbi import (
    HBIConfig, load_molly_matrix, load_and_cut_catalog, build_fine_grid,
    regenerate_molly_counts, make_C_interpolator, build_pathlength,
    _build_qso_lookup, v2_refit, truth_reductions,
    make_fp_model, make_rho_interpolator,
)

# ---- inputs (same defaults as run_phase3d_postkernel.main) -------------------
KERNEL = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
          "phase3d_postkernel_out/posterior_kernel_2lpt0.npz")
CATDIR = ("/scratch/cavestru_root/cavestru0/mfho/"
          "gl_prod_2lpt0_v1_20260526/combined_catalog/")
TRUTH = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
         "qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits")
BAL = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
       "qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits")
MOLLY = ("/scratch/cavestru_root/cavestru0/mfho/"
         "gl_prod_2lpt0_v1_20260526/figures_molly/molly_matrix.tsv")
OUT = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
       "phase3d_experiments/freebin/")

# args mirror run_phase3d defaults exactly
args = SimpleNamespace(
    host_truth_floor=19.0,
    fit_floor=19.5,            # -> v2_logN_fit_floor=19.5 (same as v3 floor)
)

ZBINS = (2.0, 2.5, 3.0, 3.5)
REPORT = (20.0, 20.3, 20.6)


def make_cfg():
    cfg = HBIConfig(
        catalog_dir=CATDIR, truth_path=TRUTH, bal_cat_path=BAL,
        molly_tsv=MOLLY, out_dir=OUT, mockdir=os.path.dirname(TRUTH),
        zbins=ZBINS, n_mc=0, rng_seed=0,
        fp_estimator="purity_mixture", no_bal=True,
        report_logN_limits=REPORT,
        v2_logN_fit_floor=args.fit_floor,
        v2_z_fit_lo=ZBINS[0], v2_z_fit_hi=ZBINS[-1], v2_z_fit_step=0.1,
        # CRITICAL: route the v2 forward build through the SAME corrected posterior
        # kernel v3 used (default "gaussian" would NOT use cfg._posterior_kernel_2d).
        v2_kernel="posterior",
        # leave v2_lambda_smooth=None -> L-curve choice (the standard v2 operating pt)
    )
    return cfg


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    cfg = make_cfg()

    # attach the SAME cached corrected kernel v3 used
    d = np.load(KERNEL, allow_pickle=True)
    cfg._posterior_kernel_2d = d["kappa"].astype(np.float32)
    print(f"[freebin] attached kernel {cfg._posterior_kernel_2d.shape} from {KERNEL}")

    mm = load_molly_matrix(cfg.molly_tsv)
    truth_floor = float(mm.nhi_edges[0])
    print(f"[freebin] molly truth floor (matrix N-edge[0]) = {truth_floor}")
    qso_lookup = _build_qso_lookup(cfg)
    cat_cut, truth_cut, is_TP, good_mask, meta = load_and_cut_catalog(
        cfg, truth_nhi_floor=truth_floor, qso_lookup=qso_lookup,
        host_truth_floor=min(args.host_truth_floor, truth_floor))
    print(f"[freebin] cat_cut meta: {meta}")

    mm = regenerate_molly_counts(mm, cat_cut, is_TP, truth_cut, good_mask, cfg)
    C_interp = make_C_interpolator(mm)
    X_tot, n_sl, qzl, qzh, qsn, Xcalc = build_pathlength(
        cfg, qso_lookup=qso_lookup, return_per_sl=True)
    logN_lo, logN_hi, N_b, dN_b = build_fine_grid(cfg)

    # build the purity-mixture FP model (v2_refit/fit_forward_hbi need a real fp_model;
    # v3x builds its own internally — this is the ONLY extra wiring vs stage2). Same
    # construction as run_pipeline_v2 step [5].
    s2n = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"], float)
    op_mask = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask
    fp_model, _ = make_fp_model(cfg, cat_cut, op_mask,
                                rho_interp=make_rho_interpolator(mm))

    # alignment guard (the v2 forward build asserts kernel rows == op rows)
    kappa = cfg._posterior_kernel_2d
    print(f"[freebin] kernel rows={kappa.shape[0]} ; op rows={int(op_mask.sum())} ; "
          f"cat_cut rows={len(cat_cut)}")

    rng = np.random.default_rng(cfg.rng_seed)
    print(f"[freebin] running v2_refit (FREE-BIN, v2_kernel=posterior) ... "
          f"({time.time()-t0:.0f}s)")
    res = v2_refit(cat_cut, is_TP, good_mask, C_interp, fp_model, X_tot,
                   logN_lo, logN_hi, N_b, dN_b, truth_cut, cfg,
                   mm=mm, qso_per_sl=(qzl, qzh, qsn), Xcalc=Xcalc, rng=rng)
    lam = res.get("_v2", {}).get("lam_chosen", None)
    print(f"[freebin] v2 lambda_chosen = {lam}")

    tr = truth_reductions(cfg, truth_cut, logN_lo, logN_hi, N_b, dN_b, X_tot)

    print("=" * 70)
    print("[freebin] RESULT — v2 free-bin dN/dX vs truth (kernel ON, same corrected kernel)")
    print("=" * 70)
    print(f"{'limit':>8} {'v2_freebin':>14} {'truth':>14} {'R0=fit/truth':>14}")
    for lim in REPORT:
        v2v = float(res["dndx_total"][lim])
        tv = float(tr["dndx_total"][lim])
        r0 = v2v / tv if tv > 0 else float("nan")
        print(f"{lim:>8} {v2v:>14.5f} {tv:>14.5f} {r0:>14.4f}")
    print("-" * 70)
    print(f"[freebin] DONE ({time.time()-t0:.0f}s)")

    np.savez(os.path.join(OUT, "freebin_result.npz"),
             lam_chosen=(lam if lam is not None else np.nan),
             **{f"v2_dndx_{l}": float(res["dndx_total"][l]) for l in REPORT},
             **{f"truth_dndx_{l}": float(tr["dndx_total"][l]) for l in REPORT})


if __name__ == "__main__":
    main()
