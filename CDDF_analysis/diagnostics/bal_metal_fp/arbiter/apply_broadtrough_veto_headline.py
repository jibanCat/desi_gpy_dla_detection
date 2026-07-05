#!/usr/bin/env python
"""apply_broadtrough_veto_headline.py — re-derive the REAL-LOA Track-C headline
dN/dX & Omega with a SYMMETRIC broad-trough BAL veto applied as an INPUT PRE-FILTER.

WHAT THIS DOES (reduce-only; no re-inference; estimator byte-frozen):
  * Reuses track_c_tf_loa.py's own build_frozen_calibration / build_loa_ingredients /
    run_measurement (the committed real-LOA headline path). The estimator
    (cddf_catalog_hbi.py) and gpy_dla_detection/ are NOT touched.
  * The veto is implemented purely by AUGMENTING the real-LOA bal_cat.fits TARGETID
    list. The estimator already drops bal_cat TIDs SYMMETRICALLY from BOTH the
    detection catalog (load_and_cut_catalog -> make_lambda_z_BAL_cuts on cat AND
    truth) AND the forest pathlength (build_pathlength skips t in bal_tids). Adding
    the broad-trough TIDs to that same list therefore removes those full sightlines
    from numerator and denominator together — the required symmetry, using the
    estimator's own existing mechanism, with zero estimator edits.

THE VETO (task spec): a TARGETID is broad-trough-vetoed iff, in the v2 VAC,
    WIDEST_CIV_450 > 2000 km/s  AND  significant AI
        (AI_CIV>0 & ERR_AI_CIV>0 & AI_CIV > 3*ERR_AI_CIV).
  WIDEST_CIV_450 is derived from the official v2 VAC per-system arrays as
    max_i |VMAX_CIV_450[i] - VMIN_CIV_450[i]|  (cross-checks our own VAC at 99.99%).
  Applied IN ADDITION to the production BI_CIV>0 veto already baked into the staged
  bal_cat (the augmented set is the UNION).

PRIVACY: writes only an augmented bal_cat.fits (a BAL TARGETID classification list)
and the augmented-run JSON to SCRATCH. No real-LOA dN/dX/Omega values are written to
the repo. This script is committed to the repo (721b2da); it writes ONLY aggregate
prints to stdout and aggregate JSON to SCRATCH. Aggregate-only prints go to stdout.

Env: conda gpdla; OMP/OPENBLAS/MKL_NUM_THREADS=1.
"""
from __future__ import annotations
import argparse
import os
import sys
import types

import numpy as np
import fitsio

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.hbi import track_c_tf_loa as TF
from CDDF_analysis.hbi import ab_loa0_fp_baseline as AB
from CDDF_analysis.hbi.cddf_catalog_hbi import make_fp_model, make_rho_interpolator

V2_VAC = "/nfs/turbo/lsa-cavestru/mfho/DESI/loa/QSO_cat_loa_main_dark_healpix_v2-altbal.fits"
SCRATCH_OUT = "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/tf_loa/bal_veto_arbiter"
# lya-only (lam_rf_min=1025) loa-0 forest-FP product (the DLA-tier FP for the headline).
# The FP background is measured on the metals-off loa-0 mock (n_sl_loa0=2255) and
# extrapolated to production volume via vol_scale = cfg.n_sl_prod / n_sl_loa0, where
# cfg.n_sl_prod is set by build_pathlength to the REAL-LOA op sightline count — so the
# veto's sightline drop scales the FP background down symmetrically with the pathlength.
LOA0_LYAONLY = ("/scratch/cavestru_root/cavestru0/mfho/gl_loa0_fp_v1_20260615/"
                "outputs/loa0_fp_product_lyaonly1025.npz")


def _git_commit():
    """Return the repo HEAD hash for provenance. On failure (e.g. git missing or a
    weird checkout) return "unknown" AND print a loud WARNING — never crash, but the
    failure must be visible so an "unknown" stamp is never shipped silently."""
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception as e:
        print(f"  [WARN] _git_commit() failed ({type(e).__name__}: {e}); "
              f"code_commit will be stamped 'unknown' (cwd={_REPO}).", file=sys.stderr)
        return "unknown"


def preflight_loa0_product(product_path, cfg, molly_tsv):
    """Provenance guards for the loa-0 forest-FP headline (task fixes #6/#7). Hard-fail
    (SystemExit) if the loa-0 FP product or the resolved molly matrix does not match the
    lya-only (lam_rf_min=1025) headline config — this makes a silent wrong-product /
    wrong-molly substitution (each shown by the panel to shift dN/dX) impossible to ship.

    Checks, all against the product's OWN self-documenting fields:
      * product exists;
      * product.lya_only_lam_rf_min == 1025 == cfg.lam_rf_min  (NOT a full-forest product);
      * product.snr_min == cfg.snr_min ; product.p_dla_min == cfg.p_dla_min ;
      * the resolved 2LPT-0 calibration molly path is the lya_only-nhi195 matrix
        (path contains both 'nhi195' and 'lya_only') — a full-forest molly dropped into
        the resolved dir silently shifts dN/dX ~+1.7% (panel finding).
    Returns the product's recorded molly_tsv (for logging/provenance)."""
    if not os.path.exists(product_path):
        raise SystemExit(f"loa0 FP product not found: {product_path}")
    d = np.load(product_path, allow_pickle=True)
    prod_lam = float(d["lya_only_lam_rf_min"]) if "lya_only_lam_rf_min" in d.files else None
    prod_snr = float(d["snr_min"]) if "snr_min" in d.files else None
    prod_pdla = float(d["p_dla_min"]) if "p_dla_min" in d.files else None
    prod_molly = str(d["molly_tsv"]) if "molly_tsv" in d.files else "<absent>"
    print(f"  [PREFLIGHT] loa0 product = {product_path}")
    print(f"  [PREFLIGHT]   product self-doc: lya_only_lam_rf_min={prod_lam} "
          f"snr_min={prod_snr} p_dla_min={prod_pdla} molly_tsv={prod_molly}")
    print(f"  [PREFLIGHT]   cfg: lam_rf_min={cfg.lam_rf_min} snr_min={cfg.snr_min} "
          f"p_dla_min={cfg.p_dla_min}")
    print(f"  [PREFLIGHT]   resolved 2LPT-0 calibration molly = {molly_tsv}")
    tol = 1e-9
    if prod_lam is None or abs(prod_lam - 1025.0) > tol:
        raise SystemExit(
            f"loa0 product lya_only_lam_rf_min={prod_lam} != 1025.0 — this is NOT the "
            f"lya-only headline product (a full-forest product over-estimates the "
            f"sub-DLA/LLS FP). Product: {product_path}")
    if abs(prod_lam - float(cfg.lam_rf_min)) > tol:
        raise SystemExit(
            f"loa0 product lya_only_lam_rf_min={prod_lam} != cfg.lam_rf_min="
            f"{cfg.lam_rf_min} — product/analysis lam_rf_min mismatch.")
    if prod_snr is None or abs(prod_snr - float(cfg.snr_min)) > tol:
        raise SystemExit(
            f"loa0 product snr_min={prod_snr} != cfg.snr_min={cfg.snr_min} — the FP "
            f"background was measured against a different op selection.")
    if prod_pdla is None or abs(prod_pdla - float(cfg.p_dla_min)) > tol:
        raise SystemExit(
            f"loa0 product p_dla_min={prod_pdla} != cfg.p_dla_min={cfg.p_dla_min} — the "
            f"FP background was measured against a different op selection.")
    mlow = str(molly_tsv).lower()
    if not ("nhi195" in mlow and "lya_only" in mlow):
        raise SystemExit(
            f"resolved calibration molly is not the expected lya_only-nhi195 matrix "
            f"(path must contain 'nhi195' and 'lya_only'): {molly_tsv} — a full-forest "
            f"molly silently shifts dN/dX (~+1.7%).")
    return prod_molly


def default_args():
    """Namespace mirroring track_c_tf_loa.main()'s committed headline defaults."""
    a = types.SimpleNamespace(
        catalog_dir=TF._C0_CAT, truth=TF._C0_TRUTH, bal_cat=TF._C0_BAL,
        molly_tsv=None, kernel=AB.DEF_KERNEL, forward_model=TF._DEF_FORWARD,
        resp_family="empirical", resp_kind="forward", loa_kernel=None,
        loa_processed_glob=("/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/"
                            "loa_main_dark_v1/processed/processed-main-dark-*.h5"),
        loa_pw_samples=("/scratch/cavestru_root/cavestru0/mfho/DESI/"
                        "desi_gpy_dla_detection/data/dr12q/processed/"
                        "pw_samples_a3_172_225_50000.mat"),
        loa_cat=TF._LOA_CAT, loa_truth=TF._LOA_TRUTH, loa_bal=TF._LOA_BAL,
        loa_mockdir=TF._LOA_MOCKDIR, out=SCRATCH_OUT,
        report_out=os.path.join(SCRATCH_OUT, "_report.md"),
        zbins="2.0,2.5,3.0,3.5", v2_z_fit_hi=3.5, report_limits="20.0,20.3",
        family="bspbody", fit_floor=19.5, fit_ceil=99.0, lambda_bspbody=30.0,
        lam_rf_min=1025.0, edge_slope_lam=40.0, gl_nodes=1, host_truth_floor=19.0,
        n_mc=12, workers=4, seed=0, cz_min_count=30.0,
        band_recenter=True, omega_slope_extrap=True,
        omega_slope_extrap_integrated=True, slope_edge=21.2, slope_fit_dex=0.6,
        sigma_slope=0.5,
    )
    a._limits = tuple(float(x) for x in a.report_limits.split(","))
    return a


def derive_broadtrough_tids(vac_path=V2_VAC):
    """Broad-trough veto set from the official v2 VAC (task's named source)."""
    v = fitsio.read(vac_path, columns=["TARGETID", "AI_CIV", "ERR_AI_CIV",
                                       "VMIN_CIV_450", "VMAX_CIV_450"])
    tid = np.asarray(v["TARGETID"], np.int64)
    ai = np.asarray(v["AI_CIV"], float)
    eai = np.asarray(v["ERR_AI_CIV"], float)
    vmin = np.asarray(v["VMIN_CIV_450"], float)
    vmax = np.asarray(v["VMAX_CIV_450"], float)
    widest = np.abs(vmax - vmin).max(axis=1)     # -1 sentinels -> 0 width
    sigai = (ai > 0) & (eai > 0) & (ai > 3 * eai)
    broad = (widest > 2000.0) & sigai
    return set(tid[broad].tolist())


def build_augmented_bal(orig_bal_path, broad_tids, out_path):
    """Union the staged (BI_CIV>0) bal_cat TIDs with the broad-trough TIDs, write to
    SCRATCH. Returns (n_orig, n_broad, n_new_added, n_union)."""
    ob = fitsio.read(orig_bal_path)
    ob_tids = set(int(t) for t in ob["TARGETID"])
    union = ob_tids | broad_tids
    n_new = len(broad_tids - ob_tids)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    arr = np.array(sorted(union), dtype=np.int64)
    fitsio.write(out_path, np.rec.fromarrays([arr], names="TARGETID"),
                 clobber=True)
    return len(ob_tids), len(broad_tids), n_new, len(union)


def measure(args, frozen, fp_estimator="purity_mixture"):
    ing = TF.build_loa_ingredients(args, frozen)
    if fp_estimator == "loa0":
        # CONFIG-ONLY override (no estimator edit): switch the DLA-tier FP from the
        # driver's hard-set purity_mixture to the loa-0 forest-FP product. build_loa_
        # ingredients already set cfg.n_sl_prod = the real-LOA op sightline count (from
        # build_pathlength), so vol_scale = n_sl_prod/n_sl_loa0 uses the REAL denominator
        # and tracks the veto's sightline drop. make_fp_model stashes cfg._loa0_fp, which
        # _forward_fp_terms reads; run_measurement never resets fp_estimator/_loa0_fp.
        cfg = ing["cfg"]
        cfg.fp_estimator = "loa0"
        cfg.loa0_product_path = LOA0_LYAONLY
        # PRE-FLIGHT provenance guards (task #6/#7): the loa0 product must be the
        # lya-only (1025) headline product AND the resolved molly the lya_only-nhi195
        # matrix — hard-fail on any silent substitution.
        preflight_loa0_product(LOA0_LYAONLY, cfg, args.molly_tsv)
        rho = make_rho_interpolator(ing["mm"])
        loa0_model, _ = make_fp_model(cfg, ing["cat_cut"], ing["op_mask"], rho)
        ing["fp_model"] = loa0_model
        assert getattr(cfg, "_loa0_fp", None) is not None, "loa0 FP not attached"
        # n_sl_prod guard (task #5): the estimator silently falls back to the product's
        # stored MOCK n_sl_prod if cfg.n_sl_prod is None. build_loa_ingredients sets
        # cfg.n_sl_prod = int(n_sl) (the REAL-LOA op sightline count); assert the FP model
        # actually picked that up so the FP volume-scale can never silently mis-scale.
        assert loa0_model.n_sl_prod == ing["n_sl"], (
            f"n_sl_prod mismatch {loa0_model.n_sl_prod} != {ing['n_sl']} — the loa0 FP "
            f"volume-scale did not pick up the real-LOA op sightline count (silent "
            f"fallback to the product's stored mock n_sl_prod).")
        print(f"  [FP=loa0] product={os.path.basename(LOA0_LYAONLY)} "
              f"n_sl_loa0={loa0_model.n_sl_loa0:.0f} n_sl_prod={loa0_model.n_sl_prod:.0f} "
              f"vol_scale={loa0_model.vol_scale:.3f}")
    res = TF.run_measurement(args, ing, args._limits, args.seed, frozen=frozen)
    out = {"n_op_sl": int(res["n_op_sl"]),
           "n_op_det": int(res["n_op_detections"]),
           "X_tot": np.asarray(res["X_tot"], float),
           "zbins": np.asarray(res["zbins"], float)}
    for l in args._limits:
        out[("dndx", l)] = float(res["dndx"][l]["integrated"]["MAP"])
        out[("omega", l)] = float(res["omega"][l]["integrated"]["MAP"])
        out[("dndx_perz", l)] = [res["dndx"][l]["perz"][k]["MAP"]
                                 for k in range(res["n_zc"])]
        out[("omega_perz", l)] = [res["omega"][l]["perz"][k]["MAP"]
                                  for k in range(res["n_zc"])]
    return out


def residual_broad_leak(vac_path, limits):
    """After the veto, on-VAC residual broad-BAL Omega leak in the clean op catalog.
    By construction (union veto) this is 0; we verify empirically on the post-veto set."""
    v = fitsio.read(vac_path, columns=["TARGETID", "BI_CIV", "AI_CIV", "ERR_AI_CIV",
                                       "VMIN_CIV_450", "VMAX_CIV_450"])
    tid = np.asarray(v["TARGETID"], np.int64)
    bi = np.asarray(v["BI_CIV"], float)
    ai = np.asarray(v["AI_CIV"], float)
    eai = np.asarray(v["ERR_AI_CIV"], float)
    vmin = np.asarray(v["VMIN_CIV_450"], float)
    vmax = np.asarray(v["VMAX_CIV_450"], float)
    widest = np.abs(vmax - vmin).max(axis=1)
    sigai = (ai > 0) & (eai > 0) & (ai > 3 * eai)
    broad = (widest > 2000.0) & sigai
    veto = broad | (bi > 0)
    veto_set = set(tid[veto].tolist())
    broad_set = set(tid[broad].tolist())

    d = fitsio.read(TF._LOA_CAT.rstrip("/") + "/dlacat-loa-main-dark-v1.fits",
                    columns=["TARGETID", "NHI", "Z_DLA", "Z_QSO", "SNR_REDSIDE",
                             "P_DLA", "DLAFLAG"])
    dt = np.asarray(d["TARGETID"], np.int64)
    nhi = np.asarray(d["NHI"], float)
    zd = np.asarray(d["Z_DLA"], float); zq = np.asarray(d["Z_QSO"], float)
    snr = np.asarray(d["SNR_REDSIDE"], float); p = np.asarray(d["P_DLA"], float)
    fl = np.asarray(d["DLAFLAG"], int)
    lr = 1215.67 * (1 + zd) / (1 + zq)
    op = (snr > 2) & (p > 0.99) & (lr >= 1025.0) & (fl == 0)
    kept = op & np.array([int(t) not in veto_set for t in dt])   # post-veto clean set
    res = {}
    for lim in limits:
        m = kept & (nhi >= lim)
        w = 10.0 ** nhi[m]
        stillbroad = np.array([int(dt[i]) in broad_set for i in np.where(m)[0]])
        res[lim] = 100.0 * (w[stillbroad].sum() / w.sum() if w.sum() > 0 else 0.0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mc", type=int, default=12,
                    help="MC band draws (MAP point is n_mc-independent).")
    ap.add_argument("--fp", choices=["purity_mixture", "loa0"], default="loa0",
                    help="DLA-tier FP estimator (PI headline = loa0).")
    a = ap.parse_args()

    args = default_args()
    args.n_mc = a.n_mc
    fp = a.fp
    os.makedirs(args.out, exist_ok=True)
    limits = args._limits

    print("=" * 78)
    print(f"BROAD-TROUGH BAL VETO — real-LOA Track-C headline (FP={fp})")
    print("=" * 78)

    # frozen 2LPT-0 calibration (independent of the LOA bal_cat AND the FP estimator) — ONCE.
    frozen = TF.build_frozen_calibration(args)
    args.molly_tsv = frozen["molly_tsv"]

    # ---- STEP 1 (gate): un-vetoed baseline (staged bal_cat = production BI_CIV>0) ----
    print(f"\n[BASELINE] un-vetoed headline (staged bal_cat = BI_CIV>0, FP={fp}) ...")
    base = measure(args, frozen, fp_estimator=fp)

    # ---- build augmented bal_cat (union of BI>0 and broad-trough) ----
    broad_tids = derive_broadtrough_tids()
    aug_path = os.path.join(args.out, "bal_cat_broadtrough_augmented.fits")
    n_orig, n_broad, n_new, n_union = build_augmented_bal(
        args.loa_bal, broad_tids, aug_path)
    print(f"\n[VETO SET] staged BI>0={n_orig}  broad-trough={n_broad}  "
          f"NEW added={n_new}  union={n_union}  -> {aug_path}")

    # ---- STEP 2: vetoed measurement (augmented bal_cat) ----
    print(f"\n[VETOED] broad-trough veto applied symmetrically (num+denom, FP={fp}) ...")
    args.loa_bal = aug_path
    vet = measure(args, frozen, fp_estimator=fp)

    # ---- pathlength (Delta X) symmetry verification ----
    Xb = base["X_tot"].sum(); Xv = vet["X_tot"].sum()
    dX_frac = 100.0 * (Xb - Xv) / Xb
    n_sl_drop = base["n_op_sl"] - vet["n_op_sl"]
    print("\n" + "=" * 78)
    print("PATHLENGTH (Delta X) SYMMETRY")
    print(f"  n_op_sl  baseline={base['n_op_sl']}  vetoed={vet['n_op_sl']}  "
          f"dropped={n_sl_drop}")
    print(f"  n_op_det baseline={base['n_op_det']}  vetoed={vet['n_op_det']}  "
          f"dropped={base['n_op_det'] - vet['n_op_det']}")
    print(f"  Sum Delta X baseline={Xb:.2f}  vetoed={Xv:.2f}  "
          f"removed={Xb - Xv:.2f} ({dX_frac:.2f}%)")
    perz_rm = [100.0 * (bx - vx) / bx for bx, vx in zip(base["X_tot"], vet["X_tot"])]
    print("  per-zbin Delta X removed %: "
          + ", ".join(f"{r:.2f}" for r in perz_rm))

    # ---- headline deltas ----
    print("\n" + "=" * 78)
    print("HEADLINE dN/dX & Omega (integrated, z 2-3.5) — MAP point")
    print(f"  {'quantity':>16s}{'baseline':>14s}{'vetoed':>14s}{'delta':>14s}{'pct':>9s}")
    for l in limits:
        for q in ("dndx", "omega"):
            b = base[(q, l)]; w = vet[(q, l)]
            print(f"  {q+' >='+str(l):>16s}{b:>14.6g}{w:>14.6g}"
                  f"{w-b:>14.3g}{100*(w-b)/b:>8.2f}%")

    # ---- residual on-VAC broad-BAL Omega leak after veto ----
    leak = residual_broad_leak(V2_VAC, limits)
    print("\nResidual on-VAC broad-BAL Omega leak AFTER veto (should be ~0):")
    for l in limits:
        print(f"  N>={l}: {leak[l]:.4f}%")

    # aggregate JSON to scratch (no repo write)
    import json
    rec = dict(
        code_commit=_git_commit(),
        fp_estimator=fp, limits=list(limits), n_mc=args.n_mc,
        veto=dict(n_bi=n_orig, n_broad=n_broad, n_new=n_new, n_union=n_union),
        dX=dict(sum_base=float(Xb), sum_vet=float(Xv), frac_removed_pct=float(dX_frac),
                n_sl_base=base["n_op_sl"], n_sl_vet=vet["n_op_sl"],
                per_zbin_removed_pct=[float(100*(bx-vx)/bx)
                                      for bx, vx in zip(base["X_tot"], vet["X_tot"])]),
        headline={f"{q}_{l}": dict(base=base[(q, l)], vet=vet[(q, l)],
                                   delta=vet[(q, l)] - base[(q, l)],
                                   pct=100*(vet[(q, l)]-base[(q, l)])/base[(q, l)])
                  for l in limits for q in ("dndx", "omega")},
        residual_leak_pct={str(l): leak[l] for l in limits},
    )
    jpath = os.path.join(args.out, f"broadtrough_veto_result_{fp}.json")
    with open(jpath, "w") as fh:
        json.dump(rec, fh, indent=2, default=float)
    print(f"\n[DONE] aggregate JSON -> {jpath}")


if __name__ == "__main__":
    main()
