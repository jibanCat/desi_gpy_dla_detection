#!/usr/bin/env python
"""run_paper1_lowz_bins.py — the FINAL low-z Paper-1 bins B1–B5, actually re-run.

PI ruling 2026-08-15 (evening): mechanically execute the final low-z Paper-1
redshift architecture

    B1 [2.15,2.35)  B2 [2.35,2.56)  B3 [2.56,2.96)  B4 [2.96,3.40)  B5 [3.40,3.80)

(BH [3.80,5.00) comes from the separate high-z catalog run, track_c_tf_hz.py).
This is a CONFIG-ONLY re-bin RE-RUN of the loa0 headline configuration
(run_loa0_headline_full.py pattern): same frozen ingredients, same canonical
sample (S2N_RED>2, P_DLA>0.99, DLAFLAG==0, lya_only [1025,1216], BI_CIV>0
bal_cat, z_qso in (2.0,4.25) strict), same Molly C/rho + frozen g(N,z) +
forward kernel + loa0 FP; the ONLY changes are zbins -> the B-edges and
v2_z_fit_hi -> 3.8 (the fine z grid must cover the bin ceiling; interior
edges 2.56/2.96 nest inside the frozen response z-cells by design).

NOT a relabeling of the old [2.0,2.5,3.0,3.5,4.0,4.25] artifacts — the per-z
pathlength, per-bin dN/dX/Omega and the band are regenerated under the new
edges. The old artifacts remain untouched.

Estimand: DIAGNOSTIC_RECENTERED / paper_facing=false (the 2026-07-28 band
retirement applies until the forward-closure successor lands) — B1–B5 rows
are CANDIDATE table inputs exactly like BH.

Usage: python CDDF_analysis/hbi/run_paper1_lowz_bins.py [--fp loa0|pm]
       [--n-mc 2000] [--force]
Real values go to SCRATCH only. Env gpdla; *_NUM_THREADS=1.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_ARB = os.path.join(_REPO, "CDDF_analysis", "diagnostics", "bal_metal_fp", "arbiter")
_spec = importlib.util.spec_from_file_location(
    "bt_helper", os.path.join(_ARB, "apply_broadtrough_veto_headline.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
TF = H.TF

OUT_DIR = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/"
           "tf_loa_paper1bins")
PAPER1_ZBINS = "2.15,2.35,2.56,2.96,3.40,3.80"


def _git_commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp", choices=["loa0", "pm"], default="loa0")
    ap.add_argument("--n-mc", type=int, default=2000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"track_c_tf_loa_paper1bins_{a.fp}.json")
    if os.path.exists(out_path) and not a.force:
        raise SystemExit(f"refusing to overwrite {out_path} (pass --force).")

    args = H.default_args()
    args.zbins = PAPER1_ZBINS
    args.v2_z_fit_hi = 3.8
    args.n_mc = a.n_mc
    args.out = OUT_DIR
    args.report_out = os.path.join(OUT_DIR, f"_report_{a.fp}.md")
    limits = (20.0, 20.3)
    args._limits = limits
    args.report_limits = "20.0,20.3"

    t0 = time.time()
    frozen = TF.build_frozen_calibration(args)
    args.molly_tsv = frozen["molly_tsv"]
    ing = TF.build_loa_ingredients(args, frozen)
    cfg = ing["cfg"]
    if a.fp == "loa0":
        from CDDF_analysis.hbi.cddf_catalog_hbi import (
            make_fp_model, make_rho_interpolator)
        cfg.fp_estimator = "loa0"
        cfg.loa0_product_path = H.LOA0_LYAONLY
        H.preflight_loa0_product(H.LOA0_LYAONLY, cfg, args.molly_tsv)
        rho = make_rho_interpolator(ing["mm"])
        loa0_model, _ = make_fp_model(cfg, ing["cat_cut"], ing["op_mask"], rho)
        ing["fp_model"] = loa0_model
        assert loa0_model.n_sl_prod == ing["n_sl"], "loa0 n_sl_prod guard"

    res = TF.run_measurement(args, ing, limits, args.seed, frozen=frozen)
    wall = time.time() - t0
    out_json = dict(
        metadata=dict(
            estimand="DIAGNOSTIC_RECENTERED", paper_facing=False,
            status="CANDIDATE Paper-1 low-z bins B1-B5 (PI-approved re-bin re-run 2026-08-15)",
            sample="P1_PRIMARY_LYA", fp_estimator=cfg.fp_estimator,
            contract="CANONICAL_PURITY_COMPLETENESS_CONTRACT v1.1",
            n_mc=args.n_mc, seed=args.seed, limits=list(limits),
            resp_kind="forward", molly_tsv=args.molly_tsv,
            loa_cat=args.loa_cat, v2_z_fit_hi=3.8,
            n_op_detections=res["n_op_detections"], n_op_sl=res["n_op_sl"],
            consistency_err=res["consistency_err"],
            z_extrapolated=[bool(x) for x in np.asarray(res.get("z_extrapolated", []))],
            truth_counts_perz=res.get("truth_counts_perz"),
            max_truth_z=float(res.get("max_truth_z", float("nan"))),
            wallclock_s=float(wall), code_commit=_git_commit()),
        measurement={
            str(l): dict(
                dndx=dict(
                    perz=[res["dndx"][l]["perz"][k] for k in range(res["n_zc"])],
                    integrated=res["dndx"][l]["integrated"]),
                omega=dict(
                    perz=[res["omega"][l]["perz"][k] for k in range(res["n_zc"])],
                    integrated=res["omega"][l]["integrated"]),
            ) for l in limits},
        zbins=list(map(float, res["zbins"])))
    out_json["perz_fN"] = TF.assemble_perz_fN(res, limits)
    with open(out_path, "w") as fh:
        json.dump(out_json, fh, indent=2, default=float)
    print(f"[paper1bins:{a.fp}] wrote {out_path} ({wall:.0f}s)")
    for l in limits:
        pz = [res["dndx"][l]["perz"][k]["MAP"] for k in range(res["n_zc"])]
        print(f"  >= {l}: perz dN/dX MAP = {[round(p,4) for p in pz]}")


if __name__ == "__main__":
    main()
