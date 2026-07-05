#!/usr/bin/env python
"""run_loa0_headline_full.py — produce the FULL real-LOA loa0 headline JSON in the
SAME schema as track_c_tf_loa.json (measurement[20.0/20.3] per-z + integrated with
MC band + perz_fN f(N) arrays + zbins + metadata), for the paper figures + provenance.

This is a CONFIG-ONLY FP-model override of committed job 52266001 (which ran
purity_mixture): it reuses track_c_tf_loa.py's own build_frozen_calibration /
build_loa_ingredients / run_measurement on the IDENTICAL frozen ingredients (same
forward-response kernel, z-resolved completeness g(N,z), molly C/rho, lam_rf_min=1025,
zbins, real dlacat, cut bundle), and the ONLY change is cfg.fp_estimator "purity_mixture"
-> "loa0" (+ cfg.loa0_product_path) via the estimator's own make_fp_model dispatch.
No estimator edit; no re-inference; gpy_dla_detection/ + cddf_catalog_hbi.py untouched.

Real-LOA numbers go to SCRATCH only (never the code repo). Aggregate values only.

Env: conda gpdla; OMP/OPENBLAS/MKL_NUM_THREADS=1; HDF5_USE_FILE_LOCKING=FALSE.
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
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(_HERE)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# reuse the untracked veto helper (default_args + measure/loa0-override) and the driver
_spec = importlib.util.spec_from_file_location(
    "bt_helper", os.path.join(_HERE, "apply_broadtrough_veto_headline.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
TF = H.TF

OUT_JSON = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/tf_loa/"
            "track_c_tf_loa_loa0.json")


def _git_commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mc", type=int, default=120, help="MC band draws (match job 52266001).")
    a = ap.parse_args()

    args = H.default_args()
    # MATCH job 52266001 config exactly (only FP differs):
    args.zbins = "2.0,2.5,3.0,3.5,4.0,4.25"
    args.v2_z_fit_hi = 3.5
    args.n_mc = a.n_mc
    limits = (20.0, 20.3)
    args._limits = limits
    args.report_limits = "20.0,20.3"

    t0 = time.time()
    frozen = TF.build_frozen_calibration(args)
    args.molly_tsv = frozen["molly_tsv"]

    # build ingredients (un-vetoed / staged bal_cat = BI_CIV>0) + loa0 FP override
    ing = TF.build_loa_ingredients(args, frozen)
    cfg = ing["cfg"]
    cfg.fp_estimator = "loa0"
    cfg.loa0_product_path = H.LOA0_LYAONLY
    from CDDF_analysis.hbi.cddf_catalog_hbi import make_fp_model, make_rho_interpolator
    rho = make_rho_interpolator(ing["mm"])
    loa0_model, _ = make_fp_model(cfg, ing["cat_cut"], ing["op_mask"], rho)
    ing["fp_model"] = loa0_model
    assert getattr(cfg, "_loa0_fp", None) is not None
    print(f"[loa0] product n_sl_loa0={loa0_model.n_sl_loa0:.0f} "
          f"n_sl_prod={loa0_model.n_sl_prod:.0f} vol_scale={loa0_model.vol_scale:.3f}")

    res = TF.run_measurement(args, ing, limits, args.seed, frozen=frozen)
    wall = time.time() - t0

    # ---- assemble the SAME out_json schema as track_c_tf_loa.main() ----
    out_json = dict(
        metadata=dict(
            fp_estimator="loa0",
            source_job_id="52266001",
            provenance=("config-only FP-model override of job 52266001 (purity_mixture); "
                        "frozen ingredients identical, only cfg.fp_estimator=loa0 + "
                        "loa0_product_path differ"),
            loa0_product=H.LOA0_LYAONLY,
            n_mc=args.n_mc, seed=args.seed, limits=list(limits),
            resp_kind="forward", forward_model=args.forward_model,
            molly_tsv=args.molly_tsv, loa_cat=args.loa_cat,
            n_op_detections=res["n_op_detections"], n_op_sl=res["n_op_sl"],
            consistency_err=res["consistency_err"], v2_z_fit_hi=float(args.v2_z_fit_hi),
            z_extrapolated=[bool(x) for x in np.asarray(res.get("z_extrapolated", []))],
            z_thin=[bool(x) for x in np.asarray(res.get("z_thin", []))],
            truth_counts_perz=res.get("truth_counts_perz"),
            max_truth_z=float(res.get("max_truth_z", float("nan"))),
            support_limit=float(res.get("support_limit", max(limits))),
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

    with open(OUT_JSON, "w") as fh:
        json.dump(out_json, fh, indent=2, default=float)
    print(f"[loa0] wrote {OUT_JSON}  ({wall:.0f}s)")
    # aggregate echo
    for l in limits:
        di = res["dndx"][l]["integrated"]["MAP"]; oi = res["omega"][l]["integrated"]["MAP"]
        print(f"  >= {l}: integ dN/dX={di:.4f}  1e3*Om={1e3*oi:.3f}")


if __name__ == "__main__":
    main()
