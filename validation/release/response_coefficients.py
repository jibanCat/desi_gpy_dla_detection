#!/usr/bin/env python
"""response_coefficients.py -- recover the FITTED COEFFICIENTS of the response
finalists E and B by a deterministic refit, for the Zenodo release.

Why this exists
---------------
The ladder objects of record are the EVALUATED tensors ``rows_unit`` /
``Mg_<var>_<fam>.npz``; the fitting code never persisted the parameter vectors
(E: alpha (R x P) and the phi_r basis (R x C), plus the R1c base kernel it
deforms; B: beta (6 x P)).  A release that ships only a 16x8x3x29 table cannot
be re-fitted or extrapolated by a reader, so the coefficients are recovered
POST HOC by re-running the SAME fitting code on the SAME full 2LPT-0 sample
with the same seeds and starts, and the recovered coefficients are then
VERIFIED by re-evaluating them and comparing to the delivered tensors.

Discipline
----------
* the delivered candidate files are NEVER modified or overwritten;
* if the refit does not reproduce the delivered rows to the tolerance, the
  coefficients are still shipped but labelled "reproduction to X" and the
  delivered tensors remain the objects of record;
* the E base kernel is labelled EXACTLY as built -- the R1c spec actually used
  is ``estimator="sample"`` with ``marginalise={"n_quad": 17, "deg": 4,
  "weight": "flat"}``, i.e. NOT the ML / deg-3 R1c of record elsewhere.

VALIDATION-ONLY.  Mock calibration only, no sampler, no real data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_CAND = os.path.join(_REPO, "validation", "absorber_ladder",
                     "response_review", "candidates")
_RESP = os.path.join(_REPO, "validation", "absorber_ladder", "response")
for _p in (_CAND, _RESP, _REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hashutil import sha256_file                              # noqa: E402

SCHEMA = "zenodo_release/response_coefficients/v1"
DEFAULT_TOL = 1e-8


def refit(products, tol=DEFAULT_TOL, variants=("E", "B")):
    import candlib as CL
    import build_variants as BV
    import run_candidates as RC

    geom = CL.load_geom(
        os.path.join(products, "support", "empirical_ops_2lpt0_A0.npz"),
        os.path.join("/scratch/cavestru_root/cavestru0/mfho/"
                     "fp_ladder_2026-09-12/packs", "scanpack_2lpt0_b300.npz"))
    ev = CL.load_events(os.path.join(products, "response",
                                     "calib_events_2lpt0.npz"), geom)
    N_ref = float(np.load(RC.ADOPTED_NPZ, allow_pickle=True)["N_ref"])
    full = np.ones(ev["n"], bool)
    cand = os.path.join(products, "response_review", "candidates")

    out = {"schema": SCHEMA, "n_events": int(ev["n"]), "N_ref": N_ref,
           "geometry": {k: int(geom[k]) for k in ("B", "S", "K", "C")},
           "quadrature": {
               "n_quad_u": int(CL.N_QUAD_U), "n_quad_v": int(CL.N_QUAD_V),
               "weight": "flat (predeclaration I2b)",
               "u": "linspace over the latent bin, interior nodes, minus "
                    "N_REF_U = %g" % CL.N_REF_U,
               "v": "linspace in log10(S/N) over the stratum, interior nodes, "
                    "minus V_REF = %g" % CL.V_REF,
               "snr_bot_floor": float(CL.SNR_BOT_FLOOR),
               "snr_top_cap": float(CL.SNR_TOP_CAP),
               "n_restart": int(CL.N_RESTART), "ridge": float(CL.RIDGE),
               "base_floor": float(CL.BASE_FLOOR)},
           "variants": {}, "arrays": {}}

    base_rows = None
    base_obj = None
    if "E" in variants:
        t0 = time.time()
        base_rows, base_obj = RC.parametric_rows(ev, full, geom,
                                                 BV.specs()["R1c"], N_ref)
        out["E_base"] = {
            "role": "the parametric kernel E deforms (log-base of the "
                    "low-rank residual)",
            "label_as_built": "R1c with estimator='sample' and "
                              "marginalise={'n_quad': 17, 'deg': 4, "
                              "'weight': 'flat'} -- NOT the ML / deg-3 R1c "
                              "used elsewhere; label it exactly as built",
            "spec": {k: v for k, v in BV.specs()["R1c"].items()
                     if k != "defect"},
            "n_coef": base_obj.get("n_coef"),
            "wall_s": time.time() - t0,
            "released_as": "base_rows_R1c in response_coefficients_E.npz "
                           "(the kernel's own %s coefficients are a moment "
                           "model whose evaluation needs the internal "
                           "respfit/opmetrics code; the evaluated base tensor "
                           "is therefore released alongside them)"
                           % (base_obj.get("n_coef", {})
                              .get("total_moments") if isinstance(
                                  base_obj.get("n_coef"), dict) else "?"),
        }
        for key in ("coef", "coefs", "beta", "moment_coef", "theta"):
            if key in base_obj:
                out["arrays"]["base_%s" % key] = np.asarray(base_obj[key],
                                                            float)

    for var in variants:
        t0 = time.time()
        fit = RC.fit_candidate(var, ev, full, geom, N_ref,
                               base_rows=base_rows if var == "E" else None)
        par = fit["par"]
        delivered = np.asarray(np.load(os.path.join(
            cand, "rows_%s.npz" % var), allow_pickle=True)["rows"], float)
        dev = float(np.max(np.abs(np.asarray(fit["rows"], float) - delivered)))
        blob = {
            "family": var,
            "form": ("Q = normalise_c[ base(c) * exp( sum_r a_r(x) phi_r(c) ) ]"
                     " with base = the R1c kernel row (E)"
                     if var == "E" else
                     "Q = normalise_c[ bin mass of  pi*SplitNormal(N+d1, w1, "
                     "kappa) + (1-pi)*Normal(N+d1+d2, w2) ]  (B)"),
            "design": ("[1, u, v, 1[K=1], 1[K=2]] (kind 'e', P=5)"
                       if var == "E" else
                       "[1, u, u^2, v, 1[K=1], 1[K=2]] (kind 'full', P=6)"),
            "design_covariates": {
                "u": "N_true - %g" % CL.N_REF_U,
                "v": "log10(S/N) - %g" % CL.V_REF,
                "K": "coarse redshift block index"},
            "standardisation": "the design is standardised with the stored "
                               "xmu/xsd BEFORE the coefficients act",
            "rank_R": int(par.get("R", 0)) or None,
            "train_loss": par.get("train_loss"),
            "start_objectives": par.get("start_objectives"),
            "start_spread": par.get("start_spread"),
            "nit": par.get("nit"),
            "nominal_dof_as_stored": fit["meta"].get("nominal_dof"),
            "wall_s": time.time() - t0,
            "reproduces_delivered_rows_to": dev,
            "reproduces_within_tolerance": bool(dev <= tol),
            "tolerance": tol,
        }
        if var == "E":
            blob["nominal_dof_corrected"] = (
                int(fit["meta"].get("nominal_dof_own", 0))
                + int((base_obj.get("n_coef") or {}).get("total_moments", 0)))
            blob["nominal_dof_note"] = (
                "the stored nominal_dof=%s is STALE: E's own identifiable "
                "dimension (%s) plus the %s coefficients of the R1c base it "
                "deforms is %s"
                % (fit["meta"].get("nominal_dof"),
                   fit["meta"].get("nominal_dof_own"),
                   (base_obj.get("n_coef") or {}).get("total_moments"),
                   blob["nominal_dof_corrected"]))
            out["arrays"]["E_alpha"] = np.asarray(par["alpha"], float)
            out["arrays"]["E_phi"] = np.asarray(par["phi"], float)
            out["arrays"]["E_sv"] = np.asarray(par["sv"], float)
            out["arrays"]["E_xmu"] = np.asarray(par["xmu"], float)
            out["arrays"]["E_xsd"] = np.asarray(par["xsd"], float)
            out["arrays"]["base_rows_R1c"] = np.asarray(base_rows, float)
        else:
            out["arrays"]["B_beta"] = np.asarray(par["beta"], float)
            out["arrays"]["B_xmu"] = np.asarray(par["xmu"], float)
            out["arrays"]["B_xsd"] = np.asarray(par["xsd"], float)
        out["arrays"]["%s_rows_refit" % var] = np.asarray(fit["rows"], float)
        out["variants"][var] = blob
        print("[coef] %s refit in %.0fs; max|Q_refit - Q_delivered| = %.3e"
              % (var, blob["wall_s"], dev), flush=True)

    out["arrays"]["ntrue_edges"] = np.asarray(geom["ntrue"], float)
    out["arrays"]["nhat_edges"] = np.asarray(geom["nhat"], float)
    out["arrays"]["snr_edges"] = np.asarray(geom["snr"], float)
    return out


def write(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    arrays = res.pop("arrays")
    written = []
    for var in res["variants"]:
        keys = {k: v for k, v in arrays.items()
                if k.startswith("%s_" % var) or k.startswith("base_")
                or k in ("ntrue_edges", "nhat_edges", "snr_edges")}
        if var != "E":
            keys = {k: v for k, v in keys.items()
                    if not k.startswith("base_")}
        p = os.path.join(out_dir, "response_coefficients_%s.npz" % var)
        np.savez_compressed(p, **keys)
        written.append(p)
    meta = dict(res)
    meta["files"] = [os.path.basename(p) for p in written]
    meta["file_sha256"] = {os.path.basename(p): sha256_file(p)
                           for p in written}
    mp = os.path.join(out_dir, "response_coefficients.json")
    with open(mp, "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")
    return written + [mp], meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--products", required=True)
    ap.add_argument("--out", required=True, help="release/response directory")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL)
    ap.add_argument("--variants", default="E,B")
    a = ap.parse_args(argv)
    res = refit(a.products, tol=a.tol,
                variants=tuple(v for v in a.variants.split(",") if v))
    paths, meta = write(res, a.out)
    for var, blob in meta["variants"].items():
        print("%s: reproduction %.3e (tolerance %.0e) -> %s"
              % (var, blob["reproduces_delivered_rows_to"], a.tol,
                 "WITHIN" if blob["reproduces_within_tolerance"]
                 else "OUTSIDE -- delivered tensors remain the objects of "
                      "record"))
    for p in paths:
        print("  wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
