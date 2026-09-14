#!/usr/bin/env python
"""response_release.py -- build the Zenodo response product (PI §19).

The forward response operator is TWO distinct conditional objects (§3):

    M(c | b, s, K) = phi(b, s, K) * Q(c | b, s, K)

``Q`` is the in-grid row SHAPE (sums to one over the 29 observed N_hat bins)
and ``phi`` the in-grid HAD MASS.  Both finalists of the closed response search
are released as CANDIDATES -- E (low-rank residual deformation of the R1c
parametric kernel) and B (split-normal / normal mixture) -- because the model
of record is a PI adoption decision that has not been made.

What is released is the TABULATED Q on the calibration grid, not the internal
optimiser's parameter vector: the fitted alpha/phi_r of E and the 36 mixture
coefficients of B are not persisted by the fitting code, and the tabulated rows
ARE the object the inference consumes.  The 6-coefficient smooth phi IS a
closed-form object and is released as coefficients plus its basis, with a
standalone evaluator that rebuilds it exactly.

VALIDATION-ONLY: frozen mock calibration in, release tree out.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "standalone"))

from hashutil import (verify_against_sums, write_sha256sums,
                      parse_accept_stale,
                      ManifestIntegrityError)                  # noqa: E402
import evaluate_response as ER                                 # noqa: E402

SCHEMA = "zenodo_release/response/v1"
FINALISTS = ("E", "B")
CAL_FAMILY = "2lpt0"
FAMILIES = ("2lpt0", "london0", "saclay0")
ROWSUM_ATOL = 1e-12


def _git_stamp(repo):
    def _run(args):
        try:
            return subprocess.check_output(args, cwd=repo,
                                           stderr=subprocess.DEVNULL
                                           ).decode().strip()
        except Exception:                                    # pragma: no cover
            return None
    dirty = _run(["git", "status", "--porcelain"])
    return {"commit": _run(["git", "rev-parse", "HEAD"]),
            "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(dirty) if dirty is not None else None}


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    return path


def _vmid(snr_edges, n_quad_v=5, v_ref=0.7, bot=1.0, top=40.0):
    """The stratum flat-quadrature midpoint of log10(S/N) - V_REF.

    Reproduces ``candlib.quad_grid``'s v nodes averaged over the stratum: the
    covariate the 6-coefficient smooth phi was fitted on (predeclaration I2b,
    a FLAT within-stratum convention, not an occupancy median).
    """
    out = []
    for s in range(len(snr_edges) - 1):
        lo = max(float(snr_edges[s]), bot)
        hi = float(snr_edges[s + 1])
        hi = top if not np.isfinite(hi) else min(hi, top)
        hi = max(hi, lo * 1.0001)
        out.append(float(np.mean(np.linspace(np.log10(lo), np.log10(hi),
                                             n_quad_v + 2)[1:-1] - v_ref)))
    return np.asarray(out, float)


def build(products, out_root, repo=None, finalists=FINALISTS,
          accept_stale=None):
    repo = repo or os.path.abspath(os.path.join(_HERE, "..", ".."))
    out = os.path.join(out_root, "response")
    os.makedirs(out, exist_ok=True)
    sums_cache = {}
    accept_stale = accept_stale or {}
    cand = os.path.join(products, "response_review", "candidates")

    meta = {
        "schema": SCHEMA,
        "product": "DESI GP-DLA Paper-1 forward response operator "
                   "M = phi(b,s,K) * Q(c|b,s,K)",
        "status": "CANDIDATES -- the response model of record is PENDING PI "
                  "adoption (PI ruling 2026-09-13d §1/§15: the search is "
                  "CLOSED to new families and the ladder adjudicates E vs B)",
        "finalists": list(finalists),
        "calibration_family": CAL_FAMILY,
        "calibration_events": None,
        "fitted_to_real_data": False,
        "grid": {}, "families": {}, "checks": {},
        "phi": {}, "cross_validation": {}, "anti_tautology": {},
        "evaluation_code": {"file": "evaluate_response.py",
                            "imports": ["numpy"],
                            "entry_points": ["response_row", "phi_at",
                                             "phi_smooth", "operator_M",
                                             "row_sum_check"],
                            "note": "Q evaluation is an exact cell lookup -- "
                                    "the same operation the sampler performs; "
                                    "phi_smooth is closed form"},
        "builder": "validation/release/response_release.py",
        "git": _git_stamp(repo),
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "inputs": {},
    }

    # ---- inputs and their hashes -----------------------------------------
    ce = os.path.join(products, "response", "calib_events_%s.npz" % CAL_FAMILY)
    d_ce, st_ce = verify_against_sums(ce, sums_cache, accept_stale)
    if st_ce == "MISSING":
        raise ManifestIntegrityError("calibration events missing: %s" % ce)
    ce_prov = json.loads(str(np.load(ce, allow_pickle=True)["provenance"]))
    meta["inputs"]["calib_events"] = {"path": ce, "sha256": d_ce,
                                      "sha256sums": st_ce,
                                      "n_events": ce_prov.get("n_events"),
                                      "n_uniq_tids": ce_prov.get("n_uniq_tids")}
    meta["calibration_events"] = ce_prov.get("n_events")

    for name, rel in (("candidate_cv_table",
                       os.path.join(cand, "candidate_cv_table.json")),
                      ("antitautology_AB",
                       os.path.join(cand, "antitautology_AB.json")),
                      ("antitautology_results",
                       os.path.join(products, "response_review",
                                    "antitautology",
                                    "antitautology_results.json"))):
        dg, stt = verify_against_sums(rel, sums_cache, accept_stale)
        if stt == "MISSING":
            raise ManifestIntegrityError("response input missing: %s" % rel)
        meta["inputs"][name] = {"path": rel, "sha256": dg, "sha256sums": stt}

    cv_tab = json.load(open(meta["inputs"]["candidate_cv_table"]["path"]))
    anti = json.load(open(meta["inputs"]["antitautology_results"]["path"]))
    meta["cross_validation"] = {
        "protocol": "full-distribution 2-fold CV on the %d matched 2LPT-0 "
                    "calibration events; sealed opening rule %s"
                    % (ce_prov.get("n_events"),
                       cv_tab.get("sealed_rule_sha256")),
        "implementation_predeclaration_sha256":
            cv_tab.get("implementation_predeclaration_sha256"),
        "per_family": {k: {kk: v[kk] for kk in
                           ("ll_fixed", "row_kl_wmean", "row_kl_wmean_ge200",
                            "row_dev_per_dof_wmean", "d_mean_wmean")
                           if kk in v}
                       for k, v in cv_tab.get("table", {}).items()},
    }
    meta["anti_tautology"] = {
        "authority": anti.get("authority"),
        "tests": [k for k in anti if k.startswith("test_")],
        "mutation_controls": bool(anti.get("MUTATION_CONTROLS")),
        "statement": "the estimand is the CONDITIONAL Q(N_hat | N_true, S/N, z),"
                     " not the joint p(N_true, N_hat); E is invariant under "
                     "calibration-population reweighting and B has the smaller "
                     "equalised-occupancy sensitivity (PI §5)",
    }

    # ---- per finalist -----------------------------------------------------
    edges = None
    phi_written = False
    for fam in finalists:
        src = os.path.join(cand, "Mg_%s_%s.npz" % (fam, CAL_FAMILY))
        rowsrc = os.path.join(cand, "rows_%s.npz" % fam)
        dg, stt = verify_against_sums(src, sums_cache, accept_stale)
        if stt == "MISSING":
            raise ManifestIntegrityError("response candidate missing: %s" % src)
        dg2, stt2 = verify_against_sums(rowsrc, sums_cache, accept_stale)
        if stt2 == "MISSING":
            raise ManifestIntegrityError("row product missing: %s" % rowsrc)

        Z = np.load(src, allow_pickle=True)
        RW = np.load(rowsrc, allow_pickle=True)
        prov = json.loads(str(Z["provenance"]))
        coefmeta = json.loads(str(Z["coefficients"]))
        Q = np.asarray(Z["rows_unit"], float)               # (B, S, K, C)
        phi = np.asarray(Z["phi_bsK"], float)
        phi_s = np.asarray(Z["phi_bsK_smooth"], float)
        phi_fam = np.asarray(Z["phi_bsK_family_measured"], float)
        phi_ref = np.asarray(Z["phi_ref_pack_gathered"], float)
        Mg = np.asarray(Z["Mg"], float)                     # (S, kf, C, B)
        rows_alt = np.asarray(RW["rows"], float)

        ntrue_edges = np.asarray(RW["ntrue_edges"], float)
        nhat_edges = np.asarray(RW["nhat_edges"], float)
        snr_edges = np.asarray(RW["snr_edges"], float)
        zc_edges = np.asarray(RW["zc_edges"], float)
        bcen = 0.5 * (ntrue_edges[:-1] + ntrue_edges[1:])
        ccen = 0.5 * (nhat_edges[:-1] + nhat_edges[1:])
        vmid = _vmid(snr_edges)
        edges = (ntrue_edges, nhat_edges, snr_edges, zc_edges)

        # --- checks --------------------------------------------------------
        rowsum_dev = float(np.max(np.abs(Q.sum(axis=-1) - 1.0)))
        rows_equal = float(np.max(np.abs(Q - rows_alt)))
        # Mg[s, kf, c, b] == phi[b, s, K(kf)] * Q[b, s, K(kf), c]
        kz = np.asarray(json_kz(products), int)
        Mg_rec = np.zeros_like(Mg)
        for kf in range(Mg.shape[1]):
            Mg_rec[:, kf, :, :] = np.einsum(
                "bs,bsc->scb", phi[:, :, kz[kf]], Q[:, :, kz[kf], :])
        mg_dev = float(np.max(np.abs(Mg_rec - Mg)))
        phi_coef = prov["phi"]["smooth_alternative"]["coef"]
        phi_s_rec = np.array([[[ER.phi_smooth(b, s, k, phi_coef, bcen, vmid)
                                for k in range(phi.shape[2])]
                               for s in range(phi.shape[1])]
                              for b in range(phi.shape[0])])
        phi_s_dev = float(np.max(np.abs(phi_s_rec - phi_s)))
        for label, dev, tol in (("Q row sums", rowsum_dev, ROWSUM_ATOL),
                                ("rows_unit vs rows_%s.npz" % fam,
                                 rows_equal, 0.0),
                                ("Mg == phi * Q", mg_dev, 0.0),
                                ("phi_smooth rebuild", phi_s_dev, 1e-12)):
            if dev > tol:
                raise ManifestIntegrityError(
                    "release check FAILED for %s: %s deviation %g > %g"
                    % (fam, label, dev, tol))
        meta["checks"][fam] = {"max_abs_row_sum_minus_one": rowsum_dev,
                               "rows_unit_equals_rows_file": rows_equal,
                               "Mg_equals_phi_times_Q": mg_dev,
                               "phi_smooth_rebuild": phi_s_dev}

        # --- row moments (compact validation table) ------------------------
        mom = []
        for b in range(Q.shape[0]):
            for s in range(Q.shape[1]):
                for k in range(Q.shape[2]):
                    p = Q[b, s, k]
                    m1 = float(p @ ccen)
                    m2 = float(p @ (ccen - m1) ** 2)
                    sd = float(np.sqrt(max(m2, 0.0)))
                    sk = (float(p @ (ccen - m1) ** 3 / sd ** 3)
                          if sd > 0 else float("nan"))
                    mom.append([b, s, k, float(bcen[b]), m1,
                                m1 - float(bcen[b]), sd, sk,
                                float(phi[b, s, k]), float(phi_s[b, s, k])])
        _write_csv(os.path.join(out, "row_moments_%s.csv" % fam),
                   ["b", "s", "K", "logN_true_centre", "mean_Nhat",
                    "bias_mean_Nhat", "sd_Nhat", "skew_Nhat",
                    "phi_measured", "phi_smooth"], mom)

        _write_csv(os.path.join(out, "Q_rows_%s.csv" % fam),
                   ["b", "s", "K", "c", "logN_true_lo", "logN_true_hi",
                    "snr_lo", "snr_hi", "z_lo", "z_hi", "Nhat_lo", "Nhat_hi",
                    "Q"],
                   [[b, s, k, c, float(ntrue_edges[b]),
                     float(ntrue_edges[b + 1]), float(snr_edges[s]),
                     float(snr_edges[s + 1]), float(zc_edges[k]),
                     float(zc_edges[k + 1]), float(nhat_edges[c]),
                     float(nhat_edges[c + 1]), float(Q[b, s, k, c])]
                    for b in range(Q.shape[0]) for s in range(Q.shape[1])
                    for k in range(Q.shape[2]) for c in range(Q.shape[3])])

        np.savez_compressed(
            os.path.join(out, "response_model_%s.npz" % fam),
            Q=Q, phi=phi, phi_smooth=phi_s, phi_family_measured=phi_fam,
            phi_ref_pack=phi_ref, phi_smooth_coef=np.asarray(phi_coef, float),
            ntrue_edges=ntrue_edges, nhat_edges=nhat_edges,
            snr_edges=snr_edges, zc_edges=zc_edges,
            ntrue_centres=bcen, nhat_centres=ccen, vmid_s=vmid,
            kz_to_K=kz)

        meta["families"][fam] = {
            "name": {"E": "low-rank residual deformation of the R1c "
                          "parametric kernel (rank 2)",
                     "B": "split-normal / normal two-component mixture"}[fam],
            "nominal_dof": coefmeta.get("meta", {}).get("nominal_dof"),
            "identifiable_dof": coefmeta.get("meta", {}).get("nominal_dof_own"),
            "singular_values": coefmeta.get("meta", {}).get("sv"),
            "source": {"path": src, "sha256": dg, "sha256sums": stt},
            "rows_source": {"path": rowsrc, "sha256": dg2, "sha256sums": stt2},
            "fitted_on": prov.get("fitted_on"),
            "sealed_rule": prov.get("sealed_rule"),
            "code_commit": prov.get("git", {}).get("commit"),
            "released_object": "tabulated Q[b, s, K, c]; the fitting "
                               "parameter vector is not persisted by the "
                               "fitting code and is an implementation detail",
            "released_file": "response_model_%s.npz" % fam,
        }

        if not phi_written:
            meta["phi"] = {
                "definition": "P(N_hat on the >= %.1f observed grid | detected,"
                              " b, s, K)" % float(nhat_edges[0]),
                "default": prov["phi"]["default"],
                "baseline_for_final_ladder":
                    "2LPT-calibrated MEASURED per-cell phi for all three mock "
                    "families (PI §9); family-specific measured phi is an "
                    "ORACLE DIAGNOSTIC only",
                "smooth_alternative": prov["phi"]["smooth_alternative"],
                "smooth_basis": {
                    "form": "logistic(a0 + a1 w + a2 w^2 + a3 v_s "
                            "+ a4 1[K=1] + a5 1[K=2])",
                    "w": "N_true bin centre - 20.5",
                    "v_s": "stratum flat-quadrature midpoint of log10(S/N) "
                           "- 0.7 (predeclaration I2b); shipped as vmid_s",
                },
                "cv": prov["phi"]["cv"],
                "summary": prov["phi"]["summary"],
                "note": prov["phi"]["note"],
            }
            _write_csv(os.path.join(out, "phi_table.csv"),
                       ["b", "s", "K", "logN_true_lo", "logN_true_hi",
                        "snr_lo", "snr_hi", "z_lo", "z_hi", "phi_measured",
                        "phi_smooth", "phi_pack_reference"],
                       [[b, s, k, float(ntrue_edges[b]),
                         float(ntrue_edges[b + 1]), float(snr_edges[s]),
                         float(snr_edges[s + 1]), float(zc_edges[k]),
                         float(zc_edges[k + 1]), float(phi[b, s, k]),
                         float(phi_s[b, s, k]), float(phi_ref[b, s, k])]
                        for b in range(phi.shape[0])
                        for s in range(phi.shape[1])
                        for k in range(phi.shape[2])])
            phi_written = True

    # ---- fitted coefficients recovered by deterministic refit -------------
    cpath = os.path.join(out, "response_coefficients.json")
    if os.path.isfile(cpath):
        cj = json.load(open(cpath))
        meta["coefficients"] = {
            "status": "recovered POST HOC by a deterministic refit of the "
                      "same fitting code on the same full 2LPT-0 sample; the "
                      "ladder objects were the EVALUATED tensors and those "
                      "remain the objects of record",
            "files": cj.get("files"),
            "file_sha256": cj.get("file_sha256"),
            "quadrature": cj.get("quadrature"),
            "E_base": cj.get("E_base"),
            "per_variant": {k: {kk: v.get(kk) for kk in
                                ("form", "design", "design_covariates",
                                 "standardisation", "rank_R", "train_loss",
                                 "start_spread", "nominal_dof_as_stored",
                                 "nominal_dof_corrected", "nominal_dof_note",
                                 "reproduces_delivered_rows_to",
                                 "reproduces_within_tolerance", "tolerance")
                                if kk in v}
                            for k, v in cj.get("variants", {}).items()},
        }
        for fam, blob in meta["coefficients"]["per_variant"].items():
            if fam in meta["families"]:
                meta["families"][fam]["coefficients_file"] = \
                    "response_coefficients_%s.npz" % fam
                meta["families"][fam]["coefficients_reproduce_rows_to"] = \
                    blob.get("reproduces_delivered_rows_to")
                if blob.get("nominal_dof_corrected") is not None:
                    meta["families"][fam]["nominal_dof_corrected"] = \
                        blob["nominal_dof_corrected"]
                    meta["families"][fam]["nominal_dof_note"] = \
                        blob.get("nominal_dof_note")
            meta["families"][fam]["released_object"] = (
                "tabulated Q[b, s, K, c] (object of record) PLUS the fitted "
                "coefficients recovered post hoc by deterministic refit")
    else:
        meta["coefficients"] = {
            "status": "PENDING -- run validation/release/"
                      "response_coefficients.py; only the evaluated tensors "
                      "are released"}

    nt, nh, se, ze = edges
    meta["grid"] = {
        "b_truth_logN_edges": nt.tolist(),
        "c_observed_logN_edges": nh.tolist(),
        "s_snr_edges": [None if not np.isfinite(v) else float(v) for v in se],
        "K_z_edges": ze.tolist(),
        "shape": {"B": int(len(nt) - 1), "S": int(len(se) - 1),
                  "K": int(len(ze) - 1), "C": int(len(nh) - 1)},
    }
    shutil.copy2(os.path.join(_HERE, "standalone", "evaluate_response.py"),
                 os.path.join(out, "evaluate_response.py"))
    with open(os.path.join(out, "response_model.json"), "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
        fh.write("\n")
    _write_readme(out, meta)
    return out, meta


def json_kz(products):
    """``kz_to_K`` (fine z bin -> coarse block) from the v3 operator pack."""
    p = os.path.join(products, "support_v3", "empirical_ops_2lpt0_v3.npz")
    if not os.path.isfile(p):
        p = os.path.join(products, "support", "empirical_ops_2lpt0_A0.npz")
    return np.load(p, allow_pickle=True)["kz_to_K"]


def _coefblock(meta):
    c = meta.get("coefficients", {})
    if not c.get("per_variant"):
        return ("**PENDING.** Only the evaluated tensors are released; run "
                "`validation/release/response_coefficients.py` to recover the "
                "fitted coefficients.")
    lines = [c["status"], "",
             "| family | coefficients | rebuilds the delivered rows to |",
             "|---|---|---|"]
    for fam, b in c["per_variant"].items():
        lines.append("| %s | `response_coefficients_%s.npz` | %.3e (tolerance "
                     "%.0e, %s) |"
                     % (fam, fam, b.get("reproduces_delivered_rows_to",
                                        float("nan")),
                        b.get("tolerance", 1e-8),
                        "WITHIN" if b.get("reproduces_within_tolerance")
                        else "OUTSIDE"))
    for fam, b in c["per_variant"].items():
        if b.get("nominal_dof_note"):
            lines += ["", "**dof correction (%s).** %s" % (fam,
                                                           b["nominal_dof_note"])]
    if c.get("E_base"):
        lines += ["", "**E base kernel, labelled exactly as built.** %s"
                  % c["E_base"]["label_as_built"]]
    lines += ["", "Rebuild from coefficients with "
              "`evaluate_response.rebuild_Q_E(np.load('response_coefficients_"
              "E.npz'))` / `rebuild_Q_B(...)`."]
    return "\n".join(lines)


def _write_readme(out, meta):
    fams = meta["families"]
    txt = """# DESI GP-DLA Paper-1 — forward response operator Q and phi

    M(c | b, s, K) = phi(b, s, K) * Q(c | b, s, K)

`Q` is the SHAPE of the reported column-density distribution for a detected
truth absorber whose reported value lands on the observed grid — rows sum to
one exactly over the {C} observed bins.  `phi` is the in-grid HAD MASS, the
probability that a detected absorber's reported column lands on that grid at
all.  They are two distinct conditional objects and must not be conflated.

Both were calibrated on **mock** natural-pair matched events ({nev} events,
family `{cal}`) and frozen before any survey inference.

## Status — PENDING PI adoption

The response-model search is CLOSED.  Two finalists are released as
**candidates**; the model of record is a PI decision that has not been made:

| family | form | nominal dof | file |
|---|---|---|---|
{table}

## Grid

| axis | meaning | edges |
|---|---|---|
| `b` | truth log10 N_HI | {B} bins, {nt0} … {nt1} |
| `s` | sightline S/N | {S} strata |
| `K` | redshift block | {K} blocks, {z} |
| `c` | observed (reported) log10 N_HI | {C} bins, {nh0} … {nh1} |

## phi

Baseline for the final ladder: the **2LPT-calibrated measured per-cell phi**
applied to all three mock families (a production-like transfer question).
A 6-coefficient smooth alternative is shipped as the predeclared response-
calibration sensitivity, with its basis and coefficients, and
`evaluate_response.phi_smooth` rebuilds it exactly.

## Fitted coefficients — recovered post hoc

{coefblock}

## Files

| file | contents |
|---|---|
| `response_model_E.npz`, `response_model_B.npz` | Q, phi (measured / smooth / family-measured oracle / pack reference), coefficients of smooth phi, grid edges and centres |
| `response_coefficients_E.npz`, `response_coefficients_B.npz` | the fitted family coefficients (E: alpha, phi_r, standardisation, and the evaluated R1c base tensor; B: beta) |
| `response_coefficients.json` | coefficient metadata, quadrature settings, dof correction, reproduction error |
| `evaluate_response.py` | standalone pure-numpy evaluator |
| `response_model.json` | metadata, dof, CV, anti-tautology summary, provenance hashes |
| `Q_rows_E.csv`, `Q_rows_B.csv` | the full operator in long format |
| `phi_table.csv` | phi per cell, measured vs smooth vs the pack's frozen reference |
| `row_moments_E.csv`, `row_moments_B.csv` | per-cell mean / sd / skew of the reported column and the mean bias |

## Released checks (recomputed at build time, all exact)

{checks}

## Use

```python
import numpy as np, evaluate_response as er
m = np.load("response_model_E.npz")
row = er.response_row(logN=20.35, snr=4.0, z=2.6, model=m)   # 29-vector
M   = er.operator_M(m)                                       # phi * Q
```

## Caveats

* The objects consumed by the ladder were the EVALUATED tensors; the fitting
  code never persisted the parameter vectors.  The coefficients in this release
  were recovered afterwards by re-running the same fitting code on the same
  full sample with the same seeds and starts, and are shipped with the measured
  reproduction error against the delivered tensors.  Where they disagree, the
  DELIVERED TENSORS are the objects of record.
* Anti-tautology: the estimand is the conditional Q(N_hat | N_true, S/N, z),
  not the joint.  See `response_model.json -> anti_tautology`.
* `phi_family_measured` is an ORACLE diagnostic (each mock's own phi) and must
  not be used as a production object.

Licence: <PLACEHOLDER — CC-BY-4.0 proposed, PI to confirm>
""".format(
        C=meta["grid"]["shape"]["C"], B=meta["grid"]["shape"]["B"],
        S=meta["grid"]["shape"]["S"], K=meta["grid"]["shape"]["K"],
        nev=meta["calibration_events"], cal=meta["calibration_family"],
        nt0=meta["grid"]["b_truth_logN_edges"][0],
        nt1=meta["grid"]["b_truth_logN_edges"][-1],
        nh0=meta["grid"]["c_observed_logN_edges"][0],
        nh1=meta["grid"]["c_observed_logN_edges"][-1],
        z=meta["grid"]["K_z_edges"],
        table="\n".join("| %s | %s | %s | `%s` |"
                        % (k, v["name"], v["nominal_dof"], v["released_file"])
                        for k, v in fams.items()),
        checks="\n".join("* `%s`: %s" % (k, json.dumps(v))
                         for k, v in meta["checks"].items()),
        coefblock=_coefblock(meta))
    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--products", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-sums", action="store_true")
    ap.add_argument("--accept-stale", action="append", default=[],
                    metavar="NAME=SHA256",
                    help="explicitly accept a file whose SHA256SUMS entry is "
                         "stale; the observed digest must be typed out")
    a = ap.parse_args(argv)
    out, meta = build(a.products, a.out,
                      accept_stale=parse_accept_stale(a.accept_stale))
    if not a.no_sums:
        write_sha256sums(out)
    print("response release ->", out)
    for fam, chk in meta["checks"].items():
        print("  %s: %s" % (fam, chk))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
