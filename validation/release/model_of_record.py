#!/usr/bin/env python
"""model_of_record.py -- the FROZEN configuration of record (PI ruling
2026-09-14 sec.16: "after the freeze, science results must not choose or modify
these objects"), written as one machine-readable JSON in which EVERY object is
named by path and sha256.

    MODEL OF RECORD = B (36-coefficient split-normal mixture response)
                      x phi_2LPT (measured per-cell in-grid fraction)
                      x C1nsadd (6-coefficient completeness)
                      + M1CUT FP (fixed loa-0 Perks template, a0 = 1/K,
                                  cut-feedback Lambda, coarse t_K only)
                      + fixed TRANSPORTED sub-floor-host term
                      on the corrected v3 support contract.

FAIL CLOSED: a referenced object that does not exist, or whose recorded
``SHA256SUMS`` digest disagrees with the file, aborts the build.  A sealed
predeclaration whose ``.sha256`` sidecar disagrees with the document aborts the
build as well -- a broken seal is never downgraded to a warning.

VALIDATION-ONLY.  Mock/calibration products only; no real data; no sampler.

    python -m validation.release.model_of_record \
        --products /scratch/.../absorber_ladder_2026-09-13 \
        --out      /scratch/.../absorber_ladder_2026-09-13/release
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from hashutil import (sha256_file, verify_against_sums, write_sha256sums,
                      ManifestIntegrityError)                   # noqa: E402

SCHEMA = "zenodo_release/model_of_record/v1"
FAMILIES = ("2lpt0", "london0", "saclay0")
GOV = "/home/mfho/desi_gpy_dla_notes/governance"
LOA0_FP = ("/nfs/turbo/lsa-cavestru/mfho/paper1_durable_inputs/"
           "loa0_fp_v1_20260615_outputs/loa0_fp_product.npz")
RULING = os.path.join(GOV, "PI_RULING_2026-09-14_MODEL_FREEZE_ADOPT_B.md")
RECORD_RUN_DIR = "final/runs/B-phi2lpt-C1nsadd-M1CUTj1"
RECORD_RUN_STEM = "B-phi2lpt-C1nsadd-M1CUTj1"

PREDECLARATIONS = {
    "final_ladder": ("final_campaign_2026-09-13",
                     "FINAL_LADDER_PREDECLARATION"),
    "j8_certification": ("final_campaign_2026-09-13",
                         "J8_CERTIFICATION_PREDECLARATION"),
    "response_family_opening_rule": ("response_review_2026-09-13",
                                     "RESPONSE_FAMILY_OPENING_RULE_PREDECLARATION"),
    "candidate_implementation_choices": ("response_review_2026-09-13",
                                         "CANDIDATE_IMPLEMENTATION_CHOICES_PREDECLARATION"),
    "absorber_ladder": ("absorber_ladder_2026-09-13",
                        "ABSORBER_LADDER_PREDECLARATION"),
}

GENERATIVE_MODEL = (
    "n_cks ~ Poisson(mu^TP_cks + mu^FP_cks + mu^subfloor_cks);  "
    "mu^TP_cks = dX_ks * sum_b f_bk * dN_b * C_bs * g_bk * phi_bsK * "
    "Q_{c|bsK};  M_skcb = phi_bsK * Q_{c|bsK}.  Completeness C, loss outside "
    "the observed grid phi, and migration within the grid Q are THREE "
    "DIFFERENT calibration objects.")


def _git(repo, args):
    try:
        return subprocess.check_output(["git"] + args, cwd=repo,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                        # pragma: no cover
        return None


def _node(path, role, note=None, sums_cache=None, required=True):
    """One fail-closed object record."""
    if not os.path.isfile(path):
        if required:
            raise ManifestIntegrityError(
                "FAIL CLOSED: model-of-record object missing: %s (%s)"
                % (path, role))
        return {"path": path, "role": role, "exists": False, "sha256": None,
                "bytes": None, "sha256sums_status": "MISSING", "note": note}
    digest, status = verify_against_sums(path, sums_cache if sums_cache
                                         is not None else {})
    return {"path": path, "role": role, "exists": True, "sha256": digest,
            "bytes": os.path.getsize(path), "sha256sums_status": status,
            "note": note}


def _predeclaration(directory, stem):
    doc = os.path.join(GOV, directory, stem + ".md")
    sidecar = os.path.join(GOV, directory, stem + ".sha256")
    if not os.path.isfile(doc):
        raise ManifestIntegrityError(
            "FAIL CLOSED: sealed predeclaration missing: %s" % doc)
    digest = sha256_file(doc)
    sealed = None
    if os.path.isfile(sidecar):
        sealed = open(sidecar).read().split()[0].strip().lower()
        if sealed != digest:
            raise ManifestIntegrityError(
                "FAIL CLOSED: BROKEN SEAL on %s: sidecar says %s, document "
                "hashes to %s" % (doc, sealed, digest))
    stamp = os.path.join(GOV, directory, stem + ".timestamp")
    return {"path": doc, "sha256": digest, "sealed_sha256": sealed,
            "sealed_utc": (open(stamp).read().strip()
                           if os.path.isfile(stamp) else None)}


def _record_run(products, family="2lpt0", seed=20260811):
    path = os.path.join(products, RECORD_RUN_DIR,
                        "RUN_%s_%s_s%d.json" % (RECORD_RUN_STEM, family, seed))
    if not os.path.isfile(path):
        raise ManifestIntegrityError(
            "FAIL CLOSED: model-of-record run JSON missing: %s" % path)
    return path, json.load(open(path))


def build(products, out_root, repo=None):
    repo = repo or os.path.abspath(os.path.join(_HERE, "..", ".."))
    out = os.path.join(out_root, "model_of_record")
    os.makedirs(out, exist_ok=True)
    sums = {}

    run_path, run = _record_run(products)
    lam = run["diagnostics"]["lam_cut"]

    objects = {}

    def add(key, path, role, note=None, required=True):
        objects[key] = _node(path, role, note, sums, required)

    for fam in FAMILIES:
        add("response/Mg_B_%s" % fam,
            os.path.join(products, "response_review", "candidates",
                         "Mg_B_%s.npz" % fam),
            "response operator of record: key Mg = Q_B x phi_2LPT,measured")
        add("completeness/C_C1nsadd_%s" % fam,
            os.path.join(products, "completeness", "C_C1nsadd_%s.npz" % fam),
            "completeness of record C(N, S/N), 6 coefficients")
        add("subfloor/P6b_rate_%s" % fam,
            os.path.join(products, "completeness", "P6b_rate_%s.npz" % fam),
            "sub-floor-host rate; the TRANSPORTED 2LPT-0 rate is the "
            "survey-facing prescription (truth-pinned P6b was the "
            "mock-certification convention)")
        for kind in ("scanpack_%s_b300_v3", "fp_census_%s_v3",
                     "empirical_ops_%s_v3", "fp_counts_%s_v3"):
            name = kind % fam
            add("support_v3/%s" % name,
                os.path.join(products, "support_v3", name + ".npz"),
                "v3 support contract object")

    add("completeness/C1nsadd_covariance_2lpt0",
        os.path.join(products, "completeness",
                     "C1nsadd_covariance_2lpt0.npz"),
        "completeness coefficient covariance (Fisher / bootstrap / "
        "half-split) -- category-3 calibration uncertainty")
    add("response/rows_B",
        os.path.join(products, "response_review", "candidates", "rows_B.npz"),
        "frozen Q_B tensor (the object of record; coefficients reproduce it "
        "to 1.9e-7)")
    add("fp/loa0_fp_product", LOA0_FP,
        "loa-0 false-positive product (fixed template; no survey-learned "
        "shape)")
    add("governance/pi_ruling_2026-09-14", RULING,
        "the adoption ruling this configuration is frozen under")

    # release-side products (present once the other builders have run)
    for rel, role in (
            ("completeness/completeness_model.npz", "released completeness"),
            ("response/response_model_B.npz", "released response B"),
            ("response/response_coefficients_B.npz", "released B coefficients"),
            ("systematics/SYSTEMATICS_TABLE.json", "released systematics table"),
            ("fp/fp_template.npz", "released loa-0 FP template"),
            ("PROVENANCE_MANIFEST.json", "released provenance manifest")):
        add("release/" + rel, os.path.join(out_root, rel), role,
            required=False)

    missing = [k for k, v in objects.items() if not v["exists"]]

    K = _live_cells(products)
    fpc = np.load(os.path.join(products, "support_v3",
                               "fp_counts_2lpt0_v3.npz"), allow_pickle=True)
    n_fp = int(np.asarray(fpc["fp_counts"]).sum())
    ell_eff = float(fpc["fp_ell_eff"])

    doc = {
        "schema": SCHEMA,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ADOPTED / FROZEN 2026-09-14 (PI ruling)",
        "authority": {
            "ruling": "PI_RULING_2026-09-14_MODEL_FREEZE_ADOPT_B.md",
            "sha256": sha256_file(RULING) if os.path.isfile(RULING) else None,
            "frozen_items": [
                "B architecture + coefficients", "2LPT measured phi",
                "C1nsadd", "frozen g", "M1CUT", "a0 = 1/K", "t_K prior",
                "Lambda cut posterior", "sub-floor prescription",
                "v3 support", "HBI population prior",
                "reduction definitions", "systematics definitions"],
            "closed": ["new response families", "R2", "C2",
                       "FP-shape freedom", "Lambda feedback",
                       "a0 optimisation", "completeness development",
                       "mock-driven tuning"],
        },
        "model_of_record": "B + phi_2LPT + C1nsadd + M1CUT (a0 = 1/K)",
        "generative_model": GENERATIVE_MODEL,
        "named_sensitivity": {
            "response_form": "E (rank-2 residual correction on the R1c-form "
                             "base) -- NOT a second model of record; both "
                             "posteriors preserved; signed asymmetric "
                             "envelope; never |B - E| / 2 as 1 sigma",
        },
        "configuration": {
            "response_family": "B (split-normal + normal mixture; 36 "
                               "calibration coefficients on the basis "
                               "{1, u, u^2, v, I_K1, I_K2}, u = N - 20.5, "
                               "v = log10 S/N - 0.7)",
            "phi": "2LPT-0 measured per-cell in-grid fraction (Jeffreys +1/2 "
                   "count ratio), applied to every family",
            "completeness": "C1nsadd: logit C = sum_{i<=3} beta_i (N - 20)^i "
                            "+ beta_4 log10 S/N + beta_5 (log10 S/N)^2, with "
                            "the calibrated high-S/N clamp",
            "g": "frozen redshift shape of completeness (no S/N dependence -- "
                 "the identified candidate cause of systematic S1)",
            "fp_model": "M1CUT: fixed loa-0 Perks template, cut-feedback "
                        "Lambda, coarse t_K the only survey freedom",
            "fp_a0": 1.0 / K,
            "fp_a0_rule": "a0 = 1/K with K = %d live cells" % K,
            "fp_K_live_cells": K,
            "fp_events": n_fp,
            "fp_ell_eff": ell_eff,
            "lambda_posterior": "Gamma(%.1f, ell_eff = %.6f) = "
                                "Gamma(N_FP + 1/2, ell_eff)"
                                % (n_fp + 0.5, ell_eff),
            "lambda_imputation_rule": "Lambda_j = Q_Gamma((j + 1/2)/J), "
                                      "j = 0..J-1 (deterministic, seed-free)",
            "lambda_J_screen": int(lam.get("J", 1)),
            "lambda_J_production": 8,
            "t_K_prior": "t_K ~ Normal(0, 1), 3 coarse-z sites",
            "subfloor": "fixed TRANSPORTED calibration term for survey use "
                        "(the truth-pinned P6b form was the mock-certification "
                        "convention only)",
            "support_contract": run["support_gate"]["support_id"],
            "support_level": run["support_gate"]["level"],
            "sampler": {"kernel": "NUTS",
                        "target_accept": run["run_config"]["target_accept"],
                        "chains": run["run_config"]["chains"],
                        "warmup": run["run_config"]["warmup"],
                        "samples": run["run_config"]["samples"],
                        "seed_primary": run["run_config"]["seed"]},
            "population_prior": "non-centred two-dimensional random walk on "
                                "f_bk (hyper-parameters sigma_N, sigma_z, "
                                "level, slope); 242 latent sites",
        },
        "reductions": {
            "dndx_ge20.0": "dN/dX for log10 N_HI >= 20.0, all z",
            "dndx_ge20.3": "dN/dX for log10 N_HI >= 20.3, all z",
            "omega": run["thresholds"]["omega_allz"]["key"],
            "paper1_z_bins": [
                {"bin": c["bin"], "z": c["z"]} for c in
                run["perz_recovery"]["estimand"]["ge20.3"]["paper1_bins"]],
            "coarse_blocks": [
                {"bin": c["bin"], "z": c["z"]} for c in
                run["perz_recovery"]["estimand"]["ge20.3"]["coarse_blocks"]],
            "code": "CDDF_analysis/hbi_mcmc/hbi_reduction.py",
        },
        "code": {
            "worktree": repo,
            "commit": _git(repo, ["rev-parse", "HEAD"]),
            "branch": _git(repo, ["rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(_git(repo, ["status", "--porcelain"])),
            "run_commit": run["run_config"]["code_commit"],
            "python": run["run_config"]["python"],
            "note": "the sampler state of the frozen runs is the run_commit; "
                    "later commits in this worktree are read-out and release "
                    "tooling only",
        },
        "predeclarations": {k: _predeclaration(*v)
                            for k, v in PREDECLARATIONS.items()},
        "record_run_example": {"path": run_path,
                               "sha256": sha256_file(run_path)},
        "objects": objects,
        "n_objects": len(objects),
        "missing_objects": missing,
        "fail_closed": "every object above is verified against its directory's "
                       "SHA256SUMS at build time; a mismatch aborts the build "
                       "and a missing frozen object aborts the build",
    }
    path = os.path.join(out, "MODEL_OF_RECORD.json")
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    return out, doc


def _live_cells(products, family="2lpt0"):
    pack = os.path.join(products, "support_v3",
                        "scanpack_%s_b300_v3.npz" % family)
    if not os.path.isfile(pack):
        raise ManifestIntegrityError("FAIL CLOSED: pack missing: %s" % pack)
    with np.load(pack, allow_pickle=True) as z:
        dX = np.asarray(z["dX"], float)
        counts = np.asarray(z["counts"], float)
    return int(counts.shape[0] * (dX.sum(axis=0) > 0).sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--products", required=True)
    ap.add_argument("--out", required=True, help="the release root")
    ap.add_argument("--no-sums", action="store_true")
    a = ap.parse_args(argv)
    out, doc = build(a.products, a.out)
    if not a.no_sums:
        write_sha256sums(out)
    print("model of record ->", out)
    print("  %d objects, %d missing (optional release-side products)"
          % (doc["n_objects"], len(doc["missing_objects"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
