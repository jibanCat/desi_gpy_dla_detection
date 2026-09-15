#!/usr/bin/env python
"""release_readme.py -- the top-level README of the Zenodo package.

It is BUILT, not written: the file inventory is enumerated from the release
tree itself (so it can never drift from what is actually shipped), and the
frozen numbers it quotes are read out of ``model_of_record/MODEL_OF_RECORD.json``
and ``systematics/SYSTEMATICS_TABLE.json``.

VALIDATION-ONLY.  Mock/calibration products only; no real data.

    python -m validation.release.release_readme \
        --out /scratch/.../absorber_ladder_2026-09-13/release
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from hashutil import write_sha256sums, ManifestIntegrityError   # noqa: E402

README = "README.md"
SKIP = {README, "SHA256SUMS"}

APPROVED_WORDING = [
    "\"integrated incidence recovered to approximately sub-percent accuracy "
    "across three independent mock families; individual redshift bins retain "
    "a coherent percent-level transfer residual, propagated as a "
    "redshift-dependent calibration systematic.\"",
    "\"the FP / low-N decomposition remains partially degenerate while the "
    "integrated incidence is substantially more stable; FP-model and prior "
    "sensitivities quantified explicitly.\"",
]
FORBIDDEN_WORDING = [
    "\"all mock closure tests passed\" -- the frozen per-bin CRIT v2 gate is "
    "formally FAILED; it must never be described as passed.",
    "\"the HBI accurately recovers the FP population\" -- the FP / low-N "
    "decomposition is only weakly identified.",
]


def _inventory(root):
    """``[(relative path, bytes), ...]`` for everything shipped, sorted."""
    rows = []
    for base, _dirs, files in os.walk(root):
        for name in files:
            if name in SKIP and base == root:
                continue
            if name == "SHA256SUMS":
                continue
            full = os.path.join(base, name)
            rows.append((os.path.relpath(full, root), os.path.getsize(full)))
    rows.sort()
    return rows


def _load(root, rel):
    path = os.path.join(root, rel)
    if not os.path.isfile(path):
        raise ManifestIntegrityError(
            "FAIL CLOSED: the release README needs %s; build it first" % rel)
    return json.load(open(path))


def build(out_root):
    mor = _load(out_root, "model_of_record/MODEL_OF_RECORD.json")
    sysd = _load(out_root, "systematics/SYSTEMATICS_TABLE.json")
    inv = _inventory(out_root)
    cfg = mor["configuration"]

    L = []
    A = L.append
    A("# Paper-1 forward observation model — release package")
    A("")
    A("**Status: %s.** Model of record: **%s**." % (mor["status"],
                                                    mor["model_of_record"]))
    A("Generated %s. Authority: `%s`."
      % (_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
         mor["authority"]["ruling"]))
    A("")
    A("No real-data product appears anywhere in this package: everything here "
      "is a calibration object or a mock-closure diagnostic.")
    A("")
    A("## 1. The model")
    A("")
    A("```")
    A(mor["generative_model"])
    A("```")
    A("")
    A("`C` (completeness), `phi` (loss outside the observed grid) and `Q` "
      "(migration within the grid) are three different calibration objects "
      "and are released as such. `M = phi Q`.")
    A("")
    A("| component | frozen object |")
    A("|---|---|")
    for key in ("response_family", "phi", "completeness", "g", "fp_model",
                "fp_a0_rule", "lambda_posterior", "lambda_imputation_rule",
                "t_K_prior", "subfloor", "population_prior"):
        A("| `%s` | %s |" % (key, cfg[key]))
    A("| support contract | `%s` |" % cfg["support_contract"])
    A("| sampler | NUTS, target_accept %s, %s chains x (%s warmup + %s) |"
      % (cfg["sampler"]["target_accept"], cfg["sampler"]["chains"],
         cfg["sampler"]["warmup"], cfg["sampler"]["samples"]))
    A("")
    A("**E is a named response-form SENSITIVITY, not a second model of "
      "record.** Both posteriors are preserved; the difference is reported "
      "signed and asymmetric. `|B - E| / 2` is not a 1 sigma and the two must "
      "never be averaged.")
    A("")
    A("## 2. Contents")
    A("")
    A("| file | bytes |")
    A("|---|---|")
    for rel, size in inv:
        A("| `%s` | %d |" % (rel, size))
    A("")
    A("`SHA256SUMS` (at the root and in each product directory) carries the "
      "sha256 of every file above; `PROVENANCE_MANIFEST.json` links "
      "calibration data -> fitted object -> HBI pack -> posterior -> paper "
      "number, fail-closed on any hash mismatch.")
    A("")
    A("## 3. Reconstruction recipe")
    A("")
    A("A reader with this package and numpy can rebuild the forward count "
      "model; the two shipped evaluators import numpy and nothing else.")
    A("")
    A("```python")
    A("import numpy as np, evaluate_completeness as ec, evaluate_response as er")
    A("cm = np.load('completeness/completeness_model.npz')")
    A("C  = ec.evaluate_completeness(logN, snr, cm['coef'], float(cm['N0']))")
    A("rm = np.load('response/response_model_B.npz')   # B is the model of record")
    A("er.row_sum_check(rm)                            # |sum_c Q - 1| ~ 1e-16")
    A("M  = er.operator_M(rm)                          # phi * Q, shape (b,s,K,c)")
    A("Mg = M[:, :, rm['kz_to_K'], :]                  # expand to the 15 fine z bins")
    A("mu_TP = np.einsum('skcb,sb,bk->cks', Mg_skcb, C_sb, w_bk)")
    A("fp = np.load('fp/fp_template.npz')              # fixed FP template")
    A("pi = fp['perks_share']                          # sums to 1 over live cells")
    A("```")
    A("")
    A("1. Completeness: propagate the coefficient uncertainty with "
      "`cm['cov_bootstrap']` or the 200 `cm['beta_bootstrap']` replicates.")
    A("2. Response: `response_model_B.npz` is the object of record; "
      "`response_coefficients_B.npz` rebuilds it to 1.9e-7 (optimiser noise). "
      "`response_model_E.npz` is the alternate sensitivity, whose "
      "coefficients reproduce its tensor only to 1.2e-2, so for E the FROZEN "
      "TENSOR is the object of record.")
    A("3. phi: `er.operator_M(rm, smooth_phi=True)` gives the predeclared "
      "smooth-phi sensitivity; the family's own measured phi is an ORACLE "
      "diagnostic and never a production object.")
    A("4. False positives: `fp/fp_template.npz` plus "
      "`fp/lambda_spec.json`; set the FP term to zero to reproduce the ORACLE "
      "diagnostic configuration.")
    A("5. Check yourself: `sum_c Q = 1` exactly; `Mg` rows sum to `phi`, not "
      "to one — the missing mass is the fraction of detections whose reported "
      "column falls off the >= 19.5 observed grid, and it is NOT a "
      "completeness loss.")
    A("")
    A("## 4. Systematics")
    A("")
    A("`systematics/SYSTEMATICS_TABLE.{json,csv,md}` carries the %d named "
      "effects. %s" % (sysd["n_systematics"], sysd["combination_rule"]))
    A("")
    A("| id | effect | treatment |")
    A("|---|---|---|")
    for s in sysd["systematics"]:
        A("| %s | %s | %s |" % (s["id"], s["name"],
                                s["treatment"].split(".")[0]))
    A("")
    A("## 5. Wording constraints for reuse")
    A("")
    A("These are binding on any description of these products.")
    A("")
    A("**Approved formulations.**")
    A("")
    for w in APPROVED_WORDING:
        A("* %s" % w)
    A("")
    A("**Forbidden formulations.**")
    A("")
    for w in FORBIDDEN_WORDING:
        A("* NOT %s" % w)
    A("")
    A("**Completeness presentation.** Never a single scalar completeness. "
      "Show `C(N_HI, S/N)`: the N dependence, the S/N dependence, the "
      "calibrated support and the fitted-vs-raw validation. Completeness and "
      "purity are separate components and must be presented separately; "
      "`1 - C` is not contamination.")
    A("")
    A("## 6. Licence")
    A("")
    A("**PENDING PI — the release licence has not yet been chosen.** The line "
      "below is a PLACEHOLDER, not a decision, and must not be read as one.")
    A("")
    A("<PLACEHOLDER — CC-BY-4.0 proposed for data and text, BSD-3-Clause "
      "proposed for the evaluation code; PI to confirm before submission.>")
    A("")

    path = os.path.join(out_root, README)
    with open(path, "w") as fh:
        fh.write("\n".join(L))
    return path, inv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="the release root")
    ap.add_argument("--no-sums", action="store_true")
    a = ap.parse_args(argv)
    path, inv = build(a.out)
    if not a.no_sums:
        write_sha256sums(a.out)
    print("release README ->", path, "(%d files listed)" % len(inv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
