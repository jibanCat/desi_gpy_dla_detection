#!/usr/bin/env python
"""completeness_release.py -- build the Zenodo completeness product (PI §7).

The fitted selection function C(N_HI, S/N) is itself a science product, so the
release must contain everything a reader needs to USE it and to judge it:
coefficients, the pivot N0, the basis definition, both covariances (Fisher and
bootstrap, plus the half-split as a third, honest, noisier estimate), the valid
N and S/N domain, the production clamp rule, a pure-numpy evaluator that
imports nothing, the fitted surface on a grid, C vs N at representative S/N,
C vs S/N at representative N, the calibration density (truth counts per cell)
that says WHERE the fit is actually informed, and provenance hashes.

VALIDATION-ONLY: reads frozen mock calibration products, writes a release tree.
No sampler, no real data, no tracked file is modified.

    python -m validation.release.completeness_release \
        --products /scratch/.../absorber_ladder_2026-09-13 \
        --out      /scratch/.../absorber_ladder_2026-09-13/release
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

from hashutil import (sha256_file, verify_against_sums, write_sha256sums,
                      ManifestIntegrityError)                  # noqa: E402
import evaluate_completeness as EC                             # noqa: E402

SCHEMA = "zenodo_release/completeness/v1"
VARIANT = "C1nsadd"
CAL_FAMILY = "2lpt0"
FAMILIES = ("2lpt0", "london0", "saclay0")

SNR_REPRESENTATIVE = (2.5, 3.5, 5.0, 7.0, 10.0)
N_REPRESENTATIVE = (19.7, 20.0, 20.3, 20.6, 21.0)


# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
def load_completeness(products, family=CAL_FAMILY, sums_cache=None):
    """The fitted object, its covariance pack and the calibration table."""
    sums_cache = {} if sums_cache is None else sums_cache
    comp = os.path.join(products, "completeness")
    paths = {
        "C_fixed": os.path.join(comp, "C_%s_%s.npz" % ("C" + VARIANT[1:]
                                                       if False else VARIANT,
                                                       family)),
        "covariance": os.path.join(comp,
                                   "C1nsadd_covariance_%s.npz" % CAL_FAMILY),
        "cal_table": os.path.join(comp, "cal_table_%s.npz" % family),
    }
    paths["C_fixed"] = os.path.join(comp, "C_%s_%s.npz" % (VARIANT, family))
    hashes = {}
    for key, p in paths.items():
        digest, status = verify_against_sums(p, sums_cache)
        if status == "MISSING":
            raise ManifestIntegrityError("completeness input missing: %s" % p)
        hashes[key] = {"path": p, "sha256": digest, "sha256sums": status}
    C = np.load(paths["C_fixed"], allow_pickle=True)
    V = np.load(paths["covariance"], allow_pickle=True)
    T = np.load(paths["cal_table"], allow_pickle=True)
    prov = json.loads(str(C["provenance"].item()))
    return C, V, T, prov, hashes


def build(products, out_root, repo=None, families=FAMILIES):
    repo = repo or os.path.abspath(os.path.join(_HERE, "..", ".."))
    out = os.path.join(out_root, "completeness")
    os.makedirs(out, exist_ok=True)
    sums_cache = {}

    C, V, T, prov, hashes = load_completeness(products, CAL_FAMILY, sums_cache)
    beta = np.asarray(V["beta"], float)
    N0 = float(prov["x_pivot"])
    U0 = float(prov["u_pivot_log10_snr"])
    logmed = np.asarray(prov["snr_logmed_s"], float)
    live = np.asarray(V["live_idx"], int)
    snr_med = np.asarray(T["snr_med_s"], float)
    ntrue_edges = np.asarray(C["ntrue_edges"], float)
    snr_edges = np.asarray(C["snr_edges"], float)
    x_b = np.asarray(V["x_b"], float)
    bcen = 0.5 * (ntrue_edges[:-1] + ntrue_edges[1:])

    snr_lo = float(snr_med[live[0]])
    snr_hi = float(snr_med[live[-1]])
    u_clamp = float(logmed[live[-1]])          # the clamp lives in log10 S/N
    u_lo = float(logmed[live[0]])

    # ---- the fitted surface reproduced by the STANDALONE evaluator --------
    # (the release must not depend on the project code to be correct)
    C_fixed = np.asarray(C["C_fixed"], float)
    repro = np.zeros_like(C_fixed)
    for s in live:
        repro[s] = EC.evaluate_completeness(
            bcen, coef=beta, N0=N0, U0=U0, clamp=True,
            log10_snr_clamp_hi=u_clamp, log10_snr=float(logmed[s]))
    max_repro_err = float(np.max(np.abs(repro[live] - C_fixed[live])))
    if max_repro_err > 1e-10:
        raise ManifestIntegrityError(
            "standalone evaluator does not reproduce C_fixed (max |dC| = %g)"
            % max_repro_err)

    # ---- calibration density (truth counts per (b, s), summed over z) -----
    truth_bks = np.asarray(T["truth_bks"], float)          # (B, kf, S)
    det_bks = np.asarray(T["det_bks"], float)
    truth_bs = truth_bks.sum(axis=1)                       # (B, S)
    det_bs = det_bks.sum(axis=1)

    # ---- grids ------------------------------------------------------------
    grid_logN = np.round(np.arange(19.0, 22.4001, 0.05), 4)
    grid_snr = np.round(np.concatenate([
        np.arange(2.4, 10.7001, 0.1), [12.0, 15.0, 20.0, 30.0]]), 4)
    GG_N, GG_S = np.meshgrid(grid_logN, grid_snr, indexing="ij")
    surface = EC.evaluate_completeness(GG_N, GG_S, beta, N0=N0, U0=U0,
                                       clamp=True, log10_snr_clamp_hi=u_clamp)
    surface_noclamp = EC.evaluate_completeness(GG_N, GG_S, beta, N0=N0, U0=U0,
                                               clamp=False)

    # ---- coefficient-uncertainty band on the surface (Fisher + bootstrap) -
    boots = np.asarray(V["beta_bootstrap"], float)         # (200, 6)

    def _band(logN, snr):
        draws = np.stack([EC.evaluate_completeness(
            logN, snr, b_, N0=N0, U0=U0, clamp=True,
            log10_snr_clamp_hi=u_clamp) for b_ in boots])
        return (np.percentile(draws, 16, axis=0),
                np.percentile(draws, 50, axis=0),
                np.percentile(draws, 84, axis=0))

    # ---- representative tables -------------------------------------------
    rows_CvN = []
    for snr in SNR_REPRESENTATIVE:
        lo, med, hi = _band(grid_logN, float(snr))
        c = EC.evaluate_completeness(grid_logN, float(snr), beta, N0=N0, U0=U0,
                                     clamp=True, log10_snr_clamp_hi=u_clamp)
        for i, n in enumerate(grid_logN):
            rows_CvN.append([float(n), float(snr), float(c[i]),
                             float(lo[i]), float(med[i]), float(hi[i])])
    rows_CvS = []
    for n in N_REPRESENTATIVE:
        lo, med, hi = _band(float(n), grid_snr)
        c = EC.evaluate_completeness(float(n), grid_snr, beta, N0=N0, U0=U0,
                                     clamp=True, log10_snr_clamp_hi=u_clamp)
        for i, s in enumerate(grid_snr):
            rows_CvS.append([float(n), float(s), float(c[i]),
                             float(lo[i]), float(med[i]), float(hi[i]),
                             bool(s > snr_hi)])

    # ---- write ------------------------------------------------------------
    npz_path = os.path.join(out, "completeness_model.npz")
    np.savez_compressed(
        npz_path,
        coef=beta, N0=np.array(N0), U0=np.array(U0),
        cov_fisher=np.asarray(V["cov_fisher"], float),
        cov_bootstrap=np.asarray(V["cov_bootstrap"], float),
        cov_halfsplit=np.asarray(V["cov_halfsplit"], float),
        beta_bootstrap=boots,
        beta_half_even=np.asarray(V["beta_half_even"], float),
        beta_half_odd=np.asarray(V["beta_half_odd"], float),
        sd_fisher=np.asarray(V["sd_fisher"], float),
        sd_bootstrap=np.asarray(V["sd_bootstrap"], float),
        sd_halfsplit=np.asarray(V["sd_halfsplit"], float),
        ntrue_edges=ntrue_edges, ntrue_centres=bcen, x_b=x_b,
        snr_edges=snr_edges, snr_median_stratum=snr_med,
        log10_snr_median_stratum=logmed, live_strata=live,
        snr_clamp_hi=np.array(snr_hi), snr_domain_lo=np.array(snr_lo),
        log10_snr_clamp_hi=np.array(u_clamp),
        log10_snr_domain_lo=np.array(u_lo),
        C_calibration_grid=C_fixed, C_calibration_grid_sd=
        np.asarray(C["C_fixed_sd"], float),
        live_strata_mask=np.asarray(C["live_strata_mask"], bool),
        grid_logN=grid_logN, grid_snr=grid_snr,
        C_surface=surface, C_surface_noclamp=surface_noclamp,
        truth_counts_bs=truth_bs, det_counts_bs=det_bs,
    )

    _write_csv(os.path.join(out, "completeness_surface.csv"),
               ["log10_NHI", "snr", "C", "C_no_clamp", "snr_clamped"],
               [[float(GG_N[i, j]), float(GG_S[i, j]), float(surface[i, j]),
                 float(surface_noclamp[i, j]), int(GG_S[i, j] > snr_hi)]
                for i in range(GG_N.shape[0]) for j in range(GG_N.shape[1])])
    _write_csv(os.path.join(out, "C_vs_N_at_representative_snr.csv"),
               ["log10_NHI", "snr", "C", "C_p16", "C_p50", "C_p84"], rows_CvN)
    _write_csv(os.path.join(out, "C_vs_snr_at_representative_N.csv"),
               ["log10_NHI", "snr", "C", "C_p16", "C_p50", "C_p84",
                "snr_clamped"], rows_CvS)
    _write_csv(os.path.join(out, "calibration_density.csv"),
               ["b", "logN_lo", "logN_hi", "s", "snr_lo", "snr_hi",
                "snr_median", "n_truth", "n_detected", "C_raw", "C_fitted",
                "live_stratum"],
               [[b, float(ntrue_edges[b]), float(ntrue_edges[b + 1]), s,
                 float(snr_edges[s]), float(snr_edges[s + 1]),
                 float(snr_med[s]), float(truth_bs[b, s]), float(det_bs[b, s]),
                 (float(det_bs[b, s] / truth_bs[b, s])
                  if truth_bs[b, s] > 0 else ""),
                 float(C_fixed[s, b]), int(s in set(live.tolist()))]
                for b in range(len(bcen)) for s in range(len(snr_med))])

    shutil.copy2(os.path.join(_HERE, "standalone", "evaluate_completeness.py"),
                 os.path.join(out, "evaluate_completeness.py"))

    meta = {
        "schema": SCHEMA,
        "product": "DESI GP-DLA Paper-1 fitted completeness (selection) "
                   "function C(N_HI, S/N)",
        "status": "FINAL (PI ruling 2026-09-13d §6: C1nsadd CARRIED as the "
                  "baseline completeness model; no further development)",
        "variant": VARIANT,
        "variant_id": prov.get("variant_id"),
        "calibration_family": CAL_FAMILY,
        "transfer_families": [f for f in families if f != CAL_FAMILY],
        "fitted_to_real_data": False,
        "estimand": prov.get("convention"),
        "basis": {
            "form": "logit C = poly3(x) + poly2(u), NO interaction",
            "x": "log10 N_HI - N0",
            "u": "log10(S/N) - U0",
            "N0": N0,
            "U0": U0,
            "columns": ["1", "x", "x^2", "x^3", "u", "u^2"],
            "n_coef": 6,
            "link": "logit",
            "estimator": "IRLS binomial GLM, ridge %g on non-intercept columns"
                         % prov.get("ridge", 0.0),
            "response": "Jeffreys-consistent logit of detected/truth counts, "
                        "eta_hat = log((d+1/2)/(t-d+1/2))",
        },
        "coefficients": {"names": ["b0", "b1", "b2", "b3", "b4", "b5"],
                         "values": beta.tolist(),
                         "sd_fisher": np.asarray(V["sd_fisher"],
                                                 float).tolist(),
                         "sd_bootstrap": np.asarray(V["sd_bootstrap"],
                                                    float).tolist(),
                         "sd_halfsplit": np.asarray(V["sd_halfsplit"],
                                                    float).tolist()},
        "covariance": {
            "cov_fisher": "6x6, IRLS observed information (in completeness_"
                          "model.npz)",
            "cov_bootstrap": "6x6, %d sightline-half bootstrap replicates"
                             % boots.shape[0],
            "cov_halfsplit": "6x6, TARGETID-parity half-split; noisier, "
                             "reported for honesty not for propagation",
            "recommended_for_propagation": "cov_bootstrap",
        },
        "valid_domain": {
            "log10_NHI": [float(ntrue_edges[0]), float(ntrue_edges[-1])],
            "snr_calibrated": [snr_lo, snr_hi],
            "snr_calibrated_note": "stratum S/N MEDIANS of the lowest and "
                                   "highest live strata; the fit is informed "
                                   "only between them",
            "live_strata": live.tolist(),
            "stratum_snr_medians": [None if not np.isfinite(v) else float(v)
                                    for v in snr_med],
            "stratum_log10_snr_medians": [None if not np.isfinite(v)
                                          else float(v) for v in logmed],
            "support_snr_cut": 2.0,
        },
        "production_clamp_rule": {
            "predeclared": True,
            "rule": "evaluate at log10(S/N) -> min(log10(S/N), %.16f) "
                    "(= log10 %.7f, the highest calibrated stratum median)"
                    % (u_clamp, snr_hi),
            "clamp_snr": snr_hi,
            "clamp_log10_snr": u_clamp,
            "direction": "UPPER only; inert below the cap",
            "low_end": "not clamped; below the lowest calibrated stratum "
                       "median (%.4f) the surface is an extrapolation and is "
                       "flagged out-of-domain by the evaluator" % snr_lo,
            "authority": "PI ruling 2026-09-13d §6",
        },
        "evaluation_code": {
            "file": "evaluate_completeness.py",
            "imports": ["numpy"],
            "entry_point": "evaluate_completeness(logN, snr, coef, N0, "
                           "clamp=True)",
            "reproduces_calibration_table_to": max_repro_err,
        },
        "validation": {
            "heldout_logloss_total": prov.get("heldout", {}).get("total"),
            "heldout_reported_window": prov.get("heldout", {}).get(
                "reported_window"),
            "cv": prov.get("cv"),
            "c0_gate": prov.get("c0_gate"),
        },
        "files": {
            "completeness_model.npz": "coefficients, covariances, bootstrap "
                                      "replicates, grids, surface, "
                                      "calibration counts",
            "evaluate_completeness.py": "standalone pure-numpy evaluator",
            "completeness_surface.csv": "fitted surface on the released grid",
            "C_vs_N_at_representative_snr.csv": "C vs N at S/N in %s"
                                                % (SNR_REPRESENTATIVE,),
            "C_vs_snr_at_representative_N.csv": "C vs S/N at log N in %s"
                                                % (N_REPRESENTATIVE,),
            "calibration_density.csv": "truth / detected counts per (N, S/N) "
                                       "cell = where the fit is informed",
            "README.md": "this product in prose",
        },
        "inputs": hashes,
        "per_family_objects": {},
        "builder": "validation/release/completeness_release.py",
        "git": _git_stamp(repo),
        "numpy": np.__version__,
        "python": sys.version.split()[0],
    }
    for fam in families:
        p = os.path.join(products, "completeness", "C_%s_%s.npz"
                         % (VARIANT, fam))
        digest, status = verify_against_sums(p, sums_cache)
        meta["per_family_objects"][fam] = {
            "path": p, "sha256": digest, "sha256sums": status,
            "note": "the SAME 6 coefficients; the per-family file differs only "
                    "in the pack's cell geometry / gather indices"}

    with open(os.path.join(out, "completeness_model.json"), "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
        fh.write("\n")
    _write_readme(out, meta, beta, N0, snr_lo, snr_hi)
    return out, meta


def _write_readme(out, meta, beta, N0, snr_lo, snr_hi):
    txt = """# DESI GP-DLA Paper-1 — fitted completeness C(N_HI, S/N)

Detection probability for a truth H I absorber as a function of its column
density and the sightline signal-to-noise, fitted on mock calibration only
(`{cal}`), frozen **before** any survey inference, and used as a FIXED object
by the hierarchical CDDF inference.  This is the object PI ruling 2026-09-13d
§7 requires to be released as a science product in its own right.

## The model

    x = log10 N_HI - {N0}
    u = log10(S/N) - {U0:.16f}
    logit C = b0 + b1 x + b2 x^2 + b3 x^3 + b4 u + b5 u^2

Six coefficients; cubic in N, quadratic in log S/N, additive in the logit (no
interaction, i.e. the N shape is common to every S/N stratum and S/N only
shifts the logit).  Fitted coefficients:

    b = {beta}

## Valid domain and the production clamp rule

The calibration is informed between the S/N stratum medians **{lo:.4f}** and
**{hi:.4f}** and over log10 N_HI in [{nlo}, {nhi}].  Production evaluation
applies the predeclared clamp

    log10(S/N) -> min( log10(S/N), log10({hi:.7f}) )

so a sightline better than the top calibrated stratum receives that stratum's
completeness.  The clamp is inert below the cap and is NOT applied at the low
end; below {lo:.4f} the surface is an extrapolation and `evaluate_completeness`
flags it out of domain.

## Files

| file | contents |
|---|---|
| `completeness_model.npz` | coefficients, Fisher / bootstrap / half-split covariance, bootstrap replicates, grids, fitted surface, calibration counts |
| `evaluate_completeness.py` | standalone evaluator — imports numpy and nothing else |
| `completeness_model.json` | machine-readable metadata, domain, clamp rule, provenance hashes |
| `completeness_surface.csv` | the fitted surface on the released grid |
| `C_vs_N_at_representative_snr.csv` | C vs N at S/N = {snrrep} with 16/50/84 bootstrap bands |
| `C_vs_snr_at_representative_N.csv` | C vs S/N at log N = {nrep} with bands |
| `calibration_density.csv` | truth and detected counts per (N, S/N) cell — where the fit is actually informed |

## Use

```python
import numpy as np, evaluate_completeness as ec
m = np.load("completeness_model.npz")
C = ec.evaluate_completeness(20.3, 4.0, m["coef"], float(m["N0"]))
```

The standalone evaluator reproduces the internal calibration table to
{err:.3e} at the stratum medians.

## Caveats

* Completeness is NOT purity.  False positives are a separate calibration
  object (the loa-0 FP template); do not read 1 - C as a contamination rate.
* The surface is a DETECTION probability on truth systems with no observed-grid
  restriction.  The in-grid fraction of the reported column is carried by the
  response operator's `phi`, not here.
* Covariance for propagation: use `cov_bootstrap`.  `cov_halfsplit` is a
  two-sample estimate and is much noisier; it is shipped for transparency.

Licence: {lic}
""".format(cal=meta["calibration_family"], N0=N0,
           U0=meta["basis"]["U0"],
           beta=np.array2string(beta, precision=8),
           lo=snr_lo, hi=snr_hi,
           nlo=meta["valid_domain"]["log10_NHI"][0],
           nhi=meta["valid_domain"]["log10_NHI"][1],
           snrrep=", ".join(str(s) for s in SNR_REPRESENTATIVE),
           nrep=", ".join(str(s) for s in N_REPRESENTATIVE),
           err=meta["evaluation_code"]["reproduces_calibration_table_to"],
           lic="<PLACEHOLDER — CC-BY-4.0 proposed, PI to confirm>")
    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--products", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-sums", action="store_true")
    a = ap.parse_args(argv)
    out, meta = build(a.products, a.out)
    if not a.no_sums:
        write_sha256sums(out)
    print("completeness release ->", out)
    print("  evaluator reproduces calibration table to %.3e"
          % meta["evaluation_code"]["reproduces_calibration_table_to"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
