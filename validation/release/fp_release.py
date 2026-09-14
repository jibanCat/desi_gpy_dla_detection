#!/usr/bin/env python
"""fp_release.py -- the loa-0 FALSE-POSITIVE template of record (PI ruling
2026-09-14 sec.1, sec.8, sec.9, sec.11).

Released objects
----------------
* ``fp/fp_template.npz``     -- the 29 x 8 ``fp_counts`` block on its own
  support contract, the observed-bin and S/N-stratum edges, the LIVE-stratum
  mask, ``a0 = 1/K``, the Perks log-share template ``m_cs``, and the
  calibration posterior ``p(Lambda | D_loa0) = Gamma(N_FP + 1/2, ell_eff)``
  with the eight stratified imputations of the J = 8 production rule.
* ``fp/fp_counts.csv``       -- the same counts in long format, one row per
  (observed bin, S/N stratum), round-trippable.
* ``fp/lambda_spec.json``    -- the posterior specification and imputation rule.
* ``fp/README.md``           -- what the template is and, explicitly, what it
  is NOT (no survey-learned shape, no accurate FP reconstruction claim).

The FP shape is FIXED: there is no survey-learned shape, no saturated logits,
no survey update of Lambda and no per-cell fields (PI sec.8).  The only
survey-fitted FP freedom is the coarse-z transfer ``t_K`` (3 sites).

VALIDATION-ONLY.  Mock/calibration products only; no real data; no sampler.

    python -m validation.release.fp_release \
        --products /scratch/.../absorber_ladder_2026-09-13 \
        --out      /scratch/.../absorber_ladder_2026-09-13/release
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from hashutil import (sha256_file, verify_against_sums, write_sha256sums,
                      ManifestIntegrityError)                   # noqa: E402

SCHEMA = "zenodo_release/fp_template/v1"
FAMILIES = ("2lpt0", "london0", "saclay0")
CAL_FAMILY = "2lpt0"
J_PRODUCTION = 8
LOA0_FP = ("/nfs/turbo/lsa-cavestru/mfho/paper1_durable_inputs/"
           "loa0_fp_v1_20260615_outputs/loa0_fp_product.npz")

CSV_HEADER = ("nhat_lo", "nhat_hi", "snr_lo", "snr_hi", "live_stratum",
              "fp_counts", "perks_share")


def _load(products, family, sums):
    fp = os.path.join(products, "support_v3", "fp_counts_%s_v3.npz" % family)
    pack = os.path.join(products, "support_v3",
                        "scanpack_%s_b300_v3.npz" % family)
    for p in (fp, pack):
        if not os.path.isfile(p):
            raise ManifestIntegrityError("FAIL CLOSED: missing %s" % p)
        verify_against_sums(p, sums)
    return np.load(fp, allow_pickle=True), np.load(pack, allow_pickle=True)


def build(products, out_root):
    out = os.path.join(out_root, "fp")
    os.makedirs(out, exist_ok=True)
    sums = {}

    # the committed production routines -- copied, never re-implemented
    from CDDF_analysis.hbi_mcmc.fp_ladder import (
        perks_log_share, lambda_calibration_posterior_quantiles)

    F, P = _load(products, CAL_FAMILY, sums)
    fp_counts = np.asarray(F["fp_counts"], np.int64)
    nhat_edges = np.asarray(P["nhat_edges"], float)
    snr_edges = np.asarray(P["snr_edges"], float)
    live = np.asarray(P["dX"], float).sum(axis=0) > 0
    C, S = fp_counts.shape
    K = int(C * live.sum())
    a0 = 1.0 / K
    n_fp = int(fp_counts.sum())
    m = perks_log_share(fp_counts, live)                 # a0 = 1/K by default
    share = np.zeros_like(m)
    share[:, live] = np.exp(m[:, live])

    # every family shares the loa-0 counts; only ell_eff differs by support
    per_family = {}
    for fam in FAMILIES:
        Ff, _Pf = _load(products, fam, sums)
        if not np.array_equal(np.asarray(Ff["fp_counts"], np.int64), fp_counts):
            raise ManifestIntegrityError(
                "FAIL CLOSED: fp_counts differ between families (%s)" % fam)
        ell = float(Ff["fp_ell_eff"])
        per_family[fam] = {
            "fp_ell_eff": ell,
            "fp_w_sightline_ratio": float(Ff["fp_w_sightline_ratio"]),
            "support_id": str(Ff["support_id"]),
            "lambda_gamma_shape": n_fp + 0.5,
            "lambda_gamma_scale": 1.0 / ell,
            "lambda_median_J1": float(lambda_calibration_posterior_quantiles(
                fp_counts, ell, 1)[0]),
            "lambda_imputations_J8": [float(x) for x in
                                      lambda_calibration_posterior_quantiles(
                                          fp_counts, ell, J_PRODUCTION)],
        }

    npz = os.path.join(out, "fp_template.npz")
    np.savez(
        npz,
        fp_counts=fp_counts, nhat_edges=nhat_edges, snr_edges=snr_edges,
        live_stratum=live, a0=np.float64(a0), K_live_cells=np.int64(K),
        n_fp_events=np.int64(n_fp), perks_log_share=m, perks_share=share,
        lambda_gamma_shape=np.float64(n_fp + 0.5),
        **{("fp_ell_eff_%s" % f): np.float64(per_family[f]["fp_ell_eff"])
           for f in FAMILIES},
        **{("lambda_imputations_J8_%s" % f):
           np.asarray(per_family[f]["lambda_imputations_J8"], float)
           for f in FAMILIES})

    rows = []
    for c in range(C):
        for s in range(S):
            rows.append((nhat_edges[c], nhat_edges[c + 1], snr_edges[s],
                         snr_edges[s + 1], int(bool(live[s])),
                         int(fp_counts[c, s]), float(share[c, s])))
    csv_path = os.path.join(out, "fp_counts.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        for r in rows:
            w.writerow(r)

    spec = {
        "schema": SCHEMA,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "authority": "PI ruling 2026-09-14 sec.1, sec.8, sec.9, sec.11",
        "template": {
            "source": "loa-0 twin false-positive calibration",
            "n_events": n_fp,
            "grid": "29 observed log10 Nhat bins x 8 S/N strata "
                    "(%d live strata, K = %d live cells)"
                    % (int(live.sum()), K),
            "a0": a0,
            "a0_rule": "Perks pseudo-count a0 = 1/K (PI sec.9: baseline; the "
                       "factor-4 bracket is a named systematic and Jeffreys "
                       "a0 = 1/2 is an outer stress envelope only)",
            "log_share": "m_cs = log((n_cs + a0) / (N_FP + K a0)) on live "
                         "cells; sum of exp(m) over live cells = 1",
            "shape_freedom": "NONE. Fixed template: no survey-learned shape, "
                             "no saturated logits, no survey update of "
                             "Lambda, no per-cell fields",
            "survey_freedom": "coarse-z transfer t_K only (3 sites), "
                              "t_K ~ Normal(0, 1)",
            "calibration_support_note": "the calibration has 0 events at "
                                        "Nhat >= 20.3, so the FP block there "
                                        "is 100 % pseudo-count",
        },
        "lambda_posterior": {
            "form": "p(Lambda | D_loa0) = Gamma(N_FP + 1/2, ell_eff) "
                    "(shape %.1f)" % (n_fp + 0.5),
            "imputation_rule": "Lambda_j = Q_Gamma((j + 1/2)/J), j = 0..J-1 "
                               "(deterministic, seed-free); J = 1 is the "
                               "median screen, J = %d the production posterior"
                               % J_PRODUCTION,
            "per_family": per_family,
        },
        "fold": "mu^FP_cks = fp_w * ell_eff * (1 - eta_c) * exp(t_K(k)) * "
                "Lambda * pi_cs * E_ks, pi_cs = exp(m_cs)",
        "upstream_product": {
            "path": LOA0_FP,
            "sha256": (sha256_file(LOA0_FP) if os.path.isfile(LOA0_FP)
                       else None),
        },
        "files": {"fp_template.npz": None, "fp_counts.csv": None},
    }
    for name in list(spec["files"]):
        spec["files"][name] = sha256_file(os.path.join(out, name))
    with open(os.path.join(out, "lambda_spec.json"), "w") as fh:
        json.dump(spec, fh, indent=1)
        fh.write("\n")

    _readme(out, spec, per_family)
    return out, spec


def _readme(out, spec, per_family):
    t = spec["template"]
    lam = per_family[CAL_FAMILY]
    txt = """# loa-0 false-positive template — Paper 1 model of record

**Status: FROZEN (PI ruling 2026-09-14).** The false-positive block of the
forward model is a *fixed* calibration object: {n} loa-0 events on a
{grid}, smoothed with the Perks pseudo-count **a0 = 1/K = {a0:.9g}**.

    m_cs = log((n_cs + a0) / (N_FP + K a0))        (live cells only)
    pi_cs = exp(m_cs),   sum over live cells = 1
    mu^FP_cks = fp_w * ell_eff * (1 - eta_c) * exp(t_K(k)) * Lambda * pi_cs * E_ks

The calibration enters the survey fit ONLY through the cut posterior

    p(Lambda | D_loa0) = Gamma(N_FP + 1/2, ell_eff) = Gamma({shape:.1f}, {ell:.6f})

realised by the deterministic stratified imputations
`Lambda_j = Q_Gamma((j + 1/2) / J)`.  For the {J} production imputations of
this family: {imps}.

**What this object is NOT.**

* It is not a fitted FP population.  There is no survey-learned shape, no
  saturated logits, no survey update of Lambda and no per-cell fields.  The
  only survey-fitted FP quantity is the coarse-z transfer `t_K` (3 sites,
  `t_K ~ Normal(0, 1)`).
* It must not be read as an accurate reconstruction of the false-positive
  population.  The FP / low-N absorber decomposition is only weakly
  identified; what is defensible is the *robustness of the integrated
  incidence* to the bounded FP ambiguity (systematics S3 and S4).
* {supp}
* Completeness is NOT purity.  `1 - C` is not contamination; the FP term is a
  separate component of the model and must be presented separately.

Files: `fp_template.npz` (counts, edges, live mask, a0, Perks shares, Lambda
imputations per family), `fp_counts.csv` (long format, round-trippable),
`lambda_spec.json` (the posterior specification and the fold equation).

Licence: <PLACEHOLDER — CC-BY-4.0 proposed for data, PI to confirm>
""".format(n=t["n_events"], grid=t["grid"], a0=t["a0"],
           shape=lam["lambda_gamma_shape"], ell=lam["fp_ell_eff"], J=J_PRODUCTION,
           imps=", ".join("%.4f" % x for x in lam["lambda_imputations_J8"]),
           supp=t["calibration_support_note"][0].upper()
           + t["calibration_support_note"][1:] + ".")
    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--products", required=True)
    ap.add_argument("--out", required=True, help="the release root")
    ap.add_argument("--no-sums", action="store_true")
    a = ap.parse_args(argv)
    out, spec = build(a.products, a.out)
    if not a.no_sums:
        write_sha256sums(out)
    print("fp template ->", out)
    print("  %d events, a0 = %.9g, K = %s"
          % (spec["template"]["n_events"], spec["template"]["a0"],
             spec["template"]["grid"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
