#!/usr/bin/env python
"""systematics_table.py -- the EIGHT NAMED systematics of PI ruling 2026-09-14
sec.18, built from the frozen mock run JSONs (never typed in by hand).

    (1) redshift-dependent transfer residual      (signed, per z bin -- NOT a scalar)
    (2) B-vs-E response-form sensitivity          (signed; never |B-E|/2)
    (3) FP-scale / absorber decomposition         (ORACLE -> M1CUT shift)
    (4) high-Nhat FP pseudo-count sensitivity     (a0 battery, factor-4 bracket)
    (5) measured-vs-smooth phi sensitivity
    (6) completeness calibration covariance       (category 3; estimand-level PENDING)
    (7) sub-floor transport sensitivity           (first ladder A0-P6bcal vs A0)
    (8) sampler-geometry disclosure               (E-BFMI, divergences, seed spread)

Design rules (as for the rest of ``validation/release``): FAIL CLOSED -- a run
that the table needs and cannot find is an error, never a silently dropped row;
every number is a pure function of the run JSONs on disk; no quadrature, no
averaging of B and E, no collapse of the z-bin residual to one number.

VALIDATION-ONLY.  Mock products only; no real data; no sampler.

    python -m validation.release.systematics_table \
        --products /scratch/.../absorber_ladder_2026-09-13 \
        --out      /scratch/.../absorber_ladder_2026-09-13/release
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from hashutil import write_sha256sums, ManifestIntegrityError   # noqa: E402

SCHEMA = "zenodo_release/systematics/v1"
FAMILIES = ("2lpt0", "london0", "saclay0")
THRESHOLDS = ("ge20.0", "ge20.3")
MODEL_OF_RECORD = "B + phi_2LPT + C1nsadd + M1CUT (a0 = 1/K)"

# arm -> (runs subdirectory, run-directory name, file stem)
ARMS = {
    "REF":          ("final/runs", "REF-A0+C1nsadd",              "A0+C1nsadd"),
    "B_oracle":     ("final/runs", "B-phi2lpt-C1nsadd",           "B-phi2lpt-C1nsadd"),
    "E_oracle":     ("final/runs", "E-phi2lpt-C1nsadd",           "E-phi2lpt-C1nsadd"),
    "B_m1cut":      ("final/runs", "B-phi2lpt-C1nsadd-M1CUTj1",   "B-phi2lpt-C1nsadd-M1CUTj1"),
    "E_m1cut":      ("final/runs", "E-phi2lpt-C1nsadd-M1CUTj1",   "E-phi2lpt-C1nsadd-M1CUTj1"),
    "B_phismooth":  ("final/runs", "B-phismooth-C1nsadd",         "B-phismooth-C1nsadd"),
    "E_phismooth":  ("final/runs", "E-phismooth-C1nsadd",         "E-phismooth-C1nsadd"),
    "B_phifamily":  ("final/runs", "B-phifamilyDIAG-C1nsadd",     "B-phifamilyDIAG-C1nsadd"),
    "A0_base":      ("runs",       "A0",                          "A0"),
    "A0_P6bcal":    ("runs",       "A0-P6bcal",                   "A0-P6bcal"),
}
A0_BATTERY_ARM = ("final/runs", "B-phi2lpt-C1nsadd-M1CUTj1-a0batt",
                  "B-phi2lpt-C1nsadd-M1CUTj1-a0batt")
# the battery ran at one seed; the a0 = 1/K reference is the F2 run at that seed
A0_BATTERY_SEED = 20260811


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _load_arm(products, arm, families=FAMILIES, required=True):
    """``{family: [run_json, ...]}`` over every seed present, sorted by seed."""
    sub, directory, stem = ARMS[arm]
    root = os.path.join(products, sub, directory)
    out = {}
    for fam in families:
        pat = os.path.join(root, "RUN_%s_%s_s*.json" % (stem, fam))
        paths = sorted(p for p in glob.glob(pat) if "_a0" not in os.path.basename(p))
        if not paths:
            if required:
                raise ManifestIntegrityError(
                    "FAIL CLOSED: no run JSON for arm %r family %r under %s"
                    % (arm, fam, root))
            continue
        out[fam] = [(_seed_of(p), json.load(open(p)), p) for p in paths]
    return out


def _seed_of(path):
    tag = os.path.basename(path).rsplit("_s", 1)[-1]
    return int(tag.split("_")[0].split(".")[0])


def _load_a0_battery(products, families=FAMILIES):
    """``{family: {a0: run_json}}`` for the battery runs (a0 != 1/K)."""
    sub, directory, stem = A0_BATTERY_ARM
    root = os.path.join(products, sub, directory)
    out = {}
    for fam in families:
        pat = os.path.join(root, "RUN_%s_%s_s%d_a0*.json"
                           % (stem, fam, A0_BATTERY_SEED))
        paths = sorted(glob.glob(pat))
        if not paths:
            raise ManifestIntegrityError(
                "FAIL CLOSED: no a0-battery runs for family %r under %s"
                % (fam, root))
        fam_out = {}
        for p in paths:
            tag = os.path.basename(p).rsplit("_a0", 1)[-1][:-len(".json")]
            fam_out[float(tag)] = json.load(open(p))
        out[fam] = fam_out
    return out


# --------------------------------------------------------------------------
# per-run readouts
# --------------------------------------------------------------------------
def bias_pct(run, threshold):
    return float(run["thresholds"][threshold]["median_bias_pct"])


def hw68_pct(run, threshold):
    """The posterior 68 % HALF-width as a percentage of the median."""
    p16, p50, p84 = run["thresholds"][threshold]["post_p16_50_84"]
    return 50.0 * (p84 - p16) / p50


def paper1_bins(run, threshold):
    """``[(bin, (z_lo, z_hi), signed bias %, truth_in_68), ...]``."""
    cells = run["perz_recovery"]["estimand"][threshold]["paper1_bins"]
    return [(c["bin"], tuple(c["z"]), float(c["median_bias_pct"]),
             bool(c["truth_in_68"])) for c in cells if c.get("available", True)]


def _mean(values):
    return float(np.mean(np.asarray(values, float)))


def _seed_rollup(runs, fn):
    """Mean over seeds of ``fn(run)`` plus the seed list and the spread."""
    vals = [fn(r) for _s, r, _p in runs]
    seeds = [s for s, _r, _p in runs]
    return {"value": _mean(vals), "per_seed": vals, "seeds": seeds,
            "seed_spread": float(max(vals) - min(vals)) if len(vals) > 1 else 0.0}


def _arm_headlines(arm_runs):
    """``{family: {threshold: rollup}}``."""
    return {fam: {t: _seed_rollup(runs, lambda r, t=t: bias_pct(r, t))
                  for t in THRESHOLDS}
            for fam, runs in arm_runs.items()}


# --------------------------------------------------------------------------
# the eight systematics
# --------------------------------------------------------------------------
def _sys1_zbin(b_m1cut, b_oracle):
    """(1) redshift-dependent transfer residual -- SIGNED, per bin, never a scalar."""
    rows = []
    for label, arm in (("model_of_record_M1CUT", b_m1cut),
                       ("ORACLE_FP_diagnostic", b_oracle)):
        for fam, runs in arm.items():
            # seed-mean per bin, bins keyed by name
            per_bin = {}
            for t in THRESHOLDS:
                acc = {}
                for _s, run, _p in runs:
                    for name, z, val, in68 in paper1_bins(run, t):
                        acc.setdefault((name, z), []).append((val, in68))
                for (name, z), vals in acc.items():
                    per_bin[(t, name)] = {
                        "threshold": t, "bin": name, "z_lo": z[0], "z_hi": z[1],
                        "bias_pct": _mean([v for v, _ in vals]),
                        "truth_in_68_all_seeds": all(i for _v, i in vals),
                        "n_seeds": len(vals)}
            for key in sorted(per_bin):
                r = dict(per_bin[key]); r["family"] = fam; r["arm"] = label
                rows.append(r)
    mor = [r for r in rows if r["arm"] == "model_of_record_M1CUT"]
    hi = [r for r in mor if r["threshold"] == "ge20.3"]
    return {
        "id": "S1",
        "name": "redshift-dependent transfer residual",
        "ruling": "PI 2026-09-14 sec.5, sec.6, sec.18(1)",
        "treatment": "propagated / disclosed as a signed per-bin calibration "
                     "systematic; the frozen per-bin CRIT v2 gate is formally "
                     "FAILED and must never be described as passed; NOT fitted "
                     "away (R2 and C2 are closed for Paper 1)",
        "presentation": "signed per-bin table; do not collapse to one scalar",
        "summary": {
            "ge20.3_min_pct": min(r["bias_pct"] for r in hi),
            "ge20.3_max_pct": max(r["bias_pct"] for r in hi),
            "n_bins_truth_outside_68_ge20.3": sum(
                0 if r["truth_in_68_all_seeds"] else 1 for r in hi),
            "identified_candidate_cause":
                "the frozen g surface carries no S/N dependence while the "
                "truth's redshift shape does",
        },
        "rows": rows,
    }


def _sys2_response_form(b_oracle, e_oracle, b_m1cut, e_m1cut, ref):
    rows = []
    for fp_label, bb, ee in (("ORACLE", b_oracle, e_oracle),
                             ("M1CUT", b_m1cut, e_m1cut)):
        for fam in FAMILIES:
            if fam not in bb or fam not in ee:
                continue
            for t in THRESHOLDS:
                b = _seed_rollup(bb[fam], lambda r, t=t: bias_pct(r, t))
                e = _seed_rollup(ee[fam], lambda r, t=t: bias_pct(r, t))
                hw_rec = _seed_rollup(bb[fam], lambda r, t=t: hw68_pct(r, t))
                hw_ref = (_seed_rollup(ref[fam], lambda r, t=t: hw68_pct(r, t))
                          if fam in ref else None)
                d = e["value"] - b["value"]
                rows.append({
                    "fp_configuration": fp_label, "family": fam,
                    "threshold": t,
                    "baseline_B_bias_pct": b["value"],
                    "alternate_E_bias_pct": e["value"],
                    "signed_E_minus_B_pp": d,
                    "in_hw68_model_of_record": d / hw_rec["value"],
                    "in_hw68_reference_ladder": (d / hw_ref["value"]
                                                 if hw_ref else None),
                    "n_seeds_B": len(bb[fam]), "n_seeds_E": len(ee[fam])})
    orc = [r for r in rows if r["fp_configuration"] == "ORACLE"]
    return {
        "id": "S2",
        "name": "B-vs-E response-form sensitivity",
        "ruling": "PI 2026-09-14 sec.1, sec.13, sec.18(2)",
        "treatment": "two preserved posteriors; signed, one-sided, asymmetric "
                     "envelope. E is NOT a second model of record; no model "
                     "averaging; |B - E| / 2 is NOT a 1 sigma",
        "summary": {
            "envelope_ORACLE_ge20.0_pp": max(
                r["signed_E_minus_B_pp"] for r in orc if r["threshold"] == "ge20.0"),
            "envelope_ORACLE_ge20.3_pp": max(
                r["signed_E_minus_B_pp"] for r in orc if r["threshold"] == "ge20.3"),
            "sign": "one-sided: E >= B on every family and both thresholds "
                    "under ORACLE",
        },
        "rows": rows,
    }


def _sys3_fp_scale(b_oracle, b_m1cut):
    rows = []
    for fam in FAMILIES:
        for t in THRESHOLDS:
            o = _seed_rollup(b_oracle[fam], lambda r, t=t: bias_pct(r, t))
            m = _seed_rollup(b_m1cut[fam], lambda r, t=t: bias_pct(r, t))
            rows.append({"family": fam, "threshold": t,
                         "oracle_truth_pinned_FP_bias_pct": o["value"],
                         "rule_compliant_FP_bias_pct": m["value"],
                         "shift_M1CUT_minus_ORACLE_pp": m["value"] - o["value"]})
    for fam in FAMILIES:
        run = b_m1cut[fam][0][1]
        d = run["diagnostics"]
        mu_fp = d["fp_by_block"]["mu_fp_total_p16_50_84"][1]
        census = float(np.sum(d["fp_truth"]["hostless_block"]))
        rows.append({"family": fam, "threshold": "n/a",
                     "fp_total_over_hostless_census": mu_fp / census,
                     "t_K0_posterior_mean": d["t_post_mean"][0]})
    sh = [r for r in rows if "shift_M1CUT_minus_ORACLE_pp" in r]
    return {
        "id": "S3",
        "name": "FP-scale / absorber decomposition sensitivity",
        "ruling": "PI 2026-09-14 sec.7, sec.8, sec.18(3)",
        "treatment": "caveat / named systematic. The FP <-> low-N absorber "
                     "decomposition is weakly identified; the integrated "
                     "estimands are much better identified. The paper must not "
                     "claim accurate FP reconstruction",
        "summary": {
            "shift_ge20.0_pp_min": min(r["shift_M1CUT_minus_ORACLE_pp"]
                                       for r in sh if r["threshold"] == "ge20.0"),
            "shift_ge20.0_pp_max": max(r["shift_M1CUT_minus_ORACLE_pp"]
                                       for r in sh if r["threshold"] == "ge20.0"),
            "shift_ge20.3_pp_min": min(r["shift_M1CUT_minus_ORACLE_pp"]
                                       for r in sh if r["threshold"] == "ge20.3"),
            "shift_ge20.3_pp_max": max(r["shift_M1CUT_minus_ORACLE_pp"]
                                       for r in sh if r["threshold"] == "ge20.3"),
        },
        "rows": rows,
    }


def _sys4_a0(battery, b_m1cut, products):
    """(4) a0 pseudo-count.  Reference = the a0 = 1/K run at the battery seed."""
    K = _live_cells(products)
    a0_record = 1.0 / K
    rows = []
    for fam in FAMILIES:
        ref_runs = [r for s, r, _p in b_m1cut[fam] if s == A0_BATTERY_SEED]
        if not ref_runs:
            raise ManifestIntegrityError(
                "FAIL CLOSED: no a0 = 1/K reference run for family %r at seed %d"
                % (fam, A0_BATTERY_SEED))
        ref = ref_runs[0]
        for a0 in sorted(battery[fam]) + [a0_record]:
            run = ref if a0 == a0_record else battery[fam][a0]
            fp_hi = _fp_counts_above(run, 20.3)
            row = {"family": fam, "a0": a0,
                   "is_record": bool(a0 == a0_record),
                   "in_factor4_bracket": bool(a0_record / 4.0 - 1e-12 <= a0
                                              <= 4.0 * a0_record + 1e-12
                                              or a0 == 0.0),
                   "fp_counts_ge20.3": fp_hi}
            for t in THRESHOLDS:
                row["bias_%s_pct" % t] = bias_pct(run, t)
                row["delta_vs_record_%s_pp" % t] = (bias_pct(run, t)
                                                    - bias_pct(ref, t))
            rows.append(row)
    bracket = [r for r in rows if r["in_factor4_bracket"] and not r["is_record"]]
    outer = [r for r in rows if r["a0"] == 0.5]
    return {
        "id": "S4",
        "name": "high-Nhat FP prior / pseudo-count sensitivity (a0)",
        "ruling": "PI 2026-09-14 sec.9, sec.18(4)",
        "treatment": "named FP-prior systematic inside the predeclared "
                     "factor-4 bracket; Jeffreys a0 = 1/2 is an OUTER STRESS "
                     "ENVELOPE only, not part of the bracket. a0 search closed",
        "summary": {
            "a0_record": a0_record, "K_live_cells": K,
            "bracket_max_abs_delta_ge20.0_pp": max(
                abs(r["delta_vs_record_ge20.0_pp"]) for r in bracket),
            "bracket_max_abs_delta_ge20.3_pp": max(
                abs(r["delta_vs_record_ge20.3_pp"]) for r in bracket),
            "jeffreys_delta_ge20.0_pp": [r["delta_vs_record_ge20.0_pp"]
                                         for r in outer],
            "jeffreys_delta_ge20.3_pp": [r["delta_vs_record_ge20.3_pp"]
                                         for r in outer],
        },
        "rows": rows,
    }


def _sys5_phi(b_oracle, b_phismooth, b_phifamily):
    rows = []
    for label, arm in (("phi_smooth_6coef", b_phismooth),
                       ("phi_family_measured_ORACLE_DIAGNOSTIC", b_phifamily)):
        for fam in FAMILIES:
            seeds = [s for s, _r, _p in arm[fam]]
            base = [(s, r) for s, r, _p in b_oracle[fam] if s in seeds]
            for t in THRESHOLDS:
                alt = _mean([bias_pct(r, t) for _s, r, _p in arm[fam]])
                ref = _mean([bias_pct(r, t) for _s, r in base])
                rows.append({"variant": label, "family": fam, "threshold": t,
                             "baseline_measured_phi_bias_pct": ref,
                             "variant_bias_pct": alt,
                             "shift_pp": alt - ref,
                             "seeds_compared": seeds})
    sm = [r for r in rows if r["variant"] == "phi_smooth_6coef"]
    orc = [r for r in rows if r["variant"].startswith("phi_family")]
    return {
        "id": "S5",
        "name": "measured-vs-smooth phi sensitivity",
        "ruling": "PI 2026-09-14 sec.1, sec.18(5)",
        "treatment": "predeclared response-calibration sensitivity. The "
                     "2LPT-measured per-cell phi is the object of record; the "
                     "family's own measured phi is an ORACLE diagnostic and "
                     "never a production object",
        "summary": {
            "phi_smooth_shift_ge20.0_pp": [r["shift_pp"] for r in sm
                                           if r["threshold"] == "ge20.0"],
            "phi_smooth_shift_ge20.3_pp": [r["shift_pp"] for r in sm
                                           if r["threshold"] == "ge20.3"],
            "phi_transfer_error_max_abs_pp": max(abs(r["shift_pp"])
                                                 for r in orc),
        },
        "rows": rows,
    }


def _sys6_completeness(products):
    path = os.path.join(products, "completeness",
                        "C1nsadd_covariance_2lpt0.npz")
    if not os.path.isfile(path):
        raise ManifestIntegrityError(
            "FAIL CLOSED: completeness covariance missing: %s" % path)
    V = np.load(path, allow_pickle=True)
    beta = np.asarray(V["beta"], float)
    keys = set(V.files)
    if "sd_fisher" not in keys or "cov_fisher" not in keys:
        raise ManifestIntegrityError(
            "FAIL CLOSED: completeness covariance lacks the Fisher block: %s"
            % sorted(keys))
    cov_key = "cov_fisher"
    sd = np.asarray(V["sd_fisher"], float)
    sd_boot = np.asarray(V["sd_bootstrap"], float)
    sd_half = np.asarray(V["sd_halfsplit"], float)
    rows = []
    names = ["b0", "b1 (x)", "b2 (x^2)", "b3 (x^3)",
             "b4 (log10 S/N)", "b5 (log10 S/N)^2"]
    for i, nm in enumerate(names):
        rows.append({"coefficient": nm, "value": float(beta[i]),
                     "fisher_sd": float(sd[i]),
                     "bootstrap_sd": float(sd_boot[i]),
                     "halfsplit_sd": float(sd_half[i])})
    return {
        "id": "S6",
        "name": "completeness calibration covariance (category 3)",
        "ruling": "PI 2026-09-14 sec.10, sec.11, sec.18(6)",
        "treatment": "category-3 calibration uncertainty. The coefficient "
                     "covariance (Fisher + 200-draw bootstrap + half-split) is "
                     "preserved and released for propagation; it was NOT "
                     "propagated through the final mock ladder",
        "summary": {
            "covariance_source": os.path.basename(path),
            "covariance_keys": ["cov_fisher", "cov_bootstrap",
                                "cov_halfsplit", "beta_bootstrap"],
            "estimand_level_size_pp": None,
            "estimand_level_status": "PENDING -- not propagated in the final "
                                     "ladder; the covariance is released so "
                                     "the Paper lane can propagate it",
        },
        "rows": rows,
    }


def _sys7_subfloor(a0_base, a0_p6bcal):
    rows = []
    for fam in FAMILIES:
        seeds = [s for s, _r, _p in a0_p6bcal[fam]]
        base = [(s, r) for s, r, _p in a0_base[fam] if s in seeds]
        for t in THRESHOLDS:
            tp = _mean([bias_pct(r, t) for _s, r, _p in a0_p6bcal[fam]])
            tr = _mean([bias_pct(r, t) for _s, r in base])
            rows.append({"family": fam, "threshold": t,
                         "truth_pinned_subfloor_bias_pct": tr,
                         "transported_subfloor_bias_pct": tp,
                         "shift_pp": tp - tr, "seeds_compared": seeds})
    return {
        "id": "S7",
        "name": "sub-floor transport sensitivity",
        "ruling": "PI 2026-09-14 sec.1, sec.18(7)",
        "treatment": "the sub-floor-host term is a FIXED TRANSPORTED "
                     "calibration term for survey use; truth-pinned was the "
                     "mock-certification convention only",
        "summary": {"max_abs_shift_pp": max(abs(r["shift_pp"]) for r in rows),
                    "measured_on": "first ladder, A0-P6bcal vs A0 "
                                   "(the arm that isolates the sub-floor term)"},
        "rows": rows,
    }


def _sys8_sampler(b_m1cut):
    rows = []
    for fam in FAMILIES:
        for seed, run, _p in b_m1cut[fam]:
            d = run["diagnostics"]
            mix = d.get("estimand_mixing", {})
            rows.append({
                "family": fam, "seed": seed,
                "divergences": int(run["divergences"]),
                "ebfmi_min": float(min(d["ebfmi_per_chain"])),
                "ebfmi_max": float(max(d["ebfmi_per_chain"])),
                "split_rhat_ge20.0": mix.get("dndx_dla_20p0_allz", {}).get("split_rhat"),
                "split_rhat_ge20.3": mix.get("dndx_dla_20p3_allz", {}).get("split_rhat"),
                "ess_ge20.0": mix.get("dndx_dla_20p0_allz", {}).get("ess"),
                "ess_ge20.3": mix.get("dndx_dla_20p3_allz", {}).get("ess"),
            })
    spread = []
    for fam in FAMILIES:
        for t in THRESHOLDS:
            roll = _seed_rollup(b_m1cut[fam], lambda r, t=t: bias_pct(r, t))
            spread.append({"family": fam, "threshold": t,
                           "seed_spread_pp": roll["seed_spread"],
                           "seeds": roll["seeds"]})
    return {
        "id": "S8",
        "name": "sampler-geometry disclosure",
        "ruling": "PI 2026-09-14 sec.15, sec.18(8)",
        "treatment": "DISCLOSURE, not a bias budget. Low E-BFMI alone is not a "
                     "reason to reopen the model; the J = 8 certification asks "
                     "whether different imputations give the same science "
                     "marginal",
        "summary": {
            "ebfmi_min_over_runs": min(r["ebfmi_min"] for r in rows),
            "ebfmi_max_over_runs": max(r["ebfmi_max"] for r in rows),
            "divergences_max": max(r["divergences"] for r in rows),
            "headline_seed_spread_max_pp": max(r["seed_spread_pp"]
                                               for r in spread),
        },
        "rows": rows + [dict(r, kind="seed_spread") for r in spread],
    }


# --------------------------------------------------------------------------
def _live_cells(products, family="2lpt0"):
    """K = 29 observed bins x (number of live S/N strata) from the v3 pack."""
    pack = os.path.join(products, "support_v3",
                        "scanpack_%s_b300_v3.npz" % family)
    if not os.path.isfile(pack):
        raise ManifestIntegrityError("FAIL CLOSED: pack missing: %s" % pack)
    with np.load(pack, allow_pickle=True) as z:
        dX = np.asarray(z["dX"], float)
        counts = np.asarray(z["counts"], float)
    live = dX.sum(axis=0) > 0
    return int(counts.shape[0] * live.sum())


def _fp_counts_above(run, threshold_logN):
    """Posterior-median FP counts in observed bins with centre >= threshold."""
    d = run["diagnostics"]["fp_by_block"]
    grp = d.get("mu_fp_nhat_group_p16_50_84", {})
    for key in grp:
        lo = key.split("_")[0]
        try:
            if abs(float(lo) - float(threshold_logN)) < 1e-9:
                return float(grp[key][1])
        except ValueError:
            continue
    raise ManifestIntegrityError(
        "FAIL CLOSED: no FP Nhat group starting at %.1f in %r"
        % (threshold_logN, sorted(grp)))


# --------------------------------------------------------------------------
def build(products, out_root):
    out = os.path.join(out_root, "systematics")
    os.makedirs(out, exist_ok=True)

    arms = {k: _load_arm(products, k) for k in ARMS}
    battery = _load_a0_battery(products)

    systematics = [
        _sys1_zbin(arms["B_m1cut"], arms["B_oracle"]),
        _sys2_response_form(arms["B_oracle"], arms["E_oracle"],
                            arms["B_m1cut"], arms["E_m1cut"], arms["REF"]),
        _sys3_fp_scale(arms["B_oracle"], arms["B_m1cut"]),
        _sys4_a0(battery, arms["B_m1cut"], products),
        _sys5_phi(arms["B_oracle"], arms["B_phismooth"], arms["B_phifamily"]),
        _sys6_completeness(products),
        _sys7_subfloor(arms["A0_base"], arms["A0_P6bcal"]),
        _sys8_sampler(arms["B_m1cut"]),
    ]

    doc = {
        "schema": SCHEMA,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "authority": "PI ruling 2026-09-14 (MODEL FREEZE: ADOPT B + phi_2LPT "
                     "+ C1nsadd + M1CUT), sec.18",
        "model_of_record": MODEL_OF_RECORD,
        "products": products,
        "families": list(FAMILIES),
        "thresholds": list(THRESHOLDS),
        "units": {"bias_pct": "per cent of the mock truth",
                  "pp": "percentage points of that bias (a difference of two "
                        "bias percentages)",
                  "hw68": "the 68 % posterior half-width of the same estimand, "
                          "in per cent of the median"},
        "combination_rule": "NO blind quadrature. The eight effects are named, "
                            "signed and reported separately; the Paper lane "
                            "decides presentation (PI sec.18).",
        "n_systematics": len(systematics),
        "systematics": systematics,
    }
    with open(os.path.join(out, "SYSTEMATICS_TABLE.json"), "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=False)
        fh.write("\n")
    _write_csv(os.path.join(out, "SYSTEMATICS_TABLE.csv"), doc)
    _write_md(os.path.join(out, "SYSTEMATICS_TABLE.md"), doc)
    return out, doc


_CSV_HEADER = ("systematic_id", "name", "arm_or_variant", "family",
               "threshold", "bin", "quantity", "value", "unit")


def _write_csv(path, doc):
    rows = []
    for s in doc["systematics"]:
        for r in s["rows"]:
            arm = (r.get("arm") or r.get("variant") or r.get("fp_configuration")
                   or r.get("kind") or "")
            fam = r.get("family", "")
            thr = r.get("threshold", "")
            b = r.get("bin", "")
            for k, v in r.items():
                if k in ("arm", "variant", "fp_configuration", "family",
                         "threshold", "bin", "kind", "z_lo", "z_hi",
                         "seeds", "seeds_compared", "n_seeds", "n_seeds_B",
                         "n_seeds_E"):
                    continue
                unit = ("pp" if k.endswith("_pp") else
                        "pct" if k.endswith("_pct") else
                        "hw68" if "hw68" in k else "")
                rows.append((s["id"], s["name"], arm, fam, thr, b, k,
                             "" if v is None else v, unit))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_CSV_HEADER)
        for r in rows:
            w.writerow(r)
    return path


def _fmt(v, nd=2):
    if v is None:
        return "PENDING"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        return ("%+.*f" % (nd, v)) if abs(v) < 1e4 else "%.3e" % v
    return str(v)


def _write_md(path, doc):
    L = ["# SYSTEMATICS TABLE — the eight named effects (PI ruling "
         "2026-09-14 §18)", "",
         "**Model of record:** %s." % doc["model_of_record"],
         "Generated %s from the frozen run JSONs under `final/runs` and `runs`; "
         "every number is a pure function of those files." % doc["generated_utc"],
         "", "**Combination rule.** %s" % doc["combination_rule"], ""]
    for s in doc["systematics"]:
        L += ["## %s — %s" % (s["id"], s["name"]), "",
              "*Ruling:* %s. *Treatment:* %s." % (s["ruling"], s["treatment"]),
              ""]
        if s["id"] == "S1":
            L += ["Signed per-bin residual (mean over the available seeds of "
                  "the median bias %; \\* = truth outside the 68 % interval "
                  "in every seed). The published `F1_PAPER1_ZBIN_TABLE.md` "
                  "quotes seed 20260811 alone, so single-seed entries agree "
                  "exactly and two-seed entries differ by the seed noise "
                  "(<= 0.05 pp). "
                  "This table is the systematic; it is never collapsed to one "
                  "number.", ""]
            for arm in ("model_of_record_M1CUT", "ORACLE_FP_diagnostic"):
                L += ["### %s" % arm, "",
                      "| family | threshold | " + " | ".join(
                          b for b in _bins_of(s, arm)) + " |",
                      "|---" * (2 + len(_bins_of(s, arm))) + "|"]
                for fam in doc["families"]:
                    for t in doc["thresholds"]:
                        cells = [r for r in s["rows"]
                                 if r["arm"] == arm and r["family"] == fam
                                 and r["threshold"] == t]
                        cells.sort(key=lambda r: r["bin"])
                        L.append("| %s | %s | " % (fam, t) + " | ".join(
                            "%s%s" % (_fmt(c["bias_pct"]),
                                      "" if c["truth_in_68_all_seeds"] else "*")
                            for c in cells) + " |")
                L.append("")
        elif s["id"] == "S2":
            L += ["| FP config | family | threshold | B (record) % | E (alt) % "
                  "| signed E − B (pp) | in hw68(record) |",
                  "|---|---|---|---|---|---|---|"]
            for r in s["rows"]:
                L.append("| %s | %s | %s | %s | %s | %s | %.2f |" % (
                    r["fp_configuration"], r["family"], r["threshold"],
                    _fmt(r["baseline_B_bias_pct"]),
                    _fmt(r["alternate_E_bias_pct"]),
                    _fmt(r["signed_E_minus_B_pp"]),
                    r["in_hw68_model_of_record"]))
            L.append("")
        elif s["id"] == "S3":
            L += ["| family | threshold | ORACLE (truth-pinned FP) % | M1CUT "
                  "(rule-compliant FP) % | shift (pp) |",
                  "|---|---|---|---|---|"]
            for r in s["rows"]:
                if "shift_M1CUT_minus_ORACLE_pp" not in r:
                    continue
                L.append("| %s | %s | %s | %s | %s |" % (
                    r["family"], r["threshold"],
                    _fmt(r["oracle_truth_pinned_FP_bias_pct"]),
                    _fmt(r["rule_compliant_FP_bias_pct"]),
                    _fmt(r["shift_M1CUT_minus_ORACLE_pp"])))
            L.append("")
            for r in s["rows"]:
                if "fp_total_over_hostless_census" in r:
                    L.append("* %s: FP total / hostless census = %.3f, "
                             "t_K0 = %.2f" % (r["family"],
                                              r["fp_total_over_hostless_census"],
                                              r["t_K0_posterior_mean"]))
            L.append("")
        elif s["id"] == "S4":
            L += ["| family | a0 | in bracket | ≥20.0 % | Δ (pp) | ≥20.3 % | "
                  "Δ (pp) | FP counts ≥20.3 |",
                  "|---|---|---|---|---|---|---|---|"]
            for r in sorted(s["rows"], key=lambda r: (r["family"], r["a0"])):
                L.append("| %s | %.7g%s | %s | %s | %s | %s | %s | %s |" % (
                    r["family"], r["a0"], " (record)" if r["is_record"] else "",
                    "yes" if r["in_factor4_bracket"] else "OUTER",
                    _fmt(r["bias_ge20.0_pct"]),
                    _fmt(r["delta_vs_record_ge20.0_pp"]),
                    _fmt(r["bias_ge20.3_pct"]),
                    _fmt(r["delta_vs_record_ge20.3_pp"]),
                    ("%.1f" % r["fp_counts_ge20.3"])))
            L.append("")
        elif s["id"] == "S5":
            L += ["| variant | family | threshold | baseline (measured φ) % | "
                  "variant % | shift (pp) |", "|---|---|---|---|---|---|"]
            for r in s["rows"]:
                L.append("| %s | %s | %s | %s | %s | %s |" % (
                    r["variant"], r["family"], r["threshold"],
                    _fmt(r["baseline_measured_phi_bias_pct"]),
                    _fmt(r["variant_bias_pct"]), _fmt(r["shift_pp"])))
            L.append("")
        elif s["id"] == "S6":
            L += ["| coefficient | value | Fisher sd | bootstrap sd | "
                  "half-split sd |", "|---|---|---|---|---|"]
            for r in s["rows"]:
                L.append("| %s | %s | %s | %s | %s |" % (
                    r["coefficient"], _fmt(r["value"], 4),
                    _fmt(r["fisher_sd"], 4), _fmt(r["bootstrap_sd"], 4),
                    _fmt(r["halfsplit_sd"], 4)))
            L += ["", "Estimand-level size: **%s**."
                  % s["summary"]["estimand_level_status"], ""]
        elif s["id"] == "S7":
            L += ["| family | threshold | truth-pinned % | transported % | "
                  "shift (pp) |", "|---|---|---|---|---|"]
            for r in s["rows"]:
                L.append("| %s | %s | %s | %s | %s |" % (
                    r["family"], r["threshold"],
                    _fmt(r["truth_pinned_subfloor_bias_pct"]),
                    _fmt(r["transported_subfloor_bias_pct"]),
                    _fmt(r["shift_pp"])))
            L.append("")
        elif s["id"] == "S8":
            L += ["| family | seed | divergences | E-BFMI min | E-BFMI max | "
                  "split-R̂ ≥20.0 | split-R̂ ≥20.3 | ESS ≥20.0 | ESS ≥20.3 |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for r in s["rows"]:
                if r.get("kind") == "seed_spread":
                    continue
                L.append("| %s | %s | %d | %.2f | %.2f | %s | %s | %s | %s |" % (
                    r["family"], r["seed"], r["divergences"], r["ebfmi_min"],
                    r["ebfmi_max"], r["split_rhat_ge20.0"],
                    r["split_rhat_ge20.3"], r["ess_ge20.0"], r["ess_ge20.3"]))
            L += ["", "Headline seed spread (max over family × threshold): "
                  "%.3f pp." % s["summary"]["headline_seed_spread_max_pp"], ""]
        L += ["Summary: `%s`" % json.dumps(s["summary"]), ""]
    with open(path, "w") as fh:
        fh.write("\n".join(L))
    return path


def _bins_of(s, arm):
    seen = []
    for r in s["rows"]:
        if r["arm"] == arm and r["bin"] not in seen:
            seen.append(r["bin"])
    return sorted(seen)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--products", required=True)
    ap.add_argument("--out", required=True, help="the release root")
    ap.add_argument("--no-sums", action="store_true")
    a = ap.parse_args(argv)
    out, doc = build(a.products, a.out)
    if not a.no_sums:
        write_sha256sums(out)
    print("systematics table ->", out)
    for s in doc["systematics"]:
        print("  %s %-52s %d rows" % (s["id"], s["name"], len(s["rows"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
