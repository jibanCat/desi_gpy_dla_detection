#!/usr/bin/env python
"""systematics_table.py -- the EIGHT NAMED systematics of PI ruling 2026-09-14
sec.18, built from the frozen mock run JSONs (never typed in by hand).

    (1) redshift-dependent transfer residual      (signed, per z bin -- NOT a scalar;
                                                  arm of record = the J = 8 production runs)
    (2) B-vs-E response-form sensitivity          (signed; never |B-E|/2)
    (3) FP-scale / absorber decomposition         (ORACLE -> M1CUT shift)
    (4) high-Nhat FP pseudo-count sensitivity     (a0 battery, factor-4 bracket)
    (5) measured-vs-smooth phi sensitivity
    (6) completeness calibration covariance       (category 3; estimand-level PENDING)
    (7) sub-floor transport sensitivity           (first ladder A0-P6bcal vs A0)
    (8) sampler-geometry disclosure               (E-BFMI, divergences, rank-Rhat/ESS,
                                                  t_K mixing, imputation spread)

Design rules (as for the rest of ``validation/release``): FAIL CLOSED -- a run
that the table needs and cannot find is an error, never a silently dropped row;
every number is a pure function of the run JSONs on disk; no quadrature, no
averaging of B and E, no collapse of the z-bin residual to one number.

VALIDATION-ONLY.  Every SCIENCE number is a mock number and no sampler is run.
Two carefully bounded exceptions, both required by PI ruling 2026-09-14b:

  * ``--real-runs-dir`` adds the real survey runs' SAMPLER DIAGNOSTICS to S8
    (divergences, E-BFMI, rank-R-hat, ESS, t_K mixing).  No real median,
    interval, bias or any other estimand value is copied -- sec.11 and
    sec.18(8) require the disclosure to be accurate and current.
  * ``--real-pooled`` + ``--private-out`` write a SEPARATE PRIVATE companion
    (notes repository) expressing the same named sizes in units of the REAL
    pooled 68 % half-width.  The real half-widths are real-data values and
    never enter the release tree; the release table's ``in_hw68_*`` ratios are
    MOCK half-widths throughout.

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

from hashutil import (write_sha256sums, sha256_file,           # noqa: E402
                      ManifestIntegrityError)


def _sha256(path):
    return sha256_file(path) if path and os.path.isfile(path) else None

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
    # the PRODUCTION model of record: J = 8 stratified Lambda imputations,
    # one seed (20260811).  The J = 1 arm above is the FINAL-LADDER (F2)
    # single-imputation arm and is kept as HISTORY only.
    "B_m1cut_J8":   ("final/runs", "B-phi2lpt-C1nsadd-M1CUTJ8",   "B-phi2lpt-C1nsadd-M1CUTJ8"),
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


#: the J = 8 production campaign ran ONE seed; ``j`` indexes the imputation.
J8_SEED = 20260811


def _load_j8(products, families=FAMILIES, required=True):
    """``{family: [(seed, j, run_json, path), ...]}`` for the J = 8 production arm.

    The J = 8 runs are ``RUN_<stem>_<fam>_s<seed>_J8j<j>.json``.  They are NOT
    seed replicates: they are eight stratified draws of Lambda from
    p(Lambda | D_loa0) at one seed, which the sealed J = 8 rule pools with
    EQUAL WEIGHT.  Keeping ``j`` explicit is what stops the builder from
    silently reporting an imputation spread as seed noise (the defect this
    rebuild corrects).
    """
    sub, directory, stem = ARMS["B_m1cut_J8"]
    root = os.path.join(products, sub, directory)
    out = {}
    for fam in families:
        pat = os.path.join(root, "RUN_%s_%s_s*_J8j*.json" % (stem, fam))
        paths = sorted(glob.glob(pat))
        if not paths:
            if required:
                raise ManifestIntegrityError(
                    "FAIL CLOSED: no J = 8 production runs for family %r under %s"
                    % (fam, root))
            continue
        recs = []
        for pth in paths:
            j = json.load(open(pth))
            lam = (j["diagnostics"].get("lam_cut") or {})
            if int(lam.get("J", 0)) != 8:
                raise ManifestIntegrityError(
                    "FAIL CLOSED: %s is not a J = 8 run (lam_cut.J = %r)"
                    % (os.path.basename(pth), lam.get("J")))
            recs.append((_seed_of(pth), int(lam["j"]), j, pth))
        recs.sort(key=lambda r: (r[0], r[1]))
        js = [r[1] for r in recs]
        if sorted(js) != list(range(8)):
            raise ManifestIntegrityError(
                "FAIL CLOSED: family %r has imputations %r, expected 0..7"
                % (fam, sorted(js)))
        out[fam] = recs
    return out


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
            fam_out[float(tag)] = (json.load(open(p)), p)
        out[fam] = fam_out
    return out


# --------------------------------------------------------------------------
# Omega_HI[20.3, 21.6] -- deterministic read-out of the stored f draws
# --------------------------------------------------------------------------
#: the run JSONs' ``thresholds.omega_allz`` is the SUB-DLA window
#: [19.5, 20.3) (``key = omega_subdla_195_203_allz``); it is NOT the paper's
#: Omega_HI[20.3, 21.6].  Every Omega number in this table is therefore read
#: back from the stored ``_fdraws.npz`` with the COMMITTED helper
#: ``validation.fp_ladder.ladder_table.paper_omega_20p3_21p6``, which imports
#: the paper's own reduction weights read-only.  No definition is re-derived.
_OMEGA_CACHE = {}


def _ladder_table():
    """The committed Omega helper module (imported lazily, read-only)."""
    repo = os.path.dirname(os.path.dirname(_HERE))       # .../wt_abs_diag_2026-09
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from validation.fp_ladder import ladder_table as LT          # noqa: E402
    return LT


def _fdraws_of(run_path):
    f = run_path[:-len(".json")] + "_fdraws.npz"
    if not os.path.isfile(f):
        raise ManifestIntegrityError(
            "FAIL CLOSED: no stored f draws beside %s" % run_path)
    return f


def omega_allz(run_path):
    """``paper_omega_20p3_21p6`` on this run's stored draws (memoised)."""
    if run_path in _OMEGA_CACHE:
        return _OMEGA_CACHE[run_path]
    LT = _ladder_table()
    om = LT.paper_omega_20p3_21p6(_fdraws_of(run_path), pack=None)
    if om is None or "unavailable" in om or "blocked" in om:
        raise ManifestIntegrityError(
            "FAIL CLOSED: Omega[20.3,21.6] unavailable for %s: %r"
            % (os.path.basename(run_path), om))
    _OMEGA_CACHE[run_path] = om
    return om


def omega_bias_pct(run_path):
    return float(omega_allz(run_path)["median_bias_pct"])


def omega_hw68_pct(run_path):
    p16, p50, p84 = omega_allz(run_path)["post_p16_50_84"]
    return 50.0 * (p84 - p16) / p50


_OMEGA_BIN_CACHE = {}


def omega_paper1_bins(run_path):
    """``{bin: {"bias_pct", "hw68_pct", "truth_in_68", "dX", "coverage"}}``.

    Omega_HI[20.3, 21.6] restricted to each Paper-1 redshift bin.  The N_HI
    weight, the redshift weight and the prefactor are the PAPER's own
    (``hbi_reduction``, imported read-only exactly as
    ``ladder_table.paper_omega_20p3_21p6`` does); the per-bin redshift overlap
    weight is the committed ``reduce_truthfree._overlap_w``, i.e. the same
    weight the runner uses for the dN/dX per-bin numbers.  Nothing is
    re-derived here and no sampler is run.
    """
    if run_path in _OMEGA_BIN_CACHE:
        return _OMEGA_BIN_CACHE[run_path]
    LT = _ladder_table()
    if not os.path.isdir(LT.PAPER_FIGURES):
        raise ManifestIntegrityError(
            "FAIL CLOSED: paper_figures not found at %s" % LT.PAPER_FIGURES)
    if LT.PAPER_FIGURES not in sys.path:
        sys.path.insert(0, LT.PAPER_FIGURES)
    import hbi_reduction as HR                                   # noqa: E402
    from validation.real_c1 import reduce_truthfree as RT        # noqa: E402

    with np.load(_fdraws_of(run_path)) as z:
        f = np.asarray(z["f"], float)
        ft = np.asarray(z["truth_f"], float)
        n_edges = np.asarray(z["ntrue_edges"], float)
        zf = np.asarray(z["zf_edges"], float)
        dX = np.asarray(z["dX_k"], float)
    P = HR.Posterior.__new__(HR.Posterior)
    P.f, P.n_edges, P.z_edges, P.dX = f, n_edges, zf, dX
    ow = P._omega_weight(*HR.OMEGA_NHI)
    pre = float(HR.OMEGA_PREFACTOR_CM2)
    out = {}
    for name, lo, hi in RT.PAPER1_LOWZ_BINS:
        w = RT._overlap_w(zf, dX, lo, hi)
        ws = float(np.asarray(w, float).sum())
        cov = float(np.clip(min(hi, zf[-1]) - max(lo, zf[0]), 0, None) / (hi - lo))
        if ws <= 0.0:
            out[name] = {"available": False, "coverage": cov,
                         "z_lo": float(lo), "z_hi": float(hi)}
            continue
        post = pre * np.einsum("dbk,b,k->d", f, ow, w) / ws
        truth = float(pre * np.einsum("bk,b,k->", ft, ow, w) / ws)
        q = np.percentile(post, [16, 50, 84])
        out[name] = {"available": True, "z_lo": float(lo), "z_hi": float(hi),
                     "dX": ws, "coverage": cov, "truth": truth,
                     "median": float(q[1]),
                     "bias_pct": float(round(100.0 * (q[1] / truth - 1.0), 3)),
                     "hw68_pct": float(50.0 * (q[2] - q[0]) / q[1]),
                     "truth_in_68": bool(q[0] <= truth <= q[2])}
    _OMEGA_BIN_CACHE[run_path] = out
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


def _seed_rollup_path(runs, fn):
    """As ``_seed_rollup`` but for read-outs that need the run PATH (Omega)."""
    vals = [fn(pth) for _s, _r, pth in runs]
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
def _sys1_zbin(j8, b_m1cut_j1, b_oracle):
    """(1) redshift-dependent transfer residual -- SIGNED, per bin, never a scalar.

    ARM OF RECORD = the PRODUCTION J = 8 M1CUT runs, pooled with equal weight
    over the eight Lambda imputations (sealed J = 8 rule; PI 2026-09-14b sec.6:
    "use the production J = 8 mock-systematic read-out, not the J = 1 S1
    table").  HISTORY: the released table until 2026-09-15 carried the
    single-imputation J = 1 (F2) arm under the label
    ``model_of_record_M1CUT``; those rows are preserved verbatim below under
    the label ``M1CUT_J1_HISTORY`` so the correction is auditable.  The
    truth-pinned ORACLE (F1) arm is the third, diagnostic, arm.

    Each row also carries Omega_HI[20.3, 21.6] restricted to the same z bin,
    read back from the stored f draws (PI 2026-09-14b sec.10).
    """
    rows = []
    # ---- arm of record: J = 8 production, equal-weight over imputations ----
    for fam, recs in j8.items():
        for t in THRESHOLDS:
            acc = {}
            for _seed, jj, run, pth in recs:
                for name, z, val, in68 in paper1_bins(run, t):
                    acc.setdefault((name, z), []).append((jj, val, in68))
            for (name, z), vals in sorted(acc.items()):
                v = [x for _j, x, _i in vals]
                cov = _coverage_of(recs[0][2], t, name)
                row = {"arm": "model_of_record_M1CUT_J8_production",
                       "family": fam, "threshold": t, "bin": name,
                       "z_lo": z[0], "z_hi": z[1],
                       "nominal_coverage": cov,
                       "bias_pct": _mean(v),
                       "imputation_spread_pp": float(max(v) - min(v)),
                       "truth_in_68_all_imputations": all(i for _j, _x, i in vals),
                       "n_imputations": len(vals),
                       "seeds": sorted({sd for sd, _j, _r, _p in recs})}
                ob = [omega_paper1_bins(pp)[name]
                      for _sd, _jj, _rr, pp in recs]
                ob = [o for o in ob if o.get("available")]
                if ob:
                    ovals = [o["bias_pct"] for o in ob]
                    row["omega_20p3_21p6_bias_pct"] = _mean(ovals)
                    row["omega_20p3_21p6_imputation_spread_pp"] = float(
                        max(ovals) - min(ovals))
                    row["omega_20p3_21p6_hw68_pct"] = _mean(
                        [o["hw68_pct"] for o in ob])
                    row["omega_20p3_21p6_truth_in_68_all_imputations"] = all(
                        o["truth_in_68"] for o in ob)
                rows.append(row)
    # ---- history arms: J = 1 (the rows the released table used to carry)
    #      and the truth-pinned ORACLE F1 diagnostic ------------------------
    for label, arm in (("M1CUT_J1_HISTORY", b_m1cut_j1),
                       ("ORACLE_FP_diagnostic_F1", b_oracle)):
        for fam, runs in arm.items():
            for t in THRESHOLDS:
                acc = {}
                for _s, run, _p in runs:
                    for name, z, val, in68 in paper1_bins(run, t):
                        acc.setdefault((name, z), []).append((val, in68))
                for (name, z), vals in sorted(acc.items()):
                    v = [x for x, _i in vals]
                    row = {"arm": label, "family": fam, "threshold": t,
                           "bin": name, "z_lo": z[0], "z_hi": z[1],
                           "nominal_coverage": _coverage_of(runs[0][1], t, name),
                           "bias_pct": _mean(v),
                           "seed_spread_pp": float(max(v) - min(v)) if len(v) > 1 else 0.0,
                           "truth_in_68_all_seeds": all(i for _x, i in vals),
                           "n_seeds": len(vals),
                           "seeds": [sd for sd, _r, _p in runs]}
                    ovals = [omega_paper1_bins(pp)[name] for _s, _r, pp in runs]
                    ovals = [o for o in ovals if o.get("available")]
                    if ovals:
                        row["omega_20p3_21p6_bias_pct"] = _mean(
                            [o["bias_pct"] for o in ovals])
                        row["omega_20p3_21p6_hw68_pct"] = _mean(
                            [o["hw68_pct"] for o in ovals])
                    rows.append(row)
    rec = [r for r in rows if r["arm"] == "model_of_record_M1CUT_J8_production"]
    hi = [r for r in rec if r["threshold"] == "ge20.3"]
    j1 = [r for r in rows if r["arm"] == "M1CUT_J1_HISTORY"]
    b5_seed = {"%s_%s" % (r["family"], r["threshold"]): r["seed_spread_pp"]
               for r in j1 if r["bin"] == "B5"}
    # the J1 -> J8 relabelling, bin by bin, so the correction is quantified
    delta = []
    for r in rec:
        m = [q for q in j1 if q["family"] == r["family"]
             and q["threshold"] == r["threshold"] and q["bin"] == r["bin"]]
        if m:
            delta.append({"family": r["family"], "threshold": r["threshold"],
                          "bin": r["bin"],
                          "J8_minus_J1_pp": r["bias_pct"] - m[0]["bias_pct"]})
    return {
        "id": "S1",
        "name": "redshift-dependent transfer residual",
        "ruling": "PI 2026-09-14 sec.5, sec.6, sec.18(1); PI 2026-09-14b "
                  "sec.6 (use the production J = 8 read-out), sec.10 (Omega)",
        "treatment": "propagated / disclosed as a signed per-bin calibration "
                     "systematic; the frozen per-bin CRIT v2 gate is formally "
                     "FAILED and must never be described as passed; NOT fitted "
                     "away (R2 and C2 are closed for Paper 1)",
        "presentation": "signed per-bin table; do not collapse to one scalar",
        "arm_of_record": "model_of_record_M1CUT_J8_production",
        "history_note":
            "Until 2026-09-15 the rows labelled `model_of_record_M1CUT` in the "
            "released table were the J = 1 (F2) single-imputation arm, not the "
            "J = 8 production runs. The J = 1 rows are PRESERVED here under "
            "`M1CUT_J1_HISTORY` and the per-bin J8 - J1 difference is given in "
            "`summary.J8_minus_J1_pp_max_abs` / `rows_J8_minus_J1`; nothing was "
            "rewritten. The J = 8 campaign ran ONE seed (20260811), so its "
            "spread column is an IMPUTATION spread, never seed noise.",
        "summary": {
            "arm_of_record_n_imputations": 8,
            "arm_of_record_seeds": sorted({sd for recs in j8.values()
                                           for sd, _j, _r, _p in recs}),
            "ge20.3_min_pct": min(r["bias_pct"] for r in hi),
            "ge20.3_max_pct": max(r["bias_pct"] for r in hi),
            "n_bins_truth_outside_68_ge20.3": sum(
                0 if r["truth_in_68_all_imputations"] else 1 for r in hi),
            "max_imputation_spread_pp": max(r["imputation_spread_pp"]
                                            for r in rec),
            "J8_minus_J1_pp_max_abs": max(abs(d["J8_minus_J1_pp"])
                                          for d in delta) if delta else None,
            "B5_seed_spread_pp_J1_two_seed_arm": b5_seed,
            "omega_20p3_21p6_ge20.3_bin_bias_pct_range": [
                min(r["omega_20p3_21p6_bias_pct"] for r in hi
                    if "omega_20p3_21p6_bias_pct" in r),
                max(r["omega_20p3_21p6_bias_pct"] for r in hi
                    if "omega_20p3_21p6_bias_pct" in r)],
            "identified_candidate_cause":
                "the frozen g surface carries no S/N dependence while the "
                "truth's redshift shape does",
        },
        "rows": rows,
        "rows_J8_minus_J1": delta,
    }


def _coverage_of(run, threshold, name):
    """The stored ``coverage`` of a Paper-1 bin (B5 is only 25 % covered)."""
    for c in run["perz_recovery"]["estimand"][threshold]["paper1_bins"]:
        if c["bin"] == name:
            return c.get("coverage")
    return None


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
            # Omega_HI[20.3, 21.6] is a single all-z estimand: one row per
            # (FP configuration, family), threshold "omega_20p3_21p6".
            ob = _seed_rollup_path(bb[fam], omega_bias_pct)
            oe = _seed_rollup_path(ee[fam], omega_bias_pct)
            ohw = _seed_rollup_path(bb[fam], omega_hw68_pct)
            rows.append({
                "fp_configuration": fp_label, "family": fam,
                "threshold": "omega_20p3_21p6",
                "baseline_B_bias_pct": ob["value"],
                "alternate_E_bias_pct": oe["value"],
                "signed_E_minus_B_pp": oe["value"] - ob["value"],
                "in_hw68_model_of_record": (oe["value"] - ob["value"]) / ohw["value"],
                "in_hw68_reference_ladder": None,
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
            "envelope_ORACLE_omega_20p3_21p6_pp": max(
                r["signed_E_minus_B_pp"] for r in orc
                if r["threshold"] == "omega_20p3_21p6"),
            "omega_20p3_21p6_signed_E_minus_B_pp": {
                "%s_%s" % (r["fp_configuration"], r["family"]):
                    r["signed_E_minus_B_pp"] for r in rows
                if r["threshold"] == "omega_20p3_21p6"},
            "sign": "one-sided: E >= B on every family and both thresholds "
                    "under ORACLE",
        },
        "rows": rows,
    }


def _sys3_fp_scale(b_oracle, b_m1cut, j8=None):
    rows = []
    for fam in FAMILIES:
        for t in THRESHOLDS:
            o = _seed_rollup(b_oracle[fam], lambda r, t=t: bias_pct(r, t))
            m = _seed_rollup(b_m1cut[fam], lambda r, t=t: bias_pct(r, t))
            rows.append({"arm": "M1CUT_J1", "family": fam, "threshold": t,
                         "oracle_truth_pinned_FP_bias_pct": o["value"],
                         "rule_compliant_FP_bias_pct": m["value"],
                         "shift_M1CUT_minus_ORACLE_pp": m["value"] - o["value"]})
        # Omega_HI[20.3, 21.6]: the same ORACLE -> M1CUT shift (PI 14b sec.10)
        oo = _seed_rollup_path(b_oracle[fam], omega_bias_pct)
        om = _seed_rollup_path(b_m1cut[fam], omega_bias_pct)
        rows.append({"arm": "M1CUT_J1", "family": fam,
                     "threshold": "omega_20p3_21p6",
                     "oracle_truth_pinned_FP_bias_pct": oo["value"],
                     "rule_compliant_FP_bias_pct": om["value"],
                     "shift_M1CUT_minus_ORACLE_pp": om["value"] - oo["value"]})
    if j8:
        # the PRODUCTION arm: equal-weight over the eight Lambda imputations
        for fam, recs in j8.items():
            for t in THRESHOLDS:
                o = _seed_rollup(b_oracle[fam], lambda r, t=t: bias_pct(r, t))
                m = _mean([bias_pct(r, t) for _s, _j, r, _p in recs])
                rows.append({"arm": "M1CUT_J8_production", "family": fam,
                             "threshold": t,
                             "oracle_truth_pinned_FP_bias_pct": o["value"],
                             "rule_compliant_FP_bias_pct": m,
                             "shift_M1CUT_minus_ORACLE_pp": m - o["value"]})
            oo = _seed_rollup_path(b_oracle[fam], omega_bias_pct)
            om = _mean([omega_bias_pct(pp) for _s, _j, _r, pp in recs])
            rows.append({"arm": "M1CUT_J8_production", "family": fam,
                         "threshold": "omega_20p3_21p6",
                         "oracle_truth_pinned_FP_bias_pct": oo["value"],
                         "rule_compliant_FP_bias_pct": om,
                         "shift_M1CUT_minus_ORACLE_pp": om - oo["value"]})
    for fam in FAMILIES:
        run = b_m1cut[fam][0][1]
        d = run["diagnostics"]
        mu_fp = d["fp_by_block"]["mu_fp_total_p16_50_84"][1]
        census = float(np.sum(d["fp_truth"]["hostless_block"]))
        rows.append({"arm": "M1CUT_J1", "family": fam, "threshold": "n/a",
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
            "shift_omega_20p3_21p6_pp_min": min(
                r["shift_M1CUT_minus_ORACLE_pp"] for r in sh
                if r["threshold"] == "omega_20p3_21p6"),
            "shift_omega_20p3_21p6_pp_max": max(
                r["shift_M1CUT_minus_ORACLE_pp"] for r in sh
                if r["threshold"] == "omega_20p3_21p6"),
        },
        "rows": rows,
    }


def _sys4_a0(battery, b_m1cut, products):
    """(4) a0 pseudo-count.  Reference = the a0 = 1/K run at the battery seed."""
    K = _live_cells(products)
    a0_record = 1.0 / K
    rows = []
    for fam in FAMILIES:
        ref_runs = [(r, pth) for s, r, pth in b_m1cut[fam]
                    if s == A0_BATTERY_SEED]
        if not ref_runs:
            raise ManifestIntegrityError(
                "FAIL CLOSED: no a0 = 1/K reference run for family %r at seed %d"
                % (fam, A0_BATTERY_SEED))
        ref, ref_path = ref_runs[0]
        for a0 in sorted(battery[fam]) + [a0_record]:
            run, run_path = ((ref, ref_path) if a0 == a0_record
                             else battery[fam][a0])
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
            row["bias_omega_20p3_21p6_pct"] = omega_bias_pct(run_path)
            row["delta_vs_record_omega_20p3_21p6_pp"] = (
                omega_bias_pct(run_path) - omega_bias_pct(ref_path))
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
            "bracket_max_abs_delta_omega_20p3_21p6_pp": max(
                abs(r["delta_vs_record_omega_20p3_21p6_pp"]) for r in bracket),
            "jeffreys_delta_omega_20p3_21p6_pp": [
                r["delta_vs_record_omega_20p3_21p6_pp"] for r in outer],
        },
        "rows": rows,
    }


def _sys5_phi(b_oracle, b_phismooth, b_phifamily):
    rows = []
    for label, arm in (("phi_smooth_6coef", b_phismooth),
                       ("phi_family_measured_ORACLE_DIAGNOSTIC", b_phifamily)):
        for fam in FAMILIES:
            seeds = [s for s, _r, _p in arm[fam]]
            base = [(s, r, pth) for s, r, pth in b_oracle[fam] if s in seeds]
            for t in THRESHOLDS:
                alt = _mean([bias_pct(r, t) for _s, r, _p in arm[fam]])
                ref = _mean([bias_pct(r, t) for _s, r, _p in base])
                rows.append({"variant": label, "family": fam, "threshold": t,
                             "baseline_measured_phi_bias_pct": ref,
                             "variant_bias_pct": alt,
                             "shift_pp": alt - ref,
                             "seeds_compared": seeds})
            oalt = _mean([omega_bias_pct(pth) for _s, _r, pth in arm[fam]])
            oref = _mean([omega_bias_pct(pth) for _s, _r, pth in base])
            rows.append({"variant": label, "family": fam,
                         "threshold": "omega_20p3_21p6",
                         "baseline_measured_phi_bias_pct": oref,
                         "variant_bias_pct": oalt,
                         "shift_pp": oalt - oref,
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
            "phi_smooth_shift_omega_20p3_21p6_pp": [
                r["shift_pp"] for r in sm
                if r["threshold"] == "omega_20p3_21p6"],
        },
        "rows": rows,
    }


#: the S6 propagation is the ONLY additional science computation the PI
#: authorised (ruling 2026-09-14b sec.9).  It is run elsewhere; this table
#: carries a MACHINE-READABLE PLACEHOLDER and fills itself the moment the
#: result file lands.  Nothing is ever invented here.
S6_RESULT_BASENAME = "S6_COMPLETENESS_PROPAGATION_RESULT.json"


def _s6_search_paths(products, explicit=None):
    cand = []
    if explicit:
        cand.append(explicit)
    cand += [os.path.join(products, S6_RESULT_BASENAME),
             os.path.join(products, "completeness", S6_RESULT_BASENAME),
             os.path.join(products, "release", "systematics", S6_RESULT_BASENAME),
             os.path.join(os.path.expanduser("~"), "desi_gpy_dla_notes",
                          "governance", "final_campaign_2026-09-13",
                          S6_RESULT_BASENAME)]
    return cand


def _load_s6_result(products, explicit=None, search_defaults=True):
    """``(path, payload)`` of the S6 propagation result, or ``(None, None)``."""
    cand = _s6_search_paths(products, explicit) if search_defaults \
        else ([explicit] if explicit else [])
    for c in cand:
        if c and os.path.isfile(c):
            with open(c) as fh:
                return c, json.load(fh)
    return None, None


def _sys6_completeness(products, s6_path=None, search_defaults=True):
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
    found, payload = _load_s6_result(products, s6_path, search_defaults)
    if payload is None:
        estimand = {
            "estimand_level_size_pp": None,
            "estimand_level_size_omega_20p3_21p6_pp": None,
            "estimand_level_status": "PENDING -- not propagated in the final "
                                     "ladder; the covariance is released so "
                                     "the Paper lane can propagate it",
            "result_file_expected": S6_RESULT_BASENAME,
            "result_file_found": None,
            "result_file_searched": _s6_search_paths(products, s6_path),
            "authority": "PI 2026-09-14b sec.9 (propagation APPROVED, no refit)",
        }
    else:
        # PRIVACY GATE.  The S6 propagation is evaluated against the REAL
        # pooled posterior, so its result file carries real-data estimand
        # values.  The release product therefore records that the propagation
        # EXISTS, where it lives and its digest -- and copies numbers only from
        # an explicit ``release_safe`` block that the result itself declares.
        # The full sizes go to the PRIVATE companion instead.  Never dump the
        # payload here: a release table is a public product.
        safe = payload.get("release_safe") or {}
        classification = str(payload.get("classification", ""))
        private = ("PRIVATE" in classification.upper()
                   or "real" in classification.lower())
        estimand = {
            "estimand_level_size_pp": safe.get("estimand_level_size_pp"),
            "estimand_level_size_omega_20p3_21p6_pp":
                safe.get("estimand_level_size_omega_20p3_21p6_pp"),
            "estimand_level_status": (
                safe.get("estimand_level_status")
                or ("PROPAGATED -- the result of record is PRIVATE (it is "
                    "evaluated against the real pooled posterior and carries "
                    "real-data estimand values); this release product records "
                    "its existence and digest only. Sizes: see the notes-repo "
                    + S6_RESULT_BASENAME + " and the private companion "
                    "SYSTEMATICS_TABLE_REAL_HW.{json,md}"
                    if private else "PROPAGATED")),
            "result_file_expected": S6_RESULT_BASENAME,
            "result_file_found": os.path.basename(found),
            "result_file_sha256": _sha256(found),
            "result_classification": classification,
            "result_is_private": bool(private),
            "result_method": (payload.get("method") or {}).get("fit"),
            "frozen_objects_unchanged":
                (payload.get("method") or {}).get("frozen_objects_unchanged"),
            "beta_updated_with_real_data":
                (payload.get("method") or {}).get("beta_updated_with_real_data"),
            "release_safe_block_present": bool(safe),
            "authority": "PI 2026-09-14b sec.9 (propagation APPROVED, no refit)",
        }
    return {
        "id": "S6",
        "name": "completeness calibration covariance (category 3)",
        "ruling": "PI 2026-09-14 sec.10, sec.11, sec.18(6)",
        "treatment": "category-3 calibration uncertainty. The coefficient "
                     "covariance (Fisher + 200-draw bootstrap + half-split) is "
                     "preserved and released for propagation; it was not "
                     "propagated through the final mock ladder and was propagated "
                     "afterwards to the estimands under PI ruling 2026-09-14b sec.9 "
                     "(see the PROPAGATED status below; release-safe sizes in "
                     "S6_RELEASE_SAFE_SUMMARY.md)",
        "summary": dict({
            "covariance_source": os.path.basename(path),
            "covariance_keys": ["cov_fisher", "cov_bootstrap",
                                "cov_halfsplit", "beta_bootstrap"],
        }, **estimand),
        "rows": rows,
    }


def _sys7_subfloor(a0_base, a0_p6bcal):
    rows = []
    for fam in FAMILIES:
        seeds = [s for s, _r, _p in a0_p6bcal[fam]]
        base = [(s, r, pth) for s, r, pth in a0_base[fam] if s in seeds]
        for t in THRESHOLDS:
            tp = _mean([bias_pct(r, t) for _s, r, _p in a0_p6bcal[fam]])
            tr = _mean([bias_pct(r, t) for _s, r, _p in base])
            rows.append({"family": fam, "threshold": t,
                         "truth_pinned_subfloor_bias_pct": tr,
                         "transported_subfloor_bias_pct": tp,
                         "shift_pp": tp - tr, "seeds_compared": seeds})
        otp = _mean([omega_bias_pct(pth) for _s, _r, pth in a0_p6bcal[fam]])
        otr = _mean([omega_bias_pct(pth) for _s, _r, pth in base])
        rows.append({"family": fam, "threshold": "omega_20p3_21p6",
                     "truth_pinned_subfloor_bias_pct": otr,
                     "transported_subfloor_bias_pct": otp,
                     "shift_pp": otp - otr, "seeds_compared": seeds})
    return {
        "id": "S7",
        "name": "sub-floor transport sensitivity",
        "ruling": "PI 2026-09-14 sec.1, sec.18(7)",
        "treatment": "the sub-floor-host term is a FIXED TRANSPORTED "
                     "calibration term for survey use; truth-pinned was the "
                     "mock-certification convention only",
        "summary": {"max_abs_shift_pp": max(
                        abs(r["shift_pp"]) for r in rows
                        if r["threshold"] in THRESHOLDS),
                    "max_abs_shift_omega_20p3_21p6_pp": max(
                        abs(r["shift_pp"]) for r in rows
                        if r["threshold"] == "omega_20p3_21p6"),
                    "measured_on": "first ladder, A0-P6bcal vs A0 "
                                   "(the arm that isolates the sub-floor term)"},
        "rows": rows,
    }


def _headline_rank_rhat_ess(run_path):
    """Rank-normalised split-R-hat and bulk/tail ESS of the two headlines.

    The runner stores the PLAIN split-R-hat; the sealed J = 8 rule asks for
    rank-R-hat (see the J = 8 certification's "implemented as" note).  The
    headline series per chain is recovered from the stored ``_fdraws.npz``:
    the draw axis is chain-major (verified against the runner's own
    ``perchain_median``), so ``reshape(chains, -1)`` is the chain view.  Pure
    read-out of stored draws; no sampler is run.
    """
    repo = os.path.dirname(os.path.dirname(_HERE))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from validation.fp_ladder.j8_certify import rank_rhat_ess    # noqa: E402
    with np.load(_fdraws_of(run_path)) as z:
        f = np.asarray(z["f"], float)
        e = np.asarray(z["ntrue_edges"], float)
        dX = np.asarray(z["dX_k"], float)
    run = json.load(open(run_path))
    nch = int(run.get("chains") or run["run_config"]["chains"])
    lo, hi = e[:-1], e[1:]
    out = {}
    for thr, nmin in (("ge20.0", 20.0), ("ge20.3", 20.3)):
        w = np.clip(hi - np.maximum(lo, nmin), 0.0, None)
        sel = w > 0
        v = (f[:, sel, :] * w[sel][None, :, None]).sum(1)
        v = (v * dX[None, :]).sum(1) / dX.sum()
        x = v.reshape(nch, -1)
        per_chain = np.median(x, axis=1)
        stored = ((run["diagnostics"].get("estimand_mixing") or {})
                  .get("dndx_dla_20p0_allz" if thr == "ge20.0"
                       else "dndx_dla_20p3_allz") or {})
        pcm = stored.get("perchain_median")
        if pcm is not None and not np.allclose(np.round(per_chain, 5),
                                               np.asarray(pcm, float),
                                               rtol=0.0, atol=1e-5):
            raise ManifestIntegrityError(
                "FAIL CLOSED: chain view of %s disagrees with the runner's "
                "per-chain medians" % os.path.basename(run_path))
        r, bulk, tail = rank_rhat_ess(x)
        out[thr] = {"rank_split_rhat": float(r), "ess_bulk": float(bulk),
                    "ess_tail": float(tail),
                    "split_rhat_runner": stored.get("split_rhat"),
                    "ess_runner": stored.get("ess")}
    return out


def _tk_rank_rhat_ess(run_path):
    """Rank-R-hat / bulk ESS of the coarse-z FP nuisance sites t_K."""
    repo = os.path.dirname(os.path.dirname(_HERE))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from validation.fp_ladder.j8_certify import rank_rhat_ess    # noqa: E402
    bc = run_path[:-len(".json")] + "_bychain.npz"
    if not os.path.isfile(bc):
        return None
    with np.load(bc) as z:
        if "t" not in z.files:
            return None
        t = np.asarray(z["t"], float)                 # (chains, draws, K)
    out = []
    for k in range(t.shape[2]):
        r, bulk, tail = rank_rhat_ess(t[:, :, k])
        out.append({"site": "t_K%d" % k, "rank_split_rhat": float(r),
                    "ess_bulk": float(bulk), "ess_tail": float(tail)})
    return out


def _real_sampler_disclosure(real_runs_dir):
    """Sampler DIAGNOSTICS ONLY of the real C1 runs -- never an estimand.

    Read straight out of the stored ``sampler_health`` block of each real run
    JSON (the runner already writes rank-R-hat and bulk/tail ESS there).  No
    posterior value, median, interval or science number is copied: this block
    carries divergences, E-BFMI, R-hat and ESS, which are the quantities PI
    2026-09-14b sec.11 and sec.18(8) require to be disclosed.
    """
    rows = []
    for pth in sorted(glob.glob(os.path.join(real_runs_dir, "RUN_*.json"))):
        j = json.load(open(pth))
        sh = j.get("sampler_health") or {}
        em = sh.get("estimand_mixing") or {}
        tm = sh.get("t_mixing_per_K") or []
        rows.append({
            "run": os.path.basename(pth),
            "seed": (j.get("run_config") or {}).get("seed"),
            "imputation_j": (j.get("lam_cut") or {}).get("j"),
            "divergences": sh.get("divergences"),
            "ebfmi_min": (min(sh["ebfmi_per_chain"])
                          if sh.get("ebfmi_per_chain") else None),
            "ebfmi_max": (max(sh["ebfmi_per_chain"])
                          if sh.get("ebfmi_per_chain") else None),
            "headline_rank_rhat": {k: v.get("rank_split_rhat")
                                   for k, v in em.items()},
            "headline_ess_bulk": {k: v.get("ess_bulk") for k, v in em.items()},
            "t_K_rank_rhat": [x.get("rank_split_rhat") for x in tm],
            "t_K_ess_bulk": [x.get("ess_bulk") for x in tm],
        })
    return rows


def _sys8_sampler(j8, b_m1cut_j1, real_runs_dir=None):
    """(8) sampler-geometry DISCLOSURE, built from the PRODUCTION runs.

    HISTORY: the released S8 block until 2026-09-15 was built from the J = 1
    (F2) arm alone and understated both tails (E-BFMI minimum 0.0812,
    divergences maximum 10).  The J = 1 rows are preserved below under
    ``arm = "M1CUT_J1_HISTORY"``; the production J = 8 rows are the arm of
    record.  When ``real_runs_dir`` is given, the real survey runs'
    SAMPLER DIAGNOSTICS (and nothing else) are appended.
    """
    rows = []
    for fam, recs in j8.items():
        for seed, jj, run, pth in recs:
            d = run["diagnostics"]
            mix = d.get("estimand_mixing", {})
            rk = _headline_rank_rhat_ess(pth)
            tk = _tk_rank_rhat_ess(pth)
            rows.append({
                "arm": "M1CUT_J8_production", "family": fam, "seed": seed,
                "imputation_j": jj,
                "divergences": int(run["divergences"]),
                "ebfmi_min": float(min(d["ebfmi_per_chain"])),
                "ebfmi_max": float(max(d["ebfmi_per_chain"])),
                "split_rhat_ge20.0": mix.get("dndx_dla_20p0_allz", {}).get("split_rhat"),
                "split_rhat_ge20.3": mix.get("dndx_dla_20p3_allz", {}).get("split_rhat"),
                "ess_ge20.0": mix.get("dndx_dla_20p0_allz", {}).get("ess"),
                "ess_ge20.3": mix.get("dndx_dla_20p3_allz", {}).get("ess"),
                "rank_rhat_ge20.0": rk["ge20.0"]["rank_split_rhat"],
                "rank_rhat_ge20.3": rk["ge20.3"]["rank_split_rhat"],
                "ess_bulk_ge20.0": rk["ge20.0"]["ess_bulk"],
                "ess_bulk_ge20.3": rk["ge20.3"]["ess_bulk"],
                "ess_tail_ge20.0": rk["ge20.0"]["ess_tail"],
                "ess_tail_ge20.3": rk["ge20.3"]["ess_tail"],
                "t_K_rank_rhat": [x["rank_split_rhat"] for x in (tk or [])],
                "t_K_ess_bulk": [x["ess_bulk"] for x in (tk or [])],
            })
    for fam, runs in b_m1cut_j1.items():
        for seed, run, _pth in runs:
            d = run["diagnostics"]
            mix = d.get("estimand_mixing", {})
            rows.append({
                "arm": "M1CUT_J1_HISTORY", "family": fam, "seed": seed,
                "divergences": int(run["divergences"]),
                "ebfmi_min": float(min(d["ebfmi_per_chain"])),
                "ebfmi_max": float(max(d["ebfmi_per_chain"])),
                "split_rhat_ge20.0": mix.get("dndx_dla_20p0_allz", {}).get("split_rhat"),
                "split_rhat_ge20.3": mix.get("dndx_dla_20p3_allz", {}).get("split_rhat"),
                "ess_ge20.0": mix.get("dndx_dla_20p0_allz", {}).get("ess"),
                "ess_ge20.3": mix.get("dndx_dla_20p3_allz", {}).get("ess"),
            })
    prod = [r for r in rows if r["arm"] == "M1CUT_J8_production"]
    hist = [r for r in rows if r["arm"] == "M1CUT_J1_HISTORY"]
    spread = []
    for fam, recs in j8.items():
        for t in THRESHOLDS:
            v = [bias_pct(r, t) for _s, _j, r, _p in recs]
            spread.append({"kind": "imputation_spread", "family": fam,
                           "threshold": t,
                           "imputation_spread_pp": float(max(v) - min(v)),
                           "n_imputations": len(v)})
    for fam, runs in b_m1cut_j1.items():
        for t in THRESHOLDS:
            roll = _seed_rollup(runs, lambda r, t=t: bias_pct(r, t))
            spread.append({"kind": "seed_spread_J1_history", "family": fam,
                           "threshold": t,
                           "seed_spread_pp": roll["seed_spread"],
                           "seeds": roll["seeds"]})
    summary = {
        "arm_of_record": "M1CUT_J8_production",
        "ebfmi_min_over_production_runs": min(r["ebfmi_min"] for r in prod),
        "ebfmi_max_over_production_runs": max(r["ebfmi_max"] for r in prod),
        "divergences_max_production": max(r["divergences"] for r in prod),
        "headline_rank_rhat_max_production": max(
            max(r["rank_rhat_ge20.0"], r["rank_rhat_ge20.3"]) for r in prod),
        "headline_ess_bulk_min_production": min(
            min(r["ess_bulk_ge20.0"], r["ess_bulk_ge20.3"]) for r in prod),
        "t_K_rank_rhat_max_production": max(
            max(r["t_K_rank_rhat"]) for r in prod if r["t_K_rank_rhat"]),
        "t_K_ess_bulk_min_production": min(
            min(r["t_K_ess_bulk"]) for r in prod if r["t_K_ess_bulk"]),
        "imputation_spread_max_pp": max(r["imputation_spread_pp"]
                                        for r in spread
                                        if r["kind"] == "imputation_spread"),
        "history_J1_ebfmi_min": min(r["ebfmi_min"] for r in hist),
        "history_J1_divergences_max": max(r["divergences"] for r in hist),
        "history_note":
            "the released S8 block until 2026-09-15 quoted the J = 1 (F2) arm "
            "alone (E-BFMI min 0.0812, divergences max 10); the production "
            "J = 8 runs reach the values above. The J = 1 rows are preserved, "
            "not rewritten.",
    }
    if real_runs_dir:
        real = _real_sampler_disclosure(real_runs_dir)
        rows += [dict(r, arm="REAL_C1_sampler_diagnostics_only") for r in real]
        eb = [r["ebfmi_min"] for r in real if r["ebfmi_min"] is not None]
        tk = [x for r in real for x in (r["t_K_rank_rhat"] or []) if x]
        tke = [x for r in real for x in (r["t_K_ess_bulk"] or []) if x]
        hr = [v for r in real for v in r["headline_rank_rhat"].values() if v]
        he = [v for r in real for v in r["headline_ess_bulk"].values() if v]
        summary["real_survey_disclosure"] = {
            "note": "SAMPLER DIAGNOSTICS ONLY -- no real estimand value "
                    "appears in this table",
            "n_runs": len(real),
            "ebfmi_min": min(eb) if eb else None,
            "ebfmi_max": max(r["ebfmi_max"] for r in real
                             if r["ebfmi_max"] is not None),
            "divergences_max": max(r["divergences"] for r in real
                                   if r["divergences"] is not None),
            "headline_rank_rhat_max": max(hr) if hr else None,
            "headline_ess_bulk_min": min(he) if he else None,
            "t_K_rank_rhat_max": max(tk) if tk else None,
            "t_K_ess_bulk_min": min(tke) if tke else None,
        }
    return {
        "id": "S8",
        "name": "sampler-geometry disclosure",
        "ruling": "PI 2026-09-14 sec.15, sec.18(8); PI 2026-09-14b sec.11 "
                  "(t_K mixing disclosed with the t_K location)",
        "treatment": "DISCLOSURE, not a bias budget. Low E-BFMI alone is not a "
                     "reason to reopen the model; the J = 8 certification asks "
                     "whether different imputations give the same science "
                     "marginal. The t_K nuisance sites are NOT converged in "
                     "part of the runs and must never be quoted as a measured "
                     "FP transfer",
        "summary": summary,
        "rows": rows + spread,
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
def build(products, out_root, real_runs_dir=None, s6_path=None,
          real_pooled=None, private_out=None, s6_search=True):
    """Build the release systematics table (and, optionally, a PRIVATE copy).

    ``real_runs_dir``  -- real C1 run JSONs; only their SAMPLER DIAGNOSTICS
                          (divergences, E-BFMI, R-hat, ESS) enter S8.
    ``s6_path``        -- explicit path to the S6 propagation result.
    ``real_pooled``    -- ``real_c1/REAL_C1_POOLED.json``.  Its 68 % half-widths
                          are REAL values, so they NEVER enter the release
                          table: they are used only for the private companion
                          written to ``private_out``.
    """
    out = os.path.join(out_root, "systematics")
    os.makedirs(out, exist_ok=True)

    arms = {k: _load_arm(products, k) for k in ARMS if k != "B_m1cut_J8"}
    j8 = _load_j8(products)
    battery = _load_a0_battery(products)

    systematics = [
        _sys1_zbin(j8, arms["B_m1cut"], arms["B_oracle"]),
        _sys2_response_form(arms["B_oracle"], arms["E_oracle"],
                            arms["B_m1cut"], arms["E_m1cut"], arms["REF"]),
        _sys3_fp_scale(arms["B_oracle"], arms["B_m1cut"], j8),
        _sys4_a0(battery, arms["B_m1cut"], products),
        _sys5_phi(arms["B_oracle"], arms["B_phismooth"], arms["B_phifamily"]),
        _sys6_completeness(products, s6_path, s6_search),
        _sys7_subfloor(arms["A0_base"], arms["A0_P6bcal"]),
        _sys8_sampler(j8, arms["B_m1cut"], real_runs_dir),
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
                          "in per cent of the median",
                  "hw68_reference": "MOCK. Every `in_hw68_*` ratio in THIS "
                                    "file is in units of the MOCK posterior "
                                    "half-width. The real survey's pooled "
                                    "half-widths are not reproduced in this "
                                    "release product; the private companion "
                                    "SYSTEMATICS_TABLE_REAL_HW.{json,md} "
                                    "carries the same sizes in real-half-width "
                                    "units."},
        "estimands": {"ge20.0": "dN/dX(N_HI >= 20.0), all z (secondary)",
                      "ge20.3": "dN/dX(N_HI >= 20.3), all z (primary)",
                      "omega_20p3_21p6":
                          "Omega_HI[20.3, 21.6], all z -- read back from the "
                          "stored f draws with the committed "
                          "ladder_table.paper_omega_20p3_21p6 (the run JSONs' "
                          "thresholds.omega_allz is the SUB-DLA window "
                          "[19.5, 20.3) and is NOT this quantity)"},
        "arm_of_record": "B + phi_2LPT + C1nsadd + M1CUT, J = 8 production "
                         "(3 families x 8 Lambda imputations, seed 20260811)",
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
    if real_pooled and private_out:
        _write_private_real_hw(private_out, doc, real_pooled,
                               _load_s6_result(products, s6_path, s6_search))
    return out, doc


# --------------------------------------------------------------------------
# PRIVATE companion: the same sizes in REAL pooled half-width units.
# The real half-widths are REAL-DATA values; they must never appear in the
# release tree, in stdout or in a commit message (real-data privacy rule).
# --------------------------------------------------------------------------
def _real_halfwidths(real_pooled_path):
    with open(real_pooled_path) as fh:
        pool = json.load(fh)
    allp = pool["pools"]["all"]
    out = {}
    for key, tag in (("ge20.0", "ge20.0"), ("ge20.3", "ge20.3")):
        q = allp["thresholds_allz"][key]
        p16, p50, p84 = q["post_p16_50_84"]
        out[tag] = 50.0 * (p84 - p16) / p50            # half-width in % of median
    q = allp["omega_20p3_21p6_allz"]["post_p16_50_84"]
    out["omega_20p3_21p6"] = 50.0 * (q[2] - q[0]) / q[1]
    return out


def _sizes_in_pp(doc):
    """``[(id, label, family, estimand, signed size in pp), ...]``."""
    rows = []
    for s in doc["systematics"]:
        for r in s["rows"]:
            thr = r.get("threshold")
            if thr not in ("ge20.0", "ge20.3", "omega_20p3_21p6"):
                continue
            for k in ("signed_E_minus_B_pp", "shift_M1CUT_minus_ORACLE_pp",
                      "shift_pp"):
                if k in r and r[k] is not None:
                    rows.append((s["id"], r.get("arm") or r.get("variant")
                                 or r.get("fp_configuration") or "",
                                 r.get("family", ""), thr, float(r[k]), k))
    for r in _sysid(doc, "S4")["rows"]:
        if r.get("is_record"):
            continue
        for thr, k in (("ge20.0", "delta_vs_record_ge20.0_pp"),
                       ("ge20.3", "delta_vs_record_ge20.3_pp"),
                       ("omega_20p3_21p6",
                        "delta_vs_record_omega_20p3_21p6_pp")):
            if r.get(k) is not None:
                rows.append(("S4", "a0=%.7g" % r["a0"], r["family"], thr,
                             float(r[k]), k))
    return rows


def _sysid(doc, sid):
    for s in doc["systematics"]:
        if s["id"] == sid:
            return s
    raise KeyError(sid)


def _write_private_real_hw(private_out, doc, real_pooled_path, s6=(None, None)):
    """PRIVATE notes product: the named sizes in REAL pooled half-width units."""
    hw = _real_halfwidths(real_pooled_path)
    s6_path, s6_payload = s6
    os.makedirs(private_out, exist_ok=True)
    rows = []
    for sid, label, fam, thr, pp, key in _sizes_in_pp(doc):
        h = hw.get(thr)
        rows.append({"systematic": sid, "arm": label, "family": fam,
                     "estimand": thr, "size_pp": pp, "quantity": key,
                     "size_in_real_hw68": (pp / h) if h else None})
    # S1: the signed per-bin residual of the arm of record
    for r in _sysid(doc, "S1")["rows"]:
        if r["arm"] != "model_of_record_M1CUT_J8_production":
            continue
        h = hw.get(r["threshold"])
        rows.append({"systematic": "S1", "arm": r["arm"] + "/" + r["bin"],
                     "family": r["family"], "estimand": r["threshold"],
                     "size_pp": r["bias_pct"], "quantity": "bias_pct",
                     "size_in_real_hw68": (r["bias_pct"] / h) if h else None})
    # S6: the propagated calibration-uncertainty sizes live ONLY here.
    s6_rows = []
    if s6_payload:
        for name, blk in sorted((s6_payload.get("estimands") or {}).items()):
            q = blk.get("quoted_S6") or {}
            if not isinstance(q, dict):
                continue
            pct = q.get("pct_of_record_median")
            in_hw = q.get("in_hw68")
            if pct is None and in_hw is None:
                continue
            s6_rows.append({
                "systematic": "S6", "arm": "completeness_covariance",
                "family": "real (calibration covariance)", "estimand": name,
                "size_pp": (float(pct) if pct is not None else None),
                "quantity": "quoted_S6 as per cent of the record median (%s)"
                            % q.get("source", "linearised"),
                "size_in_real_hw68": (float(in_hw) if in_hw is not None
                                      else None),
                "nonlinearity_triggered":
                    (blk.get("nonlinearity") or {}).get("triggered")})
    rows += s6_rows

    payload = {
        "schema": "zenodo_release/systematics_private_real_hw/v1",
        "PRIVACY": "CONTAINS REAL-DATA DERIVED QUANTITIES (the real pooled "
                   "68 % half-widths and every ratio to them). NOTES REPO "
                   "ONLY -- never in the release tree, stdout or a commit "
                   "message.",
        "generated_utc": doc["generated_utc"],
        "source_release_table": "systematics/SYSTEMATICS_TABLE.json",
        "real_pooled_source": real_pooled_path,
        "real_pooled_hw68_pct_of_median": hw,
        "s6_result_source": s6_path,
        "s6_result_sha256": _sha256(s6_path),
        "s6_note": ("S6 sizes are per cent of the real pooled median and, in "
                    "the last column, in units of the real pooled 68 % "
                    "half-width. They appear ONLY in this private product; the "
                    "release table records the propagation's existence and "
                    "digest, never its numbers."),
        "rows": rows,
    }
    jpath = os.path.join(private_out, "SYSTEMATICS_TABLE_REAL_HW.json")
    with open(jpath, "w") as fh:
        json.dump(payload, fh, indent=1)
        fh.write("\n")
    L = ["# PRIVATE — named systematics in REAL pooled half-width units",
         "",
         "**PRIVATE (notes repo only).** " + payload["PRIVACY"], "",
         "Real pooled 68 % half-widths (per cent of the median): "
         + ", ".join("%s = %.3f %%" % (k, v) for k, v in sorted(hw.items())),
         "",
         "| systematic | arm | family | estimand | size (pp) | size (real hw68) |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| %s | %s | %s | %s | %s | %s |" % (
            r["systematic"], r["arm"], r["family"], r["estimand"],
            "n/a" if r["size_pp"] is None else "%+.3f" % r["size_pp"],
            "%+.2f" % r["size_in_real_hw68"]
            if r["size_in_real_hw68"] is not None else "n/a"))
    with open(os.path.join(private_out,
                           "SYSTEMATICS_TABLE_REAL_HW.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    return jpath


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
            b5 = s["summary"]["B5_seed_spread_pp_J1_two_seed_arm"]
            L += ["**Arm of record: `%s`** — the PRODUCTION J = 8 runs "
                  "(3 families x 8 Lambda imputations), equal-weight over the "
                  "imputations, seed %s. Signed per-bin median bias %%; "
                  "\\* = truth outside the 68 %% interval in EVERY imputation. "
                  "This table is the systematic; it is never collapsed to one "
                  "number." % (s["arm_of_record"],
                               s["summary"]["arm_of_record_seeds"]), "",
                  "**History.** %s" % s["history_note"], "",
                  "**Seed noise.** The J = 8 campaign ran a single seed, so the "
                  "spread column above is an IMPUTATION spread (max %.3f pp "
                  "over every family, threshold and bin). Seed noise is "
                  "measured on the two-seed J = 1 M1CUT arm and is NOT "
                  "uniformly small: in **B5** it is "
                  % s["summary"]["max_imputation_spread_pp"]
                  + ", ".join("%s %.2f pp" % (k.replace("_", " "), v)
                              for k, v in sorted(b5.items()))
                  + ". The earlier caption's \"seed noise <= 0.05 pp\" held "
                    "for the all-z headlines, never for B5.", "",
                  "**B5 coverage.** B5 is nominally [3.40, 3.80) but the pack's "
                  "absorber-z grid ends at 3.50, so its stored `coverage` is "
                  "0.25 and it is effectively a [3.40, 3.50) measurement; the "
                  "`nominal_coverage` column carries the stored value for "
                  "every bin.", ""]
            for arm in ("model_of_record_M1CUT_J8_production",
                        "M1CUT_J1_HISTORY", "ORACLE_FP_diagnostic_F1"):
                bins = _bins_of(s, arm)
                if not bins:
                    continue
                L += ["### %s" % arm, "",
                      "| family | estimand | " + " | ".join(bins) + " |",
                      "|---" * (2 + len(bins)) + "|"]
                for fam in doc["families"]:
                    for t in list(doc["thresholds"]):
                        cells = [r for r in s["rows"]
                                 if r["arm"] == arm and r["family"] == fam
                                 and r["threshold"] == t]
                        cells.sort(key=lambda r: r["bin"])
                        if not cells:
                            continue
                        L.append("| %s | %s | " % (fam, t) + " | ".join(
                            "%s%s" % (_fmt(c["bias_pct"]),
                                      "" if c.get("truth_in_68_all_imputations",
                                                  c.get("truth_in_68_all_seeds"))
                                      else "*")
                            for c in cells) + " |")
                    # the Omega row for the same family / arm
                    cells = [r for r in s["rows"]
                             if r["arm"] == arm and r["family"] == fam
                             and r["threshold"] == doc["thresholds"][-1]
                             and "omega_20p3_21p6_bias_pct" in r]
                    cells.sort(key=lambda r: r["bin"])
                    if cells:
                        L.append("| %s | Omega[20.3,21.6] | " % fam + " | ".join(
                            "%s%s" % (_fmt(c["omega_20p3_21p6_bias_pct"]),
                                      "" if c.get(
                                          "omega_20p3_21p6_truth_in_68_all_"
                                          "imputations", True) else "*")
                            for c in cells) + " |")
                L.append("")
            L += ["Coverage of each bin (stored `coverage` field): "
                  + ", ".join("%s = %s" % (b, _fmt(
                      next((r["nominal_coverage"] for r in s["rows"]
                            if r["bin"] == b), None), 2))
                      for b in _bins_of(s, "model_of_record_M1CUT_J8_production")),
                  "",
                  "Per-bin J8 - J1 relabelling difference (max |.| = %s pp)."
                  % _fmt(s["summary"]["J8_minus_J1_pp_max_abs"]), ""]
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
            L += ["| arm | family | estimand | ORACLE (truth-pinned FP) % | "
                  "M1CUT (rule-compliant FP) % | shift (pp) |",
                  "|---|---|---|---|---|---|"]
            for r in s["rows"]:
                if "shift_M1CUT_minus_ORACLE_pp" not in r:
                    continue
                L.append("| %s | %s | %s | %s | %s | %s |" % (
                    r.get("arm", ""), r["family"], r["threshold"],
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
                  "Δ (pp) | Ω[20.3,21.6] % | Δ (pp) | FP counts ≥20.3 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for r in sorted(s["rows"], key=lambda r: (r["family"], r["a0"])):
                L.append("| %s | %.7g%s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    r["family"], r["a0"], " (record)" if r["is_record"] else "",
                    "yes" if r["in_factor4_bracket"] else "OUTER",
                    _fmt(r["bias_ge20.0_pct"]),
                    _fmt(r["delta_vs_record_ge20.0_pp"]),
                    _fmt(r["bias_ge20.3_pct"]),
                    _fmt(r["delta_vs_record_ge20.3_pp"]),
                    _fmt(r["bias_omega_20p3_21p6_pct"]),
                    _fmt(r["delta_vs_record_omega_20p3_21p6_pp"]),
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
            L += ["Arm of record = the PRODUCTION J = 8 runs. History: %s"
                  % s["summary"]["history_note"], "",
                  "| arm | family | seed | j | div | E-BFMI min | E-BFMI max | "
                  "rank-R̂ ≥20.0 | rank-R̂ ≥20.3 | ESS_bulk ≥20.0 | "
                  "ESS_bulk ≥20.3 | t_K rank-R̂ | t_K ESS_bulk |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in s["rows"]:
                if r.get("kind") or r["arm"].startswith("REAL_"):
                    continue
                na = lambda v, nd=4: ("n/a" if v is None else _fmt(v, nd))
                L.append("| %s | %s | %s | %s | %s | %.3f | %.2f | %s | %s | "
                         "%s | %s | %s | %s |" % (
                             r["arm"], r["family"], r.get("seed", ""),
                             r.get("imputation_j", "n/a"), r["divergences"],
                             r["ebfmi_min"], r["ebfmi_max"],
                             na(r.get("rank_rhat_ge20.0")),
                             na(r.get("rank_rhat_ge20.3")),
                             na(r.get("ess_bulk_ge20.0"), 0),
                             na(r.get("ess_bulk_ge20.3"), 0),
                             ([round(x, 3) for x in r["t_K_rank_rhat"]]
                              if r.get("t_K_rank_rhat") else "n/a"),
                             ([round(x) for x in r["t_K_ess_bulk"]]
                              if r.get("t_K_ess_bulk") else "n/a")))
            L += ["", "Imputation spread of the headline bias (max over family "
                  "× threshold): %.3f pp. J = 1 history: E-BFMI min %.4f, "
                  "divergences max %d."
                  % (s["summary"]["imputation_spread_max_pp"],
                     s["summary"]["history_J1_ebfmi_min"],
                     s["summary"]["history_J1_divergences_max"]), ""]
            rs = s["summary"].get("real_survey_disclosure")
            if rs:
                L += ["**Real survey (sampler diagnostics only; no real "
                      "estimand value appears in this release product).** "
                      "%d runs: E-BFMI %.3f–%.3f, divergences max %s, headline "
                      "rank-R̂ max %.4f with bulk ESS min %.0f; the coarse-z "
                      "FP nuisance sites t_K reach rank-R̂ %.2f with bulk ESS "
                      "%.0f — NOT converged, disclosed, never quoted as a "
                      "measured FP transfer (PI 2026-09-14b §11)."
                      % (rs["n_runs"], rs["ebfmi_min"], rs["ebfmi_max"],
                         rs["divergences_max"], rs["headline_rank_rhat_max"],
                         rs["headline_ess_bulk_min"], rs["t_K_rank_rhat_max"],
                         rs["t_K_ess_bulk_min"]), ""]
        L += ["Summary: `%s`" % json.dumps(s["summary"]), ""]
    with open(path, "w") as fh:
        fh.write("\n".join(L))
    return path


def _bins_of(s, arm):
    seen = []
    for r in s["rows"]:
        if r.get("arm") == arm and r.get("bin") and r["bin"] not in seen:
            seen.append(r["bin"])
    return sorted(seen)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--products", required=True)
    ap.add_argument("--out", required=True, help="the release root")
    ap.add_argument("--no-sums", action="store_true")
    ap.add_argument("--real-runs-dir", default=None,
                    help="real C1 run JSONs -- SAMPLER DIAGNOSTICS ONLY (S8)")
    ap.add_argument("--s6-result", default=None,
                    help="path to %s" % S6_RESULT_BASENAME)
    ap.add_argument("--no-s6-search", action="store_true",
                    help="do not look for %s in the default locations"
                         % S6_RESULT_BASENAME)
    ap.add_argument("--real-pooled", default=None,
                    help="real_c1/REAL_C1_POOLED.json -- used ONLY for the "
                         "private companion (--private-out)")
    ap.add_argument("--private-out", default=None,
                    help="directory for the PRIVATE real-half-width companion "
                         "(notes repo only; never the release tree)")
    a = ap.parse_args(argv)
    if bool(a.real_pooled) != bool(a.private_out):
        ap.error("--real-pooled and --private-out must be given together: the "
                 "real half-widths may only be written to the private "
                 "directory")
    if a.private_out and os.path.abspath(a.private_out).startswith(
            os.path.abspath(a.out) + os.sep):
        ap.error("--private-out must NOT live inside the release tree")
    out, doc = build(a.products, a.out, real_runs_dir=a.real_runs_dir,
                     s6_path=a.s6_result, real_pooled=a.real_pooled,
                     private_out=a.private_out,
                     s6_search=not a.no_s6_search)
    if not a.no_sums:
        write_sha256sums(out)
    print("systematics table ->", out)
    for s in doc["systematics"]:
        print("  %s %-52s %d rows" % (s["id"], s["name"], len(s["rows"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
