#!/usr/bin/env python
"""b5_presentation.py -- everything a Paper-1 caption needs for bin B5.

PI ruling 2026-09-14b sec.6 and sec.18: B5 stays in the main redshift figure
with its NOMINAL bin 3.40 < z < 3.80 stated, its ACTUAL support 3.40 < z < 3.50
stated, its nominal coverage (25 %) stated, and its uncertainty built from the
correct statistical posterior plus the FINAL J = 8 transfer systematic -- not
the J = 1 S1 table.

PRIVACY.  The real pooled statistical interval is a REAL-DATA value.  This
builder therefore writes to a PRIVATE directory (the notes repository) and
never into the Zenodo release tree; it refuses an output path inside the
release root.  Nothing here is fitted: every number is a deterministic read-out
of stored run JSONs and of the freshly built systematics table.

    python -m validation.release.b5_presentation \\
        --products /scratch/.../absorber_ladder_2026-09-13 \\
        --systematics /scratch/.../release/systematics/SYSTEMATICS_TABLE.json \\
        --real-pooled /scratch/.../real_c1/REAL_C1_POOLED.json \\
        --out /home/mfho/desi_gpy_dla_notes/governance/final_campaign_2026-09-13
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from hashutil import ManifestIntegrityError                      # noqa: E402

BIN = "B5"
FAMILIES = ("2lpt0", "london0", "saclay0")
THRESHOLDS = ("ge20.0", "ge20.3")
NOMINAL = (3.40, 3.80)


def _sysid(doc, sid):
    for s in doc["systematics"]:
        if s["id"] == sid:
            return s
    raise ManifestIntegrityError("FAIL CLOSED: systematic %r missing" % sid)


def _rows(s, arm, threshold):
    return {r["family"]: r for r in s["rows"]
            if r.get("arm") == arm and r.get("bin") == BIN
            and r.get("threshold") == threshold}


def _support_from_pack(products, family="2lpt0"):
    """The fine-z cells with dX > 0 that actually fall inside the nominal bin."""
    pack = os.path.join(products, "support_v3",
                        "scanpack_%s_b300_v3.npz" % family)
    if not os.path.isfile(pack):
        raise ManifestIntegrityError("FAIL CLOSED: pack missing: %s" % pack)
    with np.load(pack, allow_pickle=True) as z:
        zf = np.asarray(z["zf_edges"], float)
        dX = np.asarray(z["dX"], float).sum(axis=1)
    lo, hi = NOMINAL
    cells = [(float(zf[k]), float(zf[k + 1]), float(dX[k]))
             for k in range(len(dX))
             if dX[k] > 0 and zf[k] < hi and zf[k + 1] > lo]
    return cells, float(zf[0]), float(zf[-1])


def _ab_difference(products):
    """B vs E in B5 on the ORACLE F1 arm (signed E - B), per family/threshold."""
    out = {}
    for tag, sub in (("B", "B-phi2lpt-C1nsadd"), ("E", "E-phi2lpt-C1nsadd")):
        for fam in FAMILIES:
            pat = os.path.join(products, "final", "runs", sub,
                               "RUN_%s_%s_s*.json" % (sub, fam))
            paths = sorted(glob.glob(pat))
            if not paths:
                raise ManifestIntegrityError(
                    "FAIL CLOSED: no F1 runs for %s / %s" % (tag, fam))
            for t in THRESHOLDS:
                vals = []
                for p in paths:
                    j = json.load(open(p))
                    for c in j["perz_recovery"]["estimand"][t]["paper1_bins"]:
                        if c["bin"] == BIN and c.get("available", True):
                            vals.append(float(c["median_bias_pct"]))
                out[(tag, fam, t)] = (float(np.mean(vals)), len(vals))
    diff = {}
    for fam in FAMILIES:
        for t in THRESHOLDS:
            b, nb = out[("B", fam, t)]
            e, ne = out[("E", fam, t)]
            diff[(fam, t)] = {"B_bias_pct": b, "E_bias_pct": e,
                              "signed_E_minus_B_pp": e - b,
                              "n_seeds": min(nb, ne)}
    return diff


def _real_b5(real_pooled_path):
    with open(real_pooled_path) as fh:
        pool = json.load(fh)
    est = pool["pools"]["all"]["perz_posterior"]["estimand"]
    out = {}
    for t in THRESHOLDS:
        cell = [c for c in est[t]["paper1_bins"] if c["bin"] == BIN]
        if not cell:
            raise ManifestIntegrityError("FAIL CLOSED: no real B5 cell for " + t)
        c = cell[0]
        q = c["post_p2p5_16_50_84_97p5"]
        out[t] = {"z_nominal": c["z"], "coverage": c.get("coverage"),
                  "dX": c.get("dX"), "p2p5": q[0], "p16": q[1], "median": q[2],
                  "p84": q[3], "p97p5": q[4],
                  "hw68_pct_of_median": 50.0 * (q[3] - q[1]) / q[2]}
    # the same for every other bin, so a caption can say "3-5x the other bins"
    ratio = {}
    for t in THRESHOLDS:
        hws = {}
        for c in est[t]["paper1_bins"]:
            if not c.get("available"):
                continue
            q = c["post_p2p5_16_50_84_97p5"]
            hws[c["bin"]] = 50.0 * (q[3] - q[1]) / q[2]
        other = [v for k, v in hws.items() if k != BIN]
        ratio[t] = {"b5_hw68_pct": hws[BIN],
                    "other_bins_hw68_pct": hws,
                    "b5_over_median_other": hws[BIN] / float(np.median(other))}
    return out, ratio, pool


def build(products, systematics_json, real_pooled, out_dir, release_root=None):
    if release_root and os.path.abspath(out_dir).startswith(
            os.path.abspath(release_root) + os.sep):
        raise ManifestIntegrityError(
            "FAIL CLOSED: B5 presentation table carries REAL values and must "
            "not be written inside the release tree")
    with open(systematics_json) as fh:
        doc = json.load(fh)
    s1 = _sysid(doc, "S1")
    j8 = {t: _rows(s1, "model_of_record_M1CUT_J8_production", t)
          for t in THRESHOLDS}
    j1 = {t: _rows(s1, "M1CUT_J1_HISTORY", t) for t in THRESHOLDS}
    orc = {t: _rows(s1, "ORACLE_FP_diagnostic_F1", t) for t in THRESHOLDS}
    cells, zmin, zmax = _support_from_pack(products)
    ab = _ab_difference(products)
    real, ratio, pool = _real_b5(real_pooled)

    payload = {
        "schema": "paper1/b5_presentation/v1",
        "PRIVACY": "CONTAINS REAL-DATA VALUES (the real pooled B5 posterior). "
                   "NOTES REPO ONLY -- never the release tree, stdout or a "
                   "commit message.",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "authority": "PI ruling 2026-09-14b sec.6, sec.18",
        "bin": BIN,
        "nominal_z": list(NOMINAL),
        "actual_support_z": [cells[0][0], cells[-1][1]] if cells else None,
        "fine_z_cells_with_dX_gt_0": [{"z": [a, b], "dX": d}
                                      for a, b, d in cells],
        "nominal_coverage": (cells[-1][1] - cells[0][0]) / (NOMINAL[1] - NOMINAL[0])
                            if cells else None,
        "pack_z_grid": [zmin, zmax],
        "real_statistical": real,
        "real_halfwidth_context": ratio,
        "transfer_systematic_J8_production": {
            t: {fam: {"bias_pct": r["bias_pct"],
                      "imputation_spread_pp": r.get("imputation_spread_pp"),
                      "truth_in_68_all_imputations":
                          r.get("truth_in_68_all_imputations")}
                for fam, r in j8[t].items()} for t in THRESHOLDS},
        "transfer_systematic_J1_history": {
            t: {fam: {"bias_pct": r["bias_pct"],
                      "seed_spread_pp": r.get("seed_spread_pp"),
                      "seeds": r.get("seeds")}
                for fam, r in j1[t].items()} for t in THRESHOLDS},
        "transfer_systematic_ORACLE_F1": {
            t: {fam: {"bias_pct": r["bias_pct"]} for fam, r in orc[t].items()}
            for t in THRESHOLDS},
        "B_vs_E_F1": {"%s/%s" % (fam, t): v for (fam, t), v in ab.items()},
        "omega_20p3_21p6_B5_J8_production": {
            fam: r.get("omega_20p3_21p6_bias_pct")
            for fam, r in j8["ge20.3"].items()},
        "sources": {"systematics_table": systematics_json,
                    "real_pooled": real_pooled,
                    "products": products,
                    "real_pool_n_runs": pool["pools"]["all"]["n_runs"]},
        "caption_rules": [
            "state the NOMINAL bin 3.40 < z < 3.80 and the ACTUAL support "
            "3.40 < z < 3.50 together with the nominal coverage 25 %",
            "use the PRODUCTION J = 8 transfer systematic, never the J = 1 "
            "S1 table (PI 2026-09-14b sec.6)",
            "never merge the signed transfer residual into the statistical "
            "bar (PI 2026-09-14b sec.7)",
            "the two-seed spread quoted below is measured on the J = 1 arm; "
            "the J = 8 campaign ran a single seed",
        ],
    }
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, "B5_PRESENTATION_TABLE.json")
    with open(jpath, "w") as fh:
        json.dump(payload, fh, indent=1)
        fh.write("\n")
    _write_md(os.path.join(out_dir, "B5_PRESENTATION_TABLE.md"), payload)
    return jpath, payload


def _f(v, nd=2):
    return "n/a" if v is None else ("%+.*f" % (nd, v))


def _write_md(path, p):
    real = p["real_statistical"]
    L = ["# B5 — final presentation table (PRIVATE)", "",
         "**PRIVATE (notes repo only).** " + p["PRIVACY"], "",
         "*Authority:* %s. Generated %s. Deterministic read-out of stored "
         "products; no fit, no sampler, no frozen object changed."
         % (p["authority"], p["generated_utc"]), "",
         "## 1. What B5 actually is", "",
         "| property | value |", "|---|---|",
         "| nominal bin | [%.2f, %.2f) |" % tuple(p["nominal_z"]),
         "| actual support (fine-z cells with dX > 0) | [%.2f, %.2f) |"
         % tuple(p["actual_support_z"]),
         "| nominal coverage | %.0f %% |" % (100 * p["nominal_coverage"]),
         "| fine-z cells inside the nominal bin | %d (%s) |"
         % (len(p["fine_z_cells_with_dX_gt_0"]),
            ", ".join("[%.2f, %.2f)" % tuple(c["z"])
                      for c in p["fine_z_cells_with_dX_gt_0"])),
         "| pack absorber-z grid | [%.2f, %.2f] |" % tuple(p["pack_z_grid"]),
         "",
         "B5 is therefore an **effective [3.40, 3.50) measurement carrying a "
         "[3.40, 3.80) label**. Every published table and caption must say so.",
         "",
         "## 2. Statistical posterior (REAL, pooled over %d runs)"
         % p["sources"]["real_pool_n_runs"], "",
         "| estimand | 2.5 % | 16 % | median | 84 % | 97.5 % | 68 % half-width "
         "(% of median) |", "|---|---|---|---|---|---|---|"]
    for t in ("ge20.0", "ge20.3"):
        r = real[t]
        L.append("| %s | %.5g | %.5g | %.5g | %.5g | %.5g | %.2f %% |"
                 % (t, r["p2p5"], r["p16"], r["median"], r["p84"], r["p97p5"],
                    r["hw68_pct_of_median"]))
    L += ["",
          "B5's statistical half-width is %.1f× (≥20.0) and %.1f× (≥20.3) the "
          "median of the other four Paper-1 bins — consistent with its dX."
          % (p["real_halfwidth_context"]["ge20.0"]["b5_over_median_other"],
             p["real_halfwidth_context"]["ge20.3"]["b5_over_median_other"]),
          "",
          "## 3. Transfer systematic in B5 — PRODUCTION J = 8 (the arm of "
          "record)", "",
          "Signed median bias % of the mock truth, equal-weight over the eight "
          "Λ imputations, seed 20260811; \\* = truth outside the 68 % interval "
          "in every imputation. **This is the number a caption uses** (PI "
          "2026-09-14b §6).", "",
          "| family | ≥20.0 | ≥20.3 | Ω[20.3,21.6] |", "|---|---|---|---|"]
    for fam in FAMILIES:
        a = p["transfer_systematic_J8_production"]["ge20.0"][fam]
        b = p["transfer_systematic_J8_production"]["ge20.3"][fam]
        L.append("| %s | %s%s (spread %s pp) | %s%s (spread %s pp) | %s |"
                 % (fam, _f(a["bias_pct"]),
                    "" if a["truth_in_68_all_imputations"] else "*",
                    _f(a["imputation_spread_pp"]),
                    _f(b["bias_pct"]),
                    "" if b["truth_in_68_all_imputations"] else "*",
                    _f(b["imputation_spread_pp"]),
                    _f(p["omega_20p3_21p6_B5_J8_production"][fam])))
    L += ["",
          "## 4. History and cross-checks (never the caption number)", "",
          "| quantity | ≥20.0 | ≥20.3 |", "|---|---|---|"]
    for fam in FAMILIES:
        h = p["transfer_systematic_J1_history"]
        L.append("| J = 1 (F2) arm, %s | %s | %s |"
                 % (fam, _f(h["ge20.0"][fam]["bias_pct"]),
                    _f(h["ge20.3"][fam]["bias_pct"])))
    for fam in FAMILIES:
        h = p["transfer_systematic_J1_history"]
        L.append("| **two-seed spread** (J = 1, seeds %s), %s | %s pp | %s pp |"
                 % (h["ge20.0"][fam].get("seeds"), fam,
                    _f(h["ge20.0"][fam]["seed_spread_pp"]),
                    _f(h["ge20.3"][fam]["seed_spread_pp"])))
    for fam in FAMILIES:
        o = p["transfer_systematic_ORACLE_F1"]
        L.append("| ORACLE (F1, truth-pinned FP), %s | %s | %s |"
                 % (fam, _f(o["ge20.0"][fam]["bias_pct"]),
                    _f(o["ge20.3"][fam]["bias_pct"])))
    L += ["", "| B − E in B5 (F1 ORACLE arm) | ≥20.0 signed E − B (pp) | "
          "≥20.3 signed E − B (pp) |", "|---|---|---|"]
    for fam in FAMILIES:
        a = p["B_vs_E_F1"]["%s/ge20.0" % fam]
        b = p["B_vs_E_F1"]["%s/ge20.3" % fam]
        L.append("| %s | %s | %s |" % (fam, _f(a["signed_E_minus_B_pp"]),
                                       _f(b["signed_E_minus_B_pp"])))
    L += ["", "## 5. Caption rules", ""]
    L += ["* " + r for r in p["caption_rules"]]
    L.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(L))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--products", required=True)
    ap.add_argument("--systematics", required=True)
    ap.add_argument("--real-pooled", required=True)
    ap.add_argument("--out", required=True, help="PRIVATE output directory")
    ap.add_argument("--release-root", default=None)
    a = ap.parse_args(argv)
    jpath, _p = build(a.products, a.systematics, a.real_pooled, a.out,
                      a.release_root)
    print("B5 presentation table ->", os.path.dirname(jpath))


if __name__ == "__main__":
    main()
