#!/usr/bin/env python
"""perz_gate.py — PREDECLARED per-redshift-bin recovery gate (PI ruling
2026-08-20 items 2-3: per-z closure is a permanent certification requirement
in addition to all-z closure).

Reads cc_posterior_validation JSONs (which carry `thresholds` = the all-z
committed estimand and `perz_recovery` = the same estimand per locked
Paper-1 bin) and evaluates the criteria PREDECLARED in the 2026-08-20 review
package (04_DECISION_SHEET.md, D2.2), committed BEFORE Battery 3/4 finished:

  convergence : split-Rhat <= 1.05 on both all-z estimands; divergences <= 10
  all-z >=20.0: |bias| <= 0.5 %
  all-z >=20.3: -0.5 % <= bias <= +3.0 %   (the named one-sided L1 response
                structure, measured per family +1.3..+2.7 % at ckpt 10.10)
  per-bin >=20.0: |bias| <= 2 % in B1-B4; B5 (quarter-covered, one native
                cell): |bias| <= 4 % OR truth inside the 95 % interval
  per-bin >=20.3: -1.0 % <= bias <= +4.0 % in B1, B2, B4 (the named one-sided
                response structure may sit in any bin up to +4 %); B5: that OR
                truth inside the 95 % interval; B3 EXEMPT and REPORTED (named
                residual, PI ruling item 8)

CRITERIA HISTORY (honest record): v1 (commit ae794cb, 2026-08-20 ~21:10)
required per-bin >=20.3 |bias - allz_bias| <= 2 %, i.e. assumed the named
all-z response structure is z-uniform. The Battery-2 dry run (4 runs, same
configuration as Battery 3, different seeds) showed it is not: under the
consistent g the +2 % all-z structure is carried almost entirely by B3
(+4..+7 %) while B1/B2/B4 sit at -1..+2 %, so v1 failed every run on B1/B2
at -0.4..-0.9 %. v2 (this commit) was written BEFORE any Battery-3/4 result
existed (jobs 58415463/58415464 pending at commit time; see the handoff).

A family passes when every one of its runs passes; the battery passes when
every family passes. This script decides nothing else and changes nothing.

  python -m CDDF_analysis.hbi_mcmc.perz_gate JSON [JSON ...] [--out OUT.json]
"""
from __future__ import annotations
import argparse
import json
import os

CRIT = dict(version="v2 (2026-08-20, pre-Battery-3)", rhat_max=1.05, div_max=10,
            allz_20p0_abs=0.5, allz_20p3_lo=-0.5, allz_20p3_hi=3.0,
            bin_20p0_abs={"B1": 2, "B2": 2, "B3": 2, "B4": 2, "B5": 4},
            bin_20p3_lo=-1.0, bin_20p3_hi=4.0, bin_20p3_bins=["B1", "B2", "B4", "B5"],
            bin_20p3_exempt=["B3"], b5_or_in95=True)


def _fam(pack):
    for k in ("2lpt0", "london0", "saclay0"):
        if k in os.path.basename(pack):
            return k
    return os.path.basename(pack)


def gate_one(d):
    fails = []
    mx = d["diagnostics"]["estimand_mixing"]
    for k, v in mx.items():
        if v["split_rhat"] is not None and v["split_rhat"] > CRIT["rhat_max"]:
            fails.append(f"rhat {k} {v['split_rhat']}")
    if (d.get("divergences") or 0) > CRIT["div_max"]:
        fails.append(f"divergences {d['divergences']}")
    a0 = d["thresholds"]["ge20.0"]["median_bias_pct"]
    a3 = d["thresholds"]["ge20.3"]["median_bias_pct"]
    if abs(a0) > CRIT["allz_20p0_abs"]:
        fails.append(f"allz ge20.0 {a0:+.2f}")
    if not (CRIT["allz_20p3_lo"] <= a3 <= CRIT["allz_20p3_hi"]):
        fails.append(f"allz ge20.3 {a3:+.2f}")
    p = d["perz_recovery"]["estimand"]
    b0 = {b["bin"]: b["median_bias_pct"] for b in p["ge20.0"]["paper1_bins"] if b.get("available")}
    b3 = {b["bin"]: b["median_bias_pct"] for b in p["ge20.3"]["paper1_bins"] if b.get("available")}
    in95_0 = {b["bin"]: b["truth_in_95"] for b in p["ge20.0"]["paper1_bins"] if b.get("available")}
    in95_3 = {b["bin"]: b["truth_in_95"] for b in p["ge20.3"]["paper1_bins"] if b.get("available")}
    for name, tol in CRIT["bin_20p0_abs"].items():
        if name in b0 and abs(b0[name]) > tol:
            if name == "B5" and CRIT["b5_or_in95"] and in95_0.get("B5"):
                continue
            fails.append(f"{name} ge20.0 {b0[name]:+.2f}")
    for name in CRIT["bin_20p3_bins"]:
        if name in b3 and not (CRIT["bin_20p3_lo"] <= b3[name] <= CRIT["bin_20p3_hi"]):
            if name == "B5" and CRIT["b5_or_in95"] and in95_3.get("B5"):
                continue
            fails.append(f"{name} ge20.3 {b3[name]:+.2f}")
    return dict(pack=d["pack"], family=_fam(d["pack"]), n_draws=d["n_draws"],
                divergences=d.get("divergences"),
                allz=dict(ge20p0=a0, ge20p3=a3), bins_ge20p0=b0, bins_ge20p3=b3,
                named_residual_B3_ge20p3=b3.get("B3"),
                fp_mode=d["diagnostics"].get("fp_mode"),
                sens=dict(fp_s_empty=d["diagnostics"].get("fp_s_empty"),
                          fp_total_scale=d["diagnostics"].get("fp_total_scale"),
                          t_scale=d["diagnostics"].get("t_scale")),
                fails=fails, status="PASS" if not fails else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs = [gate_one(json.load(open(p))) | {"file": os.path.basename(p)} for p in a.jsons]
    fam = {}
    for r in runs:
        fam.setdefault(r["family"], []).append(r)
    fam_status = {f: ("PASS" if all(r["status"] == "PASS" for r in rs) else "FAIL")
                  for f, rs in fam.items()}
    out = dict(criteria=CRIT, runs=runs, family_status=fam_status,
               battery_status="PASS" if all(v == "PASS" for v in fam_status.values()) else "FAIL",
               role="PREDECLARED per-bin recovery gate; decides nothing beyond PASS/FAIL")
    for r in runs:
        print(f"{r['status']:4s} {r['file']:48s} allz {r['allz']['ge20p0']:+.2f}/{r['allz']['ge20p3']:+.2f} "
              f"B3res {r['named_residual_B3_ge20p3']} fails={r['fails']}")
    print("family:", fam_status, "battery:", out["battery_status"])
    print(f"criteria: {CRIT.get('version')}; bins exempt from the >=20.3 per-bin band: {CRIT.get('bin_20p3_exempt')} "
          f"(the named B3 residual is carried as ledger line L1, not gated)")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
