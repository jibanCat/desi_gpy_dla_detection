#!/usr/bin/env python3
"""p1_gate.py -- the acceptance gate for the diff machinery.

COMPONENT CLASS: **VALIDATION-ONLY**.

Runs `manifest_diff(M_frozen_2026-08-26, M_surgical_C1)` and requires that its output
reproduces, exactly and without hand-editing, the answer already independently established by
the OPUS-D / OPUS-D1 workers.  P1 certifies the DIFF, not the science.

Every expectation below is read from a file that predates this tool:
  comparison/D2_figdata_diff.json          P1.1, P1.2, P1.3
  systematics/TAB10_DEPENDENCY_2026-09-10.json::summary_counts   P1.4
  comparison/D1_reduction_comparison.json  P1.5
  systematics/validation/CONFIG_AMBIGUITY_*.json                 P1.6
  the C1 selection contract / GATE1 adjudication                 P1.7
  tools/check_additions.py:1063-1070 + the Phase-0b audit        P1.9, P1.10
A FAIL is a defect in the diff or in the manifest's recorded edges, never in the expectation.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import manifest_diff as MD  # noqa: E402

CANDIDATE = pathlib.Path("/nfs/turbo/lsa-cavestru/mfho/paper1_science_handoff/"
                         "LOWZ_CLEAN_C1_CANDIDATE_2026-09-10")

FIG41_EXPECT = {"allz_dndx_20p0", "allz_dndx_20p3", "allz_omega_20p3", "closure_max_rel_diff",
                "cmp_ho21_ratio_to_reference", "dX_per_bin", "l15_dndx_20p0", "l15_dndx_20p3",
                "lowz_dndx_20p0", "lowz_dndx_20p3", "lowz_omega_20p3"}
FIG40_EXPECT = {"closure_max_rel_diff", "cumulative", "dndx_allz_20p0", "dndx_allz_20p3",
                "env_truth_side_f_per_N_hi", "env_truth_side_f_per_N_lo", "f_bin_integral",
                "f_per_dex", "f_per_linear_N", "l15_f_per_linear_N", "omega_20p0", "omega_20p3",
                "ref_spine_binavg_ratio", "stat_ratio_to_median"}
TAB10_EXPECT = {
    "carry_over": {"L1", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L11", "L13",
                   "OMEGA_L1", "OMEGA_L13"},
    "refresh_mechanical": {"L12", "L14", "OMEGA_STAT", "OMEGA_UPPER_LIMIT"},
    "reclassify": {"L2", "L10", "L15", "L16", "OMEGA_L15", "OMEGA_L2"},
    "refresh_campaign": set(),
}
HEADLINES = ["dndx_20p3_allz", "dndx_20p0_allz", "omega_20p3_21p6_allz",
             "dndx_20p3_B1", "dndx_20p3_B2", "dndx_20p3_B3", "dndx_20p3_B4", "dndx_20p3_B5",
             "dndx_20p0_B1", "dndx_20p0_B2", "dndx_20p0_B3", "dndx_20p0_B4", "dndx_20p0_B5",
             "omega_20p3_21p6_B1", "omega_20p3_21p6_B2", "omega_20p3_21p6_B3",
             "omega_20p3_21p6_B4"]
CDDF = [f"cddf_{b}" for b in
        ["19.7_19.9", "19.9_20.1", "20.1_20.3", "20.3_20.5", "20.5_20.7", "20.7_20.9",
         "20.9_21.1", "21.1_21.3", "21.3_21.5", "21.5_21.7", "21.7_21.9", "21.9_22.1",
         "22.1_22.4"]]


def check(results, item, cond, detail):
    results.append({"item": item, "verdict": "PASS" if cond else "FAIL", "detail": detail})
    return cond


def stale_keys(R, prefix):
    return {c["consumer_id"][len(prefix):] for c in R["consumers"]
            if c["consumer_id"].startswith(prefix) and c["verdict"] in MD.STALE_VERDICTS}


def run(old, new, exit_code):
    R = MD.run(old, new, None)
    R["exit_code_observed"] = exit_code
    res = []
    q = R["quantities"]
    C = R["consumers"]

    # ---- P1.1 TAB-12: 18 of 21 value cells STALE-REPRINT, the 3 unchanged are the BH row
    cells = [c for c in C if c["consumer_kind"] == "table_cell"]
    reprint = [c for c in cells if c["verdict"] == "STALE-REPRINT"]
    unchanged = sorted(c["consumer_id"] for c in cells if c["verdict"] == "UNCHANGED")
    check(res, "P1.1",
          len(cells) == 21 and len(reprint) == 18
          and unchanged == ["TAB-12.BH.dndx_20p0", "TAB-12.BH.dndx_20p3", "TAB-12.BH.omega"],
          f"{len(cells)} value cells, {len(reprint)} STALE-REPRINT; unchanged = {unchanged}")

    # ---- P1.2 FIG-41: 55 keys both sides, 0 added/removed, 11 stale, exact set
    f41 = [c for c in C if c["consumer_id"].startswith("FIG-41.")]
    s41 = stale_keys(R, "FIG-41.")
    added41 = [c for c in f41 if c["verdict_note"]]
    check(res, "P1.2",
          len(f41) == 55 and not added41 and s41 == FIG41_EXPECT,
          f"{len(f41)} keys, {len(added41)} added/removed, {len(s41)} stale; "
          f"set matches D2: {s41 == FIG41_EXPECT}"
          + ("" if s41 == FIG41_EXPECT else f"; symmetric difference {s41 ^ FIG41_EXPECT}"))

    # ---- P1.3 FIG-40: 39 keys, 0 added/removed, 14 stale, exact set
    f40 = [c for c in C if c["consumer_id"].startswith("FIG-40.")]
    s40 = stale_keys(R, "FIG-40.")
    added40 = [c for c in f40 if c["verdict_note"]]
    check(res, "P1.3",
          len(f40) == 39 and not added40 and s40 == FIG40_EXPECT,
          f"{len(f40)} keys, {len(added40)} added/removed, {len(s40)} stale; "
          f"set matches D2: {s40 == FIG40_EXPECT}"
          + ("" if s40 == FIG40_EXPECT else f"; symmetric difference {s40 ^ FIG40_EXPECT}"))

    # ---- P1.4 TAB-10 dispositions reproduce the D1 classification exactly
    t10 = {c["consumer_id"][len("TAB-10."):]: c["disposition"]
           for c in C if c["consumer_id"].startswith("TAB-10.")}
    got = {d: {k for k, v in t10.items() if v == d} for d in TAB10_EXPECT}
    ok = len(t10) == 22 and all(got[d] == TAB10_EXPECT[d] for d in TAB10_EXPECT)
    check(res, "P1.4", ok,
          f"{len(t10)} low-z lines; " + "; ".join(
              f"{d} {len(got[d])}{'' if got[d] == TAB10_EXPECT[d] else ' MISMATCH ' + str(got[d] ^ TAB10_EXPECT[d])}"
              for d in TAB10_EXPECT))

    # ---- P1.5 the 17 headline quantities, half-width ratios, the 13 CDDF bins
    hl_pdc = [h for h in HEADLINES if q.get(h, {}).get("printed_digits_change") is True]
    ratios = [v["halfwidth_ratio"] for v in q.values() if v["halfwidth_ratio"] is not None]
    cd_pct = [q[c]["delta_pct"] for c in CDDF if q.get(c, {}).get("delta_pct") is not None]
    # the published bounds are quoted at the precision of the handoff table, so the
    # comparison is made at that precision rather than against the unrounded float
    ok = (len(hl_pdc) == 17
          and ratios and round(min(ratios), 4) >= 0.9175 and round(max(ratios), 4) <= 1.0039
          and len(cd_pct) == 13 and all(x < 0 for x in cd_pct)
          and round(min(cd_pct), 2) >= -3.46 and round(max(cd_pct), 2) <= -1.29)
    check(res, "P1.5", ok,
          f"{len(hl_pdc)}/17 headlines printed_digits_change=true; half-width ratios "
          f"[{min(ratios):.4f}, {max(ratios):.4f}] (expected within [0.9175, 1.0039]); "
          f"{len(cd_pct)} CDDF bins, all decreasing = {all(x < 0 for x in cd_pct)}, "
          f"d% in [{min(cd_pct):.2f}, {max(cd_pct):.2f}]")

    # ---- P1.6 L15: 6.66 -> 7.02 at p=1 (6.7 -> 7.0), verdict RECLASSIFY
    l15 = q.get("TAB10.L15.ge20p3_allz", {})
    row = next((c for c in C if c["consumer_id"] == "TAB-10.L15"), {})
    ok = (round(l15.get("old", 0), 2) == 6.66 and round(l15.get("new", 0), 2) == 7.02
          and l15.get("printed_precision") == 1 and l15.get("printed_digits_change") is True
          and l15.get("printed_old") == "6.7" and l15.get("printed_new") == "7.0"
          and row.get("verdict") == "RECLASSIFY")
    check(res, "P1.6", ok,
          f"old {l15.get('old')} -> new {l15.get('new')}; printed "
          f"{l15.get('printed_old')} -> {l15.get('printed_new')} at p="
          f"{l15.get('printed_precision')}; TAB-10.L15 verdict {row.get('verdict')}")

    # ---- P1.7 CONTRACT-BREAK on the path leg's statistic and on spectype_cut; and the
    #      break reaches real-pack/real-posterior products only, along recorded edges
    fields = {r["field"]: r for r in R["contract_break"]}
    leg = fields.get("contract.selection.snr.legs[path].statistic", {})
    spec = fields.get("contract.selection.population.spectype_cut", {})
    hz_keys = {c["consumer_id"][len("FIG-41."):] for c in C
               if c["consumer_id"].startswith("FIG-41.")
               and c["verdict"] == "UNCHANGED"}
    broken_products = {p["product_id"] for p in R["products"] if p["contract_broken"]}
    ok = (leg.get("old") == "median" and leg.get("new") == "mean"
          and spec.get("old") == "QSO" and spec.get("new") is None
          and "product:pack" in broken_products
          and "product:pooled_posterior" in broken_products
          and "product:hz2_posterior" not in broken_products
          and "product:literature_dla_data" not in broken_products
          and "product:mock_envelope" not in broken_products
          and len(hz_keys) == 44 and len(f40) - len(s40) == 25)
    check(res, "P1.7", ok,
          f"legs[path].statistic {leg.get('old')} -> {leg.get('new')}; spectype_cut "
          f"{spec.get('old')} -> {spec.get('new')}; {len(broken_products)} products broken "
          f"(hz2/literature/mock_envelope NOT among them: "
          f"{'product:hz2_posterior' not in broken_products}); "
          f"{len(hz_keys)} FIG-41 and {len(f40) - len(s40)} FIG-40 arrays come back UNCHANGED")

    # ---- P1.8 exit code 1; untraceable == 0 on the NEW side, non-zero and reported on the old
    sc = R["summary_counts"]
    check(res, "P1.8",
          exit_code == 1 and sc["untraceable"] == 0 and sc["untraceable_old_side"] > 0,
          f"exit code {exit_code}; untraceable new {sc['untraceable']}, "
          f"old {sc['untraceable_old_side']} (reported, not suppressed: "
          f"{R['untraceable_old']})")

    # ---- P1.9 the pending-literal clearance list: 9 rows, 8 cleared, :2214 UNRESOLVED
    pend = R["pending_literals"]
    by_line = {p["file_line"].rsplit(":", 1)[1]: p for p in pend}
    r2253 = by_line.get("2253", {})
    vals2253 = {v["quantity_id"]: (v.get("old_value"), v.get("new_value"))
                for v in r2253.get("replacing", [])}
    ok = (len(pend) == 9
          and sc["pending_literals_cleared"] == 8 and sc["pending_literals_unresolved"] == 1
          and by_line.get("2214", {}).get("resolution", "").startswith("UNRESOLVED")
          and "mean-plane refresh" in by_line.get("2214", {}).get("resolution", "")
          and vals2253.get("TAB10.L15.ge20p3_allz") == ("6.66", "7.02")
          and vals2253.get("TAB10.L15.ge20p0_allz") == ("12.87", "13.37"))
    check(res, "P1.9", ok,
          f"{len(pend)} rows; {sc['pending_literals_cleared']} cleared, "
          f"{sc['pending_literals_unresolved']} unresolved; :2214 -> "
          f"\"{by_line.get('2214', {}).get('resolution', '')[:60]}\"; :2253 -> {vals2253}")

    # ---- P1.10 SHIPPING / SUPPRESSED split; TAB-10 L8 and L10 are SHIPPING
    l8 = next((c for c in C if c["consumer_id"] == "TAB-10.L8"), {})
    l10 = next((c for c in C if c["consumer_id"] == "TAB-10.L10"), {})
    ok = (sc["consumers_stale_shipping"] > 0 and sc["consumers_stale_suppressed"] > 0
          and l8.get("ships") is True and l10.get("ships") is True
          and l8.get("verdict") in MD.STALE_VERDICTS and l10.get("verdict") in MD.STALE_VERDICTS
          and "2279-2323" in (l8.get("block") or ""))
    check(res, "P1.10", ok,
          f"SHIPPING-STALE {sc['consumers_stale_shipping']}, SUPPRESSED-STALE "
          f"{sc['consumers_stale_suppressed']}; TAB-10.L8 ships={l8.get('ships')} "
          f"verdict={l8.get('verdict')}; TAB-10.L10 ships={l10.get('ships')} "
          f"verdict={l10.get('verdict')}; L8 block cites the suppressed prose range")
    return R, res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--exit-code", type=int, default=None,
                    help="the exit code manifest_diff returned; defaults to recomputing it")
    a = ap.parse_args()
    ec = a.exit_code
    if ec is None:
        ec = MD.run(a.old, a.new, None)["exit_code"]
    R, res = run(a.old, a.new, ec)
    npass = sum(1 for r in res if r["verdict"] == "PASS")
    order = ["P1.1", "P1.2", "P1.3", "P1.4", "P1.5", "P1.6", "P1.7", "P1.9", "P1.10", "P1.8"]
    res.sort(key=lambda r: order.index(r["item"]) if r["item"] in order else 99)
    out = {"_class": "VALIDATION-ONLY. P1 certifies the diff machinery, not the science.",
           "gate": "P1 (MANIFEST_DIFF_SPEC.md s3)",
           "old": R["old"], "new": R["new"],
           "verdict": "PASS" if npass == len(res) else "FAIL",
           "passed": npass, "total": len(res), "items": res,
           "summary_counts": R["summary_counts"]}
    od = pathlib.Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    with open(od / "P1_GATE.json", "w") as f:
        json.dump(out, f, indent=1)
    lines = ["# GATE P1 -- acceptance test for `manifest_diff`", "",
             "**VALIDATION-ONLY.** P1 certifies the diff, not the science. Every expectation is "
             "read from a file that predates this tool; a FAIL is a defect in the diff or in the "
             "manifest's recorded edges, never in the expectation.", "",
             f"`{R['old']['manifest_id']}` -> `{R['new']['manifest_id']}`", "",
             f"## VERDICT: {out['verdict']}  ({npass}/{len(res)})", "",
             MD.md_table(["item", "verdict", "detail"],
                         [[r["item"], r["verdict"], r["detail"]] for r in res]), ""]
    (od / "P1_GATE.md").write_text("\n".join(lines))
    for r in res:
        print(f"  {r['item']:6s} {r['verdict']:4s}  {r['detail']}")
    print(f"P1 {out['verdict']} ({npass}/{len(res)})")
    raise SystemExit(0 if out["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
