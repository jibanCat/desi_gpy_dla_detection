#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""§23 realized-support MATERIAL-degradation check (PI 2026-08-13).

Compares the realized (post-collision-freeze) plan against the approved
H2-v2 design. Material criteria (per the ruling — approximate benchmarks,
not third-decimal gates): FAIL only if a decision-relevant stratum loses
>25% of its designed count or any important stratum becomes nearly empty
(<10 injections), or a whole regime loses coverage. Otherwise PASS.
"""
import csv
import json
import sys

ZB = [(3.8, 4.25), (4.25, 4.5), (4.5, 5.0)]
NRE = {"19.5-20.0": ("19.5", "19.75"), "20.0-20.3": ("20.0", "20.25"),
       "20.3-20.7": ("20.5",), "20.7-21.1": ("20.75", "21.0"),
       "21.1-21.5": ("21.25", "21.5")}


def table(rows):
    t = {}
    for r in rows:
        z = float(r["z_inj"])
        zb = next((f"[{a},{b})" for a, b in ZB if a <= z < b), "?")
        nr = next((k for k, pts in NRE.items() if r["logN"] in pts
                   or f"{float(r['logN']):.2f}".rstrip("0").rstrip(".") in pts), "?")
        for key in [("total",), ("arm", r["cell"][0]), ("cell", r["cell"]),
                    ("zbin", zb), ("nre", nr), ("arm_zbin", r["cell"][0], zb),
                    ("cell_zbin", r["cell"], zb),
                    ("zbin_nre", zb, nr)]:
            t[key] = t.get(key, 0) + 1
    return t


def main():
    frozen = list(csv.DictReader(open(sys.argv[1])))   # h2_injection_plan.csv (v2)
    realized = list(csv.DictReader(open(sys.argv[2]))) # h2_realized_plan.csv
    tf, tr = table(frozen), table(realized)
    report, fails = [], []
    for key in sorted(tf, key=str):
        f, r = tf[key], tr.get(key, 0)
        frac = r / f if f else 1.0
        row = dict(stratum=":".join(map(str, key)), designed=f, realized=r,
                   frac=round(frac, 3))
        # decision-relevant strata: everything except the finest cell_zbin slices
        important = key[0] in ("total", "arm", "cell", "zbin", "nre",
                               "arm_zbin", "zbin_nre")
        if important and (frac < 0.75 or (f >= 20 and r < 10)):
            row["MATERIAL"] = True
            fails.append(row)
        report.append(row)
    n_sl_f = len({r["TARGETID"] for r in frozen})
    n_sl_r = len({r["TARGETID"] for r in realized})
    out = dict(n_frozen=len(frozen), n_realized=len(realized),
               n_sightlines_frozen=n_sl_f, n_sightlines_realized=n_sl_r,
               material_failures=fails, strata=report,
               _verdict="MATERIAL-DEGRADATION" if fails else "PASS")
    json.dump(out, open(sys.argv[3], "w"), indent=1)
    print(f"support check: {len(realized)}/{len(frozen)} injections, "
          f"{n_sl_r}/{n_sl_f} sightlines, material failures: {len(fails)} "
          f"-> {out['_verdict']}")
    for f in fails:
        print("  MATERIAL:", f)


if __name__ == "__main__":
    main()
