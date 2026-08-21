#!/usr/bin/env python
"""cp3_envelope_check.py — measure a pooled real posterior against the
PREDECLARED envelope of expected calibration-induced change (notes
2026-08-21_CP3_PREDECLARATION.md §5; PI ruling 2026-08-21 #18: freeze path
B if only the expected changes appear, escalate to C otherwise — the
determination is the PI's). Reports in/out per line; decides nothing.
jax-free; real values go to stdout/JSON in scratch only.

  python CDDF_analysis/hbi_mcmc/cp3_envelope_check.py --new POOLED_new.json \
      --old POOLED_superseded.json [--env ENV.json] --out REPORT.json
"""
from __future__ import annotations
import argparse
import json

ENVELOPE = dict(
    allz_pct=1.0,
    bins_pts={"ge20.0": {"B1": (-6, -3), "B2": (-2, 2), "B3": (-2, 2),
                         "B4": (11, 16), "B5": (19, 28)},
              "ge20.3": {"B1": (-7, -3), "B2": (-2, 2), "B3": (1, 4),
                         "B4": (10, 17), "B5": (13, 33)}})


def _med(q):
    return float(q[2])


def envelope_report(new, old, env):
    lines = dict(allz={}, bins={})
    ok = True
    for k, tag in (("dndx_dla_20p0_allz", "ge20.0"), ("dndx_dla_20p3_allz", "ge20.3")):
        mo = _med(old["thresholds"][k]["post_p2p5_16_50_84_97p5"])
        mn = _med(new["thresholds"][k]["post_p2p5_16_50_84_97p5"])
        shift = 100.0 * (mn / mo - 1.0)
        inside = abs(shift) <= env["allz_pct"]
        ok &= inside
        lines["allz"][tag] = dict(shift_pct=shift, tol_pct=env["allz_pct"], inside=bool(inside))
    for tag in ("ge20.0", "ge20.3"):
        lines["bins"][tag] = {}
        ob = {b["bin"]: b for b in old.get("perz_paper1", {}).get(tag, {}).get("paper1_bins", []) if b.get("available")}
        nb = {b["bin"]: b for b in new.get("perz_paper1", {}).get(tag, {}).get("paper1_bins", []) if b.get("available")}
        for name, (lo, hi) in env["bins_pts"][tag].items():
            if name not in ob or name not in nb:
                lines["bins"][tag][name] = dict(available=False)
                continue
            shift = 100.0 * (_med(nb[name]["post_p2p5_16_50_84_97p5"]) / _med(ob[name]["post_p2p5_16_50_84_97p5"]) - 1.0)
            inside = lo <= shift <= hi
            ok &= inside
            lines["bins"][tag][name] = dict(shift_pts=shift, expected=[lo, hi], inside=bool(inside))
    return dict(lines=lines, all_inside=bool(ok), envelope=env,
                role="measurement vs the predeclared envelope; the B/C determination is the PI's")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--env", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    env = json.load(open(a.env)) if a.env else ENVELOPE
    rep = envelope_report(json.load(open(a.new)), json.load(open(a.old)), env)
    rep["inputs"] = dict(new=a.new, old=a.old)
    json.dump(rep, open(a.out, "w"), indent=1)
    for tag, v in rep["lines"]["allz"].items():
        print(f"allz {tag}: shift {v['shift_pct']:+.2f} % (tol ±{v['tol_pct']}) -> {'IN' if v['inside'] else 'OUT'}")
    for tag, bins in rep["lines"]["bins"].items():
        for name, v in bins.items():
            if v.get("available", True):
                print(f"{tag} {name}: shift {v['shift_pts']:+.1f} pts (expected {v['expected']}) -> {'IN' if v['inside'] else 'OUT'}")
    print("ALL INSIDE ENVELOPE:", rep["all_inside"])


if __name__ == "__main__":
    main()
