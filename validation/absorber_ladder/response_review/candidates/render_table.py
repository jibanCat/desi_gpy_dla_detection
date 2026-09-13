#!/usr/bin/env python
"""render_table.py -- render the candidate CV table as markdown for
RESPONSE_REPRESENTATION_VARIANTS.md.  Pure formatting; no science decision."""
from __future__ import annotations

import argparse
import json
import os
import sys

ORDER = ["R0", "R1c", "R1d-raw", "A-small", "E", "B", "C", "D"]


def g(d, *ks, default=None):
    for k in ks:
        if d is None:
            return default
        d = d.get(k) if isinstance(d, dict) else None
    return default if d is None else d


def f(v, n=4):
    if v is None:
        return "--"
    try:
        return f"{float(v):+.{n}f}"
    except Exception:
        return str(v)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    J = json.load(open(a.json))
    T = J["table"]
    names = [n for n in ORDER if n in T]

    print("### Table 1 -- held-out predictive score (sealed section 3 "
          "primary metric)\n")
    print("| candidate | nominal DOF | effective DOF | held-out l (fixed "
          "object) | l (conditional) | marginalisation cost | dl vs R1d-raw "
          "(+-SE) | dl vs R1c (+-SE) | in-sample - held-out | wall (s) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        t = T[n]
        ed = t.get("effective_dof")
        eds = (f"{ed['identifiable_rank']} (rank), {ed['p_eff_ridge']:.1f} "
               f"(ridge)" if isinstance(ed, dict) and "identifiable_rank" in ed
               else (f"{ed:.1f}" if isinstance(ed, (int, float)) else "--"))
        d1 = t.get("dll_vs_R1d-raw", {}); d2 = t.get("dll_vs_R1c", {})
        print(f"| {n} | {t.get('nominal_dof')} | {eds} | "
              f"{f(t.get('ll_fixed'))} | {f(t.get('ll_conditional'))} | "
              f"{f(t.get('marginalisation_cost'))} | "
              f"{f(d1.get('mean'))} +- {d1.get('se', 0):.4f} | "
              f"{f(d2.get('mean'))} +- {d2.get('se', 0):.4f} | "
              f"{f(t.get('insample_heldout_gap'))} | "
              f"{t.get('wall_s', float('nan')):.0f} |")

    print("\n### Table 2 -- held-out row deviance / KL and moment residuals\n")
    print("| candidate | rows >=20 | rows >=200 | row KL (wmean) | row KL "
          "(rows >=200) | deviance/dof | d_mean | width ratio-1 | d_skew | "
          "d_skew (b>=21.3) | d_skew (b<19.7) | d_tail(>0.3 dex) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        t = T[n]
        print(f"| {n} | {t.get('rows_ge20')} | {t.get('rows_ge200')} | "
              f"{t.get('row_kl_wmean', float('nan')):.4f} | "
              f"{t.get('row_kl_wmean_ge200', float('nan')):.4f} | "
              f"{t.get('row_dev_per_dof_wmean', float('nan')):.3f} | "
              f"{f(t.get('d_mean_wmean'))} | {f(t.get('r_sd_wmean'))} | "
              f"{f(t.get('d_skew_wmean'), 3)} | "
              f"{f(t.get('d_skew_wmean_b_ge_21p3'), 3)} | "
              f"{f(t.get('d_skew_wmean_b_lt_19p7'), 3)} | "
              f"{f(t.get('d_tail_wmean'))} |")

    print("\n### Table 3 -- boundary leakage, CDF calibration, transfer\n")
    print("| candidate | 20.0 frac>3SE | 20.0 wmean d | 20.3 frac>3SE | "
          "20.3 wmean d | 21.0 frac>3SE | 21.0 wmean d | PIT chi2 (19 dof) | "
          "KL London-0 | KL Saclay-0 | d_skew(b>=21.3) vs M_true 2LPT-0 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        t = T[n]; b = t.get("boundary", {})
        tr = t.get("transfer", {})
        mo = g(tr, "2lpt0", "moments") or {}
        row = [n]
        for k in ("20.0", "20.3", "21.0"):
            v = b.get(k, {})
            row.append(f"{v.get('frac_bad', float('nan')):.2f}")
            row.append(f(v.get("wmean_d")))
        row.append(f"{g(t, 'pit', 'chi2', default=float('nan')):.0f}")
        row.append(f"{g(tr, 'london0', 'wmean_kl', default=float('nan')):.4f}")
        row.append(f"{g(tr, 'saclay0', 'wmean_kl', default=float('nan')):.4f}")
        row.append(f(mo.get("d_skew_b_ge_21p3"), 3))
        print("| " + " | ".join(row) + " |")

    print("\n### Table 4 -- the sealed section 5 verdict\n")
    print("| candidate | level | (i) dl vs R1d-raw | (i) fires | (ii) fires | "
          "(iii) row skew b>=21.3 | (iii) fires | FAIL? | opened |")
    print("|---|---|---|---|---|---|---|---|---|")
    lvl = {"R0": 0, "R1c": 0, "R1d-raw": 0, "A-small": 1, "E": 1, "B": 2,
           "C": 2, "D": 3}
    for n in names:
        t = T[n]; s5 = t.get("section5", {})
        d = s5.get("detail", {})
        print(f"| {n} | {lvl.get(n)} | "
              f"{f(d.get('dll_mean'))} +- {d.get('dll_se', 0):.4f} | "
              f"{d.get('i_fail')} | {d.get('ii_fail')} "
              f"{','.join(d.get('ii_boundaries', []))} | "
              f"{f(d.get('skew_ge_21p3'), 3)} | {d.get('iii_fail')} | "
              f"**{s5.get('fail')}** | {t.get('opened')} |")

    ph = J.get("phi", {})
    print("\n### Table 5 -- the row HAD MASS phi (post-seal addition; NOT in "
          "section 5)\n")
    print("| estimator | nominal DOF | held-out binomial l per trial | "
          "in-sample l per trial |")
    print("|---|---|---|---|")
    cv = ph.get("cv", {}).get("heldout_ll_per_trial", {})
    ins = ph.get("insample_ll_per_trial", {})
    print(f"| measured per-cell (Jeffreys +1/2) **default** | "
          f"{ph.get('summary', {}).get('n_cells_populated')} populated cells "
          f"({ph.get('summary', {}).get('n_cells_phi_lt_0p98')} with phi < "
          f"0.98) | {f(cv.get('percell'), 5)} | {f(ins.get('percell'), 5)} |")
    print(f"| smooth logistic (poly2 N, poly1 log S/N, K offsets) | 6 | "
          f"{f(cv.get('smooth'), 5)} | {f(ins.get('smooth'), 5)} |")
    print(f"| frozen parametric pack phi_ref (for reference) | -- | -- | "
          f"{f(ins.get('pack_phi_ref'), 5)} |")
    s = ph.get("summary", {})
    print(f"\nphi / phi_ref ratio over populated cells: "
          f"{s.get('ratio_to_phi_ref_min', float('nan')):.3f} to "
          f"{s.get('ratio_to_phi_ref_max', float('nan')):.3f}; below "
          f"N_true = 19.7: {s.get('ratio_below_19p7_min', float('nan')):.3f} "
          f"to {s.get('ratio_below_19p7_max', float('nan')):.3f}. "
          f"phi(b=0, s=2, K=0) = {s.get('phi_b0_s2', float('nan')):.3f} vs "
          f"phi_ref {s.get('phi_ref_b0_s2', float('nan')):.3f}; "
          f"phi(b=0, s=7, K=0) = {s.get('phi_b0_s7', float('nan')):.3f} vs "
          f"phi_ref {s.get('phi_ref_b0_s7', float('nan')):.3f}. "
          f"Trials = {s.get('n_trials', float('nan')):.0f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
