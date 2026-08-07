#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1 holdout ADJUDICABILITY + POWER gate — DESIGN-SIDE computation.

Rulings: the holdout may be opened only after a frozen adjudicability +
power gate (effective size, support coverage, power vs material defects,
false-pass/false-fail); underpowered ⇒ do NOT open.

This script reads ONLY generation-time design data:
  * the frozen roles map (which inj_id is held-out-evaluation);
  * `injection_truth.fits` DESIGN columns (logN_true, native_snr) —
    written by the generator BEFORE any GP run.
It never opens gp_out, never touches detection/matching/N̂ outcomes for
ANY role, and produces no dx statistic. Expected per-bin precision is
projected from the CALIBRATION-role pair yields and robust widths
already committed in t2_power.json / t2_pairing.json.

Power targets (material defects, frozen in P1_ESTIMAND_SPEC.md §7):
  D-mean-31: a +0.031 dex kernel mean error over [20.7,21.1)
             (== the §9 G3 effect size, 450 counts);
  D-mean-50: a 0.050 dex error (clamp-scale, the Stage-2A finding);
  D-comp-5:  a 5-point completeness drop from the calibration level.
"""
import json
import os
import time

import numpy as np
from astropy.table import Table

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
ARM = "/scratch/cavestru_root/cavestru0/mfho/phaseC_resp/prod_v1"
ROLES = os.path.join(_REPO, "diagnostics_phaseC/stage2A/roles_prod_v1.json")

# frozen kernel-support bins (P1_ESTIMAND_SPEC.md §4)
BINS = [(19.5, 20.0), (20.0, 20.4), (20.4, 20.7), (20.7, 21.0),
        (21.0, 21.3), (21.3, 21.7), (21.7, 22.4)]
ALPHA_Z = {0.05: 1.959964, 0.01: 2.575829}


def _phi(z):
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def main():
    roles = json.load(open(ROLES))
    rmap = {int(k): v["role"] for k, v in roles["roles"].items()}
    man = Table.read(os.path.join(ARM, "injection_truth.fits"))
    role = np.array([rmap[int(i)] for i in man["inj_id"]])
    N = np.asarray(man["logN_true"], float)
    S = np.asarray(man["native_snr"], float)
    hold = role == "held-out-evaluation"
    calib = (role == "bridge") | (role == "production-calibration")

    # calibration pair yield per bin: committed matched-pair counts
    # (t2_pairing/t2_power runs) over design counts, live-SNR side
    pw = json.load(open(os.path.join(_HERE, "t2_power.json")))
    widths = {tuple(r["N"]): r["injected"] for r in pw["R3_widths"]}
    npairs = {tuple(r["N"]): r["injected"]["n"] for r in pw["R3_widths"]}

    out = {"schema": "p1_holdout_gate/v1", "date": time.strftime("%Y-%m-%d"),
           "design_side_only": True, "n_holdout_design": int(hold.sum()),
           "n_calib_design": int(calib.sum()), "bins": []}
    print(f"holdout design n = {hold.sum()}, calib design n = {calib.sum()}")
    for lo, hi in BINS:
        mh = hold & (N >= lo) & (N < hi)
        mhl = mh & (S > 2.0)
        mc = calib & (N >= lo) & (N < hi)
        mcl = mc & (S > 2.0)
        # projected op-matched pair yield from the calibration roles;
        # bins outside R3's committed range fall back to the Tier-1
        # injected completeness table (flagged): completeness-only power
        T1_FALLBACK_YIELD = {(19.5, 20.0): 0.81, (20.0, 20.4): 0.90,
                             (21.7, 22.4): 0.98}
        key = (lo, hi)
        y = (npairs.get(key, np.nan) / mcl.sum()) if mcl.sum() else np.nan
        yield_fallback = False
        if not np.isfinite(y) and key in T1_FALLBACK_YIELD:
            y = T1_FALLBACK_YIELD[key]
            yield_fallback = True
        n_eff = float(mhl.sum() * y) if np.isfinite(y) else np.nan
        # robust width of the injected kernel in this bin (nearest committed)
        if key in widths:
            w = widths[key]["robust"]
        else:
            w = 0.10  # nearest-bin conservative default, flagged
        sig = w / np.sqrt(n_eff) if n_eff and n_eff > 4 else np.nan
        row = {"N": [lo, hi], "n_holdout": int(mh.sum()),
               "n_holdout_live": int(mhl.sum()),
               "calib_pair_yield": None if not np.isfinite(y) else float(y),
               "n_eff_pairs": None if not np.isfinite(n_eff) else n_eff,
               "robust_width_used": w,
               "sigma_mean_dx": None if not np.isfinite(sig) else float(sig),
               "width_flagged_default": key not in widths,
               "yield_from_t1_fallback": yield_fallback}
        for name, d in [("D-mean-31", 0.031), ("D-mean-50", 0.050)]:
            if np.isfinite(sig):
                row[f"power_{name}"] = {
                    a: float(_phi(d / sig - z)) for a, z in ALPHA_Z.items()}
        # completeness defect: 5-point drop detectability
        if mhl.sum() >= 10 and np.isfinite(y):
            c0 = min(max(y, 0.05), 0.99)
            sc = np.sqrt(c0 * (1 - c0) / mhl.sum())
            row["power_D-comp-5"] = {
                a: float(_phi(0.05 / sc - z)) for a, z in ALPHA_Z.items()}
        out["bins"].append(row)
        p31 = row.get("power_D-mean-31", {}).get(0.05)
        p50 = row.get("power_D-mean-50", {}).get(0.05)
        print(f"[{lo},{hi}): n_hold={mh.sum()} live={mhl.sum()} "
              f"n_eff={n_eff if np.isfinite(n_eff) else float('nan'):.0f} "
              f"σ={sig if np.isfinite(sig) else float('nan'):.4f} "
              f"P(.031)={p31 if p31 else float('nan'):.2f} "
              f"P(.050)={p50 if p50 else float('nan'):.2f}")

    # pooled critical window [20.7,21.1): σ on the pooled mean
    mh = hold & (N >= 20.7) & (N < 21.1) & (S > 2.0)
    # pooled yield/width from the two straddling committed bins
    y = np.mean([out["bins"][3]["calib_pair_yield"] or np.nan,
                 out["bins"][4]["calib_pair_yield"] or np.nan])
    w = np.mean([widths[(20.7, 21.0)]["robust"],
                 widths[(21.0, 21.3)]["robust"]])
    n_eff = mh.sum() * y
    sig = w / np.sqrt(n_eff)
    out["critical_window_20p7_21p1"] = {
        "n_holdout_live": int(mh.sum()), "n_eff_pairs": float(n_eff),
        "sigma_mean_dx": float(sig),
        "power_D-mean-31": {a: float(_phi(0.031 / sig - z))
                            for a, z in ALPHA_Z.items()},
        "power_D-mean-50": {a: float(_phi(0.050 / sig - z))
                            for a, z in ALPHA_Z.items()}}
    print(f"critical [20.7,21.1): live={mh.sum()} n_eff={n_eff:.0f} "
          f"σ={sig:.4f} P(.031|.05)="
          f"{out['critical_window_20p7_21p1']['power_D-mean-31'][0.05]:.2f} "
          f"P(.050|.05)="
          f"{out['critical_window_20p7_21p1']['power_D-mean-50'][0.05]:.2f}")

    with open(os.path.join(_HERE, "p1_holdout_gate.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
