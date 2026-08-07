#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tier-2 POWER + ROBUSTNESS ADDENDUM (post-specified diagnostics).

STATUS: everything here was specified AFTER the frozen t2_pairing.py
verdict was seen. Nothing below re-adjudicates the frozen rule; the
verdict stands as committed. These are the power/sensitivity statements
the attribution checkpoint requires (rulings §16/§19: post-selection
variables and post-specified contrasts DIAGNOSE, never decide), plus
deterministic projections from committed artifacts.

P1  power of D2 against the MATERIAL coupling alternative: what slope
    difference would catalogued-shell-mediated coupling need to produce
    the observed [21.0,21.3) offset, vs what the data allow at 95%.
P2  injected dx vs PRE-injection forest flux at the trough centre
    (`forest_flux_frac`, a design covariate recorded at generation) —
    a mechanical bound on the near-field forest-coupling channel that
    the shell proxy cannot see: if pre-existing central absorption does
    not move injected dx, correlated near-field absorption lacks a
    mechanism to inflate natural N-hat at fixed profile.
R1  joint dependence: D2 slopes per frozen N bin (same proxy, same bins).
R2  D1 offsets restricted to shell==0 pairs on BOTH sides (environment-
    free common support).
R3  kernel WIDTH comparison per frozen N bin (natural vs injected dx
    std + robust sigma) — decision input for the estimand freeze.
R4  G3 projection of the frozen D1 offsets through the committed
    preimage sensitivities, with propagated uncertainty (labeled
    back-of-envelope; the exact number remains the GATED P1 refold).

No holdout rows (roles filter inherited from t2_pairing); no forced
fits; no Stage-2B; no P2 campaign.
"""
import json
import os
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

import sys                                                    # noqa: E402
sys.path.insert(0, _HERE)
from t2_pairing import (truth_neighbors, injected_pairs,      # noqa: E402
                        natural_pairs, wmean, NB, ARM)
from astropy.table import Table                               # noqa: E402

PREIMAGE = os.path.join(_HERE, "..", "preimage", "preimage.json")
D2_DOMAIN = (20.4, 21.7)


def ols(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 30 or len(set(xs.tolist())) < 2:
        return None
    X = np.vstack([np.ones_like(xs), xs]).T
    co, *_ = np.linalg.lstsq(X, ys, rcond=None)
    resid = ys - X @ co
    s2 = float(resid @ resid) / (len(ys) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    return {"slope": float(co[1]), "slope_sigma": float(np.sqrt(cov[1, 1])),
            "n": int(len(ys))}


def d2_arrays(recs, lo=D2_DOMAIN[0], hi=D2_DOMAIN[1]):
    xs = [min(r["shell"], 2) for r in recs if lo <= r["N"] < hi]
    ys = [r["dx"] for r in recs if lo <= r["N"] < hi]
    return np.asarray(xs, float), np.asarray(ys, float)


def robust_sigma(v):
    v = np.asarray(v, float)
    q16, q84 = np.percentile(v, [15.865, 84.135])
    return float(0.5 * (q84 - q16))


def main():
    t0 = time.time()
    by = truth_neighbors()
    inj = injected_pairs(by)
    nat = natural_pairs(by)
    out = {"schema": "p1_t2_power/v1", "date": time.strftime("%Y-%m-%d"),
           "label": "POST-SPECIFIED power/robustness addendum; frozen "
                    "verdict NOT re-adjudicated"}

    # ---- P1: power of D2 against the material coupling alternative ----
    xn, yn = d2_arrays(nat)
    xi, yi = d2_arrays(inj)
    sn, si = ols(xn, yn), ols(xi, yi)
    dsl = sn["slope"] - si["slope"]
    sds = float(np.hypot(sn["slope_sigma"], si["slope_sigma"]))
    mean_shell_nat = (float(xn.mean()), float(xn.std(ddof=1) / np.sqrt(len(xn))))
    mean_shell_inj = (float(xi.mean()), float(xi.std(ddof=1) / np.sqrt(len(xi))))
    lever = mean_shell_nat[0] - mean_shell_inj[0]
    upper95 = dsl + 1.96 * sds
    # frozen D1 offset at [21.0,21.3) (t2_pairing.json)
    OFF2110 = 0.045362552080292086
    max_contrib_conservative = upper95 * mean_shell_nat[0]
    req_slope_via_meanS = OFF2110 / mean_shell_nat[0]
    out["P1_d2_power"] = {
        "mean_shell_natural": mean_shell_nat,
        "mean_shell_injected": mean_shell_inj,
        "lever_nat_minus_inj": lever,
        "slope_diff": {"value": dsl, "sigma": sds, "upper95": upper95},
        "offset_to_explain_21.0_21.3": OFF2110,
        "required_slope_diff_if_channel_via_meanS": req_slope_via_meanS,
        "required_over_measured_sigma": (req_slope_via_meanS - dsl) / sds,
        "max_channel_contribution_dex_at_95": max_contrib_conservative,
        "note": ("Isolation-mirrored naturals do NOT sit at higher "
                 "catalogued shell density than injected sightlines "
                 "(lever ~ 0 or negative); even crediting the channel "
                 "with the FULL natural mean shell count at the 95% "
                 "upper slope difference, its contribution is "
                 "max_channel_contribution_dex_at_95 — the catalogued-"
                 "neighbor channel is excluded with large margin, i.e. "
                 "D2 is NOT underpowered for that channel. The channel "
                 "with NO catalog proxy (sub-17.2 absorption within "
                 "5,000 km/s) is bounded mechanically by P2 below.")}

    # ---- P2: injected dx vs pre-injection forest flux at the trough ----
    roles_man = Table.read(os.path.join(ARM, "injection_truth.fits"))
    n_blend = n_nan = 0
    # injected_pairs() does not return target_id; join by z_true (checked
    # unique at 6 dp across the manifest via n_dup_zkey below)
    zkeys = {}
    dup = 0
    for r in roles_man:
        k = round(float(r["z_true"]), 6)
        if k in zkeys:
            dup += 1
        zkeys[k] = (float(r["forest_flux_frac"]), bool(r["forest_blend"]))
    xs_f, ys_f = [], []
    for r in inj:
        k = round(r["z"], 6)
        if k not in zkeys:
            continue
        fff, fb = zkeys[k]
        if fb:
            n_blend += 1
        if not np.isfinite(fff):
            n_nan += 1
            continue
        if D2_DOMAIN[0] <= r["N"] < D2_DOMAIN[1]:
            xs_f.append(fff)
            ys_f.append(r["dx"])
    sf = ols(xs_f, ys_f)
    xs_f = np.asarray(xs_f, float)
    ys_f = np.asarray(ys_f, float)
    terc = []
    if len(xs_f) >= 30:
        qs = np.percentile(xs_f, [33.3, 66.7])
        for lo, hi in [(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], np.inf)]:
            m = (xs_f >= lo) & (xs_f < hi)
            terc.append({"fff_range": [float(lo), float(hi)],
                         "mean_dx": wmean(ys_f[m])})
    out["P2_forest_flux_mechanical"] = {
        "n_pairs_domain": int(len(xs_f)), "n_dup_zkey": dup,
        "n_forest_blend_flagged": n_blend, "n_fff_nan": n_nan,
        "ols_dx_vs_fff": sf, "terciles": terc,
        "fff_spread_sd": float(xs_f.std(ddof=1)) if len(xs_f) > 1 else None,
        "note": ("forest_flux_frac = PRE-injection forest flux at the "
                 "trough centre / local pseudo-continuum (design "
                 "covariate, coadd_injection.py). A flat dx vs fff "
                 "bounds the mechanical channel by which correlated "
                 "near-field absorption could inflate fitted N-hat; "
                 "implied max shift = |slope|x(fff spread) reported in "
                 "the completion note.")}

    # ---- R1: per-frozen-bin D2 slopes (joint dependence) ----
    out["R1_slopes_by_N"] = []
    for lo, hi in NB:
        xnb, ynb = d2_arrays(nat, lo, hi)
        xib, yib = d2_arrays(inj, lo, hi)
        snb, sib = ols(xnb, ynb), ols(xib, yib)
        row = {"N": [lo, hi], "natural": snb, "injected": sib}
        if snb and sib:
            d = snb["slope"] - sib["slope"]
            s = float(np.hypot(snb["slope_sigma"], sib["slope_sigma"]))
            row["difference"] = {"value": d, "sigma": s, "z": d / s}
        out["R1_slopes_by_N"].append(row)

    # ---- R2: D1 restricted to shell==0 on both sides ----
    out["R2_offset_shell0"] = []
    from collections import defaultdict
    for lo, hi in NB:
        cells = defaultdict(lambda: {"i": [], "n": []})
        from t2_pairing import z2zr, s2sr
        for r in inj:
            if lo <= r["N"] < hi and r["shell"] == 0:
                cells[(z2zr(r["z"]), s2sr(r["snr"]))]["i"].append(r["dx"])
        for r in nat:
            if lo <= r["N"] < hi and r["shell"] == 0:
                cells[(z2zr(r["z"]), s2sr(r["snr"]))]["n"].append(r["dx"])
        num = den = var = 0.0
        for c, v in cells.items():
            if len(v["i"]) >= 3 and len(v["n"]) >= 3:
                w = len(v["i"])
                mi, si_, _ = wmean(v["i"])
                mn, sn_, _ = wmean(v["n"])
                num += w * (mn - mi)
                var += (w ** 2) * (si_ ** 2 + sn_ ** 2)
                den += w
        if den:
            out["R2_offset_shell0"].append(
                {"N": [lo, hi], "offset": num / den,
                 "sigma": float(np.sqrt(var) / den),
                 "z": num / den / (np.sqrt(var) / den)})

    # ---- R3: width comparison per frozen bin ----
    out["R3_widths"] = []
    for lo, hi in NB:
        vn = np.asarray([r["dx"] for r in nat if lo <= r["N"] < hi])
        vi = np.asarray([r["dx"] for r in inj if lo <= r["N"] < hi])
        if len(vn) > 5 and len(vi) > 5:
            out["R3_widths"].append(
                {"N": [lo, hi],
                 "natural": {"sd": float(vn.std(ddof=1)),
                             "robust": robust_sigma(vn), "n": len(vn)},
                 "injected": {"sd": float(vi.std(ddof=1)),
                              "robust": robust_sigma(vi), "n": len(vi)}})

    # ---- R4: G3 projection of the frozen D1 offsets ----
    pj = json.load(open(PREIMAGE))
    dg3 = np.asarray(pj["mocks"]["2lpt0"]["sensitivity"]
                     ["dG3_per_bin_mean_shift"], float)
    delta = float(pj["mocks"]["2lpt0"]["sensitivity"]["delta_mean_dex"])
    S = dg3 / delta                       # counts per dex, per 0.2-dex bin
    edges = 19.1 + 0.2 * np.arange(len(S) + 1)
    D1 = [((20.4, 20.7), 0.017702285657740312, 0.007378111443459617),
          ((20.7, 21.0), 0.024874019979816447, 0.006675277839882104),
          ((21.0, 21.3), 0.045362552080292086, 0.005669118191513693),
          ((21.3, 21.7), 0.037563641458363405, 0.00817767356666321)]
    W, proj, var = {}, 0.0, 0.0
    rows = []
    for (tlo, thi), off, sig in D1:
        w = 0.0
        for b in range(len(S)):
            blo, bhi = edges[b], edges[b + 1]
            ov = max(0.0, min(thi, bhi) - max(tlo, blo))
            w += S[b] * (ov / (bhi - blo))
        rows.append({"N": [tlo, thi], "W_counts_per_dex": w,
                     "offset": off, "sigma": sig,
                     "counts": w * off, "counts_sigma": abs(w) * sig})
        proj += w * off
        var += (w * sig) ** 2
    out["R4_g3_projection"] = {
        "per_bin": rows, "total_counts": proj,
        "total_sigma": float(np.sqrt(var)),
        "note": ("BACK-OF-ENVELOPE, labeled: the frozen D1 natural-minus-"
                 "injected offsets folded through the committed preimage "
                 "mean-shift sensitivities (0.2-dex bins, overlap-"
                 "weighted). Sub-20.4 contribution excluded (bounded "
                 "<=~25 counts, Tier-1). This projects what an "
                 "UNCORRECTED injected-anchored kernel would omit; the "
                 "exact number is the GATED P1 refold, not performed.")}

    out["wall_s"] = round(time.time() - t0, 1)
    with open(os.path.join(_HERE, "t2_power.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"P1: lever {lever:+.4f}; slope diff {dsl:+.4f}±{sds:.4f} "
          f"(95% up {upper95:+.4f}); required {req_slope_via_meanS:+.3f} "
          f"({out['P1_d2_power']['required_over_measured_sigma']:.0f}σ "
          f"above measured); max channel {max_contrib_conservative:+.4f} dex")
    if sf:
        print(f"P2: dx vs fff slope {sf['slope']:+.4f}±{sf['slope_sigma']:.4f}"
              f" (n={sf['n']}); implied max shift over fff sd "
              f"{abs(sf['slope']) * out['P2_forest_flux_mechanical']['fff_spread_sd']:.4f} dex")
    for r in out["R1_slopes_by_N"]:
        d = r.get("difference")
        print(f"R1 {r['N']}: diff z = {d['z']:+.2f}" if d else
              f"R1 {r['N']}: insufficient")
    for r in out["R2_offset_shell0"]:
        print(f"R2 {r['N']}: shell0 offset {r['offset']:+.4f} ± "
              f"{r['sigma']:.4f} (z={r['z']:+.1f})")
    for r in out["R3_widths"]:
        print(f"R3 {r['N']}: sd nat {r['natural']['sd']:.3f} "
              f"(rob {r['natural']['robust']:.3f}) vs inj "
              f"{r['injected']['sd']:.3f} (rob {r['injected']['robust']:.3f})")
    print(f"R4: ΔG3 = {proj:+.0f} ± {np.sqrt(var):.0f} counts")


if __name__ == "__main__":
    main()
