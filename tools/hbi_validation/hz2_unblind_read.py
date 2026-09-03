#!/usr/bin/env python
"""HZ2 controlled unblinding — NARROW READ ONLY (PI ruling 2026-09-03 late afternoon, §4–§6, §11; predeclared comparison thresholds from
MAX4_HZ2_HBI_CLOSURE_GATE_2026-09-03.md §4, frozen 5c8d08f).

Prints, in the ruling's order, and NOTHING ELSE from the run records:
  1. dN/dX(>=20.3) pooled median, 68 %, 95 %;  2. dN/dX(>=20.0) idem;  3./4. intervals;  5. per-seed (retained) medians and per-chain medians for
  the two estimands;  6. split-R-hat, ESS (computed here from the retained seeds' draws of the two estimands), divergences of the retained pool;
  7. G_A real-mode level per retained seed;  then the predeclared comparison with the independent P0 diagnostic, whose value and band are
  read from a PRIVATE JSON (--diag-json {value, p16, p84, p2p5, p97p5, source}; never committed here):
  COMPATIBLE iff the diagnostic lies inside the pooled 95 % AND |median - diag| <= 0.0068; MILDLY SHIFTED iff |median - diag| <= 0.0136; else MATERIALLY DIFFERENT.
Fine reporting bins, per-z structure, nuisance posteriors, corner data are NOT touched.
"""
import argparse
import glob
import json
import os

import numpy as np

THR_COMPAT, THR_MILD = 0.0068, 0.0136
EST = {"ge20.3": "dndx_dla_20p3_allz", "ge20.0": "dndx_dla_20p0_allz"}


def ess_1d(x):
    """Effective sample size of a 1-d chain via initial positive sequence (Geyer); x: (n,)."""
    x = np.asarray(x, float); n = x.size; x = x - x.mean(); v = np.dot(x, x) / n
    if v <= 0:
        return float(n)
    rho = []
    for k in range(1, n):
        r = np.dot(x[:-k], x[k:]) / n / v
        rho.append(r)
        if k >= 2 and k % 2 == 0 and (rho[-1] + rho[-2]) <= 0:
            rho = rho[:-2]; break
    tau = 1.0 + 2.0 * sum(rho)
    return float(n / max(tau, 1e-9))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--pooled", required=True); ap.add_argument("--run-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--diag-json", required=True, help="PRIVATE JSON with the independent diagnostic: {value, p16, p84, p2p5, p97p5, source}")
    a = ap.parse_args(argv)
    DIAG = json.load(open(a.diag_json))
    P = json.load(open(a.pooled)); out = dict(pooled=a.pooled, pooled_sha256_recorded_separately=True)
    thr = P["thresholds"]
    for k, key in EST.items():
        q = thr[key]["post_p2p5_16_50_84_97p5"]; out[k] = dict(median=q[2], p16=q[1], p84=q[3], p2p5=q[0], p97p5=q[4], sigma68_rel=0.5 * (q[3] - q[1]) / q[2])
    print(f"1. dN/dX(>=20.3) pooled median {out['ge20.3']['median']:.4f}")
    print(f"2. dN/dX(>=20.0) pooled median {out['ge20.0']['median']:.4f}")
    for k in ("ge20.3", "ge20.0"):
        o = out[k]; print(f"3./4. {k}: 68 % [{o['p16']:.4f}, {o['p84']:.4f}]  95 % [{o['p2p5']:.4f}, {o['p97p5']:.4f}]  (sigma68/median {100*o['sigma68_rel']:.1f} %)")
    # 5. per-seed consistency (retained pool per the pooled artifact's selection)
    sel = P.get("selection") or P
    def norm(e):
        if isinstance(e, dict):
            return (int(e.get("seed")), bool(e.get("deep")), str(e.get("reason", "")))
        return (int(e[0]), bool(e[1]), str(e[2]) if len(e) > 2 else "")
    inc = [norm(e) for e in sel.get("included", [])]; exc = [norm(e) for e in sel.get("excluded", [])]
    out["selection"] = dict(included=inc, excluded=exc, n_included=len(inc), n_excluded=len(exc))
    print(f"5. retained seeds {[(s[0], 'deep' if s[1] else 'base') for s in inc]}; excluded {[(s[0], 'deep' if s[1] else 'base', s[2]) for s in exc]}")
    per = {}
    for seed, deep, _ in inc:
        p = os.path.join(a.run_dir, f"REAL_ln_{'deep_' if deep else ''}s{seed}.json"); j = json.load(open(p)); d = j["diagnostics"]
        row = dict(deep=bool(deep), divergences=int(d["divergences"]), split_rhat=d.get("split_rhat"), perchain=d.get("perchain_estimand_medians"),
                   G_A_real=(j.get("guards", {}).get("G_A_real_mode") or j.get("G_A_real_mode")))
        for k, key in EST.items():
            row[k + "_median"] = j["thresholds"][key]["post_p2p5_16_50_84_97p5"][2]
        # ESS from the saved draws of the two estimands (fdraws -> integrated series), if the fdraws file exists
        fp = p.replace(".json", "_fdraws.npz")
        if os.path.exists(fp):
            z = np.load(fp); f = z["f"]; ntrue = z["ntrue_edges"]; dX = z["dX_k"] if "dX_k" in z.files else None
            if dX is not None:
                dN = np.diff(ntrue); W = dX / dX.sum()
                for k, lo in (("ge20.3", 20.3), ("ge20.0", 20.0)):
                    m = ntrue[:-1] >= lo - 1e-9; series = ((f[:, m, :] * dN[None, m, None]).sum(axis=1) * W[None, :]).sum(axis=1)
                    n = series.size; half = n // 2   # draws are chain-concatenated (chain 0 then chain 1)
                    row[k + "_ess"] = float(ess_1d(series[:half]) + ess_1d(series[half:]))
        per[seed] = row
        print(f"   seed {seed} ({'deep' if deep else 'base'}): >=20.3 {row['ge20.3_median']:.4f} >=20.0 {row['ge20.0_median']:.4f} | div {row['divergences']} R-hat {row['split_rhat']} | "
              f"ESS 20.3 {row.get('ge20.3_ess', float('nan')):.0f} 20.0 {row.get('ge20.0_ess', float('nan')):.0f} | per-chain {row['perchain']} | G_A_real {row['G_A_real']}")
    out["per_seed"] = per
    med = np.array([per[s]["ge20.3_median"] for s in per]); out["between_seed_spread_ge20p3"] = dict(min=float(med.min()), max=float(med.max()), range_over_pooled_median=float((med.max() - med.min()) / out["ge20.3"]["median"]))
    print(f"6. between-seed spread of >=20.3 medians: {med.min():.4f}–{med.max():.4f} (range/pooled median {100*out['between_seed_spread_ge20p3']['range_over_pooled_median']:.1f} %)")
    # 7. comparison with the independent diagnostic (predeclared)
    m3 = out["ge20.3"]["median"]; diff = m3 - DIAG["value"]; sig = 0.5 * (out["ge20.3"]["p84"] - out["ge20.3"]["p16"]); dsig = 0.5 * (DIAG["p84"] - DIAG["p16"])
    inside95 = out["ge20.3"]["p2p5"] <= DIAG["value"] <= out["ge20.3"]["p97p5"]
    cls = "COMPATIBLE" if (inside95 and abs(diff) <= THR_COMPAT) else ("MILDLY SHIFTED" if abs(diff) <= THR_MILD else "MATERIALLY DIFFERENT")
    out["comparison_0p1009"] = dict(diagnostic=DIAG, hz2_median=m3, abs_diff=diff, frac_diff=diff / DIAG["value"], diff_over_hz2_sigma=diff / sig, diff_over_diag_sigma=diff / dsig,
                                    diag_inside_hz2_95=bool(inside95), thresholds=dict(compatible=THR_COMPAT, mild=THR_MILD, source="MAX4_HZ2_HBI_CLOSURE_GATE_2026-09-03.md §4 (frozen 5c8d08f)"), classification=cls)
    m0 = out["ge20.0"]["median"]
    print(f"7. vs diagnostic {DIAG['value']:.4f}: HZ2 {m3:.4f} - diag = {diff:+.4f} ({100*diff/DIAG['value']:+.1f} %); /HZ2 sigma68 {diff/sig:+.2f}; /diagnostic sigma68 {diff/dsig:+.2f}; diag inside HZ2 95 %: {inside95} → {cls}")
    print(f"   companion >=20.0: HZ2 {m0:.4f} (68 % [{out['ge20.0']['p16']:.4f},{out['ge20.0']['p84']:.4f}])")
    json.dump(out, open(a.out, "w"), indent=1, default=float); print("wrote", a.out)


if __name__ == "__main__":
    main()
