#!/usr/bin/env python
"""r041_multihcd_pair_arms.py — PAIRED comparison of two multi-HCD control arms scored by r041_multihcd_score.py
(units CSVs), for the predeclared clustering-control gate (MAX4_MULTIHCD_CLUSTERING_CONTROL_GATE_2026-09-02.md §3-§4).
Units are paired by (wave, TARGETID, kind, member index) — the SAME truth system / absorber appears in both arms with the same
N multiset and internal separations (only the placement differs). Primary = production-weighted dC_w^(B - A) over the leveraged
truth-N cells (frozen weights g(N) x s(stratum)), paired bootstrap over sightlines (B = 4000, seed 20260906), 68/95 %;
per separation class x cell dC; dN-hat migration; pair recovery / merged / split fractions from the absorbers CSVs; propagated
delta = f_multi x dC_w / C_A; verdict per gate §4 (INSENSITIVE / BOUNDED / MATERIAL).
"""
from __future__ import annotations
import argparse, csv, json, os
import numpy as np

LEV = {"[20.3,20.5)": "1", "[20.5,21.0)": "2", "[21.0,21.5)": "3"}
KNOWN = {("moderate", "[20.3,20.5)")}                       # the already-known regime (3000-4000 km/s, [20.3,20.5), low S/N)


def cell_of(logN):
    for lo, hi, name in ((20.3, 20.5, "[20.3,20.5)"), (20.5, 21.0, "[20.5,21.0)"), (21.0, 21.5, "[21.0,21.5)")):
        if lo <= logN < hi: return name
    return None


def load_units(path):
    out = {}
    for r in csv.DictReader(open(path)):
        key = (int(r.get("wave", 0) or 0), int(r["TARGETID"]), r["kind"], round(float(r["logN"]), 4), round(float(r["z"]), 3) if r["kind"] == "single" else r.get("members", ""))
        out.setdefault((int(r.get("wave", 0) or 0), int(r["TARGETID"])), []).append(dict(kind=r["kind"], logN=float(r["logN"]), stratum=int(r["stratum"]), sep_class=r["sep_class"],
                                                                                  matched=r["matched"] == "True", dN=(float(r["dN"]) if r["dN"] not in ("", "nan") else np.nan), m_true=int(float(r["m_true"]))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="units CSV of arm A (reference, e.g. sysrandom)"); ap.add_argument("--b", required=True, help="units CSV of arm B (e.g. syscluster)")
    ap.add_argument("--a-label", default="A"); ap.add_argument("--b-label", default="B"); ap.add_argument("--weights", required=True)
    ap.add_argument("--f-multi", type=float, nargs="+", default=[0.155, 0.476]); ap.add_argument("--n-boot", type=int, default=4000); ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    A, B = load_units(a.a), load_units(a.b); W = json.load(open(a.weights)); g = W["g_cell"]; s_str = W["s_stratum"]
    pairs = []                                          # one record per matched unit: (sightline key, cell, stratum, class, yA, yB, dNA, dNB)
    lost = 0
    for k in sorted(set(A) & set(B)):
        ua = sorted(A[k], key=lambda u: (u["kind"], u["logN"])); ub = sorted(B[k], key=lambda u: (u["kind"], u["logN"]))
        if len(ua) != len(ub) or any(x["kind"] != y["kind"] or abs(x["logN"] - y["logN"]) > 1e-6 for x, y in zip(ua, ub)):
            lost += 1; continue                            # a system re-scored with a different resolvability (shift crossed 3000 km/s) is reported, not paired
        for x, y in zip(ua, ub):
            pairs.append(dict(sl=k, cell=cell_of(x["logN"]), stratum=x["stratum"], clsA=x["sep_class"], clsB=y["sep_class"], yA=int(x["matched"]), yB=int(y["matched"]), dNA=x["dN"], dNB=y["dN"], kind=x["kind"], m=x["m_true"]))
    lev = [p for p in pairs if p["cell"] in LEV]
    w = {(c, s): float(g[LEV[c]]) * float(s_str[s]) for c in LEV for s in range(len(s_str))}
    def dcw(P):
        num = den = 0.0; numC = 0.0
        cells = {}
        for p in P: cells.setdefault((p["cell"], p["stratum"]), []).append(p)
        for key, ps in cells.items():
            if len(ps) < 5: continue
            ya = np.mean([p["yA"] for p in ps]); yb = np.mean([p["yB"] for p in ps]); ww = w[key]
            num += ww * (yb - ya); den += ww; numC += ww * ya
        return (num / den if den else np.nan), (numC / den if den else np.nan)
    d0, C_A = dcw(lev)
    rng = np.random.default_rng(a.seed); sls = sorted({p["sl"] for p in lev}); by = {}
    for p in lev: by.setdefault(p["sl"], []).append(p)
    boots = np.array([dcw([p for i in rng.choice(len(sls), len(sls), replace=True) for p in by[sls[i]]])[0] for _ in range(a.n_boot)])
    ci = [float(np.nanpercentile(boots, q)) for q in (16, 84, 2.5, 97.5)]
    rel = d0 / C_A if C_A else np.nan
    deltas = {f"f_multi={f}": float(f * rel) for f in a.f_multi}; dcons = max(a.f_multi) * rel; dcons_ci = [max(a.f_multi) * v / C_A for v in ci]
    # per class x cell (class of arm A = the truth geometry class; identical in both arms for syscluster/sysrandom)
    cc = {}
    for p in lev:
        cc.setdefault((p["clsA"], p["cell"]), []).append(p)
    per = {f"{k[0]}|{k[1]}": dict(n=len(v), C_A=float(np.mean([p["yA"] for p in v])), C_B=float(np.mean([p["yB"] for p in v])), dC=float(np.mean([p["yB"] - p["yA"] for p in v])),
                                 dN_A=float(np.nanmean([p["dNA"] for p in v])) if any(p["dNA"] == p["dNA"] for p in v) else None, dN_B=float(np.nanmean([p["dNB"] for p in v])) if any(p["dNB"] == p["dNB"] for p in v) else None)
           for k, v in cc.items()}
    # 68 % intervals per class x cell (bootstrap over pairs)
    for k, v in cc.items():
        d = np.array([p["yB"] - p["yA"] for p in v]); bs = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(1000)])
        per[f"{k[0]}|{k[1]}"]["ci68"] = [float(np.percentile(bs, 16)), float(np.percentile(bs, 84))]
    new_fail = [k for k, v in per.items() if v["n"] >= 20 and abs(v["dC"]) > 0.15 and not (v["ci68"][0] <= 0 <= v["ci68"][1]) and tuple(k.split("|")) not in KNOWN]
    mig = {c: dict(dN_A=float(np.nanmean([p["dNA"] for p in lev if p["cell"] == c])), dN_B=float(np.nanmean([p["dNB"] for p in lev if p["cell"] == c])), n=sum(1 for p in lev if p["cell"] == c)) for c in LEV}
    by_m = {str(m): dict(n=sum(1 for p in lev if p["m"] == m), C_A=float(np.mean([p["yA"] for p in lev if p["m"] == m])) if any(p["m"] == m for p in lev) else None,
                         C_B=float(np.mean([p["yB"] for p in lev if p["m"] == m])) if any(p["m"] == m for p in lev) else None) for m in sorted({p["m"] for p in lev})}
    d = abs(dcons) if dcons == dcons else np.inf
    if d >= 0.05 or new_fail:
        tier = "MATERIAL"
    elif abs(d0) <= 0.03 and d <= 0.02:
        tier = "INSENSITIVE"
    else:
        tier = "BOUNDED"
    out = dict(a=a.a_label, b=a.b_label, n_pairs=len(pairs), n_leveraged=len(lev), n_sightlines_lost_resolvability=lost,
               primary=dict(dC_w=float(d0), ci68=ci[:2], ci95=ci[2:], C_A_w=float(C_A), rel_shift=float(rel), delta_by_f_multi=deltas, delta_conservative=float(dcons), delta_conservative_ci68_95=[float(x) for x in dcons_ci]),
               per_class_x_cell=per, migration=mig, by_m_true=by_m, new_failures_outside_known_regime=new_fail, verdict=tier,
               rules=dict(n_boot=a.n_boot, seed=a.seed, known_regime=[list(k) for k in KNOWN], tiers="INSENSITIVE |dC_w|<=0.03 & |delta|<=2%; BOUNDED |delta|<5%; MATERIAL |delta|>=5% or a new class x cell failure (|dC|>0.15, 68% excl. 0) outside the known regime"))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(dict(primary=out["primary"], verdict=tier, new_failures=new_fail, lost=lost, n_lev=len(lev)), indent=1))


if __name__ == "__main__":
    main()
