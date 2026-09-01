#!/usr/bin/env python3
"""r041_prescription_gate.py — the PREDECLARED paired injection-prescription gate (private notes:
governance/MAX4_INJECTION_PRESCRIPTION_GATE_SPEC_2026-09-02.md; PI ruling 2026-09-01 late §9–§11). Inputs: the
per-injection CSVs written by tools/r041_analyze.py for arm A and arm B (same plan; pairing key TARGETID + inj_idx + wave).
Computes per pair d = y_B − y_A; cellwise ΔC with a paired bootstrap (percentile 68/95 %) and the exact two-sided binomial
p-value on the discordant counts; the production-weighted ΔC_w over the leveraged region with the FIXED weights passed in
--weights (JSON: g per molly cell, s per stratum, q candidate-bearing fraction per stratum); the tier verdict per the spec.
No verdict is computed until both arms are complete; nothing here changes the spec.

    python tools/r041_prescription_gate.py --a a_per_injection.csv [...] --b b_per_injection.csv [...] --weights weights.json --out gate.json
"""
import argparse, csv, json, sys
import numpy as np

MOLLY = [19.5, 20.0, 20.3, 20.5, 21.0, 21.5, 22.0, np.inf]
LEVERAGED_POINTS = (20.3, 20.5, 21.0)


def mcell(ln):
    return int(np.searchsorted(MOLLY, ln, side="right") - 1)


def load(paths):
    rows = []
    for p in paths:
        for r in csv.DictReader(open(p)):
            rows.append(dict(key=(int(r["TARGETID"]), int(r["wave"]), int(r["inj_idx"])), logN=float(r["logN"]), stratum=int(r["stratum"]), cand=int(r.get("has_cand_ge20", 0) or 0),
                             det=int(str(r["detected"]).lower() in ("true", "1")), nhat=(float(r["nhat"]) if r.get("nhat") not in (None, "", "nan") else None),
                             n_acc=(int(float(r["n_accepted"])) if r.get("n_accepted") not in (None, "") else None), TARGETID=int(r["TARGETID"]), wave=int(r["wave"]), inj_idx=int(r["inj_idx"]), z=float(r["z_inj"])))
    return rows


def exact_binom_p(nplus, nminus):
    from math import comb
    n = nplus + nminus
    if n == 0:
        return 1.0
    k = min(nplus, nminus); p = sum(comb(n, i) for i in range(0, k + 1)) / 2**n
    return float(min(1.0, 2 * p))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True); ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--weights", required=True, help="JSON with g_cell (molly cell index -> weight), s_stratum (list), q_cand (list of candidate-bearing fractions per stratum)")
    ap.add_argument("--n-boot", type=int, default=4000); ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--out", required=True); ap.add_argument("--label", default="stage1")
    a = ap.parse_args(argv)
    A = {r["key"]: r for r in load(a.a)}; B = {r["key"]: r for r in load(a.b)}
    keys = sorted(set(A) & set(B)); lostA = sorted(set(B) - set(A)); lostB = sorted(set(A) - set(B))
    W = json.load(open(a.weights)); g = {int(k): float(v) for k, v in W["g_cell"].items()}; s = np.asarray(W["s_stratum"], float); q = np.asarray(W["q_cand"], float)
    pairs = []
    for k in keys:
        ra, rb = A[k], B[k]
        assert abs(ra["logN"] - rb["logN"]) < 1e-9 and ra["stratum"] == rb["stratum"] and abs(ra["z"] - rb["z"]) < 1e-9, f"unpaired truth for {k}"
        pairs.append(dict(TARGETID=k[0], wave=k[1], inj_idx=k[2], injection_id=f"cmp:{k[1]}:{k[0]}:{k[2]}", logN=ra["logN"], cell=mcell(ra["logN"]), stratum=ra["stratum"], cand=ra["cand"], z=ra["z"],
                          yA=ra["det"], yB=rb["det"], d=rb["det"] - ra["det"], nhatA=ra["nhat"], nhatB=rb["nhat"], n_accA=ra["n_acc"], n_accB=rb["n_acc"],
                          w=g.get(mcell(ra["logN"]), 0.0) * s[ra["stratum"]] * (q[ra["stratum"]] if ra["cand"] else 1 - q[ra["stratum"]])))
    P = pairs; n = len(P); rng = np.random.default_rng(a.seed)
    d = np.array([p["d"] for p in P]); w = np.array([p["w"] for p in P]); logN = np.array([p["logN"] for p in P]); st = np.array([p["stratum"] for p in P]); cd = np.array([p["cand"] for p in P]); yA = np.array([p["yA"] for p in P]); yB = np.array([p["yB"] for p in P])
    lev = np.isin(logN, LEVERAGED_POINTS)

    def cellstat(mask):
        m = np.where(mask)[0]
        if m.size == 0:
            return None
        dd = d[m]; boots = np.array([dd[rng.integers(0, m.size, m.size)].mean() for _ in range(a.n_boot)])
        return dict(n=int(m.size), C_A=float(yA[m].mean()), C_B=float(yB[m].mean()), dC=float(dd.mean()), n_plus=int((dd == 1).sum()), n_minus=int((dd == -1).sum()),
                    ci68=[float(np.percentile(boots, 16)), float(np.percentile(boots, 84))], ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                    p_exact=exact_binom_p(int((dd == 1).sum()), int((dd == -1).sum())), underpowered=bool(m.size < 5))
    cells = {}
    for ln in sorted(set(logN.tolist())):
        cells[f"logN={ln}"] = cellstat(logN == ln)
        for sidx in range(5):
            cells[f"logN={ln}|stratum={sidx}"] = cellstat((logN == ln) & (st == sidx))
        for c in (0, 1):
            cells[f"logN={ln}|cand={c}"] = cellstat((logN == ln) & (cd == c))
    # production-weighted over the leveraged region, joint bootstrap (resample pairs within each (logN, stratum, cand) cell)
    def weighted(idx):
        ww = w[idx]; return float((ww * d[idx]).sum() / ww.sum()) if ww.sum() > 0 else float("nan")
    groups = {}
    for i in np.where(lev)[0]:
        groups.setdefault((logN[i], st[i], cd[i]), []).append(i)
    boots = []
    for _ in range(a.n_boot):
        idx = np.concatenate([np.array(v)[rng.integers(0, len(v), len(v))] for v in groups.values()])
        boots.append(weighted(idx))
    boots = np.array(boots); dCw = weighted(np.where(lev)[0])
    ci95 = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]; ci68 = [float(np.percentile(boots, 16)), float(np.percentile(boots, 84))]
    # tiers (spec §5)
    Lo, Up = ci95; amax = max(abs(Lo), abs(Up))
    coherent = all((cells[f"logN={ln}"] is not None) and (np.sign(cells[f"logN={ln}"]["dC"]) == np.sign(dCw) != 0) and not (cells[f"logN={ln}"]["ci68"][0] <= 0 <= cells[f"logN={ln}"]["ci68"][1]) for ln in LEVERAGED_POINTS)
    c2035 = cells.get("logN=20.3"); big2035 = bool(c2035 and abs(c2035["dC"]) > 0.10 and not (c2035["ci95"][0] <= 0 <= c2035["ci95"][1]))
    if abs(dCw) <= 0.03 and amax < 0.05:
        tier = "INSENSITIVE"
    elif (abs(dCw) > 0.05 and not (Lo <= 0 <= Up)) or coherent or big2035:
        tier = "REBUILD / STOP"
    elif abs(dCw) <= 0.05 and amax <= 0.08:
        tier = "ACCEPTABLE WITH NAMED BOUND"
    else:
        tier = "INCONCLUSIVE -> ADD INJECTIONS (stage 2)"
    out = dict(label=a.label, n_pairs=n, n_leveraged=int(lev.sum()), lost_in_A=[list(k) for k in lostA], lost_in_B=[list(k) for k in lostB], weights=W,
               production_weighted=dict(dC_w=dCw, ci68=ci68, ci95=ci95, n_boot=a.n_boot, seed=a.seed, coherent_structure=bool(coherent), big_2035=big2035, tier=tier,
                                        by_cand={c: weighted(np.where(lev & (cd == c))[0]) for c in (0, 1)}, by_stratum={s_: weighted(np.where(lev & (st == s_))[0]) for s_ in range(5)},
                                        discordance_rate_leveraged=float((d[lev] != 0).mean())),
               cells=cells, pairs=P)
    json.dump(out, open(a.out, "w"), indent=1)
    with open(a.out[:-5] + "_pairs.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(P[0].keys())); wr.writeheader(); wr.writerows(P)
    print(json.dumps({k: out["production_weighted"][k] for k in ("dC_w", "ci68", "ci95", "tier", "discordance_rate_leveraged", "by_cand")}, indent=1))
    print("cells (logN):", {k: (v["n"], round(v["C_A"], 3), round(v["C_B"], 3), round(v["dC"], 3), v["ci95"]) for k, v in cells.items() if "|" not in k and v})
    print("lost pairs:", len(lostA), len(lostB))
    return 0


if __name__ == "__main__":
    sys.exit(main())
