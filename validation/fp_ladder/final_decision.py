"""Mechanical application of the SEALED E-vs-B decision rule (FINAL_LADDER_PREDECLARATION.md §4, sha 739afa37) and the
response-model envelope report (PI ruling 2026-09-13d §16). VALIDATION-ONLY read-out of RUN JSONs; selects nothing that the
sealed text does not dictate.

Inputs: the final-ladder runs directory (RUN_<TAG>_<fam>_s<seed>.json under <TAG>/), the two finalist TAGs, the reference TAG
(first ladder A0+C1nsadd, linked in), and the frozen gate outcome per run (read from the VARIANT_TABLE.json produced by
variant_table.py on the same directory).
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np

FAMS = ("2lpt0", "london0", "saclay0")
THR = ("ge20.0", "ge20.3")


def load_runs(runs_dir, tag):
    out = {}
    for p in sorted(glob.glob(os.path.join(runs_dir, tag, "RUN_*_s*.json"))):   # any RUN in the tag dir (REF dirs keep their own names)
        j = json.load(open(p)); fam = next(f for f in FAMS if f"_{f}_" in os.path.basename(p))
        out.setdefault(fam, []).append(j)
    return out


def headline(j, thr):
    t = j["thresholds"][thr]; tr = t["truth"]; p16, p50, p84 = t["post_p16_50_84"]
    return dict(bias_pct=100 * (p50 / tr - 1), hw68_pct=100 * 0.5 * (p84 - p16) / tr, in68=bool(t["truth_in_68"]))


def zigzag_range(j):
    b = [r["median_bias_pct"] for r in j["reporting_bins"] if r.get("median_bias_pct") is not None]
    return float(max(b) - min(b)) if b else float("nan")


def k1_bias(j, thr="ge20.3"):
    cells = j["perz_recovery"]["estimand"][thr]["native_cells"]
    # coarse block K1 = z in [2.5, 3.0): path-weighted median bias over the cells in that block
    w = np.array([c["dX"] for c in cells if 2.5 <= c["z"][0] < 3.0 and c.get("available")])
    b = np.array([c["median_bias_pct"] for c in cells if 2.5 <= c["z"][0] < 3.0 and c.get("available")])
    return float(np.sum(w * b) / np.sum(w)) if w.size else float("nan")


def health(j):
    g = j["diagnostics"]
    return dict(div=int(j["divergences"]), ebfmi_min=float(min(g["ebfmi_per_chain"])))


def summarise(runs, gate_by_unit):
    """Per family: seeds, headline mean/seed-sd, hw68 mean, zigzag, K1, health, gate PASS count."""
    S = {}
    for fam, js in runs.items():
        h = {thr: [headline(j, thr) for j in js] for thr in THR}
        S[fam] = dict(
            n_seeds=len(js),
            bias={thr: float(np.mean([x["bias_pct"] for x in h[thr]])) for thr in THR},
            bias_sd={thr: (float(np.std([x["bias_pct"] for x in h[thr]], ddof=1)) if len(js) > 1 else float("nan")) for thr in THR},
            hw68={thr: float(np.mean([x["hw68_pct"] for x in h[thr]])) for thr in THR},
            zigzag=float(np.mean([zigzag_range(j) for j in js])),
            k1_ge20p3=float(np.mean([k1_bias(j) for j in js])),
            div_max=max(health(j)["div"] for j in js),
            ebfmi_min=min(health(j)["ebfmi_min"] for j in js),
            gate_pass=[gate_by_unit.get(os.path.basename(j.get("_path", "")), gate_by_unit.get(j.get("_file", ""), None)) for j in js],
        )
    return S


def decide(SE, SB, seed_sd_floor=0.05):
    """Sealed §4. Returns (outcome, carried, baseline, alternate, trail)."""
    trail = []
    passE = sum(1 for f in SE for g in SE[f]["gate_pass"] if g == "PASS"); passB = sum(1 for f in SB for g in SB[f]["gate_pass"] if g == "PASS")
    famE = sum(1 for f in SE if all(g == "PASS" for g in SE[f]["gate_pass"])); famB = sum(1 for f in SB if all(g == "PASS" for g in SB[f]["gate_pass"]))
    trail.append(f"frozen-gate families passing (all seeds): E {famE}/3, B {famB}/3 (unit PASS counts E {passE}, B {passB})")

    def not_worse_structured(SX, SY):
        # X not worse than Y on any structured residual by more than 2 seed-sd (zigzag range, |K1|), per family
        bad = []
        for f in SX:
            sd = max(SX[f]["bias_sd"].get("ge20.3", 0.0) or 0.0, seed_sd_floor)
            if SX[f]["zigzag"] > SY[f]["zigzag"] + 2 * max(0.3, sd): bad.append(f"{f}: zigzag {SX[f]['zigzag']:.1f} vs {SY[f]['zigzag']:.1f}")
            if abs(SX[f]["k1_ge20p3"]) > abs(SY[f]["k1_ge20p3"]) + 2 * sd: bad.append(f"{f}: |K1| {abs(SX[f]['k1_ge20p3']):.2f} vs {abs(SY[f]['k1_ge20p3']):.2f}")
        return bad

    if famE != famB:
        win, lose, SW, SL = (("E", "B", SE, SB) if famE > famB else ("B", "E", SB, SE))
        bad = not_worse_structured(SW, SL)
        if not bad:
            trail.append(f"Outcome A: {win} passes strictly more families and is not worse on any structured residual")
            return "A", [win], win, lose, trail
        trail.append(f"{win} passes more families but is worse on structured residuals: {bad} -> not Outcome A")
    # Outcome B test: same families passing AND every headline differs by < 1 hw68 on every family
    consistent = all(abs(SE[f]["bias"][t] - SB[f]["bias"][t]) < min(SE[f]["hw68"][t], SB[f]["hw68"][t]) for f in SE for t in THR)
    if famE == famB and consistent:
        # combined rank: calibration prediction (B), conditionality (E) fixed; closure mean |bias|; transfer spread; zigzag; |K1|; health; parsimony (B)
        def score(S, cal, cond, pars):
            mean_abs = np.mean([abs(S[f]["bias"][t]) for f in S for t in THR])
            spread = max(max(S[f]["bias"][t] for f in S) - min(S[f]["bias"][t] for f in S) for t in THR)
            zz = np.mean([S[f]["zigzag"] for f in S]); k1 = np.mean([abs(S[f]["k1_ge20p3"]) for f in S])
            hb = max(S[f]["div_max"] for f in S)
            return dict(mean_abs_bias=mean_abs, transfer_spread=spread, zigzag=zz, k1=k1, div_max=hb, calibration_rank=cal, conditionality_rank=cond, parsimony_rank=pars)
        sE, sB = score(SE, 2, 1, 2), score(SB, 1, 2, 1)
        keys = ("mean_abs_bias", "transfer_spread", "zigzag", "k1", "div_max", "calibration_rank", "conditionality_rank", "parsimony_rank")
        rE = sum(1 for k in keys if sE[k] < sB[k]); rB = sum(1 for k in keys if sB[k] < sE[k])
        trail.append(f"Outcome B: both pass {famE}/3 and headlines agree within 1 hw68 on every family; wins E {rE} / B {rB} over {len(keys)} classes")
        base = "E" if rE > rB else ("B" if rB > rE else "B")  # tie -> parsimony (B, 36 coef)
        trail.append(f"baseline = {base} (tie rule: parsimony)")
        return "B", ["E", "B"], base, ("B" if base == "E" else "E"), trail
    mE = np.mean([abs(SE[f]["bias"]["ge20.0"]) + abs(SE[f]["bias"]["ge20.3"]) for f in SE]); mB = np.mean([abs(SB[f]["bias"]["ge20.0"]) + abs(SB[f]["bias"]["ge20.3"]) for f in SB])
    base = "E" if mE < mB else "B"
    trail.append(f"Outcome C: mean(|>=20.0| + |>=20.3|) over families E {mE:.2f} vs B {mB:.2f} pp -> carry {base}; the other becomes the response-model envelope")
    return "C", [base], base, ("B" if base == "E" else "E"), trail


def envelope(Sbase, Salt, Sref):
    """PI §16: baseline, alternate, signed difference, max |shift| in old (REF) and new (baseline) hw68 units, conservative envelope."""
    rows = []; mx_old = 0.0; mx_new = 0.0; mx_pp = 0.0
    for f in FAMS:
        for t in THR:
            d = Salt[f]["bias"][t] - Sbase[f]["bias"][t]
            old = Sref[f]["hw68"][t] if f in Sref else float("nan"); new = Sbase[f]["hw68"][t]
            rows.append(dict(family=f, threshold=t, baseline_bias_pct=Sbase[f]["bias"][t], alternate_bias_pct=Salt[f]["bias"][t],
                             signed_diff_pp=d, diff_in_old_hw68=d / old if old else float("nan"), diff_in_new_hw68=d / new))
            mx_old = max(mx_old, abs(d / old)) if old else mx_old; mx_new = max(mx_new, abs(d / new)); mx_pp = max(mx_pp, abs(d))
    return dict(rows=rows, max_abs_shift_pp=mx_pp, max_abs_shift_old_hw68=mx_old, max_abs_shift_new_hw68=mx_new,
                conservative_envelope_pp=mx_pp, note="signed differences preserved; half-difference is NOT reported as a 1-sigma")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--table", required=True)
    ap.add_argument("--tag-e", default="E-phi2lpt-C1nsadd"); ap.add_argument("--tag-b", default="B-phi2lpt-C1nsadd")
    ap.add_argument("--tag-ref", default="REF-A0+C1nsadd"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tab = json.load(open(a.table)); gate = {u["file"]: u["gate_status"] for u in tab["units"]}
    def load(tag):
        R = load_runs(a.runs, tag)
        for fam, js in R.items():
            for j, p in zip(js, sorted(glob.glob(os.path.join(a.runs, tag, f"RUN_*_{fam}_s*.json")))):
                j["_file"] = os.path.basename(p)
        return R
    RE, RB, RR = load(a.tag_e), load(a.tag_b), load(a.tag_ref)
    SE, SB, SR = summarise(RE, gate), summarise(RB, gate), summarise(RR, gate)
    outcome, carried, base, alt, trail = decide(SE, SB)
    Sb, Sa = (SE, SB) if base == "E" else (SB, SE)
    env = envelope(Sb, Sa, SR)
    res = dict(sealed_rule="FINAL_LADDER_PREDECLARATION.md §4 (sha 739afa37)", outcome=outcome, carried_to_F2=carried, baseline=base,
               alternate=alt, trail=trail, summary=dict(E=SE, B=SB, REF=SR), envelope=env)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(dict(outcome=outcome, carried=carried, baseline=base, trail=trail), indent=1))
    for k in ("E", "B", "REF"):
        S = res["summary"][k]
        for f in FAMS:
            if f in S:
                s = S[f]; print(f"{k:4s} {f:8s} seeds {s['n_seeds']} gate {s['gate_pass']} >=20.0 {s['bias']['ge20.0']:+.2f}% (hw {s['hw68']['ge20.0']:.2f}) >=20.3 {s['bias']['ge20.3']:+.2f}% (hw {s['hw68']['ge20.3']:.2f}) zigzag {s['zigzag']:.1f} K1 {s['k1_ge20p3']:+.2f} div {s['div_max']} ebfmi {s['ebfmi_min']:.2f}")
    print("envelope: max|shift| %.2f pp = %.2f old hw68 = %.2f new hw68" % (env["max_abs_shift_pp"], env["max_abs_shift_old_hw68"], env["max_abs_shift_new_hw68"]))


if __name__ == "__main__":
    main()
