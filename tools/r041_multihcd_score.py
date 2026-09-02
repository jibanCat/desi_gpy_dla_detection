#!/usr/bin/env python
"""r041_multihcd_score.py — per-absorber / per-system scoring of truth-known multi-HCD sightlines under the
predeclared MAX4 multi-HCD generalization gate (notes: MAX4_MULTI_HCD_GENERALIZATION_GATE_2026-09-02.md §3-§5 +
Amendment 1; written before any P1 pairs output was read).

Inputs
  --truth      CSV(s) with one row per TRUTH absorber: TARGETID, wave, inj_idx, logN, z_inj, stratum, snr, [pair_class,
               dv_kms, pair_logN, has_cand_ge20] (the R-041D truth tables) — or a native-truth CSV with the same columns.
  --outputs    finder output dir(s) (dlacat-*.fits), matched in order with --truth.
  --reference  the m = 1 reference per-injection CSV (P0 fiducial `analysis_fid_MAX4_per_injection.csv`; rows with
               has_cand_ge20 == 0 are the same sightline class as the pairs).
  --weights    the frozen gate weights JSON (g_cell, s_stratum); --population for zlo/zhi (only truth inside the window counts).
Rules (fixed by the spec): accepted rows = P_DLA >= p_thr (0.99); one-to-one greedy matching by |dz|/(1+z) <= 0.01, closest
first; RESOLVABLE pairs (dv >= 3000 km/s = the configured MIN_Z_SEPARATION) scored per absorber; UNRESOLVABLE pairs (dv < 3000)
scored as one SYSTEM with N_sys = log10(N1 + N2) at the N-weighted mean z (matched if a row lies within tolerance of either
member); labels: matched / missed / captured (nearest row within tolerance already claimed by a sibling) / split (extra
accepted row inside the pair span). Per-cell completeness with Jeffreys intervals; the primary scalar dC_w^multi and the
propagated delta with bootstrap over sightlines (B, seed fixed); verdict PASS / BOUNDED / FAIL per §5.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

C_KMS = 299792.458
LEV = [(20.3, 20.5), (20.5, 21.0), (21.0, 21.5)]          # leveraged truth-N cells
MIN_SEP_KMS = 3000.0                                       # the finder's configured MIN_Z_SEPARATION
CLASSES = [("close", 0.0, 1000.0), ("moderate", 1000.0, 4000.0), ("gap", 4000.0, 8000.0), ("wide", 8000.0, np.inf)]


def jeffreys(k, n):
    from scipy.stats import beta
    if n == 0:
        return [None, None]
    return [float(beta.ppf(0.16, k + 0.5, n - k + 0.5)), float(beta.ppf(0.84, k + 0.5, n - k + 0.5))]


def load_accepted(outdirs, p_thr, waves):
    """{(wave, TARGETID): [(z, logN, P), ...]} from every dlacat in the output dirs; outputs dir i belongs to wave waves[i]
    (a sightline re-used in several waves is a DIFFERENT injected spectrum per wave and must never be pooled)."""
    from astropy.io import fits
    acc = {}
    for d, w in zip(outdirs, waves):
        for f in sorted(glob.glob(os.path.join(d, "dlacat-*.fits"))):
            t = fits.open(f)[1].data
            for r in t:
                if float(r["P_DLA"]) >= p_thr:
                    acc.setdefault((int(w), int(r["TARGETID"])), []).append((float(r["Z_DLA"]), float(r["NHI"]), float(r["P_DLA"])))
    return acc


def sep_class(dv):
    for name, lo, hi in CLASSES:
        if lo <= dv < hi:
            return name
    return "wide"


def cell_of(logN):
    for lo, hi in LEV:
        if lo <= logN < hi:
            return f"[{lo},{hi})"
    if 20.0 <= logN < 20.3:
        return "[20.0,20.3)"
    if logN >= 21.5:
        return "[21.5,inf)"
    return "[<20.0)"


def match_sightline(truth, rows, tol):
    """One-to-one greedy matching (closest first). truth: list of dicts with z; rows: list of (z, N, P).
    Returns per-truth dicts (matched, row index, dz, Nhat) and the set of unused row indices."""
    cand = []
    for i, t in enumerate(truth):
        for j, (z, N, P) in enumerate(rows):
            d = abs(z - t["z"]) / (1.0 + t["z"])
            if d <= tol:
                cand.append((d, i, j))
    cand.sort()
    used_t, used_r, out = set(), set(), {}
    for d, i, j in cand:
        if i in used_t or j in used_r:
            continue
        used_t.add(i); used_r.add(j); out[i] = dict(row=j, dz=d, Nhat=rows[j][1])
    for i, t in enumerate(truth):
        if i not in out:
            nearest = [(abs(z - t["z"]) / (1.0 + t["z"]), j) for j, (z, N, P) in enumerate(rows)]
            nearest = [x for x in nearest if x[0] <= tol]
            out[i] = dict(row=None, dz=None, Nhat=None, captured=bool(nearest))      # nearest row within tolerance exists but was claimed
    return out, set(range(len(rows))) - used_r


def score(truth_by_tid, acc, tol):
    """Returns the list of scored UNITS (absorbers for resolvable, systems for unresolvable) + descriptive per-absorber rows."""
    units, absorbers = [], []
    for key, tl in truth_by_tid.items():
        wave, tid = key
        rows = acc.get(key, [])
        tl = sorted(tl, key=lambda t: t["z"])
        if len(tl) == 1:
            m, unused = match_sightline(tl, rows, tol); r = m[0]
            units.append(dict(TARGETID=tid, wave=wave, kind="single", m_true=1, logN=tl[0]["logN"], z=tl[0]["z"], stratum=tl[0]["stratum"], sep_class="none", dv=np.nan,
                              matched=r["row"] is not None, Nhat=r["Nhat"], dN=(r["Nhat"] - tl[0]["logN"]) if r["Nhat"] is not None else np.nan, n_acc=len(rows), split=0))
            continue
        # pairs / multiples: nearest-neighbour separation per absorber; resolvability per adjacent pair
        dvs = [C_KMS * (tl[i + 1]["z"] - tl[i]["z"]) / (1.0 + tl[i]["z"]) for i in range(len(tl) - 1)]
        if len(tl) == 2 and dvs[0] < MIN_SEP_KMS:
            n1, n2 = 10 ** tl[0]["logN"], 10 ** tl[1]["logN"]
            zs = (tl[0]["z"] * n1 + tl[1]["z"] * n2) / (n1 + n2); Nsys = np.log10(n1 + n2)
            # matched if a row lies within tolerance of either member (one row for the system)
            m, unused = match_sightline([dict(z=tl[0]["z"]), dict(z=tl[1]["z"])], rows, tol)
            hits = [m[i] for i in (0, 1) if m[i]["row"] is not None]
            split = int(len({h["row"] for h in hits}) >= 2)
            Nhat = hits[0]["Nhat"] if hits else None
            if len(hits) == 2:                                    # two rows within the pair -> take the higher-N one as the system match
                Nhat = max(h["Nhat"] for h in hits)
            units.append(dict(TARGETID=tid, wave=wave, kind="system", m_true=2, logN=Nsys, z=zs, stratum=tl[0]["stratum"], sep_class=sep_class(dvs[0]), dv=dvs[0],
                              matched=bool(hits), Nhat=Nhat, dN=(Nhat - Nsys) if Nhat is not None else np.nan, n_acc=len(rows), split=split,
                              members=f"{tl[0]['logN']}+{tl[1]['logN']}", pair_class=tl[0].get("pair_class", "")))
            for i, t in enumerate(tl):
                absorbers.append(dict(TARGETID=tid, wave=wave, inj_idx=t.get("inj_idx"), logN=t["logN"], z=t["z"], stratum=t["stratum"], m_true=2, resolvable=False, dv_nn=dvs[0],
                                      sep_class=sep_class(dvs[0]), matched=m[i]["row"] is not None, captured=m[i].get("captured", False), Nhat=m[i]["Nhat"],
                                      dN=(m[i]["Nhat"] - t["logN"]) if m[i]["Nhat"] is not None else np.nan, n_acc=len(rows)))
            continue
        m, unused = match_sightline(tl, rows, tol)
        zmin, zmax = tl[0]["z"], tl[-1]["z"]
        extra = [j for j in unused if zmin - tol * (1 + zmin) <= rows[j][0] <= zmax + tol * (1 + zmax)]
        for i, t in enumerate(tl):
            dv_nn = min([dvs[i - 1]] if i > 0 else [np.inf] + ([dvs[i]] if i < len(dvs) else []))
            dv_nn = min(([dvs[i - 1]] if i > 0 else []) + ([dvs[i]] if i < len(dvs) else []))
            res = dv_nn >= MIN_SEP_KMS
            rec = dict(TARGETID=tid, wave=wave, inj_idx=t.get("inj_idx"), logN=t["logN"], z=t["z"], stratum=t["stratum"], m_true=len(tl), resolvable=res, dv_nn=dv_nn,
                       sep_class=sep_class(dv_nn), matched=m[i]["row"] is not None, captured=m[i].get("captured", False), Nhat=m[i]["Nhat"],
                       dN=(m[i]["Nhat"] - t["logN"]) if m[i]["Nhat"] is not None else np.nan, n_acc=len(rows), split=len(extra))
            absorbers.append(rec)
            if res:
                units.append(dict(TARGETID=tid, wave=wave, kind="absorber", m_true=len(tl), logN=t["logN"], z=t["z"], stratum=t["stratum"], sep_class=rec["sep_class"], dv=dv_nn,
                                  matched=rec["matched"], Nhat=rec["Nhat"], dN=rec["dN"], n_acc=len(rows), split=len(extra), pair_class=t.get("pair_class", "")))
            else:                                                 # unresolvable member of a >2 system: system-level with its nearest neighbour
                units.append(dict(TARGETID=tid, wave=wave, kind="unresolved_member", m_true=len(tl), logN=t["logN"], z=t["z"], stratum=t["stratum"], sep_class=rec["sep_class"], dv=dv_nn,
                                  matched=rec["matched"] or rec["captured"], Nhat=rec["Nhat"], dN=rec["dN"], n_acc=len(rows), split=len(extra), pair_class=t.get("pair_class", "")))
    return units, absorbers


def completeness_table(units, key):
    tab = {}
    for u in units:
        k = key(u)
        t = tab.setdefault(k, dict(n=0, k=0, dN=[]))
        t["n"] += 1; t["k"] += int(u["matched"])
        if u["matched"] and u["dN"] == u["dN"]:
            t["dN"].append(u["dN"])
    out = {}
    for k, t in tab.items():
        out[k] = dict(n=t["n"], k=t["k"], C=t["k"] / t["n"] if t["n"] else None, C68=jeffreys(t["k"], t["n"]), dN_mean=float(np.mean(t["dN"])) if t["dN"] else None,
                      dN_median=float(np.median(t["dN"])) if t["dN"] else None, dN_sd=float(np.std(t["dN"])) if len(t["dN"]) > 1 else None,
                      frac_below_203=float(np.mean([x < 0 for x in t["dN"]])) if t["dN"] else None)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", nargs="+", required=True); ap.add_argument("--outputs", nargs="+", required=True)
    ap.add_argument("--reference", required=True, help="m=1 reference per-injection CSV (P0 fiducial)"); ap.add_argument("--reference-candfree-only", action="store_true", default=True)
    ap.add_argument("--population", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--f-multi", type=float, nargs="+", default=[0.155, 0.476], help="brackets of the real multi-HCD fraction (mock truth; real MAP upper bound)")
    ap.add_argument("--p-thr", type=float, default=0.99); ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--n-boot", type=int, default=4000); ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--out", required=True); ap.add_argument("--label", default="pairs")
    a = ap.parse_args(argv)
    pop = {int(r["TARGETID"]): (float(r["zlo"]), float(r["zhi"])) for r in csv.DictReader(open(a.population))}
    truth_by = {}
    n_outside = 0
    waves = []
    assert len(a.truth) == len(a.outputs), "--truth and --outputs must be paired in order (one finder output dir per truth file / wave)"
    for tf in a.truth:
        wv = None
        for r in csv.DictReader(open(tf)):
            tid = int(r["TARGETID"]); z = float(r["z_inj"]); w = int(r.get("wave", 0) or 0); wv = w if wv is None else wv
            assert w == wv, f"{tf}: mixed wave values in one truth file"
            lo, hi = pop.get(tid, (-np.inf, np.inf))
            if not (lo <= z <= hi):
                n_outside += 1; continue
            truth_by.setdefault((w, tid), []).append(dict(z=z, logN=float(r["logN"]), stratum=int(r["stratum"]), inj_idx=int(r.get("inj_idx", 0)), pair_class=r.get("pair_class", ""), dv_plan=r.get("dv_kms", "")))
        waves.append(0 if wv is None else wv)
    acc = load_accepted(a.outputs, a.p_thr, waves)
    units, absorbers = score(truth_by, acc, a.tol)
    # reference (m = 1): P0 fiducial per-injection table, candidate-free rows
    ref = [r for r in csv.DictReader(open(a.reference)) if (not a.reference_candfree_only) or int(r.get("has_cand_ge20", 0) or 0) == 0]
    ref_units = [dict(matched=r["detected"] == "True", logN=float(r["logN"]), stratum=int(r["stratum"]), dN=(float(r["nhat"]) - float(r["logN"])) if r["detected"] == "True" and r["nhat"] not in ("", "nan") else np.nan) for r in ref]
    W = json.load(open(a.weights))
    g = {str(k): float(v) for k, v in W["g_cell"].items()} if isinstance(W.get("g_cell"), dict) else None
    s_str = W["s_stratum"]
    def wcell(cell, st):
        gk = {"[20.3,20.5)": 1, "[20.5,21.0)": 2, "[21.0,21.5)": 3}.get(cell)
        gg = (g.get(str(gk), 0.0) if g else {"[20.3,20.5)": 0.00312, "[20.5,21.0)": 0.00653, "[21.0,21.5)": 0.00250}[cell])
        return gg * s_str[st]
    lev_units = [u for u in units if cell_of(u["logN"]) in ("[20.3,20.5)", "[20.5,21.0)", "[21.0,21.5)")]
    lev_ref = [u for u in ref_units if cell_of(u["logN"]) in ("[20.3,20.5)", "[20.5,21.0)", "[21.0,21.5)")]
    ref_tab = completeness_table(lev_ref, lambda u: (cell_of(u["logN"]), u["stratum"]))
    def primary(us, rng=None):
        """dC_w^multi and delta over leveraged cells; optional bootstrap resample over sightlines."""
        if rng is not None:
            tids = sorted({(u["wave"], u["TARGETID"]) for u in us}); pick = rng.choice(len(tids), len(tids), replace=True)
            by = {}
            for u in us: by.setdefault((u["wave"], u["TARGETID"]), []).append(u)
            us = [u for i in pick for u in by[tids[i]]]
        tab = completeness_table(us, lambda u: (cell_of(u["logN"]), u["stratum"]))
        num = den = 0.0; dnum = 0.0
        for key, t in tab.items():
            r = ref_tab.get(key)
            if r is None or t["n"] < 5 or r["n"] < 5 or r["C"] in (None, 0):
                continue
            w = wcell(key[0], key[1]); dm = t["C"] - r["C"]
            num += w * dm; den += w; dnum += w * dm / r["C"]
        return (num / den if den else np.nan, dnum / den if den else np.nan)
    dCw, rel = primary(lev_units)
    rng = np.random.default_rng(a.seed)
    boots = np.array([primary(lev_units, rng) for _ in range(a.n_boot)])
    ci = lambda x: [float(np.nanpercentile(x, q)) for q in (16, 84, 2.5, 97.5)]
    deltas = {f"f_multi={f}": dict(delta=float(f * rel) if rel == rel else None, ci68_95=[float(f * v) for v in ci(boots[:, 1])]) for f in a.f_multi}
    fmax = max(a.f_multi); dmax = fmax * rel if rel == rel else np.nan; dci = [fmax * v for v in ci(boots[:, 1])]
    # per class / per cell / per combination tables
    by_class = completeness_table(lev_units, lambda u: u["sep_class"]); by_class_ref = {"m=1": completeness_table(lev_ref, lambda u: "all")}
    by_cell = completeness_table(lev_units, lambda u: cell_of(u["logN"])); by_cell_ref = completeness_table(lev_ref, lambda u: cell_of(u["logN"]))
    by_combo = completeness_table([u for u in units], lambda u: u.get("pair_class", "") + "|" + (u.get("members", "") if u["kind"] == "system" else str(u["logN"])))
    by_kind = completeness_table(units, lambda u: (u["kind"], u["sep_class"]))
    cls_cell = completeness_table(lev_units, lambda u: (u["sep_class"], cell_of(u["logN"])))
    # merged / split / pair recovery
    pairs = {}
    for ab in absorbers: pairs.setdefault((ab["wave"], ab["TARGETID"]), []).append(ab)
    def rates(sel):
        ps = [v for v in pairs.values() if sel(v)]
        if not ps: return dict(n=0)
        return dict(n=len(ps), pair_recovery=float(np.mean([all(x["matched"] for x in v) for v in ps])), any_captured=float(np.mean([any(x.get("captured") for x in v) for v in ps])),
                    none_matched=float(np.mean([not any(x["matched"] for x in v) for v in ps])), split=float(np.mean([any(x.get("split", 0) for x in v) for v in ps])))
    merge = {c: rates(lambda v, c=c: v[0]["sep_class"] == c) for c, _, _ in CLASSES}
    merge["resolvable_all"] = rates(lambda v: all(x["resolvable"] for x in v)); merge["unresolvable_all"] = rates(lambda v: not all(x["resolvable"] for x in v))
    # migration at [20.3,20.5)
    mig_m = by_cell.get("[20.3,20.5)", {}); mig_r = by_cell_ref.get("[20.3,20.5)", {})
    dmu = (mig_m.get("dN_mean") - mig_r.get("dN_mean")) if mig_m.get("dN_mean") is not None and mig_r.get("dN_mean") is not None else None
    # verdict (§5 + Amendment 1)
    wide_merge = merge.get("wide", {}).get("any_captured")
    class_fail = any((v["C"] is not None) and (v["n"] >= 20) and abs((v["C"] - ((by_cell_ref.get(k[1]) or {}).get("C") or v["C"]))) > 0.30 and k[1] == "[20.3,20.5)" for k, v in cls_cell.items())
    class_gt015 = [k for k, v in cls_cell.items() if v["n"] >= 20 and by_cell_ref.get(k[1], {}).get("C") is not None and abs(v["C"] - by_cell_ref[k[1]]["C"]) > 0.15
                   and not (v["C68"][0] <= by_cell_ref[k[1]]["C"] <= v["C68"][1])]
    d = abs(dmax) if dmax == dmax else np.inf; d95 = max(abs(dci[2]), abs(dci[3])) if dci else np.inf
    if d <= 0.02 and d95 < 0.05 and not class_gt015 and (dmu is None or abs(dmu) <= 0.05) and (wide_merge is None or wide_merge <= 0.10):
        verdict = "PASS"
    elif d > 0.05 and min(abs(dci[2]), abs(dci[3])) > 0.02 or (wide_merge is not None and wide_merge > 0.30) or class_fail:
        verdict = "FAIL"
    elif d <= 0.05 and d95 < 0.08:
        verdict = "BOUNDED"
    else:
        verdict = "INCONCLUSIVE"
    sk = lambda d: {("|".join(str(x) for x in k) if isinstance(k, tuple) else str(k)): v for k, v in d.items()}   # JSON-safe keys
    ref_tab_s, by_kind, cls_cell_s = sk(ref_tab), sk(by_kind), sk(cls_cell)
    class_gt015 = ["|".join(str(x) for x in k) for k in class_gt015]
    res = dict(label=a.label, n_truth_units=len(units), n_absorbers=len(absorbers), n_truth_outside_window=n_outside, n_sightline_spectra=len(truth_by), n_sightlines=len({k[1] for k in truth_by}), waves=waves, n_reference=len(lev_ref),
               primary=dict(dC_w_multi=float(dCw) if dCw == dCw else None, dC_w_ci68_95=ci(boots[:, 0]), rel_shift=float(rel) if rel == rel else None, rel_ci68_95=ci(boots[:, 1]),
                            delta_by_f_multi=deltas, delta_conservative=float(dmax) if dmax == dmax else None, delta_conservative_ci68_95=dci, f_multi_used=a.f_multi),
               by_cell=by_cell, by_cell_reference=by_cell_ref, by_class=by_class, by_class_x_cell=cls_cell_s, by_combination=by_combo, by_kind=by_kind, reference_cell_x_stratum=ref_tab_s,
               merge_split=merge, migration_2035=dict(multi=mig_m, reference=mig_r, dmu=dmu), verdict=dict(tier=verdict, classes_gt_015=class_gt015, wide_merged=wide_merge),
               rules=dict(p_thr=a.p_thr, tol=a.tol, min_sep_kms=MIN_SEP_KMS, classes=[(n, lo, (hi if np.isfinite(hi) else 'inf')) for n, lo, hi in CLASSES], n_boot=a.n_boot, seed=a.seed))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=lambda o: o if not isinstance(o, (np.floating, np.integer)) else float(o))
    with open(os.path.splitext(a.out)[0] + "_units.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for u in units for k in u})); w.writeheader(); w.writerows(units)
    with open(os.path.splitext(a.out)[0] + "_absorbers.csv", "w", newline="") as fh:
        if absorbers:
            w = csv.DictWriter(fh, fieldnames=sorted({k for u in absorbers for k in u})); w.writeheader(); w.writerows(absorbers)
    print(json.dumps(dict(primary=res["primary"], verdict=res["verdict"], merge_split=res["merge_split"]), indent=1))


if __name__ == "__main__":
    main()
