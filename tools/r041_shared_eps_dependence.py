#!/usr/bin/env python
"""r041_shared_eps_dependence.py — the correlation / variance part (b1-b4) of the predeclared shared-epsilon
micro-audit of prescription A (notes: MAX4_SHARED_EPSILON_MICROAUDIT_SPEC_2026-09-02.md §3, frozen before any A_ind
output existed).

Inputs: the paired archives of the two arms (A_shared = the historical seed-0 archives; A_ind = the same injections
rebuilt with independent per-sightline seeds), their truth CSVs (byte-identical by construction), the SOURCE archive
(to recover T_i F_i, i.e. the injected increment F' - T F), and optionally the gate's per-pair table (d_i = y_ind - y_shared)
to compute the outcome-dependence statistics b2/b3. Without --pairs only the geometry (b1, b4) is computed.

b1  pixel overlap of the profile supports {(1 - T^2) > thr} between every pair of injections of the same wave; f_o, mean
    partners; the correlation of the normalised increments u_i = (F'_i - T_i F_i) / (sqrt(1 - T_i^2) sigma_i) over the
    overlap (== 1 for a shared eps by construction; ~0 for independent eps).
b2  eps-sensitivity p_eps = P(y_ind != y_shared) overall / leveraged / per N point / per stratum.
b3  gamma = mean_{overlapping pairs} d_i d_j vs gamma0 = mean_{non-overlapping} d_i d_j, permutation p (10,000, seed 20260903);
    design effect DE = 1 + (n_lev - 1) f_o,lev max(gamma_lev - gamma0_lev, 0) / Vbar, n_eff/n = 1/DE.
b4  distribution of the profile-centre pixel index (same OBSERVED wavelength <-> same eps index, not same profile-relative position).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

LEV_POINTS = (20.3, 20.5, 21.0)


def load_wave(src, shared, ind, truth, plan_label, wave):
    import h5py
    from injection.noise_preserving import transmission
    rows = [r for r in csv.DictReader(open(truth)) if int(r["wave"]) == wave]
    with h5py.File(src, "r") as S, h5py.File(shared, "r") as A, h5py.File(ind, "r") as B:
        wl = S["wavelength"][:].astype(float)
        cat = A["catalog"][:]; tid_a = [int(t) for t in cat["TARGETID"]]
        catb = B["catalog"][:]; tid_b = [int(t) for t in catb["TARGETID"]]
        assert tid_a == tid_b, "A_shared / A_ind catalog order differs"
        scat = S["catalog"][:]; sidx = {int(t): i for i, t in enumerate(scat["TARGETID"])}
        out = []
        by_tid = {}
        for r in rows:
            by_tid.setdefault(int(r["TARGETID"]), []).append(r)
        for j, t in enumerate(tid_a):
            i = sidx[t]
            F = S["flux"][i].astype(float); iv = S["ivar"][i].astype(float); mk = S["mask"][i]
            good = np.isfinite(iv) & (iv > 0) & (mk == 0)
            sig = np.zeros_like(F); sig[good] = 1.0 / np.sqrt(iv[good])
            absorbers = [{"nhi": 10.0 ** float(r["logN"]), "z_dla": float(r["z_inj"]), "num_lines": 3} for r in by_tid[t]]
            T = transmission(wl, absorbers)
            amp = np.sqrt(np.clip(1.0 - T ** 2, 0.0, 1.0))
            Fa = A["flux"][j].astype(float); Fb = B["flux"][j].astype(float)
            ua = np.full(F.size, np.nan); ub = np.full(F.size, np.nan)
            m = good & (amp > 1e-3)
            ua[m] = (Fa[m] - T[m] * F[m]) / (amp[m] * sig[m]); ub[m] = (Fb[m] - T[m] * F[m]) / (amp[m] * sig[m])
            for r in by_tid[t]:
                out.append(dict(injection_id=f"{plan_label}:{wave}:{t}:{int(r['inj_idx'])}", TARGETID=t, wave=wave, logN=float(r["logN"]),
                                z_inj=float(r["z_inj"]), stratum=int(r["stratum"]), cand=int(r.get("has_cand_ge20", 0)),
                                centre_px=int(np.argmin(np.abs(wl - 1215.67 * (1.0 + float(r["z_inj"]))))), T=T, amp=amp, ua=ua, ub=ub, good=good))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True); ap.add_argument("--plan-label", default="cmp")
    ap.add_argument("--wave", action="append", required=True, help="WAVE:shared.h5:ind.h5:truth.csv (repeatable)")
    ap.add_argument("--pairs", default=None, help="gate pairs CSV (injection_id, d, logN, stratum, cand, yA, yB) for b2/b3")
    ap.add_argument("--thr", type=float, default=0.1, help="profile support = (1 - T^2) > thr")
    ap.add_argument("--n-perm", type=int, default=10000); ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    inj = []
    for w in a.wave:
        wave, shared, ind, truth = w.split(":")
        inj += load_wave(a.source, shared, ind, truth, a.plan_label, int(wave))
    n = len(inj)
    ids = [x["injection_id"] for x in inj]
    lev = np.array([x["logN"] in LEV_POINTS for x in inj])
    # ---- b1 / b4: geometry + realised increment correlation --------------------------------------------------------
    supp = np.array([(x["amp"] ** 2 > a.thr) for x in inj])            # (n, npix) boolean
    waves = np.array([x["wave"] for x in inj])
    O = (supp.astype(np.int32) @ supp.astype(np.int32).T)             # pixel overlaps
    same_wave = waves[:, None] == waves[None, :]
    np.fill_diagonal(O, 0)
    ov = (O > 0) & same_wave
    iu = np.triu_indices(n, 1)
    pairs_mask = same_wave[iu]
    f_o = float(ov[iu][pairs_mask].mean())
    partners = ov.sum(1)
    f_o_lev = float(ov[np.ix_(lev, lev)][np.triu_indices(lev.sum(), 1)].mean()) if lev.sum() > 1 else float("nan")
    partners_lev = ov[np.ix_(lev, lev)].sum(1)
    corr_a, corr_b, ov_px = [], [], []
    for i, j in zip(*np.where(np.triu(ov))):
        m = supp[i] & supp[j] & inj[i]["good"] & inj[j]["good"] & np.isfinite(inj[i]["ua"]) & np.isfinite(inj[j]["ua"])
        if m.sum() < 5:
            continue
        ov_px.append(int(m.sum()))
        corr_a.append(float(np.corrcoef(inj[i]["ua"][m], inj[j]["ua"][m])[0, 1]))
        corr_b.append(float(np.corrcoef(inj[i]["ub"][m], inj[j]["ub"][m])[0, 1]))
    corr_a = np.array(corr_a); corr_b = np.array(corr_b)
    def q(x):
        return dict(n=int(x.size), mean=float(np.mean(x)) if x.size else None, median=float(np.median(x)) if x.size else None,
                    p05=float(np.percentile(x, 5)) if x.size else None, p95=float(np.percentile(x, 95)) if x.size else None,
                    min=float(x.min()) if x.size else None, max=float(x.max()) if x.size else None)
    supp_px = supp.sum(1)
    res = dict(n_injections=n, n_leveraged=int(lev.sum()), support_threshold=a.thr,
               b1=dict(support_px_per_injection=q(supp_px.astype(float)), f_overlap_all_pairs_same_wave=f_o, f_overlap_leveraged=f_o_lev,
                       partners_per_injection=q(partners.astype(float)), partners_per_injection_leveraged=q(partners_lev.astype(float)),
                       n_overlapping_pairs=int(ov[iu].sum()), overlap_px=q(np.array(ov_px, float)),
                       corr_u_shared=q(corr_a), corr_u_independent=q(corr_b),
                       note="u_i = (F'_i - T_i F_i)/(sqrt(1-T_i^2) sigma_i) restricted to the joint support; shared eps -> corr == 1 by construction"),
               b4=dict(centre_px=q(np.array([x["centre_px"] for x in inj], float)),
                       note="eps is indexed by OBSERVED-grid pixel; the same index is the same observed wavelength, not the same profile-relative position"))
    # ---- b2 / b3: outcome dependence (needs the gate pairs) -------------------------------------------------------
    if a.pairs:
        P = {r["injection_id"]: r for r in csv.DictReader(open(a.pairs))}
        missing = [k for k in ids if k not in P]
        d = np.array([int(float(P[k]["d"])) if k in P else 0 for k in ids])
        have = np.array([k in P for k in ids])
        ya = np.array([float(P[k]["yA"]) if k in P else np.nan for k in ids])
        flips = d != 0
        def rate(m):
            m = m & have
            return dict(n=int(m.sum()), p_eps=float(flips[m].mean()) if m.sum() else None, n_plus=int((d[m] == 1).sum()), n_minus=int((d[m] == -1).sum()))
        logN = np.array([x["logN"] for x in inj]); st = np.array([x["stratum"] for x in inj]); cd = np.array([x["cand"] for x in inj])
        b2 = dict(all=rate(np.ones(n, bool)), leveraged=rate(lev),
                  per_N={str(v): rate(logN == v) for v in sorted(set(logN.tolist()))},
                  per_stratum={str(v): rate(st == v) for v in sorted(set(st.tolist()))},
                  per_cand={str(v): rate(cd == v) for v in sorted(set(cd.tolist()))}, missing_in_pairs=missing)
        rng = np.random.default_rng(a.seed)
        def gamma_stats(sel):
            idx = np.where(sel & have)[0]
            if idx.size < 3:
                return dict(n=int(idx.size))
            dd = d[idx]; ovs = ov[np.ix_(idx, idx)]; sw = same_wave[np.ix_(idx, idx)]
            tri = np.triu_indices(idx.size, 1)
            prod = np.outer(dd, dd)[tri]; o = ovs[tri]; s = sw[tri]
            g = float(prod[o].mean()) if o.sum() else float("nan"); g0 = float(prod[(~o) & s].mean()) if ((~o) & s).sum() else float("nan")
            obs = g - g0
            cnt = 0; perm = np.empty(a.n_perm)
            for k in range(a.n_perm):
                dp = rng.permutation(dd); pp = np.outer(dp, dp)[tri]
                perm[k] = pp[o].mean() - pp[(~o) & s].mean()
            p_val = float((perm >= obs).mean())
            C = float(np.nanmean(ya[idx])); V = C * (1.0 - C)
            fo = float(o[s].mean())
            DE = 1.0 + (idx.size - 1) * fo * max(obs, 0.0) / V if V > 0 else float("nan")
            return dict(n=int(idx.size), n_overlapping_pairs=int(o.sum()), gamma_overlap=g, gamma0_nonoverlap=g0, excess=obs,
                        perm_p_one_sided=p_val, perm_null_sd=float(perm.std()), C_shared=C, Vbar=V, f_o=fo, design_effect=DE, n_eff_over_n=(1.0 / DE if DE == DE else None))
        b3 = dict(all=gamma_stats(np.ones(n, bool)), leveraged=gamma_stats(lev))
        res.update(b2=b2, b3=b3)
    res["inputs"] = dict(source=a.source, waves=a.wave, pairs=a.pairs, plan_label=a.plan_label, n_perm=a.n_perm, seed=a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k in ("n_injections", "n_leveraged", "b1", "b2", "b3")}, indent=1)[:6000])


if __name__ == "__main__":
    main()
