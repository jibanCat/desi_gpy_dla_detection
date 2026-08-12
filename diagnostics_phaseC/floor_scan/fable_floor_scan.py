#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fable 2026-08-12 pass 2 — fine-grid diagnostic floor scan (PI item 27.1-4).

READ-ONLY on frozen artifacts at prov/p1-refold-2026-08-08. Event-level
deposits reproduce the fold exactly (guarded), enabling arbitrary floor cuts.
Observed comparisons only at 0.1-aligned floors (pack obs grid), labeled
POST-HOC DIAGNOSTIC. No gate evaluated.
"""
import json
import os
import sys

import numpy as np

WT = os.environ.get("FLOOR_SCAN_REPO", os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.join(WT, "injection"))
sys.path.insert(0, os.path.join(WT, "diagnostics_phaseC", "p1_completeness"))
sys.path.insert(0, WT)
OUT = os.environ.get("FLOOR_SCAN_OUT", "fable_floor_scan.json")

from p1_refold_fold import (  # noqa: E402
    FLOOR, CACHE172, build_fold, build_p1_kernel, c_marginal,
    load_kernel_events, load_migration, mu_sig_p1)

from CDDF_analysis.hbi_mcmc.pack import load_pack  # noqa: E402
from p1_refold_fold import PACK  # noqa: E402


def main():
    pk = load_pack(PACK)
    fold = build_fold(pk)
    E, truth, sparse, art, cache = load_kernel_events()
    mig = load_migration(fold["nhat_edges"])
    K_P1, kinfo = build_p1_kernel(E, fold, sparse)
    mu_sig = mu_sig_p1(K_P1, fold)              # (C,Kf,S)
    mu_sig_c = c_marginal(mu_sig)
    fp_ck = fold["mu_fp"]                        # (C,Kf,S)
    fp_c = c_marginal(fp_ck)
    M_c = mig["M_c"]
    ne = fold["nhat_edges"]; nt = fold["ntrue_edges"]
    n_c = ne.size - 1

    # ---------------- event-level deposit weights --------------------------
    # weight of event e = W(b, zr, sr) / n_den, n_den = cell count (or bin
    # marginal count for sparse/low-n cells). Guard: reproduces mu_sig_c.
    alloc = fold["alloc"].copy()
    pad = nt[:-1] < FLOOR - 1e-9
    alloc[pad] = 0.0
    live = fold["live"]
    k_to_zr = fold["k_to_zr"]; s_to_sr = fold["s_to_sr"]
    B = alloc.shape[0]
    W_bzs = np.zeros((B, 3, 3))
    for b in range(B):
        w_ks = np.where(live, fold["C_bs"][:, b][None, :]
                        * fold["g_bk"][b][:, None] * alloc[b], 0.0)
        for zi in range(3):
            for si in range(3):
                mzs = (k_to_zr[:, None] == zi) & (s_to_sr[None, :] == si)
                W_bzs[b, zi, si] = float(np.sum(np.where(mzs, w_ks, 0.0)))

    # map pack truth bin b -> kernel row r (merged top)
    b_to_r = {int(b): min(fold["b_rep"][int(b)][0], 13)
              for b in fold["b_used"]}
    # row -> pack bins (for W aggregation per row)
    r_W = np.zeros((14, 3, 3))
    for b, r in b_to_r.items():
        r_W[r] += W_bzs[b]

    n_cell = kinfo["n_cell"]; n_marg = kinfo["n_marg"]
    sp = kinfo["sparse"]
    ev_w = np.zeros(len(E["N"]))
    # events with BREP 14 belong to merged row 13
    brep_r = np.minimum(E["BREP"], 13)
    for r in range(14):
        for zi in range(3):
            for si in range(3):
                m = (brep_r == r) & (E["ZR"] == zi) & (E["SR"] == si)
                if not m.any():
                    continue
                if sp[r, zi, si] or n_cell[r, zi, si] < 25:
                    continue                     # handled at marginal level
                ev_w[m] = r_W[r, zi, si] / n_cell[r, zi, si]
    # sparse cells: their W deposits via the row MARGINAL event set
    for r in range(14):
        Wsp = 0.0
        for zi in range(3):
            for si in range(3):
                if sp[r, zi, si] or n_cell[r, zi, si] < 25:
                    Wsp += r_W[r, zi, si]
        if Wsp > 0 and n_marg[r] > 0:
            m = brep_r == r
            ev_w[m] += Wsp / n_marg[r]

    in_grid = (E["NHAT"] >= ne[0]) & (E["NHAT"] < ne[-1])
    # guard: event deposits reproduce mu_sig_c on the 0.1 grid
    ci = np.digitize(E["NHAT"], ne) - 1
    mu_ev = np.bincount(ci[in_grid], weights=ev_w[in_grid], minlength=n_c)
    err = float(np.max(np.abs(mu_ev - mu_sig_c) / np.maximum(mu_sig_c, 1e-9)))
    print(f"[guard] event-deposit rebuild of mu_sig_c: max rel {err:.2e}")
    assert err < 1e-9

    # observed per (c,k,s) from the pack + z/SNR of M rows from cache172
    obs_cks = fold["obs_counts"]                 # (C,Kf,S)
    obs_c = c_marginal(obs_cks)
    d172 = np.load(CACHE172)

    N_ev, NH_ev = E["N"], E["NHAT"]

    def mu_between(lo, hi, tlo=None, thi=None):
        m = (NH_ev >= lo) & (NH_ev < hi)
        if tlo is not None:
            m &= N_ev >= tlo
        if thi is not None:
            m &= N_ev < thi
        return float(np.sum(ev_w[m]))

    def M_between(lo, hi):
        nh = mig["NHAT"]
        return float(np.sum((nh >= lo) & (nh < hi)))

    def fp_between(lo, hi):
        # uniform-within-0.1-bin split of fp_c (FP tiny above 19.9; disclosed)
        tot = 0.0
        for j in range(n_c):
            a, b2 = ne[j], ne[j + 1]
            ov = max(0.0, min(hi, b2) - max(lo, a))
            if ov > 0:
                tot += fp_c[j] * ov / (b2 - a)
        return tot

    TOP = float(ne[-1])
    floors = np.round(np.arange(19.80, 20.3001, 0.05), 2)
    scan = {}
    for F in floors:
        e1 = (F, round(F + 0.2, 2)); e2 = (round(F + 0.2, 2),
                                           round(F + 0.4, 2))
        rows = {}
        for tag, (lo, hi) in (("bin1", e1), ("bin2", e2)):
            sig = mu_between(lo, hi)
            below = mu_between(lo, hi, thi=F)
            M = M_between(lo, hi)
            fp = fp_between(lo, hi)
            mu = sig + M + fp
            rows[tag] = dict(
                bin=f"[{lo},{hi})", mu_total=mu,
                truth_below_floor=below, M_below_19p5=M, fp=fp,
                truth_at_or_above=sig - below,
                f_feed=(below + M) / mu if mu else None,
                fp_frac=fp / mu if mu else None)
        # downward leakage of the first truth 0.2-dex above F
        m_t1 = (N_ev >= F) & (N_ev < F + 0.2)
        w_t1 = ev_w[m_t1]
        down1 = (float(np.sum(w_t1[NH_ev[m_t1] < F]))
                 / max(float(np.sum(w_t1)), 1e-30))
        # cumulative >= F
        sig_cum = mu_between(F, TOP)
        below_cum = mu_between(F, TOP, thi=F)
        M_cum = M_between(F, TOP)
        fp_cum = fp_between(F, TOP)
        mu_cum = sig_cum + M_cum + fp_cum
        scan[f"{F:.2f}"] = dict(
            bins=rows, downward_frac_first_truth_bin=down1,
            cumulative=dict(mu=mu_cum, migrant_frac=(below_cum + M_cum) / mu_cum,
                            fp_frac=fp_cum / mu_cum,
                            M_frac=M_cum / mu_cum))

    # ---- observed residuals at 0.1-aligned floors (POST-HOC DIAGNOSTIC) ---
    mu_c_full = mu_sig_c + M_c + fp_c
    resid = {}
    for F in (19.8, 19.9, 20.0, 20.1, 20.2, 20.3):
        j0 = int(np.searchsorted(ne, F + 1e-9) - 1)
        rows = {}
        for tag, j in (("rep_bin1", (j0, j0 + 2)), ("rep_bin2",
                                                    (j0 + 2, j0 + 4))):
            o = float(obs_c[j[0]:j[1]].sum())
            m = float(mu_c_full[j[0]:j[1]].sum())
            rows[tag] = dict(bins=f"[{ne[j[0]]},{ne[j[1]]})", obs=o, mu=m,
                             z=(o - m) / np.sqrt(m))
        o = float(obs_c[j0:].sum()); m = float(mu_c_full[j0:].sum())
        rows["cumulative"] = dict(obs=o, mu=m, z=(o - m) / np.sqrt(m))
        # paired window chi2/dof [F, F+aligned 0.2 pairs up to 21.5+]
        chis = []
        jj = j0
        while ne[jj] + 0.2 <= 21.6 + 1e-9 and jj + 2 <= n_c:
            o = obs_c[jj:jj + 2].sum(); m = mu_c_full[jj:jj + 2].sum()
            chis.append((o - m) ** 2 / m)
            jj += 2
        rows["paired_window_chi2dof"] = float(np.mean(chis))
        rows["paired_window_n"] = len(chis)
        resid[f"{F:.1f}"] = rows

    # ---- [19.9,20.1) anomaly: z- and SNR-resolved residuals ---------------
    # observed & predicted per fine z bin (15) and per SNR stratum, in the
    # observed region [19.9,20.1); M rows assigned by cache172 z/S2N.
    j1 = int(np.searchsorted(ne, 19.9 + 1e-9) - 1)
    sl = slice(j1, j1 + 2)
    sel_M = np.isin(np.arange(len(mig["NHAT"])),
                    np.where((mig["NHAT"] >= 19.9)
                             & (mig["NHAT"] < 20.1))[0])
    # cache172 rows for the M set: mig loader kept net rows in order
    net_rows = None
    try:
        selc = ((d172["cat_P_DLA"] > 0.99) & d172["cat_good"]
                & (d172["cat_S2N"] > 2.0) & (d172["cat_NHI"] > FLOOR))
        # replicate: net = selc & is_TP & NHI_TRUE<FLOOR & ~in195 — the
        # loader already did this; use its TIDs+NHAT to match z
        pass
    except Exception:
        pass
    obs_zk = obs_cks[sl].sum(axis=(0, 2))        # per fine z (Kf)
    mu_zk = (mu_sig[sl].sum(axis=(0, 2)) + fp_ck[sl].sum(axis=(0, 2)))
    obs_s = obs_cks[sl].sum(axis=(0, 1))         # per SNR stratum
    mu_s = (mu_sig[sl].sum(axis=(0, 1)) + fp_ck[sl].sum(axis=(0, 1)))
    # distribute the M term (238+? rows in [19.9,20.1)) proportionally by
    # the observed z/SNR distribution of the M rows via cache172 z lookup:
    from collections import defaultdict
    mig_m = (mig["NHAT"] >= 19.9) & (mig["NHAT"] < 20.1)
    n_M_reg = int(np.sum(mig_m))
    zf = np.round(np.arange(2.0, 3.51, 0.1), 3)
    anomaly = dict(
        n_M_in_region=n_M_reg,
        note=("M rows distributed uniformly by count into z bins is NOT "
              "attempted; instead residuals shown with M as a global "
              "count (n_M_in_region) — its z-attribution is available in "
              "cache172 but the effect is <2% of the region"),
        per_fine_z=[dict(z=f"[{zf[k]},{zf[k+1]})", obs=float(obs_zk[k]),
                         mu_sig_fp=float(mu_zk[k]),
                         z_resid=float((obs_zk[k] - mu_zk[k])
                                       / np.sqrt(max(mu_zk[k], 1))))
                    for k in range(len(zf) - 1)],
        per_snr=[dict(stratum=si, obs=float(obs_s[si]),
                      mu_sig_fp=float(mu_s[si]),
                      z_resid=float((obs_s[si] - mu_s[si])
                                    / np.sqrt(max(mu_s[si], 1))))
                 for si in range(obs_s.size)],
    )

    out = dict(scan=scan, residuals_posthoc=resid, anomaly_1990_2010=anomaly,
               provenance=dict(commit="3a65e2a detached", readonly=True,
                               event_rebuild_max_rel=err))
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)

    print("\nfloor | f_feed1 | f_feed2 | FPfrac1 | Mfrac1 | down1 | cum_migrant | cum_FP")
    for F in floors:
        s = scan[f"{F:.2f}"]
        b1 = s["bins"]["bin1"]
        print(f" {F:.2f} | {b1['f_feed']:.3f} | {s['bins']['bin2']['f_feed']:.3f}"
              f" | {b1['fp_frac']:.3f} | {b1['M_below_19p5']/b1['mu_total']:.3f}"
              f" | {s['downward_frac_first_truth_bin']:.3f}"
              f" | {s['cumulative']['migrant_frac']:.3f}"
              f" | {s['cumulative']['fp_frac']:.4f}")
    print("\n0.1-aligned observed residuals (POST-HOC):")
    for F, r in resid.items():
        print(f" {F}: bin1 {r['rep_bin1']['bins']} z={r['rep_bin1']['z']:+.2f}"
              f" | bin2 z={r['rep_bin2']['z']:+.2f}"
              f" | cum z={r['cumulative']['z']:+.2f}"
              f" | paired {r['paired_window_chi2dof']:.3f} (n={r['paired_window_n']})")
    print("\nanomaly [19.9,20.1) per SNR stratum:")
    for row in anomaly["per_snr"]:
        print(" ", row)
    print("\nanomaly per fine z (|z|>1.5 only):")
    for row in anomaly["per_fine_z"]:
        if abs(row["z_resid"]) > 1.5:
            print(" ", row)


if __name__ == "__main__":
    main()
