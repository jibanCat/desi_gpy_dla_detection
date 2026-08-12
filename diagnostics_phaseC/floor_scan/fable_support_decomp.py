#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fable 2026-08-12 — reporting-floor vs operator-support decomposition.

READ-ONLY recomputation from the frozen refold machinery at
prov/p1-refold-2026-08-08 (worktree, detached).  Nothing is written into
the repo; no gate is (re-)evaluated; no observed count is compared to any
threshold.  Every committed guard in the fold library stays live:
  * deployed-fold bit-level rebuild guard (<=1e-8),
  * kernel per-cell integer identity vs the loaded artifact,
  * migration group totals == committed 4088/144/0,
  * contrib decomposition == mu_sig_c (<=1e-8),
  * mu_total per bin == frozen closure record (checked here, report-only).

Outputs (scratchpad JSON):
  1. per-observed-bin composition: truth-bin contributions (0.2 dex),
     M_<19.5, FP;
  2. f_feed,1 / f_feed,2 for reporting floors 20.0, 20.1, 20.3 (and the
     19.7 status quo), with the straddling truth bin [19.9,20.1) split at
     event level (frozen kernel event set; sparse-cell share disclosed);
  3. downward migration (truth above floor -> N-hat below floor);
  4. latent-support truncation sensitivity of the lowest two reported
     bins as the latent floor is moved;
  5. FP / upward-migrant / in-domain decomposition of cumulative observed
     counts above 20.0 and 20.3;
  6. granularity parity diagnostic of the frozen residuals.
"""
import json
import os
import sys

import numpy as np

WT = os.environ.get("FLOOR_SCAN_REPO", os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
HERE = os.path.join(WT, "diagnostics_phaseC", "p1_completeness")
sys.path.insert(0, HERE)
OUT = os.environ.get("FLOOR_SCAN_OUT", "fable_support_decomp.json")

from p1_refold_fold import (  # noqa: E402
    FLOOR, build_fold, build_p1_kernel, c_marginal, load_kernel_events,
    load_migration, mu_sig_p1)

CLOSURE = json.load(open(os.path.join(HERE, "p1_refold_closure.json")))


def main():
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from p1_refold_fold import PACK

    pk = load_pack(PACK)
    fold = build_fold(pk)
    E, truth, sparse, art, cache = load_kernel_events()
    mig = load_migration(fold["nhat_edges"])
    K_P1, kinfo = build_p1_kernel(E, fold, sparse)
    mu_sig = mu_sig_p1(K_P1, fold)
    mu_sig_c = c_marginal(mu_sig)
    fp_c = c_marginal(fold["mu_fp"])
    M_c = mig["M_c"]
    mu_c = mu_sig_c + M_c + fp_c

    ne = fold["nhat_edges"]          # observed 0.1-dex edges
    nt = fold["ntrue_edges"]         # pack truth edges
    n_c = ne.size - 1

    # ---- guard: frozen closure record reproduced -------------------------
    mu_ref = np.asarray(CLOSURE["per_bin"]["mu_total"], float)
    rel = float(np.max(np.abs(mu_c - mu_ref) / np.maximum(mu_ref, 1e-12)))
    print(f"[guard] mu_total vs frozen closure: max rel diff = {rel:.3e}")
    assert rel < 1e-6, "recomputation does not reproduce the frozen record"

    # ---- per-truth-bin -> per-observed-bin contributions -----------------
    alloc = fold["alloc"].copy()
    pad = fold["ntrue_edges"][:-1] < FLOOR - 1e-9
    alloc[pad] = 0.0
    live = fold["live"]
    B = alloc.shape[0]
    contrib_bc = np.zeros((B, n_c))
    # and per response-cell weights W(b, zr, sr) for the event-level split
    W_bzs = np.zeros((B, 3, 3))
    k_to_zr = fold["k_to_zr"]; s_to_sr = fold["s_to_sr"]
    for b in range(B):
        w_ks = (fold["C_bs"][:, b][None, :] * fold["g_bk"][b][:, None]
                * alloc[b])                                    # (Kf,S)
        w_ks = np.where(live, w_ks, 0.0)
        contrib_bc[b] = np.einsum("skc,ks->c", K_P1[:, :, :, b], w_ks)
        for zi in range(3):
            for si in range(3):
                mzs = (k_to_zr[:, None] == zi) & (s_to_sr[None, :] == si)
                W_bzs[b, zi, si] = float(np.sum(np.where(mzs, w_ks, 0.0)))
    err = np.max(np.abs(contrib_bc.sum(axis=0) - mu_sig_c))
    print(f"[guard] contrib_bc sum vs mu_sig_c: {err:.3e}")
    assert err < 1e-8

    # ---- event-level split of a truth bin at an interior N cut ----------
    # For pack truth bin b (report row r), fraction of each landing-row
    # entry attributable to events with N < ncut, per (zr,sr) cell; sparse
    # cells use the marginal-row event split (frozen inheritance rule).
    ci_ev = np.digitize(E["NHAT"], ne) - 1
    in_grid_ev = (ci_ev >= 0) & (ci_ev < n_c) & (E["NHAT"] < ne[-1])

    def split_contrib(b, ncut):
        """contrib of pack truth bin b to each observed bin, split into
        (below ncut, at/above ncut) by the frozen kernel events."""
        r = min(fold["b_rep"][int(b)][0], 13)
        below = np.zeros(n_c); above = np.zeros(n_c)
        sparse_mass = 0.0
        for zi in range(3):
            for si in range(3):
                Wc = W_bzs[b, zi, si]
                if Wc == 0.0:
                    continue
                use_marginal = (kinfo["sparse"][r, zi, si]
                                or kinfo["n_cell"][r, zi, si] < 25)
                if use_marginal:
                    m = (E["BREP"] == r) & in_grid_ev
                    n_den = kinfo["n_marg"][r]
                    sparse_mass += Wc
                else:
                    m = ((E["BREP"] == r) & (E["ZR"] == zi)
                         & (E["SR"] == si) & in_grid_ev)
                    n_den = kinfo["n_cell"][r, zi, si]
                if n_den == 0:
                    continue
                mb = m & (E["N"] < ncut)
                ma = m & (E["N"] >= ncut)
                below += np.bincount(ci_ev[mb], minlength=n_c) / n_den * Wc
                above += np.bincount(ci_ev[ma], minlength=n_c) / n_den * Wc
        return below, above, sparse_mass

    # which pack truth bin straddles 20.0?
    b_straddle = [b for b in range(B)
                  if nt[b] < 20.0 - 1e-9 < 20.0 + 1e-9 < nt[b + 1]
                  and not pad[b]]
    assert len(b_straddle) == 1, b_straddle
    bS = b_straddle[0]
    bel20, abv20, spl_mass = split_contrib(bS, 20.0)
    tot_S = contrib_bc[bS]
    chk = np.max(np.abs(bel20 + abv20 - tot_S))
    print(f"[guard] event split of bin [{nt[bS]},{nt[bS+1]}) closes: "
          f"{chk:.3e}; sparse-inherited weight share = "
          f"{spl_mass / max(W_bzs[bS].sum(), 1e-30):.4f}")

    def bin_idx(x):
        return int(np.searchsorted(ne, x + 1e-9) - 1)

    # contributions from truth below a cut, per observed bin
    def truth_below(ncut):
        v = np.zeros(n_c)
        for b in range(B):
            if pad[b]:
                continue
            if nt[b + 1] <= ncut + 1e-9:
                v += contrib_bc[b]
            elif nt[b] < ncut - 1e-9 < nt[b + 1]:
                bel, _, _ = split_contrib(b, ncut)
                v += bel
        return v

    results = {"floors": {}}
    for F in (19.7, 20.0, 20.1, 20.3):
        tb = truth_below(F)
        rows = {}
        for j, lab in ((bin_idx(F), "lowest"), (bin_idx(F) + 1, "second")):
            c0, c1 = ne[j], ne[j + 1]
            comp = dict(
                obs_bin=f"[{c0},{c1})",
                mu_total=float(mu_c[j]),
                mu_from_truth_below_floor=float(tb[j]),
                mu_from_M_below_19p5=float(M_c[j]),
                mu_fp=float(fp_c[j]),
                mu_from_truth_at_or_above_floor=float(
                    mu_sig_c[j] - tb[j]),
            )
            comp["f_feed"] = float((tb[j] + M_c[j]) / mu_c[j])
            comp["f_feed_excl_fp_denom"] = float(
                (tb[j] + M_c[j]) / (mu_c[j] - fp_c[j]))
            comp["fp_frac"] = float(fp_c[j] / mu_c[j])
            rows[lab] = comp
        # cumulative decomposition above F
        sel = np.arange(n_c) >= bin_idx(F)
        cum = dict(
            mu_total=float(mu_c[sel].sum()),
            truth_below_floor=float(tb[sel].sum()),
            M_below_19p5=float(M_c[sel].sum()),
            fp=float(fp_c[sel].sum()),
            truth_at_or_above_floor=float((mu_sig_c - tb)[sel].sum()),
        )
        cum["migrant_frac_of_total"] = float(
            (cum["truth_below_floor"] + cum["M_below_19p5"])
            / cum["mu_total"])
        cum["fp_frac_of_total"] = float(cum["fp"] / cum["mu_total"])
        results["floors"][str(F)] = dict(lowest_two=rows, cumulative=cum)

    # ---- downward migration ---------------------------------------------
    # per truth bin: fraction of its landed (in-grid) mass below each floor
    down = {}
    for F in (20.0, 20.3):
        j0 = bin_idx(F)
        per = {}
        for b in range(B):
            if pad[b] or nt[b] < F - 1e-9:
                continue
            tot = contrib_bc[b].sum()
            if tot <= 0:
                continue
            lost = contrib_bc[b][:j0].sum()
            per[f"[{nt[b]},{nt[b+1]})"] = dict(
                frac_below_floor=float(lost / tot),
                mu_below_floor=float(lost), mu_total_landed=float(tot))
        # event-level for the straddling piece at 20.0 (truth in [20.0,20.1))
        if F == 20.0:
            _, abv, _ = split_contrib(bS, 20.0)
            tot = abv.sum()
            if tot > 0:
                per[f"[20.0,{nt[bS+1]}) (event-split)"] = dict(
                    frac_below_floor=float(abv[:j0].sum() / tot),
                    mu_below_floor=float(abv[:j0].sum()),
                    mu_total_landed=float(tot))
        down[str(F)] = per
    results["downward_migration"] = down

    # raw-event counting cross-check (unweighted kernel events)
    raw = {}
    for F in (20.0, 20.3):
        for lo, hi, tag in ((F, F + 0.2, "first_0.2dex"),
                            (F + 0.2, F + 0.4, "second_0.2dex")):
            m = (E["N"] >= lo) & (E["N"] < hi)
            if m.sum():
                raw[f"truth[{lo:.1f},{hi:.1f})_land_below_{F}"] = dict(
                    frac=float(np.mean(E["NHAT"][m] < F)),
                    n_events=int(m.sum()))
            m2 = (E["N"] < F) & (E["NHAT"] >= F)
            raw[f"events_true_below_{F}_landing_at_or_above_{F}"] = dict(
                n=int(np.sum(m2)),
                frac_of_below_floor_events=float(
                    np.mean(E["NHAT"][E["N"] < F] >= F))
                if np.sum(E["N"] < F) else None)
    results["raw_event_crosscheck"] = raw

    # ---- latent-support truncation sensitivity ---------------------------
    # variants: drop M; truncate truth support at successive edges
    variants = {}
    watch = [bin_idx(20.0), bin_idx(20.0) + 1, bin_idx(20.3),
             bin_idx(20.3) + 1]
    watch_lab = [f"[{ne[j]},{ne[j+1]})" for j in watch]

    def record(name, mu_var):
        variants[name] = {
            lab: dict(mu=float(mu_var[j]),
                      delta_vs_full=float(mu_var[j] - mu_c[j]),
                      rel=float((mu_var[j] - mu_c[j]) / mu_c[j]))
            for lab, j in zip(watch_lab, watch)}

    record("full_support_19p5_plus_M", mu_c)
    record("drop_M_only", mu_sig_c + fp_c)
    for cut in (19.7, 19.9, 20.1, 20.3):
        v = mu_c.copy()
        v -= M_c                              # truncation removes M too
        for b in range(B):
            if pad[b]:
                continue
            if nt[b + 1] <= cut + 1e-9:
                v -= contrib_bc[b]
        record(f"truncate_truth_below_{cut}", v)
    # exact truncation at 20.0 via the event split
    v = mu_c - M_c
    for b in range(B):
        if pad[b]:
            continue
        if nt[b + 1] <= 20.0 + 1e-9:
            v -= contrib_bc[b]
    v -= bel20
    record("truncate_truth_below_20.0_eventsplit", v)
    results["latent_truncation_sensitivity"] = dict(
        watched_bins=watch_lab, variants=variants)

    # ---- granularity parity diagnostic (frozen residuals, no gate) ------
    obs = np.asarray(CLOSURE["per_bin"]["observed"], float)
    z = (obs - mu_ref) / np.sqrt(mu_ref)
    par = {}
    for j in range(n_c):
        if 19.7 - 1e-9 <= ne[j] and ne[j + 1] <= 21.6 + 1e-9:
            off = int(round((ne[j] - 19.5) / 0.1))
            par[f"[{ne[j]},{ne[j+1]})"] = dict(z=float(z[j]),
                                               parity=off % 2)
    ev_z = [v["z"] for v in par.values() if v["parity"] == 0]
    od_z = [v["z"] for v in par.values() if v["parity"] == 1]
    results["granularity_parity"] = dict(
        note=("parity 0 = observed bin starts on a 0.2-dex truth edge "
              "(19.5+0.2k); window [19.7,21.6] frozen residuals"),
        mean_z_parity0=float(np.mean(ev_z)),
        mean_z_parity1=float(np.mean(od_z)),
        per_bin=par)

    # ---- bookkeeping ------------------------------------------------------
    results["provenance"] = dict(
        worktree_commit="3a65e2a (prov/p1-refold-2026-08-08, detached)",
        readonly=True,
        guards="rebuild<=1e-8; K integer identity; migration groups exact; "
               "contrib==mu_sig; mu_total==frozen closure (report above)",
        mu_total_vs_frozen_rel=rel,
        sparse_share_straddle_bin=float(
            spl_mass / max(W_bzs[bS].sum(), 1e-30)),
        n_kernel_events=int(len(E["N"])),
    )
    json.dump(results, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")

    # ---- console summary --------------------------------------------------
    for F, r in results["floors"].items():
        print(f"\n== floor {F} ==")
        for lab in ("lowest", "second"):
            c = r["lowest_two"][lab]
            print(f"  {lab} {c['obs_bin']}: mu {c['mu_total']:.1f} | "
                  f"f_feed {c['f_feed']:.3f} (belowN {c['mu_from_truth_below_floor']:.1f} "
                  f"+ M {c['mu_from_M_below_19p5']:.1f}) | FP {c['fp_frac']:.3f}")
        cu = r["cumulative"]
        print(f"  cumulative >= {F}: migrant_frac {cu['migrant_frac_of_total']:.4f}, "
              f"fp_frac {cu['fp_frac_of_total']:.4f}")


if __name__ == "__main__":
    main()
