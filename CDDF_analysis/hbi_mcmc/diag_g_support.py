#!/usr/bin/env python
"""diag_g_support.py — DIAGNOSTIC (2026-08-20, finding N1): the z-shape surface
g(N,z) of the frozen 2LPT-0 calibration was built by ``measure_c_nz`` with a
TRUTH denominator that carries NO SNR cut, while (i) its TP numerator is the
op-cut (S2N_RED > snr_min & P_DLA > p_dla_min & good_mask) and (ii) the fold's
own truth support (``build_truth_counts``) applies S2N_RED > snr_min. The
excess truth is z-dependent, and g is normalised per N row, so the mismatch
appears as a spurious z-tilt of g that an all-z validation cannot see.

This script MEASURES that: it reloads the 2LPT-0 bundle through the committed
loader, (a) reproduces the deployed g (bit-identity check against a pack),
(b) rebuilds g with the truth table restricted to the SAME support as the
fold's truth_counts (S2N_RED > snr_min), (c) reports the denominator excess
per z, and (d) re-runs the per-z closure of the fold at mock truth with the
rebuilt g, for every pack given. Nothing is written into any pack; the rebuilt
surface is saved to the --out-dir as a diagnostic artifact only.

Env: gpdla-hbi (mock-only inputs; refuses real packs for the closure).
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import jax.numpy as jnp

from CDDF_analysis.hbi_mcmc.pack import load_pack
from CDDF_analysis.hbi_mcmc.extract_pack import load_mock_bundle, build_g_block
from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors
from CDDF_analysis.hbi_mcmc.perz_fold_closure import fold_at_truth, closure_table
from CDDF_analysis.hbi.cddf_catalog_hbi import build_cnz_resolved, _fine_z_grid
from CDDF_analysis.hbi.znz_kernel import measure_c_nz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", nargs="+", required=True,
                    help="mock packs to re-close (first one also used for the "
                         "bit-identity check of the deployed g rows)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window", default="lya_only")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    bundle = load_mock_bundle("2lpt0", a.out_dir, window=a.window)
    cfg, mm = bundle["cfg"], bundle["mm"]
    zf = _fine_z_grid(cfg)
    g_dep, occ_dep = build_g_block(bundle)                    # deployed recipe
    t = bundle["truth_cut"]
    s2n_t = np.asarray(t["S2N_RED"], float)
    keep = s2n_t > cfg.snr_min                                 # build_truth_counts's cut
    t_cons = t[keep]
    meas_dep = measure_c_nz(bundle["cat_cut"], t, cfg, mm, zf,
                            good_mask=bundle["good_mask"])
    meas_cons = measure_c_nz(bundle["cat_cut"], t_cons, cfg, mm, zf,
                             good_mask=bundle["good_mask"])
    cnz_cons = build_cnz_resolved(cfg, bundle["cat_cut"], t_cons,
                                  bundle["good_mask"], mm)
    g_cons = np.asarray(cnz_cons.g_grid, float)
    n_true_dep = np.asarray(meas_dep["n_true"]); n_true_cons = np.asarray(meas_cons["n_true"])
    n_rec = np.asarray(meas_dep["n_rec"])
    assert np.array_equal(n_rec, np.asarray(meas_cons["n_rec"])), "numerator must not move"

    rep = dict(role=("DIAGNOSTIC: g(N,z) denominator-support audit; nothing "
                     "applied, no pack modified"),
               snr_min=float(cfg.snr_min), n_truth_cut=int(len(t)),
               n_truth_cut_snr_gt_min=int(keep.sum()),
               molly_nhi_edges=[float(x) for x in mm.nhi_edges],
               zf_edges=[float(x) for x in zf],
               per_row={})
    me = np.asarray(mm.nhi_edges, float)
    for j in range(len(me) - 1):
        with np.errstate(divide="ignore", invalid="ignore"):
            rep["per_row"][f"[{me[j]},{me[j+1]})"] = dict(
                n_true_deployed=[int(x) for x in n_true_dep[j]],
                n_true_consistent=[int(x) for x in n_true_cons[j]],
                n_rec=[int(x) for x in n_rec[j]],
                excess_ratio=[round(float(x), 4) for x in
                              np.where(n_true_cons[j] > 0,
                                       n_true_dep[j] / np.maximum(n_true_cons[j], 1), np.nan)],
                g_deployed=[round(float(x), 4) for x in g_dep[j]],
                g_consistent=[round(float(x), 4) for x in g_cons[j]],
                raw_C_consistent=[round(float(x), 4) for x in
                                  np.where(n_true_cons[j] > 0,
                                           n_rec[j] / np.maximum(n_true_cons[j], 1), np.nan)])
    # bit-identity of the deployed recipe against the first pack's >=19.5 rows
    pk0 = load_pack(a.packs[0])
    pme = np.asarray(pk0.molly_nhi_edges, float)
    off = int(np.flatnonzero(np.isclose(pme, me[0]))[0])
    rep["deployed_g_bit_identical_to_pack_rows"] = bool(
        np.allclose(np.asarray(pk0.g_grid)[off:], g_dep, rtol=0, atol=1e-12))
    rep["truth_counts_vs_consistent_denominator_per_z"] = {}
    # the fold's truth support vs the rebuilt denominator (should agree row-wise)
    nt = np.asarray(pk0.ntrue_edges, float); tc = np.asarray(pk0.truth_counts, float)
    agg = np.zeros((len(me) - 1, tc.shape[1]))
    for b in range(len(nt) - 1):
        c = 0.5 * (nt[b] + nt[b + 1])
        if c < me[0]:
            continue
        j = int(np.clip(np.searchsorted(me, c, side="right") - 1, 0, len(me) - 2))
        agg[j] += tc[b]
    for j in range(len(me) - 1):
        rep["truth_counts_vs_consistent_denominator_per_z"][f"[{me[j]},{me[j+1]})"] = \
            [round(float(x), 4) for x in np.where(agg[j] > 0, n_true_cons[j] / np.maximum(agg[j], 1), np.nan)]
    np.savez(os.path.join(a.out_dir, "g_consistent_diag.npz"), g_deployed=g_dep,
             g_consistent=g_cons, n_true_deployed=n_true_dep,
             n_true_consistent=n_true_cons, n_rec=n_rec, nhi_edges=me, zf_edges=zf)

    # per-z closure at truth with the rebuilt g substituted on the >=19.5 rows
    rep["closure"] = {}
    for path in a.packs:
        pk = load_pack(path)
        if pk.truth_counts is None:
            raise SystemExit("closure is MOCK-ONLY")
        consts, Mg = build_cc_tensors(pk)
        pme = np.asarray(pk.molly_nhi_edges, float)
        off = int(np.flatnonzero(np.isclose(pme, me[0]))[0])
        g_new = np.asarray(pk.g_grid, float).copy()
        g_new[off:] = g_cons                                   # sub-floor rows untouched
        g_bk_new = jnp.asarray(g_new[np.asarray(consts.b_to_cell), :])
        obs = np.asarray(pk.counts, float)
        ne = np.asarray(pk.nhat_edges, float)
        masks = {"nhat_ge20.3": ne[:-1] >= 20.3 - 1e-9,
                 "nhat_ge20.0": ne[:-1] >= 20.0 - 1e-9,
                 "nhat_all": np.ones(len(ne) - 1, bool)}
        fam = os.path.basename(path)
        rep["closure"][fam] = {}
        for gname, gfac in (("deployed_g", consts.g_bk), ("g_consistent", g_bk_new),
                            ("g_equals_1", jnp.ones_like(consts.g_bk))):
            tp, fp = fold_at_truth(pk, consts, Mg, gfac)
            rep["closure"][fam][gname] = {mk: closure_table(pk, tp + fp, obs, mm_, f"{gname}/{mk}")
                                          for mk, mm_ in masks.items()}
        for gname in ("deployed_g", "g_consistent", "g_equals_1"):
            r = rep["closure"][fam][gname]["nhat_ge20.3"]
            print(f"{fam} {gname:12s} ge20.3 blocks {r['ratio_coarse']} "
                  f"bins {r['ratio_paper1_bins']} allz {r['ratio_allz']}")
    json.dump(rep, open(os.path.join(a.out_dir, "diag_g_support.json"), "w"), indent=1)
    print("bit-identity deployed g vs pack:", rep["deployed_g_bit_identical_to_pack_rows"])
    print("n_truth_cut", rep["n_truth_cut"], "-> S2N>min", rep["n_truth_cut_snr_gt_min"])
    return rep


if __name__ == "__main__":
    main()
