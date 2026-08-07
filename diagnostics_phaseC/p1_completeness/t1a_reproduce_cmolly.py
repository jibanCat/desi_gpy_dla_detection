#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tier 1a — event-level reproduction of the DEPLOYED C_molly (rulings §7).

Rebuilds the deployed completeness counts from the immutable production
catalogue + truth through THE committed chain (`HBIConfig` →
`load_and_cut_catalog` → `regenerate_molly_counts`, exactly as
`ff_fp_estimator.build_molly_counts_cache` wires it, with the pack's own
molly-172 TSV), and compares against the frozen Phase-B pack's
`molly_n_det` / `molly_n_tot` under the FROZEN tolerance
(`docs/P1_STOPPING_RULE.md` criterion 1: integer equality per cell, or a
per-cell diagnosis).

Also caches the event-level intermediates (catalogue rows + truth rows +
match flags) for Tiers 1b–1d, so the expensive load runs once. Natural
production data only — the injection holdout cannot be touched here by
construction (the production catalogue predates all injection arms).

Run:  gpdla-hbi python t1a_reproduce_cmolly.py
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)

CACHE = ("/scratch/cavestru_root/cavestru0/mfho/phaseC_resp/"
         "p1_completeness_cache.npz")
PACK = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phaseB_packs/"
        "modelA_pack_2lpt0_winlya_only_pad19p0_molly172_bw0p2.npz")
MOLLY_TSV_172 = ("/scratch/cavestru_root/cavestru0/mfho/"
                 "gl_prod_2lpt0_v1_20260526/figures_molly_nhi172/"
                 "molly_matrix.tsv")


MOLLY_TSV_195 = ("/scratch/cavestru_root/cavestru0/mfho/"
                 "gl_prod_2lpt0_v1_20260526/figures_molly_nhi195/lya_only/"
                 "molly_matrix.tsv")


def _chain(tsv):
    """One event-level molly regeneration chain (the committed wiring)."""
    from CDDF_analysis.hbi import ab_loa0_fp_baseline as AB
    from CDDF_analysis.hbi.cddf_catalog_hbi import (
        HBIConfig, load_molly_matrix, regenerate_molly_counts,
        load_and_cut_catalog, _build_qso_lookup)
    cfg = HBIConfig(
        catalog_dir=AB.DEF_CAT, truth_path=AB.DEF_TRUTH,
        bal_cat_path=AB.DEF_BAL, molly_tsv=tsv,
        out_dir=_HERE, mockdir=os.path.dirname(AB.DEF_TRUTH),
        lam_rf_min=1025.0, no_bal=True)
    mm = load_molly_matrix(tsv)
    truth_floor = float(mm.nhi_edges[0])
    qso_lookup = _build_qso_lookup(cfg)
    cat_cut, truth_cut, is_TP, good_mask, meta = load_and_cut_catalog(
        cfg, truth_nhi_floor=truth_floor, qso_lookup=qso_lookup,
        host_truth_floor=min(19.0, truth_floor))
    mm = regenerate_molly_counts(mm, cat_cut, is_TP, truth_cut, good_mask,
                                 cfg)
    return mm, cat_cut, truth_cut, is_TP, good_mask, meta


def main():
    t0 = time.time()
    # THE DEPLOYED ARTIFACT IS A SPLICE (extract_pack.py:556-589): cells
    # >= 19.5 from the CANONICAL nhi195 chain; cells < 19.5 from the
    # floor-17.2 chain. Reproduce BOTH chains event-level and splice.
    mm195, cat195, tr195, tp195, good195, meta = _chain(MOLLY_TSV_195)
    mm172, cat172, tr172, tp172, good172, _ = _chain(MOLLY_TSV_172)
    e172 = np.asarray(mm172.nhi_edges, float)
    e195 = np.asarray(mm195.nhi_edges, float)
    n_sub = len(e172) - len(e195)
    det_r = np.concatenate(
        [np.asarray(mm172.cmp_nfound, float)[:, :n_sub],
         np.asarray(mm195.cmp_nfound, float)], axis=1)
    tot_r = np.concatenate(
        [np.asarray(mm172.cmp_nfid, float)[:, :n_sub],
         np.asarray(mm195.cmp_nfid, float)], axis=1)

    pk = np.load(PACK)
    det_p = np.asarray(pk["molly_n_det"], float)
    tot_p = np.asarray(pk["molly_n_tot"], float)
    same_edges = (np.allclose(pk["molly_snr_edges"], mm195.snr_edges)
                  and np.allclose(pk["molly_nhi_edges"], e172))
    mm = mm195
    cat_cut, truth_cut, is_TP, good_mask = cat195, tr195, tp195, good195
    d_det = det_r - det_p
    d_tot = tot_r - tot_p
    bad = [
        {"snr_i": int(i), "nhi_j": int(j),
         "pack_det": det_p[i, j], "recon_det": det_r[i, j],
         "pack_tot": tot_p[i, j], "recon_tot": tot_r[i, j]}
        for i in range(det_p.shape[0]) for j in range(det_p.shape[1])
        if d_det[i, j] != 0 or d_tot[i, j] != 0]
    out = {
        "schema": "p1_t1a_cmolly_reproduction/v1",
        "date": time.strftime("%Y-%m-%d"),
        "edges_match": bool(same_edges),
        "n_cells": int(det_p.size),
        "n_cells_nonreproducing": len(bad),
        "max_abs_ddet": float(np.max(np.abs(d_det))),
        "max_abs_dtot": float(np.max(np.abs(d_tot))),
        "tsv_ratio_guard_max_c_diff": float(mm._max_c_diff),
        "nonreproducing_cells": bad,
        "deployed_C": (det_p / np.maximum(tot_p, 1)).tolist(),
        "catalog_meta": {k: v for k, v in meta.items()
                         if isinstance(v, (int, float, str, bool))},
        "REPRODUCED": bool(same_edges and not bad),
        "wall_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(_HERE, "t1a_cmolly.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in
                      ("edges_match", "n_cells_nonreproducing",
                       "max_abs_ddet", "max_abs_dtot",
                       "tsv_ratio_guard_max_c_diff", "REPRODUCED",
                       "wall_s")}, indent=1))

    # ---- event-level cache for Tiers 1b-1d (one expensive load) ----
    def col(t, name, dt=float):
        return np.asarray(t[name], dt) if name in t.colnames else None
    save = dict(
        cat_TARGETID=np.asarray(cat_cut["TARGETID"], np.int64),
        cat_Z_DLA=col(cat_cut, "Z_DLA"),
        cat_NHI=col(cat_cut, "NHI"),
        cat_P_DLA=col(cat_cut, "P_DLA"),
        cat_S2N=col(cat_cut, "S2N_RED"),
        cat_NHI_TRUE=col(cat_cut, "NHI_TRUE"),
        cat_Z_TRUE=col(cat_cut, "Z_TRUE"),
        cat_is_TP=np.asarray(is_TP, bool),
        cat_good=np.asarray(good_mask, bool),
        tr_TARGETID=np.asarray(truth_cut["TARGETID"], np.int64),
        tr_NHI=col(truth_cut, "NHI"),
        tr_Z=(col(truth_cut, "Z_TRUTH") if "Z_TRUTH" in truth_cut.colnames
              else col(truth_cut, "Z_DLA")),
        tr_SNR=col(truth_cut, "SNR"),
    )
    if "DLAFLAG" in cat_cut.colnames:
        save["cat_DLAFLAG"] = np.asarray(cat_cut["DLAFLAG"], float)
    np.savez(CACHE, **{k: v for k, v in save.items() if v is not None})
    print("cached event-level intermediates ->", CACHE)


if __name__ == "__main__":
    main()
