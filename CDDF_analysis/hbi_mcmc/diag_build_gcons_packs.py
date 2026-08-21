#!/usr/bin/env python
"""diag_build_gcons_packs.py — DIAGNOSTIC pack copies with g(N,z) rebuilt on the
consistent truth support (2026-08-20 finding; see diag_g_support.py).

For each input MOCK pack: copy every array byte-for-byte except ``g_grid`` and
``g_occupancy``, which are rebuilt from the frozen 2LPT-0 bundles through the
committed builders (``build_cnz_resolved`` / ``measure_c_nz``) with the truth
table restricted to S2N_RED > snr_min — the same cut ``build_truth_counts``
applies to the fold's truth support and the op cut applies to the numerator.
The sub-floor rows (molly172 splice) are rebuilt the same way from the
floor-17.2 bundle, exactly as build_frozen_calibration splices them.

Output names carry ``DIAGPACK_gcons`` and a provenance sidecar marking them
NOT ADOPTED. Mock packs only (refuses packs without truth_counts); these
copies are for mock posterior validation of the candidate fix, nothing else.
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import os

import numpy as np

from CDDF_analysis.hbi_mcmc.pack import load_pack, save_pack
from CDDF_analysis.hbi_mcmc.extract_pack import (load_mock_bundle, window_spec,
                                                 DEF_WINDOW)
from CDDF_analysis.hbi.cddf_catalog_hbi import build_cnz_resolved, _fine_z_grid
from CDDF_analysis.hbi.znz_kernel import measure_c_nz


def rebuild_g(bundle):
    cfg, mm = bundle["cfg"], bundle["mm"]
    t = bundle["truth_cut"]
    keep = np.asarray(t["S2N_RED"], float) > cfg.snr_min
    t_cons = t[keep]
    cnz = build_cnz_resolved(cfg, bundle["cat_cut"], t_cons, bundle["good_mask"], mm)
    meas = measure_c_nz(bundle["cat_cut"], t_cons, cfg, mm, _fine_z_grid(cfg),
                        good_mask=bundle["good_mask"])
    return (np.asarray(cnz.g_grid, float), np.asarray(meas["n_true"], float),
            int(len(t)), int(keep.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window", default=DEF_WINDOW)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    w = window_spec(a.window)
    b195 = load_mock_bundle("2lpt0", a.out_dir, window=a.window)
    g195, occ195, n_t, n_keep = rebuild_g(b195)
    b172 = load_mock_bundle("2lpt0", a.out_dir, molly_tsv=w["molly_tsv_172"],
                            window=a.window)
    g172, occ172, n_t172, n_keep172 = rebuild_g(b172)
    n_sub = g172.shape[0] - g195.shape[0]
    assert n_sub > 0
    g_new = np.concatenate([g172[:n_sub], g195], axis=0)
    occ_new = np.concatenate([occ172[:n_sub], occ195], axis=0)
    written = []
    for path in a.packs:
        pk = load_pack(path)
        if pk.truth_counts is None:
            raise SystemExit(f"{path}: mock packs only")
        assert np.asarray(pk.g_grid).shape == g_new.shape, (pk.g_grid.shape, g_new.shape)
        prov = dict(pk.provenance or {})
        prov.update(DIAGNOSTIC=True, adopted=False,
                    role=("DIAGPACK: g_grid/g_occupancy rebuilt on the consistent "
                          "truth support (S2N_RED>snr_min on the denominator); "
                          "every other array byte-identical to the source pack; "
                          "for mock per-z recovery validation ONLY"),
                    source_pack=path,
                    g_rebuild=dict(n_truth_cut_195=n_t, n_truth_kept_195=n_keep,
                                   n_truth_cut_172=n_t172, n_truth_kept_172=n_keep172,
                                   n_subfloor_rows=int(n_sub)))
        new = dataclasses.replace(pk, g_grid=g_new, g_occupancy=occ_new, provenance=prov)
        name = os.path.basename(path).replace(".npz", "_DIAGPACK_gcons.npz")
        out = os.path.join(a.out_dir, name)
        save_pack(new, out)
        # byte-identity of everything else
        src = np.load(path); dst = np.load(out)
        changed = [k for k in src.files if k in dst.files and not np.array_equal(src[k], dst[k])]
        missing = [k for k in src.files if k not in dst.files]
        print(f"{name}: changed={changed} missing={missing}")
        assert set(changed) == {"g_grid", "g_occupancy"}, changed
        written.append(out)
    json.dump(dict(written=written, g_new=g_new.tolist()),
              open(os.path.join(a.out_dir, "DIAGPACK_gcons_manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
