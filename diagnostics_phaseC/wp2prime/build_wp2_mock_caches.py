#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""WP-2′ per-mock event-level caches (London-0 / Saclay), PI-approved.

Runs THE committed molly chain (`HBIConfig → load_and_cut_catalog →
regenerate_molly_counts`, exactly as `t1a_reproduce_cmolly.py` wires it for
2LPT-0) on a HELD-OUT mock's own catalog/truth, for BOTH the nhi195 and
nhi172 chains, and writes the two event-level caches that
`p1_refold_fold.load_migration` consumes (`…_cache.npz` / `…_cache_172.npz`).

Differences from t1a, both deliberate and protocol-consistent:
  * no pack-molly integer gate — the held-out packs carry the 2LPT-0
    calibration blocks BY DESIGN (transport), so reproducing the pack's C
    from the mock's own events is neither expected nor meaningful; instead
    the mock's own molly counts are recorded in the cache for reference;
  * input paths are the per-mock values pinned in the pack's
    provenance.json (catalog_dir / truth / bal_cat / mockdir), with the
    molly TSVs = the 2LPT-0 ones (the pack convention).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)

MOLLY_TSV_195 = ("/scratch/cavestru_root/cavestru0/mfho/"
                 "gl_prod_2lpt0_v1_20260526/figures_molly_nhi195/lya_only/"
                 "molly_matrix.tsv")
MOLLY_TSV_172 = ("/scratch/cavestru_root/cavestru0/mfho/"
                 "gl_prod_2lpt0_v1_20260526/figures_molly_nhi172/"
                 "molly_matrix.tsv")

MOCKS = {
    "london0": dict(
        catalog_dir=("/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/"
                     "london0_jura124_v1"),
        truth=("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
               "track_c/tf_london0/mockdir/dla_cat.fits"),
        bal_cat=("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                 "track_c/tf_london0/mockdir/bal_cat.fits"),
        mockdir=("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                 "track_c/tf_london0/mockdir")),
    "saclay0": dict(
        catalog_dir=("/scratch/cavestru_root/cavestru0/mfho/"
                     "gl_prod_saclay0_v1_20260630/combined_catalog"),
        truth=("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/saclay/qq_desi_y3/"
               "v4.7.5/mock-0/juraLy8-124/hcd_truth_cat.fits"),
        bal_cat=("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/saclay/"
                 "qq_desi_y3/v4.7.5/mock-0/juraLy8-124/bal_cat.fits"),
        mockdir=("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/saclay/"
                 "qq_desi_y3/v4.7.5/mock-0/juraLy8-124")),
}


def _chain(mock, tsv):
    from CDDF_analysis.hbi.cddf_catalog_hbi import (
        HBIConfig, load_molly_matrix, regenerate_molly_counts,
        load_and_cut_catalog, _build_qso_lookup)
    mk = MOCKS[mock]
    cfg = HBIConfig(
        catalog_dir=mk["catalog_dir"], truth_path=mk["truth"],
        bal_cat_path=mk["bal_cat"], molly_tsv=tsv,
        out_dir=_HERE, mockdir=mk["mockdir"],
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


def _save(cache_path, mm, cat_cut, truth_cut, is_TP, good_mask, meta,
          mock, tsv):
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
        tr_S2N=col(truth_cut, "S2N_RED"),
        tr_ZQSO=col(truth_cut, "Z_QSO"),
        own_molly_nfound=np.asarray(mm.cmp_nfound, float),
        own_molly_nfid=np.asarray(mm.cmp_nfid, float),
    )
    if "DLAFLAG" in cat_cut.colnames:
        save["cat_DLAFLAG"] = np.asarray(cat_cut["DLAFLAG"], float)
    np.savez(cache_path, **{k: v for k, v in save.items() if v is not None})
    return dict(cache=cache_path, mock=mock, tsv=os.path.basename(tsv),
                n_cat=int(len(cat_cut)), n_truth=int(len(truth_cut)),
                n_TP=int(np.sum(is_TP)),
                meta={k: v for k, v in meta.items()
                      if isinstance(v, (int, float, str, bool))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", required=True, choices=sorted(MOCKS))
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()
    report = {"schema": "wp2prime_mock_cache/v1", "mock": args.mock,
              "date": time.strftime("%Y-%m-%d")}
    for tag, tsv in (("cache", MOLLY_TSV_195), ("cache_172",
                                                MOLLY_TSV_172)):
        r = _chain(args.mock, tsv)
        path = os.path.join(args.outdir,
                            f"wp2_{args.mock}_completeness_{tag}.npz")
        report[tag] = _save(path, *r, args.mock, tsv)
        print(f"[{args.mock}] {tag} done: n_cat={report[tag]['n_cat']}",
              flush=True)
    report["wall_s"] = round(time.time() - t0, 1)
    json.dump(report, open(os.path.join(
        args.outdir, f"wp2_{args.mock}_cache_report.json"), "w"), indent=1)
    print(json.dumps({k: report[k] for k in ("mock", "wall_s")}))


if __name__ == "__main__":
    main()
