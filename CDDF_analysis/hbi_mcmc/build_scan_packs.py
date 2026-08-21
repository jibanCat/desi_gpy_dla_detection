#!/usr/bin/env python
"""build_scan_packs.py — the observable-only (collar 3000 + b km/s) MOCK pack
variants used by the predeclared collar scan (@8edd3b1, ckpt 10.8) and by
every mock posterior validation since (ckpt 10.10, Batteries 1-4).

Counts + dX + fp_E_alloc are rebuilt at the collar from the mock catalog;
EVERY OTHER ARRAY is copied byte-identically from the source (adopted v2)
pack. Truth never enters the selection (post-hoc only, inside
cc_posterior_validation's recovery evaluation).

History: this recipe ran 2026-08-17 as an uncommitted scratch script
(archived in the notes repo: figures/2026-08-17_stilt_diag/inputs/ckpt10p5/
scripts/collar_scan.py). Committed here 2026-08-21 (CP-1) so the scan packs
can be regenerated from the corrected-g adopted packs through a committed
routine; ``--regress-against`` proves it IS that recipe by requiring
byte-identity with the existing scan packs when pointed at the old inputs.

Env: gpdla (jax-free; extract_pack loaded file-directly). NOTE (2026-08-21):
byte-identity of ``dX_coarse_committed`` is numpy-version sensitive at the
1-3 ULP level (total_DeltaX_in_zbins accumulation; gpdla numpy 2.4.4 vs
gpdla-hbi 2.2.6) — the 2026-08-17 packs reproduce IDENTICALLY under gpdla
and differ by <= 3 ULP under gpdla-hbi. The field is a provenance carrier
(pack.py optional), not a model input.
  python CDDF_analysis/hbi_mcmc/build_scan_packs.py --src-dir V2DIR --out-dir OUT
      [--families 2lpt0 london0 saclay0] [--buffers 300]
      [--regress-against DIR]   # assert byte-identity with DIR/scanpack_*.npz
"""
from __future__ import annotations
import argparse
import glob
import importlib.util as ilu
import json
import os
import subprocess
import sys
import time

import numpy as np
import fitsio

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
_spec = ilu.spec_from_file_location("_scan_ep", os.path.join(_HERE, "extract_pack.py"))
EP = ilu.module_from_spec(_spec)
sys.modules["_scan_ep"] = EP
_spec.loader.exec_module(EP)
from CDDF_analysis.hbi.cddf_catalog_hbi import (   # noqa: E402
    HBIConfig, load_molly_matrix, build_fine_grid, build_M_b,
    _build_qso_lookup, AbsorptionDistance, total_DeltaX_in_zbins)

LYA = 1215.67
C_KMS = 299792.458
TAG = "bw0p2_pad19p0_molly172"
DATA_PLANE = ("counts", "dX", "dX_coarse_committed", "fp_E_alloc")


def build_family(fam, src_dir, out_dir, buffers):
    m = EP.MOCKS[fam]
    w = EP.window_spec("lya_only")
    cfg = HBIConfig(catalog_dir=m["catalog_dir"], truth_path="",
                    bal_cat_path=m["bal_cat_path"], molly_tsv=w["molly_tsv"],
                    out_dir="/tmp", mockdir=m["mockdir"],
                    zbins=tuple(EP.ZC_EDGES.tolist()), lam_rf_min=1025.0,
                    no_bal=True, rng_seed=0)
    mm = load_molly_matrix(w["molly_tsv"])
    lookup = _build_qso_lookup(cfg)
    catf = sorted(glob.glob(os.path.join(m["catalog_dir"], "dlacat*.fits")))
    cat = (np.concatenate([fitsio.read(f, ext=1) for f in catf])
           if len(catf) > 1 else fitsio.read(catf[0], ext=1))
    bal = np.unique(fitsio.read(m["bal_cat_path"], ext=1,
                                columns=["TARGETID"])["TARGETID"].astype(np.int64))
    bal_set = set(bal.tolist())
    tid = cat["TARGETID"].astype(np.int64)
    zqc = np.asarray(cat["Z_QSO"], float)
    zd = np.asarray(cat["Z_DLA"], float)
    nhi = np.asarray(cat["NHI"], float)
    snr = np.asarray(cat["SNR_REDSIDE"], float)
    flag_ok = (np.asarray(cat["DLAFLAG"], int) == 0)
    p_ok = (np.asarray(cat["P_DLA"], float) > cfg.p_dla_min)
    src = os.path.join(src_dir, f"modelA_pack_{fam}_{TAG}_v2.npz")
    raw0 = dict(np.load(src, allow_pickle=False))
    zq_all, snr_all = [], []
    for t, (s, z) in lookup.items():
        if s <= cfg.snr_min or not (cfg.z_qso_min < z < cfg.z_qso_max) \
                or t in bal_set:
            continue
        zq_all.append(z)
        snr_all.append(s)
    zq_all = np.asarray(zq_all)
    snr_all = np.asarray(snr_all)
    logN_lo, logN_hi, N_b, dN_b = build_fine_grid(cfg)
    zf = np.round(np.arange(2.0, 3.5 + 1e-9, 0.1), 10)
    written = []
    for b in buffers:
        coll = (3000.0 + b) / C_KMS
        qzl = np.maximum(3600.0 / LYA - 1.0,
                         cfg.lam_rf_min * (1 + zq_all) / LYA - 1.0 + coll)
        qzh = np.minimum(zq_all - coll,
                         cfg.lam_rf_max * (1 + zq_all) / LYA - 1.0 - coll)
        ok = np.isfinite(qzl) & np.isfinite(qzh) & (qzh > qzl)
        Xc = AbsorptionDistance(zmax=float(qzh[ok].max()), Omega_m=cfg.Omega_m)
        X_tot = total_DeltaX_in_zbins(np.asarray(cfg.zbins), qzl[ok], qzh[ok], Xc)
        M_meta = build_M_b(qzl[ok], qzh[ok], snr_all[ok], mm, logN_lo,
                           logN_hi, N_b, dN_b, zf, Xc, cfg)
        dX = np.asarray(M_meta["PX"], float).T
        col = dX.sum(axis=0)
        fpE = np.zeros_like(dX)
        nz = col > 0
        fpE[:, nz] = dX[:, nz] / col[nz]
        z_lo = np.maximum(3600.0 / LYA - 1.0,
                          1025.0 * (1 + zqc) / LYA - 1.0 + coll)
        z_hi = np.minimum(zqc - coll, 1216.0 * (1 + zqc) / LYA - 1.0 - coll)
        keep = ((zqc > cfg.z_qso_min) & (zqc < cfg.z_qso_max)
                & (zd > z_lo) & (zd < z_hi) & ~np.isin(tid, bal))
        op = keep & flag_ok & p_ok & (snr > cfg.snr_min)
        counts, _ = EP.bin_counts_cks(nhi[op], zd[op], snr[op])
        raw = dict(raw0)
        raw["counts"] = counts.astype(np.int64)
        raw["dX"] = dX
        raw["dX_coarse_committed"] = np.asarray(X_tot, float)
        raw["fp_E_alloc"] = fpE
        out = os.path.join(out_dir, f"scanpack_{fam}_b{b}.npz")
        np.savez_compressed(out, **raw)
        written.append(dict(out=out, src=src, b=b, counts=int(counts.sum()),
                            dX=float(dX.sum())))
        print(fam, f"b={b}: counts={int(counts.sum())} dX={float(dX.sum()):.1f}",
              flush=True)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--families", nargs="+", default=["2lpt0", "london0", "saclay0"])
    ap.add_argument("--buffers", nargs="+", type=int, default=[300])
    ap.add_argument("--regress-against", default=None,
                    help="directory holding scanpack_<fam>_b<b>.npz to which the "
                         "outputs must be byte-identical (recipe proof)")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=_REPO).decode().strip()
    except Exception:
        commit = "unknown"
    written = []
    for fam in a.families:
        written += build_family(fam, a.src_dir, a.out_dir, a.buffers)
    regress = None
    if a.regress_against:
        regress = []
        for wr in written:
            ref = os.path.join(a.regress_against, os.path.basename(wr["out"]))
            with np.load(ref) as z1, np.load(wr["out"]) as z2:
                diff = [k for k in sorted(set(z1.files) | set(z2.files))
                        if k not in z1.files or k not in z2.files
                        or not np.array_equal(z1[k], z2[k])]
            regress.append(dict(out=wr["out"], ref=ref, differing_keys=diff,
                                identical=(diff == [])))
            print(f"[regress] {os.path.basename(wr['out'])}: "
                  f"{'IDENTICAL' if not diff else 'DIFFERS ' + str(diff)}")
    prov = dict(role=("observable-only collar-scan MOCK packs: counts/dX/fp_E "
                      "rebuilt at collar 3000+b; every other array byte-identical "
                      "to the source adopted v2 pack"),
                recipe="CDDF_analysis/hbi_mcmc/build_scan_packs.py",
                recipe_history=("ran 2026-08-17 as an uncommitted scratch script "
                                "(notes repo figures/2026-08-17_stilt_diag/inputs/"
                                "ckpt10p5/scripts/collar_scan.py); committed "
                                "2026-08-21 (CP-1)"),
                src_dir=a.src_dir, written=written, regress=regress,
                data_plane_keys=list(DATA_PLANE), code_commit=commit,
                wall_s=round(time.time() - t0, 1))
    for wr in written:
        with open(wr["out"][:-4] + ".provenance.json", "w") as f:
            json.dump(dict(prov, this=wr), f, indent=1)
    with open(os.path.join(a.out_dir, "scan_packs_manifest.json"), "w") as f:
        json.dump(prov, f, indent=1)
    if regress is not None and not all(r["identical"] for r in regress):
        raise SystemExit("REGRESSION FAILED: outputs differ from the reference packs")
    print("SCAN PACKS DONE", f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
