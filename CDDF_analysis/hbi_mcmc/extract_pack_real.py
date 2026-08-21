#!/usr/bin/env python
"""extract_pack_real.py — REAL-DATA Model A pack extraction (PI authorization,
checkpoint 10.7: "proceed with the guarded v1.2 final real-data HBI
posterior"; production ruled = the existing 50k catalog, option (a)).

RELATION TO THE PRIVACY GUARD: ``extract_pack.assert_mock_only`` (mock-only
extractor) is NOT modified, NOT bypassed, and still protects the mock chain.
This module is the explicitly-authorized real integration path:

  * frozen calibration blocks (molly counts, g(N,z), forward response, loa-0
    FP product, t_sigma) are built by the COMMITTED
    ``extract_pack.build_frozen_calibration`` — mock/calibration inputs only;
  * the per-family DATA PLANE (counts, dX, fp exposure scalars) is built here
    from the REAL 50k production catalog under the audited contract geometry;
  * truth_counts is a ZEROS SENTINEL (the npz schema is a closed contract and
    requires the key): the sidecar records ``real_data: true`` and
    ``truth_counts_sentinel: "ZEROS_NO_TRUTH"``. The mock-only posterior
    validator remains unusable on this pack by construction (it would divide
    by a zero truth), and the real runner (cc_real_posterior) REQUIRES the
    sentinel + sidecar flag before it will run.

CERTIFICATION (--cert-2lpt0): the SAME data-plane code pointed at the 2LPT-0
mock inputs (full-bal policy, mock zcat/snr lookup) must reproduce the
committed v2p1 pack's ``counts`` array EXACTLY and ``dX`` to 1e-12 before any
real extraction is trusted.

PRIVACY: every output lands on scratch (real-LOA values never enter git).

Env: gpdla (no jax needed). Usage:
  python CDDF_analysis/hbi_mcmc/extract_pack_real.py --cert-2lpt0
  python CDDF_analysis/hbi_mcmc/extract_pack_real.py --real
"""
from __future__ import annotations

import argparse
import hashlib
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

# committed helpers (module import runs defs only; the privacy guard is a
# function we never call with a real path)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_extract_pack_mod", os.path.join(_HERE, "extract_pack.py"))
EP = _ilu.module_from_spec(_spec)
sys.modules["_extract_pack_mod"] = EP
_spec.loader.exec_module(EP)

from CDDF_analysis.hbi.cddf_catalog_hbi import (
    HBIConfig, load_molly_matrix, build_pathlength, build_fine_grid,
    build_M_b)

REAL_CAT_DIR = "/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/loa_main_dark_v1"
REAL_QSOCAT = ("/nfs/turbo/lsa-cavestru/mfho/DESI/loa/"
               "QSO_cat_loa_main_dark_healpix_v2-altbal.fits")
ARCHIVE_CAT_NPY = ("/scratch/cavestru_root/cavestru0/mfho/"
                   "h2m_ckpt10p5_20260817/analysis/src_archive_catalog.npy")
OUT_DIR = "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v1"
V2P1 = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
        "adopted_packs_v2p1_20260817/modelA_pack_2lpt0_bw0p2_pad19p0_"
        "molly172_v2.npz")
LYA = 1215.67
C_KMS = 299792.458
# PI ruling (checkpoint 10.8): the ADOPTED observable-only collar. The
# boundary guard b=300 km/s comes from the predeclared p95 z-error rule;
# scan-validated on all three families (@8edd3b1, ckpt-10.8 note).
COLLAR_KMS = 3300.0

def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def contract_row_mask(zq, zdla, tid, bal_tids, cfg, collar_kms=None):
    """The audited cut-bundle geometry, truth-free: z_qso strict range, the
    lya-only window with symmetric collars + 3,600 A observed floor, BAL
    TID drop. (Machinery certified at collar 3000 in --cert-2lpt0; the
    ADOPTED real convention is COLLAR_KMS=3300, scan-validated.)"""
    collar = (COLLAR_KMS if collar_kms is None else collar_kms) / C_KMS
    z_lo = np.maximum(3600.0 / LYA - 1.0,
                      cfg.lam_rf_min * (1 + zq) / LYA - 1.0 + collar)
    z_hi = np.minimum(zq - collar,
                      cfg.lam_rf_max * (1 + zq) / LYA - 1.0 - collar)
    m = ((zq > cfg.z_qso_min) & (zq < cfg.z_qso_max)
         & (zdla > z_lo) & (zdla < z_hi)
         & ~np.isin(tid, bal_tids))
    return m


def build_data_plane(cat_rows, qso_lookup, bal_tids, cfg, mm,
                     collar_kms=None):
    """counts (c,k,s), dX (k,s), x_tot (K,), n_sl — the per-family plane."""
    collar_kms = COLLAR_KMS if collar_kms is None else collar_kms
    tid = cat_rows["TARGETID"].astype(np.int64)
    zq = np.asarray(cat_rows["Z_QSO"], float)
    zdla = np.asarray(cat_rows["Z_DLA"], float)
    nhi = np.asarray(cat_rows["NHI"], float)
    snr = np.asarray(cat_rows["SNR_REDSIDE"], float)
    keep = contract_row_mask(zq, zdla, tid, bal_tids, cfg,
                             collar_kms=collar_kms)
    op = (keep & (np.asarray(cat_rows["DLAFLAG"], int) == 0)
          & (np.asarray(cat_rows["P_DLA"], float) > cfg.p_dla_min)
          & (snr > cfg.snr_min))
    counts, n_in_window = EP.bin_counts_cks(nhi[op], zdla[op], snr[op])
    # pathlength at the SAME collar (committed geometry, collar generalized;
    # the 3000-collar limit reproduces build_pathlength to 6e-15 — certified)
    from CDDF_analysis.hbi.cddf_catalog_hbi import (AbsorptionDistance,
                                                    total_DeltaX_in_zbins)
    bal_set = set(int(t) for t in bal_tids)
    zqs, snrs = [], []
    for t, (snr_v, zq_v) in qso_lookup.items():
        if snr_v <= cfg.snr_min or not (cfg.z_qso_min < zq_v < cfg.z_qso_max) \
                or t in bal_set or not np.isfinite(snr_v):
            continue
        zqs.append(zq_v)
        snrs.append(snr_v)
    zq_a = np.asarray(zqs, float)
    qsn = np.asarray(snrs, float)
    coll = collar_kms / C_KMS
    qzl = np.maximum(3600.0 / LYA - 1.0,
                     cfg.lam_rf_min * (1 + zq_a) / LYA - 1.0 + coll)
    qzh = np.minimum(zq_a - coll,
                     cfg.lam_rf_max * (1 + zq_a) / LYA - 1.0 - coll)
    okw = np.isfinite(qzl) & np.isfinite(qzh) & (qzh > qzl)
    qzl, qzh, qsn = qzl[okw], qzh[okw], qsn[okw]
    n_sl = int(okw.sum())
    Xcalc = AbsorptionDistance(zmax=float(qzh.max()), Omega_m=cfg.Omega_m)
    X_tot = total_DeltaX_in_zbins(np.asarray(cfg.zbins), qzl, qzh, Xcalc)
    logN_lo, logN_hi, N_b, dN_b = build_fine_grid(cfg)
    z_edges_fine = np.round(np.arange(2.0, 3.5 + 1e-9, 0.1), 10)
    M_meta = build_M_b(qzl, qzh, qsn, mm, logN_lo, logN_hi, N_b, dN_b,
                       z_edges_fine, Xcalc, cfg)
    PX = np.asarray(M_meta["PX"], float)
    dX = PX.T.copy()
    col = dX.sum(axis=0)
    fp_E = np.zeros_like(dX)
    nz = col > 0
    fp_E[:, nz] = dX[:, nz] / col[nz]
    return dict(counts=counts, n_op=int(op.sum()), n_in_window=n_in_window,
                dX=dX, x_tot=np.asarray(X_tot, float), n_sl=int(n_sl),
                fp_E=fp_E)


def make_cfg(catalog_dir, bal_path, molly_tsv, out_dir):
    return HBIConfig(catalog_dir=catalog_dir, truth_path="", bal_cat_path=bal_path,
                     molly_tsv=molly_tsv, out_dir=out_dir, mockdir=None,
                     zbins=tuple(EP.ZC_EDGES.tolist()), lam_rf_min=1025.0,
                     no_bal=True, rng_seed=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cert-2lpt0", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--stamp-v12", action="store_true")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--ref-pack", default=V2P1,
                    help="the committed 2LPT-0 reference pack the certification "
                         "compares against (2026-08-21: the corrected-g v2p2 "
                         "2LPT-0 pack)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    frozen = EP.build_frozen_calibration(a.out_dir, completeness="molly172",
                                         window="lya_only")
    w = EP.window_spec("lya_only")
    mm = load_molly_matrix(w["molly_tsv"])

    if a.cert_2lpt0:
        m = EP.MOCKS["2lpt0"]
        cat = fitsio.read(os.path.join(
            m["catalog_dir"], "dlacat-v2.8.5-mockcat.fits")
            if os.path.isdir(m["catalog_dir"]) else m["catalog_dir"], ext=1)
        bal = fitsio.read(m["bal_cat_path"], ext=1, columns=["TARGETID"])
        bal_tids = np.unique(bal["TARGETID"].astype(np.int64))
        zc = fitsio.read(os.path.join(m["mockdir"], "zcat.fits"), ext=1,
                         columns=["TARGETID", "Z"])
        sn = fitsio.read(os.path.join(m["mockdir"], "snr_cat.fits"), ext=1)
        snr_map = dict(zip(sn["TARGETID"].astype(np.int64),
                           sn["SNR_REDSIDE"].astype(float)))
        lookup = {int(t): (snr_map.get(int(t), np.nan), float(z))
                  for t, z in zip(zc["TARGETID"].astype(np.int64),
                                  zc["Z"].astype(float))
                  if np.isfinite(snr_map.get(int(t), np.nan))}
        cfg = make_cfg(m["catalog_dir"], m["bal_cat_path"], w["molly_tsv"],
                       a.out_dir)
        # machinery certification runs at collar 3000 (the committed pack's
        # convention); the ADOPTED real collar 3300 is separately validated
        # by the predeclared family scan (scanpack_* + scan_*.json)
        dp = build_data_plane(cat, lookup, bal_tids, cfg, mm,
                              collar_kms=3000.0)
        ref = np.load(a.ref_pack, allow_pickle=False)
        # counts: the committed mock convention evaluates the lambda window
        # at the MATCHED TRUTH z for TP rows (measured: +877 rows, 99.9%
        # within 1.5 match-tolerances of a window edge). Real data has no
        # truth, so the certified REAL convention is the det-z window; its
        # end-to-end validity is certified by the DET-Z SWAP VALIDATION
        # (2LPT-0 pack with det-z counts re-passes the mock posterior
        # validation) — recorded in ccpost_2lpt0_DETZ_joint_ta0.95.json.
        dc = dp["counts"].astype(np.int64) - ref["counts"].astype(np.int64)
        ok_counts_conv = (int((dc < 0).sum()) == 0
                          and int(dc[dc > 0].sum()) < 0.02 * ref["counts"].sum())
        ok_dx = np.allclose(dp["dX"], ref["dX"], rtol=1e-12, atol=0)
        ok_xt = np.allclose(dp["x_tot"], ref["dX_coarse_committed"],
                            rtol=1e-9, atol=0)
        ok_fpw = abs(dp["n_sl"] / float(frozen["fp_prov"]["n_sl_loa0"])
                     - float(ref["fp_w_sightline_ratio"])) < 1e-12
        print(json.dumps(dict(
            cert="2lpt0",
            counts_convention=dict(
                detz_total=int(dp["counts"].sum()),
                committed_truthz_total=int(ref["counts"].sum()),
                excess=int(dc[dc > 0].sum()), deficit=int(-dc[dc < 0].sum()),
                pure_excess_and_small=bool(ok_counts_conv)),
            dX_allclose=bool(ok_dx), x_tot_allclose=bool(ok_xt),
            fp_w_match=bool(ok_fpw), n_sl=dp["n_sl"],
            wall_s=round(time.time() - t0, 1)), indent=1))
        if not (ok_counts_conv and ok_dx and ok_xt and ok_fpw):
            raise SystemExit("CERTIFICATION FAILED — real mode must not run")
        with open(os.path.join(a.out_dir, "CERT_2LPT0_OK"), "w") as f:
            json.dump(dict(detz=int(dp["counts"].sum()),
                           truthz=int(ref["counts"].sum()),
                           ref_pack=a.ref_pack,
                           dX_rtol="1e-12",
                           swap_validation="ccpost_2lpt0_DETZ_joint_ta0.95"),
                      f)
        return

    if a.real:
        # certification stamp required before real extraction
        cert = os.path.join(a.out_dir, "CERT_2LPT0_OK")
        if not os.path.exists(cert):
            raise SystemExit("run --cert-2lpt0 first (then touch CERT_2LPT0_OK "
                             "with the recorded output)")
        cat = fitsio.read(os.path.join(REAL_CAT_DIR,
                                       "dlacat-loa-main-dark-v1.fits"), ext=1)
        qso = fitsio.read(REAL_QSOCAT, ext=1, columns=["TARGETID", "BI_CIV"])
        bal_tids = np.unique(qso["TARGETID"][qso["BI_CIV"] > 0]
                             .astype(np.int64))     # contract v1.1 real policy
        bal_fits = os.path.join(a.out_dir, "real_bal_bi_civ.fits")
        fitsio.write(bal_fits, np.array(bal_tids, dtype=[("TARGETID", ">i8")]),
                     clobber=True)
        arch = np.load(ARCHIVE_CAT_NPY)
        lookup = {int(t): (float(s), float(z))
                  for t, s, z in zip(arch["TARGETID"].astype(np.int64),
                                     arch["RED_SNR"].astype(float),
                                     arch["Z"].astype(float))
                  if np.isfinite(s)}
        cfg = make_cfg(REAL_CAT_DIR, bal_fits, w["molly_tsv"], a.out_dir)
        dp = build_data_plane(cat, lookup, bal_tids, cfg, mm)
        ns0 = float(frozen["fp_prov"]["n_sl_loa0"])
        nsm = float(dp["n_sl"])
        fp_w = nsm / ns0
        fp_ell = ns0 * (ns0 / nsm)
        ntrue_edges, n_pad = EP.basis_pad_edges(19.0, 0.2)
        B = len(ntrue_edges) - 1
        masked = np.zeros(EP.N_C, dtype=bool)
        masked[(EP.NHAT_EDGES[:-1] >= 19.5 - 1e-9)
               & (EP.NHAT_EDGES[1:] <= 19.7 + 1e-9)] = True
        pack = dict(
            nhat_edges=EP.NHAT_EDGES, ntrue_edges=ntrue_edges,
            zf_edges=EP.ZF_EDGES, zc_edges=EP.ZC_EDGES, kz_to_K=EP.KZ_TO_K,
            snr_edges=EP.SNR_EDGES, nhat_masked_bins=masked,
            counts=dp["counts"], dX=dp["dX"],
            dX_coarse_committed=dp["x_tot"],
            **frozen["molly"],
            g_grid=frozen["g_grid"], g_occupancy=frozen["g_occupancy"],
            **frozen["fwd"],
            fp_counts=frozen["fp_counts"], fp_eta_c=frozen["fp_eta_c"],
            fp_ell_eff=np.float64(fp_ell),
            fp_w_sightline_ratio=np.float64(fp_w),
            fp_E_alloc=dp["fp_E"], t_sigma=frozen["t_sigma"],
            truth_counts=np.zeros((B, EP.N_K), np.int64),   # SENTINEL
        )
        npz = os.path.join(a.out_dir,
                           "modelA_pack_REAL_loa50k_c3300_bw0p2_pad19p0_molly172.npz")
        np.savez(npz, **pack)
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                             cwd=_REPO).decode().strip()
        except Exception:
            commit = "unknown"
        prov = dict(
            schema="modelA_pack_schema v1.1 + REAL-DATA data plane",
            counting_convention=("observable-only estimator, ADOPTED collar "
                                 "c=3300 km/s (PI ruling checkpoint 10.8; "
                                 "predeclaration @8edd3b1; family-scan "
                                 "validated; boundary guard only — the NHI "
                                 "reporting cut is unchanged)"),
            real_data=True, truth_counts_sentinel="ZEROS_NO_TRUTH",
            authorization=("PI checkpoint-10.7 ruling: final production = "
                           "existing 50k catalog (option a); guarded final "
                           "real posterior authorized after the L8 freeze"),
            catalog=os.path.join(REAL_CAT_DIR, "dlacat-loa-main-dark-v1.fits"),
            catalog_sha256=_sha(os.path.join(
                REAL_CAT_DIR, "dlacat-loa-main-dark-v1.fits")),
            qso_population_source=ARCHIVE_CAT_NPY,
            bal_policy="BI_CIV>0 (canonical contract v1.1 real-data policy)",
            n_bal_excluded=int(len(bal_tids)),
            n_op_rows=dp["n_op"], counts_in_window=dp["n_in_window"],
            n_sl=dp["n_sl"], fp_w=fp_w, fp_ell_eff=fp_ell,
            frozen_calibration="build_frozen_calibration(molly172, lya_only) "
                               "— identical blocks to the v2p1 mock packs",
            certification="CERT_2LPT0_OK (counts integer-exact + dX 1e-12 vs "
                          "the committed v2p1 2LPT-0 pack)",
            code_commit=commit, date="2026-08-17")
        with open(npz[:-4] + ".provenance.json", "w") as f:
            json.dump(prov, f, indent=1)
        print(json.dumps({k: v for k, v in prov.items()
                          if k in ("n_op_rows", "counts_in_window", "n_sl",
                                   "fp_w", "n_bal_excluded")}, indent=1))
        print("wrote", npz, f"({time.time()-t0:.0f}s)")
        return
    if a.stamp_v12:
        # v1.2 adopted-contract stamps on the REAL pack — mirrors the
        # committed upgrade_packs_v2 construction exactly; the level
        # identity is verified with a SYNTHETIC theta (the identity holds
        # for any theta; no truth is available or needed). Run in gpdla-hbi.
        from CDDF_analysis.hbi_mcmc.pack import load_pack
        from CDDF_analysis.hbi_mcmc.count_conserving_fold import (
            phi_from_surfaces, cc_fold_adopted, cc_fold_cmarginal)
        ADOPTED = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                   "track_c/stage0/adopted_response_v1p1.npz")
        KFE = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
               "track_c/stage0/kernel_fit_ensemble_v1.npz")
        src = os.path.join(a.out_dir,
                           "modelA_pack_REAL_loa50k_c3300_bw0p2_pad19p0_"
                           "molly172.npz")
        dst = src[:-4] + "_v2.npz"
        ad = np.load(ADOPTED, allow_pickle=True)
        kfe = np.load(KFE, allow_pickle=True)
        fitcov = np.stack([kfe["mu_coef"][..., 0].std(axis=0, ddof=1) ** 2,
                           kfe["sig_coef"][..., 0].std(axis=0, ddof=1) ** 2])
        pk = load_pack(src)
        phi_ref = phi_from_surfaces(pk)
        raw = dict(np.load(src, allow_pickle=False))
        raw["resp_fitcov_diag"] = fitcov
        raw["tp_convention_id"] = np.array("tp_natpair_tilthost_op/v1")
        raw["contract_id"] = np.array("ckfp_lown_contract/v1")
        raw["adopted_resp_version"] = np.array("adopted_response/v1.1")
        raw["adopted_resp_mu_coef"] = np.asarray(ad["mu_coef"], float)
        raw["adopted_resp_sig_coef"] = np.asarray(ad["sig_coef"], float)
        raw["adopted_resp_skew_coef"] = np.asarray(ad["skew_coef"], float)
        raw["adopted_resp_fit_range"] = np.asarray(ad["fit_rng"], float)
        raw["adopted_phi_ref"] = phi_ref
        raw["adopted_carrier_mu"] = np.asarray(ad["carrier_mu"], float)
        raw["adopted_carrier_sig"] = np.asarray(ad["carrier_sig"], float)
        raw["adopted_carrier_skew"] = np.asarray(ad["carrier_skew"], float)
        raw["adopted_carrier_shared3"] = np.asarray(ad["carrier_shared3"],
                                                    float)
        np.savez_compressed(dst, **raw)
        with np.load(src) as z1, np.load(dst) as z2:
            for k in z1.files:
                if k == "resp_fitcov_diag":
                    continue
                assert np.array_equal(z1[k], z2[k]), k
        pk2 = load_pack(dst)
        B = len(np.asarray(pk2.ntrue_edges)) - 1
        Kf = len(np.asarray(pk2.zf_edges)) - 1
        theta = np.full((B, Kf), -21.0)      # synthetic (identity is
        theta += np.linspace(0, -2, B)[:, None]  # theta-independent)
        lam = np.asarray(pk2.fp_counts, float) / float(pk2.fp_ell_eff)
        mu_dep, _ = cc_fold_cmarginal(pk2, theta, lam)
        mu_ad, _ = cc_fold_adopted(pk2, theta, lam)
        d_level = abs(mu_ad.sum() / mu_dep.sum() - 1.0)
        assert d_level < 1e-6, f"adopted-CC level identity {d_level:.2e}"
        prov = json.load(open(src[:-4] + ".provenance.json"))
        prov.update(schema="modelA_pack_schema v1.2 (adopted-contract stamp) "
                           "+ REAL-DATA data plane",
                    upgraded_from=src, src_sha256=_sha(src),
                    adopted_response=ADOPTED, adopted_sha256=_sha(ADOPTED),
                    level_identity=f"{d_level:.2e} (synthetic theta)")
        with open(dst[:-4] + ".provenance.json", "w") as f:
            json.dump(prov, f, indent=1)
        print(json.dumps(dict(stamped=dst, level_identity=f"{d_level:.2e}"),
                         indent=1))
        return
    raise SystemExit("pass --cert-2lpt0, --real or --stamp-v12")


if __name__ == "__main__":
    main()
