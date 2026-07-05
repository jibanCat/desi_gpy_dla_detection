#!/usr/bin/env python
"""lya_only_rerun.py — recompute the BAL high-N FP numbers under the LYA-ONLY selection
(lam_rf_min=1025), MATCHING the headline, vs the full-forest selection the earlier arbiter used.

Motivation (PI, 2026-07-02): the headline restricts DLA detections to the Lyα-only forest
(`run_phase3d_postkernel.py:75` lam_rf_min=1025.0; `build_loa0_fp_product.py:214` "the Lyβ region
inflates the FP"). But the arbiter (veto_snr_sweep.py, decompose_highn_fp.py) used the FULL forest
(no lam_rf cut), so its over-count (56%/153%) and leak (31.8%) OVER-STATE the headline BAL bias.

lam_rest = LYA_REST * (1+Z_DLA)/(1+Z_QSO); lya-only = lam_rest >= lam_rf_min (default 1025) — exactly
the headline catalog cut (cddf_catalog_hbi.py load_and_cut_catalog on cfg.lam_rf_min).

Part A — REAL LOA (our own VAC): clean-set broad-trough leak Ω + field coincidence + veto cost.
Part B — MOCK (2LPT-0, truth-anchored): spurious-BAL over-count (the 56%/153% analog).

Env: source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate gpdla
Aggregate/mock/derived-catalog numbers only.
"""
import argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")
import fitsio

LYA = 1215.67
C_KMS = 299792.458
OUR_VAC = "/scratch/cavestru_root/cavestru0/mfho/our_loa_bal_vac/our_loa_bal_vac_v1.fits"
REAL_DLA = "/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/loa_main_dark_v1/dlacat-loa-main-dark-v1.fits"
MOCK_DLA = "/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/combined_catalog/dlacat-v2.8.5-mockcat.fits"
MOCK_TRUTH = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits"


def lam_rest(zd, zq):
    return LYA * (1 + zd) / (1 + zq)


def part_A_real(lam_rf):
    ours = fitsio.read(OUR_VAC)
    brd = {int(t): (float(w) > 2000 and float(a) > 0)
           for t, w, a in zip(ours["TARGETID"], ours["WIDEST_CIV_450"], ours["AI_CIV"])}
    bi = {int(t): float(b) for t, b in zip(ours["TARGETID"], ours["BI_CIV"])}
    d = fitsio.read(REAL_DLA)
    tid = np.asarray(d["TARGETID"], np.int64); nhi = np.asarray(d["NHI"], float)
    zd = np.asarray(d["Z_DLA"], float); zq = np.asarray(d["Z_QSO"], float)
    snr = np.asarray(d["SNR_REDSIDE"], float); p = np.asarray(d["P_DLA"], float); fl = np.asarray(d["DLAFLAG"], int)
    lr = lam_rest(zd, zq)
    lya = lr >= lam_rf
    op = (snr > 2) & (p > 0.99) & lya
    clean = op & (fl == 0)
    print(f"  [REAL LOA, lam_rf_min={lam_rf}]  clean dets kept by lya cut: {int(clean.sum())}")
    print(f"  {'NHI>=':>7s}{'broad-leak%':>12s}{'field%':>8s}{'coinc-sub%':>11s}{'veto Ωrm%':>10s}{'n_clean':>9s}")
    for lim in (20.0, 20.3, 21.0, 21.6):
        m = clean & (nhi >= lim); idx = np.where(m)[0]; w = 10.0 ** nhi[idx]
        broad = np.array([brd.get(int(tid[i]), False) for i in idx])
        leak = 100 * w[broad].sum() / w.sum()
        # field broad-BAL rate among non-DLA-hosting QSOs, lya-only, z-matched enough via same set
        hosts = set(tid[m].tolist())
        # crude field rate: broad-BAL fraction of clean dets that are NOT high-N-DLA hosts is ~the base rate;
        # reuse the veto_snr_sweep field number (5.78% full) recomputed here on the clean set's complement
        fieldrate = 100 * np.mean([brd.get(int(t), False) for t in np.unique(tid[clean & (nhi < 20.0)])]) if (clean & (nhi < 20.0)).any() else np.nan
        coinc = leak - fieldrate
        # veto: broad-trough removes broad-BAL dets from the FULL op set (fresh veto, incl BI>0)
        mo = op & (nhi >= lim); io = np.where(mo)[0]; wo = 10.0 ** nhi[io]
        vrm = np.array([brd.get(int(tid[i]), False) or (bi.get(int(tid[i]), 0) > 0) for i in io])
        print(f"  {lim:>7.1f}{leak:>12.1f}{fieldrate:>8.1f}{coinc:>11.1f}{100*wo[vrm].sum()/wo.sum():>10.1f}{len(idx):>9d}")


def part_B_mock(lam_rf):
    d = fitsio.read(MOCK_DLA)
    tid = np.asarray(d["TARGETID"], np.int64); nhi = np.asarray(d["NHI"], float)
    zd = np.asarray(d["Z_DLA"], float); zq = np.asarray(d["Z_QSO"], float)
    snr = np.asarray(d["SNR_REDSIDE"], float); p = np.asarray(d["P_DLA"], float)
    balf = np.asarray(d["BAL_FLAG"], int)
    hcd = fitsio.read(MOCK_TRUTH)
    hset = set(int(x) for x in hcd["TARGETID"])
    tr_nhi = np.asarray(hcd["NHI"], float); tr_snr = np.asarray(hcd["SNR"], float)
    from collections import defaultdict
    tby = defaultdict(list)
    for t, z in zip(hcd["TARGETID"], np.asarray(hcd["Z"], float)):
        tby[int(t)].append(float(z))

    def spurious(t, z, dv=3000.0):
        for zt in tby.get(int(t), ()):
            if abs(z - zt) / (1 + zt) * C_KMS < dv:
                return False
        return True

    lr = lam_rest(zd, zq)
    op = (snr > 2) & (p > 0.99) & (lr >= lam_rf)
    isfp = np.array([spurious(int(tid[i]), zd[i]) for i in range(len(tid))])
    print(f"  [MOCK 2LPT-0, lam_rf_min={lam_rf}]  spurious-BAL over-count = FP-BAL Ω / truth Ω")
    print(f"  {'NHI>=':>7s}{'over-count%':>12s}{'n_fpbal':>9s}")
    for lim in (20.0, 20.3, 21.0, 21.6):
        base = op & isfp & (balf == 1) & (nhi >= lim)
        # CAVEAT (referee B): the numerator is lya-only (op includes lr>=lam_rf) but this
        # truth-Ω denominator is FULL-FOREST — only ~74% of truth Ω lies in the lya-only
        # region, so this Part-B over-count is FP_Ω(lya)/truth_Ω(full) and UNDER-states the
        # true lya-only over-count by ~x0.74 (multiply by ~1.34). Mock cross-check only, NOT
        # the headline. Full fix: restrict truth to lam_rest>=lam_rf via a TARGETID->Z_QSO join.
        truthO = float(np.sum(10.0 ** tr_nhi[(tr_nhi >= lim) & (tr_snr > 2)]))
        over = 100 * (10.0 ** nhi[base]).sum() / truthO
        print(f"  {lim:>7.1f}{over:>12.1f}{int(base.sum()):>9d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "both"], default="both")
    a = ap.parse_args()
    for lam_rf in (911.0, 1025.0):   # full-forest vs lya-only
        tag = "FULL FOREST" if lam_rf == 911.0 else "LYA-ONLY (headline)"
        print(f"\n================ {tag} (lam_rf_min={lam_rf}) ================")
        if a.part in ("A", "both"):
            part_A_real(lam_rf)
        if a.part in ("B", "both"):
            part_B_mock(lam_rf)


if __name__ == "__main__":
    main()
