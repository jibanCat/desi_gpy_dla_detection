#!/usr/bin/env python
"""snr_matched_decomp.py — SNR-matched decomposition of the AI-mini-BAL high-N excess (task #14, Lens 2/4).

Question: is the AI>0-mini-BAL enrichment among clean high-N DLA detections explained by SNR selection
(the "benign" reading — BAL QSOs are brighter, so more real DLAs are detectable AND more mini-BALs are
detected), or is there a residual excess that SNR-matching cannot remove?

FINDING (this reproduction, 2026-07-03): the residual is REAL and robust, NOT ~0. Lens 2's "integrated
excess ~0 after SNR-matching" does NOT hold up. The balfinder AI-detection completeness saturates above
SNR_CIV~2 (parent BI=0 AI rate: 0% below SNR_CIV~1 -> ~24% by SNR_CIV~2 -> flat), and the high-N dets sit
at median SNR_CIV~5.3 (deep in the saturated regime), so the "brighter -> more AI detected" mechanism is
EXHAUSTED. Matching on SNR_CIV (the balfinder's own SNR) AND on RED_SNR (red-side continuum) BOTH leave a
+6.5pp residual at >=20.3, rising to ~+9pp in the deep tail. So the excess is NOT an SNR-selection artifact.

Interpretation: the ~6.5pp (>=20.3) / ~9pp (deep-tail) excess is a REAL over-representation of AI>0
mini-BAL sightlines among high-N DLA detections. Its benign-ness (real DLAs preferentially toward mini-BAL
QSOs = astrophysical over-density, NOT false positives) rests on the width/EW PHYSICS (narrow shallow
troughs cannot mimic a >=20.3 damped profile -- inject_minibal_gp.py) + the STACK (AI>0 dets carry real
damped cores -- stack_ai_minibal.py). As a CONSERVATIVE FP upper bound (if none of the excess were
astrophysical), the residual is the upper end of the BAL band: ~6% (>=20.3), ~9% (deep tail, E1-limited).
=> This is the upper end of the reported benign-direction band ~2-6% at >=20.3.

Method (catalog-only, no GP):
  * Parent = z>2 QSOs, BI_CIV==0 (the op-cut removes BI>0 via DLAFLAG). AI-host = AI_CIV>0.
  * Parent AI-host rate as a function of SNR (two axes: SNR_CIV from our VAC = the AI-completeness axis;
    RED_SNR from the archive = red-side continuum, a cross-check).
  * Clean high-N dets = headline op-cut (SNR_REDSIDE>2 & P_DLA>0.99 & lam_rest>=1025 & DLAFLAG==0), per NHI.
  * SNR-matched EXPECTED AI-host frac = mean over the bin's dets of parent_AI_rate(SNR_i). Residual =
    observed - expected (binomial CIs). Reported for BOTH SNR axes (they agree ~ +6.5pp).

Aggregate/derived numbers only (no per-object real spectra). conda gpdla; HDF5_USE_FILE_LOCKING=FALSE.
"""
import os, argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")
import h5py, fitsio

LYA = 1215.67
ARCHIVE = "/scratch/cavestru_root/cavestru0/mfho/nersc/loa_archives/loa_full_z2_noR_v2.h5"
OUR_VAC = "/scratch/cavestru_root/cavestru0/mfho/our_loa_bal_vac/our_loa_bal_vac_v1.fits"
REAL_DLA = "/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/loa_main_dark_v1/dlacat-loa-main-dark-v1.fits"


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    if n == 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (lo, hi)


def parent_rate_by_snr(snr_par, ai_par, edges):
    """parent AI-host rate in each SNR bin."""
    b = np.digitize(snr_par, edges) - 1
    n = len(edges) - 1
    return np.array([ai_par[b == j].mean() if (b == j).any() else np.nan for j in range(n)])


def matched_expect(snr_det, prate, edges):
    """SNR-matched expected AI-host frac = mean parent rate at the dets' SNR."""
    b = np.clip(np.digitize(snr_det, edges) - 1, 0, len(edges) - 2)
    return np.nanmean(prate[b])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nhi-edges", default="20.0,20.3,20.6,21.0,21.5,22.5")
    ap.add_argument("--snrciv-edges", default="0,1,2,3,4,6,10,1e9")
    ap.add_argument("--redsnr-edges", default="2,2.5,3,4,6,10,1e9")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "figures"))
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    nhi_edges = np.array([float(x) for x in a.nhi_edges.split(",")])
    civ_edges = np.array([float(x) for x in a.snrciv_edges.split(",")])
    red_edges = np.array([float(x) for x in a.redsnr_edges.split(",")])

    # --- VAC: AI/BI/SNR_CIV per TARGETID ---
    ours = fitsio.read(OUR_VAC)
    vt = np.asarray(ours["TARGETID"], np.int64)
    v_ai = np.asarray(ours["AI_CIV"], float) > 0
    v_bi = np.asarray(ours["BI_CIV"], float) > 0
    v_civ = np.asarray(ours["SNR_CIV"], float)
    ai_by_tid = dict(zip(vt.tolist(), v_ai.tolist()))
    civ_by_tid = dict(zip(vt.tolist(), v_civ.tolist()))

    # --- archive: RED_SNR per QSO ---
    with h5py.File(ARCHIVE, "r") as H:
        cat = H["catalog"][:]
    a_tid = np.asarray(cat["TARGETID"], np.int64); a_red = np.asarray(cat["RED_SNR"], float)
    red_by_tid = dict(zip(a_tid.tolist(), a_red.tolist()))

    # --- parent BI=0 populations + AI rate vs each SNR axis ---
    par = ~v_bi
    p_civ = v_civ[par]; p_ai_civ = v_ai[par]
    prate_civ = parent_rate_by_snr(p_civ, p_ai_civ, civ_edges)
    print("Parent BI=0 AI-host rate vs SNR_CIV (the AI-completeness axis):")
    for j in range(len(civ_edges) - 1):
        m = (np.digitize(p_civ, civ_edges) - 1) == j
        print(f"  SNR_CIV [{civ_edges[j]:g},{civ_edges[j+1]:g}): {100*prate_civ[j]:.1f}%  (n={int(m.sum())})")
    # RED_SNR parent (archive, BI=0) — cross-check axis
    bi_by_tid = dict(zip(vt.tolist(), v_bi.tolist()))
    a_ai_arr = np.array([ai_by_tid.get(int(t), False) for t in a_tid])
    a_bi_arr = np.array([bi_by_tid.get(int(t), False) for t in a_tid])
    apar = (a_red > 2) & ~a_bi_arr
    prate_red = parent_rate_by_snr(a_red[apar], a_ai_arr[apar], red_edges)

    # --- dets: headline op-cut clean high-N ---
    d = fitsio.read(REAL_DLA)
    tid = np.asarray(d["TARGETID"], np.int64); nhi = np.asarray(d["NHI"], float)
    zd = np.asarray(d["Z_DLA"], float); zq = np.asarray(d["Z_QSO"], float)
    snr = np.asarray(d["SNR_REDSIDE"], float); p = np.asarray(d["P_DLA"], float); fl = np.asarray(d["DLAFLAG"], int)
    lam = LYA * (1 + zd) / (1 + zq)
    op = (snr > 2) & (p > 0.99) & (lam >= 1025) & (fl == 0)
    ii = np.where(op)[0]
    dt = tid[ii]; d_nhi = nhi[ii]
    d_ai = np.array([ai_by_tid.get(int(t), False) for t in dt])
    d_civ = np.array([civ_by_tid.get(int(t), np.nan) for t in dt])
    d_red = np.array([red_by_tid.get(int(t), np.nan) for t in dt])
    print(f"\nSNR_CIV: high-N(>=20.3) dets median={np.nanmedian(d_civ[d_nhi>=20.3]):.2f} vs parent BI=0 median={np.nanmedian(p_civ):.2f}")
    print(f"clean op-cut dets: {len(ii)}\n")

    print(f"{'NHI bin':>13s}{'n':>7s}{'obs AI%':>9s}{'obs CI':>15s}{'civ-exp%':>9s}{'resid':>7s}{'z':>6s}{'red-exp%':>9s}{'resid':>7s}")
    rows = []
    for j in range(len(nhi_edges) - 1):
        lo, hi = nhi_edges[j], nhi_edges[j + 1]
        m = (d_nhi >= lo) & (d_nhi < hi)
        n = int(m.sum())
        if n < 20:
            continue
        k = int(d_ai[m].sum()); obs = k / n
        cl, ch = clopper_pearson(k, n)
        eciv = matched_expect(d_civ[m], prate_civ, civ_edges)
        ered = matched_expect(d_red[m], prate_red, red_edges)
        se = np.sqrt(obs * (1 - obs) / n); z = (obs - eciv) / se if se > 0 else np.nan
        rows.append((lo, hi, n, obs, cl, ch, eciv, ered, z))
        print(f"  [{lo:.1f},{hi:.1f}){n:>7d}{100*obs:>9.1f}{f'[{100*cl:.1f},{100*ch:.1f}]':>15s}"
              f"{100*eciv:>9.1f}{100*(obs-eciv):>+7.1f}{z:>6.1f}{100*ered:>9.1f}{100*(obs-ered):>+7.1f}")

    for lim in (20.3, 21.0):
        m = d_nhi >= lim; n = int(m.sum()); k = int(d_ai[m].sum()); obs = k / n
        eciv = matched_expect(d_civ[m], prate_civ, civ_edges); ered = matched_expect(d_red[m], prate_red, red_edges)
        cl, ch = clopper_pearson(k, n)
        tag = "INTEGRATED >=20.3" if lim == 20.3 else "DEEP TAIL  >=21.0"
        print(f"\n{tag}: obs {100*obs:.1f}% [{100*cl:.1f},{100*ch:.1f}]  SNR_CIV-matched {100*eciv:.1f}% "
              f"(resid {100*(obs-eciv):+.2f}pp) | RED_SNR-matched {100*ered:.1f}% (resid {100*(obs-ered):+.2f}pp)  n={n}")
    print("\n=> residual is REAL and robust to the SNR axis; it is the conservative FP UPPER bound "
          "(band upper end ~6% at >=20.3, ~9% deep tail). Benign-ness rests on the width/EW physics + the stack.")

    # --- figure ---
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    mids = [0.5 * (r[0] + r[1]) for r in rows]
    obsv = np.array([r[3] for r in rows]); clv = np.array([r[4] for r in rows]); chv = np.array([r[5] for r in rows])
    ecivv = np.array([r[6] for r in rows]); eredv = np.array([r[7] for r in rows])
    ax[0].errorbar(mids, 100 * obsv, yerr=[100 * (obsv - clv), 100 * (chv - obsv)], fmt="o-", color="C3",
                   capsize=3, label="observed AI-host frac (dets)")
    ax[0].plot(mids, 100 * ecivv, "s--", color="C0", label="SNR_CIV-matched parent")
    ax[0].plot(mids, 100 * eredv, "^:", color="C1", label="RED_SNR-matched parent")
    ax[0].set_xlabel("log N_HI"); ax[0].set_ylabel("AI-host fraction [%]")
    ax[0].set_title("AI-host fraction vs N_HI:\nobserved vs SNR-matched parent (two axes)"); ax[0].legend(fontsize=8)
    resv = obsv - ecivv
    ax[1].axhline(0, color="grey", ls=":")
    ax[1].errorbar(mids, 100 * resv, yerr=[100 * (obsv - clv), 100 * (chv - obsv)], fmt="o-", color="C2", capsize=3,
                   label="obs - SNR_CIV-matched")
    ax[1].plot(mids, 100 * (obsv - eredv), "^:", color="C1", label="obs - RED_SNR-matched")
    ax[1].set_xlabel("log N_HI"); ax[1].set_ylabel("residual [pp]")
    ax[1].set_title("SNR-matched residual is REAL (~+6.5pp >=20.3, ~+9pp deep tail):\n= conservative FP upper bound (band upper end)")
    ax[1].legend(fontsize=8)
    fig.suptitle("SNR-matched decomposition of the AI-mini-BAL high-N excess (real LOA, lya-only)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{a.outdir}/snr_matched_decomp.png", dpi=130); plt.close(fig)
    print(f"\nfig -> {a.outdir}/snr_matched_decomp.png")


if __name__ == "__main__":
    main()
