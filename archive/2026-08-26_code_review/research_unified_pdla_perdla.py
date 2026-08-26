"""Per-DLA molly evaluation — baseline vs unified p_DLA.

Two tables: classical (truth NHI≥20.3) and sub-DLA (truth 19≤NHI<20.3).
For each: compares baseline p_DLA (sum DLAs only) vs unified (SubDLA + DLAs).

Matches molly_faithful_pc_plots.py conventions: BAL-excl, |Δz|/(1+z) < 0.01 greedy,
lya_lyb window [911, 1216], SNR_REDSIDE > 2.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
import h5py
import fitsio
from astropy.table import Table, vstack

ROOT = "/pscratch/sd/j/jibancat/desi_gpy_dla_detection"
sys.path.insert(0, os.path.join(ROOT, "examples"))
from gp_native_pc_plots import match_truth_to_cat

CAT_DIR = "/pscratch/sd/j/jibancat/prod533_5k_20260511/london_v3_loa124_pw14_tau_eb"
TRUTH_FN = "/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/dla_cat.fits"
BAL_FN = "/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/bal_cat.fits"

LYA = 1215.67
LYB = 1025.7222
SNR_MIN = 2.0
DZ_REL = 0.01
LAM_RF_MIN = 911.0    # lya_lyb window inner
LAM_RF_MAX = 1216.0   # outer

THRESHOLDS = [0.99, 0.999, 0.99999]


def load_per_spec_scores():
    """Per-targetid: baseline + unified p_DLA from h5."""
    files = sorted(glob.glob(f"{CAT_DIR}/processed/processed-spectra-16-*.h5"))
    out = {}
    for fn in files:
        with h5py.File(fn, "r") as h:
            tids = h["target_ids"][:]
            mp = h["model_posteriors"][:]
            for i, t in enumerate(tids):
                out[int(t)] = (
                    float(np.nansum(mp[i, 2:])),
                    float(np.nansum(mp[i, 1:])),
                    float(mp[i, 1]),
                )
    print(f"loaded scores for {len(out)} spectra")
    return out


def load_catalog():
    files = sorted(glob.glob(f"{CAT_DIR}/dlacat-*.fits"))
    tbls = [Table(fitsio.read(f, ext=1)) for f in files]
    cat = vstack(tbls)
    print(f"loaded {len(cat)} per-DLA rows from {len(files)} files")
    return cat


def load_truth(nhi_min=None, nhi_max=None):
    t = Table(fitsio.read(TRUTH_FN, ext=1))
    # Detect NHI scale and convert if linear
    nhi_col = "NHI" if "NHI" in t.columns.keys() else "N_HI"
    nhi = np.asarray(t[nhi_col], dtype=float)
    if np.nanmedian(nhi) > 100:
        nhi = np.log10(nhi)
        t[nhi_col] = nhi
    if nhi_min is not None:
        t = t[nhi >= nhi_min]
        nhi = np.asarray(t[nhi_col], dtype=float)
    if nhi_max is not None:
        t = t[nhi < nhi_max]
    # match_truth_to_cat expects Z_TRUTH; rename Z_DLA→Z_TRUTH if needed
    if "Z_TRUTH" not in t.columns.keys() and "Z_DLA" in t.columns.keys():
        t.rename_column("Z_DLA", "Z_TRUTH")
    print(f"truth rows after NHI filter [{nhi_min},{nhi_max}): {len(t)}, cols={t.columns.keys()}")
    return t


def apply_bal_cut(cat: Table, truth: Table):
    bal = fitsio.read(BAL_FN, ext=1, columns=["TARGETID", "BI_CIV"])
    bal_tids = set(int(r["TARGETID"]) for r in bal if r["BI_CIV"] > 0)
    cat = cat[~np.isin(np.asarray(cat["TARGETID"]), list(bal_tids))]
    truth = truth[~np.isin(np.asarray(truth["TARGETID"]), list(bal_tids))]
    print(f"BAL excl → cat={len(cat)}, truth={len(truth)}")
    return cat, truth


def apply_window_and_snr(cat: Table, truth: Table):
    """Apply molly's lambda_rf window and SNR cut."""
    z_qso = np.asarray(cat["Z_QSO"], dtype=float)
    z_dla = np.asarray(cat["Z_DLA"], dtype=float)
    snr = np.asarray(cat["SNR_REDSIDE"], dtype=float)
    lam_rf = LYA * (1.0 + z_dla) / (1.0 + z_qso)
    keep = (lam_rf >= LAM_RF_MIN) & (lam_rf <= LAM_RF_MAX) & (snr > SNR_MIN)
    cat = cat[keep]
    print(f"after window + SNR>{SNR_MIN}: cat={len(cat)}")

    # Truth: apply same SNR_RED cut (from h5 SNRs joined to truth's TARGETIDs)
    return cat


def truth_snr_filter(truth: Table, per_spec_snr: dict):
    """Filter truth to TARGETIDs with SNR > SNR_MIN."""
    tids = np.asarray(truth["TARGETID"])
    snr = np.array([per_spec_snr.get(int(t), -1.0) for t in tids])
    truth = truth[snr > SNR_MIN]
    print(f"truth after SNR>{SNR_MIN}: {len(truth)}")
    return truth


def truth_window_filter(truth: Table, per_spec_zqso: dict):
    """Filter truth to DLAs in the lya_lyb window."""
    tids = np.asarray(truth["TARGETID"])
    z_dla = np.asarray(truth["Z_TRUTH"], dtype=float)
    z_qso = np.array([per_spec_zqso.get(int(t), -1.0) for t in tids])
    lam_rf = LYA * (1.0 + z_dla) / (1.0 + z_qso)
    keep = (lam_rf >= LAM_RF_MIN) & (lam_rf <= LAM_RF_MAX)
    truth = truth[keep]
    print(f"truth after window: {len(truth)}")
    return truth


def evaluate(cat: Table, truth: Table, score_col: str, threshold: float,
             pred_nhi_range, label: str):
    """One row of P/C at threshold."""
    p = np.asarray(cat[score_col], dtype=float)
    nhi_pred = np.asarray(cat["NHI"], dtype=float)
    nhi_lo, nhi_hi = pred_nhi_range
    pred_mask = (p > threshold) & (nhi_pred >= nhi_lo) & (nhi_pred < nhi_hi)
    cat_kept = cat[pred_mask]
    n_pred = len(cat_kept)
    is_tp, _, _, truth_matched = match_truth_to_cat(cat_kept, truth, DZ_REL)
    tp = int(is_tp.sum())
    n_truth = len(truth)
    fp = n_pred - tp
    P = tp / n_pred if n_pred > 0 else float("nan")
    C = int(truth_matched.sum()) / n_truth if n_truth > 0 else float("nan")
    return dict(label=label, threshold=threshold, n_pred=n_pred, tp=tp, fp=fp,
                n_truth=n_truth, P=P, C=C)


def main():
    scores = load_per_spec_scores()
    cat = load_catalog()

    # Inject baseline + unified scores
    base = np.array([scores.get(int(t), (np.nan, np.nan, np.nan))[0] for t in cat["TARGETID"]])
    unif = np.array([scores.get(int(t), (np.nan, np.nan, np.nan))[1] for t in cat["TARGETID"]])
    cat["P_DLA_BASELINE"] = base
    cat["P_DLA_UNIFIED"] = unif

    # Pull per-spec SNR and z_qso (from cat itself — same value across all DLAs of a spec)
    per_spec_snr = {}
    per_spec_zqso = {}
    for r in cat:
        tid = int(r["TARGETID"])
        per_spec_snr[tid] = float(r["SNR_REDSIDE"])
        per_spec_zqso[tid] = float(r["Z_QSO"])

    # Truth: full NHI floor 17.2 to start (we'll filter by NHI per table)
    truth_full = load_truth(nhi_min=17.0)
    cat_be, truth_be = apply_bal_cut(cat, truth_full)
    cat_be = apply_window_and_snr(cat_be, truth_be)
    truth_be = truth_snr_filter(truth_be, per_spec_snr)
    truth_be = truth_window_filter(truth_be, per_spec_zqso)
    print()

    # --- TABLE 1: Classical DLA (truth NHI≥20.3, predicted NHI≥20.3)
    truth_classical = truth_be[(np.asarray(truth_be["NHI"], dtype=float) >= 20.3)]
    print(f"=== Table 1: Classical DLA, truth NHI≥20.3, n_truth={len(truth_classical)} ===")
    print(f"{'thr':>10s}  {'score':>9s}  {'n_pred':>6s}  {'tp':>5s}  {'fp':>5s}  {'P':>7s}  {'C':>7s}")
    rows = []
    for thr in THRESHOLDS:
        for col_label, col in [("baseline", "P_DLA_BASELINE"), ("unified", "P_DLA_UNIFIED")]:
            r = evaluate(cat_be, truth_classical, col, thr, (20.3, 99.0), "classical")
            print(f"{thr:>10.5f}  {col_label:>9s}  {r['n_pred']:>6d}  {r['tp']:>5d}  {r['fp']:>5d}  {r['P']:>7.4f}  {r['C']:>7.4f}")
            r["score"] = col_label
            rows.append(r)

    # --- TABLE 2: Sub-DLA from DLA-model MAP (truth NHI ∈ [19, 20.3), pred NHI ∈ [19, 20.3))
    # NOTE: this is the DLA model's MAP landing in the sub-DLA range — a wrong/poor sub-DLA detector
    truth_nhi = np.asarray(truth_be["NHI"], dtype=float)
    truth_subdla = truth_be[(truth_nhi >= 19.0) & (truth_nhi < 20.3)]
    print(f"\n=== Table 2: Sub-DLA via DLA-model MAP, truth 19≤NHI<20.3, n_truth={len(truth_subdla)} ===")
    print(f"(Note: this uses DLA-model predictions that landed in [19,20.3] — the wrong detector)")
    print(f"{'thr':>10s}  {'score':>9s}  {'n_pred':>6s}  {'tp':>5s}  {'fp':>5s}  {'P':>7s}  {'C':>7s}")
    for thr in THRESHOLDS:
        for col_label, col in [("baseline", "P_DLA_BASELINE"), ("unified", "P_DLA_UNIFIED")]:
            r = evaluate(cat_be, truth_subdla, col, thr, (19.0, 20.3), "subdla_via_dla")
            print(f"{thr:>10.5f}  {col_label:>9s}  {r['n_pred']:>6d}  {r['tp']:>5d}  {r['fp']:>5d}  {r['P']:>7.4f}  {r['C']:>7.4f}")
            r["score"] = col_label
            rows.append(r)

    # --- TABLE 3: Sub-DLA via the dedicated SubDLA model (per-spec presence detection)
    # SubDLA's per-spec score = model_posteriors[:, 1].
    # Build per-spec data from h5 directly (covers ALL spectra, not just those in cat).
    tids_all, snr_all, zqso_all, p_sub_all, p_dla_all = [], [], [], [], []
    for fn in sorted(glob.glob(f"{CAT_DIR}/processed/processed-spectra-16-*.h5")):
        with h5py.File(fn, "r") as h:
            mp = h["model_posteriors"][:]
            tids_all.append(h["target_ids"][:])
            snr_all.append(h["snrs"][:])
            zqso_all.append(h["z_qsos"][:])
            p_sub_all.append(mp[:, 1])
            p_dla_all.append(np.nansum(mp[:, 2:], axis=1))
    tids_scored = np.concatenate(tids_all).astype(np.int64)
    snr_per_spec = np.concatenate(snr_all)
    zqso_per_spec = np.concatenate(zqso_all)
    p_subdla_per_spec = np.concatenate(p_sub_all)
    p_dla_per_spec = np.concatenate(p_dla_all)
    print(f"loaded per-spec scores: {len(tids_scored)} spectra")

    # Build per-spec truth membership using FULL truth + per-spec zqso for window check
    truth_full = fitsio.read(TRUTH_FN, ext=1)
    t_tid = np.asarray(truth_full["TARGETID"])
    t_z = np.asarray(truth_full["Z_DLA"], dtype=float)
    t_nhi = np.asarray(truth_full["NHI"], dtype=float)
    if np.nanmedian(t_nhi) > 100:
        t_nhi = np.log10(t_nhi)
    spec_zq = dict(zip(tids_scored, zqso_per_spec))
    truth_by_tid = {}
    for tt, tz, tn in zip(t_tid, t_z, t_nhi):
        truth_by_tid.setdefault(int(tt), []).append((float(tz), float(tn)))

    SPEED = 299792.458
    MARGIN_KMS = 3000.0

    def in_window_for(tid, z):
        zq = spec_zq.get(int(tid))
        if zq is None or zq <= 0:
            return False
        z_min = (1.0 + zq) * LYB / LYA - 1.0
        z_min_m = z_min * (1.0 + MARGIN_KMS / SPEED)
        z_max_m = zq * (1.0 - MARGIN_KMS / SPEED)
        return z_min_m <= z <= z_max_m

    has_subdla_truth = np.zeros(len(tids_scored), dtype=bool)
    has_classical_truth = np.zeros(len(tids_scored), dtype=bool)
    for i, t in enumerate(tids_scored):
        abs_in_win = [(z, n) for z, n in truth_by_tid.get(int(t), [])
                      if in_window_for(int(t), z)]
        if any(19.1 <= n < 20.0 for _, n in abs_in_win):
            has_subdla_truth[i] = True
        if any(n >= 20.3 for _, n in abs_in_win):
            has_classical_truth[i] = True

    bal_tids = set(int(rr["TARGETID"]) for rr in
                   fitsio.read(BAL_FN, ext=1, columns=["TARGETID", "BI_CIV"])
                   if rr["BI_CIV"] > 0)
    is_bal = np.array([int(t) in bal_tids for t in tids_scored])
    snr_mask = (snr_per_spec > SNR_MIN) & ~is_bal & (zqso_per_spec > 0)

    pure_subdla_truth = has_subdla_truth & ~has_classical_truth

    print(f"\n=== Table 3: Sub-DLA per-spec via SubDLA model (P(SubDLA|D) score) ===")
    print(f"truth=spec has absorber in [19.1, 20.0) and NO classical DLA (NHI>=20.3)")
    print(f"in scope (SNR>{SNR_MIN}, BAL-excl): n_spec={int(snr_mask.sum())}")
    print(f"pure sub-DLA truth in scope: {int((pure_subdla_truth & snr_mask).sum())}")
    print(f"p_SubDLA stats on in-scope: median={np.nanmedian(p_subdla_per_spec[snr_mask]):.4e}, p95={np.nanpercentile(p_subdla_per_spec[snr_mask], 95):.4e}, p99={np.nanpercentile(p_subdla_per_spec[snr_mask], 99):.4e}")
    SUBDLA_THR = [0.5, 0.7, 0.9, 0.95, 0.99]
    print(f"\nA) P(SubDLA|D) alone:")
    print(f"{'thr':>10s}  {'n_pred':>6s}  {'tp':>5s}  {'fp':>5s}  {'P':>7s}  {'C':>7s}")
    for thr in SUBDLA_THR:
        pred = (p_subdla_per_spec > thr) & snr_mask
        truth = pure_subdla_truth & snr_mask
        tp = int(np.sum(pred & truth))
        fp = int(np.sum(pred & ~truth))
        n_pred = tp + fp
        P = tp / n_pred if n_pred > 0 else float("nan")
        C = tp / int(truth.sum()) if int(truth.sum()) > 0 else float("nan")
        Pstr = f"{P:>7.4f}" if not np.isnan(P) else "    nan"
        print(f"{thr:>10.5f}  {n_pred:>6d}  {tp:>5d}  {fp:>5d}  {Pstr}  {C:>7.4f}")
        rows.append(dict(label="subdla_alone", threshold=thr, n_pred=n_pred, tp=tp, fp=fp, P=P, C=C, score="P_SubDLA"))

    print(f"\nB) P(SubDLA|D) > thr AND P(DLA|D) < 0.5 (filter out DLA-dominant specs):")
    print(f"{'thr':>10s}  {'n_pred':>6s}  {'tp':>5s}  {'fp':>5s}  {'P':>7s}  {'C':>7s}")
    for thr in SUBDLA_THR:
        pred = (p_subdla_per_spec > thr) & (p_dla_per_spec < 0.5) & snr_mask
        truth = pure_subdla_truth & snr_mask
        tp = int(np.sum(pred & truth))
        fp = int(np.sum(pred & ~truth))
        n_pred = tp + fp
        P = tp / n_pred if n_pred > 0 else float("nan")
        C = tp / int(truth.sum()) if int(truth.sum()) > 0 else float("nan")
        Pstr = f"{P:>7.4f}" if not np.isnan(P) else "    nan"
        print(f"{thr:>10.5f}  {n_pred:>6d}  {tp:>5d}  {fp:>5d}  {Pstr}  {C:>7.4f}")
        rows.append(dict(label="subdla_filtered", threshold=thr, n_pred=n_pred, tp=tp, fp=fp, P=P, C=C, score="P_SubDLA_filt"))

    print(f"\nC) P(SubDLA|D) > P(DLA|D) AND P(SubDLA|D) > thr (dominant-model test):")
    print(f"{'thr':>10s}  {'n_pred':>6s}  {'tp':>5s}  {'fp':>5s}  {'P':>7s}  {'C':>7s}")
    for thr in SUBDLA_THR:
        pred = (p_subdla_per_spec > thr) & (p_subdla_per_spec > p_dla_per_spec) & snr_mask
        truth = pure_subdla_truth & snr_mask
        tp = int(np.sum(pred & truth))
        fp = int(np.sum(pred & ~truth))
        n_pred = tp + fp
        P = tp / n_pred if n_pred > 0 else float("nan")
        C = tp / int(truth.sum()) if int(truth.sum()) > 0 else float("nan")
        Pstr = f"{P:>7.4f}" if not np.isnan(P) else "    nan"
        print(f"{thr:>10.5f}  {n_pred:>6d}  {tp:>5d}  {fp:>5d}  {Pstr}  {C:>7.4f}")
        rows.append(dict(label="subdla_dominant", threshold=thr, n_pred=n_pred, tp=tp, fp=fp, P=P, C=C, score="P_SubDLA_dom"))

    # Save
    out = "/pscratch/sd/j/jibancat/prod533_5k_20260511/null_quantile_map_combined/unified_pdla_perdla.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
