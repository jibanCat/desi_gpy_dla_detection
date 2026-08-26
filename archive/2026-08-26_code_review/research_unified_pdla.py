"""Cheap test: re-aggregate p_DLA to include SubDLA (treat SubDLA as a DLA detection).

Compares two scores on v3_loa124 London 8f:
  baseline: p_DLA = sum(model_posteriors[:, 2:])   # cols 2+ = 1DLA, 2DLA, 3DLA
  unified:  p_DLA = sum(model_posteriors[:, 1:])   # col 1 = SubDLA, cols 2+ = DLAs

Per-spec evaluation against `dla_cat.fits` truth (full NHI floor 17.2).
Truth-positive = spectrum with at least one truth DLA in [z_Lyβ + 3000 km/s, z_qso - 3000 km/s].
"""

from __future__ import annotations
import glob, json
import numpy as np
import h5py
from astropy.io import fits

H5_GLOB = "/pscratch/sd/j/jibancat/prod533_5k_20260511/london_v3_loa124_pw14_tau_eb/processed/processed-spectra-16-*.h5"
TRUTH_FITS = "/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/dla_cat.fits"

SPEED_OF_LIGHT_KM_S = 299792.458

# search-window margin (matches existing pipeline convention)
Z_MARGIN_KM_S = 3000.0

# Lyman series transition wavelengths (Å)
LYA = 1215.67
LYB = 1025.7222
LYG = 972.5368


def load_truth_dla_cat():
    """Return dict tid -> list of (z_dla, log_nhi) from truth catalog."""
    with fits.open(TRUTH_FITS) as h:
        t = h[1].data
        # column names vary; try common ones
        cols = {c.upper(): c for c in t.columns.names}
        tid_col = cols.get("TARGETID", cols.get("TID"))
        z_col = cols.get("Z_DLA", cols.get("Z"))
        nhi_col = cols.get("NHI", cols.get("LOG_NHI", cols.get("N_HI")))
        if tid_col is None or z_col is None or nhi_col is None:
            raise RuntimeError(f"truth columns: {t.columns.names}")
        tids = np.asarray(t[tid_col])
        zs = np.asarray(t[z_col])
        nhis = np.asarray(t[nhi_col])
        # detect if NHI is linear (~1e20) or log
        med = float(np.nanmedian(nhis))
        if med > 100:
            nhis = np.log10(nhis)
        print(f"truth: {len(tids)} rows; nhi range [{nhis.min():.2f}, {nhis.max():.2f}]")
    by_tid: dict[int, list[tuple[float, float]]] = {}
    for tid, z, nh in zip(tids, zs, nhis):
        by_tid.setdefault(int(tid), []).append((float(z), float(nh)))
    return by_tid


def truth_in_window(truth_dlas, z_qso, nhi_min=None, nhi_max=None):
    """Return list of (z, nhi) truth DLAs in the [z_Lyβ + 3000 km/s, z_qso - 3000 km/s] window."""
    if not truth_dlas:
        return []
    # search window: redshift such that Lyα observed is between Lyβ rest of the qso (lower) and Lyα rest of qso (upper)
    # Equivalent (used by molly): z_min ≈ (1+z_qso)*LYB/LYA - 1 + margin; z_max ≈ z_qso - margin
    z_min = (1.0 + z_qso) * LYB / LYA - 1.0
    z_min_with_margin = z_min * (1.0 + Z_MARGIN_KM_S / SPEED_OF_LIGHT_KM_S)
    z_max_with_margin = z_qso * (1.0 - Z_MARGIN_KM_S / SPEED_OF_LIGHT_KM_S)
    out = []
    for z, nh in truth_dlas:
        if z < z_min_with_margin or z > z_max_with_margin:
            continue
        if nhi_min is not None and nh < nhi_min:
            continue
        if nhi_max is not None and nh > nhi_max:
            continue
        out.append((z, nh))
    return out


def aggregate():
    """Load all 8 h5 files, return arrays."""
    files = sorted(glob.glob(H5_GLOB))
    print(f"loaded {len(files)} h5 files")
    tids, zqsos, snrs, p_base, p_unif, p_subdla = [], [], [], [], [], []
    for fn in files:
        with h5py.File(fn, "r") as h:
            mp = h["model_posteriors"][:]
            tids.append(h["target_ids"][:])
            zqsos.append(h["z_qsos"][:])
            snrs.append(h["snrs"][:])
            p_base.append(np.nansum(mp[:, 2:], axis=1))
            p_unif.append(np.nansum(mp[:, 1:], axis=1))
            p_subdla.append(mp[:, 1])
    return (
        np.concatenate(tids).astype(np.int64),
        np.concatenate(zqsos),
        np.concatenate(snrs),
        np.concatenate(p_base),
        np.concatenate(p_unif),
        np.concatenate(p_subdla),
    )


def pc_at_threshold(p_score, truth_pos, snr_mask, threshold):
    """Per-spec P/C for spectra in snr_mask, score >= threshold counts as detected."""
    in_scope = snr_mask
    pred = (p_score >= threshold) & in_scope
    truth = truth_pos & in_scope
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    n_pred = tp + fp
    n_truth = tp + fn
    P = tp / n_pred if n_pred > 0 else float("nan")
    C = tp / n_truth if n_truth > 0 else float("nan")
    return dict(threshold=threshold, n_pred=n_pred, n_truth=n_truth, tp=tp, fp=fp, fn=fn, P=P, C=C)


def main():
    truth_by_tid = load_truth_dla_cat()
    tids, zqsos, snrs, p_base, p_unif, p_subdla = aggregate()
    print(f"aggregated: {len(tids)} spectra")

    # Define per-spec truth-positive: at least one truth DLA with NHI >= 20.3 in the window
    # (matches the "real DLA" convention; sub-DLAs counted separately below)
    truth_dla_full = np.zeros(len(tids), dtype=bool)        # any truth absorber NHI>=17.2 in window
    truth_dla_classical = np.zeros(len(tids), dtype=bool)   # NHI>=20.3 in window
    truth_dla_subdla = np.zeros(len(tids), dtype=bool)      # 19.0<=NHI<20.3 in window
    for i, (tid, zq) in enumerate(zip(tids, zqsos)):
        td = truth_by_tid.get(int(tid), [])
        if not td:
            continue
        in_win_all = truth_in_window(td, zq)
        in_win_dla = truth_in_window(td, zq, nhi_min=20.3)
        in_win_sub = [t for t in in_win_all if 19.0 <= t[1] < 20.3]
        truth_dla_full[i] = len(in_win_all) > 0
        truth_dla_classical[i] = len(in_win_dla) > 0
        truth_dla_subdla[i] = len(in_win_sub) > 0

    snr_gt_2 = snrs > 2
    snr_gt_1 = snrs > 1

    print(f"\n=== Truth stats (SNR>2): ===")
    n_scope = int(snr_gt_2.sum())
    print(f"  in scope SNR>2: {n_scope}")
    print(f"  classical DLA truth (NHI>=20.3, in window): {int((truth_dla_classical & snr_gt_2).sum())}")
    print(f"  sub-DLA truth (19.0<=NHI<20.3, in window):  {int((truth_dla_subdla & snr_gt_2).sum())}")
    print(f"  any-absorber truth (NHI>=17.2, in window):  {int((truth_dla_full & snr_gt_2).sum())}")

    # Score distributions in nulls (true: no DLA in window)
    null_mask_classical = snr_gt_2 & ~truth_dla_classical
    null_mask_strict = snr_gt_2 & ~truth_dla_full
    print(f"\n=== Score distribution stats: ===")
    print(f"  n_null_classical (no NHI>=20.3 truth): {int(null_mask_classical.sum())}")
    print(f"  n_null_strict (no NHI>=17.2 truth):    {int(null_mask_strict.sum())}")
    print(f"  baseline p_dla on classical nulls: median={np.nanmedian(p_base[null_mask_classical]):.3e}, p95={np.nanpercentile(p_base[null_mask_classical], 95):.3e}, p99={np.nanpercentile(p_base[null_mask_classical], 99):.3e}")
    print(f"  unified  p_dla on classical nulls: median={np.nanmedian(p_unif[null_mask_classical]):.3e}, p95={np.nanpercentile(p_unif[null_mask_classical], 95):.3e}, p99={np.nanpercentile(p_unif[null_mask_classical], 99):.3e}")

    thresholds = [0.99, 0.999, 0.99999]

    print(f"\n=== SNR>2, truth=classical DLA (NHI>=20.3) ===")
    print(f"{'cut':>10s}  {'score':>8s}  {'n_pred':>6s}  {'n_truth':>7s}  {'tp':>5s}  {'fp':>5s}  {'P':>7s}  {'C':>7s}")
    rows = []
    for thr in thresholds:
        for label, score in [("baseline", p_base), ("unified", p_unif)]:
            r = pc_at_threshold(score, truth_dla_classical, snr_gt_2, thr)
            print(f"{thr:>10.5f}  {label:>8s}  {r['n_pred']:>6d}  {r['n_truth']:>7d}  {r['tp']:>5d}  {r['fp']:>5d}  {r['P']:>7.4f}  {r['C']:>7.4f}")
            rows.append({**r, "score": label, "snr": "snr_gt_2", "truth": "classical_DLA"})

    print(f"\n=== SNR>1, truth=classical DLA (NHI>=20.3) ===")
    print(f"{'cut':>10s}  {'score':>8s}  {'n_pred':>6s}  {'n_truth':>7s}  {'tp':>5s}  {'fp':>5s}  {'P':>7s}  {'C':>7s}")
    for thr in thresholds:
        for label, score in [("baseline", p_base), ("unified", p_unif)]:
            r = pc_at_threshold(score, truth_dla_classical, snr_gt_1, thr)
            print(f"{thr:>10.5f}  {label:>8s}  {r['n_pred']:>6d}  {r['n_truth']:>7d}  {r['tp']:>5d}  {r['fp']:>5d}  {r['P']:>7.4f}  {r['C']:>7.4f}")
            rows.append({**r, "score": label, "snr": "snr_gt_1", "truth": "classical_DLA"})

    # ALSO evaluate with truth = any absorber NHI>=19 in window (i.e., counting sub-DLA truth as positive too)
    truth_dla_or_sub = np.zeros(len(tids), dtype=bool)
    for i, (tid, zq) in enumerate(zip(tids, zqsos)):
        td = truth_by_tid.get(int(tid), [])
        if td and len(truth_in_window(td, zq, nhi_min=19.0)) > 0:
            truth_dla_or_sub[i] = True
    print(f"\n=== SNR>2, truth=DLA+SubDLA (NHI>=19) ===")
    print(f"{'cut':>10s}  {'score':>8s}  {'n_pred':>6s}  {'n_truth':>7s}  {'tp':>5s}  {'fp':>5s}  {'P':>7s}  {'C':>7s}")
    for thr in thresholds:
        for label, score in [("baseline", p_base), ("unified", p_unif)]:
            r = pc_at_threshold(score, truth_dla_or_sub, snr_gt_2, thr)
            print(f"{thr:>10.5f}  {label:>8s}  {r['n_pred']:>6d}  {r['n_truth']:>7d}  {r['tp']:>5d}  {r['fp']:>5d}  {r['P']:>7.4f}  {r['C']:>7.4f}")
            rows.append({**r, "score": label, "snr": "snr_gt_2", "truth": "DLA_or_SubDLA"})

    # Save full table
    out = "/pscratch/sd/j/jibancat/prod533_5k_20260511/null_quantile_map_combined/unified_pdla_test.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
