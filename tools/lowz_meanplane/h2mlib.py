"""Checkpoint-10.5 H2-M analysis library — re-implemented molly-exact join +
weighting, validated against the recorded H2-M per-injection outcomes before
any new use (PI ruling 10.5 requires the SAME analysis contract everywhere).

Join semantics (molly-exact, from the checkpoint-10 record + natpair pack
kernel_condition): accepted row = P_DLA > 0.99 & DLAFLAG == 0 & NHI_pred > 19.5;
greedy 1-to-1 match per sightline, candidates sorted desc predicted NHI,
|dz|/(1+z_true) < 0.01. TARGETIDs strictly int64 everywhere.
"""
from __future__ import annotations
import glob
import os

import numpy as np
import fitsio

P_CUT = 0.99
DZ_REL = 0.01
# molly172 matrix floor: the completeness join accepts predicted NHI down to
# the C-matrix axis floor (17.2), NOT the 19.5 kernel-condition floor
# (validated: 19.5 breaks 9/900 recorded H2-M outcomes, 17.2 gives 900/900).
NHI_PRED_FLOOR = 17.2

SNR_EDGES = [2.0, 3.5, 6.5, np.inf]
Z_EDGES = [2.1, 2.56, 2.96, 3.79]


def load_dlacat_rows(outputs_dir, pattern="dlacat-*.fits"):
    """Concatenate per-hpx dlacat FITS into one structured array (int64 TID)."""
    files = sorted(glob.glob(os.path.join(outputs_dir, pattern)))
    assert files, f"no dlacat files under {outputs_dir}"
    parts = [fitsio.read(f, ext=1) for f in files]
    rows = np.concatenate(parts)
    assert rows["TARGETID"].dtype == np.int64
    return rows


def greedy_match_sightline(cand_z, cand_nhi, true_z, true_n):
    """Greedy 1-to-1 desc-NHI match. Returns rec flags + matched cand idx
    per truth entry (-1 = unmatched). Candidates already acceptance-filtered."""
    order = np.argsort(-cand_nhi)
    used_true = np.zeros(len(true_z), bool)
    match_idx = np.full(len(true_z), -1, np.int64)
    for ci in order:
        dz = np.abs(cand_z[ci] - true_z) / (1.0 + true_z)
        dz[used_true] = np.inf
        j = int(np.argmin(dz)) if len(dz) else -1
        if j >= 0 and dz[j] < DZ_REL:
            used_true[j] = True
            match_idx[j] = ci
    return match_idx


def join_injections(rows, truth, p_cut=P_CUT, nhi_floor=NHI_PRED_FLOOR,
                    require_flag0=True):
    """truth: structured/dict-like with int64 TARGETID, z_inj, logN arrays.
    Returns per-injection (rec, dlogN, dv_kms, matched_nhi, matched_p)."""
    acc = (rows["P_DLA"] > p_cut) & (rows["NHI"] > nhi_floor)
    if require_flag0:
        acc &= (rows["DLAFLAG"] == 0)
    arows = rows[acc]
    tids = np.asarray(truth["TARGETID"], np.int64)
    z_inj = np.asarray(truth["z_inj"], float)
    logn = np.asarray(truth["logN"], float)
    n = len(tids)
    rec = np.zeros(n, bool)
    dlogN = np.full(n, np.nan)
    dv = np.full(n, np.nan)
    by_tid = {}
    for i, t in enumerate(arows["TARGETID"]):
        by_tid.setdefault(int(t), []).append(i)
    for tid in np.unique(tids):
        tmask = tids == tid
        idxs = by_tid.get(int(tid), [])
        if not idxs:
            continue
        cz = arows["Z_DLA"][idxs]
        cn = arows["NHI"][idxs]
        m = greedy_match_sightline(cz, cn, z_inj[tmask], logn[tmask])
        ti = np.where(tmask)[0]
        for k, ci in enumerate(m):
            if ci >= 0:
                i_glob = ti[k]
                rec[i_glob] = True
                dlogN[i_glob] = cn[ci] - logn[i_glob]
                dv[i_glob] = 2.998e5 * (cz[ci] - z_inj[i_glob]) / (1.0 + z_inj[i_glob])
    return rec, dlogN, dv


def cell_of(snr, zq):
    si = np.digitize(snr, SNR_EDGES[1:3])   # 0,1,2
    zi = np.digitize(zq, Z_EDGES[1:3])
    return si, zi


def cell_name(si, zi):
    return f"s{si}z{zi}"


def ratio_summary(w, rec, mockC, sel):
    """Weighted completeness ratio in a selection mask."""
    w = w[sel]; r = rec[sel].astype(float); mc = mockC[sel]
    if w.sum() == 0:
        return None
    Cw = float(np.sum(w * r) / np.sum(w))
    mCw = float(np.sum(w * mc) / np.sum(w))
    # binomial SE on the weighted real completeness (effective n)
    neff = w.sum() ** 2 / np.sum(w ** 2)
    se_C = np.sqrt(max(Cw * (1 - Cw), 1e-12) / neff)
    ratio = Cw / mCw if mCw > 0 else np.nan
    return dict(n=int(sel.sum()), k=int(rec[sel].sum()),
                unw=float(rec[sel].mean()), Cw=round(Cw, 4),
                mockCw=round(mCw, 4), ratio=round(ratio, 4),
                ratio_se=round(se_C / mCw, 4) if mCw > 0 else None,
                neff=round(float(neff), 1))
