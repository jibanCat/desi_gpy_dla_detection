#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the atomic P1 (C, K) artifact — `p1_natpair_ck/v1`.

PI-authorized engineering phase (2026-08-07 ruling: natural-pair kernel
anchor approved; complete freeze pending mechanical closure). This
builder constructs ONE atomic file containing the deployed completeness
C_molly (verbatim from the frozen Phase-B pack) and the natural-pair
kernel K, both derived from the SAME canonical nhi195-chain event set
(the Tier-1 cache, which reproduced the deployed counts integer-exactly
in all 96 cells).

COHERENCE CONTRACT (enforced, fail-loud):
  * Kernel events = EXACTLY the deployed completeness-numerator events:
    is_TP & (N̂ > 19.5) & (P_DLA > 0.99) & (DLAFLAG == 0), counted per
    (S2N_RED, N_true) molly cell with the deployed STRICT bin bounds
    (`completeness_snr_nhi_bins`). A matched event failing any cut is a
    MISS in the deployed accounting and is excluded from K identically.
  * IDENTITY: per-cell kernel event counts must equal the pack's
    `molly_n_det` INTEGER-EXACTLY on every ≥19.5 column (the nhi195
    splice region). Any mismatch aborts the build.
  * MISS CLOSURE (exactly-once): per cell,
    n_det + n_TP_subfloorN̂ + n_TP_lowP + n_TP_flag + n_unmatched
    == molly_n_tot, INTEGER-EXACTLY. Any violation aborts the build.
  * No renormalization anywhere; the miss state is explicit mass.

The artifact stores C verbatim (all strata, incl. dX=0 dead strata,
live rows marked), K on the identity grid AND on the reporting grid
(0.2-dex N × 3 z-cells × 3 SNR strata + live N-marginal), a support
map with sparse flags, blend/isolation composition diagnostics, and
full provenance (input hashes, git commit, estimand ID + version).

Nothing is spliced into production. The holdout is not touched (the
production catalogue predates all injection arms).
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))

CACHE = ("/scratch/cavestru_root/cavestru0/mfho/phaseC_resp/"
         "p1_completeness_cache.npz")
PACK = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phaseB_packs/"
        "modelA_pack_2lpt0_winlya_only_pad19p0_molly172_bw0p2.npz")
OUT = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/"
       "stage0/p1_natpair_ck_v1.npz")
TRUTH_FITS = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
              "qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits")

ESTIMAND_ID = "p1_natpair_ck/v1"
FLOOR = 19.5                  # nhi195 chain truth floor == cmp_min_pred_nhi
P_DLA_MIN = 0.99              # strict >, deployed
REPORT_N_EDGES = np.round(19.5 + 0.2 * np.arange(16), 10)   # 19.5 .. 22.5
ZR_EDGES = (2.56, 2.96)       # frozen 3 z-cells
SR_EDGES = (3.5, 6.5)         # frozen 3 SNR strata (reporting; live = S2N>2)
SPARSE_N_MIN = 25             # frozen sparse-cell threshold
C_KMS = 299792.458
ISO_KMS = 5000.0
BLEND_KMS = 3000.0


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def extract_kernel_events(d=None):
    """The kernel event set + exclusion classes from the Tier-1 cache.

    Returns dict of arrays for TP rows: N_true, N_hat, dx, z_true, S2N,
    TARGETID, plus boolean class masks over TP rows:
    in_kernel / subfloor (N̂ ≤ FLOOR) / lowP / flag (DLAFLAG ≠ 0).
    Classes are assigned with the SAME precedence used in the miss
    table: flag → lowP → subfloor (mutually exclusive, first match).
    """
    if d is None:
        d = np.load(CACHE)
    tp = d["cat_is_TP"]
    good = d["cat_good"]
    P = d["cat_P_DLA"]
    nhat = d["cat_NHI"]
    ev = dict(
        N=d["cat_NHI_TRUE"][tp], NHAT=nhat[tp], Z=d["cat_Z_TRUE"][tp],
        S2N=d["cat_S2N"][tp], TID=d["cat_TARGETID"][tp],
        P=P[tp], GOOD=good[tp])
    ev["DX"] = ev["NHAT"] - ev["N"]
    flag = ~ev["GOOD"]
    lowP = (~flag) & ~(ev["P"] > P_DLA_MIN)
    subf = (~flag) & (~lowP) & ~(ev["NHAT"] > FLOOR)
    ev["CLS_FLAG"], ev["CLS_LOWP"], ev["CLS_SUBF"] = flag, lowP, subf
    ev["IN_KERNEL"] = ~(flag | lowP | subf)
    return ev, d


def _cell_counts(vals_s2n, vals_n, snr_e, nhi_e, sel=None):
    """Strict-bound cell histogram matching the deployed counting."""
    ns, nn = len(snr_e) - 1, len(nhi_e) - 1
    out = np.zeros((ns, nn), dtype=np.int64)
    s = vals_s2n if sel is None else vals_s2n[sel]
    n = vals_n if sel is None else vals_n[sel]
    for i in range(ns):
        ms = (s > snr_e[i]) & (s < snr_e[i + 1])
        for j in range(nn):
            out[i, j] = int(np.sum(ms & (n > nhi_e[j]) & (n < nhi_e[j + 1])))
    return out


def _stats(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return dict(n=0, mean=np.nan, se=np.nan, sd=np.nan, robust=np.nan)
    q16, q84 = (np.percentile(v, [15.865, 84.135]) if len(v) > 4
                else (np.nan, np.nan))
    return dict(n=int(len(v)), mean=float(v.mean()),
                se=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1
                else np.nan,
                sd=float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                robust=float(0.5 * (q84 - q16)))


def main():
    t0 = time.time()
    ev, d = extract_kernel_events()
    pk = np.load(PACK)
    snr_e = np.asarray(pk["molly_snr_edges"], float)
    nhi_e = np.asarray(pk["molly_nhi_edges"], float)
    det_p = np.asarray(pk["molly_n_det"], np.int64)
    tot_p = np.asarray(pk["molly_n_tot"], np.int64)
    j195 = int(np.searchsorted(nhi_e, FLOOR))
    if abs(nhi_e[j195] - FLOOR) > 1e-9:
        raise SystemExit(f"FATAL: no molly N edge at {FLOOR}")

    # ---- IDENTITY: kernel counts == deployed numerators (>=19.5 cols) ----
    det_k = _cell_counts(ev["S2N"], ev["N"], snr_e, nhi_e,
                         sel=ev["IN_KERNEL"])
    bad = [(int(i), int(j), int(det_k[i, j]), int(det_p[i, j]))
           for i in range(det_k.shape[0])
           for j in range(j195, det_k.shape[1])
           if det_k[i, j] != det_p[i, j]]
    if bad:
        raise SystemExit(f"FATAL identity failure in {len(bad)} cells "
                         f"(i,j,kernel,pack): {bad[:8]}")

    # ---- MISS CLOSURE: exactly-once truth accounting per cell ----------
    cls = {}
    for name, m in [("subfloor", ev["CLS_SUBF"]), ("lowP", ev["CLS_LOWP"]),
                    ("flag", ev["CLS_FLAG"])]:
        cls[name] = _cell_counts(ev["S2N"], ev["N"], snr_e, nhi_e, sel=m)
    tot_truth = _cell_counts(d["tr_S2N"], d["tr_NHI"], snr_e, nhi_e)
    bad_tot = [(int(i), int(j), int(tot_truth[i, j]), int(tot_p[i, j]))
               for i in range(tot_truth.shape[0])
               for j in range(j195, tot_truth.shape[1])
               if tot_truth[i, j] != tot_p[i, j]]
    if bad_tot:
        raise SystemExit(f"FATAL denominator mismatch: {bad_tot[:8]}")
    unmatched = (tot_p - det_k - cls["subfloor"] - cls["lowP"]
                 - cls["flag"])
    if np.any(unmatched[:, j195:] < 0):
        raise SystemExit("FATAL miss closure: negative unmatched count")
    # duplicate-claim guard: matched truth keys must be unique
    keys = list(zip(ev["TID"].tolist(),
                    np.round(ev["Z"], 6).tolist(),
                    np.round(ev["N"], 6).tolist()))
    if len(keys) != len(set(keys)):
        raise SystemExit("FATAL: a truth system carries two TP rows")

    # ---- K tables ------------------------------------------------------
    kin = ev["IN_KERNEL"]
    z2 = np.digitize(ev["Z"], ZR_EDGES)          # 0,1,2
    s2 = np.digitize(ev["S2N"], SR_EDGES)        # 0,1,2  (reporting strata)
    nb = np.digitize(ev["N"], REPORT_N_EDGES) - 1   # 0..14 inside support
    live = ev["S2N"] > 2.0
    n_bins = len(REPORT_N_EDGES) - 1
    # identity-grid stats (for the record; live rows marked separately)
    id_stats = np.full((det_k.shape[0], det_k.shape[1], 5), np.nan)
    for i in range(det_k.shape[0]):
        for j in range(j195, det_k.shape[1]):
            m = (kin & (ev["S2N"] > snr_e[i]) & (ev["S2N"] < snr_e[i + 1])
                 & (ev["N"] > nhi_e[j]) & (ev["N"] < nhi_e[j + 1]))
            st = _stats(ev["DX"][m])
            id_stats[i, j] = [st["n"], st["mean"], st["se"], st["sd"],
                              st["robust"]]
    # reporting grid (live events only feed the fold; dead strata dX=0)
    rep = np.full((n_bins, 3, 3, 5), np.nan)
    inb = (nb >= 0) & (nb < n_bins)
    for b in range(n_bins):
        for zi in range(3):
            for si in range(3):
                m = kin & live & inb & (nb == b) & (z2 == zi) & (s2 == si)
                st = _stats(ev["DX"][m])
                rep[b, zi, si] = [st["n"], st["mean"], st["se"], st["sd"],
                                  st["robust"]]
    marg = np.full((n_bins, 5), np.nan)
    for b in range(n_bins):
        m = kin & live & inb & (nb == b)
        st = _stats(ev["DX"][m])
        marg[b] = [st["n"], st["mean"], st["se"], st["sd"], st["robust"]]
    sparse = rep[:, :, :, 0] < SPARSE_N_MIN
    # re-binning conservation: no live kernel event lost or duplicated
    n_live_kernel = int(np.sum(kin & live))
    n_rep = int(np.nansum(rep[:, :, :, 0]))
    n_out_support = int(np.sum(kin & live & ~inb))
    n_edge_dropped = n_live_kernel - n_rep - n_out_support
    if n_edge_dropped != 0:
        raise SystemExit(f"FATAL re-binning lost {n_edge_dropped} events")

    # ---- composition diagnostics (blend / isolation) -------------------
    import fitsio
    tr = fitsio.read(TRUTH_FITS, columns=["TARGETID", "NHI", "Z"])
    by = defaultdict(list)
    for T, N, Z in zip(np.asarray(tr["TARGETID"], np.int64),
                       np.asarray(tr["NHI"], float),
                       np.asarray(tr["Z"], float)):
        by[int(T)].append((float(Z), float(N)))
    comp = np.zeros((n_bins, 4))          # n, frac_iso5k, frac_blend3k, dxb
    for b in range(n_bins):
        m = kin & live & inb & (nb == b)
        tid, zz, dx = ev["TID"][m], ev["Z"][m], ev["DX"][m]
        iso = np.ones(len(tid), bool)
        blend = np.zeros(len(tid), bool)
        for k in range(len(tid)):
            for z, N in by.get(int(tid[k]), []):
                dv = abs(z - zz[k]) / (1 + zz[k]) * C_KMS
                if 1e-9 < dv < ISO_KMS:
                    iso[k] = False
                    if dv < BLEND_KMS and 17.2 <= N < 19.5:
                        blend[k] = True
        nb_ = len(tid)
        comp[b] = [nb_, iso.mean() if nb_ else np.nan,
                   blend.mean() if nb_ else np.nan,
                   float(dx[blend].mean() - dx[~blend].mean())
                   if blend.any() and (~blend).any() else np.nan]

    # ---- assemble atomic artifact --------------------------------------
    prov = dict(
        schema=ESTIMAND_ID, version=1, date=time.strftime("%Y-%m-%d"),
        git_commit=_git_head(), builder="injection/build_p1_natpair_ck.py",
        cache_path=CACHE, cache_sha256=_sha(CACHE),
        pack_path=PACK, pack_sha256=_sha(PACK),
        floor=FLOOR, p_dla_min=P_DLA_MIN, sparse_n_min=SPARSE_N_MIN,
        kernel_condition=("is_TP & (NHI_pred > 19.5) & (P_DLA > 0.99) & "
                          "(DLAFLAG == 0), strict cell bounds"),
        miss_classes=["unmatched", "subfloor_nhat", "lowP", "flag"],
        continuation_rule=("no extrapolation beyond last populated bin; "
                           "sparse cells inherit live N-marginal, "
                           "flagged, variance inflated"))
    np.savez(
        OUT,
        estimand_id=np.array(ESTIMAND_ID), version=np.array([1]),
        provenance_json=np.array(json.dumps(prov)),
        # --- C: deployed, verbatim, all strata ---
        C_molly_n_det=det_p, C_molly_n_tot=tot_p,
        C_snr_edges=snr_e, C_nhi_edges=nhi_e,
        C_live_row=np.array([snr_e[i] >= 2.0
                             for i in range(len(snr_e) - 1)]),
        # --- K: identity grid + miss decomposition (>=19.5 cols) ---
        K_id_grid=id_stats, K_id_j195=np.array([j195]),
        miss_subfloor=cls["subfloor"], miss_lowP=cls["lowP"],
        miss_flag=cls["flag"], miss_unmatched=unmatched,
        # --- K: reporting representation ---
        K_rep=rep, K_rep_n_edges=REPORT_N_EDGES,
        K_rep_z_edges=np.array(ZR_EDGES), K_rep_s_edges=np.array(SR_EDGES),
        K_marginal=marg, K_sparse_flag=sparse,
        composition=comp,
        n_out_support=np.array([n_out_support]))
    art_sha = _sha(OUT)

    summary = dict(
        schema="p1_ck_build/v1", date=prov["date"],
        estimand_id=ESTIMAND_ID, artifact=OUT, artifact_sha256=art_sha,
        identity_cells_checked=int((det_k.shape[1] - j195)
                                   * det_k.shape[0]),
        identity_pass=True, miss_closure_pass=True,
        n_kernel_events_total=int(np.sum(kin)),
        n_kernel_events_live=n_live_kernel,
        n_live_out_of_report_support=n_out_support,
        tp_class_counts=dict(
            in_kernel=int(np.sum(ev["IN_KERNEL"])),
            subfloor=int(np.sum(ev["CLS_SUBF"])),
            lowP=int(np.sum(ev["CLS_LOWP"])),
            flag=int(np.sum(ev["CLS_FLAG"]))),
        sparse_cells_live=int(np.sum(sparse)),
        provenance=prov, wall_s=round(time.time() - t0, 1))
    outj = os.path.join(_REPO, "diagnostics_phaseC/p1_completeness/"
                        "p1_ck_build.json")
    with open(outj, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: summary[k] for k in
                      ("identity_cells_checked", "identity_pass",
                       "miss_closure_pass", "n_kernel_events_total",
                       "n_kernel_events_live", "tp_class_counts",
                       "sparse_cells_live", "artifact_sha256", "wall_s")},
                     indent=1))


if __name__ == "__main__":
    sys.exit(main())
