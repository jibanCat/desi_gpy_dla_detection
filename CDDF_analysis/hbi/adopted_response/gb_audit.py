#!/usr/bin/env python
"""G-B atomic C/K event-set certification (PI ruling item 4). Env gpdla.
PASS is a prerequisite for real-data inference; FAIL => stop, return to PI.

Checks (2LPT-0, adopted lya_only-195 configuration):
 B1. C-side regen: rebuild the molly (n_found, n_fid) counts fresh via the
     committed regenerate_molly_counts path; compare INTEGER-EXACT against
     (a) the cached npz the pack consumed and (b) the pack's
     molly_n_det/molly_n_tot on the >=19.5 (195-chain) cells.
 B2. Event-level C/K identity per molly cell m>=19.5:
       {kernel events: op-cut & tilt-host in cell & xhat>=19.5}
    == {C-numerator events: op-cut & is_TP & NHI_TRUE in cell & xhat>19.5}
     with zero unexplained members either way.
 B3. The contract statement domain: report phi_ref per cell so the exact
     R = C.K region (phi ~ 1, N>=19.8) is documented against the ruled
     low-N convention region.
Writes gb_audit.json.
"""
import json, os, sys
import numpy as np

REPO = "/home/mfho/wt_forward_2026_08"
sys.path.insert(0, REPO)
os.chdir(REPO)
HERE = os.path.dirname(os.path.abspath(__file__))

import CDDF_analysis.hbi.track_c_tf_loa as TF
import CDDF_analysis.hbi.ab_loa0_fp_baseline as AB
from CDDF_analysis.hbi.cddf_catalog_hbi import (
    HBIConfig, load_molly_matrix, load_and_cut_catalog, _build_qso_lookup,
    regenerate_molly_counts)
from CDDF_analysis.hbi_mcmc.pack import load_pack

CACHE = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
         "ff_fp_cache/molly_counts_2lpt0_lyaonly195.npz")
PACK = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
        "adopted_packs_20260816/modelA_pack_2lpt0_bw0p2_pad19p0_molly172.npz")

class _A:
    molly_tsv = None
molly_tsv = AB._resolve_molly(_A)
cfg = HBIConfig(catalog_dir=TF._C0_CAT, truth_path=TF._C0_TRUTH,
                bal_cat_path=TF._C0_BAL, molly_tsv=molly_tsv,
                out_dir="/tmp", mockdir=os.path.dirname(TF._C0_TRUTH),
                fp_estimator="purity_mixture", no_bal=True, lam_rf_min=1025.0)
mm = load_molly_matrix(molly_tsv)
floor = float(mm.nhi_edges[0])
qso_lookup = _build_qso_lookup(cfg)
cat_cut, truth_cut, is_TP, good_mask, meta = load_and_cut_catalog(
    cfg, truth_nhi_floor=floor, qso_lookup=qso_lookup,
    host_truth_floor=19.0)
res = {"config": dict(molly_tsv=molly_tsv, matrix_floor=floor,
                      host_truth_floor=19.0)}

# ---- B1: fresh regen vs cache vs pack ------------------------------------
mm = regenerate_molly_counts(mm, cat_cut, is_TP, truth_cut, good_mask, cfg)
res["B1_ratio_guard"] = dict(max_p_diff=float(mm._max_p_diff),
                             max_c_diff=float(mm._max_c_diff),
                             threshold=5e-3,
                             status="PASS" if max(mm._max_p_diff,
                                                  mm._max_c_diff) <= 5e-3
                             else "FAIL")
ch = np.load(CACHE, allow_pickle=True)
print("cache keys:", list(ch.keys()), flush=True)
nd_cache = np.asarray(ch[[k for k in ch.files if "det" in k or "found" in k][0]])
nt_cache = np.asarray(ch[[k for k in ch.files if "tot" in k or "fid" in k][0]])
b1_cache_nd = bool(np.array_equal(np.asarray(mm.cmp_nfound, int), nd_cache.astype(int)))
b1_cache_nt = bool(np.array_equal(np.asarray(mm.cmp_nfid, int), nt_cache.astype(int)))
pk = load_pack(PACK)
pnd = np.asarray(pk.molly_n_det); pnt = np.asarray(pk.molly_n_tot)
pedges = np.asarray(pk.molly_nhi_edges, float)
# map pack cells (spliced grid) to the 195-chain cells for j >= floor
j195 = [j for j in range(len(pedges) - 1) if pedges[j] >= floor - 1e-9]
mmj = {round(float(mm.nhi_edges[j]), 3): j for j in range(len(mm.nhi_edges) - 1)}
pairs = [(jp, mmj[round(float(pedges[jp]), 3)]) for jp in j195
         if round(float(pedges[jp]), 3) in mmj]
snr_ok = np.array_equal(np.asarray(mm.snr_edges, float),
                        np.asarray(pk.molly_snr_edges, float)) \
    if pk.molly_snr_edges is not None else None
d_nd = max(abs(int(pnd[i, jp]) - int(mm.cmp_nfound[i, jm]))
           for i in range(pnd.shape[0]) for jp, jm in pairs)
d_nt = max(abs(int(pnt[i, jp]) - int(mm.cmp_nfid[i, jm]))
           for i in range(pnt.shape[0]) for jp, jm in pairs)
res["B1_integer_exact"] = dict(
    fresh_vs_cache_ndet=b1_cache_nd, fresh_vs_cache_ntot=b1_cache_nt,
    fresh_vs_pack_maxdiff_ndet=d_nd, fresh_vs_pack_maxdiff_ntot=d_nt,
    n_cells_compared=len(pairs) * pnd.shape[0],
    snr_edges_equal=snr_ok,
    status="PASS" if (b1_cache_nd and b1_cache_nt and d_nd == 0
                      and d_nt == 0) else "FAIL")
print("B1:", res["B1_integer_exact"], flush=True)

# ---- B2: event-level identity per >=19.5 cell ----------------------------
s2n = np.asarray(cat_cut["S2N_RED"], float)
pdla = np.asarray(cat_cut["P_DLA"], float)
xhat = np.asarray(cat_cut["NHI"], float)
ntr = np.asarray(cat_cut["NHI_TRUE"], float)
tilt = np.asarray(cat_cut["NHI_TILT_HOST"], float)
tid = np.asarray(cat_cut["TARGETID"], np.int64)
zd = np.asarray(cat_cut["Z_DLA"], float)
op = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask
tp = np.asarray(is_TP, bool)

mism_total, cells_checked, kernel_total = 0, 0, 0
mism_examples = []
for i in range(len(mm.snr_edges) - 1):
    s_lo, s_hi = mm.snr_edges[i], mm.snr_edges[i + 1]
    in_s = (s2n > s_lo) & (s2n < s_hi)
    for j in range(len(mm.nhi_edges) - 1):
        n_lo, n_hi = mm.nhi_edges[j], mm.nhi_edges[j + 1]
        if n_lo < floor - 1e-9:
            continue
        c_set = op & tp & in_s & (ntr > n_lo) & (ntr < n_hi) & (xhat > floor)
        k_set = op & in_s & (tilt > n_lo) & (tilt < n_hi) & (xhat >= floor)
        diff = int((c_set != k_set).sum())
        cells_checked += 1
        kernel_total += int(k_set.sum())
        if diff:
            mism_total += diff
            idx = np.where(c_set != k_set)[0][:3]
            for k in idx:
                mism_examples.append(dict(
                    cell=[i, j], tid=int(tid[k]), z=round(float(zd[k]), 4),
                    xhat=round(float(xhat[k]), 3),
                    nhi_true=round(float(ntr[k]), 3),
                    tilt=round(float(tilt[k]), 3),
                    in_C=bool(c_set[k]), in_K=bool(k_set[k])))
res["B2_event_identity"] = dict(
    cells_checked=cells_checked, kernel_events_total=kernel_total,
    mismatched_events=mism_total,
    examples=mism_examples[:10],
    status="PASS" if mism_total == 0 else "FAIL")
print("B2:", {k: v for k, v in res["B2_event_identity"].items()
              if k != "examples"}, flush=True)

# ---- B3: contract domain -------------------------------------------------
from CDDF_analysis.hbi_mcmc.count_conserving_fold import phi_from_surfaces
phi = phi_from_surfaces(pk)
ntrue = np.asarray(pk.ntrue_edges, float)
Nc = 0.5 * (ntrue[:-1] + ntrue[1:])
res["B3_domain"] = dict(
    note=("R = C_op.K~ holds EXACTLY (B1+B2) on the cells with phi ~ 1; "
          "below N~19.8 the pairing is the ruled low-N convention with its "
          "named envelope"),
    phi_meancell_per_bin={f"{Nc[b]:.2f}": round(float(phi[:, :, b].mean()), 4)
                          for b in range(len(Nc)) if Nc[b] < 20.0})
overall = (res["B1_ratio_guard"]["status"] == "PASS"
           and res["B1_integer_exact"]["status"] == "PASS"
           and res["B2_event_identity"]["status"] == "PASS")
res["G_B_STATUS"] = "PASS" if overall else "FAIL"
json.dump(res, open(os.path.join(HERE, "gb_audit.json"), "w"), indent=1)
print("G-B:", res["G_B_STATUS"])
