# REVIEW-ONLY (Phase A) — does not alter production behavior.
"""r03 — Estimand audit of the ceiling claim: "corrected mu_FP exceeds each
mock's unmatched-detection count (upper bound on forest-FP supply)".

Written independently of the previous session's probe path: the census below
re-derives the unmatched counts from the committed bundle loader with its OWN
vectorized ANY-host interval matcher (not the greedy 1-to-1), plus a
chance-coincidence rate estimate that the greedy matcher's bound direction
depends on.

Questions answered per mock:
  (a) SUPPORT — do "mu_FP" and "unmatched" live on the same support?
      mu_FP support: op cut (SNR>2, P_DLA>0.99, good_mask), N-hat in
      [19.5, 22.4), z in [2.0, 3.5), all SNR strata, dX>0 cells.
      The census here applies exactly that selection to the mock catalog.
  (b) ESTIMAND — 'unmatched' under the committed PRIMARY convention is
      "no greedy-1-to-1 truth match against truth FLOORED AT 19.5 within
      |dz|/(1+z_t) < 0.01".  The loa-0-comparable forest-FP estimand is
      "no physical host at ANY truth N (catalog floor 17.2)".  Both are
      computed, with the host-N histogram of the difference.
  (c) BOUND DIRECTION — two opposing contaminations:
        + unmatched gains rows whose host is real but < 19.5 (LLS-host
          channel) or whose truth was claimed by a stronger sibling row
          (multi-candidate channel)  -> unmatched OVER-counts forest FP;
        - a genuine forest FP within 0.01*(1+z_t) of an unclaimed >=19.5
          truth is chance-matched OUT of 'unmatched'  -> UNDER-count.
      The chance rate is estimated from the truth density per unit search
      path; the multi-candidate channel from the ANY-19.5-host-but-unmatched
      rows; availability from the claimed-truth fraction.
  (d) the corrected per-mock mu_FP (from the pack, = fp_w * sum fp_counts)
      against each census number, with the loa-0 Poisson width (10.6%), the
      t_sigma transport prior, and the (1-eta) omission for context.
"""
import os
import sys
import json
import time

import numpy as np

sys.path.insert(0, "/home/mfho/wt_review_phaseA")

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ep", "/home/mfho/wt_review_phaseA/CDDF_analysis/hbi_mcmc/extract_pack.py")
EP = importlib.util.module_from_spec(_spec)
sys.modules["ep"] = EP
_spec.loader.exec_module(EP)

import fitsio                     # noqa: E402
from astropy.table import Table   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = ("/tmp/claude-114399728/-home-mfho-desi-gpy-dla-detection/"
           "b10b5e23-575d-487e-811d-479f51611f63/scratchpad/r03_bundles")
os.makedirs(SCRATCH, exist_ok=True)

DZ_REL = 0.01
NHAT_LO, NHAT_HI = 19.5, 22.4
Z_LO, Z_HI = 2.0, 3.5

# corrected per-mock mu_FP totals measured in r01 (counting argument == fold)
MU_FP = {"2lpt0": 14767.961419068737, "london0": 14716.376940133037,
         "saclay0": 14707.062527716187}
N0 = 89.0
ETA_SUBDLA = 0.005756532459300326       # loa-0 product (r01)


def any_host_mask(c_tid, c_z, t_tid, t_z, t_nhi, floor, dz_rel=DZ_REL):
    """True for cat rows having ANY truth with N>=floor, same TID,
    |z_c - z_t| <= dz_rel*(1+z_t). Vectorized composite-key interval search."""
    keep = t_nhi >= floor - 1e-9
    tt, tz = t_tid[keep], t_z[keep]
    order = np.lexsort((tz, tt))
    tt, tz = tt[order], tz[order]
    key_t = tt.astype(np.float64) * 8.0 + tz
    key_c = c_tid.astype(np.float64) * 8.0 + c_z
    PAD = 0.06                     # > max dz 0.01*(1+4.5); < inter-TID gap
    lo = np.searchsorted(key_t, key_c - PAD)
    hi = np.searchsorted(key_t, key_c + PAD)
    n_pairs = hi - lo
    has = np.zeros(len(c_tid), dtype=bool)
    nz = np.where(n_pairs > 0)[0]
    if len(nz) == 0:
        return has
    reps = n_pairs[nz]
    cat_idx = np.repeat(nz, reps)
    # ragged index of the truth row per pair
    offs = np.concatenate([np.arange(l, h) for l, h in zip(lo[nz], hi[nz])])
    ok = (tt[offs] == c_tid[cat_idx]) & \
         (np.abs(c_z[cat_idx] - tz[offs]) <= dz_rel * (1.0 + tz[offs]))
    np.logical_or.at(has, cat_idx[ok], True)
    return has


def host_nhi_best(c_tid, c_z, t_tid, t_z, t_nhi, floor, dz_rel=DZ_REL):
    """Max host NHI within the window per cat row (NaN if none)."""
    keep = t_nhi >= floor - 1e-9
    tt, tz, tn = t_tid[keep], t_z[keep], t_nhi[keep]
    order = np.lexsort((tz, tt))
    tt, tz, tn = tt[order], tz[order], tn[order]
    key_t = tt.astype(np.float64) * 8.0 + tz
    key_c = c_tid.astype(np.float64) * 8.0 + c_z
    PAD = 0.06
    lo = np.searchsorted(key_t, key_c - PAD)
    hi = np.searchsorted(key_t, key_c + PAD)
    n_pairs = hi - lo
    best = np.full(len(c_tid), np.nan)
    nz = np.where(n_pairs > 0)[0]
    if len(nz) == 0:
        return best
    cat_idx = np.repeat(nz, n_pairs[nz])
    offs = np.concatenate([np.arange(l, h) for l, h in zip(lo[nz], hi[nz])])
    ok = (tt[offs] == c_tid[cat_idx]) & \
         (np.abs(c_z[cat_idx] - tz[offs]) <= dz_rel * (1.0 + tz[offs]))
    ci, hv = cat_idx[ok], tn[offs][ok]
    np.fmax.at(best, ci, hv)
    return best


out = {}
for mock in ("2lpt0", "london0", "saclay0"):
    t0 = time.time()
    b = EP.load_mock_bundle(mock, out_dir=SCRATCH)
    cat, op, truth = b["cat_cut"], b["op_mask"], b["truth_cut"]
    cfg = b["cfg"]

    c_tid = np.asarray(cat["TARGETID"], np.int64)
    c_z = np.asarray(cat["Z_DLA"], float)
    c_nhi = np.asarray(cat["NHI"], float)
    ntr = np.asarray(cat["NHI_TRUE"], float)         # committed primary match

    # full truth (catalog floor; measures its own floor)
    tf = Table(fitsio.read(cfg.truth_path, ext=1))
    zcol = next(c for c in ("Z_DLA", "Z_DLA_NO_RSD", "Z") if c in tf.colnames)
    t_tid = np.asarray(tf["TARGETID"], np.int64)
    t_z = np.asarray(tf[zcol], float)
    t_nhi = np.asarray(tf["NHI"], float)
    truth_file_floor = float(np.nanmin(t_nhi))

    # (a) the pack support
    sel = op & (c_nhi >= NHAT_LO) & (c_nhi < NHAT_HI) & \
        (c_z >= Z_LO) & (c_z < Z_HI)
    n_sel = int(sel.sum())

    # (b) estimands
    unm_primary = sel & ~np.isfinite(ntr)
    n_unm_primary = int(unm_primary.sum())
    has195 = any_host_mask(c_tid, c_z, t_tid, t_z, t_nhi, 19.5)
    has172 = any_host_mask(c_tid, c_z, t_tid, t_z, t_nhi, truth_file_floor)
    n_hostless195 = int((sel & ~has195).sum())
    n_hostless172 = int((sel & ~has172).sum())
    # multi-candidate channel: unmatched under greedy but a >=19.5 host exists
    n_sibling = int((unm_primary & has195).sum())
    # LLS-host channel: no >=19.5 host, but a sub-19.5 host exists
    n_lls_host = int((sel & ~has195 & has172).sum())
    hostN = host_nhi_best(c_tid, c_z, t_tid, t_z, t_nhi, truth_file_floor)
    lls_hist = {}
    m_lls = sel & ~has195 & has172
    for lo_, hi_ in ((17.0, 18.0), (18.0, 18.5), (18.5, 19.0), (19.0, 19.5)):
        lls_hist[f"[{lo_},{hi_})"] = int(
            (m_lls & (hostN >= lo_) & (hostN < hi_)).sum())

    # per-N-hat-bin unmatched (FP fold has NO support above n-hat 20.2)
    unm_by_nhat = {}
    for c0 in np.round(np.arange(19.5, 22.4, 0.1), 1):
        m = unm_primary & (c_nhi >= c0) & (c_nhi < c0 + 0.1)
        if m.sum():
            unm_by_nhat[f"{c0:.1f}"] = int(m.sum())
    n_unm_above202 = int((unm_primary & (c_nhi >= 20.2)).sum())

    # (c) chance-coincidence rate: truth window density per unit search path
    qzl, qzh = np.asarray(b["qzl"], float), np.asarray(b["qzh"], float)
    seg = np.clip(np.minimum(qzh, Z_HI) - np.maximum(qzl, Z_LO), 0.0, None)
    total_dz = float(seg.sum())
    tz_c = np.asarray(truth["Z_DLA"], float)
    in_win = (tz_c >= Z_LO) & (tz_c < Z_HI)
    p_chance_195 = float((2.0 * DZ_REL * (1.0 + tz_c[in_win])).sum() / total_dz)
    # availability of >=19.5 truths for a chance claim (greedy 1-to-1)
    n_claimed = int(np.isfinite(ntr).sum())
    availability = 1.0 - n_claimed / max(len(truth), 1)
    p_chance_eff = p_chance_195 * availability

    mu = MU_FP[mock]
    rec = dict(
        n_cat_cut=len(cat), n_op=int(op.sum()), n_sl=int(b["n_sl"]),
        n_truth_cut_ge195=len(truth), truth_file_floor=truth_file_floor,
        n_sel_pack_support=n_sel,
        unmatched_primary=n_unm_primary,
        hostless_any195=n_hostless195,
        hostless_any172_forestFP_candidates=n_hostless172,
        sibling_channel_unm_with_195_host=n_sibling,
        lls_host_channel=n_lls_host,
        lls_host_hist=lls_hist,
        unm_by_nhat=unm_by_nhat,
        unmatched_above_nhat_20p2=n_unm_above202,
        chance_match=dict(p_upper=p_chance_195, availability=availability,
                          p_eff=p_chance_eff,
                          total_search_dz=total_dz),
        mu_fp_corrected=mu,
        excess_vs_unmatched_primary=mu / n_unm_primary - 1.0,
        excess_vs_hostless172=mu / n_hostless172 - 1.0,
        mu_fp_poisson_rel_sd=1.0 / np.sqrt(N0 + 0.5),
        mu_fp_after_eta_and_chance=mu * (1.0 - ETA_SUBDLA)
        * (1.0 - p_chance_eff),
        secs=round(time.time() - t0, 1),
    )
    out[mock] = rec
    print(json.dumps({mock: rec}, indent=1, default=float), flush=True)

with open(os.path.join(HERE, "r03_out.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print("wrote r03_out.json")
