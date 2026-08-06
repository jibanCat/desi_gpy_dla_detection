# REVIEW-ONLY (Phase A) — does not alter production behavior.
"""r03b — Direct measurement of the chance-coincidence rate that decides the
ceiling claim's bound direction.

The ceiling claim's comparator (r03 shows it is 'hostless at the truth-file
floor ~17.2') can only be an UPPER bound on the forest-FP supply if genuine
forest FPs are almost never removed from it by CHANCE z-coincidence with a
truth system.  The truth catalog down to 17.2 is DENSE (~5x the >=19.5
population) and the match window |dz| <= 0.01*(1+z_t) is +-0.030-0.045 in z,
so this must be measured, not assumed.

METHOD (z-scramble): take the op detections on the pack support, displace
their redshifts by +-0.10 / +-0.15 (2-4x the match window, small vs the
forest window), and re-run the SAME any-host matcher.  A displaced position
retains its sightline and approximate z-weighting but destroys any physical
association, so its match rate IS the chance rate p_c.  Then
    S_forest_FP  =  hostless_172 / (1 - p_c172)
is the de-blurred empirical forest-FP supply, and the ceiling comparison is
mu_FP vs S, not mu_FP vs hostless_172.

Also emitted: the chance-expected host-N composition of the 'hosted but
unmatched-at-19.5' class, to decompose the LLS-host channel into chance
overlap vs genuine sub-floor hosts.
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

DZ_REL = 0.01
NHAT_LO, NHAT_HI = 19.5, 22.4
Z_LO, Z_HI = 2.0, 3.5
MU_FP = {"2lpt0": 14767.961419068737, "london0": 14716.376940133037,
         "saclay0": 14707.062527716187}
HOSTLESS_172 = {"2lpt0": 13491, "london0": 9253, "saclay0": 10273}   # r03
ETA_SUBDLA = 0.005756532459300326


def any_host_mask(c_tid, c_z, t_tid, t_z, t_nhi, floor, dz_rel=DZ_REL):
    keep = t_nhi >= floor - 1e-9
    tt, tz = t_tid[keep], t_z[keep]
    order = np.lexsort((tz, tt))
    tt, tz = tt[order], tz[order]
    key_t = tt.astype(np.float64) * 8.0 + tz
    key_c = c_tid.astype(np.float64) * 8.0 + c_z
    PAD = 0.06
    lo = np.searchsorted(key_t, key_c - PAD)
    hi = np.searchsorted(key_t, key_c + PAD)
    n_pairs = hi - lo
    has = np.zeros(len(c_tid), dtype=bool)
    nz = np.where(n_pairs > 0)[0]
    if len(nz) == 0:
        return has
    cat_idx = np.repeat(nz, n_pairs[nz])
    offs = np.concatenate([np.arange(l, h) for l, h in zip(lo[nz], hi[nz])])
    ok = (tt[offs] == c_tid[cat_idx]) & \
         (np.abs(c_z[cat_idx] - tz[offs]) <= dz_rel * (1.0 + tz[offs]))
    np.logical_or.at(has, cat_idx[ok], True)
    return has


out = {}
for mock in ("2lpt0", "london0", "saclay0"):
    t0 = time.time()
    b = EP.load_mock_bundle(mock, out_dir=SCRATCH)
    cat, op = b["cat_cut"], b["op_mask"]
    cfg = b["cfg"]
    c_tid = np.asarray(cat["TARGETID"], np.int64)
    c_z = np.asarray(cat["Z_DLA"], float)
    c_nhi = np.asarray(cat["NHI"], float)

    tf = Table(fitsio.read(cfg.truth_path, ext=1))
    zcol = next(c for c in ("Z_DLA", "Z_DLA_NO_RSD", "Z") if c in tf.colnames)
    t_tid = np.asarray(tf["TARGETID"], np.int64)
    t_z = np.asarray(tf[zcol], float)
    t_nhi = np.asarray(tf["NHI"], float)
    floor172 = float(np.nanmin(t_nhi))

    sel = op & (c_nhi >= NHAT_LO) & (c_nhi < NHAT_HI) & \
        (c_z >= Z_LO) & (c_z < Z_HI)

    # ---- z-scramble chance rates -------------------------------------------
    rates172, rates195 = [], []
    per_off = {}
    for off in (0.10, -0.10, 0.15, -0.15):
        zs = c_z + off
        keep = sel & (zs >= Z_LO) & (zs < Z_HI)
        h172 = any_host_mask(c_tid[keep], zs[keep], t_tid, t_z, t_nhi, floor172)
        h195 = any_host_mask(c_tid[keep], zs[keep], t_tid, t_z, t_nhi, 19.5)
        r172 = float(h172.mean())
        r195 = float(h195.mean())
        rates172.append(r172)
        rates195.append(r195)
        per_off[f"{off:+.2f}"] = dict(n=int(keep.sum()), p172=r172, p195=r195)
    p172 = float(np.mean(rates172))
    p195 = float(np.mean(rates195))

    # ---- chance-expected host-N composition (per truth band weights) -------
    # weight ∝ sum over in-window truth of the match-window width 2*dz*(1+z)
    cat_tids = np.unique(c_tid)
    on_searched = np.isin(t_tid, cat_tids)
    tw = on_searched & (t_z >= Z_LO) & (t_z < Z_HI)
    band_w = {}
    for lo_, hi_ in ((17.2, 18.0), (18.0, 18.5), (18.5, 19.0), (19.0, 19.5)):
        m = tw & (t_nhi >= lo_ - 1e-9) & (t_nhi < hi_)
        band_w[f"[{lo_},{hi_})"] = float(
            (2.0 * DZ_REL * (1.0 + t_z[m])).sum())
    wsum = sum(band_w.values())
    band_frac = {k: v / wsum for k, v in band_w.items()}

    # ---- the corrected supply and the re-done ceiling comparison -----------
    S = HOSTLESS_172[mock] / (1.0 - p172)
    mu = MU_FP[mock]
    rec = dict(
        chance_p172_scrambled=p172, chance_p195_scrambled=p195,
        per_offset=per_off,
        hostless_172=HOSTLESS_172[mock],
        forest_fp_supply_chance_corrected=S,
        mu_fp_corrected=mu,
        mu_over_hostless_172=mu / HOSTLESS_172[mock],
        mu_over_supply_corrected=mu / S,
        mu_after_eta=mu * (1.0 - ETA_SUBDLA),
        chance_band_window_fracs_sub195=band_frac,
        secs=round(time.time() - t0, 1),
    )
    out[mock] = rec
    print(json.dumps({mock: rec}, indent=1, default=float), flush=True)

with open(os.path.join(HERE, "r03b_out.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print("wrote r03b_out.json")
