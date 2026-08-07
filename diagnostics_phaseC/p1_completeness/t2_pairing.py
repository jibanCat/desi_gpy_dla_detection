#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tier-2 completion — the bounded catalog-level discrimination test.

THE SOLE GATE (handoff, PI checkpoint 3): attribute or bound the
pair-level natural−injected dx offset (+0.00…+0.06, largest +0.059 at
21.0) between the two §18 candidate mechanisms, using catalogs only.

PREDICTED SIGNATURES (stated BEFORE the run):
* HOST-ENVIRONMENT COUPLING (natural HCDs sit in correlated forest
  overdensities that inflate fitted N̂; injections have no host halo):
  (D1) the reweighted offset RISES with host N_true; and — the sharper
  discriminant — (D2) natural dx rises with the CATALOGUED absorber
  density in the 5,000–10,000 km/s shell around the host (an
  environment proxy OUTSIDE the blend/matching window) with a STEEPER
  slope than injected dx shows against the same shell density on its
  sightlines (the shared shell-absorption effect cancels in the slope
  DIFFERENCE; the host's own correlated halo does not).
* IMPRINT REALISM (quickquasars' imprint vs inject_voigt's profile at
  fixed catalog NHI): the offset is FLAT in shell density at fixed N
  (slope difference ≈ 0), with any N-dependence smooth and
  environment-independent.
VERDICT RULE (frozen here): coupling-supported if the natural-minus-
injected shell-density slope difference is positive at ≥2σ AND D1 rises;
imprint-supported if the slope difference is consistent with 0 (|z|<2)
while the reweighted offset stays significant; otherwise
NOT-SEPARABLE-AT-CATALOG-LEVEL → STOP, return the Tier-3 budget to the
PI. No forced fits; no holdout rows (roles bridge/production-calibration
only, holdout healpix excluded by the roles file); no Stage-2B; no P2.

Common-substrate/reweighting discipline (rulings §16–§17): natural
pairs restricted to ISOLATED (no catalogued ≥17.2 neighbor within
5,000 km/s — mirroring the injection eligibility exactly); comparison
per (response z-cell × SNR stratum) with the offset combined using the
INJECTED cells' weights (pre-selection covariates only: z-cell,
stratum, N anchor).
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import fitsio

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "injection"))

from examples.molly_faithful_pc_plots import match_truth_to_cat_molly  # noqa: E402
from gen_phaseC_resp import analysis_window, ZQSO_MIN, ZQSO_MAX        # noqa: E402
from astropy.table import Table, vstack                                # noqa: E402
import glob as _glob                                                   # noqa: E402

CACHE = ("/scratch/cavestru_root/cavestru0/mfho/phaseC_resp/"
         "p1_completeness_cache.npz")
ARM = "/scratch/cavestru_root/cavestru0/mfho/phaseC_resp/prod_v1"
ROLES = os.path.join(_REPO, "diagnostics_phaseC/stage2A/roles_prod_v1.json")
M124 = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
        "qq_desi_y3/v2.8.5/mock-0/loa-124")
C_KMS = 299792.458
DZ_REL = 0.01
ISO_KMS = 5000.0
SHELL = (5000.0, 10000.0)
NB = [(20.4, 20.7), (20.7, 21.0), (21.0, 21.3), (21.3, 21.7)]
s2sr = lambda s: 0 if s <= 3.5 else (1 if s <= 6.5 else 2)
z2zr = lambda z: 0 if z < 2.56 else (1 if z < 2.96 else 2)


def truth_neighbors():
    tr = fitsio.read(M124 + "/hcd_truth_cat.fits",
                     columns=["TARGETID", "NHI", "Z"])
    by = defaultdict(list)
    for T, N, Z in zip(np.asarray(tr["TARGETID"], np.int64),
                       np.asarray(tr["NHI"], float),
                       np.asarray(tr["Z"], float)):
        by[int(T)].append((float(Z), float(N)))
    return by


def shell_count(by, T, Z):
    n_iso = n_shell = 0
    for z, N in by.get(int(T), []):
        dv = abs(z - Z) / (1 + Z) * C_KMS
        if 1e-9 < dv < ISO_KMS:
            n_iso += 1
        elif SHELL[0] <= dv < SHELL[1]:
            n_shell += 1
    return n_iso, n_shell


def injected_pairs(by):
    """Per-pair injected records (roles bridge+production only)."""
    roles = json.load(open(ROLES))
    rmap = {int(k): v["role"] for k, v in roles["roles"].items()}
    man = Table.read(os.path.join(ARM, "injection_truth.fits"))
    keep = np.array([rmap[int(i)] in ("bridge", "production-calibration")
                     for i in man["inj_id"]])
    man = man[keep]
    dla = vstack([Table.read(p) for p in sorted(
        _glob.glob(os.path.join(ARM, "gp_out", "dlacat-*.fits")))],
        metadata_conflicts="silent")
    if "NHI_ERR" in dla.colnames:
        s = ((np.asarray(dla["NHI_ERR"], float) == -1)
             | (np.asarray(dla["Z_DLA_ERR"], float) == -1))
        dla = dla[~s]
    cat = Table()
    cat["TARGETID"] = np.asarray(dla["TARGETID"], np.int64)
    cat["Z_DLA"] = np.asarray(dla["Z_DLA"], float)
    cat["NHI"] = np.asarray(dla["NHI"], float)
    cat["P_DLA"] = np.asarray(dla["P_DLA"], float)
    truth = Table()
    truth["TARGETID"] = np.asarray(man["target_id"], np.int64)
    truth["Z_TRUTH"] = np.asarray(man["z_true"], float)
    truth["NHI"] = np.asarray(man["logN_true"], float)
    is_tp, nhi_tr, z_tr, _ = match_truth_to_cat_molly(
        cat, truth, DZ_REL, cat_iter_order="nhi_desc")
    tid2 = {int(t): (float(s), float(zq)) for t, s, zq in
            zip(man["target_id"], man["native_snr"], man["z_qso"])}
    recs = []
    dflag = (np.asarray(dla["DLAFLAG"], float) == 0
             if "DLAFLAG" in dla.colnames else np.ones(len(dla), bool))
    for ci in np.where(is_tp)[0]:
        T = int(cat["TARGETID"][ci])
        snr, zq = tid2[T]
        lo, hi = analysis_window(zq)
        zd = float(cat["Z_DLA"][ci])
        if not (float(cat["P_DLA"][ci]) > 0.99 and snr > 2.0
                and dflag[ci] and ZQSO_MIN < zq < ZQSO_MAX
                and lo < zd < hi):
            continue
        _, nsh = shell_count(by, T, float(z_tr[ci]))
        recs.append(dict(N=float(nhi_tr[ci]), z=float(z_tr[ci]), snr=snr,
                         dx=float(cat["NHI"][ci]) - float(nhi_tr[ci]),
                         shell=nsh))
    return recs


def natural_pairs(by):
    d = np.load(CACHE)
    sel = (d["cat_is_TP"] & (d["cat_P_DLA"] > 0.99)
           & (d["cat_S2N"] > 2.0))
    recs = []
    for T, N, Z, S, nh in zip(d["cat_TARGETID"][sel],
                              d["cat_NHI_TRUE"][sel],
                              d["cat_Z_TRUE"][sel], d["cat_S2N"][sel],
                              d["cat_NHI"][sel]):
        niso, nsh = shell_count(by, int(T), float(Z))
        if niso > 0:
            continue                       # ISOLATED mirror of injections
        recs.append(dict(N=float(N), z=float(Z), snr=float(S),
                         dx=float(nh) - float(N), shell=nsh))
    return recs


def wmean(v):
    v = np.asarray(v, float)
    return (float(v.mean()),
            float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan,
            len(v))


def main():
    t0 = time.time()
    by = truth_neighbors()
    inj = injected_pairs(by)
    nat = natural_pairs(by)
    print(f"pairs: injected {len(inj)}, natural-isolated {len(nat)}")

    out = {"schema": "p1_t2_pairing/v1", "date": time.strftime("%Y-%m-%d"),
           "n_injected": len(inj), "n_natural_isolated": len(nat),
           "D1_offset_by_hostN": [], "D2_shell_slopes": {},
           "verdict": None}

    # D1: reweighted offset per host-N bin (injected cell weights)
    for lo, hi in NB:
        cells = defaultdict(lambda: {"i": [], "n": []})
        for r in inj:
            if lo <= r["N"] < hi:
                cells[(z2zr(r["z"]), s2sr(r["snr"]))]["i"].append(r["dx"])
        for r in nat:
            if lo <= r["N"] < hi:
                cells[(z2zr(r["z"]), s2sr(r["snr"]))]["n"].append(r["dx"])
        num = den = var = 0.0
        for c, v in cells.items():
            if len(v["i"]) >= 3 and len(v["n"]) >= 3:
                w = len(v["i"])
                mi, si, _ = wmean(v["i"])
                mn, sn, _ = wmean(v["n"])
                num += w * (mn - mi)
                var += (w ** 2) * (si ** 2 + sn ** 2)
                den += w
        if den:
            off, soff = num / den, np.sqrt(var) / den
            out["D1_offset_by_hostN"].append(
                {"N": [lo, hi], "offset_nat_minus_inj": off,
                 "sigma": soff, "z": off / soff})
            print(f"D1 [{lo},{hi}): offset {off:+.4f} ± {soff:.4f} "
                  f"(z={off/soff:+.1f})")

    # D2: dx vs shell count (0 / 1 / 2+), fixed N in [20.4, 21.7), slopes
    def slope(recs):
        xs, ys = [], []
        for r in recs:
            if 20.4 <= r["N"] < 21.7:
                xs.append(min(r["shell"], 2))
                ys.append(r["dx"])
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if len(xs) < 30 or len(set(xs.tolist())) < 2:
            return None
        X = np.vstack([np.ones_like(xs), xs]).T
        co, res, *_ = np.linalg.lstsq(X, ys, rcond=None)
        resid = ys - X @ co
        s2 = float(resid @ resid) / (len(ys) - 2)
        cov = s2 * np.linalg.inv(X.T @ X)
        groups = {int(k): wmean(ys[xs == k]) for k in sorted(set(xs))}
        return {"slope": float(co[1]),
                "slope_sigma": float(np.sqrt(cov[1, 1])),
                "groups": {str(k): v for k, v in groups.items()},
                "n": len(ys)}
    sn = slope(nat)
    si = slope(inj)
    out["D2_shell_slopes"] = {"natural": sn, "injected": si}
    if sn and si:
        dsl = sn["slope"] - si["slope"]
        sds = float(np.hypot(sn["slope_sigma"], si["slope_sigma"]))
        out["D2_shell_slopes"]["difference"] = {"value": dsl, "sigma": sds,
                                                "z": dsl / sds}
        print(f"D2 slopes: natural {sn['slope']:+.4f}±{sn['slope_sigma']:.4f}"
              f"  injected {si['slope']:+.4f}±{si['slope_sigma']:.4f}"
              f"  diff z = {dsl/sds:+.2f}")
    elif sn:
        print("D2: injected slope unavailable (insufficient shell spread)")
        out["D2_shell_slopes"]["difference"] = None

    # frozen verdict rule
    d1_rises = (len(out["D1_offset_by_hostN"]) >= 2
                and out["D1_offset_by_hostN"][-1]["offset_nat_minus_inj"]
                > out["D1_offset_by_hostN"][0]["offset_nat_minus_inj"])
    diff = out["D2_shell_slopes"].get("difference")
    d1_sig = any(r["z"] > 2 for r in out["D1_offset_by_hostN"])
    if diff and diff["z"] >= 2 and d1_rises:
        out["verdict"] = "COUPLING-SUPPORTED"
    elif diff is not None and abs(diff["z"]) < 2 and d1_sig:
        out["verdict"] = "IMPRINT-SUPPORTED (environment-flat)"
    else:
        out["verdict"] = "NOT-SEPARABLE-AT-CATALOG-LEVEL"
    out["wall_s"] = round(time.time() - t0, 1)
    with open(os.path.join(_HERE, "t2_pairing.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("VERDICT:", out["verdict"])


if __name__ == "__main__":
    main()
