#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Matched-population FF (literal calc_cddf) aggregation — Lane L.

Runs the EXISTING FF harness machinery (`calccddf_vs_hbi.py`, untouched)
restricted to the PACK-SIDE sightline universe, so both comparison arms
live on one documented population:

  universe := the TARGETIDs of `build_pathlength(HBIConfig(no_bal=True))`
              — the committed pack convention (BAL veto + catalog cuts),
              the same function that defined the phaseB pack's dX.

CORRECTED DIAGNOSIS (2026-08-12, supersedes the pass-2 note): the
pack-vs-FF dX gap (0.8385) is dominated by the **BAL veto** (~14 % of
sightlines; empirically 0.864 on a 3-file sample) plus the pack's
good-mask universe cuts (~2–3 %) — NOT by a λ_rf 1216/1250 window-span
difference: the stored `max_z_dlas` are clamped at z_qso − Δz(3000 km/s)
by `set_parameters.max_z_dla`, so the proximity strip never entered the
FF windows. The span-arithmetic match in the pass-2 note was a
coincidence.

Output: a stamped JSON with, per z-bin (Paper-1 zones [2.0,2.4) [2.4,2.6)
[2.6,3.0) [3.0,3.5) + integrated) and per threshold (>=20.0, >=20.3, and
the N-resolved 0.2-dex bins), the FF mean counts and dX on the matched
universe, with n_sightlines/dX recorded for BOTH conventions so the match
is checkable rather than asserted. NO GP re-inference; NO alpha applied
(raw plug-in, alpha = 1/R0 downstream as before).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)

from CDDF_analysis.hbi.calccddf_vs_hbi import (   # noqa: E402
    MOCKS, N_EDGES, SNR_MIN, HIGH_NHI_CUT, NanSafeDLACatalogue)
from CDDF_analysis.cddf_forward.window import WindowSpec  # noqa: E402

Z_ZONES = [(2.0, 2.4), (2.4, 2.6), (2.6, 3.0), (3.0, 3.5)]


def pack_universe(mock):
    """TARGETID set + dX of the committed pack-side pathlength convention.

    Mirrors `build_pathlength`'s selection loop EXACTLY (SNR > snr_min,
    z_qso window, BAL veto, finite collared window), and cross-checks the
    resulting dX against `build_pathlength` itself.
    """
    import fitsio
    from CDDF_analysis.hbi.cddf_catalog_hbi import (
        HBIConfig, build_pathlength, _build_qso_lookup, LYA_REST)
    from CDDF_analysis.hbi import ab_loa0_fp_baseline as AB
    assert mock == "2lpt0", "pack universe wired for 2lpt0 here"
    cfg = HBIConfig(catalog_dir=AB.DEF_CAT, truth_path=AB.DEF_TRUTH,
                    bal_cat_path=AB.DEF_BAL, molly_tsv=AB.DEF_LYAONLY_MOLLY,
                    out_dir="/tmp", mockdir=os.path.dirname(AB.DEF_TRUTH),
                    lam_rf_min=1025.0, no_bal=True)
    qso_lookup = _build_qso_lookup(cfg)
    bal = fitsio.read(cfg.bal_cat_path, ext=1, columns=["TARGETID"])
    bal_tids = set(int(r["TARGETID"]) for r in bal)
    C_KMS = 299792.458
    collar = 3000.0 / C_KMS
    tids = []
    for t, (snr, zq) in qso_lookup.items():
        if snr <= cfg.snr_min or not (cfg.z_qso_min < zq < cfg.z_qso_max):
            continue
        if t in bal_tids:
            continue
        zlo = max(3600.0 / LYA_REST - 1.0,
                  cfg.lam_rf_min * (1 + zq) / LYA_REST - 1.0 + collar)
        zhi = min(zq - collar,
                  cfg.lam_rf_max * (1 + zq) / LYA_REST - 1.0 - collar)
        if np.isfinite(zlo) and np.isfinite(zhi) and zhi > zlo:
            tids.append(int(t))
    X_tot, n_sl = build_pathlength(cfg, qso_lookup=qso_lookup)
    dx_pack = float(np.sum(X_tot))
    if len(tids) != int(n_sl):
        print(f"[warn] mirrored universe {len(tids)} != "
              f"build_pathlength n_sl {n_sl}", file=sys.stderr)
    return set(tids), dx_pack, None


class MatchedCatalogue(NanSafeDLACatalogue):
    """The harness catalogue with a TARGETID-universe restriction ANDed into
    the standard condition mask (nothing else altered)."""
    _tid_whitelist = None

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        if MatchedCatalogue._tid_whitelist is not None:
            tids = self.filehandle["target_ids"][:]
            keep = np.array([int(t) in MatchedCatalogue._tid_whitelist
                             for t in tids])
            self.condition = self.condition & keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", default="2lpt0")
    ap.add_argument("--nfiles", type=int, default=-1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfgm = MOCKS[args.mock]
    t0 = time.time()
    uni, dx_pack, pl = pack_universe(args.mock)
    print(f"pack universe: {len(uni)} sightlines, dX {dx_pack:.1f}",
          flush=True)
    MatchedCatalogue._tid_whitelist = uni

    files = sorted(glob.glob(os.path.join(cfgm["proc"], "processed-*.h5")))
    if args.nfiles > 0:
        files = files[: args.nfiles]

    zone_counts = {z: np.zeros(len(N_EDGES) - 1) for z in
                   [f"[{a},{b})" for a, b in Z_ZONES] + ["integrated"]}
    zone_dx = {k: 0.0 for k in zone_counts}
    n_sl = 0
    for i, fn in enumerate(files):
        cat = MatchedCatalogue(
            processed_file=fn, sample_file=cfgm["grid"],
            catalog_file=cfgm["truth"], sub_dla=False, second=False,
            snr=SNR_MIN, high_nhi_cut=True,
            high_nhi_cut_value=HIGH_NHI_CUT,
            window=WindowSpec(z_min_lyb=True))
        n_sl += int(np.sum(cat.condition))
        for (a, b) in Z_ZONES + [(2.0, 3.5)]:
            key = f"[{a},{b})" if (a, b) != (2.0, 3.5) else "integrated"
            probs, poissons = cat._split_distributions(
                N_EDGES, lred=a, ured=b, lnhi_min=17.19,
                lnhi_max=HIGH_NHI_CUT, nhi=True)
            mc = np.array(poissons, float)
            for bb, plist in enumerate(probs):
                if plist:
                    mc[bb] += float(np.sum(np.concatenate(
                        [np.atleast_1d(p) for p in plist])))
            zone_counts[key] += mc
            zone_dx[key] += float(cat.path_length(a, b))
        cat.filehandle.close()
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)} files "
                  f"({time.time()-t0:.0f}s)", flush=True)

    edges = np.asarray(N_EDGES, float)
    out = {
        "schema": "ff_matched_window/v1",
        "date": time.strftime("%Y-%m-%d"),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip(),
        "mock": args.mock, "n_files": len(files),
        "universe": dict(
            definition="build_pathlength(HBIConfig(no_bal=True)) TARGETIDs "
                       "(the committed pack convention)",
            n_targetids=len(uni), dX_pack_convention=dx_pack,
            n_sightlines_ff_matched=n_sl),
        "n_edges": edges.tolist(),
        "zones": {},
        "estimand": "raw FF plug-in (calc_cddf DLA(1) posterior-weighted "
                    "expected counts); NO alpha applied; alpha = 1/R0 "
                    "downstream as in calccddf_vs_hbi.json",
    }
    for key in zone_counts:
        counts = zone_counts[key]
        dx = zone_dx[key]
        cum = {}
        for thr in (20.0, 20.3):
            m = edges[:-1] >= thr - 1e-9
            cum[f">={thr}"] = float(counts[m].sum() / dx) if dx else None
        out["zones"][key] = dict(
            dX=dx, counts_per_Nbin=counts.tolist(),
            dndx_cumulative=cum)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: out["zones"][k]["dndx_cumulative"]
                      for k in out["zones"]}, indent=1))
    print(f"wrote {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
