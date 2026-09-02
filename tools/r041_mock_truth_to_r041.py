#!/usr/bin/env python
"""r041_mock_truth_to_r041.py — convert an R-041C/E mock arm's injection_truth.fits (+ plan) into the R-041 truth CSV and
population CSV consumed by tools/r041_analyze.py / r041_multihcd_score.py (P1 reductions, 2026-09-02).
Schema map: TARGETID=target_id, wave=0, inj_idx=0 (single injection per sightline), logN=logN_true, z_inj=z_true,
stratum=snr_bin, snr=native_snr, z_qso, has_cand_ge20=0, pair_class='', method=arm, meanflux_model='extrapolated';
population: TARGETID, z_qso, snr, zlo=zlo_bin, zhi=zhi_bin, zlo_bin, zhi_bin, dX_bin, stratum, forest_blend."""
from __future__ import annotations
import argparse, csv, os
import numpy as np
from astropy.io import fits


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-fits", required=True); ap.add_argument("--out-truth", required=True); ap.add_argument("--out-population", required=True)
    a = ap.parse_args(argv)
    t = fits.open(a.truth_fits)[1].data
    rows, pop = [], {}
    for r in t:
        if not np.isfinite(float(r["logN_true"])):
            continue
        tid = int(r["target_id"])
        rows.append(dict(TARGETID=tid, wave=0, inj_idx=0, logN=float(r["logN_true"]), z_inj=float(r["z_true"]), stratum=int(r["snr_bin"]), snr=float(r["native_snr"]),
                         z_qso=float(r["z_qso"]), has_cand_ge20=0, pair_class="", dv_kms="", pair_logN="", method=str(r["arm"]), meanflux_model="extrapolated",
                         forest_blend=bool(r["forest_blend"]) if "forest_blend" in t.columns.names else False))
        pop[tid] = dict(TARGETID=tid, z_qso=float(r["z_qso"]), snr=float(r["native_snr"]), zlo=float(r["zlo_bin"]), zhi=float(r["zhi_bin"]), zlo_bin=float(r["zlo_bin"]),
                        zhi_bin=float(r["zhi_bin"]), dX_bin=float(r["dX_bin"]), stratum=int(r["snr_bin"]), n_cand=0, has_cand_ge20=0, cand="")
    for path, data in ((a.out_truth, rows), (a.out_population, list(pop.values()))):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
    print(f"{len(rows)} truth rows, {len(pop)} sightlines -> {a.out_truth}, {a.out_population}")


if __name__ == "__main__":
    main()
