#!/usr/bin/env python
"""Campaign D — inject a KNOWN non-PW100 truth CDDF (the ANTI-CIRCULAR gate).

Injects absorbers whose log N_HI is drawn from a deliberately-chosen test CDDF
(default: a falling power law dn/dlogN ∝ 10^(−β·logN)) — NOT the PW100 inference
prior.  After the GP runs, deconvolving the recovered counts with the response
matrix R (built from Campaign A) and checking the result returns this KNOWN
distribution is the unbiasedness test the same-mock closure can't provide
(self-calibration on its own sample is circular — see README §intro).  Reuses the
validated clean-select + tree-write path; only the injection builder differs
(`build_injection_sample`).
"""
import argparse, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
from _gen_common import load_clean_sightlines, finalize_tree, report_restframe
from campaign_grid import (build_injection_sample, build_control_rows, validate_manifest,
                           default_zqso_bins, LOGN_MIN, LOGN_MAX)

DEFAULT_MOCK = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                "qq_desi_y3/v2.8.5/mock-0/loa-124")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mockdir", default=DEFAULT_MOCK)
    ap.add_argument("--n_per_cell", type=int, default=50)
    ap.add_argument("--n_controls", type=int, default=2000)
    ap.add_argument("--n_healpix", type=int, default=200)
    ap.add_argument("--snr_cut", type=float, default=2.0)
    ap.add_argument("--num_lines", type=int, default=31)
    ap.add_argument("--seed", type=int, default=20260612)
    ap.add_argument("--snr_bins", type=float, nargs="+", default=[2.0, 4.0, 8.0, 1e9])
    ap.add_argument("--beta", type=float, default=0.7,
                    help="test-CDDF slope: dn/dlogN ∝ 10^(-beta*logN). Default 0.7 "
                         "(falling; clearly NOT the PW100 prior).")
    ap.add_argument("--logN_min", type=float, default=LOGN_MIN)
    ap.add_argument("--logN_max", type=float, default=LOGN_MAX)
    a = ap.parse_args()

    zqso_bins = list(default_zqso_bins())
    beta = float(a.beta)
    logN_pdf = lambda ln: 10.0 ** (-beta * np.asarray(ln, float))   # the KNOWN truth

    clean, csl = load_clean_sightlines(a.mockdir, snr_cut=a.snr_cut, n_healpix=a.n_healpix)
    inj = build_injection_sample(
        csl, snr_bins=a.snr_bins, n_per_cell=a.n_per_cell, logN_pdf=logN_pdf,
        logN_range=(a.logN_min, a.logN_max), zqso_bins=zqso_bins, seed=a.seed,
        campaign="D", method="coadd", num_lines=a.num_lines)
    if not inj:
        raise SystemExit("[manifest] ERROR: zero injections built — widen healpix/grid.")
    ctrl = build_control_rows(
        csl, snr_bins=a.snr_bins, target_controls=a.n_controls, seed=a.seed + 1,
        inj_id_start=len(inj), exclude_target_ids={int(r["target_id"]) for r in inj},
        zqso_bins=zqso_bins)
    manifest = list(inj) + list(ctrl)
    validate_manifest(manifest)

    nlt = np.array([r["logN_true"] for r in inj])
    # provenance: the injected truth histogram (deconvolution target)
    edges = np.arange(a.logN_min, a.logN_max + 1e-6, 0.3)
    hist, _ = np.histogram(nlt, bins=edges)
    print(f"[manifest] {len(inj)} inj (D, β={beta}) + {len(ctrl)} ctrl on "
          f"{len(set(int(r['healpix']) for r in inj))} healpix; "
          f"logN [{nlt.min():.2f},{nlt.max():.2f}] frac<19={np.mean(nlt<19):.2f}", flush=True)
    print(f"[truth-CDDF] injected logN histogram (Δ=0.3, edges {a.logN_min}-{a.logN_max}): "
          f"{hist.tolist()}", flush=True)
    report_restframe(inj, zqso_bins)
    finalize_tree(manifest, clean, out_root=a.out, mockdir=a.mockdir, num_lines=a.num_lines)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
