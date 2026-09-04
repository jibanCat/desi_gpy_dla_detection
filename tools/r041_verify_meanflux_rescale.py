#!/usr/bin/env python
"""Verification of the P1 mean-flux-only high-z emulation ("mean-flux-rescaled native mock"; PI ruling 2026-09-03 §11–§14). Mock-only, read-only.

Measures the Lyα-forest mean transmitted flux F̄(z) = ΣF / ΣC (TRUE_CONT continuum) per Δz = 0.05 bin in the NATIVE frame over rest 1045–1195 Å
(unmasked, ivar > 0), for: the original loa-0 (HCD-free) spectra, the rescaled loa-0 injection arm (P1 2LPT random; injected DLA windows masked),
the original loa-124 (native HCDs masked) and the rescaled loa-124 native arm (native HCDs masked, BAL sightlines excluded), and compares with the
targets: the measured low-z table (original) and the finder-fiducial τ_eff(z + Δz) (rescaled arms). Also checks the exact transformation on individual
sightlines: F' − F = (r(λ) − 1) · S(λ) with S the noise-preserving signal estimate, r = exp(−(τ_model(z+Δz) − τ_meas(z))) on forest pixels, ivar and mask
unchanged, DLA transmission applied once (T = 1 for the native arm).
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, REPO)
from injection.noise_preserving import signal_estimate, taueff, DEFAULT_MEDIAN_PX, DEFAULT_SIGMA_PX  # noqa: E402

LYA = 1215.67; C_KMS = 299792.458
MOCK0 = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-0"
MOCK124 = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124"
RANDOM_ARM = "/scratch/cavestru_root/cavestru0/mfho/r041_highz_repair_2026-08-28/mock/2lpt/random"
NATIVE_ARM = "/scratch/cavestru_root/cavestru0/mfho/r041_max4_highz_2026-09/p1/mock_native/2lpt/native"
MEANFLUX_JSON = "/scratch/cavestru_root/cavestru0/mfho/r041_highz_repair_2026-08-28/mock/meanflux_2lpt_loa0.json"
EDGES = np.arange(2.8, 3.85, 0.05)


def read_spectra(path):
    from astropy.io import fits
    h = fits.open(path, memmap=True); fm = h["FIBERMAP"].data; tid = np.asarray(fm["TARGETID"], np.int64)
    cams = {c: (np.asarray(h[f"{c}_WAVELENGTH"].data, float), h[f"{c}_FLUX"].data, h[f"{c}_IVAR"].data, h[f"{c}_MASK"].data) for c in ("B", "R", "Z") if f"{c}_FLUX" in h}
    return tid, cams


def true_cont(truth_path):
    from astropy.io import fits
    tr = fits.open(truth_path, memmap=True); cont = tr["TRUE_CONT"].data; hdr = tr["TRUE_CONT"].header
    cwave = float(hdr["WMIN"]) + float(hdr["DWAVE"]) * np.arange(cont["TRUE_CONT"].shape[1])
    return {int(t): np.asarray(cont["TRUE_CONT"][i], float) for i, t in enumerate(np.asarray(cont["TARGETID"], np.int64))}, cwave, {int(t): float(z) for t, z in zip(np.asarray(tr["TRUTH"].data["TARGETID"], np.int64), np.asarray(tr["TRUTH"].data["Z"], float))}


def accumulate(acc, wave, flux, ivar, mask, cont, zq, windows):
    """Add ΣF and ΣC per z bin over rest 1045–1195 Å, unmasked, ivar > 0, outside the masked windows [(z_lo, z_hi)...] in absorption redshift."""
    z = wave / LYA - 1.0; rest = wave / (1 + zq)
    ok = (rest > 1045) & (rest < 1195) & np.isfinite(flux) & (ivar > 0) & (mask == 0)
    for lo, hi in windows:
        ok &= ~((z > lo) & (z < hi))
    b = np.clip(np.searchsorted(EDGES, z, side="right") - 1, -1, len(EDGES) - 2)
    for k in range(len(EDGES) - 1):
        m = ok & (b == k)
        if m.any():
            acc[0][k] += float(flux[m].sum()); acc[1][k] += float(cont[m].sum()); acc[2][k] += int(m.sum())


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--max-pixels", type=int, default=60)
    a = ap.parse_args(argv); os.makedirs(a.out, exist_ok=True)
    mf = json.load(open(MEANFLUX_JSON)); zt = np.array(mf["z_center"]); tt = np.array(mf["taueff"], float)
    tau_meas = lambda z: np.interp(z, zt[np.isfinite(tt)], tt[np.isfinite(tt)]); tau_model = taueff("finder_fiducial")
    # sightlines and windows
    plan = list(csv.DictReader(open(f"{RANDOM_ARM}/plan.csv")))
    # 2026-09-03 FIX: the arm files carry the ORIGINAL rows for every sightline not in the plan (the generator rewrites only the plan
    # sightlines), so the measurement and the exact-equation checks must be restricted to the plan sightlines of each arm — the first
    # run averaged all rows and understated the rescale by construction (disclosed in the ledger).
    plan_tids_random = {int(r["target_id"]) for r in plan}
    plan_tids_native = {int(r["target_id"]) for r in csv.DictReader(open(f"{NATIVE_ARM}/plan.csv"))}
    n_sl = {"random": 0, "native": 0}
    inj_by = {}
    for r in plan:
        inj_by.setdefault(int(r["target_id"]), []).append(float(r["z_true"]))
    pop_nat = {int(r["TARGETID"]): r for r in csv.DictReader(open(f"{NATIVE_ARM}/../population_native.csv"))}
    from astropy.io import fits
    hcd = fits.open(f"{MOCK124}/hcd_truth_cat.fits")[1].data; hN = np.asarray(hcd["NHI"], float); hZ = np.asarray(hcd["Z"], float); hT = np.asarray(hcd["TARGETID"], np.int64)
    hcd_by = {}
    for t, n, z in zip(hT, hN, hZ):
        if n >= 19.0:
            hcd_by.setdefault(int(t), []).append(float(z))
    bal = set(np.asarray(fits.open(f"{MOCK124}/bal_cat.fits")[1].data["TARGETID"], np.int64).tolist())
    dv = 5000.0 / C_KMS
    arms = {k: [np.zeros(len(EDGES) - 1), np.zeros(len(EDGES) - 1), np.zeros(len(EDGES) - 1, int)] for k in ("loa0_original", "loa0_random_arm_rescaled", "loa124_original", "loa124_native_arm_rescaled")}
    checks = dict(exact_equation=[], ivar_mask_identical=[], native_no_injection=[])
    pix_random = sorted({int(os.path.basename(os.path.dirname(p))) for p in glob.glob(f"{RANDOM_ARM}/spectra-16/*/*/spectra-16-*.fits")})[:a.max_pixels]
    pix_native = sorted({int(os.path.basename(os.path.dirname(p))) for p in glob.glob(f"{NATIVE_ARM}/spectra-16/*/*/spectra-16-*.fits")})[:a.max_pixels]
    n_checked = 0
    for pix in pix_random:
        cont, cwave, zq_by = true_cont(f"{MOCK0}/spectra-16/{pix // 100}/{pix}/truth-16-{pix}.fits")
        t0, c0 = read_spectra(f"{MOCK0}/spectra-16/{pix // 100}/{pix}/spectra-16-{pix}.fits"); tr, cr = read_spectra(f"{RANDOM_ARM}/spectra-16/{pix // 100}/{pix}/spectra-16-{pix}.fits")
        i0 = {int(t): i for i, t in enumerate(t0)}
        for i, t in enumerate(tr):
            if int(t) not in i0 or int(t) not in cont or int(t) not in plan_tids_random:
                continue
            n_sl["random"] += 1
            zq = zq_by[int(t)]; win = [(z - dv * (1 + z), z + dv * (1 + z)) for z in inj_by.get(int(t), [])]
            for cam in cr:
                w, fr, ivr, mkr = cr[cam][0], np.asarray(cr[cam][1][i], float), np.asarray(cr[cam][2][i], float), np.asarray(cr[cam][3][i])
                f0, iv0, mk0 = np.asarray(c0[cam][1][i0[int(t)]], float), np.asarray(c0[cam][2][i0[int(t)]], float), np.asarray(c0[cam][3][i0[int(t)]])
                C = np.interp(w, cwave, cont[int(t)])
                accumulate(arms["loa0_original"], w, f0, iv0, mk0, C, zq, win); accumulate(arms["loa0_random_arm_rescaled"], w, fr, ivr, mkr, C, zq, win)
                if n_checked < 200 and cam == "B":
                    # exact-equation check outside injected windows: F' - F = (r - 1) S, with S from the same signal estimate on the ORIGINAL flux
                    S = signal_estimate(f0, iv0, mk0, DEFAULT_MEDIAN_PX, DEFAULT_SIGMA_PX); z = w / LYA - 1.0; forest = (w < LYA * (1 + zq)) & (z > 0)
                    r = np.ones(w.size); r[forest] = np.exp(-(tau_model(z[forest] + 1.0) - tau_meas(z[forest])))
                    ok = np.isfinite(S) & (iv0 > 0) & (mk0 == 0) & forest
                    for lo, hi in win:
                        ok &= ~((z > lo - 0.03) & (z < hi + 0.03))
                    pred = f0 + (r - 1.0) * S; dev = np.abs(fr - pred)[ok]
                    checks["exact_equation"].append(float(np.max(dev)) if ok.any() else np.nan); checks["ivar_mask_identical"].append(bool(np.array_equal(ivr, iv0) and np.array_equal(mkr, mk0))); n_checked += 1
    n_nat = 0
    for pix in pix_native:
        cont, cwave, zq_by = true_cont(f"{MOCK124}/spectra-16/{pix // 100}/{pix}/truth-16-{pix}.fits")
        t0, c0 = read_spectra(f"{MOCK124}/spectra-16/{pix // 100}/{pix}/spectra-16-{pix}.fits"); tn, cn = read_spectra(f"{NATIVE_ARM}/spectra-16/{pix // 100}/{pix}/spectra-16-{pix}.fits")
        i0 = {int(t): i for i, t in enumerate(t0)}
        for i, t in enumerate(tn):
            if int(t) not in i0 or int(t) not in cont or int(t) in bal or int(t) not in plan_tids_native:
                continue
            n_sl["native"] += 1
            zq = zq_by[int(t)]; win = [(z - dv * (1 + z), z + dv * (1 + z)) for z in hcd_by.get(int(t), [])]
            for cam in cn:
                w, fn, ivn, mkn = cn[cam][0], np.asarray(cn[cam][1][i], float), np.asarray(cn[cam][2][i], float), np.asarray(cn[cam][3][i])
                f0, iv0, mk0 = np.asarray(c0[cam][1][i0[int(t)]], float), np.asarray(c0[cam][2][i0[int(t)]], float), np.asarray(c0[cam][3][i0[int(t)]])
                C = np.interp(w, cwave, cont[int(t)])
                accumulate(arms["loa124_original"], w, f0, iv0, mk0, C, zq, win); accumulate(arms["loa124_native_arm_rescaled"], w, fn, ivn, mkn, C, zq, win)
                if n_nat < 200 and cam == "B":
                    S = signal_estimate(f0, iv0, mk0, DEFAULT_MEDIAN_PX, DEFAULT_SIGMA_PX); z = w / LYA - 1.0; forest = (w < LYA * (1 + zq)) & (z > 0)
                    r = np.ones(w.size); r[forest] = np.exp(-(tau_model(z[forest] + 1.0) - tau_meas(z[forest])))
                    ok = np.isfinite(S) & (iv0 > 0) & (mk0 == 0) & forest; pred = f0 + (r - 1.0) * S; dev = np.abs(fn - pred)[ok]
                    checks["native_no_injection"].append(float(np.max(dev)) if ok.any() else np.nan); n_nat += 1
    zc = 0.5 * (EDGES[:-1] + EDGES[1:]); out = dict(z_center=zc.tolist(), target_measured_lowz=tau_meas(zc).tolist(), target_model_zplus1=tau_model(zc + 1.0).tolist(), arms={})
    print(f"{'z':>6} {'tau_meas(z)':>12} {'tau_model(z+1)':>14} | " + " | ".join(f"{k:>26}" for k in arms))
    for k, (SF, SC, n) in arms.items():
        F = np.where(SC > 0, SF / np.maximum(SC, 1e-12), np.nan); out["arms"][k] = dict(meanflux=F.tolist(), taueff=(-np.log(np.clip(F, 1e-6, None))).tolist(), n_pix=n.tolist())
    for k_ in range(len(zc)):
        print(f"{zc[k_]:6.3f} {tau_meas(zc[k_]):12.4f} {tau_model(zc[k_] + 1.0):14.4f} | " + " | ".join(f"{out['arms'][k]['taueff'][k_]:8.4f} (n {out['arms'][k]['n_pix'][k_]:7d})" for k in arms))
    ee = np.array(checks["exact_equation"]); nn = np.array(checks["native_no_injection"])
    out["checks"] = dict(exact_equation_max_dev=float(np.nanmax(ee)) if ee.size else None, exact_equation_n=int(ee.size), ivar_mask_identical_all=bool(all(checks["ivar_mask_identical"])),
                         native_arm_equation_max_dev=float(np.nanmax(nn)) if nn.size else None, native_n=int(nn.size))
    out["n_sightlines_used"] = n_sl; print("sightlines used (plan rows only):", n_sl)
    print("checks:", out["checks"])
    json.dump(out, open(os.path.join(a.out, "meanflux_rescale_verification.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
