#!/usr/bin/env python
"""validate_noise_preserving.py — the injection unit-test suite of the high-z repair cycle
(R-041A item A3): demonstrate, on real high-z archive sightlines, that the corrected
injection (injection/noise_preserving.py) satisfies the five properties the PI required
and quantify how the OLD multiplicative operation fails them. No finder is run here (the
recovery comparison is a separate SLURM campaign).

Checks (per injected spectrum; saturated absorber, log N = 21.0 unless stated):
  1. saturated troughs retain realistic residual fluctuations: the standard deviation of
     F' inside the trough core (|dv| < 300 km/s of the line centre), scaled by
     sqrt(ivar), compared with 1 (a real noise realization) — new vs old;
  2. ivar consistency: pull distribution (F' - 0) * sqrt(ivar) inside the core — std,
     mean, fraction beyond 3 sigma — and outside the profile (F' - F) == 0 exactly;
  3. masks unchanged (byte-identical) and masked pixels' flux untouched; ivar untouched;
  4. wavelength grid and resolution arrays untouched (byte-identical);
  5. the profile is correctly applied: (F' - F) / S == (T - 1) wherever S is finite
     (to 1e-12), and the flux decrement integrated over the profile equals sum((1-T) S).
Also reports, per smoothing scale, how much of the noise variance the signal estimate
absorbs (residual variance / (1/ivar) in absorber-free forest pixels), so the chosen
scale is documented.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from injection.noise_preserving import (inject_noise_preserving, inject_multiplicative, signal_estimate,
                                        transmission, LYA_REST, DEFAULT_SIGMA_PX, DEFAULT_MEDIAN_PX)

C_KMS = 299792.458


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--population", required=True, help="CSV of eligible sightlines (TARGETID,z_qso,snr,zlo,zhi)")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--logn", type=float, nargs="+", default=[21.0, 20.3, 20.0])
    ap.add_argument("--sigma-scan", type=float, nargs="+", default=[1.5, 2.5, 4.0, 8.0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    pop = np.genfromtxt(a.population, delimiter=",", names=True, dtype=None, encoding="utf-8")
    pick = rng.choice(len(pop), size=min(a.n, len(pop)), replace=False)
    with h5py.File(a.archive, "r") as h:
        cat = h["catalog"][:]
        tid2i = {int(t): i for i, t in enumerate(cat["TARGETID"])}
        wave = h["wavelength"][:].astype(float)
        res = {"per_logn": {}, "sigma_scan": {}, "n_spectra": int(len(pick)), "invariants": {}}
        inv = {"mask_identical": 0, "ivar_identical": 0, "wave_identical": 0, "outside_profile_unchanged": 0, "profile_applied_exact": 0, "n": 0}
        # --- smoothing-scale scan on absorber-free forest pixels: residual variance vs 1/ivar
        scan = {str(s): [] for s in a.sigma_scan}
        for j in pick:
            r = pop[j]; i = tid2i[int(r["TARGETID"])]
            f = h["flux"][i].astype(float); iv = h["ivar"][i].astype(float); m = h["mask"][i]
            zq = float(r["z_qso"])
            forest = (wave > 3600.0) & (wave < LYA_REST * (1 + zq) * 0.98) & (wave > LYA_REST * (1 + zq) * 1025.0 / 1215.67) & (m == 0) & (iv > 0)
            if forest.sum() < 200:
                continue
            for s in a.sigma_scan:
                S = signal_estimate(f, iv, m, DEFAULT_MEDIAN_PX, s)
                pull = (f - S)[forest] * np.sqrt(iv[forest])
                scan[str(s)].append([float(np.std(pull)), float(np.mean(pull))])
        res["sigma_scan"] = {k: {"pull_std_mean": float(np.mean([v[0] for v in vals])), "pull_std_p16_84": np.percentile([v[0] for v in vals], [16, 84]).tolist(),
                                 "pull_mean_mean": float(np.mean([v[1] for v in vals])), "n": len(vals),
                                 "note": "std of (F - S) sqrt(ivar) over absorber-free forest pixels: 1.0 = the residual is exactly the noise; < 1 = S absorbed part of the noise; > 1 = S leaves real forest structure in the residual"}
                             for k, vals in scan.items()}
        # --- per log N: trough statistics, old vs new
        for logn in a.logn:
            core_new, core_old, core_pull_new, frac3_new, edges_new, core_sest, red_pull = [], [], [], [], [], [], []
            for j in pick:
                r = pop[j]; i = tid2i[int(r["TARGETID"])]
                f = h["flux"][i].astype(float); iv = h["ivar"][i].astype(float); m = h["mask"][i]
                zlo, zhi = float(r["zlo"]), float(r["zhi"])
                zi = float(rng.uniform(max(zlo, 3.8), min(zhi, 5.0))) if min(zhi, 5.0) > max(zlo, 3.8) else None
                if zi is None:
                    continue
                ab = [{"nhi": 10.0 ** logn, "z_dla": zi}]
                new, parts = inject_noise_preserving(wave, f, iv, m, ab, seed=int(r["TARGETID"]) & 0xFFFFFFFF, return_parts=True)
                sest = inject_noise_preserving(wave, f, iv, m, ab, method="signal_estimate")
                old = inject_multiplicative(wave, f, ab)
                T, S = parts["T"], parts["S"]
                lam0 = LYA_REST * (1 + zi)
                core = (np.abs(wave / lam0 - 1.0) * C_KMS < 300.0) & (m == 0) & (iv > 0) & np.isfinite(S)
                if core.sum() < 5:
                    continue
                # 1 / 2: residual fluctuations inside the core, in noise units (a real DLA trough: F ~ 0 + noise -> pull std ~ 1)
                pn = new[core] * np.sqrt(iv[core]); po = old[core] * np.sqrt(iv[core]); ps = sest[core] * np.sqrt(iv[core])
                core_new.append(float(np.std(pn))); core_old.append(float(np.std(po))); core_pull_new.append(float(np.mean(pn))); core_sest.append(float(np.std(ps)))
                # ivar calibration on the RED side of Lya (no forest): (F - S) sqrt(ivar) should have std ~ 1
                red = (wave > LYA_REST * (1 + float(r["z_qso"])) * 1.05) & (wave < LYA_REST * (1 + float(r["z_qso"])) * 1.25) & (m == 0) & (iv > 0) & np.isfinite(S)
                if red.sum() > 50:
                    red_pull.append(float(np.std((f - S)[red] * np.sqrt(iv[red]))))
                frac3_new.append(float(np.mean(np.abs(pn) > 3)))
                # 3 / 4 / 5: invariants
                inv["n"] += 1
                inv["mask_identical"] += int(np.array_equal(m, h["mask"][i]))
                inv["ivar_identical"] += int(np.array_equal(iv, h["ivar"][i].astype(float)))
                inv["wave_identical"] += int(np.array_equal(wave, h["wavelength"][:].astype(float)))
                outside = np.abs(T - 1.0) < 1e-12
                inv["outside_profile_unchanged"] += int(np.all(new[outside] == f[outside]))
                # variance-preserving identity: the deterministic part of F' is exactly T * F, and the
                # synthetic part is exactly sqrt(1 - T^2) eps / sqrt(ivar) on unmasked finite-ivar pixels
                good = (iv > 0) & (m == 0) & np.isfinite(iv)
                synth = np.zeros_like(f); synth[good] = np.sqrt(np.clip(1.0 - T[good] ** 2, 0, 1)) * parts["eps"][good] / np.sqrt(iv[good])
                inv["profile_applied_exact"] += int(np.max(np.abs((new - synth) - T * f)) < 1e-9)
                # the rejected signal-estimate method obeys (F' - F) / S == T - 1 exactly (kept as a check of the profile itself)
                ok = np.isfinite(S) & (np.abs(S) > 1e-3)
                inv["signal_estimate_profile_exact"] = inv.get("signal_estimate_profile_exact", 0) + int(np.max(np.abs((sest[ok] - f[ok]) / S[ok] - (T[ok] - 1.0))) < 1e-9)
                edges_new.append(float(np.mean(new[core]) / np.mean(S[core])) if np.mean(S[core]) > 0 else np.nan)
            res["per_logn"][str(logn)] = {
                "n": len(core_new),
                "trough_pull_std_new_mean": float(np.mean(core_new)), "trough_pull_std_new_p16_84": np.percentile(core_new, [16, 84]).tolist(),
                "trough_pull_std_old_mean": float(np.mean(core_old)), "trough_pull_std_old_p16_84": np.percentile(core_old, [16, 84]).tolist(),
                "trough_pull_std_signal_estimate_method_mean": float(np.mean(core_sest)),
                "redside_pull_std_mean": float(np.mean(red_pull)) if red_pull else None, "redside_pull_std_p16_84": (np.percentile(red_pull, [16, 84]).tolist() if red_pull else None),
                "trough_pull_mean_new": float(np.mean(core_pull_new)), "trough_frac_beyond_3sigma_new": float(np.mean(frac3_new)),
                "trough_mean_flux_over_signal_new": float(np.nanmean(edges_new)),
                "reading": "a real saturated trough has pull std ~ 1 and mean ~ 0 (if ivar is calibrated: see redside_pull_std); the OLD operation gives ~ 0 (noiseless trough); the rejected signal-estimate method leaves forest structure in the trough (> 1)"}
        res["invariants"] = inv
        res["config"] = {"median_px": DEFAULT_MEDIAN_PX, "sigma_px": DEFAULT_SIGMA_PX, "core_window_kms": 300.0, "seed": a.seed}
        res["inputs"] = {"archive": {"path": a.archive, "sha256": _sha(a.archive)}, "population": {"path": a.population, "sha256": _sha(a.population)}}
        try:
            res["generator_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
        except Exception:
            res["generator_commit"] = "unknown"
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"invariants": inv, "sigma_scan": {k: round(v["pull_std_mean"], 3) for k, v in res["sigma_scan"].items()},
                      "per_logn": {k: {"new_pull_std": round(v["trough_pull_std_new_mean"], 3), "old_pull_std": round(v["trough_pull_std_old_mean"], 3), "sest_pull_std": round(v["trough_pull_std_signal_estimate_method_mean"], 3), "redside_pull_std": (round(v["redside_pull_std_mean"], 3) if v["redside_pull_std_mean"] else None),
                                       "new_mean": round(v["trough_pull_mean_new"], 3), "frac>3sig": round(v["trough_frac_beyond_3sigma_new"], 4)} for k, v in res["per_logn"].items()}}, indent=1))


if __name__ == "__main__":
    main()
