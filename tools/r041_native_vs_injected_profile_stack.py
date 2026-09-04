#!/usr/bin/env python
"""Native-vs-injected profile stack (PI continuation ruling 2026-09-02 §7): a direct, mock-only test of the construction difference between the
mock's native DLA imprint and the science lane's injection primitive, using the HCD-bearing / HCD-free twin spectra of the SAME sightlines.

For each isolated single native DLA (truth N, z from the full-support matched table) with S/N in the requested strata:
  F_124 = twin-with-HCDs flux, F_0 = HCD-free twin flux (same TARGETID, same pixel file, same seed), T_inj = the injection primitive's transmission
  (gpy_dla_detection.inject_absorber.inject_voigt, num_lines as configured for R-041 = 3) at the truth (N, z) on the same wavelength grid.
Stacks in the DLA rest frame (median over sightlines, 0.5-Å rest bins):
  S_raw  = F_124 / F_0                 (the native imprint; includes forest metals/BALs of the -124 twin)
  S_res  = F_124 / (F_0 * T_inj)       (native imprint divided by the injected profile: 1 if the constructions agree)
Also: equivalent widths of (1 - F_124/F_0) and (1 - T_inj) over +-25 A rest around Ly-alpha (the damping-wing region that sets N_hat), the
implied effective N difference from the EW ratio in the damping-wing regime (EW ∝ sqrt(N) for saturated damped profiles), and the stacks at the
Ly-beta position of the DLA and at the SiII 1190/1193, SiIII 1207, SiII 1260 positions. Mock-only; outputs JSON + PNG.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, REPO)
from gpy_dla_detection.inject_absorber import inject_voigt  # noqa: E402

ROOT = os.environ.get("ROOT_MAX4", "/scratch/cavestru_root/cavestru0/mfho/r041_max4_highz_2026-09")
MOCK = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0"
LYA, LYB = 1215.6701, 1025.7223
LINES = {"SiII1190": 1190.42, "SiII1193": 1193.29, "SiIII1207": 1206.50, "Lya": LYA, "SiII1260": 1260.42, "Lyb": LYB}
REST_EDGES = np.arange(1000.0, 1300.0 + 1e-9, 0.5)


def read_pixel(pix, twin):
    from astropy.io import fits
    p = f"{MOCK}/{twin}/spectra-16/{pix // 100}/{pix}/spectra-16-{pix}.fits"
    h = fits.open(p, memmap=True); fm = h["FIBERMAP"].data
    tid = np.asarray(fm["TARGETID"], np.int64)
    out = {}
    for cam in ("B", "R"):
        out[cam] = (np.asarray(h[f"{cam}_WAVELENGTH"].data, float), h[f"{cam}_FLUX"].data, h[f"{cam}_IVAR"].data)
    return tid, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/response_estimator/profile_stack")
    ap.add_argument("--matches", default=f"{ROOT}/response_study/matches_N2_native_full.csv")
    ap.add_argument("--pop", default=f"{ROOT}/p1/mock_native/2lpt/population_native.csv")
    ap.add_argument("--min-stratum", type=int, default=2, help="use strata >= this (S/N >= 4) for the stack")
    ap.add_argument("--num-lines", type=int, default=3, help="R-041 injection configs use num_lines 3")
    ap.add_argument("--max", type=int, default=10000)
    a = ap.parse_args(argv); os.makedirs(a.out, exist_ok=True)
    pop = {int(r["TARGETID"]): r for r in csv.DictReader(open(a.pop))}
    rows = [r for r in csv.DictReader(open(a.matches)) if int(r["isolated"]) == 1 and float(r["logN"]) >= 20.0 and int(r["stratum"]) >= a.min_stratum]
    rows = rows[:a.max]
    by_pix = {}
    for r in rows:
        by_pix.setdefault(int(pop[int(r["TARGETID"])]["healpix"]), []).append(r)
    stacks = {k: [] for k in ("raw", "res")}; ew = []; n_used = 0; per = []
    for pix, rs in sorted(by_pix.items()):
        tid1, s1 = read_pixel(pix, "loa-124"); tid0, s0 = read_pixel(pix, "loa-0")
        for r in rs:
            t = int(r["TARGETID"]); z = float(r["z"]); N = float(r["logN"]); lam_a = LYA * (1 + z)
            cam = "B" if lam_a < 5750 else "R"
            i1 = np.where(tid1 == t)[0]; i0 = np.where(tid0 == t)[0]
            if len(i1) != 1 or len(i0) != 1:
                continue
            w, F1, I1 = s1[cam][0], np.asarray(s1[cam][1][i1[0]], float), np.asarray(s1[cam][2][i1[0]], float)
            F0 = np.asarray(s0[cam][1][i0[0]], float); I0 = np.asarray(s0[cam][2][i0[0]], float)
            good = (I1 > 0) & (I0 > 0) & np.isfinite(F1) & np.isfinite(F0)
            T = inject_voigt(w, np.ones_like(w), 10 ** N, z, a.num_lines)
            # local continuum normalisation of the HCD-free twin near Lya (median of F0 over 30-60 A either side, excluding the trough)
            side = good & (np.abs(w - lam_a) > 30 * (1 + z) / 1.0) & (np.abs(w - lam_a) < 60 * (1 + z))
            c0 = np.median(F0[side]) if side.sum() > 20 else np.nan
            if not np.isfinite(c0) or c0 <= 0:
                continue
            rest = w / (1 + z); ib = np.clip(np.searchsorted(REST_EDGES, rest, side="right") - 1, -1, len(REST_EDGES) - 2)
            raw = np.where(good & (F0 > 0.3 * c0), F1 / np.maximum(F0, 1e-6), np.nan)          # ratio only where the HCD-free twin is not itself dark
            res = np.where(good & (F0 * T > 0.3 * c0), F1 / np.maximum(F0 * T, 1e-6), np.nan)
            for key, arr in (("raw", raw), ("res", res)):
                prof = np.full(len(REST_EDGES) - 1, np.nan)
                for b in np.unique(ib[ib >= 0]):
                    v = arr[ib == b]; v = v[np.isfinite(v)]
                    if len(v):
                        prof[b] = np.mean(v)
                stacks[key].append(prof)
            # equivalent widths over +-25 A rest around Lya (damping wings), on the normalised flux
            win = good & (np.abs(rest - LYA) <= 25)
            ew_nat = float(np.sum((1 - np.clip(F1[win] / np.maximum(F0[win], 1e-6), 0, 1.5)) * np.gradient(rest)[win])) if win.sum() > 5 else np.nan
            ew_inj = float(np.sum((1 - T[win]) * np.gradient(rest)[win])) if win.sum() > 5 else np.nan
            ew.append((N, z, int(r["stratum"]), ew_nat, ew_inj)); n_used += 1
            per.append(dict(TARGETID=t, logN=N, z=z, stratum=int(r["stratum"]), ew_native=ew_nat, ew_injected=ew_inj, Nhat=(float(r["Nhat"]) if r["Nhat"] not in ("", "nan") else None)))
    med = {k: np.nanmedian(np.array(v), axis=0).tolist() for k, v in stacks.items()}
    n_per_bin = {k: np.sum(np.isfinite(np.array(v)), axis=0).tolist() for k, v in stacks.items()}
    ewa = np.array(ew, float); ok = np.isfinite(ewa[:, 3]) & np.isfinite(ewa[:, 4]) & (ewa[:, 4] > 0)
    ratio = ewa[ok, 3] / ewa[ok, 4]
    # damped regime: EW ∝ N^{1/2} → ΔlogN_eff = 2 log10(EW_nat / EW_inj)
    dlogN = 2 * np.log10(np.clip(ratio, 1e-3, None))
    centres = 0.5 * (REST_EDGES[:-1] + REST_EDGES[1:])
    def window(key, lo, hi):
        m = (centres >= lo) & (centres < hi); v = np.array(med[key])[m]; v = v[np.isfinite(v)]
        return round(float(np.median(v)), 4) if len(v) else None
    summ = dict(n_dlas=n_used, num_lines_injected=a.num_lines, min_stratum=a.min_stratum,
                ew_ratio_native_over_injected=dict(median=round(float(np.median(ratio)), 4), p16=round(float(np.percentile(ratio, 16)), 4), p84=round(float(np.percentile(ratio, 84)), 4), n=int(ok.sum())),
                implied_dlogN_eff=dict(median=round(float(np.median(dlogN)), 4), mean=round(float(np.mean(dlogN)), 4), p16=round(float(np.percentile(dlogN, 16)), 4), p84=round(float(np.percentile(dlogN, 84)), 4)),
                implied_dlogN_by_N=[dict(bin=[lo, hi], n=int(((ewa[ok, 0] >= lo) & (ewa[ok, 0] < hi)).sum()), median=round(float(np.median(dlogN[(ewa[ok, 0] >= lo) & (ewa[ok, 0] < hi)])), 4)) for lo, hi in ((20.0, 20.3), (20.3, 20.5), (20.5, 21.0), (21.0, 22.5)) if ((ewa[ok, 0] >= lo) & (ewa[ok, 0] < hi)).sum() >= 5],
                stack_res_windows=dict(core_pm2A=window("res", LYA - 2, LYA + 2), blue_wing_5_15A=window("res", LYA - 15, LYA - 5), red_wing_5_15A=window("res", LYA + 5, LYA + 15), wing_15_30A=window("res", LYA + 15, LYA + 30),
                                       control_1150_1170=window("res", 1150, 1170), control_1270_1290=window("res", 1270, 1290),
                                       SiII1190_pm1=window("raw", 1189.4, 1191.4), SiII1193_pm1=window("raw", 1192.3, 1194.3), SiIII1207_pm1=window("raw", 1205.5, 1207.5), SiII1260_pm1=window("raw", 1259.4, 1261.4),
                                       Lyb_pm2=window("raw", LYB - 2, LYB + 2), Lyb_control_1030_1050=window("raw", 1030, 1050)),
                rest_edges=REST_EDGES.tolist(), stack_raw=med["raw"], stack_res=med["res"], n_per_bin=n_per_bin)
    json.dump(dict(summary=summ, per_dla=per), open(os.path.join(a.out, "profile_stack.json"), "w"), indent=1)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        ax[0].plot(centres, med["raw"], lw=1, label="F_124 / F_0 (native imprint)"); ax[0].set_ylabel("ratio"); ax[0].set_ylim(0, 1.3); ax[0].legend()
        ax[1].plot(centres, med["res"], lw=1, color="C3", label="F_124 / (F_0 T_inj)  (native / injected)"); ax[1].axhline(1, color="k", lw=0.5); ax[1].set_ylim(0.6, 1.4); ax[1].set_ylabel("ratio"); ax[1].legend()
        for k, v in LINES.items():
            for a_ in ax:
                a_.axvline(v, color="gray", lw=0.4, ls=":")
        ax[1].set_xlabel("rest wavelength [A] (DLA frame)"); ax[0].set_title(f"2LPT loa-124 vs loa-0 twin, {n_used} isolated single native DLAs (strata >= {a.min_stratum}), median stack")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "profile_stack.png"), dpi=120)
    except Exception as e:
        print("plot skipped:", e)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("stack_raw", "stack_res", "rest_edges", "n_per_bin")}, indent=1))


if __name__ == "__main__":
    main()
