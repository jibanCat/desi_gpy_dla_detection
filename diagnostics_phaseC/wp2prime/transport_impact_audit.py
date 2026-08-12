#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""No-refit transport-impact propagation (PI ruling 2026-08-12, §10–§18).

Uses ONLY already-read prospective evidence: the committed WP-2′ frozen
predictions and the held-out packs' observed counts (read at the one-shot
close). Nothing is refit; no operator element is touched; no new mock is
read. Represents the observed family-transport structure as explicit,
signed, physically named modes + per-bin measured shifts (never a
family sample covariance), and propagates it into the Paper-1 observables
in the actual reporting strategies motivated BEFORE the WP-2′ read
(19.9 / 20.0 / 20.1 / 20.3 — no post-hoc threshold optimization).
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from wp2_runner import assemble, PACKS, Z_ZONES  # noqa: E402

REAL_CAT = ("/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/"
            "loa_cddf_main_dark_v1/dlacat-loa-cddf-main-dark-v1.fits")
CACHEDIR = "/scratch/cavestru_root/cavestru0/mfho/wp2prime_2026_08_12"
STRATEGIES = {"19.9": 19.9, "20.0": 20.0, "20.1": 20.1, "20.3": 20.3}
# non-transport systematics carried per group (committed records):
# sigma_CKM fractional (142.4/38159, 89.5/22998, 47.2/6196) and the FP
# transfer half-width 10.6% applied to the FP fraction of each bin.
CKM_FRAC = {"G1": 0.00373, "G2": 0.00389, "G3": 0.00762}


def zfrac(a, lo_c, hi_c, zlo, zhi, with_M=True):
    """(obs-mu)/mu for observed N-hat in [lo_c,hi_c) x z in [zlo,zhi)."""
    fold = a["fold"]
    ne = fold["nhat_edges"]
    cm = (ne[:-1] >= lo_c - 1e-9) & (ne[1:] <= hi_c + 1e-9)
    kcent = 2.05 + 0.1 * np.arange(a["mu_sig"].shape[1])
    km = (kcent >= zlo) & (kcent < zhi)
    mu = float(a["mu_sig"][cm][:, km, :].sum()
               + a["fp_ck"][cm][:, km, :].sum())
    if with_M:
        mm = ((a["mig"]["NHAT"] >= lo_c) & (a["mig"]["NHAT"] < hi_c)
              & (a["mig"]["Z"] >= zlo) & (a["mig"]["Z"] < zhi))
        mu += float(mm.sum())
    obs = float(a["fold"]["obs_counts"][cm][:, km, :].sum())
    return (obs - mu) / mu if mu > 0 else np.nan, mu, obs


def main():
    from astropy.io import fits
    A = {m: assemble(m, CACHEDIR) for m in ("london0", "saclay0")}
    real = fits.open(REAL_CAT)[1].data
    clean = ((real["DLAFLAG"] == 0) & (real["P_DLA"] > 0.99)
             & (real["SNR_REDSIDE"] > 2.0))

    out = {"schema": "wp2_transport_impact/v1",
           "inputs": "committed WP-2' predictions + already-read pack "
                     "observed counts; real stat errors from "
                     "loa_cddf_main_dark_v1 clean counts",
           "modes": {}, "cddf": {}, "dndx": {}, "omega": {}}

    # ---------------- modes (signed, per mock) ---------------------------
    # Mode Z: fractional residual >=20.3 per z zone; linear-in-z amplitude
    modeZ = {}
    for m, a in A.items():
        rows = []
        for zlo, zhi in Z_ZONES:
            f, mu, obs = zfrac(a, 20.3, 22.4, zlo, zhi)
            rows.append(dict(z=f"[{zlo},{zhi})", frac=f, mu=mu))
        zc = np.array([0.5 * (z[0] + z[1]) for z in Z_ZONES])
        fr = np.array([r["frac"] for r in rows])
        w = np.array([r["mu"] for r in rows])
        slope = float(np.polyfit(zc, fr, 1, w=np.sqrt(w))[0])
        modeZ[m] = dict(zones=rows, linear_slope_per_unit_z=slope)
    out["modes"]["Z_tilt_ge20p3"] = modeZ

    # Mode N: z-integrated fractional residual per aligned 0.2-dex bin
    modeN = {}
    nb_edges = [19.9, 20.1, 20.3, 20.5, 20.7, 20.9, 21.1, 21.3, 21.5]
    for m, a in A.items():
        rows = []
        for lo, hi in zip(nb_edges[:-1], nb_edges[1:]):
            f, mu, obs = zfrac(a, lo, hi, 2.0, 3.5)
            rows.append(dict(N=f"[{lo},{hi})", frac=f, mu=mu))
        nc = np.array([0.5 * (lo + hi)
                       for lo, hi in zip(nb_edges[:-1], nb_edges[1:])])
        fr = np.array([r["frac"] for r in rows])
        w = np.array([r["mu"] for r in rows])
        slope = float(np.polyfit(nc - 20.7, fr, 1, w=np.sqrt(w))[0])
        modeN[m] = dict(bins=rows, linear_slope_per_dex=slope)
    out["modes"]["N_rotation"] = modeN

    # Mode S and C: read from the committed closures (already extracted)
    out["modes"]["S_lowN_highSNR"] = "see wp2_*_closure.json snr rows " \
        "(s7 -6.9/-7.2 sigma; fractional deficits below)"
    for m, a in A.items():
        f, mu, obs = zfrac(a, 19.9, 20.1, 2.0, 3.5)
        out["modes"].setdefault("S_1990_2010_total", {})[m] = f
    modeC = {}
    for m, a in A.items():
        f, mu, obs = zfrac(a, 21.3, 21.5, 2.0, 3.5)
        modeC[m] = dict(frac=f, mu=mu)
    out["modes"]["C_ceiling_2130_2150"] = modeC

    # ---------------- CDDF per strategy, first bins + z-resolved --------
    for sname, F in STRATEGIES.items():
        rows = []
        for b in range(2):                     # first and second 0.2-dex
            lo, hi = round(F + 0.2 * b, 2), round(F + 0.2 * (b + 1), 2)
            n_real = int(np.sum(clean & (real["NHI"] >= lo)
                                & (real["NHI"] < hi)))
            stat = 1.0 / np.sqrt(max(n_real, 1))
            ent = dict(bin=f"[{lo},{hi})", n_real=n_real,
                       stat_frac=stat)
            for m, a in A.items():
                f, mu, obs = zfrac(a, lo, hi, 2.0, 3.5)
                ent[f"{m}_shift"] = f
                # z-resolved
                zres = []
                for zlo, zhi in Z_ZONES:
                    fz, muz, _ = zfrac(a, lo, hi, zlo, zhi)
                    zres.append(round(float(fz), 4))
                ent[f"{m}_z_shifts"] = zres
            env = max(abs(ent["london0_shift"]), abs(ent["saclay0_shift"]))
            other = CKM_FRAC["G1" if lo < 20.3 else "G2"] + 0.02 \
                if lo < 20.3 else CKM_FRAC["G2"]
            # feed systematic ~1-2% for bins near the floor (pass-2), FP
            # 10.6% of the bin FP fraction (4.3% at 20.0, ~0 above 20.2)
            fpfrac = {19.9: 0.115, 20.0: 0.043, 20.1: 0.016,
                      20.2: 0.0, 20.3: 0.0}.get(lo, 0.0)
            other = np.hypot(other, 0.106 * fpfrac)
            ent["transport_envelope"] = env
            ent["other_syst_frac"] = float(other)
            ent["dominates"] = ("transport" if env > max(stat, other)
                               else ("stat" if stat > other else "other"))
            ent["useful"] = bool(env < 5 * stat + 0.05)  # descriptive tag
            rows.append(ent)
        out["cddf"][sname] = rows

    # ---------------- dN/dX(z) at 20.3 and 20.0 -------------------------
    for thr in (20.3, 20.0):
        rows = []
        for zlo, zhi in Z_ZONES:
            n_real = int(np.sum(clean & (real["NHI"] >= thr)
                                & (real["Z_DLA"] >= zlo)
                                & (real["Z_DLA"] < zhi)))
            ent = dict(z=f"[{zlo},{zhi})", n_real=n_real,
                       stat_frac=1.0 / np.sqrt(max(n_real, 1)))
            for m, a in A.items():
                f, mu, obs = zfrac(a, thr, 22.4, zlo, zhi)
                ent[f"{m}_shift"] = f
            ent["transport_envelope"] = max(abs(ent["london0_shift"]),
                                            abs(ent["saclay0_shift"]))
            rows.append(ent)
        # z-integrated (cancellation display)
        entI = dict(z="[2.0,3.5) integrated")
        for m, a in A.items():
            f, mu, obs = zfrac(a, thr, 22.4, 2.0, 3.5)
            entI[f"{m}_shift"] = f
        rows.append(entI)
        out["dndx"][f">={thr}"] = rows

    # ---------------- band-limited Omega(>=20.3, z) ----------------------
    # weight per-bin residuals by 10^(Nc): Omega shift per zone per mock
    om = {}
    ne_o = np.arange(20.3, 22.41, 0.1)
    for m, a in A.items():
        fold = a["fold"]
        ne = fold["nhat_edges"]
        kcent = 2.05 + 0.1 * np.arange(a["mu_sig"].shape[1])
        zrows = []
        for zlo, zhi in Z_ZONES + [(2.0, 3.5)]:
            km = (kcent >= zlo) & (kcent < zhi)
            num = den = 0.0
            for lo, hi in zip(ne_o[:-1], ne_o[1:]):
                cm = (ne[:-1] >= lo - 1e-9) & (ne[1:] <= hi + 1e-9)
                mu = float(a["mu_sig"][cm][:, km, :].sum()
                           + a["fp_ck"][cm][:, km, :].sum())
                mm = ((a["mig"]["NHAT"] >= lo) & (a["mig"]["NHAT"] < hi)
                      & (a["mig"]["Z"] >= zlo) & (a["mig"]["Z"] < zhi))
                mu += float(mm.sum())
                obs = float(fold["obs_counts"][cm][:, km, :].sum())
                wgt = 10.0 ** (0.5 * (lo + hi) - 20.3)
                num += wgt * (obs - mu)
                den += wgt * mu
            zrows.append(dict(z=f"[{zlo},{zhi})",
                              omega_shift=num / den if den else np.nan))
        om[m] = zrows
    out["omega"][">=20.3"] = om
    # ceiling-bin contribution shares (from the mocks' own mu)
    a = A["saclay0"]
    ne = a["fold"]["nhat_edges"]
    shares = {}
    for lo, hi in zip(ne_o[:-1], ne_o[1:]):
        cm = (ne[:-1] >= lo - 1e-9) & (ne[1:] <= hi + 1e-9)
        mu = float(a["mu_sig"][cm].sum() + a["fp_ck"][cm].sum())
        shares[f"[{lo:.1f},{hi:.1f})"] = mu * 10.0 ** (0.5 * (lo + hi)
                                                       - 20.3)
    tot = sum(shares.values())
    out["omega"]["bin_weight_shares_saclay_mu"] = {
        k: round(v / tot, 4) for k, v in
        sorted(shares.items(), key=lambda kv: -kv[1])[:8]}

    path = os.path.join(_HERE, "wp2_transport_impact.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print("wrote", path)

    # ------------- console decision table --------------------------------
    print("\n=== Mode summary (signed) ===")
    for m in ("london0", "saclay0"):
        print(f" {m}: z-slope {modeZ[m]['linear_slope_per_unit_z']:+.3f}/z"
              f"  N-slope {modeN[m]['linear_slope_per_dex']:+.3f}/dex"
              f"  S[19.9,20.1) {out['modes']['S_1990_2010_total'][m]:+.3f}"
              f"  C[21.3,21.5) {modeC[m]['frac']:+.3f}")
    print("\n=== CDDF first two bins per strategy ===")
    for sname, rows in out["cddf"].items():
        for r in rows:
            print(f" {sname} {r['bin']}: Lon {r['london0_shift']:+.3f} "
                  f"Sac {r['saclay0_shift']:+.3f} env {r['transport_envelope']:.3f} "
                  f"stat {r['stat_frac']:.4f} other {r['other_syst_frac']:.4f} "
                  f"drives={r['dominates']}")
    print("\n=== dN/dX per z ===")
    for thr, rows in out["dndx"].items():
        for r in rows:
            s = f" {thr} {r['z']}: Lon {r['london0_shift']:+.3f} Sac {r['saclay0_shift']:+.3f}"
            if "stat_frac" in r:
                s += f" stat {r['stat_frac']:.4f}"
            print(s)
    print("\n=== Omega(>=20.3) shifts ===")
    for m, rows in om.items():
        print(f" {m}: " + "  ".join(f"{r['z']}:{r['omega_shift']:+.3f}"
                                    for r in rows))


if __name__ == "__main__":
    main()
