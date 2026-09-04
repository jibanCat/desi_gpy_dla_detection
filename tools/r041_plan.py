#!/usr/bin/env python
"""r041_plan.py — realized injection plans for the R-041 high-z campaigns (fiducial single
injection, the old-vs-corrected comparison, the mean-flux variants, the real-spectrum pair
stress test). Deterministic (seeded per TARGETID/injection) so a plan is reproducible from
its config; every injection is one row with the sightline, wave, log N, z_inj, stratum and
the design provenance.

Design rules (PI, R-041A A1/A2):
  * ONE injected absorber per sightline per wave (a sightline reused across waves carries a
    different injection in a different archive; never two in one spectrum in the fiducial);
  * SNR_REDSIDE strata from the population (population CSV `stratum`);
  * z_inj drawn proportional to the absorption path of the measurement: sightlines are drawn
    with probability ∝ dX_bin within a stratum and z_inj ∝ dX/dz inside [zlo_bin, zhi_bin]
    (constant-Δz 3000 km/s collar geometry, identical to the measurement);
  * N grid with direct support in [20.3, 20.5) and denser sampling around 20.3; trials
    per (N, stratum) from the config (adaptive weights);
  * collision rule (pre-declared, unchanged from H2): re-draw z_inj if within 5000 km/s of
    any existing production candidate (P_DLA > 0.5) on that sightline, ≤ 100 attempts, else
    drop and log. Sightlines WITH an existing candidate ≥ 20.0 stay in the population (the
    measurement's own population; MAX_DLAS = 1 competition is part of the data-matched
    completeness) and are flagged `has_cand_ge20` for stratified reporting.
Pair mode (R-041D): two absorbers per sightline on candidate-free sightlines only, with a
prescribed velocity separation class and column pair.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os

import numpy as np

C_KMS = 299792.458
COLLISION_KMS = 5000.0
OM = 0.279


def dXdz(z):
    return (1.0 + z) ** 2 / np.sqrt(OM * (1.0 + z) ** 3 + 1.0 - OM)


def seed_for(tid, k, salt):
    h = hashlib.sha256(f"{salt}:{tid}:{k}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def draw_z(rng, zlo, zhi, cands, n_attempt=100):
    """z ∝ dX/dz in [zlo, zhi] with the collision rule; returns (z, n_redraw) or (None, n)."""
    for i in range(n_attempt):
        # inverse-CDF on a fine grid of dX/dz
        g = np.linspace(zlo, zhi, 400)
        w = dXdz(g); cdf = np.cumsum(w); cdf /= cdf[-1]
        z = float(np.interp(rng.random(), cdf, g))
        if all(abs(z - zc) / (1.0 + zc) * C_KMS > COLLISION_KMS for zc in cands):
            return z, i
    return None, n_attempt


def parse_cands(s):
    if not s:
        return []
    return [float(x.split(":")[0]) for x in s.split(";")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    pop = list(csv.DictReader(open(a.population)))
    for r in pop:
        for k in ("z_qso", "snr", "zlo", "zhi", "zlo_bin", "zhi_bin", "dX_bin"):
            r[k] = float(r[k])
        r["stratum"] = int(r["stratum"]); r["has_cand_ge20"] = int(r["has_cand_ge20"]); r["TARGETID"] = int(r["TARGETID"])
        r["cands"] = parse_cands(r.get("cand", ""))
    salt = cfg["seed_salt"]
    rng_master = np.random.default_rng(seed_for(0, 0, salt))
    mode = cfg.get("mode", "single")
    n_strata = int(cfg.get("n_strata", 5))
    plan, dropped = [], []
    if mode in ("single",):
        grid = cfg["logN_grid"]; trials = cfg["trials_per_logN_per_stratum"]
        assert len(grid) == len(trials)
        only_clean = bool(cfg.get("only_candidate_free_sightlines", False))
        for s in range(n_strata):
            rows = [r for r in pop if r["stratum"] == s and (not only_clean or r["has_cand_ge20"] == 0)]
            if not rows:
                continue
            w0 = np.array([r["dX_bin"] for r in rows])
            # per-sightline wave counters guarantee one injection per sightline per wave; a
            # sightline is retired after `max_uses_per_sightline` draws (bounds the number of
            # archive waves; truncates the path weighting only for the highest-path sightlines)
            cap = int(cfg.get("max_uses_per_sightline", 3))
            use = {r["TARGETID"]: 0 for r in rows}
            for logn, ntr in zip(grid, trials):
                for k in range(int(ntr)):
                    p = w0 * np.array([use[r["TARGETID"]] < cap for r in rows], float)
                    if p.sum() <= 0:
                        raise SystemExit(f"stratum {s}: all sightlines exhausted at cap {cap}")
                    p /= p.sum()
                    j = int(rng_master.choice(len(rows), p=p))
                    r = rows[j]
                    wave = use[r["TARGETID"]]; use[r["TARGETID"]] += 1
                    rng = np.random.default_rng(seed_for(r["TARGETID"], 1000 * int(round(logn * 100)) + wave, salt))
                    z, nre = draw_z(rng, r["zlo_bin"], r["zhi_bin"], r["cands"])
                    rec = dict(TARGETID=r["TARGETID"], wave=wave, inj_idx=0, logN=float(logn), z_inj=None if z is None else round(z, 6),
                               stratum=s, snr=r["snr"], z_qso=r["z_qso"], has_cand_ge20=r["has_cand_ge20"], n_redraw=nre,
                               zlo_bin=r["zlo_bin"], zhi_bin=r["zhi_bin"], dX_bin=r["dX_bin"])
                    (plan if z is not None else dropped).append(rec)
    elif mode == "pairs":
        # candidate-free sightlines only; each row = a pair (two injections, same sightline, same wave)
        classes = cfg["separation_classes_kms"]          # {"wide": [10000, 30000], "partial": [1500, 4000], "blend": [200, 1000]}
        pairs = cfg["logN_pairs"]                         # [[20.5, 20.5], [21.0, 20.3], ...]
        ntr = int(cfg["trials_per_class_per_pair"])
        for s in range(n_strata):
            rows = [r for r in pop if r["stratum"] == s and r["has_cand_ge20"] == 0]
            if not rows:
                continue
            w0 = np.array([r["dX_bin"] for r in rows])
            cap = int(cfg.get("max_uses_per_sightline", 3))
            use = {r["TARGETID"]: 0 for r in rows}
            for cname, (dv_lo, dv_hi) in classes.items():
                for (n1, n2) in pairs:
                    for k in range(ntr):
                        p = w0 * np.array([use[r["TARGETID"]] < cap for r in rows], float); p /= p.sum()
                        j = int(rng_master.choice(len(rows), p=p)); r = rows[j]
                        wave = use[r["TARGETID"]]; use[r["TARGETID"]] += 1
                        rng = np.random.default_rng(seed_for(r["TARGETID"], hash((cname, n1, n2, wave)) & 0xFFFF, salt))
                        dv = float(rng.uniform(dv_lo, dv_hi)) * (1 if rng.random() < 0.5 else -1)
                        # first absorber ∝ path, the partner at dv (must stay inside the bin window)
                        z1, nre = draw_z(rng, r["zlo_bin"], r["zhi_bin"], r["cands"])
                        if z1 is None:
                            dropped.append(dict(TARGETID=r["TARGETID"], wave=wave, reason="collision")); continue
                        z2 = z1 + dv / C_KMS * (1.0 + z1)
                        if not (r["zlo_bin"] <= z2 <= r["zhi_bin"]):
                            z2 = z1 - dv / C_KMS * (1.0 + z1)
                        if not (r["zlo_bin"] <= z2 <= r["zhi_bin"]):
                            dropped.append(dict(TARGETID=r["TARGETID"], wave=wave, reason="partner outside window")); continue
                        for idx, (nn, zz) in enumerate(((n1, z1), (n2, z2))):
                            plan.append(dict(TARGETID=r["TARGETID"], wave=wave, inj_idx=idx, logN=float(nn), z_inj=round(zz, 6), stratum=s, snr=r["snr"],
                                             z_qso=r["z_qso"], has_cand_ge20=0, n_redraw=nre, pair_class=cname, dv_kms=round(abs(dv), 1),
                                             pair_logN=f"{n1}+{n2}", zlo_bin=r["zlo_bin"], zhi_bin=r["zhi_bin"], dX_bin=r["dX_bin"]))
    else:
        raise SystemExit(f"unknown mode {mode}")
    plan.sort(key=lambda r: (r["wave"], r["TARGETID"], r["inj_idx"]))
    fields = sorted({k for r in plan for k in r}, key=lambda k: (k != "TARGETID", k))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(plan)
    waves = sorted({r["wave"] for r in plan})
    summ = dict(n_rows=len(plan), n_dropped=len(dropped), n_waves=len(waves), rows_per_wave={str(wv): sum(1 for r in plan if r["wave"] == wv) for wv in waves},
                n_sightlines=len({r["TARGETID"] for r in plan}), config=cfg, config_sha256=hashlib.sha256(open(a.config, "rb").read()).hexdigest(),
                population_sha256=hashlib.sha256(open(a.population, "rb").read()).hexdigest(),
                plan_sha256=hashlib.sha256(open(a.out, "rb").read()).hexdigest(), dropped=dropped[:50])
    if mode == "single":
        summ["rows_per_stratum"] = {str(s): sum(1 for r in plan if r["stratum"] == s) for s in range(n_strata)}
        summ["rows_with_candidate_ge20"] = sum(r["has_cand_ge20"] for r in plan)
        zb = [3.8, 4.25, 4.5, 5.0]
        summ["rows_per_zbin"] = {f"[{zb[i]},{zb[i+1]})": sum(1 for r in plan if zb[i] <= r["z_inj"] < zb[i + 1]) for i in range(3)}
    with open(a.out + ".summary.json", "w") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("config", "dropped")}, indent=1))


if __name__ == "__main__":
    main()
