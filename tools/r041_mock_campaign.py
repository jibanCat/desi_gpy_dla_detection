#!/usr/bin/env python
"""r041_mock_campaign.py — R-041C / R-041E: the corrected single-injection calibration
transferred to HCD-free mock substrates (2LPT loa-0 twin; London jura-124 clean
sightlines) with the mean-flux-only high-z extrapolation

    F_high = F_low · F̄_high(z + Δz) / F̄_low(z),

where F̄_low is the substrate's MEASURED forest mean flux (tools/r041_mock_meanflux.py,
truth continuum) and F̄_high the high-z mean-flux model evaluated at z + Δz. The
injection is the same noise-preserving operation as the real campaign
(injection/noise_preserving.py through injection/coadd_injection.py), the N grid /
trials, SNR_REDSIDE strata, path-proportional z draw and constant-Δz collar are those of
the real fiducial config, and the finder configuration is the resolved BASELINE.env of the
real high-z run (A4) with only the mock-mode inputs and the τ_eff seed changed:

  * PREV_TAU_0 is rescaled so the finder's mean-flux seed matches the extrapolated forest
    at the path-weighted mean injected z (τ0' = τ0 ((1+z̄+Δz)/(1+z̄))^β) — on real high-z
    data the seed is evaluated at the true (high) pixel z; without this the mock arm would
    add a seed/forest mismatch absent from the real analysis. τ-EB then refits per
    spectrum exactly as in production. This choice is recorded and PI-visible.

Arms (R-041E, 2LPT only — the loa-0 twin shares every skewer with loa-124):
  random    : z_inj ∝ dX/dz inside the window (as the real campaign);
  clustered : the SAME sightlines and log N, z_inj = the loa-124 truth DLA position
              (log N_truth ≥ 20.0) on that sightline — the LyaCoLoRe b_DLA = 2 placement
              (density-peak correlated forest) on the HCD-free twin.
Outputs (per arm): injected spectra-16 tree, injection_truth.fits, plan CSV, qsocat FITS,
GL env, build summary with hashes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "tools"))
from r041_plan import dXdz, seed_for, draw_z, C_KMS  # noqa: E402
from injection.coadd_injection import build_clean_table, write_campaign  # noqa: E402

LYA = 1215.67
COLLAR_KMS = 3000.0
LAM_RF = (1025.0, 1216.0)
ZBIN_REAL = (3.8, 5.0)
SNR_STRATA = [2.0, 3.0, 4.0, 5.0, 7.0, np.inf]
OM = 0.279
# native multi-HCD arm (P1, PI ruling 2026-09-02 §5-§9): the HCD-BEARING mock itself (no injection; only the mean-flux
# extrapolation), truth = the mock's own absorber catalogue, sightlines selected by their TRUTH multiplicity in the window.
NATIVE = {
    "2lpt": dict(mockdir="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124",
                 truth="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits", zcol="Z", ncol="NHI",
                 snr="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/snr_cat.fits", release="v2.8.5"),
    "london": dict(mockdir="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/london/qq_desi_y3/v5.9.5/mock-0/jura-124",
                   truth="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/london/qq_desi_y3/v5.9.5/mock-0/jura-124/dla_cat.fits", zcol="Z_DLA", ncol="NHI",
                   snr=None, release="v5.9.5"),
}
FAMILIES = {
    "2lpt": dict(mockdir="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-0",
                 twin_truth="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits",
                 bal="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits",
                 snr="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-0/snr_cat.fits",
                 exclude=[], release="v2.8.5",
                 note="loa-0 = HCD-free twin of loa-124 (same TARGETIDs/skewers); BAL TARGETIDs of loa-124 excluded"),
    "london": dict(mockdir="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/london/qq_desi_y3/v5.9.5/mock-0/jura-124",
                   twin_truth=None,
                   bal="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/london/qq_desi_y3/v5.9.5/mock-0/jura-124/bal_cat.fits",
                   snr=None, exclude=["/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/london/qq_desi_y3/v5.9.5/mock-0/jura-124/dla_cat.fits"],
                   release="v5.9.5", note="clean = zcat - dla_cat - bal_cat (no HCD-free twin exists)"),
}


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def _X(z):
    g = np.linspace(0.0, z, 2000)
    return float(np.trapezoid(dXdz(g), g))


def window(zq, dz):
    coll = COLLAR_KMS / C_KMS
    zlo = max(3600.0 / LYA - 1.0, LAM_RF[0] * (1 + zq) / LYA - 1.0 + coll)
    zhi = min(zq - coll, LAM_RF[1] * (1 + zq) / LYA - 1.0 - coll)
    lo, hi = max(zlo, ZBIN_REAL[0] - dz), min(zhi, ZBIN_REAL[1] - dz)
    return zlo, zhi, lo, hi


def parse_env(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if v != "(unset)":
            out[k] = v
    return out


def native_selection(zcat_tids, zcat_z, snr_by_tid, bal_tids, truth_by_tid, *, dz, zqso_min, companion_logN=20.0, primary_logN=20.3,
                     singles_per_multi=1.0, max_multi=None, rng=None, n_strata=5):
    """Select native multi-HCD sightlines (truth multiplicity >= 2 inside the high-z-emulation bin window) and a
    stratum-matched single-HCD reference (exactly one truth absorber >= companion_logN in the window, that absorber
    >= primary_logN). Returns (population rows, truth rows) in the R-041 schemas. Deterministic given rng."""
    rng = rng or np.random.default_rng(0)
    multi, single = [], []
    for tid, zq in zip(zcat_tids, zcat_z):
        tid = int(tid); zq = float(zq)
        if zq < zqso_min or tid in bal_tids:
            continue
        snr = snr_by_tid.get(tid)
        if snr is None or not np.isfinite(snr) or snr <= 2.0:
            continue
        zlo, zhi, lo, hi = window(zq, dz)
        if hi <= lo:
            continue
        ab = sorted([(z, n) for z, n in truth_by_tid.get(tid, []) if lo <= z <= hi and n >= companion_logN])
        if not ab:
            continue
        s = int(np.digitize(snr, SNR_STRATA) - 1)
        rec = dict(TARGETID=tid, z_qso=zq, snr=snr, stratum=s, zlo=lo, zhi=hi, zlo_bin=lo, zhi_bin=hi, dX_bin=_X(hi) - _X(lo), m_true=len(ab), absorbers=ab)
        if len(ab) >= 2:
            multi.append(rec)
        elif ab[0][1] >= primary_logN:
            single.append(rec)
    if max_multi is not None and len(multi) > max_multi:
        idx = rng.choice(len(multi), max_multi, replace=False); multi = [multi[i] for i in sorted(idx)]
    chosen = list(multi)
    for s in range(n_strata):
        need = int(round(singles_per_multi * sum(1 for r in multi if r["stratum"] == s)))
        cand = [r for r in single if r["stratum"] == s]
        if need and cand:
            idx = rng.choice(len(cand), min(need, len(cand)), replace=False); chosen += [cand[i] for i in sorted(idx)]
    pop_rows, truth_rows = [], []
    for r in chosen:
        pop_rows.append(dict(TARGETID=r["TARGETID"], z_qso=r["z_qso"], snr=r["snr"], zlo=r["zlo"], zhi=r["zhi"], zlo_bin=r["zlo_bin"], zhi_bin=r["zhi_bin"],
                             dX_bin=r["dX_bin"], stratum=r["stratum"], m_true=r["m_true"], has_cand_ge20=0))
        zs = [z for z, n in r["absorbers"]]
        for k, (z, n) in enumerate(r["absorbers"]):
            others = [abs(z2 - z) for z2 in zs if z2 != z]
            dv = (C_KMS * min(others) / (1.0 + z)) if others else ""
            truth_rows.append(dict(TARGETID=r["TARGETID"], wave=0, inj_idx=k, logN=round(float(n), 4), z_inj=round(float(z), 6), stratum=r["stratum"], snr=r["snr"],
                                   z_qso=r["z_qso"], has_cand_ge20=0, pair_class="native", dv_kms=(round(dv, 1) if dv != "" else ""), pair_logN="",
                                   m_true=r["m_true"], method="native", meanflux_model="extrapolated"))
    return pop_rows, truth_rows


def build_native_arm(a, fam, cfg):
    """P1 native multi-HCD arm: rescale-only campaign on the HCD-bearing mock; truth = native absorbers in window."""
    nat = NATIVE[a.family]
    os.makedirs(a.out_root, exist_ok=True)
    rng = np.random.default_rng(seed_for(0, 1, a.seed_salt))
    zcat = Table(fits.open(os.path.join(nat["mockdir"], "zcat.fits"))[1].data)
    bal = set(np.asarray(Table(fits.open(fam["bal"])[1].data)["TARGETID"], dtype=np.int64).tolist())
    snr_path = a.snr_cat or nat["snr"] or fam["snr"]
    snr_t = Table(fits.open(snr_path)[1].data)
    snr_col = "SNR_REDSIDE" if "SNR_REDSIDE" in snr_t.colnames else [c for c in snr_t.colnames if "SNR" in c.upper()][0]
    snr_by = {int(t): float(v) for t, v in zip(np.asarray(snr_t["TARGETID"], dtype=np.int64), snr_t[snr_col])}
    tt = fits.open(nat["truth"])[1].data
    N = np.asarray(tt[nat["ncol"]], float); N = np.log10(N) if np.nanmax(N) > 100 else N
    truth_by = {}
    for t, z, n in zip(np.asarray(tt["TARGETID"], dtype=np.int64), np.asarray(tt[nat["zcol"]], float), N):
        truth_by.setdefault(int(t), []).append((float(z), float(n)))
    pop, truth = native_selection(np.asarray(zcat["TARGETID"], dtype=np.int64), np.asarray(zcat["Z"], float), snr_by, bal, truth_by, dz=a.delta_z,
                                  zqso_min=a.zqso_min, singles_per_multi=a.native_singles_per_multi, max_multi=a.native_max_multi, rng=rng)
    # healpix from the mock's snr/zcat carrier if present, else from RA/DEC (DESI nside 16 nested) via build_clean_table's convention
    hp_by = {}
    if "HEALPIX" in snr_t.colnames:
        hp_by = {int(t): int(h) for t, h in zip(np.asarray(snr_t["TARGETID"], dtype=np.int64), snr_t["HEALPIX"])}
    else:
        import healpy
        ra_col = "RA" if "RA" in zcat.colnames else "TARGET_RA"; dec_col = "DEC" if "DEC" in zcat.colnames else "TARGET_DEC"
        for t, ra, dec in zip(np.asarray(zcat["TARGETID"], dtype=np.int64), zcat[ra_col], zcat[dec_col]):
            hp_by[int(t)] = int(healpy.ang2pix(16, np.radians(90.0 - float(dec)), np.radians(float(ra)), nest=True))
    for r in pop:
        r["healpix"] = hp_by[r["TARGETID"]]
    if a.n_healpix and len({r["healpix"] for r in pop}) > a.n_healpix:      # bound the number of coadds: keep the seeded-first healpix
        hps = sorted({r["healpix"] for r in pop}); keep = set(rng.permutation(hps)[: a.n_healpix].tolist())
        pop = [r for r in pop if r["healpix"] in keep]; kept = {r["TARGETID"] for r in pop}; truth = [t for t in truth if t["TARGETID"] in kept]
    arm_root = os.path.join(a.out_root, "native"); os.makedirs(arm_root, exist_ok=True)
    with open(os.path.join(a.out_root, "population_native.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pop[0].keys())); w.writeheader(); w.writerows(pop)
    truth_csv = os.path.join(arm_root, "native_truth.csv")
    with open(truth_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(truth[0].keys())); w.writeheader(); w.writerows(truth)
    zbar = float(np.average([0.5 * (r["zlo"] + r["zhi"]) for r in pop], weights=[r["dX_bin"] for r in pop]))
    base = parse_env(a.baseline_env)
    tau0 = float(base.get("PREV_TAU_0", 0.00246)); beta = float(base.get("PREV_BETA", 3.62))
    tau0_matched = tau0 * ((1 + zbar + a.delta_z) / (1 + zbar)) ** beta
    mf = json.load(open(a.meanflux_json))
    meanflux = {"fiducial": {"z": mf["z_center"], "taueff": mf["taueff"]}, "model": a.meanflux_model, "delta_z": a.delta_z}
    manifest = [dict(inj_id=i, campaign="N", method="coadd", target_id=r["TARGETID"], healpix=r["healpix"], z_qso=r["z_qso"], snr_bin=r["stratum"], native_snr=r["snr"],
                     logN_true=np.nan, z_true=np.nan, num_lines=3, arm="native", rescale_only=True, m_true=r["m_true"], zlo_bin=r["zlo_bin"], zhi_bin=r["zhi_bin"],
                     dX_bin=r["dX_bin"], delta_z=a.delta_z) for i, r in enumerate(pop)]
    plan_path = os.path.join(arm_root, "plan.csv")
    with open(plan_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys())); w.writeheader(); w.writerows(manifest)
    truth_fits = write_campaign(manifest, None, out_root=arm_root, mockdir=nat["mockdir"], num_lines=3, method="variance_preserving", meanflux=meanflux,
                                seed_salt=f"{a.seed_salt}:native")
    tids = np.array([m["target_id"] for m in manifest], dtype=np.int64)
    zc = zcat[np.isin(np.asarray(zcat["TARGETID"], dtype=np.int64), tids)]
    q = Table({"TARGETID": np.asarray(zc["TARGETID"], dtype=np.int64), "Z": np.asarray(zc["Z"], dtype=float)})
    ra_col = "RA" if "RA" in zc.colnames else "TARGET_RA"; dec_col = "DEC" if "DEC" in zc.colnames else "TARGET_DEC"
    q["RA"] = np.asarray(zc[ra_col], dtype=float); q["DEC"] = np.asarray(zc[dec_col], dtype=float); q["TARGET_RA"] = q["RA"]; q["TARGET_DEC"] = q["DEC"]
    qpath = arm_root + "_qsocat.fits"; q.write(qpath, overwrite=True)
    n_files = len({m["healpix"] for m in manifest})
    env_path = arm_root + ".env"
    keep = ["LEARNED_FILE", "CATALOG_NAME", "LOS_CATALOG", "DLA_CATALOG", "DLA_SAMPLES_FILE", "SUB_DLA_SAMPLES_FILE", "NUM_DLA_SAMPLES",
            "NUM_SUBDLA_SAMPLES", "MAX_DLAS", "SINGLE_ABSORBER_MODEL", "FILTER_LOW_LIKELIHOOD", "MAX_LAMBDA", "MIN_LAMBDA", "DLAMBDA", "K",
            "NUM_FOREST_LINES", "NUM_LINES", "BALMASK", "PREV_BETA", "MAX_NOISE_VARIANCE", "BATCH_SIZE", "ENABLE_TAU_EB", "TAU_EB_OBJECTIVE", "EARLY_STOP_MODE"]
    with open(env_path, "w") as fh:
        fh.write(f'# P1 native multi-HCD arm — family {a.family}; HCD-bearing mock {nat["mockdir"]} (no injection; mean-flux extrapolation only);\n'
                 f'# finder settings = resolved BASELINE.env of the MAX4 P0 real run ({a.baseline_env}); PREV_TAU_0 matched to the extrapolated forest\n'
                 f'source "{REPO}/slurm/greatlakes/production/_base_gl.env"\nMODE="mock"\nRUN_DATE="${{RUN_DATE:-$(date +%Y%m%d)}}"\n'
                 f'RUN_NAME="r041_mock_{a.family}_native_MAX4"\nMOCKDIR="{arm_root}"\nQSOCAT="{qpath}"\nRELEASE="{nat["release"]}"\nOUTDIR="{arm_root}_outputs/"\n')
        for k in keep:
            if k in base:
                fh.write(f'{k}="{base[k]}"\n')
        fh.write(f'PREV_TAU_0="{tau0_matched:.6g}"\nPAIR_PRIOR_MODE="off"\nDLA_BIAS="2.0"\nMAX_WORKERS={a.gl_cpus}\nGL_SLURM_MEM=64G\nGL_SLURM_TIME=08:00:00\n'
                 f'OUTER_MAX_INDEX={n_files - 1}\nOUTER_STEP={a.outer_window}\nOUTER_WINDOW={a.outer_window}\nTRUTH_CAT="{truth_fits}"\nBAL_CAT="{fam["bal"]}"\nREPO_ROOT="{REPO}"\n')
    summ = dict(family=a.family, arm="native", native_mockdir=nat["mockdir"], native_truth=nat["truth"], native_truth_sha256=_sha(nat["truth"]), snr_cat=snr_path,
                n_sightlines=len(pop), n_multi=sum(1 for r in pop if r["m_true"] >= 2), n_single_reference=sum(1 for r in pop if r["m_true"] == 1),
                m_true_hist={str(m): sum(1 for r in pop if min(r["m_true"], 3) == m) for m in (1, 2, 3)}, n_truth_absorbers=len(truth), n_healpix=n_files,
                zbar_window=zbar, delta_z=a.delta_z, PREV_TAU_0_matched=tau0_matched, PREV_TAU_0_production=tau0, meanflux_model_high=a.meanflux_model,
                meanflux_low_measured=a.meanflux_json, baseline_env=a.baseline_env, seed_salt=a.seed_salt, companion_logN=20.0, primary_logN=20.3,
                singles_per_multi=a.native_singles_per_multi, max_multi=a.native_max_multi, plan=plan_path, plan_sha256=_sha(plan_path), truth_csv=truth_csv,
                truth_csv_sha256=_sha(truth_csv), truth_manifest=truth_fits, truth_manifest_sha256=_sha(truth_fits), qsocat=qpath, qsocat_sha256=_sha(qpath),
                env=env_path, generator_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip())
    json.dump(summ, open(os.path.join(a.out_root, "build_summary_native.json"), "w"), indent=1)
    print(json.dumps({k: summ[k] for k in ("family", "n_sightlines", "n_multi", "n_single_reference", "m_true_hist", "n_truth_absorbers", "n_healpix", "PREV_TAU_0_matched")}, indent=1))
    print(f"[native] built: {n_files} files, {len(pop)} sightlines -> {arm_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=list(FAMILIES), required=True)
    ap.add_argument("--config", required=True, help="real fiducial config (logN grid, trials, strata)")
    ap.add_argument("--meanflux-json", required=True, help="tools/r041_mock_meanflux.py output for this substrate")
    ap.add_argument("--meanflux-model", default="finder_fiducial")
    ap.add_argument("--delta-z", type=float, default=1.0)
    ap.add_argument("--zqso-min", type=float, default=3.0)
    ap.add_argument("--n-healpix", type=int, default=600, help="upper bound on coadds to build")
    ap.add_argument("--need-factor", type=float, default=1.3, help="population per stratum >= need_factor x trials")
    ap.add_argument("--arms", default="random", help="comma list: random[,clustered] (clustered: 2lpt only)")
    ap.add_argument("--trial-scale", type=float, default=1.0, help="multiply the config's trials per (N, stratum)")
    ap.add_argument("--baseline-env", required=True, help="resolved BASELINE.env of the real high-z run (A4)")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--seed-salt", required=True)
    ap.add_argument("--snr-cat", default=None, help="override SNR catalog (London: built from the previous production run)")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--gl-cpus", type=int, default=16)
    ap.add_argument("--outer-window", type=int, default=20)
    ap.add_argument("--native-singles-per-multi", type=float, default=1.0, help="native arm: single-HCD reference sightlines per multi sightline, per stratum")
    ap.add_argument("--native-max-multi", type=int, default=None, help="native arm: cap on multi-HCD sightlines (seeded subsample)")
    a = ap.parse_args()
    fam = FAMILIES[a.family]
    cfg = json.load(open(a.config))
    arms = a.arms.split(",")
    if arms == ["native"]:
        return build_native_arm(a, fam, cfg)
    if "native" in arms:
        raise SystemExit("--arms native must be run on its own (separate product root)")
    if "clustered" in arms and fam["twin_truth"] is None:
        raise SystemExit("clustered arm needs an HCD-free twin with a truth catalogue (2lpt only)")
    os.makedirs(a.out_root, exist_ok=True)
    rng = np.random.default_rng(seed_for(0, 0, a.seed_salt))

    # ---- population: clean sightlines, z_qso >= zqso_min, SNR_REDSIDE > 2, healpix subset
    zcat = Table(fits.open(os.path.join(fam["mockdir"], "zcat.fits"))[1].data)
    excl = Table({"TARGETID": np.zeros(0, dtype=np.int64)})
    for e in fam["exclude"]:
        t = Table(fits.open(e)[1].data)
        excl = Table({"TARGETID": np.concatenate([np.asarray(excl["TARGETID"], dtype=np.int64), np.asarray(t["TARGETID"], dtype=np.int64)])})
    bal = Table(fits.open(fam["bal"])[1].data)
    snr_path = a.snr_cat or fam["snr"]
    snr = Table(fits.open(snr_path)[1].data)
    clean = build_clean_table(zcat, excl, bal, snr)
    ok = (np.asarray(clean["Z"]) >= a.zqso_min) & np.isfinite(np.asarray(clean["SNR_REDSIDE"])) & (np.asarray(clean["SNR_REDSIDE"]) > 2.0)
    clean = clean[ok]
    truth_by_tid = {}
    if fam["twin_truth"]:
        tt = fits.open(fam["twin_truth"])[1].data
        for t, z, n in zip(np.asarray(tt["TARGETID"], dtype=np.int64), tt["Z"], tt["NHI"]):
            truth_by_tid.setdefault(int(t), []).append((float(z), float(n)))
    # healpix are added in a seeded random order until every stratum holds >= need_factor x
    # the trials it must serve (one use per sightline), bounding the number of coadds to build
    grid = cfg["logN_grid"]; trials = [max(1, int(round(t * a.trial_scale))) for t in cfg["trials_per_logN_per_stratum"]]
    n_strata = len(SNR_STRATA) - 1
    need = int(np.ceil(sum(trials) * a.need_factor))
    hp_all = np.unique(np.asarray(clean["HEALPIX"])); hp_order = rng.permutation(hp_all)
    hp_col = np.asarray(clean["HEALPIX"])
    pop, hp_sel, count = [], [], np.zeros(n_strata, int)
    for hp in hp_order:
        if (count >= need).all() or len(hp_sel) >= a.n_healpix:
            break
        hp_sel.append(int(hp))
        for r in clean[hp_col == hp]:
            zq = float(r["Z"]); zlo, zhi, lo, hi = window(zq, a.delta_z)
            if hi <= lo:
                continue
            s = int(np.digitize(float(r["SNR_REDSIDE"]), SNR_STRATA) - 1)
            truth_in = [(z, n) for z, n in truth_by_tid.get(int(r["TARGETID"]), []) if lo <= z <= hi and n >= 20.0]
            if "clustered" in arms and not truth_in:
                continue
            count[s] += 1
            pop.append(dict(TARGETID=int(r["TARGETID"]), healpix=int(r["HEALPIX"]), z_qso=zq, snr=float(r["SNR_REDSIDE"]), zlo=zlo, zhi=zhi,
                            zlo_bin=lo, zhi_bin=hi, dX_bin=_X(hi) - _X(lo), stratum=s,
                            z_truth=(max(truth_in, key=lambda x: x[1])[0] if truth_in else None),
                            logN_truth=(max(truth_in, key=lambda x: x[1])[1] if truth_in else None)))
    hp_sel = np.array(sorted(hp_sel))
    if not (count >= need).all():
        raise SystemExit(f"population per stratum {count.tolist()} < need {need} after {len(hp_sel)} healpix; raise --n-healpix")
    pop_path = os.path.join(a.out_root, "population.csv")
    with open(pop_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pop[0].keys())); w.writeheader(); w.writerows(pop)

    # ---- plan (single mode, one use per sightline; identical draws for every arm)
    plan = []
    for s in range(n_strata):
        rows = [r for r in pop if r["stratum"] == s]
        if not rows:
            continue
        w0 = np.array([r["dX_bin"] for r in rows]); used = np.zeros(len(rows), bool)
        for logn, ntr in zip(grid, trials):
            for k in range(ntr):
                p = w0 * (~used)
                if p.sum() <= 0:
                    raise SystemExit(f"stratum {s}: population exhausted (n={len(rows)}); raise --n-healpix")
                p /= p.sum(); j = int(rng.choice(len(rows), p=p)); used[j] = True; r = rows[j]
                rz = np.random.default_rng(seed_for(r["TARGETID"], 1000 * int(round(logn * 100)), a.seed_salt))
                z, nre = draw_z(rz, r["zlo_bin"], r["zhi_bin"], [])
                plan.append(dict(TARGETID=r["TARGETID"], healpix=r["healpix"], wave=0, inj_idx=0, logN=float(logn), z_random=round(z, 6),
                                 z_clustered=(None if r["z_truth"] is None else round(r["z_truth"], 6)), logN_truth_at_site=r["logN_truth"],
                                 stratum=s, snr=r["snr"], z_qso=r["z_qso"], has_cand_ge20=0, n_redraw=nre,
                                 zlo_bin=r["zlo_bin"], zhi_bin=r["zhi_bin"], dX_bin=r["dX_bin"]))
    plan.sort(key=lambda r: (r["healpix"], r["TARGETID"]))
    zbar = float(np.average([r["z_random"] for r in plan], weights=[r["dX_bin"] for r in plan]))
    base = parse_env(a.baseline_env)
    tau0 = float(base.get("PREV_TAU_0", 0.00246)); beta = float(base.get("PREV_BETA", 3.62))
    tau0_matched = tau0 * ((1 + zbar + a.delta_z) / (1 + zbar)) ** beta
    mf = json.load(open(a.meanflux_json))
    meanflux = {"fiducial": {"z": mf["z_center"], "taueff": mf["taueff"]}, "model": a.meanflux_model, "delta_z": a.delta_z}
    summ = dict(family=a.family, mockdir=fam["mockdir"], note=fam["note"], n_population=len(pop), population_per_stratum=count.tolist(), n_healpix=int(len(hp_sel)),
                healpix=hp_sel.tolist(), n_plan=len(plan), rows_per_stratum={str(s): sum(1 for r in plan if r["stratum"] == s) for s in range(n_strata)},
                z_random_path_weighted_mean=zbar, delta_z=a.delta_z, effective_z_mean=zbar + a.delta_z, PREV_TAU_0_matched=tau0_matched,
                PREV_TAU_0_production=tau0, PREV_BETA=beta, meanflux_model_high=a.meanflux_model, meanflux_low_measured=a.meanflux_json,
                config=a.config, config_sha256=_sha(a.config), baseline_env=a.baseline_env, snr_cat=snr_path, seed_salt=a.seed_salt,
                trial_scale=a.trial_scale, arms=arms, generator_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip())
    print(json.dumps({k: summ[k] for k in ("family", "n_population", "n_healpix", "n_plan", "rows_per_stratum", "z_random_path_weighted_mean", "effective_z_mean", "PREV_TAU_0_matched")}, indent=1))
    if a.plan_only:
        json.dump(summ, open(os.path.join(a.out_root, "plan_summary.json"), "w"), indent=1)
        return

    # ---- per arm: manifest -> injected tree -> qsocat -> env
    for arm in arms:
        arm_root = os.path.join(a.out_root, arm)
        os.makedirs(arm_root, exist_ok=True)
        zkey = "z_random" if arm == "random" else "z_clustered"
        manifest = []
        for i, r in enumerate(plan):
            manifest.append(dict(inj_id=i, campaign="A", method="coadd", target_id=r["TARGETID"], healpix=r["healpix"], z_qso=r["z_qso"],
                                 snr_bin=r["stratum"], native_snr=r["snr"], logN_true=r["logN"], z_true=r[zkey], num_lines=3,
                                 arm=arm, z_random=r["z_random"], z_clustered=(np.nan if r["z_clustered"] is None else r["z_clustered"]),
                                 zlo_bin=r["zlo_bin"], zhi_bin=r["zhi_bin"], dX_bin=r["dX_bin"], delta_z=a.delta_z))
        plan_path = os.path.join(arm_root, "plan.csv")
        with open(plan_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys())); w.writeheader(); w.writerows(manifest)
        truth = write_campaign(manifest, clean, out_root=arm_root, mockdir=fam["mockdir"], num_lines=3,
                               method="variance_preserving", meanflux=meanflux, seed_salt=f"{a.seed_salt}:{arm}")
        tids = np.array([m["target_id"] for m in manifest], dtype=np.int64)
        zc = zcat[np.isin(np.asarray(zcat["TARGETID"], dtype=np.int64), tids)]
        q = Table({"TARGETID": np.asarray(zc["TARGETID"], dtype=np.int64), "Z": np.asarray(zc["Z"], dtype=float)})
        ra_col = "RA" if "RA" in zc.colnames else "TARGET_RA"; dec_col = "DEC" if "DEC" in zc.colnames else "TARGET_DEC"
        q["RA"] = np.asarray(zc[ra_col], dtype=float); q["DEC"] = np.asarray(zc[dec_col], dtype=float)
        q["TARGET_RA"] = q["RA"]; q["TARGET_DEC"] = q["DEC"]
        qpath = os.path.join(arm_root + "_qsocat.fits"); q.write(qpath, overwrite=True)
        n_files = len(set(m["healpix"] for m in manifest))
        env_path = arm_root + ".env"
        keep = ["LEARNED_FILE", "CATALOG_NAME", "LOS_CATALOG", "DLA_CATALOG", "DLA_SAMPLES_FILE", "SUB_DLA_SAMPLES_FILE", "NUM_DLA_SAMPLES",
                "NUM_SUBDLA_SAMPLES", "MAX_DLAS", "SINGLE_ABSORBER_MODEL", "FILTER_LOW_LIKELIHOOD", "MAX_LAMBDA", "MIN_LAMBDA", "DLAMBDA", "K",
                "NUM_FOREST_LINES", "NUM_LINES", "BALMASK", "PREV_BETA", "MAX_NOISE_VARIANCE", "BATCH_SIZE", "ENABLE_TAU_EB", "TAU_EB_OBJECTIVE", "EARLY_STOP_MODE"]
        with open(env_path, "w") as fh:
            fh.write(f'# R-041C/E mock campaign env — family {a.family}, arm {arm}; finder settings = resolved BASELINE.env of the real\n'
                     f'# high-z run ({a.baseline_env}); PREV_TAU_0 matched to the extrapolated forest (see plan_summary.json)\n'
                     f'source "{REPO}/slurm/greatlakes/production/_base_gl.env"\nMODE="mock"\nRUN_DATE="${{RUN_DATE:-$(date +%Y%m%d)}}"\n'
                     f'RUN_NAME="r041_mock_{a.family}_{arm}"\nMOCKDIR="{arm_root}"\nQSOCAT="{qpath}"\nRELEASE="{fam["release"]}"\nOUTDIR="{arm_root}_outputs/"\n')
            for k in keep:
                if k in base:
                    fh.write(f'{k}="{base[k]}"\n')
            fh.write(f'PREV_TAU_0="{tau0_matched:.6g}"\nPAIR_PRIOR_MODE="off"\nDLA_BIAS="2.0"\n'
                     f'MAX_WORKERS={a.gl_cpus}\nGL_SLURM_MEM=64G\nGL_SLURM_TIME=08:00:00\n'
                     # launch_gl.sh loops i = 0 .. OUTER_MAX_INDEX inclusive in steps of OUTER_STEP; the inner
                     # script slices the sorted spectra list by index, so the last chunk must start < n_files
                     f'OUTER_MAX_INDEX={n_files - 1}\nOUTER_STEP={a.outer_window}\nOUTER_WINDOW={a.outer_window}\n'
                     f'TRUTH_CAT="{truth}"\nBAL_CAT="{fam["bal"]}"\n')
        summ.setdefault("arms_built", {})[arm] = dict(root=arm_root, plan=plan_path, plan_sha256=_sha(plan_path), truth_manifest=truth,
                                                    truth_sha256=_sha(truth), qsocat=qpath, qsocat_sha256=_sha(qpath), env=env_path, n_files=n_files,
                                                    n_injections=len(manifest))
        print(f"[{arm}] built: {n_files} files, {len(manifest)} injections -> {arm_root}")
    json.dump(summ, open(os.path.join(a.out_root, "build_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
