#!/usr/bin/env python
"""Paired real-spectrum associated-absorption response campaign (PI ruling 2026-09-03 §3–§10).

Arms: A = the existing A_shared fiducial (no rebuild); B = representative associated absorption; C = plausible upper envelope. Every arm re-uses the
fiducial plan (same TARGETID / wave / inj_idx / N_HI / z_DLA / stratum), the same source archive, the same A_shared prescription and the same noise
seed policy (shared0), so the DLA transmission and the synthetic noise are IDENTICAL across arms; only the metal transmission (plan column
metals_json, built from the FROZEN model JSON with per-object deterministic seeds) differs.

Stages: plans | build | envs | launch | verify
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, REPO)
from injection.associated_absorption import LINES, metal_transmission, equivalent_width_A  # noqa: E402

ROOT_MAX4 = "/scratch/cavestru_root/cavestru0/mfho/r041_max4_highz_2026-09"
FID = "/scratch/cavestru_root/cavestru0/mfho/r041_highz_repair_2026-08-28/fid"
PLAN_FID = "/scratch/cavestru_root/cavestru0/mfho/r041_highz_repair_2026-08-28/plans/r041a_fiducial.plan.csv"
WAVES = (0, 1, 2)
C_KMS = 299792.458


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def draw_lognormal(rng, median, sigma_ln, lo, hi):
    x = median * np.exp(sigma_ln * rng.standard_normal())
    return float(np.clip(x, lo, hi))


def draw_normal(rng, mu, sigma, lo, hi):
    return float(np.clip(mu + sigma * rng.standard_normal(), lo, hi))


def ew_rest_1526(logN_SiII, comps, z=3.0):
    """Rest EW [Å] of Si II 1526 for a total Si II column split over components comps = [(frac, b_kms, dv_kms), ...] (LSF-free, EW is LSF-invariant)."""
    lam0 = LINES["SiII1526"]["lambda0"] * (1 + z); wave = np.arange(lam0 - 40.0, lam0 + 40.0, 0.2)
    lines = [dict(line="SiII1526", logN=float(np.log10(10 ** logN_SiII * f)), b_kms=b, dv_kms=dv) for f, b, dv in comps]
    return equivalent_width_A(wave, metal_transmission(wave, z, lines, lsf_fwhm_A=None, fine_dlam_A=0.01, pad_A=12.0)) / (1 + z)


def solve_logN_for_W1526(target_W, comps, lo=11.0, hi=16.0):
    """Bisection on the (monotonic) curve of growth: log N(Si II) such that EW_rest(1526) = target_W for the given velocity structure."""
    f_lo = ew_rest_1526(lo, comps) - target_W; f_hi = ew_rest_1526(hi, comps) - target_W
    if f_lo > 0:
        return lo
    if f_hi < 0:
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if ew_rest_1526(mid, comps) - target_W > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def realize(model, arm, logN_HI, key):
    """One associated-absorption realization, deterministic in (model sha, arm, key). Frozen model MAX4_ASSOCIATED_ABSORPTION_MODEL_2026-09-03.md:
    the observable anchor is the per-object rest EW of Si II 1526 (lognormal; N_HI scaling), the velocity structure is Δv90 (lognormal) split into components
    with per-component b; the total Si II column is SOLVED from the drawn W1526 and structure (curve of growth); Si III follows Si II with a scatter; the
    injected lines are Si II 1190/1193/1260 and Si III 1206. Arm C draws W1526 and Δv90 from the upper half of their distributions."""
    seed = int.from_bytes(hashlib.sha256(f"assoc:{arm}:{model['model_sha']}:{key}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    m = model; c = m.get("arm_C", {}) if arm == "C" else {}
    from scipy.stats import norm
    # --- W1526 target (rest Å): lognormal; median scales with N_HI; arm C: quantile restricted to [u_min, u_max]
    w = m["W1526_rest_A"]; med = w["median"] * (10 ** (logN_HI - w["logNHI_ref"])) ** w["NHI_exponent"]
    u = rng.uniform(c.get("u_min", 0.0), c.get("u_max", 1.0)) if c else rng.uniform()
    W1526 = float(np.clip(med * np.exp(w["sigma_ln"] * norm.ppf(u)), w["min"], w["max"]))
    # --- velocity structure: Δv90 lognormal (arm C: upper half), components, per-component b, metal-H I centroid offset
    # model Amendment 1: W1526 and Δv90 are drawn COMONOTONICALLY (same quantile u): the two verified relations [X/H] = 1.55 log ΔV − 4.33 (Ledoux+2006)
    # and [M/H] = 1.46 log W1526 − 0.71 (Jorgenson+2013) imply W1526 ∝ ΔV^1.06, and a saturated line's EW scales with its velocity extent; independent draws
    # produced unphysical (large W1526, narrow structure) pairs whose solved columns hit the search ceiling.
    d = m["dv90_kms"]; u2 = u
    dv90 = float(np.clip(d["median"] * np.exp(d["sigma_ln"] * norm.ppf(u2)), d["min"], d["max"]))
    n_comp = int(min(m["n_comp"]["max"], 1 + int(dv90 // m["n_comp"]["kms_per_component"])))
    bp = m["b_comp_kms"]; b = [draw_lognormal(rng, bp["median"], bp["sigma_ln"], bp["min"], bp["max"]) for _ in range(n_comp)]
    off = draw_normal(rng, 0.0, m["dv_offset_kms"]["sigma"], -m["dv_offset_kms"]["max_abs"], m["dv_offset_kms"]["max_abs"])
    vels = [off] if n_comp == 1 else list(off + np.linspace(-dv90 / 2.0, dv90 / 2.0, n_comp) + rng.uniform(-0.15, 0.15, n_comp) * dv90 / max(n_comp - 1, 1))
    frac = rng.dirichlet(np.ones(n_comp) * m["n_comp"].get("dirichlet_alpha", 2.0))
    comps = [(float(frac[k]), float(b[k]), float(vels[k])) for k in range(n_comp)]
    logN_SiII = solve_logN_for_W1526(W1526, comps)
    q = m["logN_SiIII_minus_SiII"]; logN_SiIII = float(np.clip(logN_SiII + draw_normal(rng, q["mu"], q["sigma"], -2.0, 2.0), 11.0, 16.0))
    has_SiII = rng.uniform() < m["occurrence"]["SiII"]; has_SiIII = rng.uniform() < m["occurrence"]["SiIII"]
    lines = []
    for k in range(n_comp):
        if has_SiII:
            for ln in ("SiII1190", "SiII1193", "SiII1260"):
                lines.append(dict(line=ln, logN=round(float(np.log10(10 ** logN_SiII * frac[k])), 4), b_kms=round(b[k], 2), dv_kms=round(float(vels[k]), 2)))
        if has_SiIII:
            lines.append(dict(line="SiIII1206", logN=round(float(np.log10(10 ** logN_SiIII * frac[k])), 4), b_kms=round(b[k], 2), dv_kms=round(float(vels[k]), 2)))
    summ = dict(W1526_target=round(W1526, 4), W1526_achieved=round(ew_rest_1526(logN_SiII, comps), 4), logN_SiII=round(logN_SiII, 3), logN_SiIII=round(logN_SiIII, 3), dv90=round(dv90, 1), n_comp=n_comp, dv_offset=round(off, 1),
                has_SiII=bool(has_SiII), has_SiIII=bool(has_SiIII), seed=seed)
    return lines, summ


def stage_plans(a, model):
    rows = list(csv.DictReader(open(PLAN_FID)))
    wave = np.arange(3600.0, 9824.0 + 1e-6, 0.8)
    out_rows = []; ews = {ln: [] for ln in LINES}
    for r in rows:
        key = f"{r['TARGETID']}:{r['wave']}:{r['inj_idx']}"; z = float(r["z_inj"]); N = float(r["logN"])
        lines, summ = realize(model, a.arm, N, key)
        T = metal_transmission(wave, z, lines, lsf_fwhm_A=None)
        assert all(LINES[x["line"]]["verified"] for x in lines)
        # rest-frame EW per line (isolated evaluation; 1190/1193 blend is accounted in the joint transmission used for injection)
        ew = {}
        for ln in LINES:
            sub = [x for x in lines if x["line"] == ln]
            ew[ln] = round(equivalent_width_A(wave, metal_transmission(wave, z, sub, lsf_fwhm_A=None)) / (1 + z), 4) if sub else 0.0
            ews[ln].append(ew[ln])
        rr = dict(r); rr["metals_json"] = json.dumps(lines); rr["assoc_summary"] = json.dumps(summ); rr["metal_lsf_fwhm_A"] = round(1233.0 * (1 + z) / model["lsf_R"], 3)
        for ln in LINES:
            rr[f"ew_rest_{ln}"] = ew[ln]
        rr["ew_rest_total_window"] = round(equivalent_width_A(wave, T) / (1 + z), 4)
        out_rows.append(rr)
    os.makedirs(a.out_root, exist_ok=True); plan_out = os.path.join(a.out_root, f"plan_assoc{a.arm}.csv")
    with open(plan_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
    stats = {ln: dict(median=float(np.median(v)), p16=float(np.percentile(v, 16)), p84=float(np.percentile(v, 84)), frac_nonzero=float(np.mean(np.array(v) > 0))) for ln, v in ews.items()}
    prov = dict(arm=a.arm, model=a.model, model_sha256=model["model_sha"], plan_fid=PLAN_FID, plan_fid_sha256=_sha(PLAN_FID), plan_out=plan_out, plan_out_sha256=_sha(plan_out),
                n_injections=len(out_rows), ew_rest_stats=stats, code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip())
    json.dump(prov, open(os.path.join(a.out_root, f"plan_assoc{a.arm}.provenance.json"), "w"), indent=1)
    print(json.dumps(dict(arm=a.arm, n=len(out_rows), ew_rest_stats=stats), indent=1))


def build_cmd(a, wave, plan):
    summ = json.load(open(f"{FID}/r041_fid_wave{wave}.h5.build_summary.json"))
    return ["python", "tools/r041_build_archive.py", "--plan", plan, "--wave", str(wave), "--archive", summ["source_archive"], "--qsocat", summ["qsocat"],
            "--out-dir", a.out_root, "--tag", f"assoc{a.arm}", "--method", summ["method"], "--num-lines", str(summ["num_lines"]), "--sigma-px", str(summ["sigma_px"]),
            "--median-px", str(summ["median_px"]), "--noise-seed-policy", "shared0", "--metal-lsf-fwhm-A", str(a.lsf_fwhm_A)]


def stage_build(a, model):
    """Submit one small sbatch job per wave that runs the archive builder with the fiducial build's own arguments + the metal option."""
    plan = os.path.join(a.out_root, f"plan_assoc{a.arm}.csv"); os.makedirs(os.path.join(a.out_root, "logs"), exist_ok=True)
    strip = [f"-u{k}" for k in os.environ if k.startswith("SLURM_")]
    for w in WAVES:
        cmd = " ".join(build_cmd(a, w, plan))
        sb = os.path.join(a.out_root, f"build_wave{w}.sbatch")
        open(sb, "w").write(f"""#!/bin/bash
#SBATCH --job-name=assoc{a.arm}_build{w}
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh && conda activate gpdla
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1
cd {REPO}
{cmd}
echo BUILD_DONE arm={a.arm} wave={w} $(date -Is)
""")
        out = subprocess.check_output(["env"] + strip + ["SBATCH_CONSTRAINT=", "sbatch", "--parsable", f"--output={a.out_root}/logs/build_wave{w}_%j.log", sb]).decode().strip()
        print(f"arm {a.arm} wave {w} build job {out}: {cmd}")


def stage_envs(a, model):
    for w in WAVES:
        src = f"{ROOT_MAX4}/fid_max4/r041_fid_wave{w}_MAX4.env"; txt = open(src).read()
        txt = txt.replace(f"{FID}/r041_fid_wave{w}.h5", f"{a.out_root}/r041_assoc{a.arm}_wave{w}.h5")
        txt = txt.replace(f"{FID}/r041_fid_wave{w}_hpx_list.txt", f"{a.out_root}/r041_assoc{a.arm}_wave{w}_hpx_list.txt")
        txt = txt.replace(f"{FID}/r041_fid_wave{w}_qsocat.fits", f"{a.out_root}/r041_assoc{a.arm}_wave{w}_qsocat.fits")
        txt = txt.replace(f"{ROOT_MAX4}/fid_max4/r041_fid_wave{w}_MAX4_outputs", f"{a.out_root}/r041_assoc{a.arm}_wave{w}_MAX4_outputs")
        dst = f"{a.out_root}/r041_assoc{a.arm}_wave{w}_MAX4.env"; open(dst, "w").write(txt)
        for needle in (f"r041_assoc{a.arm}_wave{w}.h5", f"r041_assoc{a.arm}_wave{w}_hpx_list.txt", f"r041_assoc{a.arm}_wave{w}_MAX4_outputs"):
            assert needle in txt, needle
        print("env", dst)


def stage_launch(a, model):
    strip = [f"-u{k}" for k in os.environ if k.startswith("SLURM_")]
    rec = open(os.path.join(a.out_root, "LAUNCH_RECORD.txt"), "a")
    for w in WAVES:
        env = f"{a.out_root}/r041_assoc{a.arm}_wave{w}_MAX4.env"
        base = ["env"] + strip + ["SBATCH_CONSTRAINT=", "SBATCH_PARTITION=standard", "SBATCH_MEM_PER_CPU=5g", "bash", "slurm/nersc/production/launch_nersc.sh", env,
                                  "--start", "0", "--end", str(a.end), "--window", str(a.end), "--time", a.time, "--no-sleep"]
        dry = subprocess.run(base + ["--dry-run"], cwd=REPO, capture_output=True, text=True).stdout
        print(f"[dry-run wave {w}]", " | ".join(l for l in dry.splitlines() if "MAX_DLAS" in l or "sbatch" in l or "chdir" in l)[:400])
        if a.dry_run:
            continue
        out = subprocess.run(base, cwd=REPO, capture_output=True, text=True).stdout
        ids = [l.split()[-1] for l in out.splitlines() if "Submitted batch job" in l]
        line = f"assoc{a.arm} wave {w} jobs {ids} env {env} {subprocess.check_output(['date', '-Is']).decode().strip()}"
        print(line); rec.write(line + "\n")
    rec.close()


def stage_verify(a, model):
    """Pairing check: the arm's archive must equal the fiducial archive outside the metal windows (identical DLA transmission + identical eps)."""
    import h5py
    for w in WAVES:
        fa = f"{a.out_root}/r041_assoc{a.arm}_wave{w}.h5"; ff = f"{FID}/r041_fid_wave{w}.h5"
        if not os.path.exists(fa):
            print("missing", fa); continue
        A = h5py.File(fa, "r"); F = h5py.File(ff, "r"); wave = A["wavelength"][:]
        ta = {int(t): i for i, t in enumerate(A["catalog"][:]["TARGETID"])}; tf = {int(t): i for i, t in enumerate(F["catalog"][:]["TARGETID"])}
        plan = {(int(r["TARGETID"])): r for r in csv.DictReader(open(os.path.join(a.out_root, f"plan_assoc{a.arm}.csv"))) if int(r["wave"]) == w}
        n_same = n_diff_in = n_bad = 0; max_out = 0.0
        for t, i in list(ta.items())[:400]:
            j = tf[t]; fa_ = A["flux"][i][:]; ff_ = F["flux"][j][:]; r = plan[t]; z = float(r["z_inj"])
            win = np.zeros(wave.size, bool)
            for ln, p in LINES.items():
                lc = p["lambda0"] * (1 + z); win |= np.abs(wave - lc) < 30.0
            d = np.abs(fa_ - ff_); ok = np.isfinite(d)
            out_max = float(np.nanmax(d[~win & ok])) if (~win & ok).any() else 0.0; max_out = max(max_out, out_max)
            n_same += int(out_max == 0.0); n_diff_in += int(np.nanmax(d[win & ok]) > 0); n_bad += int(out_max > 0)
        print(f"wave {w}: sightlines checked {min(400, len(ta))}; identical outside metal windows {n_same}; differ inside {n_diff_in}; DIFFER OUTSIDE (must be 0) {n_bad}; max |dF| outside {max_out:.3e}")
        ta_ = {(int(r["TARGETID"]), int(r["inj_idx"])): (float(r["logN"]), float(r["z_inj"])) for r in csv.DictReader(open(f"{a.out_root}/r041_assoc{a.arm}_wave{w}.h5.truth.csv"))}
        tf_ = {(int(r["TARGETID"]), int(r["inj_idx"])): (float(r["logN"]), float(r["z_inj"])) for r in csv.DictReader(open(f"{FID}/r041_fid_wave{w}.h5.truth.csv"))}
        print(f"   truth identical to fid: keys {set(ta_) == set(tf_)}, values {all(abs(ta_[k][0] - tf_[k][0]) < 1e-9 and abs(ta_[k][1] - tf_[k][1]) < 1e-9 for k in ta_)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["plans", "build", "envs", "launch", "verify"])
    ap.add_argument("--arm", choices=["B", "C"], required=True); ap.add_argument("--model", required=True, help="frozen model JSON")
    ap.add_argument("--out-root", default=None); ap.add_argument("--lsf-fwhm-A", type=float, default=None, dest="lsf_fwhm_A")
    ap.add_argument("--end", type=int, default=1519); ap.add_argument("--time", default="04:00:00"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    model = json.load(open(a.model)); model["model_sha"] = _sha(a.model)
    for ln in model["lines"]:
        assert LINES[ln]["verified"], f"{ln} atomic data not verified"
    if a.lsf_fwhm_A is None:
        a.lsf_fwhm_A = -1.0   # per-row value from the plan column metal_lsf_fwhm_A (lambda_obs / R); the builder reads it when this is negative
    a.out_root = a.out_root or f"{ROOT_MAX4}/assoc/arm{a.arm}"
    {"plans": stage_plans, "build": stage_build, "envs": stage_envs, "launch": stage_launch, "verify": stage_verify}[a.stage](a, model)


if __name__ == "__main__":
    main()
