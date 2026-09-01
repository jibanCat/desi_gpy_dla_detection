#!/usr/bin/env python
"""r041_build_archive.py — build one injected LoaArchive WAVE from a realized R-041 plan,
with the corrected noise-preserving injection (default; prescription A), the residual-preserving
alternative (prescription B of the injection-prescription gate, PI ruling 2026-08-28/29 item 6)
or the old multiplicative one (for the old-vs-corrected comparison), optionally with a mean-flux
rescaling of the forest signal (R-041B), and emit everything the archive-route finder launch
needs: the injected archive
(schema-identical to the source; only the wave's sightlines), the truth manifest, the
per-wave QSO catalogue (rows of the production QSO catalogue), the healpix list, the launch
env file, and a build summary with hashes.

The source archive is opened read-only and never modified. ivar / mask / fwhm_pix are copied
unchanged (the corrected injection preserves the noise realization, so the noise model stays
valid by construction).
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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# --method choices -> (injection/noise_preserving.py method, human-readable prescription). The
# builder's method names are the CAMPAIGN names (kept for the existing plans/truth CSVs);
# "noise_preserving" is prescription A = the injector's "variance_preserving".
METHODS = {
    "noise_preserving": ("variance_preserving",
                         "A (variance-preserving): F' = T*(F + (r-1)*S) + sqrt(1 - T^2)*eps/sqrt(ivar), eps ~ N(0,1) from "
                         "numpy default_rng(seed) with seed = 0 for EVERY sightline of the wave (the injector default; the "
                         "archive route passes no per-sightline seed); S = signal_estimate(F; median_px, sigma_px); "
                         "ivar / mask unchanged; T = frozen Voigt transmission (num_lines)"),
    "residual_preserving": ("residual_preserving",
                            "B (residual-preserving): F' = T*S_r + (F_r - S_r) with F_r = F + (r-1)*S and S_r = r*S, i.e. the "
                            "observed residual F - S is carried through unchanged; no synthetic noise, no seed; pixels with "
                            "T == 1 exactly or undefined S keep F_r bit-for-bit; ivar / mask unchanged"),
    "multiplicative": (None,
                       "OLD (rejected): F' = T*F — the observed noise is attenuated together with the signal (noiseless "
                       "saturated troughs); kept only for the old-vs-corrected comparison"),
}


def inject_sightline(method, wave, flux, ivar, mask, absorbers, *, z_qso, alt, fid, num_lines, median_px, sigma_px):
    """Per-sightline dispatch on the campaign method name (see METHODS). `alt` / `fid` are the
    tau_eff callables of the mean-flux rescaling (alt None = fiducial, r = 1)."""
    from injection.noise_preserving import inject_noise_preserving, inject_multiplicative, meanflux_ratio
    if method not in METHODS:
        raise ValueError(f"unknown --method {method!r}; choices {sorted(METHODS)}")
    if method == "multiplicative":
        return inject_multiplicative(wave, flux, absorbers, num_lines)
    r_mf = meanflux_ratio(wave, z_qso, alt, fid) if alt is not None else None
    return inject_noise_preserving(wave, flux, ivar, mask, absorbers, z_qso=z_qso, r=r_mf, num_lines=num_lines,
                                   median_px=median_px, sigma_px=sigma_px, method=METHODS[method][0])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--wave", type=int, required=True)
    ap.add_argument("--archive", required=True, help="source LoaArchive (read-only)")
    ap.add_argument("--qsocat", required=True, help="production QSO catalogue (rows copied for the wave's TARGETIDs)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True, help="campaign tag, e.g. fid, cmp_old, cmp_new, mf_fg2008")
    ap.add_argument("--method", choices=list(METHODS), default="noise_preserving",
                    help="noise_preserving = prescription A (default, variance-preserving); residual_preserving = "
                         "prescription B; multiplicative = the old F*T operation (comparison only)")
    ap.add_argument("--meanflux-model", default=None, help="TAUEFF_MODELS key for the forest rescaling (R-041B); None = fiducial (r = 1)")
    ap.add_argument("--meanflux-fiducial", default="finder_fiducial")
    ap.add_argument("--sigma-px", type=float, default=None)
    ap.add_argument("--median-px", type=int, default=None)
    ap.add_argument("--num-lines", type=int, default=3)
    ap.add_argument("--base-env", default=os.path.join(REPO, "slurm/greatlakes/production/loa_cddf_hz_gl_v1.env"))
    a = ap.parse_args(argv)
    import h5py, fitsio
    from injection.noise_preserving import taueff, DEFAULT_SIGMA_PX, DEFAULT_MEDIAN_PX
    sigma = DEFAULT_SIGMA_PX if a.sigma_px is None else a.sigma_px
    median = DEFAULT_MEDIAN_PX if a.median_px is None else a.median_px
    os.makedirs(a.out_dir, exist_ok=True)
    plan = [r for r in csv.DictReader(open(a.plan)) if int(r["wave"]) == a.wave]
    if not plan:
        raise SystemExit(f"no rows for wave {a.wave}")
    by_tid = {}
    for r in plan:
        by_tid.setdefault(int(r["TARGETID"]), []).append(r)
    tids = sorted(by_tid)
    if a.method != "multiplicative" and any(len(v) > 1 for v in by_tid.values()) and "pair_class" not in plan[0]:
        raise SystemExit("fiducial waves must carry ONE injection per sightline")
    stem = f"r041_{a.tag}_wave{a.wave}"
    out_h5 = os.path.join(a.out_dir, stem + ".h5")
    truth_rows = []
    with h5py.File(a.archive, "r") as src, h5py.File(out_h5, "w") as dst:
        cat = src["catalog"][:]
        idx = {int(t): i for i, t in enumerate(cat["TARGETID"])}
        wave = src["wavelength"][:].astype(np.float64)
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        dst.attrs["r041_tag"] = a.tag; dst.attrs["r041_method"] = a.method; dst.attrs["r041_wave"] = a.wave
        dst.create_dataset("wavelength", data=src["wavelength"][:])
        rows = np.array([idx[t] for t in tids])
        dst.create_dataset("catalog", data=cat[rows])
        for name in ("ivar", "mask", "fwhm_pix"):
            if name in src:
                dst.create_dataset(name, data=src[name][:][rows], compression="gzip", compression_opts=4)
        fl_all = src["flux"][:][rows].astype(np.float64)
        iv_all = src["ivar"][:][rows]; mk_all = src["mask"][:][rows]
        out_flux = np.empty_like(fl_all, dtype=np.float32)
        fid = taueff(a.meanflux_fiducial); alt = taueff(a.meanflux_model) if a.meanflux_model else None
        for j, t in enumerate(tids):
            zq = float(cat[idx[t]]["Z"])
            absorbers = [{"nhi": 10.0 ** float(r["logN"]), "z_dla": float(r["z_inj"]), "num_lines": a.num_lines} for r in by_tid[t]]
            fl = inject_sightline(a.method, wave, fl_all[j], iv_all[j], mk_all[j], absorbers, z_qso=zq, alt=alt, fid=fid,
                                  num_lines=a.num_lines, median_px=median, sigma_px=sigma)
            out_flux[j] = fl.astype(np.float32)
            for r in by_tid[t]:
                truth_rows.append(dict(TARGETID=t, wave=a.wave, inj_idx=int(r["inj_idx"]), logN=float(r["logN"]), z_inj=float(r["z_inj"]),
                                       stratum=int(r["stratum"]), snr=float(r["snr"]), z_qso=zq, has_cand_ge20=int(r.get("has_cand_ge20", 0)),
                                       pair_class=r.get("pair_class", ""), dv_kms=r.get("dv_kms", ""), pair_logN=r.get("pair_logN", ""),
                                       method=a.method, meanflux_model=a.meanflux_model or "fiducial"))
        dst.create_dataset("flux", data=out_flux, compression="gzip", compression_opts=4)
    truth = out_h5 + ".truth.csv"
    with open(truth, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(truth_rows[0])); w.writeheader(); w.writerows(truth_rows)
    # per-wave QSO catalogue + healpix list (archive route: healpix mode, external hpx list)
    q = fitsio.read(a.qsocat, ext=1)
    sel = np.isin(q["TARGETID"].astype(np.int64), np.array(tids, np.int64))
    qsub = q[sel]
    assert len(qsub) == len(tids), f"QSO catalogue rows {len(qsub)} != wave sightlines {len(tids)}"
    qso_out = os.path.join(a.out_dir, stem + "_qsocat.fits")
    fitsio.write(qso_out, qsub, clobber=True)
    hpx = sorted(set(int(p) for p in qsub["HPXPIXEL"]))
    hpx_out = os.path.join(a.out_dir, stem + "_hpx_list.txt")
    with open(hpx_out, "w") as fh:
        fh.write("\n".join(str(p) for p in hpx) + "\n")
    outdir = os.path.join(a.out_dir, stem + "_outputs")
    env_out = os.path.join(a.out_dir, stem + ".env")
    with open(env_out, "w") as fh:
        fh.write(f"# R-041 {a.tag} wave {a.wave} — archive-route finder run on the injected archive (production high-z config, untouched)\n")
        fh.write(f'source "{a.base_env}"\n')
        fh.write(f'QSOCAT="{qso_out}"\nOUTDIR="{outdir}"\nOUTER_MAX_INDEX={len(hpx) - 1}\n')
        fh.write(f'export GPDLA_SPECTRA_ARCHIVE="{out_h5}"\nexport EXTERNAL_HPX_LIST="{hpx_out}"\n')
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        commit = "unknown"
    summ = dict(tag=a.tag, wave=a.wave, method=a.method, injection_prescription=METHODS[a.method][1],
                injector_method=METHODS[a.method][0],
                noise_seed_policy=("constant seed 0 for every sightline (inject_noise_preserving default; deterministic, NOT per-sightline)"
                                   if a.method == "noise_preserving" else "no synthetic noise"),
                meanflux_model=a.meanflux_model, sigma_px=sigma, median_px=median, num_lines=a.num_lines,
                n_sightlines=len(tids), n_injections=len(truth_rows), n_hpx=len(hpx), source_archive=a.archive, source_archive_sha256=_sha(a.archive),
                injected_archive=out_h5, injected_archive_sha256=_sha(out_h5), truth=truth, truth_sha256=_sha(truth), qsocat=qso_out, qsocat_sha256=_sha(qso_out),
                hpx_list=hpx_out, hpx_list_sha256=_sha(hpx_out), env=env_out, plan=a.plan, plan_sha256=_sha(a.plan), code_commit=commit,
                launch=f"EXTERNAL_HPX_LIST={hpx_out} GPDLA_SPECTRA_ARCHIVE={out_h5} SBATCH_PARTITION=standard bash slurm/nersc/production/launch_nersc.sh {env_out} --start 0 --end {len(hpx)} --window {len(hpx)} --time <hh:mm:ss>")
    with open(out_h5 + ".build_summary.json", "w") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if "sha256" not in k}, indent=1))


if __name__ == "__main__":
    main()
