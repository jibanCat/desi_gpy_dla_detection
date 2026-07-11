# -*- coding: utf-8 -*-
"""calccddf_vs_hbi.py  — LITERAL calc_cddf (Bird-2017 recipe) vs catalog-HBI, MOCK-ONLY.

STATUS 2026-07-11 (Queue 1): both former blockers RESOLVED with evidence; do
not quote numbers until the 2LPT-0 closure artifact is stamped.
  (1) NaN semantics — SETTLED as NaN-as-zero + keep rows in dX (the writer's
      own convention: ``bayesian_model_selection.py:246-253`` normalizes over
      non-NaN entries so stored rows nansum to exactly 1; ``:279`` nansums
      p_dlas). NaN = "model k never evaluated: a lower-order model already won
      the sequential early-stop" (``dla_gp.py:1020-1071``, 3 stop conditions +
      a FILTER edge case). calc_cddf's row-drop (``calc_cddf.py:522-524``) is
      REJECTED for the comparison: at SNR>2 it deletes 50.2%% of p>0.9
      detections (essentially all 1-absorber systems) — behavior pinned as-is
      in ``tests/test_cddf_characterization_q1.py``. This class's
      nan_to_num+renormalize is numerically identical to the writer convention
      on real rows (guards below enforce that per file, fail-closed).
  (2) sub_dla offset — SETTLED: these prod files are SINGLE_ABSORBER_MODEL=1
      (layout [null,1abs,2abs,3abs,4abs]); constructor now sub_dla=False.
      The old sub_dla=True misassignment is pinned in the same test module.
Remaining caveat before quoting: FILTER=1 truncates the per-spectrum sample
softmax to region-A samples (~4-7%% of the grid) — the N-resolved shape needs
the mock truth-closure run (this script's purpose) to validate.

Runs the *literal* fixed-Lambda
posterior CDDF (``CDDF_analysis.calc_cddf.DLACatalogue``) on the per-spectrum
processed HDF5 files of a mock, aggregating N-resolved expected DLA counts and
absorption path dX across all healpix files (the full combined file would be TBs
and calc_cddf bulk-loads the sample axis, so we run per-file and SUM — the
posterior-weighted expected count `probs+poissons` and dX are both additive).

NO GP re-inference.  NO alpha.  NO hard P_DLA cut (posterior-weighted).  The DLA
sample grid is the SAME pw_samples_a3_172_225 grid the inference used (support
[17.2,22.5]), so the DLA(1..k) posterior reaches the sub-DLA band natively.

Truth is windowed IDENTICALLY to the estimator (Lyβ blue edge = lymanbeta(z_qso),
clamped to the stored [min_z_dla,max_z_dla] search window, z in [ZMIN,ZMAX],
sightline SNR_REDSIDE>2), so R0=est/truth is on one footing.

MOCK values only — public-OK.  Never reads real-LOA (loa main-dark).
"""
import os
import sys
import glob
import json
import time
import argparse
import tempfile

import numpy as np
import h5py
from astropy.table import Table
from scipy.special import logsumexp

# repo root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from CDDF_analysis.calc_cddf import DLACatalogue, rho_crit  # noqa: E402
from CDDF_analysis.cddf_forward.window import WindowSpec  # noqa: E402
from CDDF_analysis.calc_cddf import lyb_wavelength, lya_wavelength  # noqa: E402

GRID_DIR = "/scratch/cavestru_root/cavestru0/mfho/DESI/desi_gpy_dla_detection/data/dr12q/processed"

MOCKS = {
    "2lpt0": dict(
        proc="/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/outputs/figures/processed",
        grid=os.path.join(GRID_DIR, "pw_samples_a3_172_225_100000.mat"),
        truth="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits",
        zcol="Z",
    ),
    "saclay0": dict(
        proc="/scratch/cavestru_root/cavestru0/mfho/gl_prod_saclay0_v1_20260630/outputs/figures/processed",
        grid=os.path.join(GRID_DIR, "pw_samples_a3_172_225_50000.mat"),
        truth="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124/hcd_truth_cat.fits",
        zcol="Z",
    ),
    "london0": dict(
        proc="/scratch/cavestru_root/cavestru0/mfho/gl_prod_london0_v1_preclustering_20260522/outputs/figures/processed",
        grid=os.path.join(GRID_DIR, "pw_samples_a3_172_225_100000.mat"),
        truth="/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/tf_london0/mockdir/dla_cat.fits",
        zcol="Z_DLA",
    ),
}

ZMIN, ZMAX = 2.0, 3.5          # matches HBI forward-path self baseline zbins [2.0..3.5]
SNR_MIN = 2.0                  # HBI SNR_REDSIDE > 2.0 (strict)
N_EDGES = np.round(np.arange(17.2, 22.40001, 0.1), 3)   # centers 17.25..22.35 (== HBI perz_fN)
N_CENT = 0.5 * (N_EDGES[:-1] + N_EDGES[1:])
HIGH_NHI_CUT = 22.5

# Omega conversion constant (calc_cddf.omega_dla_cddf), hubble=0.7
_PROTON = 1.67262178e-24
_H100 = 3.2407789e-18 * 0.7
_LIGHT = 2.99e10
OMEGA_CONV = _PROTON / _LIGHT * _H100 / rho_crit(0.7)

LIMITS = [19.5, 20.0, 20.3]
Z_SPLITS = [(2.0, 2.5), (2.5, 3.0), (3.0, 3.5)]   # C4 stress splits
SNR_HI = 4.0                                       # C4 high-SNR stratum


class NanSafeDLACatalogue(DLACatalogue):
    """LITERAL calc_cddf with faithful NaN handling + a memory-light DLA(1) load.

    The DESI processed files store NaN in a DLA posterior column / sample
    likelihood to mean 'negligible or invalid' (~0), NOT missing data (`p_dlas`
    is stored clean).  calc_cddf's DLA(1) path lacks the NaN mask its multi-DLA
    path (`_do_norm_log_norm_like_k`) already applies, and its `condition`
    NaN-filter would drop ~83% of sightlines (incl. most detections) from BOTH
    the counts AND the dX denominator.  We override the two setup methods so:
      * NaN model-posteriors -> 0, re-normalized; ALL real sightlines kept for dX;
      * the DLA(1) softmax cache is built from a PARTIAL load of only the
        detection rows (p_dla>p_thresh_spec), with NaN samples -> -inf.
    All estimator math (`_split_distributions`, `path_length`) is calc_cddf's own.
    """

    def renormalise_occams_razor(self, occams_razor=1):
        mp_raw = self.filehandle["model_posteriors"][()]
        # fail-closed writer-convention guards (Queue-1 NaN trace, 2026-07-11):
        # (G1) non-NaN columns of every row sum to exactly 1 (bayesian_model_
        #      selection.py:246-253 normalizes over non-NaN entries);
        # (G2) NaN pattern is monotone in model order (sequential early-stop);
        # (G3) stored p_dlas == nansum over absorber columns.
        row_sum = np.nansum(mp_raw, axis=1)
        if not np.allclose(row_sum, 1.0, rtol=0, atol=1e-6):
            bad = int(np.sum(np.abs(row_sum - 1.0) > 1e-6))
            raise ValueError(
                f"writer convention violated: {bad} rows with nansum(mp)!=1 in "
                f"{self.filehandle.filename} — NaN-as-zero is not safe here.")
        nan_pat = np.isnan(mp_raw[:, 1:])
        if np.any(nan_pat[:, :-1] & ~nan_pat[:, 1:]):
            raise ValueError(
                f"non-monotone NaN pattern in {self.filehandle.filename} — "
                "not the sequential early-stop; investigate before proceeding.")
        if "p_dlas" in self.filehandle:
            stored = self.filehandle["p_dlas"][()]
            if not np.allclose(np.nansum(mp_raw[:, 1:], axis=1), stored, atol=1e-6):
                raise ValueError(
                    f"stored p_dlas disagrees with nansum(mp[:,1:]) in "
                    f"{self.filehandle.filename} — column layout mismatch?")
        mp = np.nan_to_num(mp_raw, nan=0.0)
        mp[:, 1:] = mp[:, 1:] / occams_razor
        norm = mp.sum(axis=1, keepdims=True)
        self.model_posteriors = mp / norm
        self.condition = self.condition * (self.real_index != -1) * (norm.ravel() > 0)
        self.p_dla = self.model_posteriors[:, 1 + self.sub_dla:].sum(axis=1)
        self.p_no_dla = self.model_posteriors[:, : 1 + self.sub_dla].sum(axis=1)

    def get_first_dla_attrs(self):
        self.log_norm_like_cache = {}
        dla_ind = self.filter_dla_spectra(second=False)[0]
        if len(dla_ind) == 0:
            return
        S = self.filehandle["sample_log_likelihoods_dla"]
        # ONE contiguous full-slice read (fast); h5py fancy row-indexing on a
        # chunked dataset is ~10x slower.  Then NaN-mask + softmax per detection.
        arr = (S[:, :, 0] if S.ndim > 2 else S[:]).astype(float)   # (N, Ssamples)
        for spec in np.unique(dla_ind):
            ll = arr[int(spec)].copy()
            ll[~np.isfinite(ll)] = -np.inf
            self.log_norm_like_cache[int(spec)] = ll - logsumexp(ll)
        del arr


def _lyb(z):
    return (1.0 + z) * (lyb_wavelength / lya_wavelength) - 1.0


def estimate_one_file(proc_file, grid, catalog_file, second, splits=False):
    """Return {tag: (mean_counts_N[nbin], dX)} from LITERAL calc_cddf for one file.

    tag 'full' always present; with splits=True also per z-bin (Z_SPLITS) from the
    SAME catalogue build, and an SNR>SNR_HI full-range stratum from a second build
    (strata by construction; [2,4] follows by differencing full - snr_gt4).
    """
    cat = NanSafeDLACatalogue(
        processed_file=proc_file,
        sample_file=grid,
        catalog_file=catalog_file,
        # single-absorber prod layout [null,1abs,2abs,3abs,4abs] — there is NO
        # sub-DLA column; sub_dla=True would misassign 1abs into p_no_dla
        # (pinned in tests/test_cddf_characterization_q1.py).
        sub_dla=False,
        second=second,
        snr=SNR_MIN,
        high_nhi_cut=True,
        high_nhi_cut_value=HIGH_NHI_CUT,
        window=WindowSpec(z_min_lyb=True),   # Lyα-only: blue edge = lymanbeta(z_qso); no prox/tail re-cut
    )
    mean_counts, dX = _counts_dx(cat, ZMIN, ZMAX)
    out = {"full": (mean_counts, dX)}
    if splits:
        for lo, hi in Z_SPLITS:
            out[f"z_{lo}_{hi}"] = _counts_dx(cat, lo, hi)
    cat.filehandle.close()
    if splits:
        cat4 = NanSafeDLACatalogue(
            processed_file=proc_file, sample_file=grid, catalog_file=catalog_file,
            sub_dla=False, second=second, snr=SNR_HI,
            high_nhi_cut=True, high_nhi_cut_value=HIGH_NHI_CUT,
            window=WindowSpec(z_min_lyb=True),
        )
        out[f"snr_gt{SNR_HI:g}"] = _counts_dx(cat4, ZMIN, ZMAX)
        cat4.filehandle.close()
    return out


def _counts_dx(cat, zlo, zhi):
    """(mean N-counts, dX) over one z window from an already-built catalogue."""
    probs, poissons = cat._split_distributions(
        N_EDGES, lred=zlo, ured=zhi, lnhi_min=17.19, lnhi_max=HIGH_NHI_CUT, nhi=True
    )
    mean_counts = np.array(poissons, dtype=float)
    for b, plist in enumerate(probs):
        if plist:
            mean_counts[b] += float(np.sum(np.concatenate([np.atleast_1d(p) for p in plist])))
    return mean_counts, float(cat.path_length(zlo, zhi))


def truth_one_file(proc_file, truth_by_tid, splits=False):
    """Convention-matched truth N-histogram for one processed file's sightlines."""
    with h5py.File(proc_file, "r") as f:
        tids = np.asarray(f["target_ids"][:]).astype(np.int64).ravel()
        zq = np.asarray(f["z_qsos"][:]).astype(float).ravel()
        zlo = np.asarray(f["min_z_dlas"][:]).astype(float).ravel()
        zhi = np.asarray(f["max_z_dlas"][:]).astype(float).ravel()
        snr = np.asarray(f["snrs"][:]).astype(float).ravel()
    tags = ["full"] + ([f"z_{lo}_{hi}" for lo, hi in Z_SPLITS] + [f"snr_gt{SNR_HI:g}"]
                       if splits else [])
    counts = {t: np.zeros(len(N_CENT), dtype=float) for t in tags}
    for tid, zqso, mn, mx, s in zip(tids, zq, zlo, zhi, snr):
        if not (s > SNR_MIN):
            continue
        lower = max(mn, _lyb(zqso), ZMIN)
        lower = min(lower, mx)
        upper = min(mx, ZMAX)
        if not (upper > lower):
            continue
        rows = truth_by_tid.get(int(tid))
        if rows is None:
            continue
        za, na = rows
        nok = (na >= N_EDGES[0]) & (na < N_EDGES[-1])
        m = (za > lower) & (za < upper) & nok
        if m.any():
            h = np.histogram(na[m], bins=N_EDGES)[0]
            counts["full"] += h
            if splits and s > SNR_HI:
                counts[f"snr_gt{SNR_HI:g}"] += h
        if splits:
            for lo_z, hi_z in Z_SPLITS:
                mz = (za > max(lower, lo_z)) & (za < min(upper, hi_z)) & nok
                if mz.any():
                    counts[f"z_{lo_z}_{hi_z}"] += np.histogram(na[mz], bins=N_EDGES)[0]
    return counts


def load_truth_by_tid(path, zcol):
    t = Table.read(path)
    tid = np.asarray(t["TARGETID"]).astype(np.int64)
    z = np.asarray(t[zcol]).astype(float)
    n = np.asarray(t["NHI"]).astype(float)
    order = np.argsort(tid, kind="stable")
    tid, z, n = tid[order], z[order], n[order]
    uniq, start = np.unique(tid, return_index=True)
    end = np.append(start[1:], len(tid))
    out = {}
    for u, a, b in zip(uniq, start, end):
        out[int(u)] = (z[a:b], n[a:b])
    return out


def cumulative_dndx_omega(counts_N, dX):
    dndx, omega = {}, {}
    for lim in LIMITS:
        sel = N_CENT >= (lim - 1e-9)
        c = counts_N[sel]
        dndx[str(lim)] = float(c.sum() / dX) if dX > 0 else float("nan")
        omega[str(lim)] = float(OMEGA_CONV * np.sum(10.0 ** N_CENT[sel] * c) / dX) if dX > 0 else float("nan")
    # sub-DLA band [19.5,20.3)
    selb = (N_CENT >= 19.5 - 1e-9) & (N_CENT < 20.3 - 1e-9)
    cb = counts_N[selb]
    dndx["band_195_203"] = float(cb.sum() / dX) if dX > 0 else float("nan")
    omega["band_195_203"] = float(OMEGA_CONV * np.sum(10.0 ** N_CENT[selb] * cb) / dX) if dX > 0 else float("nan")
    return dndx, omega


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", required=True, choices=list(MOCKS))
    ap.add_argument("--nfiles", type=int, default=-1, help="subset of processed files (-1=all)")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth file (representative subset)")
    ap.add_argument("--second", type=int, default=0,
                    help="RETIRED: only 0 (DLA1) is accepted — see the guard below")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", action="store_true",
                    help="C4 stress splits: also accumulate per-z-bin (2.0/2.5/3.0/3.5) "
                         "and SNR>4 stratum histograms (roughly 2x runtime)")
    args = ap.parse_args()

    if args.second != 0:
        raise SystemExit(
            "--second != 0 is RETIRED (PI decision C3, 2026-07-11): calc_cddf's "
            "multi-DLA increment path has been broken since 2020 (b00e6e4 — spectra "
            "axis indexed by sample indices; -1e30 accumulator), so slot-0 IS the "
            "literal estimator this code has always computed. Fixing the path is "
            "separate, referee-reviewed debt. See notes 2026-07-11_q1_gate.md (C3).")

    cfg = MOCKS[args.mock]
    assert "main_dark" not in cfg["proc"] and "main_dark" not in cfg["truth"], "REAL-LOA guard"
    files = sorted(glob.glob(os.path.join(cfg["proc"], "processed-*.h5")))
    files = files[:: args.stride]
    if args.nfiles > 0:
        files = files[: args.nfiles]
    print(f"[{args.mock}] {len(files)} processed files; second={args.second}; z[{ZMIN},{ZMAX}]; grid={os.path.basename(cfg['grid'])}", flush=True)

    truth_by_tid = load_truth_by_tid(cfg["truth"], cfg["zcol"])
    print(f"[{args.mock}] truth: {len(truth_by_tid)} sightlines w/ injected absorbers", flush=True)

    second = False if args.second == 0 else args.second
    tmpdir = tempfile.mkdtemp(prefix="calccddf_")
    cat_path = os.path.join(tmpdir, "cat.fits")

    tags = ["full"] + ([f"z_{lo}_{hi}" for lo, hi in Z_SPLITS] + [f"snr_gt{SNR_HI:g}"]
                       if args.splits else [])
    est_T = {t: np.zeros(len(N_CENT)) for t in tags}
    tru_T = {t: np.zeros(len(N_CENT)) for t in tags}
    dX_T = {t: 0.0 for t in tags}
    n_sl = 0
    n_skip = 0
    t0 = time.time()
    for i, pf in enumerate(files):
        with h5py.File(pf, "r") as f:
            tids = np.asarray(f["target_ids"][:]).astype(np.int64).ravel()
        Table({"TARGETID": tids}).write(cat_path, overwrite=True)
        try:
            est_blocks = estimate_one_file(pf, cfg["grid"], cat_path, second,
                                           splits=args.splits)
        except Exception as e:
            n_skip += 1
            print(f"  [skip {os.path.basename(pf)}] {type(e).__name__}: {e}", flush=True)
            # fail-closed: a systematic per-file failure must kill the run, not
            # silently degrade it to a tiny subsample (2026-07-11 lesson: the
            # broken calc_cddf multi-slot path skipped 1148/1150 files).
            if n_skip > max(5, 0.01 * len(files)):
                raise RuntimeError(
                    f"{n_skip} files skipped by {i+1} — systematic failure, aborting "
                    f"(last: {type(e).__name__}: {e})")
            continue
        tru_blocks = truth_one_file(pf, truth_by_tid, splits=args.splits)
        for t in tags:
            mc_t, dx_t = est_blocks[t]
            est_T[t] += mc_t
            dX_T[t] += dx_t
            tru_T[t] += tru_blocks[t]
        n_sl += len(tids)
        if (i + 1) % 25 == 0 or i == len(files) - 1:
            print(f"  {i+1}/{len(files)} dX={dX_T['full']:.2f} est(>=20.3)={est_T['full'][N_CENT>=20.3-1e-9].sum():.1f} "
                  f"tru(>=20.3)={tru_T['full'][N_CENT>=20.3-1e-9].sum():.1f} ({time.time()-t0:.0f}s)", flush=True)
            _write_out(args, cfg, files, est_T, tru_T, dX_T, n_sl, i + 1, time.time() - t0, n_skip)

    out = _write_out(args, cfg, files, est_T, tru_T, dX_T, n_sl, len(files), time.time() - t0, n_skip)
    print(json.dumps(out["cumulative"], indent=1), flush=True)
    print(f"[{args.mock}] wrote {args.out}  ({time.time()-t0:.0f}s)", flush=True)


def _write_out(args, cfg, files, est_T, tru_T, dX_T, n_sl, n_done, wall, n_skip=0):
    est_N, tru_N, dX_tot = est_T["full"], tru_T["full"], dX_T["full"]
    est_dndx, est_om = cumulative_dndx_omega(est_N, dX_tot)
    tru_dndx, tru_om = cumulative_dndx_omega(tru_N, dX_tot)
    r0_dndx = {k: (est_dndx[k] / tru_dndx[k] if tru_dndx[k] else float("nan")) for k in est_dndx}
    r0_om = {k: (est_om[k] / tru_om[k] if tru_om[k] else float("nan")) for k in est_om}
    dN_lin = 10.0 ** N_EDGES[1:] - 10.0 ** N_EDGES[:-1]
    fN_est = (est_N / dX_tot / dN_lin).tolist() if dX_tot > 0 else []
    fN_tru = (tru_N / dX_tot / dN_lin).tolist() if dX_tot > 0 else []
    out = dict(
        mock=args.mock, n_files=n_done, n_files_total=len(files), n_sightlines=n_sl, n_files_skipped=n_skip,
        second=args.second, z_range=[ZMIN, ZMAX], snr_min=SNR_MIN, dX_total=dX_tot,
        grid=os.path.basename(cfg["grid"]), truth=cfg["truth"], checkpoint=(n_done < len(files)),
        N_centers=N_CENT.tolist(), fN_calccddf=fN_est, fN_truth=fN_tru,
        counts_calccddf_N=est_N.tolist(), counts_truth_N=tru_N.tolist(),
        cumulative=dict(
            calccddf=dict(dndx=est_dndx, omega=est_om),
            truth=dict(dndx=tru_dndx, omega=tru_om),
            R0_calccddf=dict(dndx=r0_dndx, omega=r0_om),
        ),
        wallclock_s=wall,
    )
    if len(est_T) > 1:
        out["splits"] = {}
        for t in est_T:
            if t == "full":
                continue
            e, tr, dx = est_T[t], tru_T[t], dX_T[t]
            ed, eo = cumulative_dndx_omega(e, dx)
            td, to = cumulative_dndx_omega(tr, dx)
            out["splits"][t] = dict(
                dX=dx, counts_est=e.tolist(), counts_truth=tr.tolist(),
                R0_dndx={k: (ed[k] / td[k] if td[k] else float("nan")) for k in ed},
                R0_omega={k: (eo[k] / to[k] if to[k] else float("nan")) for k in eo},
            )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    main()
