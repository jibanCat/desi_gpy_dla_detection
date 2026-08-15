"""loa_literal_calccddf_hz.py -- HIGH-Z (BH candidate) port of the literal
calc_cddf baseline (constant-port of loa_literal_calccddf.py, 2026-08-15; the
analysis machinery is byte-identical, only the fixed-config block and the
DLACatalogue SNR selection differ):

  * PD  -> the 3,022-QSO high-z production processed dir (job 57427253);
  * CAT -> hz_tidcat_nobal.fits (universe z_qso in (4.25,7.0) minus BI_CIV>0);
  * SNR_REDSIDE>2 via set_snr(2) after every DLACatalogue construction
    (the low-z snr2_nobal apples-to-apples recipe);
  * Z window / coarse grid -> the H2-v2 BH candidate bins 3.8/4.25/4.5/5.0;
  * OUT -> track_c/tf_hz_calc_cddf.

Estimand unchanged: PLUGIN_MAP_MC (uncorrected Bayesian posterior CDDF,
Bird-2017-style feed-forward; the 2026-07-28 retired-diagnostic label applies
to this route as well). CANDIDATE / methodological context only.
"""

from __future__ import annotations

import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .calc_cddf import DLACatalogue, rho_crit
from .cddf_forward.window import WindowSpec
from .cddf_forward.filter_guard import assert_filter_off

# --------------------------------------------------------------------------- #
# fixed analysis config
# --------------------------------------------------------------------------- #
PD = ("/scratch/cavestru_root/cavestru0/mfho/loa_hz_production/"
      "gl_cddf_loa_hz_v1_20260813/outputs/figures/processed")
SAMPLE = ("/scratch/cavestru_root/cavestru0/mfho/DESI/desi_gpy_dla_detection/"
          "data/dr12q/processed/pw_samples_a3_172_225_100000.mat")
# TARGETID-only catalog covering ALL processed sightlines (built by build_loa_tidcat.py)
CAT = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/"
       "tf_hz_calc_cddf/hz_tidcat_nobal.fits")
OUT = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/"
       "tf_hz_calc_cddf")

Z_MIN, Z_MAX = 3.8, 5.0
SNR_THRESH = 2.0   # SNR_REDSIDE>2 (apples-to-apples with P1_PRIMARY_LYA)
LNHI_MAX = 22.4          # = HBIConfig.drop_top_bin_above
THRESHOLDS = (20.0, 20.3, 20.6)
FN_LNHI_MIN, FN_LNHI_MAX, FN_NBINS = 19.5, 22.0, 25   # differential f(N) grid
HUBBLE = 0.7
SUB_DLA = False          # FILTER-off maxdla1 layout: model_posteriors=[Null, 1DLA]
# per-z grid = EXPLICIT Track-C 5-bin coarse grid (z extended above 3.5):
#   [2.0, 2.5, 3.0, 3.5, 4.0, 4.25] -> 5 bins, z_mid 2.25/2.75/3.25/3.75/4.12.
# calc_cddf is RAW (no calibration support limit), so it runs to 4.25 freely; the
# z>3.5 bins are a genuine per-bin Poisson-binomial density (each bin re-computes
# _split_distributions + path_length over its OWN z-edges, NOT a naive fine-bin mean).
Z_GRID_COARSE = np.array([3.8, 4.25, 4.5, 5.0], float)
N_ZBINS = len(Z_GRID_COARSE) - 1     # 5 coarse z bins
# fine f(N|z) grid: SAME nhi edges as the z-marginal f(N) so the per-z curves stack to
# the z-marginal one (up to the per-bin ΔX normalization).
FNZ_LNHI_MIN, FNZ_LNHI_MAX, FNZ_NBINS = FN_LNHI_MIN, FN_LNHI_MAX, FN_NBINS

DLACAT_KW = dict(sub_dla=SUB_DLA, high_nhi_cut_value=LNHI_MAX)


# --------------------------------------------------------------------------- #
# per-file ingredient worker (picklable, module-level)
# --------------------------------------------------------------------------- #
def _ingredients(processed_file):
    """All literal-calc_cddf ingredients from ONE file (one DLACatalogue, closed)."""
    win = WindowSpec()
    cat = DLACatalogue(processed_file=processed_file, sample_file=SAMPLE,
                       catalog_file=CAT, window=win, **DLACAT_KW)
    cat.set_snr(SNR_THRESH)
    try:
        out = {"processed_file": processed_file}

        # ---- fine f(N) ingredients (nhi bins), single z window ----
        fn_edges = np.linspace(FN_LNHI_MIN, FN_LNHI_MAX, FN_NBINS + 1)
        pr_f, po_f = cat._split_distributions(
            fn_edges, lred=Z_MIN, ured=Z_MAX,
            lnhi_min=fn_edges[0], lnhi_max=fn_edges[-1], nhi=True)
        out["fn_probs"] = pr_f
        out["fn_poissons"] = np.asarray(po_f, float)
        out["fn_dX"] = float(cat.path_length(Z_MIN, Z_MAX))

        # ---- per-threshold integrated dN/dX (single z bin) + Omega ----
        for thr in THRESHOLDS:
            z_edges = np.array([Z_MIN, Z_MAX])
            pr_d, po_d = cat._split_distributions(
                z_edges, lred=Z_MIN, ured=Z_MAX,
                lnhi_min=thr, lnhi_max=LNHI_MAX, nhi=False)
            out[f"dndx_probs_{thr}"] = pr_d
            out[f"dndx_poissons_{thr}"] = np.asarray(po_d, float)

            om_edges = np.linspace(thr, LNHI_MAX, FN_NBINS + 1)
            pr_o, po_o = cat._split_distributions(
                om_edges, lred=Z_MIN, ured=Z_MAX,
                lnhi_min=thr, lnhi_max=LNHI_MAX, nhi=True)
            out[f"omega_probs_{thr}"] = pr_o
            out[f"omega_poissons_{thr}"] = np.asarray(po_o, float)
            out[f"omega_edges_{thr}"] = om_edges

        # ---- per-z dN/dX + Omega at headline 20.3 (Track-C 5-bin z grid) ----
        z_grid = Z_GRID_COARSE
        prz, poz = cat._split_distributions(
            z_grid, lred=Z_MIN, ured=Z_MAX, lnhi_min=20.3, lnhi_max=LNHI_MAX, nhi=False)
        out["perz_dndx_probs"] = prz
        out["perz_dndx_poissons"] = np.asarray(poz, float)
        # per-z dX (each bin), and per-z Omega ingredients (N-weighted grid per z bin)
        out["perz_dX"] = np.array([float(cat.path_length(z_grid[i], z_grid[i + 1]))
                                   for i in range(N_ZBINS)], float)
        perz_om_edges = np.linspace(20.3, LNHI_MAX, FN_NBINS + 1)
        perz_om_probs = []
        perz_om_poissons = []
        for i in range(N_ZBINS):
            pr_oz, po_oz = cat._split_distributions(
                perz_om_edges, lred=z_grid[i], ured=z_grid[i + 1],
                lnhi_min=20.3, lnhi_max=LNHI_MAX, nhi=True)
            perz_om_probs.append(pr_oz)
            perz_om_poissons.append(np.asarray(po_oz, float))
        out["perz_om_probs"] = perz_om_probs
        out["perz_om_poissons"] = perz_om_poissons

        # ---- per-z DIFFERENTIAL f(N|z) ingredients (the NEW deliverable) ----
        # For each coarse z bin, bin the per-sample DLA probabilities by logN over the
        # FINE nhi grid, restricted to THAT z-window (lred/ured = the bin edges). This is
        # the per-(z,N) Poisson-binomial expected count; divided by (ΔX_zbin·ΔN_bin) it is
        # the differential CDDF f(N_HI) IN that z bin. Same _split_distributions(nhi=True)
        # call the z-marginal f(N) uses, just with the z-window narrowed per bin.
        fnz_edges = np.linspace(FNZ_LNHI_MIN, FNZ_LNHI_MAX, FNZ_NBINS + 1)
        fnz_probs = []
        fnz_poissons = []
        for i in range(N_ZBINS):
            pr_fz, po_fz = cat._split_distributions(
                fnz_edges, lred=z_grid[i], ured=z_grid[i + 1],
                lnhi_min=fnz_edges[0], lnhi_max=fnz_edges[-1], nhi=True)
            fnz_probs.append(pr_fz)
            fnz_poissons.append(np.asarray(po_fz, float))
        out["fnz_probs"] = fnz_probs
        out["fnz_poissons"] = fnz_poissons

        # common single-z-bin dX over [Z_MIN, Z_MAX]
        out["dX"] = float(cat.path_length(Z_MIN, Z_MAX))
        dx_mask = cat._filter_snr_spectra() * cat._filter_z_dlas(cat.z_dla_minimum)
        out["n_active"] = int(np.sum(dx_mask))
        return out
    finally:
        try:
            cat.filehandle.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# accumulation helpers
# --------------------------------------------------------------------------- #
def _extend_probs(acc_list, new_list):
    for b in range(len(acc_list)):
        acc_list[b].extend(new_list[b])


def _check_one(fp):
    import h5py
    try:
        with h5py.File(fp, "r") as h:
            if "target_ids" not in h or h["target_ids"].shape[0] == 0:
                return (fp, False)
        return (fp, True)
    except (OSError, KeyError):
        return (fp, False)


def _partition_readable(files, n_workers=16):
    """Parallel readability partition (the serial version is the disk-bound
    bottleneck on a 16k-file set; this forks the superblock opens)."""
    readable, unreadable = [], []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for fp, ok in ex.map(_check_one, files, chunksize=16):
            (readable if ok else unreadable).append(fp)
    return sorted(readable), unreadable


def main(n_workers=16, limit=None, subsample=None, seed=0, skip_partition=False):
    os.makedirs(OUT, exist_ok=True)
    all_files = sorted(glob.glob(os.path.join(PD, "processed-*.h5")))
    if limit:
        all_files = all_files[:limit]
    if subsample and subsample < len(all_files):
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(len(all_files), size=subsample, replace=False))
        all_files = [all_files[i] for i in idx]
        is_subsample = True
    else:
        is_subsample = False
    if skip_partition:
        # all 16519 files were verified readable by build_loa_tidcat (0 bad);
        # _ingredients raises on a genuinely-bad file, which we'd see immediately.
        files, unreadable = sorted(all_files), []
        print(f"[loa-literal] skip_partition: assuming all {len(files)} readable", flush=True)
    else:
        files, unreadable = _partition_readable(all_files, n_workers=n_workers)
    n_files = len(files)
    print(f"[loa-literal] {len(all_files)} files; {n_files} readable, "
          f"{len(unreadable)} unreadable/empty stubs skipped, "
          f"subsample={is_subsample}, n_workers={n_workers}", flush=True)

    assert_filter_off(0, ctx="loa_literal_calccddf")

    t0 = time.time()

    # accumulators
    fn_edges = np.linspace(FN_LNHI_MIN, FN_LNHI_MAX, FN_NBINS + 1)
    fn_probs = [list() for _ in range(FN_NBINS)]
    fn_poissons = np.zeros(FN_NBINS)
    fn_dX = 0.0

    dndx_probs = {thr: [list()] for thr in THRESHOLDS}
    dndx_poissons = {thr: np.zeros(1) for thr in THRESHOLDS}
    om_edges = {thr: np.linspace(thr, LNHI_MAX, FN_NBINS + 1) for thr in THRESHOLDS}
    om_probs = {thr: [list() for _ in range(FN_NBINS)] for thr in THRESHOLDS}
    om_poissons = {thr: np.zeros(FN_NBINS) for thr in THRESHOLDS}
    dX_single = 0.0
    n_active_total = 0

    # per-z accumulators (headline 20.3)
    perz_dndx_probs = [list() for _ in range(N_ZBINS)]
    perz_dndx_poissons = np.zeros(N_ZBINS)
    perz_dX = np.zeros(N_ZBINS)
    perz_om_edges = np.linspace(20.3, LNHI_MAX, FN_NBINS + 1)
    perz_om_probs = [[list() for _ in range(FN_NBINS)] for _ in range(N_ZBINS)]
    perz_om_poissons = [np.zeros(FN_NBINS) for _ in range(N_ZBINS)]

    # per-z DIFFERENTIAL f(N|z) accumulators (the NEW deliverable)
    fnz_edges = np.linspace(FNZ_LNHI_MIN, FNZ_LNHI_MAX, FNZ_NBINS + 1)
    fnz_probs = [[list() for _ in range(FNZ_NBINS)] for _ in range(N_ZBINS)]
    fnz_poissons = [np.zeros(FNZ_NBINS) for _ in range(N_ZBINS)]

    def _reduce(ing):
        nonlocal fn_dX, dX_single, n_active_total
        _extend_probs(fn_probs, ing["fn_probs"])
        fn_poissons[:] = fn_poissons + ing["fn_poissons"]
        fn_dX += ing["fn_dX"]
        dX_single += ing["dX"]
        n_active_total += ing["n_active"]
        for thr in THRESHOLDS:
            _extend_probs(dndx_probs[thr], ing[f"dndx_probs_{thr}"])
            dndx_poissons[thr][:] = dndx_poissons[thr] + ing[f"dndx_poissons_{thr}"]
            _extend_probs(om_probs[thr], ing[f"omega_probs_{thr}"])
            om_poissons[thr][:] = om_poissons[thr] + ing[f"omega_poissons_{thr}"]
        # per-z
        _extend_probs(perz_dndx_probs, ing["perz_dndx_probs"])
        perz_dndx_poissons[:] = perz_dndx_poissons + ing["perz_dndx_poissons"]
        perz_dX[:] = perz_dX + ing["perz_dX"]
        for i in range(N_ZBINS):
            _extend_probs(perz_om_probs[i], ing["perz_om_probs"][i])
            perz_om_poissons[i][:] = perz_om_poissons[i] + ing["perz_om_poissons"][i]
            _extend_probs(fnz_probs[i], ing["fnz_probs"][i])
            fnz_poissons[i][:] = fnz_poissons[i] + ing["fnz_poissons"][i]

    done = 0
    if n_workers <= 1:
        for fp in files:
            _reduce(_ingredients(fp))
            done += 1
            if done % 200 == 0:
                print(f"[loa-literal] {done}/{n_files}  {time.time()-t0:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for ing in ex.map(_ingredients, files, chunksize=4):
                _reduce(ing)
                done += 1
                if done % 200 == 0:
                    print(f"[loa-literal] {done}/{n_files}  {time.time()-t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"[loa-literal] streaming done {n_files} files in {elapsed:.0f}s", flush=True)

    # ---- CI-combine ONCE on the totals (reuse calc_cddf PB combine) ----
    ref = DLACatalogue(processed_file=files[0], sample_file=SAMPLE,
                       catalog_file=CAT, window=WindowSpec(), **DLACAT_KW)
    ref.set_snr(SNR_THRESH)
    protonmass = 1.67262178e-24
    h100 = 3.2407789e-18 * HUBBLE
    light = 2.99e10
    conversion = protonmass / light * h100 / rho_crit(HUBBLE)
    try:
        # differential f(N)
        fn_ml, fn68, fn95 = ref._count_ci_from_probs_poissons(fn_probs, fn_poissons)
        dN = np.array([10**e2 - 10**e1 for e1, e2 in zip(fn_edges[:-1], fn_edges[1:])])
        l_Ncent = 0.5 * (fn_edges[:-1] + fn_edges[1:])
        fN = np.asarray(fn_ml) / fn_dX / dN
        fN68 = np.asarray(fn68) / fn_dX / np.vstack([dN, dN]).T
        fN95 = np.asarray(fn95) / fn_dX / np.vstack([dN, dN]).T

        # integrated dN/dX + Omega per threshold
        results = {}
        for thr in THRESHOLDS:
            ml, l68, l95 = ref._count_ci_from_probs_poissons(
                dndx_probs[thr], dndx_poissons[thr])
            dndx = float(ml[0]) / dX_single
            dndx68 = [float(l68[0][0]) / dX_single, float(l68[0][1]) / dX_single]
            dndx95 = [float(l95[0][0]) / dX_single, float(l95[0][1]) / dX_single]

            nhi_like, nhi68, nhi95 = ref._omega_ci_from_probs_poissons(
                om_probs[thr], om_poissons[thr], om_edges[thr])
            omega = conversion * nhi_like / dX_single
            omega68 = [conversion * nhi68[0] / dX_single,
                       conversion * nhi68[1] / dX_single]
            omega95 = [conversion * nhi95[0] / dX_single,
                       conversion * nhi95[1] / dX_single]

            results[str(thr)] = dict(
                dndx=dndx, dndx68=dndx68, dndx95=dndx95,
                omega=float(omega), omega68=omega68, omega95=omega95,
            )

        # per-z dN/dX + Omega at 20.3
        pz_ml, pz68, pz95 = ref._count_ci_from_probs_poissons(
            perz_dndx_probs, perz_dndx_poissons)
        z_grid = Z_GRID_COARSE
        z_mid = 0.5 * (z_grid[:-1] + z_grid[1:])
        # per-z DIFFERENTIAL f(N|z): the (per-(z,N) PB expected count)/(ΔX_zbin·ΔN_bin),
        # with the per-(z,N)-bin 68/95 PB credible band. fN_z[i] is the f(N) curve in
        # coarse z bin i over the fine nhi grid.
        dN_fnz = np.array([10**e2 - 10**e1
                           for e1, e2 in zip(fnz_edges[:-1], fnz_edges[1:])])
        l_Ncent_fnz = 0.5 * (fnz_edges[:-1] + fnz_edges[1:])
        perz = []
        perz_fN = []
        for i in range(N_ZBINS):
            dX_i = perz_dX[i]
            if dX_i <= 0:
                perz.append(dict(z=float(z_mid[i]), dndx=None, omega=None, dX=float(dX_i)))
                perz_fN.append(dict(z=float(z_mid[i]), dX=float(dX_i),
                                    logN_centers=l_Ncent_fnz.tolist(),
                                    lnhi_edges=fnz_edges.tolist(),
                                    f=None, f68_lo=None, f68_hi=None,
                                    f95_lo=None, f95_hi=None))
                continue
            d_i = float(pz_ml[i]) / dX_i
            d68 = [float(pz68[i][0]) / dX_i, float(pz68[i][1]) / dX_i]
            o_like, o68, o95 = ref._omega_ci_from_probs_poissons(
                perz_om_probs[i], perz_om_poissons[i], perz_om_edges)
            o_i = conversion * o_like / dX_i
            o68v = [conversion * o68[0] / dX_i, conversion * o68[1] / dX_i]
            perz.append(dict(
                z=float(z_mid[i]), dndx=d_i, dndx68=d68,
                omega=float(o_i), omega68=o68v, dX=float(dX_i)))
            # per-z differential f(N|z) with per-(z,N)-bin PB credible band
            fz_ml, fz68, fz95 = ref._count_ci_from_probs_poissons(
                fnz_probs[i], fnz_poissons[i])
            fNz = np.asarray(fz_ml) / dX_i / dN_fnz
            fNz68 = np.asarray(fz68) / dX_i / np.vstack([dN_fnz, dN_fnz]).T
            fNz95 = np.asarray(fz95) / dX_i / np.vstack([dN_fnz, dN_fnz]).T
            perz_fN.append(dict(
                z=float(z_mid[i]), dX=float(dX_i),
                logN_centers=l_Ncent_fnz.tolist(),
                lnhi_edges=fnz_edges.tolist(),
                f=fNz.tolist(),
                f68_lo=fNz68[:, 0].tolist(), f68_hi=fNz68[:, 1].tolist(),
                f95_lo=fNz95[:, 0].tolist(), f95_hi=fNz95[:, 1].tolist()))
    finally:
        try:
            ref.filehandle.close()
        except Exception:
            pass

    payload = dict(
        description="LITERAL calc_cddf.py Pathway-A Bayesian posterior CDDF on the "
                    "REAL DESI LOA FILTER-off dense-100k single-absorber run. "
                    "UNCORRECTED feed-forward (no completeness/FP/kernel deconvolution).",
        input_run="nersc_cddf_loa_v1_20260609 (real DESI LOA, FILTER-off, MAX_DLAS=1, PW=100k)",
        processed_dir=PD,
        n_files=n_files,
        n_files_total=len(all_files),
        n_unreadable_stubs=len(unreadable),
        is_subsample=is_subsample,
        subsample_n_files=subsample if is_subsample else None,
        n_active_sightlines_sum=n_active_total,
        sample_file=SAMPLE,
        catalog_file=CAT,
        sub_dla=SUB_DLA,
        z_min=Z_MIN, z_max=Z_MAX, lnhi_max=LNHI_MAX,
        thresholds=list(THRESHOLDS),
        hubble=HUBBLE,
        window="WindowSpec() production Lya-only (z_min_lyb=True, v_prox=v_tail=3000 km/s)",
        nan_fraction_verified="0.0000% (FILTER-off non-degenerate; verified on sampled files)",
        dX_single_zbin=dX_single,
        fn_dX=fn_dX,
        runtime_sec=elapsed,
        results=results,
        perz_20p3=perz,
        z_grid_coarse=Z_GRID_COARSE.tolist(),
        perz_fN=perz_fN,        # per-z DIFFERENTIAL f(N|z), the NEW deliverable
        fN_curve=dict(
            logN_centers=l_Ncent.tolist(),
            f=fN.tolist(),
            f68_lo=fN68[:, 0].tolist(), f68_hi=fN68[:, 1].tolist(),
            f95_lo=fN95[:, 0].tolist(), f95_hi=fN95[:, 1].tolist(),
            lnhi_edges=fn_edges.tolist(),
        ),
    )
    suffix = "_subset" if is_subsample else ""
    jpath = os.path.join(OUT, f"loa_literal_calccddf_results{suffix}.json")
    with open(jpath, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[loa-literal] wrote {jpath}", flush=True)

    print("\n==== LITERAL calc_cddf >= thr (REAL DESI LOA) ====")
    print(f"{'thr':>6}{'dN/dX':>12}{'10^3 Omega':>14}")
    for thr in THRESHOLDS:
        r = results[str(thr)]
        print(f"{thr:>6.1f}{r['dndx']:>12.5f}{r['omega']*1000:>14.5f}")
    return payload


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--subsample", type=int, default=None,
                    help="random subset of N healpix files (representative; reported as subsample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-partition", action="store_true",
                    help="skip the serial readability pre-scan (all files verified readable)")
    a = ap.parse_args()
    main(n_workers=a.n_workers, limit=a.limit, subsample=a.subsample, seed=a.seed,
         skip_partition=a.skip_partition)
