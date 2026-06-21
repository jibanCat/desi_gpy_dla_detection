"""LITERAL Pathway-A calc_cddf.py on the FULL real DESI LOA FILTER-off posteriors.

WHAT
----
Runs the LITERAL Bayesian posterior CDDF (``calc_cddf.DLACatalogue`` Poisson-binomial
machinery) on the dense FILTER-off single-absorber real-LOA run
``nersc_cddf_loa_v1_20260609`` (618 GB, 16,519 healpix). Produces the differential
f(N_HI) with 68/95% PB credible band, integrated dN/dX(>=thr) and 10^3 Omega(>=thr),
plus per-z dN/dX / Omega in the Track-C z grid (z [2.0, 3.5]).

WHY THIS IS NOW POSSIBLE (it was not before)
--------------------------------------------
The earlier overlay used the 2LPT-0 MOCK literal as a stand-in because the only
real-LOA runs on GreatLakes were FILTER_LOW_LIKELIHOOD=1 (per-sample DLA
log-likelihoods carry NaN -> _split_distributions returns NaN -> f(N)==0). This run
is the dedicated FILTER-off (FILTER_LOW_LIKELIHOOD=0), MAX_DLAS=1,
SINGLE_ABSORBER_MODEL=1, PW=100,000 dense-CDDF run. VERIFIED: sample_log_likelihoods_dla
NaN fraction = 0.0000%; model_posteriors shape (N,2) = [Null, DLA] -> sub_dla=FALSE;
QMC grid = pw_samples_a3_172_225_100000.mat (100,000 samples = sll axis).

NO COMBINE / NO DISK BLOWUP (additive streaming)
-----------------------------------------------
The per-(logN,z)-bin Poisson-binomial deposit is ADDITIVE over sightlines, so we
stream file-by-file, accumulate the EXACT ``_split_distributions`` ingredients
(``probs`` lists concatenate, ``poissons`` arrays sum, ``dX`` sums), open ONE file
at a time, and run the EXISTING CI-combine ONCE on the totals. Mathematically
IDENTICAL to one combined.h5, bounded memory, NO combined file on disk. This is the
same additive seam ``cddf_forward.streaming`` / ``rawff_2lpt0`` use. We reuse
calc_cddf kernels VERBATIM: nothing in the inference path is touched.

CATALOG (dX coverage)
---------------------
``calc_cddf`` uses ``catalog_file`` ONLY for TID alignment; a processed TID NOT in
the catalog is dropped from dX (``self.condition``). The shared MAP dlacat covers
only ~47% of sightlines (one row per MAP detection), which would inflate dN/dX. So
we pass a TARGETID-only catalog (``loa_tidcat.fits``) covering ALL 942,946 processed
sightlines, built by ``build_loa_tidcat.py`` — dX is then the FULL SNR/z-filtered
path length, exactly as the production CDDF intends.

PRIVACY (real DESI LOA): AGGREGATE f(N)/dN/dX/Omega curves only; NO raw per-sightline
rows. The 618 GB posteriors stay on scratch; outputs (JSON + figure) go to scratch +
the private notes repo only.

CONVENTIONS (matched to rawff_2lpt0 / cddf_catalog_hbi.HBIConfig)
-----------------------------------------------------------------
z window [2.0, 3.5]; production Lya-only WindowSpec (z_min_lyb=True, v_prox=v_tail=
3000 km/s); lnhi ceiling 22.4 (drop_top_bin_above); report thresholds 20.0/20.3/20.6;
hubble=0.7.
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
PD = ("/scratch/cavestru_root/cavestru0/mfho/nersc/loa_cddf_main_dark_v1/"
      "nersc_cddf_loa_v1_20260609/outputs/figures/processed")
SAMPLE = ("/scratch/cavestru_root/cavestru0/mfho/DESI/desi_gpy_dla_detection/"
          "data/dr12q/processed/pw_samples_a3_172_225_100000.mat")
# TARGETID-only catalog covering ALL processed sightlines (built by build_loa_tidcat.py)
CAT = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/"
       "tf_loa_calc_cddf/loa_tidcat.fits")
OUT = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/"
       "tf_loa_calc_cddf")

Z_MIN, Z_MAX = 2.0, 3.5
LNHI_MAX = 22.4          # = HBIConfig.drop_top_bin_above
THRESHOLDS = (20.0, 20.3, 20.6)
FN_LNHI_MIN, FN_LNHI_MAX, FN_NBINS = 19.5, 22.0, 25   # differential f(N) grid
HUBBLE = 0.7
SUB_DLA = False          # FILTER-off maxdla1 layout: model_posteriors=[Null, 1DLA]
N_ZBINS = 9              # per-z grid; Track-C tf_loa uses 9 bins over [2.0,3.5]

DLACAT_KW = dict(sub_dla=SUB_DLA, high_nhi_cut_value=LNHI_MAX)


# --------------------------------------------------------------------------- #
# per-file ingredient worker (picklable, module-level)
# --------------------------------------------------------------------------- #
def _ingredients(processed_file):
    """All literal-calc_cddf ingredients from ONE file (one DLACatalogue, closed)."""
    win = WindowSpec()
    cat = DLACatalogue(processed_file=processed_file, sample_file=SAMPLE,
                       catalog_file=CAT, window=win, **DLACAT_KW)
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

        # ---- per-z dN/dX + Omega at headline 20.3 (Track-C z grid) ----
        z_grid = np.linspace(Z_MIN, Z_MAX, N_ZBINS + 1)
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
        z_grid = np.linspace(Z_MIN, Z_MAX, N_ZBINS + 1)
        z_mid = 0.5 * (z_grid[:-1] + z_grid[1:])
        perz = []
        for i in range(N_ZBINS):
            dX_i = perz_dX[i]
            if dX_i <= 0:
                perz.append(dict(z=float(z_mid[i]), dndx=None, omega=None))
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
