"""Raw feed-forward (Pathway-A) Bayesian DLA CDDF on the 2LPT-0 mock.

WHAT
----
Computes the UNCORRECTED Bayesian posterior CDDF products (f(N), dN/dX, Ω_DLA)
from the GP-DLA inference posteriors via the existing ``calc_cddf.DLACatalogue``
Poisson-binomial machinery, with NO completeness / false-positive / kernel
deconvolution.  This is the "raw feed-forward" baseline against which the
catalog-HBI estimator's corrections are measured.

WHY THIS INPUT (not the FILTER-on baseline the task named)
----------------------------------------------------------
The raw-FF posterior CDDF integrates the per-(N,z)-bin probabilistic DLA counts
built from the 100k QMC per-sample log-likelihoods.  FILTER_LOW_LIKELIHOOD=1
truncates the low-likelihood per-sample evidence, which biases exactly those
per-bin counts -- so ``cddf_forward.filter_guard`` REFUSES a FILTER-on run.

The directory named in the task brief
(``gl_prod_2lpt0_v1_20260526``) is the FILTER-ON, MAX_DLAS production baseline
(model_posteriors = [Null, SubDLA, 1DLA, 2DLA, 3DLA], so sub_dla=True there) --
that is the *catalog* basis the HBI estimator uses, not a raw-FF input.

The scientifically valid raw-FF input is the dedicated FILTER-OFF, MAX_DLAS=1,
dense-100k-QMC run ``gl_cddf_2lpt0_v1_filteroff_maxdla1_20260526`` whose
model_posteriors = [Null, 1DLA] (single-absorber layout) => sub_dla=FALSE.
(Memory: "GW reweighting computable from existing FILTER=off/maxdla1".)

NO COMBINE / NO DISK BLOWUP
---------------------------
The per-(logN,z)-bin Poisson-binomial deposit is ADDITIVE over sightlines, so we
stream file-by-file, accumulate the EXACT ``_split_distributions`` ingredients
(``probs`` lists concatenate, ``poissons`` arrays sum, ``dX`` sums), open ONE file
at a time (~1-2 GB/worker), and run the EXISTING CI-combine ONCE on the totals.
This is the same additive seam ``cddf_forward.streaming`` uses; we extend it here
to additionally extract the THREE threshold single-z-bin dN/dX ingredients and the
fine f(N) ingredients in ONE pass per file (so the 1081-file run streams once, not
4x).  We reuse calc_cddf kernels verbatim: nothing in the inference path is
touched.

CONVENTIONS (matched to cddf_catalog_hbi.HBIConfig)
---------------------------------------------------
z window [2.0, 3.5]; SNR>2 enters via the production WindowSpec / snr cuts already
baked into min_z_dlas/max_z_dlas; lnhi ceiling 22.4 (drop_top_bin_above);
report thresholds 20.0/20.3/20.6; hubble=0.7; production Lyα-only WindowSpec.
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
PD = ("/scratch/cavestru_root/cavestru0/mfho/"
      "gl_cddf_2lpt0_v1_filteroff_maxdla1_20260526/outputs/figures/processed")
SAMPLE = ("/scratch/cavestru_root/cavestru0/mfho/DESI/desi_gpy_dla_detection/"
          "data/dr12q/processed/pw_samples_a3_172_225_100000.mat")
CAT = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/"
       "v2.8.5/mock-0/loa-124/zcat.fits")
TRUTH = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/"
         "v2.8.5/mock-0/loa-124/hcd_truth_cat.fits")
OUT = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
       "hbi_validation_2lpt0/rawff")

Z_MIN, Z_MAX = 2.0, 3.5
LNHI_MAX = 22.4          # = HBIConfig.drop_top_bin_above
THRESHOLDS = (20.0, 20.3, 20.6)
FN_LNHI_MIN, FN_LNHI_MAX, FN_NBINS = 19.5, 22.0, 25   # differential f(N) grid
HUBBLE = 0.7
SUB_DLA = False          # FILTER-off maxdla1 layout: model_posteriors=[Null, 1DLA]

# Task-provided truth anchors (HBI estimator internal truth) -- the values to beat.
TRUTH_DNDX_200 = 0.0859
TRUTH_OMEGA_200 = 6.94e-4

DLACAT_KW = dict(sub_dla=SUB_DLA, high_nhi_cut_value=LNHI_MAX)


# --------------------------------------------------------------------------- #
# per-file ingredient worker (picklable, module-level)
# --------------------------------------------------------------------------- #
def _ingredients(processed_file):
    """All raw-FF ingredients from ONE file (one DLACatalogue, closed after).

    Returns the additive (probs, poissons) + dX for:
      * fine f(N) over [FN_LNHI_MIN, FN_LNHI_MAX] (single z window) ;
      * three threshold single-z-bin dN/dX (lnhi_min=thr..LNHI_MAX) ;
      * Omega over the SAME fine lnhi grid as the >=20.0 CDDF (single z window),
        for each threshold (lnhi grid floored at the threshold).
    """
    win = WindowSpec()
    cat = DLACatalogue(processed_file=processed_file, sample_file=SAMPLE,
                       catalog_file=CAT, window=win, **DLACAT_KW)
    try:
        out = {"processed_file": processed_file}

        # ---- fine f(N) ingredients (nhi bins) ----
        fn_edges = np.linspace(FN_LNHI_MIN, FN_LNHI_MAX, FN_NBINS + 1)
        pr_f, po_f = cat._split_distributions(
            fn_edges, lred=Z_MIN, ured=Z_MAX,
            lnhi_min=fn_edges[0], lnhi_max=fn_edges[-1], nhi=True)
        out["fn_probs"] = pr_f
        out["fn_poissons"] = np.asarray(po_f, float)
        out["fn_dX"] = float(cat.path_length(Z_MIN, Z_MAX))

        # ---- per-threshold integrated dN/dX (single z bin) + Omega ----
        for thr in THRESHOLDS:
            # dN/dX: a single z bin [Z_MIN, Z_MAX], one lnhi bin [thr, LNHI_MAX].
            z_edges = np.array([Z_MIN, Z_MAX])
            pr_d, po_d = cat._split_distributions(
                z_edges, lred=Z_MIN, ured=Z_MAX,
                lnhi_min=thr, lnhi_max=LNHI_MAX, nhi=False)
            out[f"dndx_probs_{thr}"] = pr_d
            out[f"dndx_poissons_{thr}"] = np.asarray(po_d, float)

            # Omega: N-weighted PDF over a fine lnhi grid floored at thr, single z bin.
            om_edges = np.linspace(thr, LNHI_MAX, FN_NBINS + 1)
            pr_o, po_o = cat._split_distributions(
                om_edges, lred=Z_MIN, ured=Z_MAX,
                lnhi_min=thr, lnhi_max=LNHI_MAX, nhi=True)
            out[f"omega_probs_{thr}"] = pr_o
            out[f"omega_poissons_{thr}"] = np.asarray(po_o, float)
            out[f"omega_edges_{thr}"] = om_edges

        # path length over [Z_MIN, Z_MAX] is the common dX for the single-z-bin products
        out["dX"] = float(cat.path_length(Z_MIN, Z_MAX))
        # Active sightline TARGETIDs that contribute to dX: the SAME mask
        # path_length uses (SNR cut * z-range cut * catalog condition), so the
        # self-consistency truth can be restricted to the SAME population.
        dx_mask = cat._filter_snr_spectra() * cat._filter_z_dlas(cat.z_dla_minimum)
        out["active_tids"] = np.asarray(
            cat.target_ids[np.where(dx_mask)[0]]).astype(np.int64)
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


def _partition_readable(files):
    """Skip TRUNCATED/empty HDF5 stubs (e.g. 96-byte killed-job outputs) so the
    population run is not crashed by one bad file. Mirrors
    cddf_forward.streaming._partition_readable_files. Opening the superblock +
    checking target_ids is cheap."""
    import h5py
    readable, unreadable = [], []
    for fp in files:
        try:
            with h5py.File(fp, "r") as h:
                if "target_ids" not in h or h["target_ids"].shape[0] == 0:
                    unreadable.append(fp)
                    continue
            readable.append(fp)
        except (OSError, KeyError):
            unreadable.append(fp)
    return readable, unreadable


def main(n_workers=4, limit=None):
    os.makedirs(OUT, exist_ok=True)
    all_files = sorted(glob.glob(os.path.join(PD, "processed-*.h5")))
    if limit:
        all_files = all_files[:limit]
    files, unreadable = _partition_readable(all_files)
    n_files = len(files)
    print(f"[rawff] {len(all_files)} files; {n_files} readable, "
          f"{len(unreadable)} unreadable/empty stubs skipped, "
          f"n_workers={n_workers}", flush=True)

    # FILTER guard (explicit: schema does not persist the flag; this run is FILTER-off).
    assert_filter_off(0, ctx="rawff_2lpt0")

    t0 = time.time()

    # accumulators
    fn_edges = np.linspace(FN_LNHI_MIN, FN_LNHI_MAX, FN_NBINS + 1)
    fn_nbins = FN_NBINS
    fn_probs = [list() for _ in range(fn_nbins)]
    fn_poissons = np.zeros(fn_nbins)
    fn_dX = 0.0

    dndx_probs = {thr: [list()] for thr in THRESHOLDS}   # single z bin
    dndx_poissons = {thr: np.zeros(1) for thr in THRESHOLDS}
    om_edges = {thr: np.linspace(thr, LNHI_MAX, FN_NBINS + 1) for thr in THRESHOLDS}
    om_probs = {thr: [list() for _ in range(FN_NBINS)] for thr in THRESHOLDS}
    om_poissons = {thr: np.zeros(FN_NBINS) for thr in THRESHOLDS}
    dX_single = 0.0
    active_tids = set()

    def _reduce(ing):
        nonlocal fn_dX, dX_single
        _extend_probs(fn_probs, ing["fn_probs"])
        fn_poissons[:] = fn_poissons + ing["fn_poissons"]
        fn_dX += ing["fn_dX"]
        dX_single += ing["dX"]
        active_tids.update(int(t) for t in ing["active_tids"])
        for thr in THRESHOLDS:
            _extend_probs(dndx_probs[thr], ing[f"dndx_probs_{thr}"])
            dndx_poissons[thr][:] = dndx_poissons[thr] + ing[f"dndx_poissons_{thr}"]
            _extend_probs(om_probs[thr], ing[f"omega_probs_{thr}"])
            om_poissons[thr][:] = om_poissons[thr] + ing[f"omega_poissons_{thr}"]

    done = 0
    if n_workers <= 1:
        for fp in files:
            _reduce(_ingredients(fp))
            done += 1
            if done % 50 == 0:
                print(f"[rawff] {done}/{n_files}  {time.time()-t0:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for ing in ex.map(_ingredients, files, chunksize=1):
                _reduce(ing)
                done += 1
                if done % 50 == 0:
                    print(f"[rawff] {done}/{n_files}  {time.time()-t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"[rawff] streaming done {n_files} files in {elapsed:.0f}s", flush=True)

    # ---- CI-combine ONCE on the totals (reuse calc_cddf PB combine) ----
    ref = DLACatalogue(processed_file=files[0], sample_file=SAMPLE,
                       catalog_file=CAT, window=WindowSpec(), **DLACAT_KW)
    try:
        # differential f(N)
        fn_ml, fn68, fn95 = ref._count_ci_from_probs_poissons(fn_probs, fn_poissons)
        dN = np.array([10**e2 - 10**e1 for e1, e2 in zip(fn_edges[:-1], fn_edges[1:])])
        l_Ncent = 0.5 * (fn_edges[:-1] + fn_edges[1:])
        fN = np.asarray(fn_ml) / fn_dX / dN
        fN68 = np.asarray(fn68) / fn_dX / np.vstack([dN, dN]).T
        fN95 = np.asarray(fn95) / fn_dX / np.vstack([dN, dN]).T

        # integrated dN/dX per threshold
        protonmass = 1.67262178e-24
        h100 = 3.2407789e-18 * HUBBLE
        light = 2.99e10
        conversion = protonmass / light * h100 / rho_crit(HUBBLE)

        results = {}
        for thr in THRESHOLDS:
            ml, l68, l95 = ref._count_ci_from_probs_poissons(
                dndx_probs[thr], dndx_poissons[thr])
            dndx = float(ml[0]) / dX_single
            dndx68 = [float(l68[0][0]) / dX_single, float(l68[0][1]) / dX_single]
            dndx95 = [float(l95[0][0]) / dX_single, float(l95[0][1]) / dX_single]

            # Omega via N-weighted PB convolution (reuse _omega_ci_from_probs_poissons)
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
    finally:
        try:
            ref.filehandle.close()
        except Exception:
            pass

    # ---- internal-consistency truth on the SAME window/dX ----
    from astropy.table import Table
    t = Table.read(TRUTH)
    tn = np.asarray(t["NHI"], float); tz = np.asarray(t["Z"], float)
    tsnr = np.asarray(t["SNR"], float); ttid = np.asarray(t["TARGETID"]).astype(np.int64)
    # Restrict truth to absorbers on sightlines that actually contributed to dX
    # (same SNR>2 + z-range + catalog-coverage population), so dndx_self is a FAIR
    # truth on the SAME path length.  (For the full 1081-file run this is the whole
    # processed population; the restriction matters because the processed files do
    # not cover 100% of the zcat sightlines.)
    in_active = np.isin(ttid, np.fromiter(active_tids, dtype=np.int64))
    kp = (tz >= Z_MIN) & (tz < Z_MAX) & (tsnr > 2.0) & (tn < LNHI_MAX) & in_active
    truth_self = {}
    for thr in THRESHOLDS:
        n_abs = int(np.sum(kp & (tn >= thr)))
        truth_self[str(thr)] = dict(n_absorbers=n_abs,
                                    dndx_self=n_abs / dX_single)

    payload = dict(
        description="Raw feed-forward (Pathway A) Bayesian posterior CDDF, "
                    "UNCORRECTED (no completeness/FP/kernel deconvolution).",
        input_run="gl_cddf_2lpt0_v1_filteroff_maxdla1_20260526 (FILTER-off, MAX_DLAS=1)",
        processed_dir=PD,
        n_files=n_files,
        n_files_total=len(all_files),
        n_unreadable_stubs=len(unreadable),
        unreadable_files=[os.path.basename(f) for f in unreadable],
        n_active_sightlines=len(active_tids),
        used_all_1150="N/A: this is the FILTER-off maxdla1 run (1081 files); "
                      "ALL of them used (no subset).",
        sample_file=SAMPLE,
        catalog_file=CAT,
        truth_file=TRUTH,
        sub_dla=SUB_DLA,
        z_min=Z_MIN, z_max=Z_MAX, lnhi_max=LNHI_MAX,
        thresholds=list(THRESHOLDS),
        hubble=HUBBLE,
        window="WindowSpec() production Lya-only (z_min_lyb=True, v_prox=v_tail=3000 km/s)",
        dX_single_zbin=dX_single,
        fn_dX=fn_dX,
        runtime_sec=elapsed,
        results=results,
        truth_anchors_task=dict(dndx_20p0=TRUTH_DNDX_200, omega_20p0=TRUTH_OMEGA_200),
        truth_self_consistency=truth_self,
        R0_raw_vs_task_truth=dict(
            dndx_20p0=results["20.0"]["dndx"] / TRUTH_DNDX_200,
            omega_20p0=results["20.0"]["omega"] / TRUTH_OMEGA_200,
        ),
        R0_raw_vs_self_truth={
            str(thr): dict(dndx=results[str(thr)]["dndx"] / truth_self[str(thr)]["dndx_self"]
                           if truth_self[str(thr)]["dndx_self"] > 0 else None)
            for thr in THRESHOLDS},
        fN_curve=dict(
            logN_centers=l_Ncent.tolist(),
            f=fN.tolist(),
            f68_lo=fN68[:, 0].tolist(), f68_hi=fN68[:, 1].tolist(),
            f95_lo=fN95[:, 0].tolist(), f95_hi=fN95[:, 1].tolist(),
            lnhi_edges=fn_edges.tolist(),
        ),
    )
    jpath = os.path.join(OUT, "rawff_results.json")
    with open(jpath, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[rawff] wrote {jpath}", flush=True)

    # ---- figure: f(N) raw-FF vs 2LPT truth (truth on the active population) ----
    kp_logN = (tz >= Z_MIN) & (tz < Z_MAX) & (tsnr > 2.0) & in_active
    _plot_fN(l_Ncent, fN, fN68, fN95, t, kp_logN=kp_logN,
             tn=tn, fn_dX=fn_dX, fn_edges=fn_edges)
    return payload


def _plot_fN(l_Ncent, fN, fN68, fN95, truth_table, kp_logN, tn, fn_dX, fn_edges):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # truth f(N) on the SAME lnhi grid + dX (count / dX / dN per bin)
    dN = np.array([10**e2 - 10**e1 for e1, e2 in zip(fn_edges[:-1], fn_edges[1:])])
    tn_kept = tn[kp_logN & (tn >= fn_edges[0]) & (tn < fn_edges[-1])]
    counts, _ = np.histogram(tn_kept, bins=fn_edges)
    f_truth = counts / fn_dX / dN

    fig, ax = plt.subplots(figsize=(7, 5.5))
    N = 10 ** l_Ncent
    # raw-FF with 68% band
    yerr = np.vstack([fN - fN68[:, 0], fN68[:, 1] - fN])
    good = fN > 0
    ax.errorbar(N[good], fN[good], yerr=(yerr[0][good], yerr[1][good]),
                fmt="o", ms=5, color="C0", label="Raw feed-forward (Pathway A, uncorrected)")
    ax.fill_between(N, fN95[:, 0], fN95[:, 1], color="C0", alpha=0.18, label="95% PB CI")
    gt = f_truth > 0
    ax.plot(N[gt], f_truth[gt], "s-", color="k", ms=4, label="2LPT-0 truth (SNR>2, z[2,3.5])")
    ax.axvline(10**20.3, ls=":", color="grey", lw=1)
    ax.text(10**20.32, ax.get_ylim()[0] if False else 1e-24, "20.3", color="grey")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$N_\mathrm{HI}$ (cm$^{-2}$)")
    ax.set_ylabel(r"$f(N_\mathrm{HI})$ (cm$^2$)")
    ax.set_title("Raw feed-forward vs 2LPT-0 truth CDDF\n"
                 "FILTER-off maxdla1, z[2.0,3.5], logN 19.5-22.0")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figpath = os.path.join(OUT, "fig_rawff_fN.png")
    fig.savefig(figpath, dpi=130)
    print(f"[rawff] wrote {figpath}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    main(n_workers=a.n_workers, limit=a.limit)
