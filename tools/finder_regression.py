#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Production-finder invariance regression (PI 2026-08-12 §9).

Runs the FORWARD-state finder (worktree at prov/p1-refold-2026-08-08 +
nothing else) on real LOA spectra already processed by the HISTORICAL
production (loa_main_dark_v1, commit 84fa654, 2026-06-06, NERSC), and
compares the per-sightline outputs field by field.

Entry level: dlasearch.dlasearch_hpx — the exact function the production
driver calls per healpix; the only thing above it in main() is catalog
reading and the CFS datapath constant (I/O only; GL substitutes the local
mirror path). Config = the recorded BASELINE.env + slurm/configs/_base.env
defaults + the h5-recorded attrs (pair_prior_mode=off, dla_bias=2.0).

PRE-DECLARED tolerances (stated before any comparison is run):
  * integer fields / flags / grid endpoints ............ EXACT
  * MAP_z_dlas, MAP_log_nhis (sample-grid argmax picks) . EXACT (a tie
    within 1e-9 in sample log-likelihood is reported, not failed)
  * probabilities (p_dlas, p_no_dlas, model_posteriors) . |Δ| <= 1e-6
  * log-likelihood / log-prior / log-posterior fields ... rel <= 1e-6
  * z_dla_errs / log_nhi_errs ........................... rel <= 1e-5
Cross-platform floating-point (NERSC vs GL BLAS, summation order) is the
only expected source of nonzero Δ. BLAS threads pinned to 1.
"""
import os
import sys
import json

import numpy as np

WT = os.environ.get("REG_WT", "/tmp/claude-114399728/"
                    "-home-mfho-desi-gpy-dla-detection/"
                    "8c3eb4bd-d1c1-4d39-9a36-184f4269509d/scratchpad/"
                    "wt_fable_reg")
OUT = os.environ.get("REG_OUT")
PIXES = [int(x) for x in os.environ.get("REG_PIXES", "24,147,150").split(",")]

sys.path.insert(0, WT)

QSOCAT = ("/nfs/turbo/lsa-cavestru/mfho/DESI/loa/"
          "QSO_cat_loa_main_dark_healpix_v2-altbal.fits")
MIRROR = "/nfs/turbo/lsa-cavestru/mfho/DESI/loa/healpix/main/dark"
HIST = ("/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/"
        "loa_main_dark_v1/processed")
GLDATA = "/scratch/cavestru_root/cavestru0/mfho/DESI/desi_gpy_dla_detection"
MODEL = ("/nfs/turbo/lsa-cavestru/mfho/DESI/GP_trained/"
         "DEPLOYED_phase2_2lpt_loa124_nohcd_nobal_wide_m/phase2_result.h5")


def build_model_params():
    """Replicates desi-DLAGP.py main()'s params/model dict assembly
    verbatim, from the recorded historical configuration."""
    params_dict = dict(
        loading_min_lambda=910.0, loading_max_lambda=1550.0,
        normalization_min_lambda=1425.0, normalization_max_lambda=1475.0,
        min_lambda=911.75, max_lambda=1250.0, dlambda=0.15, k=30,
        max_noise_variance=9.0, max_z_cut=3000.0, min_z_cut=3000.0,
        num_forest_lines=31, num_lines=3, num_dla_samples=50000)
    params_subdla_dict = dict(params_dict)
    params_subdla_dict["num_dla_samples"] = 50000
    return dict(
        learned_file=MODEL,
        catalog_name=f"{GLDATA}/data/dr12q/processed/catalog.mat",
        los_catalog=(f"{GLDATA}/data/dla_catalogs/dr9q_concordance/"
                     "processed/los_catalog"),
        dla_catalog=(f"{GLDATA}/data/dla_catalogs/dr9q_concordance/"
                     "processed/dla_catalog"),
        dla_samples_file=(f"{GLDATA}/data/dr12q/processed/"
                          "pw_samples_a3_172_225_50000.mat"),
        sub_dla_samples_file=(f"{GLDATA}/data/dr12q/processed/"
                              "subdla_samples_a03_191_200_50000.mat"),
        params_dict=params_dict, params_subdla_dict=params_subdla_dict,
        min_z_separation=3000.0, prev_tau_0=0.00246, prev_beta=3.62,
        max_dlas=4, plot_figures=False, max_workers=6, batch_size=1250,
        figure_dir=OUT, filter_low_likelihood=True,
        filter_n_initial_floor=5000, filter_empty_mask_fallthrough=False,
        single_absorber_model=True, enable_tau_eb=True,
        tau_eb_factors=(0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0),
        tau_eb_apply_hcd_mask=False, tau_eb_mask_threshold_sigma=1.5,
        tau_eb_objective="null", early_stop_mode="baseline",
        pair_prior_mode="off", dla_bias=2.0)


def run():
    os.makedirs(OUT, exist_ok=True)
    os.chdir(WT)
    import dlasearch
    from astropy.table import Table
    import fitsio
    # read_catalog equivalent: the driver reads QSOCAT, applies z cut from
    # constants (2.0 < z < 4.25), BAL handling off (BALMASK=false) — but it
    # KEEPS the BAL columns for DLAFLAG checks. Reuse the driver's function.
    sys.argv = ["regression"]
    import importlib
    spec = importlib.util.spec_from_file_location(
        "desi_dlagp", os.path.join(WT, "desi-DLAGP.py"))
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)
    catalog = drv.read_catalog(QSOCAT, False, False)
    mp = build_model_params()
    for pix in PIXES:
        hpxcat = catalog[catalog["HPXPIXEL"] == pix]
        print(f"[pix {pix}] {len(hpxcat)} spectra", flush=True)
        res = dlasearch.dlasearch_hpx(pix, "main", "dark", MIRROR, hpxcat, mp)
        n = len(res) if res is not None and len(res) else 0
        print(f"[pix {pix}] fitresults rows: {n}", flush=True)


def compare():
    import h5py
    FIELDS_EXACT = ["target_ids", "detection_flags"]
    FIELDS_PROB = ["p_dlas", "p_no_dlas", "model_posteriors"]
    FIELDS_LOG = ["log_likelihoods_dla", "log_likelihoods_no_dla",
                  "log_priors_dla", "log_priors_no_dla",
                  "log_posteriors_dla", "log_posteriors_no_dla"]
    FIELDS_MAP = ["MAP_z_dlas", "MAP_log_nhis"]
    FIELDS_ERR = ["z_dla_errs", "log_nhi_errs"]
    FIELDS_GRID = ["min_z_dlas", "max_z_dlas", "z_qsos", "snrs"]
    summary = {}
    worst = {}
    fail = False
    for pix in PIXES:
        new = os.path.join(OUT, "processed",
                           f"processed-main-dark-{pix}.h5")
        old = os.path.join(HIST, f"processed-main-dark-{pix}.h5")
        with h5py.File(new) as hn, h5py.File(old) as ho:
            tn = np.asarray(hn["target_ids"])
            to = np.asarray(ho["target_ids"])
            common, in_, io_ = np.intersect1d(tn, to, return_indices=True)
            row = dict(n_new=len(tn), n_old=len(to), n_common=len(common))
            if len(tn) != len(to):
                row["note"] = "row-count differs"
            for f in FIELDS_EXACT:
                a, b = np.asarray(hn[f])[in_], np.asarray(ho[f])[io_]
                ok = np.array_equal(a, b)
                row[f] = "EXACT" if ok else "MISMATCH"
                fail |= not ok
            for f in FIELDS_GRID:
                a, b = np.asarray(hn[f])[in_], np.asarray(ho[f])[io_]
                d = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
                row[f] = d
                fail |= d > 1e-9
            for f in FIELDS_MAP:
                a, b = np.asarray(hn[f])[in_], np.asarray(ho[f])[io_]
                m = np.isfinite(a) & np.isfinite(b)
                d = float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0
                row[f] = d
                fail |= d > 1e-9          # EXACT (grid picks)
            for f in FIELDS_PROB:
                a, b = np.asarray(hn[f])[in_], np.asarray(ho[f])[io_]
                m = np.isfinite(a) & np.isfinite(b)
                d = float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0
                row[f] = d
                worst[f] = max(worst.get(f, 0.0), d)
                fail |= d > 1e-6
            for f in FIELDS_LOG:
                a, b = np.asarray(hn[f])[in_], np.asarray(ho[f])[io_]
                m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-30)
                d = float(np.max(np.abs((a[m] - b[m]) / b[m]))) if m.any() \
                    else 0.0
                row[f] = d
                worst[f] = max(worst.get(f, 0.0), d)
                fail |= d > 1e-6
            for f in FIELDS_ERR:
                a, b = np.asarray(hn[f])[in_], np.asarray(ho[f])[io_]
                m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-30)
                d = float(np.max(np.abs((a[m] - b[m]) / b[m]))) if m.any() \
                    else 0.0
                row[f] = d
                fail |= d > 1e-5
            summary[str(pix)] = row
    summary["_verdict"] = "FAIL" if fail else "PASS"
    summary["_worst"] = worst
    json.dump(summary, open(os.path.join(OUT, "regression_summary.json"),
                            "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    if sys.argv[-1] == "compare":
        compare()
    else:
        run()
        compare()
