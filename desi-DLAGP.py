#!/usr/bin/env python

"""
desi-DLAGP.py

Search for DLAs in DESI quasar spectra using Gaussian Processes.
"""

from astropy.table import Table, vstack
import numpy as np
from scipy.interpolate import interp1d
import fitsio

import os
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import dlasearch
import constants

from desiutil.log import log

# GP-DLA imports
from run_bayes_select import DLAHolder
from gpy_dla_detection.set_parameters import Parameters


def parse(options=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""search for DLAs in DESI quasar spectra""",
    )

    parser.add_argument(
        "-q",
        "--qsocat",
        type=str,
        default=None,
        required=True,
        help="path to quasar catalog",
    )

    parser.add_argument(
        "-r",
        "--release",
        type=str,
        default=None,
        required=True,
        help="DESI redux version (e.g. iron)",
    )

    parser.add_argument(
        "-p",
        "--program",
        type=str,
        default="dark",
        required=False,
        help="observing program, default is dark",
    )

    parser.add_argument(
        "-s",
        "--survey",
        type=str,
        default="main",
        required=False,
        help="survey, default is main",
    )

    parser.add_argument(
        "--mocks",
        default=False,
        required=False,
        action="store_true",
        help="is this a mock catalog? Default is False",
    )

    parser.add_argument(
        "--mockdir",
        type=str,
        default=None,
        required=False,
        help="path to mock directory",
    )

    parser.add_argument(
        "--tilebased",
        default=False,
        required=False,
        action="store_true",
        help="use tile based coadds, default is False",
    )

    parser.add_argument(
        "--balmask",
        default=False,
        required=False,
        action="store_true",
        help="should BALs be masked using AI_CIV? Default is False but recommended setting is True",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=None,
        required=True,
        help="output directory for DLA catalog",
    )

    ###======== GP-DLA specific arguments =========###
    parser.add_argument(
        "--learned_file",
        default="data/dr12q/processed/learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat",
        help="Learned QSO model file path.",
    )
    parser.add_argument(
        "--catalog_name",
        default="data/dr12q/processed/catalog.mat",
        help="Catalog file path.",
    )
    parser.add_argument(
        "--los_catalog",
        default="data/dla_catalogs/dr9q_concordance/processed/los_catalog",
        help="Line-of-sight catalog file path.",
    )
    parser.add_argument(
        "--dla_catalog",
        default="data/dla_catalogs/dr9q_concordance/processed/dla_catalog",
        help="DLA catalog file path.",
    )
    parser.add_argument(
        "--dla_samples_file",
        default="data/dr12q/processed/dla_samples_a03.mat",
        help="DLA samples file path.",
    )
    parser.add_argument(
        "--sub_dla_samples_file",
        default="data/dr12q/processed/subdla_samples.mat",
        help="Sub-DLA samples file path.",
    )

    # DLA-related arguments
    parser.add_argument(
        "--min_z_separation",
        type=float,
        default=3000.0,
        help="Minimum redshift separation for DLA models.",
    )
    parser.add_argument(
        "--prev_tau_0", type=float, default=0.00554, help="Previous value for tau_0."
    )
    parser.add_argument(
        "--prev_beta", type=float, default=3.182, help="Previous value for beta."
    )
    parser.add_argument(
        "--max_dlas", type=int, default=3, help="Maximum number of DLAs to model."
    )
    parser.add_argument(
        "--plot_figures",
        type=int,
        default=0,
        help="Set to 1 to generate plots, 0 otherwise.",
    )
    # figure directory
    parser.add_argument(
        "--figure_dir",
        type=str,
        default="figures",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=32,
        help="Number of workers for parallel processing.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=313,
        help="Batch size for parallel model evidence computation.",
    )

    # DLA-related arguments
    parser.add_argument(
        "--filter_low_likelihood",
        type=int,
        default=0,
        help="Set to 1 to filter out low likelihood samples during model evidence computation, 0 otherwise.",
        dest="filter_low_likelihood"
    )

    # FILTER=1 knobs — see docs/notes/2026-05-13_filter1_knob_tuning.md
    parser.add_argument(
        "--filter_n_initial_floor",
        type=int, default=5000,
        help="Knob 1: minimum size of the FILTER=1 coarse-scan budget. The actual "
             "n_initial = max(num_dla_samples // 20, this_floor). Default 5000 "
             "matches the historical hardcoded value. Raise to widen coarse "
             "coverage when weak-DLA recall is the issue.",
        dest="filter_n_initial_floor",
    )
    parser.add_argument(
        "--filter_empty_mask_fallthrough",
        type=int, default=0,
        help="Knob 4: 1 = when the FILTER=1 coarse-scan yields an empty valid_mask, "
             "fall through to the FILTER=0 full-sample path instead of early-stopping "
             "with only the coarse samples. 0 (default) preserves historical early-stop.",
        dest="filter_empty_mask_fallthrough",
    )

    # single absorber model only
    parser.add_argument(
        "--single_absorber_model",
        type=int,
        default=0,
        help="Set to 1 to use only single absorber model (no subDLA), 0 otherwise.",
        dest="single_absorber_model"
    )

    # Per-spectrum empirical-Bayes τ_eff fit
    # (see gpy_dla_detection/tau_eb.py + docs/tau_eb_hcd_mask.md +
    #  docs/notes/2026-04-29_tau_eb_n90_unbiasedness.md)
    parser.add_argument(
        "--enable_tau_eb", type=int, default=0,
        help="Set to 1 to enable per-spectrum τ_0 fit; default 0.",
        dest="enable_tau_eb",
    )
    parser.add_argument(
        "--tau_eb_factors",
        type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
        help="τ-grid factors for the EB scan (multiplied by --prev_tau_0).",
    )
    parser.add_argument(
        "--tau_eb_apply_hcd_mask", type=int, default=0,
        help="Set to 1 to mask HCD pixels during τ-fit; default 0 (at scale "
             "the mask over-corrects — see 2026-04-29_tau_eb_n90_unbiasedness.md).",
        dest="tau_eb_apply_hcd_mask",
    )
    parser.add_argument(
        "--tau_eb_mask_threshold_sigma",
        type=float, default=1.5,
        help="HCD-flag threshold N: pixels with (y-μ_pred)/σ < -N are masked "
             "during the τ-fit step (only when --tau_eb_apply_hcd_mask=1).",
    )
    parser.add_argument(
        "--tau_eb_objective",
        choices=["null", "dla"], default="null",
        help='"null" (default, cheap): fit τ on null-model log evidence. '
             '"dla": match the validated diagnostic at higher cost.',
    )

    # Multi-DLA early-stop policy
    # (see docs/notes/2026-05-12_multidla_early_stop_bug.md)
    _early_stop_default = os.environ.get("EARLY_STOP_MODE", "baseline")
    if _early_stop_default not in ("baseline", "A", "D"):
        _early_stop_default = "baseline"
    parser.add_argument(
        "--early_stop_mode",
        choices=["baseline", "A", "D"], default=_early_stop_default,
        help='Multi-DLA early-stop policy. "baseline" (default) keeps the '
             'historical penalized-likelihood-vs-null heuristic. "A" disables '
             'the null-vs-current early-stop entirely (max_dlas / NaN / lik-decreased '
             'early-stops still apply). "D" compares the PRE-Occam likelihood to '
             'null instead of the penalized one. Default also reads from env '
             'EARLY_STOP_MODE.',
        dest="early_stop_mode",
    )

    # DLA velocity-separation clustering prior
    parser.add_argument(
        "--pair_prior_mode", dest="pair_prior_mode", default="off",
        choices=["off", "clustering"],
        help="DLA velocity-separation clustering prior on the multi-DLA evidence "
             "(default off => byte-identical). 'clustering' is mock-calibrated (b_DLA=2); "
             "on real data propagate b-uncertainty and exclude the catalog from "
             "clustering/bias science (spec section 3).")
    parser.add_argument(
        "--dla_bias", dest="dla_bias", type=float, default=2.0,
        help="Linear DLA bias for the clustering prior (xi ~ b^2). 2.0 = LyaCoLoRe mock value.")

    # Parameter-related arguments
    # These are the values used in the trained GP model, don't change them unless you change the trained model
    parser.add_argument(
        "--loading_min_lambda",
        type=float,
        default=910,
        help="Range of rest wavelengths to load (Å).",
    )
    parser.add_argument(
        "--loading_max_lambda",
        type=float,
        default=1550,
        help="Range of rest wavelengths to load (Å).",
    )
    parser.add_argument(
        "--normalization_min_lambda",
        type=float,
        default=1425,
        help="Range of rest wavelengths for flux normalization.",
    )
    parser.add_argument(
        "--normalization_max_lambda",
        type=float,
        default=1475,
        help="Range of rest wavelengths for flux normalization.",
    )
    parser.add_argument(
        "--min_lambda",
        type=float,
        default=911.75,
        help="Range of rest wavelengths to model (Å).",
    )
    parser.add_argument(
        "--max_lambda",
        type=float,
        default=1216.75,
        help="Range of rest wavelengths to model (Å).",
    )
    parser.add_argument(
        "--dlambda", type=float, default=0.25, help="Separation of wavelength grid (Å)."
    )
    parser.add_argument(
        "--k", type=int, default=20, help="Rank of non-diagonal contribution."
    )
    parser.add_argument(
        "--max_noise_variance",
        type=float,
        default=9,
        help="Maximum pixel noise allowed during model training.",
    )
    parser.add_argument(
        "--max_z_cut",
        type=float,
        default=3000,
        help="Maximum redshift cut for DLA models.",
    )
    parser.add_argument(
        "--min_z_cut",
        type=float,
        default=3000,
        help="Minimum redshift cut for DLA models.",
    )
    # Additional parameters for the DLA model
    parser.add_argument(
        "--num_forest_lines",
        type=int,
        default=31,
        help="Number of forest lines to model.",
    )
    parser.add_argument(
        "--num_lines",
        type=int,
        default=3,
        help="Number of members of the Lyman series to use.",
    )
    # Number of DLA samples to generate
    parser.add_argument(
        "--num_dla_samples",
        type=int,
        default=10000,
        help="Number of DLA samples to generate.",
    )
    parser.add_argument(
        "--num_subdla_samples",
        type=int,
        default=10000,
        help="Number of sub-DLA samples to generate.",
    )

    # process range
    parser.add_argument(
        "--hpx_start",
        type=int,
        default=0,
        help="start healpix pixel",
    )
    parser.add_argument(
        "--hpx_end",
        type=int,
        default=1,
        help="end healpix pixel",
    )
    parser.add_argument(
        "--level2_start",
        type=int,
        default=0,
        help="start level2 folder",
    )
    parser.add_argument(
        "--level2_end",
        type=int,
        default=1,
        help="end level2 folder",
    )

    # external healpix list
    parser.add_argument(
        "--use_external_hpx_list",
        default=False,
        required=False,
        action="store_true",
        help="use external healpix list",
    )
    parser.add_argument(
        "--external_hpx_list",
        type=str,
        default=None,
        help="external healpix list",
    )

    # external TARGETID subset (mock mode only). When provided, the mock
    # catalog is restricted to these TARGETIDs *after* the standard z-cut /
    # BAL handling. Used for cheap targeted re-inference subsets (e.g. the
    # tau_eb high-z falsifier). Additive + config-only: it filters the QSO
    # catalog passed to dlasearch; the per-spectrum inference is unchanged.
    parser.add_argument(
        "--external_tid_list",
        type=str,
        default=None,
        help="path to a text/csv file of TARGETIDs (one per line) to restrict "
             "the mock catalog to. Mock mode only; ignored for healpix/tile runs.",
    )

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args


def main(args=None):
    if isinstance(args, (list, tuple, type(None))):
        args = parse(args)

    # print out the flags line by line for logging
    log.info("running with the following flags:")
    for arg in vars(args):
        log.info(f"{arg}: {getattr(args, arg)}")

    # Check if catalog exists
    if not os.path.isfile(args.qsocat):
        log.error(f"{args.qsocat} does not exist")
        exit(1)

    # TODO: check if outdir exists, if not create it
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)
        log.info(f"created output directory: {args.outdir}")
    # check if figure dir exists, if not create it
    if not os.path.exists(args.figure_dir):
        os.makedirs(args.figure_dir, exist_ok=True)
        log.info(f"created figure directory: {args.figure_dir}")

    # if catalog is healpix based, we must have program & survey
    if not (args.tilebased) and not (args.mocks):
        log.info(
            f"expecting healpix catalog for redux={args.release}, survey={args.survey}, program={args.program}; confirm this matches the catalog provided!"
        )
        log.info(f"running in between healpix pixels {args.hpx_start} - {args.hpx_end}")

    # confirm BAL masking choice
    if not (args.balmask):
        log.warning(
            f"BALs will not be masked! The only good reason to do this is if you do not have a BAL catalog, set --balmask to turn on masking."
        )

    # check if mock data
    if args.mocks and (args.mockdir is None):
        log.error(f"mocks argument set to true but no mock data path provided")
    elif args.mocks and not (os.path.exists(args.mockdir)):
        log.error(f"{args.mockdir} does not exist")
        exit(1)

    tini = time.time()

    # read in quasar catalog and intrinsic flux model
    # TODO: Get the total number of spectra
    # For real data, count the number of healpix pixels
    # For mock data, count the number of level1 folders
    if args.mocks:
        #  Mock section: count the total number of spectra.fits files
        datapath = f"{args.mockdir}/spectra-16"
        # list of .fits files, each ~ 800 spectra
        speclist = []
        all_level2 = []
        for level1 in os.listdir(f"{datapath}"):
            for level2 in os.listdir(f"{datapath}/{level1}"):
                if os.path.exists(
                    f"{datapath}/{level1}/{level2}/spectra-16-{level2}.fits"
                ):
                    speclist.append(
                        f"{datapath}/{level1}/{level2}/spectra-16-{level2}.fits"
                    )
                    all_level2.append(level2)

        # reorder speclist by level2
        argsortind = np.argsort(list(map(int, all_level2)))
        speclist = np.array(speclist)[
            argsortind
        ]  # these would be by order from 0 - 3071
        all_level2 = np.array(list(map(int, all_level2)))[argsortind]

        # running in between mock level2 folders: level2_start - level2_end
        log.info(
            "running in between mock level2 folders {} - {}; Total level2: {}".format(
                args.level2_start, args.level2_end, all_level2[-1]
            )
        )
        # TODO: So all_level2 is discontinuous, so it might make more sense to just indexing the speclist
        # ind = (all_level2 >= args.level2_start) & (all_level2 < args.level2_end)
        all_level2 = all_level2[args.level2_start : args.level2_end]
        speclist = speclist[args.level2_start : args.level2_end]

        log.info(f"Specfiles to process: {' '.join(speclist)}")
        log.info(f"level2 from {all_level2[0]} to {all_level2[-1]}")

        catalog = read_mock_catalog(args.qsocat, args.balmask, args.mockdir)

        # Optional TARGETID subset (cheap targeted re-inference, e.g. the
        # tau_eb high-z falsifier). Filters the QSO catalog only; inference
        # is byte-identical. dlasearch_mock further intersects this with each
        # spectra-16 file's TARGETIDs, so empty-overlap files are skipped fast.
        if args.external_tid_list is not None:
            ext_tids = np.loadtxt(args.external_tid_list, dtype=np.int64, ndmin=1)
            ext_tids = np.unique(ext_tids)
            n_before = len(catalog)
            tidmask = np.isin(np.asarray(catalog["TARGETID"]), ext_tids)
            catalog = catalog[tidmask]
            log.info(
                f"external_tid_list={args.external_tid_list}: restricting mock "
                f"catalog from {n_before} to {len(catalog)} spectra "
                f"({len(ext_tids)} unique TARGETIDs requested)"
            )
            if len(catalog) < 1:
                log.error("external_tid_list left 0 spectra in catalog; aborting")
                exit(1)
    else:
        # running in between healpix pixels: hpx_start - hpx_end
        catalog = read_catalog(args.qsocat, args.balmask, args.tilebased)

        if args.use_external_hpx_list:
            # read in healpix list
            all_hpxs = np.loadtxt(args.external_hpx_list).astype(int)

            log.info(
                f"running in between external healpix list: {args.external_hpx_list}; Total {len(all_hpxs)} pixels"
            )
            this_hpxs = all_hpxs[args.hpx_start : args.hpx_end]
            log.info(
                f"healpix pixels to process: from {this_hpxs[0]} to {this_hpxs[-1]}"
            )
        else:
            all_hpxs = np.unique(catalog["HPXPIXEL"])
            log.info(
                "running in between healpix pixels {} - {}; Total {}".format(
                    args.hpx_start, args.hpx_end, len(all_hpxs)
                )
            )

            # TODO: So hxps are ALSO discontinuous, so it might make more sense to just indexing the catalog
            # ind = (all_hpxs >= args.hpx_start) & (all_hpxs < args.hpx_end)
            this_hpxs = all_hpxs.data[args.hpx_start : args.hpx_end]

    # Convert Parameters to a dictionary
    params_dict = {
        "loading_min_lambda": args.loading_min_lambda,
        "loading_max_lambda": args.loading_max_lambda,
        "normalization_min_lambda": args.normalization_min_lambda,
        "normalization_max_lambda": args.normalization_max_lambda,
        "min_lambda": args.min_lambda,
        "max_lambda": args.max_lambda,
        "dlambda": args.dlambda,
        "k": args.k,
        "max_noise_variance": args.max_noise_variance,
        "max_z_cut": args.max_z_cut,
        "min_z_cut": args.min_z_cut,
        "num_forest_lines" : args.num_forest_lines, # 3, # Number of forest lines to model
        "num_lines": args.num_lines, # 3,  # number of members of the Lyman series to use
        "num_dla_samples": args.num_dla_samples, # 10000,  # Number of DLA samples to generate
    }
    params_subdla_dict = {
        "loading_min_lambda": args.loading_min_lambda,
        "loading_max_lambda": args.loading_max_lambda,
        "normalization_min_lambda": args.normalization_min_lambda,
        "normalization_max_lambda": args.normalization_max_lambda,
        "min_lambda": args.min_lambda,
        "max_lambda": args.max_lambda,
        "dlambda": args.dlambda,
        "k": args.k,
        "max_noise_variance": args.max_noise_variance,
        "max_z_cut": args.max_z_cut,
        "min_z_cut": args.min_z_cut,
        "num_forest_lines" : args.num_forest_lines, # 3, # Number of forest lines to model
        "num_lines": args.num_lines, # 3,  # number of members of the Lyman series to use
        "num_dla_samples": args.num_subdla_samples, # 10000,  # Number of DLA samples to generate
    }

    # Convert DLAHolder to a dictionary
    model_params = {
        "learned_file": args.learned_file,
        "catalog_name": args.catalog_name,
        "los_catalog": args.los_catalog,
        "dla_catalog": args.dla_catalog,
        "dla_samples_file": args.dla_samples_file,
        "sub_dla_samples_file": args.sub_dla_samples_file,
        "params_dict": params_dict,  # Pass the Parameters dictionary instead of the instance
        "params_subdla_dict": params_subdla_dict,  # Pass the Sub-DLA Parameters dictionary instead of the instance
        "min_z_separation": args.min_z_separation,
        "prev_tau_0": args.prev_tau_0,
        "prev_beta": args.prev_beta,
        "max_dlas": args.max_dlas,
        "plot_figures": bool(args.plot_figures),
        "max_workers": args.max_workers,
        "batch_size": args.batch_size,
        "figure_dir": args.figure_dir,
        "filter_low_likelihood": bool(args.filter_low_likelihood),
        "filter_n_initial_floor": int(args.filter_n_initial_floor),
        "filter_empty_mask_fallthrough": bool(args.filter_empty_mask_fallthrough),
        "single_absorber_model": bool(args.single_absorber_model),  # single absorber model only
        "enable_tau_eb": bool(args.enable_tau_eb),
        "tau_eb_factors": tuple(args.tau_eb_factors),
        "tau_eb_apply_hcd_mask": bool(args.tau_eb_apply_hcd_mask),
        "tau_eb_mask_threshold_sigma": float(args.tau_eb_mask_threshold_sigma),
        "tau_eb_objective": args.tau_eb_objective,
        "early_stop_mode": args.early_stop_mode,
        "pair_prior_mode": args.pair_prior_mode,
        "dla_bias": args.dla_bias,
    }

    if args.pair_prior_mode == "clustering":
        log.warning(
            "pair_prior_mode=clustering: mock-calibrated prior (b_DLA=%.2f). On REAL data this "
            "fixes b and is over-confident; propagate b-uncertainty and DO NOT use this catalog "
            "for DLA clustering/bias measurements (spec section 3 / C-1).", args.dla_bias)

    # Set up for nested multiprocessing
    # nproc_futures = int(os.cpu_count() / args.max_workers)
    nproc_futures = 1
    log.info(f"using {nproc_futures} high-level processes")

    if not (args.tilebased) and not (args.mocks):
        datapath = f"/global/cfs/cdirs/desi/spectro/redux/{args.release}/healpix/{args.survey}/{args.program}"

        if nproc_futures == 1:
            results = [
                dlasearch.dlasearch_hpx(
                    hpx,
                    args.survey,
                    args.program,
                    datapath,
                    catalog[catalog["HPXPIXEL"] == hpx],
                    model_params,  # Pass the model parameters dictionary here
                )
                for hpx in this_hpxs
            ]

        else:
            arguments = [
                {
                    "healpix": hpx,
                    "survey": args.survey,
                    "program": args.program,
                    "datapath": datapath,
                    "hpxcat": catalog[catalog["HPXPIXEL"] == hpx],
                    "model_params": model_params,
                }
                for hpx in this_hpxs
            ]
            # Create a high-level executor
            with ProcessPoolExecutor(max_workers=nproc_futures) as high_level_executor:
                results = list(high_level_executor.map(_dlasearchhpx, arguments))

    elif args.mocks:
        if nproc_futures == 1:
            results = [
                dlasearch.dlasearch_mock(specfile, catalog, model_params)
                for specfile in speclist
            ]
        else:
            arguments = [
                {
                    "specfile": specfile,
                    "catalog": catalog,
                    "model_params": model_params,
                }
                for specfile in speclist
            ]
            # Create a high-level executor
            with ProcessPoolExecutor(max_workers=nproc_futures) as high_level_executor:
                results = list(high_level_executor.map(_dlasearchmock, arguments))

    results = vstack(results)
    results.meta["EXTNAME"] = "DLACAT"

    # remove extra column from hpx with no detections
    if "col0" in results.columns:
        results.remove_column("col0")

    # filename for output include release, survey, program and healpix range
    if not (args.tilebased) and not (args.mocks):
        outfile = os.path.join(
            args.outdir,
            f"dlacat-{args.release}-{args.survey}-{args.program}-hpx-{args.hpx_start}-{args.hpx_end}.fits",
        )
        if os.path.isfile(outfile):
            log.warning(
                f"dlacat-{args.release}-{args.survey}-{args.program}-hpx-{args.hpx_start}-{args.hpx_end}.fits already exists in {args.outdir}, overwriting"
            )
        results.write(outfile, overwrite=True)

    elif args.mocks:
        # filename for output include release, survey, program and folder range
        outfile = os.path.join(
            args.outdir,
            f"dlacat-{args.release}-mockcat-{args.level2_start}-{args.level2_end}.fits",
        )
        if os.path.isfile(outfile):
            log.warning(
                f"dlacat-{args.release}-mockcat-{args.level2_start}-{args.level2_end}.fits already exists in {args.outdir}, overwriting"
            )
        results.write(outfile, overwrite=True)

    tfin = time.time()
    total_time = tfin - tini

    print(f"total run time: {np.round(total_time/60,1)} minutes")


def read_catalog(qsocat, balmask, bytile):
    """
    read quasar catalog

    Arguments
    ---------
    qsocat (str) : path to quasar catalog
    balmask (bool) : should BAL attributes from baltools be read in?
    bytile (bool) : catalog is tilebased, default assumption is healpix

    Returns
    -------
    table of relevant attributes for quasars defined in constants.py

    """
    if constants.no_bal:
        balmask = True

    if balmask:
        try:
            # read the following columns from qsocat
            cols = [
                "TARGETID",
                "TARGET_RA",
                "TARGET_DEC",
                "Z",
                "HPXPIXEL",
                "AI_CIV",
                "NCIV_450",
                "VMIN_CIV_450",
                "VMAX_CIV_450",
                "SPECTYPE",
                "ZWARN",
            ]
            if bytile:
                cols = [
                    "TARGETID",
                    "TARGET_RA",
                    "TARGET_DEC",
                    "Z",
                    "TILEID",
                    "PETAL_LOC",
                    "AI_CIV",
                    "NCIV_450",
                    "VMIN_CIV_450",
                    "VMAX_CIV_450",
                    "SPECTYPE",
                    "ZWARN",
                ]
            catalog = Table(fitsio.read(qsocat, ext=1, columns=cols))
        except:
            log.error(f"cannot find {cols} in quasar catalog")
            exit(1)
    else:
        # read the following columns from qsocat
        cols = [
            "TARGETID",
            "TARGET_RA",
            "TARGET_DEC",
            "Z",
            "HPXPIXEL",
            "SPECTYPE",
            "ZWARN",
        ]
        if bytile:
            cols = [
                "TARGETID",
                "TARGET_RA",
                "TARGET_DEC",
                "Z",
                "TILEID",
                "PETAL_LOC",
                "SPECTYPE",
                "ZWARN",
            ]
        catalog = Table(fitsio.read(qsocat, ext=1, columns=cols))

    log.info(f"Successfully read quasar catalog: {qsocat}")

    # Apply redshift cuts
    zmask = (catalog["Z"] > constants.zmin_qso) & (catalog["Z"] < constants.zmax_qso)
    log.info(f"objects in catalog: {len(catalog)} ")
    log.info(
        f"restricting to {constants.zmin_qso} < z < {constants.zmax_qso}: {np.sum(zmask)} objects remain"
    )

    # Apply bal mask
    if constants.no_bal:
        balind = catalog["NCIV_450"] > 0
        zmask = zmask & ~balind
        log.info(f"objects in catalog without BAL: {np.sum(zmask)}")

    # Apply zwarning mask
    if constants.zwarning:
        zmask = zmask & (catalog["ZWARN"] == 0)
        log.info(f"objects in catalog without ZWARN: {np.sum(zmask)}")

    # Apply spectype mask
    if constants.is_qso:
        zmask = zmask & (catalog["SPECTYPE"] == "QSO")
        log.info(f"objects in catalog with SPECTYPE QSO: {np.sum(zmask)}")

    catalog = catalog[zmask]

    return catalog


def read_mock_catalog(qsocat, balmask, mockpath):
    """
    read quasar catalog

    Arguments
    ---------
    qsocat (str) : path to quasar catalog
    balmask (bool) : should BAL attributes be read in?
    mockpath (str) : path to mock data

    Returns
    -------
    table of relevant attributes for z>2 quasars

    """
    # read the following columns from qsocat
    cols = ["TARGETID", "RA", "DEC", "Z"]
    try:
        catalog = Table(fitsio.read(qsocat, ext=1, columns=cols))
    except:
        print("[Warning] cannot find TARGETID, RA, DEC, Z in quasar catalog")
        print("... using Saclay cols instead.")
        cols = ['TARGETID', 'TARGET_RA', 'TARGET_DEC', 'Z']
        catalog = Table(fitsio.read(qsocat, ext=1, columns=cols))

    log.info(f"Successfully read mock quasar catalog: {qsocat}")

    # Apply redshift cuts
    zmask = (catalog["Z"] > constants.zmin_qso) & (catalog["Z"] < constants.zmax_qso)
    log.info(f"objects in catalog: {len(catalog)} ")
    log.info(
        f"restricting to {constants.zmin_qso} < z < {constants.zmax_qso}: {np.sum(zmask)} objects remain"
    )

    catalog = catalog[zmask]

    if balmask:
        try:
            # open bal catalog
            balcat = os.path.join(mockpath, "bal_cat.fits")
            cols = ["TARGETID", "AI_CIV", "NCIV_450", "VMIN_CIV_450", "VMAX_CIV_450"]
            balcat = Table(fitsio.read(balcat, ext=1, columns=cols))

            # add columns to catalog
            ai = np.full(len(catalog), 0.0)
            nciv = np.full(len(catalog), 0)
            vmin = np.full((len(catalog), balcat["VMIN_CIV_450"].shape[1]), -1.0)
            vmax = np.full((len(catalog), balcat["VMIN_CIV_450"].shape[1]), -1.0)

            for i, tid in enumerate(catalog["TARGETID"]):
                if np.any(tid == balcat["TARGETID"]):
                    match = balcat[balcat["TARGETID"] == tid]
                    ai[i] = match["AI_CIV"]
                    nciv[i] = match["NCIV_450"]
                    vmin[i] = match["VMIN_CIV_450"]
                    vmax[i] = match["VMAX_CIV_450"]

            catalog.add_columns(
                [ai, nciv, vmin, vmax],
                names=["AI_CIV", "NCIV_450", "VMIN_CIV_450", "VMAX_CIV_450"],
            )

        except:
            log.error(f"cannot find mock bal_cat.fits in {mockpath}")
            exit(1)

    return catalog


# for parallelization over hpx
def _dlasearchhpx(arguments):
    return dlasearch.dlasearch_hpx(**arguments)


# for parallelization over mock spectra files
def _dlasearchmock(arguments):
    return dlasearch.dlasearch_mock(**arguments)


if __name__ == "__main__":
    main()
