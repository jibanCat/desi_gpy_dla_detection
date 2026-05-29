"""
examples/demo_desi_spectrum.py
==============================
Minimal reproducible demo: GP-DLA detection on DESI-format spectra.

This script demonstrates end-to-end single-spectrum DLA detection using:
  - London mock DESI spectra  (eBOSS-0.0.0 format, included in the repo)
  - A learned QSO model trained on SDSS DR16Q (eBOSS)

It is a cleaned-up version of ``notebooks/Demo DESI Spectra-Copy1.ipynb``,
adapted to use local data paths instead of NERSC paths.

Data required (relative to repo root)
--------------------------------------
  data/desi/eboss-0.0.0/spectra-16/7/705/spectra-16-705.fits
  data/desi/eboss-0.0.0/zcat.fits
  data/dr12q/processed/learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat
  data/dr12q/processed/catalog.mat
  data/dla_catalogs/dr9q_concordance/processed/los_catalog
  data/dla_catalogs/dr9q_concordance/processed/dla_catalog
  data/dr12q/processed/dla_samples_a03.mat
  data/dr12q/processed/subdla_samples.mat

Usage
-----
    cd <repo_root>
    python examples/demo_desi_spectrum.py

    # Process only N spectra (faster for testing):
    python examples/demo_desi_spectrum.py --num-spectra 3

    # Save results:
    python examples/demo_desi_spectrum.py --output results_demo.h5

    # Plot diagnostic figures:
    python examples/demo_desi_spectrum.py --plot --figure-dir figures/

Output
------
For each spectrum the script prints:
  - Target ID and z_QSO
  - p(DLA): posterior probability of at least one DLA
  - p(null): posterior probability of no absorber
  - MAP z_DLA and log N_HI (if DLA detected)
  - model_posteriors vector [Null, SubDLA, DLA(1), DLA(2), DLA(3)]

Notes
-----
- The eBOSS DR16Q model (loading_min_lambda=911, max_lambda=1420 Å) is
  used here as a substitute for the DESI-trained model.  For production
  inference, replace ``LEARNED_FILE`` with the DESI Y3 trained model.
- Parameters are set for the eBOSS wavelength range.  For a DESI-trained
  model, adjust ``dlambda``, ``k``, and the wavelength range accordingly.
- This demo uses ``max_workers=4`` to avoid exhausting RAM on a laptop.
  On a cluster, use the default (None = all CPUs).
"""

import argparse
import os
import sys

import fitsio
import numpy as np

# ---------------------------------------------------------------------------
# Locate repo root
# The script walks upward from examples/ to find a directory that contains
# "data/dr12q/processed", which is the canonical data directory.  This
# handles git worktrees, where the worktree root differs from the main repo.
# ---------------------------------------------------------------------------
def _find_repo_root(start: str) -> str:
    candidate = os.path.dirname(start)
    for _ in range(6):  # walk at most 6 levels up
        if os.path.isdir(os.path.join(candidate, "data", "dr12q", "processed")):
            return candidate
        candidate = os.path.dirname(candidate)
    # Fallback: two levels up from examples/
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPO_ROOT = _find_repo_root(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Data file paths (relative to repo root)
# ---------------------------------------------------------------------------
SPECTRA_FILE = os.path.join(
    REPO_ROOT,
    "data/desi/eboss-0.0.0/spectra-16/7/705/spectra-16-705.fits",
)
ZCAT_FILE = os.path.join(
    REPO_ROOT,
    "data/desi/eboss-0.0.0/zcat.fits",
)
LEARNED_FILE = os.path.join(
    REPO_ROOT,
    "data/dr12q/processed/"
    "learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat",
)
CATALOG_FILE   = os.path.join(REPO_ROOT, "data/dr12q/processed/catalog.mat")
LOS_CATALOG    = os.path.join(REPO_ROOT, "data/dla_catalogs/dr9q_concordance/processed/los_catalog")
DLA_CATALOG    = os.path.join(REPO_ROOT, "data/dla_catalogs/dr9q_concordance/processed/dla_catalog")
DLA_SAMPLES    = os.path.join(REPO_ROOT, "data/dr12q/processed/dla_samples_a03.mat")
SUBDLA_SAMPLES = os.path.join(REPO_ROOT, "data/dr12q/processed/subdla_samples.mat")

# ---------------------------------------------------------------------------
# Default parameters (eBOSS DR16Q model range)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = dict(
    loading_min_lambda=910.0,
    loading_max_lambda=1550.0,
    normalization_min_lambda=1425.0,
    normalization_max_lambda=1475.0,
    min_lambda=910.75,
    max_lambda=1216.75,
    dlambda=0.25,
    k=20,
    max_noise_variance=9.0,
    num_dla_samples=10_000,
    num_lines=3,
    max_z_cut=3000.0,
    min_z_cut=3000.0,
    num_forest_lines=3,
)

# IGM mean-flux parameters (Kim+2007, arXiv:0711.1862; legacy default for eBOSS
# model). NOT Kamble+2020 (that is 0.00554/3.182).
PREV_TAU_0 = 0.0023
PREV_BETA  = 3.65

# Minimum redshift separation between two DLAs (km/s)
MIN_Z_SEP = 3000.0


def parse_args():
    p = argparse.ArgumentParser(
        description="GP-DLA detection demo on London mock DESI spectra."
    )
    p.add_argument(
        "--num-spectra", type=int, default=5,
        help="Number of spectra to process (default 5).",
    )
    p.add_argument(
        "--max-dlas", type=int, default=3,
        help="Maximum number of DLAs per spectrum (default 3).",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="If given, save results to this HDF5 file.",
    )
    p.add_argument(
        "--plot", action="store_true",
        help="Generate diagnostic plots.",
    )
    p.add_argument(
        "--figure-dir", type=str, default="figures/",
        help="Directory for diagnostic plots (default figures/).",
    )
    p.add_argument(
        "--max-workers", type=int, default=4,
        help="Max parallel workers for evidence computation (default 4).",
    )
    p.add_argument(
        "--batch-size", type=int, default=100,
        help="Batch size for evidence computation (default 100).",
    )
    return p.parse_args()


def check_data_files():
    """Verify all required data files exist before loading heavy modules."""
    missing = []
    for path in [SPECTRA_FILE, ZCAT_FILE, LEARNED_FILE, CATALOG_FILE,
                 DLA_SAMPLES, SUBDLA_SAMPLES]:
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        print("ERROR: Required data files not found:")
        for p in missing:
            print(f"  {p}")
        print(
            "\nTo download the London mock data, run:\n"
            "  bash data/scripts/download_mocks.sh\n"
            "For the SDSS GP model, run:\n"
            "  bash data/scripts/download_gp_files.sh"
        )
        sys.exit(1)


def load_spectra_and_zcat(num_spectra: int):
    """
    Load DESI-format spectra and the corresponding redshift catalog.

    Returns a list of (target_id, z_qso, wavelengths, flux, noise_var, mask) tuples.
    Skips spectra with bad redshift warnings or z < 2.

    Uses desispec.io.read_spectra for proper band-combination.
    """
    from desispec.io import read_spectra
    from desispec.coaddition import coadd_cameras

    # Read z catalog
    zcat = fitsio.read(ZCAT_FILE, ext="ZCATALOG")
    z_by_tid = {int(row["TARGETID"]): float(row["Z"]) for row in zcat if row["ZWARN"] == 0}

    # Read spectra (all targets in the file)
    spectra = read_spectra(SPECTRA_FILE)
    try:
        spectra = coadd_cameras(spectra)
        band = "brz"
    except Exception:
        band = "b"  # fallback if coadd fails

    target_ids = spectra.fibermap["TARGETID"]
    wavelengths_all = spectra.wave[band]

    results = []
    for i, tid in enumerate(target_ids):
        tid = int(tid)
        if tid not in z_by_tid:
            continue
        z_qso = z_by_tid[tid]
        if z_qso < 2.0:
            continue

        flux = spectra.flux[band][i].astype(np.float64)
        ivar = spectra.ivar[band][i].astype(np.float64)
        mask = spectra.mask[band][i].astype(bool)

        # Convert inverse variance to noise variance
        noise_var = np.full_like(flux, np.inf)
        good = ivar > 0
        noise_var[good] = 1.0 / ivar[good]

        results.append((tid, z_qso, wavelengths_all.copy(), flux, noise_var, mask))
        if len(results) >= num_spectra:
            break

    if not results:
        print("No spectra with valid redshifts found. Check data files.")
        sys.exit(1)

    print(f"Loaded {len(results)} spectra from {os.path.basename(SPECTRA_FILE)}")
    return results


def main():
    args = parse_args()

    # Check data before importing heavy modules
    check_data_files()

    # Import GP-DLA modules (after path check to give clear errors)
    sys.path.insert(0, REPO_ROOT)
    from gpy_dla_detection.set_parameters import Parameters
    from run_bayes_select import DLAHolder

    # ---------------------------------------------------------------------------
    # Build Parameters
    # ---------------------------------------------------------------------------
    params = Parameters(
        loading_min_lambda=DEFAULT_PARAMS["loading_min_lambda"],
        loading_max_lambda=DEFAULT_PARAMS["loading_max_lambda"],
        normalization_min_lambda=DEFAULT_PARAMS["normalization_min_lambda"],
        normalization_max_lambda=DEFAULT_PARAMS["normalization_max_lambda"],
        min_lambda=DEFAULT_PARAMS["min_lambda"],
        max_lambda=DEFAULT_PARAMS["max_lambda"],
        dlambda=DEFAULT_PARAMS["dlambda"],
        k=DEFAULT_PARAMS["k"],
        max_noise_variance=DEFAULT_PARAMS["max_noise_variance"],
        num_dla_samples=DEFAULT_PARAMS["num_dla_samples"],
        num_lines=DEFAULT_PARAMS["num_lines"],
        max_z_cut=DEFAULT_PARAMS["max_z_cut"],
        min_z_cut=DEFAULT_PARAMS["min_z_cut"],
        num_forest_lines=DEFAULT_PARAMS["num_forest_lines"],
    )

    # ---------------------------------------------------------------------------
    # Load spectra
    # ---------------------------------------------------------------------------
    spectra = load_spectra_and_zcat(args.num_spectra)

    # ---------------------------------------------------------------------------
    # Initialize DLAHolder (loads GP model, priors, sample grids once)
    # ---------------------------------------------------------------------------
    print(f"\nInitializing DLA detection pipeline...")
    print(f"  Learned model : {os.path.basename(LEARNED_FILE)}")
    print(f"  DLA samples   : {os.path.basename(DLA_SAMPLES)}")
    print(f"  Run mode      : multi-DLA (max {args.max_dlas} DLAs per spectrum)")

    holder = DLAHolder(
        learned_file=LEARNED_FILE,
        catalog_name=CATALOG_FILE,
        los_catalog=LOS_CATALOG,
        dla_catalog=DLA_CATALOG,
        dla_samples_file=DLA_SAMPLES,
        sub_dla_samples_file=SUBDLA_SAMPLES,
        params=params,
        min_z_separation=MIN_Z_SEP,
        prev_tau_0=PREV_TAU_0,
        prev_beta=PREV_BETA,
        max_dlas=args.max_dlas,
        broadening=True,
        plot_figures=args.plot,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        figure_dir=args.figure_dir,
        single_absorber_model=False,  # DLA run: includes SubDLA model
    )
    holder.initialize_results(len(spectra))

    # ---------------------------------------------------------------------------
    # Process spectra
    # ---------------------------------------------------------------------------
    print(f"\nProcessing {len(spectra)} spectrum/spectra...\n")
    for idx, (tid, z_qso, wavelengths, flux, noise_var, mask) in enumerate(spectra):
        holder.process_qso(
            idx=idx,
            target_id=str(tid),
            wavelengths=wavelengths,
            flux=flux,
            noise_variance=noise_var,
            pixel_mask=mask,
            z_qso=z_qso,
        )

    # ---------------------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------------------
    res = holder.results
    print("\n" + "=" * 70)
    print(f"{'TARGETID':>15}  {'z_QSO':>6}  {'p(DLA)':>8}  {'p(null)':>8}  "
          f"{'MAP_z_DLA':>10}  {'MAP_logNHI':>11}")
    print("-" * 70)
    for i in range(len(spectra)):
        tid, z_qso = spectra[i][0], spectra[i][1]
        p_dla   = res["p_dlas"][i]
        p_null  = res["p_no_dlas"][i]
        z_dla   = res["MAP_z_dlas"][i, 0]
        log_nhi = res["MAP_log_nhis"][i, 0]

        z_str   = f"{z_dla:.3f}" if not np.isnan(z_dla) else "---"
        nhi_str = f"{log_nhi:.2f}" if not np.isnan(log_nhi) else "---"

        print(f"{tid:>15}  {z_qso:>6.3f}  {p_dla:>8.3f}  {p_null:>8.3f}  "
              f"{z_str:>10}  {nhi_str:>11}")
    print("=" * 70)

    # Print model_posteriors layout reminder
    print("\nmodel_posteriors column layout (multi-DLA run):")
    print("  [0] Null  [1] SubDLA  [2] 1-DLA  [3] 2-DLA  [4] 3-DLA")
    for i in range(len(spectra)):
        tid = spectra[i][0]
        mp = res["model_posteriors"][i]
        vals = "  ".join(f"{v:.3f}" for v in mp)
        print(f"  {tid}: [{vals}]")

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    if args.output:
        holder.save_results(args.output)
        print(f"\nResults saved to {args.output}")
        print("HDF5 keys available:")
        import h5py
        with h5py.File(args.output, "r") as hf:
            for k in hf.keys():
                print(f"  {k}: shape={hf[k].shape}, dtype={hf[k].dtype}")


if __name__ == "__main__":
    main()
