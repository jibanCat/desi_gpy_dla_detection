"""Sharper τ_eff test: scan log L on a fine NHI grid at fixed truth z,
varying τ_factor, to see if the argmax NHI shifts continuously."""
import sys, os, time
sys.path.insert(0, "/home/mfho/desi_gpy_dla_detection")
import numpy as np

from gpy_dla_detection.voigt_v2_inject import inject
inject(kernel="boss-log-r2000", num_lines=3)

from examples.smoke_one_spectrum import load_one_desi_spectrum, lookup_z_qso, PRESETS
from gpy_dla_detection.set_parameters import Parameters
from gpy_dla_detection.dla_gp import DLAGPMAT
from gpy_dla_detection.model_priors import PriorCatalog
from gpy_dla_detection.dla_samples import DLASamplesMAT

spec = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/spectra-16/7/789/spectra-16-789.fits"
zcat = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/zcat.fits"
tid = 120046865
truth_z, truth_n = 2.7730, 21.263

wave, flux, nv, mask = load_one_desi_spectrum(spec, tid)
z_qso = lookup_z_qso(zcat, tid)

preset = PRESETS["y3"]
data_root = "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection"
common = dict(
    loading_min_lambda=preset.loading_min_lambda, loading_max_lambda=preset.loading_max_lambda,
    normalization_min_lambda=preset.normalization_min_lambda, normalization_max_lambda=preset.normalization_max_lambda,
    min_lambda=preset.min_lambda, max_lambda=preset.max_lambda,
    dlambda=preset.dlambda, k=preset.k,
    max_noise_variance=9.0, num_lines=3, max_z_cut=3000.0, min_z_cut=3000.0,
    num_forest_lines=preset.num_forest_lines,
)
params = Parameters(num_dla_samples=100000, **common)
prior = PriorCatalog(params,
    os.path.join(data_root, "data/dr12q/processed/catalog.mat"),
    os.path.join(data_root, "data/dla_catalogs/dr9q_concordance/processed/los_catalog"),
    os.path.join(data_root, "data/dla_catalogs/dr9q_concordance/processed/dla_catalog"))
dla_samples = DLASamplesMAT(params, prior,
    os.path.join(data_root, "data/dr12q/processed/dla_samples_a03_100000.mat"))

# fine continuous NHI grid at fixed z = truth z
nhi_grid = np.arange(20.30, 22.01, 0.025)
z_grid = np.full_like(nhi_grid, truth_z)
print(f"τ_eff fine-NHI test on TID {tid}")
print(f"  z fixed at truth = {truth_z}, NHI grid: {nhi_grid[0]:.2f} → {nhi_grid[-1]:.2f} step 0.025 ({len(nhi_grid)} pts)")
print()
print(f"{'τ_factor':>9} {'τ_0':>8} {'argmax NHI':>11} {'log L(argmax)':>14} {'log L(truth=21.263)':>20}")
print("-"*70)

learned = os.path.join(data_root, preset.learned_file)
for tau_factor in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
    prev_tau_0 = preset.prev_tau_0 * tau_factor
    dla_gp = DLAGPMAT(params, prior, dla_samples,
        min_z_separation=3000.0, learned_file=learned,
        broadening=True, prev_tau_0=prev_tau_0, prev_beta=preset.prev_beta)
    rest_w = params.emitted_wavelengths(wave, z_qso)
    dla_gp.set_data(np.atleast_2d(rest_w), np.atleast_2d(flux),
                    np.atleast_2d(nv), np.atleast_2d(mask),
                    np.array([z_qso]), build_model=True)
    log_l = np.full(len(nhi_grid), np.nan)
    for i, ln in enumerate(nhi_grid):
        try:
            log_l[i] = dla_gp.sample_log_likelihood_k_dlas(
                np.array([truth_z]), np.array([10**ln]))
        except Exception:
            pass
    argmax_idx = int(np.nanargmax(log_l))
    L_truth = float(np.interp(truth_n, nhi_grid, log_l))
    print(f"{tau_factor:9.2f} {prev_tau_0:8.5f} {nhi_grid[argmax_idx]:11.3f} "
          f"{log_l[argmax_idx]:14.2f} {L_truth:20.2f}")
