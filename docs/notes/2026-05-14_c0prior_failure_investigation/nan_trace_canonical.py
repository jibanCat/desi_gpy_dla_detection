"""Instrument the DLA likelihood path for the canonical TID 120046865 with
the c0prior model. Where does NaN first appear in the multi-DLA evidence?

This script:
  1. Loads the canonical spectrum and the c0prior model.
  2. Runs through process_qso to populate the GP state.
  3. Then directly calls dla_gp.sample_log_likelihood_k_dlas for k=1,2,3,4
     at the MAP location, and inspects:
       - the absorbed mu, M, omega² (any NaNs?)
       - the Woodbury B matrix and its Cholesky
       - the log-determinant pieces
       - the final log-likelihood
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

REPO = Path('/home/mfho/desi_gpy_dla_detection')
sys.path.insert(0, str(REPO))

import scipy.linalg.lapack as lapack

# --- load model and target -------------------------------------------------
TID = 120046865
TRUTH_NHI = 21.263
SPEC = '/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/spectra-16/7/789/spectra-16-789.fits'
ZCAT = '/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/zcat.fits'
MODEL = str(REPO / 'docs/notes/2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_c0prior/phase2_result.h5')
DATA_ROOT = '/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection'

from examples.smoke_one_spectrum import load_one_desi_spectrum, lookup_z_qso, PRESETS
from gpy_dla_detection.set_parameters import Parameters
from run_bayes_select import DLAHolder

preset = PRESETS['y3']
wave, flux, nv, mask = load_one_desi_spectrum(SPEC, TID)
z_qso = lookup_z_qso(ZCAT, TID)
print(f'spectrum: {len(wave)} pixels, z_qso={z_qso:.4f}')

common = dict(
    loading_min_lambda=preset.loading_min_lambda,
    loading_max_lambda=preset.loading_max_lambda,
    normalization_min_lambda=preset.normalization_min_lambda,
    normalization_max_lambda=preset.normalization_max_lambda,
    min_lambda=preset.min_lambda, max_lambda=preset.max_lambda,
    dlambda=preset.dlambda, k=30,
    max_noise_variance=9.0, num_lines=3,
    max_z_cut=3000.0, min_z_cut=3000.0,
    num_forest_lines=preset.num_forest_lines,
)
params = Parameters(num_dla_samples=100000, **common)
params_subdla = Parameters(num_dla_samples=100000, **common)
holder = DLAHolder(
    learned_file=MODEL,
    catalog_name=os.path.join(DATA_ROOT, 'data/dr12q/processed/catalog.mat'),
    los_catalog=os.path.join(DATA_ROOT, 'data/dla_catalogs/dr9q_concordance/processed/los_catalog'),
    dla_catalog=os.path.join(DATA_ROOT, 'data/dla_catalogs/dr9q_concordance/processed/dla_catalog'),
    dla_samples_file=os.path.join(DATA_ROOT, 'data/dr12q/processed/dla_samples_a03_100000.mat'),
    sub_dla_samples_file=os.path.join(DATA_ROOT, 'data/dr12q/processed/subdla_samples_a03_191_200_100000.mat'),
    params=params, params_subdla=params_subdla,
    min_z_separation=3000.0,
    prev_tau_0=preset.prev_tau_0, prev_beta=preset.prev_beta,
    max_dlas=4, broadening=True,
    plot_figures=False, max_workers=8, batch_size=12500,
    figure_dir='/tmp',
    single_absorber_model=False,
    filter_low_likelihood=True,
)
holder.initialize_results(1)
holder.process_qso(idx=0, target_id=str(TID),
                   wavelengths=wave, flux=flux,
                   noise_variance=nv, pixel_mask=mask, z_qso=z_qso)
print('post-process posteriors:', holder.results['model_posteriors'][0])
print('sample_log_likelihoods shape:', holder.dla_gp.sample_log_likelihoods.shape)
sll = holder.dla_gp.sample_log_likelihoods
for k in range(sll.shape[1]):
    col = sll[:, k]
    n_nan = np.isnan(col).sum()
    print(f'  k={k+1}: n_nan={n_nan}/{len(col)}  max={np.nanmax(col) if n_nan < len(col) else float("nan"):.3f}')

# Now manually probe with a single (z_dla, log_nhi) at truth, and progressively at 2 DLAs etc.
dla_gp = holder.dla_gp
print('\n--- Manual probe at truth (z=2.7748, log_NHI=21.26) ---')
z_arr = np.array([2.7748])
nhi_arr = np.array([10**21.26])
ll = dla_gp.sample_log_likelihood_k_dlas(z_arr, nhi_arr)
print(f'  k=1 at truth: log_L = {ll:.4f}')

# Inspect the absorbed model and the K-matrix Cholesky
dla_mu, dla_M, dla_omega2 = dla_gp.this_dla_gp(z_arr, nhi_arr)
v = dla_gp.v
d = dla_omega2 + v
print(f'  n_pixels = {len(d)}')
print(f'  any NaN in dla_mu? {np.isnan(dla_mu).any()}')
print(f'  any NaN in dla_M?  {np.isnan(dla_M).any()}')
print(f'  any NaN in dla_omega2? {np.isnan(dla_omega2).any()}')
print(f'  d (diagonal) min={d.min():.6g} max={d.max():.6g}')
print(f'  any d <= 0? {(d <= 0).any()}')
print(f'  any d NaN? {np.isnan(d).any()}')

# Repeat for k=2 — pick a separated 2nd DLA
print('\n--- Manual probe at truth + spurious 2nd DLA ---')
z2 = np.array([2.7748, 2.6])
nhi2 = np.array([10**21.26, 10**20.5])
try:
    ll2 = dla_gp.sample_log_likelihood_k_dlas(z2, nhi2)
    print(f'  k=2: log_L = {ll2:.4f}')
except Exception as ex:
    print(f'  k=2 raised: {ex!r}')
dla_mu2, dla_M2, dla_omega2_2 = dla_gp.this_dla_gp(z2, nhi2)
d2 = dla_omega2_2 + v
print(f'  dla_omega2_2 min={dla_omega2_2.min():.6g} max={dla_omega2_2.max():.6g}')
print(f'  d2 min={d2.min():.6g} max={d2.max():.6g}')
print(f'  any d2 <= 0? {(d2 <= 0).any()}')
print(f'  any d2 NaN? {np.isnan(d2).any()}')

# Inspect the Woodbury B matrix
n, k = dla_M2.shape
y = dla_gp.y - dla_mu2
d_inv = 1 / d2
D_inv_M = d_inv[:, None] * dla_M2
B = dla_M2.T @ D_inv_M
B.ravel()[0::(k+1)] += 1
print(f'  B condition number = {np.linalg.cond(B):.4g}')
print(f'  B eigvals min={np.linalg.eigvalsh(B).min():.4g}  max={np.linalg.eigvalsh(B).max():.4g}')
try:
    L = np.linalg.cholesky(B)
    print(f'  Cholesky OK, diag(L) min={np.abs(np.diag(L)).min():.4g}  max={np.abs(np.diag(L)).max():.4g}')
except Exception as ex:
    print(f'  Cholesky FAILED: {ex!r}')

# Also look at the QMC-sampled posterior columns
print('\n--- QMC samples passed for k=2 and beyond? ---')
for kk in range(sll.shape[1]):
    col = sll[:, kk]
    n_nan = np.isnan(col).sum()
    print(f'  k={kk+1}: total={len(col)} n_nan={n_nan} valid={len(col)-n_nan}')
    if n_nan > 0 and n_nan < len(col):
        idx_nan = np.where(np.isnan(col))[0][:5]
        idx_ok = np.where(~np.isnan(col))[0][:5]
        print(f'     first-5 NaN indices: {idx_nan}')
        print(f'     first-5 valid indices: {idx_ok}')
