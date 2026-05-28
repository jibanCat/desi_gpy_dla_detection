"""Reproduce the FULL canonical-TID inference with the c0prior model — same
DLAHolder configuration as the original recovery test — and capture the
exact p_DLA, posterior columns, and any NaN trace.

If this reproduces p_DLA = 0.042 with NaN 2-DLA posteriors, the failure is
deterministic. If it produces a NORMAL p_DLA ~ 0.8 (matching the
reproducible single-sample probe), the original recovery test had a
transient state issue (e.g., a different code branch, or a stale model
file).
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np

REPO = Path('/home/mfho/desi_gpy_dla_detection')
sys.path.insert(0, str(REPO))

TID = 120046865
TRUTH_NHI = 21.263
SPEC = '/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/spectra-16/7/789/spectra-16-789.fits'
ZCAT = '/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/zcat.fits'
DATA_ROOT = '/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection'

MODEL = str(REPO / 'docs/notes/2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_c0prior/phase2_result.h5')

from examples.smoke_one_spectrum import load_one_desi_spectrum, lookup_z_qso, PRESETS
from gpy_dla_detection.set_parameters import Parameters
from run_bayes_select import DLAHolder

preset = PRESETS['y3']
wave, flux, nv, mask = load_one_desi_spectrum(SPEC, TID)
z_qso = lookup_z_qso(ZCAT, TID)

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
t0 = time.time()
holder.process_qso(idx=0, target_id=str(TID),
                   wavelengths=wave, flux=flux,
                   noise_variance=nv, pixel_mask=mask, z_qso=z_qso)
print(f'\nelapsed: {time.time() - t0:.1f}s')
print('p_dla =', float(holder.results['p_dlas'][0]))
print('MAP z =', float(holder.results['MAP_z_dlas'][0, 0]))
print('MAP log_NHI =', float(holder.results['MAP_log_nhis'][0, 0]))
post = [float(x) for x in holder.results['model_posteriors'][0]]
print('posteriors =', post)
