"""Run the initial-scan-equivalent diagnostic on the c0prior model for
TID 120046865, to understand why no samples cross null_evidence.

Recompute the first 5000 QMC samples' log-likelihoods manually, compare
to null_evidence, and look at the histogram of (logL - null) values
within ±0.05 z of the truth.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

REPO = Path('/home/mfho/desi_gpy_dla_detection')
sys.path.insert(0, str(REPO))

TID = 120046865
TRUTH_NHI = 21.263
TRUTH_Z = 2.7748
SPEC = '/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/spectra-16/7/789/spectra-16-789.fits'
ZCAT = '/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/zcat.fits'
DATA_ROOT = '/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection'
MODELS = {
    'c0prior': str(REPO / 'docs/notes/2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_c0prior/phase2_result.h5'),
    'm_baseline': str(REPO / 'docs/notes/2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_m/phase2_result.h5'),
}

from examples.smoke_one_spectrum import load_one_desi_spectrum, lookup_z_qso, PRESETS
from gpy_dla_detection.set_parameters import Parameters
from gpy_dla_detection.null_gp import NullGPMAT
from gpy_dla_detection.dla_gp import DLAGPMAT
from gpy_dla_detection.dla_samples import DLASamplesMAT
from gpy_dla_detection.model_priors import PriorCatalog

preset = PRESETS['y3']
wave, flux, nv, mask = load_one_desi_spectrum(SPEC, TID)
z_qso = lookup_z_qso(ZCAT, TID)
print(f'spectrum: {len(wave)} pixels, z_qso={z_qso:.4f}')

def build_params():
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
    return Parameters(num_dla_samples=100000, **common)


def diagnose(name, model_path):
    print(f'\n=== {name} ===')
    params = build_params()
    prior = PriorCatalog(
        params,
        catalog_name=os.path.join(DATA_ROOT, 'data/dr12q/processed/catalog.mat'),
        los_catalog=os.path.join(DATA_ROOT, 'data/dla_catalogs/dr9q_concordance/processed/los_catalog'),
        dla_catalog=os.path.join(DATA_ROOT, 'data/dla_catalogs/dr9q_concordance/processed/dla_catalog'),
    )
    dla_samples = DLASamplesMAT(params, prior,
        os.path.join(DATA_ROOT, 'data/dr12q/processed/dla_samples_a03_100000.mat'))
    null_gp = NullGPMAT(params, prior, learned_file=model_path,
                       prev_tau_0=preset.prev_tau_0, prev_beta=preset.prev_beta)
    dla_gp = DLAGPMAT(params, prior, dla_samples,
                      min_z_separation=3000.0,
                      learned_file=model_path,
                      broadening=True,
                      prev_tau_0=preset.prev_tau_0, prev_beta=preset.prev_beta)
    rest = wave / (1 + z_qso)
    null_gp.set_data(rest, flux, nv, mask, z_qso, normalize=True)
    dla_gp.set_data(rest, flux, nv, mask, z_qso, normalize=True)
    null_ev = null_gp.log_model_evidence()
    print(f'  null evidence = {null_ev:.4f}')
    # Sample z_dla
    sample_z = dla_samples.sample_z_dlas(dla_gp.this_wavelengths, z_qso)
    sample_lognhi = dla_samples.log_nhi_samples  # full set
    print(f'  sample_z range: [{sample_z.min():.4f}, {sample_z.max():.4f}], n={len(sample_z)}')
    print(f'  sample_lognhi range: [{sample_lognhi.min():.3f}, {sample_lognhi.max():.3f}], n={len(sample_lognhi)}')
    # Distance to truth in (z, log_NHI) — find closest sample
    d = (sample_z - TRUTH_Z)**2 + (sample_lognhi - TRUTH_NHI)**2 * 0.001  # scale logNHI to ~unitless
    closest = np.argmin(d)
    print(f'  closest sample: idx={closest}  z={sample_z[closest]:.4f}  logNHI={sample_lognhi[closest]:.3f}')
    # Run first 5000 samples
    n_init = 5000
    logL = np.empty(n_init)
    nhis = 10 ** sample_lognhi[:n_init]
    for i in range(n_init):
        logL[i] = dla_gp.sample_log_likelihood_k_dlas(sample_z[i:i+1], np.array([nhis[i]]))
    above_null = (logL > null_ev).sum()
    print(f'  initial 5000 samples: above null = {above_null}/{n_init}')
    print(f'  logL stats: min={logL.min():.3f} median={np.median(logL):.3f} max={logL.max():.3f}')
    delta = logL - null_ev
    print(f'  Δ-vs-null stats: min={delta.min():.3f} median={np.median(delta):.3f} max={delta.max():.3f}')
    # In what z window is logL > null?
    if above_null > 0:
        good_z = sample_z[:n_init][logL > null_ev]
        good_nhi = sample_lognhi[:n_init][logL > null_ev]
        good_logL = logL[logL > null_ev]
        print(f'  "good" samples z range: [{good_z.min():.3f}, {good_z.max():.3f}]')
        print(f'  "good" samples logNHI range: [{good_nhi.min():.3f}, {good_nhi.max():.3f}]')
        print(f'  "good" samples max logL = {good_logL.max():.4f}  (Δ = {(good_logL.max() - null_ev):+.3f})')
    # Look at samples near truth z
    near_truth = np.where(np.abs(sample_z[:n_init] - TRUTH_Z) < 0.05)[0]
    print(f'  samples within Δz=0.05 of truth: n={len(near_truth)}')
    if len(near_truth) > 0:
        near_logL = logL[near_truth]
        near_above = (near_logL > null_ev).sum()
        print(f'    of those, above null = {near_above}/{len(near_truth)}')
        print(f'    max logL = {near_logL.max():.3f}  Δ = {near_logL.max() - null_ev:+.3f}')
        best = np.argmax(near_logL)
        print(f'    best: idx={near_truth[best]}  z={sample_z[near_truth[best]]:.4f}  '
              f'logNHI={sample_lognhi[near_truth[best]]:.3f}  logL={near_logL[best]:.3f}')
    return dict(null_ev=null_ev, above_null=int(above_null), max_logL=float(logL.max()))


r_c = diagnose('c0prior', MODELS['c0prior'])
r_m = diagnose('m_baseline', MODELS['m_baseline'])

print('\n=== SUMMARY ===')
print(f'  c0prior:    null = {r_c["null_ev"]:.3f}  max_initial_logL = {r_c["max_logL"]:.3f}  above_null = {r_c["above_null"]}/5000')
print(f'  m_baseline: null = {r_m["null_ev"]:.3f}  max_initial_logL = {r_m["max_logL"]:.3f}  above_null = {r_m["above_null"]}/5000')
