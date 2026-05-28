"""Reproduce EXACTLY what parallel_log_model_evidences does on the c0prior
model for TID 120046865 to find why valid_mask.sum() == 0 in production.
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
from gpy_dla_detection.dla_gp import process_sample, select_region_indices_searchsorted

preset = PRESETS['y3']
wave, flux, nv, mask = load_one_desi_spectrum(SPEC, TID)
z_qso = lookup_z_qso(ZCAT, TID)


def reproduce(name, model_path):
    print(f'\n=== {name} ===')
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
    print(f'  null evidence (my probe) = {null_ev:.4f}')

    # Now mimic process_sample
    sample_z_dlas = dla_samples.sample_z_dlas(dla_gp.this_wavelengths, z_qso)
    base_sample_inds = np.zeros((3, params.num_dla_samples), dtype=np.int32)
    base_sample_inds_T = base_sample_inds.T

    # 5000 initial samples — process_sample subtracts log(num_dla_samples)
    n_init = 5000
    raw_logL = np.empty(n_init)
    shifted_logL = np.empty(n_init)
    for i in range(n_init):
        res = process_sample(
            i, 0, sample_z_dlas, base_sample_inds_T,
            dla_samples, params,
            dla_gp.sample_log_likelihood_k_dlas,
            3000.0,
        )
        shifted_logL[i] = res
        raw_logL[i] = res + np.log(params.num_dla_samples)

    lognorm = np.log(params.num_dla_samples)
    print(f'  lognorm correction = {lognorm:.4f}')
    print(f'  raw logL: max={raw_logL.max():.3f}  median={np.median(raw_logL):.3f}')
    print(f'  shifted logL (after -lognorm): max={shifted_logL.max():.3f}  median={np.median(shifted_logL):.3f}')

    # The FILTER uses these shifted values vs prior-weighted null_evidence.
    # In production: null_evidence = log_lik(no_dla) + log_priors[0] - log_prior_dla
    # We don't easily reproduce log_priors here, so use the value printed in
    # the production log: -3027.48 (for c0prior) or compute from p(no DLA) /
    # p(DLA) ratio printed by the Bayes module:
    #   c0prior: p(no DLA) = 0.958, p(DLA) = ? from log p(DLA|zQSO)=-2.616
    # The simplest approach: scan a range of plausible thresholds.
    print(f'  --- threshold scan ---')
    lognorm = np.log(params.num_dla_samples)
    null_ev_shifted = null_ev - lognorm
    for thresh_label, thresh in [
        ('null_ev (raw)', null_ev),
        ('null_ev_shifted (raw - lognorm)', null_ev_shifted),
        ('prod null_ev (prior-weighted)',
         -3027.477 if 'c0prior' in name else -2877.0),
    ]:
        n_above = (shifted_logL > thresh).sum()
        valid_mask = select_region_indices_searchsorted(
            z_all=sample_z_dlas,
            initial_logL=shifted_logL,
            initial_z=sample_z_dlas[:n_init],
            z_tol=0.02,
            logL_null=thresh,
        )
        vm = int(np.sum(valid_mask)) if len(valid_mask) else 0
        print(f'    thresh={thresh:>10.3f}  ({thresh_label}):  above={n_above:>5d}  valid_mask.sum()={vm}')

    # What if we use null_ev as MY computed value (no broadening difference)?
    # Also check using -lognorm-corrected null
    null_ev_shifted = null_ev - lognorm
    print(f'  null_ev_shifted (null - lognorm) = {null_ev_shifted:.3f}')
    print(f'  raw_logL > null_ev (NO shift on either side): {(raw_logL > null_ev).sum()}/{n_init}')
    print(f'  shifted_logL > null_ev_shifted: {(shifted_logL > null_ev_shifted).sum()}/{n_init}')

    return dict(name=name, null_ev=null_ev, max_raw=float(raw_logL.max()),
                max_shifted=float(shifted_logL.max()),
                above_null_shifted=int(above_null))


r_c = reproduce('c0prior', MODELS['c0prior'])
r_m = reproduce('m_baseline', MODELS['m_baseline'])
