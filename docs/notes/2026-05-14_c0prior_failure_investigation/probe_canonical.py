"""Probe the canonical-TID inference path for the c0prior model.

Goals:
  (1) Build NullGPMAT + DLAGPMAT just like the production pipeline.
  (2) Compute log p(D|no DLA), then evaluate log p(D|1 DLA) at the truth
      (z=2.7748, log_NHI=21.26), to see how far below the null evidence
      the model's preferred DLA fit really is.
  (3) Compute log_L for a 10x10 grid of (z_DLA, log_NHI) around the truth,
      to see the shape of the likelihood surface.
  (4) Repeat (1)-(3) for the m_baseline model on the same spectrum.
  (5) c_0 OVERRIDE experiment: take the c0prior model, replace its
      log_c_0 in-memory with the m_baseline value, and re-evaluate the
      same 1-DLA-at-truth likelihood. If detection "recovers" (log_L >
      null) → c_0 anchoring is the bug. Otherwise → kernel-shape issue.
"""
from __future__ import annotations
import os, sys, copy
from pathlib import Path
import numpy as np
import h5py

REPO = Path('/home/mfho/desi_gpy_dla_detection')
sys.path.insert(0, str(REPO))

TID = 120046865
TRUTH_NHI = 21.263
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


def setup(model_path, override_log_c_0=None):
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
    if override_log_c_0 is not None:
        old = float(null_gp.log_c_0)
        # mutate the in-memory attribute
        null_gp.log_c_0 = float(override_log_c_0)
        dla_gp.log_c_0 = float(override_log_c_0)
        print(f'  OVERRIDE: log_c_0  {old:+.4f} -> {override_log_c_0:+.4f}  (c_0  {np.exp(old):.5f} -> {np.exp(override_log_c_0):.5f})')

    # set_data (calls get_interp internally if build_model=True)
    rest = wave / (1 + z_qso)
    null_gp.set_data(rest, flux, nv, mask, z_qso, normalize=True)
    dla_gp.set_data(rest, flux, nv, mask, z_qso, normalize=True)
    return params, null_gp, dla_gp


def evaluate(name, model_path, override_log_c_0=None):
    print(f'\n=== {name} ===')
    print(f'  model: {model_path}')
    params, null_gp, dla_gp = setup(model_path, override_log_c_0=override_log_c_0)
    # 1. null evidence
    null_ev = null_gp.log_model_evidence()
    print(f'  log p(D|no DLA) = {null_ev:.4f}')
    # endpoint scalars actually used
    print(f'  in-memory log_c_0 = {dla_gp.log_c_0:.4f}  (c_0 = {np.exp(dla_gp.log_c_0):.5f})')
    print(f'  in-memory log_tau_0 = {dla_gp.log_tau_0:.4f}')
    print(f'  in-memory log_beta = {dla_gp.log_beta:.4f}')
    # 2. 1-DLA at truth
    z_truth = 2.7748
    ll_truth = dla_gp.sample_log_likelihood_k_dlas(np.array([z_truth]), np.array([10**TRUTH_NHI]))
    print(f'  log p(D|1 DLA at truth z={z_truth}, log_NHI={TRUTH_NHI}) = {ll_truth:.4f}')
    print(f'  Δ vs null = {ll_truth - null_ev:+.4f}  (>0 means DLA preferred)')
    # 3. Likelihood surface: scan z in z_DLA min..z_qso, log_NHI 20.3..22
    # but cheaper: pick a coarse grid
    # First find the actual sample-z range
    z_min_dla = params.min_z_dla(dla_gp.this_wavelengths, z_qso)
    z_max_dla = params.max_z_dla(dla_gp.this_wavelengths, z_qso)
    print(f'  z_dla scan range: [{z_min_dla:.3f}, {z_max_dla:.3f}]')
    z_grid = np.linspace(max(2.1, z_min_dla), z_max_dla, 20)
    nhi_grid = np.linspace(20.3, 22.0, 8)
    ll_surf = np.empty((len(z_grid), len(nhi_grid)))
    for i, zd in enumerate(z_grid):
        for j, lnhi in enumerate(nhi_grid):
            ll_surf[i, j] = dla_gp.sample_log_likelihood_k_dlas(np.array([zd]), np.array([10**lnhi]))
    print(f'  surf log L: max={ll_surf.max():.3f}  min={ll_surf.min():.3f}  median={np.median(ll_surf):.3f}')
    print(f'  count(logL > null) = {(ll_surf > null_ev).sum()}/{ll_surf.size}')
    # Best in surf
    bi, bj = np.unravel_index(np.argmax(ll_surf), ll_surf.shape)
    print(f'  best in surf: z={z_grid[bi]:.4f}  logNHI={nhi_grid[bj]:.3f}  logL={ll_surf[bi, bj]:.3f}  Δ={ll_surf[bi,bj]-null_ev:+.3f}')
    return dict(name=name, null_ev=null_ev, ll_truth=ll_truth, delta_truth=ll_truth-null_ev,
                ll_surf_max=float(ll_surf.max()), ll_surf=ll_surf.tolist(),
                z_grid=z_grid.tolist(), nhi_grid=nhi_grid.tolist())


results = {}
# 1. c0prior native
results['c0prior'] = evaluate('c0prior', MODELS['c0prior'])
# 2. m_baseline native
results['m_baseline'] = evaluate('m_baseline', MODELS['m_baseline'])
# 3. c0prior with c_0 overridden to m_baseline value
with h5py.File(MODELS['m_baseline'], 'r') as f:
    m_log_c_0 = float(f['log_c_0'][()])
print(f'\n[m_baseline log_c_0 = {m_log_c_0:.4f}, c_0 = {np.exp(m_log_c_0):.5f}]')
results['c0prior_override_c_0'] = evaluate(
    'c0prior_override_c_0 (use m\'s c_0)', MODELS['c0prior'],
    override_log_c_0=m_log_c_0)
# 4. c0prior with c_0 = 0.1 (the prior anchor)
results['c0prior_override_c0_01'] = evaluate(
    'c0prior_override_c_0_0.1 (anchor)', MODELS['c0prior'],
    override_log_c_0=np.log(0.1))

import json
out = REPO / 'docs/notes/2026-05-14_c0prior_failure_investigation/probe_canonical.json'
# strip large surfaces for JSON
slim = {k: {kk: vv for kk, vv in v.items() if kk not in ('ll_surf',)} for k, v in results.items()}
json.dump(slim, open(out, 'w'), indent=2)
print(f'\nWrote {out}')

print('\n=== FINAL TABLE ===')
print(f'{"variant":40s}  {"null":>10s}  {"L@truth":>10s}  {"Δ":>10s}  {"max surf":>10s}')
for k, v in results.items():
    print(f'{k:40s}  {v["null_ev"]:>10.3f}  {v["ll_truth"]:>10.3f}  {v["delta_truth"]:>+10.3f}  {v["ll_surf_max"]:>10.3f}')
