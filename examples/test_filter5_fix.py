"""Verify FILTER fix #5: 1-DLA evidence should not be poisoned by a
degenerate initial-scan valid_mask.

Test target: 2lpt TID 120046865 (truth log_NHI = 21.263; the canonical
target where production-FILTER returned p_DLA = 0.05 despite a real DLA).

Expected:
- BEFORE the fix (filter_low_likelihood=True, no FILTER #5 fix):
    p_DLA ≈ 0.05  (broken: initial scan rejected the prior)
- AFTER the fix (filter_low_likelihood=True, with FILTER #5 fix):
    p_DLA ≈ 1.0  (1-DLA evidence comes from the unbiased initial-scan,
                  which clearly prefers the DLA model over null)
- BASELINE (filter_low_likelihood=False — no truncation):
    p_DLA ≈ 1.0  (sanity check that the data has a real DLA)

Usage::
    python examples/test_filter5_fix.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np


def _run_one(filter_flag: bool, label: str):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from examples.smoke_one_spectrum import (
        load_one_desi_spectrum, lookup_z_qso, PRESETS,
    )
    from gpy_dla_detection.set_parameters import Parameters
    from run_bayes_select import DLAHolder

    spec = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/spectra-16/7/789/spectra-16-789.fits"
    zcat = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/zcat.fits"
    tid = 120046865

    wave, flux, nv, mask = load_one_desi_spectrum(spec, tid)
    z_qso = lookup_z_qso(zcat, tid)
    preset = PRESETS["y3"]
    data_root = "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection"
    common = dict(
        loading_min_lambda=preset.loading_min_lambda,
        loading_max_lambda=preset.loading_max_lambda,
        normalization_min_lambda=preset.normalization_min_lambda,
        normalization_max_lambda=preset.normalization_max_lambda,
        min_lambda=preset.min_lambda, max_lambda=preset.max_lambda,
        dlambda=preset.dlambda, k=preset.k,
        max_noise_variance=9.0, num_lines=3,
        max_z_cut=3000.0, min_z_cut=3000.0,
        num_forest_lines=preset.num_forest_lines,
    )
    params = Parameters(num_dla_samples=100000, **common)
    params_subdla = Parameters(num_dla_samples=100000, **common)

    holder = DLAHolder(
        learned_file=os.path.join(data_root, preset.learned_file),
        catalog_name=os.path.join(data_root, "data/dr12q/processed/catalog.mat"),
        los_catalog=os.path.join(data_root, "data/dla_catalogs/dr9q_concordance/processed/los_catalog"),
        dla_catalog=os.path.join(data_root, "data/dla_catalogs/dr9q_concordance/processed/dla_catalog"),
        dla_samples_file=os.path.join(data_root, "data/dr12q/processed/dla_samples_a03_100000.mat"),
        sub_dla_samples_file=os.path.join(data_root, "data/dr12q/processed/subdla_samples_a03_191_200_100000.mat"),
        params=params, params_subdla=params_subdla,
        min_z_separation=3000.0,
        prev_tau_0=preset.prev_tau_0, prev_beta=preset.prev_beta,
        max_dlas=4, broadening=True,
        plot_figures=False, max_workers=8, batch_size=12500,
        figure_dir="/tmp",
        single_absorber_model=False,
        filter_low_likelihood=filter_flag,
    )
    holder.initialize_results(1)
    t0 = time.time()
    holder.process_qso(idx=0, target_id=str(tid),
                       wavelengths=wave, flux=flux,
                       noise_variance=nv, pixel_mask=mask, z_qso=z_qso)
    dt = time.time() - t0
    res = holder.results
    print(f"\n[{label}]  filter_low_likelihood={filter_flag}  ({dt:.1f}s)")
    print(f"  p_DLA       = {float(res['p_dlas'][0]):.4f}")
    print(f"  MAP z_DLA   = {float(res['MAP_z_dlas'][0, 0]):.4f}")
    print(f"  MAP log NHI = {float(res['MAP_log_nhis'][0, 0]):.4f}")


def main():
    print("FILTER fix #5 verification on TID 120046865 (truth log_NHI = 21.263)")
    print("================================================================")
    _run_one(False, "BASELINE (FILTER off, full QMC integration)")
    _run_one(True,  "WITH FILTER fix #5 (FILTER=1, 1-DLA from initial scan)")


if __name__ == "__main__":
    main()
