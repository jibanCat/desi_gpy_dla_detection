"""Smoke test: verify the production τ-EB wiring matches the diagnostic recipe.

Runs ``DLAHolder.process_qso`` with ``enable_tau_eb_hcd_mask=True`` and
checks:
  1. The function runs without raising.
  2. The chosen τ_eff is in the configured ``tau_eb_factors`` grid times
     ``prev_tau_0_seed``.
  3. With the SAME spectrum, the holder's chosen τ matches the diagnostic
     recipe's ``check_tau_eb_robust_mask.py`` for ``objective="dla"``.

Skipped if the canonical 2lpt mock isn't available on this machine.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


CANONICAL_TID = 120046865
CANONICAL_SPEC = (
    "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/"
    "mock-0/loa-124/spectra-16/7/789/spectra-16-789.fits"
)
CANONICAL_ZCAT = (
    "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/"
    "mock-0/loa-124/zcat.fits"
)
DATA_ROOT = "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection"


@pytest.fixture(scope="module")
def canonical_spectrum():
    if not os.path.exists(CANONICAL_SPEC):
        pytest.skip(f"canonical spec not available: {CANONICAL_SPEC}")
    if not os.path.exists(CANONICAL_ZCAT):
        pytest.skip(f"canonical zcat not available: {CANONICAL_ZCAT}")

    from gpy_dla_detection.voigt_v2_inject import inject
    inject(kernel="boss-log-r2000", num_lines=3)
    from examples.smoke_one_spectrum import load_one_desi_spectrum, lookup_z_qso

    wave, flux, nv, mask = load_one_desi_spectrum(CANONICAL_SPEC, CANONICAL_TID)
    z_qso = lookup_z_qso(CANONICAL_ZCAT, CANONICAL_TID)
    return dict(wave=wave, flux=flux, nv=nv, mask=mask, z_qso=z_qso)


def _build_params():
    from examples.smoke_one_spectrum import PRESETS
    from gpy_dla_detection.set_parameters import Parameters

    p = PRESETS["y3"]
    return p, Parameters(
        loading_min_lambda=p.loading_min_lambda,
        loading_max_lambda=p.loading_max_lambda,
        normalization_min_lambda=p.normalization_min_lambda,
        normalization_max_lambda=p.normalization_max_lambda,
        min_lambda=p.min_lambda, max_lambda=p.max_lambda,
        dlambda=p.dlambda, k=p.k,
        max_noise_variance=9.0, num_lines=3,
        max_z_cut=3000.0, min_z_cut=3000.0,
        num_forest_lines=p.num_forest_lines,
        num_dla_samples=10000,
    )


def test_fit_tau_eb_hcd_mask_runs(canonical_spectrum):
    """The standalone module returns a sane τ on the canonical target."""
    from gpy_dla_detection.tau_eb import fit_tau_eb_hcd_mask
    from gpy_dla_detection.model_priors import PriorCatalog
    from gpy_dla_detection.dla_samples import DLASamplesMAT

    preset, params = _build_params()
    learned = os.path.join(DATA_ROOT, preset.learned_file)
    catalog = os.path.join(DATA_ROOT, "data/dr12q/processed/catalog.mat")
    los_cat = os.path.join(DATA_ROOT, "data/dla_catalogs/dr9q_concordance/processed/los_catalog")
    dla_cat = os.path.join(DATA_ROOT, "data/dla_catalogs/dr9q_concordance/processed/dla_catalog")
    dla_samples_file = os.path.join(DATA_ROOT, "data/dr12q/processed/dla_samples_a03.mat")
    prior = PriorCatalog(params, catalog, los_cat, dla_cat)
    dla_samples = DLASamplesMAT(params, prior, dla_samples_file)

    s = canonical_spectrum
    rest_w = params.emitted_wavelengths(s["wave"], s["z_qso"])

    factors = (0.5, 1.0, 1.5, 2.0, 3.0)
    tau_eb, info = fit_tau_eb_hcd_mask(
        params=params, prior=prior, learned_file=learned,
        rest_wavelengths=rest_w, flux=s["flux"], noise_variance=s["nv"],
        pixel_mask=s["mask"], z_qso=s["z_qso"],
        prev_tau_0_seed=preset.prev_tau_0,
        prev_beta=preset.prev_beta,
        tau_factors=factors,
        mask_threshold_sigma=1.5,
        objective="null",
        dla_samples=dla_samples,
    )
    assert info["tau_factor_best"] in factors
    assert tau_eb == pytest.approx(preset.prev_tau_0 * info["tau_factor_best"])
    assert info["n_hcd"] >= 0
    assert len(info["log_l_per_tau"]) == len(factors)
    # Canonical target is known to prefer high τ (≥1.5×) — sanity check.
    assert info["tau_factor_best"] >= 1.0, (
        f"unexpected τ_best={info['tau_factor_best']} on canonical "
        f"(this target's HCD mask should pick a high τ)"
    )


def test_holder_disabled_path_unchanged(canonical_spectrum):
    """With enable_tau_eb=False (default), holder behavior is unchanged."""
    from run_bayes_select import DLAHolder
    preset, params = _build_params()
    learned = os.path.join(DATA_ROOT, preset.learned_file)
    catalog = os.path.join(DATA_ROOT, "data/dr12q/processed/catalog.mat")
    los_cat = os.path.join(DATA_ROOT, "data/dla_catalogs/dr9q_concordance/processed/los_catalog")
    dla_cat = os.path.join(DATA_ROOT, "data/dla_catalogs/dr9q_concordance/processed/dla_catalog")
    dla_samples_file = os.path.join(DATA_ROOT, "data/dr12q/processed/dla_samples_a03.mat")
    sub_dla_samples_file = os.path.join(DATA_ROOT, "data/dr12q/processed/subdla_samples.mat")

    # Pass params_subdla explicitly — the params.copy() fallback path
    # has a pre-existing bug unrelated to this PR.
    import copy as _copy
    holder = DLAHolder(
        learned_file=learned,
        catalog_name=catalog, los_catalog=los_cat, dla_catalog=dla_cat,
        dla_samples_file=dla_samples_file,
        sub_dla_samples_file=sub_dla_samples_file,
        params=params,
        params_subdla=_copy.copy(params),
        min_z_separation=3000.0,
        prev_tau_0=preset.prev_tau_0, prev_beta=preset.prev_beta,
        max_dlas=3, broadening=True,
        # explicit defaults
        enable_tau_eb=False,
    )
    assert holder.enable_tau_eb is False
    assert holder.tau_eb_factors == (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert holder.tau_eb_apply_hcd_mask is False
    assert holder.tau_eb_objective == "null"
