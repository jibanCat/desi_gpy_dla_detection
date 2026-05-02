"""HCD-masked empirical-Bayes τ_eff fit for production use.

Implements the per-spectrum τ_0 fit validated on n=18/54 DLA targets
across 3 mocks (closes 81 % of median DLA-regime N_HI bias). Companion
documentation: ``docs/tau_eb_hcd_mask.md``.

The recipe:
  1. Build a null GP at a seed τ_0 (Turner+2024 by default).
  2. Identify HCD pixels where (y - μ_pred)/σ < -mask_threshold_sigma.
  3. With the HCD-extended pixel mask, scan τ on a small grid; at each
     τ, compute a log-likelihood scalar (objective; see below) and pick
     τ_best = argmax.
  4. The caller then runs the production inference at τ_best with the
     ORIGINAL pixel mask (the HCD mask is only used to pick τ).

Objectives for the τ-fit step:
  - "null"  : log p(D | null GP) on HCD-masked pixels. Cheap (K null
              builds, no DLA forward model). Closest to the textbook
              Becker / Faucher-Giguère mean-flux convention.
  - "dla"   : max over a (z_DLA, log N_HI) grid of log p(D | 1-DLA GP)
              on HCD-masked pixels. Matches the validated diagnostic
              recipe. ~K times more expensive than "null" because each
              τ point requires a DLAGPMAT build + grid scan.

Both objectives use the same HCD mask. Empirically (single-target
canonical 120046865) they pick the same τ_best ⇒ the cheaper "null"
objective is the recommended production default. The "dla" objective
is preserved for parity with the diagnostic and for spectra where the
two might diverge (open question).

Usage::

    from gpy_dla_detection.tau_eb import fit_tau_eb

    tau_eb, info = fit_tau_eb(
        params=params,
        prior=prior,
        learned_file=learned_file,
        rest_wavelengths=rest_w,
        flux=flux,
        noise_variance=nv,
        pixel_mask=mask,
        z_qso=z_qso,
        prev_tau_0_seed=0.00246,    # Turner+2024
        prev_beta=3.62,
        tau_factors=(0.5, 1.0, 1.5, 2.0),
        mask_threshold_sigma=1.5,
        objective="null",
    )

    # Then run production inference at τ_eb with the ORIGINAL mask.
"""
from __future__ import annotations

from typing import Tuple, Sequence, Optional, Dict, Any

import numpy as np

from .null_gp import NullGPMAT


def _build_hcd_mask(
    rest_wavelengths: np.ndarray,
    flux: np.ndarray,
    noise_variance: np.ndarray,
    pixel_mask: np.ndarray,
    z_qso: float,
    *,
    params,
    prior,
    learned_file: str,
    prev_tau_0_seed: float,
    prev_beta: float,
    mask_threshold_sigma: float,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Build the seed null GP and identify HCD pixels by negative residual.

    Returns
    -------
    new_mask : np.ndarray of bool, same shape as pixel_mask
        Original pixel_mask ∪ HCD-flagged pixels. To be used as the
        pixel_mask for the τ-fit step ONLY.
    n_hcd : int
        Number of HCD-flagged pixels (≥ 0).
    residuals_sigma : np.ndarray
        (y - μ_pred) / σ on the unmasked-and-in-range subset, for caller
        diagnostics.
    """
    null_gp = NullGPMAT(params, prior, learned_file=learned_file,
                        prev_tau_0=prev_tau_0_seed, prev_beta=prev_beta)
    null_gp.set_data(rest_wavelengths, flux, noise_variance, pixel_mask,
                     z_qso, build_model=True)
    pred = null_gp.this_mu
    y = null_gp.y
    sigma2 = null_gp.this_omega2 + null_gp.v
    residuals_sigma = (y - pred) / np.sqrt(sigma2)
    hcd_mask_inner = residuals_sigma < -mask_threshold_sigma
    n_hcd = int(hcd_mask_inner.sum())

    # Map the inner-grid HCD flags back to the full pixel_mask grid.
    full_idx_in_range = np.flatnonzero(null_gp.ind_unmasked)
    survived = ~pixel_mask[full_idx_in_range]
    full_idx_used = full_idx_in_range[survived]
    new_mask = pixel_mask.copy()
    new_mask[full_idx_used[hcd_mask_inner]] = True
    return new_mask, n_hcd, residuals_sigma


def _log_l_null_at_tau(
    rest_wavelengths: np.ndarray,
    flux: np.ndarray,
    noise_variance: np.ndarray,
    pixel_mask: np.ndarray,
    z_qso: float,
    *,
    params,
    prior,
    learned_file: str,
    prev_tau_0: float,
    prev_beta: float,
) -> float:
    """Build a null GP at a given τ_0 and return its log model evidence."""
    g = NullGPMAT(params, prior, learned_file=learned_file,
                  prev_tau_0=prev_tau_0, prev_beta=prev_beta)
    g.set_data(rest_wavelengths, flux, noise_variance, pixel_mask, z_qso,
               build_model=True)
    return float(g.log_model_evidence())


def _log_l_dla_max_over_grid_at_tau(
    rest_wavelengths: np.ndarray,
    flux: np.ndarray,
    noise_variance: np.ndarray,
    pixel_mask: np.ndarray,
    z_qso: float,
    *,
    params,
    prior,
    dla_samples,
    learned_file: str,
    prev_tau_0: float,
    prev_beta: float,
    z_dla_grid: np.ndarray,
    log_nhi_grid: np.ndarray,
) -> float:
    """Build a 1-DLA GP at given τ_0 and return max log L over (z, NHI) grid.

    Mirrors the diagnostic recipe ``examples/check_tau_eb_robust_mask.py``,
    but scans z over ``z_dla_grid`` rather than fixing at truth_z.
    """
    # Late import: DLAGPMAT pulls in voigt_fast, which can be expensive to
    # initialize. Keep it lazy so the cheaper "null" objective doesn't
    # have to pay for it.
    from .dla_gp import DLAGPMAT

    g = DLAGPMAT(params, prior, dla_samples,
                 min_z_separation=3000.0, learned_file=learned_file,
                 broadening=True,
                 prev_tau_0=prev_tau_0, prev_beta=prev_beta)
    g.set_data(rest_wavelengths, flux, noise_variance, pixel_mask, z_qso,
               build_model=True)
    best = -np.inf
    for z_dla in z_dla_grid:
        for ln in log_nhi_grid:
            try:
                ll = g.sample_log_likelihood_k_dlas(
                    np.array([z_dla]), np.array([10**ln]))
                if ll > best:
                    best = ll
            except Exception:
                pass
    return float(best)


def fit_tau_eb(
    *,
    params,
    prior,
    learned_file: str,
    rest_wavelengths: np.ndarray,
    flux: np.ndarray,
    noise_variance: np.ndarray,
    pixel_mask: np.ndarray,
    z_qso: float,
    prev_tau_0_seed: float,
    prev_beta: float,
    tau_factors: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0),
    apply_hcd_mask: bool = False,
    mask_threshold_sigma: float = 1.5,
    objective: str = "null",
    dla_samples=None,
    z_dla_grid: Optional[np.ndarray] = None,
    log_nhi_grid: Optional[np.ndarray] = None,
    return_diagnostics: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """Per-spectrum empirical-Bayes fit for τ_0 (optionally with HCD masking).

    Parameters
    ----------
    params, prior, learned_file
        Same objects/path used to construct production NullGPMAT and
        DLAGPMAT.
    rest_wavelengths, flux, noise_variance, pixel_mask, z_qso
        1-D arrays (and a scalar z_qso) for the spectrum being fit. SAME
        shapes/types as expected by ``NullGPMAT.set_data``.
    prev_tau_0_seed, prev_beta
        The production τ_0 and β. The seed τ_0 is used as the multiplicative
        center for ``tau_factors`` (and for the seed null GP if HCD masking
        is enabled).
    tau_factors : sequence of float
        τ-grid: τ_0 candidates are ``tau_factors[i] * prev_tau_0_seed``.
        Default (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0). At n=5000 random
        2LPT spectra (no cherry-picking) the median is 3.0×, mean 2.78×;
        the histogram decays past 4× (τ=4: 17 %, τ=5: 7 %, τ=6: 4 %),
        so 6× is a defensible ceiling. Earlier (...,4.0) grid caused
        28 % ceiling pile-up at τ=4.
    apply_hcd_mask : bool, default False
        If True, run a seed null-GP fit at ``prev_tau_0_seed``, identify
        pixels with negative standardized residual ``< -mask_threshold_sigma``,
        and add them to the pixel mask for the τ-fit step ONLY.
        At n=90 representative DLA targets, the mask **systematically
        over-corrects** the τ_0 fit (median bias −0.131 dex vs +0.026 dex
        without the mask). Default OFF — see
        ``docs/notes/2026-04-29_tau_eb_n90_unbiasedness.md``. Keep True
        only for saturated-DLA-dominated targets where the canonical
        recipe was validated (n=18 sample at seed=42).
    mask_threshold_sigma : float
        Negative-residual threshold for HCD flagging; only used when
        ``apply_hcd_mask=True``. Default 1.5 σ.
    objective : {"null", "dla"}
        See module docstring. "null" is cheaper (K null builds), "dla"
        matches the validated diagnostic at higher cost.
    dla_samples, z_dla_grid, log_nhi_grid
        Required when ``objective == "dla"``. ``z_dla_grid`` defaults
        to 5 evenly-spaced points across the search window;
        ``log_nhi_grid`` defaults to 35 points in [20.3, 22.0].
    return_diagnostics : bool
        If True, the second tuple element contains intermediate arrays
        (residuals, per-tau log L, hcd mask flags). Default False to
        reduce per-spectrum memory.

    Returns
    -------
    tau_eb : float
        Best τ_0 ( = tau_factors[j_best] * prev_tau_0_seed ).
    info : dict
        Always contains ``tau_factor_best``, ``n_hcd``, ``log_l_per_tau``;
        more arrays when ``return_diagnostics=True``.
    """
    if objective not in ("null", "dla"):
        raise ValueError(f"objective must be 'null' or 'dla', got {objective!r}")
    if objective == "dla" and dla_samples is None:
        raise ValueError("objective='dla' requires dla_samples")

    tau_factors = np.asarray(tau_factors, dtype=float)

    # Step 1+2: optional HCD-mask. Default OFF — see module docstring +
    # 2026-04-29_tau_eb_n90_unbiasedness.md.
    if apply_hcd_mask:
        new_mask, n_hcd, residuals_sigma = _build_hcd_mask(
            rest_wavelengths, flux, noise_variance, pixel_mask, z_qso,
            params=params, prior=prior, learned_file=learned_file,
            prev_tau_0_seed=prev_tau_0_seed, prev_beta=prev_beta,
            mask_threshold_sigma=mask_threshold_sigma,
        )
    else:
        new_mask = pixel_mask
        n_hcd = 0
        residuals_sigma = None

    # Step 3: τ-grid scan with HCD-extended mask.
    log_l_per_tau = np.full(tau_factors.size, -np.inf)
    if objective == "null":
        for j, tf in enumerate(tau_factors):
            log_l_per_tau[j] = _log_l_null_at_tau(
                rest_wavelengths, flux, noise_variance, new_mask, z_qso,
                params=params, prior=prior, learned_file=learned_file,
                prev_tau_0=prev_tau_0_seed * tf, prev_beta=prev_beta,
            )
    else:  # "dla"
        if z_dla_grid is None:
            # Default: 5 evenly spaced z within search window.
            z_min = float(params.min_z_dla(rest_wavelengths * (1 + z_qso), z_qso))
            z_max = float(params.max_z_dla(rest_wavelengths * (1 + z_qso), z_qso))
            z_dla_grid = np.linspace(z_min, z_max, 5)
        if log_nhi_grid is None:
            log_nhi_grid = np.arange(20.30, 22.01, 0.05)
        for j, tf in enumerate(tau_factors):
            log_l_per_tau[j] = _log_l_dla_max_over_grid_at_tau(
                rest_wavelengths, flux, noise_variance, new_mask, z_qso,
                params=params, prior=prior, dla_samples=dla_samples,
                learned_file=learned_file,
                prev_tau_0=prev_tau_0_seed * tf, prev_beta=prev_beta,
                z_dla_grid=z_dla_grid, log_nhi_grid=log_nhi_grid,
            )

    j_best = int(np.argmax(log_l_per_tau))
    tau_factor_best = float(tau_factors[j_best])
    tau_eb = float(prev_tau_0_seed * tau_factor_best)

    info: Dict[str, Any] = {
        "tau_factor_best": tau_factor_best,
        "tau_eb": tau_eb,
        "n_hcd": n_hcd,
        "log_l_per_tau": log_l_per_tau.tolist(),
        "objective": objective,
    }
    if return_diagnostics:
        if residuals_sigma is not None:
            info["residuals_sigma"] = residuals_sigma
        info["hcd_pixel_mask"] = new_mask
        info["tau_factors"] = tau_factors.tolist()
    return tau_eb, info


# Backward-compatible alias (was the original name when HCD masking was
# the default; the mask is now an opt-in flag, so the API name is broader).
fit_tau_eb_hcd_mask = fit_tau_eb
