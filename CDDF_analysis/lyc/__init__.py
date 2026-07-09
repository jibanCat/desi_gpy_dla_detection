"""CDDF_analysis.lyc — reusable Lyman-continuum (bound-free) opacity toolkit.

Small, dependency-light core for the LLS Lyman-limit-drop programme, shared by:
  * the injection campaign (add a physical LyC drop to mock spectra from an HCD catalog),
  * the composite-drop estimator (τ_eff,LL → κ₉₁₂ → λ_mfp),
  * the break-aware GP-finder feasibility tests,
  * the joint counting+drop HBI drop-likelihood term.

Everything numeric (σ₉₁₂, the cross-section index β, the cosmology) lives HERE so the counting
and drop channels share ONE convention (fixes the cosmology-mismatch blocker from the joint
design review). Physics anchors: PWO09 (0910.0009), Worseck+2014 (1402.4154), PMOF14
(1310.0052), Verner+1996 (σ₉₁₂), Zuo & Phinney 1993 (recovery power law).
"""
from .opacity import (
    SIGMA_912, LYMAN_LIMIT, BETA_LL, Cosmology, DEFAULT_COSMO,
    sigma_ll, tau_ll, lyc_optical_depth, lyc_transmission,
    effective_opacity, tau_eff_kernel_basis, lambda_mfp_from_kappa, fit_kappa,
    break_matched_filter_snr,
)
from .survival import (
    blue_cutoff_z, proximity_z_max, build_break_census,
    ell_nelson_aalen, ell_direct_incidence, ell_per_dz_to_dX,
)

__all__ = [
    "SIGMA_912", "LYMAN_LIMIT", "BETA_LL", "Cosmology", "DEFAULT_COSMO",
    "sigma_ll", "tau_ll", "lyc_optical_depth", "lyc_transmission",
    "effective_opacity", "tau_eff_kernel_basis", "lambda_mfp_from_kappa", "fit_kappa",
    "break_matched_filter_snr",
    # survival / incidence estimator (Nelson-Aalen g(z))
    "blue_cutoff_z", "proximity_z_max", "build_break_census",
    "ell_nelson_aalen", "ell_direct_incidence", "ell_per_dz_to_dX",
]
