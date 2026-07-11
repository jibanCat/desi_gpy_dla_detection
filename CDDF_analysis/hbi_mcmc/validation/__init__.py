"""CDDF_analysis.hbi_mcmc.validation — pure-synthetic validation ladder (rungs 1-3).

rung1_conjugate      : analytic Poisson-Gamma conjugate rate model vs NUTS.
rung2_binned_poisson : binned Poisson CDDF toy (direct bins + response-matrix variant B).
rung3_recycling      : per-sightline FGMC (k<=1) prior-QMC recycling core vs quadrature.

Import submodules directly, e.g.
``from CDDF_analysis.hbi_mcmc.validation import rung1_conjugate``.
"""
# x64 is enabled by the parent package __init__ (always executed first).
