"""M3 DLA-injection campaign — Bayesian-modeling owner's modules.

Pure python/numpy (NO desispec, NO coadd I/O — that is the CS agent's
``coadd_injection.py``).  This subpackage owns the statistical core of the M3
injection campaign:

* :mod:`campaign_grid`  — the (logN_true × z_true × SNR_bin) injection grid, the
  clean-sightline sampler, and the MANIFEST SCHEMA the CS injector consumes.
* :mod:`measurements`   — the three recovered-vs-injected estimators
  (detection completeness, N_HI bias, off-diagonal response matrix + b_FP).

See ``2026-06-10_m3_injection_campaign_design.md`` (notes repo) for the design.
"""
