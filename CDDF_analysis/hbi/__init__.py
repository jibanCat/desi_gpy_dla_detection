"""CDDF_analysis.hbi — selection-corrected catalog-HBI CDDF estimator + Track-C kernel rebuild.

Reduce-only: reuses frozen GP-DLA catalog posteriors, never re-runs inference. Produces
dN/dX, f(N), Omega_DLA from a catalog. See README.md and QUICKSTART.md in this directory.

Import submodules directly, e.g. `from CDDF_analysis.hbi.cddf_catalog_hbi import ...`.
Back-compat: the pre-2026-06 top-level paths (CDDF_analysis.cddf_catalog_hbi, etc.) still
resolve to these submodules via sys.modules-alias shims at the package root.
"""
# Intentionally NO eager re-exports: the estimator modules cross-import each other lazily,
# and tests import submodules directly. Keeping __init__ a bare marker avoids import cycles.
