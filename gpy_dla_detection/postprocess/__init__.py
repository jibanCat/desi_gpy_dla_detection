"""Post-processing utilities for GP-DLA catalogs.

These helpers run AFTER the main inference pipeline. They take a DLA catalog
(per-spectrum HDF5 from `combine_processed_h5.py`, or a FITS catalog from
`dlasearch.py`) and add veto / cross-reference flags without re-running
inference.

Modules:
- :mod:`gpy_dla_detection.postprocess.lyb_veto` — flag MAP DLAs that match
  the Lyβ-shifted z of another (stronger, higher-z) MAP DLA on the same LOS.
- :mod:`gpy_dla_detection.postprocess.lls_cross_reference` — pull the
  matching LLS-mode posterior and downgrade DLA-mode detections that
  the LLS-mode catalog explains better.

See the README in this directory for usage and the rationale.
"""
