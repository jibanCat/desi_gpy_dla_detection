"""validation/release -- Zenodo release-package machinery (PI 2026-09-13d §7/§17/§19).

VALIDATION-ONLY.  Nothing here touches ``CDDF_analysis/``, no sampler, no jax,
no real data.  Three entry points:

* ``provenance_manifest``   -- the machine-readable calibration -> fitted object
  -> HBI pack -> posterior -> paper-number manifest (§17);
* ``completeness_release``  -- the fitted C(N_HI, S/N) science product (§7);
* ``response_release``      -- the Q/phi response science product (§19).

``standalone/`` holds the evaluator sources that are SHIPPED IN THE RELEASE and
therefore import nothing but numpy.
"""
