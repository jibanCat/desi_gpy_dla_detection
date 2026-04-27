"""
gpy_dla_detection/voigt_v2_inject.py
=====================================
Importable shim that swaps ``gpy_dla_detection.voigt_fast.VoigtProfile``
for ``voigt_v2.VoigtProfileV2`` *before* ``dla_gp`` imports it.

Usage (from a runner / smoke script, before any GP imports):

    from gpy_dla_detection.voigt_v2_inject import inject
    inject(kernel="desi-linear-r3000", num_lines=3)

After this, ``from gpy_dla_detection.dla_gp import DLAGPMAT`` will pick up
the v2 forward model with the requested LSF kernel and number of Lyman
series lines. The production C extension is left on disk untouched.
"""

from __future__ import annotations

import sys
from typing import Literal

import numpy as np


def inject(kernel: str = "boss-log-r2000", num_lines: int = 3):
    """Replace voigt_fast.VoigtProfile with a v2 wrapper.

    The wrapper has the same surface as the production class so dla_gp.py's
    `voigt_absorption = VoigtProfile().compute_voigt_profile` line picks it
    up transparently. Must be called before dla_gp / null_gp are imported.
    """
    from gpy_dla_detection import voigt_v2
    from gpy_dla_detection import voigt_fast as _vf

    base = voigt_v2.VoigtProfileV2(kernel=kernel)

    class _ShimVoigtProfile:
        def __init__(self):
            pass

        def compute_voigt_profile(self, wavelengths, nhi, z_dla,
                                  num_lines=num_lines):
            return base.compute_voigt_profile(
                wavelengths, nhi, z_dla, num_lines=num_lines,
            )

    _vf.VoigtProfile = _ShimVoigtProfile
    # Also reload dla_gp / null_gp if they've already been imported, so
    # their `voigt_absorption = VoigtProfile().compute_voigt_profile`
    # binding refreshes.
    for mod in ["gpy_dla_detection.dla_gp",
                "gpy_dla_detection.null_gp"]:
        if mod in sys.modules:
            import importlib
            importlib.reload(sys.modules[mod])
    return base
