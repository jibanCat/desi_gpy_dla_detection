"""gpy_dla_detection.lls — the LLS (Lyman-limit system) sub-module.

Everything LLS-specific lives here, DISENTANGLED from the main DLA / sub-DLA path so the
NERSC-proven DLA inference stays byte-identical. The LLS regime (17.2 <~ logN <~ 19) needs
extra machinery the DLA search does not:

  gp      — break-aware GP: the GP+Voigt forward model WITH the 912 A Lyman-limit drop folded
            into ONE likelihood (SubDLAGP{,MAT}LymanBreak). See gp.load_lls_gp to load the
            LLS-only relearned model (extended blueward to cover the drop) — never used for
            DLA/sub-DLA, which keep their own frozen model.
  mirror  — mirror-quickquasar LyC injection: add the bound-free 912 A drop to 2LPT spectra
            (which carry quickquasars' HCD Lyman-SERIES lines only), writing a mirror
            spectra-16 tree the finder reads unchanged. No quickquasars re-run.
  train   — relearn the QSO GP down to the drop (loading/min_lambda extended to ~800 A rest),
            producing the LLS-only learned model. Wraps the frozen trainer; config-only.

The break signal for a foreground LLS lands at quasar-rest 912*(1+z_abs)/(1+z_qso) < 912 A,
i.e. BLUEWARD of the standard model window [911.75, 1215.75] (~99.6% of LLS breaks fall below
it). So the LLS model is relearned down to ~800 A to bring the drop inside the modelled range.
"""
from __future__ import annotations

from .gp import (
    SubDLAGPLymanBreak,
    SubDLAGPMATLymanBreak,
    _LymanBreakMixin,
    load_lls_gp,
    extend_window_to_drop,
    LLS_MODEL_DEFAULT,
    LLS_REST_MIN_DEFAULT,
)

__all__ = [
    "SubDLAGPLymanBreak",
    "SubDLAGPMATLymanBreak",
    "_LymanBreakMixin",
    "load_lls_gp",
    "extend_window_to_drop",
    "LLS_MODEL_DEFAULT",
    "LLS_REST_MIN_DEFAULT",
]
