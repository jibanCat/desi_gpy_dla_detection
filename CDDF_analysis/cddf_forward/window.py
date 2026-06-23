"""Shared CDDF search-window specification.

``WindowSpec`` is the SINGLE source of truth for the DLA search window, so that
the *measurement* (``calc_cddf.DLACatalogue``), the *truth* (mock HCD catalog),
and any *injection* path all apply an identical window.

Motivating mismatch — three different proximity *values*, all expressible in the
same velocity→z convention the inference actually used:

  * inference (``set_parameters.kms_to_z``) converts a velocity cut to a
    **constant** Δz: ``kms_to_z(v) = v*1000 / c = v_kms / C_KMS`` (the small-z
    approximation, i.e. NOT ``(1+z)``-scaled), and applies ``z_qso - kms_to_z(v)``.
    Production used ``max_z_cut = min_z_cut = 3000`` km/s → Δz ≈ 0.0100.
  * ``calc_cddf.py:324/327`` instead hard-code ``proximity_zone = tail_zone = 0.1``
    in Δz.  Under the same constant convention that is ≈ 30000 km/s (the inline
    "30000 km/s" comment is therefore *correct*) — i.e. ~10× too wide a cut, not a
    convention error.
  * ``CDDF_analysis/cddf_mock.py`` defaults ``v_prox_kms = 10000``.

So the windows differ in *magnitude*; ``WindowSpec`` fixes them to one value
(3000 km/s) in the inference's convention.

Convention (``velocity_scaled``): the default ``False`` reproduces the inference's
constant ``v/c`` (Δz independent of z_qso) so the spec is byte-identical to the
GP's stored ``min_z_dlas`` / ``max_z_dlas`` — REQUIRED, since the GP can only find
DLAs inside the window it actually searched.  ``velocity_scaled=True`` gives the
physically-preferred ``(1+z)·v/c`` proximity width, but that would only be correct
if inference is re-run with a matching ``kms_to_z`` — do NOT enable it against the
existing posteriors.

M0 scope: deliver ONLY the object (plus :meth:`assert_equal`).  Re-wiring
``calc_cddf``'s binning to consume a ``WindowSpec`` is a later milestone, at which
point also unify the Lyβ rest wavelength (``calc_cddf.py`` 1026.72 vs
``cddf_mock.LYB_REST`` 1025.72 vs ``set_parameters.lyb`` 1025.7223 → use one),
apply the proximity cut ONCE (not stacked on the inference truncation), and extend
the contract with the QSO z-range and ``lambda_obs_min`` so it is the full window.

TODO(post-M0): the processed HDF5 does not persist ``filter_low_likelihood`` or the
window params (only ``pair_prior_mode`` + ``dla_bias``) — add them to the writer so
a catalog is self-describing and the guard/window can be asserted against the file
rather than trusted from the driver.
TODO(injection milestone): inject the real DESI ``ZERR`` z-scatter into mock/
injection z' before measuring R, so the response carries the z-scatter off-diagonal
that perfect mock redshifts lack.
"""

import math
from dataclasses import dataclass, fields
from typing import Optional

# Speed of light in km/s.  This MUST equal the inference convention,
# ``gpy_dla_detection.set_parameters.Parameters.speed_of_light / 1000`` (i.e.
# 299792458 m/s / 1000), so that ``prox_dz(velocity_scaled=False)`` ==
# ``Parameters.kms_to_z(v)`` byte-for-byte.  The equivalence is pinned by
# ``tests/test_cddf_window.py::test_prox_dz_matches_inference_kms_to_z`` — if the
# inference constant ever changes, that test fails rather than this drifting silently.
C_KMS: float = 299792.458


def _windowspec_field_equal(va, vb) -> bool:
    """Field equality for :meth:`WindowSpec.assert_equal`.

    Exact ``==`` on floats is wrong for a cross-path consistency guard: two NaNs
    compare unequal (``nan != nan``) and float values derived by different
    arithmetic (``0.1 + 0.2`` vs ``0.3``) differ at ~1e-16.  So compare numeric
    fields with ``math.isclose`` (and treat two NaNs as equal); everything else
    (bool, None) with exact equality.
    """
    a_num = isinstance(va, float) and not isinstance(va, bool)
    b_num = isinstance(vb, float) and not isinstance(vb, bool)
    if a_num and b_num:
        if math.isnan(va) and math.isnan(vb):
            return True
        return math.isclose(va, vb, rel_tol=1e-9, abs_tol=0.0)
    return va == vb


@dataclass
class WindowSpec:
    """Single source of truth for the DLA search window.

    Parameters
    ----------
    v_prox_kms : float, default 3000.0
        Proximity-zone velocity cut (km/s) measured blueward of the QSO Lyα
        emission; excludes the proximate region (fewer DLAs than average).
    v_tail_kms : float, default 3000.0
        Tail-zone velocity cut (km/s) at the red/blue spectral edge; excludes
        the dubious tail (more spurious DLAs than average).
    z_min_lyb : bool, default True
        Lyα-only for the CDDF: shift the minimum search edge to the Lyβ peak.
    z_max_lyb : bool, default False
        Shift the maximum search edge to the Lyβ peak (Lyman-limit analyses).
    lambda_obs_min : Optional[float], default None
        Optional minimum observed wavelength (Å) to clip the dubious blue tail;
        ``None`` means no observed-wavelength cut.
    velocity_scaled : bool, default False
        If ``False`` (default), proximity/tail Δz is the inference's constant
        ``v/C_KMS`` (matches the stored ``min_z_dlas`` / ``max_z_dlas``).  If
        ``True``, use ``(1+z_qso)·v/C_KMS`` (the physically-preferred form) — only
        valid against inference re-run with a matching ``kms_to_z``.
    """

    v_prox_kms: float = 3000.0
    v_tail_kms: float = 3000.0
    z_min_lyb: bool = True
    z_max_lyb: bool = False
    lambda_obs_min: Optional[float] = None
    velocity_scaled: bool = False

    def _dz(self, v_kms: float, z_qso: float) -> float:
        """z-width of a velocity cut ``v_kms`` at ``z_qso`` (shared by prox/tail).

        ``velocity_scaled=False`` (default) → constant ``v_kms / C_KMS`` ==
        ``set_parameters.kms_to_z(v_kms)`` (what the stored search edges used).
        ``velocity_scaled=True`` → ``(1+z_qso)·v_kms / C_KMS``.
        """
        scale = (1.0 + z_qso) if self.velocity_scaled else 1.0
        return scale * v_kms / C_KMS

    def prox_dz(self, z_qso: float) -> float:
        """z-equivalent of the proximity velocity cut at ``z_qso`` (see :meth:`_dz`)."""
        return self._dz(self.v_prox_kms, z_qso)

    def tail_dz(self, z_qso: float) -> float:
        """z-equivalent of the tail velocity cut at ``z_qso`` (see :meth:`_dz`)."""
        return self._dz(self.v_tail_kms, z_qso)

    @staticmethod
    def assert_equal(a: "WindowSpec", b: "WindowSpec", *, ctx: str = "") -> None:
        """Raise ``ValueError`` if any field of ``a`` and ``b`` differs.

        Use to enforce that measurement / truth / injection share one window.
        Float fields are compared with tolerance (see :func:`_windowspec_field_equal`)
        so float-arithmetic drift / NaN do not cause a spurious mismatch.
        ``ctx`` is prepended to the error message for caller-side context.
        """
        diffs = []
        for fld in fields(WindowSpec):
            va = getattr(a, fld.name)
            vb = getattr(b, fld.name)
            if not _windowspec_field_equal(va, vb):
                diffs.append(f"{fld.name}: {va!r} != {vb!r}")
        if diffs:
            prefix = f"{ctx}: " if ctx else ""
            raise ValueError(
                prefix + "WindowSpec mismatch — " + "; ".join(diffs)
            )
