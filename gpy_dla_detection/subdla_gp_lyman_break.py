"""subdla_gp_lyman_break.py — LLS/sub-DLA GP model whose forward model includes the
Lyman-limit continuum BREAK, so ONE GP likelihood scores both the Lyman-series lines and
the 912 Å drop of a candidate absorber.

Motivation: in the LLS regime (17.2 <~ log N_HI <~ 19) the Lyα line is saturated but not
damped (flat curve of growth) -> line-only detection is ~5% pure. The bound-free Lyman-limit
break tau_LL = N_HI * sigma_912 * (lambda_rest/912)^3 (lambda_rest<912) is a distinctive,
N-dependent signature the line lacks (per-sightline matched-filter S/N ~10-18 on 2LPT mocks),
so a forward model that includes it should detect LLS with far higher purity.

FROZEN-CODE DISCIPLINE: this is a SEPARATE class in a NEW file. It reuses the byte-frozen
`SubDLAGP` / `DLAGP` verbatim and only *overrides* `this_dla_gp` to route the per-absorber
absorption through `voigt_lls.voigt_absorption` (which already computes
    raw_profile = exp( nhi * sum_l(-c_l * Voigt) - tau_LLS_break )
= exp(-(tau_lines + tau_LL))) instead of the plain line-only `voigt_absorption`. `dla_gp.py`,
`subdla_gp.py`, `voigt.py` are NOT modified, so the production inference path is byte-identical.

Usage: swap `SubDLAGPMAT` -> `SubDLAGPMATLymanBreak` in the LLS run; everything else (priors,
samples, model-selection) is inherited unchanged.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np

from .subdla_gp import SubDLAGP, SubDLAGPMAT
from .voigt_lls import voigt_absorption as voigt_absorption_lls


class _LymanBreakMixin:
    """Override this_dla_gp to include the Lyman-limit break in the per-absorber absorption.

    This is a faithful copy of DLAGP.this_dla_gp with the SINGLE change that the Voigt
    absorption is computed by voigt_lls.voigt_absorption (lines + 912 Å break) rather than the
    line-only voigt_absorption. Kept in sync with dla_gp.DLAGP.this_dla_gp."""

    def this_dla_gp(
        self, z_dlas: np.ndarray, nhis: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert len(z_dlas) == len(nhis)
        k_dlas = len(z_dlas)

        # to retain only unmasked pixels from the computed absorption profile
        mask_ind = ~self.pixel_mask[self.ind_unmasked]

        # [broadening] use the padded wavelengths for convolution
        if self.broadening:
            wavelengths = self.padded_wavelengths
        else:
            wavelengths = self.unmasked_wavelengths

        absorption = np.ones(self.unmasked_wavelengths.shape[0])

        for j in range(k_dlas):
            # cache tagged 'llb' so a break-aware instance never reuses a line-only profile
            cache_key = (z_dlas[j], nhis[j], self.broadening, "llb")
            if cache_key in self.voigt_cache:
                cached_absorption = self.voigt_cache[cache_key]
            else:
                cached_absorption = voigt_absorption_lls(
                    wavelengths,
                    z_lls=z_dlas[j],
                    nhi=nhis[j],
                    num_lines=self.params.num_lines,
                )
                self.voigt_cache[cache_key] = cached_absorption
            absorption *= cached_absorption

        absorption = absorption[mask_ind]
        assert len(absorption) == len(self.this_mu)

        dla_mu = self.this_mu * absorption
        dla_M = self.this_M * absorption[:, None]
        dla_omega2 = self.this_omega2 * absorption ** 2

        return dla_mu, dla_M, dla_omega2


class SubDLAGPLymanBreak(_LymanBreakMixin, SubDLAGP):
    """SubDLAGP with the Lyman-limit break folded into the forward model. Drop-in for SubDLAGP."""


class SubDLAGPMATLymanBreak(_LymanBreakMixin, SubDLAGPMAT):
    """SubDLAGPMAT (the .mat-sample production LLS model) with the Lyman-limit break folded in.
    Drop-in replacement for SubDLAGPMAT in the LLS finder run."""
