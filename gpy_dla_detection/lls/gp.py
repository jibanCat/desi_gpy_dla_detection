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

from ..subdla_gp import SubDLAGP, SubDLAGPMAT
from ..dla_gp import DLAGPMAT
from ..voigt_lls import voigt_absorption as voigt_absorption_lls


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
    Drop-in replacement for SubDLAGPMAT in the (obsolete) 3-way LLS finder run."""


class DLAGPMATLymanBreak(_LymanBreakMixin, DLAGPMAT):
    """DLAGPMAT with the Lyman-limit break folded in — the SINGLE-ABSORBER LLS model.

    In the single-absorber finder (null vs one absorber; the 3-way null/subDLA/DLA split is
    retired) the absorber IS the DLA model. Point its samples at the LLS N-range (e.g. the
    subdla_samples 17.2-20.3 grid, or a 17.2-floor DLA-samples file) and this model scores the
    912 A break of each candidate. Drop-in replacement for DLAGPMAT."""


# ---------------------------------------------------------------------------
# LLS-only model loading + blueward window extension (bring the drop in-window)
# ---------------------------------------------------------------------------
# The production DESI model's rest grid already reaches ~851 A (Lyman-continuum region);
# the standard inference just clips it to min_lambda=911.75. Extending the LLS window to
# 850.9 (Tier 1) is therefore CONFIG-ONLY (no retrain) and brings ~43% of LLS breaks into
# the modelled range. A retrain down to ~800 A (Tier 2) would raise that to ~66%.
# This model is loaded ONLY here (LLS), never for DLA/sub-DLA.
LLS_MODEL_DEFAULT = (
    "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/"
    "desi_gpy_dla_detection/learnlogs/model_epoch_920.h5"
)
LLS_REST_MIN_DEFAULT = 850.9  # model_epoch_920 grid floor; extend window here (Tier 1)


def extend_window_to_drop(params, rest_min: float = LLS_REST_MIN_DEFAULT):
    """Extend the GP inference window blueward so an LLS's 912 A break — which lands at
    quasar-rest 912*(1+z_abs)/(1+z_qso) < 912 — falls inside the modelled range.

    CONFIG-ONLY: mutates params.min_lambda and params.loading_min_lambda in place. The loaded
    model grid must already cover rest_min (model_epoch_920 covers 850.9). Returns params."""
    params.min_lambda = float(rest_min)
    params.loading_min_lambda = min(float(params.loading_min_lambda), float(rest_min))
    return params


def load_lls_gp(
    params,
    prior,
    dla_samples,
    learned_file: str = LLS_MODEL_DEFAULT,
    rest_min: float = LLS_REST_MIN_DEFAULT,
    **kw,
):
    """Construct the break-aware LLS GP (SubDLAGPMATLymanBreak) with (a) the inference window
    extended blueward into the Lyman-continuum region and (b) the LLS-only model loaded.

    Use this for the LLS finder run; DLA/sub-DLA keep their own model + window untouched."""
    extend_window_to_drop(params, rest_min=rest_min)
    return SubDLAGPMATLymanBreak(
        params=params,
        prior=prior,
        dla_samples=dla_samples,
        learned_file=learned_file,
        **kw,
    )
