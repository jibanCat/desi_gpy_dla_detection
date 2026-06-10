# -*- coding: utf-8 -*-
"""
CDDF_analysis/calc_cddf.py — Bayesian DLA statistical products from GP-DLA inference outputs.

Overview
--------
Computes three downstream statistical products from GP-DLA model posteriors:

  1. **CDDF** — Column Density Distribution Function:
       f(N_HI) = d²n_DLA / (dN_HI dX)
     where n_DLA is the expected number of absorbers per sightline with column density N_HI
     and X is the absorption distance (dimensionless comoving path length).
     Units: cm² (since N_HI has units cm⁻²).

  2. **dN/dX** — Line density (number of DLAs per unit absorption distance):
       dN/dX = sum_spectra P(DLA | z_DLA in [z, z+dz], D) / dX(z, z+dz)
     Also called the incidence rate.

  3. **Omega_DLA** — Neutral hydrogen mass density in DLAs:
       Omega_DLA = (m_p H_0 / c rho_c) * sum_NHI N_HI * f(N_HI) dN_HI / dX
     Computed both by summing the CDDF (omega_dla_cddf) and by direct histogram (omega_dla).

Input files
-----------
  processed_file : HDF5 output of the GP-DLA inference pipeline (process_helpers.py).
      Contains per-spectrum: model_posteriors, sample_log_likelihoods_dla,
      log_likelihoods_dla, min_z_dlas, max_z_dlas, z_qsos, snrs, target_ids.

  sample_file : QMC sample grid (.mat, MATLAB v7.3 HDF5).
      Contains: offset_samples, log_nhi_samples, nhi_samples.
      For DLA runs: dla_samples_a03.mat (Ho+2020 grid, log NHI ∈ [20.3, 23]).
      For sub-DLA/LLS runs: generated via gpy_dla_detection/generate_samples.py.

  catalog_file : FITS QSO catalog.
      Used to align target IDs between the processed file and the reference catalog.

model_posteriors index layout
------------------------------
The `model_posteriors` array in the processed file has shape (num_qsos, num_models):

  DLA run (sub_dla=True, default):
      index 0   → Null model       (no absorber)
      index 1   → Sub-DLA model    (log NHI ∈ [19, 20.3])
      index 2   → DLA(1) model     (1 DLA, log NHI > 20.3)
      index 3   → DLA(2) model     (2 DLAs)
      index 4   → DLA(3) model     (3 DLAs)

  Sub-DLA / LLS run (sub_dla=False, single_absorber_model=True):
      index 0   → Null model       (no absorber)
      index 1   → DLA(1) model     (1 absorber)

The ``sub_dla`` parameter in DLACatalogue accounts for this shift:
    p_DLA   := model_posteriors[:, 1 + sub_dla:]  (sum of all DLA models)
    p_no_DLA := model_posteriors[:, :1 + sub_dla]  (null + sub-DLA if present)

Path length formula
-------------------
The absorption distance X is the comoving path length per unit redshift:

    dX = (1 + z)^2 * H_0 / H(z) * dz

where H(z) = H_0 * sqrt(Omega_m (1+z)^3 + Omega_Lambda).
This is integrated numerically for each spectrum's [z_min, z_max] range
(see ``path_length_int`` and ``path_length``).

Bayesian confidence intervals
------------------------------
Each DLA detection contributes a probability p_i to each statistics bin.
The aggregate count in a bin is a sum of Bernoulli variables → Poisson-binomial
distribution. Two approximations are used:

  - For p_i < p_switch (default 0.25): Poisson approximation (Le Cam 1960).
    Error bounded by sum(p_i²) / sum(p_i).
  - For p_i >= p_switch: exact Poisson-binomial PDF via FFT (Fernandez & Williams 2010).

68% and 95% credible intervals are extracted from the combined PDF.

Key classes
-----------
DLACatalogue   : main analysis class; holds file handles, filters, and stat methods.

Key functions (module-level)
-----------------------------
path_length_int(z)       : integrand dX/dz for scipy.integrate.quad
HubbleByH0(z)            : H(z)/H_0 (WMAP9 cosmology)
rho_crit(hubble)         : critical density at z=0 in g cm⁻³
get_poisson_binomial_pdf : exact Poisson-binomial PDF via FFT
pdf_confidence           : MAP + 68/95% credible intervals from a PDF array
interval                 : confidence interval extraction from a CDF array

References
----------
Ho, Bird & Garnett (2020) https://arxiv.org/abs/2003.11036
Prochaska et al. (2014)   https://arxiv.org/abs/1402.0548
Noterdaeme et al. (2012)  — dN/dX comparison bins
Le Cam (1960)             — Poisson approximation theorem
Fernandez & Williams (2010) — Poisson-binomial FFT algorithm
WMAP9 cosmological parameters (Omega_m=0.279, h=0.7 default)
"""
from typing import Optional, Union

import math

# Complex number
import cmath
import operator
import h5py
import numpy as np
import scipy.integrate as integrate
from scipy.special import logsumexp
from scipy.stats import poisson
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

from .set_parameters import *

# WindowSpec (the shared search-window spec) is annotation-only here, imported under
# TYPE_CHECKING with a string forward-reference annotation. A runtime
# ``from .cddf_forward.window import WindowSpec`` would run ``cddf_forward/__init__.py``,
# which eagerly imports ``driver`` → ``from ..calc_cddf import DLACatalogue`` → a
# circular import while ``calc_cddf`` is still initializing. The constructor only ever
# duck-types a passed window (``.prox_dz``/``.v_prox_kms``/``.z_min_lyb``), so no
# runtime class reference is needed and there is exactly ONE WindowSpec class.
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .cddf_forward.window import WindowSpec

# prevent cluster session plotting issue
import matplotlib

matplotlib.use("pdf")

# TODO: remove samples with nhi > 22.5 : [requrie re-calculate the model_posteriors]
# TODO: remove samples with min_z_dlas < min_z_dlas + 0.1 : [requrie re-calculate the model_posteriors]
# TODO: higher SNR thresh tests
# TODO: zQSO split tests : make sure avoid the assert error


class DLACatalogue(object):
    """GP-DLA statistical catalogue: computes CDDF, dN/dX, and Omega_DLA.

    This class loads the GP-DLA inference outputs (processed HDF5 file) and
    QMC sample grids, applies quality filters, and provides methods for
    computing three key DLA statistics: the column density distribution
    function (CDDF), the incidence rate (dN/dX), and the matter density
    (Omega_DLA).

    Run modes and ``sub_dla`` flag
    -------------------------------
    The ``sub_dla`` parameter controls how ``model_posteriors`` columns are
    interpreted.  The processed file stores model posteriors with this layout:

      DLA run (sub_dla=True, default) — shape (num_qsos, 2 + max_dlas):
          col 0  → Null model (no absorber)
          col 1  → Sub-DLA model (log NHI ∈ [19, 20.3])
          col 2  → DLA(1): exactly 1 DLA
          col 3  → DLA(2): exactly 2 DLAs
          col 4  → DLA(3): exactly 3 DLAs

      Sub-DLA / LLS run (sub_dla=False) — shape (num_qsos, 1 + max_dlas):
          col 0  → Null model (no absorber)
          col 1  → DLA(1): 1 absorber

    With sub_dla=True (default):
        p_DLA    = model_posteriors[:, 2:]   # DLA columns only
        p_no_DLA = model_posteriors[:, :2]   # Null + Sub-DLA

    With sub_dla=False:
        p_DLA    = model_posteriors[:, 1:]
        p_no_DLA = model_posteriors[:, :1]

    Quality filters
    ---------------
    Spectra are filtered by:
      - SNR > snr_thresh  (set via ``set_snr(snr_thresh)``; default -2 = no cut)
      - z_DLA search range > z_dla_minimum (avoids degenerate short windows)
      - proximity zone: exclude absorption within ``proximity_zone`` dz of QSO
      - tail zone: exclude absorption within ``tail_zone`` dz of spectrum start
      - Optionally: Lyman-beta forest region, minimum observed wavelength cut,
        high N_HI cut (log NHI > 22, for modeling quality)

    Parameters
    ----------
    processed_file : str
        HDF5 output from the GP-DLA inference pipeline.  Must contain:
        ``model_posteriors``, ``sample_log_likelihoods_dla``,
        ``log_likelihoods_dla``, ``min_z_dlas``, ``max_z_dlas``,
        ``z_qsos``, ``snrs``, ``target_ids``.
    sample_file : str
        QMC sample grid (.mat HDF5).  Must contain:
        ``offset_samples`` (shape N×1) and ``log_nhi_samples`` (shape N×1).
        Use ``dla_samples_a03.mat`` for DLA runs (Ho+2020); use
        ``gpy_dla_detection.generate_samples`` output for sub-DLA/LLS runs.
    catalog_file : str
        FITS QSO catalog with ``TARGETID`` column.  Used to align
        target IDs between the processed file and the reference catalog.
    snr : int, optional
        Minimum SNR threshold.  Default -2 means no SNR cut.
        Call ``set_snr(snr_thresh)`` to change after construction.
    lowzcut : bool, optional
        If True (default), exclude DLA candidates within ``proximity_zone``
        dz of the QSO redshift (removes proximity-zone contamination).
    highzcut : bool, optional
        If True (default), exclude DLA candidates within ``tail_zone``
        dz of the minimum search redshift (removes tail artifacts).
    second : int or bool, optional
        Maximum k for multi-DLA models to include in statistics.
        False or 0: only DLA(1); 1: DLA(1) + DLA(2); 2: up to DLA(3).
        Loads additional sample caches for each DLA(k) model.
    sub_dla : bool, optional
        If True (default), the processed file includes a Sub-DLA column in
        ``model_posteriors`` (col 1).  Set to False for single-absorber
        (sub-DLA/LLS) runs where the layout shifts.
    occams_razor : int, optional
        Additional Occam's razor penalty applied to DLA/Sub-DLA model
        posteriors.  Default 1 means no additional penalty.  Higher values
        penalize multi-DLA models more aggressively.
    z_dla_minimum : float, optional
        Minimum required width (in z) of the DLA search window.  Spectra
        with a shorter window are excluded.  Default 0.1.
    z_max_lyb : bool, optional
        If True, restrict the DLA search to the Ly-limit to Ly-beta range
        (excludes absorption redward of the Ly-beta wavelength).
    z_min_lyb : bool, optional
        If True, restrict to the Ly-beta to Ly-alpha range.
    min_obs_wavelength_cut : bool, optional
        If True, exclude DLA candidates below ``min_obs_wavelength`` in
        observed wavelength.  Useful for removing blue-end artifacts.
    min_obs_wavelength : float, optional
        Minimum observed wavelength in Angstroms (default 4000 Å).
    high_nhi_cut : bool, optional
        If True (default), exclude QMC samples with log NHI > ``high_nhi_cut_value``.
        Reduces contamination from pathological large-NHI detections where
        the eBOSS-trained GP model may be unreliable.
    high_nhi_cut_value : float, optional
        Upper log NHI cut applied when ``high_nhi_cut=True`` (default 22.0).
    bins_per_z : int, optional
        Number of redshift bins per unit z for dN/dX and Omega_DLA plots
        (default 6).

    Attributes
    ----------
    p_dla : np.ndarray, shape (num_qsos,)
        Probability of at least one DLA in each spectrum,
        p(M_DLA(>=1) | D, z_QSO).
    p_no_dla : np.ndarray, shape (num_qsos,)
        Probability of no DLA (Null + Sub-DLA if sub_dla=True).
    model_posteriors : np.ndarray, shape (num_qsos, num_models)
        Re-normalized model posteriors after Occam's razor penalty.
        Column layout described in the class docstring above.
    log_norm_like_cache : dict
        Per-spectrum cache of normalized sample log-likelihoods for DLA(1):
        p(D | M, z_QSO, θ) / p(M | z_QSO, D) / num_dla_samples.
    z_offsets : np.ndarray, shape (num_dla_samples,)
        Halton sequence offsets in [0, 1] for DLA redshift sampling.
    lnhi_vals : np.ndarray, shape (num_dla_samples,)
        log10(N_HI) values for each QMC sample.
    """

    def __init__(
        self,
        processed_file: str = "/pscratch/sd/j/jibancat/desi-kibo-gpdla-nobal-2_15-7-nozwarn/processed-main-dark.h5",
        sample_file: str = "dla_samples_a03.mat",
        # raw file not needed
        # raw_file: str = "preloaded_qsos.mat",
        # no need for snrs files since this is in the processed file
        # snrs_file: str = "snrs_qsos_multi_meanflux_dr16q.mat",
        catalog_file: str = "catalog.fits",  # reduced from QSO_cat_kibo_main_dark_healpix_v3-altbal.fits
        snr: int = -2,
        lowzcut: bool = True,
        highzcut: bool = True,
        second: Union[int, bool] = False,
        sub_dla: bool = True,
        occams_razor: int = 1,
        z_dla_minimum: float = 0.1,
        # raw_distfile: str = "DR16Q_v4.fits",  # DR16Q only
        # zestimate_cut: bool = False,  # DR16Q only; remove zestimate disagreements
        # delta_z_qso: float = 0.1,  # DR16Q only; remove zestimate disagreements
        # is_qso_final_cut: bool = False,  # DR16Q only; only take final QSO samples
        # class_person_cut: bool = False,  # DR16Q only; only take non-BAL samples
        # z_source_cut: bool = False,  # DR16Q only; remove source_z='pipe' and z > 5
        z_max_lyb: bool = False,  # Lylimit only: in case you want to shift the maximum search range to lyb peak
        z_min_lyb: bool = False,  # Lya only: in case you want to shift the minimum search range to lyb peak
        min_obs_wavelength_cut: bool = False,  # Cut out the tail part below certain obs lambda, default 4000 A
        min_obs_wavelength: float = 4000,  # A
        high_nhi_cut: bool = True,  # Cut out the high NHI samples
        high_nhi_cut_value: float = 22.5,  # log10(cm^-2)
        bins_per_z: int = 6, # number of bins of dNdX or Omega_DLA to plot per unit z interval
        window: "Optional[WindowSpec]" = None,  # shared search-window spec (None = legacy behaviour)
    ):
        # Should we include the second DLA?
        self.second_dla = (
            second  # False or 0: DLA(1); True or 1: DLA(2); 2: DLA(3); ...; k-1: DLA(k)
        )

        # Does model_posteriors include a Sub-DLA column?
        # -------------------------------------------------------
        # DLA run (sub_dla=True):
        #   model_posteriors columns: [Null, SubDLA, DLA(1), DLA(2), ...]
        #                              [  0,      1,      2,      3, ...]
        #   p_DLA    = model_posteriors[:, 2:]   (DLA columns start at index 2)
        #   p_no_DLA = model_posteriors[:, :2]   (Null + SubDLA)
        #
        # Sub-DLA / LLS run (sub_dla=False):
        #   model_posteriors columns: [Null, DLA(1)]
        #                              [  0,      1]
        #   p_DLA    = model_posteriors[:, 1:]
        #   p_no_DLA = model_posteriors[:, :1]
        #
        # The general rule used throughout this class is:
        #   DLA column k starts at index: k + sub_dla
        #   p_DLA  := model_posteriors[:, 1 + sub_dla:]
        #   p_sub_dla (if present) := model_posteriors[:, 1]
        self.sub_dla = sub_dla

        # the Occam's implementation for different DLA models is in self._log_norm_like
        # the additional Occam's razor implementation is in self.renormalise_occams_razor
        self.occams_razor = occams_razor

        # [min_z_dla] the minimum requirement for the sampling range of zDLA;
        # sometimes the sampling range is too small.
        self.z_dla_minimum = z_dla_minimum

        # Spectra with a DLA probability below this value are assumed to have p = 0, as an optimization.
        # Can be set as high as 0.1 without changing results much.
        # Can be increased, but never decreased
        self.p_thresh_spec = 5e-2
        # This excludes *samples* whose probability is below this value
        self.p_thresh_sample = 1e-4
        # p value to switch from the Poisson approximation to direct summation.
        # 0.25 is the value given in Le Cam 1960. In practice 0.5 seems not terrible.
        self.p_switch = 0.25
        # Exclude spectra closer to the DLA than this, which has fewer DLAs than average.
        self.lowzcut = lowzcut
        self.proximity_zone = 0.1  # 30000 km/s
        # Exclude spectra closer to the tail of the spectrum, which has more dubious DLAs than average.
        self.highzcut = highzcut
        self.tail_zone = 0.1  # 30000 km/s
        # Exclude spectra between lymanbeta to lymanalpha
        self.z_max_lyb = z_max_lyb
        # Exclude spectra between lymanlimit to lymanbeta
        self.z_min_lyb = z_min_lyb  # TODO implement it

        # [Shared WindowSpec] When a window is supplied it becomes the single
        # source of truth for the proximity/tail cut and the Lyβ edges. We keep
        # the legacy constant-0.1 path EXACTLY when ``window is None`` so the
        # existing CDDF numbers (golden + 131 tests) are byte-identical.
        self.window = window
        if window is not None:
            if window.velocity_scaled:
                raise NotImplementedError(
                    "WindowSpec(velocity_scaled=True) would require re-running "
                    "inference with a matching kms_to_z; it does not match the "
                    "existing posteriors' stored min_z_dlas/max_z_dlas."
                )
            # CRITICAL: the stored min_z_dlas/max_z_dlas ALREADY encode the inference
            # proximity/tail cut (set_parameters.kms_to_z(v_prox)). So we must NOT
            # re-apply proximity/tail here (that double-cuts: z_qso - 2*v/c) and must
            # NOT force lowzcut/highzcut (they also trip the lyb-branch asserts). The
            # WindowSpec's ONLY measurement-side effect is the Lyα-only / Lyman-limit
            # edge selection; the proximity is the stored edge, and cddf_mock reproduces
            # that stored edge (z_qso - v/c off raw z_qso) from the SAME spec.
            self.z_min_lyb = window.z_min_lyb
            self.z_max_lyb = window.z_max_lyb
            # The lyb branches assert highzcut/lowzcut == False (and the stored edges
            # already carry the tail/proximity), so disable the conflicting cut to keep
            # a window with a lyb mode self-consistent regardless of the ctor default.
            if window.z_min_lyb:
                self.highzcut = False
            if window.z_max_lyb:
                self.lowzcut = False
        # Exclude the dubious part of the obs wavelengths
        self.min_obs_wavelength_cut = min_obs_wavelength_cut
        self.min_obs_wavelength = min_obs_wavelength  # A
        # Exclude the high NHI samples
        self.high_nhi_cut = high_nhi_cut
        self.high_nhi_cut_value = high_nhi_cut_value  # log10(cm^-2)

        # self.raw_file = raw_file
        self.processed_file = processed_file
        self.catalog_file = catalog_file
        self.tophat_prior = False

        # Load data from the file
        self.filehandle = h5py.File(processed_file, "r")
        # First load small arrays
        self._z_min = self.filehandle["min_z_dlas"][:]
        self._z_max = self.filehandle["max_z_dlas"][:]
        self.z_qsos = self.filehandle["z_qsos"][:]

        # self.test_ind = self.filehandle["test_ind"][0, :].astype(np.bool)
        # Index of each spectrum in the file containing the flux: raw_file
        # It's the indices we selected from `preloaded_qsos.mat` based on our `test_ind`

        # Target IDs of the spectra
        self.target_ids = self.filehandle["target_ids"][:]

        # BAL catalog file : the reference file for processing DLAs
        # Find the corresponding index in the BAL catalog
        catalog = Table.read(catalog_file)
        target_ids_catalog = catalog["TARGETID"].data.astype(int)

        # Step 1: Create a mapping of target_ids to their positions in the reference order
        order_mapping = {val: idx for idx, val in enumerate(target_ids_catalog)}
        # Prevent situation where the run fails, which returns -1
        order_mapping[-1] = -1  # TODO: Make sure those -1 are not valid data

        # Step 2: Generate sorting indices for target_ids_to_sort
        real_index = np.array([order_mapping[val] for val in self.target_ids])

        self.real_index = real_index

        # number of bins of dNdX or Omega_DLA to plot per unit z interval
        self.bins_per_z = bins_per_z
        # Exclude things which have a low SNR. This is tested to be converged on DR7.
        self.filter_noisy_pixels = False
        self.noise_thresh = 0.5**2

        # Check if the `.mat` file was saved from MATLAB format matrix
        # DESI Y3: snrs are saved in the processed file
        self.snrs = self.filehandle["snrs"][:]

        if self.filter_noisy_pixels:
            self.pixel_noise = self.snrs  # TODO: placeholder

        self.set_snr(snr)
        self.do_resample = False
        # This allows us to filter by quasar redshift later
        self.condition = np.ones_like(self._z_min, dtype=bool)
        # filter out those detection with target_ids not in the DLA catalog
        self.condition = self.condition * (self.real_index != -1)

        # [Occam's razor] set up model_posteriors attr and put an additional occam's razor
        self.renormalise_occams_razor(occams_razor=self.occams_razor)

        # get dla attributes
        self.get_first_dla_attrs()

        if self.second_dla:
            # get DLA attrs up to Model DLA(k)
            for k in range(2, second + 2):
                self.get_kth_dla_attrs(k)

        # Load samples
        samplefilehandle = h5py.File(sample_file, "r")
        # Get the redshift of each sample
        self.z_offsets = samplefilehandle["offset_samples"][:, 0]
        # Get the value of NHI at each sample: we do not want to include samples with a column density below the cut.
        self.lnhi_vals = samplefilehandle["log_nhi_samples"][:, 0]
        samplefilehandle.close()

    # [Occam's razor] an additional occam's razor to penalise the DLA/subDLA detections
    def renormalise_occams_razor(self, occams_razor=10000):
        """Re-normalize model posteriors with an additional Occam's razor penalty.

        Applies a penalty factor to DLA/Sub-DLA model posteriors to suppress
        detections and test robustness against DLA over-counting:

            P'(M_DLA | D) ∝ P(M_DLA | D) / occams_razor
            P'(M_null | D) ∝ P(M_null | D)  [unchanged]

        The posteriors are then re-normalized so they sum to 1.

        Note: The Occam's penalty in ``sample_log_likelihoods_dla`` cancels
        with the same factor in ``log_likelihoods_dla`` when computing
        ``log_norm_like``, so normalizing the sample likelihoods here is not
        necessary — only ``model_posteriors`` and the derived ``p_dla`` need
        updating.

        Default occams_razor=1 means no additional penalty (identity transform).
        The original pipeline uses occams_razor=10000 for aggressive suppression.

        Parameters
        ----------
        occams_razor : int or float
            Penalty factor applied to all non-Null model posteriors (default 10000).
            1 means no penalty.  Higher values make DLA detection harder.
        """
        # TODO: it's assumed to be re-calculated for the filtered samples
        self.model_posteriors = self.filehandle["model_posteriors"][()]
        self.model_posteriors = self._occams_model_posteriors(
            self.model_posteriors, occams_razor
        )

        self.p_dla = self.model_posteriors[:, 1 + self.sub_dla :].sum(axis=1)
        self.p_no_dla = self.model_posteriors[:, : 1 + self.sub_dla].sum(axis=1)

    def _occams_model_posteriors(self, model_posteriors, occams_razor=10000):
        """
        re-calculate the model posteriors based on an additional occams_razor penalty

        P(DLA | D) = P(DLA | D) / occams_razor
                    / ( P(noDLA | D) + P(DLA | D) / occams_razor + P(subDLAs | D) / occams_razor  )

        Parameters:
        ----
        model_posteriors (np.ndarray) : shape (num_qsos, sub_dla + noDLA + k DLAs)

        """
        # all subDLAs + DLAs needed to be normalised
        model_posteriors[:, 1:] = model_posteriors[:, 1:] / occams_razor

        # calculate normalisation factor
        normalisation = (
            np.sum(model_posteriors, axis=1)[:, None]
            * np.ones(model_posteriors.shape[1])[None, :]
        )

        model_posteriors = model_posteriors / normalisation

        # [condition] filter out NaN p_dlas
        condition = ~np.isnan(np.sum(model_posteriors, axis=1))
        self.condition = self.condition * condition

        assert np.all(
            (
                (0.8 < np.sum(model_posteriors, axis=1))
                * (np.sum(model_posteriors, axis=1) < 1.2)
            )[condition]
        )
        return model_posteriors

    def get_first_dla_attrs(self):
        """
        get the attributes for DLA(1)
        """
        # Probability of at least one DLA in each spectrum
        # self.p_dla = self.filehandle["p_dlas"][0]
        # already assigned in the occams razor method

        # First do the DLA1 likelihoods
        # Load normalization constant for the DLA likelihoods
        self.log_norm_like_cache = {}
        dla_ind = self.filter_dla_spectra(second=False)
        if len(np.shape(self.filehandle["sample_log_likelihoods_dla"])) > 2:
            log_norm_like = self.filehandle["sample_log_likelihoods_dla"][
                :, :, 0
            ].T  # DESI: (num_qsos, num_samples, k)
        else:
            log_norm_like = self.filehandle["sample_log_likelihoods_dla"][:].T
        # Normalize by the total likelihood of a DLA in each spectrum, so that sum_spectrum ( like) == 1
        # Each DLA in a spectrum is a different column
        log_dla_like = self.filehandle["log_likelihoods_dla"][
            :, 0
        ]  # DESI: shape (num_qsos, k)
        # log_norm_like -= (log_dla_like + np.log(np.shape(self.log_norm_like)[0]))
        for spec in dla_ind[0]:
            # prevent IndexError while using a small test set for sample_log_likelihoods
            try:
                self.log_norm_like_cache[spec] = np.array(
                    log_norm_like[:, spec]
                    - (log_dla_like[spec] + np.log(np.shape(log_norm_like)[0]))
                )
            except IndexError as e:
                print("The sizes of dla_ind and log_norm_like don't match!")
                print(e)
                break
        del log_norm_like
        del log_dla_like

    def get_kth_dla_attrs(self, k=2):
        """
        get the attributes for DLA(k)

        Parameters:
        ----
        k (int) : the DLA(k) model, exactly k DLAs in each spectrum, you want to consider.
            default k = 2.

        Get Attributes:
        ----
        p_dla_k (np.ndarray) : model posterior, p(Mdla(k)|D,zqso), for each spectrum
        log_norm_like_k_cache (dict{np.ndarray}) : log sample likelihoods, p(D|Mdla(k),θ,zqso),
            for each spectrum
        base_sample_inds_k_cache (dict{np.ndarray}) : base sample inds resampled from Halton sequence
            of `sample_z_dlas` and `log_nhi_samples` in `generate_dla_samples.m`
        """
        # Probability of exactly k DLAs in each spectrum
        # `model_posteriors` := (no dla, sub dla, dla(1), dla(2), ...)
        #                      [     0,       1,      2,      3, ...]
        assert k > 1 and isinstance(k, (int, np.integer))
        setattr(self, "p_dla_{}".format(k), self.model_posteriors[:, k + self.sub_dla])

        # Now build caches for the DLA2 likelihoods and base_sample values
        # second := k - 1 so that second == True for DLA(k=2)
        dla_ind_k = self.filter_dla_spectra(second=k - 1)
        # First the log_likelihood of DLA(k)
        log_norm_like_k_cache = {}
        log_norm_like_k = self.filehandle["sample_log_likelihoods_dla"][
            :, :, k - 1
        ].T  # DESI: (num_qsos, num_samples, k)
        for spec in dla_ind_k[0]:
            log_norm_like_k_cache[spec] = self._do_norm_log_norm_like_k(
                log_norm_like_k[:, spec], spec, k - 1
            )
        setattr(self, "log_norm_like_{}_cache".format(k), log_norm_like_k_cache)
        del log_norm_like_k
        # Build a cache for the base_sample_ind values we will use
        # mitigate the memory consumption by not loading the full matrix
        try:
            # if max_dlas > 2, the `base_sample_inds` file will have one additional axis
            base_sample_inds_k = self.filehandle["base_sample_inds"][
                :, k - 2, :
            ].T  # DESI: (num_qsos, k, num_samples)
        except TypeError as e:
            print(e)
            base_sample_inds_k = np.array(self.filehandle["base_sample_inds"][:].T)

        base_sample_inds_k_cache = {}
        for spec in dla_ind_k[0]:
            base_sample_inds_k_cache[spec] = np.array(base_sample_inds_k[:, spec]) - 1
        setattr(self, "base_sample_inds_{}_cache".format(k), base_sample_inds_k_cache)
        del base_sample_inds_k

    def resample(self, do_it=True, nspec=0):
        """Generate a new sample (with replacement) of the same size as the original."""
        assert not self.second_dla  # not implemented
        assert not self.filter_noisy_pixels  # not implemented
        # z_max, z_min, p_dla, snrs and log_norm_like will now be sampled from the new set.
        self.do_resample = do_it
        # Stop if we aren't resampling
        if not do_it:
            return
        # Get the new sample set
        if nspec == 0:
            nspec = np.size(self.p_dla)
        self._resample = np.empty(nspec, dtype=int)
        # Find the redshift above which there are only 5 DLAs,
        # so that we don't have overly small sized bins
        newmax = np.max(self._z_max) - 0.2
        while np.sum(self._z_max > newmax) * nspec / np.size(self.p_dla) < 10:
            newmax -= 0.2
        newmin = np.min(self._z_min) + 0.2
        while np.sum(self._z_min > newmin) * nspec / np.size(self.p_dla) < 10:
            newmin += 0.2
        # This extends the last bin over a wider redshift range
        z_bins = np.linspace(newmin, newmax, 10)
        z_bins[-1] = np.max(self._z_max)
        z_bins[0] = np.min(self._z_min)
        # Roughly preserve the redshift distribution of the quasars.
        # Because high redshift quasars are quite rare,
        # if we just resample entirely randomly we could end up with very few of them.
        total = 0
        for zm, zp in zip(z_bins[:-1], z_bins[1:]):
            ii = np.where(np.logical_and(self._z_max > zm, self._z_max <= zp))
            nthisbin = np.min(
                [
                    int(np.floor(np.size(ii) / np.size(self.p_dla) * nspec)),
                    nspec - total,
                ]
            )
            assert nthisbin >= 10
            rand = np.random.randint(0, nthisbin, nthisbin)
            self._resample[total : total + nthisbin] = ii[0][rand]
            total += nthisbin
        assert total == nspec

    def get_sample_errors(self, *, z_min=2, z_max=5, nsample=5):
        """Do a number of resamplings to get error bars on omega_dla and dNdX."""
        dndx_sample = []
        om_sample = []
        self.resample(True)
        for _ in range(nsample):
            (_, dNdX, _, _, _) = self.line_density(z_min=z_min, z_max=z_max)
            (_, omega_dla, _, _, _) = self.omega_dla_cddf(
                z_min=z_min, z_max=z_max, lnhi_nbins=15.0
            )
            om_sample.append(1000 * omega_dla)
            dndx_sample.append(dNdX)
            self.resample(True)
        self.resample(False)
        dndx_sample = np.array(dndx_sample)
        om_sample = np.array(om_sample)
        self.dndx_68_sample = np.array(
            (
                np.percentile(dndx_sample, 100 - 32 / 2, axis=0),
                np.percentile(dndx_sample, 32 / 2, axis=0),
            )
        )
        assert np.shape(self.dndx_68_sample)[1] == np.shape(dNdX)[0]
        self.dndx_95_sample = np.array(
            (
                np.percentile(dndx_sample, 100 - 5 / 2, axis=0),
                np.percentile(dndx_sample, 5 / 2, axis=0),
            )
        )
        self.omega_68_sample = np.array(
            (
                np.percentile(om_sample, 100 - 32 / 2, axis=0),
                np.percentile(om_sample, 32 / 2, axis=0),
            )
        )
        self.omega_95_sample = np.array(
            (
                np.percentile(om_sample, 100 - 5 / 2, axis=0),
                np.percentile(om_sample, 5 / 2, axis=0),
            )
        )
        self.omega_sample = np.median(om_sample, axis=0)
        self.dndx_sample = np.median(dndx_sample, axis=0)

    def plot_dndx_sample_errors(self, *, z_min=2, z_max=5, nsample=5):
        """Plot the sample errors"""
        try:
            self.dndx_68_sample
        except AttributeError:
            self.get_sample_errors(z_min=z_min, z_max=z_max, nsample=nsample)
        (z_cent, dNdX, dndx68, dndx95, xerrs) = self.line_density(
            z_min=z_min, z_max=z_max
        )
        plt.fill_between(z_cent, dndx95[:, 0], dndx95[:, 1], color="grey", alpha=0.5)
        yerr = (dNdX - dndx68[:, 0], dndx68[:, 1] - dNdX)
        plt.errorbar(z_cent, dNdX, yerr=yerr, xerr=xerrs, fmt="o", label="Total")
        yerr = (
            self.dndx_sample - self.dndx_68_sample[0, :],
            self.dndx_68_sample[1, :] - self.dndx_sample,
        )
        plt.errorbar(
            z_cent, self.dndx_sample, yerr=yerr, xerr=xerrs, fmt="o", label="Resampled"
        )
        plt.xlabel(r"z")
        plt.ylabel(r"dN/dX")
        plt.xlim(z_min, z_max)

    def plot_omega_sample_errors(self, *, z_min=2, z_max=5, nsample=5):
        """Plot the sample errors"""
        try:
            self.omega_68_sample
        except AttributeError:
            self.get_sample_errors(z_min=z_min, z_max=z_max, nsample=nsample)
        (z_cent, omega_dla, omega68, omega95, xerrs) = self.omega_dla_cddf(
            z_min=z_min, z_max=z_max
        )
        plt.fill_between(
            z_cent, 1000 * omega95[:, 0], 1000 * omega95[:, 1], color="grey", alpha=0.5
        )
        yerr = (
            1000 * omega_dla - 1000 * omega68[:, 0],
            1000 * omega68[:, 1] - 1000 * omega_dla,
        )
        plt.errorbar(
            z_cent, 1000 * omega_dla, yerr=yerr, xerr=xerrs, fmt="o", label="Total"
        )
        yerr = (
            self.omega_sample - self.omega_68_sample[0, :],
            self.omega_68_sample[1, :] - self.omega_sample,
        )
        plt.errorbar(
            z_cent, self.omega_sample, yerr=yerr, xerr=xerrs, fmt="o", label="Resampled"
        )
        plt.xlabel(r"z")
        plt.ylabel(r"$10^3 \times \Omega_\mathrm{DLA}$")
        plt.xlim(z_min, z_max)

    def _base_sample_inds(self, spec, k=2):
        """
        Load the base_sample index to look up NHI for the second DLA, for spectrum spec

        Parameters:
        ----
        spec (int) : qso_ind
        k (int)    : which model (M_dla(k)) you want to query the base inds, range from k = (2, max_dlas).
            default k = 2
        """
        try:
            return getattr(self, "base_sample_inds_{}_cache".format(k))[spec]
        except KeyError:
            # base_sample_inds starts off zero indexed and needs to be 1-indexed.
            try:
                # if max_dlas > 2, the base_sample_inds file will have one additional axis
                getattr(self, "base_sample_inds_{}_cache".format(k))[spec] = (
                    np.array(self.filehandle["base_sample_inds"][spec, k - 2, :]) - 1
                )  # DESI: (num_qsos, k, num_samples)

            except IndexError as e:
                print("max_dlas < 3")
                print(e)
                assert k == 2  # this situation happened only if max_dlas == 2
                getattr(self, "base_sample_inds_{}_cache".format(k))[spec] = (
                    np.array(self.filehandle["base_sample_inds"][spec, :]) - 1
                )  # DESI: (num_qsos, k, num_samples)

            return getattr(self, "base_sample_inds_{}_cache".format(k))[spec]

    def _log_norm_like(self, spec, *, second=False):
        """Get the probability (normalised likelihood) values for the samples in a particular spectrum from the disc"""
        # Loading this from the disc each time is unreasonably slow
        if self.do_resample:
            spec = self._resample[spec]
        if not second:
            try:
                return self.log_norm_like_cache[spec]
            except KeyError:
                if len(np.shape(self.filehandle["sample_log_likelihoods_dla"])) > 2:
                    log_norm_like = self.filehandle["sample_log_likelihoods_dla"][
                        spec,
                        :,
                        0,
                    ]  # DESI: (num_qsos, num_samples, k)
                else:
                    log_norm_like = self.filehandle["sample_log_likelihoods_dla"][
                        spec, :
                    ]  # DESI: (num_qsos, num_samples, k)
                # Normalize by the total likelihood of a DLA in each spectrum, so that sum_spectrum ( like) == 1
                # Each DLA in a spectrum is a different column
                log_dla_like = self.filehandle["log_likelihoods_dla"][spec, 0]
                log_norm_like -= log_dla_like + np.log(np.shape(log_norm_like)[0])
                self.log_norm_like_cache[spec] = log_norm_like
                assert 0.95 < np.sum(np.exp(log_norm_like)) < 1.05
                return log_norm_like
        # Or get for the second DLA:
        # We will want P(DLA @ q = (N,z)) = P(n_DLA >= 1) P(DLA1 @ q) + P(n_DLA == 2) P(DLA2 @ q)
        # and P(DLA2 @ q ) = sum(DLA1 @ q') P(DLA1 @ q' and DLA2 @ q | data ) P(DLA1 @ q' | data)
        #                  = sum(DLA1 @ q') P(data | DLA1 @ q' and DLA2 @ q ) P( data | DLA1 @ q' ) P(DLA2 @ q | DLA1 @ q') P(DLA1 @ q')
        # P( data | DLA1 @ q' )  is sample_log_likelihood_dla[0] P(data | DLA1 @ q' and DLA2 @ q ) is sample_log_likelihood_dla[1].
        # Note that the parameters for DLA1 sample_log_likelihood_dla[1] are the same as sample_log_likelihood_dla[0]
        # So this sum is over all DLA1 samples.
        # then we have P(DLA2 @ q | DLA1 @ q') == P(DLA1 @ q') == 1/Nsample
        # The parameters of DLA1 are sample j are nhi[j], z[j]
        # The parameters of DLA2 are spectrum dependent and given by nhi[base_sample_inds[i,j]], z[base_sample_inds[i, j]]
        # Mask out nan values by making them very low probability: these correspond to samples where the DLAs are too close.
        try:
            return getattr(self, "log_norm_like_{}_cache".format(int(second) + 1))[
                spec
            ]
            # return self.log_norm_like_2_cache[spec]
        except KeyError:
            # log_nhi_like_k (np.ndarray) : dimension, (k-1, num_dla_samples)
            # if it is a DLA(2) model, we will still get a 2-dim array with shape == (1, num_dla_samples)
            # so that we can sum(axis=0) to eliminate the 0th axis.
            # log_nhi_like_k = self.filehandle["sample_log_likelihoods_dla"][1:int(second) + 1, :, spec]
            log_nhi_like_k = self.filehandle["sample_log_likelihoods_dla"][
                spec,
                :,
                int(second),
            ]  # DESI: (num_qsos, num_samples, k)
            getattr(self, "log_norm_like_{}_cache".format(int(second) + 1))[spec] = (
                self._do_norm_log_norm_like_k(log_nhi_like_k, spec, int(second))
            )
            # self.log_norm_like_2_cache[spec] = self._do_norm_log_norm_like_2(log_nhi_like, spec)
            return getattr(self, "log_norm_like_{}_cache".format(int(second) + 1))[
                spec
            ]

    def _do_norm_log_norm_like_k(self, log_nhi_like_k, spec, second):
        """
        Compute the normalized probabilities for DLA(k) samples from the likelihood values for a spectrum.

        Parameters:
        ----
        log_nhi_like_k (np.ndarray) : (max_num_dlas - 1, num_dla_samples)
        spec (int)                  : quasar_ind
        """
        log_nhi_like_k[np.isnan(log_nhi_like_k)] = -1e30
        # log_norm_like_k = np.sum( log_nhi_like_k, axis=0 ) + self._log_norm_like(spec, second=False)
        log_dla_like_k = self.filehandle["log_likelihoods_dla"][
            spec, second
        ]  # DESI: shape (num_qsos, k)
        log_norm_like_k = log_nhi_like_k - (
            log_dla_like_k + np.log(np.shape(log_nhi_like_k)[0]) * (second + 1)
        )
        # Normalize so that the sum of these likelihoods is unity.
        # First add something so we don't underflow our floating points.
        # This has the bonus that for peaked distributions, the normalization constant will be basically one already.
        # log_norm_like_k -= np.max( log_norm_like_k )
        # norm = logsumexp(log_norm_like_k)
        # assert np.isfinite(norm)
        # log_norm_like_k -= norm
        return log_norm_like_k

    def filter_dla_spectra(self, *, second=False):
        """
        Find the spectra we are not interested in, because the probability of a DLA is below the desired threshold.
        Or because the SNR is insufficient
        """
        inds_p_thresh = self._p_dla(second=second) > self.p_thresh_spec
        inds_snr_thresh = self._filter_snr_spectra()
        ind_z_dlas = self._filter_z_dlas(self.z_dla_minimum)

        # select snrs with the same length as p_dla because it is possible we are running on a truncated file
        if len(inds_p_thresh) != len(inds_snr_thresh):
            print(
                "[Warning] log_likelihoods_dla ({}) and snr ({}) do not have the same size".format(
                    len(inds_p_thresh), len(inds_snr_thresh)
                )
            )
            inds_snr_thresh = inds_snr_thresh[: len(inds_p_thresh)]

        return np.where(inds_p_thresh * inds_snr_thresh * ind_z_dlas)

    def _filter_snr_spectra(self):
        """Helper function to get SNR mask."""
        snrs = self.snrs
        if self.do_resample:
            snrs = self.snrs[self._resample]
        return (snrs > self.snr_thresh) * self.condition

    def filter_snr_spectra(self):
        """Remove spectra whose SNR is below snr_thresh"""
        inds_snr_thresh = self._filter_snr_spectra()

        # select snrs with the same length as p_dla because it is possible we are running on a truncated file
        if len(self._p_dla()) != len(inds_snr_thresh):
            print(
                "[Warning] log_likelihoods_dla ({}) and snr ({}) do not have the same size".format(
                    len(self._p_dla()), len(inds_snr_thresh)
                )
            )
            inds_snr_thresh = inds_snr_thresh[: len(self._p_dla())]

        return np.where(inds_snr_thresh)

    def set_snr(self, snr_thresh):
        """Set the value of SNR to be used, loading the SNR array if needed"""
        self.snr_thresh = snr_thresh

    def _filter_z_dlas(self, z_dla_minimum: float = 0.1):
        """Filter out the spectra without enough sampling in zDLAs."""
        return ((self._z_max - self._z_min) > z_dla_minimum) * self.condition

    def filter_z_dlas(self, z_dla_minimum: float = 0.1):
        """Filter out the spectra without enough sampling in zDLAs."""
        return np.where(self._filter_z_dlas(z_dla_minimum))

    def _p_dla(self, *, second=False):
        """Get the probability of a DLA. If second=False, return the probabilities of at least one DLA in each spectrum.
        If second=True, return the probability of exactly two DLAs in each spectrum.
        If second=k, k is an integer, return the probability of exactly (k+1) DLAs in each spectrum.
        """
        assert second >= 0 and isinstance(second, (int, np.integer, bool, np.bool_))
        if not second:
            if self.do_resample:
                return self.p_dla[self._resample]
            return self.p_dla
        else:
            return getattr(self, "p_dla_{}".format(int(second) + 1))

    def z_max(self, spec=None):
        """Returns the maximum redshift of the quasar spectrum."""
        if spec is None:
            if self.do_resample:
                return self._z_max[self._resample]
            return self._z_max
        else:
            if self.do_resample:
                return self._z_max[self._resample[spec]]
            return self._z_max[spec]

    def z_min(self, spec=None):
        """Returns the minimum redshift of the quasar spectrum."""
        if spec is None:
            if self.do_resample:
                return self._z_min[self._resample]
            return self._z_min
        else:
            if self.do_resample:
                return self._z_min[self._resample[spec]]
            return self._z_min[spec]

    def path_length(self, z_min, z_max):
        """Compute the total comoving absorption path length dX searched for DLAs.

        The absorption distance X is the dimensionless comoving path length
        used to normalize DLA counts:

            dX = (1 + z)^2 * H_0 / H(z) * dz

        where H(z) = H_0 * sqrt(Omega_m (1+z)^3 + Omega_Lambda).
        This convention ensures that dN/dX is redshift-independent for a
        population with a constant comoving number density.

        Only spectra passing SNR and z_DLA range cuts contribute to dX.
        For each contributing spectrum, only the overlap of [z_min_dla, z_max_dla]
        with [z_min, z_max] is integrated.  Spectra whose entire path falls
        within [z_min, z_max] are accelerated via a pre-computed bin integral.

        Optional cuts applied before integration (controlled by instance flags):
          - lowzcut: truncate at proximity zone (exclude z > z_QSO - proximity_zone)
          - highzcut: truncate at tail zone (exclude z < z_min + tail_zone)
          - z_max_lyb: cap at Ly-beta wavelength (for Ly-limit to Ly-beta region)
          - z_min_lyb: floor at Ly-beta wavelength (for Ly-beta to Ly-alpha region)
          - min_obs_wavelength_cut: exclude below observed wavelength threshold

        Parameters
        ----------
        z_min : float
            Lower redshift bound of the integration bin.
        z_max : float
            Upper redshift bound of the integration bin.  Must be > z_min.

        Returns
        -------
        float
            Total dX for the path over which we searched for DLAs in [z_min, z_max].
        """
        assert z_min < z_max
        # Make a clean copy
        # Filter spectra that don't make the SNR cut
        ind = self._filter_snr_spectra() * self._filter_z_dlas(self.z_dla_minimum)
        max_z_dlas = np.array(self.z_max())[ind]
        min_z_dlas = np.array(self.z_min())[ind]
        # remove the lyman alpha forest region to test lyinf-lybeta detections
        if self.z_max_lyb:
            print("[Info] testing on the range lyinf-lybeta")
            max_z_dlas = np.max(
                [np.min([max_z_dlas, self.lymanbeta(max_z_dlas)], axis=0), min_z_dlas],
                axis=0,
            )
        # remove the lyman beta forest region to test lybeta-lya (Lyα-only) detections.
        # Floor the blue edge at the QSO Lyβ EMISSION redshift = lymanbeta(z_qso) (NOT
        # lymanbeta(min_z_dla), which is < min_z_dla and a no-op). This matches
        # cddf_mock.qso_blue_edge_to_z_abs(z_qso).
        if self.z_min_lyb:
            print("[Info] testing on the range lybeta-lya (Lyα-only)")
            z_qsos = np.array(self.z_qsos)[ind]
            min_z_dlas = np.minimum(
                np.maximum(min_z_dlas, self.lymanbeta(z_qsos)), max_z_dlas
            )
        # Increase the minimum redshift to remove spectra contaminated by the lyman beta forest.
        if self.lowzcut:
            print("[Info] testing z_max - proximity dz {}".format(self.proximity_zone))
            max_z_dlas = np.max(
                [np.min([max_z_dlas, self.proximity(max_z_dlas)], axis=0), min_z_dlas],
                axis=0,
            )
        if self.highzcut:
            print("[Info] test z_min + tail dz {}".format(self.tail_zone))
            min_z_dlas = np.min(
                [np.max([min_z_dlas, self.tail(min_z_dlas)], axis=0), max_z_dlas],
                axis=0,
            )
        if self.min_obs_wavelength_cut:
            # obs lambda to z sampling -> obs lambda / (1 + zQSO)
            z_obs_min = self.min_obs_wavelength / (lya_wavelength) - 1
            z_obs_min = np.ones_like(min_z_dlas) * z_obs_min
            print(
                "[Info] test exlcuding everything lower than obs lambda {}A".format(
                    self.min_obs_wavelength
                )
            )
            min_z_dlas = np.min(
                [np.max([min_z_dlas, z_obs_min], axis=0), max_z_dlas],
                axis=0,
            )

        assert np.all(max_z_dlas - min_z_dlas >= 0)
        # Filter spectra that aren't in our redshift range
        i2 = np.where(np.logical_and(min_z_dlas < z_max, max_z_dlas > z_min))
        max_z_dlas = max_z_dlas[i2]
        min_z_dlas = min_z_dlas[i2]
        total = 0
        # Shortcut for spectra which cross the whole bin
        whole_bin = np.logical_and(max_z_dlas > z_max, min_z_dlas < z_min)
        # Find spectra where all pixels pass noise cuts
        if self.filter_noisy_pixels:
            pixel_noise = self.pixel_noise[ind][i2]
            no_filters = np.array(
                [np.all(ftrns < self.noise_thresh) for ftrns in pixel_noise]
            )
            whole_bin = np.logical_and(whole_bin, no_filters)
        i3 = np.where(whole_bin)
        (tbin, err) = integrate.quad(path_length_int, z_min, z_max)
        total += np.size(i3) * tbin
        # Integrate only remaining spectra
        i3 = np.where(np.logical_not(whole_bin))
        max_z_dlas = max_z_dlas[i3]
        min_z_dlas = min_z_dlas[i3]
        if not self.filter_noisy_pixels:
            for zmin, zmax in zip(min_z_dlas, max_z_dlas):
                assert zmin <= zmax
                # Do the spectra
                pathzmax = np.min([z_max, zmax])
                pathzmin = np.max([z_min, zmin])
                (ans, err) = integrate.quad(path_length_int, pathzmin, pathzmax)
                total += ans
                assert err < 1e-6
        else:
            total += self._do_filtered_path(
                z_max, z_min, min_z_dlas, max_z_dlas, pixel_noise, no_filters, i3
            )
        # The total dX for the path length we looked in
        return total

    def _do_filtered_path(
        self, z_max, z_min, min_z_dlas, max_z_dlas, pixel_noise, no_filters, i3
    ):
        """Compute the path length for spectra where certain pixels have been filtered due to their SNR."""
        total = 0.0
        pixel_noise = pixel_noise[i3]
        no_filters = no_filters[i3]
        # Clamp remaining max and min to limits
        # max_z_dlas[np.where(max_z_dlas > z_max)] = z_max
        # min_z_dlas[np.where(min_z_dlas < z_min)] = z_min
        for zmin, zmax, pn, nf in zip(min_z_dlas, max_z_dlas, pixel_noise, no_filters):
            assert zmin < zmax
            # Do the spectra that have good noise properties
            pathzmax = np.min([z_max, zmax])
            pathzmin = np.max([z_min, zmin])
            if nf:
                (ans, err) = integrate.quad(path_length_int, pathzmin, pathzmax)
            # Do the others
            else:
                zzs = zmin + (zmax - zmin) * np.arange(np.size(pn)) / (np.size(pn) - 1)
                # This will contain a list of contiguous regions with good noise properties
                regions = []
                # Find the first pixel within the redshift range which has good noise.
                ii = np.where(np.logical_and(zzs >= pathzmin, pn < self.noise_thresh))
                if np.size(ii) == 0:
                    continue
                ii = ii[0][0]
                # As long as there is more spectrum to look at within our redshift range
                while np.logical_and(ii < np.size(pn) - 1, zzs[ii] <= pathzmax):
                    # Find the next pixel which exceeds the noise bound
                    ie = np.where(
                        np.logical_and(pn[ii:] > self.noise_thresh, zzs[ii:] < pathzmax)
                    )
                    # If no more pixels exceed the noise bound, exit the loop
                    if np.size(ie) == 0:
                        regions += [(zzs[ii], pathzmax)]
                        break
                    # If this pixel exists, mark it as the end of the region
                    ie = ie[0][0] + ii
                    regions += [(zzs[ii], zzs[ie - 1])]
                    # Find the start of the next regions with low noise
                    ind = np.where(pn[ie:] < self.noise_thresh)
                    # If it doesn't exist, exit the loop
                    if np.size(ind) == 0:
                        break
                    ii = ind[0][0] + ie
                ans = 0
                err = 0
                # Do it piecewise: first argument is the start of each bin, second is the end.
                for zrr in regions:
                    (a1, e1) = integrate.quad(path_length_int, zrr[0], zrr[1])
                    ans += a1
                    err += e1
            total += ans
            assert err < 1e-6
        return total

    def column_density_function(
        self, z_min=1.0, z_max=6.0, lnhi_nbins=30, lnhi_min=20.0, lnhi_max=23.0
    ):
        """Compute the HI column density distribution function (CDDF), f(N_HI).

        The CDDF is defined as the number of DLA absorbers per sightline per
        unit column density per unit absorption distance:

            f(N_HI) = d²n_DLA / (dN_HI dX)
                     = n_DLA(N_HI, N_HI + dN_HI) / dN_HI / dX

        where:
          - n_DLA is the Bayesian expected number of DLAs with N_HI in each bin
            (sum of DLA probabilities weighted by the QMC sample distribution)
          - dN_HI is the linear bin width  (10^lnhi_max − 10^lnhi_min per bin)
          - dX = path_length(z_min, z_max) is the total comoving path searched

        Units: cm² (N_HI in cm⁻², X dimensionless).

        68% and 95% Bayesian credible intervals are computed via the
        Poisson-binomial method (see ``_get_confidence_intervals``).

        Parameters
        ----------
        z_min : float
            Lower redshift bound (default 1.0).
        z_max : float
            Upper redshift bound (default 6.0).
        lnhi_nbins : int
            Number of log10(N_HI) bins (default 30).
        lnhi_min : float
            Lower log10(N_HI) bound (default 20.0).
        lnhi_max : float
            Upper log10(N_HI) bound (default 23.0).

        Returns
        -------
        l_Ncent : np.ndarray
            Bin-centre log10(N_HI) values.
        cddf : np.ndarray
            MAP f(N_HI) in each bin [cm²].
        cddf68 : np.ndarray, shape (nbins, 2)
            68% credible interval [lower, upper] for f(N_HI).
        cddf95 : np.ndarray, shape (nbins, 2)
            95% credible interval [lower, upper] for f(N_HI).
        xerrs : tuple of np.ndarray
            (lower_xerr, upper_xerr) for N_HI bin widths.
        """
        # Get the NHI bins
        l_nhi = np.linspace(lnhi_min, lnhi_max, num=lnhi_nbins + 1)
        # Get the mean and variance of the probability distribution of DLAs.
        (ndlas, l68, l95) = self._get_confidence_intervals(
            q_bins=l_nhi, lred=z_min, ured=z_max, lnhi_min=lnhi_min, nhi=True
        )
        dX = self.path_length(z_min, z_max)
        dN = np.array(
            [10**lnhi_x - 10**lnhi_m for (lnhi_m, lnhi_x) in zip(l_nhi[:-1], l_nhi[1:])]
        )
        cddf = np.array(ndlas) / dX / dN
        # Broadcasting failure
        cddf68 = np.array(l68) / dX / np.vstack([dN, dN]).T
        cddf95 = np.array(l95) / dX / np.vstack([dN, dN]).T
        l_Ncent = np.array(
            [(lnhi_x + lnhi_m) / 2.0 for (lnhi_m, lnhi_x) in zip(l_nhi[:-1], l_nhi[1:])]
        )
        xerrs = (10**l_Ncent - 10 ** l_nhi[:-1], 10 ** l_nhi[1:] - 10**l_Ncent)
        return (l_Ncent, cddf, cddf68, cddf95, xerrs)

    def column_density_function_counts(
        self, z_min=1.0, z_max=6.0, lnhi_nbins=30, lnhi_min=20.0, lnhi_max=23.0
    ):
        """COUNT-SPACE CDDF accessor (additive; for the O3 diagonal correction).

        Surfaces the per-bin Poisson-binomial expected count (MAP) + 68/95 COUNT
        intervals the estimator already computes internally in
        ``_get_confidence_intervals`` — *before* the ΔN·ΔX normalization that
        ``column_density_function`` applies.  This is the count basis the O3
        diagonal soft-completeness correction operates in (``(F − b_FP)/C``); the
        same ``ΔN``/``ΔX`` are returned so a caller can re-normalize back to f(N).

        ADDITIVE: this method does NOT change ``column_density_function``'s output.
        Re-normalizing the returned counts by ``counts / dX / dN`` reproduces the
        O1 f(N) byte-identically (pinned by ``test_cddf_count_accessor``).

        Returns
        -------
        dict
            ``logN``      : (nbins,) bin-centre log10(N_HI);
            ``counts``    : (nbins,) MAP expected DLA count per bin;
            ``counts68``  : (nbins, 2) 68% COUNT interval [lo, hi];
            ``counts95``  : (nbins, 2) 95% COUNT interval [lo, hi];
            ``dN``        : (nbins,) linear N_HI bin width;
            ``dX``        : float, total absorption path length over [z_min, z_max].
        """
        l_nhi = np.linspace(lnhi_min, lnhi_max, num=lnhi_nbins + 1)
        (ndlas, l68, l95) = self._get_confidence_intervals(
            q_bins=l_nhi, lred=z_min, ured=z_max, lnhi_min=lnhi_min, nhi=True
        )
        dX = self.path_length(z_min, z_max)
        dN = np.array(
            [10**lnhi_x - 10**lnhi_m for (lnhi_m, lnhi_x) in zip(l_nhi[:-1], l_nhi[1:])]
        )
        l_Ncent = np.array(
            [(lnhi_x + lnhi_m) / 2.0 for (lnhi_m, lnhi_x) in zip(l_nhi[:-1], l_nhi[1:])]
        )
        return {
            "logN": l_Ncent,
            "counts": np.array(ndlas),
            "counts68": np.array(l68),
            "counts95": np.array(l95),
            "dN": dN,
            "dX": dX,
        }

    def plot_cddf(
        self, zmin=1.0, zmax=6.0, label="GP", color=None, moment=False, twosigma=True,
        lnhi_nbins=30, lnhi_min=20.0, lnhi_max=23.0
    ):
        """Plot the column density function"""
        (l_N, cddf, cddf68, cddf95, xerrs) = self.column_density_function(
            z_min=zmin, z_max=zmax, lnhi_nbins=lnhi_nbins, lnhi_min=lnhi_min, lnhi_max=lnhi_max
        )
        if moment:
            cddf *= 10**l_N
            for x in (0, 1):
                cddf68[:, x] *= 10**l_N
                cddf95[:, x] *= 10**l_N
        # 2 sigma contours.
        if twosigma:
            plt.fill_between(
                10**l_N, cddf95[:, 0], cddf95[:, 1], color="grey", alpha=0.5
            )
        yerr = (cddf - cddf68[:, 0], cddf68[:, 1] - cddf)
        ii = np.where(cddf68[:, 0] > 0.0)
        if np.size(ii) > 0:
            plt.errorbar(
                10 ** l_N[ii],
                cddf[ii],
                yerr=(yerr[0][ii], yerr[1][ii]),
                xerr=(xerrs[0][ii], xerrs[1][ii]),
                fmt="o",
                label=label,
                color=color,
            )
        i2 = np.where(cddf68[:, 0] == 0)
        if np.size(i2) > 0:
            plt.errorbar(
                10 ** l_N[i2],
                cddf[i2] + yerr[1][i2],
                yerr=yerr[1][i2] / 2.0,
                xerr=(xerrs[0][i2], xerrs[1][i2]),
                fmt="o",
                label=None,
                uplims=True,
                color=color,
                lw=2,
            )
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel(r"$N_\mathrm{HI}$ (cm$^{-2}$)")
        plt.ylabel(r"$f(N_\mathrm{HI})$")
        return (l_N, cddf, cddf68, cddf95)

    def line_density(self, z_min=2, z_max=4, lnhi_min=20.3, lnhi_max=23):
        """Compute the DLA line density dN/dX as a function of redshift.

        The line density (or incidence rate) is the expected number of DLAs
        per unit absorption distance:

            dN/dX(z) = sum_{spectra} P(DLA in [z, z+dz], log NHI in [lnhi_min, lnhi_max] | D)
                       / dX(z, z+dz)

        Default redshift bins and NHI limits are chosen to match
        Noterdaeme et al. (2012) for direct comparison.

        Parameters
        ----------
        z_min : float
            Lower redshift bound (default 2).
        z_max : float
            Upper redshift bound (default 4).
        lnhi_min : float
            Lower log10(N_HI) cut for counting DLAs (default 20.3 — DLA threshold).
        lnhi_max : float
            Upper log10(N_HI) cut (default 23).

        Returns
        -------
        z_cent : np.ndarray
            Bin-centre redshifts.
        dNdX : np.ndarray
            MAP dN/dX in each redshift bin.
        dndx68 : np.ndarray, shape (nbins, 2)
            68% credible interval [lower, upper] on dN/dX.
        dndx95 : np.ndarray, shape (nbins, 2)
            95% credible interval [lower, upper] on dN/dX.
        xerrs : tuple of np.ndarray
            (lower_xerr, upper_xerr) bin half-widths.
        """
        # Get the redshifts
        nbins = np.max([int((z_max - z_min) * self.bins_per_z), 1])
        z_bins = np.linspace(z_min, z_max, nbins + 1)
        # Get the mean and variance of the probability distribution of DLAs.
        (maxlike, l68, l95) = self._get_confidence_intervals(
            q_bins=z_bins, lred=z_min, ured=z_max, lnhi_min=lnhi_min, lnhi_max=lnhi_max, nhi=False
        )
        # Check the outputs are reasonably ordered.
        dX = np.array(
            [self.path_length(z_m, z_x) for (z_m, z_x) in zip(z_bins[:-1], z_bins[1:])]
        )
        ii = np.where(dX > 0)
        dX = dX[ii]
        dNdX = np.array(maxlike)[ii] / dX
        dndx68 = np.array(l68)[ii] / np.vstack([dX, dX]).T
        dndx95 = np.array(l95)[ii] / np.vstack([dX, dX]).T
        z_cent = np.array(
            [(z_x + z_m) / 2.0 for (z_m, z_x) in zip(z_bins[:-1], z_bins[1:])]
        )
        xerrs = (z_cent[ii] - z_bins[:-1][ii], z_bins[1:][ii] - z_cent[ii])
        return (z_cent[ii], dNdX, dndx68, dndx95, xerrs)

    def line_density_counts(self, z_min=2, z_max=4, lnhi_min=20.3, lnhi_max=23):
        """COUNT-SPACE dN/dX accessor (additive; for the O3 diagonal correction).

        Surfaces the per-redshift-bin Poisson-binomial MAP count + 68/95 COUNT
        intervals before the ``/dX`` normalization that ``line_density`` applies, so
        the O3 diagonal correction can operate in count space and re-normalize.

        ADDITIVE: does NOT change ``line_density``'s output.  Re-normalizing by
        ``counts / dX`` reproduces the O1 dN/dX byte-identically.

        Returns
        -------
        dict
            ``z``        : (nbins,) bin-centre redshifts (only dX>0 bins kept,
                           matching ``line_density``);
            ``counts``   : (nbins,) MAP expected DLA count per z bin;
            ``counts68`` : (nbins, 2) 68% COUNT interval;
            ``counts95`` : (nbins, 2) 95% COUNT interval;
            ``dX``       : (nbins,) absorption path length per z bin.
        """
        nbins = np.max([int((z_max - z_min) * self.bins_per_z), 1])
        z_bins = np.linspace(z_min, z_max, nbins + 1)
        (maxlike, l68, l95) = self._get_confidence_intervals(
            q_bins=z_bins, lred=z_min, ured=z_max, lnhi_min=lnhi_min,
            lnhi_max=lnhi_max, nhi=False,
        )
        dX = np.array(
            [self.path_length(z_m, z_x) for (z_m, z_x) in zip(z_bins[:-1], z_bins[1:])]
        )
        ii = np.where(dX > 0)
        dX = dX[ii]
        z_cent = np.array(
            [(z_x + z_m) / 2.0 for (z_m, z_x) in zip(z_bins[:-1], z_bins[1:])]
        )
        return {
            "z": z_cent[ii],
            "counts": np.array(maxlike)[ii],
            "counts68": np.array(l68)[ii],
            "counts95": np.array(l95)[ii],
            "dX": dX,
        }

    def plot_line_density(self, zmin=2, zmax=4, label="GP", lnhi_min=20.3, lnhi_max=23):
        """Plot the line density as a function of redshift"""
        (z_cent, dNdX, dndx68, dndx95, xerrs) = self.line_density(
            z_min=zmin, z_max=zmax, lnhi_min=lnhi_min, lnhi_max=lnhi_max
        )
        # 2 sigma contours.
        plt.fill_between(z_cent, dndx95[:, 0], dndx95[:, 1], color="grey", alpha=0.5)
        yerr = (dNdX - dndx68[:, 0], dndx68[:, 1] - dNdX)
        plt.errorbar(z_cent, dNdX, yerr=yerr, xerr=xerrs, fmt="o", label=label)
        plt.xlabel(r"z")
        plt.ylabel(r"dN/dX")
        plt.xlim(zmin, zmax)
        return (z_cent, dNdX, dndx68, dndx95)

    def omega_dla_cddf(self, z_min=2, z_max=4, hubble=0.7, lnhi_nbins=30, lnhi_min=20.3, lnhi_max=23.0):
        """Compute Omega_DLA as a function of redshift by integrating the CDDF.

        Omega_DLA is the neutral hydrogen mass density in DLAs relative to
        the critical density:

            Omega_DLA(z) = (m_p H_0 / c rho_c) * int_{N_min}^{N_max} N_HI f(N_HI) dN_HI / dX

        where:
          - m_p = proton mass (1.67262178e-24 g)
          - H_0 = hubble * 100 km/s/Mpc (converted to 1/s)
          - rho_c = critical density at z=0 (g/cm³; see ``rho_crit``)
          - f(N_HI) = CDDF (cm²)
          - dX = path length (dimensionless comoving distance)

        This is the CDDF-based estimator (integrating N * f(N) over the
        column density function), which gives full Bayesian credible intervals
        via the Poisson-binomial method.  See also ``omega_dla`` for the
        simpler histogram-based variance estimator.

        Parameters
        ----------
        z_min : float
            Lower redshift bound (default 2).
        z_max : float
            Upper redshift bound (default 4).
        hubble : float
            Hubble constant H_0 / (100 km/s/Mpc) (default 0.7).
        lnhi_nbins : int
            Number of log10(N_HI) bins for the CDDF integration (default 30).
        lnhi_min : float
            Lower log10(N_HI) integration limit (default 20.3 — DLA threshold).
        lnhi_max : float
            Upper log10(N_HI) integration limit (default 23.0).

        Returns
        -------
        z_cent : np.ndarray
            Bin-centre redshifts.
        omega_dla : np.ndarray
            MAP Omega_DLA in each redshift bin (dimensionless).
        omega_dla_68 : np.ndarray, shape (nbins, 2)
            68% credible interval [lower, upper].
        omega_dla_95 : np.ndarray, shape (nbins, 2)
            95% credible interval [lower, upper].
        xerrs : np.ndarray, shape (2, nbins)
            [lower_xerr, upper_xerr] for redshift bin widths.

        Notes
        -----
        Multiply by 1000 to plot as 10³ × Omega_DLA (common convention).
        """
        nbins = np.max([int((z_max - z_min) * self.bins_per_z), 1])
        z_bins = np.linspace(z_min, z_max, nbins + 1)
        protonmass = 1.67262178e-24
        # H0 in 1/s units
        h100 = 3.2407789e-18 * hubble
        # Speed of light in cm/s
        light = 2.99e10
        omega_dla = np.array([])
        omega_dla_68 = np.array([]).reshape(0, 2)
        omega_dla_95 = np.empty_like(omega_dla_68)
        xerrs = np.empty_like(omega_dla_68)
        z_cent = np.array([])
        conversion = protonmass / light * h100 / rho_crit(hubble)
        # Get the NHI bins
        # The old recipe used a fixed lower limit of 20.3 for DLAs; to adjust for LLS use lnhi_min here.
        # lnhi_bins = np.linspace(20.3, self.high_nhi_cut_value, num=lnhi_nbins + 1)
        lnhi_bins = np.linspace(lnhi_min, lnhi_max, num=lnhi_nbins + 1)
        for zz in range(nbins):
            dX = self.path_length(z_bins[zz], z_bins[zz + 1])
            if dX == 0.0:
                continue
            (nhi_like, nhi_68, nhi_95) = self._get_omega_confidence_intervals(
                lnhi_bins=lnhi_bins, lred=z_bins[zz], ured=z_bins[zz + 1]
            )
            # Check the outputs are reasonably ordered.
            assert nhi_95[0] <= nhi_68[0] <= nhi_like
            assert nhi_95[1] >= nhi_68[1] >= nhi_like
            # The 1+z factor converts lightspeed to comoving
            omega_dla = np.append(omega_dla, conversion * nhi_like / dX)
            omega_dla_68 = np.append(
                omega_dla_68, conversion * np.array(nhi_68).reshape(1, 2) / dX, axis=0
            )
            omega_dla_95 = np.append(
                omega_dla_95, conversion * np.array(nhi_95).reshape(1, 2) / dX, axis=0
            )
            z_c = (z_bins[zz] + z_bins[zz + 1]) / 2.0
            z_cent = np.append(z_cent, z_c)
            xerrs = np.append(
                xerrs,
                np.array([z_c - z_bins[zz], z_bins[zz + 1] - z_c]).reshape(1, 2),
                axis=0,
            )
        assert np.shape(omega_dla_68) == (np.shape(omega_dla)[0], 2)
        return (z_cent, omega_dla, omega_dla_68, omega_dla_95, xerrs.T)

    def _get_omega_confidence_intervals(
        self, lnhi_bins, lred=2.0, ured=4.0, tailprob=5e-4
    ):
        """
        Get the confidence interval on the total abundance of HI in DLAs in a given redshift range (this should be called for each bin in Omega_DLA).
        We do this be computing the CDDF in NHI bins and then summing the PDFs for each one.
        Returns: (maximum a posteriori likelihoods, lower 68 % confidence levels, upper 68% confidence levels, lower and upper 95 % confidence levels)
        """
        (probs, poissons) = self._split_distributions(
            lnhi_bins,
            lred=lred,
            ured=ured,
            lnhi_min=lnhi_bins[0],
            lnhi_max=lnhi_bins[-1],
            nhi=True,
        )
        # probs[i] now contains a list of arrays
        # Now we have built a list of probabilities in each z bin of interest and we want to solve for the Poisson binomial coefficients.
        # to get each combined pdf.
        # Empty pdf: P(NHI=0) = 1
        pdf_comb = np.ones(1)
        nhi_comb = np.zeros(1)
        # We could probably get more accuracy by doing some sort of interpolation and then integrating...
        nhi_cent = 10 ** np.array(
            [
                (lnhi_x + lnhi_m) / 2.0
                for (lnhi_m, lnhi_x) in zip(lnhi_bins[:-1], lnhi_bins[1:])
            ]
        )
        # Loop over bins in the column density function
        for pp, pmean, nhi_cc in zip(probs, poissons, nhi_cent):
            pdf = get_poisson_binomial_pdf(pp)
            # Get the pdf for this NHI bin
            (pdf_one_bin, offset_one_bin) = self._get_combined_levels(pdf, pmean)
            # If the last CDDF bin is consistent with zero, stop.
            if self.tophat_prior:
                (lowtest, _) = interval(np.cumsum(pdf_one_bin), 0.68)
                if lowtest < 1:
                    continue
            # Store the PDFs
            (dlow, dhigh) = interval(np.cumsum(pdf_one_bin), 1 - 1e-4)
            # We want to include dhigh, as long as it is in the array
            maxr = np.min([dhigh + 1, np.size(pdf_one_bin)])
            pdf_comb = np.ravel(
                np.array(
                    [
                        [pdf_comb[j] * pdf_one_bin[i] for i in range(dlow, maxr)]
                        for j in range(np.size(pdf_comb))
                    ]
                )
            )
            # Store the NHI values corresponding to each PDF
            nhi_comb = np.ravel(
                np.array(
                    [
                        [
                            nhi_comb[j] + (offset_one_bin + i) * nhi_cc
                            for i in range(dlow, maxr)
                        ]
                        for j in range(np.size(nhi_comb))
                    ]
                )
            )
            assert 1.01 > math.fsum(pdf_comb) > 0.99
            # Sort the pdf by increasing NHI
            sort = np.argsort(nhi_comb)
            nhi_comb = nhi_comb[sort]
            pdf_comb = pdf_comb[sort]
            # Now we want to shrink the arrays a little, by combining options within 1% of each other, as well as merging low-probability tails.
            # If we don't do this the array quickly gets out of hand.
            # First do tails.
            cdf = np.cumsum(pdf_comb)
            t1 = np.where(cdf < tailprob)
            t2 = np.where(cdf > 1 - tailprob)
            # Replace the last few points in this distribution with the sum of their pdfs
            if np.size(t2) > 0:
                pdf_comb = np.append(pdf_comb[: t2[0][0]], np.sum(pdf_comb[t2]))
                nhi_comb = np.append(nhi_comb[: t2[0][0]], np.min(nhi_comb[t2]))
            if np.size(t1) > 0:
                pdf_comb = np.insert(pdf_comb[t1[0][-1] + 1 :], 0, np.sum(pdf_comb[t1]))
                nhi_comb = np.insert(nhi_comb[t1[0][-1] + 1 :], 0, np.max(nhi_comb[t1]))
            assert 1.01 > math.fsum(pdf_comb) > 0.99
            # Now find options which are indistinguishable for all reasonable purposes
            new_pdf = [
                pdf_comb[0],
            ]
            new_nhi = [
                nhi_comb[0],
            ]
            # Here we need a 'real' for loop
            low_ind = 1
            while low_ind < np.size(pdf_comb) - 1:
                i3 = np.where(
                    np.logical_and(
                        nhi_comb[low_ind:-1] / nhi_comb[low_ind] < 1 + 1e-3,
                        np.cumsum(pdf_comb[low_ind:-1]) < pdf_comb[low_ind:-1] + 0.04,
                    )
                )
                new_pdf.append(math.fsum(pdf_comb[low_ind:-1][i3]))
                new_nhi.append(np.median(nhi_comb[low_ind:-1][i3]))
                low_ind += i3[0][-1] + 1
            # Add the last sample
            if np.size(pdf_comb) > 1:
                new_pdf.append(pdf_comb[-1])
                new_nhi.append(nhi_comb[-1])
            assert np.size(new_pdf) == np.size(new_nhi)
            assert np.abs(math.fsum(new_pdf) - math.fsum(pdf_comb)) < 1e-4
            pdf_comb = np.array(new_pdf)
            nhi_comb = np.array(new_nhi)
        # Unpack maximum likelihoods and 68/95% contours
        (maxlikes, levels68, levels95) = pdf_confidence(pdf_comb, 0)
        # Edge case
        if levels95[1] >= np.size(nhi_comb):
            levels95 = (levels95[0], levels95[1] - 1)
        return (
            nhi_comb[maxlikes],
            (nhi_comb[levels68[0]], nhi_comb[levels68[1]]),
            (nhi_comb[levels95[0]], nhi_comb[levels95[1]]),
        )

    def omega_dla(self, z_min=2, z_max=4, hubble=0.7, lnhi_max=23.0, lnhi_min=20.3):
        """Compute Omega_DLA by direct histogram summation (Gaussian error approximation).

        Alternative to ``omega_dla_cddf``.  Uses Gaussian error propagation
        (mean ± sqrt(variance)) instead of the full Poisson-binomial PDF:

            Omega_DLA(z) = (m_p H_0 / c rho_c) * sum_i( P_i * N_HI_i ) / dX

        where P_i is the per-sample DLA probability weighted by N_HI (the
        first moment).  The variance is sum_i( P_i * (1 - P_i) * N_HI_i² ).

        This method is faster but underestimates errors compared to the full
        Bayesian Poisson-binomial approach in ``omega_dla_cddf``.  Prefer
        ``omega_dla_cddf`` for publication-quality error bars.

        Parameters
        ----------
        z_min, z_max : float
            Redshift range (default 2–4).
        hubble : float
            H_0 / (100 km/s/Mpc) (default 0.7).
        lnhi_min, lnhi_max : float
            log10(N_HI) integration limits (default 20.3–23).

        Returns
        -------
        z_cent : np.ndarray
            Bin-centre redshifts.
        omega_DLA : np.ndarray
            Omega_DLA in each bin (dimensionless).
        err : np.ndarray
            Gaussian 1-sigma error estimates.
        z_bins : np.ndarray
            Redshift bin edges.
        """
        # Get the redshifts
        nbins = np.max([int((z_max - z_min) * self.bins_per_z), 1])
        z_bins = np.linspace(z_min, z_max, nbins + 1)
        # Get the mean and variance of the probability distribution of DLAs.
        (mean, variance) = self._get_z_nhi_hist(
            q_bins=z_bins,
            lred=z_min,
            ured=z_max,
            lnhi_min=lnhi_min,
            lnhi_max=lnhi_max,
            nhi=False,
            moment=True,
        )
        # This returns the total matter in DLAs at each redshift in atoms/cm^2.
        # Need to turn this into g/cm^2, divide by path length in (comoving) cm, and then divide by rho_crit.
        # proton mass in g
        protonmass = 1.67262178e-24
        dX = np.array(
            [self.path_length(z_m, z_x) for (z_m, z_x) in zip(z_bins[:-1], z_bins[1:])]
        )
        # H0 in 1/s units
        h100 = 3.2407789e-18 * hubble
        # Speed of light in cm/s
        light = 2.99e10
        conversion = protonmass * h100 / light / dX / rho_crit()
        omega_DLA = mean * conversion
        err = np.sqrt(variance) * conversion
        z_cent = np.array(
            [(z_x + z_m) / 2.0 for (z_m, z_x) in zip(z_bins[:-1], z_bins[1:])]
        )
        return (z_cent, omega_DLA, err, z_bins)

    def plot_omega_dla_var(self, zmin=2, zmax=4, label="GP", color=None):
        """Plot omega_DLA as a function of redshift, with errors given by (an approximation to) the distribution variance"""
        (z_cent, omega_DLA, err, z_bins) = self.omega_dla(z_min=zmin, z_max=zmax)
        xerrs = (z_cent - z_bins[:-1], z_bins[1:] - z_cent)
        plt.errorbar(
            z_cent,
            1000 * omega_DLA,
            yerr=1000 * err,
            xerr=xerrs,
            fmt="s",
            label=label,
            color=color,
        )
        plt.xlabel(r"z")
        plt.ylabel(r"$10^3 \times \Omega_\mathrm{DLA}$")

    def plot_omega_dla(self, zmin=2, zmax=4, label="GP", color=None, twosigma=True,
                       lnhi_nbins=30, lnhi_min=20.3, lnhi_max=23.0):
        """Plot omega_DLA as a function of redshift, with full Bayesian errors"""
        (z_cent, omega_dla, omega_dla_68, omega_dla_95, xerrs) = self.omega_dla_cddf(
            z_min=zmin, z_max=zmax,
            hubble=0.7, lnhi_nbins=lnhi_nbins, lnhi_min=lnhi_min, lnhi_max=lnhi_max,
        )
        if twosigma:
            plt.fill_between(
                z_cent,
                1000 * omega_dla_95[:, 0],
                1000 * omega_dla_95[:, 1],
                color="grey",
                alpha=0.5,
            )
        omega_dla *= 1000
        yerr = (
            omega_dla - 1000 * omega_dla_68[:, 0],
            1000 * omega_dla_68[:, 1] - omega_dla,
        )
        plt.errorbar(z_cent, omega_dla, yerr=yerr, xerr=xerrs, fmt="s", label=label)
        plt.xlabel(r"z")
        plt.ylabel(r"$10^3 \times \Omega_\mathrm{DLA}$")
        plt.xlim(zmin, zmax)
        return (z_cent, omega_dla, omega_dla_68, omega_dla_95)

    def _get_sample_params(self, spec, *, second=False):
        """Get the (n,z) values for each sample in this spectrum. spec is the spectrum number,
        second denotes whether to return the parameters of the second DLA.

        Parameters:
        ----
        spec (int)           : quasar_ind
        second (bool or int) : consider up to model DLA(second + 1)
        """
        # Compute redshift of each sample
        redshifts = (
            self.z_min(spec) + (self.z_max(spec) - self.z_min(spec)) * self.z_offsets
        )
        lnhi_vals = self.lnhi_vals
        # Get N,z values for this spectrum
        if second:
            base_sample = self._base_sample_inds(spec, k=second + 1)
            lnhi_vals = lnhi_vals[base_sample]
            redshifts = redshifts[base_sample]
        return (lnhi_vals, redshifts)

    def _get_prob_dla_this_bin(self, spec, index, *, second=False):
        """Get the probability of a DLA with the samples specified in index.

        Parameters:
        ----
        spec (int)           : quasar_ind
        index (int)          : the index of `sample_log_likelihoods_dla`
        second (bool or int) : consider up to model DLA(second + 1)
        """
        log_norm_posteriors = (
            np.exp(self._log_norm_like(spec, second=False)[index])
            * self._p_dla(second=False)[spec]
        )

        if second == False:
            return log_norm_posteriors
        else:
            log_norm_posteriors_k = np.empty(log_norm_posteriors.shape)
            log_norm_posteriors_k = -1e30

            for i in range(int(second) + 1):
                p_dla_k = self.model_posteriors[index, i + 1 + self.sub_dla]
                log_norm_posteriors_k += (
                    np.exp(self._log_norm_like(spec, second=second)[index]) * p_dla_k
                )

            return log_norm_posteriors_k

    def _split_distributions(
        self, q_bins, lred=2.0, ured=4.0, lnhi_min=20.3, lnhi_max=23.0, *, nhi=False
    ):
        """Split the distributions for both the first and second DLA, in turn
        and combine distributions up to kth DLA (k := second + 1).
        """
        (probs, poissons) = self._split_distributions_single(
            q_bins,
            lred=lred,
            ured=ured,
            lnhi_min=lnhi_min,
            lnhi_max=lnhi_max,
            nhi=nhi,
            second=False,
        )
        if self.second_dla:
            for k in range(2, self.second_dla + 2):
                (probs2, poissons2) = self._split_distributions_single(
                    q_bins,
                    lred=lred,
                    ured=ured,
                    lnhi_min=lnhi_min,
                    lnhi_max=lnhi_max,
                    nhi=nhi,
                    second=k - 1,
                )
                # List addition is concatenation, but we want a list of lists.
                probs = list(map(operator.add, probs, probs2))
                # Array addition is element-wise addition
                poissons += poissons2
        return (probs, poissons)

    def lymanbeta(self, zqso):
        """Compute the redshift at which the lyman beta forest at the redshift of the quasar will show up.

        Uses the UNIFIED Lyβ/Lyα rest wavelengths from ``set_parameters`` (imported
        via ``from .set_parameters import *``: ``lyb_wavelength = 1025.7223``,
        ``lya_wavelength = 1215.6701``) — the same values as
        ``gpy_dla_detection.set_parameters.Parameters.lyb_wavelength`` /
        ``cddf_mock.LYB_REST`` and the inference — instead of the legacy hard-coded
        ``1026.72 / 1215.67``. This shifts the Lyβ edge by ~0.0037 in z at z_qso=3.5
        (i.e. ``Δz ≈ (1+z)·(1026.72-1025.7223)/1215.67``), making all three CDDF
        pathways agree on the Lyβ blue edge.
        """
        waveratios = lyb_wavelength / lya_wavelength
        zlyb = (1 + zqso) * waveratios - 1
        return zlyb

    def proximity(self, zqso):
        """Remove a redshift range close to the quasar"""
        dz = self.proximity_zone
        return zqso - dz

    def tail(self, zmin):
        """Remove a redshift range close to the tail of the spectrum"""
        dz = self.tail_zone
        return zmin + dz

    def _split_distributions_single(
        self,
        q_bins,
        lred=2.0,
        ured=4.0,
        lnhi_min=20.3,
        lnhi_max=23.0,
        *,
        nhi=False,
        second=False
    ):
        """
        Split the sampled probabilities (in the desired bin) into two sets; those with small probabilities, for which we just keep the mean and sum of squares
        and will model with a Poisson distribution, and those with large probabilities, which we keep exactly for further computation.

        Parameters:
        ----
        q_bins (np.ndarray)  : redshift bins or log_nhi bins we want to compute
        lred (float)         : low cutoff redshift
        ured (float)         : upper cutoff redshift
        lnhi_min (float)     : low cutoff log_nhi
        lnhi_max (float)     : high cutoff log_nhi
        nhi (bool)           : use log_nhi bins or not
        second (bool or int) : consider up to model DLA(second + 1)

        Attributes:
        ----
        lowzcut   : controls the redshift cutoff of the proximity zone of quasar
        lybetacut : controls the lymanbeta cutoff (discard the regions contaminated by lyman beta forest)
        """
        # A list of probabilities for each redshift bin
        probs = [list() for _ in q_bins[:-1]]
        poisson_list = [list() for _ in q_bins[:-1]]
        dla_ind = self.filter_dla_spectra(second=second)
        for spec in dla_ind[0]:
            # Compute redshift of each sample
            (lnhi_vals, redshifts) = self._get_sample_params(spec, second=second)

            upper_z = ured
            lower_z = lred
            # test lyinf - lybeta only
            if self.z_max_lyb:
                assert self.lowzcut == False
                upper_z = np.min([self.lymanbeta(self.z_max(spec)), ured])
            # test lybeta - lyalpha only (Lyα-only): floor at the QSO Lyβ EMISSION
            # = lymanbeta(z_qso), matching cddf_mock (NOT lymanbeta(z_min), a no-op).
            if self.z_min_lyb:
                assert self.highzcut == False
                lower_z = np.max([self.lymanbeta(self.z_qsos[spec]), lred])
            # The low cutoff redshift.
            if self.lowzcut:
                upper_z = np.min([self.proximity(self.z_max(spec)), ured])
            if self.highzcut:
                lower_z = np.max([self.tail(self.z_min(spec)), lred])
            if self.min_obs_wavelength_cut:
                # obs lambda to z sampling -> obs lambda / (1 + zQSO)
                z_obs_min = self.min_obs_wavelength / (lya_wavelength) - 1
                lower_z = np.max([z_obs_min, lred])

            # [DESI] The high cutoff nhi : too many failed modeling large DLAs due to GP is not trained on DESI spectra
            if self.high_nhi_cut:
                lnhi_max = self.high_nhi_cut_value

            # Select only samples with a DLA value, within the redshift we want.
            desired_samples = (
                (lnhi_vals > lnhi_min)
                * (lnhi_vals < lnhi_max)
                * (redshifts < upper_z)
                * (redshifts > lower_z)
            )
            if self.filter_noisy_pixels:
                # Exclude pixels which have too large noise within them
                # These are the indexes of the samples in the pixel noise vector
                pn = self.pixel_noise[spec]
                pind = np.array(
                    (redshifts - self.z_min(spec))
                    / (self.z_max(spec) - self.z_min(spec))
                    * np.size(pn),
                    dtype=int,
                )
                desired_samples *= pn[pind] < self.noise_thresh
            ind = np.where(desired_samples)
            if np.size(ind) == 0:
                continue
            # Find the probability that we have a DLA from this spectrum in each redshift bin
            p_dla_each_bin = self._get_prob_dla_this_bin(spec, ind[0], second=second)
            ind2 = np.where(p_dla_each_bin > self.p_thresh_sample)
            if np.size(ind2) == 0:
                continue
            # If this is computing the CDDF, use lnhi_vals. Otherwise use redshift for dN/dX and omega_DLA
            if nhi:
                quantity = lnhi_vals[ind]
            else:
                quantity = redshifts[ind]
            for iz in range(np.size(q_bins) - 1):
                p_dla_this_z = p_dla_each_bin[ind2][
                    np.where(
                        (quantity[ind2] > q_bins[iz])
                        * (quantity[ind2] < q_bins[iz + 1])
                    )
                ]
                if np.size(p_dla_this_z) == 0:
                    continue
                #                 assert np.all(p_dla_this_z > 1e-4)
                # Add small probability events to the Poisson approximation: use a stable sum as this is probably *very* unstable.
                ipois = np.where(p_dla_this_z < self.p_switch)
                if np.size(ipois) > 0:
                    poisson_list[iz].append(math.fsum(p_dla_this_z[ipois]))
                # Add large probability events to the direct compute chain
                idla = np.where(p_dla_this_z >= self.p_switch)
                if np.size(idla) > 0:
                    probs[iz].append(p_dla_this_z[idla])
        poissons = np.array([math.fsum(pl) for pl in poisson_list])
        # Check that the Poisson approximation is a reasonable one; in practice this seems pretty good.
        # poissonsquare= np.array([math.fsum(pl**2) for pl in poisson_list])
        # assert np.all(poissonsquare/poissons < 0.2)
        return probs, poissons

    def _get_combined_levels(self, pdf_pb, pmean):
        """Get the combined pdf of a Poisson binomial process and a Poisson distribution with parameter pmean"""
        cdf_dla = np.cumsum(pdf_pb)
        # Properties of a zero poisson distribution are not defined.
        if pmean == 0.0:
            return (pdf_pb, 0)
        weak = poisson(pmean)
        # So now we have the PDF of the likely DLAs (which may not be Poisson). Add in the PDF of the Poisson process describing the others
        # Neglect the tails where either CDF is < 1e-4
        (plow, phigh) = weak.interval(1 - 1e-4)
        plow = int(plow)
        phigh = int(phigh)
        (dlow, dhigh) = interval(cdf_dla, 1 - 1e-4)
        # print(pmean, plow, phigh, np.argmax(pdf_pb), dlow, dhigh)
        # Note that in practice a not terrible approximation is just to sum the confidence intervals.
        # But that marginally overestimates the errors!
        pdf_comb = np.array(
            [
                math.fsum(
                    [
                        weak.pmf(N - i) * pdf_pb[i]
                        for i in range(dlow, np.min([dhigh + 1, np.size(pdf_pb)]))
                    ]
                )
                for N in range(plow + dlow, phigh + dhigh + 1)
            ]
        )
        assert 1.00 > math.fsum(pdf_comb) > 0.99
        return (pdf_comb, plow + dlow)

    def _get_confidence_intervals(
        self, q_bins, lred=2.0, ured=4.0, lnhi_min=20.3, lnhi_max=23.0, nhi=False
    ):
        """Compute MAP + Bayesian 68/95% CIs on DLA counts per bin.

        Each DLA contributes a probability p_i of falling in a given bin,
        making the total count a sum of Bernoulli variables — a
        Poisson-binomial distribution.

        Algorithm:
          1. ``_split_distributions`` partitions samples into two groups:
             - High-p samples (p >= p_switch=0.25): kept exactly for FFT.
             - Low-p samples (p < p_switch): approximated as Poisson
               (Le Cam 1960; error bounded by sum(p²)/sum(p)).
          2. For high-p samples, the exact Poisson-binomial PDF is computed
             via FFT (Fernandez & Williams 2010): see ``get_poisson_binomial_pdf``.
          3. The Poisson component is convolved in via ``_get_combined_levels``.
          4. MAP, 68%, and 95% credible intervals are extracted from the
             combined CDF.

        Parameters
        ----------
        q_bins : np.ndarray
            Bin edges — either redshift bins (nhi=False) or log10(N_HI) bins (nhi=True).
        lred, ured : float
            Redshift range to include.
        lnhi_min, lnhi_max : float
            log10(N_HI) range to include.
        nhi : bool
            If True, bin by log10(N_HI); if False, bin by redshift.

        Returns
        -------
        maxlikes : list of int
            MAP DLA count in each bin.
        levels68 : list of (int, int)
            68% credible interval (lower, upper) index pairs.
        levels95 : list of (int, int)
            95% credible interval (lower, upper) index pairs.
        """
        (probs, poissons) = self._split_distributions(
            q_bins, lred=lred, ured=ured, lnhi_min=lnhi_min, lnhi_max=lnhi_max, nhi=nhi
        )
        # probs[i] now contains a list of arrays
        # Now we have built a list of probabilities in each z bin of interest and we want to solve for the Poisson binomial coefficients.
        maxlikes = []
        levels68 = []
        levels95 = []
        for pp, pmean in zip(probs, poissons):
            pdf = get_poisson_binomial_pdf(pp)
            (pdf_comb, offset) = self._get_combined_levels(pdf, pmean)
            (maxlike, ll68, ll95) = pdf_confidence(pdf_comb, offset)
            # Check correctly ordered
            assert ll95[0] <= ll68[0] <= maxlike
            assert ll95[1] >= ll68[1] >= maxlike
            # Unpack maximum likelihoods and 68/95% contours
            maxlikes.append(maxlike)
            levels68.append(ll68)
            levels95.append(ll95)
        return (maxlikes, levels68, levels95)

    def _get_z_nhi_hist(
        self,
        q_bins,
        lred=2.0,
        ured=4.0,
        lnhi_min=20.3,
        lnhi_max=23.0,
        nhi=False,
        moment=False,
    ):
        """
        Estimate the mean and standard deviation on the number of DLAs in a given redshift bin.
        Since each DLA has some probability of being in a given bin, p_dla * p_in_this_bin,
        each DLA is a binomial process, and the sum is a binomial poisson process.
        Thus the mean is sum(p_dla * p_in_this_bin) and the variance sum[p(1-p)]
        Ignore spectra with p_DLA < p_thresh, as an optimization.
        """
        dla_ind = self.filter_dla_spectra()
        means = np.zeros(np.size(q_bins) - 1)
        variances = np.zeros(np.size(q_bins) - 1)
        for spec in dla_ind[0]:
            # Compute redshift of each sample
            (lnhi_vals, redshifts) = self._get_sample_params(spec)
            # Select only samples with a DLA value, within the redshift we want.
            ind = np.where(
                (lnhi_vals > lnhi_min)
                * (lnhi_vals < lnhi_max)
                * (redshifts < ured)
                * (redshifts > lred)
            )
            if np.size(ind) == 0:
                continue
            # Find the probability that we have a DLA from this spectrum in each redshift bin
            p_dla_each_bin = self._get_prob_dla_this_bin(spec, ind[0], second=False)
            # Multiply by the column density to get total amount of HI instead of the number of DLAs
            if moment:
                weight = 10 ** lnhi_vals[ind]
            else:
                weight = 1.0
            # If this is computing the CDDF, use lnhi_vals. Otherwise use redshift for dN/dX and omega_DLA
            if nhi:
                quantity = lnhi_vals[ind]
            else:
                quantity = redshifts[ind]
            # These are the means
            (t_hist, _) = np.histogram(
                quantity, bins=q_bins, weights=weight * p_dla_each_bin
            )
            means += t_hist
            # These are the variances
            (t_var, _) = np.histogram(
                quantity,
                bins=q_bins,
                weights=weight * weight * (1 - p_dla_each_bin) * p_dla_each_bin,
            )
            variances += t_var
        # Don't forget Poisson term from sample variance.
        # The variance before this indicates the uncertainty arising from our imperfect knowledge of the properties of the DLAs
        # in our spectra; this term indicates our imperfect *sampling* of the total population
        # If we had one spectrum which we were certain contained a DLA, this would estimate the error.
        variances += means
        return means, variances

    def find_delta_NHI(self, nspec):
        """Find the range of NHI values in nspec with a likelihood 1/2e times the max.
        This is an easily calculable value which is the 2-sigma contour if the likelihood is Gaussian
        """
        likes = self._log_norm_like(nspec)
        mlike = np.max(likes)
        nvals = self.lnhi_vals[np.where(likes > mlike - 2)]
        return np.max(nvals) - np.min(nvals)

    def find_delta_z(self, nspec):
        """Find the range of redshift values in nspec with a likelihood 1/2e times the max.
        This is an easily calculable value which is the 2-sigma contour if the likelihood is Gaussian
        """
        likes = self._log_norm_like(nspec)
        mlike = np.max(likes)
        nvals = (self.z_max(nspec) - self.z_min(nspec)) * self.z_offsets[
            np.where(likes > mlike - 2)
        ] + self.z_min(nspec)
        return np.max(nvals) - np.min(nvals)

    def find_max_like(self, nspec, *, second=False):
        """Find the maximum likelihood values of NHI and redshift"""
        likes = self._log_norm_like(nspec, second=second)
        mlike = np.argmax(likes)
        (lnhi_vals, redshifts) = self._get_sample_params(nspec, second=second)
        return lnhi_vals[mlike], redshifts[mlike]

    def find_real(self, nspec, *, field="flux"):
        """Find the index of a quasar in the raw datafile"""
        # Load the indices of the quasars we have data for in the raw file
        nspec_real = self.real_index[nspec]
        hh = h5py.File(self.raw_file, "r")
        flux = hh[hh["all_" + field][0][nspec_real]][0]
        nflux = np.size(flux)
        zzs = (self.z_max(nspec) - self.z_min(nspec)) * range(
            nflux
        ) / nflux + self.z_min(nspec)
        hh.close()
        return zzs, flux


def find_snr(nspec, real_index, raw_file, zmin, zmax):
    """Find the signal to noise ratio, according to the definition where it is the flux/s.d. noise."""
    # Get noise variance
    _ = zmin
    nspec_real = real_index[nspec]
    hh = h5py.File(raw_file, "r")
    wavelengths = hh[hh["all_wavelengths"][0][nspec_real]][0]
    # ipix = np.where(np.logical_and(wavelengths > 1215.67*(1+ zmin), wavelengths < 1215.67*(1+zmax)))
    ipix = np.where(wavelengths > 1215.67 * (1 + zmax))
    flux = np.array(hh[hh["all_flux"][0][nspec_real]][0])[ipix]
    try:
        norm = hh["all_normalizers"][0][nspec_real]
        # This is so that we don't have an unrealistically low noise threshold inside of absorbers.
        flux[np.where(flux / norm < 0.1)] = norm * 0.1
    except KeyError:
        flux[np.where(flux < 0.1)] = 0.1
    noise_var = np.array(hh[hh["all_noise_variance"][0][nspec_real]][0])[ipix]
    hh.close()
    return 1 / np.median(np.sqrt(noise_var) / np.abs(flux))


def find_pixel_noise(nspec, real_index, raw_file, zmin, zmax):
    """Find pixels where the absolute value of the noise is below thresh a particular value.
    So we want pixels with: all_noise_variance/all_normalizers^2 < thresh^2
    where all_noise_variance is the noise and defined in preloaded_qsos."""
    nspec_real = real_index[nspec]
    hh = h5py.File(raw_file, "r")
    norm = hh["all_normalizers"][0][nspec_real]
    wavelengths = hh[hh["all_wavelengths"][0][nspec_real]][0]
    ipix = np.where(
        np.logical_and(
            wavelengths > 1215.67 * (1 + zmin), wavelengths < 1215.67 * (1 + zmax)
        )
    )
    noise_var = np.array(hh[hh["all_noise_variance"][0][nspec_real]][0])[ipix]
    hh.close()
    return noise_var / norm**2


def find_pixel_snr(nspec, real_index, raw_file, zmin, zmax):
    """Find pixels where the absolute value of the noise is below thresh a particular value.
    So we want pixels with: all_noise_variance/all_normalizers^2 < thresh^2
    where all_noise_variance is the noise and defined in preloaded_qsos."""
    nspec_real = real_index[nspec]
    hh = h5py.File(raw_file, "r")
    wavelengths = hh[hh["all_wavelengths"][0][nspec_real]][0]
    ipix = np.where(
        np.logical_and(
            wavelengths > 1215.67 * (1 + zmin), wavelengths < 1215.67 * (1 + zmax)
        )
    )
    flux = np.array(hh[hh["all_flux"][0][nspec_real]][0])[ipix]
    noise_var = np.array(hh[hh["all_noise_variance"][0][nspec_real]][0])[ipix]
    try:
        norm = hh["all_normalizers"][0][nspec_real]
        # This is so that we don't have an unrealistically low noise threshold inside of absorbers.
        flux[np.where(flux / norm < 0.1)] = norm * 0.1
    except KeyError:
        flux[np.where(flux < 0.1)] = 0.1
    hh.close()
    return np.sqrt(noise_var) / np.abs(flux)


def compute_all_snrs(
    *,
    raw_file="preloaded_qsos.mat",
    processed_file="processed_qsos_dr12q_lyb_lya.mat",
    save_file="snrs_qsos_dr12.mat"
):
    """Compute the SNR for all spectra and save to a separate file"""
    ff = h5py.File(processed_file, "r")
    real_index = np.where(ff["test_ind"][0] != 0)[0]
    min_z_dla = np.array(ff["min_z_dlas"][0])
    max_z_dla = np.array(ff["max_z_dlas"][0])

    # if this is a partial file, take the first min_z_dla.shape[0] index
    size = min_z_dla.shape[0]
    if size != real_index.shape[0]:
        print(
            "[Warning] size preload {} and size processed {} not match, take first {} index.".format(
                real_index.shape[0],
                size,
                size,
            )
        )
    else:
        size = np.size(real_index)

    ff.close()
    snrs = np.array(
        [
            find_snr(nn, real_index, raw_file, min_z_dla[nn], max_z_dla[nn])
            for nn in range(size)
        ]
    )
    f = h5py.File(save_file, "w")
    f["snrs"] = snrs
    #     dt = h5py.special_dtype(vlen=np.dtype('float64'))
    #     dset = f.create_dataset('pixel_noise', (np.size(real_index),), dtype=dt)
    #     for nn in range(np.size(real_index)):
    #         dset[nn] = find_pixel_noise(nn, real_index, raw_file, min_z_dla[nn], max_z_dla[nn])
    #     dset = f.create_dataset('pixel_snr', (np.size(real_index),), dtype=dt)
    #     for nn in range(np.size(real_index)):
    #         dset[nn] = find_pixel_snr(nn, real_index, raw_file, min_z_dla[nn], max_z_dla[nn])
    f.close()


def HubbleByH0(z, Omega_m=0.279):
    """Hubble function divided by H0, H/H0(z).
    H/H0(z)**2 = Omega_m/a^3 + Omega_lambda
    We neglect curvature and radiation, and assume Omega_lambda = 1- Omega_m.
    Omega_m is WMAP 9 by default
    """
    return math.sqrt(Omega_m * (1 + z) ** 3 + (1 - Omega_m))


def interval(cdf, level, offset=0):
    """Return a tuple with the confidence interval at level for the given cdf.
    level should be between 0 and 1. Larger values mean wider intervals"""
    if np.size(cdf) == 1:
        return (offset, offset)
    ii = np.where((cdf <= 0.5 + level / 2.0) * (cdf >= 0.5 - level / 2.0))
    # This can happen when all the cdf is in one bin, and it is on the edge.
    if True or np.size(ii) == 0:
        high = 1 + offset
        low = offset
        idown = np.where(cdf < 0.5 - level / 2)
        if np.size(idown) != 0:
            low += idown[0][-1] + 1
        iup = np.where(cdf > 0.5 + level / 2)
        if np.size(iup) != 0:
            high += iup[0][0]
        else:
            high = np.size(cdf)
        return (low, high)
    return (ii[0][0] + offset, 1 + ii[0][-1] + offset)


def pdf_confidence(pdf_comb, offset):
    """Get the maximum likelihood value, and 68 and 95 % confidence levels of a probability density function.
    offset is a value to add to all derived quantities
    """
    # Find the cumulative distribution function
    cdf_comb = np.cumsum(pdf_comb)
    # Find the maximum a posteriori likelihood.
    maxlike = interval(cdf_comb, 0.0, offset=offset)[0]
    ll68 = interval(cdf_comb, 0.68, offset=offset)
    ll95 = interval(cdf_comb, 0.95, offset=offset)
    assert maxlike >= ll68[0] >= ll95[0]
    assert maxlike <= ll68[1] <= ll95[1]
    return maxlike, ll68, ll95


def get_poisson_binomial_pdf(pp):
    """Compute the exact PDF of a Poisson-binomial distribution via FFT.

    The Poisson-binomial distribution is the distribution of the sum of
    independent Bernoulli variables with (different) probabilities p_i.
    Its PDF P(N=n) = sum_{S ⊆ [m], |S|=n} prod_{i∈S} p_i prod_{i∉S} (1-p_i).

    This is computed using the DFT-based algorithm of Fernandez & Williams (2010):
        C_n = exp(2πi n / (m+1)) − 1
        Φ_n = prod_i (1 + p_i C_n)       [characteristic function coefficients]
        PDF = IFFT(Φ) / (m+1)

    The symmetry of the DFT is used to compute only the first half of the
    coefficient array.

    Parameters
    ----------
    pp : list of np.ndarray
        List of 1D arrays of DLA probabilities in this bin.
        Will be concatenated into a single array.

    Returns
    -------
    np.ndarray
        PDF array of length m+1 where m = total number of samples.
        pdf[n] = probability of exactly n DLAs in this bin.

    References
    ----------
    Fernandez, M. and Williams, S. (2010). "Closed-Form Expression for the
    Poisson-Binomial Probability Density Function."
    IEEE Transactions on Reliability, 59(3), 615–616.
    """
    # Check input is reasonable
    if np.size(pp) == 0:
        return np.ones(1)
    ppa = np.array(np.concatenate(pp))
    assert ppa.dtype == np.float64
    assert np.size(np.shape(ppa)) == 1
    Nsamp = np.size(ppa)
    # Compute the coefficients of a DFT which we will use to find the poisson-binomial coefficients.
    # See Fernandez, M. and Williams, S. 2010 or wikipedia
    nco = lambda nn: cmath.exp(-2 * math.pi * nn * 1j / (Nsamp + 1)) - 1
    # Use the symmetry of the fourier transform to only compute the first half of the array: we know that the pdf is real.
    coeffs = np.array(
        [
            stable_complex_product(1 + ppa * nco(nn))
            for nn in range((Nsamp + 1) // 2 + 1)
        ]
    )
    # Check for roundoff; should be ok as all the coefficients that are multiplied are within the unit circle
    # Almost all coeffs should be complex...
    assert np.any(np.absolute(coeffs) > 0)
    # Do the FFT
    pdf = np.fft.irfft(coeffs, n=Nsamp + 1)
    # Make sure we got a reasonable answer
    assert np.all(np.logical_not(np.isinf(pdf)))
    # Check correctly normalized
    assert np.abs(math.fsum(pdf) - 1.0) < 1e-7
    return pdf


def stable_complex_product(iterable):
    """
    Compute the product of a large array of complex numbers using logs and python's stable summation routine.
    Some shenanigans are necessary because we want to avoid taking a complex logarithm.
    We say that if z = r e ^i theta , then
    prod(z) = prod(r) exp(sum i theta) = exp( sum log (r) + sum i theta )
    Returns a long double result so we can store very small values!
    """
    rr = np.absolute(iterable)
    theta = np.angle(iterable)
    return np.exp(math.fsum(np.log(rr)) + 1j * math.fsum(theta), dtype=np.complex256)


def path_length_int(z, Omega_m=0.279):
    """Integrand for the comoving absorption path length integral.

    Returns dX/dz = (1 + z)^2 / (H(z)/H_0) = (1 + z)^2 * H_0 / H(z).

    This convention (Bahcall & Peebles 1969) ensures that dN/dX is
    redshift-independent for a population with constant comoving density.

    Parameters
    ----------
    z : float
        Redshift.
    Omega_m : float
        Matter density parameter (default 0.279, WMAP9).

    Returns
    -------
    float
        dX/dz at redshift z.
    """
    return (1 + z) ** 2 / HubbleByH0(z, Omega_m)


def rho_crit(hubble=0.7):
    """Critical density at z=0 in g cm⁻³.

    rho_c = 3 H_0^2 / (8 π G)

    Parameters
    ----------
    hubble : float
        Dimensionless Hubble constant h = H_0 / (100 km/s/Mpc) (default 0.7).

    Returns
    -------
    float
        Critical density rho_c(z=0) in g cm⁻³.
    """
    # H in units of 1/s
    # h * 100 km/s/Mpc in h/s
    h100 = 3.2407789e-18 * hubble
    gravcgs = 6.674e-8
    rho_c = 3 * h100**2 / (8 * math.pi * gravcgs)
    return rho_c


def z_cent_fill(z_cent, xerrs):
    """Small function to find an expanded filling region.
    Used in plotting 2 sigma contours."""
    filler = np.array(z_cent)
    filler[0] -= xerrs[0][0]
    filler[-1] += xerrs[-1][-1]
    return filler
