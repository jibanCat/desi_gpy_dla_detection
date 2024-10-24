"""
A class to handle the QMC samples of the DLA parameters:
theta = (z_dla, logNHI) = (redshift of DLA, column density of DLA)
"""

import numpy as np
import scipy.stats as stats
from scipy.integrate import quad
import h5py
from .set_parameters import Parameters
from .model_priors import PriorCatalog


class DLASamples:
    """
    A class to generate and store the QMC samples for DLAs:
    theta = (z_dla, logNHI) = (redshift of DLA, column density of DLA)

    :attr offset_samples: used for z_dla samples
    :attr log_nhi_samples: log_nhi samples
    """

    def __init__(self, params: Parameters, prior: PriorCatalog):
        self.params = params
        self.prior = prior

        # extract data-driven prior paramters
        self.num_dla_samples = params.num_dla_samples
        self.uniform_min_log_nhi = params.uniform_min_log_nhi
        self.uniform_max_log_nhi = params.uniform_max_log_nhi
        self.fit_min_log_nhi = params.fit_min_log_nhi
        self.fit_max_log_nhi = params.fit_max_log_nhi
        self.alpha = params.alpha

    def log_nhi_prior(self):
        NotImplementedError

    def z_dla_prior(self):
        NotImplementedError

    @property
    def offset_samples(self):
        NotImplementedError

    @property
    def log_nhi_samples(self):
        NotImplementedError

    @property
    def nhi_samples(self):
        NotImplementedError


class DLASamplesMAT(DLASamples):
    """
    Load DLA samples from .mat file, which is generated from
    Roman's generate_dla_samples.m.
    """

    def __init__(
        self,
        params: Parameters,
        prior: PriorCatalog,
        dla_samples_file: str = "dla_samples_a03.mat",
    ):
        super().__init__(params, prior)

        dla_samples = h5py.File(dla_samples_file, "r")

        assert self.alpha == dla_samples["alpha"][0, 0]
        assert self.uniform_min_log_nhi == dla_samples["uniform_min_log_nhi"][0, 0]

        self._offset_samples = dla_samples["offset_samples"][:, 0]
        self._log_nhi_samples = dla_samples["log_nhi_samples"][:, 0]
        self._nhi_samples = dla_samples["nhi_samples"][:, 0]

        self.uniform_min_log_nhi = dla_samples["uniform_min_log_nhi"][0, 0]
        self.uniform_max_log_nhi = dla_samples["uniform_max_log_nhi"][0, 0]

        # # build the pdf function
        # self._pdf()

    @property
    def offset_samples(self) -> np.ndarray:
        return self._offset_samples

    @property
    def log_nhi_samples(self) -> np.ndarray:
        return self._log_nhi_samples

    @property
    def nhi_samples(self) -> np.ndarray:
        return self._nhi_samples

    def sample_z_dlas(self, wavelengths: np.ndarray, z_qso: float) -> np.ndarray:
        sample_z_dlas = (
            self.params.min_z_dla(wavelengths, z_qso)
            + (
                self.params.max_z_dla(wavelengths, z_qso)
                - self.params.min_z_dla(wavelengths, z_qso)
            )
            * self._offset_samples
        )

        return sample_z_dlas

    # TODO: This part makes the Python pickle file issue during multiprocessing
    # def _pdf(self):
    #     """
    #     Make the normalized pdf and store it in class.
    #     """
    #     # uniform component of column density prior
    #     u = stats.uniform(loc=self.uniform_min_log_nhi,
    #         scale=self.uniform_max_log_nhi - self.uniform_min_log_nhi)

    #     # directly use the fitted poly values in the Garnett (2017)
    #     unnormalized_pdf = lambda nhi: (np.exp(
    #         -1.2695 * nhi**2 + 50.863 * nhi -509.33
    #     ))
    #     Z = quad(unnormalized_pdf, self.fit_min_log_nhi, 25.0)[0] # hard-coded 25.0

    #     # create the PDF of the mixture between the unifrom distribution and
    #     # the distribution fit to the data
    #     normalized_pdf = lambda nhi: self.alpha  * (unnormalized_pdf(nhi) / Z
    #         ) + (1 - self.alpha) * (u.pdf(nhi))

    #     self.normalized_pdf = normalized_pdf

    # def pdf(self, log_nhi: float) -> float:
    #     """
    #     The logNHI pdf used in Garnett (2017) paper.
    #     """
    #     return self.normalized_pdf(log_nhi)
