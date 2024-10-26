# null_meanflux_gp.py

"""
NullMFGP: Null GP model with mean-flux marginalization for DLA detection.
This class builds on NullGP to incorporate marginalization over `prev_beta` 
and `prev_tau_0`, following a Monte Carlo approach.

"""
import numpy as np
import h5py
from .set_parameters import Parameters
from .model_priors import PriorCatalog

from .null_gp import NullGP
from .effective_optical_depth import effective_optical_depth


class NullMFGP(NullGP):
    """
    Null GP model with mean-flux marginalization.

    This class extends the NullGP to marginalize over `prev_beta` and `prev_tau_0`
    by caching interpolated `this_mu` and `this_M` values for each combination
    and performing a Monte Carlo summation to compute log model evidence.

    Attributes:
        precomputed_mu: Cached mean vectors for each (prev_tau_0, prev_beta) combination.
        precomputed_M: Cached low-rank decomposition of covariance for each combination.
    """

    def __init__(self, *args, **kwargs):
        # Initialize the parent class
        super().__init__(*args, **kwargs)

        # Placeholder for cached `this_mu` and `this_M` based on `prev_beta` and `prev_tau_0` values
        self.precomputed_mu = None
        self.precomputed_M = None
        self.precomputed_omega2 = None

    def get_interp(
        self, x: np.ndarray, y: np.ndarray, wavelengths: np.ndarray, z_qso: float
    ) -> None:
        """
        Precompute and cache `this_mu` and `this_M` for each pair of `prev_beta` and `prev_tau_0`
        to enable efficient marginalization in log model evidence calculation.

        Args:
            x: Emission wavelengths of the observed data.
            y: Flux of the observed data.
            wavelengths: Observed wavelengths, derived from x and z_qso.
            z_qso: Redshift of the quasar.
        """
        # Assume `self.prev_beta` and `self.prev_tau_0` are arrays of shape (num_dla_samples,)
        num_dla_samples = self.prev_beta.shape[0]

        # Prepare arrays to store precomputed values
        self.precomputed_mu = np.zeros((num_dla_samples, x.shape[0]))
        self.precomputed_M = np.zeros((num_dla_samples, x.shape[0], self.params.k))
        self.precomputed_omega2 = np.zeros((num_dla_samples, x.shape[0]))

        _this_mu = self.mu_interpolator(x)
        _this_M = self.M_interpolator(x)

        # Interpolate noise parameters
        this_log_omega = self.log_omega_interpolator(x)
        this_omega2 = np.exp(2 * this_log_omega)

        # Compute scaling factor for absorption noise
        lya_optical_depth = effective_optical_depth(
            wavelengths,
            np.exp(self.log_beta),
            np.exp(self.log_tau_0),
            z_qso,
            self.params.num_forest_lines,
        )
        scaling_factor = (
            1 - np.exp(-np.sum(lya_optical_depth, axis=1)) + np.exp(self.log_c_0)
        )
        self.this_omega2 = this_omega2 * scaling_factor**2

        # Precompute `this_mu` and `this_M` for each sample
        for i in range(num_dla_samples):
            # Compute effective optical depth based on sampled `prev_tau_0` and `prev_beta`
            total_optical_depth = effective_optical_depth(
                wavelengths,
                self.prev_beta[i],
                self.prev_tau_0[i],
                z_qso,
                self.params.num_forest_lines,
            )
            lya_absorption = np.exp(-np.sum(total_optical_depth, axis=1))

            # Interpolate mean vector and covariance decomposition
            this_mu = _this_mu * lya_absorption
            this_M = _this_M * lya_absorption[:, None]
            # re-adjust (K + Ω) to the level of μ .* exp( -optical_depth ) = μ .* a_lya
            # now the null model likelihood is:
            # p(y | λ, zqso, v, ω, M_nodla) = N(y; μ .* a_lya, A_lya (K + Ω) A_lya + V)
            this_omega2 = this_omega2 * lya_absorption**2

            # Cache results
            self.precomputed_mu[i, :] = this_mu
            self.precomputed_M[i, :, :] = this_M
            self.precomputed_omega2[i, :] = this_omega2

    def log_model_evidence(self) -> float:
        """
        Compute log model evidence by marginalizing over `prev_beta` and `prev_tau_0`
        using Monte Carlo summation over precomputed `this_mu` and `this_M`.

        Returns:
            float: The log model evidence for the null model.
        """
        num_dla_samples = self.prev_beta.shape[0]
        log_likelihoods = np.zeros(num_dla_samples)

        # Iterate over all samples of `prev_beta` and `prev_tau_0`
        for i in range(num_dla_samples):
            # Extract precomputed values
            this_mu = self.precomputed_mu[i, :]
            this_M = self.precomputed_M[i, :, :]
            this_omega2 = self.precomputed_omega2[i, :]

            # Calculate log-likelihood for this sample
            log_likelihoods[i] = self.log_mvnpdf_low_rank(
                self.y, this_mu, this_M, this_omega2 + self.v
            )

        # Monte Carlo average to get final log model evidence
        # Perform the log-sum-exp trick to avoid numerical instability
        max_log_likelihood = np.nanmax(log_likelihoods)
        sample_probabilities = np.exp(log_likelihoods - max_log_likelihood)
        log_likelihood_no_dla = max_log_likelihood + np.log(
            np.nanmean(sample_probabilities)
        )

        return log_likelihood_no_dla


class NullMFGPMAT(NullMFGP):
    """
    Load learned model from .mat file and marginalize over mean-flux parameters.

    This class reads the primary learned file containing GP parameters, as well as
    `tau_0_samples_30000.mat` for sampled `prev_tau_0` and `prev_beta` arrays.
    """

    def __init__(
        self,
        params: Parameters,
        prior: PriorCatalog,
        learned_file: str = "learned_qso_model_lyseries_variance_kim_dr9q_minus_concordance.mat",
        tau_beta_file: str = "data/dr12q/processed/tau_0_samples_30000.mat",
    ):
        # Load the main learned GP parameters
        with h5py.File(learned_file, "r") as learned:
            rest_wavelengths = learned["rest_wavelengths"][:, 0]
            mu = learned["mu"][:, 0]
            M = learned["M"][()].T
            log_omega = learned["log_omega"][:, 0]
            log_c_0 = learned["log_c_0"][0, 0]
            log_tau_0 = learned["log_tau_0"][0, 0]
            log_beta = learned["log_beta"][0, 0]

        # Load `prev_tau_0` and `prev_beta` arrays for marginalization
        with h5py.File(tau_beta_file, "r") as tau_beta_data:
            prev_tau_0 = tau_beta_data["tau_0_samples"][:]
            prev_beta = tau_beta_data["beta_samples"][:]

        # Validate shape of `prev_tau_0` and `prev_beta` to ensure consistency
        if prev_tau_0.shape != prev_beta.shape:
            raise ValueError(
                "Mismatch in shape between `prev_tau_0` and `prev_beta` arrays."
            )

        # Initialize the superclass with loaded parameters
        super().__init__(
            params=params,
            prior=prior,
            rest_wavelengths=rest_wavelengths,
            mu=mu,
            M=M,
            log_omega=log_omega,
            log_c_0=log_c_0,
            log_tau_0=log_tau_0,
            log_beta=log_beta,
            prev_tau_0=prev_tau_0,
            prev_beta=prev_beta,
        )
