"""
subdla_meanflux_gp.py

A GP class for handling subDLAs intervening in a given sightline. 
This inherits from `DLAMFGP` for handling mean-flux marginalization.
"""

from typing import Tuple, Optional
import numpy as np
import h5py

from .set_parameters import Parameters
from .model_priors import PriorCatalog
from .dla_meanflux_gp import DLAMFGP  # Updated inheritance

# Attempt to import VoigtProfile from voigt_fast, and fall back to voigt if it fails
try:
    from .voigt_fast import VoigtProfile

    voigt_absorption = VoigtProfile().compute_voigt_profile
except ImportError:
    from .voigt import voigt_absorption

from .subdla_samples import SubDLASamplesMAT  # for convenient autocomplete


class SubDLAMFGP(DLAMFGP):
    """
    SubDLA GP model for QSO emission + subDLA intervening:
        p(y | λ, σ², M, ω, c₀, τ₀, β, τ_kim, β_kim, z_dla, logNHI)

    Additional parameters (z_dla, logNHI) control the position and strength of the
    absorption intervening on the QSO emission.

    SubDLA parameter prior: logNHI ~ U(19.5, 20)

    We use Quasi Monte Carlo (QMC) for approximating the model evidence.

    :param rest_wavelengths: λ, the range of λ you model your GP on QSO emission
    :param mu: mu, the mean model of the GP.
    :param M: M, the low-rank decomposition of the covariance kernel: K = MM^T.
    :param log_omega: log ω, the pixel-wise noise of the model. Used to model absorption noise.
    :param log_c_0: log c₀, the constant in the Lyman forest noise model.
    :param log_tau_0: log τ₀, the scale factor of effective optical depth in the absorption noise.
    :param log_beta: log β, the exponent of the effective optical depth in the absorption noise.
    :param prev_tau_0: τ_kim, the scale factor of effective optical depth used in mean-flux suppression.
    :param prev_beta: β_kim, the exponent of the effective optical depth used in mean-flux suppression.

    Future: MCMC can be embedded in the class as an instance method.
    """

    def __init__(
        self,
        params: Parameters,
        prior: PriorCatalog,
        dla_samples: SubDLASamplesMAT,
        rest_wavelengths: np.ndarray,
        mu: np.ndarray,
        M: np.ndarray,
        log_omega: np.ndarray,
        log_c_0: float,
        log_tau_0: float,
        log_beta: float,
        prev_tau_0: float = 0.0023,
        prev_beta: float = 3.65,
        min_z_separation: float = 3000.0,
        broadening: bool = True,
    ):
        # Initialize the superclass with all parameters
        super().__init__(
            params=params,
            prior=prior,
            dla_samples=dla_samples,
            rest_wavelengths=rest_wavelengths,
            mu=mu,
            M=M,
            log_omega=log_omega,
            log_c_0=log_c_0,
            log_tau_0=log_tau_0,
            log_beta=log_beta,
            prev_tau_0=prev_tau_0,
            prev_beta=prev_beta,
            min_z_separation=min_z_separation,
            broadening=broadening,
        )

    def log_priors(self, z_qso: float, max_dlas: int) -> float:
        """
        Get the model prior for the SubDLA model, defined as:
            P(k subDLA | zQSO) = P(at least k subDLAs | zQSO) - P(at least (k + 1) subDLAs | zQSO)

        Where:
            P(at least 1 subDLA | zQSO) = Z_lls / Z_dla * M / N

        Here:
        - M is the number of subDLAs below this zQSO.
        - N is the number of quasars below this zQSO.
        - Z_lls and Z_dla are normalization factors for subDLAs and DLAs.

        Args:
            z_qso (float): The redshift of the quasar.
            max_dlas (int): The maximum number of subDLAs considered.

        Returns:
            log_priors_dla (float): The log prior for each subDLA.
        """
        this_num_dlas, this_num_quasars = self.prior.less_ind(z_qso)

        # Adjust the prior for subDLAs using the Z_lls / Z_dla ratio
        p_dlas = (
            self.dla_samples._Z_lls
            / self.dla_samples._Z_dla
            * (this_num_dlas / this_num_quasars) ** np.arange(1, max_dlas + 1)
        )

        # Adjust the probabilities to account for P(k subDLA | zQSO)
        for i in range(max_dlas - 1):
            p_dlas[i] = p_dlas[i] - p_dlas[i + 1]

        log_priors_dla = np.log(p_dlas)

        return log_priors_dla


class SubDLAMFGPMAT(SubDLAMFGP):
    """
    Load a learned model from a .mat file for SubDLA GP.

    The learned model file structure is the same as DLAGP.
    The sample file differs for subDLAs.
    """

    def __init__(
        self,
        params: Parameters,
        prior: PriorCatalog,
        dla_samples: SubDLASamplesMAT,
        min_z_separation: float = 3000.0,
        learned_file: str = "learned_qso_model_lyseries_variance_kim_dr9q_minus_concordance.mat",
        tau_beta_file: str = "data/dr12q/processed/tau_0_samples_30000.mat",
        broadening: bool = True,
        prev_tau_0: float = 0.0023,  # Not used
        prev_beta: float = 3.65,  # Not used
    ):
        # Load the learned model from the .mat file
        with h5py.File(learned_file, "r") as learned:
            rest_wavelengths = learned["rest_wavelengths"][:, 0]
            mu = learned["mu"][:, 0]
            M = learned["M"][()].T
            log_omega = learned["log_omega"][:, 0]
            log_c_0 = learned["log_c_0"][0, 0]
            log_tau_0 = learned["log_tau_0"][0, 0]
            log_beta = learned["log_beta"][0, 0]

        # Load tau_0 and beta samples for marginalization
        try:
            with h5py.File(tau_beta_file, "r") as tau_beta_data:
                prev_tau_0 = tau_beta_data["tau_0_samples"][:]
                prev_beta = tau_beta_data["beta_samples"][:]
        except (OSError, KeyError) as e:
            raise RuntimeError(
                f"Error loading tau/beta samples from {tau_beta_file}: {e}"
            )

        # Validate shape of `prev_tau_0` and `prev_beta` to ensure consistency
        if prev_tau_0.shape != prev_beta.shape:
            raise ValueError(
                "Mismatch in shape between `prev_tau_0` and `prev_beta` arrays."
            )

        # Initialize the superclass with all parameters
        super().__init__(
            params=params,
            prior=prior,
            dla_samples=dla_samples,
            rest_wavelengths=rest_wavelengths,
            mu=mu,
            M=M,
            log_omega=log_omega,
            log_c_0=log_c_0,
            log_tau_0=log_tau_0,
            log_beta=log_beta,
            prev_tau_0=prev_tau_0,
            prev_beta=prev_beta,
            min_z_separation=min_z_separation,
            broadening=broadening,
        )
