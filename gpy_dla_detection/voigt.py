"""
voigt.py : python version of the Voigt profile for 
    Roman's voigt.c file, including:

- number of Lyman series members as an option
- parameterized using : (1) z_dla, (2) nhi;
- input wavelengths: observed wavelengths

Note:
I keep the variables of Lyman series in this file to
reflect the same structure as Roman's code, but they
could be moved to set_parameters.py in the future.

TODO: instrumental boardening
"""
import numpy as np
from scipy.special import wofz

# physical constants in cgs
c: float = 2.99792458e10

# quantities of Lyman series
transition_wavelengths: np.ndarray = np.array(
    [  # lambda_ul, cm
        1.2156701e-05,  # Lya
        1.0257223e-05,  # Lyb ...
        9.725368e-06,
        9.497431e-06,
        9.378035e-06,
        9.307483e-06,
        9.262257e-06,
        9.231504e-06,
        9.209631e-06,
        9.193514e-06,
        9.181294e-06,
        9.171806e-06,
        9.16429e-06,
        9.15824e-06,
        9.15329e-06,
        9.14919e-06,
        9.14576e-06,
        9.14286e-06,
        9.14039e-06,
        9.13826e-06,
        9.13641e-06,
        9.13480e-06,
        9.13339e-06,
        9.13215e-06,
        9.13104e-06,
        9.13006e-06,
        9.12918e-06,
        9.12839e-06,
        9.12768e-06,
        9.12703e-06,
        9.12645e-06,
    ]
)

oscillator_strengths: np.ndarray = np.array(
    [  # oscillator strengths f_ul, dimensionless
        0.416400,
        0.079120,
        0.029000,
        0.013940,
        0.007799,
        0.004814,
        0.003183,
        0.002216,
        0.001605,
        0.00120,
        0.000921,
        0.0007226,
        0.000577,
        0.000469,
        0.000386,
        0.000321,
        0.000270,
        0.000230,
        0.000197,
        0.000170,
        0.000148,
        0.000129,
        0.000114,
        0.000101,
        0.000089,
        0.000080,
        0.000071,
        0.000064,
        0.000058,
        0.000053,
        0.000048,
    ]
)

Gammas: np.ndarray = np.array(
    [  # transition rates s⁻¹
        6.265e08,
        1.897e08,
        8.127e07,
        4.204e07,
        2.450e07,
        1.236e07,
        8.255e06,
        5.785e06,
        4.210e06,
        3.160e06,
        2.432e06,
        1.911e06,
        1.529e06,
        1.243e06,
        1.024e06,
        8.533e05,
        7.186e05,
        6.109e05,
        5.237e05,
        4.523e05,
        3.933e05,
        3.443e05,
        3.030e05,
        2.679e05,
        2.382e05,
        2.127e05,
        1.907e05,
        1.716e05,
        1.550e05,
        1.405e05,
        1.277e05,
    ]
)

# Garnett (2016): the width of Gaussian is fixed, with
# the assumption that the gas temperature fixed to 10^4 K
# this imparts a thermal broadening of 13 km s⁻¹
sigma: float = 9.08537121627923800e05  # cm s⁻¹

# leading constants[i] =
#        M_PI * e * e * oscillator_strengths[i] * transition_wavelengths[i] / (m_e * c)
leading_constants: np.ndarray = np.array(
    [  # cm²
        1.34347262962625339e-07,
        2.15386482180851912e-08,
        7.48525170087141461e-09,
        3.51375347286007472e-09,
        1.94112336271172934e-09,
        1.18916112899713152e-09,
        7.82448627128742997e-10,
        5.42930932279390593e-10,
        3.92301197282493829e-10,
        2.92796010451409027e-10,
        2.24422239410389782e-10,
        1.75895684469038289e-10,
        1.40338556137474778e-10,
        1.13995374637743197e-10,
        9.37706429662300083e-11,
        7.79453203101192392e-11,
        6.55369055970184901e-11,
        5.58100321584169051e-11,
        4.77895916635794548e-11,
        4.12301389852588843e-11,
        3.58872072638707592e-11,
        3.12745536798214080e-11,
        2.76337116167110415e-11,
        2.44791750078032772e-11,
        2.15681362798480253e-11,
        1.93850080479346101e-11,
        1.72025364178111889e-11,
        1.55051698336865945e-11,
        1.40504672409331934e-11,
        1.28383057589411395e-11,
        1.16264059622218997e-11,
    ]
)

# Lorentzian widths:
#   gammas[i] = Gammas[i] * transition_wavelengths[i] / (4 * M_PI);
gammas: np.ndarray = np.array(
    [
        6.06075804241938613e02,  # cm s⁻¹
        1.54841462408931704e02,
        6.28964942715328164e01,
        3.17730561586147395e01,
        1.82838676775503330e01,
        9.15463131005758157e00,
        6.08448802613156925e00,
        4.24977523573725779e00,
        3.08542121666345803e00,
        2.31184525202557767e00,
        1.77687796208123139e00,
        1.39477990932179852e00,
        1.11505539984541979e00,
        9.05885451682623022e-01,
        7.45877170715450677e-01,
        6.21261624902197052e-01,
        5.22994533400935269e-01,
        4.44469874827484512e-01,
        3.80923210837841919e-01,
        3.28912390446060132e-01,
        2.85949711597237033e-01,
        2.50280032040928802e-01,
        2.20224061101442048e-01,
        1.94686521675913549e-01,
        1.73082093051965591e-01,
        1.54536566013816490e-01,
        1.38539175663870029e-01,
        1.24652675945279762e-01,
        1.12585442799479921e-01,
        1.02045988802423507e-01,
        9.27433783998286437e-02,
    ]
)

# fixed width of convolution
width: int = 3  # dimensionless

# instrumental profile
instrument_profile: np.ndarray = np.array(
    [
        2.17460992138080811e-03,
        4.11623059580451742e-02,
        2.40309364651846963e-01,
        4.32707438937454059e-01,  # center pixel
        2.40309364651846963e-01,
        4.11623059580451742e-02,
        2.17460992138080811e-03,
    ]
)


def Gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    """
    G(x; sigma) = 1 / sqrt(2 pi sigma^2) * exp( - x^2 / 2 sigma^2 )
    """
    return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x ** 2) / 2 / sigma ** 2)


def Lorentzian(x: np.ndarray, gamma: float) -> np.ndarray:
    """
    L(x; gamma) = (gamma / pi ) / (x**2 + gamma**2)
    """
    return gamma / np.pi / (x ** 2 + gamma ** 2)


def Voigt(x: np.ndarray, sigma: float, gamma: float) -> np.ndarray:
    """
    Vogit line profile

    V(x; sigma, gamma) = Re[ w(z) ] / sqrt( 2 pi sigma^2 ) 
    """
    z = (x + 1j * gamma) / (np.sqrt(2) * sigma)
    return np.real(wofz(z)) / (np.sqrt(2 * np.pi) * sigma)


def voigt_absorption(
    wavelengths: np.ndarray,
    nhi: float,
    z_dla: float,
    num_lines: int = 3,
    boardening: bool = True,
) -> np.ndarray:
    """
    Voigt line profile for absorptions

    Parameters:
    ----
    wavelengths (np.ndarray) : observed wavelengths (Å)
    nhi (float) : column density of this absorber   (cm⁻²)
    z_dla (float) : the redshift of this absorber   (dimensionless)

    raw_profile =
        exp( nhi * ( - leading_constants[j] * Voigt(velocity, sigma, gammas[j] ) )  ) 

    for the relative velocity:
    velocity = 
        c * ( wavelengths * / ( transition_wavelengths[j] * (1 + z) ) - 1 )
    
    for the leading constants:
    leading_constants[i] =
       M_PI * e * e * oscillator_strengths[i] * transition_wavelengths[i] / (m_e * c)


    /* instrumental broadening */
    for (i = 0; i < num_points; i++)
        for (j = i, k = 0; j <= i + 2 * width; j++, k++)
        profile[i] += raw_profile[j] * instrument_profile[k];

    Note: 
    ----
    unit conversion from cm to A is 10^-8
    """
    # number of pixels within the input spectrum
    num_points = wavelengths.shape[0]

    # initialize a profile
    # absorption profile : dimensionless
    profile = np.zeros((num_points - 2 * width))

    # raw_profile before convolve with the instrumental profile
    raw_profile = np.empty((num_points,))

    # build the multipliers for the relative velocity
    multipliers = c / (transition_wavelengths[:num_lines] * (1 + z_dla)) / 1e8

    # compute raw Voigt profile
    total = np.empty((num_lines, raw_profile.shape[0]))

    for l in range(num_lines):
        velocity = wavelengths * multipliers[l] - c

        total[l, :] = -leading_constants[l] * Voigt(velocity, sigma, gammas[l])

    raw_profile[:] = np.exp(np.float(nhi) * np.nansum(total, axis=0))

    if boardening:
        # num_points = len(profile)

        # # instrumental broadening
        # for i in range(num_points):
        #     for k,j in enumerate(range(i, i + 2 * width + 1)):
        #         profile[i] += raw_profile[j] * instrument_profile[k]
        # return  profile
        profile[:] = np.convolve(raw_profile, instrument_profile, "valid")
        return profile

    return raw_profile
