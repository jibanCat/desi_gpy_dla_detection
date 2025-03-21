"""
plot_visual_inspect.py
======================

This script is used to plot the visual inspection of the spectra. It reads the spectra from the DESI data release and
plots the spectrum in the observed and rest-frame wavelengths. It also plots the zoomed-in Lyman-alpha region and
highlights the Lyman-alpha, Lyman-beta, and Lyman-gamma emission lines. The script also detects DLAs in the spectrum
using the Gaussian Process model and plots the posterior space and the real spectrum space. The script also compares
the DLA detections with other DLA finders, such as the CNN and TEMP.
"""
import os, re
import numpy as np

# plotting styles
import matplotlib as mpl
from matplotlib import pyplot as plt
# include .. in the path
import sys
import os

sys.path.insert(0, "..")

from astropy.table import Table


import desispec.io
from desispec.interpolation import resample_flux
from desispec.coaddition import coadd_cameras, resample_spectra_lin_or_log
from desiutil.log import log
import constants


# load QSO catalog
import desispec.io
from desispec.interpolation import resample_flux
from desispec.coaddition import coadd_cameras, resample_spectra_lin_or_log
from desiutil.log import log
import constants

from desiutil.log import log
from astropy.table import Table, vstack
import numpy as np
from scipy.interpolate import interp1d
import fitsio
import constants
import desispec.io

def read_catalog(qsocat, balmask, bytile):
    """
    read quasar catalog

    Arguments
    ---------
    qsocat (str) : path to quasar catalog
    balmask (bool) : should BAL attributes from baltools be read in?
    bytile (bool) : catalog is tilebased, default assumption is healpix

    Returns
    -------
    table of relevant attributes for quasars defined in constants.py

    """
    if constants.no_bal:
        balmask = True

    if balmask:
        try:
            # read the following columns from qsocat
            cols = [
                "TARGETID",
                "TARGET_RA",
                "TARGET_DEC",
                "Z",
                "HPXPIXEL",
                "AI_CIV",
                "NCIV_450",
                "VMIN_CIV_450",
                "VMAX_CIV_450",
                "SPECTYPE",
                "ZWARN",
            ]
            if bytile:
                cols = [
                    "TARGETID",
                    "TARGET_RA",
                    "TARGET_DEC",
                    "Z",
                    "TILEID",
                    "PETAL_LOC",
                    "AI_CIV",
                    "NCIV_450",
                    "VMIN_CIV_450",
                    "VMAX_CIV_450",
                    "SPECTYPE",
                    "ZWARN",
                ]
            catalog = Table(fitsio.read(qsocat, ext=1, columns=cols))
        except:
            log.error(f"cannot find {cols} in quasar catalog")
            exit(1)
    else:
        # read the following columns from qsocat
        cols = [
            "TARGETID",
            "TARGET_RA",
            "TARGET_DEC",
            "Z",
            "HPXPIXEL",
            "SPECTYPE",
            "ZWARN",
        ]
        if bytile:
            cols = [
                "TARGETID",
                "TARGET_RA",
                "TARGET_DEC",
                "Z",
                "TILEID",
                "PETAL_LOC",
                "SPECTYPE",
                "ZWARN",
            ]
        catalog = Table(fitsio.read(qsocat, ext=1, columns=cols))

    log.info(f"Successfully read quasar catalog: {qsocat}")

    # Apply redshift cuts
    zmask = (catalog["Z"] > constants.zmin_qso) & (catalog["Z"] < constants.zmax_qso)
    log.info(f"objects in catalog: {len(catalog)} ")
    log.info(
        f"restricting to {constants.zmin_qso} < z < {constants.zmax_qso}: {np.sum(zmask)} objects remain"
    )

    # Apply bal mask
    if constants.no_bal:
        balind = catalog["NCIV_450"] > 0
        zmask = zmask & ~balind
        log.info(f"objects in catalog without BAL: {np.sum(zmask)}")

    # Apply zwarning mask
    if constants.zwarning:
        zmask = zmask & (catalog["ZWARN"] == 0)
        log.info(f"objects in catalog without ZWARN: {np.sum(zmask)}")

    # Apply spectype mask
    if constants.is_qso:
        zmask = zmask & (catalog["SPECTYPE"] == "QSO")
        log.info(f"objects in catalog with SPECTYPE QSO: {np.sum(zmask)}")

    catalog = catalog[zmask]

    return catalog


qsocat = "/global/cfs/cdirs/desi/users/martini/bal-catalogs/kibo/QSO_cat_kibo_main_dark_healpix_v3-altbal.fits"
balmask = True
tilebased = False
release = "kibo"
program = "dark"
survey = "main"


catalog = read_catalog(qsocat, balmask, tilebased)

############## Routine to load the CNN and TEMP results ##############
from astropy.table import Table
import numpy as np
mollycat = "/global/cfs/cdirs/desi/users/mwolfson/DLA_cat/loa_combined_cat_raw.fits"
mollycat_gp = "/global/cfs/cdirs/desi/users/mwolfson/DLA_cat/loa_dla_cat_close_gp_bal_col.fits"
# mollycat_not_gp = "/global/cfs/cdirs/desi/users/mwolfson/DLA_cat/loa_dla_cat_not_gp_snr_15_bal_col_gt_910.fits"

# This loads the raw combined catalog
mollycat = Table.read(mollycat)
# This loads the GP with overlapping DLAs
mollycat_gp = Table.read(mollycat_gp)
# This loads the DLAs not in GP but in the CNN and TEMP
# mollycat_not_gp = Table.read(mollycat_not_gp)

# Some cuts to make the plot cleaner
snr = mollycat_gp["SNR_REDSIDE"]
bal_mask = mollycat_gp["AI_CIV"]
ind = snr > 15
ind = ind & (bal_mask == 0)
ind = ind & (mollycat_gp["NHI"] > 20.3)

# These are the target_ids to run
target_ids = np.unique(mollycat_gp[ind]["TARGETID"])
print(mollycat_gp[ind])

# This is the index of the target_id to run
i = 0 # Change this to run a different target_id

tid = target_ids[i]

# Load the CNN and TEMP results
ind = mollycat["TARGETID"] == tid
# CNN
z_dla_cnn = mollycat[ind]["Z_DLA_CNN"]
_ind_nan =  ~np.isnan(z_dla_cnn.value.data)
z_dla_cnn = z_dla_cnn.value.data[_ind_nan]
nhi_cnn = mollycat[ind]["NHI_CNN"]
nhi_cnn = nhi_cnn.value.data[_ind_nan]
# TEMP
z_dla_temp = mollycat[ind]["Z_DLA_TEMP"]
_ind_nan =  ~np.isnan(z_dla_temp.value.data)
z_dla_temp = z_dla_temp.value.data[_ind_nan]
nhi_temp = mollycat[ind]["NHI_TEMP"]
nhi_temp = nhi_temp.value.data[_ind_nan]

############### Steps to read a spectrum from a healpix pixel ###############
# Make the tid as an argument
# tid =  39627666508219798  
idx = np.where(catalog["TARGETID"] == tid)[0][0]

# Double check this is correct
print(catalog[idx])


# read spectra from healpix
hpx = catalog[idx]["HPXPIXEL"]
healpix = hpx

datapath = f"/global/cfs/cdirs/desi/spectro/redux/{release}/healpix/{survey}/{program}"

coaddname = f"coadd-{survey}-{program}-{str(healpix)}.fits"
coadd = os.path.join(datapath, str(healpix // 100), str(healpix), coaddname)

hpxcatalog = catalog[catalog["HPXPIXEL"] == hpx]

specobj = desispec.io.read_spectra(
    coadd,
    targetids=hpxcatalog["TARGETID"],
    skip_hdus=["EXP_FIBERMAP", "SCORES", "EXTRA_CATALOG"],
)




specobj = coadd_cameras(specobj)

################ Plot the spectrum ################

from collections import namedtuple

z_qso = catalog[idx]["Z"]

this_cat = catalog["TARGETID"][catalog["HPXPIXEL"] == hpx]
this_idx = np.where(this_cat == tid)[0][0]

# this_idx = 37

# Define a namedtuple for storing each spectrum's data
SpectrumData = namedtuple(
    "SpectrumData", ["wavelengths", "flux", "noise_variance", "pixel_mask"]
)

ivar = specobj.ivar["brz"][this_idx]
noise_variance = np.zeros(ivar.shape)
ind = ivar == 0
noise_variance[:] = np.nan
noise_variance[~ind] = 1 / ivar[~ind]

pixel_mask = specobj.mask["brz"][this_idx].astype(np.bool_)
pixel_mask[ind] = True

spectrum_data = SpectrumData(
    wavelengths = specobj.wave["brz"],
    flux = specobj.flux["brz"][this_idx],
    noise_variance = noise_variance,
    pixel_mask = pixel_mask
)


# ## DESI Spectra on the GP-DLA Finder
# Access wavelengths, flux, and noise variance from the DESI reader
wavelengths = spectrum_data.wavelengths
flux = spectrum_data.flux
noise_variance = spectrum_data.noise_variance
pixel_mask = spectrum_data.pixel_mask

rest_wavelengths = wavelengths / (1 + z_qso)

# Plot the spectrum in "observed wavelengths"
plt.figure(figsize=(16, 5))
plt.plot(wavelengths, flux)
plt.xlabel("Observed Wavelengths [$\AA$]")
plt.ylabel("Flux")
plt.title(f"Spectrum {tid} in Observed Wavelengths (z = {z_qso:.2f})")
plt.grid(True)
plt.savefig("spectrum_observed.png")
plt.clf()
plt.close()


# Emission lines for Lyα, Lyβ, and Lyman Limit
lya_wavelength = 1215.24
lyb_wavelength = 1025.72
ly_limit_wavelength = 911.76

# Zoom in on the Lyα region (wavelengths 900 Å to 1216 Å)
plt.figure(figsize=(16, 5))
plt.plot(rest_wavelengths, flux / np.mean(flux))
plt.xlabel("Rest-frame Wavelengths [$\AA$]")
plt.ylabel("Normalized Flux")
plt.ylim(-1, 5)
plt.xlim(750, 1415)
plt.title(f"Zoomed-in Lya Region for Spectrum {tid}")
plt.grid(True)

# Highlight the Lyα, Lyβ, and Lyman Limit emission lines
plt.vlines(lya_wavelength, -1, 5, color="C3", ls="--")
plt.text(lya_wavelength, 4, r"Lyα", rotation="vertical", color="C3")

plt.vlines(lyb_wavelength, -1, 5, color="C2", ls="--")
plt.text(lyb_wavelength, 4, r"Lyβ", rotation="vertical", color="C2")

plt.vlines(ly_limit_wavelength, -1, 5, color="C1", ls="--")
plt.text(ly_limit_wavelength, 4, r"Lyman Limit", rotation="vertical", color="C1")
plt.savefig("spectrum.png")
plt.clf()
plt.close()

############### Detect DLAs ###############


import numpy as np
import matplotlib.pyplot as plt
from run_bayes_select import process_single_spectrum
from gpy_dla_detection.set_parameters import Parameters
from gpy_dla_detection.model_priors import PriorCatalog
from gpy_dla_detection.dla_samples import DLASamplesMAT
from gpy_dla_detection.subdla_samples import SubDLASamplesMAT
from gpy_dla_detection.bayesian_model_selection import BayesModelSelect
from gpy_dla_detection.null_gp import NullGPMAT
from gpy_dla_detection.dla_gp import DLAGPMAT
from gpy_dla_detection.subdla_gp import SubDLAGPMAT
from gpy_dla_detection.process_helpers import initialize_results

class SpectrumProcessor:
    def __init__(self, spectra_filename, zbest_filename, learned_file, catalog_name,
                 los_catalog, dla_catalog, dla_samples_file, sub_dla_samples_file,
                 max_dlas=3, min_z_separation=3000.0, prev_tau_0=0.00554, prev_beta=3.182, k=20, dlambda=0.25,
                min_lambda=912.75,
                max_lambda=1216.75,):
        
        self.spectra_filename = spectra_filename
        self.zbest_filename = zbest_filename
        self.learned_file = learned_file
        self.catalog_name = catalog_name
        self.los_catalog = los_catalog
        self.dla_catalog = dla_catalog
        self.dla_samples_file = dla_samples_file
        self.sub_dla_samples_file = sub_dla_samples_file
        self.max_dlas = max_dlas
        self.min_z_separation = min_z_separation
        self.prev_tau_0 = prev_tau_0
        self.prev_beta = prev_beta
        
        # Initialize parameters
        self.params = Parameters(
            loading_min_lambda=910,
            loading_max_lambda=1550,
            normalization_min_lambda=1425,
            normalization_max_lambda=1475,
            min_lambda=min_lambda,
            max_lambda=max_lambda,
            dlambda=dlambda,
            k=k,
            max_noise_variance=3 ** 2,
        )
        
        # Initialize priors and sample catalogs
        self.prior = PriorCatalog(self.params, self.catalog_name, self.los_catalog, self.dla_catalog)
        self.dla_samples = DLASamplesMAT(self.params, self.prior, self.dla_samples_file)
        self.subdla_samples = SubDLASamplesMAT(self.params, self.prior, self.sub_dla_samples_file)
        
        # Bayesian model selection
        self.bayes = BayesModelSelect([0, 1, self.max_dlas], 2)
        
        # Instantiate models
        self.null_gp = NullGPMAT(self.params, self.prior, self.learned_file, prev_tau_0=self.prev_tau_0, prev_beta=self.prev_beta)
        self.dla_gp = DLAGPMAT(
            params=self.params, prior=self.prior, dla_samples=self.dla_samples, min_z_separation=self.min_z_separation,
            learned_file=self.learned_file, broadening=True, prev_tau_0=self.prev_tau_0, prev_beta=self.prev_beta
        )
        self.subdla_gp = SubDLAGPMAT(
            params=self.params, prior=self.prior, dla_samples=self.subdla_samples, min_z_separation=self.min_z_separation,
            learned_file=self.learned_file, broadening=True, prev_tau_0=self.prev_tau_0, prev_beta=self.prev_beta
        )
        
        # Initialize results dictionary
        num_spectra = 1
        num_dla_samples = self.dla_samples.log_nhi_samples.shape[0]
        self.results = initialize_results(num_spectra, self.max_dlas, num_dla_samples=num_dla_samples)

    def process_spectrum(self, idx, target_id, z_qso, wavelengths, rest_wavelengths, flux, noise_variance, pixel_mask):
        process_single_spectrum(
            idx=idx,
            target_id=target_id,
            z_qso=z_qso,
            wavelengths=wavelengths,
            rest_wavelengths=rest_wavelengths,
            flux=flux,
            noise_variance=noise_variance,
            pixel_mask=pixel_mask,
            params=self.params,
            prior=self.prior,
            dla_samples=self.dla_samples,
            subdla_samples=self.subdla_samples,
            bayes=self.bayes,
            results=self.results,
            max_dlas=self.max_dlas,
            broadening=True,
            gp=self.null_gp,
            dla_gp=self.dla_gp,
            subdla_gp=self.subdla_gp,
            min_z_separation=self.min_z_separation,
            plot_figures=False,
            max_workers=32,
            batch_size=313,
            figure_dir="figures"
        )
        return self.results
    
    def plot_results(self, z_qso, fig=None, ax=None, return_variables=False):
        results = self.results
        # How many absorbers searches you want to plot
        if self.bayes.p_dla > 0.9:
            nth_lya = 1 + results["model_posteriors"][0, 2:].argmax() # here we plot all of the searches
        else:
            nth_lya = 0
        
        lya_gp = self.dla_gp
        gp = self.null_gp

        sample_z_dlas = lya_gp.dla_samples.sample_z_dlas(
                lya_gp.this_wavelengths, lya_gp.z_qso
        )
        
        # [color sequence] convert sample log likelihoods to values in (0, 1)
        sample_log_likelihoods = lya_gp.sample_log_likelihoods[
            :, 0
        ]  # only query the DLA(1) likelihoods
        # TODO: marginalize over k DLAs
        max_like = np.nanmax(sample_log_likelihoods)
        min_like = np.nanmin(sample_log_likelihoods)
        
        colours = (sample_log_likelihoods - min_like) / (max_like - min_like)
        
        # scale to make the colour more visible
        # TODO: make it more reasonable. scatter only takes values between [0, 1].
        colours = colours * 5 - 4
        colours[colours < 0] = 0
        

        # Canvas with two panels
        if fig is None:
            fig, ax = plt.subplots(2, 1, figsize=(16, 10))

        # 1. Real spectrum space
        # N * (1~k models) * (1~k MAP dlas)
        MAP_z_dla, MAP_log_nhi = lya_gp.maximum_a_posteriori()
        # make them to be 1-D array
        map_z_dlas = MAP_z_dla[nth_lya - 1, :nth_lya]
        map_log_nhis = MAP_log_nhi[nth_lya - 1, :nth_lya]
        # feed in MAP values and get the absorption profile given (z_dlas, nhis)
        lya_mu, lya_M, lya_omega2 = lya_gp.this_dla_gp(map_z_dlas, 10 ** map_log_nhis)
        absorption = lya_mu / lya_gp.this_mu
        _this_mu = lya_gp.mu_interpolator(lya_gp.X)
        # replace the lya_mu with the mu without meanflux suppression
        lya_mu = _this_mu * absorption

        # Only plot the spectrum within the search range
        this_rest_wavelengths = lya_gp.x
        ind = (this_rest_wavelengths < lya_gp.params.lya_wavelength)
    
        this_rest_wavelengths = this_rest_wavelengths[ind]
        lya_mu = lya_mu[ind]
        # _this_mu = _this_mu[ind]

        ax[0].plot(
            (this_rest_wavelengths * (1 + z_qso)) / lya_gp.params.lya_wavelength - 1,
            lya_gp.Y[ind]
        )
        ax[0].plot(
            (this_rest_wavelengths * (1 + z_qso)) / lya_gp.params.lya_wavelength - 1,
            lya_gp.this_mu[ind],
            color="red",
            label="GP meanflux",
            ls="--",
        )
        ax[0].plot(
            (this_rest_wavelengths * (1 + z_qso)) / lya_gp.params.lya_wavelength - 1,
            lya_mu,
            label=r"$\mathcal{M}$"
            + r" HCD({n}); ".format(n=nth_lya)
            + "z_dlas = ({}); ".format(",".join("{:.3g}".format(z) for z in map_z_dlas))
            + "lognhi = ({})".format(
                ",".join("{:.3g}".format(n) for n in map_log_nhis)
            ),
            color="red",
        )
        ax[0].fill_between(
            (this_rest_wavelengths * (1 + z_qso)) / lya_gp.params.lya_wavelength - 1,
            gp.Y[ind] - 2*np.sqrt(gp.v[ind]),
            gp.Y[ind] + 2*np.sqrt(gp.v[ind]),
            label="SDSS Instrumental Uncertainty (95%)",
            color="C0",
            alpha=0.3,
        )
        
        
        # 2. Posterior space
        ax[1].scatter(
            sample_z_dlas, lya_gp.dla_samples.log_nhi_samples, c=colours,
            marker="o", alpha=0.5,
        )
        # MAP estimate
        ax[1].scatter(
            map_z_dlas, map_log_nhis,
            marker="*",
            s=100,
            color="C3",
        )
        
        # [min max sample zDLAs] instead of using min max from sample_z_dlas
        # using the zDLAs converted from wavelengths will better reflect the
        # range of wavelengths range in the this_mu plot.
        ax[1].set_xlim(sample_z_dlas.min(), z_qso)
        ax[1].set_ylim(
            lya_gp.dla_samples.log_nhi_samples.min(),
            lya_gp.dla_samples.log_nhi_samples.max(),
        )
        ax[1].set_xlabel(r"$z_{Lya}$")
        ax[1].set_ylabel(r"$log N_{HI}$")
        
        # You want the first panel has the same range
        ax[0].set_xlim(sample_z_dlas.min(), z_qso)
        ax[0].set_ylim(-1, 5)
        ax[0].legend()

        if return_variables:
            return fig, ax, nth_lya, map_z_dlas, map_log_nhis, this_rest_wavelengths, lya_mu

        return fig, ax


########## Run eBOSS trained model ##########
# Define the paths to your necessary files
spectra_filename = "/path/to/spectra-16-724.fits" # placeholder
zbest_filename = "/path/to/zbest-16-724.fits" # placeholder
learned_file = "../data/dr12q/processed/learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat"
catalog_name = "../data/dr12q/processed/catalog.mat"
los_catalog = "../data/dla_catalogs/dr9q_concordance/processed/los_catalog"
dla_catalog = "../data/dla_catalogs/dr9q_concordance/processed/dla_catalog"
dla_samples_file = "../data/dr12q/processed/dla_samples_a03.mat"
sub_dla_samples_file = "../data/dr12q/processed/subdla_samples.mat"

# Initialize the processor
processor = SpectrumProcessor(
    spectra_filename=spectra_filename,
    zbest_filename=zbest_filename,
    learned_file=learned_file,
    catalog_name=catalog_name,
    los_catalog=los_catalog,
    dla_catalog=dla_catalog,
    dla_samples_file=dla_samples_file,
    sub_dla_samples_file=sub_dla_samples_file,
    max_dlas=3,
    min_z_separation=3000.0, prev_tau_0=0.00554, prev_beta=3.182
)

# Example input parameters
idx = 0 # has to be zero

# Process the spectrum
import time

start_time = time.time()
results = processor.process_spectrum(
    idx=idx,
    target_id=tid,
    z_qso=z_qso,
    wavelengths=wavelengths,
    rest_wavelengths=rest_wavelengths,
    flux=flux,
    noise_variance=noise_variance,
    pixel_mask=pixel_mask
)

# Print results
print("p(Null) = ", results["model_posteriors"][0, 0])
print("p(SubDLA/Alternative) = ", results["model_posteriors"][0, 1])
print("p(DLA+) = ", results["model_posteriors"][0, 2:])

print("End time {}".format(time.time() - start_time))

############# Run DESI trained model #############

# Define the paths to your necessary files
learned_file = "../learnlogs/model_epoch_682.h5"
catalog_name = "../data/dr12q/processed/catalog.mat"
los_catalog = "../data/dla_catalogs/dr9q_concordance/processed/los_catalog"
dla_catalog = "../data/dla_catalogs/dr9q_concordance/processed/dla_catalog"
dla_samples_file = "../data/dr12q/processed/dla_samples_a03.mat"
sub_dla_samples_file = "../data/dr12q/processed/subdla_samples.mat"

# Set parameters dynamically, including max_dlas
max_dlas = 3  # Can change this value to reflect how many DLA models you want to process
min_z_separation = 3000.0  # Minimum redshift separation for DLA models
prev_tau_0 = 0.00246
prev_beta = 3.62

# Initialize the processor
processor_desi = SpectrumProcessor(
    spectra_filename=spectra_filename,
    zbest_filename=zbest_filename,
    learned_file=learned_file,
    catalog_name=catalog_name,
    los_catalog=los_catalog,
    dla_catalog=dla_catalog,
    dla_samples_file=dla_samples_file,
    sub_dla_samples_file=sub_dla_samples_file,
    max_dlas=3,
    min_z_separation=3000.0, prev_tau_0=prev_tau_0, prev_beta=prev_beta, dlambda=0.15, k=30,
    min_lambda=912.75,
    max_lambda=1420,
)

# Example input parameters
idx = 0

# Process the spectrum
import time

start_time = time.time()

results_desi = processor_desi.process_spectrum(
    idx=idx,
    target_id=tid,
    z_qso=z_qso,
    wavelengths=wavelengths,
    rest_wavelengths=rest_wavelengths,
    flux=flux,
    noise_variance=noise_variance,
    pixel_mask=pixel_mask
)

# Print results
print("p(Null) = ", results_desi["model_posteriors"][0, 0])
print("p(SubDLA/Alternative) = ", results_desi["model_posteriors"][0, 1])
print("p(DLA+) = ", results_desi["model_posteriors"][0, 2:])

print("End time {}".format(time.time() - start_time))


############ Plotting ############

fig, ax, nth_lya, map_z_dlas, map_log_nhis, this_rest_wavelengths, lya_mu = processor.plot_results(z_qso, return_variables=True)
plt.title("TARGETID = " + str(tid))
plt.close()
plt.clf()

_fig, _ax = processor_desi.plot_results(z_qso,)
# MAP estimate
_ax[1].scatter(
    map_z_dlas, map_log_nhis,
    marker="o",
    s=100,
    color="black",
    alpha=0.8,
)
_ax[0].plot(
    (this_rest_wavelengths * (1 + z_qso)) / processor.dla_gp.params.lya_wavelength - 1,
    lya_mu,
    label=r"$\mathcal{M}$"
    + r"eBOSS HCD({n}); ".format(n=nth_lya)
    + "z_dlas = ({}); ".format(",".join("{:.3g}".format(z) for z in map_z_dlas))
    + "lognhi = ({})".format(
        ",".join("{:.3g}".format(n) for n in map_log_nhis)
    ),
    color="black",
    ls="--",
)
_ax[0].legend()

# Print results
print("DESI Model:")
print("p(Null) = ", results_desi["model_posteriors"][0, 0])
print("p(SubDLA/Alternative) = ", results_desi["model_posteriors"][0, 1])
print("p(DLA+) = ", results_desi["model_posteriors"][0, 2:])

# Print results
print("eBOSS Model:")
print("p(Null) = ", results["model_posteriors"][0, 0])
print("p(SubDLA/Alternative) = ", results["model_posteriors"][0, 1])
print("p(DLA+) = ", results["model_posteriors"][0, 2:])

plt.savefig("plot_tid_{}.pdf".format(tid), format="pdf", dpi=150, )
plt.close()
plt.clf()

############ Comparison with other DLA Finders ############

from gpy_dla_detection.voigt import voigt_absorption



def plot_added_finder(ax, lya_gp, map_z_dlas:np.ndarray, map_log_nhis:np.ndarray, num_lines=2, marker="x", label="CNN", color="C2"):
    # lya_mu, lya_M, lya_omega2 = lya_gp.this_dla_gp(map_z_dlas, 10 ** map_log_nhis)

    nth_lya = len(map_z_dlas)
    absorption = np.ones_like(lya_gp.X)
    for i in range(len(map_log_nhis)):
        absorption *= voigt_absorption(
            lya_gp.X * (lya_gp.z_qso + 1),
            10**map_log_nhis[i],
            map_z_dlas[i],
            broadening=False,
            num_lines=num_lines,
        )
    
    _this_mu = lya_gp.mu_interpolator(lya_gp.X)
    # replace the lya_mu with the mu without meanflux suppression
    lya_mu = _this_mu * absorption

    # Only plot the spectrum within the search range
    this_rest_wavelengths = lya_gp.x
    ind = (this_rest_wavelengths < lya_gp.params.lya_wavelength)
    
    this_rest_wavelengths = this_rest_wavelengths[ind]
    lya_mu = lya_mu[ind]
    # _this_mu = _this_mu[ind]
    
    ax[0].plot(
        (this_rest_wavelengths * (1 + z_qso)) / lya_gp.params.lya_wavelength - 1,
        lya_mu,
        label=label
        + r"({n}); ".format(n=nth_lya)
        + "z_dlas = ({}); ".format(",".join("{:.3g}".format(z) for z in map_z_dlas))
        + "lognhi = ({})".format(
            ",".join("{:.3g}".format(n) for n in map_log_nhis)
        ),
        color=color,
    )
    
    # 2. Posterior space
    # MAP estimate
    ax[1].scatter(
        map_z_dlas, map_log_nhis,
        marker=marker,
        s=100,
        color=color,
    )
    return ax


############ Comparison with other DLA Finders ############
_fig, _ax = processor_desi.plot_results(z_qso,)

_ax = plot_added_finder(_ax, processor_desi.dla_gp, z_dla_cnn, nhi_cnn, num_lines=1, marker="v", color="C4")

_ax = plot_added_finder(_ax, processor_desi.dla_gp, z_dla_temp, nhi_temp, num_lines=2, marker="x", color="C2", label="TEMP")


###### GP ########
# Label the lya series lines
nth_lya = np.argmax(processor_desi.bayes.model_posteriors) - 1
MAP_z_dla, MAP_log_nhi = processor_desi.dla_gp.maximum_a_posteriori()
# make them to be 1-D array
map_z_dlas = MAP_z_dla[nth_lya - 1, :nth_lya]
map_log_nhis = MAP_log_nhi[nth_lya - 1, :nth_lya]

## Lya
_ax[0].vlines(map_z_dlas, 1.5, 2.5, color="red", )
for z in map_z_dlas:
    _ax[0].text(z, 2.4, r"Ly$\alpha$", rotation=90, color="red")

## Lyb
map_z_dlbs = (map_z_dlas  + 1)*  1025.7 / 1215.67 - 1
ind = map_z_dlbs > 912 * (1 + z_qso) / 1216 - 1 
_ax[0].vlines(map_z_dlbs[ind], 1.5, 2, ls="--", color="red", )
for z in map_z_dlbs[ind]:
    _ax[0].text(z, 2, r"Ly$\beta$", rotation=90, color="red")

## Lyg
map_z_dlgs = (map_z_dlas  + 1)*  972.5 / 1215.67 - 1
ind = map_z_dlgs > 912 * (1 + z_qso) / 1216 - 1 
_ax[0].vlines(map_z_dlgs[ind], 1.5, 1.7, ls="dotted", color="red", )
for z in map_z_dlgs[ind]:
    _ax[0].text(z, 1.7, r"Ly$\gamma$", rotation=90, color="red")

###### CNN ########
## Lya
_ax[0].vlines(z_dla_cnn, 1.5, 2.5, color="C4", )
for z in z_dla_cnn:
    _ax[0].text(z, 2.4, r"Ly$\alpha$", rotation=90, color="C4")


###### TEMP #######
## Lya
_ax[0].vlines(z_dla_temp, -0.2, -0.9, color="C2", )
for z in z_dla_temp:
    _ax[0].text(z, -0.5, r"Ly$\alpha$", rotation=90, color="C2")

## Lyb
z_dlb_temp = (z_dla_temp  + 1)*  1025.7 / 1215.67 - 1
ind = z_dlb_temp > 912 * (1 + z_qso) / 1216 - 1 
_ax[0].vlines(z_dlb_temp[ind], -0.4, -0.9, ls="--", color="C2", )
for z in z_dlb_temp[ind]:
    _ax[0].text(z, -0.5, r"Ly$\beta$", rotation=90, color="C2")


_ax[0].legend()
_ax[1].set_ylim(19.7,23)

plt.savefig("plot_tid_{}_comparison.pdf".format(tid), format="pdf", dpi=150, )
plt.close()
plt.clf()
