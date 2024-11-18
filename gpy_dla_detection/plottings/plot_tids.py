import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from desispec.io import read_spectra
from desispec.coaddition import coadd_cameras
from collections import namedtuple
from gpy_dla_detection.null_gp import NullGPMAT
from gpy_dla_detection.dla_gp import DLAGPMAT


def plot_dla_spectrum(
    target_id,
    catalog,
    processed_file,
    datapath,
    release,
    survey,
    program,
    params,
    prior,
    dla_samples,
    learned_file,  # Added learned_file as a required argument
    max_dlas=3,
    out_dir="plot_spec",
):
    """
    Plot DLA-related spectrum visualizations for a given target ID.

    Parameters:
        target_id (int): The target ID of the spectrum.
        catalog (astropy.table.Table): Catalog containing TARGETID, Z, and HPXPIXEL.
        processed_file (str): Path to the processed HDF5 file.
        datapath (str): Base path to the spectra directory.
        release (str): Data release name.
        survey (str): Survey name.
        program (str): Observing program.
        params (Parameters): Parameters object for GP modeling.
        prior (PriorCatalog): Prior catalog object.
        dla_samples (DLASamplesMAT): DLA samples object.
        learned_file (str): Path to the learned QSO model file.  # Added description
        max_dlas (int): Maximum number of DLAs to consider in modeling.
    """
    # Load processed data
    with h5py.File(processed_file, "r") as f:
        target_ids = f["target_ids"][:]
        z_qsos = f["z_qsos"][:]
        p_dlas = f["p_dlas"][:]
        model_posteriors = f["model_posteriors"][:]
        MAP_log_nhis = f["MAP_log_nhis"][:]
        MAP_z_dlas = f["MAP_z_dlas"][:]

        # Locate target in processed data
        idx_processed = np.where(target_ids == target_id)[0][0]
        z_qso = z_qsos[idx_processed]

        # Load sample log likelihoods only for this spectrum
        sample_log_likelihoods = f["sample_log_likelihoods_dla"][idx_processed, :, 0]

    # Locate target in catalog
    idx_catalog = np.where(catalog["TARGETID"] == target_id)[0][0]
    healpix = catalog[idx_catalog]["HPXPIXEL"]

    # Load the spectrum from Healpix data
    coaddname = f"coadd-{survey}-{program}-{healpix}.fits"
    coadd_path = os.path.join(datapath, str(healpix // 100), str(healpix), coaddname)
    specobj = read_spectra(coadd_path, targetids=[target_id])
    specobj = coadd_cameras(specobj)

    # Extract spectrum data
    SpectrumData = namedtuple(
        "SpectrumData", ["wavelengths", "flux", "noise_variance", "pixel_mask"]
    )
    idx_spec = 0  # Assuming one target per Healpix file
    ivar = specobj.ivar["brz"][idx_spec]
    noise_variance = np.zeros_like(ivar)
    noise_variance[ivar != 0] = 1 / ivar[ivar != 0]
    pixel_mask = specobj.mask["brz"][idx_spec].astype(bool)

    spectrum_data = SpectrumData(
        wavelengths=specobj.wave["brz"],
        flux=specobj.flux["brz"][idx_spec],
        noise_variance=noise_variance,
        pixel_mask=pixel_mask,
    )
    z_qso = z_qsos[idx_processed]
    rest_wavelengths = spectrum_data.wavelengths / (1 + z_qso)

    # Initialize models with learned_file
    null_gp = NullGPMAT(
        params, prior, learned_file=learned_file
    )  # Modified to include learned_file
    dla_gp = DLAGPMAT(
        params=params,
        prior=prior,
        dla_samples=dla_samples,
        learned_file=learned_file,  # Modified to include learned_file
    )
    # Set data for the Null, DLA, and Sub-DLA models
    for model, name in zip([null_gp, dla_gp], ["Null", "DLA"]):
        model.set_data(
            rest_wavelengths,
            spectrum_data.flux,
            spectrum_data.noise_variance,
            spectrum_data.pixel_mask,
            z_qso,
            build_model=True,
        )

    # Determine the number of absorbers to plot
    if p_dlas[idx_processed] > 0.9:
        nth_lya = 1 + model_posteriors[idx_processed, 2:].argmax()
    else:
        nth_lya = 0

    # Compute MAP and absorption profile
    map_z_dlas = MAP_z_dlas[idx_processed, :nth_lya]
    map_log_nhis = MAP_log_nhis[idx_processed, :nth_lya]
    lya_mu, _, _ = dla_gp.this_dla_gp(map_z_dlas, 10**map_log_nhis)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    spec_dir = os.path.join(out_dir, f"spec_{target_id}")
    if not os.path.exists(spec_dir):
        os.makedirs(spec_dir, exist_ok=True)

    # Plot 1: Observed spectrum
    plt.figure(figsize=(16, 5))
    plt.plot(spectrum_data.wavelengths, spectrum_data.flux, label="Observed Flux")
    plt.xlabel("Observed Wavelengths [$\AA$]")
    plt.ylabel("Flux")
    plt.title(f"Spectrum {target_id} in Observed Wavelengths (z = {z_qso:.2f})")
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig(
        os.path.join(spec_dir, f"01_{target_id}_observed.pdf"), dpi=150, format="pdf"
    )
    plt.clf()
    plt.close()

    # Plot 2: Absorption model combined with MAP
    plt.figure(figsize=(16, 5))
    plt.plot(rest_wavelengths, spectrum_data.flux, label="Observed Flux")
    plt.plot(dla_gp.X, lya_mu, label="DLA Model", color="red")
    plt.fill_between(
        dla_gp.X,
        lya_mu - 2 * np.sqrt(null_gp.v),
        lya_mu + 2 * np.sqrt(null_gp.v),
        alpha=0.3,
        label="95% Confidence Interval",
    )
    plt.xlabel("Rest-frame Wavelengths [$\AA$]")
    plt.ylabel("Flux")
    plt.ylim(-1, 5)
    plt.legend()
    plt.grid(True)
    plt.title(f"DLA Model for Spectrum {target_id}")
    # plt.show()
    plt.savefig(
        os.path.join(spec_dir, f"02_{target_id}_model.pdf"), dpi=150, format="pdf"
    )
    plt.clf()
    plt.close()

    # Plot 4: Posterior space with sample likelihoods

    # # [color sequence] convert sample log likelihoods to values in (0, 1)
    # sample_log_likelihoods = lya_gp.sample_log_likelihoods[
    #     :, 0
    # ]  # only query the DLA(1) likelihoods
    # TODO: marginalize over k DLAs
    max_like = np.nanmax(sample_log_likelihoods)
    min_like = np.nanmin(sample_log_likelihoods)

    colours = (sample_log_likelihoods - min_like) / (max_like - min_like)

    # scale to make the colour more visible
    # TODO: make it more reasonable. scatter only takes values between [0, 1].
    colours = colours * 5 - 4
    colours[colours < 0] = 0

    # Canvas with two panels
    fig, ax = plt.subplots(2, 1, figsize=(16, 10))

    # 1. Real spectrum space
    # N * (1~k models) * (1~k MAP dlas)
    map_z_dlas = MAP_z_dlas[idx_processed, :nth_lya]
    map_log_nhis = MAP_log_nhis[idx_processed, :nth_lya]
    # feed in MAP values and get the absorption profile given (z_dlas, nhis)
    lya_mu, lya_M, lya_omega2 = dla_gp.this_dla_gp(map_z_dlas, 10**map_log_nhis)

    # Only plot the spectrum within the search range
    this_rest_wavelengths = dla_gp.x
    ind = this_rest_wavelengths < dla_gp.params.lya_wavelength

    this_rest_wavelengths = this_rest_wavelengths[ind]
    lya_mu = lya_mu[ind]

    ax[0].plot(
        (this_rest_wavelengths * (1 + z_qso)) / dla_gp.params.lya_wavelength - 1,
        dla_gp.Y[ind],
    )
    ax[0].plot(
        (this_rest_wavelengths * (1 + z_qso)) / dla_gp.params.lya_wavelength - 1,
        lya_mu,
        label=r"$\mathcal{M}$"
        + r" HCD({n}); ".format(n=nth_lya)
        + "z_dlas = ({}); ".format(",".join("{:.3g}".format(z) for z in map_z_dlas))
        + "lognhi = ({})".format(",".join("{:.3g}".format(n) for n in map_log_nhis)),
        color="red",
    )
    ax[0].fill_between(
        (this_rest_wavelengths * (1 + z_qso)) / dla_gp.params.lya_wavelength - 1,
        dla_gp.Y[ind] - 2 * np.sqrt(dla_gp.v[ind]),
        dla_gp.Y[ind] + 2 * np.sqrt(dla_gp.v[ind]),
        label="SDSS Instrumental Uncertainty (95%)",
        color="C0",
        alpha=0.3,
    )

    # 2. Posterior space
    sample_z_dlas = dla_gp.dla_samples.sample_z_dlas(
        dla_gp.this_wavelengths, dla_gp.z_qso
    )

    ax[1].scatter(
        sample_z_dlas,
        dla_gp.dla_samples.log_nhi_samples,
        c=colours,
        marker="o",
        alpha=0.5,
    )
    # MAP estimate
    ax[1].scatter(
        map_z_dlas,
        map_log_nhis,
        marker="*",
        s=100,
        color="C3",
    )

    # [min max sample zDLAs] instead of using min max from sample_z_dlas
    # using the zDLAs converted from wavelengths will better reflect the
    # range of wavelengths range in the this_mu plot.
    ax[1].set_xlim(sample_z_dlas.min(), z_qso)
    ax[1].set_ylim(
        dla_gp.dla_samples.log_nhi_samples.min(),
        dla_gp.dla_samples.log_nhi_samples.max(),
    )
    ax[1].set_xlabel(r"$z_{Lya}$")
    ax[1].set_ylabel(r"$log N_{HI}$")

    # You want the first panel has the same range
    ax[0].set_xlim(sample_z_dlas.min(), z_qso)
    ax[0].set_ylim(-1, 5)
    ax[0].legend()

    # plt.show()
    plt.savefig(
        os.path.join(spec_dir, f"03_{target_id}_posteriors.pdf"), dpi=150, format="pdf"
    )
    plt.clf()
    plt.close()

    # 4. Wide-spectrum plot
    # N * (1~k models) * (1~k MAP dlas)
    # MAP_z_dla, MAP_log_nhi = lya_gp.maximum_a_posteriori()
    # # make them to be 1-D array
    # map_z_dlas = MAP_z_dla[nth_lya - 1, :nth_lya]
    # map_log_nhis = MAP_log_nhi[nth_lya - 1, :nth_lya]
    # feed in MAP values and get the absorption profile given (z_dlas, nhis)
    lya_mu, lya_M, lya_omega2 = dla_gp.this_dla_gp(map_z_dlas, 10**map_log_nhis)
    absorption = lya_mu / dla_gp.this_mu

    plt.figure(figsize=(16, 5))

    # Mean function
    plt.plot(
        null_gp.X,  # quasar spectrum's rest-frame wavelengths
        null_gp.Y,  # quasar spectrum's flux
        label="Data",
    )
    plt.fill_between(
        null_gp.X,
        null_gp.Y - 2 * np.sqrt(null_gp.v),
        null_gp.Y + 2 * np.sqrt(null_gp.v),
        label="SDSS Instrumental Uncertainty (95%)",
        color="C0",
        alpha=0.3,
    )
    plt.plot(
        null_gp.rest_wavelengths,
        null_gp.mu,
        label="GP without DLA",
        color="C3",
        ls="--",
    )
    _this_mu = null_gp.mu_interpolator(null_gp.X)
    plt.plot(
        null_gp.X,
        _this_mu * absorption,
        label="GP * absorption",
        color="C3",
    )

    # Plot the Lyman-alpha GP model's mean function (absorption)
    plt.plot(
        null_gp.X,
        null_gp.this_mu,
        label="GP mean function",
        color="C1",
        ls="--",
    )
    plt.xlabel("Rest-frame Wavelengths [$\AA$]")
    plt.ylabel("Normalized Flux")
    plt.legend()
    plt.ylim(-1, 5)
    # plt.show()
    plt.savefig(
        os.path.join(spec_dir, f"04_{target_id}_wide_spectrum.pdf"),
        dpi=150,
        format="pdf",
    )
    plt.clf()
    plt.close()

    # Save metadata
    metadata_file = os.path.join(spec_dir, "metadata.txt")
    with open(metadata_file, "w") as f:
        f.write(f"Target ID: {target_id}\n")
        f.write(f"z_qso: {z_qso}\n")
        f.write(f"p_dla: {p_dlas[idx_processed]}\n")
        f.write(f"MAP_z_dlas: {MAP_z_dlas[idx_processed, :max_dlas]}\n")
        f.write(f"MAP_log_nhis: {MAP_log_nhis[idx_processed, :max_dlas]}\n")
        f.write(f"Model posteriors: {model_posteriors[idx_processed, :]}\n")
