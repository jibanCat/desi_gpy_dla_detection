"""Make plots for the DLA dNdX estimation paper"""

import os.path as path
import numpy as np
import matplotlib

matplotlib.use("PDF")
import matplotlib.pyplot as plt
from . import calc_cddf
from .dla_data import dla_data
from .calc_cddf import DLACatalogue

from scipy.interpolate import PchipInterpolator

############## PW14 CDDF spline ##############
# From Table 2 (Results for Spline Model — Figure 7)
logN_nodes = np.array([12.0, 15.0, 17.0, 18.0, 
                       20.0, 21.0, 21.5, 22.0])
logf_nodes = np.array([-9.72, -14.41, -17.94, -19.39,
                       -21.28, -22.82, -23.95, -25.50])

# Paper specifies cubic Hermite spline (Fritsch & Carlson 1980)
cddf_spline = PchipInterpolator(logN_nodes, logf_nodes)

def log10_f_cddf(logN):
    """
    CDDF used in Figure 7 of Prochaska et al. 2014.
    A cubic Hermite spline defined by Table 2 (Spline Model).
    """
    logN = np.asarray(logN)
    logN_clip = np.clip(logN, logN_nodes[0], logN_nodes[-1])
    return cddf_spline(logN_clip)

def f_cddf(logN):
    return 10**log10_f_cddf(logN)

################ End PW14 CDDF spline ##############

save_figure = lambda filename: plt.savefig(
    "{}.pdf".format(filename), format="pdf", dpi=300
)

def do_data_plots(cat: DLACatalogue, subdir, z_dla_max=5, z_dla_cddf_min=1, z_dla_dndx_min=2,
                  lnhi_nbins=30, lnhi_min=20.0, lnhi_max=23.0, lnhi_min_dndx=20.3, lnhi_max_dndx=22.5):
    """Make a set of plots
    
    Parameters
    ----------
    cat : DLACatalogue
        The DLA catalogue object containing the data.
    subdir : str
        The output subdirectory to save plots.
    z_dla_max : float, optional
        Maximum DLA redshift for plots, by default 5
    z_dla_cddf_min : float, optional
        Minimum DLA redshift for CDDF plots, by default 1
    z_dla_dndx_min : float, optional
        Minimum DLA redshift for dNdX plots, by default 2
    lnhi_nbins : int, optional
        Number of bins for ln(NHI) histograms, by default 30
    lnhi_min : float, optional
        Minimum value for ln(NHI) histograms, by default 20.0
    lnhi_max : float, optional
        Maximum value for ln(NHI) histograms, by default 23.0

    """
    dla_data.noterdaeme_12_data()
    (l_N, cddf, cddf68, cddf95) = cat.plot_cddf(
        zmin=z_dla_cddf_min, zmax=z_dla_max, color="blue",
        lnhi_nbins=lnhi_nbins, lnhi_min=lnhi_min, lnhi_max=lnhi_max
    )
    np.savetxt(
        path.join(subdir, "cddf_all.txt"),
        (l_N, cddf, cddf68[:, 0], cddf68[:, 1], cddf95[:, 0], cddf95[:, 1]),
    )

    # Plot the PW14 spline for comparison
    logN_plot = np.linspace(lnhi_min, lnhi_max, 100)
    plt.plot(10**logN_plot, f_cddf(logN_plot), color="grey", ls="--", label="Prochaska+2014 Spline")

    # xlim (10**lnhi_min, 10**lnhi_max)
    plt.xlim(10**lnhi_min, 10**lnhi_max)
    plt.ylim(1e-28, 5e-21)
    plt.legend(loc=0)
    save_figure(path.join(subdir, "cddf_gp"))
    plt.clf()

    (l_N, cddf, cddf68, cddf95) = cat.plot_cddf(
        zmin=z_dla_cddf_min, zmax=z_dla_max, color="blue", moment=True,
        lnhi_nbins=lnhi_nbins, lnhi_min=lnhi_min, lnhi_max=lnhi_max
    )
    # Plot the PW14 spline moment for comparison
    # logN_plot = np.linspace(lnhi_min, lnhi_max, 100)
    plt.plot(10**logN_plot, 10**logN_plot * f_cddf(logN_plot), color="grey", ls="--", label="Prochaska+2014 Spline")

    plt.xlim(10**lnhi_min, 10**lnhi_max)
    plt.legend(loc=0)
    save_figure(path.join(subdir, "cddf_moment_gp"))
    plt.clf()

    # Evolution with redshift
    (l_N, cddf, cddf68, cddf95) = cat.plot_cddf(4, 5, label="4-5", color="brown",
        lnhi_nbins=lnhi_nbins, lnhi_min=lnhi_min, lnhi_max=lnhi_max
    )
    np.savetxt(
        path.join(subdir, "cddf_z45.txt"),
        (l_N, cddf, cddf68[:, 0], cddf68[:, 1], cddf95[:, 0], cddf95[:, 1]),
    )
    (l_N, cddf, cddf68, cddf95) = cat.plot_cddf(3, 4, label="3-4", color="black",
        lnhi_nbins=lnhi_nbins, lnhi_min=lnhi_min, lnhi_max=lnhi_max
    )
    np.savetxt(
        path.join(subdir, "cddf_z34.txt"),
        (l_N, cddf, cddf68[:, 0], cddf68[:, 1], cddf95[:, 0], cddf95[:, 1]),
    )
    (l_N, cddf, cddf68, cddf95) = cat.plot_cddf(2.5, 3, label="2.5-3", color="green",
        lnhi_nbins=lnhi_nbins, lnhi_min=lnhi_min, lnhi_max=lnhi_max
    )
    np.savetxt(
        path.join(subdir, "cddf_z253.txt"),
        (l_N, cddf, cddf68[:, 0], cddf68[:, 1], cddf95[:, 0], cddf95[:, 1]),
    )
    (l_N, cddf, cddf68, cddf95) = cat.plot_cddf(2, 2.5, label="2-2.5", color="blue",
        lnhi_nbins=lnhi_nbins, lnhi_min=lnhi_min, lnhi_max=lnhi_max
    )
    np.savetxt(
        path.join(subdir, "cddf_z225.txt"),
        (l_N, cddf, cddf68[:, 0], cddf68[:, 1], cddf95[:, 0], cddf95[:, 1]),
    )
    # Plot the PW14 spline for comparison
    plt.plot(10**logN_plot, f_cddf(logN_plot), color="grey", ls="--", label="Prochaska+2014 Spline")

    plt.xlim(10**lnhi_min, 10**lnhi_max)
    plt.ylim(1e-28, 5e-21)
    plt.legend(loc=0)
    save_figure(path.join(subdir, "cddf_zz_gp"))
    plt.clf()

    # dNdX
    dla_data.dndx_not()
    dla_data.dndx_pro()
    (z_cent, dNdX, dndx68, dndx95) = cat.plot_line_density(
        zmin=z_dla_dndx_min, zmax=z_dla_max, lnhi_min=lnhi_min_dndx, lnhi_max=lnhi_max_dndx
    )
    np.savetxt(
        path.join(subdir, "dndx_all.txt"),
        (z_cent, dNdX, dndx68[:, 0], dndx68[:, 1], dndx95[:, 0], dndx95[:, 1]),
    )
    plt.legend(loc=0)
    plt.ylim(0, 0.16)
    save_figure(path.join(subdir, "dndx_gp"))
    plt.clf()

    # Omega_DLA
    dla_data.omegahi_not()
    dla_data.omegahi_pro()
    dla_data.crighton_omega()
    (z_cent, omega_dla, omega_dla_68, omega_dla_95) = cat.plot_omega_dla(
        zmin=z_dla_dndx_min, zmax=z_dla_max
    )
    #     cat.tophat_prior = True
    #     cat.plot_omega_dla(zmax=5, label="Tophat Prior", twosigma=False)
    #     cat.tophat_prior = False
    np.savetxt(
        path.join(subdir, "omega_dla_all.txt"),
        (
            z_cent,
            omega_dla,
            omega_dla_68[:, 0],
            omega_dla_68[:, 1],
            omega_dla_95[:, 0],
            omega_dla_95[:, 1],
        ),
    )
    plt.legend(loc=0)
    plt.xlim(2, 5)
    plt.ylim(0, 2.5)
    save_figure(path.join(subdir, "omega_gp"))
    plt.clf()


def do_sample_error_check(cat, subdir, z_dla_max=5):
    """Do a bunch of resamplings to check the effect of sample variance."""
    # dNdX/Omega_DLA
    cat.plot_dndx_sample_errors(z_max=z_dla_max, nsample=13)
    plt.legend(loc=0)
    plt.ylim(0, 0.16)
    save_figure(path.join(subdir, "dndx_gp_resample"))
    plt.clf()
    cat.plot_omega_sample_errors(z_max=z_dla_max, nsample=13)
    plt.legend(loc=0)
    plt.ylim(0, 2.5)
    save_figure(path.join(subdir, "omega_gp_resample"))
    plt.clf()


def do_check_p_thresh(cat, subdir, z_dla_max=5):
    """Check the effect of very unlikely samples"""
    cat.p_thresh_sample = 1e-4
    cat.plot_line_density(zmax=z_dla_max, label=r"$p_\mathrm{sample} = 10^{-4}$")
    cat.p_thresh_sample = 1e-2
    cat.plot_line_density(zmax=z_dla_max, label=r"$p_\mathrm{sample} = 10^{-2}$")
    cat.p_thresh_sample = 1e-4
    cat.p_thresh_spec = 0.1
    cat.plot_line_density(zmax=z_dla_max, label=r"$p_\mathrm{spec} = 10^{-1}$")
    plt.legend(loc=0)
    save_figure(path.join(subdir, "dndx_p_thresh"))
    plt.clf()


def do_pixel_noise_check(cat, subdir, z_dla_max=5):
    """Check effect of removing spectra with a low SNR."""
    cat.set_snr(1)
    nt = cat.noise_thresh
    cat.filter_noisy_pixels = True
    cat.plot_omega_dla(zmax=z_dla_max, label="N < 0.5")
    cat.noise_thresh = 1.0
    cat.plot_omega_dla(zmax=z_dla_max, label="N < 1")
    cat.noise_thresh = 0.25**2
    cat.plot_omega_dla(zmax=z_dla_max, label="N < 0.25")
    plt.legend(loc=0)
    save_figure(path.join(subdir, "omega_gp_pix_noise"))
    plt.clf()

    cat.plot_line_density(zmax=z_dla_max, label="N < 0.5")
    cat.noise_thresh = 1.0
    cat.plot_line_density(zmax=z_dla_max, label="N < 1")
    cat.noise_thresh = 0.25**2
    cat.plot_line_density(zmax=z_dla_max, label="N < 0.25")
    plt.legend(loc=0)
    save_figure(path.join(subdir, "dndx_gp_pix_noise"))
    plt.clf()
    cat.noise_thresh = nt
    cat.filter_noisy_pixels = False


def do_snr_check(cat: DLACatalogue, subdir, z_dla_max=5, z_dla_cddf_min=1, z_dla_dndx_min=2,
                 lnhi_nbins=30, lnhi_min=20.0, lnhi_max=23.0, lnhi_min_dndx=20.3, lnhi_max_dndx=22.5):
    """Check effect of removing spectra with a low SNR."""
    first_snr = cat.snr_thresh

    # [CDDF]
    dla_data.noterdaeme_12_data()
    cat.set_snr(-2)
    cat.plot_cddf(zmin=z_dla_cddf_min, zmax=z_dla_max, label="ALL GP", color="C0",
                  lnhi_nbins=lnhi_nbins, lnhi_min=lnhi_min, lnhi_max=lnhi_max)
    cat.set_snr(2)
    cat.plot_cddf(zmin=z_dla_cddf_min, zmax=z_dla_max, label="SNR > 2", color="C1",
                  lnhi_nbins=lnhi_nbins, lnhi_min=lnhi_min, lnhi_max=lnhi_max)
    cat.set_snr(4)
    cat.plot_cddf(zmin=z_dla_cddf_min, zmax=z_dla_max, label="SNR > 4", color="C2",
                  lnhi_nbins=lnhi_nbins, lnhi_min=lnhi_min, lnhi_max=lnhi_max)
    # Plot the PW14 spline for comparison
    logN_plot = np.linspace(lnhi_min, lnhi_max, 100)
    plt.plot(10**logN_plot, f_cddf(logN_plot), color="grey", ls="--", label="Prochaska+2014 Spline")

    plt.xlim(10**lnhi_min, 10**lnhi_max)
    plt.ylim(1e-28, 5e-21)
    plt.legend(loc=0)
    save_figure(path.join(subdir, "cddf_gp_snr"))
    plt.clf()

    # [Omega_DLA]
    dla_data.omegahi_not()
    dla_data.omegahi_pro()
    dla_data.crighton_omega()
    cat.set_snr(-2)
    cat.plot_omega_dla(zmin=z_dla_dndx_min, zmax=z_dla_max, label="All GP")
    cat.set_snr(2)
    cat.plot_omega_dla(zmin=z_dla_dndx_min, zmax=z_dla_max, label="SNR > 2")
    cat.set_snr(4)
    cat.plot_omega_dla(zmin=z_dla_dndx_min, zmax=z_dla_max, label="SNR > 4")
    #     cat.set_snr(8)
    #     cat.plot_omega_dla(zmax=5,label="SNR > 8")
    plt.legend(loc=0)
    save_figure(path.join(subdir, "omega_gp_snr"))
    plt.clf()

    # [dNdX]
    dla_data.dndx_not()
    dla_data.dndx_pro()
    cat.set_snr(-2)
    cat.plot_line_density(zmin=z_dla_dndx_min, zmax=z_dla_max, label="All GP",
                          lnhi_min=lnhi_min_dndx, lnhi_max=lnhi_max_dndx)
    cat.set_snr(2)
    cat.plot_line_density(zmin=z_dla_dndx_min, zmax=z_dla_max, label="SNR > 2",
                          lnhi_min=lnhi_min_dndx, lnhi_max=lnhi_max_dndx)
    cat.set_snr(4)
    cat.plot_line_density(zmin=z_dla_dndx_min, zmax=z_dla_max, label="SNR > 4",
                          lnhi_min=lnhi_min_dndx, lnhi_max=lnhi_max_dndx)
    #     cat.set_snr(8)
    #     cat.plot_line_density(zmax=5, label="SNR > 8")
    plt.legend(loc=0)
    save_figure(path.join(subdir, "dndx_gp_snr"))
    plt.clf()
    cat.set_snr(first_snr)


def do_lowzcut_check(cat, subdir, z_dla_max=5, z_dla_dndx_min=2, lnhi_min_dndx=20.3, lnhi_max_dndx=22.5):
    """Check effect of the low-z cut."""
    lowzcut = cat.lowzcut
    cat.lowzcut = True
    cat.plot_omega_dla(zmin=z_dla_dndx_min, zmax=z_dla_max, label="Cutting")
    cat.lowzcut = False
    cat.plot_omega_dla(zmin=z_dla_dndx_min, zmax=z_dla_max, label="Not cutting")
    plt.legend(loc=0)
    save_figure(path.join(subdir, "omega_gp_lowz"))
    plt.clf()

    cat.lowzcut = True
    cat.plot_line_density(zmin=z_dla_dndx_min, zmax=z_dla_max, label="Cutting",
                          lnhi_min=lnhi_min_dndx, lnhi_max=lnhi_max_dndx)
    cat.lowzcut = False
    cat.plot_line_density(zmin=z_dla_dndx_min, zmax=z_dla_max, label="Not cutting",
                          lnhi_min=lnhi_min_dndx, lnhi_max=lnhi_max_dndx)
    plt.ylim(0, 0.12)
    plt.legend(loc=0)
    save_figure(path.join(subdir, "dndx_gp_lowz"))
    plt.clf()
    cat.lowzcut = lowzcut


def do_highzcut_check(cat, subdir, z_dla_max=5, z_dla_dndx_min=2, lnhi_min_dndx=20.3, lnhi_max_dndx=22.5):
    """Check effect of the high-z cut."""
    highzcut = cat.highzcut
    cat.highzcut = True
    cat.plot_omega_dla(zmin=z_dla_dndx_min, zmax=z_dla_max, label="Tail cutting")
    cat.highzcut = False
    cat.plot_omega_dla(zmin=z_dla_dndx_min, zmax=z_dla_max, label="Not tail cutting")
    plt.legend(loc=0)
    save_figure(path.join(subdir, "omega_gp_lowz"))
    plt.clf()

    cat.highzcut = True
    cat.plot_line_density(zmin=z_dla_dndx_min, zmax=z_dla_max, label="Tail cutting",
                          lnhi_min=lnhi_min_dndx, lnhi_max=lnhi_max_dndx)
    cat.highzcut = False
    cat.plot_line_density(zmin=z_dla_dndx_min, zmax=z_dla_max, label="Not tail cutting",
                          lnhi_min=lnhi_min_dndx, lnhi_max=lnhi_max_dndx)
    plt.ylim(0, 0.12)
    plt.legend(loc=0)
    save_figure(path.join(subdir, "dndx_gp_highz"))
    plt.clf()
    cat.lowzcut = highzcut


def do_2dla_plots(cat, subdir, z_dla_max=5):
    """Check the effect of a second DLA. No longer included in catalogue"""
    # Omega_DLA in variance vs bayesian mode
    cat.second_dla = False
    cat.plot_omega_dla(zmax=z_dla_max, label="Confidence interval")
    cat.second_dla = True
    cat.plot_omega_dla_var(zmax=z_dla_max, label="Variance")
    plt.legend(loc=0)
    save_figure(path.join(subdir, "omega_gp_diff"))
    plt.clf()

    # dNdX
    # Check effect of the second DLA
    cat.plot_line_density(zmax=z_dla_max, label="Two-DLA")
    cat.second_dla = False
    cat.plot_line_density(zmax=z_dla_max, label="One-DLA")
    cat.second_dla = True
    plt.legend(loc=0)
    save_figure(path.join(subdir, "dndx_2dla"))
    plt.clf()

    cat.plot_omega_dla(zmax=z_dla_max, label="Two-DLA")
    cat.second_dla = False
    cat.plot_omega_dla(zmax=z_dla_max, label="One-DLA")
    cat.second_dla = True
    plt.legend(loc=0)
    save_figure(path.join(subdir, "omega_2dla"))
    plt.clf()


def do_qso_split(cat, subdir, z_dla_max=5.0, z_dla_dndx_min=2, lnhi_min_dndx=20.3, lnhi_max_dndx=22.5):
    """Check the effect of the quasar redshift."""
    # Check z_qso split
    oldcond = cat.condition
    high_z = (2.5, 3.0, 3.5, 5.0)
    low_z = (2.0, 2.5, 3.0, 3.5)
    for high_z_qso, z_qso_split in zip(high_z, low_z):
        # here should actually use cat.z_qsos, since there is 3000 km/s difference z_qso and z_max
        # [oldcond] you should select based on oldcond
        cat.condition = (
            oldcond * (cat.z_max() < high_z_qso) * (cat.z_max() > z_qso_split)
        )
        cat.plot_omega_dla(
            label="$"
            + str(high_z_qso)
            + " > z_\mathrm{QSO} > "
            + str(z_qso_split)
            + "$",
            zmin=z_dla_dndx_min,
            zmax=z_dla_max,
        )
    plt.ylim(ymin=0)
    plt.legend(loc=0)
    save_figure(path.join(subdir, "omega_gp_zqso" + str(cat.lowzcut)))
    plt.clf()

    for high_z_qso, z_qso_split in zip(high_z, low_z):
        cat.condition = (
            oldcond * (cat.z_max() < high_z_qso) * (cat.z_max() > z_qso_split)
        )
        cat.plot_line_density(
            label="$"
            + str(high_z_qso)
            + " > z_\mathrm{QSO} > "
            + str(z_qso_split)
            + "$",
            zmin=z_dla_dndx_min,
            zmax=z_dla_max,
            lnhi_min=lnhi_min_dndx,
            lnhi_max=lnhi_max_dndx,
        )
    plt.ylim(ymin=0, ymax=0.15)
    plt.legend(loc=0)
    save_figure(path.join(subdir, "dndx_gp_zqso" + str(cat.lowzcut)))
    plt.clf()

    # [DR16Q] check lowzcut + zQSO splits
    lowzcut = cat.lowzcut
    cat.lowzcut = True
    if cat.z_max_lyb == False:
        high_z = (2.5, 3.0, 3.5, 5.0)
        low_z = (2.0, 2.5, 3.0, 3.5)
        for high_z_qso, z_qso_split in zip(high_z, low_z):
            # here should actually use cat.z_qsos, since there is 3000 km/s difference z_qso and z_max
            # [oldcond] you should select based on oldcond
            cat.condition = (
                oldcond * (cat.z_max() < high_z_qso) * (cat.z_max() > z_qso_split)
            )
            cat.plot_omega_dla(
                label="$"
                + str(high_z_qso)
                + " > z_\mathrm{QSO} > "
                + str(z_qso_split)
                + "$"
                + " Cutting",
                zmin=z_dla_dndx_min,
                zmax=z_dla_max,
            )
        plt.ylim(ymin=0)
        plt.legend(loc=0)
        save_figure(path.join(subdir, "omega_gp_zqso" + str(cat.lowzcut)))
        plt.clf()

        for high_z_qso, z_qso_split in zip(high_z, low_z):
            cat.condition = (
                oldcond * (cat.z_max() < high_z_qso) * (cat.z_max() > z_qso_split)
            )
            cat.plot_line_density(
                label="$"
                + str(high_z_qso)
                + " > z_\mathrm{QSO} > "
                + str(z_qso_split)
                + "$"
                + " Cutting",
                zmin=z_dla_dndx_min,
                zmax=z_dla_max,
                lnhi_min=lnhi_min_dndx,
                lnhi_max=lnhi_max_dndx
            )
        plt.ylim(ymin=0, ymax=0.15)
        plt.legend(loc=0)
        save_figure(path.join(subdir, "dndx_gp_zqso" + str(cat.lowzcut)))
        plt.clf()

    # [DR16Q] check SNR cut + zQSO cuts
    cat.lowzcut = False
    old_snr = cat.snr_thresh
    cat.set_snr(4)

    high_z = (2.5, 3.0, 3.5, 5.0)
    low_z = (2.0, 2.5, 3.0, 3.5)
    for high_z_qso, z_qso_split in zip(high_z, low_z):
        # here should actually use cat.z_qsos, since there is 3000 km/s difference z_qso and z_max
        # [oldcond] you should select based on oldcond
        cat.condition = (
            oldcond * (cat.z_max() < high_z_qso) * (cat.z_max() > z_qso_split)
        )
        cat.plot_omega_dla(
            label="$"
            + str(high_z_qso)
            + " > z_\mathrm{QSO} > "
            + str(z_qso_split)
            + "$"
            + " SNR > 4",
            zmin=z_dla_dndx_min,
            zmax=z_dla_max,
        )
    plt.ylim(ymin=0)
    plt.legend(loc=0)
    save_figure(path.join(subdir, "omega_gp_zqso_snr"))
    plt.clf()

    for high_z_qso, z_qso_split in zip(high_z, low_z):
        cat.condition = (
            oldcond * (cat.z_max() < high_z_qso) * (cat.z_max() > z_qso_split)
        )
        cat.plot_line_density(
            label="$"
            + str(high_z_qso)
            + " > z_\mathrm{QSO} > "
            + str(z_qso_split)
            + "$"
            + " SNR > 4",
            zmin=z_dla_dndx_min,
            zmax=z_dla_max,
            lnhi_min=lnhi_min_dndx,
            lnhi_max=lnhi_max_dndx,
        )
    plt.ylim(ymin=0, ymax=0.15)
    plt.legend(loc=0)
    save_figure(path.join(subdir, "dndx_gp_zqso_snr"))
    plt.clf()

    # [DR16Q] check highzcut + zQSO splits
    highzcut = cat.highzcut
    cat.highzcut = True
    if cat.z_max_lyb == False:

        high_z = (2.5, 3.0, 3.5, 5.0)
        low_z = (2.0, 2.5, 3.0, 3.5)
        for high_z_qso, z_qso_split in zip(high_z, low_z):
            # here should actually use cat.z_qsos, since there is 3000 km/s difference z_qso and z_max
            # [oldcond] you should select based on oldcond
            cat.condition = (
                oldcond * (cat.z_max() < high_z_qso) * (cat.z_max() > z_qso_split)
            )
            cat.plot_omega_dla(
                label="$"
                + str(high_z_qso)
                + " > z_\mathrm{QSO} > "
                + str(z_qso_split)
                + "$"
                + "Tail cutting",
                zmin=z_dla_dndx_min,
                zmax=z_dla_max,
            )
        plt.ylim(ymin=0)
        plt.legend(loc=0)
        save_figure(path.join(subdir, "omega_gp_zqso_tail" + str(cat.highzcut)))
        plt.clf()

        for high_z_qso, z_qso_split in zip(high_z, low_z):
            cat.condition = (
                oldcond * (cat.z_max() < high_z_qso) * (cat.z_max() > z_qso_split)
            )
            cat.plot_line_density(
                label="$"
                + str(high_z_qso)
                + " > z_\mathrm{QSO} > "
                + str(z_qso_split)
                + "$"
                + "Tail cutting",
                zmin=z_dla_dndx_min,
                zmax=z_dla_max,
                lnhi_min=lnhi_min_dndx,
                lnhi_max=lnhi_max_dndx,
            )
        plt.ylim(ymin=0, ymax=0.15)
        plt.legend(loc=0)
        save_figure(path.join(subdir, "dndx_gp_zqso_tail" + str(cat.highzcut)))
        plt.clf()

    # assign the old conditions
    cat.snr_thresh = old_snr
    cat.lowzcut = lowzcut
    cat.condition = oldcond
    cat.highzcut = highzcut


def do_length_split(cat, subdir, z_dla_max=5, lnhi_min_dndx=20.3, lnhi_max_dndx=22.5):
    """Check the effect of the quasar redshift."""
    # Check z_qso split
    oldcond = cat.condition
    high_z = (0.2, 0.4, 0.6, 0.8, 2)
    low_z = (0.0, 0.2, 0.4, 0.6, 0.8)
    z_diff = cat.z_max() - cat.z_min()
    for high_z_qso, z_qso_split in zip(high_z, low_z):
        cat.condition = oldcond * (z_diff < high_z_qso) * (z_diff > z_qso_split)
        cat.plot_omega_dla(
            label=str(high_z_qso) + " > zQSO > " + str(z_qso_split), zmax=z_dla_max
        )
    plt.ylim(ymin=0)
    plt.legend(loc=0)
    save_figure(path.join(subdir, "omega_gp_zdiff"))
    plt.clf()

    for high_z_qso, z_qso_split in zip(high_z, low_z):
        cat.condition = oldcond * (z_diff < high_z_qso) * (z_diff > z_qso_split)
        cat.plot_line_density(
            label=str(high_z_qso) + " > zQSO > " + str(z_qso_split), zmax=z_dla_max,
            lnhi_min=lnhi_min_dndx, lnhi_max=lnhi_max_dndx,
        )
    plt.ylim(ymin=0, ymax=0.1)
    plt.legend(loc=0)
    save_figure(path.join(subdir, "dndx_gp_zdiff"))
    plt.clf()
    cat.condition = oldcond


def do_compare_plots(cat7, cat7s, subdir, label, z_dla_max=5,
                     lnhi_nbins=30, lnhi_min=20.0, lnhi_max=23.0,
                     lnhi_min_dndx=20.3, lnhi_max_dndx=22.5):
    """Plots to compare two cddfs"""
    # Check the effect of the 5km/s split
    # dNdX
    cat7.plot_line_density(zmax=z_dla_max, lnhi_min=lnhi_min_dndx, lnhi_max=lnhi_max_dndx)
    cat7s.plot_line_density(zmax=z_dla_max, label=label, lnhi_min=lnhi_min_dndx, lnhi_max=lnhi_max_dndx)
    plt.legend(loc=0)
    save_figure(path.join(subdir, "dndx_" + label))
    plt.clf()

    # Omega_DLA
    cat7.plot_cddf(zmax=z_dla_max, color="blue",
                   lnhi_nbins=lnhi_nbins, lnhi_min=lnhi_min, lnhi_max=lnhi_max)
    cat7s.plot_cddf(zmax=z_dla_max, color="red", label=label,
                   lnhi_nbins=lnhi_nbins, lnhi_min=lnhi_min, lnhi_max=lnhi_max)
    # Plot the PW14 spline for comparison
    logN_plot = np.linspace(lnhi_min, lnhi_max, 100)
    plt.plot(10**logN_plot, f_cddf(logN_plot), color="grey", ls="--", label="Prochaska+2014 Spline")

    plt.xlim(10**lnhi_min, 10**lnhi_max)
    plt.ylim(1e-28, 5e-21)
    plt.legend(loc=0)
    save_figure(path.join(subdir, "cddf_" + label))
    plt.clf()

    # Omega_DLA
    cat7.plot_omega_dla(zmax=z_dla_max)
    cat7s.plot_omega_dla(zmax=z_dla_max, label=label)
    plt.legend(loc=0)
    save_figure(path.join(subdir, "omega_" + label))
    plt.clf()


def do_dla_statistics_plots(
    cat12: calc_cddf.DLACatalogue,
    subdir: str,
    z_dla_cddf_min: float = 1.0,
    z_dla_dndx_min: float = 2.0,
    z_dla_max: float = 5.0,
    high_z_qso: float = 7.0,
    low_z_qso: float = 2.0,
    lnhi_nbins: int = 30,
    lnhi_min: float = 20.0,
    lnhi_max: float = 23.0,
    lnhi_min_dndx: float = 20.3,
    lnhi_max_dndx: float = 22.5
):
    """
    Do the plotting for CDDF, dN/dX, OmegaDLA,
    including zQSO splitting checks, snr checks, and lowz cut checks.

    Parameters
    ----------
    cat12 : calc_cddf.DLACatalogue
        The DLA catalogue object containing the data.
    subdir : str
        The output subdirectory to save plots.
    z_dla_cddf_min : float, optional
        Minimum DLA redshift for CDDF plots, by default 1.0
    z_dla_dndx_min : float, optional
        Minimum DLA redshift for dNdX plots, by default 2.0
    z_dla_max : float, optional
        Maximum DLA redshift for plots, by default 5.0
    high_z_qso : float, optional
        Upper limit for quasar redshift split, by default 7.0
    low_z_qso : float, optional
        Lower limit for quasar redshift split, by default 2.0
    lnhi_nbins : int, optional
        Number of bins for ln(NHI) histograms, by default 30
    lnhi_min : float, optional
        Minimum value for ln(NHI) histograms, by default 20.0
    lnhi_max : float, optional
        Maximum value for ln(NHI) histograms, by default 23.0
    lnhi_min_dndx : float, optional
        Minimum ln(NHI) for dNdX calculations, by default 20.3 (DLA threshold)
    lnhi_max_dndx : float, optional
        Maximum ln(NHI) for dNdX calculations, by default 22.5
    """
    oldcond = cat12.condition

    # instead of using z_map like sbird's original code, I use z_qsos
    # [oldcond] here should select within the old cond not create a new cond
    cat12.condition = oldcond * (cat12.z_qsos < high_z_qso) * (cat12.z_qsos > low_z_qso)

    do_data_plots(
        cat12,
        subdir,
        z_dla_max=z_dla_max,
        z_dla_cddf_min=z_dla_cddf_min,
        z_dla_dndx_min=z_dla_dndx_min,
        lnhi_nbins=lnhi_nbins,
        lnhi_min=lnhi_min,
        lnhi_max=lnhi_max,
        lnhi_min_dndx=lnhi_min_dndx,
        lnhi_max_dndx=lnhi_max_dndx,
    )
    do_snr_check(
        cat12,
        subdir,
        z_dla_max=z_dla_max,
        z_dla_cddf_min=z_dla_cddf_min,
        z_dla_dndx_min=z_dla_dndx_min,
        lnhi_nbins=lnhi_nbins,
        lnhi_min=lnhi_min,
        lnhi_max=lnhi_max,
        lnhi_min_dndx=lnhi_min_dndx,
        lnhi_max_dndx=lnhi_max_dndx,
    )
    do_qso_split(cat12, subdir, z_dla_max=z_dla_max, z_dla_dndx_min=z_dla_dndx_min, lnhi_min_dndx=lnhi_min_dndx, lnhi_max_dndx=lnhi_max_dndx)
    do_lowzcut_check(cat12, subdir, z_dla_max=z_dla_max, z_dla_dndx_min=z_dla_dndx_min, lnhi_min_dndx=lnhi_min_dndx, lnhi_max_dndx=lnhi_max_dndx)
    do_highzcut_check(cat12, subdir, z_dla_max=z_dla_max, z_dla_dndx_min=z_dla_dndx_min, lnhi_min_dndx=lnhi_min_dndx, lnhi_max_dndx=lnhi_max_dndx)

    cat12.condition = oldcond


if __name__ == "__main__":
    # DR7 data
    # Using old samples
    # cat7ss = calc_cddf.DLACatalogue(processed_file="processed_qsos_dr7q.mat", snr=-2)
    # do_snr_check(cat7ss, "DR7")

    # cat7 = calc_cddf.DLACatalogue(processed_file="processed_qsos_dr7q.mat")
    # do_data_plots(cat7,"DR7")
    # print("Done data plots")
    # do_check_plots(cat7,"DR7")
    # print("Done check plots")
    # cat7p = calc_cddf.DLACatalogue(processed_file="processed_qsos_dr7q.mat")
    # do_check_p_thresh(cat7p, "DR7")
    # del cat7p
    # print("Done p_thresh")
    # do_pixel_noise_check(cat7ss, "DR7")
    # del cat7ss
    # print("Done SNR")

    # cat7s = calc_cddf.DLACatalogue(processed_file="processed_qsos_dr7q_5kms_separation.mat")

    # do_compare_plots(cat7,cat7s,"DR7", label="5kms")

    # DR12 data
    cat12 = calc_cddf.DLACatalogue(
        processed_file="processed_qsos_dr12q_lyb_lya.mat",
        snrs_file="snrs_qsos_dr12.mat",
    )
    do_data_plots(cat12, "DR12")
    cat12.lowzcut = False
    do_qso_split(cat12, "DR12")
    cat12.lowzcut = True
    do_qso_split(cat12, "DR12")
    cat12.lowzcut = False
    #     do_length_split(cat12, "DR12")
    do_lowzcut_check(cat12, "DR12")
    do_snr_check(cat12, "DR12")
    do_sample_error_check(cat12, "DR12")
    # do_pixel_noise_check(cat12, "DR12")
    # cat12p = calc_cddf.DLACatalogue(processed_file="processed_qsos_dr12q.mat", snrs_file = "snrs_qsos_dr12.mat")
    # do_check_p_thresh(cat12p, "DR12")
