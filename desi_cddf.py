import argparse
import os
from CDDF_analysis.calc_cddf import DLACatalogue
from CDDF_analysis.make_plots import do_dla_statistics_plots

def main():
    parser = argparse.ArgumentParser(description="Run DLA Catalogue analysis and generate statistics plots.")

    # File path arguments
    parser.add_argument("--processed_file", type=str, required=True, help="Path to the processed file.")
    parser.add_argument("--sample_file", type=str, required=True, help="Path to the sample file.")
    parser.add_argument("--catalog_file", type=str, required=True, help="Path to the catalog file.")

    # Output directory prefix
    parser.add_argument("--output_prefix", type=str, default="CDDF_analysis", 
                        help="Prefix for the output directory (default: 'CDDF_analysis').")

    # Plotting and analysis parameters
    parser.add_argument("--z_dla_max", type=float, default=5.5, help="Maximum DLA redshift.")
    parser.add_argument("--high_z_qso", type=float, default=7, help="Upper limit for QSO redshift.")
    parser.add_argument("--low_z_qso", type=float, default=2, help="Lower limit for QSO redshift.")
    parser.add_argument("--occams_razor", type=float, default=1, help="Occam's razor penalty.")
    parser.add_argument("--sub_dla", action="store_true", help="Include sub-DLAs in analysis.")
    parser.add_argument("--snr", type=float, default=-2, help="Signal-to-noise ratio cut.")
    parser.add_argument("--lowzcut", action="store_true", help="Apply low redshift cut.")
    parser.add_argument("--highzcut", action="store_true", help="Apply high redshift cut.")
    parser.add_argument("--z_max_lyb", action="store_true", help="Adjust for Ly-beta peak.")
    parser.add_argument("--z_min_lyb", action="store_true", help="Adjust for Ly-alpha range.")
    parser.add_argument("--min_obs_wavelength_cut", action="store_true", help="Apply minimum observed wavelength cut.")
    parser.add_argument("--min_obs_wavelength", type=float, default=4000, help="Minimum observed wavelength in Angstroms.")
    parser.add_argument("--second", type=int, default=1, help="Allow up to `second + 1` DLAs per QSO.")
    parser.add_argument("--high_nhi_cut", action="store_true", help="Apply high NHI cut.")
    parser.add_argument("--high_nhi_cut_value", type=float, default=22.0, help="High NHI cut value.")
    parser.add_argument("--bins_per_z", type=int, default=6, help="Number of bins per redshift.")
    parser.add_argument("--lnhi_nbins", type=int, default=30, help="Number of ln(NHI) bins.")
    parser.add_argument("--lnhi_min", type=float, default=20.0, help="Minimum ln(NHI) for histograms.")
    parser.add_argument("--lnhi_max", type=float, default=23.0, help="Maximum ln(NHI) for histograms.")
    parser.add_argument("--lnhi_min_dndx", type=float, default=20.3, help="Minimum ln(NHI) for dNdX calculations.")
    parser.add_argument("--lnhi_max_dndx", type=float, default=22.5, help="Maximum ln(NHI) for dNdX calculations.")

    args = parser.parse_args()

    # Generate a dynamic subdirectory name with a user-defined prefix
    subdir = os.path.join(
        args.output_prefix,
        f"desi_snr{args.snr}_zqsos_{args.low_z_qso}-{args.high_z_qso}_dla_{args.z_dla_max}_"
        f"occam_{args.occams_razor}_dla_model_{args.second+1}"
        f"{'_subdla' if args.sub_dla else ''}"
        f"{'_lowzcut' if args.lowzcut else ''}"
        f"{'_highzcut' if args.highzcut else ''}"
        f"{'_zmaxlyb' if args.z_max_lyb else ''}"
        f"{'_zminlyb' if args.z_min_lyb else ''}"
        f"{'_highnhi' if args.high_nhi_cut else ''}_{args.high_nhi_cut_value}"
        f"{'_minobswave' if args.min_obs_wavelength_cut else ''}_{args.min_obs_wavelength}"
        f"_bins_{args.bins_per_z}"
        f"_lnhi_{args.lnhi_min}-{args.lnhi_max}"
        f"_lnhi_dndx_{args.lnhi_min_dndx}-{args.lnhi_max_dndx}"
    )

    os.makedirs(subdir, exist_ok=True)

    # Initialize the DLA Catalogue
    dla_catalog = DLACatalogue(
        processed_file=args.processed_file,
        sample_file=args.sample_file,
        catalog_file=args.catalog_file,
        snr=args.snr,
        lowzcut=args.lowzcut,
        highzcut=args.highzcut,
        second=args.second,
        sub_dla=args.sub_dla,
        occams_razor=args.occams_razor,
        z_dla_minimum=0.1,
        z_max_lyb=args.z_max_lyb,
        z_min_lyb=args.z_min_lyb,
        min_obs_wavelength_cut=args.min_obs_wavelength_cut,
        min_obs_wavelength=args.min_obs_wavelength,
        high_nhi_cut=args.high_nhi_cut,
        bins_per_z=args.bins_per_z,
    )

    # Generate DLA statistics plots
    do_dla_statistics_plots(
        dla_catalog,
        subdir=subdir,
        z_dla_max=args.z_dla_max,
        high_z_qso=args.high_z_qso,
        low_z_qso=args.low_z_qso,
        lnhi_nbins=args.lnhi_nbins,
        lnhi_min=args.lnhi_min,
        lnhi_max=args.lnhi_max,
        lnhi_min_dndx=args.lnhi_min_dndx,
        lnhi_max_dndx=args.lnhi_max_dndx,
    )

    print(f"DLA statistics plots saved to subdirectory: {subdir}")

if __name__ == "__main__":
    main()