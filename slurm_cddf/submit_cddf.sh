#!/bin/bash
#SBATCH --job-name=cddf                      # Job name
#SBATCH --output=logs/cddf_%j.out            # Standard output log (%j expands to job ID)
#SBATCH --error=logs/cddf_%j.err             # Standard error log
#SBATCH --time=01:30:00                      # Time limit
#SBATCH --nodes=1                            # Use one node
#SBATCH -C cpu                               # Use CPU node
#SBATCH -q regular                             # Use debug queue (change if needed)
#SBATCH -A desi                              # Account name

# Load the required environment
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# Ensure the logs directory exists
mkdir -p logs

# Define input files (can be overridden at submission)
PROCESSED_FILE=${PROCESSED_FILE:-"/pscratch/sd/j/jibancat/desi-loa-gpdla-20250222-desi-learned/processed-main-dark.h5"}
SAMPLE_FILE=${SAMPLE_FILE:-"data/dr12q/processed/dla_samples_a03.mat"}
CATALOG_FILE=${CATALOG_FILE:-"data/desi/processed/catalog.fits"}

# Define hyperparameters (can be changed at submission)
SNR=${SNR:--2}  # Default to -2 if unset
SECOND=${SECOND:-1}  # Default to 1 if unset
OCCAMS_RAZOR=${OCCAMS_RAZOR:-1}  # Default to 1 if unset
OUTPUT_PREFIX=${OUTPUT_PREFIX:-"/pscratch/sd/j/jibancat/dla_cddf"}
HIGH_Z_QSO=${HIGH_Z_QSO:-6}  # Default to 6 if unset
LOW_Z_QSO=${LOW_Z_QSO:-2.1}  # Default to 2.1 if unset

# Optional flags (only included if set)
EXTRA_FLAGS=""
[ "${SUB_DLA:-0}" -eq 1 ] && EXTRA_FLAGS+=" --sub_dla"
[ "${HIGH_NHI_CUT:-0}" -eq 1 ] && EXTRA_FLAGS+=" --high_nhi_cut"
[ "${LOWZCUT:-0}" -eq 1 ] && EXTRA_FLAGS+=" --lowzcut"
[ "${HIGHZCUT:-0}" -eq 1 ] && EXTRA_FLAGS+=" --highzcut"

# extra for z_min_lyb
[ "${Z_MIN_LYB:-0}" -eq 1 ] && EXTRA_FLAGS+=" --z_min_lyb"

# high_nhi_cut_value
HIGH_NHI_CUT_VALUE=${HIGH_NHI_CUT_VALUE:-22.0}

# Bins per z
BINS_PER_Z=${BINS_PER_Z:-6}

# lnhi parameters
LNHI_NBINS=${LNHI_NBINS:-30}
LNHI_MIN=${LNHI_MIN:-20.0}
LNHI_MAX=${LNHI_MAX:-23.0}
LNHI_MIN_DNDX=${LNHI_MIN_DNDX:-20.3}
LNHI_MAX_DNDX=${LNHI_MAX_DNDX:-22.5}

# DLA redshift minimums
Z_DLA_CDDF_MIN=${Z_DLA_CDDF_MIN:-1.0}
Z_DLA_DNDX_MIN=${Z_DLA_DNDX_MIN:-2.0}

# Run the Python script with parameters
python desi_cddf.py \
    --processed_file "$PROCESSED_FILE" \
    --sample_file "$SAMPLE_FILE" \
    --catalog_file "$CATALOG_FILE" \
    --snr "$SNR" \
    --second "$SECOND" \
    --output_prefix "$OUTPUT_PREFIX" \
    --occams_razor "$OCCAMS_RAZOR" \
    --high_z_qso "$HIGH_Z_QSO" \
    --low_z_qso "$LOW_Z_QSO" \
    $EXTRA_FLAGS \
    --min_obs_wavelength_cut \
    --min_obs_wavelength 3700 \
    --high_nhi_cut_value $HIGH_NHI_CUT_VALUE \
    --bins_per_z $BINS_PER_Z \
    --lnhi_nbins $LNHI_NBINS \
    --lnhi_min $LNHI_MIN \
    --lnhi_max $LNHI_MAX \
    --lnhi_min_dndx $LNHI_MIN_DNDX \
    --lnhi_max_dndx $LNHI_MAX_DNDX \
    --z_dla_cddf_min $Z_DLA_CDDF_MIN \
    --z_dla_dndx_min $Z_DLA_DNDX_MIN

# # # NHI 17.2 loa - LLS + subDLA
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=4,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/dla_cddf_20260107",CATALOG_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/loa/cddf-qsocat-altbal_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=3,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=17.2,LNHI_MAX_DNDX=20.3 slurm_cddf/submit_cddf.sh

# # ############# LLS only #############
# # # NHI 17.2 loa - LLS only
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=4,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/dla_cddf_20260107",CATALOG_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/loa/cddf-qsocat-altbal_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=3,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=17.2,LNHI_MAX_DNDX=19.0 slurm_cddf/submit_cddf.sh
# # ## more bins
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=4,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/dla_cddf_20260107",CATALOG_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/loa/cddf-qsocat-altbal_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=17.2,LNHI_MAX_DNDX=19.0 slurm_cddf/submit_cddf.sh
# # ## More SNR > 6
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=6,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/dla_cddf_20260107",CATALOG_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/loa/cddf-qsocat-altbal_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=17.2,LNHI_MAX_DNDX=19.0 slurm_cddf/submit_cddf.sh
# # ## More SNR > 6 and lower z cut
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=6,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/dla_cddf_20260107",CATALOG_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/loa/cddf-qsocat-altbal_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=1,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=17.2,LNHI_MAX_DNDX=19.0 slurm_cddf/submit_cddf.sh
# # ## More SNR > 8
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=8,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/dla_cddf_20260107",CATALOG_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/loa/cddf-qsocat-altbal_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=17.2,LNHI_MAX_DNDX=19.0 slurm_cddf/submit_cddf.sh
# # ## More SNR > 8 and lower z cut
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=8,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/dla_cddf_20260107",CATALOG_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/loa/cddf-qsocat-altbal_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=1,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=17.2,LNHI_MAX_DNDX=19.0 slurm_cddf/submit_cddf.sh

# # ############ subDLA only #############
# ## More SNR > 6 and subDLA only
# sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=6,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/dla_cddf_20260107",CATALOG_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/loa/cddf-qsocat-altbal_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=19.0,LNHI_MAX_DNDX=20.3 slurm_cddf/submit_cddf.sh
# ## More SNR > 8 and subDLA only
# sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=8,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/dla_cddf_20260107",CATALOG_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/loa/cddf-qsocat-altbal_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=19.0,LNHI_MAX_DNDX=20.3 slurm_cddf/submit_cddf.sh
# # ## More SNR > 8 and subDLA only and lower z cut
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=8,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/dla_cddf_20260107",CATALOG_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/loa/cddf-qsocat-altbal_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=1,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=19.0,LNHI_MAX_DNDX=20.3 slurm_cddf/submit_cddf.sh
# # ## More SNR > 6 and subDLA only and lower z cut
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=6,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/dla_cddf_20260107",CATALOG_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/loa/cddf-qsocat-altbal_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=1,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=19.0,LNHI_MAX_DNDX=20.3 slurm_cddf/submit_cddf.sh

# # ######## LLS only #############
# # #### Cut z > 3 and SNR > 6
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=6,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/dla_cddf_20260107",CATALOG_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/loa/cddf-qsocat-altbal_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=3,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=17.2,LNHI_MAX_DNDX=19.0 slurm_cddf/submit_cddf.sh
# #### Cut z > 2.15 and SNR > 6
# sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=6,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/dla_cddf_20260107",CATALOG_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/loa/cddf-qsocat-altbal_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=17.2,LNHI_MAX_DNDX=19.0 slurm_cddf/submit_cddf.sh
# #### Fumagalli cut: Cut z > 3.5 and SNR > 6
# sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=6,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/cddf_all/dla_cddf_20260107",CATALOG_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/loa/cddf-qsocat-altbal_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=3.5,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=17.2,LNHI_MAX_DNDX=19.0,Z_DLA_CDDF_MIN=3.0,Z_DLA_DNDX_MIN=3.0 slurm_cddf/submit_cddf.sh
# #### Fumagalli cut: Cut z > 3.5 and SNR > 8
# sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=8,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/cddf_all/dla_cddf_20260107",CATALOG_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/loa/cddf-qsocat-altbal_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=3.5,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=17.2,LNHI_MAX_DNDX=19.0,Z_DLA_CDDF_MIN=3.0,Z_DLA_DNDX_MIN=3.0 slurm_cddf/submit_cddf.sh


# # # NHI 17.2 mock - LLS + subDLA
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-mock-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=4,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/cddf_all/dla_london_cddf_nhi172_20260107",CATALOG_FILE="data/london/cddf-qsocat_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=17.2,LNHI_MAX_DNDX=20.3 slurm_cddf/submit_cddf.sh
# # # LLS only
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-mock-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=4,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/cddf_all/dla_london_cddf_nhi172_20260107",CATALOG_FILE="data/london/cddf-qsocat_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=17.2,LNHI_MAX_DNDX=19.0 slurm_cddf/submit_cddf.sh
# # # subDLA only
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-mock-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=4,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/cddf_all/dla_london_cddf_nhi172_20260107",CATALOG_FILE="data/london/cddf-qsocat_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=19.0,LNHI_MAX_DNDX=20.3 slurm_cddf/submit_cddf.sh
# # ## More SNR > 6
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-mock-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=6,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/cddf_all/dla_london_cddf_nhi172_20260107",CATALOG_FILE="data/london/cddf-qsocat_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=17.2,LNHI_MAX_DNDX=19.0 slurm_cddf/submit_cddf.sh
# # ## More SNR >6 and subDLA only
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-mock-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=6,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/cddf_all/dla_london_cddf_nhi172_20260107",CATALOG_FILE="data/london/cddf-qsocat_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=19.0,LNHI_MAX_DNDX=20.3 slurm_cddf/submit_cddf.sh
# # ## More SNR > 8 and subDLA only
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-mock-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=8,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/cddf_all/dla_london_cddf_nhi172_20260107",CATALOG_FILE="data/london/cddf-qsocat_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=19.0,LNHI_MAX_DNDX=20.3 slurm_cddf/submit_cddf.sh
# # ## More SNR > 6 and subDLA only and lower z cut
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-mock-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=6,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/cddf_all/dla_london_cddf_nhi172_20260107",CATALOG_FILE="data/london/cddf-qsocat_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=1,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=19.0,LNHI_MAX_DNDX=20.3 slurm_cddf/submit_cddf.sh
# # ## More SNR > 8 and subDLA only and lower z cut
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-mock-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=8,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/cddf_all/dla_london_cddf_nhi172_20260107",CATALOG_FILE="data/london/cddf-qsocat_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=1,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=19.0,LNHI_MAX_DNDX=20.3 slurm_cddf/submit_cddf.sh
# # ## More SNR > 6 and LLS only and lower z cut
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-mock-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=6,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/cddf_all/dla_london_cddf_nhi172_20260107",CATALOG_FILE="data/london/cddf-qsocat_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=1,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=17.2,LNHI_MAX_DNDX=19 slurm_cddf/submit_cddf.sh
# # ## More SNR > 8 and LLS only and lower z cut
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-mock-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=8,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/cddf_all/dla_london_cddf_nhi172_20260107",CATALOG_FILE="data/london/cddf-qsocat_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=1,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=17.2,LNHI_MAX_DNDX=19,Z_DLA_CDDF_MIN=3.0,Z_DLA_DNDX_MIN=3.0 slurm_cddf/submit_cddf.sh
# #### Fumagalli cut: Cut z > 3.5 and SNR > 6
# sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-mock-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=6,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/cddf_all/dla_london_cddf_nhi172_20260107",CATALOG_FILE="data/london/cddf-qsocat_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=3.5,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=17.2,LNHI_MAX_DNDX=19.0,Z_DLA_CDDF_MIN=3.0,Z_DLA_DNDX_MIN=3.0 slurm_cddf/submit_cddf.sh
# #### Fumagalli cut: Cut z > 3.5 and SNR > 8
# sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-mock-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/processed-main-dark.h5",SNR=8,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/cddf_all/dla_london_cddf_nhi172_20260107",CATALOG_FILE="data/london/cddf-qsocat_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",LOW_Z_QSO=3.5,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=17.2,LNHI_MAX=22,LNHI_MIN_DNDX=17.2,LNHI_MAX_DNDX=19.0,Z_DLA_CDDF_MIN=3.0,Z_DLA_DNDX_MIN=3.0 slurm_cddf/submit_cddf.sh





# # # # NHI 19 loa
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi190/processed-main-dark.h5",SNR=4,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/dla_cddf_nhi190_20260107",CATALOG_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/loa/cddf-qsocat-altbal_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_190_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=3,LNHI_NBINS=30,LNHI_MIN=19.0,LNHI_MAX=22,LNHI_MIN_DNDX=19.0,LNHI_MAX_DNDX=20.3 slurm_cddf/submit_cddf.sh
# # ## More bins
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi190/processed-main-dark.h5",SNR=4,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/dla_cddf_nhi190_20260107",CATALOG_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/loa/cddf-qsocat-altbal_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_190_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=19.0,LNHI_MAX=22,LNHI_MIN_DNDX=19.0,LNHI_MAX_DNDX=20.3 slurm_cddf/submit_cddf.sh
# # ## More SNR > 6
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi190/processed-main-dark.h5",SNR=6,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/cddf_all/dla_cddf_nhi190_20260107",CATALOG_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/loa/cddf-qsocat-altbal_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_190_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=19.0,LNHI_MAX=22,LNHI_MIN_DNDX=19.0,LNHI_MAX_DNDX=20.3 slurm_cddf/submit_cddf.sh
# # ## More SNR > 8
# # sbatch --export=ALL,PROCESSED_FILE="/pscratch/sd/j/jibancat/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi190/processed-main-dark.h5",SNR=8,SECOND=0,OUTPUT_PREFIX="/pscratch/sd/j/jibancat/cddf_all/dla_cddf_nhi190_20260107",CATALOG_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/loa/cddf-qsocat-altbal_zgt2.15_zlt6.fits",SAMPLE_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_190_220_50000.mat",LOW_Z_QSO=2.15,HIGH_Z_QSO=5,HIGH_NHI_CUT=0,SUB_DLA=0,LOWZCUT=0,HIGHZCUT=0,HIGH_NHI_CUT_VALUE=22,Z_MIN_LYB=1,BINS_PER_Z=5,LNHI_NBINS=30,LNHI_MIN=19.0,LNHI_MAX=22,LNHI_MIN_DNDX=19.0,LNHI_MAX_DNDX=20.3 slurm_cddf/submit_cddf.sh


