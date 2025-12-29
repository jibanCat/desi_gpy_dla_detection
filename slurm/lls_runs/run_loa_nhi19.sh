#!/bin/bash

# Define the maximum range for the healpix start and end indices
MAX_HPX_INDEX=16519
STEP=1664 # 32 * 52 = 1664

# Loop over the healpix start indices and calculate corresponding end indices
for (( HPX_START_INDEX=0; HPX_START_INDEX<MAX_HPX_INDEX; HPX_START_INDEX+=STEP )); do
    HPX_END_INDEX=$((HPX_START_INDEX + 1612)) # 31 * 52 = 1612

    # Print the command to be executed for reference
    echo "sbatch --export=ALL,QSOCAT=\"/global/cfs/cdirs/desi/science/lya/y3/loa/catalogs/QSO_cat_loa_main_dark_healpix_v2-altbal-20241115.fits\",\
OUTDIR=\"/pscratch/sd/j/jibancat/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi190/\",\
LEARNED_FILE=\"/pscratch/sd/j/jibancat/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5\",\
PREV_TAU_0=0.00246,\
PREV_BETA=3.62,\
DLAMBDA=0.15,\
K=30,\
MAX_DLAS=1,\
NUM_FOREST_LINES=3,\
PLOT_FIGURES=0,\
BATCH_SIZE=6250,\
MAX_WORKERS=8,\
FILTER_LOW_LIKELIHOOD=0,\
SINGLE_ABSORBER_MODEL=1,\
NUM_DLA_SAMPLES=50000,\
NUM_SUBDLA_SAMPLES=100000,\
HPX_START_INDEX=$HPX_START_INDEX,\
HPX_END_INDEX=$HPX_END_INDEX slurm/submit_desi_loa.sh"

    # Submit the job using sbatch
    sbatch --export=ALL,QSOCAT="/global/cfs/cdirs/desi/science/lya/y3/loa/catalogs/QSO_cat_loa_main_dark_healpix_v2-altbal-20241115.fits",\
OUTDIR="/pscratch/sd/j/jibancat/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi190/",\
LEARNED_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5",\
DLA_SAMPLES_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_190_220_50000.mat",\
SUB_DLA_SAMPLES_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/subdla_samples_a03_191_200_100000.mat",\
PREV_TAU_0=0.00246,\
PREV_BETA=3.62,\
DLAMBDA=0.15,\
K=30,\
MAX_DLAS=1,\
NUM_FOREST_LINES=3,\
PLOT_FIGURES=0,\
BATCH_SIZE=6250,\
MAX_WORKERS=8,\
FILTER_LOW_LIKELIHOOD=0,\
SINGLE_ABSORBER_MODEL=1,\
NUM_DLA_SAMPLES=50000,\
NUM_SUBDLA_SAMPLES=100000,\
HPX_START_INDEX=$HPX_START_INDEX,\
HPX_END_INDEX=$HPX_END_INDEX slurm/submit_desi_loa.sh

    # Sleep for 1 minute before the next sbatch
    sleep 1m

done