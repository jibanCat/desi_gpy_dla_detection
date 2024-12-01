#!/bin/bash

# Define the maximum range for the healpix start and end indices
MAX_HPX_INDEX=16471
STEP=1664 # 32 * 52 = 1664

# Loop over the healpix start indices and calculate corresponding end indices
for (( HPX_START_INDEX=0; HPX_START_INDEX<MAX_HPX_INDEX; HPX_START_INDEX+=STEP )); do
    HPX_END_INDEX=$((HPX_START_INDEX + 1612)) # 31 * 52 = 1612

    # Print the command to be executed for reference
    echo "sbatch --export=ALL,QSOCAT=\"/global/cfs/cdirs/desi/users/martini/bal-catalogs/kibo/QSO_cat_kibo_main_dark_healpix_v3-altbal.fits\",\
OUTDIR=\"/pscratch/sd/j/jibancat/desi-kibo-gpdla-nobal-2_15-7-nozwarn/\",\
MAX_DLAS=3,\
PLOT_FIGURES=0,\
BATCH_SIZE=1250,\
MAX_WORKERS=8,\
MAX_Z_CUT=4500.0,\
MIN_Z_CUT=30000.0,\
HPX_START_INDEX=$HPX_START_INDEX,\
HPX_END_INDEX=$HPX_END_INDEX slurm/submit_desi_kibo.sh"

    # Submit the job using sbatch
    sbatch --export=ALL,QSOCAT="/global/cfs/cdirs/desi/users/martini/bal-catalogs/kibo/QSO_cat_kibo_main_dark_healpix_v3-altbal.fits",\
OUTDIR="/pscratch/sd/j/jibancat/desi-kibo-gpdla-nobal-2_15-7-nozwarn/",\
MAX_DLAS=3,\
PLOT_FIGURES=0,\
BATCH_SIZE=1250,\
MAX_WORKERS=8,\
MAX_Z_CUT=4500.0,\
MIN_Z_CUT=30000.0,\
HPX_START_INDEX=$HPX_START_INDEX,\
HPX_END_INDEX=$HPX_END_INDEX slurm/submit_desi_kibo.sh

    # Sleep for 1 minute before the next sbatch
    sleep 1m

done