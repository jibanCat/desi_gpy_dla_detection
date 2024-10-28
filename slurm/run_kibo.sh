#!/bin/bash

# Define the maximum range for the healpix start and end indices
MAX_HPX_INDEX=16519
STEP=320

# Loop over the healpix start indices and calculate corresponding end indices
for (( HPX_START_INDEX=0; HPX_START_INDEX<MAX_HPX_INDEX; HPX_START_INDEX+=STEP )); do
    HPX_END_INDEX=$((HPX_START_INDEX + 280))

    # Print the command to be executed for reference
    echo "sbatch --export=ALL,QSOCAT=\"/global/cfs/cdirs/desi/users/martini/bal-catalogs/kibo/QSO_cat_kibo_main_dark_healpix_v3-altbal.fits\",\
OUTDIR=\"/pscratch/sd/j/jibancat/desi-kibo-gpdla/\",\
MAX_DLAS=3,\
PLOT_FIGURES=0,\
BATCH_SIZE=313,\
MAX_WORKERS=32,\
HPX_START_INDEX=$HPX_START_INDEX,\
HPX_END_INDEX=$HPX_END_INDEX slurm/submit_desi_kibo.sh"

    # Submit the job using sbatch
    sbatch --export=ALL,QSOCAT=\"/global/cfs/cdirs/desi/users/martini/bal-catalogs/kibo/QSO_cat_kibo_main_dark_healpix_v3-altbal.fits\",\
OUTDIR=\"/pscratch/sd/j/jibancat/desi-kibo-gpdla/\",\
MAX_DLAS=3,\
PLOT_FIGURES=0,\
BATCH_SIZE=313,\
MAX_WORKERS=32,\
HPX_START_INDEX=$HPX_START_INDEX,\
HPX_END_INDEX=$HPX_END_INDEX slurm/submit_desi_kibo.sh

    # Sleep for 1 minute before the next sbatch
    sleep 1m

done