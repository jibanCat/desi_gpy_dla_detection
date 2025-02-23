#!/bin/bash

# Define the maximum range for start and end indices
MAX_START_INDEX=1150
STEP=64

# Loop over the start indices and calculate corresponding end indices
for (( START_INDEX=0; START_INDEX<=MAX_START_INDEX; START_INDEX+=STEP )); do
    END_INDEX=$((START_INDEX + 62))

    # Print the command to be executed for reference
    echo "sbatch --export=ALL,QSOCAT=\"/global/cfs/projectdirs/desi/mocks/lya_forest/develop/london/qq_desi_y3/v5.9.5/mock-0/jura-124/zcat.fits\",\
MOCKDIR=\"/global/cfs/projectdirs/desi/mocks/lya_forest/develop/london/qq_desi_y3/v5.9.5/mock-0/jura-124/\",\
OUTDIR=\"/pscratch/sd/j/jibancat/desi-mock-gpdla-20250223-mock-learned/\",\
LEARNED_FILE=\"/pscratch/sd/j/jibancat/desi_gpy_dla_detection/learnlogs_london/model_epoch_86.h5\",\
PREV_TAU_0=0.00246,\
PREV_BETA=3.62,\
DLAMBDA=0.15,\
K=30,\
MAX_DLAS=3,\
PLOT_FIGURES=0,\
BATCH_SIZE=1250,\
MAX_WORKERS=8,\
START_INDEX=$START_INDEX,\
END_INDEX=$END_INDEX,\
STEP=2 slurm/submit_desi_mock.sh"

    # Submit the job using sbatch
    sbatch --export=ALL,QSOCAT="/global/cfs/projectdirs/desi/mocks/lya_forest/develop/london/qq_desi_y3/v5.9.5/mock-0/jura-124/zcat.fits",\
MOCKDIR="/global/cfs/projectdirs/desi/mocks/lya_forest/develop/london/qq_desi_y3/v5.9.5/mock-0/jura-124/",\
OUTDIR="/pscratch/sd/j/jibancat/desi-mock-gpdla-20250223-mock-learned/",\
LEARNED_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/learnlogs_london/model_epoch_86.h5",\
PREV_TAU_0=0.00246,\
PREV_BETA=3.62,\
DLAMBDA=0.15,\
K=30,\
MAX_DLAS=3,\
PLOT_FIGURES=0,\
BATCH_SIZE=1250,\
MAX_WORKERS=8,\
START_INDEX=$START_INDEX,\
END_INDEX=$END_INDEX,\
STEP=2 slurm/submit_desi_mock.sh

    # Sleep for 1 minute before the next sbatch
    sleep 1m

done