#!/bin/bash
# Saclay mock-1, LLS run — NHI range 17.2–22.0
# Model: epoch_920, same settings as London mock LLS runs (lls_runs/)
# Absorber mode: SINGLE_ABSORBER_MODEL=1 (single-absorber LLS/subDLA/DLA)
# DLA samples: pw_samples_a3_172_220 (NHI floor 17.2)

# Define the maximum range for start and end indices
MAX_START_INDEX=1127
STEP=64

# Loop over the start indices and calculate corresponding end indices
for (( START_INDEX=0; START_INDEX<=MAX_START_INDEX; START_INDEX+=STEP )); do
    END_INDEX=$((START_INDEX + 62))

    # Print the command to be executed for reference
    echo "sbatch --export=ALL,QSOCAT=\"/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/saclay/qq_desi_y3/v4.7.5/mock-1/jura-124/zcat.fits\",\
MOCKDIR=\"/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/saclay/qq_desi_y3/v4.7.5/mock-1/jura-124/\",\
OUTDIR=\"/pscratch/sd/j/jibancat/desi-mock-saclay-1-gpdla-20260415-y3-learned-epoch920-lls_run-nhi172/\",\
LEARNED_FILE=\"/pscratch/sd/j/jibancat/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5\",\
DLA_SAMPLES_FILE=\"/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat\",\
SUB_DLA_SAMPLES_FILE=\"/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/subdla_samples_a03_191_200_100000.mat\",\
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
START_INDEX=$START_INDEX,\
END_INDEX=$END_INDEX,\
STEP=2 slurm/submit_desi_mock.sh"

    # Submit the job using sbatch
    sbatch --export=ALL,QSOCAT="/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/saclay/qq_desi_y3/v4.7.5/mock-1/jura-124/zcat.fits",\
MOCKDIR="/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/saclay/qq_desi_y3/v4.7.5/mock-1/jura-124/",\
OUTDIR="/pscratch/sd/j/jibancat/desi-mock-saclay-1-gpdla-20260415-y3-learned-epoch920-lls_run-nhi172/",\
LEARNED_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5",\
DLA_SAMPLES_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_220_50000.mat",\
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
START_INDEX=$START_INDEX,\
END_INDEX=$END_INDEX,\
STEP=2 slurm/submit_desi_mock.sh

    # Sleep for 1 minute before the next sbatch
    sleep 1m

done
