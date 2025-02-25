#!/bin/bash
#SBATCH --job-name=combine_processed_h5     # Job name
#SBATCH --output=logs/combine_%j.out       # Standard output log (%j expands to job ID)
#SBATCH --error=logs/combine_%j.err        # Standard error log (%j expands to job ID)
#SBATCH --time=03:00:00                    # Time limit
#SBATCH --nodes=1                          # Use one node
#SBATCH -C cpu                             # Use CPU node
#SBATCH -q regular                         # Use regular queue
#SBATCH -A desi                            # Account name

# Load the required environment
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# Ensure the logs directory exists
mkdir -p logs

# Allow setting variables from command line or use defaults
CATALOG=${CATALOG:-"/global/cfs/cdirs/desi/users/martini/bal-catalogs/loa/QSO_cat_loa_main_dark_healpix_v2-altbal.fits"}
PROCESSED_DIR=${PROCESSED_DIR:-"/pscratch/sd/j/jibancat/desi-loa-gpdla-20250222-desi-learned/processed"}
OUTPUT_FILE=${OUTPUT_FILE:-"/pscratch/sd/j/jibancat/desi-loa-gpdla-20250222-desi-learned/processed-main-dark.h5"}
SURVEY=${SURVEY:-"main"}
PROGRAM=${PROGRAM:-"dark"}
MOCK_FLAG=${MOCK_FLAG:-""}

# Check if mock mode is enabled
if [[ "$MOCK_FLAG" == "--mock" ]]; then
    echo "Running in mock mode: extracting files from processed directory."
else
    echo "Running in catalog mode: using catalog file to determine healpix values."
fi

# Run the combine script
python -u combine_processed_h5.py \
    --catalog "$CATALOG" \
    --processed_dir "$PROCESSED_DIR" \
    --output_file "$OUTPUT_FILE" \
    --survey "$SURVEY" \
    --program "$PROGRAM" \
    $MOCK_FLAG

exit
