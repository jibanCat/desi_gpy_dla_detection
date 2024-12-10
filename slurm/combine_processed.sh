#!/bin/bash
#SBATCH --job-name=combine_processed_h5     # Job name
#SBATCH --output=logs/combine_%j.out       # Standard output log (%j expands to job ID)
#SBATCH --error=logs/combine_%j.err        # Standard error log (%j expands to job ID)
#SBATCH --time=02:00:00                    # Time limit
#SBATCH --nodes=1                          # Use one node
#SBATCH -C cpu                             # Use CPU node
#SBATCH -q regular                         # Use debug queue
#SBATCH -A desi                            # Account name

# Load the required environment
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# Ensure the logs directory exists
mkdir -p logs

# Run the combine script
python combine_processed_h5.py

exit
