#!/bin/bash
#SBATCH -A yueyingn0
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -t 2:00:00
#SBATCH -J gp_inject_pilot
#SBATCH -o /scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/gp_inject_%j.log
#SBATCH -e /scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/gp_inject_%j.log

# Run the UNMODIFIED GP on an injectable tree, matching the 2LPT-0 FILTER-off run
# being corrected (single_absorber_model=1, max_dlas=1, filter_low_likelihood=0,
# num_forest_lines=31, 100k samples, MAX_LAMBDA=1250). --qsocat is restricted to the
# injected/control TARGETIDs (keeps it ~7 CPU-h, not ~314). Pass CAMPAIGN (tree root) +
# N_HEALPIX via sbatch --export.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /home/mfho/desi_gpy_dla_detection
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate gpdla
export LD_LIBRARY_PATH="$HOME/.local/usr/local/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CAMPAIGN="${CAMPAIGN:?set CAMPAIGN=<injectable tree root>}"
N_HEALPIX="${N_HEALPIX:-6}"
OUTDIR="${OUTDIR:-$CAMPAIGN/gp_out}"
DR=/scratch/cavestru_root/cavestru0/mfho/DESI/desi_gpy_dla_detection/data
mkdir -p "$OUTDIR"

echo "[gp-inject] host=$(hostname) campaign=$CAMPAIGN n_healpix=$N_HEALPIX start=$(date)"
python desi-DLAGP.py \
    --qsocat "$CAMPAIGN/pilot_qsocat.fits" \
    --release v5.9.5 --program dark --survey main \
    --mocks --mockdir "$CAMPAIGN" \
    --outdir "$OUTDIR" \
    --learned_file /scratch/cavestru_root/cavestru0/mfho/phase2_desi/2lpt_loa124_nohcd_nobal_wide_m/phase2_result.h5 \
    --catalog_name "$DR/dr12q/processed/catalog.mat" \
    --los_catalog "$DR/dla_catalogs/dr9q_concordance/processed/los_catalog" \
    --dla_catalog "$DR/dla_catalogs/dr9q_concordance/processed/dla_catalog" \
    --dla_samples_file "$DR/dr12q/processed/pw_samples_a3_172_225_100000.mat" \
    --sub_dla_samples_file "$DR/dr12q/processed/subdla_samples_a03_191_200_100000.mat" \
    --min_z_separation 3000.0 --prev_tau_0 0.00246 --prev_beta 3.62 \
    --max_dlas 1 --single_absorber_model 1 --filter_low_likelihood 0 \
    --num_dla_samples 100000 --num_subdla_samples 100000 \
    --num_forest_lines 31 --max_lambda 1250 --plot_figures 0 \
    --k 30 \
    --level2_start 0 --level2_end "$N_HEALPIX"
echo "[gp-inject] DONE rc=$?  end=$(date)"
