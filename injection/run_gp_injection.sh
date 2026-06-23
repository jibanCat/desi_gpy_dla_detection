#!/bin/bash
#SBATCH -A yueyingn0
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -J gp_inject
#SBATCH -o /scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/gp_inject_%A_%a.log
#SBATCH -e /scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/gp_inject_%A_%a.log

# Run the UNMODIFIED GP on an injectable tree, BYTE-MATCHING the 2LPT-0 FILTER-off
# CDDF run being corrected (2lpt0_gl_v1_filteroff_maxdla1.env): single_absorber_model=1,
# max_dlas=1, filter_low_likelihood=0, num_forest_lines=31, 100k samples, MAX_LAMBDA=1250,
# DLAMBDA=0.15, MIN_LAMBDA=911.75, K=30, τ-EB ON/null objective, early_stop=baseline.
# pair_prior_mode/dla_bias/max_noise_variance/num_lines and the loading/normalization
# bands are all left at the CLI defaults, which equal the production env values
# (pair_prior unset→off, dla_bias 2.0, max_noise 9, num_lines 3, 910/1550/1425/1475).
# --qsocat is restricted to the injected/control TARGETIDs (keeps it cheap). Pass
# CAMPAIGN (tree root); N_HEALPIX is derived from the tree (override via --export).
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /home/mfho/desi_gpy_dla_detection
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate gpdla
export LD_LIBRARY_PATH="$HOME/.local/usr/local/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CAMPAIGN="${CAMPAIGN:?set CAMPAIGN=<injectable tree root>}"
# Derive the healpix count from the injectable tree itself — never silently
# default to a fixed number (which would undercount a larger campaign). The GP's
# --level2_end is exclusive, so this count == number of spectra-16-*.fits files.
if [ -z "${N_HEALPIX:-}" ]; then
    # 2>/dev/null so a missing spectra-16/ dir doesn't abort the command-substitution
    # under `set -e` BEFORE the friendly guard below can fire.
    N_HEALPIX=$(find "$CAMPAIGN/spectra-16" -name 'spectra-16-*.fits' 2>/dev/null | wc -l)
fi
[ "${N_HEALPIX:-0}" -ge 1 ] 2>/dev/null || { echo "[gp-inject] ERROR: no spectra-16-*.fits under $CAMPAIGN/spectra-16" >&2; exit 1; }
OUTDIR="${OUTDIR:-$CAMPAIGN/gp_out}"
QSOCAT="${QSOCAT:-$CAMPAIGN/pilot_qsocat.fits}"
DR=/scratch/cavestru_root/cavestru0/mfho/DESI/desi_gpy_dla_detection/data
mkdir -p "$OUTDIR"

# Healpix range for THIS run. In a SLURM job array, each task processes a CHUNK of
# healpix [task*CHUNK, (task+1)*CHUNK) → one dlacat-*-<start>-<end>.fits per chunk
# (measure_recovery reads them all). Outside an array, do the whole tree (or honor
# an explicit LEVEL2_START/END override).
CHUNK="${CHUNK:-$N_HEALPIX}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    LEVEL2_START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
    LEVEL2_END=$(( LEVEL2_START + CHUNK ))
    [ "$LEVEL2_END" -gt "$N_HEALPIX" ] && LEVEL2_END=$N_HEALPIX
    if [ "$LEVEL2_START" -ge "$N_HEALPIX" ]; then
        echo "[gp-inject] array task $SLURM_ARRAY_TASK_ID: start $LEVEL2_START >= N_HEALPIX $N_HEALPIX — nothing to do"; exit 0
    fi
else
    LEVEL2_START="${LEVEL2_START:-0}"
    LEVEL2_END="${LEVEL2_END:-$N_HEALPIX}"
fi

echo "[gp-inject] host=$(hostname) campaign=$CAMPAIGN n_healpix=$N_HEALPIX level2=[$LEVEL2_START,$LEVEL2_END) start=$(date)"
python desi-DLAGP.py \
    --qsocat "$QSOCAT" \
    --release v2.8.5 --program dark --survey main \
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
    --enable_tau_eb 1 --tau_eb_objective null \
    --dlambda 0.15 --early_stop_mode baseline --min_lambda 911.75 \
    --k 30 --max_workers 16 --batch_size 1250 \
    --figure_dir "$OUTDIR/figures" \
    --level2_start "$LEVEL2_START" --level2_end "$LEVEL2_END"
echo "[gp-inject] DONE rc=$?  end=$(date)"
