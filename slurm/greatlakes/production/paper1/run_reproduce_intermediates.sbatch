#!/bin/bash
#SBATCH --job-name=p1_intermediates
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/code_review_20260826/intermediates_%j.log
# Pre-tag review 2026-08-26 req. 7: re-derive every deterministic frozen intermediate into a
# scratch work dir and print the reproducibility matrix; then run the two dispositioned
# Track-C cache contracts (znz tolerance gate; r_emp strict xfail) in gpdla-hbi.
set +u; set -eo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
REPO=${REPO:-/home/mfho/wt_forward_2026_08}; cd $REPO
W=${W:-/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/code_review_20260826/intermediates}
echo "code=$(git rev-parse --short HEAD) date=$(date -Is) work=$W"
PY=$HOME/.conda/envs/gpdla-hbi/bin/python
$PY tools/paper1/reproduce_intermediates.py --work $W real-pack pool zdomain config recovery compare
echo "=== cache-contract tests (gpdla-hbi)"
$PY -m pytest tests/test_znz_kernel.py tests/test_remp_rebind.py -q -rxX -p no:cacheprovider 2>&1 | tail -15 || true
echo "INTERMEDIATES_DONE $(date -Is)"
