#!/bin/bash
#SBATCH --job-name=p1_tests
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/code_review_20260826/tests_%x_%j.log
# Run one Paper-1 test profile (tests/profiles/README.md) on a compute node: the hbi profile
# includes scratch-backed integration tests that exceed the login node's memory.
# Usage: PROFILE=hbi sbatch --job-name=p1_tests_hbi run_test_profile.sbatch
set +u
REPO=${REPO:-/home/mfho/wt_forward_2026_08}; cd $REPO
echo "profile=${PROFILE:-hbi} code=$(git rev-parse --short HEAD) date=$(date -Is) host=$(hostname)"
bash tools/paper1/run_tests.sh ${PROFILE:-hbi} --continue-on-collection-errors
echo "PROFILE_EXIT=$? $(date -Is)"
