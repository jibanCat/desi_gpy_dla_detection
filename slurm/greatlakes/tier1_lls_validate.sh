#!/bin/bash
#SBATCH -A cavestru0
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -J tier1_lls
#SBATCH -o /nfs/turbo/lsa-cavestru/mfho/DESI/lls_relearn/tier1_%j.log
#SBATCH -e /nfs/turbo/lsa-cavestru/mfho/DESI/lls_relearn/tier1_%j.log
# Tier-1 LLS break-aware validation: line-only (production) vs break-aware finder
# on the mirror mock (in-window-break LLS + clean). Runs on a compute node so the
# 50k-sample marginalization has memory + isn't reaped with a login session.
set -eo pipefail
cd /home/mfho/desi_gpy_dla_detection
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate gpdla
export LD_LIBRARY_PATH="$HOME/.local/usr/local/lib64:$LD_LIBRARY_PATH"   # C Voigt (libcerf)
export N_LLS="${N_LLS:-8}" N_CLEAN="${N_CLEAN:-8}" PYTHONUNBUFFERED=1
export OUT_DIR=/nfs/turbo/lsa-cavestru/mfho/DESI/lls_relearn/tier1_out
mkdir -p "$OUT_DIR"
echo "=== Tier-1 validation start: N_LLS=$N_LLS N_CLEAN=$N_CLEAN ==="
python -u /nfs/turbo/lsa-cavestru/mfho/DESI/lls_relearn/tier1_validate.py
echo "=== done ==="
