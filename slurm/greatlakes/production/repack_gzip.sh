#!/bin/bash
# slurm/greatlakes/production/repack_gzip.sh
#
# Lossless gzip-repack of processed-spectra h5 files, in place, in parallel.
#
# WHY: the processed h5 is dominated by sample_log_likelihoods_dla (n,100k,4)
# f64 and base_sample_inds (n,3,100k) i32, written UNCOMPRESSED by
# process_helpers.py. sample_log_likelihoods_dla is ~93-96% NaN (FILTER
# truncation + early-stop leave most QMC samples un-computed). gzip collapses
# the long identical-NaN byte runs -> measured 15-25x smaller, fully lossless
# (proven byte-identical, equal_nan, across all 22 datasets on a sample file).
#
# Each file: skip if already gzip; else h5repack -> .repack.tmp; verify the tmp
# opens with matching keys/shapes/dtypes (repack_verify.py); atomic mv over the
# original; on ANY failure leave the original untouched and log [FAIL...]. Safe
# to resubmit (idempotent via the iscompressed skip).
#
# Submit (from repo root):
#   PROCDIR=/scratch/.../outputs/figures/processed \
#     sbatch --export=ALL,PROCDIR slurm/greatlakes/production/repack_gzip.sh
#
# Tunables (env): GZIP_LEVEL (default 4), NPAR (default 16).

#SBATCH -A cavestru0
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -J h5repack_gzip
#SBATCH -o slurm/greatlakes/production/logs/repack_%j.log
#SBATCH -e slurm/greatlakes/production/logs/repack_err_%j.log

set -uo pipefail

PROCDIR="${PROCDIR:?set PROCDIR to the processed/ dir holding the *.h5}"
GZIP_LEVEL="${GZIP_LEVEL:-4}"
NPAR="${NPAR:-16}"
REPO_ROOT="${REPO_ROOT:-/home/mfho/desi_gpy_dla_detection}"
H5REPACK="${H5REPACK:-/home/mfho/.conda/envs/emu-3.9/bin/h5repack}"
PYBIN="${PYBIN:-/home/mfho/.conda/envs/gpdla/bin/python}"
VERIFY="${VERIFY:-$REPO_ROOT/slurm/greatlakes/production/repack_verify.py}"

export H5REPACK PYBIN VERIFY GZIP_LEVEL

process_one() {
    local f="$1"
    local base; base="$(basename "$f")"
    local tmp="${f}.repack.tmp"

    if "$PYBIN" "$VERIFY" iscompressed "$f"; then
        echo "[skip] $base (already gzip)"
        return 0
    fi

    rm -f "$tmp"
    if ! "$H5REPACK" -f GZIP="$GZIP_LEVEL" "$f" "$tmp" >/dev/null 2>&1; then
        echo "[FAIL-repack] $base"; rm -f "$tmp"; return 1
    fi

    local ssz tsz
    ssz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    tsz=$(stat -c %s "$tmp" 2>/dev/null || echo 0)
    if [ "$tsz" -le 0 ]; then
        echo "[FAIL-empty] $base"; rm -f "$tmp"; return 1
    fi

    if ! "$PYBIN" "$VERIFY" verify "$f" "$tmp"; then
        echo "[FAIL-verify] $base"; rm -f "$tmp"; return 1
    fi

    mv "$tmp" "$f"
    echo "[ok] $base  $((ssz/1000000))MB -> $((tsz/1000000))MB"
    return 0
}
export -f process_one

mapfile -t FILES < <(ls "$PROCDIR"/*.h5 2>/dev/null)
echo "[repack] $(date) PROCDIR=$PROCDIR  files=${#FILES[@]}  NPAR=$NPAR  GZIP=$GZIP_LEVEL"
echo "[repack] before:"; du -sh "$PROCDIR" 2>/dev/null
echo "[repack] free:"; df -h "$PROCDIR" 2>/dev/null | tail -1

printf '%s\n' "${FILES[@]}" | xargs -P "$NPAR" -I{} bash -c 'process_one "$@"' _ {}

echo "[repack] $(date) all files processed"
echo "[repack] after:"; du -sh "$PROCDIR" 2>/dev/null
# Final summary: how many remain uncompressed (= failures), plus failure list.
nleft=0
for f in "${FILES[@]}"; do
    "$PYBIN" "$VERIFY" iscompressed "$f" || { nleft=$((nleft+1)); echo "[remaining-uncompressed] $(basename "$f")"; }
done
echo "[repack] DONE  remaining-uncompressed=$nleft / ${#FILES[@]}"
