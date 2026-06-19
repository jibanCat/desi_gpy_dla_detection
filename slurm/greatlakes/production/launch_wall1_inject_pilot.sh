#!/usr/bin/env bash
# launch_wall1_inject_pilot.sh — WALL-1 FULL-INJECTION pilot (mandatory gate).
#
# notes/2026-06-17_wall1_full_injection_design.md §5 step 2: inject ONE arm
# (Δα=+0.5) into ~200 loa-0 sightlines across ~4 healpix, re-run the UNMODIFIED
# production GP (wall1_inject_gl_v1.env -> loa0_fp_gl_v1.env -> 2lpt0_gl_v1.env),
# and confirm the round-trip works end-to-end (injected truth -> injected spectra
# -> GP detections -> HBI recovery) + re-measure the per-spec CPU cost.
#
# This launcher:
#   1. generates the pilot arm (cheap, on-node) IF the arm tree is absent, via
#      injection/gen_wall1_inject.py (--dalpha 0.5 --n_inj 200 --n_healpix 4);
#   2. derives the healpix file count from the arm tree;
#   3. submits ONE production-config sbatch covering all the arm's healpix.
#
# Cost (design §6, loa-124 production ~167 CPU-s/spec): ~200 spec × 167 s ≈ 9 CPU-h,
# wall ~6 h on one N=2×W=16 sbatch (under the 12 h -t). The pilot RE-MEASURES this.
#
# Usage:
#   bash slurm/greatlakes/production/launch_wall1_inject_pilot.sh            # gen+submit
#   bash slurm/greatlakes/production/launch_wall1_inject_pilot.sh --dry-run  # preview
#   WALL1_ARM=<tree> ... (override the arm root)
#
# DO NOT submit during cluster maintenance / without the PI go-ahead. The full
# 2-arm × 4000-spec run is gated on this pilot (launch_wall1_inject_full.sh).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DRY_RUN=0
PASSTHRU=()
for a in "$@"; do
    case "$a" in
        --dry-run) DRY_RUN=1; PASSTHRU+=("$a") ;;
        *) PASSTHRU+=("$a") ;;
    esac
done

# Pilot arm root (Δα=+0.5). Override via the WALL1_ARM env var.
GL_SCRATCH="/scratch/cavestru_root/cavestru0/mfho"
WALL1_ARM="${WALL1_ARM:-${GL_SCRATCH}/wall1_inject/pilot_dalpha+0.5}"
DALPHA="${DALPHA:-0.5}"
N_INJ="${N_INJ:-200}"
N_HEALPIX="${N_HEALPIX:-4}"

echo "[wall1-pilot] arm   = $WALL1_ARM"
echo "[wall1-pilot] dalpha=$DALPHA  n_inj=$N_INJ  n_healpix=$N_HEALPIX"

# 1) Generate the pilot arm if absent (cheap; needs the gpdla env for desispec/voigt).
ARM_TREE="${WALL1_ARM}/spectra-16"
if [ ! -d "$ARM_TREE" ]; then
    echo "[wall1-pilot] generating pilot arm (on-node) ..."
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  python injection/gen_wall1_inject.py --out $WALL1_ARM --dalpha $DALPHA --n_inj $N_INJ --n_healpix $N_HEALPIX"
    else
        GL_CONDA_SETUP="${GL_CONDA_SETUP:-/sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh}"
        # shellcheck disable=SC1090
        source "$GL_CONDA_SETUP"; conda activate "${GL_CONDA_ENV:-gpdla}"
        export LD_LIBRARY_PATH="${GL_LIBCERF_PATH:-$HOME/.local/usr/local/lib64}:${LD_LIBRARY_PATH:-}"
        export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
        ( cd "$REPO_ROOT" && python injection/gen_wall1_inject.py \
            --out "$WALL1_ARM" --dalpha "$DALPHA" --n_inj "$N_INJ" --n_healpix "$N_HEALPIX" )
    fi
else
    echo "[wall1-pilot] arm tree already present — reusing $ARM_TREE"
fi

# 2) Derive the healpix file count (positional level2 index space).
if [ "$DRY_RUN" -eq 1 ] && [ ! -d "$ARM_TREE" ]; then
    N_FILES=4   # preview only
else
    N_FILES=$(find "$ARM_TREE" -name 'spectra-16-*.fits' 2>/dev/null | wc -l)
fi
[ "${N_FILES:-0}" -ge 1 ] 2>/dev/null || { echo "[wall1-pilot] ERROR: no spectra-16 files under $ARM_TREE" >&2; exit 1; }
echo "[wall1-pilot] arm has $N_FILES healpix files -> level2 window [0,$N_FILES)"

# 3) Submit ONE production-config sbatch covering all the arm's healpix.
#    --start 0 --end 0 => exactly one sbatch; --window N_FILES => END_INDEX=N_FILES
#    (the inner STEP=2 loop tiles it). WALL1_ARM is exported so the env resolves
#    MOCKDIR/QSOCAT/OUTDIR/OUTER_* off the arm tree.
export WALL1_ARM
exec bash "${SCRIPT_DIR}/launch_gl.sh" wall1_inject_gl_v1.env \
    --start 0 --end 0 --window "$N_FILES" --no-sleep "${PASSTHRU[@]}"
