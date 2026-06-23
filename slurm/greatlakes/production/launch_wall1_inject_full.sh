#!/usr/bin/env bash
# launch_wall1_inject_full.sh — WALL-1 FULL-INJECTION full run (2 arms × 4000 spec).
#
# notes/2026-06-17_wall1_full_injection_design.md §5 step 3. GATED ON THE PILOT
# (launch_wall1_inject_pilot.sh job 51948919) — do NOT run until the pilot has:
#   (a) confirmed per-spec cost ≈ 167 CPU-s (re-measured from the pilot log),
#   (b) confirmed injected troughs recovered (dlacat NHI ≈ injected),
#   (c) confirmed the R_emp re-bind + HBI reduce runs end-to-end
#       (CDDF_analysis/wall1_full_injection.py --arm <pilot_arm>),
#   AND a PI go.
#
# Two injected arms (the Δα=0 control REUSES the loa-124 production catalog — zero
# new compute, design §3.1):
#   +tilt  Δα = +0.5   4000 injections into loa-0  (the FAIL is clearest here)
#   −tilt  Δα = −0.5   4000 injections into loa-0  (opposite-sign pull)
#
# Cost (design §6, ~167 CPU-s/spec): 2 × 4000 = 8000 spec ≈ 371 CPU-h, wall ~6 h
# across multiple sbatch (one launch_gl.sh window-loop per arm; the arm's healpix
# count sets the window). Each arm is generated on-node first (minutes), then the
# production-config GP runs via launch_gl.sh wall1_inject_gl_v1.env.
#
# Usage:
#   bash slurm/greatlakes/production/launch_wall1_inject_full.sh            # gen+submit BOTH arms
#   bash slurm/greatlakes/production/launch_wall1_inject_full.sh --dry-run
#   N_INJ=5000 bash .../launch_wall1_inject_full.sh    # if more high-z margin wanted

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DRY_RUN=0; PASSTHRU=()
for a in "$@"; do
    case "$a" in
        --dry-run) DRY_RUN=1; PASSTHRU+=("$a") ;;
        *) PASSTHRU+=("$a") ;;
    esac
done

GL_SCRATCH="/scratch/cavestru_root/cavestru0/mfho"
N_INJ="${N_INJ:-4000}"
N_HEALPIX="${N_HEALPIX:-0}"   # 0 = all clean healpix (need the high-z fill, design §4)

for DALPHA in 0.5 -0.5; do
    ARM="${GL_SCRATCH}/wall1_inject/full_dalpha${DALPHA}"
    echo "==============================================================="
    echo "[wall1-full] arm Δα=${DALPHA}  n_inj=${N_INJ}  -> ${ARM}"
    echo "==============================================================="

    ARM_TREE="${ARM}/spectra-16"
    if [ ! -d "$ARM_TREE" ]; then
        echo "[wall1-full] generating arm (on-node) ..."
        if [ "$DRY_RUN" -eq 1 ]; then
            echo "  python injection/gen_wall1_inject.py --out $ARM --dalpha $DALPHA --n_inj $N_INJ --n_healpix $N_HEALPIX"
        else
            GL_CONDA_SETUP="${GL_CONDA_SETUP:-/sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh}"
            # shellcheck disable=SC1090
            source "$GL_CONDA_SETUP"; conda activate "${GL_CONDA_ENV:-gpdla}"
            export LD_LIBRARY_PATH="${GL_LIBCERF_PATH:-$HOME/.local/usr/local/lib64}:${LD_LIBRARY_PATH:-}"
            export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
            ( cd "$REPO_ROOT" && python injection/gen_wall1_inject.py \
                --out "$ARM" --dalpha "$DALPHA" --n_inj "$N_INJ" --n_healpix "$N_HEALPIX" )
        fi
    else
        echo "[wall1-full] arm tree present — reusing $ARM_TREE"
    fi

    if [ "$DRY_RUN" -eq 1 ] && [ ! -d "$ARM_TREE" ]; then
        N_FILES=64   # preview placeholder
    else
        N_FILES=$(find "$ARM_TREE" -name 'spectra-16-*.fits' 2>/dev/null | wc -l)
    fi
    echo "[wall1-full] arm has $N_FILES healpix files"

    # Submit the production-config GP over the arm's healpix. The launch_gl.sh window
    # loop (OUTER_STEP=12, OUTER_WINDOW=10) tiles the file index space [0, N_FILES).
    export WALL1_ARM="$ARM"
    bash "${SCRIPT_DIR}/launch_gl.sh" wall1_inject_gl_v1.env \
        --start 0 --end "$N_FILES" "${PASSTHRU[@]}"
done

echo "[wall1-full] both arms submitted. Reduce each with:"
echo "  python CDDF_analysis/wall1_full_injection.py --arm ${GL_SCRATCH}/wall1_inject/full_dalpha0.5  --label plus"
echo "  python CDDF_analysis/wall1_full_injection.py --arm ${GL_SCRATCH}/wall1_inject/full_dalpha-0.5 --label minus"
