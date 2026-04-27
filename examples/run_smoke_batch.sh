#!/bin/bash
# Run the smoke test on every target listed in out/smoke/targets.tsv.
#
# Usage:
#   bash examples/run_smoke_batch.sh [PRESET] [FILTER] [N_DLA] [N_SUBDLA]
#
#   PRESET    eboss | y3 | london     (default eboss)
#   FILTER    0 | 1                   (default 1)
#   N_DLA     # DLA samples           (default 10000)
#   N_SUBDLA  # sub-DLA samples       (default 10000)
#
# Outputs:
#   out/smoke/batch/<preset>_filter<F>_n<N_DLA>/<mock>_<tid>.{h5,pkl,log}
#   figures/smoke_v2/<mock>_<tid>/spec-*.png   (canonical project plot via --plot)
#
# Note: per-condition output dirs let multiple sweeps coexist.
set -euo pipefail
# Defaults below match GreatLakes; override via env vars on NERSC, e.g.:
#   REPO_DIR=/pscratch/sd/j/jibancat/desi_gpy_dla_detection \
#   DATA_ROOT=/pscratch/sd/j/jibancat/desi_gpy_dla_detection \
#   PYTHON=python \
#   bash examples/run_smoke_batch.sh y3 1 10000 10000 out/smoke/targets.tsv
REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
PY="${PYTHON:-${PY:-python}}"
DATA="${DATA_ROOT:-${DATA:-/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection}}"

PRESET=${1:-eboss}
FILTER=${2:-1}
N_DLA=${3:-10000}
N_SUBDLA=${4:-10000}
TARGETS=${5:-out/smoke/targets.tsv}
DLA_MAT_OVERRIDE=${6:-}

# Pick the right DLA-sample .mat for the requested N (override-able for prior-edge tests)
if [[ -n "$DLA_MAT_OVERRIDE" ]]; then
  DLA_MAT="$DLA_MAT_OVERRIDE"
else
  case "$N_DLA" in
    10000)  DLA_MAT="$DATA/data/dr12q/processed/dla_samples_a03.mat" ;;
    100000) DLA_MAT="$DATA/data/dr12q/processed/dla_samples_a03_100000.mat" ;;
    *) DLA_MAT="$DATA/data/dr12q/processed/dla_samples_a03_${N_DLA}.mat" ;;
  esac
fi
case "$N_SUBDLA" in
  10000)  SUBDLA_MAT="$DATA/data/dr12q/processed/subdla_samples.mat" ;;
  100000) SUBDLA_MAT="$DATA/data/dr12q/processed/subdla_samples_a03_191_200_100000.mat" ;;
  *) SUBDLA_MAT="$DATA/data/dr12q/processed/subdla_samples_${N_SUBDLA}.mat" ;;
esac

[[ -f "$DLA_MAT"    ]] || { echo "missing $DLA_MAT"; exit 2; }
[[ -f "$SUBDLA_MAT" ]] || { echo "missing $SUBDLA_MAT"; exit 2; }

SUFFIX=""
if [[ -n "$DLA_MAT_OVERRIDE" ]]; then
  base=$(basename "$DLA_MAT_OVERRIDE" .mat)
  SUFFIX="_${base}"
fi
TARGSUFFIX=""
if [[ "$TARGETS" != "out/smoke/targets.tsv" ]]; then
  TARGSUFFIX="_$(basename "$TARGETS" .tsv)"
fi
OUT=out/smoke/batch/${PRESET}_filter${FILTER}_n${N_DLA}${SUFFIX}${TARGSUFFIX}
mkdir -p "$OUT" figures/smoke_v2

# Read the targets TSV. The 100-target file has extra columns (all_truth_z,
# all_truth_nhi); drop them with `cut -f1-9` so this loop's positional
# IFS-read works regardless of which file we're given.
while IFS=$'\t' read -r mock tid z_qso z_dla_t nhi_t snr hpx spec zcat; do
  [[ "$mock" == "mock" ]] && continue
  echo "[batch] $PRESET filt=$FILTER N=$N_DLA  $mock  TID=$tid  truth z=$z_dla_t NHI=$nhi_t SNR=$snr"
  fig=figures/smoke_v2/${mock}_${tid}
  mkdir -p "$fig"
  log=$OUT/${mock}_${tid}.log
  $PY examples/smoke_one_spectrum.py \
    --specfile "$spec" --zcat "$zcat" --target-id "$tid" \
    --preset "$PRESET" --data-root "$DATA" \
    --dla-samples-file     "$DLA_MAT" \
    --sub-dla-samples-file "$SUBDLA_MAT" \
    --single-absorber-model 0 --max-dlas 4 --filter-low-likelihood "$FILTER" \
    --num-dla-samples "$N_DLA" --num-subdla-samples "$N_SUBDLA" \
    --max-workers 8 --batch-size 1250 \
    --output     "$OUT/${mock}_${tid}.h5" \
    --output-pkl "$OUT/${mock}_${tid}.pkl" \
    >"$log" 2>&1 || { echo "  FAILED — see $log"; continue; }
done < <(cut -f1-9 "$TARGETS")
echo "[batch] $PRESET filter=$FILTER N=$N_DLA  DONE — outputs in $OUT"
