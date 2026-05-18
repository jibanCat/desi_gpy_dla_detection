#!/bin/bash
#SBATCH -A cavestru0
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 4:00:00
#SBATCH -J stack_real_loa
#SBATCH -o slurm/greatlakes/stack_real_loa_%j.log
#SBATCH -e slurm/greatlakes/stack_real_loa_%j.log

# Real-LOA LLS / sub-DLA / DLA metal-line stacking (PR #8).
#
# Runs examples/stack_real_loa_dlas.py end-to-end: reads the DLA catalog
# + altbal BAL catalog, builds per-NHI-bin median composites (non-BAL +
# BAL split), the production/diagnostic bin sets, the Lyman-limit-break
# figure, and the z-scrambled real-vs-control plots. Writes the cached
# stack_curves.npz + figures into docs/notes/2026-05-15_stack_real_loa_dlas/.
#
# Single-threaded Python; bottleneck is random HDF5 reads off /scratch.
# Expect ~45-90 min wall.

set -eo pipefail
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$HOME/.local/usr/local/lib64:${LD_LIBRARY_PATH:-}"

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpdla

echo "===================================================="
echo "  Real-LOA absorber metal-line stacking"
echo "  job: ${SLURM_JOB_ID:-<interactive>}  cpus: $(nproc)"
echo "  repo: $REPO_DIR"
echo "===================================================="

# Preflight: imports + input files reachable.
python -c "
import numpy, h5py, matplotlib
from astropy.io import fits
from numpy.lib import recfunctions
print('[preflight] imports OK')
" || { echo '[error] preflight import failed' >&2; exit 1; }

python - <<'PY' || { echo '[error] input files not reachable' >&2; exit 2; }
import sys, importlib.util
spec = importlib.util.spec_from_file_location('stk', 'examples/stack_real_loa_dlas.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import os
for p in (m.DLACAT, m.LOA_ARCHIVE, m.BAL_CATALOG):
    ok = os.path.exists(p)
    print(f'[preflight] {"OK " if ok else "MISSING"} {p}')
    if not ok:
        sys.exit(1)
PY

echo
echo "=== running full stack ==="
python -u examples/stack_real_loa_dlas.py

echo
echo "=== STACK COMPLETE ==="
echo "  outputs: docs/notes/2026-05-15_stack_real_loa_dlas/"
ls -la docs/notes/2026-05-15_stack_real_loa_dlas/
