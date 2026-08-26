#!/usr/bin/env bash
# Paper-1 test profiles (see tests/profiles/README.md). Usage: tools/paper1/run_tests.sh hbi|finder|training [extra pytest args]
set -euo pipefail
P=${1:?profile}; shift || true
cd "$(dirname "$0")/../.."
case "$P" in
  hbi) ENV=gpdla-hbi ;;
  finder|training) ENV=gpdla ;;
  *) echo "unknown profile $P"; exit 2 ;;
esac
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mapfile -t FILES < "tests/profiles/$P.txt"
exec "$HOME/.conda/envs/$ENV/bin/python" -m pytest -q -p no:cacheprovider -rfEs "${FILES[@]}" "$@"
