#!/bin/bash
#SBATCH --job-name=hbiv_collect
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
# 2026-09-02 HBI identifiability campaign collector = the frozen run_cp3_collect.sbatch
# logic (predeclared selection rule, one deep rerun per failing seed, stage-2 pooling with
# the frozen-pack guard), pointed at a validation run directory.
#   env: RUN_ID OUT SEEDS_FILE EXTRA REPO PACK STAGE(1|2) SCRIPTS
set -uo pipefail
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh
conda activate gpdla-hbi
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
STAGE=${STAGE:-1}
echo "run=$RUN_ID stage=$STAGE code=$(git rev-parse HEAD) $(date -Is)"
python - <<PY > $OUT/selection_stage${STAGE}.txt
import json, os, sys
sys.path.insert(0, "$REPO")
from CDDF_analysis.hbi_mcmc.cc_pool_posterior import select_runs
R = "$OUT"
seeds = [int(x) for x in open("$SEEDS_FILE").read().split()]
recs, missing = [], []
for s in seeds:
    for deep in (False, True):
        p = f"{R}/REAL_ln_{'deep_' if deep else ''}s{s}.json"
        if os.path.exists(p):
            d = json.load(open(p)); recs.append(dict(seed=s, deep=deep, file=p, diagnostics=d["diagnostics"], guards=d["guards"]))
        elif not deep:
            missing.append(s)
sel = select_runs(recs, dict(rhat_max=1.10, div_max=10))
need = sorted(set(sel["needs_deep_rerun"]) | {s for s in missing if not os.path.exists(f"{R}/REAL_ln_deep_s{s}.json")})
print(json.dumps(dict(included=[(r["seed"], r["deep"]) for r in sel["included"]],
                      excluded=[(r["seed"], r["deep"], r["reason"]) for r in sel["excluded"]],
                      missing_base_json=missing, needs_deep_rerun=need), indent=1))
open(f"{R}/deep_seeds_stage{$STAGE}.txt", "w").write("\n".join(str(s) for s in need) + ("\n" if need else ""))
PY
cat $OUT/selection_stage${STAGE}.txt
N=$(grep -c . $OUT/deep_seeds_stage${STAGE}.txt || true)
STRIP=$(env | grep '^SLURM_' | cut -d= -f1 | sed 's/^/-u /')
if [ "$STAGE" = "1" ] && [ "$N" -gt 0 ]; then
  echo "submitting $N deep reruns (predeclared rule, warmup 3000)"
  cp $OUT/deep_seeds_stage1.txt $OUT/deep_seeds.txt
  J=$(env $STRIP SBATCH_CONSTRAINT= sbatch --parsable --array=0-$((N-1)) --job-name=hbiv_${RUN_ID}_deep \
       --output=$OUT/logs/chain_deep_%A_%a.log \
       --export=ALL,RUN_ID=$RUN_ID,OUT=$OUT,SEEDS_FILE=$OUT/deep_seeds.txt,WARMUP=3000,DEEP=1,EXTRA="${EXTRA:-}",REPO=$REPO,PACK=$PACK \
       $SCRIPTS/run_chain.sbatch)
  J2=$(env $STRIP SBATCH_CONSTRAINT= sbatch --parsable --dependency=afterany:$J --job-name=hbiv_${RUN_ID}_collect2 \
       --output=$OUT/logs/collect_stage2_%j.log \
       --export=ALL,RUN_ID=$RUN_ID,OUT=$OUT,SEEDS_FILE=$SEEDS_FILE,EXTRA="${EXTRA:-}",REPO=$REPO,PACK=$PACK,STAGE=2,SCRIPTS=$SCRIPTS \
       $SCRIPTS/run_collect.sbatch)
  echo "deep array $J; stage-2 collector $J2"
  echo "{\"deep_array\": \"$J\", \"collect_stage2\": \"$J2\"}" > $OUT/jobids_stage1_submitted.json
  exit 0
fi
echo "pooling (stage $STAGE)"
BASE=$(ls $OUT/REAL_ln_s2026????.json 2>/dev/null | tr '\n' ' ')
DEEP=$(ls $OUT/REAL_ln_deep_s2026????.json 2>/dev/null | tr '\n' ' ')
python -m CDDF_analysis.hbi_mcmc.cc_pool_posterior --runs $BASE ${DEEP:+--deep $DEEP} --rhat-max 1.10 --div-max 10 \
  --expect-pack-sha256 ${EXPECT_PACK_SHA:-219c43aaa59aeb7f070e223c3e652190b4cabe55683c45679a88522d88b920bb} \
  --out $OUT/POOLED_ln_${RUN_ID}.json | tee $OUT/POOL_summary.txt
RC=$?
(cd $OUT && sha256sum *.json *.npz *.txt 2>/dev/null > SHA256SUMS.txt)
if [ "$RUN_ID" = "R0" ]; then
  python tools/hbi_validation/r0_bitrepro.py --new-dir $OUT \
    --ref-dir /scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v2_20260821/cp3_real \
    --out $OUT/R0_BITREPRO.json | tee $OUT/R0_BITREPRO.txt
fi
echo "COLLECT_DONE run=$RUN_ID stage=$STAGE rc=$RC $(date -Is)"
