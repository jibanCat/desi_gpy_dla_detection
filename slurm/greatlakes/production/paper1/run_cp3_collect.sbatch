#!/bin/bash
#SBATCH --job-name=cp3_collect
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v2_20260821/cp3_real/logs/cp3_collect_%j.log
# PAPER-1 PRODUCT 3 OF RECORD (CP-3 collector: predeclared selection rule -> pooled frozen posterior).
# Committed copy 2026-08-26: deep_seeds.txt is no longer truncated (deep_seeds_of_record.txt written);
# the pooling call carries --expect-pack-sha256 of the certified pack (219c43aa...).
# CP-3 collector (predeclared §3/§6): apply the convergence rule; STAGE=1: submit the deep
# reruns for failing/missing seeds (then a STAGE=2 collector afterany); STAGE=2 (or no
# rerun needed): build the pooled artifact. Decides nothing else.
set -uo pipefail
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh
conda activate gpdla-hbi
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1
cd /home/mfho/wt_forward_2026_08
D=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v2_20260821
R=$D/cp3_real; STAGE=${STAGE:-1}
echo "stage=$STAGE code=$(git rev-parse --short HEAD) $(date)"
python - <<'PY' > $R/selection_stage${STAGE}.txt
import json, os, glob, sys
sys.path.insert(0, "/home/mfho/wt_forward_2026_08")
import importlib.util as ilu
spec = ilu.spec_from_file_location("pp", "/home/mfho/wt_forward_2026_08/CDDF_analysis/hbi_mcmc/cc_pool_posterior.py")
PP = ilu.module_from_spec(spec); spec.loader.exec_module(PP)
R = "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v2_20260821/cp3_real"
seeds = [20260821,20260822,20260823,20260824,20260825,20260826,20260827,20260828]
recs, missing = [], []
for s in seeds:
    for deep in (False, True):
        p = f"{R}/REAL_ln_{'deep_' if deep else ''}s{s}.json"
        if os.path.exists(p):
            d = json.load(open(p)); recs.append(dict(seed=s, deep=deep, file=p, diagnostics=d["diagnostics"], guards=d["guards"]))
        elif not deep:
            missing.append(s)
sel = PP.select_runs(recs, dict(rhat_max=1.10, div_max=10))
need = sorted(set(sel["needs_deep_rerun"]) | {s for s in missing if not os.path.exists(f"{R}/REAL_ln_deep_s{s}.json")})
print(json.dumps(dict(included=[(r["seed"], r["deep"]) for r in sel["included"]],
                      excluded=[(r["seed"], r["deep"], r["reason"]) for r in sel["excluded"]],
                      missing_base_json=missing, needs_deep_rerun=need), indent=1))
open(f"{R}/deep_seeds.txt", "w").write("\n".join(str(s) for s in need) + ("\n" if need else ""))
PY
cat $R/selection_stage${STAGE}.txt
N=$(grep -c . $R/deep_seeds.txt || true)
STRIP=$(env | grep '^SLURM_' | cut -d= -f1 | sed 's/^/-u /')
if [ "$STAGE" = "1" ] && [ "$N" -gt 0 ]; then
  echo "submitting $N deep reruns (predeclared, warmup 3000)"
  J=$(env $STRIP SBATCH_CONSTRAINT= sbatch --parsable --array=0-$((N-1)) $R/run_cp3_deep.sbatch)
  env $STRIP SBATCH_CONSTRAINT= STAGE=2 sbatch --export=ALL,STAGE=2 --dependency=afterany:$J $R/run_cp3_collect.sbatch
  echo "deep array $J; stage-2 collector queued"; exit 0
fi
echo "pooling (stage $STAGE)"
BASE=$(ls $R/REAL_ln_s2026082?.json 2>/dev/null | tr '\n' ' ')
DEEP=$(ls $R/REAL_ln_deep_s2026082?.json 2>/dev/null | tr '\n' ' ')
python -m CDDF_analysis.hbi_mcmc.cc_pool_posterior --runs $BASE ${DEEP:+--deep $DEEP} --rhat-max 1.10 --div-max 10 --expect-pack-sha256 219c43aaa59aeb7f070e223c3e652190b4cabe55683c45679a88522d88b920bb --out $R/POOLED_ln_real_v2_20260821.json | tee $R/POOL_summary.txt
echo "CP3_POOL_DONE $(date)"
