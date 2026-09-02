#!/bin/bash
# submit_run_hz.sh RUN_ID — HIGH-z HBI EXTENSION TRIAL driver (PI ruling 2026-09-02 §15-§20; predeclaration + Amendments 1-2): identical chain, PACK/ROOT from env "[EXTRA cc_real_posterior flags]" [OUT_SUBDIR]
# 2026-09-02 HBI identifiability campaign driver: writes the machine-readable manifest,
# submits the 8-seed base array (warmup 1500) + the stage-1 collector (which submits the
# deep reruns and the stage-2 pooling). Heavy work only ever runs under sbatch.
set -euo pipefail
RUN_ID=${1:?RUN_ID}; EXTRA=${2:-}; SUB=${3:-$RUN_ID}
ROOT=${ROOT:-/scratch/cavestru_root/cavestru0/mfho/r041_max4_highz_2026-09/hz_hbi/runs}
REPO=${REPO:-/home/mfho/wt_hbi_validation_2026_09}
SCRIPTS=$REPO/slurm/greatlakes/validation/hbi_identifiability_2026-09
PACK=${PACK:?PACK (the high-z arm pack) must be given}
OUT=$ROOT/$SUB; mkdir -p $OUT/logs
if ls $OUT/REAL_ln_*.json >/dev/null 2>&1; then echo "REFUSING: $OUT already holds run JSONs (never overwrite)"; exit 3; fi
SEEDS_FILE=$OUT/seeds.txt; printf "%s\n" 20260821 20260822 20260823 20260824 20260825 20260826 20260827 20260828 > $SEEDS_FILE
cd $REPO
COMMIT=$(git rev-parse HEAD); DIRTY=$(git status --porcelain -uno | wc -l)
FREEZE=$(git rev-parse 'prov/paper1-freeze-2026-08-26^{commit}')
git diff $FREEZE HEAD > $OUT/code_diff_from_freeze.patch
PACK_SHA=$(sha256sum $PACK | cut -d' ' -f1)
python3 - <<PY
import json, hashlib, subprocess, datetime, os
def sha(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()
m = dict(run_id="$RUN_ID", out_dir="$OUT", timestamp=datetime.datetime.now().astimezone().isoformat(),
         science_tag="prov/paper1-freeze-2026-08-26", base_commit="$FREEZE", validation_commit="$COMMIT",
         validation_branch=subprocess.run(["git","branch","--show-current"],capture_output=True,text=True).stdout.strip(),
         dirty_tracked_files=int("$DIRTY"), diff_from_freeze_sha256=sha("$OUT/code_diff_from_freeze.patch"),
         pack="$PACK", pack_sha256="$PACK_SHA",
         pack_provenance_sha256=sha("$PACK"[:-4]+".provenance.json"),
         env="gpdla-hbi", env_lock="slurm/greatlakes/production/env_lock_gpdla-hbi_2026-08-26.txt",
         env_lock_sha256=sha("slurm/greatlakes/production/env_lock_gpdla-hbi_2026-08-26.txt"),
         runner="CDDF_analysis/hbi_mcmc/cc_real_posterior.py", runner_sha256=sha("CDDF_analysis/hbi_mcmc/cc_real_posterior.py"),
         model="CDDF_analysis/hbi_mcmc/cc_posterior_validation.py:model_cc", model_sha256=sha("CDDF_analysis/hbi_mcmc/cc_posterior_validation.py"),
         sbatch_chain_sha256=sha("$SCRIPTS/run_chain.sbatch"), sbatch_collect_sha256=sha("$SCRIPTS/run_collect.sbatch"),
         driver_sha256=sha("$SCRIPTS/submit_run.sh"),
         command_template=("python3 -m CDDF_analysis.hbi_mcmc.cc_real_posterior --pack PACK --samples 500 --warmup {1500|3000} "
                           "--chains 2 --target-accept 0.95 --fp-mode informative_ln --seed S " + "$EXTRA" +
                           " --save-nuisance-draws ... --save-all-sites ... --out ..."),
         intervention_flags="$EXTRA", seeds=[20260821,20260822,20260823,20260824,20260825,20260826,20260827,20260828],
         chains=2, warmup_base=1500, warmup_deep=3000, samples=500, target_accept=0.95, fp_mode="informative_ln",
         pooling="cc_pool_posterior --rhat-max 1.10 --div-max 10 --expect-pack-sha256 $PACK_SHA (predeclared CP-3 rule)",
         sbatch=dict(cpus_per_task=4, mem="8G", time_base="03:00:00", partition="standard", account="cavestru0"),
         parent_run="P0 diagnostic estimator dN/dX(>=20.3,[3.8,5.0)) (real value private; closure target; not tuned)",
         jobs={})
json.dump(m, open("$OUT/manifest_$RUN_ID.json","w"), indent=1)
PY
STRIP=$(env | grep -o '^SLURM_[A-Za-z_]*' | sed 's/^/-u /')
JB=$(env $STRIP SBATCH_CONSTRAINT= sbatch --parsable --array=0-7 --job-name=hbiv_${RUN_ID}_base \
     --output=$OUT/logs/chain_base_%A_%a.log \
     --export=ALL,RUN_ID=$RUN_ID,OUT=$OUT,SEEDS_FILE=$SEEDS_FILE,WARMUP=1500,DEEP=0,EXTRA="$EXTRA",REPO=$REPO,PACK=$PACK \
     $SCRIPTS/run_chain.sbatch)
JC=$(env $STRIP SBATCH_CONSTRAINT= sbatch --parsable --dependency=afterany:$JB --job-name=hbiv_${RUN_ID}_collect1 \
     --output=$OUT/logs/collect_stage1_%j.log \
     --export=ALL,RUN_ID=$RUN_ID,OUT=$OUT,SEEDS_FILE=$SEEDS_FILE,EXTRA="$EXTRA",REPO=$REPO,PACK=$PACK,STAGE=1,SCRIPTS=$SCRIPTS,EXPECT_PACK_SHA=$PACK_SHA \
     $SCRIPTS/run_collect.sbatch)
python3 - <<PY
import json; p="$OUT/manifest_$RUN_ID.json"; m=json.load(open(p)); m["jobs"].update(base_array="$JB", collect_stage1="$JC"); json.dump(m, open(p,"w"), indent=1)
PY
echo "SUBMITTED run=$RUN_ID base_array=$JB collect_stage1=$JC out=$OUT commit=$COMMIT dirty=$DIRTY extra='$EXTRA'"
