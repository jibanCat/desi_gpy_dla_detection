#!/bin/bash
# Checkpoint-3b analysis chain (2026-09-02 HBI campaign): once R4, R2x4 (ROOT/R2b) and LC exist, run every reduction
# needed for PI checkpoint 3b/4 in one go (all light; interactive-safe). Usage: bash tools/hbi_validation/run_3b.sh
set -uo pipefail
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh; conda activate gpdla-hbi
export JAX_PLATFORMS=cpu HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/mfho/wt_hbi_validation_2026_09
ROOT=/scratch/cavestru_root/cavestru0/mfho/hbi_validation_2026-09-02
PACK=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v2_20260821/modelA_pack_REAL_loa50k_c3300_bw0p2_pad19p0_molly172_v2.npz
T=tools/hbi_validation
for spec in "R4|--expect-fixed t --expect-fixed psi_c" "R2b|--expect-t-scale 4"; do
  r=${spec%%|*}; args=${spec#*|}; mkdir -p $ROOT/$r/analysis
  [ -f $ROOT/$r/POOLED_ln_$r.json ] || { echo "$r: no pooled artifact yet"; continue; }
  python $T/check_intervention.py --run-dir $ROOT/$r --pooled $ROOT/$r/POOLED_ln_$r.json $args --out $ROOT/$r/analysis/check_intervention_$r.json 2>&1 | grep -v "NVIDIA\|WARNING" | tail -3
  EX=""; for f in $(ls $ROOT/$r/REAL_ln_deep_s2026????_allsites.npz 2>/dev/null); do EX="$EX --extra-arm $f:${r}_$(basename $f _allsites.npz | sed 's/REAL_ln_deep_//')d"; done
  python $T/geometry.py --pack $PACK --run-dir $ROOT/$r --pooled $ROOT/$r/POOLED_ln_$r.json --out $ROOT/$r/analysis/geometry_$r.json $EX > $ROOT/$r/analysis/geometry_$r.log 2>&1; echo "$r geometry exit $?"
  python $T/sci_corner.py --pack $PACK --geometry $ROOT/$r/analysis/geometry_$r.json --pooled $ROOT/$r/POOLED_ln_$r.json --out-dir $ROOT/$r/analysis/sci --run-id $r --axes-from $ROOT/R0/analysis/sci/sci_axes_R0.json > $ROOT/$r/analysis/sci_corner_$r.log 2>&1; echo "$r sci_corner exit $?"
done
# long chain: diagnostics bundle per arm, geometry on the two arms, strict audit with LC as its own family
if ls $ROOT/LC/REAL_ln_lc_s2026????_allsites.npz >/dev/null 2>&1; then
  mkdir -p $ROOT/LC/analysis/chains $ROOT/LC/analysis/lpt
  ARMS=$(ls $ROOT/LC/REAL_ln_lc_s2026????_allsites.npz | tr '\n' ' ')
  for f in $ARMS; do python $T/chain_diagnostics.py --pack $PACK --allsites $f --label LC_$(basename $f _allsites.npz | sed 's/REAL_ln_lc_//') --out-dir $ROOT/LC/analysis/chains > $ROOT/LC/analysis/chains/$(basename $f .npz).log 2>&1; echo "chain diag $(basename $f) exit $?"; done
  python $T/geometry.py --pack $PACK --run-dir $ROOT/LC --arms $ARMS --out $ROOT/LC/analysis/geometry_LC.json > $ROOT/LC/analysis/geometry_LC.log 2>&1; echo "LC geometry exit $?"
  python $T/lpt_audit.py --pack $PACK --family R0=$ROOT/R0/POOLED_ln_R0.json --family LC=$(echo $ARMS | tr ' ' ',' | sed 's/,$//') --family R2A=$ROOT/R2/POOLED_ln_R2.json --extra $ROOT/R0/REAL_ln_deep_s20260826_allsites.npz:mirror_s26d_c0:0 --out-dir $ROOT/LC/analysis/lpt > $ROOT/LC/analysis/lpt/lpt_LC.log 2>&1; echo "LC lpt exit $?"
  # the same audit with LC as the BASELINE family (its own plots/decomposition first)
  mkdir -p $ROOT/LC/analysis/lpt_LCbase
  python $T/lpt_audit.py --pack $PACK --family LC=$(echo $ARMS | tr ' ' ',' | sed 's/,$//') --family R0=$ROOT/R0/POOLED_ln_R0.json --out-dir $ROOT/LC/analysis/lpt_LCbase > $ROOT/LC/analysis/lpt_LCbase/log 2>&1; echo "LC-base lpt exit $?"
fi
# cross-run comparison with everything available
RUNS="R1=$ROOT/R1/analysis/geometry_R1.json R2A=$ROOT/R2/analysis/geometry_R2.json R2Binit=$ROOT/R2B/analysis/geometry_R2B.json R3=$ROOT/R3/analysis/geometry_R3.json"
[ -f $ROOT/R4/analysis/geometry_R4.json ] && RUNS="$RUNS R4=$ROOT/R4/analysis/geometry_R4.json"
[ -f $ROOT/R2b/analysis/geometry_R2b.json ] && RUNS="$RUNS R2x4=$ROOT/R2b/analysis/geometry_R2b.json"
[ -f $ROOT/LC/analysis/geometry_LC.json ] && RUNS="$RUNS LC=$ROOT/LC/analysis/geometry_LC.json"
python $T/compare_runs.py --baseline $ROOT/R0/analysis/geometry_R0.json --runs $RUNS --out-dir $ROOT/compare > $ROOT/compare/compare.log 2>&1; echo "compare exit $?"
echo "RUN_3B_DONE $(date -Is)"
