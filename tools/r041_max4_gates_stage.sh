#!/bin/bash
# MAX4 Gate B / Gate C stage-1 reductions (PI ruling 2026-09-01 late). Light; interactive-safe. Usage:
#   bash tools/r041_max4_gates_stage.sh gateB     # paired QMC spot check (real50 50k vs 100k; inj13 pilot 50k vs 100k with truth)
#   bash tools/r041_max4_gates_stage.sh gateC     # analyzer on A (wave0 MAX4 + pilot wave1) and B (waves 0,1), the frozen paired gate, the provenance index
#   bash tools/r041_max4_gates_stage.sh sharedeps # shared-epsilon micro-audit: analyzer on A_ind, paired gate A_shared vs A_ind, dependence statistics, provenance rows
set -uo pipefail
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh; conda activate gpdla; export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1
cd /home/mfho/wt_highz_repair
ROOT=/scratch/cavestru_root/cavestru0/mfho/r041_highz_repair_2026-08-28; M=/scratch/cavestru_root/cavestru0/mfho/r041_max4_highz_2026-09
case "${1:?gateB|gateC}" in
gateB)
  mkdir -p $M/smoke_real/qmc
  python tools/r041_qmc_compare.py --a-dir $M/smoke_real/real50_MAX4_50k_outputs --b-dir $M/smoke_real/real50_MAX4_100k_outputs --a-label 50k --b-label 100k \
     --population $ROOT/population/r041_population.csv --out $M/smoke_real/qmc/qmc_real50_50k_vs_100k.json 2>&1 | grep -v "UserWarning\|from scipy"
  python tools/r041_qmc_compare.py --a-dir $M/smoke/r041_smoke_MAX4_outputs --b-dir $M/smoke/inj13_100k/r041_smoke_MAX4_100k_outputs --a-label 50k --b-label 100k \
     --truth $ROOT/cmp/r041_cmp_new_wave1.h5.truth.csv --out $M/smoke_real/qmc/qmc_inj13_50k_vs_100k.json 2>&1 | grep -v "UserWarning\|from scipy"
  for d in real50_MAX4_50k_outputs real50_MAX4_100k_outputs; do (cd $M/smoke_real/$d && sha256sum dlacat-*.fits figures/processed/*.h5 BASELINE.env > SHA256SUMS.txt); done
  (cd $M/smoke/inj13_100k/r041_smoke_MAX4_100k_outputs && sha256sum dlacat-*.fits figures/processed/*.h5 BASELINE.env > SHA256SUMS.txt)
  ;;
gateC)
  mkdir -p $M/gate_prescription
  # arm A under MAX4: wave 0 = MAX4-GATEC1-A0, wave 1 = MAX4-PILOT-13
  python tools/r041_analyze.py --truth $ROOT/cmp/r041_cmp_new_wave0.h5.truth.csv $ROOT/cmp/r041_cmp_new_wave1.h5.truth.csv \
     --outputs $M/cmpA/r041_cmp_new_wave0_MAX4_outputs $M/smoke/r041_smoke_MAX4_outputs --population $ROOT/population/r041_population.csv \
     --out $M/gate_prescription/analysis_cmpA_MAX4.json --label cmpA_MAX4 2>&1 | grep -v "UserWarning\|from scipy" | tail -8
  python tools/r041_analyze.py --truth $M/cmpB/r041_cmpB_resid_wave0.h5.truth.csv $M/cmpB/r041_cmpB_resid_wave1.h5.truth.csv \
     --outputs $M/cmpB/r041_cmpB_resid_wave0_outputs $M/cmpB/r041_cmpB_resid_wave1_outputs --population $ROOT/population/r041_population.csv \
     --out $M/gate_prescription/analysis_cmpB_MAX4.json --label cmpB_MAX4 2>&1 | grep -v "UserWarning\|from scipy" | tail -8
  python tools/r041_prescription_gate.py --a $M/gate_prescription/analysis_cmpA_MAX4_per_injection.csv --b $M/gate_prescription/analysis_cmpB_MAX4_per_injection.csv \
     --weights $M/cmpB/gate_weights.json --out $M/gate_prescription/gate_stage1.json --label stage1
  python tools/r041_injection_provenance_index.py --plan-label cmp \
     --arm "A:$ROOT/cmp/r041_cmp_new_wave0.h5.build_summary.json:$M/cmpA/r041_cmp_new_wave0_MAX4_outputs:$M/gate_prescription/analysis_cmpA_MAX4_per_injection.csv:MAX4-GATEC1-A0+MAX4-PILOT-13" \
     --arm "B:$M/cmpB/r041_cmpB_resid_wave0.h5.build_summary.json:$M/cmpB/r041_cmpB_resid_wave0_outputs:$M/gate_prescription/analysis_cmpB_MAX4_per_injection.csv:MAX4-GATEC1-B0+B1" \
     --out $M/gate_prescription/MAX4_INJECTION_PROVENANCE_INDEX_stage1.csv
  for d in $M/cmpA/r041_cmp_new_wave0_MAX4_outputs $M/cmpB/r041_cmpB_resid_wave0_outputs $M/cmpB/r041_cmpB_resid_wave1_outputs; do (cd $d && sha256sum dlacat-*.fits figures/processed/*.h5 BASELINE.env > SHA256SUMS.txt); done
  (cd $M/gate_prescription && sha256sum *.json *.csv > SHA256SUMS.txt)
  ;;
sharedeps)
  # shared-epsilon micro-audit reduction (spec MAX4_SHARED_EPSILON_MICROAUDIT_SPEC_2026-09-02.md, frozen a0e3ff1): A_shared = cmpA MAX4 per-injection table (Gate C), A_ind = new arm
  mkdir -p $M/shared_epsilon
  python tools/r041_analyze.py --truth $M/A_ind/r041_Aind_cmp_wave0.h5.truth.csv $M/A_ind/r041_Aind_cmp_wave1.h5.truth.csv \
     --outputs $M/A_ind/r041_Aind_cmp_wave0_outputs $M/A_ind/r041_Aind_cmp_wave1_outputs --population $ROOT/population/r041_population.csv \
     --out $M/shared_epsilon/analysis_Aind_MAX4.json --label Aind_MAX4 2>&1 | grep -v "UserWarning\|from scipy" | tail -8
  python tools/r041_prescription_gate.py --a $M/gate_prescription/analysis_cmpA_MAX4_per_injection.csv --b $M/shared_epsilon/analysis_Aind_MAX4_per_injection.csv \
     --weights $M/cmpB/gate_weights.json --out $M/shared_epsilon/gate_sharedeps.json --label sharedeps
  python tools/r041_shared_eps_dependence.py --source /nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/loa_highz_2026-08-12c/archive/loa_hz_archive_v1.h5 --plan-label cmp \
     --wave 0:$ROOT/cmp/r041_cmp_new_wave0.h5:$M/A_ind/r041_Aind_cmp_wave0.h5:$M/A_ind/r041_Aind_cmp_wave0.h5.truth.csv \
     --wave 1:$ROOT/cmp/r041_cmp_new_wave1.h5:$M/A_ind/r041_Aind_cmp_wave1.h5:$M/A_ind/r041_Aind_cmp_wave1.h5.truth.csv \
     --pairs $M/shared_epsilon/gate_sharedeps_pairs.csv --out $M/shared_epsilon/dependence_sharedeps.json 2>&1 | grep -v "UserWarning\|from scipy"
  python tools/r041_injection_provenance_index.py --plan-label cmp \
     --arm "A_ind:$M/A_ind/r041_Aind_cmp_wave0.h5.build_summary.json:$M/A_ind/r041_Aind_cmp_wave0_outputs:$M/shared_epsilon/analysis_Aind_MAX4_per_injection.csv:MAX4-SHEPS-AIND0+AIND1" \
     --out $M/shared_epsilon/MAX4_INJECTION_PROVENANCE_INDEX_Aind.csv
  for d in $M/A_ind/r041_Aind_cmp_wave0_outputs $M/A_ind/r041_Aind_cmp_wave1_outputs; do (cd $d && sha256sum dlacat-*.fits figures/processed/*.h5 BASELINE.env > SHA256SUMS.txt); done
  (cd $M/A_ind && sha256sum *.h5 *.csv *.json *.env *.txt *.fits > SHA256SUMS.txt); (cd $M/shared_epsilon && sha256sum *.json *.csv > SHA256SUMS.txt)
  ;;
esac
echo "STAGE_DONE $1 $(date -Is)"
