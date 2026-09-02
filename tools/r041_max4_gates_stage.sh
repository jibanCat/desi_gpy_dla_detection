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
p0)
  # P0 post-run chain (PI rulings 2026-09-01 §22-§28, 2026-09-02): stamp checks -> combine + flags (low-z recipe) + bits 3-4 no-op + schema/multiplicity/equality
  # checks -> MAX1-vs-MAX4 catalogue comparison -> fiducial analysis (pooled + per wave) -> candidate-status gate -> provenance index -> hashes.
  # The high-z HBI reduction (track_c_tf_hz.py --variant r041cal) is launched separately by `p0hz` (sbatch).
  set -e
  HZ1=/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/gl_cddf_loa_hz_v1_20260813/outputs
  TFHZ=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/tf_hz/mockdir
  mkdir -p $M/real/combined $M/fid_max4/analysis $M/p0_checks
  # (1) BASELINE.env stamp (§23) on all four output dirs
  python - <<'PY'
import os, sys, json
M='/scratch/cavestru_root/cavestru0/mfho/r041_max4_highz_2026-09'
want={'MAX_DLAS':'4','SINGLE_ABSORBER_MODEL':'1','FILTER_LOW_LIKELIHOOD':'1','NUM_DLA_SAMPLES':'50000','CODE_DIRTY':'clean','GPDLA_ZMIN_QSO':'4.25','GPDLA_ZMAX_QSO':'7.0'}
out={}; bad=0
for d in ['real/r041_real_MAX4_outputs','fid_max4/r041_fid_wave0_MAX4_outputs','fid_max4/r041_fid_wave1_MAX4_outputs','fid_max4/r041_fid_wave2_MAX4_outputs']:
    kv={}
    for line in open(os.path.join(M,d,'BASELINE.env')):
        if '=' in line and not line.startswith('#'):
            k,v=line.rstrip('\n').split('=',1); kv[k]=v.strip('"')
    chk={k:(kv.get(k)==v) for k,v in want.items()}
    chk['DLA_SAMPLES_FILE_50000']=kv.get('DLA_SAMPLES_FILE','').endswith('pw_samples_a3_172_225_50000.mat')
    chk['archive_recorded']=bool(kv.get('GPDLA_SPECTRA_ARCHIVE')); chk['hpx_list_recorded']=bool(kv.get('EXTERNAL_HPX_LIST'))
    out[d]=dict(checks=chk, CODE_COMMIT=kv.get('CODE_COMMIT'), archive=kv.get('GPDLA_SPECTRA_ARCHIVE'), hpx=kv.get('EXTERNAL_HPX_LIST'), ok=all(chk.values()))
    bad+= (not all(chk.values()))
json.dump(out, open(os.path.join(M,'p0_checks','baseline_stamp_check.json'),'w'), indent=1)
print('STAMP CHECK', 'ALL OK' if bad==0 else f'{bad} DIR(S) FAIL'); 
for d,v in out.items(): print(' ', d, v['ok'], v['CODE_COMMIT'])
sys.exit(1 if bad else 0)
PY
  # (2) combine (gap-fatal) + flags with the low-z recipe (lyb dz 0.005; BAL cat = the staged hz bal_cat; no BF band, as in loa_main_dark_v1)
  python examples/combine_dlacat.py --procdir $M/real/r041_real_MAX4_outputs --out $M/real/combined/dlacat-loa-hz-MAX4-v1.fits --expect-positions 2179 --fail-on-gap 2>&1 | grep -v "UserWarning\|from scipy" | tail -4
  python tools/postprocess/add_dla_flags.py --catalog-dir $M/real/combined --bal-cat $TFHZ/bal_cat.fits --lyb-veto-dz 0.005 --no-bf-band 2>&1 | grep -v "UserWarning\|from scipy" | tail -4
  # (3) schema / bits 3-4 no-op / multiplicity / SNR-z-BAL equality vs the population and the low-z catalogue schema
  python - <<'PY'
import numpy as np, json, csv
from astropy.io import fits
M='/scratch/cavestru_root/cavestru0/mfho/r041_max4_highz_2026-09'
c=fits.open(f'{M}/real/combined/dlacat-loa-hz-MAX4-v1.fits')[1]; d=c.data; cols=list(c.columns.names)
lz=fits.open('/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/loa_main_dark_v1/dlacat-loa-main-dark-v1.fits', memmap=True)[1].columns.names
pop={int(r['TARGETID']):r for r in csv.DictReader(open('/scratch/cavestru_root/cavestru0/mfho/r041_highz_repair_2026-08-28/population/r041_population.csv'))}
tid=d['TARGETID'].astype(np.int64); flag=d['DLAFLAG'].astype(np.int64)
b3=int(((flag>>3)&1).sum()); b4=int(((flag>>4)&1).sum()); lyb=int(np.sum(d['LYBETA_FLAG'])); bal=int(np.sum(d['BAL_FLAG']))
per=np.unique(tid, return_counts=True)[1]
zq={}; snr={}
for t,z,s in zip(tid, d['Z_QSO'], d['SNR_REDSIDE']): zq.setdefault(int(t), set()).add(round(float(z),6)); snr.setdefault(int(t), set()).add(round(float(s),4))
dz=max(abs(float(list(zq[t])[0])-float(pop[t]['z_qso'])) for t in zq if t in pop); ds=max(abs(float(list(snr[t])[0])-float(pop[t]['snr'])) for t in snr if t in pop and np.isfinite(list(snr[t])[0]))
res=dict(n_rows=int(len(d)), n_tids=int(len(per)), n_pop=len(pop), tids_not_in_pop=int(sum(1 for t in set(tid.tolist()) if t not in pop)), pop_tids_without_row=int(sum(1 for t in pop if t not in zq)),
         schema_equals_lowz=(cols==lz), cols=cols, lowz_cols=lz, rows_per_tid_max=int(per.max()), rows_per_tid_gt4=int((per>4).sum()), DLAFLAG_bit3_set=b3, DLAFLAG_bit4_set=b4, LYBETA_FLAG_true=lyb, BAL_FLAG_true=bal,
         bits34_noop=(b3==0 and b4==0), z_qso_intra_tid_nunique_gt1=int(sum(1 for v in zq.values() if len(v)>1)), max_abs_dz_qso_vs_population=float(dz), max_abs_dsnr_vs_population=float(ds),
         equality_ok=(dz<1e-5 and ds<1e-3), pop_BI_CIV_gt0=0)
json.dump(res, open(f'{M}/p0_checks/real_catalogue_checks.json','w'), indent=1)
print('REAL CATALOGUE CHECKS:', {k:v for k,v in res.items() if k not in ('cols','lowz_cols')})
PY
  # (4) MAX1 (superseded diagnostic) vs MAX4 catalogue comparison on the population
  python tools/r041_qmc_compare.py --a-dir $HZ1 --b-dir $M/real/r041_real_MAX4_outputs --a-label MAX1_FILTER0_100k --b-label MAX4_FILTER1_50k --population $ROOT/population/r041_population.csv --out $M/p0_checks/MAX1_vs_MAX4_catalogue_comparison.json 2>&1 | grep -v "UserWarning\|from scipy" | tail -3
  # (5) fiducial analysis: pooled (the calibration) + per wave (provenance index)
  python tools/r041_analyze.py --truth $ROOT/fid/r041_fid_wave0.h5.truth.csv $ROOT/fid/r041_fid_wave1.h5.truth.csv $ROOT/fid/r041_fid_wave2.h5.truth.csv \
     --outputs $M/fid_max4/r041_fid_wave0_MAX4_outputs $M/fid_max4/r041_fid_wave1_MAX4_outputs $M/fid_max4/r041_fid_wave2_MAX4_outputs --population $ROOT/population/r041_population.csv \
     --out $M/fid_max4/analysis/analysis_fid_MAX4.json --label fid_MAX4 2>&1 | grep -v "UserWarning\|from scipy" | tail -6
  for w in 0 1 2; do python tools/r041_analyze.py --truth $ROOT/fid/r041_fid_wave$w.h5.truth.csv --outputs $M/fid_max4/r041_fid_wave${w}_MAX4_outputs --population $ROOT/population/r041_population.csv \
     --out $M/fid_max4/analysis/analysis_fid_MAX4_wave$w.json --label fid_MAX4_wave$w 2>&1 | grep -v "UserWarning\|from scipy" | tail -1; done
  # (6) the predeclared candidate-status gate (MAX1 diagnostic arm = R-041A analysis of the same archives)
  python tools/r041_candidate_status_gate.py --max1 $ROOT/r041a/analysis_fid_per_injection.csv --max4 $M/fid_max4/analysis/analysis_fid_MAX4_per_injection.csv --out $M/p0_checks/candidate_status_gate.json 2>&1 | grep -v "UserWarning\|from scipy"
  # (7) provenance index rows for all 2900 fiducial injections (arm A_shared, one arm per wave)
  python tools/r041_injection_provenance_index.py --plan-label fid \
     --arm "A_shared:$ROOT/fid/r041_fid_wave0.h5.build_summary.json:$M/fid_max4/r041_fid_wave0_MAX4_outputs:$M/fid_max4/analysis/analysis_fid_MAX4_wave0_per_injection.csv:MAX4-P0-FID0" \
     --arm "A_shared:$ROOT/fid/r041_fid_wave1.h5.build_summary.json:$M/fid_max4/r041_fid_wave1_MAX4_outputs:$M/fid_max4/analysis/analysis_fid_MAX4_wave1_per_injection.csv:MAX4-P0-FID1" \
     --arm "A_shared:$ROOT/fid/r041_fid_wave2.h5.build_summary.json:$M/fid_max4/r041_fid_wave2_MAX4_outputs:$M/fid_max4/analysis/analysis_fid_MAX4_wave2_per_injection.csv:MAX4-P0-FID2" \
     --out $M/fid_max4/analysis/MAX4_INJECTION_PROVENANCE_INDEX_fiducial.csv
  # (8) hashes
  for d in $M/real/r041_real_MAX4_outputs $M/fid_max4/r041_fid_wave0_MAX4_outputs $M/fid_max4/r041_fid_wave1_MAX4_outputs $M/fid_max4/r041_fid_wave2_MAX4_outputs; do (cd $d && sha256sum dlacat-*.fits figures/processed/*.h5 BASELINE.env > SHA256SUMS.txt); done
  (cd $M/real/combined && sha256sum *.fits > SHA256SUMS.txt); (cd $M/fid_max4/analysis && sha256sum *.json *.csv > SHA256SUMS.txt); (cd $M/p0_checks && sha256sum *.json *.md > SHA256SUMS.txt)
  set +e
  ;;
p0hz)
  # the high-z HBI reduction with the MAX4 calibration and the MAX4 real catalogue, as an sbatch job (1 core, real-LOA numbers to SCRATCH only)
  mkdir -p $M/measurement
  cat > $M/measurement/run_tf_hz_MAX4.sbatch <<SB
#!/bin/bash
#SBATCH --job-name=tf_hz_MAX4 --account=cavestru0 --partition=standard --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16g --time=04:00:00
#SBATCH --output=$M/measurement/tf_hz_MAX4_%j.log
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh; conda activate gpdla; export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /home/mfho/wt_highz_repair
python CDDF_analysis/hbi/track_c_tf_hz.py --variant r041cal --r041-analysis $M/fid_max4/analysis/analysis_fid_MAX4.json --hz-cat $M/real/r041_real_MAX4_outputs \
   --hz-mockdir /scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/tf_hz/mockdir --fp loa0 --window lya --zbins 3.8,4.25,4.5,5.0 --n-mc 2000 \
   --out-json $M/measurement/track_c_tf_hz_MAX4_r041cal_loa0_lya.json --dump-npz $M/measurement/track_c_tf_hz_MAX4_r041cal_loa0_lya.npz --force
echo "TF_HZ_DONE \$(date -Is)"
SB
  STRIP=$(env | grep -o '^SLURM_[A-Za-z0-9_]*' | sed 's/^/-u /' | tr '\n' ' ')
  env $STRIP sbatch $M/measurement/run_tf_hz_MAX4.sbatch
  ;;
p1)
  # P1 reductions (gates predeclared at notes 2ccf8db / 64d817d BEFORE any output was read): pairs (multi-HCD gate), mean-flux (cell-level vs the
  # P0 fiducial), 2LPT random vs clustered (paired by injection), London vs 2LPT random (cell-level); provenance index rows for the archive-route waves; hashes.
  set -e
  P=$M/p1; OUT=$P/reductions; mkdir -p $OUT $P/mock/truth_r041
  W=$M/cmpB/gate_weights.json; FIDPI=$M/fid_max4/analysis/analysis_fid_MAX4_per_injection.csv
  # (1) real pairs — per-absorber / per-system scoring under the multi-HCD gate (+ the analyzer's descriptive pair tables)
  python tools/r041_multihcd_score.py --truth $ROOT/pairs/r041_pairs_wave0.h5.truth.csv $ROOT/pairs/r041_pairs_wave1.h5.truth.csv $ROOT/pairs/r041_pairs_wave2.h5.truth.csv \
     --outputs $P/pairs/r041_pairs_wave0_MAX4_outputs $P/pairs/r041_pairs_wave1_MAX4_outputs $P/pairs/r041_pairs_wave2_MAX4_outputs \
     --reference $FIDPI --population $ROOT/population/r041_population.csv --weights $W --f-multi 0.155 0.476 --out $OUT/multihcd_pairs.json --label pairs_real 2>&1 | grep -v "UserWarning\|from scipy"
  python tools/r041_analyze.py --truth $ROOT/pairs/r041_pairs_wave0.h5.truth.csv $ROOT/pairs/r041_pairs_wave1.h5.truth.csv $ROOT/pairs/r041_pairs_wave2.h5.truth.csv \
     --outputs $P/pairs/r041_pairs_wave0_MAX4_outputs $P/pairs/r041_pairs_wave1_MAX4_outputs $P/pairs/r041_pairs_wave2_MAX4_outputs --population $ROOT/population/r041_population.csv \
     --out $OUT/analysis_pairs_MAX4.json --label pairs_MAX4 2>&1 | grep -v "UserWarning\|from scipy" | tail -3
  # (2) mean-flux arms — analyzer per model, cell-level comparison vs the P0 fiducial
  for m in fg2008 becker2013; do
    python tools/r041_analyze.py --truth $ROOT/mf/r041_mf_${m}_wave0.h5.truth.csv $ROOT/mf/r041_mf_${m}_wave1.h5.truth.csv $ROOT/mf/r041_mf_${m}_wave2.h5.truth.csv \
       --outputs $P/mf/r041_mf_${m}_wave0_MAX4_outputs $P/mf/r041_mf_${m}_wave1_MAX4_outputs $P/mf/r041_mf_${m}_wave2_MAX4_outputs --population $ROOT/population/r041_population.csv \
       --out $OUT/analysis_mf_${m}_MAX4.json --label mf_${m}_MAX4 2>&1 | grep -v "UserWarning\|from scipy" | tail -2
    python tools/r041_cell_compare.py --a $FIDPI --b $OUT/analysis_mf_${m}_MAX4_per_injection.csv --a-label fiducial_P0 --b-label mf_${m} --weights $W --out $OUT/cell_mf_${m}_vs_fid.json --label mf_${m} 2>&1 | grep -v "UserWarning\|from scipy" | tail -4
  done
  # (3) mock arms — truth conversion, analyzer, paired random-vs-clustered gate, cell-level London vs 2LPT random
  for arm in 2lpt/random 2lpt/clustered london/random; do n=$(echo $arm | tr '/' '_')
    python tools/r041_mock_truth_to_r041.py --truth-fits $ROOT/mock/$arm/injection_truth.fits --out-truth $P/mock/truth_r041/${n}_truth.csv --out-population $P/mock/truth_r041/${n}_population.csv
    python tools/r041_analyze.py --truth $P/mock/truth_r041/${n}_truth.csv --outputs $P/mock/${n}_MAX4_outputs --population $P/mock/truth_r041/${n}_population.csv \
       --out $OUT/analysis_mock_${n}_MAX4.json --label mock_${n}_MAX4 2>&1 | grep -v "UserWarning\|from scipy" | tail -2
  done
  python tools/r041_prescription_gate.py --a $OUT/analysis_mock_2lpt_random_MAX4_per_injection.csv --b $OUT/analysis_mock_2lpt_clustered_MAX4_per_injection.csv --weights $W --out $OUT/gate_random_vs_clustered.json --label random_vs_clustered --allow-z-mismatch
  python tools/r041_cell_compare.py --a $OUT/analysis_mock_2lpt_random_MAX4_per_injection.csv --b $OUT/analysis_mock_london_random_MAX4_per_injection.csv --a-label 2lpt_random --b-label london_random --weights $W --bounded-thr 0.10 --out $OUT/cell_london_vs_2lpt.json --label london_transfer 2>&1 | grep -v "UserWarning\|from scipy" | tail -4
  # (4) provenance index rows (archive-route waves have build summaries)
  python tools/r041_injection_provenance_index.py --plan-label pairs \
     --arm "A_shared:$ROOT/pairs/r041_pairs_wave0.h5.build_summary.json:$P/pairs/r041_pairs_wave0_MAX4_outputs:$OUT/analysis_pairs_MAX4_per_injection.csv:MAX4-P1-pairs_wave0+1+2" --out $OUT/MAX4_INJECTION_PROVENANCE_INDEX_pairs.csv
  # (5) hashes
  for d in $P/pairs/*_outputs $P/mf/*_outputs $P/mock/*_outputs; do (cd $d && sha256sum dlacat-*.fits BASELINE.env > SHA256SUMS.txt); done
  (cd $OUT && sha256sum *.json *.csv > SHA256SUMS.txt)
  set +e
  ;;
p1native_launch)
  # launch the native multi-HCD arms (built by r041_mock_campaign.py --arms native; envs carry the MAX4 config + REPO_ROOT); dry-run first
  STRIP=$(env | grep -o '^SLURM_[A-Za-z0-9_]*' | sed 's/^/-u /' | tr '\n' ' ')
  for fam in 2lpt london; do E=$M/p1/mock_native/$fam/native.env; [ -f $E ] || { echo "missing $E"; continue; }
    env $STRIP bash slurm/greatlakes/production/launch_gl.sh $E --dry-run --no-sleep 2>&1 | grep -o -E "chdir=[^ ]*|MAX_DLAS=[0-9],SINGLE_ABSORBER_MODEL=[0-9],FILTER_LOW_LIKELIHOOD=[0-9]|NUM_DLA_SAMPLES=[0-9]*|submitted [0-9]* sbatch" | sort -u | tr '\n' ' '; echo
    OUT=$(env $STRIP bash slurm/greatlakes/production/launch_gl.sh $E --no-sleep 2>&1); IDS=$(echo "$OUT" | grep -o "Submitted batch job [0-9]*" | grep -o "[0-9]*$" | tr '\n' ' ')
    echo "MAX4-P1-native_$fam jobs($(echo $IDS | wc -w)): $IDS $(date -Is)" | tee -a $M/p1/P1_LAUNCH_RECORD.txt | cut -c1-160; done
  ;;
p1native_reduce)
  P=$M/p1; OUT=$P/reductions; mkdir -p $OUT; W=$M/cmpB/gate_weights.json; FIDPI=$M/fid_max4/analysis/analysis_fid_MAX4_per_injection.csv
  for fam in 2lpt london; do R2=$P/mock_native/$fam; [ -d $R2/native_outputs ] || { echo "no outputs for $fam"; continue; }
    python tools/r041_multihcd_score.py --truth $R2/native/native_truth.csv --outputs $R2/native_outputs --reference $FIDPI --population $R2/population_native.csv --weights $W \
       --f-multi 0.155 0.476 --reference-from-singles --out $OUT/multihcd_native_${fam}.json --label native_${fam} 2>&1 | grep -v "UserWarning\|from scipy"
    (cd $R2/native_outputs && sha256sum dlacat-*.fits BASELINE.env > SHA256SUMS.txt); done
  (cd $OUT && sha256sum *.json *.csv > SHA256SUMS.txt)
  ;;
p1ctrl_launch)
  # launch the clustering-control arms and the mean-flux control variants (built by r041_mock_campaign.py; envs carry MAX4 + REPO_ROOT); dry-run first
  STRIP=$(env | grep -o '^SLURM_[A-Za-z0-9_]*' | sed 's/^/-u /' | tr '\n' ' ')
  for E in $M/p1/clustering_control/syscluster.env $M/p1/clustering_control/sysrandom.env $M/p1/clustering_control/sysshuffle.env \
           $M/p1/meanflux_control/turner2024_m1s/random.env $M/p1/meanflux_control/turner2024_p1s/random.env $M/p1/meanflux_control/ding2024_hz/random.env; do
    [ -f $E ] || { echo "missing $E"; continue; }; N=$(basename $(dirname $E))_$(basename $E .env)
    env $STRIP bash slurm/greatlakes/production/launch_gl.sh $E --dry-run --no-sleep 2>&1 | grep -o -E "chdir=[^ ]*|MAX_DLAS=[0-9],SINGLE_ABSORBER_MODEL=[0-9],FILTER_LOW_LIKELIHOOD=[0-9]|NUM_DLA_SAMPLES=[0-9]*|submitted [0-9]* sbatch" | sort -u | tr '\n' ' '; echo
    OUT=$(env $STRIP bash slurm/greatlakes/production/launch_gl.sh $E --no-sleep 2>&1); IDS=$(echo "$OUT" | grep -o "Submitted batch job [0-9]*" | grep -o "[0-9]*$" | tr '\n' ' ')
    echo "MAX4-P1-ctrl_$N jobs($(echo $IDS | wc -w)): $IDS $(date -Is)" | tee -a $M/p1/P1_LAUNCH_RECORD.txt | cut -c1-160; done
  ;;
p1ctrl_reduce)
  # clustering control: score each arm (own truth; reference = the P0 candidate-free singles, as for the real pairs) then pair arms; mean-flux control: analyzer + paired gate vs the P1 random arm
  P=$M/p1; OUT=$P/reductions; mkdir -p $OUT; W=$M/cmpB/gate_weights.json; FIDPI=$M/fid_max4/analysis/analysis_fid_MAX4_per_injection.csv
  for arm in syscluster sysrandom sysshuffle; do R2=$P/clustering_control; [ -d $R2/${arm}_outputs ] || { echo "no outputs for $arm"; continue; }
    python tools/r041_multihcd_score.py --truth $R2/$arm/systems_truth.csv --outputs $R2/${arm}_outputs --reference $FIDPI --population $M/p1/mock_native/2lpt/population_native.csv --weights $W \
       --f-multi 0.155 0.476 --out $OUT/multihcd_ctrl_${arm}.json --label ctrl_${arm} 2>&1 | grep -v "UserWarning\|from scipy" | tail -3
    (cd $R2/${arm}_outputs && sha256sum dlacat-*.fits BASELINE.env > SHA256SUMS.txt); done
  python tools/r041_multihcd_pair_arms.py --a $OUT/multihcd_ctrl_sysrandom_units.csv --b $OUT/multihcd_ctrl_syscluster_units.csv --a-label sysrandom --b-label syscluster --weights $W --out $OUT/pair_arms_cluster_vs_random.json 2>&1 | grep -v "UserWarning\|from scipy"
  [ -f $OUT/multihcd_ctrl_sysshuffle_units.csv ] && python tools/r041_multihcd_pair_arms.py --a $OUT/multihcd_ctrl_sysshuffle_units.csv --b $OUT/multihcd_ctrl_syscluster_units.csv --a-label sysshuffle --b-label syscluster --weights $W --out $OUT/pair_arms_cluster_vs_shuffle.json 2>&1 | grep -v "UserWarning\|from scipy" | tail -4
  for v in turner2024_m1s turner2024_p1s ding2024_hz; do R2=$P/meanflux_control/$v; [ -d $R2/random_outputs ] || { echo "no outputs for $v"; continue; }
    python tools/r041_mock_truth_to_r041.py --truth-fits $R2/random/injection_truth.fits --out-truth $P/mock/truth_r041/mf_${v}_truth.csv --out-population $P/mock/truth_r041/mf_${v}_population.csv
    python tools/r041_analyze.py --truth $P/mock/truth_r041/mf_${v}_truth.csv --outputs $R2/random_outputs --population $P/mock/truth_r041/mf_${v}_population.csv --out $OUT/analysis_mfctrl_${v}.json --label mfctrl_${v} 2>&1 | grep -v "UserWarning\|from scipy" | tail -2
    python tools/r041_prescription_gate.py --a $OUT/analysis_mock_2lpt_random_MAX4_per_injection.csv --b $OUT/analysis_mfctrl_${v}_per_injection.csv --weights $W --out $OUT/gate_mfctrl_${v}_vs_fid.json --label mfctrl_${v} 2>&1 | grep -E "dC_w|tier|lost" | head -4
    (cd $R2/random_outputs && sha256sum dlacat-*.fits BASELINE.env > SHA256SUMS.txt); done
  (cd $OUT && sha256sum *.json *.csv > SHA256SUMS.txt)
  ;;
esac
echo "STAGE_DONE $1 $(date -Is)"
