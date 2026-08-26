#!/bin/bash
#SBATCH --job-name=cp2_collect
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v2_20260821/cp2_validation/cp2_collect_%j.log
# PAPER-1 PRODUCT 2 OF RECORD (CP-2 collector). Committed copy 2026-08-26; the inline bit-reproduction
# heredoc of the archived script is replaced by the committed routine CDDF_analysis.hbi_mcmc.bitrepro_check
# (fails closed on a missing reference; the archived heredoc silently fell through to a literal path).
# CP-2 collector: per-z gate on the production-pack runs + bit-reproduction vs the
# Battery-2/3 DIAGPACK records (same family, same seed) + the full certification
# record with check (d). Decides nothing beyond PASS/FAIL.
set -uo pipefail
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh
conda activate gpdla
export OMP_NUM_THREADS=1
cd /home/mfho/wt_forward_2026_08
BASE=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata
NEW=$BASE/real_pack_v2_20260821; OUT=$NEW/cp2_validation
V2NEW=$BASE/adopted_packs_v2p2_20260821; OLDV2=$BASE/adopted_packs_v2p1_20260817
OLDREAL=$BASE/real_pack_v1; DIAG=$OLDREAL/diag_20260820_gsupport
B23=/gpfs/accounts/cavestru_root/cavestru0/mfho/h2m_ckpt10p5_20260817/analysis/perz_20260820
TAG=bw0p2_pad19p0_molly172
echo "=== gate v2 on the production-pack runs"
python -m CDDF_analysis.hbi_mcmc.perz_gate $OUT/cp2_ln_w1500_*_s20260811.json --out $OUT/perz_gate_v2_cp2_production.json | tee $OUT/GATE_cp2_production.txt
echo "=== bit-reproduction vs Battery-2/3 (DIAGPACK) records"
B23=/gpfs/accounts/cavestru_root/cavestru0/mfho/h2m_ckpt10p5_20260817/analysis/perz_20260820
python -m CDDF_analysis.hbi_mcmc.bitrepro_check --new-dir "$OUT" --ref-dir "$B23" --out "$OUT/bitrepro_cp2_vs_battery23.json" | tee "$OUT/BITREPRO_cp2_vs_battery23.txt"
echo "=== certification record with check (d)"
python CDDF_analysis/hbi_mcmc/certify_g_support.py \
  --new $V2NEW/modelA_pack_2lpt0_${TAG}_v2.npz $V2NEW/modelA_pack_london0_${TAG}_v2.npz $V2NEW/modelA_pack_saclay0_${TAG}_v2.npz \
        $NEW/scanpack_2lpt0_b300.npz $NEW/scanpack_london0_b300.npz $NEW/scanpack_saclay0_b300.npz \
        $NEW/modelA_pack_REAL_loa50k_c3300_${TAG}_v2.npz \
  --old $OLDV2/modelA_pack_2lpt0_${TAG}_v2.npz $OLDV2/modelA_pack_london0_${TAG}_v2.npz $OLDV2/modelA_pack_saclay0_${TAG}_v2.npz \
        $OLDREAL/scanpack_2lpt0_b300.npz $OLDREAL/scanpack_london0_b300.npz $OLDREAL/scanpack_saclay0_b300.npz \
        $OLDREAL/modelA_pack_REAL_loa50k_c3300_${TAG}_v2.npz \
  --diag $DIAG/scanpack_2lpt0_b300_DIAGPACK_gcons.npz $DIAG/scanpack_london0_b300_DIAGPACK_gcons.npz $DIAG/scanpack_saclay0_b300_DIAGPACK_gcons.npz \
  --validation-json $OUT/cp2_ln_w1500_2lpt0_s20260811.json $OUT/cp2_ln_w1500_london0_s20260811.json $OUT/cp2_ln_w1500_saclay0_s20260811.json \
  --calib-index 0 --floor 19.0 --tol-rows 2 --out $NEW/CERT_G_SUPPORT_CP1_CP2.json
echo "CP2_COLLECT_DONE $(date)"
