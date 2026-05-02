#!/bin/bash
# Run check_tau_eb_robust_mask on a representative spread of targets.
# Captures the summary table per target; writes to a TSV master.
set -e
cd /home/mfho/desi_gpy_dla_detection
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpdla
export LD_LIBRARY_PATH="$HOME/.local/usr/local/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

OUT=/tmp/hcd_mask_multi_target.log
> "$OUT"

# (mock, tid, z_qso, truth_z, truth_logNHI, spec, zcat)
declare -a targets=(
  "2lpt|120046865|2.962|2.7730|21.263|/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/spectra-16/7/789/spectra-16-789.fits|/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/zcat.fits"
  "london|50129689|2.417|2.156|21.280|/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/london/qq_desi_y3/v5.9.5/mock-0/jura-124/spectra-16/11/1111/spectra-16-1111.fits|/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/london/qq_desi_y3/v5.9.5/mock-0/jura-124/zcat.fits"
  "2lpt|50068236|2.464|2.155|20.339|/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/spectra-16/3/340/spectra-16-340.fits|/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/zcat.fits"
  "saclay|2229000465|2.231|2.026|20.757|/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124/spectra-16/16/1640/spectra-16-1640.fits|/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124/zcat.fits"
  "saclay|2385001246|2.690|2.297|19.557|/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124/spectra-16/5/546/spectra-16-546.fits|/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124/zcat.fits"
  "2lpt|260170003|2.653|2.317|17.844|/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/spectra-16/16/1682/spectra-16-1682.fits|/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/zcat.fits"
)

for entry in "${targets[@]}"; do
  IFS='|' read -r mock tid z_qso truth_z truth_n spec zcat <<< "$entry"
  echo "" >> "$OUT"
  echo "==================================================================" >> "$OUT"
  echo "[target] mock=$mock tid=$tid z_qso=$z_qso truth_z=$truth_z truth_logNHI=$truth_n" >> "$OUT"
  echo "==================================================================" >> "$OUT"
  python -u examples/check_tau_eb_robust_mask.py \
      --target-id "$tid" \
      --spec "$spec" \
      --zcat "$zcat" \
      --truth-z "$truth_z" --truth-log-nhi "$truth_n" \
      --tau-factors 0.5 0.75 1.0 1.25 1.5 2.0 \
      --mask-threshold-sigma 1.5 \
      2>&1 | grep -vE "^INFO|^ERROR|^DESI|^WARNING" | tee -a "$OUT" || \
      echo "  ERROR: target $tid failed" | tee -a "$OUT"
done
echo "" >> "$OUT"
echo "ALL_DONE" >> "$OUT"
