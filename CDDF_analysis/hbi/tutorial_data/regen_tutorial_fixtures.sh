#!/usr/bin/env bash
# Refresh the small mock-only tutorial fixtures by COPYING from the scratch
# caches. Not a recompute -- provenance is the 2LPT-0 injection validation run.
# Mock data only; this script NEVER touches a real-survey (LOA) result file.
set -euo pipefail
B=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata
V="$B/hbi_validation_2lpt0"
S0="$B/track_c/stage0"
DST="$(cd "$(dirname "$0")" && pwd)"
NOTES=/home/mfho/desi_gpy_dla_notes/notes/2026-06-24_hbi_cddf_draft/scripts
cp "$V/figures/compare_synthesis.json"            "$DST/compare_synthesis.json"
cp "$V/figures/compare_R0_table.md"               "$DST/compare_R0_table.md"
cp "$S0/znz_2lpt0.npz"                            "$DST/znz_2lpt0.npz"
cp "$S0/forward_response_2lpt0.npz"               "$DST/forward_response_2lpt0.npz"
cp "$V/hbi/pm_verify_canonical/phase3d_v3_point_kernel.npz" "$DST/point_kernel_pm.npz"
cp "$V/hbi/loa0_run/phase3d_v3_point_kernel.npz"  "$DST/point_kernel_loa0.npz"
cp "$V/stage2/stage2_band_compare.npz"            "$DST/stage2_band_compare.npz"
cp /scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/figures_molly_nhi195/lya_only/molly_matrix.tsv "$DST/molly_matrix_nhi195_lyaonly.tsv"
cp /scratch/cavestru_root/cavestru0/mfho/gl_loa0_fp_v1_20260615/outputs/loa0_fp_product_lyaonly1025.npz "$DST/loa0_fp_product_lyaonly1025.npz"
cp "$NOTES/recovery_mock_data.csv"                "$DST/recovery_mock_data.csv"
cp "$NOTES/r0_decompose_cache/h4_zstructure.tsv"  "$DST/h4_zstructure.tsv"
cp "$V/rawff/rawff_results.json"                  "$DST/rawff_results.json"
cp "$V/rawff/rawff_zresolved.json"                "$DST/rawff_zresolved.json"
cp "$B/null_field/hbi_out/null_field_hbi_results.json" "$DST/null_field_hbi_results.json"
cp "$B/sbc/sbc_realizations.npz"                  "$DST/sbc_realizations.npz"
cp /scratch/cavestru_root/cavestru0/mfho/wall1_inject/reduce_out/wall1_full_injection_baseline_divided.tsv "$DST/wall1_full_injection_baseline_divided.tsv"
cp "$B/campaignS_2lpt0/competed_completeness.npz" "$DST/competed_completeness.npz"
echo "refreshed mock-only tutorial fixtures in $DST"
