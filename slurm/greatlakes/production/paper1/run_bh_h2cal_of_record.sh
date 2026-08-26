#!/usr/bin/env bash
# Paper-1 BH (high-z) product of record — the EXACT invocation that produced
#   /scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/tf_hz/track_c_tf_hz_h2cal_loa0_lya_gapc0.496.json
#   (sha256 90264a22...), ratified as ..._RATIFIED_20260826.json (sha256 62446b47...) by
#   CDDF_analysis/hbi/bh_ratify_stamp.py under PI rulings 2026-08-26 #43-#46.
# Reconstructed 2026-08-26 (Paper-1 code review) from the artifact's own metadata block:
# variant h2cal, fp loa0, window lya, envelope none, gap_treatment frozen, gap_c 0.496,
# zbins 3.8,4.25,4.5,5.0, n_mc 2000 (NOT the CLI default 120), seed 0 (fixed in the driver).
# Deterministic: rerunning to a scratch path reproduces the measurement block exactly
# (verified 2026-08-26; only wallclock_s/code_commit/argv differ).
# Environment: gpdla (jax-free). Inputs (frozen, hashed in docs/PAPER1_FROZEN_MANIFEST.json):
#   forward_response_2lpt0.npz, molly_matrix.tsv (nhi195 lya_only), gl_cddf_loa_hz_v1_20260813 outputs,
#   h2_exec canonical tables (C_gap 0.496 from CDDF_analysis/hbi/h2_cgap_inference.py).
set -euo pipefail
OUT=${1:?usage: run_bh_h2cal_of_record.sh <out.json>}
cd "$(dirname "$0")/../../../.."
python CDDF_analysis/hbi/track_c_tf_hz.py --variant h2cal --fp loa0 --window lya --envelope none \
  --gap-treatment frozen --gap-c 0.496 --zbins 3.8,4.25,4.5,5.0 --n-mc 2000 --out-json "$OUT"
