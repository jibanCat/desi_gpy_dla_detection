#!/usr/bin/env python
"""registry.py -- the declared content of the Paper-lane handoff.

This is a DATA module: it names, once, every artifact the handoff carries or
points at, with its public/private classification.  The manifest builder and
its ``--verify`` mode both read this table, so there is exactly one place where
"what is in the handoff" is written down.

READ-ONLY over the Science lane.  Nothing here writes to a frozen product.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------- path tokens
H = "/home/mfho/desi_gpy_dla_notes/governance/paper_lane_handoff_2026-09-16"
N = "/home/mfho/desi_gpy_dla_notes"
G = N + "/governance/final_campaign_2026-09-13"
GOV = N + "/governance"
R = "/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13"
T = "/nfs/turbo/lsa-cavestru/mfho/paper1_science_handoff/ABSORBER_LADDER_2026-09-13"
CODE = "/home/mfho/wt_abs_diag_2026-09"

PRIVATE = "PRIVATE"          # carries real-survey values; notes repo only
PUBLIC = "PUBLIC_SAFE"       # mock / calibration only; releasable

HANDOFF_DATE = "2026-09-16"

ARCHIVAL_TAGS = [
    "paper1-lowz-model-freeze-2026-09-14",
    "paper1-lowz-science-closure-2026-09-15",
    "paper1-lowz-paper-handoff-2026-09-16",
]

# Files written into H by OTHER agents (the claim-ledger agent and the
# commander).  They are enumerated so the manifest can report them as present
# or PENDING; they are never authored or edited here.
EXPECTED_FROM_OTHER_AGENTS = [
    ("START_HERE.md", "commander"),
    ("SCIENCE_LANE_BOUNDARY.md", "commander"),
    ("CLAIM_LEDGER.md", "claim-ledger agent"),
    ("CLAIM_LEDGER.json", "claim-ledger agent"),
    ("LITERATURE_COMPARISON_DEFINITIONS.md", "claim-ledger agent"),
    ("ESTIMAND_NOTATION_CONTRACT.md", "claim-ledger agent"),
    ("OPEN_ITEMS_REGISTRY.md", "claim-ledger agent"),
]

# ------------------------------------------------------- curated figures in H
# (handoff relative path, canonical source absolute path, classification, note)
FIGURES = [
    ("figures/fig_sec3_real_reporting_bins.png",
     N + "/figures/2026-09-14_pi_inspection/fig_sec3_real_reporting_bins.png",
     PRIVATE, "F2 preview: real pooled 0.2-dex reporting bins"),
    ("figures/fig_sec4_real_z_bins_and_transfer.png",
     N + "/figures/2026-09-14_pi_inspection/fig_sec4_real_z_bins_and_transfer.png",
     PRIVATE, "F1 preview: real Paper-1 z bins + signed mock transfer residuals"),
    ("figures/fig_sec2_old_vs_new_by_N.png",
     N + "/figures/2026-09-14_pi_inspection/fig_sec2_old_vs_new_by_N.png",
     PRIVATE, "F4 preview: old vs new C1 per 0.2-dex bin"),
    ("figures/fig_sec2b_old_vs_new_by_z.png",
     N + "/figures/2026-09-14_pi_inspection/fig_sec2b_old_vs_new_by_z.png",
     PRIVATE, "F4 preview: old vs new C1 per Paper-1 z bin"),
    ("figures/fig_sec8_lambda_tK_ridge.png",
     N + "/figures/2026-09-14_pi_inspection/fig_sec8_lambda_tK_ridge.png",
     PRIVATE, "F8 preview: Lambda-t_K compensation ridge (real panel)"),
    ("figures/fig_sec6_response_rows.png",
     N + "/figures/2026-09-14_pi_inspection/fig_sec6_response_rows.png",
     PUBLIC, "F5 preview: conditional response rows Q(Nhat|N,s,K) + phi"),
    ("figures/fig_sec5_completeness_composite.png",
     N + "/figures/2026-09-14_pi_inspection/fig_sec5_completeness_composite.png",
     PUBLIC, "F6 preview: completeness C(N_HI, S/N) composite"),
    ("figures/fig_sec9_fp_template_and_a0.png",
     N + "/figures/2026-09-14_pi_inspection/fig_sec9_fp_template_and_a0.png",
     PUBLIC, "F7 preview: FP template (89 loa-0 events) + a0 battery"),
    ("figures/dash_systematics.png",
     N + "/figures/2026-09-14_pi_inspection/dash_systematics.png",
     PRIVATE, "T2 dashboard; carries real half-width units -> PRIVATE"),
    ("figures/dash_sampler_health.png",
     N + "/figures/2026-09-14_pi_inspection/dash_sampler_health.png",
     PRIVATE, "F11 preview: E-BFMI + per-run headlines in real hw68 units"),
    ("figures/wg_fig1_snr_predictive_real_vs_mock.png",
     N + "/figures/2026-09-15_wg_diagnostic/wg_fig1_snr_predictive_real_vs_mock.png",
     PUBLIC, "F9 preview: predictive mu/observed RATIOS only (real vs 3 mocks)"),
    ("figures/final_ALL_fig1_headline_bias_by_variant.png",
     N + "/figures/2026-09-13_final_campaign/final_ALL_fig1_headline_bias_by_variant.png",
     PUBLIC, "T5/F12 preview: mock headline bias by variant (mock only)"),
]

# -------------------------------------------------------- curated tables in H
TABLES = [
    ("tables/REAL_C1_BLIND_RESULT_2026-09-14.md",
     G + "/REAL_C1_BLIND_RESULT_2026-09-14.md", PRIVATE,
     "result of record (human-readable)"),
    ("tables/REAL_C1_BLIND_RESULT_2026-09-14.json",
     G + "/REAL_C1_BLIND_RESULT_2026-09-14.json", PRIVATE,
     "result of record (machine-readable; == real_c1/REAL_C1_POOLED.json)"),
    ("tables/SYSTEMATICS_TABLE.md",
     R + "/release/systematics/SYSTEMATICS_TABLE.md", PUBLIC,
     "final systematics table S1-S8 (mock units)"),
    ("tables/SYSTEMATICS_TABLE.json",
     R + "/release/systematics/SYSTEMATICS_TABLE.json", PUBLIC,
     "final systematics table S1-S8 (mock units)"),
    ("tables/SYSTEMATICS_TABLE.csv",
     R + "/release/systematics/SYSTEMATICS_TABLE.csv", PUBLIC,
     "final systematics table S1-S8 (mock units)"),
    ("tables/SYSTEMATICS_TABLE_REAL_HW.md",
     G + "/SYSTEMATICS_TABLE_REAL_HW.md", PRIVATE,
     "same eight effects in REAL pooled half-width units"),
    ("tables/SYSTEMATICS_TABLE_REAL_HW.json",
     G + "/SYSTEMATICS_TABLE_REAL_HW.json", PRIVATE,
     "same eight effects in REAL pooled half-width units"),
    ("tables/S6_COMPLETENESS_PROPAGATION_RESULT.md",
     G + "/S6_COMPLETENESS_PROPAGATION_RESULT.md", PRIVATE,
     "S6 result of record"),
    ("tables/S6_COMPLETENESS_PROPAGATION_RESULT.json",
     G + "/S6_COMPLETENESS_PROPAGATION_RESULT.json", PRIVATE,
     "S6 result of record"),
    ("tables/S6_RELEASE_SAFE_SUMMARY.md",
     R + "/release/systematics/S6_RELEASE_SAFE_SUMMARY.md", PUBLIC,
     "S6 sizes only (fractions and hw68 units); READY_FOR_RELEASE_PENDING_PI"),
    ("tables/B5_PRESENTATION_TABLE.md",
     G + "/B5_PRESENTATION_TABLE.md", PRIVATE, "B5 presentation of record"),
    ("tables/B5_PRESENTATION_TABLE.json",
     G + "/B5_PRESENTATION_TABLE.json", PRIVATE, "B5 presentation of record"),
    ("tables/J8_CERTIFICATION.md",
     R + "/final/J8_CERTIFICATION.md", PUBLIC,
     "J = 8 mock certification of record (post-cleanup; NOT the .pre copy)"),
    ("tables/A0_BATTERY_TABLE.md",
     G + "/A0_BATTERY_TABLE.md", PUBLIC, "S4 a0 factor-4 bracket (mock)"),
    ("tables/F1_PAPER1_ZBIN_TABLE.md",
     G + "/F1_PAPER1_ZBIN_TABLE.md", PUBLIC,
     "first-ladder ORACLE per-bin residuals, history-annotated; J = 8 supersedes"),
    ("tables/MODEL_OF_RECORD.json",
     R + "/release/model_of_record/MODEL_OF_RECORD.json", PUBLIC,
     "31 frozen objects with sha256; fail-closed"),
]

# -------------------------------- canonical sources the handoff MAPS point at
# (key, path, classification, role)
CANONICAL_SOURCES = [
    ("result_of_record.md", G + "/REAL_C1_BLIND_RESULT_2026-09-14.md", PRIVATE,
     "blind real C1 result of record"),
    ("result_of_record.json", G + "/REAL_C1_BLIND_RESULT_2026-09-14.json", PRIVATE,
     "blind real C1 result of record (machine-readable)"),
    ("result_pooled_scratch.json", R + "/real_c1/REAL_C1_POOLED.json", PRIVATE,
     "producer-side pooled read-out (byte-identical to the notes .json)"),
    ("systematics.release.md", R + "/release/systematics/SYSTEMATICS_TABLE.md", PUBLIC,
     "final systematics table, mock units"),
    ("systematics.release.json", R + "/release/systematics/SYSTEMATICS_TABLE.json", PUBLIC,
     "final systematics table, mock units"),
    ("systematics.release.csv", R + "/release/systematics/SYSTEMATICS_TABLE.csv", PUBLIC,
     "final systematics table, mock units"),
    ("systematics.real_hw.md", G + "/SYSTEMATICS_TABLE_REAL_HW.md", PRIVATE,
     "systematics in real pooled half-width units"),
    ("systematics.real_hw.json", G + "/SYSTEMATICS_TABLE_REAL_HW.json", PRIVATE,
     "systematics in real pooled half-width units"),
    ("s6.result.md", G + "/S6_COMPLETENESS_PROPAGATION_RESULT.md", PRIVATE,
     "S6 completeness calibration covariance propagation"),
    ("s6.result.json", G + "/S6_COMPLETENESS_PROPAGATION_RESULT.json", PRIVATE,
     "S6 completeness calibration covariance propagation"),
    ("s6.release_safe.md", R + "/release/systematics/S6_RELEASE_SAFE_SUMMARY.md", PUBLIC,
     "S6 release-safe summary (sizes only)"),
    ("s6.predeclaration.md",
     G + "/S6_COMPLETENESS_COVARIANCE_PROPAGATION_PREDECLARATION.md", PUBLIC,
     "sealed S6 predeclaration"),
    ("b5.md", G + "/B5_PRESENTATION_TABLE.md", PRIVATE, "B5 presentation table"),
    ("b5.json", G + "/B5_PRESENTATION_TABLE.json", PRIVATE, "B5 presentation table"),
    ("architecture.model_selection", G + "/PAPER_LANE_HBI_MODEL_SELECTION_AND_FINAL_ARCHITECTURE.md",
     PRIVATE, "model selection + final architecture memo"),
    ("architecture.mathematics", G + "/FINAL_HBI_RESPONSE_COMPLETENESS_AND_FP_MATHEMATICS.md",
     PUBLIC, "response / completeness / FP mathematics of record"),
    ("model_of_record.json", R + "/release/model_of_record/MODEL_OF_RECORD.json", PUBLIC,
     "31 frozen objects, hashed, fail-closed"),
    ("freeze_record", G + "/PAPER1_LOWZ_MODEL_FREEZE_2026-09-14.md", PUBLIC,
     "model freeze record"),
    ("methodological_history", G + "/METHODOLOGICAL_HISTORY_OF_RECORD.md", PUBLIC,
     "corrected ladder of ~30 representations"),
    ("mock_cert.md", R + "/final/J8_CERTIFICATION.md", PUBLIC,
     "J = 8 certification of record"),
    ("mock_cert.json", R + "/final/J8_CERTIFICATION.json", PUBLIC,
     "J = 8 certification of record"),
    ("mock_cert.predeclaration", G + "/J8_CERTIFICATION_PREDECLARATION.md", PUBLIC,
     "sealed J = 8 certification rule (6acf7508)"),
    ("pi_ruling.freeze", GOV + "/PI_RULING_2026-09-14_MODEL_FREEZE_ADOPT_B.md", PUBLIC,
     "freeze / adopt-B ruling"),
    ("pi_ruling.closure", GOV + "/PI_RULING_2026-09-14b_FINAL_CLOSURE_RULINGS.md", PUBLIC,
     "final closure rulings (22 items)"),
    ("read_only_declaration", G + "/SCIENCE_LANE_READ_ONLY_2026-09-16.md", PUBLIC,
     "Science lane CLOSED / READ-ONLY declaration"),
    ("closure_package", G + "/SCIENCE_LANE_CLOSURE_PACKAGE_2026-09-15.md", PRIVATE,
     "the 22-item closure package"),
    ("pi_summary", G + "/PI_SUMMARY_PACKET_2026-09-15.md", PRIVATE,
     "PI summary packet"),
    ("figure_architecture", G + "/PAPER1_FIGURE_ARCHITECTURE_PROPOSAL.md", PUBLIC,
     "approved F1-F12 / T1-T6 architecture proposal"),
    ("wg_note_draft", G + "/LYA_WG_SN_PREDICTIVE_DIAGNOSTIC_NOTE_DRAFT.md", PUBLIC,
     "Lya-WG S/N diagnostic note draft (ratios only) - parallel diagnostic product"),
    ("wg_internal_appendix", G + "/LYA_WG_SN_DIAGNOSTIC_INTERNAL_APPENDIX.md", PRIVATE,
     "WG diagnostic internal appendix T1-T4 - parallel diagnostic product"),
    ("provenance.closure", G + "/PROVENANCE_MANIFEST_CLOSURE_20260915T031406Z.json", PUBLIC,
     "provenance manifest of record (1,174 nodes / 69 runs)"),
    ("provenance.release", R + "/release/PROVENANCE_MANIFEST.json", PUBLIC,
     "released provenance manifest (same content as the closure manifest)"),
    ("provenance.realc1", G + "/PROVENANCE_MANIFEST_REALC1.json", PRIVATE,
     "provenance manifest of the 16 blind real-C1 runs"),
    ("zenodo_manifest", G + "/ZENODO_RELEASE_MANIFEST.md", PUBLIC,
     "release manifest; licence PENDING PI"),
    ("cleanup_checklist", G + "/DOCUMENTATION_CLEANUP_CHECKLIST_2026-09-15.md", PRIVATE,
     "documentation / provenance cleanup checklist"),
    ("a0_battery.md", G + "/A0_BATTERY_TABLE.md", PUBLIC, "a0 factor-4 bracket"),
    ("f1_zbin_table", G + "/F1_PAPER1_ZBIN_TABLE.md", PUBLIC,
     "first-ladder ORACLE per-bin residuals (history-annotated)"),
    ("release.readme", R + "/release/README.md", PUBLIC, "release package README"),
    ("release.licence_pending", R + "/release/LICENCE_PENDING_PI.md", PUBLIC,
     "explicit licence blocker"),
    ("real_pack_of_record", R + "/real_c1_inputs/C1_pack.npz", PRIVATE,
     "the real pack of record (row support_id 57eb8431)"),
    ("real_pack_provenance", R + "/real_c1_inputs/C1_pack.provenance.json", PRIVATE,
     "real pack provenance"),
    ("real_pack_fixed_objects", R + "/real_c1_inputs/FIXED_OBJECTS_MANIFEST.json", PRIVATE,
     "the fixed objects the real run was pinned to"),
]

# -------------------------------------------------- SUPERSEDED / NOT FOR USE
SUPERSEDED = [
    (R + "/final/J8_CERTIFICATION.md.pre20260915",
     "pre-cleanup J8 certification: Omega column was the sub-DLA Omega[19.5,20.3]",
     R + "/final/J8_CERTIFICATION.md"),
    (R + "/final/J8_CERTIFICATION.json.pre20260915",
     "pre-cleanup J8 certification (same defect)",
     R + "/final/J8_CERTIFICATION.json"),
    (G + "/PROVENANCE_MANIFEST_FINAL_20260914T042140Z.json",
     "superseded provenance manifest (pre J=8 production / pre real C1)",
     G + "/PROVENANCE_MANIFEST_CLOSURE_20260915T031406Z.json"),
    (G + "/PROVENANCE_MANIFEST_FROZEN_20260914T064113Z.json",
     "superseded provenance manifest (pre real C1)",
     G + "/PROVENANCE_MANIFEST_CLOSURE_20260915T031406Z.json"),
    (G + "/PROVENANCE_MANIFEST_CLEANUP_20260915T030130Z.json",
     "intermediate cleanup manifest; the CLOSURE manifest is the one of record",
     G + "/PROVENANCE_MANIFEST_CLOSURE_20260915T031406Z.json"),
    (G + "/PROVENANCE_MANIFEST_J8_20260914T075529Z.json",
     "superseded provenance manifest (J=8 stage only)",
     G + "/PROVENANCE_MANIFEST_CLOSURE_20260915T031406Z.json"),
    (G + "/PROVENANCE_MANIFEST_20260914T032716Z.json",
     "superseded provenance manifest (first-ladder stage)",
     G + "/PROVENANCE_MANIFEST_CLOSURE_20260915T031406Z.json"),
    (G + "/PROVENANCE_MANIFEST_20260914T035432Z.json",
     "superseded provenance manifest (first-ladder stage)",
     G + "/PROVENANCE_MANIFEST_CLOSURE_20260915T031406Z.json"),
    (R + "/release_backup_2026-09-15",
     "pre-cleanup copy of the release tree (systematics J=1 rows, stale README)",
     R + "/release"),
    (G + "/VARIANT_TABLE_F1.md",
     "first-ladder F1 variant table: NOT a first-ladder document of record and "
     "not the final variant set; superseded by VARIANT_TABLE_FINAL + J8 certification",
     G + "/VARIANT_TABLE_FINAL.md"),
    (G + "/VARIANT_TABLE_F1.json", "as above", G + "/VARIANT_TABLE_FINAL.json"),
    (R + "/VARIANT_TABLE.md",
     "first absorber-ladder variant table (pre response-family selection)",
     G + "/VARIANT_TABLE_FINAL.md"),
    (R + "/VARIANT_TABLE.json", "as above", G + "/VARIANT_TABLE_FINAL.json"),
    ("/nfs/turbo/lsa-cavestru/mfho/paper1_science_handoff/"
     "LOWZ_CLEAN_C1_DURABLE_CANDIDATE_2026-09-11/reductions/reduced_CLEAN_C1_pooled.json",
     "the OLD C1 pool (sha8 ac61ea07): historical comparison only (F4). "
     "Never a result of record and never a number to quote.",
     G + "/REAL_C1_BLIND_RESULT_2026-09-14.json"),
]

# Superseded CLASSES that are not single files (recorded as notes, not hashes).
SUPERSEDED_CLASSES = [
    ("S1 J = 1 systematics rows",
     "the released SYSTEMATICS_TABLE keeps the J = 1 arm as `M1CUT_J1_HISTORY`; "
     "it differs from production J = 8 by up to 1.4 pp at B5. Quote J = 8 only.",
     R + "/release/systematics/SYSTEMATICS_TABLE.md (S1 production J=8 rows)"),
    ("pi_inspection_sections/SEC_*.md drafts",
     "section drafts assembled into PI_INSPECTION_PACKET_POST_REALC1_2026-09-14.md. "
     "They are working read-outs, not documents of record; SEC_11b and SEC_12 are "
     "cited by the closure package for specific corrections and may be read as "
     "evidence, but no SEC_* file is a canonical source for a paper number.",
     G + "/PI_INSPECTION_PACKET_POST_REALC1_2026-09-14.md"),
    ("RESPONSE_REVIEW_RETURN: E-invariance claim",
     "the E-invariance claim in the response-review return was WITHDRAWN: E's "
     "coefficients reproduce its delivered tensor only to 1.2e-2 (a different "
     "local optimum; the original fit came from a dirty tree). The frozen E "
     "TENSOR, not the coefficient vector, is the object of record.",
     G + "/ZENODO_RELEASE_MANIFEST.md section 1.2 + "
     + N + "/governance/response_review_2026-09-13/"),
    ("real_c1 raw chains and f-draw NPZ",
     "not carried in the handoff by design (no raw chains, no NPZ draws in H). "
     "They live in the archive; the Paper lane redraws from the pooled JSON.",
     R + "/real_c1/runs/"),
]

# ------------------------------------------------------------------- archives
ARCHIVES = [
    ("turbo_forensic_archive", T, T + "/SHA256SUMS", "durable (Turbo)"),
    ("scratch_working_root", R, R + "/release/SHA256SUMS", "purgeable (scratch)"),
]


def exists(path):
    return os.path.exists(path)
