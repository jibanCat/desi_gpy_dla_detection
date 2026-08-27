# Recovery record: builder of adopted_response_v1.npz / adopted_response_v1p1.npz
Recovered 2026-08-26 (read-only forensic pass). All scripts below were reconstructed by replaying the
Write/Edit/heredoc tool calls in ONE Claude Code transcript and cross-checked byte-for-byte against the
copies the session itself committed to the notes repo.

## Source of truth
* Transcript: /home/mfho/.claude/projects/-home-mfho-desi-gpy-dla-detection/2ff9accb-47ff-421e-b947-0323e5b9a8c7.jsonl
  (session 2026-08-15 .. 2026-08-17 00:24 EDT; Claude-Session https://claude.ai/code/session_01GHxkaXdMCiKHBZwRvsqDb8)
* Session scratchpad (DELETED, not on disk any more):
  /tmp/claude-114399728/-home-mfho-desi-gpy-dla-detection/2ff9accb-47ff-421e-b947-0323e5b9a8c7/scratchpad/pi_diag/
* Surviving verbatim copies (committed by the session): /home/mfho/desi_gpy_dla_notes/figures/2026-08-17_stilt_diag/inputs/pi_diag/
  notes commits f5a1cb9 (2026-08-16 01:35 EDT) and b3ff5db (2026-08-16 12:33 EDT); later cp at transcript L2478 (2026-08-16 18:07 EDT).

## Files in ./recovered  (sha256 in ../SHA256SUMS.txt)
| file | origin (transcript line, UTC) | verified against |
|---|---|---|
| stage1b_events_full.py | Write L603 2026-08-16T05:21:20Z | notes copy IDENTICAL |
| fitlib.py (FINAL, weights+fixed_bins) | Write L619 05:22:28Z + Edit L633/L635 + python patch L1074 16:18:17Z | notes copy = pre-L1074 version (identical up to the patch); patch replayed |
| fitlib__preL1074_notesCopy.py | notes copy (01:35 EDT) | as used by run_d2b.py / v1_logo.py (weights path unused there) |
| fitlib_patch__L1074_heredoc.py | Bash heredoc L1074 | verbatim |
| run_d2b.py | Write L736 05:29:31Z | notes copy IDENTICAL |
| run_d2b_lib.py | Write L778 05:31:39Z | notes copy IDENTICAL |
| v1_logo.py | Write L882 06:06:25Z | notes copy IDENTICAL |
| boot_carrier.py (FINAL) | Write L1085 16:18:58Z + Edit L1113 16:20:00Z + Edit L1115 16:20:01Z | notes copy IDENTICAL |
| boot_carrier__asWritten_L1085_preEdit.py | Write L1085 | first run crashed: default_rng(-1) ValueError |
| gb_audit.py | Write L2067 20:47:02Z | notes copy IDENTICAL |
| chain_recert__L2183.sh | Bash heredoc L2183 21:11:32Z | verbatim |
| assemble_v1__L1214_heredoc.py | Bash heredoc L1214 16:32:33Z | verbatim; result L1220: "wrote .../adopted_response_v1.npz commit 74762e93" |
| assemble_v1p1__L2299_heredoc.py | Bash heredoc L2299 21:24:34Z | verbatim; result L2300: "wrote adopted_response_v1p1.npz commit 933b0bb9" |
| cc_audit.py, latent_study.py, final_packet.py, run_d1_d2.py, run_d4*.py, run_d5.py, fold_variants.py, v3_bump.py, make_figs*.py, upperbound_audit.py | same session | context only (not on the v1/v1p1 path) |

## Execution chain -> adopted_response_v1.npz (12:32:35 EDT 2026-08-16, sha256 f3841983...)
All runs: cwd /home/mfho/wt_forward_2026_08, env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1,
interpreter /home/mfho/.conda/envs/gpdla/bin/python (login node, foreground/background tasks, no SLURM).
1. L611 05:21Z (01:21 EDT)  python $SP/pi_diag/stage1b_events_full.py  -> events_full.npz   [HEAD 475c62b]
   output (L624): 769,833 catalog rows; op-cut detections 495,553; keys N_ref,nhi_tilt_host,nhi_true,point_*,snr,snr_edges,tid,xhat,z_edges,zdla,zqso
2. L744 05:29Z (01:29 EDT)  python $SP/pi_diag/run_d2b.py  -> d2b_variants.npz (ml_shared3__mu/sig/skew/rng, N_ref), d2b_results.json   [HEAD 475c62b]
   PROOF: notes copy figures/2026-08-17_stilt_diag/inputs/pi_diag/d2b_variants.npz (sha256 7b996935...) has
   ml_shared3__{mu,sig,skew,rng} EXACTLY equal (max|delta| = 0.0) to v1 mu_coef/sig_coef/skew_coef/fit_rng.
3. L890 06:06Z              python $SP/pi_diag/v1_logo.py -> v1_logo_results.json (15/15 positive folds)
4. L1074 16:18Z             fitlib.py patched (weights, fixed_bins)     [HEAD 1812fef; uncommitted count_conserving_fold.py in tree]
5. L1118 16:20Z (12:20 EDT) python $SP/pi_diag/boot_carrier.py (background task b1kjcayw6) -> adopted_carrier_ensemble.npz
   output (L1227): events 73845, uniq TIDs 65011, anchors/cell [22,24,24,24,24,24,23,24,24];
   unit gate (mu,sig,skew) = 1.48355891e-06 1.43901386e-06 4.41372280e-06; shared3 mean/sd [-0.03634,0.00447,-0.0582]/[0.0081,0.00143,0.01389]
6. L1203 16:26Z             git commit 74762e9 (count_conserving_fold.py + contract_guards_check.py)
7. L1214 16:32Z (12:32 EDT) plain `python - <<EOF` (cwd /home/mfho/desi_gpy_dla_detection, NO env prefix): merges d2b_variants.npz + adopted_carrier_ensemble.npz
   -> /scratch/.../track_c/stage0/adopted_response_v1.npz ; code_commit = git rev-parse HEAD of wt_forward = 74762e93f60ccfed671d4a49b1523e36b857187d
   NOTE: the tree was CLEAN of relevant changes at that moment; but the builder scripts themselves were never committed anywhere.

## Execution chain -> adopted_response_v1p1.npz (17:24:37 EDT 2026-08-16, sha256 8fb580b5...)
1. L2164 21:09Z  PI ruling: fix G-B with the hierarchical matcher.
2. L2178 21:11:00Z  Edit /home/mfho/wt_forward_2026_08/CDDF_analysis/hbi/cddf_catalog_hbi.py (load_and_cut_catalog tilt block; meta["tilt_match_mode"]="hierarchical_v2_20260817")
   -- UNCOMMITTED when the chain ran; committed 17:55 EDT as 0ecfeea. `git diff 933b0bb9 0ecfeea -- cddf_catalog_hbi.py` == this edit; block unchanged at HEAD 9113c58.
3. L2183 21:11:32Z  cp events_full.npz events_full_prehier.npz; write + run chain_recert.sh (background task bo4f2amcb), HEAD = 933b0bb9 (+ the uncommitted edit):
      stage1b_events_full.py -> run_d2b.py -> v1_logo.py -> boot_carrier.py (gpdla) -> gb_audit.py (gpdla-hbi)
   output (L2293): CV sample_deg2 0.40419 / ml_deg2 0.47472 / ml_deg2_shared3 0.48888 / ml_deg2_shared4 0.48471 per event;
   LOGO 15/15 (mu3 range -0.0404..-0.0209); carrier: events 73845, uniq TIDs 65011, gate {mu 1.484e-06, sig 1.437e-06, skew 4.416e-06},
   shared3 mean/sd [-0.0364,0.00451,-0.059]/[0.00809,0.00143,0.01381]; G-B B1 PASS (56 cells), B2 PASS (66,481 kernel events, 0 mismatches).
4. L2299 21:24:34Z (17:24 EDT)  `cd /home/mfho/wt_forward_2026_08 && python - <<PYEOF` -> adopted_response_v1p1.npz, code_commit 933b0bb9547eae44de1abf84da8df1345d5bc0e5
   (933b0bb9 = "realdata(prep): pinned-environment prescription + lockfiles", 16:50 EDT — it does NOT contain the hierarchical match; 0ecfeea does.)
5. L2304 21:24:47Z  upgrade_packs_v2.py repointed to v1p1 -> adopted_packs_v2p1_20260817; L2411 commit 0ecfeea.
v1 vs v1p1: max|delta| mu_coef 3.5e-4, sig_coef 7.2e-4, skew_coef 0.0196; N_ref identical 20.104069697852808; fit_rng identical.

## Seeds
* boot_carrier: SEED0 = 20260818; draws r=0..95 use np.random.default_rng(20260818 + r) independently; multinomial(n_u, 1/n_u) over the 65,011 unique TIDs, weights = multiplicity per event. Unit-gate draw = weights all 1, compared to d2b_variants ml_shared3__* with tolerance 5e-3.
* run_d2b.py / v1_logo.py: sightline A/B split rng = default_rng(20260817), permutation(len(uniq)) < len(uniq)//2.
* N_proc = 7 (multiprocessing.Pool), login node.

## Inputs (resolved from CDDF_analysis/hbi/ab_loa0_fp_baseline.py DEF_* and track_c_tf_loa._C0_*)
* catalog_dir  = /scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/combined_catalog/
* truth_path   = /nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits
* bal_cat_path = .../loa-124/bal_cat.fits ; snr_cat.fits + zcat.fits from the same loa-124 dir (via _build_qso_lookup)
* molly_tsv    = AB._resolve_molly(None) -> DEF_EXPER/molly_matrix.tsv if present else
                 /scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/figures_molly_nhi195/lya_only/molly_matrix.tsv (nhi195 lya_only)
* frozen response (cells + N_ref only): /scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/stage0/forward_response_2lpt0.npz
      snr_edges [2, 3.5, 6.5, inf], z_edges [0, 2.56, 2.96, inf], N_ref 20.104069697852808
* HBIConfig(..., fp_estimator="purity_mixture", no_bal=True, lam_rf_min=1025.0); load_and_cut_catalog(cfg, truth_nhi_floor=mm.nhi_edges[0] (=19.5), qso_lookup, host_truth_floor=19.0)
* op-cut: S2N_RED > cfg.snr_min (2.0) & P_DLA > cfg.p_dla_min (0.99) & good_mask
