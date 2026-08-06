# PHASE-C RUNNING HANDOFF — live document, updated as work lands
### Branch `calibration/phaseC-highN-fp-2026-08-06`, root `a56e3c8` (= Phase-B tip, frozen)
### Session 2026-08-06 (ends imminently); resume from the "NEXT COMMANDS" section

## Standing context (do not re-derive)

- **Phase-B conclusion (FROZEN by PI, do not weaken or strengthen):** the
  model fails closure through a common observed-N̂ shape tilt; the leading
  supported explanation is a response-model deficiency or calibration gap in
  a materially contributing true-N region that has not been directly
  measured — NOT an FP-normalization problem, an SNR nuisance, or a pad–FP
  identifiability failure. Do not call it a confirmed "kernel error" until
  the direct response measurement supports that.
- Phase-C directive: (1) directly re-measure the response relevant to the
  high-N̂ failure (controlled response-calibration injections AUTHORIZED —
  the old pad–FP campaign stays prohibited); (2) substantially more
  genuinely independent forest-FP events. Full rulings: the PI message of
  2026-08-06 (17 numbered sections) — reproduced in the notes repo,
  `notes/2026-08-06_phaseC_rulings.md` (verbatim paste; commit there).
- Frozen numerical criteria (PI §1.5, do not weaken): G3 = primary
  endpoint; kernel-induced 1σ on predicted G3 ≤ 1/3 of the current G3
  discrepancy; ≥90% power against a response perturbation big enough to
  explain G3; bridge tolerance = uncertainty-justified, frozen before
  production. Layer-B threshold p<0.01 PROVISIONAL (operating study
  required, must NOT use current failures to tune).
- Current G3 numbers (twin, closure_table_phaseB.json @ df29c78): residual
  +450.25 counts (obs−mu; z=+5.93 with full covariance); G1 −1760.82,
  G2 +130.46. Precision target ⇒ σ(kernel-induced G3 prediction) ≤ ~150
  counts.
- Protected: hbi-mcmc-threeroute@9d73365, lls-subdla-cddf@1533333,
  review@a420abd, repair/phaseB@a56e3c8 (all frozen). Envs: gpdla-hbi
  (jax) / gpdla (extract, jax-free); BLAS pinned =1. Packs with fp_eta_c:
  /scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phaseB_packs/.

## Work log (append-only; newest last)

- [06:0x] Branch `calibration/phaseC-highN-fp-2026-08-06` created at
  `a56e3c8`, worktree `/home/mfho/wt_calib_phaseC`, clean. This handoff
  committed as the first commit.

## NEXT COMMANDS (exact resume points)

1. If the preimage analysis (`diagnostics_phaseC/preimage/`) is present and
   committed: read its findings.md; else run
   `cd /home/mfho/wt_calib_phaseC && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/mfho/.conda/envs/gpdla-hbi/bin/python diagnostics_phaseC/preimage/run_preimage.py`
2. Same for the truth-by-SNR refold (`diagnostics_phaseC/truth_by_snr/`).
3. Draft→freeze the calibration design: `docs/PHASEC_CALIB_DESIGN.md`
   (DRAFT status until the pilot-adjustable items are fixed; the FROZEN
   items are already marked and may not be weakened).
4. Not yet started (next session, PI priority order): pilot implementation
   (§1.7 Stage 1); independent-FP expansion design + costing (§3.4: cost
   for 10.6%→5%/3%); Layer-B threshold operating study (§2.3 + MC targets
   §3.5); r5 deterministic merge contract implementation (contract doc
   committed, see governance); bridge-anchor counts (needs pilot yield).
5. Before ANY production calibration: compute/storage budget to PI (§1.8).
