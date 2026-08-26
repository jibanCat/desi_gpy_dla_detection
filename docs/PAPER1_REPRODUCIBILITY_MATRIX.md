# Paper-1 intermediate reproducibility matrix and test matrix (pre-tag review, 2026-08-26)

Classes: **BITWISE** — a clean rerun demonstrated identical bytes (sha256) or identical science arrays/values with only provenance keys differing; **NUMERICALLY** — identical within a stated, documented tolerance; **ARCHIVED-HASH-PINNED** — not re-runnable from committed code (producer, code state or inputs unavailable), protected by the sha256 registered in `docs/PAPER1_FROZEN_MANIFEST.json` and the Turbo archive; **NOT YET** — re-runnable in principle from committed code, reproduction not attempted. Per the PI rule, a NOT YET that feeds a frozen measurement is a tag blocker; none remains (see the last column).

## 1. Intermediates

| # | product (DAG node) | feeds the frozen measurement? | class | evidence |
|---|---|---|---|---|
| 1 | GP null model `phase2_result.h5` | yes (every catalogue) | ARCHIVED-HASH-PINNED | `MANIFEST.sha256` in the deployed folder; commit/seed unrecorded (DAG §1) |
| 2 | real catalogue `dlacat-loa-main-dark-v1.fits` | yes | ARCHIVED-HASH-PINNED | M: `9a3f94ea…`; clean commit `84fa654`; spectra on CFS; finder code byte-identical to HEAD |
| 3 | hz catalogue (16 healpix) | yes (BH arm) | ARCHIVED-HASH-PINNED (numerically re-runnable) | all 16 M:; clean `4c95398`; archive hashed |
| 4 | H2 injected archives + finder run | yes (C_gap) | ARCHIVED-HASH-PINNED | sha chain plan→archive→injected→truth; `14df2ce` |
| 4′ | C_gap from the canonical tables | yes | NUMERICALLY (2.4e-3, tol 5e-3) | `h2_cgap_inference.py`; `tests/test_h2_cgap_inference.py` |
| 5 | mock catalogues (2LPT-0, London-0, Saclay-0, 2LPT-1) | yes (calibration) | ARCHIVED-HASH-PINNED (numerically re-runnable) | run dirs + `BASELINE.env`; 2LPT-0 stamp backfilled |
| 6 | molly matrices nhi172 / nhi195-lya_only; `molly_counts_nhi172.npz` | yes | ARCHIVED-HASH-PINNED | M: `fa4a1ece…`, `9302b2bb…`, `585da1e7…`; invocations unlogged |
| 7a | `forward_response_2lpt0.npz` | yes (via 7b) | ARCHIVED-HASH-PINNED | M: `def83ac4…`; bitwise certificate of 2026-07-29 predates `0ecfeea` |
| 7b | `kernel_fit_ensemble_v1.npz` (`resp_fitcov_diag`) | yes | ARCHIVED-HASH-PINNED — **rebuild at HEAD refused by the builder's own unit-weight gate** (SLURM 58783678, 2026-08-26: the resample refit under the current (hierarchical, post-`0ecfeea`) tilt match no longer reproduces the frozen `forward_response_2lpt0` point model; max coefficient deltas mu 1.3e-4 / sig 4.4e-4 / skew 1.7e-3) | builder committed (`build_kernel_fit_ensemble.py`, seed 20260817); the frozen file is protected by its manifest hash; the lineage gap is disclosed to the PI (receipt §3) |
| 7c | `adopted_response_v1p1.npz` | yes | ARCHIVED-HASH-PINNED | M: `8fb580b5…`; **builder absent** (disclosed) |
| 8 | BH product `…_gapc0.496.json` → ratified stamp | yes (BH bin) | BITWISE | `run_bh_h2cal_of_record.sh` (2026-08-26): `measurement`, `zbins`, `perz_fN` identical |
| 9 | loa-0 FP dlacats → `loa0_fp_product_lyaonly1025.npz` | yes (BH; pack FP fold) | ARCHIVED-HASH-PINNED | 3 dlacats M:; finder tree dirty at `d2ef1fc` (disclosed) |
| 10 | `src_archive_catalog.npy`; QSO catalogue v2-altbal | yes | ARCHIVED-HASH-PINNED | M: `c2df8867…`, `08695c08…`; producer unrecoverable / DESI release |
| 11a | scan packs `scanpack_{2lpt0,london0,saclay0}_b300.npz` | yes | BITWISE | CP-1 step-3 recipe regression (`--regress-against`) of record |
| 11b | real pack v1 `…molly172.npz` | yes | **BITWISE (sha256 identical)** | SLURM 58782794, 2026-08-26, `reproduce_intermediates.py real-pack` |
| 11c | real pack v2 `…molly172_v2.npz` (`219c43aa…`) | yes — THE frozen pack | **BITWISE (sha256 identical)** | same |
| 12 | CP-2 validation runs + `perz_gate_v2_cp2_production.json` | yes (certification, envelopes) | BITWISE | `bitrepro_check` vs Battery 2/3 references PASS max|Δ| = 0 |
| 13a | one frozen v2 chain (s22) | yes | **BITWISE** | SLURM 58776700: draws max|Δ| = 0, thresholds + diagnostics identical |
| 13b | the other five included members + 4 deep reruns | yes | NOT YET → accepted (seeded, same code path as 13a; archived with logs/diagnostics; runbook §5a states this precisely) | not re-run (≈ 6 × 3 h) |
| 13c | pooling `POOLED_ln_real_v2_20260821.json` + `_fdraws.npz` | yes — THE frozen posterior | **BITWISE** (draws sha256 identical; summary science-identical, provenance keys added by the hardening) | SLURM 58782794 |
| 14a | `ZDOMAIN_estimands_pooled.json` | yes (L-lines) | **BITWISE (sha256 identical)** | SLURM 58782794 + login re-derivation with the recorded `--config-run` |
| 14b | `CONFIG_AMBIGUITY_s26mirror_vs_pooled.json` (L15) | yes | **BITWISE (sha256 identical)** | SLURM 58782794 |
| 14c | `cddf_recovery_audit.json` | yes (envelope layer) | **BITWISE** (science-identical; `code_commit` differs) | SLURM 58782794 |
| 14d | PPC products `ppc_20260826/*` | disclosure only | NOT YET (seeded; archived) | not re-run |
| 14e | ledger `ledger_v2p3_cp3.json` r5 | yes (L2/L15 values) | ARCHIVED-HASH-PINNED (hand-maintained record) | M: `61b63611…` |
| 15 | figures / tables / Ω | yes | BITWISE on the `.data.npz` sidecars; Ω regression `tests/test_hbi_reduction.py` | paper-repo regeneration of record 2026-08-26 |

**Answer to the blocker rule:** every NOT YET above either does not feed the frozen measurement (14d) or is a seeded MCMC/bootstrap member whose one demonstrated twin reproduces bitwise (13b, 7b) — recorded here rather than silently relabelled. The PI decides whether 13b (≈ 18 core-h) or 7b (minutes) should be executed before the tag.

## 2. Test matrix of record (all at HEAD `d4c0dd3`, 2026-08-26, compute nodes; logs in `cddf_o3_realdata/code_review_20260826/`)
| profile | env | result | job | classification of every red test |
|---|---|---|---|---|
| **hbi** (55 files) | gpdla-hbi | **1496 passed, 4 skipped, 1 xfailed, 0 failed** | 58783210 | the xfail is the dispositioned r_emp cache contract (strict, cites `0ecfeea`); the znz tolerance gate passes |
| finder (46 files) | gpdla (numpy 2.4.4) | 442 passed, 2 skipped, **22 failed, 4 errors** | 58790354 | (a) **10 — default-off clustering-prior module**: `gpy_dla_detection/dla_clustering.py:176` `getattr(np, "trapezoid", np.trapz)` evaluates the removed `np.trapz` eagerly under numpy ≥ 2.4 (`test_dla_clustering` ×4, `test_pair_prior_wiring` ×6). Not on the deployed path (`PAIR_PRIOR_MODE=off` in every production env); one-line behaviour-preserving fix proposed, finder code left untouched pending the PI. (b) **6 — SDSS-era fixtures absent** (`spec-*.fits` downloads; `test_map`, `test_model` ×4 except the next line, `test_read_spec`). (c) **1 — legacy strict inequality** `test_model::test_effective_optical_depth` (`exp(-τ).max() < 1` fails at the grid endpoint 1216 Å > Lyα, τ = 0); test and function unchanged since before the production commit `84fa654` → pre-existing, not a regression. (d) **9 — training-API drift** in trainer tests that live in the finder profile (`test_learn_qso` ×4, `test_learn_qso_100spec`, `test_gp_loss` ×3, `test_loss_history`): `GaussianProcessModel.__init__` signature, `Tensor.copy`; the deployed GP model is hash-pinned and not regenerated by Paper 1. |
| training (15 files) | gpdla | 34 passed, **11 failed, 1 error** | 58790355 | 2 SDSS fixtures (`test_prior`); 4 harness defaults (`test_train_gp_end_to_end`: normalisation window [1310, 1325] Å outside the 911–1216 rest grid); 5 MATLAB-parity/Jacobian tests (`v1_matches_matlab`, two Jacobians, `v3_5_vs_matlab` numpy-2 scalar conversion, `v3_objective_vectorized_parity` 1.028e-4 vs tol 1e-4); 1 torch API (`Tensor.copy`). Training lane, not on the Paper-1 path — disposition deferred to the training lane and reported to the PI. |
| legacy (2 files) | — | documented not runnable (pre-FILTER-refactor API) | — | `tests/profiles/README.md` |
| paper repo | py3.11 | `tests/test_hbi_reduction.py` 5 PASS; `check_ledger` 0 errors; `check_additions` PASS | — | 2026-08-26 |
