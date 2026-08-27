# Paper-1 artifact-production DAG — from spectra and mocks to figures (pre-tag review 2026-08-26)

Every edge lists: producer (script / commit / branch), configuration, inputs, outputs, seeds, hashes (sha256 prefix as registered in `docs/PAPER1_FROZEN_MANIFEST.json`, "M:" = registered there) and the **reproducibility class**:
**BITWISE** = a clean rerun demonstrated identical bytes/arrays; **NUMERICALLY** = a clean rerun demonstrated within a stated tolerance; **ARCHIVED-HASH-PINNED** = not re-runnable from committed code (producer, inputs or code state unavailable) and protected by its registered hash; **NOT YET** = re-runnable in principle from committed code, reproduction not attempted. The matrix of §10 collects the classes; the receipt in the notes repo records who verified what and when. No measurement values appear in this file.

**Governing distinction (PI ruling 5, 2026-08-26).** *Unrecoverable historical realizations* (a finder run whose working tree was dirty, a training run whose seed was not recorded, an injection whose RNG seed is lost but whose realized catalogue is frozen) may be ARCHIVED-HASH-PINNED: the realized artifact is the input of record and everything downstream must reproduce from it. *Unrecoverable analysis transformations* on the core Paper-1 path (calibration inputs → response/kernel → adopted operator → pack → chains → pool → measurements) are NOT acceptable: each must be a committed routine whose rerun is BITWISE or NUMERICALLY reproducible to a justified tolerance. Every entry below is labelled R (realization) or T (transformation).

```
 [DESI Loa spectra + QSO catalogue (CFS)]      [lyacolore 2LPT-0 / London-0 / Saclay-0 / 2LPT-1 mocks (quickquasars, seed 0)]
            │                                        │ truth: hcd_truth_cat.fits, bal_cat, snr_cat, zcat
            ▼                                        ▼
 (1) GP null model  ◄── trained on loa-124 HCD-free/BAL-free twin (2LPT-0)  ──────────┐
            │                                                                          │
            ├──► (2) real catalogue 84fa654 ──► (10) archive sightline cat ──┐          │
            ├──► (3) hz catalogue 4c95398 (z_QSO 4.25–7) ──► (8) BH arm      │          │
            ├──► (4) H2 injection finder run 14df2ce ──► C_gap ──► (8)       │          │
            ├──► (5) mock catalogues (2LPT-0 b219996, London-0 8e40399, Saclay-0 f1784fc, 2LPT-1 891db99)
            │            │                                                   │          │
            │            ├──► (6) molly completeness matrices (nhi172, nhi195/lya_only) ──┐
            │            ├──► (7) forward response (kernel) chain ──► adopted_response_v1p1 + kernel_fit_ensemble_v1
            │            └──► (9) loa-0 FP companion d2ef1fc ──► loa0_fp_product ──► (8) BH and the pack FP fold
            ▼                                                                │          │
 (11) CP-1: extract_pack (mocks) → upgrade_packs_v2 → build_scan_packs → extract_pack_real → contract guards → certify_g_support
            │
 (12) CP-2: cc_posterior_validation on the 3 scan packs → perz_gate → CERT_G_SUPPORT_CP1_CP2
            │
 (13) CP-3: cc_real_posterior ×8 seeds (+4 deep) → cc_pool_posterior (predeclared rule) → POOLED (frozen) → FROZEN_STATUS
            │
 (14) reductions: cc_zdomain_estimand, cc_config_ambiguity (L15), cddf_recovery_audit, PPC; ledger v2.3 r5
            │
 (15) paper repo: hbi_reduction (closure-guarded) → figures/tables → ledger → PDFs
```

## 1. Deployed GP null model — R (historical realization)
producer `tests/phase2_train_desi.py` via `phase2_desi_retrain.sh`, SLURM 50017770 (2026-05-14) · config `n_iters=1500`, norm band [1425,1475] Å, lr 0.005, chunk 5000 · inputs preload `2lpt_loa124_nohcd_nobal_wide_v2_1778186324/trainset.h5` (2LPT-0 loa-124 HCD-free/BAL-free twin) · outputs `DEPLOYED_phase2_2lpt_loa124_nohcd_nobal_wide_m/phase2_result.{h5,npz}` (Turbo; `MANIFEST.sha256` h5 `5e2a7691…`, npz `8d129c1d…`) · seed not recorded · **commit NOT recorded** · ~1,780 CPU-h · **ARCHIVED-HASH-PINNED** (Adam training, no seed/commit; checkpoints not preserved). Caveat of record: the training mock is also the calibration mock ⇒ 2LPT-0 recovery is a floor, not a held-out leg (PI D2, 2026-07-27).

## 2. Real DESI Loa catalogue — R
producer `desi-DLAGP.py` → `examples/combine_dlacat.py` → `tools/postprocess/add_dla_flags.py` (Lyβ dz 0.005) → `package_catalog.sh`; `slurm/nersc/production/loa_nersc_v1.env`, run `nersc_prod_loa_v1_20260606` · commit **`84fa654` (`nersc_production`, clean; FITS `CODECMT`)** · config `MODE=loa, PACKING=N32xW8, MAX_DLAS=4, SINGLE_ABSORBER_MODEL=1, FILTER_LOW_LIKELIHOOD=1, λ 911.75–1250, K=30, 31 forest lines, NUM_LINES=3, BALMASK=false, τ-EB on, PW50k` · inputs `QSO_cat_loa_main_dark_healpix_v2-altbal-20241115.fits` (CFS; Turbo copy M: `08695c08…`, DESI release product, not re-derivable), GP model (1) · outputs `loa_main_dark_v1/dlacat-loa-main-dark-v1.fits` (M: `9a3f94ea…`; 801,761 rows / 358,835 TIDs) · no seed · **ARCHIVED-HASH-PINNED** (spectra on CFS; numerically re-runnable in principle from CFS; finder inference code byte-identical to HEAD — `docs/PAPER1_BRANCH_TOPOLOGY.md` §3).

## 3. High-z catalogue (BH arm substrate) — R
producer `launch_nersc.sh slurm/greatlakes/production/loa_cddf_hz_gl_v1.env` (sources the frozen NERSC flavour; site plumbing only) with `EXTERNAL_HPX_LIST=production_hpx_list.txt`, `GPDLA_SPECTRA_ARCHIVE=loa_hz_archive_v1.h5` · commit **`4c95398` = `prov/pre-gl-highz-2026-08-13` (clean)** · config `PACKING=N16xW2, MAX_DLAS=1, FILTER_LOW_LIKELIHOOD=0, PW100k, GPDLA_ZMIN_QSO=4.25 / ZMAX 7.0` · inputs QSO catalogue v2-altbal, GP model (1), verified LoaArchive `loa_highz_2026-08-12c/archive/loa_hz_archive_v1.h5` · outputs 16 `dlacat-loa-main-dark-hpx-*.fits` (`gl_cddf_loa_hz_v1_20260813/outputs`, all M:, run-dir `production_output_manifest.sha256`) · no seed · **NOT YET** (numerically re-runnable: clean commit on the current branch, archive hashed, env named; hours of finder time; not attempted — accepted as hash-pinned for the tag).

## 4. H2 injection campaign (C_gap) — R for the injection/finder legs; T for the C_gap analysis
producer `tools/h2_select.py` → `tools/h2_inject_archive.py` → finder at **`14df2ce`** (`h2_exec/loa_cddf_hz_gl_h2arm{A,B}.env`, sourcing (3)'s env; arm A restores z_QSO 2.0–4.25) → `tools/h2_canonical_analyze.py` · plan frozen `h2_freeze_v2/h2_realized_plan.csv` (sha `efc8736e…`); arm A 179 sightlines / 298 injections (source archive `4447cb6c…` → injected `a85052c1…`, truth `ddc6b6fc…`), arm B 360 / 600 (`0628f098…` → `c94bbbde…`, truth `5716a92e…`) · outputs canonical tables `h2_exec/h2_canonical_arm{A,B}_{lya,lyab}{,_nobal}.json` + `h2_canonical_ab_transport.json` (9, all M:) · realisation-pinned (no RNG seed recorded) · injection + finder legs **ARCHIVED-HASH-PINNED**; analysis leg `CDDF_analysis/hbi/h2_cgap_inference.py --canonical h2_exec/h2_canonical_armB_lya_nobal.json` → C_gap **NUMERICALLY** (MC to 2.4e-3, documented tol 5e-3; `tests/test_h2_cgap_inference.py`).

## 5. Mock catalogues — R (2LPT-0 equivalence run: receipt §1)
| mock | run | commit / branch / stamp | config |
|---|---|---|---|
| 2LPT-0 loa-124 | `gl_prod_2lpt0_v1_20260526` | `b219996` / `production_533` / **dirty=unknown, backfilled by timestamp** | PW100k, MAX_WORKERS 16, `2lpt0_gl_v1.env` |
| London-0 jura-124 | `nersc_prod_london0_v1_20260603` | `8e40399` / `nersc_production` / clean | N32xW8 |
| Saclay-0 juraLy8-124 | `gl_prod_saclay0_v1_20260630` | `f1784fc` / `cddf-analysis-reorg` / clean | PW50k |
| 2LPT-1 loa-124 (held-out) | `nersc_prod_2lpt1_v1_20260604` | `891db99` / `nersc_production` / clean | — |
All: GP model (1), `combine_dlacat.py` + `add_dla_flags.py`; truth from quickquasars/`lyatools-make-dla-cat` (`--seed 0` both; toolchain versions NOT recorded; not in the manifest). Class **ARCHIVED-HASH-PINNED** for the tag (numerically re-runnable in principle: mocks + model on Turbo, envs committed). Caveat: the 2LPT-0 finder predates the default-off clustering hook in `dla_gp.py` (topology doc §3(b)).

## 6. Molly completeness matrices — R for the frozen TSVs (invocation unlogged); T for `molly_counts` (consumer-side re-binning)
producer `examples/molly_faithful_pc_plots.py` driven by `slurm/greatlakes/production/_gen_molly_nhi195.sh` (config-only re-binning; zero GP re-inference; `--snr-min 2.0 --gp-conf 0.99`) · inputs 2LPT-0 combined catalogue + `hcd_truth_cat.fits` + `bal_cat.fits` · outputs `figures_molly_nhi172/molly_matrix.tsv` (M: `fa4a1ece…`), `figures_molly_nhi195/lya_only/molly_matrix.tsv` (M: `9302b2bb…`) · **exact FLOOR/NHIBINS/TITLE arguments of the two frozen variants NOT logged** · downstream cache `ff_fp_cache/molly_counts_nhi172.npz` (M: `585da1e7…`; self-describing, producer invocation not recorded; consumed at `extract_pack.py`) · **ARCHIVED-HASH-PINNED**.

## 7. Response / kernel chain (all T = analysis transformations; PI ruling 3/4 reconciliation of 2026-08-26)
```
calibration events (2LPT-0 detections + hcd_truth_cat + snr_cat/zcat + bal_cat + molly nhi195/lya_only)
  → 7.1 matching / tilt assignment      cddf_catalog_hbi.load_and_cut_catalog → NHI_TILT_HOST (hierarchical since 0ecfeea)
  → 7.2 forward response                znz_kernel.build_forward_cache / measure_forward_response (host_col=NHI_TILT_HOST, lam_rf_min 1025) → forward_response_2lpt0.npz
  → 7.3 kernel fit + fit covariance     znz_kernel.fit_forward_response; hbi_mcmc.build_kernel_fit_ensemble (400-draw sightline bootstrap, seed 20260817, unit-weight gate) → kernel_fit_ensemble_v1.npz → resp_fitcov_diag
  → 7.4 adopted response operator       adopted_response_v1p1.npz (per-cell deg-2 skew-normal moment surfaces + shared cubic + 96-draw carrier, seed0 20260818) — builder: see the recovery/reconstruction record in the receipt and §7.4 below
  → 7.5 certified pack                  extract_pack_real --stamp-v12 (--adopted/--kfe default to the operators of record) → modelA_pack_REAL_…_v2.npz (adopted_* keys + resp_fitcov_diag)
```
| stage | committed producer (current) | historical producer commit | inputs | config / seed | output schema | class |
|---|---|---|---|---|---|---|
| 7.1 tilt assignment | `CDDF_analysis/hbi/cddf_catalog_hbi.py:load_and_cut_catalog` (hierarchical: primary ≥ matrix-floor assignment preserved; hostless detections only from the low-floor pool) | pre-`0ecfeea` (greedy whole-pool re-match) for 7.2/7.3 of record; `0ecfeea`/`933b0bb` for 7.4 | 2LPT-0 catalogue, `hcd_truth_cat.fits`, `bal_cat.fits`, molly nhi195/lya_only | `dz_rel`, host floor 19.0, truth floor per molly | catalogue columns `NHI_TRUE`, `NHI_TILT_HOST`, op mask | T — deterministic; contract change documented |
| 7.2 forward response | `python -m CDDF_analysis.hbi.znz_kernel build-forward-cache --lam-rf-min 1025.0 --out …` | `ecc06cb` certificate (2026-07-29) on the 2026-06-20 object | 7.1 output | `lam_rf_min 1025`, `host_col NHI_TILT_HOST`, fit floor 19.5, deg (1,2) | 15 arrays: `mu/sig/skew_coef`, `fit_rng`, `N_ref`, population counts | T — historical rebuild at `475c62b` and current rebuild at HEAD: see matrix §7 |
| 7.3 kernel-fit ensemble | `python -m CDDF_analysis.hbi_mcmc.build_kernel_fit_ensemble --n-draws 400 --seed 20260817 [--frozen-npz …] --out …` | `475c62b` (file identical to HEAD; `code_commit 70efc09` stamped) | 7.2 output + calibration events | seed 20260817; unit-weight gate | `mu/sig/skew_coef (400,3,3,3)`, `point_*`, `n_events`, `n_uniq_tids`, `seed`, `provenance` | T — see matrix §7 |
| 7.4 adopted operator | **see receipt (recovery / reconstruction of the builder)** | `74762e9` (v1) → `933b0bb` (v1.1, hierarchical) | 73,845 tilt-host pairs (op-cut, x̂ ≥ 19.5, censored anchors self-excluded) | seed0 20260818; 96-draw multinomial sightline bootstrap; LOGO 15/15 | `mu/sig/skew_coef (3,3,4)`, `fit_rng (3,3,2)`, `N_ref`, `carrier_* (96,3,3,4)`, `carrier_shared3 (96,3)`, `carrier_unit_gate (3,)`, `provenance` | T — hard pre-tag gate (PI ruling 4) |
| 7.5 certified pack | `extract_pack_real.py --stamp-v12` (gpdla-hbi) after `--cert-2lpt0`/`--real` (gpdla) | `0babe21` (CP-1) | 7.4 + 7.3 + real pack v1 | level identity ≤ 1e-6 | pack schema v1.2 (`docs/HBI_ARCHITECTURE.md`) | T — BITWISE (matrix 11c) |
Diagnostic lanes (not on the frozen path): `p1_natpair_ck_v1.npz` ← `injection/build_p1_natpair_ck.py` (`bc2564f`); Phase-C bridge `0b545cb` **QUARANTINED**, never spliced; the WALL-1 R_emp/broaden012 caches (retired; `tests/test_remp_rebind.py`, `tests/test_znz_kernel.py` dispositions).

## 8. BH / high-z arm product
producer `slurm/greatlakes/production/paper1/run_bh_h2cal_of_record.sh` = `CDDF_analysis/hbi/track_c_tf_hz.py --variant h2cal --fp loa0 --window lya --envelope none --gap-treatment frozen --gap-c 0.496 --zbins 3.8,4.25,4.5,5.0 --n-mc 2000` (env gpdla; `argv` now stamped) · inputs (3), (9) `loa0_fp_product_lyaonly1025.npz`, C_gap from (4) · output source artifact `…_gapc0.496.json` (M: `90264a22…`) → ratified stamp `bh_ratify_stamp.py` (`5e26b35`; M: `62446b47…`) · MC seeded · **BITWISE** (verified 2026-08-26: `measurement`, `zbins`, `perz_fN` identical).

## 9. loa-0 FP companion — R (dirty tree; accepted disclosed, PI ruling 2)
finder `gl_loa0_fp_v1_20260615` (`loa0_fp_gl_v1.env` = 2LPT-0 env with MOCKDIR/QSOCAT/OUTDIR swapped; loa-0 = HCD-free BAL-free byte-identical twin) · commit `d2ef1fc` (`cddf_prod`) **`CODE_DIRTY=dirty`** · outputs 3 dlacats (M: `686dc162…`, `d1ab230b…`, `6b0ed767…`) → `CDDF_analysis/hbi/build_loa0_fp_product.py` → `loa0_fp_product{,_lyaonly1025}.npz` (self-describing; build step unstamped) · finder leg **ARCHIVED-HASH-PINNED** (dirty tree unrecoverable; disclosed); product leg NOT YET (builder committed, inputs hashed).

## 10. Archive sightline catalogue and the QSO catalogue — R
`h2m_ckpt10p5_20260817/analysis/src_archive_catalog.npy` (M: `c2df8867…`; consumed by `extract_pack_real.py`) — **producer NOT recoverable** (no script, command or note) → ARCHIVED-HASH-PINNED. `QSO_cat_loa_main_dark_healpix_v2-altbal.fits` (M: `08695c08…`) — DESI release product → ARCHIVED-HASH-PINNED.

## 11. CP-1 — certified corrected-g packs
`slurm/greatlakes/production/paper1/run_cp1_regeneration.sbatch` (steps 1–2 at `ebf3787`, 3–9 at `0babe21`): `extract_pack.py --mocks 2lpt0 london0 saclay0 --basis-pad-floor 19.0 --completeness-below-floor molly172 --basis-width 0.2` (gpdla) → `upgrade_packs_v2` (v1.2 stamp; gpdla-hbi) → `build_scan_packs.py` (b=300; `rng_seed=0`; `--regress-against real_pack_v1` byte-identity proof) → `extract_pack_real.py --cert-2lpt0 / --real / --stamp-v12` → `contract_guards_check` (fail-closed) → `certify_g_support.py` · inputs (2), (5), (6), (7), (9), (10), truth catalogues · outputs `adopted_packs_gfix_v1_20260821/*`, `adopted_packs_v2p2_20260821/*`, `real_pack_v2_20260821/scanpack_{2lpt0,london0,saclay0}_b300.npz` (M: `6cd07d20…`, `dd02cf7c…`, `e45b146e…`), real pack v1 (M: `d0f673f0…`) and v2 (M: `219c43aa…`; provenance/contract/selection sidecars M:), `CERT_G_SUPPORT_CP1.json` (M:) · classes: scan packs **BITWISE** (recipe regression of record); real pack v1/v2 → §10 matrix (SLURM 58782794).

## 12. CP-2 — mock validation
`run_cp2_gate.sbatch` (`cc_posterior_validation --samples 500 --warmup 1500 --chains 2 --fp-mode informative_ln --target-accept 0.95 --seed 20260811 --save-fdraws`, 3 packs) → `run_cp2_collect.sbatch` (`perz_gate` → `perz_gate_v2_cp2_production.json`; `bitrepro_check` vs Battery-2/3 references → PASS max|Δ| = 0; `certify_g_support --validation-json` → `CERT_G_SUPPORT_CP1_CP2.json`) · seeds explicit · **BITWISE** (CP-2 bit-reproduces Battery 2/3; `tests/test_paper1_review_guards.py`).

## 13. CP-3 — the frozen real posterior
`run_cp3_real_battery.sbatch` (`cc_real_posterior --pack <v2> --samples 500 --warmup 1500 --chains 2 --target-accept 0.95 --fp-mode informative_ln --seed 20260821..28`, code `b59e0b5`) → stage-1 collector (`select_runs`) → `run_cp3_deep.sbatch` (warmup 3000; seeds 21/23/26/28; `ea4c7bb`) → stage-2 `cc_pool_posterior --rhat-max 1.10 --div-max 10 --expect-pack-sha256 219c43aa…` → `POOLED_ln_real_v2_20260821.json` (M: `ea881b5f…`) + `_fdraws.npz` (M: `e43d9148…`) + `FROZEN_STATUS.json` · every chain (8 base + 4 deep incl. excluded s23/s26), logs, selections, PPC draws are M: and in the Turbo archive; `tools/paper1/chain_ledger.py` reconstructs chain → rule → pool · classes: single member s22 **BITWISE** (clean rerun SLURM 58776700, 2026-08-26); the six-member pool from clean reruns NOT demonstrated (stated in the runbook §5a); pooling step → §10 matrix.

## 14. Reductions and systematics products
`cc_zdomain_estimand` (`ZDOMAIN_estimands_pooled.json`, M:), `cc_config_ambiguity` (`CONFIG_AMBIGUITY_s26mirror_vs_pooled.json`, M:; = L15), `cddf_recovery_audit.py` (16 per-z gate runs + pooled draws + s26 mirror; `cddf_recovery_audit.json`, M: `7c40b962…`; `08504c0`), PPC (`cp3_real/ppc_20260826/*`, M:), ledger `ledger_v2p3_cp3.json` r5 (hand-maintained, M: `61b63611…`) · deterministic (no MCMC) · classes → §10 matrix.

## 15. Paper repo (mode A)
`tools/paper1/frozen_manifest.py --verify` → `paper_figures/build_all.py` (12 figures + `emit_comparison_omega`; `hbi_posterior_bundle()` fails closed on FROZEN_STATUS sha mismatch; `hbi_reduction.require_closure` to 1e-6; Ω at h = 0.70, N_max 21.6 — `tests/test_hbi_reduction.py` regression) → `ledger_rows_from_provenance.py` → `check_ledger.py` → `check_additions.py` (draft + submission PDFs) · env `ENV_LOCK_2026-08-26.txt` (python 3.11.0, matplotlib 3.10.8, numpy 2.0.1, TeX Live 2026) · figure `.data.npz` sidecars carry the plotted arrays; PDFs embed timestamps (compare the sidecars, not the PDFs) · **NUMERICALLY/BITWISE on the sidecars** (regeneration of record 2026-08-26; ledger rows keyed on checksum + generator commit).

## 10′. Intermediate reproducibility matrix
See `docs/PAPER1_REPRODUCIBILITY_MATRIX.md` (filled from SLURM 58782794 and the verifications above).
