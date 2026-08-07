# PHASE-C RUNNING HANDOFF — live document, updated as work lands
### Branch `calibration/phaseC-highN-fp-2026-08-06`, root `a56e3c8` (= Phase-B tip, frozen)
### Session 2 closed 2026-08-06; PHASE C1 COMPLETE — resume from "CURRENT STATE"

## CURRENT STATE (read this first) — P1 BRANCH (`calibration/phaseC-p1-coherent-ck-2026-08-06`, worktree `/home/mfho/wt_p1_ck`)

**P1 phase opened per the PI's P1 rulings (notes repo
`notes/2026-08-06_phaseC_p1_rulings.md` @ `e3930cb`). Sequence: bounded
completeness investigation → P1 estimand freeze → taxonomy +
adjudicability gate → one-time holdout → verdict. Stage-2B FP spend
withheld; holdout unread; quarantine intact; P2/P3 not begun.**

P1 progress:
1. Spec + FROZEN stopping rule committed BEFORE any aggregate
   (`6e08d63`, `bfcb2e4`).
2. **Tier 1 COMPLETE (`cf4d5db`, and the t1_findings commit):**
   deployed C_molly reproduced integer-exactly (two-chain splice);
   🔴 **the 43–58%-vs-81–99% completeness gap was DEAD-STRATA
   ACCOUNTING** — 47.0% of in-window truth sits on S2N_RED ≤ 2
   sightlines the fold zeroes (dX=0); live-support natural completeness
   0.800/0.898/0.952/0.979/0.976 ≈ injected. Bridge criterion 4 was
   mis-pooled (bridge verdict UNCHANGED — criteria 1–3 pair-based);
   the −0.051±0.011 at 21.0 now stands FREE of selection (both
   pipelines ≥97.9% complete there).
3. **Next executable step — Tier 2 (bounded):** explain/bound the
   sub-20.4 pair-mean offset (+0.12 dex at 19.6 decaying to +0.004 ±
   0.013 at 20.4): mismatch/blend composition tests on cached pairs
   (natural matched pairs claimed by blended sub-floor structure vs
   injections with the 5,000 km/s exclusion; use
   `p1_completeness_cache.npz` + the stage2A pairs JSONs; per-stratum,
   PRE-selection covariates only for any reweighting). Then the §22
   consequence report → P1 estimand freeze (parent population =
   LIVE support S2N_RED>2 — decided by Tier 1), joint operator,
   taxonomy, adjudicability gate.
4. Cache: `/scratch/cavestru_root/cavestru0/mfho/phaseC_resp/
   p1_completeness_cache.npz` (cat_cut/truth_cut/is_TP + S2N/ZQSO).

## PRIOR STATE (Stage-2A, superseded but binding context)

1. **Complete:** Phase C1 in full — start-state verification; preimage
   (`diagnostics_phaseC/preimage/`); truth-by-SNR refold (allocation
   REFUTED as tilt driver); FROZEN calibration design
   (`docs/PHASEC_CALIB_DESIGN.md`) + bridge design
   (`docs/PHASEC_BRIDGE_DESIGN.md`) + sizing (σ(G3)=112, power 0.926,
   2,533 injections); Stage-1 pilot RUN AND PASSED (310 injections, GP
   jobs 56605518/9, verdict `diagnostics_phaseC/pilot/findings.md`);
   FP-expansion design+costing (`docs/PHASEC_FP_EXPANSION_DESIGN.md`);
   budget (`docs/PHASEC_BUDGET.md`); threshold operating study
   (p<0.01 not evidently too strict); H9/fallback/r5 governance
   (docs + code + tests); PI checkpoint
   (`docs/PHASEC_CHECKPOINT_2026-08-06.md`); independent code review
   (`docs/PHASEC_CODE_REVIEW_2026-08-06.md`, PASS-WITH-FINDINGS, all
   eight findings dispositioned — see the work log).
2. **Running:** nothing. Stage-2A GP jobs 56619743/56619744 COMPLETED
   (66.9 CPU-h consumed; +8.3 pilot ⇒ ≈75 of 1,850 spent).
3. **Blocked: 🔴 STAGE-2A NO-GO — the production bridge FAILED all four
   frozen criteria; the response is QUARANTINED
   (`quarantined_forward_response_2lpt0_phaseC.npz`); STOPPED FOR PI
   REVIEW per rulings §3.1. Stage 2B's main FP spend is WITHHELD.**
   Read `docs/PHASEC_STAGE2A_BRIDGE_VERDICT.md` FIRST: the failure
   decomposes into (1) a detection-conditioned estimand mismatch below
   ~20.4 — (C_molly, K_natural) is a jointly-defined pair the injected
   campaign does not share — vanishing exactly where completeness
   saturates (Δ = +0.004 at 20.4), and (2) a genuine high-N boundary
   discrepancy: −0.051 ± 0.011 at 21.0, and measured ≈0…−0.04 above
   21.2 vs the clamp's +0.03…+0.09. Precision/power themselves PASSED
   (σ(G3) = 99.0, power 0.976). PI decision paths P1/P2/P3 in the
   verdict doc; do not adopt any unilaterally.
4. **Next executable step (when jobs 56619743/56619744 complete):**
   ```
   cd /home/mfho/wt_calib_phaseC
   export LD_LIBRARY_PATH=$HOME/.local/usr/local/lib64:$LD_LIBRARY_PATH
   PYG=/home/mfho/.conda/envs/gpdla/bin/python
   ARM=/scratch/cavestru_root/cavestru0/mfho/phaseC_resp/prod_v1
   $PYG injection/measure_phaseC_pairs.py --arm $ARM --role bridge --out $ARM/pairs_bridge.json
   $PYG injection/measure_phaseC_pairs.py --arm $ARM --role production-calibration --out $ARM/pairs_production.json
   $PYG injection/measure_phaseC_pairs.py --arm ${ARM%/*}/prod_env_probe_v1 --role environment-probe --out ${ARM%/*}/prod_env_probe_v1/pairs_probe.json
   $PYG injection/build_phaseC_response.py --pairs-bridge $ARM/pairs_bridge.json \
        --pairs-production $ARM/pairs_production.json \
        --out /scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/stage0/forward_response_2lpt0_phaseC.npz
   ```
   Then the go/no-go per rulings §3.1: BRIDGE_PASS + precision/power +
   support artifact ⇒ freeze artifact, predict G1/G2/G3 (packs
   re-extracted with the new envelope), then Stage 2B; FAIL ⇒ quarantine
   + STOP for PI. Holdout is touched ONLY at the final evaluation
   (`--role held-out-evaluation --evaluation-step`). The F3/F7
   pre-production blockers are CLOSED (commits `99fb62e`, `c2154f4`).
5. **Warnings for a fresh session:**
   * The pilot's clamped-region observation (−0.03…−0.14 dex below the
     clamp) is ENGINEERING-VALIDATION ONLY (§11) — do not quote it as a
     result; the production campaign with the frozen holdout decides.
   * The ruling's "939 passed / 1 xfailed" test baseline is NOT
     reproducible in any current env (see log below); the closure-path
     suites are green; do not chase the number without reading the log
     entry.
   * The r5 stochastic test is now an EXPLAINED skip
     (RUN_R5_STOCHASTIC=1 to opt in); the deterministic width contract
     guards per-commit.
   * Pilot artifacts live on scratch
     (`/scratch/cavestru_root/cavestru0/mfho/phaseC_resp/pilot_*`,
     ~1.7 GB, hashes in `diagnostics_phaseC/pilot/input_hashes.sha256`);
     roles are in `roles.json` sidecars (the manifest schema is frozen).

## STAGE-2 AUTHORIZATION + BUDGET BREAKDOWN (PI Decision 1, 2026-08-06)

**Authorized: hard ceiling ≈1,850 CPU-h total; FP target 5% (3% NOT
authorized); response campaign ≤110 CPU-h within it. Ceiling = maximum,
not target — the sequential stopping rule can and should stop earlier.
Sequential: Stage 2A (response+bridge, go/no-go) BEFORE the majority of
the FP spend (Stage 2B). Any projected or actual overrun ⇒ NEW PI ruling
before more jobs.** Rulings verbatim: notes repo
`notes/2026-08-06_phaseC_stage2_rulings.md` (`dd1e85c`).

Reconciliation of the two quoted figures: "≈710 CPU-h" = the
common-reference FP-to-5% line ONLY; "≈1,850" = the complete envelope:

| item | CPU-h (measured basis) |
|---|---|
| **Stage 2A** high-N response injections (2,533 spec incl. bridge + env-probe + holdout, @106 CPU-s pilot-measured) | ≈ 75 |
| Stage 2A retry allowance (15%, pilot-justified) + generation | ≈ 12 |
| Stage 2A bridge/scoring/artifact analysis compute | ≈ 3 |
| *(Stage 2A authorized sub-ceiling)* | *(110 max; ≈90 projected)* |
| **Stage 2B** common-reference FP → 5% (+311 events @2.29 CPU-h/event, loa-0 mock-0 new healpix) | ≈ 710 |
| Stage 2B Saclay transport control (jura-0, ~100 ev) | ≈ 230 |
| Stage 2B Saclay method-bias pair (~100 ev) | ≈ 230 |
| Stage 2B London natural control (~100 ev; transferred label) | ≈ 230 |
| Stage 2B held-out evaluation (mock-1 loa-0, ~150 ev) | ≈ 340 |
| Covariance / null / operating-characteristic simulations (numpy-reduced) | ≤ 5 |
| Fixed overhead (pack re-extraction 21 s/mock; artifact builds; tests) | ≤ 5 |
| **Projected total** | **≈ 1,840 ≤ 1,850 ceiling** |

ACTUALS (sacct TotalCPU): pilot 8.3; Stage-2A production GP jobs
56619743+56619744 = 66.9 ⇒ **≈75 CPU-h spent**; Stage 2B NOT launched
(withheld at the Stage-2A no-go).

Spend tracking: append actual sacct TotalCPU per campaign to this table
as jobs complete; a projected breach of 1,850 stops all launches.
Invalidation rule (§4 of the rulings): outputs from a behaviorally
changed executable state are quarantined and re-run within the same
envelope; if the rerun would breach the ceiling ⇒ PI ruling first.

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
  the old pad–FP campaign stays prohibited; NOTHING below logN 19.5);
  (2) substantially more genuinely independent forest-FP events. Full
  rulings: `notes/2026-08-06_phaseC_rulings.md` in the notes repo
  (committed `29007f7` — session 2 fixed the session-1 claim that it
  existed).
- Frozen numerical criteria (PI §9, do not weaken): G3 = primary endpoint;
  σ(kernel-induced G3 prediction) ≤ 150.1 counts; ≥90% power at two-sided
  α=0.01 against the 450.25-count effect (binding: σ ≤ 116.7); bridge
  tolerance frozen pre-data in PHASEC_BRIDGE_DESIGN.md. Layer-B p<0.01
  PROVISIONAL (operating study committed; ratification = PI).
- Current G3 numbers (twin, closure_table_phaseB.json @ df29c78): residual
  +450.25 counts (z=+5.93); G1 −1760.82, G2 +130.46.
- Protected: hbi-mcmc-threeroute@9d73365, lls-subdla-cddf@1533333,
  review@a420abd, repair/phaseB@a56e3c8 (all verified untouched).
  Envs: gpdla-hbi (jax) / gpdla (desiutil, astropy, desispec); BLAS
  pinned = 1. Packs with fp_eta_c:
  /scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phaseB_packs/.
  Old response envelope:
  .../cddf_o3_realdata/track_c/stage0/forward_response_2lpt0.npz.

## Work log (append-only; newest last)

- [06:0x] Branch created at `a56e3c8`, worktree `/home/mfho/wt_calib_phaseC`,
  clean. Handoff committed as the first commit (`3802d27`).
- [06 s2] Start-state verification PASSED: all four protected tips match
  the rulings; Phase-B branch clean; all Phase-B artifact commits present
  (26a267a is in the NOTES repo, tip c0b8210 as expected); 7 phaseB packs
  + window manifest on scratch. The rulings verbatim file the first
  session claimed did NOT exist — now written (notes repo `29007f7`).
- [06 s2] 🔴 TEST-BASELINE CAVEAT: "939 passed/1 xfailed" is NOT
  reproducible as a single pytest run in any current env. gpdla-hbi lacks
  desiutil (25 collection errors); gpdla lacks jax/corner (9 errors);
  test_mcmc/test_selection/test_zestimation fail in BOTH (the latter two
  import a `process_qso` symbol that does not exist at this tip — a
  pre-existing source-level break). Measured union: gpdla side 1070
  passed / 35 failed / 35 skipped / 5 errors (failures = torch/numpy
  version drift in training paths + test_znz_kernel bit-identity off by
  1.2e-11 in an env whose numpy 2.4.4 exceeds scipy's supported <2.3);
  gpdla-hbi side 239 + 115 passed, 1 xfailed (= r5) → closure-path
  suites ALL GREEN. Phase-C code did not touch any failing path.
- [06 s2] Response-preimage analysis COMMITTED (`d07c4ff`, +G1/G2
  sensitivities `d2adbe3`): G3 feed 12.7% from true<21.0 / 84.0% in-band
  / 3.3% from >21.6; 99% feed = true [20.3,21.7); **47.3% of G3 μ on
  CLAMPED covariates** (top anchors 21.04–21.22); peak sensitivity
  14,300 counts/dex at [20.7,21.1); +450 counts ≡ +0.031 dex there (the
  §9 effect size). Closure-table sanity 1e-6 on all mocks; oracle vs
  build_K 1e-14.
- [06 s2] Truth-by-SNR refold COMMITTED (`ccf9d6d`): truth's real SNR
  allocation differs from pathlength-proportional by 6.6–6.7% L1; G3
  moves ≤5 counts of +450, χ²/dof ≤0.25, all mocks. **Allocation
  REFUTED; G3 deficit SNR-near-uniform; H10's G1 tilt =
  calibration-surface structure.** No SNR nuisance.
- [06 s2] FROZEN calibration design + sizing COMMITTED (`a6e434b`):
  estimand = deployed-pipeline response; anchors [20.5,22.4] dense at
  [20.5,21.3] + bridge [19.5,21.1]; 9-cell stratification; 2,533
  injections → σ(G3)=112, power 0.926; STAGE 2 NOT AUTHORIZED. Bridge
  design + frozen acceptance criterion COMMITTED (`b5d927e`).
- [06 s2] FP expansion design + costing COMMITTED (`13e0d46`): three
  frozen roles; independent substrates VERIFIED ON DISK (2LPT mock-1
  loa-0; Saclay jura-0 twin; London has NO twin → Saclay method-bias
  pair transfers); 2.29 CPU-h/event measured; ±5% ≈ 710 CPU-h, ±3% ≈
  2,340.
- [06 s2] Stage-1 PILOT RUN AND PASSED (`aa9963b` impl, `cd6044a`
  verdict): 208 prodlike + 102 clean injections; production-config GP
  (jobs 56605518/9, finder = primary tree @1533333, 3-docstring-line
  identity verified); matching via THE production matcher
  (match_truth_to_cat_molly dz_rel=0.01); 96% yield; measured 106
  CPU-s/spec; NO bridge cell |z|>2 vs the old envelope; clamped region
  measured −0.03…−0.14 dex below the clamp (ENGINEERING LABEL ONLY).
  Veto rate 17.5% (prodlike redraw planning). One fast-fail SLURM step
  self-covered.
- [06 s2] Threshold operating study COMMITTED (`90d5b91`): deployed
  procedure on a production-regime synthetic universe (faithfulness
  guards 1e-9/1e-12); healthy per-mock type-I 0.0167 (α=.01) vs 0.0573
  (α=.05); healthy-triple false alarm 3.4% vs 11.7%; power at the
  observed-scale tilt 0.51/0.76 per mock — immaterial (actual p≤5e-4).
  p<0.01 stands provisionally.
- [06 s2] Governance COMMITTED: H9 defined + Layer-A conditional-only
  (`f70541c`); conservative fallback reporting IMPLEMENTED — p=None in
  fallback, contingency path only, test pinned (`b63a076`); r5
  restructured — deterministic [14,16]× Jeffreys width contract in the
  suite, stochastic test → explained release-cadence skip (`c3b0941`).
- [06 s2] Budget + PI checkpoint + this handoff committed (`8bfd842`);
  branch + notes repo pushed.
- [06 s3] STAGE 2 OPENED under PI Decision 1 (1,850 CPU-h ceiling, 5%
  FP): rulings anchored (notes `dd1e85c`); ratifications recorded
  (`aab727b`); pre-production code landed F3/F7 (`99fb62e`, A1
  `c2154f4`); executable state FROZEN (`5ee7202` + A1/A2); production
  arms generated at the frozen seeds (prod_v1: 2,597 inj, 48 healpix,
  roles bridge 769 / production 1,167 / holdout 661 on 13 whole healpix,
  0 exhausted cells; env-probe: 260 inj clean substrate); role manifests
  committed BEFORE sbatch (`c5faab0`, `ff44c4b`); bridge/artifact
  builder committed (A2, `4076ad9`); **GP jobs 56619743 (prod) +
  56619744 (probe) submitted**. Budget spent so far this stage: ~8.3
  CPU-h pilot (C1) + ~77 CPU-h in flight.
- [06 s3] **STAGE-2A EXECUTED AND STOPPED AT THE NO-GO.** GP jobs
  completed (66.9 CPU-h); scoring at full fidelity (bridge 694/769,
  production 1,163/1,167 op-matched, 0 out-of-window; roles enforced;
  holdout untouched); precision/power go-condition PASSED (σ(G3)=99.0,
  power 0.976); **the frozen bridge FAILED all four criteria**
  (D = −476 ± 105 counts; z_mean 7.9; z_width 10.5; completeness far
  outside 3σ at every bridge anchor) → artifact QUARANTINED by the
  builder; diagnostic-only projection (labeled) shows wholesale
  adoption would shift groups by (−3,700, +1,200, −470) — the low-N
  estimand contamination the quarantine prevented. Diagnosis: (1)
  detection-conditioned estimand mismatch below ~20.4 (Δ tracks
  completeness 0.43→0.58 molly vs 0.81→0.99 injected; vanishes exactly
  at 20.4; clean-probe rules out substrate effects, |z|≤2.0 everywhere);
  (2) the high-N boundary: −0.051±0.011 at 21.0 and ≈0…−0.04 measured
  above 21.2 vs clamp +0.03…+0.09 — the original Phase-C question, now
  measured. Verdict + PI decision paths (P1 pair-replacement / P2
  estimand-matched natural remeasurement on mock-1 ≈1,500 CPU-h / P3
  bounded-systematic label): `docs/PHASEC_STAGE2A_BRIDGE_VERDICT.md`.
  Two mechanical builder fixes during scoring (dx_sd row field; anchor
  union) + one recorded cosmetic defect (criterion-4 21.0 row inert).
- [06 s2] Independent §21 code review RETURNED: **PASS-WITH-FINDINGS**
  (`docs/PHASEC_CODE_REVIEW_2026-08-06.md` — verbatim record +
  disposition). Every re-run reproduced committed numbers (pairs JSON
  bit-identical; preimage to 4e-16; truth-by-SNR bit-identical); no
  frozen criterion weakened; prohibition boundary enforced. Fixes
  landed same-day: F1 npz provenance (force-added + committed
  generator, bit-reproduced); F2 the checkpoint's premature
  review-record claim (this entry and the committed record correct it);
  F3 sentinel+DLAFLAG in the pair matcher (3 sentinel rows on prodlike,
  all previously unmatched — NO pilot number changed; window cuts remain
  the pre-Stage-2 item); F4 r5 floor raised to the analytic bound;
  F5 bridge CI clause replaced PRE-DATA by a real dispersion guard
  (PI to ratify, checkpoint decision 4b); F6 σ(G3) now summed over all
  re-measured bins: **113.0 counts, power 0.920** (criteria still met);
  F7 role-enforcement tracked as a hard Stage-2 blocker; F8 doc nits
  fixed.

## Exact rerun commands (any artifact on this branch)

```
cd /home/mfho/wt_calib_phaseC
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/mfho/.conda/envs/gpdla-hbi/bin/python      # jax side
PYG=/home/mfho/.conda/envs/gpdla/bin/python         # astropy/desispec side
$PY  diagnostics_phaseC/preimage/run_preimage.py
$PY  diagnostics_phaseC/truth_by_snr/run_truth_by_snr.py
$PY  diagnostics_phaseC/design_sizing/sizing.py
$PY  diagnostics_phaseC/threshold_study/run_threshold_study.py --n-rep 2000 --n-rep-power 600
$PYG injection/gen_phaseC_resp.py --out <arm> --substrate prodlike --n-per-cell 4 --n-healpix 6 --seed 20260806   # pilot regen
bash slurm/greatlakes/production/launch_phaseC_resp_pilot.sh --dry-run
$PYG injection/measure_phaseC_pairs.py --arm /scratch/cavestru_root/cavestru0/mfho/phaseC_resp/pilot_prodlike
# closure-path tests:
$PY -m pytest tests/test_modelA_rungs.py tests/test_modelA_vs_legacy.py tests/test_modelA_pack.py tests/test_gate_covariance.py tests/test_adopted_reporting.py tests/test_matching_contract.py tests/test_modelA_forward_selftest.py tests/test_hbi_mcmc_toys.py -q
```

## Files a fresh session should open first

1. `docs/PHASEC_CHECKPOINT_2026-08-06.md` (verdict/evidence/decisions)
2. `docs/PHASEC_CALIB_DESIGN.md` + `docs/PHASEC_BRIDGE_DESIGN.md`
3. `diagnostics_phaseC/pilot/findings.md`
4. `docs/PHASEC_BUDGET.md`
5. `diagnostics_phaseC/preimage/findings.md`
