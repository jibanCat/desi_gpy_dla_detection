# PHASE-C RUNNING HANDOFF — live document, updated as work lands
### Branch `calibration/phaseC-highN-fp-2026-08-06`, root `a56e3c8` (= Phase-B tip, frozen)
### Session 2 closed 2026-08-06; PHASE C1 COMPLETE — resume from "CURRENT STATE"

## CURRENT STATE (read this first)

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
2. **Running:** nothing. No SLURM jobs in flight from this stream.
3. **Blocked:** **Stage 2 (production calibration + FP expansion) is
   BLOCKED on explicit PI authorization** (§12 default rule — no
   documented envelope exists). Do NOT launch. Layer-B threshold
   ratification, r5 cadence: PI decisions pending.
4. **Next executable step (after PI authorization only):** generate the
   production arms with `injection/gen_phaseC_resp.py --substrate
   prodlike` using the FULL per-bin/per-cell counts of
   `diagnostics_phaseC/design_sizing/sizing.json` (production anchors =
   every 0.2-dex bin center in [20.5,22.4) + bridge [19.5,20.5); role
   manifest committed BEFORE the first sbatch; 25% whole-healpix holdout
   assigned at generation), then `launch_gl.sh` with
   `phaseC_resp_gl_v1.env` per arm. HARD pre-production blockers
   (review findings F3/F7): (a) add the λ_rf/z_QSO/BAL analysis-window
   cuts to `measure_phaseC_pairs.py` (sentinel + DLAFLAG already
   fixed); (b) build the labeled response-artifact writer (support
   labels per PHASEC_CALIB_DESIGN §6) WITH role-enforcement — the
   scorer must REFUSE rows whose sidecar role is not the one being
   measured, and holdout healpix must be unreadable by the calibration
   path.
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
