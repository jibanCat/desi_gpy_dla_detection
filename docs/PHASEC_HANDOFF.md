# PHASE-C RUNNING HANDOFF — live document, updated as work lands
### Branch `calibration/phaseC-highN-fp-2026-08-06`, root `a56e3c8` (= Phase-B tip, frozen)
### Session 2 closed 2026-08-06; PHASE C1 COMPLETE — resume from "CURRENT STATE"

## CURRENT STATE (read this first) — P1 BRANCH (`calibration/phaseC-p1-coherent-ck-2026-08-06`, worktree `/home/mfho/wt_p1_ck`)

### PI checkpoint 3 ACCEPTED (2026-08-06, after Tier 1 + partial Tier 2)

**Accepted: the Tier-1/2 findings and the criterion-4 correction. The
Stage-2A bridge no-go REMAINS SUPPORTED by criteria 1–3.** Ordered:
continue the BOUNDED CATALOG-LEVEL Tier-2 tests; **no forced fits, no
holdout opening, no Stage-2B, no P2**; handoff updated+pushed before
further work (this revision).

### Corrected scientific record (quote THESE, not earlier phrasings)

1. The Stage-2A bridge FAILED on criteria 1–3 (pair-based: G3-projected
   D = −476 ± 105; coherence z 7.9/10.5; local pattern). **Criterion 4's
   "completeness far outside 3σ" is RETRACTED as physics** — it pooled
   the fold's dead (dX = 0, S2N_RED ≤ 2) molly strata into C_molly; the
   PI accepted the correction. The response artifact remains QUARANTINED.
2. The natural-vs-injection completeness gap was DEAD-STRATA
   ACCOUNTING: 47.0% of in-window truth sits on S2N_RED ≤ 2 sightlines
   outside the live fold. Live-support natural completeness =
   0.800/0.898/0.952/0.979/0.976 over [19.5,20)/[20,20.4)/[20.4,21)/
   [21,21.5)/≥21.5 ≈ injected (0.81–0.99). Deployed C_molly reproduced
   integer-exactly (two-chain splice; matching competition embedded).
3. The deployed degree-2+clamp SURFACE misfits its OWN raw pairs:
   surface−pairs = +0.071/+0.020/−0.017/−0.035/−0.028/−0.004 dex over
   [19.5,19.8)…[21.0,21.3) — edge failure at BOTH boundaries; much of
   the low-N bridge Δ was this representation error.
4. Blend class (catalogued 17.2–19.5 neighbor ≤3,000 km/s): 7.5–8.0% of
   natural pairs, dx elevated +0.03…+0.10 dex — real, secondary.
5. 🔴 **SURVIVING OPEN OBJECT (the sole gate): a pair-level
   natural−injected dx offset of +0.00…+0.06 dex — isolated-natural
   minus injected ≈ +0.019/−0.004/+0.035/+0.017/+0.032/+0.059 at
   19.6/19.8/20.2/20.4/20.8/21.0 — largest at 21.0 where BOTH
   selections are ≥98% complete and blends are controlled.** Candidate
   mechanisms (§18): host-environment forest coupling (predicts
   host-N/local-density dependence) vs imprint-vs-injection profile
   realism (predicts flatness in those covariates). If coherent over
   [20.7,21.1) it projects to ~700 counts on G3 — material.

### SOLE GATE — RESOLVED (2026-08-06, `468f901`)

`t2_pairing.py` ran under its frozen verdict rule:
**IMPRINT-SUPPORTED (environment-flat)** — D1 offset rises with host N
(+0.018→+0.045, z→8.0); D2 shell-density slope difference z = +0.51
(coupling bounded ≲0.015 dex/count at 95%; proxy limitation stated).
**Level-A capstone (`t2_completion.md`): the NATURAL pairs refute the
deployed clamp directly** — pairs−surface = −0.030 (−6.5σ) /
−0.044 (−5.4σ) / −0.043 (−3.8σ) over [21.4,22.1); true mean-bias falls
+0.055→+0.01 while the clamp holds +0.05. Labeled back-of-envelope:
surface-vs-own-pairs corrections project to ≈ the full +450-count G3
discrepancy (exact number = the GATED P1 refold). Stopping rule fully
MET; the completeness investigation is COMPLETE.

### 2026-08-07 session: power addendum + PROPOSED estimand freeze — STOPPED AT THE PI CHECKPOINT

1. **State recovery VERIFIED** (fresh session): tips = origin (then
   `3c3e821`), worktree clean, notes `e3930cb`; protected tips
   unmoved; cache + quarantine intact; closure-path suites **331
   passed / 1 skipped** (the explained r5 skip); `t2_pairing.json`
   re-run reproduces bit-level up to 1–2 ulp; sacct since Aug 5 shows
   ONLY the four authorized GP jobs from this repo (all other jobs =
   `~/hcd_priya`, a separate project). No Stage-2B / P2 / forced-fit
   launches. Holdout roles intact (661 held-out; scripts filter).
2. **Post-specified power/robustness addendum (`6aa526f`,
   `t2_power.py`, labeled — verdict NOT re-adjudicated):** catalogued-
   shell coupling excluded at ~15× margin (needed slope 38σ above
   measured; lever absent); near-field channel bounded MECHANICALLY by
   the `forest_flux_frac` design covariate — wrong sign for coupling
   (cleanest sites read highest, +0.037±0.017) and ≤0.007 dex per 1σ
   population shift; offset persists at full size in shell=0 pairs
   (+0.0475 peak); **natural kernel 15–25% WIDER than injected at
   every bin**; frozen offsets project to **+387±76 G3 counts**
   (labeled; refold gated).
3. **P1 ESTIMAND SPEC — PROPOSED FREEZE (`5b63f49`,
   `docs/P1_ESTIMAND_SPEC.md`):** coherent (C_molly, K_natural-pairs)
   on live support; pairs-faithful de-clamped representation; explicit
   miss state; frozen 9-cell conditioning; blend class inside K as
   composition; **no estimand transition** ([20.4,21.1] = validation
   overlap); injected campaign = validation + frozen transfer map;
   estimand ID `p1_natpair_ck/v1` with atomic C/K + fail-loud guards.
   States plainly: within-realization; realization independence = P2
   content (separate PI decision).
4. **Failure taxonomy + holdout gate — PROPOSED FREEZE (`7fdf570`,
   `docs/P1_FAILURE_TAXONOMY.md`, `p1_holdout_gate.py/.json`,
   design-side only):** seven categories, one primary; holdout
   adjudicable at the material scale (power 0.99 vs the 0.031-dex
   defect pooled over [20.7,21.1), ≥0.99 vs clamp-scale per bin); NOT
   adjudicable ≤0.015 dex (carried as covariance systematic); frozen
   Holm-corrected battery, one-time read.

### 2026-08-07 PI RULING (same day): anchor APPROVED, engineering phase EXECUTED

PI accepted the Tier-2 checkpoint; approved the natural-pair kernel
anchor; ratified the holdout framework in principle; deferred P2 until
after the holdout; complete freeze conditional on six mechanical
gates. **All six gates were then executed and PASS:**

1. **Atomic (C,K) artifact BUILT (`f1eff35`):**
   `p1_natpair_ck_v1.npz` (sha256 in `p1_ck_build.json`); kernel
   events = EXACTLY the deployed numerator events; **identity
   integer-exact in all 56 ≥19.5 cells; miss closure exactly-once**
   (det + subfloor + lowP + flag + unmatched == tot). NEW: the
   SUBFLOOR class (matched, N̂≤19.5; 1,650 live, 1,611 at
   [19.5,20.0), 0 above 21.0) is a MISS in the estimand → C_live at
   [19.5,20.0) = 0.7504 under the deployed convention (ledger's 0.800
   counted subfloor as matched; both labeled). Loader: fail-loud
   guards (estimand/version/identity/closure/normalization/NaN),
   read-only, no renormalization path; 10/10 guard tests incl.
   scratch integration.
2. **Population coherence (spec §15.1–15.2):** K = ALL eligible
   production-matched pairs (NOT the isolated subset — that is only
   the injected transfer map); C/K one-chain compatibility PROVEN by
   the integer identity.
3. **Width checks (spec §15.3, frozen rule):** the natural width
   excess PERSISTS in iso/shell0/no-nb-30k subsets (×1.15–1.22 over
   [20.4,21.3); converges ×1.01–1.03 at [21.3,21.7)) → catalogued
   classes insufficient; all-overlap-excluded NOT claimed.
4. **Merge/split accounting (spec §15.4):** every truth counted
   exactly once; class→term table frozen; marginal per-absorber
   operator CLOSES (spec §15.5) — no multi-object model needed.
5. **Healpix jackknife (frozen gate, `p1_jackknife.json`): PASS** —
   se ratios 0.98–1.03, max single-healpix shift 0.1–0.3 se,
   ~1,100 healpix; nside-16-nest convention validated on 2,332
   shared TARGETIDs.
6. **Representation/hidden-transition audit (spec §16):** [19.5,21.7)
   = 11 bins × all 9 cells directly measured; sparse structure
   confined to ≥21.7, flagged; no refit/clamp/smoothing; one source
   chain; NO hidden transition. Holdout battery FROZEN
   (`p1_holdout_battery.json`, `b029936`) — calibration+design side
   only; holdout untouched.

### 2026-08-07 AMENDED PI RULING: two-layer ratification; holdout NOT yet authorized

PI accepted the pre-read gates but split ratification into a HIGH-N
PRIMARY OPERATOR and a LOW-BOUNDARY TRANSPORT EXTENSION, and ordered
seven work items before any read. **All seven executed same-day
(commits `14d430f`…`41545a1` + spec §18):**

1. **Chain bridge:** 17.2/nhi195 compatible — truth common support
   IDENTICAL; 4 competition reassignments of 10,687 (excluded); 0.13%
   catalogue-cut difference reported separately, never absorbed.
2. **Below-floor migration (net):** f = 22.7% at observed [19.5,20),
   15.2% at the 19.7 reporting floor, 4.1% at [20.0,20.3); **G1
   10.36% (4,088 ev) / G2 0.60% (144 ± 12) / G3 = ZERO events.**
   Primary operator keeps truth ≥ 19.5 + EXPLICIT source term; K
   never renormalized.
3. **Emission proximity (frozen regions):** Lyα-emission effect
   −0.083…−0.093 dex at every N — largely COMMON-MODE (injected
   −0.056, z −7.7); low-N completeness deficit near Lyα em (0.517 vs
   0.773), confined to low N; marginal-K mixture shift ≤ 0.003 dex
   (below materiality) ⇒ marginal K valid within-realization; frozen
   conditional table required for cross-mixture transport.
4. **Nomenclature (binding):** `C_fm` (finder-matched; 0.800) vs
   `C_paf` (production-above-floor; 0.7504) — never interchangeable.
5. **Joint (C,K) covariance:** whole-healpix jackknife; corr
   −0.04…+0.04 all bins — NOT material; ESS 242–896; stable.
6. **Hierarchical battery v2 + global verdict rule FROZEN pre-read**
   (`p1_holdout_battery_v2.json`): primary high-N family alone
   decides P1; low family maps to its own outcomes; below-floor
   migration recorded as NOT holdout-testable; exploratory subgroups
   cannot reject/tune/promote. High-N validity ≠ low-boundary
   validity.
7. **Primary support frozen: N_true ≥ 20.3** (molly edge; 99% of the
   G3 feed; zero migrant events in observed G3).

### Exact next step — AMENDED PRE-READ CHECKPOINT RETURNED; PI decision required

Requested: (1) ratify the HIGH-N primary operator (support ≥ 20.3);
(2) low-boundary extension = RESTRICTED/QUALIFIED (explicit source
term + conditional table + C_paf nomenclature; no unqualified
[19.5,22.5) closure claim); (3) authorize the one-time holdout read
under battery v2. On authorization: run the battery via the committed
measurement path, commit the full output in the same run that first
touches the rows. Stage-2B / P2 / refold / splice remain prohibited.

### Budget

Spent ≈ 75 of the 1,850 CPU-h ceiling (pilot 8.3 + Stage-2A GP 66.9 +
<1 analysis). Withheld: Stage-2B FP program (~1,740), P2 (~1,500, needs
a NEW ruling). Tier-2 catalog tests + 08-07 addendum/gate: <1 CPU-h
total, cache/design-side, login node; storage +~40 KB committed JSON.

### Repository tips (2026-08-07, this revision)

| ref | tip |
|---|---|
| `calibration/phaseC-p1-coherent-ck-2026-08-06` (THIS branch, wt `/home/mfho/wt_p1_ck`) | `b029936` + spec/handoff commits (this revision) |
| `calibration/phaseC-highN-fp-2026-08-06` (Stage-2A record, wt `/home/mfho/wt_calib_phaseC`) | `60cef40` (frozen) |
| notes repo (`~/desi_gpy_dla_notes`) | `e3930cb` |
| protected: `hbi-mcmc-threeroute` / `lls-subdla-cddf` / `review/phaseA…` / `repair/phaseB…` | `9d73365` / `1533333` / `a420abd` / `a56e3c8` (unmoved) |
| quarantined artifact | `track_c/stage0/quarantined_forward_response_2lpt0_phaseC.npz` |
| holdout | prod_v1 roles: 13 healpix / 661 injections — UNREAD |

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
3. **Tier 2 PARTIAL (`ed6865e`) — read `t2_findings.md`:** (a) Level A:
   the deployed SURFACE misfits its OWN pairs (+0.071 at [19.5,19.8),
   −0.035 at [20.4,20.7)) — much of the low-N bridge Δ was surface
   representation error (bridge FAIL stands: it tested what production
   uses); (b) blends quantified (7.5–8% of pairs, +0.03..+0.10 dex,
   secondary); (c) 🔴 SURVIVING: pair-level natural−injected offset
   +0.00..+0.06, largest +0.059 at 21.0 with BOTH selections ≥98%
   complete — candidates: host-environment forest coupling / imprint
   realism (§18 class). Stopping-rule criterion 3 NOT met (~700 counts
   on G3 if coherent over [20.7,21.1)).
4. **Next executable step — Tier 2 completion:** per-stratum
   paired/common-substrate comparisons of the 20.7–21.3 offset
   (same-healpix, same-SNR-stratum, same-z natural-vs-injected pair
   subsets; prespecified reweighting on PRE-selection covariates only:
   z_true, S2N_RED, z_qso); test the offset's dependence on host N and
   on local catalogued absorber density (host-coupling signature) vs
   flatness (imprint-realism signature). If catalogs cannot separate
   them → Tier-3 spectrum-level need goes BACK TO THE PI with a budget
   (§21: forced-fit/stack campaign not authorized). P1 estimand freeze
   BLOCKED until this attribution/bounding lands.
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
