# Phase-C independent forest-FP expansion — design + measured costing

**Status: DESIGN (Phase C1). Production generation is Phase C2 — NOT
authorized (no documented compute envelope; rulings §12).** Companion to
`docs/PHASEC_CALIB_DESIGN.md`. Implements rulings §15: substantially more
GENUINELY INDEPENDENT forest-FP events, with the three data roles kept
separate and every realization's role frozen before closure is examined.

## 1. Current state (measured)

* 89 op-cut in-support FP events from 2,255 searched loa-0 (mock-0)
  sightlines on 4 healpix (`gl_loa0_fp_v1_20260615`) → ±10.6% on the FP
  total (Poisson), the dominant FP-calibration uncertainty.
* Measured unit costs (production logs): ~51 CPU-h/healpix, ~167
  CPU-s/spectrum ⇒ **≈ 2.29 CPU-h per in-support FP event** (204 CPU-h /
  89 events, aggregate — the planning unit; the first new healpix batch
  re-measures it).
* Event rate ≈ 0.0395 in-support events per searched sightline (loa-0).
  Family rates may differ; ±50% planning band until measured.

## 2. Substrate inventory (verified on disk 2026-08-06)

| substrate | family | HCD content | role eligibility |
|---|---|---|---|
| `lyacolore_2lpt/.../mock-0/loa-0` | 2LPT | absorber-free twin | common-reference (4 healpix already spent; ~226 unused) |
| `lyacolore_2lpt/.../mock-1/loa-0` | 2LPT | absorber-free twin, **independent realization** | held-out evaluation / realization-level independence measurement |
| `saclay/.../mock-0/jura-0` | Saclay | absorber-free twin | family transport control (direct) |
| `london/.../mock-0/jura-124` | London | with-HCD only (**no twin on disk**) | family control via natural HCD-free sightlines ONLY (see §4) |
| `saclay/.../mock-0/juraLy8-124` | Saclay | with-HCD | source of the natural-HCD-free method-bias pair (§4) |

## 3. The three FP data roles (frozen at generation; §15.2)

1. **Common-reference production calibration:** loa-0 (2LPT mock-0), NEW
   healpix disjoint from the 4 spent — the same absorber-free definition
   the current 89-event product uses, so events POOL with the existing
   ones under the existing product definition (op cut, ell_eff
   normalization, (1−η) handling unchanged).
2. **Family-specific transport controls:** (a) Saclay: `jura-0` directly
   — same definition on an independent family; (b) London: natural
   HCD-free sightlines of `jura-124` (no twin exists), corrected by the
   METHOD-BIAS PAIR measured in Saclay (§4). Controls measure transport
   (amplitude, shape, z, SNR); they are never production fits.
3. **Held-out evaluation:** 2LPT `mock-1/loa-0` — a whole independent
   realization reserved for the Phase-C3 unchanged-statistic prediction
   evaluation; additionally, within every role, whole-healpix blocks are
   assigned before any GP run and recorded in a committed role manifest.

Freezing rule: the role manifest (substrate → healpix list → role) is
committed BEFORE the first Stage-2 sbatch; no reassignment afterwards;
leave-one-realization-out uses mock-0 vs mock-1 only along the frozen
split.

## 4. The London problem and the Saclay method-bias pair

A natural-HCD-free selection inside a with-HCD mock is a BIASED forest
sample (HCD placement correlates with density). Saclay uniquely has BOTH
an absorber-free twin (`jura-0`) and a with-HCD box (`juraLy8-124`), so
the bias of the natural-selection method is DIRECTLY measurable there:

    Δ_method(Saclay) = FP calibration on natural-HCD-free juraLy8-124
                       sightlines − FP calibration on jura-0

Δ_method is then carried to the London natural control as a measured
method systematic (with its own uncertainty), NOT absorbed into any
covariance. If Δ_method proves large compared to the transport effects
being measured, the London control is demoted to descriptive and the
acquisition of a London twin becomes a PI infrastructure question.

## 5. Event and uncertainty targets (measured cost; §15.4)

Naive-Poisson sizing (refined against measured covariance and effective
sample size before Stage 2 finalizes):

| target | total events | new events | new sightlines | est. CPU-h |
|---|---|---|---|---|
| ±5% | 400 | +311 | ≈ 7,900 | **≈ 710** |
| ±3% | 1,111 | +1,022 | ≈ 25,900 | ≈ 2,340 |

Plus (either target): Saclay control ~100 events ≈ 230 CPU-h; Saclay
method-bias pair ~100 events ≈ 230 CPU-h; London natural control ~100
events ≈ 230 CPU-h; mock-1 held-out ~150 events ≈ 340 CPU-h. **FP program
totals: ≈ 1,740 CPU-h (5% target) / ≈ 3,370 CPU-h (3% target).** With the
response campaign (~150 CPU-h) the full Phase-C2: **≈ 1,900 (5%) or
≈ 3,500 (3%) CPU-h** — both inside the ~5,000 CPU-h allocation cap, both
far above the ~500 CPU-h PI sign-off line, neither authorized until the
PI sets the envelope (§12).

Storage: dlacat+logs only (~1 GB/10k sightlines; no posterior stores).

## 6. Independence accounting (§15.3)

Reported with every batch: raw event count; distinct sightlines; distinct
healpix; distinct realizations; effective sample size from the
sightline-block bootstrap (the spec §4 machinery, reused unchanged);
duplicated-structure list (none expected: each sightline searched once
per role; the 4 spent healpix stay common-reference and are never
re-counted). Bootstrap draws are NEVER counted as events (§15.1); every
uncertainty statement separates event-limited from MC-limited error.

## 7. Sequential stopping rule (prespecified; §15.4)

After each batch of ~8 healpix (~100 expected events), compute, on the
POOLED common-reference sample: (i) FP-total precision; (ii) the change
in the top-3 E_cov eigenvalues vs the previous batch (<5% required twice
consecutively); (iii) Layer-B null q95/q99 stability (<2% drift twice
consecutively); (iv) the conditional N̂-profile χ² between consecutive
halves (stable); (v) the transport-vs-noise separation criterion:
family-control amplitude shifts resolved at ≥3σ OR bounded below
scientific materiality. STOP at the first batch where (i) meets the
authorized target AND (ii)–(iv) are stable — closure outcomes are never
consulted (§15.4). If the PI authorizes the 5% envelope, the expected
stop is 3–4 batches.

## 8. Monte Carlo precision targets (§15.5 restated as implementation)

Independent seed streams (extending the frozen 41001/43001/42001 family):
covariance 45001+, null 46001+, operating-characteristic 47001+, final
evaluation 48001+. Required: 95% MC half-width ≤ 0.002 near α = 0.01 (⇒
B_null ≥ ~24,000 for the operating study — cheap, numpy-only); ≤ 0.005
near 0.05; ≤ 0.02 for power curves (B ≥ ~600/point); eigenvalue
stability ≤ 5% across independent halves; unresolvable tails reported as
bounds (the existing p_is_bound convention).

## 9. What this design does NOT do

No bootstrap-as-events; no family-specific production fits (controls
measure transport only); no t-prior change; no absorption of transport
bias into covariance; no reuse of held-out realizations for calibration
or tuning; no real-DESI data anywhere.
