# Production recommendation — SNR_RED > 2 operating point

> Validation set: London mock-0 (multi-DLA mode), 8 spectra-16 files (~6.6k spectra) for sweeps,
> 32 files (26k spectra) for baseline. Truth: NHI ≥ 20.3. Full forest λ_rf ∈ [911, 1216] Å.
> BAL-excluded. P_DLA-cut family swept; SNR_RED > 2 fixed.
> P/C from Molly-faithful pipeline (`examples/molly_faithful_pc_plots.py`).

---

## At a glance

**Headline finding**: at **SNR_RED > 2**, no tested configuration passes 85% / 85%.
The limiter is *completeness*, not purity. Best achievable balance with current code
is at the **PW14 50k + τ-EB + lyb_veto** stack:

| Operating point at SNR>2 (full forest) | Purity | Completeness | Notes |
|---|---:|---:|---|
| **Balanced (recommended for cosmology DLA catalog)** P_DLA ≥ 0.99 | **83.6%** | **74.3%** | Best (P+C)/2 with the full stack |
| **High purity** P_DLA ≥ 0.99999 | **85.4%** | 63.5% | Crosses 85% purity; completeness drops |
| **Very high purity** P_DLA ≥ 0.999999 | **86.9%** | 62.3% | Diminishing returns on purity |
| Most permissive baseline P_DLA ≥ 0.99 (no extras) | 75.0% | 82.1% | What Molly's 2509 notebook reports |

**Why no 85/85 at SNR>2**: a sub-agent investigation traced the gap to a structural
bug in `FILTER_LOW_LIKELIHOOD=1` at low SNR. 74% of missed low-SNR DLAs get
P(Null) > 0.9 (rejected entirely, not stolen by sub-DLA). See "Code follow-ups"
below — the fix is in the GP, not in the catalog cuts.

---

## Recommended production knobs (at SNR_RED > 2)

For the **DLA catalog** (priority: purity > completeness, your stated goal):

```bash
ENABLE_TAU_EB=1
TAU_EB_OBJECTIVE=null
NUM_DLA_SAMPLES=50000
DLA_SAMPLES_FILE=data/dr12q/processed/pw_samples_a3_190_220_50000.mat   # PW14 NHI 19-22 joint prior
NUM_SUBDLA_SAMPLES=100000
SUB_DLA_SAMPLES_FILE=data/dr12q/processed/subdla_samples_a03_191_200_100000.mat
MAX_DLAS=3              # max_dlas=6 test in progress (eta ~20:30 today)
FILTER_LOW_LIKELIHOOD=1
# Post-process: apply lyb_veto with dz_match=0.005
# Science cut: P_DLA ≥ 0.99   (recommended for completeness-balanced catalog)
# Science cut: P_DLA ≥ 0.99999 (alternative for purity-first catalog)
```

**Cost estimate** (1M QSO production, PARALLEL_FILES=32, MAX_WORKERS=8):
- Baseline (10k, no τ-EB):          ~20 node-hr
- + PW14 50k:                        ~22 node-hr
- + τ-EB ON:                         ~27 node-hr
- + lyb_veto (post):                 negligible
- **Total**: ~**27 node-hours** — comfortably under 50 hr budget.

---

## Full SNR > 2 sweep (all stacks)

| Config | snr | P_DLA cut | Purity | Compl | Δ_P vs base | Δ_C vs base |
|---|---:|---:|---:|---:|---:|---:|
| 26k baseline (10k, no τ-EB) | 2 | 0.99 | 0.7504 | 0.8206 | — | — |
| 26k baseline | 2 | 0.999 | 0.7779 | 0.7922 | +2.8 | −2.8 |
| 26k baseline | 2 | 0.99999 | 0.8069 | 0.7209 | +5.7 | −10.0 |
| 26k baseline | 2 | 0.9999999 | 0.8260 | 0.6587 | +7.6 | −16.2 |
| 26k base + lyb_veto | 2 | 0.99 | 0.7592 | 0.8198 | +0.9 | −0.1 |
| 26k base + lyb_veto | 2 | 0.99999 | 0.8187 | 0.7201 | +6.8 | −10.0 |
| 8f PW14 50k | 2 | 0.99 | 0.7901 | 0.7924 | +4.0 | −2.8 |
| 8f PW14 50k | 2 | 0.99999 | 0.8310 | 0.6901 | +8.1 | −13.1 |
| 8f PW14 50k + lyb_veto | 2 | 0.99 | 0.7924 | 0.7924 | +4.2 | −2.8 |
| 8f τ-EB | 2 | 0.99 | 0.8084 | 0.7895 | +5.8 | −3.1 |
| 8f τ-EB | 2 | 0.99999 | 0.8339 | 0.6608 | +8.4 | −16.0 |
| 8f τ-EB + lyb_veto | 2 | 0.99 | 0.8108 | 0.7895 | +6.0 | −3.1 |
| 8f τ-EB + lyb_veto | 2 | 0.99999 | 0.8370 | 0.6608 | +8.7 | −16.0 |
| **8f PW14 + τ-EB** | 2 | 0.99 | **0.8301** | 0.7427 | +8.0 | −7.8 |
| **8f PW14 + τ-EB + lyb_veto** | 2 | 0.99 | **0.8355** | 0.7427 | +8.5 | −7.8 |
| **8f PW14 + τ-EB + lyb_veto** | 2 | 0.999 | 0.8399 | 0.6901 | +9.0 | −13.1 |
| **8f PW14 + τ-EB + lyb_veto** | 2 | 0.99999 | **0.8543** | 0.6345 | +10.4 | −18.6 |
| **8f PW14 + τ-EB + lyb_veto** | 2 | 0.999999 | **0.8694** | 0.6228 | +11.9 | −19.8 |
| **8f PW14 + τ-EB + lyb_veto** | 2 | 0.9999999 | 0.8793 | 0.5965 | +12.9 | −22.4 |

**Pattern**: every knob (PW14, τ-EB, lyb_veto) trades 2-4 pp purity gain for completeness loss
beyond a certain P_DLA cut. The combined stack maximizes the achievable purity (~88%) but
at the cost of ~60% completeness.

---

## SNR > 2 versus SNR > 4 versus SNR > 6 — the deeper story

| SNR cut | Stack | P_DLA | Purity | Completeness | Passes 85/85? |
|---|---|---:|---:|---:|---|
| > 2 | PW14+τ-EB+lyb_veto | 0.99999 | 85.4% | 63.5% | ✗ (C ceiling) |
| > 4 | PW14+τ-EB+lyb_veto | 0.99999 | 84.8% | 83.9% | ✗ (just barely) |
| > 4 | PW14+τ-EB+lyb_veto | 0.999999 | 86.0% | 82.4% | ✗ |
| > 6 | PW14+τ-EB+lyb_veto | 0.999999 | **85.0%** | 83.8% | ✗ (C 1.2 pp short) |
| > 6 | PW14+τ-EB+lyb_veto | 0.99999 | 84.1% | 85.9% | ✗ (P 0.9 pp short) |

**At SNR > 4**, the combined stack approximately straddles 85/85 — closer to your target.
**At SNR > 6**, the operating point is the closest we've measured. Neither passes both
metrics, but both are within 1-2 pp.

---

## Code follow-ups — **all three REFUTED by 2026-05-12 verification agents**

The original three proposed fixes have been **empirically refuted**. See
detail reports under `/pscratch/sd/j/jibancat/prod533_5k_20260511/`:

| Fix | Verdict | Evidence |
|---|---|---|
| #1 z_tol + 802-807 early-stop | **REFUTED** | `hypothesis1_sweep/RESULTS.md` — Pop B (118 missed in NHI [20.3, 20.6)×SNR [2,4)): V1 z_tol recovers 0/118 (+0.00 pp); V2 early-stop removal recovers 1/118 (+0.85 pp), well below the predicted +3-5 pp. The single recovery (TID 47767) is a 6-absorber spectrum where 2-DLA evidence is +11 logL above 1-DLA. Pop A: bit-identical across all variants because line 619 (no-valid-regions early-exit) fires first. |
| #2 FILTER=0 for low-SNR | **REFUTED** | `filter_sweep/RESULTS.md` — Δ changes by ≤0.4 logL on all 5 candidates, never crosses zero. Filter is downstream of the actual limiter. |
| #3 n_initial bump | **REFUTED** | `filter_sweep/RESULTS.md` — bumping `n_initial` from 5000 → 50000 (i.e. full QMC bag) still gives `n_initial_above_null = 0` for every candidate. |

**Actual mechanism (per `dilution_test/RESULTS.md`)**: for 3 of 5 candidates the
GP+DLA model **does not prefer DLA at truth** — max single-sample logL above null
inside a ±0.002×±0.05 box around truth is only +0.09 to +2.17 (TID 105798 is
*below* null by −3.04). The previous prior-subagent's claim of `Δ_at_truth ∈
[+3.7, +7.7]` is not reproduced. Prior-shape and sampling fixes therefore cannot
help these targets; the limiter is the forward model itself.

**One narrow, real fix surfaced**: gate the 802-807 early-stop on
`n_initial_above_null > 0` so genuine multi-absorber spectra (like TID 47767)
aren't blocked from computing 2-DLA evidence. This is a code-path bug, not a
completeness lift — at most a fraction of a pp.

**Hypotheses now tested (2026-05-12)**:

| Hypothesis | Outcome |
|---|---|
| τ-EB applied per-spectrum | **REFUTED** for the 5 candidates: mean ΔΔ_at_truth = −0.48 logL. τ-EB picks factor 3-4× the seed but lifts log_l_null and log_l_DLA by the same amount. (`tau_eb_5cand/RESULTS.md`) |
| **v3 phase2_desi `2lpt_loa124_nohcd_nobal_wide` GP model on London 8f population** | **POSITIVE — DEPLOYABLE.** At P_DLA≥0.99 SNR>2: **P=84.52%/C=76.61%** (vs baseline 83.55%/74.27%): +1.0pp purity, +2.3pp completeness. At P_DLA≥0.99999: 88.7%/69.0% vs 85.4%/63.5% (+3.3pp/+5.6pp). v3 wins at every P_DLA cut. (`v3_population/RESULTS.md`-equivalent, `molly/london_v3_loa124_pw14_tau_eb_8f/`) |
| MAP-based detection (blind, with Laplace) | **REFUTED as standalone**. Blind MAP recovers Δ_MAP ∈ [+8.6, +11.6] on the 5 candidates, but on n=48 null spectra has 31% FP rate at the candidate-min Δ_MAP. Laplace 22% FP. *However*: the existing prior-marginal Δ retains the ranking signal — switching to null-quantile threshold (`Δ_marg > p90(null) ≈ −7.8`) recovers candidates with ~10% FP. (`map_detection_test/RESULTS.md`) |

### Updated production recommendation at SNR > 2

```bash
LEARNED_FILE=/pscratch/sd/j/jibancat/prod533_5k_20260511/null_gp_test/converted/2lpt_loa124_nohcd_nobal_wide.h5   # v3 model
ENABLE_TAU_EB=1                  # keep — population-scale benefit from PR #5 stands
TAU_EB_OBJECTIVE=null
NUM_DLA_SAMPLES=50000
DLA_SAMPLES_FILE=data/dr12q/processed/pw_samples_a3_190_220_50000.mat
NUM_SUBDLA_SAMPLES=100000
SUB_DLA_SAMPLES_FILE=data/dr12q/processed/subdla_samples_a03_191_200_100000.mat
MAX_DLAS=3
FILTER_LOW_LIKELIHOOD=1
# Post-process: apply lyb_veto with dz_match=0.005
# Optional: switch from absolute p_DLA>0.99 to null-quantile-calibrated cut
```

**Headline at SNR > 2 with v3_loa124 stack**:

| P_DLA cut | Purity | Completeness |
|---|---:|---:|
| ≥ 0.99 | **84.5%** | 76.6% |
| ≥ 0.999 | 85.5% | 74.0% |
| ≥ 0.99999 | **88.7%** | 69.0% |

Still below 85/85 jointly, but the gap has narrowed significantly. The remaining
~9 pp on completeness is genuinely about peak narrowness (verified by Laplace
volume-penalty matching the marginal-MAP gap exactly) — addressable only via
peak-aware sampling or null-calibrated thresholds, not via prior or sample-count
or mean-flux fixes.

---

## In-flight (status as of writing)

- **max_dlas=6** + PW14 50k + τ-EB run on London 8 files: launching now, ETA ~1.5 hr.
  Expected to lift completeness specifically (more multi-absorber LOS captured) without
  trading purity. Will update this doc when done.

- Combined-stack cut sweep complete (file: `sweep_pw14_tau_eb.log`).

---

## 2026-05-12 update — narrow-prior [20, 21] test: **NEGATIVE**

The sub-agent grid-scan from the 2026-05-11 session predicted that swapping the DLA
NHI prior from PW14 [19, 22] to PW14 [20, 21] (50k samples) would lift completeness
by 8–10 pp at SNR > 2. The full 8-file London run (`london_pw14_2021_tau_eb/`,
36 min wall, 712 catalog rows / 618 truth rows post-cut) **refutes that prediction.**

| Config @ SNR>2 (lya_lyb window) | P_DLA cut | Purity | Compl |
|---|---:|---:|---:|
| PW14 [19,22] 50k + τ-EB + lyb_veto **(baseline)** | 0.99    | **83.6%** | **74.3%** |
| PW14 [20,21] 50k + τ-EB + lyb_veto **(narrow)**   | 0.99    | **77.1%** | **74.9%** |
| PW14 [19,22] 50k + τ-EB + lyb_veto                | 0.99999 | **85.4%** | 63.5% |
| PW14 [20,21] 50k + τ-EB + lyb_veto                | 0.99999 | **78.5%** | 62.9% |

Purity drops uniformly by 6–7 pp across **every** P_DLA cut. Completeness is
effectively unchanged. The narrow [20, 21] prior is a regression.

**Likely mechanism**: a uniform prior over a tight NHI range concentrates posterior
weight on the DLA model versus the (unchanged) sub-DLA / null models, inflating
false positives at sub-DLA NHI and at noise / BAL residuals. The brute-force prior
narrowing trades incompleteness for impurity rather than fixing the underlying
sampling issue.

**Implication**: the deeper fix the user flagged — 2D-stratified QMC where the
(z_offset, log NHI) sample density adapts to where the likelihood is sharp —
is **not optional**. Narrowing the prior alone introduces a different bias and
fails to recover the missed candidates.

Next experiments to consider (in order of expected impact):
1. **FILTER_LOW_LIKELIHOOD=0 ablation** on the 5 known-missed candidates (running
   in background as of 2026-05-12). Directly tests whether the bug is in FILTER
   or in the sampling density.
2. **Sample bump to 100k** at the *original* PW14 [19, 22] — isolates whether
   the early-exit-branch firing rate is the limiter.
3. **Stratified QMC**: oversample log NHI ∈ [20, 20.5] within the [19, 22] prior;
   importance-reweight in the evidence estimator. The principled fix.

Artifacts:
- `london_pw14_2021_tau_eb/` — 8 processed h5 + 8 dlacat fits, 36 min wall
- `molly/london_pw14_2021_tau_eb_8f/` — molly plots + tsv summary
- `molly_narrow_prior.log` — full molly stdout
