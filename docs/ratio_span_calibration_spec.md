# Prospective calibration spec — `ratio_span_by_z` and `ratio_span_by_snr`

**Status: UNRATIFIED.** The PI declined (decision 8, 2026-07-29) to ratify
`ratio_span_by_z_max = 0.10` and `ratio_span_by_snr_max = 0.15`, and asked that
the two statistics be *defined and calibrated prospectively*. This document is
the definition and the calibration procedure. The authoritative machine-readable
ratification record is `CDDF_analysis/hbi_mcmc/ratification.py`; the statistic
itself is `CDDF_analysis/hbi_mcmc/forward_selftest.ratio_span`; the null sampler
is `forward_selftest.ratio_span_null`.

What the code does *today*, as of this branch:

* both statistics are **computed on every run** and **stamped into every
  artifact** (`forward_gate.ratio_span_by_z`, `..._by_snr`, plus
  `..._detail`);
* they **do not contribute to pass/fail**. Exceeding the proposed number
  produces an entry in `forward_gate["advisories"]`, never in
  `forward_gate["failures"]`, and `pass` is unaffected;
* the artifact says so explicitly, so that "the span arm did not fire" can
  never be read as "the span arm passed".

The alternative behaviours were (a) keep the invented threshold armed — that is
exactly what the PI declined; (b) delete the arms — rejected, because the
reported values **are** the calibration data and deleting them means the
calibration can never be run. (c) compute-report-do-not-gate was adopted.

> 🔴 **CORRECTION (2026-07-29), read before quoting any number from §4.** An
> earlier version of this paragraph said option (a) "refuses a *third of
> perfectly correct forward models*", full stop. That measurement is real but
> it is a property of **one geometry**: the 5×4×2 calibration pack of §4,
> whose `by_z` arm has **FOUR fine-z rows**. §1.1 item 1 of this very document
> says a range statistic's null grows with the row count and that "a 15-bin
> fine-z arm and a 4-bin one do not share a threshold" — and the first
> calibration then quoted the 4-row number as though it were the tolerance's.
> On production-scale geometries the same measurement gives `by_z` false-alarm
> rates of **0.0893** (17×15×8) and **0.0819** (29×15×8). The *pair-mismatch*
> conclusion survives everywhere (`by_snr` is inert, FAR = 0.0000, on both
> production grids); the *magnitude* does not. **§4.1 is the production-scale
> table and the power comparison, and it is what a disarm/re-arm decision
> should be read off.** Every quote of a span false-alarm rate in this
> document now names its grid.
>
> Two earlier drafts of this same paragraph carried **0.0855 / 0.0810** instead.
> Those digits came from a lower draw count and are not reproducible from the
> committed artifact; they are retracted here. Everything quoted below is
> `n_draws = 20000`, `seed = 1` for the null and `4000` for the power curves,
> which is the setting `ratio_span_null_calibration.json` is generated at, and
> that artifact — not this document — is authoritative.

---

## 1. Exact definition

Let the forward self-test fold the pack's own truth `f_truth` through the pack's
own kernel at the truth-equivalent parameter point, giving predicted expected
counts `mu[c,k,s]` on the (observed-N-hat) × (fine-z) × (SNR) grid, against the
pack's recorded `obs[c,k,s]`. Cells with `dX[k,s] == 0` are zeroed out of both
before any sum (see `forward_selftest.ratio_tables`).

For one **arm** — a marginal of that grid —

| arm | one row per | summed over |
|---|---|---|
| `by_z`   | fine-z bin  | N-hat and SNR |
| `by_snr` | SNR stratum | N-hat and fine-z |

with row totals `mu_r`, `obs_r`, define the kept row set

```
R+ = { r : obs_r > 0  and  mu_r / obs_r finite }
ratio_r = mu_r / obs_r
```

and

```
ratio_span(arm) = max_{r in R+} ratio_r − min_{r in R+} ratio_r      if |R+| ≥ 2
ratio_span(arm) = 0                                                  otherwise
```

It is a **range** (max minus min), dimensionless, *not* a variance, *not* a
standard deviation, and *not* normalised by anything. `mu` is a deterministic
function of the pack: no posterior, no sampler, no estimator is involved.

### 1.1 Four properties that make a guessed threshold indefensible

1. **A range's null distribution grows with the number of rows.** For roughly
   Gaussian rows the expected range grows like `2·sd·sqrt(2 ln n)`. A 15-bin
   fine-z arm and a 4-bin one do not share a threshold. Neither 0.10 nor 0.15
   was chosen with any row count in mind.
2. **The rows are heteroscedastic.** The per-row Poisson sd of `ratio_r` is
   `≈ mu_r / obs_r^{3/2} ≈ 1/sqrt(obs_r)`. The range is therefore dominated by
   the *emptiest* row, for purely statistical reasons. The statistic mixes a
   real shape systematic with the noise of the thinnest stratum.
3. **`obs` is in the denominator.** Poisson `obs_r` has positive mass at 0, so
   `E[ratio_r]` does not exist in the strict sense; rows with `obs_r = 0` are
   dropped, making the statistic conditional on which rows happened to be
   non-empty.
4. **`|R+| < 2` returns 0, i.e. passes vacuously.** On a single-SNR-stratum grid
   (for example the SBC grid, `snr_edges = [0, inf]`) `ratio_span_by_snr` is
   *identically* 0 and its arm has never been able to fire. A vacuous 0 is
   indistinguishable downstream from a measured 0. `ratio_span` now reports
   `vacuous: true` in that case; the gate does not yet distinguish them because
   the arm does not gate.

A better-conditioned statistic would be the range or sd of `log(mu_r/obs_r)`, or
the dispersion of `poisson_z` across rows — both have a stable null and neither
divides by a Poisson count. **Nothing here silently redefines the deployed
statistic**: the calibration below calibrates what production actually computes.
Replacing the statistic is a PI decision, listed as option B in §6.

---

## 2. The null hypothesis

> **H0 — "the forward model is right."** The folded prediction `mu` *is* the true
> expected count in every cell; the nuisances (`psi_c`, `psi_k`, `t`, `lam_fp`)
> are held fixed at the point values the fold used; the observed cell counts are
> independent Poisson draws about `mu`.

```
obs*[c,k,s] ~ Poisson(mu[c,k,s]),  independently over cells
obs*[c,k,s] = 0   wherever dX[k,s] == 0
```

Each replicate is then marginalised exactly as `ratio_tables` marginalises, and
`ratio_span` is recomputed. The resulting distribution is the span produced by
**pure counting noise with a perfect forward model**, at this pack's row count
and this pack's per-row exposure. It is therefore **pack-dependent**, which is
the first reason a global constant cannot be right.

### 2.1 What the null omits (it is a *lower bound* on the null width)

* nuisance uncertainty (completeness, transfer factors, FP rate);
* response-coefficient (`resp_*_coef`) uncertainty, including `resp_fitcov_diag`;
* the Monte-Carlo error of `f_truth` itself — `mu` is built from a finite-count
  truth estimate and is treated here as exact;
* any real overdispersion of the counts (absorber clustering along sightlines
  correlates cells that this null treats as independent).

Every omission narrows the null, so a threshold read off it fires **more** often
than the stated false-alarm rate. This is anti-conservative and must be stated
whenever a threshold from §3 is quoted.

---

## 3. Sampling procedure and how a threshold would be set

Implemented: `forward_selftest.ratio_span_null(pack, n_draws=..., seed=...,
resp_clamp=...)`. Pure numpy on top of a single fold — measured **0.04 s for
2000 replicates** on the 5×4×2 synthetic pack (§4), so `n_draws = 20000` is
seconds and the calibration costs no allocation.

Threshold proposal, to be brought to the PI *with its measured false-alarm
rate*, never without:

1. Fix the pack and the run configuration (grid, `resp_clamp`, nuisance point
   values). The threshold is a property of that configuration, not a constant.
2. Draw `n_draws ≥ 20000` replicates of `obs*` under H0 and form the null
   sample `{span*_1 … span*_n}` per arm.
3. Choose a **family-wise** false-alarm rate `alpha_FW` for the pair of arms.
   Recommended `alpha_FW = 0.01`, split Bonferroni: `alpha = 0.005` per arm.
   (The two arms are correlated — they marginalise the same cells — so
   Bonferroni is conservative in the direction of *not* refusing work, which is
   the correct direction for a tripwire.)
4. Set `ratio_span_<arm>_max = quantile(span*, 1 − alpha)`.
5. Report, in the ratification request: the pack, `n_draws`, the seed, the row
   count and the median non-empty row count per arm, the null quantiles at
   {0.5, 0.9, 0.95, 0.99, 0.995, 0.999}, the proposed threshold, the measured
   false-alarm rate *at that threshold*, and the omissions in §2.1.
6. **Power.** A threshold with a stated false-alarm rate and no measured power
   is half a calibration, and this project has been burned by exactly that (see
   `feedback_coverage_tests_need_power_checks`). Alongside step 4, measure the
   detection curve: inject a known multiplicative z-tilt
   `mu_r → mu_r · (1 + d · (z_r − z̄)/Δz)` for `d` on a grid, and report the
   fraction of replicates exceeding the threshold as a function of `d`. Quote the
   smallest `d` detected at 50% and at 90%. If that `d` is larger than the
   systematic the arm was invented to catch, the arm does not do its job and
   should be replaced (§6) rather than ratified.
7. Only then does the entry move from `UNRATIFIED` to `RATIFIED` in
   `ratification.py` — a one-line edit, with **no change to the arm**, because
   the arm already computes and reports the statistic.

---

## 4. Measured result on the 5×4×2 calibration pack (**4 fine-z rows**)

**This section's numbers apply to a four-row `by_z` arm and to nothing else.**
For the production geometry go to §4.1.

Pack: `synthetic_pack(0, nhat_edges=19.9…20.4 step 0.1, zf_edges=2.0…2.4 step
0.1, zc_edges=[2.0,2.2,2.4], snr_edges=[0,3,inf], n_molly_cells=3,
fp_frac=0.15, t_true=[0.2,−0.15])`; `resp_clamp="both"`. Total `mu = 2283.95`
against total `obs = 2274`. `n_draws = 20000`, `seed = 1`.

Geometry of this table: the **5×4×2** calibration pack — `by_z` has **4 rows**,
`by_snr` has 2. Nothing below transfers to another geometry (§4.1):
| arm | rows | null q50 | null q95 | null q99 | null q99.5 | proposed max | **measured false-alarm rate at the proposed max** |
|---|---|---|---|---|---|---|---|
| `by_z`   | 4 | 0.0841 | 0.1574 | 0.1923 | 0.2058 | **0.10** | **0.3434** |
| `by_snr` | 2 | 0.0286 | 0.0825 | 0.1099 | 0.1174 | **0.15** | **0.0003** |

Read that table twice. Under a null in which the forward model is *exactly
right*, **on this 5×4×2 pack (four fine-z rows)**,
`ratio_span_by_z_max = 0.10` refuses a fraction **0.3434** of runs while
`ratio_span_by_snr_max = 0.15` refuses **0.0003** — the two numbers, presented as
a matched pair with the SNR one "wider because the strata are noisier", differ in
false-alarm rate by three orders of magnitude, and in the *opposite*
direction to the stated rationale. (An earlier draft of this sentence rendered
the second as "0.02%". `0.0003` is 0.03%. Percentages are not used here for
exactly that reason: the artifact stores fractions, and a hand-converted second
representation is a number nobody re-derives.) This is what an uncalibrated tolerance inside
a production fail-closed gate looks like.

**Do not carry the 34% out of this section.** It is the four-row figure; §4.1
measures the fifteen-row production arm, where it is ~0.08.

The same fold's *observed* `by_z` span on this pack is 0.1734 (`by_snr`:
0.0164), which the arm
would have reported as a failure and which the null puts at roughly the 98th
percentile of pure counting noise — suggestive, not decisive, and precisely the
sort of claim that needs a calibrated threshold before it can refuse anything.

At the Bonferroni `alpha = 0.005` of §3 the *pack-specific* thresholds would be
`by_z ≈ 0.206`, `by_snr ≈ 0.117` (the q99.5 column above). These
are **not** proposed for ratification here: they come from a synthetic pack and
they inherit every omission in §2.1. The step-6 power curve is now measured —
§4.1 — but on synthetic packs only, so this does not become a proposal.

Provenance: these numbers come from the committed routine
`forward_selftest.ratio_span_null_report`, not from a scratch script, and are
stamped in `CDDF_analysis/hbi_mcmc/ratio_span_null_calibration.json` with the
full 40-char SHA. Reproduce (measured 1 m 15 s / 1 m 16 s / 1 m 32 s wall over three runs, 1.6 GB
peak RSS, one core; the 1 m 32 s run was the one under `/usr/bin/time -v`):

```
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m CDDF_analysis.hbi_mcmc.forward_selftest --ratio-span-null \
    --null-draws 20000 --null-seed 1 \
    --out CDDF_analysis/hbi_mcmc/ratio_span_null_calibration.json
python -m pytest tests/test_gate_ratification.py -q -k ratio_span_null
```

---

## 4.1 Production-scale geometries, and the detection curve for **both** guards

Everything in this section is `n_draws = 20000`, `seed = 1` for the null and
`n_draws = 4000`, `seed = 1` for the power curves, from the `geometries` and
`power` blocks of `ratio_span_null_calibration.json` (schema
`ratio_span_null_calibration/v2`). This is the section a disarm/re-arm decision
should be read off; §4 is a four-row pack and is not.

**Controlled comparison.** All three packs are built with `t_true = 0` in every
coarse-z bin, so the **only** thing that varies down the table is the geometry.
That is why `calib_5x4x2` reads `by_z` FAR `0.3438` here while §4 — same grid,
but the `t_true = [0.2, −0.15]` tilt of the 5×4×2 calibration pack — reads
`0.3434`: two different folds of the same geometry, agreeing to Monte-Carlo
error, which is the point.

| geometry | `n_nhat`×`n_zf`×`n_snr` | total `mu` | arm | rows | q50 | q95 | q99 | q99.5 | proposed max | **measured FAR** |
|---|---|---|---|---|---|---|---|---|---|---|
| `calib_5x4x2`   | 5×4×2   | 2289.75  | `by_z`   | 4  | 0.0840 | 0.1563 | 0.1914 | 0.2057 | **0.10** | **0.3438** |
| `calib_5x4x2`   | 5×4×2   | 2289.75  | `by_snr` | 2  | 0.0286 | 0.0825 | 0.1087 | 0.1176 | **0.15** | **0.0004** |
| `prod_17x15x8`  | 17×15×8 | 35024.53 | `by_z`   | 15 | 0.0740 | 0.1067 | 0.1233 | 0.1292 | **0.10** | **0.0893** |
| `prod_17x15x8`  | 17×15×8 | 35024.53 | `by_snr` | 8  | 0.0446 | 0.0703 | 0.0838 | 0.0886 | **0.15** | **0.0000** |
| `prod_29x15x8`  | 29×15×8 | 35749.69 | `by_z`   | 15 | 0.0734 | 0.1054 | 0.1213 | 0.1289 | **0.10** | **0.0819** |
| `prod_29x15x8`  | 29×15×8 | 35749.69 | `by_snr` | 8  | 0.0442 | 0.0696 | 0.0817 | 0.0858 | **0.15** | **0.0000** |

Two things move and one does not:

* the `by_z` false-alarm rate at `0.10` falls by a factor ~4 from the 5×4×2
  calibration pack to production. The *number* in §4 is therefore a property of
  that pack, not of the tolerance — the whole of defect 2;
* the **pair mismatch that justified the PI's refusal survives intact**. On both
  production grids `by_snr = 0.15` sits above even the q99.9 of its own null and
  fires at `0.0000`, while `by_z = 0.10` fires at ~0.08. A "matched pair" whose
  two arms differ in false-alarm rate by more than two orders of magnitude — in
  the direction opposite to the stated rationale — is still indefensible as a
  pair at every geometry measured.

### Detection curves (step 6): what disarming actually costs

Injected systematic: `obs*` drawn from `Poisson(mu · (1 + d·(z_k − z̄)/Δz))`
while the model keeps the untilted `mu`, so `d` is the peak-to-peak fractional
z-tilt of the data relative to the forward model. `d = 0` is the null, which is
why the first row is a false-alarm rate (at 4000 draws, hence `0.0885` against
the table's `0.0893` at 20000). Both guards see the *same* replicates, so their
sensitivities are directly comparable.

`prod_17x15x8` (`prod_29x15x8` agrees to Monte-Carlo error throughout):

| `d` | P(span arm fires @0.10) | P(`z_zbin_max` fires @5) | median span | median max abs z |
|---|---|---|---|---|
| 0.00 | 0.0885 | 0.0000 | 0.0732 | 2.00 |
| 0.02 | 0.1235 | 0.0000 | 0.0766 | 2.09 |
| 0.04 | 0.2278 | 0.0000 | 0.0844 | 2.30 |
| 0.06 | 0.4632 | 0.0020 | 0.0980 | 2.70 |
| 0.08 | 0.7298 | 0.0077 | 0.1126 | 3.11 |
| 0.10 | 0.9210 | 0.0348 | 0.1290 | 3.55 |
| 0.15 | 1.0000 | 0.4007 | 0.1720 | 4.80 |
| 0.20 | 1.0000 | 0.9277 | 0.2193 | 6.14 |
| 0.30 | 1.0000 | 1.0000 | 0.3167 | 8.94 |
| 0.50 | 1.0000 | 1.0000 | 0.5377 | 14.83 |

Smallest detected tilt, by guard (17×15×8 / 29×15×8):

| guard | threshold | FAR | `d` at 50% | `d` at 90% |
|---|---|---|---|---|
| span `by_z`, **declined** value | 0.10 | 0.0885 / 0.0825 | 0.063 / 0.062 | **0.098** / 0.098 |
| span `by_z`, **calibrated** to this grid's own q99.5 | 0.1292 / 0.1289 | 0.0073 / 0.0057 | 0.100 / 0.101 | **0.142** / 0.142 |
| `z_zbin_max`, **currently armed** | 5 | 0.0000 / 0.0003 | 0.159 / 0.158 | **0.197** / 0.196 |

Read off the last column: **`z_zbin_max` does not cover what the span arm
covered.** Disarming the span arms roughly doubles the smallest z-tilt any armed
gate detects at 90%, from `d ≈ 0.10` to `d ≈ 0.20`. A threshold calibrated on
this geometry's own null recovers most of that (`d ≈ 0.14`) at a false-alarm
rate of 0.007 rather than 0.089. The calibrated value is **not** proposed for
ratification: it is synthetic, and by §2.1 its false-alarm rate is optimistic.

### 🔴 PI DECISION REQUESTED — not resolved here

Recorded in code as
`CDDF_analysis.hbi_mcmc.ratification.OPEN_PI_DECISIONS['span_arms_disarmed']`.
The instruction this stream was given was that an *unratified* tolerance must not
gate, and the span arms were accordingly moved to report-only. The measurement
above shows that this is not cost-free, so the tradeoff goes back rather than
being settled by whoever happened to be editing the file:

* **Option 1 — leave both span arms report-only (current state).** Cost: no
  armed guard fires on a z-tilt below `d ≈ 0.20`; the standing z-marginal tilt
  defect is guarded only by `z_zbin_max`, whose own scale-dependence is written
  up in `forward_selftest.poisson_z` under "WHY 5 IS NOT SCALE-FREE". Benefit:
  nothing gates on an uncalibrated number, and the accumulated reported spans
  become the calibration set.
* **Option 2 — ratify a pack-specific `ratio_span_by_z_max` computed from each
  pack's own null at `alpha = 0.005`** (§3 steps 1–6; `0.1292` on 17×15×8).
  Cost: the threshold becomes pack-dependent, so it must be recomputed and
  stamped per pack, and it is calibrated against a null that §2.1 says is too
  narrow — so it will still fire slightly too often. Benefit: recovers detection
  down to `d ≈ 0.14` at a defensible, *measured* false-alarm rate.
  Compute cost is negligible (~1.5 min, one core, no MCMC; §4).

Neither option is adopted here. Note also that `ratio_span_by_snr` is inert at
`0.15` on every production geometry measured, so nothing in this decision turns
on it; if the by_snr arm is ever to mean anything it needs its own threshold,
which is §5 bullet 3.

---

## 5. What is *not* claimed

* No threshold is proposed for ratification. §4 is a demonstration that the
  existing pair is indefensible, plus a worked example of the procedure; §4.1
  extends the demonstration to production geometry and adds the step-6 power
  curves. The `0.1292` in §4.1 is a *measured option*, not a proposal.
* Every number in §4 and §4.1 is synthetic, from `synthetic_pack`. No survey
  pack has been calibrated, so no false-alarm rate here is the one a production
  gate would actually have.
* The null in §2 is not the true null (§2.1). A calibration on a real pack must
  either widen it (propagate nuisances) or state the omission in the paper.
* `ratio_span_by_snr` on a single-stratum grid is vacuous (§1.1 item 4). Any
  ratification of that arm must state the minimum row count at which it is
  meaningful.

## 6. Options for the PI

**These are the same two live options as §4.1's decision request (A ↔ option 2,
C ↔ option 1); §4.1 has the measured costs. B is a longer-term alternative.**

* **A — ratify pack-specific thresholds** from §3 steps 1–6, recomputed on the
  production pack, each stamped with its false-alarm rate and power curve.
  Now measured: `0.1292` on 17×15×8, FAR 0.0073, 90% detection at `d ≈ 0.14`.
* **B — replace the statistic** with the range/sd of `log(mu_r/obs_r)` or the
  dispersion of `poisson_z` across rows, then calibrate *that* by the same
  procedure. Better conditioned (§1.1), but it is a new statistic and needs its
  own ratification.
* **C — leave both arms report-only indefinitely** and let the accumulated
  values across packs be the calibration set. It is the current state, but it
  does **not** cost nothing — an earlier draft of this bullet said it did, and
  §4.1 measures the price: a real z-marginal tilt is then caught only by
  `z_zbin_max`, which needs `d ≈ 0.20` peak-to-peak for 90% detection against
  the span arm's `d ≈ 0.10`. `z_zbin_max`'s own limitation is written up in
  `forward_selftest.poisson_z` under "WHY 5 IS NOT SCALE-FREE" — and it is
  itself unratified (`ratification.RESTATED_NOT_RATIFIED`), so option C leaves
  the z-marginal defect guarded by nothing that any authority has ratified.

---

*Author: gate-ratification stream, 2026-07-29. Mock/synthetic only — no survey
data or survey-derived values appear in this document.*
