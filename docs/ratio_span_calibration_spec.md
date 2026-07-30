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
exactly what the PI declined, and §4 below shows it refuses a *third of
perfectly correct forward models*; (b) delete the arms — rejected, because the
reported values **are** the calibration data and deleting them means the
calibration can never be run. (c) compute-report-do-not-gate was adopted.

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

## 4. Measured result on the synthetic pack (why the PI's refusal was right)

Pack: `synthetic_pack(0, nhat_edges=19.9…20.4 step 0.1, zf_edges=2.0…2.4 step
0.1, zc_edges=[2.0,2.2,2.4], snr_edges=[0,3,inf], n_molly_cells=3,
fp_frac=0.15, t_true=[0.2,−0.15])`; `resp_clamp="both"`. Total `mu = 2283.95`
against total `obs = 2274`. `n_draws = 20000`, `seed = 1`.

| arm | rows | null q50 | null q95 | null q99 | proposed max | **measured false-alarm rate at the proposed max** |
|---|---|---|---|---|---|---|
| `by_z`   | 4 | 0.0865 | 0.1574 | 0.1923 | **0.10** | **0.3434** |
| `by_snr` | 2 | 0.0275 | 0.0825 | 0.1099 | **0.15** | **0.0002** |

Read that table twice. Under a null in which the forward model is *exactly
right*, `ratio_span_by_z_max = 0.10` refuses **34% of runs**, while
`ratio_span_by_snr_max = 0.15` refuses **0.02%** — the two numbers, presented as
a matched pair with the SNR one "wider because the strata are noisier", differ in
false-alarm rate by more than three orders of magnitude, and in the *opposite*
direction to the stated rationale. This is what an uncalibrated tolerance inside
a production fail-closed gate looks like.

The same fold's *observed* `by_z` span on this pack is 0.1734, which the arm
would have reported as a failure and which the null puts at roughly the 98th
percentile of pure counting noise — suggestive, not decisive, and precisely the
sort of claim that needs a calibrated threshold before it can refuse anything.

At the Bonferroni `alpha = 0.005` of §3 the *pack-specific* thresholds would be
`by_z ≈ 0.198`, `by_snr ≈ 0.113` (the q0.995 column of `ratio_span_null`). These
are **not** proposed for ratification here: they come from a synthetic pack, they
inherit every omission in §2.1, and no power curve (step 6) has been measured.

Reproduce with:

```
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/mfho/.conda/envs/gpdla-hbi/bin/python -m pytest \
    tests/test_gate_ratification.py -q -k ratio_span_null
```

---

## 5. What is *not* claimed

* No threshold is proposed for ratification. §4 is a demonstration that the
  existing pair is indefensible, plus a worked example of the procedure.
* The null in §2 is not the true null (§2.1). A calibration on a real pack must
  either widen it (propagate nuisances) or state the omission in the paper.
* `ratio_span_by_snr` on a single-stratum grid is vacuous (§1.1 item 4). Any
  ratification of that arm must state the minimum row count at which it is
  meaningful.

## 6. Options for the PI

* **A — ratify pack-specific thresholds** from §3 steps 1–6, recomputed on the
  production pack, each stamped with its false-alarm rate and power curve.
* **B — replace the statistic** with the range/sd of `log(mu_r/obs_r)` or the
  dispersion of `poisson_z` across rows, then calibrate *that* by the same
  procedure. Better conditioned (§1.1), but it is a new statistic and needs its
  own ratification.
* **C — leave both arms report-only indefinitely** and let the accumulated
  values across packs be the calibration set. Costs nothing and is the current
  state; the price is that a real z-marginal tilt is caught only by
  `z_zbin_max`, whose own limitation is written up in
  `forward_selftest.poisson_z` under "WHY 5 IS NOT SCALE-FREE".

---

*Author: gate-ratification stream, 2026-07-29. Mock/synthetic only — no survey
data or survey-derived values appear in this document.*
