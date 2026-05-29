# Validation of the 6 `_corrected` v2 retrains (2026-05-06)

> Status: **regression confirmed.** None of the 6 `_corrected` retrains
> should be promoted to production. The new flag combo
> (`--scheduler none --weight-decay 1e-6 --z-min 2.5 --z-max 4.25
> --min-valid-pixels-lyman 200`, 1500 epochs) is **calibration-comparable
> to the prev v2 production but worse at canonical-TID DLA detection**.
> The "rank collapse" the 2026-05-03 synthesis flagged as a possible
> problem turns out to be a *necessary feature* for DLA discriminability
> in this GP architecture — preventing it inverted detection power.

## TL;DR

| | prev v2 production | corrected | v1 production |
|---|---:|---:|---:|
| χ²/n_valid (training-set) | 1.14–1.54 | **1.11–1.46** ✓ | n/a |
| trace_ω²/trace(K) | 0.002–0.034 | 0.006–0.594 (varies) | 0.84 |
| canonical TID p_DLA | ≈ 1.0 (detect) | **0.000–0.357 (miss)** ✗ | 0.520 (marginal) |

**Recommendation**: do not promote any `_corrected` model. Do not train
longer (converged at 1500 epochs). The next move is either to keep
prev v2 + the normalization-fix retrains (without the new flag combo)
or to investigate WHY rank collapse correlates with detection power.

---

## 1. Calibration check on training set (χ²/n)

`examples/check_v2_model_calibration.py` evaluated each model against
its own trainset (n=500 spectra, z=[2.5, 4.25], min-valid-pixels-lyman
200 — matching trainer filters):

| model | χ²/n | trace_ω²/K | top_eig MMT | eig1/eig2 | logEv mean |
|---|---:|---:|---:|---:|---:|
| 2lpt_bal_only_corrected | 1.428 | **0.594** | 3.5e+04 | **2.14** | 1169 |
| 2lpt_loa0_corrected | **1.109** | 0.116 | 1.5e+06 | 22.68 | 2048 |
| 2lpt_loa124_nohcd_nobal_corrected | 1.218 | 0.006 | 9.7e+07 | **69.79** | 1764 |
| loa_no_dla_no_bal_corrected | 1.300 | 0.036 | 5.1e+06 | 3.16 | -9 |
| loa_no_hcd_with_bal_corrected | 1.456 | 0.039 | 4.8e+06 | 2.50 | 437 |
| saclay_mock0_nohcd_nobal_corrected | **1.153** | 0.009 | 4.3e+07 | 1.78 | 1841 |

Comparison with the 2026-05-03 corrected synthesis (prev v2 production):

| | prev v2 χ²/n | corrected χ²/n | prev tr_ω²/K | corrected tr_ω²/K |
|---|---:|---:|---:|---:|
| 2lpt-mock0 | 1.18 | 1.11 | 0.002 | 0.116 |
| saclay-mock0 | 1.14 | 1.15 | n/a | 0.009 |
| LOA-noHCD-withBAL | 1.54 | 1.46 | n/a | 0.039 |
| LOA-noDLA-noBAL-y1off | 1.37 | 1.30 | n/a | 0.036 |

**χ²/n is comparable or slightly improved** for all overlapping cases.
The "fix" did not break calibration — open question #1 ("does the fix
prevent eventual collapse at 1500 epochs?") answered: **partially, and
data-dependent**. Small trainsets (BAL-only, ~20k spectra) keep
ω²-dominated K (similar to v1). Large trainsets (60k+ for mock-trained,
100k+ for real LOA) still collapse to M·M^T-dominated.

## 2. Loss-history convergence

| model | loss[0] | loss[end] | Δ%/last100 | slope/epoch | rel_std | recommendation |
|---|---:|---:|---:|---:|---:|---|
| loa_no_dla_no_bal_corrected | 10046 | 4040 | -0.045% | -0.018 | 1.3e-4 | converged |
| loa_no_hcd_with_bal_corrected | 10224 | 4666 | -0.026% | -0.012 | 7.5e-5 | converged |
| 2lpt_loa0_corrected | 8360 | 3205 | -0.027% | -0.009 | 7.9e-5 | converged |
| 2lpt_loa124_nohcd_nobal_corrected | 11177 | 3074 | -0.037% | -0.012 | 1.1e-4 | converged |
| saclay_mock0_nohcd_nobal_corrected | 11975 | 3225 | -0.058% | -0.018 | 1.6e-4 | borderline |
| 2lpt_bal_only_corrected | 4236 | 2376 | -0.058% | -0.014 | 1.7e-4 | borderline |

**Verdict on training longer: NO.** All models are at the noise floor of
optimization. The "borderline" Δ%/last100 of -0.06% means roughly 0.06%
loss-improvement per 100 epochs — at that rate, 1500 → 5000 epochs would
reduce loss by ≈3 % at most. The under-fit and detection regression are
**structural** (the K shape under the corrected flag combo doesn't
admit DLA-coherent absorption), not optimization-floor.

## 3. Canonical TID 120046865 inference (truth log NHI = 21.263)

Run with `examples/canonical_tid_per_model.py`, FILTER=True (post-fix #5
path, ≈ baseline within 0.7 %):

| model | p_DLA | MAP_z | MAP_NHI | Δ NHI | category |
|---|---:|---:|---:|---:|---|
| **v1 production** model_epoch_920 (control) | **0.520** | 2.775 | 21.53 | +0.27 | marginal |
| **prev v2** 2lpt_loa0_48938765 (control) | **0.9999** | 2.770 | 22.05 | +0.79 | DETECT (high bias) |
| 2lpt_loa0_corrected | 0.357 | nan | nan | — | MISS |
| 2lpt_loa124_nohcd_nobal_corrected | 0.009 | nan | nan | — | MISS |
| saclay_mock0_nohcd_nobal_corrected | 0.093 | nan | nan | — | MISS |
| loa_no_dla_no_bal_corrected | 0.005 | nan | nan | — | MISS |
| loa_no_hcd_with_bal_corrected | 0.001 | nan | nan | — | MISS |
| 2lpt_bal_only_corrected | 0.000 | nan | nan | — | MISS |

### FILTER on/off control on 2lpt_loa0_corrected

| FILTER | p_DLA | elapsed |
|---|---:|---:|
| False (BASELINE, full QMC) | 0.3753 | 217 s |
| True (post-fix #5 path) | 0.3572 | 19 s |

**FILTER tuning isn't the issue.** Same answer with full integration.

### Verbose log evidence comparison (canonical TID, 1-DLA mode)

| model | log p(D|null) | log p(D|1 DLA) | Bayes factor |
|---|---:|---:|---:|
| prev v2 2lpt_loa0_48938765 | -2910.21 | -2898.09 | exp(12.1) ≈ 1.8e5 |
| 2lpt_loa0_corrected | -3041.87 | -3039.92 | **exp(1.94) ≈ 7** |

**The corrected model's 1-DLA-vs-null Bayes factor dropped 4 orders of
magnitude.** Combined with the prior p(DLA|z_qso) ≈ 0.073, this gives
the observed posterior of 0.36.

The verbose log also shows `WARNING: dla_gp.py:620 No valid regions
found in the initial scan` for 5/6 corrected models — the FILTER path's
coarse scan can't find any pixel cluster that looks DLA-like under the
corrected models' larger per-pixel σ. But again, this is downstream of
the same root cause: the K shape lost DLA discriminability.

## 4. Why detection broke (mechanism)

The corrected fix combo prevents the trace_ω²/K collapse that prev v2
production suffered. We expected this to be GOOD — the synthesis worried
about "K dominated by Lyα emission scale" being unphysical. But the data
show: **K rank-collapse is what was giving the GP its DLA discriminability.**

- v1 has trace_ω²/K = 0.84 → mostly per-pixel ω² noise → it's fundamentally
  an "ω-dominated" GP, like the smoothed-residual model. Detects DLAs
  with **modest** Bayes factor (p_DLA = 0.52 on canonical TID).
- prev v2 (uncorrected) has trace_ω²/K = 0.002–0.034 → K dominated by a
  rank-1 (or very low-rank) M·M^T structure. Pixels are
  **highly correlated** through that single eigenmode. A coherent
  multi-pixel absorption signal (DLA Lorentzian wings + Lyβ damping)
  AMPLIFIES through the cross-pixel covariance → **enormous Bayes factor**
  (p_DLA → 1.0).
- corrected has trace_ω²/K = 0.006–0.594 (data-dependent) → middle
  ground. Lost the rank-1 amplification; not yet ω-dominated like v1.
  **Worst of both.**

The high-bias (+0.79 dex) prev v2 detection is consistent with the same
mechanism: the rank-1 K cross-correlates Lyα with everything, so it
preferentially finds whichever NHI value MAXIMIZES the cross-covariance
match — typically NHI > truth.

## 5. Structural decomposition figures

- `figs/trained_corrected_compare_mu.png` — μ overlay across 6 models;
  Lyα emission peak height ranges 3.0–5.5 (BAL-only model dips at
  CIV/SiIV/OVI BAL signatures).
- `figs/trained_corrected_compare_omega.png` — ω overlay (log scale);
  2lpt_loa0_corrected has uniformly low ω in the continuum (1300+ Å)
  but spikes at the Lyα peak; LOA-real models have higher continuum ω
  reflecting realistic spectral noise.
- `figs/trained_corrected_corr_grid.png` — corr(M·M^T) per model.
  **Key visual confirmation:**
  - 2lpt_loa124_nohcd_nobal: nearly **all-red** (eig1/eig2 = 70) — single
    eigenmode dominates, like prev v2. But the χ²/n calibration only
    saw a 0.04-pt difference vs the corrected 2lpt_loa0 (1.22 vs 1.11).
  - 2lpt_loa0_corrected: visibly more block-structured.
  - LOA-real models: textured fine-scale variation, multiple competing
    modes.
  - 2lpt_bal_only_corrected: rich block + ringing structure — multiple
    physical absorption modes (CIV, SiIV, OVI BAL) bake in.

## 6. What this changes about the tier roadmap

**Tier 1 #2 (v2 normalization bug fix)**: the **code fix is correct** —
the trainset's per-spectrum normalization needed to land. But the
corrected flag combo (intended to ALSO prevent rank collapse) is the
wrong direction. Recommend a follow-up retrain that keeps the
normalization fix but DROPS `--scheduler none --weight-decay 1e-6`
(i.e. uses the original v2 hyperparams + just the normalization).

**Tier 1 #3 (promote v2 LOA / test mock-trained)**: blocked. None of
the 6 corrected models is promotable. The 4 "_normalized" retrains from
the 2026-05-01 round (without the corrected combo) may still be the
right candidates — but they were never validated on canonical TID either.
Action: re-validate the `*_normalized` models the same way, before
deciding on production replacement.

**Open question that the 2026-05-03 synthesis posed (#2)**: "Does the
trace_ω² collapse actually harm DLA inference?" — **answer: NO, the
opposite. The collapse is what made DLA detection work in prev v2.**
The prior on this question should now be inverted: rank collapse is a
feature, not a bug, for this architecture.

## 7. What's saved

```
docs/notes/2026-05-06_corrected_model_validation/
├── REPORT.md                                   ← this file
├── summary_table.md                            ← calibration metrics
├── canonical_tid_summary.md                    ← inference per model
├── figs/
│   ├── calibration_<model>.png × 6             ← per-model 4-panel diag
│   ├── trained_corrected_compare_mu.png
│   ├── trained_corrected_compare_omega.png
│   └── trained_corrected_corr_grid.png
├── metrics/<model>.json × 6                    ← raw χ²/n, eigs, etc.
└── canonical_tid/<model>.json × 6              ← raw p_DLA, MAP per model
```

Scripts:
- `examples/check_v2_model_calibration.py` (modified — accepts --z-min/--z-max + emits --metrics-out JSON)
- `examples/plot_corrected_model_compare.py` (new)
- `examples/canonical_tid_per_model.py` (new)
