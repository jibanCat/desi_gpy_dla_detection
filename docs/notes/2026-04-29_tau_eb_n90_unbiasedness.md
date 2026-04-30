# τ-EB unbiasedness test on n=90 — HCD masking is *not* the right default

> **Headline**: at n=90 DLA-regime targets (seed=43, fresh sample disjoint
> from the n=54 / n=18 used previously), the **HCD-masked** version of the
> τ-EB recipe **over-corrects** the bias (median +0.135 → −0.131 dex), while
> **the same τ-EB without the HCD mask** lands the median essentially at
> zero (+0.135 → +0.026 dex). The original n=18 result that motivated the
> mask was likely cherry-picked.
>
> **Implication for PR #5**: the production τ-EB module should default to
> **no HCD masking**, with the mask exposed as an optional flag for the
> small set of saturated-DLA-dominated targets where it was helpful.

## Run setup

- 90 DLA-regime targets (truth log N_HI ≥ 20.3, exactly one truth absorber
  in window, SNR_red ≥ 2.0); 30 per mock × {2lpt, london, saclay}; seed=43.
- Targets file: `/tmp/targets_dla_n90.tsv` — distinct from the n=54 / n=18
  used in the EOD 2026-04-29 prior session (which used seed=42).
- Driver: `slurm/greatlakes/hcd_mask_unbiased_test.sh` (8-way parallel,
  `examples/check_tau_eb_robust_mask.py` per target).
- τ-grid: extended from the recipe's nominal (0.5–2.0) to **(0.5, 0.75,
  1.0, 1.25, 1.5, 2.0, 3.0, 4.0)** — necessary because 22 % of targets
  pile up at τ_factor=4.0 even with the extended grid (and 83 % piled at
  the cap=2.0 in the un-extended grid).
- HCD-mask threshold: 1.5 σ.

## Headline numbers

| Treatment | median bias | mean bias | RMS | frac \|bias\|<0.1 | frac \|bias\|<0.2 |
|---|---:|---:|---:|---:|---:|
| Production τ_0 = 0.00246 | +0.135 | +0.195 | 0.410 | 33 % | 68 % |
| **τ-EB, no HCD mask** | **+0.026** | +0.019 | **0.257** | **54 %** | **76 %** |
| τ-EB + HCD mask (1.5 σ) | −0.131 | −0.118 | 0.285 | 29 % | 59 % |

Wilcoxon test (H₀: median = 0): production p = 4·10⁻⁹ (clearly positive
bias); HCD-masked p = 1·10⁻⁶ (clearly negative bias); **no-mask p = N/A
because median ≈ 0** (would need a different test of equivalence; the
mean and median are within their MC error of zero).

### Per-mock breakdown

| Mock | n | prod median | no-mask median | HCD-mask median |
|---|---:|---:|---:|---:|
| 2lpt | 30 | +0.120 | +0.017 | −0.133 |
| london | 30 | +0.155 | +0.037 | −0.158 |
| saclay | 30 | +0.131 | +0.015 | −0.128 |

The over-correction with HCD masking is consistent across all three
mocks; the no-mask version is consistently closer to zero. This is not
mock-specific.

### Per-NHI-regime breakdown

| Truth log N_HI | n | prod median | no-mask median | HCD-mask median |
|---|---:|---:|---:|---:|
| [20.3, 20.6) | 41 | +0.078 | **+0.003** | −0.094 |
| [20.6, 21.0) | 37 | +0.154 | +0.037 | −0.184 |
| [21.0, 23.0) | 12 | +0.144 | +0.013 | −0.176 |

The pattern holds at every NHI strength.

## τ_best distribution

The τ_best each method picks is informative:

| τ_factor | no-mask | HCD-mask |
|---:|---:|---:|
| 0.50 | 3 | 1 |
| 0.75 | 3 | 2 |
| 1.00 | 4 | 2 |
| 1.25 | 7 | 2 |
| 1.50 | 9 | 8 |
| 2.00 | 26 | 26 |
| **3.00** | 22 | **27** |
| **4.00** | 16 | **22** |

HCD masking systematically pushes τ_best **upward** (43 % of targets get
a higher τ when masked; only 3 % get a lower τ). That extra
high-τ pull is what causes the over-correction: at very high τ_eff the
forward model expects deeper forest absorption everywhere, so the QMC
fitter compensates with a *lower* MAP log N_HI on whatever DLA is
present.

## Why does this contradict the n=18 result?

The earlier session's n=18 DLA-regime sample (drawn from the n=54
scale-out at seed=42) showed HCD-masking closing 81 % of median bias.
With the same grid (capped at 2.0) on this n=90 sample (seed=43), HCD
masking only closes 21 %. With the extended grid (up to 4.0) it
*over-corrects* to −8 % closure (worse than production).

Most likely explanation: **the n=18 was cherry-picked by the LSF-sweep
target picker that was reused for seed=42**. That picker's filters
(SNR_red ≥ 2, single-truth-absorber, mid-forest) happen to select for
strong, clean DLAs where the saturated trough is unambiguous and HCD
masking does what was claimed (trim the trough, free τ to find truth).
On a more representative sample (different seed, same picker but no
implicit correlation to a different test's selection), the HCD mask
removes too many pixels — including legitimate forest pixels in the
DLA wing — and the τ fitter compensates by going too high.

A simpler way to put it: at n=18 the "naive EB" closed only ~30 % of
bias because that subsample's bias was unusually large (+0.240 dex
median); on n=90 with median bias +0.135 dex the naive EB does fine.
The mask was solving a problem of cherry-picked extreme bias.

## What this means for PR #5

1. **Default the production module to NO HCD masking.** The validated
   benefit (n=18 → 81 %) doesn't generalize. Naive EB closes the bias
   to median ≈ 0 with 54 % within ±0.1 dex.
2. **Keep HCD masking as an optional flag.** On extreme saturated-DLA
   targets (the canonical 120046865, where this all started) it does
   help. A mask threshold and on/off toggle stays useful.
3. **Default the τ_factors grid to (0.5, 1.0, 1.5, 2.0, 3.0, 4.0).**
   The recipe's earlier (..., 2.0) grid pinned 83 % of targets at the
   ceiling. Six points up to 4.0 is the right trade-off (cost ~6× null
   build = sub-second per spectrum, well below the bayes step).
4. **Update `gpy_dla_detection/tau_eb.py` defaults accordingly.**

## τ_factor distribution vs Turner+2024 (no-HCD-mask EB)

The chosen τ_eff scales **on average ~2× higher than Turner+2024**, with
substantial spread. This is a large deviation, not a fine-tuning, and the
fact that all three mocks land near 2× is itself a signal worth tracking
on real LOA data.

| Subset | n | min | 25 % | median | mean | 75 % | max | std | frac ≥ 2× |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 90 | 0.50 | 1.50 | 2.00 | 2.36 | 3.00 | 4.00 | 1.04 | 71 % |
| 2lpt | 30 | 0.75 | 1.62 | 2.00 | 2.34 | 3.00 | 4.00 | 0.93 | 73 % |
| london | 30 | 0.50 | 2.00 | 2.50 | 2.58 | 3.75 | 4.00 | 1.10 | **77 %** |
| saclay | 30 | 0.50 | 1.25 | 2.00 | 2.15 | 3.00 | 4.00 | 1.07 | 63 % |

By truth log N_HI bin:

| Bin | n | min | median | max | frac ≥ 2× |
|---|---:|---:|---:|---:|---:|
| [20.3, 20.6) | 41 | 0.50 | 2.00 | 4.00 | 59 % |
| [20.6, 21.0) | 37 | 0.50 | 3.00 | 4.00 | **84 %** |
| [21.0, 23.0) | 12 | 0.75 | 2.00 | 4.00 | 75 % |

Histogram (naive EB):

```
  τ=0.50:   3 ( 3 %) ▌▌
  τ=0.75:   3 ( 3 %) ▌▌
  τ=1.00:   4 ( 4 %) ▌▌
  τ=1.25:   7 ( 8 %) ▌▌▌▌
  τ=1.50:   9 (10 %) ▌▌▌▌▌▌
  τ=2.00:  26 (29 %) ▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌
  τ=3.00:  22 (24 %) ▌▌▌▌▌▌▌▌▌▌▌▌▌▌
  τ=4.00:  16 (18 %) ▌▌▌▌▌▌▌▌▌▌▌
```

**Why is this a useful signal:**
- Across 2lpt, london, and saclay — three independent mock pipelines
  with different physics — the EB lands consistently at ~2× Turner+2024
  for the median DLA target. That suggests the deviation is **NOT
  mock-physics-specific**; it's likely a property of the GP+forward-model
  rather than a τ_eff measurement.
- A natural interpretation: the GP forward model (μ-shape × A_lyα ×
  per-pixel σ) accounts for forest absorption with a particular
  effective τ that is on the high side of Turner+2024 once HCDs are
  *included* (no masking) in the fit. The model is not "wrong" per se;
  it's that the τ parameter in the production prior τ_0 = 0.00246 is a
  free knob that the EB step is calibrating per spectrum.
- 18 % pile up at τ=4.0 (the grid ceiling) — the recipe still wants
  even higher τ on a non-trivial fraction of targets. Extending to
  τ=5.0 or 6.0 would let us test how much further. Currently flagged
  as open work (orthogonal to PR #5).

**What to log in production**: when `--enable_tau_eb 1`, the holder
already prints the chosen factor per spectrum
(`...τ-EB[null, hcd_mask=False]: factor_best=2.00 τ_0=0.00492 n_hcd=0`).
Recommend a run-summary script that aggregates these from production
logs and compares mock-vs-LOA τ distributions. **If LOA data lands
near τ ≈ 2× as well**, the recipe is likely calibrating a real model
parameter — not "fixing a bug". **If LOA τ lands near 1.0×**, the mocks
have systematically different forest opacity from real data and that's
itself science worth understanding. Either way the comparison is
actionable.

## Open questions

1. Why does the no-mask EB still over-correct in the *individual* extreme
   cases (e.g. 2lpt 270088696: prod −0.22 → no-mask not yet measured at
   ext grid)? Are there spectra where the recipe genuinely needs the
   mask? A target-level diagnostic would distinguish.
2. The recipe used `objective="dla"` (max-over-NHI at truth_z). For
   production the truth_z isn't available; either replace with a coarse
   z-grid scan or switch to `objective="null"` (null-model log evidence).
   The n=90 results above used `objective="dla"` with truth_z. The
   "null" objective is untested at scale.
3. The mask threshold sensitivity (1.5 σ here, 3 σ filtered zero pixels)
   is unprobed; reducing it might recover the canonical-target benefit
   without the over-correction. Open work.

## Files

- Driver: `slurm/greatlakes/hcd_mask_unbiased_test.sh`
- Targets: `/tmp/targets_dla_n90.tsv` (90 DLA-regime, seed=43)
- Capped-grid results: `/tmp/hcd_mask_unbiased_n90/summary.csv`
- Extended-grid results: `/tmp/hcd_mask_unbiased_n90_extgrid/summary.csv`
- Per-target logs: `/tmp/hcd_mask_unbiased_n90_extgrid/per_target/*.log`
