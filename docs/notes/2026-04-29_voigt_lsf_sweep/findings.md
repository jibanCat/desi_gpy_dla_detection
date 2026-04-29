# Voigt LSF + num_lines hypothesis test — first-pass findings

> **Run**: 18 picked targets × 4 configs (A/B/C/D) = 72 inferences.
> Both local CPU (gl3287, 16-core, 54 min wall) and SLURM `standard`
> partition (job 48947439, 45.7 min wall) gave identical results.
> Targets file: `voigt_sweep_48942458/targets.tsv` (seed=42, n_per_bin=5,
> snr_min=2.0). Master CSVs side-by-side at
> `/tmp/voigt_sweep_local/runs/master.csv` and
> `voigt_sweep_48947439/runs/master.csv`.

## Headline: hypothesis #1 (LSF mismatch) is **null** for the DLA regime

For 4 of 5 DLA-regime targets, MAP log NHI is **bit-identical across all
four configs** (BOSS-log-R2000, DESI-linear-R3000, DESI-linear-R3000 +
6 lines, no LSF). This is a real result, not a bug.

Verification that the kernel injection actually changes the forward model:
ran `voigt_v2_inject.inject(kernel=…)` in three fresh `spawn` subprocesses
and compared `dla_gp.voigt_absorption(wave, nhi=10²¹, z=2.7)` for each:

```
boss-log-r2000:    mean abs 0.06990114
desi-linear-r3000: mean abs 0.06993092   ← differs from BOSS at 10⁻⁵
none:              mean abs 0.07225413   ← differs from BOSS at 10⁻³
```

So the kernels do produce different absorption profiles. They just don't
shift the QMC argmax for DLA-regime targets, because the damping wings
(thousands of pixels) dominate the likelihood, while the kernel only
perturbs a handful of pixels at the line core. The bit-identical MAPs are
the QMC sample grid snapping to the same best sample in every case.

### Critically: a strong-DLA target (truth ≈ 21.3) shows tiny bias

| target | mock | truth log NHI | MAP A | MAP B | Δ vs truth |
|---|---|---:|---:|---:|---:|
| **50129689** | london | 21.280 | 21.319 | 21.319 | **+0.04** |
| 50068236 | 2lpt | 20.339 | 20.476 | 20.476 | +0.14 |
| 250132727 | 2lpt | 20.366 | 20.342 | 20.342 | −0.02 |
| 1327001289 | saclay | 20.613 | 20.650 | 20.650 | +0.04 |
| 2229000465 | saclay | 20.757 | 20.929 | 20.929 | +0.17 |
| 210294260 | london | 20.550 | NaN | NaN | (no detection) |

The historical **+0.37 dex bias** that motivated this experiment was
on 2LPT TID 120046865 (truth 21.26 → MAP 21.63). London 50129689 sits
in the same NHI class (21.28 truth) and shows **+0.04 dex**, an order
of magnitude smaller. So the historical bias appears to be
**target-specific, not a systemic LSF problem**.

## Hypothesis #2 (num_lines = 3 vs 6) is also null

Configs B (3 lines) and C (6 lines) give bit-identical MAP for all DLA
targets. The Lyβ-and-higher contribution is real in the forward model
(unit tests confirm) but doesn't shift MAP in the DLA regime — same
mechanism as #1 (the contribution is concentrated at specific Å-offsets
from Lyα, modifying ~tens of pixels out of thousands).

The one cell where configs differ: saclay sub-DLA target 2385001246:

| config | MAP log NHI |
|---|---:|
| A | 20.063 |
| B | 20.038 |
| C | 20.092 |
| D | 20.058 |

Spread 0.054 dex. n=1 so this is anecdotal, but it's where you'd expect
kernel sensitivity (Doppler-core-dominated regime). Worth re-running
with larger N_PER_BIN to see if it's signal.

## Other observations

### FILTER-1 lets through spurious multi-DLA in 2/6 LLS targets

| target | mock | truth | A MAP | Δ |
|---|---|---:|---:|---:|
| 2lpt 260170003 | 2lpt | 17.84 | 20.05 (3-DLA) | **+2.21** |
| saclay 2211001338 | saclay | 17.44 | 20.94 (1-DLA) | **+3.50** |
| 2lpt 50113931 | 2lpt | 17.70 | NaN ✓ | — |
| london 50224726, 280001404 | london | 17.5–17.7 | NaN ✓ | — |
| saclay 1190000850 | saclay | 17.49 | NaN ✓ | — |

Half the LLS targets get correctly rejected, half spuriously promoted to
DLA. This is the **FILTER fix #5** issue (don't filter M_DLA(1) evidence)
already on the task list. London is cleanest — possibly a mock-physics
difference, possibly that its LLS truth absorbers are at slightly
different redshifts.

### London sub-DLAs return NaN

Both london sub-DLA targets (truth 19.46, 19.63) returned p_DLA = 0
(no detection). 2lpt and saclay sub-DLAs at similar truth NHI got
detected. This could be:
- London's lyman-series scaling bug (production rescales by oscillator
  strength, not per-line Voigt) — sub-DLA features may be too weak in
  London mocks
- A sub-DLA prior boundary effect (sub-DLA prior is [19.0, 20.0); the
  19.63 truth is near the upper edge)

### Wall time scales with detection complexity, not config

| target | regime | A wall | D wall | factor |
|---|---|---:|---:|---:|
| 2lpt 50113931 | LLS (no det) | 6 s | 4 s | similar |
| 2lpt 260170003 | LLS (3-DLA) | 145 s | 122 s | similar |
| saclay 2385001246 | sub-DLA | 226 s | 179 s | D faster (no convolve) |
| any DLA | DLA | 35–50 s | 35–50 s | similar |

Voigt v2's `none` kernel is slightly faster (skips `np.convolve`) but
the dominant cost is the QMC sample loop, not the LSF convolution.

## What this experiment can't say

- **n is too small** (1–2 per cell). The null result for hypotheses
  #1/#2 in the DLA regime is robust because of the bit-identical MAPs
  (any sample-grid effect would show up regardless of n), but the
  comparison across mocks is statistically very weak.
- **Sub-DLA / LLS regime kernel sensitivity** is the natural next test
  — but FILTER + sub-DLA prior issues (separate hypotheses) confound
  the kernel test there.
- **GP continuum bias** is unchanged across all configs — this
  experiment can't isolate continuum from Voigt.
- **The historical 120046865 target** wasn't in the picked sample.
  Worth running it directly to see what its bias is now (model has
  changed since the original measurement).

## Recommended next steps

1. **Re-run with N_PER_BIN ≥ 10** to firm up the null in the DLA
   regime and probe the saclay sub-DLA hint at config-spread.
2. **Run inference on TID 120046865 directly** under all 4 configs.
   If the bias reproduces and the configs differ, hypothesis #1
   isn't falsified for that target. If configs match (no LSF effect)
   and bias persists, the bias is from elsewhere (continuum, prior,
   QMC sampling).
3. **Move to Bayesian-correctness Step 2** (QMC sample density and
   prior shape — `docs/notes/2026-04-27_bayesian_correctness_plan.md`).
   Step 1's null result steers us to the QMC integral itself rather
   than the forward model.
4. **FILTER fix #5** is independent of LSF — already on the task list,
   should land before the next sweep so LLS-regime results are clean.
