# Voigt LSF + num_lines hypothesis test — first-pass findings

> **2026-04-29 follow-up — partial retraction**: a user-flagged kernel
> truncation bug in `_kernel_for` was discovered after this report
> landed. DESI-R3000 was being clipped to a 7-pixel kernel, collapsing
> σ_eff from 4.25 → 1.92 px (45% of intent). That makes the **LLS /
> sub-DLA results below invalid** (configs A and B were both narrow
> kernels in disguise). The DLA-regime nullity is still real — the
> saturated DLA core is wider than any kernel — but the strong claim
> "hypothesis #1 falsified" is *not* supported by this run. See "Bug
> retractions" section at the end. Fix landed in commit `eda1930`;
> a re-run is queued.
>
> Three additional caveats raised by the user that compound the above:
> 1. Targets are picked for high SNR + clean truth — easy-to-fit cases,
>    biased toward small Δ. Doesn't generalize.
> 2. The QMC samples are denser at low NHI than at high NHI by design
>    (matching the prior). For a truth at log_nhi = 21.26, the nearest
>    sample is at ~21.20 or ~21.30, so the MAP is sample-grid-snapped.
>    Adaptive sampler is the right fix.
> 3. The voigt path under test is `gpy_dla_detection/voigt_v2.py` — the
>    new selectable-kernel module on this branch — not the production C
>    extension `voigt_fast.so`. v2 + voigt_v2_inject is what the sweep
>    runner injects per-config.

> **Run**: 18 picked targets × 4 configs (A/B/C/D) = 72 inferences.
> Both local CPU (gl3287, 16-core, 54 min wall) and SLURM `standard`
> partition (job 48947439, 45.7 min wall) gave identical results.
> Targets file: `voigt_sweep_48942458/targets.tsv` (seed=42, n_per_bin=5,
> snr_min=2.0). Master CSVs side-by-side at
> `/tmp/voigt_sweep_local/runs/master.csv` and
> `voigt_sweep_48947439/runs/master.csv`.

## Headline (with caveats below): hypothesis #1 looks null for DLA regime, **inconclusive elsewhere**

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

### A strong-DLA target (truth ≈ 21.3) shows tiny bias — but cherry-picked

**Caveat (raised by user)**: the picker explicitly selects targets that
(a) pass the SNR cut, (b) have exactly one truth absorber, (c) are
mid-forest. Strong, clean DLAs at SNR ≥ 2 are easy to fit by definition
— the picker selects-against the failure modes that produced the
historical +0.37 dex bias. So a small Δ here is consistent with both
"the bias is target-specific" and "we filtered out the targets that
would have shown it". Not strong evidence either way until we re-run
with looser cuts and on the historical target itself.

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

1. **Re-run the whole sweep with the kernel-fix commit `eda1930`** —
   the LLS / sub-DLA cells in this report are invalid. The DLA-regime
   columns can stand once verified against the fixed kernel (expected
   to be similar since the saturated core dominates).
2. **Run inference on TID 120046865 directly** under all 4 *fixed* configs.
   First attempt with the buggy kernel returned `p_DLA = 0.05` (no
   detection) — itself a worrying disagreement with the historical v1
   result, may be a separate v2-vs-v1 issue.
3. **Loosen the picker cuts** so the population isn't selected for
   easy-to-fit-ness. Drop the SNR floor; allow multi-absorber LOS
   (use the strongest as truth); allow targets close to the redshift
   prior edges. Sample 50+ per cell, then the bin medians have something
   to hang on.
4. **Adaptive QMC sampler** to fix the sparse-coverage problem at
   high NHI. The current `dla_samples_a03_100000.mat` has ~5e-5 dex
   spacing at log_nhi=20 (where the prior is dense) and >1e-3 dex
   spacing at log_nhi=22 — at the historical bias target's truth
   (21.26) the MAP is *forced* to snap to one of the few high-NHI
   samples. A two-stage sampler that re-densifies in the high-likelihood
   region is the right fix; this is independent of LSF.
5. **Move to Bayesian-correctness Step 2** (QMC sample density and
   prior shape — `docs/notes/2026-04-27_bayesian_correctness_plan.md`).
   This subsumes #4 and is the logical follow-up to the LSF question
   either way.
6. **FILTER fix #5** is independent of LSF — already on the task list,
   should land before the next sweep so LLS-regime results are clean.

## Bug retractions (2026-04-29)

### Kernel half_width "truncation" — partial walk-back
`_kernel_for("desi-linear-r3000", ...)` returned a 7-pixel Gaussian
even when the intended σ exceeded that. Truncating a wide Gaussian to
7 pixels collapses its effective σ — *if* you're applying it on a
fine grid. I claimed this was the cause of "configs A vs B identical"
in the sweep. **It isn't, for production data.** Trace data from a
real DESI inference shows the convolution actually happens on the
0.8 Å observed-data grid (after the GP model has been interpolated
onto that grid), where σ_pix(R=3000) is only 0.80 px and there's no
truncation issue. The half_width-auto-sizing fix in commit `eda1930`
is still useful as defensive code (if anyone runs voigt_v2 on a finer
grid, e.g. a unit test at dλ=0.15 Å, it won't silently truncate), but
it does NOT fix a real production bias. Apologies for the misdirection
— I was running the demo at dλ=0.15 (the GP rest-grid spacing) which
is not where the actual convolution happens. Updated demo PNG at
`docs/voigt_demo/voigt_kernel_demo_dl08.png` uses the correct dλ=0.8.

The actual production C ext kernel (σ_eff=0.61 px) applied on the
DESI 0.8 Å grid: σ_λ ≈ 0.49 Å, σ_v ≈ 37 km/s, **R_eff ≈ 3400** —
roughly close to the intended DESI R=3000. Slightly sharper, but not
a factor-of-7 over-sharpening as I'd claimed.

### Canonical target: confirmed forward-model bias, NOT sampler-limited
Implemented the user's "log L at truth vs log L at MAP" test
(`examples/check_truth_vs_map_likelihood.py`) on TID 120046865 with
the historical bias (truth: z=2.773, logNHI=21.263; sub-DLA at
z=2.287 logNHI=19.41). Brute-force scanned 20,000 random QMC samples
to bypass the FILTER step's initial-scan rejection.

| quantity | log L |
|---|---:|
| null (no absorber) | **−2886.1** |
| truth (z=2.773, logNHI=21.263) | **−2865.9** |
| brute-force MAP over QMC (z=2.7722, **logNHI=21.547**) | **−2864.5** |

Outcome:
- Truth fits **+20.2 log-units better** than null — the data clearly
  has the DLA, so this isn't a "DLA absent" pathology.
- The brute-force MAP sits at **logNHI = 21.547, +0.28 dex above
  truth** — closely reproducing the historical +0.37 dex bias.
- Truth fits **−1.4 log-units worse** than the brute-force MAP. So
  the GP+Voigt+continuum is genuinely happier with NHI=21.55 than
  with NHI=21.26 on this spectrum.
- The QMC sample at truth's exact logNHI=21.263 *exists* in the
  100k-sample set (Δ=0.0000 dex). **This is not a sampler-density
  problem** for the strong-DLA component.

⇒ **The +0.37 dex bias on this target is a real forward-model /
continuum bias**, not LSF and not sampler-limited. Most likely
candidates: GP mean μ overestimates flux in the DLA wing region, or
ω² (per-pixel uncertainty) is too small there, so the optimizer
compensates with stronger absorption (higher NHI). Continuum
mismatch on the DLA's blueward wing would do exactly this.

### Separate issue: FILTER initial-scan is broken on this target
With 100k samples + clear DLA detection (Δ(MAP − null) = +21.7), the
FILTER initial-scan still returned "no valid regions" and the holder
reported `p_DLA = 0.05` (≈ prior). Brute-force argmax confirms the
data clearly contains the DLA. This is FILTER fix #5 territory —
already on the task list.
