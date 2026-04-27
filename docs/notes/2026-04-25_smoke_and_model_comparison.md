# Smoke test + trained-model comparison on GreatLakes

**Date:** 2026-04-25
**Cluster:** UMich GreatLakes, login node `gl3114` (shared)
**Repo branch:** `claude/friendly-allen`
**Smoke target:** 2LPT mock-0 contaminated, TARGETID 120046865
- z_qso = 2.962
- One DLA in truth: z_DLA = 2.773, log N_HI = 21.26 (mid-forest)
- Spec file: `mocks/lyacolore_2lpt/.../mock-0/loa-124/spectra-16/7/789/spectra-16-789.fits`

## What was tested

End-to-end: env + libcerf + `_voigt.so` + 80-test suite + single-spectrum
inference under three trained GP models, in two operational modes (LLS,
multi-DLA) and both `FILTER_LOW_LIKELIHOOD` settings.

The objective was *not* a final validation; it was a smoke test on a single
spectrum to (a) prove the GreatLakes env reproduces the NERSC pipeline, and
(b) get an early read on N_HI bias and per-model performance.

All runs used:
- 10k DLA QMC samples (`dla_samples_a03.mat`)
- 10k sub-DLA samples (`subdla_samples.mat`) — baseline multi-DLA combo
- `MAX_WORKERS=8`, `BATCH_SIZE=1250`
- `FILTER_LOW_LIKELIHOOD=1` (multi-DLA)

## Results

### Operational mode — LLS-mode finds nothing on a strong DLA. Expected.

LLS-mode (`single_absorber_model=1`, `pw_samples_a3_172_220_50000.mat`) was
run twice on this DLA target:

| FILTER | wall | p(absorber) |
|---|---:|---:|
| 0 | 44.9 s | 0.366 |
| 1 |  7.6 s | 0.054 |

**LLS mode is for CDDF / sub-DLA population statistics, not DLA detection.**
The LLS prior puts essentially no weight at log N_HI ≥ 21, so even an
obvious DLA gets weak posterior here. p(absorber) drops further with
FILTER=1, and FILTER=1 is ~6× faster. This matches the user's note that
LLS-mode is *only* for sub-DLA / CDDF analysis — running it as a "did we
find the DLA?" test is the wrong tool.

### Multi-DLA mode — three trained models on the same DLA

Model | dlambda | k | wall | inference | p(DLA) | MAP z_DLA | MAP log N_HI | ΔlogN_HI vs truth (21.26)
---|---:|---:|---:|---:|---:|---:|---:|---:
**eBOSS DR16Q** | 0.25 | 20 |  **4.0 s** |  **1.4 s** | **0.992** | 2.7749 | **21.445** | **+0.18** dex
DESI Y3 epoch_920 | 0.15 | 30 | 55.6 s | 53.2 s | 0.851 | 2.7735 | 21.626 | **+0.37** dex
London-mock epoch_199 | 0.15 | 30 | 53.8 s | 51.2 s | 0.279 | (no MAP) | (no MAP) | n/a

Truth: z_DLA = 2.773, log N_HI = 21.26.

Both Y3 and eBOSS recover z to 0.001. **eBOSS recovers N_HI with half the
bias** of Y3 on this single spectrum. The London model — which has only 199
training epochs (vs 920 for Y3) and was trained on London mocks — barely
detects this 2LPT-injected DLA at all.

## Why eBOSS may be performing better than Y3 on this spectrum

This is N=1 evidence; not a definitive ranking. But the three observations
(NHI bias, p(DLA) confidence, runtime) are mutually self-consistent under a
single mechanism: **the hard-coded LSF kernel in `gpy_dla_detection/ctypes_voigt.c`
under-broadens the modeled DLA profile, and the under-broadening is worse
at smaller `dlambda`.**

### The kernel

The 7-pixel `instrument_profile[]` in `ctypes_voigt.c:250-259` was derived
for **BOSS R=2000 on a log-λ pixel grid** (pixel ratio 10^1e-4 ≈ 23 km/s,
σ ≈ 0.92 pixels ≈ 21 km/s in velocity units). The comment block at lines
224–235 acknowledges DESI has R = 2000–5000 in **linear-λ** pixels, but the
kernel values were never updated.

When that kernel is applied to a DESI linear-λ grid, the velocity width of
the kernel changes with `dlambda`:

`dlambda` | observed-frame pixel velocity (at λ_obs ≈ 4500 Å) | kernel σ_v (≈0.92 pix) | DESI true LSF σ_v (R≈3000) | shortfall
---:|---:|---:|---:|---:
0.25 Å (eBOSS preset) | 16.7 km/s | 15 km/s | 42 km/s | factor 2.8 too narrow
0.15 Å (Y3 preset)    | 10.0 km/s |  9 km/s | 42 km/s | factor 4.7 too narrow
log10 1e-4 (BOSS)     | 23.0 km/s | 21 km/s | 64 km/s (BOSS R=2000) | factor 3.0 — close to design intent

So the same kernel under-broadens worse under the Y3 preset than under the
eBOSS preset.

### Why under-broadening biases N_HI high (not low)

A DLA's intrinsic profile is dominated by the damping wing whose width
scales with N_HI. If the model under-broadens by the LSF, the modeled line
is sharper than the data:

- True observed profile width ≈ σ_intrinsic(N_HI_true) ⊕ σ_LSF_true
- Modeled profile width      ≈ σ_intrinsic(N_HI_fit)  ⊕ σ_kernel

For the modeled width to match the data, the fitter has to *increase*
σ_intrinsic, i.e. **fit a higher N_HI** to lengthen the damping wings.
The bias direction matches the user's prior "biased high" report, and the
factor-2 ratio between Y3 and eBOSS biases (+0.37 vs +0.18 dex) matches
the factor-2 ratio in kernel shortfall.

### Why eBOSS is faster (~38× on this spectrum)

The dominant factors are dimensionality of the GP linear algebra:

- `dlambda` 0.25 Å vs 0.15 Å → ~40% fewer rest-frame pixels in the model grid.
- `k` 20 vs 30 → 4/9 the rank of the Woodbury low-rank correction.
- The Bayesian model evidence integrates 10k samples × 4 DLA models per
  spectrum, and per-sample cost scales with both grid size and rank.

Naively `(0.6) × (20/30)^2 ≈ 0.27` per sample, or ~3.7× total — that's
much less than the 38× I observed. There may be additional factors
(loop overhead, ProcessPoolExecutor task granularity, login-node CPU
throttling that hits the longer-running jobs harder). I didn't profile to
isolate them. But the empirical speedup is clear and consistent across two
re-runs.

### Why London epoch_199 is so much weaker

199 epochs is a fraction of Y3's 920. The trained GP-prior on the QSO
continuum is presumably under-converged. Additionally, this model was
trained on London mock-0 spectra and is now being asked to score 2LPT
mock-0 spectra — the noise statistics, contamination model, and mean flux
in the two mocks differ. So weak performance on a non-London target is
unsurprising.

## What this implies (working hypotheses, not conclusions)

1. **The Voigt LSF kernel is the largest known bias source.** Fixing it for
   the DESI linear-λ grid + actual DESI resolving power should reduce the
   N_HI high-bias substantially, in both model presets but especially Y3.
   Plan: implement `voigt_v2` behind a `--voigt-kernel desi-static` /
   `--voigt-kernel desi-permatrix` flag, validate by re-running this same
   spectrum, then sweep injection-recovery on uncontaminated 2LPT.

2. **eBOSS may continue to look better than Y3** even after fixing the
   kernel, *if* the GP prior trained on clean SDSS DR16Q is a better
   continuum model than the GP prior trained on Y3 (which contains BAL,
   metals, and DLAs that the GP may have partially absorbed into its
   kernel). This warrants a controlled comparison on the uncontaminated
   2LPT mock and a non-trivial sample (≥100 spectra), not just N=1.

3. **Production may want to switch back to the eBOSS-trained model** for
   real DESI catalog production — but only if the bias and completeness
   numbers actually favor eBOSS over Y3 on a meaningful sample, AND the
   eBOSS preset's hyperparameter set (`dlambda`, `k`, range) matches what
   the spectra need.

## Open / next-step questions for review

1. The **London-mock model** is at epoch_199 only on GreatLakes — is the
   "production London-mock model" something later? On NERSC, is there a
   higher-epoch checkpoint?
2. **Speed**: is 1.4 s / spectrum (eBOSS) the right baseline for "the
   pipeline runs", and 53 s the right alarm bell for the Y3 preset on a
   compute node, or is there an additional optimization expected?

   **Update (2026-04-27, sbatch on `gl3051` compute node, 16 cores, 16 GB):**

   | preset | login wall | login inference | compute wall | compute inference |
   |---|---:|---:|---:|---:|
   | eBOSS  |  4.0 s | 1.4 s |  7.8 s | **1.3 s** |
   | Y3     | 55.6 s | 53.2 s | 47.7 s | **45.0 s** |

   Compute node helps Y3 by ~15% (53 → 45 s), but the bulk of the cost is
   intrinsic to the model dimensions (`dlambda=0.15` Å, `k=30`), not
   login-node throttling. The eBOSS preset on compute is actually slightly
   slower on wall time because of fixed startup overhead — but inference
   itself stays at 1.3 s. **NERSC's 2-3 s figure is plausibly the eBOSS-style
   preset on a compute node**, not the heavier Y3 preset.

   That makes sense — eBOSS has fewer pixels and a smaller GP rank, so the
   Woodbury inversion is much cheaper. If we want Y3 inference at sub-10 s
   per spectrum, the structural fix is reducing `k` or `num_dla_samples`,
   not parallelism rework.
3. **The DLA at z_DLA = 1.855 on TARGETID 110156591** got `p(DLA) = 0` even
   in multi-DLA mode (search-window-edge effect). Is the GP-DLA search
   range cut intentionally to keep clear of foreground absorbers, or
   should it cover lower z_DLA?

## Visual confirmation that the DLA is in the contaminated mock

After an initial concern that the smoke plots were showing the
*uncontaminated* spectrum, I added a unit test
(`tests/test_smoke_target_contamination.py`) that loads both
`loa-124` (contaminated) and `loa-0` (uncontaminated) for the same
TARGETID and checks that the flux drops at the truth DLA Lyα position
in the contaminated mock but not the uncontaminated one. Numbers
measured 2026-04-27:

| spectrum | flux 4567–4607 Å (DLA wing) |
|---|---:|
| loa-124 (with DLA) | mean ≈ **−0.04** (suppressed to ~0) |
| loa-0  (no DLA)    | mean ≈ **+0.31** (forest baseline) |

Both tests pass.

I also extended `examples/plot_smoke_result.py` so it can overlay the
matching uncontaminated spectrum and an analytical Voigt absorption
profile at MAP (NHI, z) on top of the inferred local continuum proxy.
The regenerated plots in `figures/smoke/` make the DLA trough visually
unmistakable, and they make the **N_HI bias visible**:

- `eboss_120046865.png` — Voigt(MAP NHI=21.45) overlay sits cleanly on the
  contaminated trough; matches the data depth.
- `y3_120046865.png` — Voigt(MAP NHI=21.63) overlay sits **deeper than the
  data**, i.e. the modelled trough wings are wider than the actual
  absorption. Direct visual confirmation of the +0.37 dex bias.
- `london_120046865.png` — no MAP overlay (London model didn't pass the
  detection threshold). The trough in the data is visible, just not fit.

## Reproducing the runs

```bash
LD_LIBRARY_PATH=$HOME/.local/usr/local/lib64:$LD_LIBRARY_PATH \
  python examples/smoke_one_spectrum.py \
    --specfile <2LPT spectra-16-789.fits> \
    --zcat <2LPT mock zcat.fits> \
    --target-id 120046865 \
    --preset {eboss,y3,london} \
    --data-root /nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection \
    --dla-samples-file <data-root>/data/dr12q/processed/dla_samples_a03.mat \
    --sub-dla-samples-file <data-root>/data/dr12q/processed/subdla_samples.mat \
    --single-absorber-model 0 --max-dlas 4 --filter-low-likelihood 1 \
    --num-dla-samples 10000 --num-subdla-samples 10000 \
    --max-workers 8 --batch-size 1250 \
    --output out/smoke/<preset>_multidla_120046865.h5
```

All HDF5 outputs and `.pkl` posteriors are under `out/smoke/`.
