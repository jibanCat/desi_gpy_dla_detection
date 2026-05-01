# τ-EB on Saclay — preliminary; multi-mock 5k Phase B in flight

> **Status (2026-04-30)**: SLURM array `49062628` is running 5000
> random Saclay QSOs (mock-0/juraLy8-124, z_qso ≥ 2, no cherry-picking)
> through both BASELINE and ENABLED τ-EB. Wall ETA ~3 h.  When it
> lands, the headline numbers below will be replaced with the
> production-bayes population result.
>
> What we have NOW: the n=18 picker subset (cherry-picked) tested
> with the *diagnostic* recipe.  Bias numbers are NOT directly
> comparable to the 2lpt 5k production Phase B; they're a sanity
> check that the recipe behaves on Saclay.

---

## Preliminary headline (n=6 DLA-truth picker subset, diagnostic recipe)

From `docs/notes/2026-04-29_voigt_lsf_sweep/scale_out/summary_n54.csv`
filtered to `mock=saclay, regime=DLA`:

| target_id | truth log NHI | prod MAP | prod bias | EB+mask MAP | EB+mask bias |
|---:|---:|---:|---:|---:|---:|
| 1377001320 | 20.88 | 21.13 | +0.24 | 20.90 | +0.02 |
| 6388000890 | 21.65 | 21.88 | +0.22 | 21.60 | −0.05 |
| 2103000740 | 20.57 | 21.38 | +0.81 | 21.28 | +0.71 |
| 4219000571 | 20.60 | 20.75 | +0.15 | 20.68 | +0.08 |
| 6397000973 | 20.72 | 20.73 | +0.01 | 20.38 | −0.34 |
| 2092000495 | 20.96 | 22.00 | +1.04 | 22.00 | +1.04 |
| **median** | | | **+0.23** | | **+0.05** |

Saclay was the closest match to the median-closure result on the
n=18 picker (81 % bias closure). The exception is TID 2092000495
where τ-EB did not move the bias at all (both stuck at NHI = 22.0,
the upper grid edge) — a saturation case.

---

## Example spectra

### Saclay 1377001320 — clean DLA closure

Truth log NHI = 20.88 at z=2.487.

![Saclay clean DLA closure (TID 1377001320)](../story_figures/saclay_01_dla_clean_close.png)

### Saclay 6388000890 — strongest DLA in the picker subset

Truth log NHI = 21.65 at z=2.078.  Heavy damping wings clearly
visible.  Production: NHI=21.88 (+0.22); EB+mask: NHI=21.60 (−0.05).

![Saclay strongest DLA (TID 6388000890)](../story_figures/saclay_02_strongest_dla.png)

### Saclay 2092000495 — failure mode: both saturate at NHI=22

Truth log NHI = 20.96 at z=1.874. Both treatments hit NHI=22.0 (the
NHI grid upper bound for these tests). z=1.874 is *below* z_qso − 0.5
in some search window definitions, so the absorber may be partly
outside the search range — at minimum, it's a low-z absorber where
the forest above it is sparse.  Hard case for any model.

![Saclay persistent-bias DLA (TID 2092000495)](../story_figures/saclay_03_dla_persistent_bias.png)

---

## Pending: Saclay 5k Phase B (job 49062628)

When this lands, expect to populate:
- median bias closure across n_DLA-truth_detected ≈ 250
- false-positive rate at p_DLA cuts ∈ {0.5, 0.9, 0.97, 0.99}
- per-NHI-regime breakdown
- τ_factor distribution (whether Saclay picks similar τ ≈ 3 to 2lpt)
- BAL-excluded analysis

---

## Mock-specific notes

- Saclay mock-0 lives at `juraLy8-124` (the `-Ly8` suffix means
  Lyman-series order 8 in the mock generation pipeline).  Saclay
  mock-1 is at `jura-124`.  This story doc uses mock-0 only.
- Truth file is `hcd_truth_cat.fits` (same convention as 2lpt).
- Saclay zcat columns: `TARGET_RA` / `TARGET_DEC` (same as 2lpt).
- Saclay was the most similar to 2lpt in the picker-subset closure
  rate (81 % vs 84 % London vs 54 % 2lpt picker).  Population-scale
  result will tell us whether that holds.
