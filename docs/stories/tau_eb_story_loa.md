# τ-EB on real DESI LOA data — what the recipe actually does on the survey

> **Status (2026-05-01)**: SLURM array `49071304`, 5000 random LOA
> QSOs (mock-0/loa-124 healpix that have spectra mirrored on
> GreatLakes), FILTER=1, max_dlas=4, 6× τ-grid. 16/16 tasks completed.
>
> **Privacy note (per `feedback_real_data_privacy` memory)**: the
> per-spectrum result TSV is **not committed** to the repo (real LOA
> spectra remain private until DESI release). Only the aggregate
> statistics in this doc are committable.

## Headline: τ-EB is essentially a no-op on real LOA data

The recipe was designed and validated on mocks, where it picks
median τ_factor ≈ 3 × Turner+2024 (closes ~60 % of the +0.09 dex
DLA-regime bias). **On real LOA, the median is τ_factor ≈ 1.5 ×
Turner+2024, with ~22 % of spectra preferring τ < 1×.** Real LOA
forest absorption is much closer to Turner+2024's prediction than
any of the three mock pipelines we tested.

| Population | median τ | mean τ | frac ≥ 2× |
|---|---:|---:|---:|
| 2lpt mock (n=49 000) | 3.0 | 2.75 | 77 % |
| london mock (n=48 000) | 3.0 | 2.72 | 76 % |
| saclay mock (n=49 000) | 3.0 | 2.64 | 76 % |
| **LOA real (n=5 000)** | **1.5** | **1.78** | **41 %** |

This is **the most important LOA finding in this PR**. The mocks all
agree at ~3× Turner; real data diverges at ~1.5×. Three independent
mock pipelines give the same answer, and that answer is wrong by a
factor of 2 in opacity vs DESI's real Y3 data.

## What this means for production

- **τ-EB is safe to run on real LOA**. The median spectrum gets a 1.5×
  τ correction — small adjustment, not the dramatic 3× the mocks
  suggested. Production catalogs will see modest changes, not wholesale
  re-fits.
- **The bias-closure mechanism that works in mocks is unlikely to fire
  the same way on real data.** Mocks have measurable τ-driven NHI bias
  because the GP forward model expects ~3× more forest absorption
  than the mock provides. Real LOA *already* matches the GP's
  expectation reasonably well — there's less for τ-EB to fix.
- **Mock-based closure numbers (50–65 %) are an UPPER bound on what
  τ-EB will do on real data.** Whatever residual bias exists in real
  data is partially driven by other factors (continuum, noise model)
  that τ-EB doesn't address.

This was always the right concern about the recipe — does it just
fix mock physics, or does it generalize? Answer: τ-EB does nothing
extreme on real data, but mocks systematically over-estimate forest
opacity by ~2× and τ-EB compensates.

## p_DLA distribution shift (BASELINE vs ENABLED, n=5000)

We don't have truth NHI on real LOA, so we can't measure bias —
only how many DLA-class detections each treatment makes:

| p_DLA cut | BASELINE detect | ENABLED detect | Δ |
|---|---:|---:|---:|
| ≥ 0.50 | 453 | 428 | −25 (−5.5 %) |
| ≥ 0.90 | 394 | 362 | −32 (−8.1 %) |
| ≥ 0.97 | 359 | 345 | −14 (−3.9 %) |
| ≥ 0.99 | 334 | 318 | −16 (−4.8 %) |

τ-EB tightens the catalog modestly — 4–8 % fewer DLAs depending on cut.
This is consistent with the mock result (where completeness drops
~3.5 pp; the percentage drops here are similar in magnitude).

Without truth catalog, we can't say which of the lost detections were
true DLAs vs false positives. The mock result implies most of the
lost detections are false positives (purity goes UP), but real data
may differ.

## τ_factor by z_qso bin — same monotonic decline as mocks

| z_qso bin | n | median τ (LOA) | median τ (2lpt mock for compare) |
|---|---:|---:|---:|
| [2.0, 2.3) | 1850 | **2.0** | 3.0 |
| [2.3, 2.6) | 1445 | 2.0 | 3.0 |
| [2.6, 3.0) | 1030 | 1.5 | 2.0 |
| [3.0, 5.5) | 675 | **1.0** | 1.5 |

At high z (≥ 3), real LOA wants exactly Turner+2024's τ_0 — the
recipe is a no-op. At low z (2.0–2.6), real LOA wants 2× Turner —
about half of what mocks want, but still a real correction.

The z-evolution is consistent across mocks and real data: τ_factor
*decreases* with z_qso. This is suggestive that something about the
low-z forest opacity in the GP forward model is mis-tuned — for both
mocks and real data, just less severely on real.

## Sample / methodology

- 5000 random TIDs from `QSO_cat_loa_main_dark_healpix_v3-altbal.fits`
  with `SPECTYPE=QSO`, `ZWARN=0`, `2.0 ≤ Z ≤ 5.5`.
- Pre-filtered to QSOs whose healpix coadd is mirrored on GreatLakes
  (only ~ 5 % of LOA is on GL; full data is on NERSC). 856
  available healpixes covered ~42 000 eligible QSOs; the 5000 sample
  is uniform-random from those.
- Each spectrum run twice: BASELINE (`enable_tau_eb=False`,
  τ_0 = Turner+2024) and ENABLED (τ-EB picked from 6× grid). Same
  inference settings as mocks: `--num_dla_samples 10000 --max_dlas 4
  --filter_low_likelihood 1`.
- BAL exclusion not applied at picker level for LOA — `bal_cat.fits`
  semantics differ slightly between LOA and mocks; for this run the
  no-truth no-BAL question is muddied by the LOA's non-BAL-flagged
  spurious-DLA emitters. Future LOA runs should add a BAL filter.

## Reproduction

```bash
python examples/pick_random_loa_targets.py --available-only \
    --n 5000 --seed 700 --out /tmp/loa_5k.tsv

sbatch --array=0-15 \
    --export=ALL,TARGETS_TSV=/tmp/loa_5k.tsv,\
OUT_BASE=$SCRATCH/phase_b_loa,\
N_TOTAL=5000,N_CHUNKS=16,\
FILTER_LOW_LIKELIHOOD=1,MAX_DLAS=4 \
    slurm/greatlakes/phase_b_5k_array.sh
```

Aggregate results land in `tests/profile/results/phase_b_aggregate.csv`
when `examples/aggregate_phase_b_results.py` is re-run.

## Open questions raised by this run

1. **Why do all three independent mock pipelines over-estimate forest
   opacity by ~2× vs real DESI data?** This is a mock-physics question
   that's outside this PR's scope, but worth flagging upstream.
2. **Should we run a 50k LOA sample at NERSC** where the full coadd
   files exist? The 5k here is statistically OK for the τ_factor
   distribution but doesn't allow per-NHI-bin detection-rate analysis
   without truth labels.
3. **What's a reasonable BAL filter for LOA?** The picker drops nothing
   right now; production-style runs use the AltBAL catalog. Worth
   adding `--exclude-bal` support to the LOA picker (currently mock-only).
