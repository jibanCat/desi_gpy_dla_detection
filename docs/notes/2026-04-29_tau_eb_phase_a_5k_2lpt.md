# Phase A: 5k random 2LPT τ-fit at production scale (no cherry-picking)

> Phase A = run the production τ-EB module (`fit_tau_eb_hcd_mask`) on a
> uniformly-random sample of 5000 2LPT QSOs (z_qso ≥ 2, all NHI regimes
> including no-truth-absorber). No SNR / multi-DLA / mid-forest filters
> like the n=90 picker. Validates that the production module makes
> sensible τ_0 choices at population scale — without paying the Phase B
> cost of running the full bayes step on each spectrum.
>
> Result CSV: `tests/profile/results/tau_eb_phase_a_5k_2lpt.tsv`.
> Targets file: `examples/pick_random_2lpt_targets.py` (seed=101, z≥2).
> Driver: `examples/run_tau_eb_phase_a.py`.
> Wall: 14.3 min on 16 CPUs ⇒ 2.71 s × 16 = 43 CPU-s per spectrum for
> the τ-fit step alone. Phase B (full production bayes on the same 5k)
> would add ~280 CPU-s/spec and is deferred.

## Headline

| Metric | NULL EB (production default) | MASK EB (HCD on, NOT default) |
|---|---:|---:|
| n_ok | 5000 | 5000 |
| median τ_factor | **3.00** | 3.00 |
| mean τ_factor | 2.63 | 2.77 |
| std | 1.11 | 1.09 |
| frac ≥ 2× | **77 %** | 81 % |
| frac at ceiling 4× | 28 % | 33 % |

Compared to the n=90 picker sample (cherry-picked: SNR≥2, single truth
absorber, mid-forest):

| Sample | n | median τ | mean τ | frac ≥ 2× | frac at 4× |
|---|---:|---:|---:|---:|---:|
| n=90 picker (DLA-only) | 90 | 2.00 | 2.36 | 71 % | 18 % |
| **n=5000 random (Phase A)** | **5000** | **3.00** | **2.63** | **77 %** | **28 %** |

The unfiltered population prefers **higher τ** than the picker subset.
That is consistent with the picker having selected for clean
single-DLA targets — those spectra have a strong absorber whose deep
trough is partly "carrying" the τ obligation. On the broader population
the recipe more often goes to τ ≥ 3.

## τ_factor distribution histogram (NULL EB, n=5000)

```
  τ=0.50:   225 ( 4.5 %) ▌▌
  τ=0.75:   111 ( 2.2 %) ▌
  τ=1.00:   164 ( 3.3 %) ▌
  τ=1.25:   245 ( 4.9 %) ▌▌
  τ=1.50:   407 ( 8.1 %) ▌▌▌▌
  τ=2.00:  1069 (21.4 %) ▌▌▌▌▌▌▌▌▌▌
  τ=3.00:  1362 (27.2 %) ▌▌▌▌▌▌▌▌▌▌▌▌▌
  τ=4.00:  1417 (28.3 %) ▌▌▌▌▌▌▌▌▌▌▌▌▌▌
```

28 % of targets pin at the τ_factor=4.0 grid ceiling. We measured 18 %
at the same ceiling on the n=90 picker sample. Suggests we should
either extend the grid (to 5.0, 6.0) and re-measure, or switch to a
continuous optimization (golden-section or scipy.optimize). Open work.

## Per-NHI-regime breakdown (NULL EB)

| Truth regime | n | median τ | mean τ | frac ≥ 2× |
|---|---:|---:|---:|---:|
| DLA (NHI ≥ 20.3) | 493 | 2.00 | 2.23 | 65 % |
| sub-DLA (19.0-20.3) | 1051 | 2.00 | 2.32 | 68 % |
| LLS (17.2-19.0) | 853 | 3.00 | 2.48 | 73 % |
| **none (no truth absorber)** | **2603** | **3.00** | **2.89** | **84 %** |

**This is a new finding worth thinking about.** The spectra with **no
truth absorber** prefer the highest τ on average, with 84 % choosing
τ ≥ 2× and a population mean of 2.89. The DLA-truth spectra prefer
lower τ (median 2.0).

Possible interpretations:

1. **τ on no-DLA spectra is "explaining away" forest fluctuations.**
   With no DLA in the model, the GP has only τ × A_lyα as a flux-
   suppression mechanism. Any forest fluctuations that look like extra
   absorption push τ up.
2. **The forward model is genuinely under-absorbed on most spectra.**
   The 2LPT mocks may have intrinsically higher forest opacity than
   Turner+2024 measured.
3. **Selection effect**: the picker that gave us n=90 selected
   z_dla ≤ z_qso − 0.05 with single absorber → those LOSes have a DLA
   close to z_qso, so most of the pixels are NOT in the DLA wing and
   the EB only has to compensate for the bulk-forest depth. The no-DLA
   sample is just bulk forest with no DLA "anchor" → τ goes higher.

If interpretation 1 is right, we may want to gate τ-EB on
"is there a putative DLA in this spectrum?" rather than running it
unconditionally. But that loses the point of EB-on-everything-uniformly.

## What Phase A does NOT measure

- **The production-bayes bias (in dex of NHI MAP)** — Phase A only
  runs the τ-fit step, not the full `bayes.model_selection`. It tells
  us *which* τ the recipe picks; it does not tell us whether using
  that τ closes the bias on the full population.
- **The "null" objective vs the "dla" objective** — only "null" was
  run here. The "dla" objective (max-over-NHI grid) is more expensive
  and was used in n=90; both should be compared at scale.

## Recommended Phase B

A SLURM array job running the full production inference (with and
without `--enable_tau_eb 1`) on the same 5000 targets. Cost estimate
(per the production-cost note): population mean baseline = 316 CPU-s,
enabled = 239 CPU-s. For 5000 spectra:

| run | total CPU-hours | NERSC node-hours (256 CPU) | Wall on 8 nodes (256 CPU each) |
|---|---:|---:|---:|
| BASELINE | 440 | 1.7 | 0.2 h |
| ENABLED | 332 | 1.3 | 0.2 h |

So 5000-spectrum Phase B is **~3 NERSC node-hours total** (1.7 baseline
+ 1.3 enabled) — well within budget. Recommend submitting this as a
SLURM array on the GreatLakes standard partition (16 array tasks ×
313 spectra each, ~1.7 h wall per task, all running in parallel).

## Cost / scaling sanity check

Phase A measured 2.71 s mean wall per spectrum on 16 CPUs. That maps
to 43 CPU-s per spectrum for the τ-fit step alone — consistent with
the per-target profile (0.25-0.42 s × 16 / N_workers). For 1 M QSOs:
43 × 1 M = 12 000 CPU-hours = **47 NERSC node-hours just for τ-EB**,
NOT including the bayes step. So the τ-EB step alone, run uniformly
across 1 M spectra, would be a ~50 node-hour budget item — close to
the user's full target. The bayes step is what makes the full pipeline
exceed budget, not τ-EB. (Reaffirms the cost-estimate doc.)

## Files

- `examples/pick_random_2lpt_targets.py` — random picker (no cherry)
- `examples/run_tau_eb_phase_a.py` — Phase A driver (multiprocessing)
- `tests/profile/results/tau_eb_phase_a_5k_2lpt.tsv` — per-target results
- `docs/notes/2026-04-29_tau_eb_n90_unbiasedness.md` — n=90 cherry-picked baseline
- `docs/notes/2026-04-29_production_cost_estimate.md` — cost projections
