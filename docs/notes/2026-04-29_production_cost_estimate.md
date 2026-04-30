# Production cost estimate: τ-EB overhead + node-hour budget for 1 M QSOs

> **TL;DR**: at GreatLakes 16-CPU profiling speeds, 1 M QSO inference costs
> ~290 node-hours on NERSC 256-CPU (Perlmutter scaling assumed).  Target is
> ~50 node-hours → we are **6× over budget**.  τ-EB itself adds essentially
> 0 (1 % of baseline); the cost is the bayes step.  This is a baseline-cost
> problem, not a τ-EB problem.

## Per-spectrum profile (GreatLakes, 10k QMC samples, max_dlas=3)

Two targets, profiled with 16-way `max_workers` parallelism:

| Target | regime | n_pix | baseline wall | τ-EB step | enabled wall | K-factor |
|---|---|---:|---:|---:|---:|---:|
| 120046865 | strong DLA (logNHI=21.26) | 1511 | 38.6 s | 0.42 s | 20.7 s | 0.54× |
| 100115792 | LLS (logNHI=18.17) — production rejects → fast | 651 | 17.7 s | 0.25 s | 14.3 s | 0.81× |

**Why ENABLED < BASELINE** on both targets: at the EB-chosen τ_best=2.0, the
1-DLA evidence drops below null, triggering the `parallel_log_model_evidences`
early-stop and skipping the 2-DLA / 3-DLA combinatorial searches. Baseline
at τ=1.0 doesn't always early-stop. On targets where production already
early-stops, the K-factor would be closer to 1 + 0.01.

## CPU-seconds per spectrum

`baseline_wall × max_workers` is a fair proxy for total CPU-time the bayes
step uses (the QMC marginalization parallelizes well over samples):

| Target | baseline CPU-s | τ-EB CPU-s | enabled CPU-s |
|---|---:|---:|---:|
| Strong DLA | 618 | 6.7 | 331 |
| LLS / no-DLA | 283 | 4.0 | 229 |

The "no-DLA spectrum runs faster" claim holds: the LLS target costs 46 % of
the strong-DLA target. Real LOA samples skew strongly to no-DLA (DLA
incidence ~5-10 % per sightline), so a population-mean cost is dominated
by the no-DLA path.

## Population-weighted cost for 1 M QSOs

Assumed mix: 90 % no-DLA-detected + 10 % DLA-detected.

| Treatment | mean CPU-s/spectrum | total CPU-hours for 1 M | 256-CPU node-hours |
|---|---:|---:|---:|
| BASELINE (no τ-EB) | 0.9 × 283 + 0.1 × 618 = **316** | 87 700 | **343** |
| ENABLED (τ-EB on) | 0.9 × 229 + 0.1 × 331 = 239 | 66 500 | 260 |

**Budget: 50 node-hours.  Current baseline: ~340 node-hours = 6.8× over.**

### Notes on this estimate

- **GreatLakes vs Perlmutter clock**: profiled on a GreatLakes 16-CPU node.
  Perlmutter CPU nodes (AMD EPYC 7763, 128 cores/socket) are typically
  20-30 % faster per core for vectorized numpy/scipy.  Best-case: 270
  node-hours.  Still 5× over.
- **Mix sensitivity**: if the no-DLA fraction is 95 % rather than 90 %,
  baseline mean drops to 300 CPU-s ⇒ 326 node-hours.  Doesn't move the
  conclusion.
- **`num_dla_samples` matters a lot**: profiles are at the production
  multi-DLA default 10k.  LLS / single-absorber runs use 100k QMC →
  baseline 4-10× higher per spectrum (we are not running 1 M spectra in
  LLS mode — that would be ~3000 node-hours at this rate).
- The I/O step (`read_spectra` + zcat lookup) is amortized in production
  via `desi-DLAGP.py` healpix batching and **not measured** in the profile.
  Production whole-pipeline numbers will be higher than the per-spectrum
  bayes-only estimate above.

## Where the time actually goes

From profile output of the strong-DLA target (38.6 s baseline):

```
build (null+subdla+dla):  0.03 s
set_data (3 models):      0.01 s
model_selection:          38.52 s   ← the entirety of the cost
```

The bayes step is 99.9 % of per-spectrum cost.  Within `model_selection`,
the dominant work is `parallel_log_model_evidences` for the 1-, 2-, and
3-DLA models — each of which runs the QMC sample evaluation for k DLA
combinations.  At max_dlas=3 with 10k samples, the 3-DLA combinatorial
search is the biggest single contributor; the FILTER-fix-#5 early-stop
helps when the data does not support 2+ DLAs but does not help on
populated DLA spectra.

## τ-EB cost is not the issue

τ-EB step alone is 0.25-0.42 s on these targets (1 % of baseline).
Enabling τ-EB does not move the budget question — at worst it adds 1 %,
at best it triggers earlier early-stops and saves time.  **Whether to
ship τ-EB does NOT depend on the budget conversation; the budget gap
is in the bayes step itself, which is unchanged by this PR.**

## How to close the 6× gap (out of scope for PR #5)

These are sketches, not commitments:

1. **Reduce `num_dla_samples`** from 10k to 2-3k, with a smarter (adaptive)
   sampler that places samples where the likelihood is high.  This is the
   user's "nested sampler" direction in the long-run plan
   (`project_long_run_sampler.md`).  Could be 3-5×.
2. **Vectorize across spectra** within a healpix.  Currently
   `bayes.model_selection` runs single-spectrum-at-a-time; batching the
   GP evaluations across (say) 16 spectra at once would amortize the
   Voigt + Woodbury costs.  Could be 2-4×.
3. **Drop max_dlas=3 → 2** for the survey-scale run.  3-DLA spectra are
   ~0.1 % of LOA; processing the long tail at max_dlas=3 costs the
   bulk of the bayes time.  Could be 2-3×.
4. **GPU port** of the QMC log-likelihood loop. A100 / H100 vs CPU is
   typically 50-100× on this kind of dense linear algebra workload.

Combining (1) + (3) might be the cheapest path to within 1× of budget.

## Profile reproducibility

```bash
python examples/profile_tau_eb_overhead.py \
    --target-id 120046865 \
    --max-workers 16 --batch-size 313 --num-dla-samples 10000
```

Adds one row to `tests/profile/results/tau_eb_overhead.csv`.  See that
CSV for any further targets profiled.
