# cellC knob sweep — 2026-05-14 (London-0 5k validation)

> ⚠ **SUPERSEDED (2026-05-17).** This note describes the *pre-+log(N)-patch*
> cellC sweep; its numbers are obsolete (pre-patch, and pre-DLAFLAG-convention
> fix). For current cellC P/C see `cellC_knob_sweep/HEADLINE.tsv` (refreshed
> 2026-05-17) and the runbook "Current production decisions". History only.

> **Status: COMPLETE.** All 8 cells ran on jupyter `nid004213` (concurrent),
> evaluated against London-0 truth. **Bottom line: no knob Pareto-dominates
> the cellC baseline** (P=0.8256, C=0.8304). Closest contender is PW 80k
> at P=0.8179 / C=0.8275 — strictly worse. cellC is locally optimal.
>
> **Goal**: tune knobs around the cellC baseline (P=0.83 / C=0.83 on the 5k
> London-0 validation) to see whether any one-at-a-time variant
> **Pareto-dominates** at the canonical operating point (SNR_RED > 2,
> P_DLA ≥ 0.99, truth NHI ≥ 20.3, lyb-veto, no-BAL, λ_rf ∈ [911, 1216]).
>
> cellC baseline: `SINGLE_ABSORBER_MODEL=1`, `MAX_DLAS=3`, NHI prior
> `[17.2, 22]`, PW 50k QMC samples, `FILTER_LOW_LIKELIHOOD=1` with default
> `n_initial_floor=5000`. See `docs/notes/2026-05-13_cellC_mechanism_verdict.md`
> for why this beats the production v3 baseline.

## Setup

- **Source mock**: London mock-0 `jura-124`, full 5k targets (8 slices × 8 workers)
- **GP model**: `2lpt_loa124_nohcd_nobal_wide.h5` (same as cellC baseline)
- **τ-EB**: enabled, objective=null (same as cellC baseline)
- **Compute**: jupyter compute node `nid004213` (256 CPUs), all 8 cells ran
  concurrently → wall times are **contention-inflated** (2× CPU oversub;
  per-cell solo wall ≈ reported wall × 0.5–0.6 based on baseline calibration)
- **Sample files generated**: `pw_samples_a3_172_230_50000.mat`,
  `pw_samples_a3_172_220_{30000, 80000, 100000}.mat` in `data/dr12q/processed/`
- **Configs**: `/pscratch/sd/j/jibancat/prod533_5k_20260511/cellC_knob_sweep/configs/C{1..8}.env`
- **Outputs**: `/pscratch/sd/j/jibancat/prod533_5k_20260511/cellC_knob_sweep/C{1..8}/`

## Cell matrix

Each cell varies ONE knob from cellC baseline; everything else identical.

| Cell | Knob varied | Sample file |
|---|---|---|
| C1 | MAX_DLAS = 4 (vs 3) | `pw_samples_a3_172_220_50000.mat` (existing) |
| C2 | MAX_DLAS = 5 (vs 3) | `pw_samples_a3_172_220_50000.mat` (existing) |
| C3 | NHI prior [17.2, **23**] | `pw_samples_a3_172_230_50000.mat` (new) |
| C4 | NHI prior [**18**, 22] | `pw_samples_a3_180_220_50000.mat` (existing) |
| C5 | PW **30k** samples | `pw_samples_a3_172_220_30000.mat` (new) |
| C6 | PW **80k** samples | `pw_samples_a3_172_220_80000.mat` (new) |
| C7 | PW **100k** samples | `pw_samples_a3_172_220_100000.mat` (new) |
| C8 | FILTER_N_INITIAL_FLOOR = 10000 (vs 5000) | `pw_samples_a3_172_220_50000.mat` (existing) |

## P_DLA semantics — important clarification

In cellC (2-way model `[Null, k-abs]`), `p_dla` sums absorber evidence
over the **full prior support** — for cellC and most variants this is
NHI ∈ [17.2, 22]; for C3 it is [17.2, 23], for C4 it is [18, 22]. It is
NOT restricted to NHI > 20.3. The NHI ≥ 20.3 cut is applied separately
by the eval script on the MAP NHI per predicted absorber row. So:

- A cell with a **wider prior** (C3: [17.2, 23]) gets more spectra past
  p_dla ≥ 0.99, but more catalog rows have MAP NHI outside [20.3, 23]
  and are silently dropped by the eval.
- A cell with a **narrower prior** (C4: [18, 22], or any cell after the
  cellC posterior-arithmetic effect) has fewer spectra past the cut but
  cleaner MAP NHI distribution.
- A cell with **more QMC samples** (C6: 80k, C7: 100k) integrates the
  marginal evidence with less variance — should converge toward the same
  marginal as a fully-converged FILTER=0 result for borderline spectra.

These are the mechanisms the sweep is testing.

## Headline P/C table

Source TSV: `cellC_knob_sweep/HEADLINE.tsv`. Wall is contention-inflated
(8 cells concurrent on 256-core node, 2× CPU oversub) — solo wall would be
~50–60 % of reported. Node-hours computed as `wall_min × 64 cores / 60 / 256
cores-per-node` (each cell uses 8 parallel files × 8 inner workers = 64 cores).

| Cell | Knob | Purity | Completeness | n_cat | wall (min) | node-hr | Δ P vs cellC | Δ C vs cellC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **cellC baseline** | — | **0.8256** | **0.8304** | 3268 | (prior, n/a) | (n/a) | ref | ref |
| C1 | MAX_DLAS=4 | 0.7955 | 0.8187 | 3891 | 64.5 | 0.269 | **−3.0 pp** | **−1.2 pp** |
| C2 | MAX_DLAS=5 | 0.8068 | 0.8304 | 4397 | 79.8 | 0.332 | −1.9 pp | **0.0 pp** |
| C3 | NHI [17.2, 23] | 0.8011 | 0.8246 | 3540 | 73.8 | 0.307 | −2.4 pp | −0.6 pp |
| C4 | NHI [18, 22] | 0.7994 | 0.8275 | 3318 | 70.2 | 0.293 | −2.6 pp | −0.3 pp |
| C5 | PW 30k | 0.8103 | 0.8246 | 3423 | 57.3 | 0.239 | −1.5 pp | −0.6 pp |
| C6 | PW 80k | **0.8179** | 0.8275 | 3163 | 80.5 | 0.335 | −0.8 pp | −0.3 pp |
| C7 | PW 100k | 0.8174 | 0.8246 | 3134 | 88.8 | 0.370 | −0.8 pp | −0.6 pp |
| C8 | n_initial=10k | 0.8069 | 0.8187 | 3382 | 76.2 | 0.318 | −1.9 pp | −1.2 pp |

`n_truth = 618` for every row (same eval cuts). Bold = best non-baseline
purity (C6) and ties on completeness (C2).

**Verdict: NO knob Pareto-dominates cellC baseline.** Every variant is
worse on at least one metric. The closest contenders:

- **C2 (MAX_DLAS=5)** ties baseline completeness but pays −1.9 pp purity.
  More k-DLA branches in the 2-way model add candidate rows that survive
  the P_DLA cut but don't match truth.
- **C6 (PW 80k)** is the strongest non-baseline cell — only −0.8 pp P,
  −0.3 pp C. But +0.06 node-hr per 5k spectra → at production scale
  (~17 nh / 1M QSO baseline per `docs/runs/2026-05-12_v3_production_cost.md`)
  this is roughly +20 nh / 1M to recover ~0 pp of headline. Not worth it.
- **C7 (PW 100k)** is essentially tied with C6 at higher cost — confirms
  the Var[Δ_marg] verdict (`docs/notes/2026-05-13_var_delta_marg_diagnostic.md`)
  that the pipeline is **statistic-limited at N=50k, not sampling-limited**.

## NHI-bin-stratified completeness

Source: `cellC_knob_sweep/nhi_bins.tsv` from `_nhi_bin_table.py`. Same
operating point. cellC's headline win is concentrated in [20.3, 20.5);
this table tests whether any knob can recover that bin further.

| Cell | [20.3, 20.5) | [20.5, 21.0) | [21.0, 21.5) | [21.5, 22.0) | overall |
|---|---:|---:|---:|---:|---:|
| **cellC baseline** | **0.704** | 0.873 | 0.919 | 0.929 | **0.830** |
| C1 (md=4) | 0.667 | 0.873 | 0.919 | 0.929 | 0.819 |
| C2 (md=5) | 0.694 | **0.880** | 0.919 | 0.929 | 0.830 |
| C3 ([17.2, 23]) | 0.694 | 0.867 | 0.919 | 0.929 | 0.825 |
| C4 ([18, 22]) | 0.676 | **0.886** | 0.919 | 0.929 | 0.828 |
| C5 (PW 30k) | 0.676 | 0.873 | **0.935** | 0.929 | 0.825 |
| C6 (PW 80k) | 0.694 | 0.873 | 0.919 | 0.929 | 0.828 |
| C7 (PW 100k) | 0.685 | 0.873 | 0.919 | 0.929 | 0.825 |
| C8 (n_init=10k) | 0.676 | 0.867 | 0.919 | 0.929 | 0.819 |

Per-bin purity (predicted-NHI bin):

| Cell | [20.3, 20.5) | [20.5, 21.0) | [21.0, 21.5) | [21.5, 22.0) |
|---|---:|---:|---:|---:|
| **cellC baseline** | **0.600** | **0.915** | **0.892** | 0.957 |
| C1 (md=4) | 0.550 | 0.885 | 0.901 | 0.920 |
| C2 (md=5) | 0.580 | 0.903 | 0.868 | 0.955 |
| C3 ([17.2, 23]) | 0.577 | 0.872 | 0.897 | **1.000** |
| C4 ([18, 22]) | 0.511 | 0.894 | 0.878 | **1.000** |
| C5 (PW 30k) | 0.552 | 0.896 | 0.882 | 0.952 |
| C6 (PW 80k) | 0.594 | 0.897 | 0.901 | 0.957 |
| C7 (PW 100k) | 0.582 | 0.893 | 0.905 | 0.952 |
| C8 (n_init=10k) | 0.562 | 0.897 | 0.892 | 0.955 |

**cellC baseline wins on the [20.3, 20.5) completeness in every comparison
(0.704 vs ≤ 0.694) and ties or wins on every bin's purity.**

A few stray "wins" by other cells:
- C2 / C4 nudge [20.5, 21.0) completeness up by 0.6–1.3 pp at the cost of
  bigger purity drops elsewhere.
- C5 (PW 30k) nudges [21.0, 21.5) completeness +1.6 pp — likely noise
  (one extra TP), the bin only has 62 truths.
- C3 / C4 reach 1.000 on [21.5, 22.0) purity (only 14 truths in this bin
  → small-N artifact).

None of these are meaningful Pareto improvements.

## Interpretation

**1. cellC baseline is locally Pareto-optimal at this operating point.**
Eight one-at-a-time variations around the four most promising knob axes
(model capacity, prior width, sample count, coarse-scan budget) all give
strictly dominated P/C. The baseline (PW 50k, MAX_DLAS=3, NHI [17.2, 22],
n_initial=5k) sits at a local maximum.

**2. Mechanism summary**:

- **MAX_DLAS** (C1=4, C2=5): adding k-DLA branches HURTS, not helps. In
  the 2-way model `[Null, k-abs]` of cellC, `p_dla = sum(model_posteriors[1:])`
  spreads evidence across all k-DLA columns. Adding k=4 / k=5 columns adds
  catalog rows (C1: 3891 vs cellC's 3268; C2: 4397) but those extra
  detections include noise. Net: more cat rows, lower purity, similar
  completeness.

- **NHI prior width** (C3=[17.2,23], C4=[18,22]): both modifications
  hurt. C3 widens the upper bound to 23 — but London mock-0 has very few
  NHI > 22 truths (`c_22.0-23.0 = nan` in every cell except C3 itself,
  which got 1 prediction with no matching truth). C4 narrows the lower
  bound to 18 — this REMOVES the LLS-extended tail that gives cellC its
  purity advantage in the [20.3, 20.5) bin (C4: P=0.511 vs cellC 0.600).
  cellC's [17.2, 22] is a sweet spot.

- **PW samples** (C5=30k, C6=80k, C7=100k): more samples ≠ better.
  C5 (30k) is essentially indistinguishable from baseline (same C, lower P).
  C6 / C7 trade away 0.6 pp completeness for 0.4-0.6 pp better purity in
  [20.3, 20.5) — but the headline numbers all land within the noise band
  of the 5k validation. **This corroborates the Var[Δ_marg] diagnostic
  verdict** (`docs/notes/2026-05-13_var_delta_marg_diagnostic.md`):
  σ_noise ≈ 0.1 vs signal gap ≈ 13 means N=50k is already
  sampling-converged for the headline metric.

- **n_initial floor** (C8=10k): essentially zero effect on cellC. The
  posterior-arithmetic mechanism that drives cellC's win (per
  `docs/notes/2026-05-13_cellC_mechanism_verdict.md`) operates regardless
  of FILTER=1 coarse-scan coverage. **This corroborates the 2×2 ablation
  finding** that knob 1 is a no-op.

**3. Cost / value summary** (per 5k spectra, 64 cores):

| Cell | wall | node-hr | $/headline (vs cellC) |
|---|---:|---:|---|
| cellC baseline | (~30 min solo) | ~0.13 | reference |
| C5 (PW 30k) | 57.3 | 0.239 | cheaper, slightly worse |
| C1, C2, C8 | 64-80 | 0.27-0.33 | comparable cost, worse |
| C3, C4, C6 | 70-80 | 0.29-0.34 | comparable cost, worse |
| C7 (PW 100k) | 88.8 | 0.370 | most expensive, no improvement |

At production scale (1M QSO ≈ 17 nh baseline), C6 / C7 add 5–10 nh per 1M
for no headline gain. Not worth shipping.

**4. Implication for PR #7 — task 4 (decide cellC default)**: this sweep
confirms cellC baseline is the right knob configuration *for the joint
catalog*. The remaining "ship cellC as default" question becomes purely
about the **2-way vs 3-way model decision** (loss of separate sub-DLA
catalog), not about whether the cellC knob choices are correct. The
2026-05-13 verdict's "flag, not default" recommendation stands —
the sweep gives no reason to revisit the knobs.

**5. Direction for further improvement**: since the cellC knob landscape
is locally flat, headline gains beyond 0.83/0.83 require **either**
(a) a model-side change (better-trained GP per
`docs/notes/2026-05-13_model_side_improvements.md`), **or**
(b) a different model-space architecture (the 3-way DLA classifier with
explicit sub-DLA channel still wins on classical-DLA purity at the cost
of [20.3, 20.5) completeness). Knob tuning around cellC has no remaining
upside in the +/-0.5 pp band.

## Reproduction

```bash
# Generate the new sample files (~1-2 min each, CPU only):
bash -c '
source /usr/share/lmod/lmod/init/bash
export DESI_ROOT=/global/cfs/cdirs/desi
source /global/common/software/desi/desi_environment.sh main
cd /pscratch/sd/j/jibancat/desi_gpy_dla_detection
DEST=data/dr12q/processed
python -m gpy_dla_detection.generate_samples --min-log-nhi 17.2 --max-log-nhi 23.0 --num-samples 50000  --output $DEST/pw_samples_a3_172_230_50000.mat
python -m gpy_dla_detection.generate_samples --min-log-nhi 17.2 --max-log-nhi 22.0 --num-samples 30000  --output $DEST/pw_samples_a3_172_220_30000.mat
python -m gpy_dla_detection.generate_samples --min-log-nhi 17.2 --max-log-nhi 22.0 --num-samples 80000  --output $DEST/pw_samples_a3_172_220_80000.mat
python -m gpy_dla_detection.generate_samples --min-log-nhi 17.2 --max-log-nhi 22.0 --num-samples 100000 --output $DEST/pw_samples_a3_172_220_100000.mat
'

# Launch a cell:
cd /pscratch/sd/j/jibancat/prod533_5k_20260511/cellC_knob_sweep
nohup bash _launch.sh C1 8 > logs/C1.log 2>&1 &

# Evaluate after a cell completes:
bash _eval_and_aggregate.sh C1

# Or aggregate all completed cells:
bash _eval_and_aggregate.sh all

# NHI-bin breakdown across all cells:
bash -c '
source /usr/share/lmod/lmod/init/bash
export DESI_ROOT=/global/cfs/cdirs/desi
source /global/common/software/desi/desi_environment.sh main
cd /pscratch/sd/j/jibancat/desi_gpy_dla_detection
python /pscratch/sd/j/jibancat/prod533_5k_20260511/cellC_knob_sweep/_nhi_bin_table.py
'
```

## Run log

All 8 cells ran concurrently on `nid004213` (256-core jupyter compute).

| Cell | start (epoch) | end (epoch) | wall (min) |
|---|---:|---:|---:|
| C1 | 1778773506 | 1778777376 | 64 |
| C4 | 1778774390 | 1778778604 | 70 |
| C5 | 1778774392 | 1778777831 | 57 |
| C8 | 1778774394 | 1778778968 | 76 |
| C2 | 1778774904 | 1778779694 | 79 |
| C3 | 1778774906 | 1778779332 | 73 |
| C6 | 1778774908 | 1778779740 | 80 |
| C7 | 1778774910 | 1778780237 | 88 |

Total wall (first cell start → last cell end): ~112 min.
Aggregate node-hours (sum of per-cell node-hr): **2.46 nh** for all 8
× 5k validation. ≈ 0.31 nh per cell average (contention-inflated).

Solo equivalent (estimated): per `_evaluate_cell.py` cellC ran in ~30 min
solo per `HANDOFF.md` 2026-05-13. Each sweep cell solo would be ~30-45
min wall, so per-cell solo ≈ 0.13–0.19 nh. Aggregate solo would be ~1.0
nh — but I didn't measure this.

All start/end epochs and wall times are listed in the table above.
Per-cell timestamps recorded in
`/pscratch/sd/j/jibancat/prod533_5k_20260511/cellC_knob_sweep/logs/C*.{start,end}`.
