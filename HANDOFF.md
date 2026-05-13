# Handoff — session 2026-05-12 (low-SNR completeness root-cause)

> Written ~12:10 PT. Jupyter session ending soon. Branch: `production_533`. Uncommitted.
> Supersedes the 2026-05-11 handoff in this file — the previous one is preserved
> in git at the prior commit if needed.

## TL;DR for next-Claude

1. **Two concrete wins that compose toward 85/85 at SNR > 2**:
   - Swap production GP for the v3 phase2_desi `2lpt_loa124_nohcd_nobal_wide`
     model. **Drop-in replacement, +1pp purity / +2.3pp completeness at
     P_DLA≥0.99 on London 8f.** Converted h5 ready to use:
     `/pscratch/sd/j/jibancat/prod533_5k_20260511/null_gp_test/converted/2lpt_loa124_nohcd_nobal_wide.h5`
   - Switch the cut from absolute `p_DLA > 0.99` (or `Δ_marg > 0`) to
     **null-quantile-calibrated threshold** (`Δ_marg > p90(null) ≈ -7.8` per
     the MAP-detection prototype). **Not yet population-tested** — this is the
     next experiment.

2. **All three "Code follow-ups" from the prior session are REFUTED**. See
   `/pscratch/sd/j/jibancat/prod533_5k_20260511/RECOMMENDATION_SNR_GT2.md`.

3. **The actual mechanism is prior-volume dilution** with peak width
   < 0.001 z × < 0.01 NHI. Laplace volume penalty `-½ log|H|` ≈ 7-10 logL
   matches the marginal-vs-MAP gap exactly. See memory note
   [`project_prior_dilution_finding.md`](../../global/homes/j/jibancat/.claude/projects/-pscratch-sd-j-jibancat-desi-gpy-dla-detection/memory/project_prior_dilution_finding.md).

---

## The 5 missed candidates (load-bearing across all 2026-05-12 tests)

| h5 file (under `/pscratch/sd/j/jibancat/prod533test-20260511_1333/london0_y3/processed/`) | row | TID | z_qso | z_truth | log NHI_truth | prod Δ_at_truth | prod Δ_marg |
|---|---:|---:|---:|---:|---:|---:|---:|
| processed-spectra-16-10.h5  | 1007 | 105798   | 2.426 | 1.9837 | 20.322 | +5.31 | -8.47 |
| processed-spectra-16-11.h5  |   14 | 1798     | 2.580 | 2.0480 | 20.543 | +7.22 | -7.11 |
| processed-spectra-16-129.h5 |  757 | 80198262 | 2.618 | 2.0999 | 20.435 | +8.57 | -6.40 |
| processed-spectra-16-2.h5   |  246 | 64988    | 2.514 | 1.9900 | 20.413 | +9.62 | -4.30 |
| processed-spectra-16-32.h5  |  598 | 20115135 | 2.248 | 2.0917 | 20.480 | +10.05 | -3.54 |

The `Δ_at_truth` is +5 to +10 → signal IS there at the peak. The `Δ_marg`
is −8 to −3 → prior averages it away. The peak is narrower than ±0.001 z × ±0.01 NHI.

---

## Investigations run this session

All artifacts under `/pscratch/sd/j/jibancat/prod533_5k_20260511/`.

| Investigation | Outdir | Verdict |
|---|---|---|
| Narrow [20, 21] prior, full inference | `london_pw14_2021_tau_eb/`, `molly/london_pw14_2021_tau_eb_8f/` | **REGRESSION** — P drops 6-7pp at every cut |
| FILTER=0 × samples × prior sweep on 5 candidates | `filter_sweep/RESULTS.md` | **REFUTED** — all 30 (TID, config) cells give Δ ∈ [-4.4, -10] |
| Hypothesis #1: SNR-aware z_tol + early-stop removal | `hypothesis1_sweep/RESULTS.md` | **REFUTED** — Pop B (118 missed) recovers 0-1/118 (+0.0 to +0.85 pp) |
| Tight-box QMC sweep ("falsify dilution") | `dilution_test/RESULTS.md` | Initially read as REFUTED but **the QMC tight box missed the peak**; v3-nullGP single-point eval corrected the Δ_at_truth numbers |
| v3 phase2_desi `.pt` model loading on 5 candidates | `null_gp_test/RESULTS.md`, `null_gp_test/converted/*.h5` | **PARTIALLY POSITIVE** — null-GP HCD-contamination hypothesis mostly refuted (prod Δ_at_truth already +5 to +10), but v3_loa124 shows +5.28 logL avg shift (outlier-driven on 5 candidates) |
| **v3_loa124 population-scale on London 8f** | `london_v3_loa124_pw14_tau_eb/`, `molly/london_v3_loa124_pw14_tau_eb_8f/` | **POSITIVE — DEPLOYABLE** (see numbers below) |
| τ-EB on 5 candidates | `tau_eb_5cand/RESULTS.md` | **REFUTED** — mean ΔΔ_at_truth = -0.48 logL; lifts log_l_null and log_l_DLA equally |
| MAP-detection prototype (blind + truth-anchored + Laplace + null FP cal) | `map_detection_test/RESULTS.md`, `map_detection_test/null_distribution.png` | **REFUTED as standalone** but produced the **null-quantile-threshold insight** |

### Headline P/C at SNR > 2 with v3_loa124 (vs baseline PW14+τ-EB+lyb_veto)

| P_DLA cut | Baseline P / C | v3_loa124 P / C | ΔP / ΔC |
|---:|---:|---:|---:|
| ≥ 0.99    | 83.55 / 74.27 | **84.52 / 76.61** | +1.0pp / +2.3pp |
| ≥ 0.999   | 83.99 / 69.01 | 85.47 / 73.98 | +1.5pp / +5.0pp |
| ≥ 0.99999 | 85.43 / 63.45 | **88.72 / 69.01** | +3.3pp / +5.6pp |

v3 wins at every cut. Inference wall 38 min for 8 files × 8 parallel.

---

## What's NOT been done — the immediate next experiments

### #1 (the user just asked for this — interrupted): null-quantile threshold sweep on London 8f

The MAP-detection agent found that on n=48 clean nulls (SNR ∈ [2, 4], no truth
DLA in window), the prior-marginal Δ has the BEST discrimination of any score —
better than Δ_MAP or Δ_Laplace. The catch is the absolute `Δ > 0` (or
`p_DLA > 0.99`) cut is mis-calibrated.

**Experiment**: on the existing v3_loa124 combined catalog at
`/pscratch/sd/j/jibancat/prod533_5k_20260511/london_v3_loa124_pw14_tau_eb/`:
1. Build a null population: truth-match against
   `/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/dla_cat.fits`,
   take rows in v3 catalog with NO truth DLA in [z_Lyβ+3000km/s, z_qso-3000km/s].
2. Compute p90, p95, p99 quantiles of Δ_marg (`log_likelihoods_dla[0] − log_likelihoods_no_dla`)
   on null population.
3. Apply each quantile as a threshold instead of P_DLA cut, compute P/C on
   the truth set. Sweep.
4. Headline: does v3 + null-quantile threshold reach 85/85 at SNR>2?

Cheap: no inference, just re-analysis. ~10-30 min wall as an agent.

Starter pattern: extend `examples/molly_faithful_pc_plots.py` to optionally
take a Δ_marg threshold instead of P_DLA. Or write a small `_nullquantile_*.py`
that reads the combined.h5, joins truth, sweeps quantiles.

### #2: v3_loa124 on Saclay mock-0 (distribution-shift check)

v3 was trained on 2LPT mock-0 with HCD/BAL exclusion. London 8f confirms it
works on London. Need to check Saclay mocks don't regress — different forest
amplitude, BAL prevalence. Spectra paths in this repo's old HANDOFF.md (preserved
in git history before this rewrite).

### #3 (further out): multi-DLA MAP + Laplace prototype

The MAP prototype only ran k=1. To extend to k=2,3:
- Greedy: at each k, build `DLAGP(...)` instance with k-1 DLAs fixed at MAP
  coords (existing code: pass `nhis=[...]`, `z_dlas=[...]` to the likelihood);
  optimize the k-th `(z, NHI)` over the same bounds excluding `|z_k - z_j| < min_z_separation`.
- Hessian in full 2k-D space at convergence (finite diffs).
- Laplace evidence `log p(D|M_k) ≈ log p(D|θ_MAP) + (k·d/2)log(2π) - ½ log|H|`.
- Plug into the existing `bayesian_model_selection.py` formula unchanged.

Note: MAP's logN=19 ghost-DLA pathology (forest noise → spurious weak absorber
at the prior boundary) means MAP-detection has ~30% null FP rate. The multi-DLA
extension will have MORE ghost DLAs in low-SNR spectra. **Don't expect this to
be a magic completeness booster** — the Laplace-corrected version (~22% null FP)
might be ok but still worse than the prior-marginal score. Worth doing for the
formalism but not the headline experiment.

---

## Repo state — uncommitted (do not commit yet)

```
M  CLAUDE.md
M  examples/analyze_production_catalog.py
?? HANDOFF.md (this file)
?? examples/_sweep_cuts.py
?? examples/gp_native_pc_plots.py  (M? not sure; check git status)
?? examples/molly_faithful_pc_plots.py
?? examples/inspect_loa_spectra.py
?? slurm/run_local.sh                 # env-overridable LEARNED_FILE, ENABLE_TAU_EB, etc.
?? slurm/configs/london0_y3.env       # env-overridable MAX_DLAS, FILTER_LOW_LIKELIHOOD, SINGLE_ABSORBER_MODEL
?? data/dr12q/processed/pw_samples_a3_200_210_50000.mat  # PW14 [20,21] (already-tested, regression)
?? data/dr12q/processed/pw_samples_a3_190_220_100000.mat # generated by filter_sweep agent
?? investigate_*.py                  # 2026-05-11 investigation scripts; obsolete
?? _sweep_*.py, _hypothesis1_*.py, _map_*.py, _v3nullgp_*.py, _taueb_*.py, _build_popB.py, _lookup_5cand.py, _smoke_min.py
?? gpy_dla_detection/_pme_patched.py # patched parallel_log_model_evidences from hypothesis1 agent
?? various slurm logs, learnlogs, notebooks
```

**Decision needed on PR scope**: The cleanest PR-worthy changes are:
- `examples/molly_faithful_pc_plots.py` (the validation script) — solid, reusable
- `slurm/run_local.sh` env-overridable additions
- `slurm/configs/london0_y3.env` env-overridable additions
- v3 model conversion path (a small `convert_pt_to_h5.py` script — the agent
  wrote one as part of the harness; PR-worthy if cleaned up)

The investigate_*.py, _sweep_*.py, _map_*.py harnesses are one-off; **delete
or move to a `tools/research/` if keeping**.

---

## v3 model conversion notes (for future use)

The `.pt` checkpoints at
`/global/cfs/cdirs/desicollab/users/jibancat/DLA/learned/phase2_desi/{2lpt_loa0_wide,2lpt_loa124_nohcd_nobal_wide}/checkpoints/phase2_desi_checkpoint_final_iter1499.pt`
have:
- `M:(5662, 30), log_omega:(5662,), log_c_0, log_tau_0, log_beta, mu:(5662,)`
- **No `rest_wavelengths` stored.**
- **Grid is `linspace(850.75, 1699.90, 5662)` at dλ=0.15** — extended to CIV.
  Verified by matching mu peaks to 8 known emission lines (the v3-nullGP agent's
  validation in `null_gp_test/mu_grid_validation.png`).

To convert to NullGPMAT-compatible h5: see
`/pscratch/sd/j/jibancat/prod533_5k_20260511/null_gp_test/_v3nullgp_convert.py`.
Just pulls fields from the .pt, adds `rest_wavelengths`, `max_noise_variance=9.0`,
`normalization_min_lambda=1310.0`, `normalization_max_lambda=1325.0`, writes h5.

Already-converted h5:
- `/pscratch/sd/j/jibancat/prod533_5k_20260511/null_gp_test/converted/2lpt_loa0_wide.h5`
- `/pscratch/sd/j/jibancat/prod533_5k_20260511/null_gp_test/converted/2lpt_loa124_nohcd_nobal_wide.h5`

---

## Memory state

`/global/homes/j/jibancat/.claude/projects/-pscratch-sd-j-jibancat-desi-gpy-dla-detection/memory/MEMORY.md`
has been updated with:
- `project_prior_dilution_finding.md` — the corrected dilution mechanism note
  (initial "filter is the cause" → corrected to peak narrowness)

The five existing memory entries (NERSC paths, test paths, dry-run, SNR cut,
verify-root-cause) are still accurate.

---

## Zombie process to clean up

```
PID 1162537, jibancat, sleeping, 146 MB
python _hypothesis1_sweep.py --pop A --variants baseline
```

Started by the hypothesis #1 agent's harness during an early debug run; agent
ran the real sweep from a different invocation. Idle, harmless, but worth a
`kill 1162537` from a fresh shell on the next salloc.

---

## Key model parameters used throughout (production defaults)

| Param | Value | Notes |
|---|---|---|
| `LEARNED_FILE` | `learnlogs/model_epoch_920.h5` | Current production; **swap to v3 h5 above** |
| `prev_tau_0` | 0.00246 | Turner+2024 |
| `prev_beta` | 3.62 | Turner+2024 |
| `NUM_DLA_SAMPLES` | 50000 | PW14 [19, 22] |
| `DLA_SAMPLES_FILE` | `data/dr12q/processed/pw_samples_a3_190_220_50000.mat` | PW14 prior |
| `MAX_DLAS` | 3 | |
| `FILTER_LOW_LIKELIHOOD` | 1 | |
| `ENABLE_TAU_EB` | 1 | per PR #5 |
| `TAU_EB_OBJECTIVE` | "null" | |
| `MIN_LAMBDA` | 911.75 | Lyα window inner |
| `MAX_LAMBDA` | 1216.75 | Lyα window outer |
| `DLAMBDA` | 0.15 | |
| `K` | 30 | GP rank |
| `NUM_FOREST_LINES` | 3 | |

---

## Compute environment for fresh sessions

The Jupyter Bash shell needs a workaround to source DESI env (sourcing fails
silently in the parent shell — module path or PYTHONPATH state pollution).
Use a clean subshell each time:

```bash
bash -c '
source /usr/share/lmod/lmod/init/bash
export DESI_ROOT=/global/cfs/cdirs/desi
source /global/common/software/desi/desi_environment.sh main
python ...
'
```

This gets `/global/common/software/desi/perlmutter/desiconda/20240425-2.2.0/conda/bin/python`
(Python 3.10.14) with `desispec 0.71.2.dev10051` and `torch`.

For compute, the user's preferred pattern (from prior handoff) remains:
```bash
salloc -N 1 -C cpu -q interactive -t 4:00:00 -A desi
```

The session today ran inside a Jupyter session (job 52862143, 256 CPUs / 487 GB,
QOS=jupyter, partition=urgent_milan_ss11). Worked fine for the 38-min v3
inference. For 1M-QSO production, sbatch the normal slurm scripts.

---

## Open questions to resume on

1. **Does v3_loa124 + null-quantile threshold reach 85/85 at SNR > 2?** (#1
   above; quickest experiment, ~30 min.)
2. **Does v3_loa124 work on Saclay mock-0?** (Distribution-shift sanity check;
   ~40 min inference + molly.)
3. **Should v3_loa124 replace the production GP for the 1M-QSO run?** Cost-wise
   v3 has 5662 pixels vs production 3798 → maybe ~1.5× slower per spectrum;
   need to measure on the existing run.
4. **MAP-detection at logN<19 boundary**: the 2/5 candidates that found a
   ghost-DLA at the boundary suggest the GP+DLA model has *intrinsic*
   sensitivity to broad weak absorbers. This is the same mechanism producing
   the 30% null FP rate. Worth a separate investigation: is the broad-weak-DLA
   sensitivity also creating false positives in the production catalog at
   strict P_DLA cuts? Might be a real audit item even if MAP-detection isn't
   deployed.
