# 2026-05-18 — trained-GP model sweep (model_sweep)

> **Status**: DONE (sbatch job 53077686, completed 2026-05-18 11:48).
> **Verdict: the GP-model swap does NOT reach 85/85.** The best healthy
> model — **V1, `2lpt_loa124_nohcd_nobal_wide_m`** — is only a mild
> Pareto improvement over the β-collapsed baseline (+0.4pp P / +2.2pp C),
> at 0.804 / 0.864. Purity stays ~0.80–0.81 across *every* model tested.
> The β-collapse was **not** the thing capping the purity frontier.
> **Recommendation: ship V1** (it must replace the deprecated β-collapsed
> baseline regardless), but 85/85 needs deeper work, not a model swap.
>
> Sweep dir: `/pscratch/sd/j/jibancat/prod533_5k_20260511/model_sweep/`

## Why

Every prior sweep used `null_gp_test/converted/2lpt_loa124_nohcd_nobal_wide.h5`
— the **β-collapsed deprecated model** (β=1.45 vs the Turner+2024 prior
3.62; memory `project_baseline_model_beta_collapse`). The standing
hypothesis was that this model caps the P/C frontier and that swapping in
a healthy PR6 `phase2_desi` `_m` model would lift it ~3pp toward 85/85.
This sweep tests that.

## Method

5 cells, London-0 5k slice, **identical** current-best config (2-way
single-absorber, MAX_LAMBDA=1250, PW 50k, NHI [17.2,22], τ-EB null,
FILTER=1, MAX_DLAS=3) — only `LEARNED_FILE` varies. Fixed molly recipe,
new DLAFLAG convention. The 4 candidates are healthy `_m` models
(MATLAB norm band [1425,1475], β ≈ 2.97–3.57); details + verification in
`docs/notes/2026-05-17` work and `model_sweep/README.md`.

## Result

| cell | model | P | C | wall (min) |
|---|---|---:|---:|---:|
| V0_baseline | β-collapsed baseline (β=1.45) | 0.8000 | 0.8421 | 44.9 |
| **V1_2lpt124m** | **2lpt loa124 nohcd nobal `_m`** | **0.8040** | **0.8638** | 50.0 |
| V2_2lpt0m | 2lpt loa0 `_m` | 0.7825 | 0.8576 | 52.2 |
| V3_loa_nodla | real-LOA no-dla no-bal `_m_normmask` 3000it | 0.7914 | 0.8576 | 50.9 |
| V4_loa_withbal | real-LOA no-hcd with-bal `_m_normmask` 3000it | 0.8047 | 0.8421 | 33.4 |

## Interpretation

**The model swap does not lift the frontier to 85/85.** Every model —
β-collapsed *and* healthy — sits at purity ≈ 0.78–0.81, completeness ≈
0.84–0.86. The hoped-for ~3pp purity lift did not happen; the purity
ceiling is robust to the model choice.

- **V1 (2lpt loa124 `_m`)** is the best: +0.4pp P / +2.2pp C over the
  β-collapsed baseline V0. The purity gain is within the ~0.6pp noise
  floor; the +2.2pp completeness is real. A mild, genuine Pareto win.
- **V4 (real-LOA, with BAL in training)** ties V1 on purity (0.805) but
  has baseline-level completeness (0.842) — the real-LOA `with-bal`
  model does not help completeness here.
- **V2 (2lpt loa0)** is the worst on purity (0.783) — the loa0 training
  subset underperforms loa124.
- **V3 (real-LOA no-dla)** is middling (0.791 / 0.858).

So the β-collapse, while a real defect, was **not** the purity-frontier
limiter. The ~4–5pp purity shortfall vs the 85% target survives the
model swap — consistent with the `p_DLA`-cut sweep
(`2026-05-17_pdla_cut_sweep.md`) and the lambda/PW sweeps. The remaining
gap is structural (NHI bias / inference), not a tuning or model-file
problem.

## Recommendation

**Adopt V1 — `2lpt_loa124_nohcd_nobal_wide_m`
(`phase2_desi/2lpt_loa124_nohcd_nobal_wide_m/phase2_result.h5`) — as the
production GP model.** Two reasons: (1) it is the best-measured model
here (mild Pareto win); (2) the current baseline is the *deprecated
β-collapsed* model and must not ship regardless. Expected P/C at the
current best config: **0.804 / 0.864** (London-0 5k).

**Runtime** (measured, V1 cell): 6766 spectra in 50.0 min on 64 cores =
0.208 node-hours. → **≈ 31 node-hours per 1M QSOs** at PW 50k. See the
runbook §"Recommended production configuration" for the full estimate.

85/85 remains unreached. The next lever is **not** another model file —
it is the NHI-bias / `NHI_ERR` recalibration work
(`2026-05-17_nhi_flag_investigation.md`) or an inference-side change.
