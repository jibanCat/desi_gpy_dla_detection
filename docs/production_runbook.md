# GP-DLA Production Runbook — NERSC Perlmutter

> **Audience**: next-Claude in this repo. You are launching full GP-DLA inference
> over a dataset (mock or real LOA). This document tells you the exact paths,
> commands, hyperparameters, expected wall-time, and expected P/C numbers.
>
> **Written**: 2026-05-13, branch `production_533`, after the Var[Δ_marg] gating
> diagnostic concluded production N=50k QMC is sampling-converged
> (see `docs/notes/2026-05-13_var_delta_marg_diagnostic.md`).
>
> **DO NOT submit new jobs without reading §10 (Gotchas) first** — the production
> sbatch scripts `slurm/submit_desi_{mock,loa}.sh` **do not forward**
> `--enable_tau_eb` / `--tau_eb_objective` / `--early_stop_mode` to the python
> CLI. You will silently lose τ-EB in production unless you patch them or use
> `slurm/run_local.sh` (which does forward them). See §10.1.

---

## Current production decisions (2026-05-17)

> This section consolidates the 2026-05-14 → 2026-05-17 work (the
> `dla_gp.py` +log(N) bias fix and the cellC/cellD/lambda/min_z/nhi/model
> knob sweeps) into one authoritative table. **It supersedes the
> "v3 high-purity stack" framing in the §"KNOWN REGRESSION" callout and
> §0 below** — the +log(N) patch rebalanced P/C and the
> regression framing predates it. Where this section and an older one
> disagree, this section wins. Per-knob evidence is in
> `docs/notes/2026-05-1{4,5,6}_*.md`.

### The recommended production configuration

**Family**: 2-way **single-absorber** model over NHI [17.2, 22.5] — the
cellC family. Chosen for the CDDF LLS use case: the single catalog is
directly usable for LLS / sub-DLA / DLA after post-hoc NHI cuts, with no
lossy 3-way channel split (3-way loses completeness — see
`2026-05-16_subdla_3way_sweep.md` and memory `project_subdla_dla_joint_design`).

| Knob | Production value | Status | Evidence |
|---|---|---|---|
| `LEARNED_FILE` (GP model) | **`2lpt_loa124_nohcd_nobal_wide_m`** (`phase2_desi/.../phase2_result.h5`) | **firm** | `2026-05-18_model_sweep.md` — V1 wins (0.804/0.864); MUST replace the deprecated β-collapsed baseline |
| `SINGLE_ABSORBER_MODEL` | **1** (2-way) | firm | `2026-05-16_config_confirmations.md` — single-absorber ≫ multi-DLA mode |
| `MAX_DLAS` | **3** | firm | London-0 truth: only **0.05%** of NHI≥20.3 QSOs have >3 classical DLAs (50 of ~101k); cellC C1/C2 (MAX_DLAS 4/5) show no P/C gain. *(Caveat: ~6% of QSOs have >3 absorbers counting sub-DLAs — see note below.)* |
| `MAX_LAMBDA` | **1250** | firm (London-strong, Saclay-mild, 2LPT-neutral) | `2026-05-16_lambda_fine_and_gp_range.md` §4 — cross-mock validated; safe everywhere |
| `MIN_LAMBDA` | **911.75** | firm | gp_range — blue-side moves inert-to-bad |
| `MIN_Z_SEPARATION` | **3000 km/s** | firm | `2026-05-15_min_z_separation_smoke.md` — inert; confirmed NO-CHANGE at 50k (M0–M3 spread ≤0.7pp) |
| `FILTER_LOW_LIKELIHOOD` | **1** | firm | cellC family runs FILTER=1 |
| +log(N) evidence patch | **ON** (in `dla_gp.py`) | firm, merged | `2026-05-14_log_evidence_bias_fix.md`; A/B `2026-05-16_logn_patch_ab.md` |
| `ENABLE_TAU_EB` / objective | **1** / `null` | firm | PR #5 |
| `EARLY_STOP_MODE` | **baseline** | firm | variants A/D not promoted |
| QMC `NUM_DLA_SAMPLES` | **100k** | **revised 2026-05-18** | refreshed cellC: C7 (100k) is **+1.1pp P / +1.9pp C** over C0 (50k) — above the ~0.6pp noise floor, *not* "within noise" as an earlier draft said. Worth the ~1.5–1.8× QMC cost for the 85/85 push. (V1 was measured at 50k; re-measure at 100k.) |
| NHI prior | **[17.2, 22.5]** (`pw_samples_a3_172_225_100000.mat`) | firm | extends the ceiling so rare NHI>22 DLAs are not clipped low. `2026-05-15_nhi_prior_extension.md` showed a ~1pp 5k P/C "cost" but that is within the noise floor — the modelling-correctness argument (no hard clip) governs. |
| BAL | included at inference, excluded at eval | firm | no `--balmask` |
| `p_DLA` catalog cut | **0.99** | convention (not optimized) | `2026-05-17_pdla_cut_sweep.md` gives the full P/C-vs-cut frontier; no cut reaches 85/85, so the cut is a free P↔C slide. 0.99 is the historical, completeness-rich end — keep it unless a purity-priority subset is wanted. |

### Recommended production configuration ("best baseline") + runtime

The full config to launch the 1M-QSO production run with:

```
LEARNED_FILE      = .../phase2_desi/2lpt_loa124_nohcd_nobal_wide_m/phase2_result.h5
SINGLE_ABSORBER_MODEL = 1      MAX_DLAS = 3      FILTER_LOW_LIKELIHOOD = 1
MAX_LAMBDA = 1250   MIN_LAMBDA = 911.75   MIN_Z_SEPARATION = 3000
NUM_DLA_SAMPLES = 100000   DLA_SAMPLES_FILE = pw_samples_a3_172_225_100000.mat   (NHI [17.2,22.5])
ENABLE_TAU_EB = 1   TAU_EB_OBJECTIVE = null   EARLY_STOP_MODE = baseline
+log(N) patch ON (in dla_gp.py)   p_DLA cut = 0.99 at catalog time
```

**Expected P/C**: the closest measured cell is `model_sweep` V1 at
**PW 50k / NHI [17.2,22] = 0.804 / 0.864** (London-0 5k). The production
config above moves to **PW 100k** (cellC C7-vs-C0 ⇒ ~+1.1pp P / +1.9pp C)
and the **[17.2,22.5]** prior — so expect roughly **~0.81 / ~0.88**, but
this exact combination has **not been measured as one cell**. A single
validation run of the final config is recommended before the 1M launch.
Either way it is **below the 85/85 target on purity** — see "Open items";
no tested knob/model/cut closes that gap.

**Runtime** — measured from `model_sweep` V1: **6766 spectra in 50.0 min
on 64 cores** = 0.208 node-hours at **PW 50k**. PW 100k (the production
choice) scales the QMC integral ~1.5–1.8× → ~0.34 nh per 6.8k.

| dataset | QSOs | node-hours @PW50k | node-hours @PW100k |
|---|---|---|---|
| 5k-slice validation | ~6.8k | 0.21 | ~0.34 |
| one full mock (London/Saclay/2LPT) | ~1.2M | ~37 | **~60** |
| **real DESI Y3 LOA (headline run)** | **~1M** | ~31 | **~50** |
| **full suite** (3 mocks + real LOA) | — | ~140 | **~230** |

At the production PW 100k: ≈ **50 node-hours per 1M QSOs** (~13 h on
4 nodes, ~3 h on 16). τ-EB `null` (the cheap objective) is included.
See §4 for the older per-dataset breakdown (now superseded by
this measured figure).

### Note on MAX_DLAS = 3

Checked against London-0 truth (`dla_cat.fits`, 2026-05-18):

- **Classical DLAs (NHI ≥ 20.3)**: of ~101k QSOs with ≥1 classical DLA,
  only **50 (0.05%)** have more than 3. For the headline NHI≥20.3 catalog,
  MAX_DLAS=3 misses essentially nothing.
- **All absorbers (any NHI, incl. sub-DLAs/LLS)**: of 461k DLA-bearing
  QSOs, **6.1%** have more than 3 absorbers. The 2-way single-absorber
  model searches the full [17.2, 22] range, so on a multi-absorber
  sightline MAX_DLAS=3 caps the recursion before all absorbers are found.

Empirically this does not cost headline P/C: the cellC sweep cells C1
(MAX_DLAS=4) and C2 (MAX_DLAS=5) show **no Pareto improvement** over
C0 (MAX_DLAS=3) — slightly lower purity, equal completeness. So MAX_DLAS=3
is kept. The 6% multi-absorber population is a candidate explanation for
some missed DLAs — see the FN/FP deep-dive (`fn_fp_deepdive/FINDINGS.md`).

### ⚠ Do NOT ship the β-collapsed sweep baseline model — use V1

`null_gp_test/converted/2lpt_loa124_nohcd_nobal_wide.h5`, used as the
baseline in every 5k sweep, is the **deprecated β=1.45 model** (Turner+2024
prior is 3.62) and forces the Garnett norm band [1310,1325]. **The
production model is the `model_sweep` winner — V1,
`phase2_desi/2lpt_loa124_nohcd_nobal_wide_m/phase2_result.h5`** (healthy
β, MATLAB norm band). Note `2026-05-18_model_sweep.md`: the swap is only a
*mild* improvement (+0.4pp P / +2.2pp C) — the β-collapse was not the
purity-frontier limiter — but the deprecated model must not ship
regardless. See memory `project_baseline_model_beta_collapse`.

### Eval recipe (the "fixed molly recipe", 2026-05-15)

P/C is measured with `examples/molly_faithful_pc_plots.py`: SNR_RED > 2,
`p_DLA ≥ 0.99`, lyb-veto on, **drop ALL `bal_cat` TIDs** (`--no-bal`),
λ_rf ∈ [911, 1216] Å, NHI ≥ 20.3 truth + predicted, external
`--snr-cat`/`--zcat`, `--restrict-truth-to-processed`. This recipe
(post-`2026-05-15_molly_eval_recipe_fix.md`) gives **n_truth = 581** on a
London-0 5k slice. Pre-fix sweeps (cellC/cellD, n_truth = 618) are **not**
P/C-comparable to post-fix sweeps.

### Expected P/C (2-way, post-patch, London-0 5k, default p_DLA ≥ 0.99)

Authoritative numbers, **refreshed 2026-05-17 under the new DLAFLAG
convention** (`NHI_INCONSISTENT` no longer gated — commit `2ae3435`;
all 12 sweeps re-postprocessed + re-evaluated, sbatch `53087827`). Fixed
molly recipe, n_truth=581, β-collapsed baseline GP model.

| Config | Purity | Completeness | source |
|---|---:|---:|---|
| **production config — V1 model + MAX_LAMBDA=1250 (PW 50k)** | **0.804** | **0.864** | `model_sweep/HEADLINE.tsv` (V1) |
| same config, β-collapsed baseline model (for reference) | 0.800 | 0.842 | `model_sweep/HEADLINE.tsv` (V0) |
| cellC C0 (β-collapsed, MAX_LAMBDA=1216.75, PW 50k) | 0.780 | 0.836 | `cellC_knob_sweep/HEADLINE.tsv` |
| cellC C7 (β-collapsed, PW 100k) | 0.791 | 0.855 | `cellC_knob_sweep/HEADLINE.tsv` |

The headline production number is **V1 = 0.804 / 0.864** (London-0 5k).
Saclay/2LPT land within ~1–2pp. This is **~4.6pp short of the 85% purity
target** — no tested model, knob, or p_DLA cut closes that gap (see Open
items).

> **History (for anyone reading older docs):** earlier drafts quoted
> NHI-gated numbers (e.g. C0 = 0.814/0.799) or the obsolete HANDOFF
> "0.779/0.877" — both superseded. Current numbers are un-NHI-gated
> (NHI_INCONSISTENT informational-only) and all sweeps are gated
> identically (LYBETA/BAL only), so cross-sweep P/C is directly
> comparable.

### Open items before the 1M production launch

**Resolved 2026-05-17/18** (all the knob sweeps have landed):
- ✅ `lambda1250_crossval` + `lambdamax_crossmock` — MAX_LAMBDA=1250 is
  London-strong / Saclay-mild / 2LPT-neutral; kept (safe everywhere).
- ✅ `min_z_separation_sweep_50k` — MIN_Z_SEPARATION inert at 50k; keep 3000.
- ✅ `model_sweep` — production model = V1 `2lpt_loa124_nohcd_nobal_wide_m`.
- ✅ QMC sample count — 50k (100k is ~1pp, not worth ~2× cost).
- ✅ DLAFLAG-gating consistency (sbatch `53087827`) — all sweeps
  re-evaluated under one convention; tables/scatter paper-citable.

**Still open / the real blocker:**

1. **The 85/85 target is not met and is not a tuning problem.** The
   production config tops out at **~0.80 P / 0.86 C** — ~4.6pp short on
   purity. The p_DLA-cut sweep (`2026-05-17_pdla_cut_sweep.md`), the
   model sweep (`2026-05-18_model_sweep.md`), and the lambda/PW/min_z
   sweeps **all** confirm the purity ceiling is robust to knobs, model
   file, and cut. Closing the gap needs **structural work** — the
   NHI-bias / `NHI_ERR` recalibration (`NHI_pred` biased +0.06 dex,
   `NHI_ERR` under-estimated ~1.4×; `2026-05-17_nhi_flag_investigation.md`)
   or an inference-side change. This is the headline pre-launch decision:
   **launch the 1M run at ~0.80/0.86 now, or hold for the purity work?**
2. `p_DLA` cut convention — 0.99 is the completeness-rich default; any
   tightening just trades down the same frontier (no 85/85 point exists).
   Pick 0.99 unless a purity-priority subset catalog is wanted.
3. NHI prior [17.2, 22.5] extension — provisional (marginal ~1pp cost at
   5k); decide with a larger-scale E0/E1 re-check, or default to [17.2,22.0].

### DLAFLAG / NHI_INCONSISTENT — decided 2026-05-17

The `NHI_INCONSISTENT` gate was investigated (`docs/notes/2026-05-17_nhi_flag_investigation.md`,
sbatch job 53078990). Result: the flag *is* FP-enriched (~3× the FP-rate of
kept rows) but **not a clean filter** — 41% of flagged rows are real DLAs and
every k>0 trades completeness for purity. **Decision: NHI gate OFF (k=0)** for
the production headline — it is a blunt P↔C knob, and purity toward the
85/85 target has a cleaner dedicated lever (the p_DLA cut); keep
`NHI_CONSISTENCY_FLAG` as an informational column only. Separately, the
investigation found `NHI_pred` biased **+0.06 dex high** and `NHI_ERR`
**under-estimated ~1.4×** — that is the scope of the deferred NHI-bias task,
to revisit after the GP-model swap.

**Implemented 2026-05-17** (commit `2ae3435`): `NHI_INCONSISTENT` is no
longer folded into `DLAFLAG` — it is now an informational column only
(`NHI_CONSISTENCY_FLAG`), like `PDLA_SATURATED_FLAG`. `DLAFLAG == 0` now
means LYBETA/BAL/bad-fit clean and is no longer swamped by the NHI knob,
so the molly headline P/C is no longer silently NHI-gated. Catalogs
stamped before this change need a re-postprocess (`add_dla_flags.py`,
cheap, no inference) to pick up the new schema — folded into the
open-item-6 consistency re-eval.

---

> **⚠ KNOWN REGRESSION — read before adopting the v3 stack as "baseline":**
> *(2026-05-17: superseded — see "Current production decisions" above. The
> +log(N) patch rebalanced P/C; this callout predates it and is kept only
> for the FILTER=1 [20.3,20.6) root-cause history.)*
>
> The "high-purity stack" documented in §0 (v3 GP + PW14 50k + τ-EB, FILTER=1)
> gives **~+9 pp purity** versus the historical baseline (`model_epoch_920` +
> 10k samples + FILTER=1, P_DLA ≥ 0.99 / SNR_RED > 2) but at the cost of
> **~−5 pp completeness** in the weakest [20.3, 20.6) NHI bin.
>
> Numbers, 26k London-0, P_DLA ≥ 0.99, SNR_RED > 2, BAL-excluded, λ_rf ∈ [911, 1216] Å:
> | Stack | Purity | Completeness |
> |---|---:|---:|
> | Historical FILTER=1 baseline (`model_epoch_920` + 10k QMC + multi-DLA) | **0.7504** | **0.8206** |
> | v3 high-purity (v3 GP + PW14 50k + τ-EB + FILTER=1) | **0.8452** | **0.7661** |
> | Δ | **+9 pp** | **−5 pp** |
>
> Source: `prod533_5k_20260511/MOLLY_TABLES_SNR_CUTS.md` (26k baseline row, SNR>2 P_DLA≥0.99).
> The 78%/80% reference the user has in memory is the same regime within
> rounding (FILTER=1 + multi-DLA + 10k; eBOSS-style historical comparison).
>
> The user's headline requirement is **≥ 85 % completeness**, so the v3 stack
> is a regression on the load-bearing metric. Root cause is documented in
> `docs/notes/2026-04-27_filter_completeness_explanation.md` — `FILTER=1`'s
> coarse-then-refine logic drops ~9.5 pp of the [20.3, 20.6) bin because the
> coarse 5k-sample null-evidence cut kills marginal-Δ_marg DLAs before the
> refine pass can rescue them. The user's *additional* memory cites a
> FILTER=0 historical baseline (`3-way [null, subDLA[19.5,20], DLA[20,23]]`
> + 10k + FILTER=0 + multi-DLA + early-stop) that recovers 87.8 % completeness
> on a 200-target dense-DLA sample. The fix is GP-side (adaptive null cut by
> NHI bin, stratified QMC, multi-pass refinement), **not** catalog-time cut
> tuning.
>
> Until that fix lands, production runs should **either**:
>   - **(a) Match the historical baseline**: keep `FILTER_LOW_LIKELIHOOD=0`
>     (per the user's 78/80 reference config) with `model_epoch_920` + 10k
>     samples + multi-DLA + early-stop. This is documented to give
>     >80 % completeness and the historical purity in the high 70s.
>   - **(b) Run the v3 stack as documented below**, accepting that it is the
>     **highest-purity** config tested but **not** the recommended production
>     baseline until the FILTER=1 completeness fix is implemented and
>     validated.
>
> §0 documents option (b). It is **not** yet promoted to "production baseline".
> The decision on whether to make it the recommended baseline is **blocked**
> on landing the FILTER=1 completeness fix (or equivalently, on demonstrating
> that an alternative `FILTER_LOW_LIKELIHOOD=0` run of the v3 stack recovers
> the historical >80 % completeness without giving back the +9 pp purity).

---

## 0. HIGH-PURITY STACK (current best-known purity at the cost of completeness)

This is the *highest-purity v3 stack tested* (2026-05-13). **It is NOT YET the
recommended production baseline** — see the "KNOWN REGRESSION" callout above.
The v3 stack lifts purity by +9 pp vs the historical FILTER=1 multi-DLA
reference (26k London-0 at SNR_RED > 2, P_DLA ≥ 0.99) but drops completeness
by ~5 pp in the [20.3, 20.6) NHI bin. Validated on London mock-0 (8 spectra-16
files, ~6.6k QSOs) and Saclay mock-0 (8 files, ~6.7k QSOs). Cross-validated by
`docs/runs/2026-05-12_saclay_v3_loa124_results.md`.

```bash
# v3 GP model (phase2_desi, 2LPT loa124 training filter, HCD- and BAL-excluded)
# Trained with num_forest_lines=3 (see /pscratch/.../v2_runs/2lpt_loa124_nohcd_nobal_52188321/config.json).
# TODO(verify before sbatch launch): NUM_FOREST_LINES must match training. Inference defaults from
# slurm/configs/_base.env=3 (matches v3 training); but slurm/submit_desi_{mock,loa}.sh default to 31.
# See §10.9 — explicitly export NUM_FOREST_LINES=3 OR patch the sbatch scripts.
export LEARNED_FILE="/pscratch/sd/j/jibancat/prod533_5k_20260511/null_gp_test/converted/2lpt_loa124_nohcd_nobal_wide.h5"
export NUM_FOREST_LINES=3                 # MUST match training (see TODO above)

# QMC sample grid: PW14 prior over NHI ∈ [19, 22], 50k samples (file rowcount MUST match)
# This is the v3 baseline file (`pw_samples_a3_190_220_50000.mat` → NHI ∈ [19, 22]).
# Do NOT confuse with `pw_samples_a3_190_230_50000.mat` (NHI ∈ [19, 23], 50k) which is used
# ONLY in the joint sub-DLA+DLA sweep (`joint_dla_subdla_sweep/cell{A,B}`) — NOT in the v3
# baseline. The [19, 22] PW14 file matches the actual `RUN_SETTINGS.md` of both v3 runs.
export DLA_SAMPLES_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_190_220_50000.mat"
export NUM_DLA_SAMPLES=50000

# Sub-DLA: keep production 10k default (the 100k file with NHI floor 19.1 is
# recommended for headline catalogs; the 10k default is what's been validated
# at scale).
# To switch: export SUB_DLA_SAMPLES_FILE=".../subdla_samples_a03_191_200_100000.mat"; export NUM_SUBDLA_SAMPLES=100000

# Per-spectrum empirical-Bayes τ_eff (PR #5) — ON
export ENABLE_TAU_EB=1
export TAU_EB_OBJECTIVE=null              # cheap; matched by "dla" objective on canonical targets
# (default tau_eb_factors = 0.5 1.0 1.5 2.0 3.0 4.0 5.0 6.0 — keep)

# Multi-DLA early-stop policy: default "baseline" matches the validated production
# configuration. Variants A/D are under evaluation (see early_stop_fix_test/)
# but have NOT been promoted to production. KEEP "baseline".
export EARLY_STOP_MODE=baseline

# DLA-mode (multi-DLA catalog, NHI ≥ 20.3)
export MAX_DLAS=3
export SINGLE_ABSORBER_MODEL=0
# FILTER_LOW_LIKELIHOOD: the v3 high-purity stack runs at 1. The historical
# multi-DLA reference that delivered 0.7504 P / 0.8206 C at SNR>2 also used
# FILTER=1; the user's separate FILTER=0 historical reference (3-way [null,
# subDLA[19.5,20], DLA[20,23]] + 10k + early-stop, per project memory) gives
# higher completeness on the [20.3, 20.6) bin. See the KNOWN REGRESSION
# callout above. The FILTER=1 [20.3, 20.6) bin drop is a GP-side issue — fix
# is queued at the GP level, not the catalog-time cut.
export FILTER_LOW_LIKELIHOOD=1

# BAL: included in inference (no --balmask). BAL exclusion is applied at eval time.
export BALMASK=false
```

**P/C at SNR_RED > 2, P_DLA ≥ 0.99, full forest λ_rf ∈ [911, 1216] Å,
BAL excluded** (`docs/runs/2026-05-12_saclay_v3_loa124_results.md`):

| Mock | Purity | Completeness | n_cat | n_truth | vs historical baseline |
|---|---:|---:|---:|---:|---|
| London 8f (v3 stack) | **0.8452** | **0.7661** | 1242 | 618 | **+9 pp P / −5.5 pp C** |
| Saclay 8f (v3 stack) | **0.8707** | **0.7710** | 1381 | 533 | (no direct historical reference run on Saclay 8f) |

**Historical reference** (`model_epoch_920` + 10k samples + FILTER=1, 26k London,
P_DLA ≥ 0.99, SNR_RED > 2 (full forest, BAL excluded);
see `docs/notes/2026-04-27_london_pdla_scan_no_bal.md` Table for SNR-free P_DLA scan
and `MOLLY_TABLES_SNR_CUTS.md` for SNR-cut breakdown):
**P = 0.7504 / C = 0.8206**. The v3 stack trades that completeness for purity in
the [20.3, 20.6) NHI bin; see the KNOWN REGRESSION callout at the top.

The user's stated reference of "~78 % / >80 %" matches this regime once SNR_RED > 2
is applied; the GP-level fix (not catalog cuts) is the right place to recover the
−5 pp completeness loss.

At stricter cuts:

| Cut | London P / C | Saclay P / C |
|---|---|---|
| ≥ 0.99    | 0.8452 / 0.7661 | 0.8707 / 0.7710 |
| ≥ 0.999   | 0.8547 / 0.7398 | 0.8845 / 0.7475 |
| ≥ 0.99999 | 0.8872 / 0.6901 | 0.9013 / 0.6768 |

**Headline: at SNR_RED > 2 nothing currently passes 85/85 jointly.** The
completeness gap is intrinsic at low SNR (model-side limiter — sampling is
already converged; see `docs/notes/2026-05-13_var_delta_marg_diagnostic.md`).
The FILTER=1 mechanism is the dominant additional loss beyond the structural floor.

Per `memory/feedback_snr_canonical.md`: report **SNR_RED > 2** as headline,
acceptable > 1. **Do NOT** report SNR > 6 as primary.

---

## 1. Datasets

### 1.1 Per-dataset spectrum counts

Counted 2026-05-13 from the zcat.fits and the spectra-16 directory tree.
Production cut applied: Z ≥ 1.96 (see `desi-DLAGP.py:read_in_each_plots_*`).

| Dataset      | zcat path                                                                                                  | rows (zcat) | Z ≥ 1.96   | spectra-16 files |
|--------------|------------------------------------------------------------------------------------------------------------|------------:|-----------:|-----------------:|
| **2LPT-0**   | `/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/zcat.fits` | 1 213 217 | 977 268 | 1 150 |
| **London-0** | `/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/zcat.fits`         | 1 217 878 | 982 313 | 1 150 |
| **London-1** | `/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-1/jura-124/zcat.fits`         | 1 218 470 | 982 672 | 1 148 |
| **Saclay-0** | `/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124/zcat.fits` | 1 221 478 | 985 282 | 1 127 |
| **Saclay-1** | `/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/saclay/qq_desi_y3/v4.7.5/mock-1/jura-124/zcat.fits`    | 1 221 416 | 984 074 | 1 130 |
| **LOA real** | `/global/cfs/cdirs/desi/science/lya/y3/loa/catalogs/QSO_cat_loa_main_dark_healpix_v2-altbal-20241115.fits`  | 2 776 520 | 1 000 854 | 16 586 unique HPXPIXEL |

**Additional cuts applied at inference time** (in `desi-DLAGP.py` / `dlasearch.py`):
- BAL inclusion (no `--balmask`) — but `BI_CIV > 0` BALs are excluded at *eval time*
- Per-spectrum pixel masking: `MAX_NOISE_VARIANCE=9`, `NUM_FOREST_LINES=3`
- `MIN_LAMBDA / MAX_LAMBDA` rest-frame window 911.75 / 1216.75 Å — spectra with
  zero in-window pixels are skipped
- SNR_RED > 1 (acceptable) / > 2 (headline) cut applied at *catalog cuts*, not
  inference time. All spectra are inferred; downstream P/C scripts apply the cut.

### 1.2 Saclay mock-0 vs mock-1 subdir naming gotcha

Saclay mock-0 lives in `juraLy8-124/`; Saclay mock-1 lives in `jura-124/`.
Note: `Ly8` vs no `Ly8`. Both config files already encode this correctly.

### 1.3 Truth catalogs (for P/C eval)

| Dataset | Truth file |
|---|---|
| London-0 / 1 | `<mockdir>/dla_cat.fits` |
| Saclay-0 / 1 | `<mockdir>/hcd_truth_cat.fits` |
| 2LPT-0 | `<mockdir>/dla_cat.fits` (verify) |
| LOA real | no truth — P/C not applicable; use eBOSS overlap or DR9/DR12 concordance for sanity-check |

---

## 2. Two run modes per dataset

### 2.1 Multi-DLA mode (the headline DLA catalog, NHI ≥ 20.3)

This is the **3-way** model: `[null, sub-DLA, k-DLA]` model-selection. Sub-DLA exists
here as a *purity guard* — sub-DLAs masquerading as DLAs near the 20.3 boundary
get absorbed by the sub-DLA hypothesis rather than entering the DLA catalog.
This 3-way is the **wrong** mode for measuring sub-DLA P/C — see §2.3 for the
2-way sub-DLA detection mode.

| Knob                    | Value                       |
|-------------------------|-----------------------------|
| `MAX_DLAS`              | 3                           |
| `SINGLE_ABSORBER_MODEL` | 0                           |
| `FILTER_LOW_LIKELIHOOD` | 1                           |
| `DLA_SAMPLES_FILE`      | `pw_samples_a3_190_220_50000.mat` (PW14, **NHI ∈ [19, 22]**, 50k) |
| `NUM_DLA_SAMPLES`       | 50000                       |
| `SUB_DLA_SAMPLES_FILE`  | `subdla_samples.mat` (10k, NHI ∈ [19.5, 20]) — production-validated |
| `NUM_SUBDLA_SAMPLES`    | 10000                       |

**Verified from `prod533_5k_20260511/{london,saclay0}_v3_loa124_pw14_tau_eb/RUN_SETTINGS.md`:**
the v3 baseline uses `pw_samples_a3_190_220_50000.mat` (NHI ∈ [19, 22]).
The separate `pw_samples_a3_190_230_50000.mat` (NHI ∈ [19, 23], 50k) file exists
on disk but is only used in `joint_dla_subdla_sweep/cell{A,B}` (an in-flight
sub-DLA+DLA joint exploration with `SINGLE_ABSORBER_MODEL=1`, NOT the multi-DLA
baseline). See §2.3 for that distinction.

Configs (use these as-is — they `source _base.env`):
| Dataset    | Config file                                          | Outer loop |
|------------|------------------------------------------------------|-----------:|
| 2LPT-0     | `slurm/configs/2lpt0_y3.env`                         | 0..1150 step 64 |
| London-0   | `slurm/configs/london0_y3.env`                       | 0..1150 step 64 |
| London-1   | (none yet — copy london0 and swap `mock-0` → `mock-1`, update `OUTER_MAX_INDEX=1148`) | 0..1148 step 64 |
| Saclay-0   | `slurm/configs/saclay0_y3.env`                       | 0..1127 step 64 |
| Saclay-1   | (none yet — copy saclay0 and swap path; OUTER=1130)  | 0..1130 step 64 |
| LOA real   | `slurm/configs/loa_y3.env`                           | 0..16519 step 1664 |

### 2.2 LLS single-absorber mode (sub-DLA + DLA, NHI ∈ [17.2, 22] or [19, 22])

| Knob                    | LLS NHI≥17.2                | LLS NHI≥19.0                |
|-------------------------|-----------------------------|-----------------------------|
| `MAX_DLAS`              | 1                           | 1                           |
| `SINGLE_ABSORBER_MODEL` | 1                           | 1                           |
| `FILTER_LOW_LIKELIHOOD` | 0                           | 0                           |
| `DLA_SAMPLES_FILE`      | `pw_samples_a3_172_220_50000.mat` | `pw_samples_a3_190_220_50000.mat` |
| `NUM_DLA_SAMPLES`       | 50000                       | 50000                       |
| `SUB_DLA_SAMPLES_FILE`  | `subdla_samples_a03_191_200_100000.mat` (100k, NHI ∈ [19.1, 20]) | same |
| `NUM_SUBDLA_SAMPLES`    | 100000                      | 100000                      |
| `BATCH_SIZE`            | 6250                        | 6250                        |

Configs:
| Dataset  | NHI 17.2 config                            | NHI 19.0 config                            |
|----------|--------------------------------------------|--------------------------------------------|
| 2LPT-0   | `slurm/configs/2lpt0_y3_lls172.env`        | `slurm/configs/2lpt0_y3_lls190.env`        |
| London-0 | `slurm/configs/london0_y3_lls172.env`      | `slurm/configs/london0_y3_lls190.env`      |
| Saclay-0 | `slurm/configs/saclay0_y3_lls172.env`      | `slurm/configs/saclay0_y3_lls190.env`      |
| LOA real | `slurm/configs/loa_y3_lls172.env`          | `slurm/configs/loa_y3_lls190.env`          |

LLS configs are **NOT** updated to v3 model + τ-EB. To deploy the high-purity
v3 stack in LLS mode, override via env vars before launching (see §6.3).

### 2.3 Sub-DLA selection modes — the 2-way vs 3-way distinction

**User-explicit policy** (from project memory):
> *"For sub-DLA detection you should model selection in two-way `[null, sub-DLA]`,
> not three-way `[null, sub-DLA, DLA]`. Three-way was used to treat sub-DLA as
> null to improve purity of DLA."*

Three modes coexist in this repo; choose the one matching the science goal.

| Mode | Model selection | `SINGLE_ABSORBER_MODEL` | DLA prior | Purpose |
|---|---|---|---|---|
| **2-way (sub-DLA P/C)** — recommended for sub-DLA detection | `[null, sub-DLA]` | 1 | `pw_samples_a3_172_220_50000.mat` or `subdla_samples_a03_191_200_100000.mat` | Sub-DLA catalog. The "absorber" hypothesis IS the sub-DLA. This is the LLS-mode subset; for sub-DLA-only set `--nhi-min 19.0 --nhi-max 20.3` at eval. |
| **3-way (DLA-purity guard)** | `[null, sub-DLA[19.5, 20], k-DLA[20, 23]]` | 0 | `pw_samples_a3_190_220_50000.mat` + 10k sub-DLA grid | DLA catalog (§2.1). Sub-DLA model exists *only* to absorb near-boundary impostors and improve DLA purity. **Not** valid for sub-DLA P/C — `model_posteriors[:,1]` (sub-DLA column) is biased low because the DLA model competes for the same evidence. |
| **2-way (joint sub-DLA+DLA, extended prior)** | `[null, k-absorber[19,23]]` | 1 | `pw_samples_a3_190_230_50000.mat` (50k) | Cell A/B of the in-flight `joint_dla_subdla_sweep/`. The "absorber" model covers NHI ∈ [19, 23] in one prior, with NHI cuts applied **post-hoc** to split into sub-DLA / DLA catalogs. This is Option B in project memory. **No P/C measured yet** — this sweep is the next data point. |

The §2.1 multi-DLA mode is 3-way. The §2.2 LLS modes are 2-way (a single absorber
hypothesis with extended NHI prior). The joint sweep cells (cellA/B/C) are 2-way
with a wider prior. **Cell C** (`pw_samples_a3_172_220_50000.mat`, NHI [17.2, 22])
is closest to the §2.2 LLS-NHI≥17.2 config but with the v3 GP + τ-EB stack.

For sub-DLA P/C (NHI ∈ [19, 20.3]) the correct measurement path is:
1. Run §2.2 LLS-mode (or joint sweep cellC) → 2-way `[null, absorber]` catalog.
2. Cut catalog to NHI ∈ [19, 20.3].
3. Compare to truth (sub-DLA truth files = the full HCD truth catalog filtered
   to NHI ∈ [19, 20.3]).

§7.X (Expected P/C) shows the currently-available sub-DLA P/C numbers under
this scheme.

---

## 3. All hyperparameters that can be tuned

### 3.1 GP model (rarely change)

| Knob | Default (Y3) | Meaning |
|---|---|---|
| `LEARNED_FILE` | `learnlogs/model_epoch_920.h5` (production); `2lpt_loa124_nohcd_nobal_wide.h5` (v3, **recommended**) | Trained GP weights (μ, M, log_omega, log_c_0, log_tau_0, log_beta). Pre-trained, you do not retrain in production. |
| `DLAMBDA` | 0.15 Å | Rest-frame pixel spacing of the GP grid. Must match the trained model. |
| `K` | 30 | GP rank. Must match the trained model. |
| `MIN_LAMBDA` / `MAX_LAMBDA` | 911.75 / 1216.75 Å | Rest-frame forest grid edges. |
| `LOADING_MIN_LAMBDA` / `LOADING_MAX_LAMBDA` | 910 / 1550 Å | Raw-spectrum load window (must contain the normalization band). |
| `NORMALIZATION_MIN_LAMBDA` / `NORMALIZATION_MAX_LAMBDA` | 1425 / 1475 Å | Rest-frame median window for flux normalization. |
| `NUM_FOREST_LINES` | 3 (`_base.env`, `run_local.sh`) / **31** (sbatch defaults `submit_desi_{mock,loa}.sh:51-52`) — **MISMATCH** | Number of forest Lyman lines in the mean-flux multiplier. **The v3 model was trained with 3** (`v2_runs/2lpt_loa124_nohcd_nobal_52188321/config.json`); see §10.9. Always pass `NUM_FOREST_LINES=3` to inference for the v3 GP. The 31 default in the sbatch scripts is a latent bug for v3. |
| `NUM_LINES` | 3 | Number of Lyman lines in the Voigt absorber model. |
| `MAX_NOISE_VARIANCE` | 9 | Pixel-level mask (pixels with noise² > 9 dropped). |

### 3.2 Mean-flux prior (Turner+2024) — production default

| Knob | Value | Meaning |
|---|---|---|
| `PREV_TAU_0` | 0.00246 | τ_0 in τ_eff = τ_0 (1+z)^β. Per-spectrum τ-EB *seeds* from this. |
| `PREV_BETA`  | 3.62  | β     in τ_eff = τ_0 (1+z)^β. |

### 3.3 Mode / sampling knobs

| Knob | Default | Meaning |
|---|---|---|
| `MAX_DLAS` | 3 (multi-DLA), 1 (LLS) | Max absorbers in the multi-DLA recursion. |
| `SINGLE_ABSORBER_MODEL` | 0 (multi-DLA), 1 (LLS) | If 1: only the 1-absorber model is evaluated; no sub-DLA branch. |
| `FILTER_LOW_LIKELIHOOD` | 1 (multi-DLA), 0 (LLS) | If 1: truncated-sampler shortcut over QMC samples. |
| `DLA_SAMPLES_FILE` | `dla_samples_a03.mat` (10k) | QMC sample grid. **`NUM_DLA_SAMPLES` MUST equal the .mat row count.** |
| `NUM_DLA_SAMPLES` | 10000 (default), 50000 (v3 high-purity stack) | Row count of `DLA_SAMPLES_FILE`. Today's Var[Δ_marg] diagnostic shows N=50k is already sampling-converged. |
| `SUB_DLA_SAMPLES_FILE` | `subdla_samples.mat` (10k) | Sub-DLA QMC grid. |
| `NUM_SUBDLA_SAMPLES` | 10000 | Row count of sub-DLA file. |

**Sample file rowcounts (must match `NUM_*_SAMPLES`):**

| File                                                     | rows  | NHI prior        | Where used |
|----------------------------------------------------------|------:|------------------|---|
| `data/dr12q/processed/dla_samples_a03.mat`               | 10000 | (legacy) [19.5, 22], extrap to 23 | Historical baseline (`model_epoch_920`) |
| `data/dr12q/processed/pw_samples_a3_190_220_50000.mat`   | 50000 | **PW14 [19.0, 22.0]** | v3 baseline (§0) and the LLS-NHI≥19 config |
| `data/dr12q/processed/pw_samples_a3_172_220_50000.mat`   | 50000 | PW14 [17.2, 22.0] | LLS-NHI≥17.2 and joint sweep cellC |
| `data/dr12q/processed/pw_samples_a3_190_230_50000.mat`   | 50000 | PW14 [19.0, **23.0**] | Joint sweep cellA/B **only**; NOT the v3 baseline |
| `data/dr12q/processed/subdla_samples.mat`                | 10000 | sub-DLA [19.5, 20], extrap to 23 | Default sub-DLA grid (the v3 baseline uses this) |
| `data/dr12q/processed/subdla_samples_a03_191_200_100000.mat` | 100000 | sub-DLA [19.1, 20], extrap to 22.6 | Recommended for headline sub-DLA catalog; LLS modes use this |

### 3.4 τ-EB (per-spectrum empirical Bayes; PR #5)

| Knob | Default | Meaning |
|---|---|---|
| `ENABLE_TAU_EB`   | 0 (CLI default) — **set 1 for the v3 high-purity stack** | Per-spectrum τ_0 fit. |
| `TAU_EB_FACTORS`  | `(0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)` | Grid factors multiplied by `PREV_TAU_0`. |
| `TAU_EB_APPLY_HCD_MASK` | 0 | HCD masking during τ-fit. At scale the mask over-corrects (see `docs/notes/2026-04-29_tau_eb_n90_unbiasedness.md`). **Keep 0.** |
| `TAU_EB_OBJECTIVE` | `"null"` | `"null"` (cheap, K=8 null-GP rebuild) or `"dla"` (more rigorous, ~5× cost). |

### 3.5 Early-stop policy (new today; commit `2c499a8`)

| Knob | Default | Meaning |
|---|---|---|
| `EARLY_STOP_MODE` | `"baseline"` | `"baseline"`: historical penalized-likelihood-vs-null heuristic. `"A"`: disable null-early-stop entirely. `"D"`: compare pre-Occam likelihood to null. |

**Status**: variants A and D have inference complete on London 8f as of today but
NO production P/C measurement yet. **Keep `baseline` for production until A/D
P/C are published.** Track `prod533_5k_20260511/early_stop_fix_test/RESULTS.md`.

### 3.6 FILTER=1 truncated-sampler knobs (new; commit `2e3642b`)

These two CLI flags expose the internals of the FILTER=1 coarse-then-refine
scheme in `gpy_dla_detection/dla_gp.py`. Defaults reproduce historical
behavior bit-for-bit.

| Knob | Default | Meaning |
|---|---|---|
| `--filter_n_initial_floor` | `5000` | Floor on the coarse-scan budget. Code: `n_initial = max(NUM_DLA_SAMPLES // 20, floor)`. At 50k samples (current production) the floor binds; raising it to 10k doubles coarse-scan coverage. |
| `--filter_empty_mask_fallthrough` | `0` (off) | If `1`: when the coarse scan returns no winners (`valid_mask.sum() == 0`), fall through to FILTER=0 (evaluate all `NUM_DLA_SAMPLES`) instead of early-stopping with the 5000-sample 1-DLA marginal. Bounds FILTER=1 completeness from below by FILTER=0 at the price of full-sample cost on unlucky spectra. |

**2×2 ablation result (London v3 8f, 5k window, evening 2026-05-13)**:

| n_initial floor | empty-mask fall-through | Purity | Completeness | Δ vs baseline |
|---|---|---:|---:|---|
| 5000 (baseline) | OFF | 0.8452 | 0.7661 | reference |
| 10000 | OFF | 0.8516 | 0.7719 | +0.6 pp P, +0.6 pp C |
| 5000 | ON | 0.8506 | 0.7661 | +0.5 pp P, **0 pp C** (knob 4 no-op) |
| 10000 | ON | 0.8511 | 0.7690 | +0.6 pp P, +0.3 pp C |

The 2×2 substantially **refuted** the knob-tuning hypothesis from
`docs/notes/2026-05-13_filter1_knob_tuning.md`: knob 1 alone gives only
+0.6 pp completeness, knob 4 is essentially a no-op. The remaining FILTER=0
vs FILTER=1 gap (~3 pp C at the cost of 2–3× node-hours) must come from
something other than coarse-scan miss / empty-mask early-stop.

**Recommendation**: keep defaults (`--filter_n_initial_floor 5000`,
`--filter_empty_mask_fallthrough 0`) for production. The cellC route
(§2.3, 2-way model) recovers the [20.3, 20.5) regression bin far more
effectively than these knobs and at no extra compute. The knobs remain
useful as a debugging tool: setting `--filter_empty_mask_fallthrough 1`
should converge FILTER=1 to FILTER=0 results in the limit `n_initial → NUM_DLA_SAMPLES`.

Plumbing: flags propagate `desi-DLAGP.py` → `dlasearch.py` → `DLAHolder` →
`BayesModelSelect.model_selection` → `DLAGP.parallel_log_model_evidences`.
`slurm/resume_local.sh` forwards via env vars `FILTER_N_INITIAL_FLOOR` and
`FILTER_EMPTY_MASK_FALLTHROUGH`; `slurm/run_local.sh` forwards extra args
directly to `desi-DLAGP.py`.

### 3.7 BAL handling

| Knob | Default | Meaning |
|---|---|---|
| `BALMASK` | `false` | If `true`: pass `--balmask`, masks BAL absorption pixels. Production has NEVER used BAL masking; BAL exclusion is applied at eval time only. |

### 3.8 Parallelism

| Knob | Default | Meaning |
|---|---|---|
| `MAX_WORKERS` | 8 | Inner-loop ThreadPool workers (per python process), used for QMC sample evaluation. |
| `BATCH_SIZE`  | 1250 (multi-DLA), 6250 (LLS) | QMC sample batch size for memory chunking. |
| `PARALLEL_FILES` (run_local.sh only) | 4 (script default), 32 (production) | Number of python processes per node. |
| `--ntasks` (sbatch) | 32 | Number of srun python procs per node. Each does one spectra-16 (mock) or 52-healpix (LOA) chunk at a time. |
| `--cpus-per-task` | 8 | = `MAX_WORKERS`. |
| Per-node total cores | 32 × 8 = **256** | Perlmutter CPU node = 256 cores. |

**Per `RECOMMENDATIONS.md` §6: PARALLEL_FILES=32 × MAX_WORKERS=8 is already optimal.**
Inner-thread parallelism saturates quickly; to go faster, scale to more nodes.

### 3.8 Outer launcher loop

| Knob | Meaning |
|---|---|
| `OUTER_MAX_INDEX` | Max index of the file/healpix axis (set per dataset). |
| `OUTER_STEP`      | Stride per sbatch job (= number of files per sbatch). 64 for mocks, 1664 for LOA. |
| `OUTER_WINDOW`    | Inner chunk-size *inside* one sbatch (typically `step - 2`). |

---

## 4. Wall-time / node-hour estimates per dataset

> **Re-validated 2026-05-13** — earlier numbers in `RECOMMENDATIONS.md` (~27 nh)
> and in earlier drafts of this runbook used different parallelism assumptions
> and were not consistently derived from the same per-spectrum measurement.
> The numbers below show the math explicitly so future agents can audit.

### 4.1 Source data: prod533 5k London/Saclay v3 runs

Per-spectrum throughput (computed from `grep "Processed spectrum" logs/local_*.log`
in each run dir):

| Source              | n_spec | total wall (s) | mean s/spec | Notes |
|---------------------|------:|---------------:|------------:|---|
| London v3 8f (50k QMC, τ-EB on, FILTER=1) | 6 766 | 12 689 | **1.875** | `prod533_5k_20260511/london_v3_loa124_pw14_tau_eb/logs/` |
| Saclay v3 8f (50k QMC, τ-EB on, FILTER=1) | 9 656 | 19 854 | **2.056** | `prod533_5k_20260511/saclay0_v3_loa124_pw14_tau_eb/logs/` (incl. relaunch slices) |
| London baseline 8f (10k QMC, no τ-EB)     | 6 766 | 11 855 | **1.752** | `prod533test-20260511_1333/london0_y3/logs/` (per `v3_production_cost.md`) |
| Mock LLS-mode (legacy 51638* / 51695* jobs) | 742 506 | 4 603 036 | **6.20** | `/pscratch/.../logs/desi_lls_runs/mock_run_*_51638*.log` |
| LOA real, LLS-mode (legacy 5169587* jobs)  | 4 970 hpx | 36 706 | **7.39** (per-hpx) | `/pscratch/.../logs/desi_lls_runs/debug_loa_run_*_5169587*.log`. Per-hpx wall, not per-spec — needs renormalization for direct comparison. |

Both v3 mock runs match within 10 % (≈ 1.9 s/spec). LLS-mode is ~3× slower because
`SINGLE_ABSORBER_MODEL=1` + `FILTER_LOW_LIKELIHOOD=0` doesn't early-stop on no-DLA
spectra and runs the full 50k-sample QMC bag.

**Early-stop telemetry (London v3 8f, from `grep "Stopping early"`):**

| Depth | count |
|---|---:|
| Stopping early at 1 DLAs | 10 972 |
| Stopping early at 2 DLAs |    678 |
| Stopping early at 3 DLAs |      0 |

Most spectra (~90 %) early-stop in the 1-DLA evaluation step (consistent with the
no-DLA mock population fraction). The 2-DLA early-stop count is the order of the
DLA-containing population that didn't early-stop at 1. Effectively very few
spectra run all the way through max_dlas=3.

### 4.2 How node-hours are computed

```
cpu_seconds_per_spec = wall_per_spec × MAX_WORKERS   (each python uses 8 cores in its QMC ThreadPool)
total_cpu_seconds    = N_spec × cpu_seconds_per_spec
node_hours           = total_cpu_seconds / (256 cores/node × 3600 s/hr)
```

This assumes the production sbatch saturates 32 pythons × 8 workers = **256 cores
per node** (Perlmutter CPU node), and that all pythons stay busy for the full job.

Worked example for London-0 at v3 cost (1.875 s/spec):
```
cpu_s/spec  = 1.875 × 8  = 15.00 cpu-s/spec
total cpu-s = 982 313 × 15.00 = 14.73 M cpu-s
node-hours  = 14.73e6 / (256 × 3600) = 15.99 node-hours
```

### 4.3 Per-dataset estimates (v3 high-purity stack, multi-DLA mode)

| Dataset    | N_spec (Z≥1.96) | wall/spec (s) | cpu-s/spec | total cpu-Ms | **node-hours** | wall on 36-node fan-out |
|------------|-----------------|---------------:|------------:|------:|------:|--------:|
| 2LPT-0     | 977 268         | 2.0 (Saclay-like, no v3 measurement yet) | 16.0 | 15.6 | **17.0** | ~3.7 h |
| London-0   | 982 313         | 1.875 (London v3) | 15.0 | 14.7 | **16.0** | ~3.6 h |
| London-1   | 982 672         | 1.875 (assume = London-0)  | 15.0 | 14.7 | **16.0** | ~3.6 h |
| Saclay-0   | 985 282         | 2.056 (Saclay v3) | 16.4 | 16.2 | **17.6** | ~3.8 h |
| Saclay-1   | 984 074         | 2.056 (assume = Saclay-0) | 16.4 | 16.1 | **17.6** | ~3.8 h |
| LOA real   | 1 000 854       | 2.5 (**estimate** — see caveat below) | 20.0 | 20.0 | **21.7** | ~4.8 h |

**Total all 6 datasets ≈ 105.9 node-hours** for the v3 high-purity stack in
multi-DLA mode. Well under any plausible NERSC budget.

#### LOA-real wall/spec caveat (unvalidated)

The 2.5 s/spec estimate for LOA-real is a **+33 % uplift on the London v3 mean**,
allowing for: (a) variable masking from real-data pixel rejection; (b) higher SNR
distribution potentially activating fewer early-stops; (c) a slightly more
DLA-rich population than mocks. The v3 stack has **not been measured on LOA-real**
in this session. Re-validate from the first LOA-real production sbatch log
(grep "Processed spectrum" for time-spent lines) and revise this row before
committing to a full LOA budget.

### 4.4 Reconciliation with `RECOMMENDATIONS.md` ~27 nh

`RECOMMENDATIONS.md` quoted "~27 node-hours" for the +PW14 50k + τ-EB stack and
"~22 node-hours" for the +PW14 50k base. Those estimates assumed
`PARALLEL_FILES=32` (i.e. 256-core saturation) and used a slightly stale
per-spectrum wall figure (~37 min per spectra-16 file → ~1.5–1.7 s/spec
extrapolated). The current per-spectrum measurement at PARALLEL_FILES=8 is
**1.875 s/spec on London v3** — that's the τ-EB-on number, already including the
~10 % τ-EB overhead.

Multiplying through with PARALLEL_FILES=32 saturation: London-0 production at v3
cost ≈ **16 node-hours**, Saclay at v3 ≈ **17.6**, both rounded to the figures
above. The "~27 nh" claim in `RECOMMENDATIONS.md` was probably padded for safety;
the math agrees once everything is normalized to the same parallelism.

### 4.5 Reconciliation with `docs/notes/2026-04-29_production_cost_estimate.md` (343 nh)

That note estimated **343 node-hours** for 1 M QSO based on a GreatLakes 16-CPU
profile of *two cherry-picked targets* (one strong DLA, one LLS, neither of which
early-stops). Perlmutter at 256-core saturation with the population-mean
early-stop rate (>90 % of mock spectra short-circuit `parallel_log_model_evidences`
at the 1-DLA layer) is **~20× cheaper** in practice. See
`prod533_5k_20260511/v3_production_cost.md` for the side-by-side reconciliation.
The 343-nh estimate is now superseded by the 16-17 nh measurements above for
multi-DLA mock production.

### 4.6 LLS-mode estimates (~3× the multi-DLA cost per spectrum)

LLS mode (`MAX_DLAS=1, SINGLE_ABSORBER_MODEL=1, FILTER_LOW_LIKELIHOOD=0`) doesn't
early-stop and runs the full 50k-sample QMC at every spectrum. Measured:
**6.20 s/spec on mock** (742k spectra across the legacy 51638* job set) and
**7.4 s per-healpix on LOA real** (5169587* debug runs).

| Dataset    | wall/spec | cpu-s/spec | **node-hours @ 1M spec** | wall on 36 nodes |
|------------|----------:|---:|---------------:|---------------------:|
| 2LPT-0     | ~6.2 s    | 49.6 | **~54**        | ~12 h |
| London-0   | ~6.2 s    | 49.6 | **~54**        | ~12 h |
| Saclay-0   | ~6.2 s    | 49.6 | **~54**        | ~12 h |
| LOA real   | ~9 s (estimate)  | 72   | **~78** (unvalidated)        | ~16 h |

**Two LLS variants per dataset (nhi172 + nhi190)** doubles this → budget
**~110 node-hours per dataset** for both, **~55** for one. LOA-real LLS cost
is extrapolated; re-measure from the first sbatch log before committing.

---

## 5. Multi-node parallelism strategy

### 5.1 The math

The outer driver `slurm/launch.sh` submits **one sbatch per `OUTER_STEP` files**.
Each sbatch is 1 node, 32 srun python processes, each handling ~`OUTER_WINDOW / 32` files.

For a mock dataset with 1150 spectra-16 files and `OUTER_STEP=64, OUTER_WINDOW=62`:
- Number of sbatch jobs = ⌈1150 / 64⌉ = **18 sbatch jobs**
- Each sbatch = 1 node × 32 srun python × ~2 files per srun
- Wall per sbatch ≈ (62 / 32) × file_wall ≈ 2 × ~30 min ≈ **~60 min per sbatch**

So **per mock dataset**: 18 sbatch jobs in parallel ≈ 1 hour wall-clock if queue
allows, with total cost ≈ 18 node-hours (within ~10% of the 16-17 node-hour
estimate from §4.3).

For LOA (1 M QSOs, 16519 healpix, `OUTER_STEP=1664, OUTER_WINDOW=1612`):
- Number of sbatch jobs = ⌈16519 / 1664⌉ = **10 sbatch jobs**
- Each sbatch = 1 node × 32 srun × 52 healpix
- Wall per sbatch ≈ (1612 / 32) × ~5 min/healpix ≈ ~4 h
- → **~22 node-hours, ~4 h wall** if all 10 run concurrently

### 5.2 Finer-grain parallelism

Override `--window` and `--end` to split sbatches into smaller chunks:
```bash
# Mock: 36 sbatches of 32 files each instead of 18 of 64
bash slurm/launch.sh slurm/configs/london0_y3.env --window 32 --end 1152
```
Doesn't reduce node-hours but lets more jobs run concurrently if queue is wide.

### 5.3 Queue selection

Perlmutter `regular` queue: max 12 h wall. All per-sbatch estimates above fit
inside 12 h with 2-3× margin. For LOA real, `--time=08:00:00` is already in
`slurm/submit_desi_loa.sh`. For mocks, `--time=05:00:00` in
`slurm/submit_desi_mock.sh`. Both are conservative.

### 5.4 Recommended dataset launch sequence

If you want all 6 datasets done in minimum wall-time and the queue allows it:

```bash
# Multi-DLA mode for all four mocks (60 sbatches total, parallel):
bash slurm/launch.sh slurm/configs/2lpt0_y3.env     --outdir /pscratch/sd/j/jibancat/desi-mock-2lpt0-prod-$(date +%Y%m%d)/
bash slurm/launch.sh slurm/configs/london0_y3.env   --outdir /pscratch/sd/j/jibancat/desi-mock-london0-prod-$(date +%Y%m%d)/
bash slurm/launch.sh slurm/configs/saclay0_y3.env   --outdir /pscratch/sd/j/jibancat/desi-mock-saclay0-prod-$(date +%Y%m%d)/
bash slurm/launch.sh slurm/configs/loa_y3.env       --outdir /pscratch/sd/j/jibancat/desi-loa-prod-$(date +%Y%m%d)/
```

This launches 72 sbatch jobs (18 × 4 mocks + 10 × LOA = 82 actually, but several
will queue). At 10-20 concurrent jobs typical NERSC dispatch, completion in
6-12 hours wall-clock for the full multi-DLA stack.

---

## 6. Exact launch commands

### 6.1 Multi-DLA, mock (v3 high-purity stack)

The `slurm/configs/*.env` files default to the historic baseline (10k samples,
model_epoch_920, no τ-EB). To deploy the v3 high-purity stack — and accept the
~5 pp completeness regression in the [20.3, 20.6) NHI bin per the top callout —
export these overrides before launch:

```bash
# 1. Source DESI env (mandatory for astropy/desispec)
source /usr/share/lmod/lmod/init/bash
export DESI_ROOT=/global/cfs/cdirs/desi
source /global/common/software/desi/desi_environment.sh main

cd /pscratch/sd/j/jibancat/desi_gpy_dla_detection

# 2. v3 high-purity stack overrides (§0 above)
export LEARNED_FILE="/pscratch/sd/j/jibancat/prod533_5k_20260511/null_gp_test/converted/2lpt_loa124_nohcd_nobal_wide.h5"
export NUM_FOREST_LINES=3                                                         # §10.9
export DLA_SAMPLES_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_190_220_50000.mat"  # PW14 NHI ∈ [19, 22]
export NUM_DLA_SAMPLES=50000
export ENABLE_TAU_EB=1
export TAU_EB_OBJECTIVE=null
export EARLY_STOP_MODE=baseline

# 3. Dry-run first (no mkdir, no submit)
bash slurm/launch.sh slurm/configs/london0_y3.env --dry-run --no-sleep | head

# 4. Submit. The launcher refuses any OUTDIR outside the allowed write roots.
bash slurm/launch.sh slurm/configs/london0_y3.env \
    --outdir /pscratch/sd/j/jibancat/desi-mock-london0-prod-$(date +%Y%m%d)-v3/
```

**WARNING — production sbatch scripts don't forward τ-EB / early_stop_mode.**
See §10.1 for details. The exports above will reach the python CLI only if you
either (a) use `slurm/run_local.sh` instead, or (b) patch the sbatch scripts as
described in §10.1. For one-node-at-a-time on a salloc'd compute node:

```bash
salloc -N 1 -C cpu -q interactive -t 04:00:00 -A desi
bash slurm/run_local.sh slurm/configs/london0_y3.env \
     --outdir /pscratch/sd/j/jibancat/desi-mock-london0-prod-$(date +%Y%m%d)-v3/ \
     --parallel-files 32 --max-workers 8
```

### 6.2 Multi-DLA, real LOA

```bash
# Same env + overrides as §6.1, then:
bash slurm/launch.sh slurm/configs/loa_y3.env \
    --outdir /pscratch/sd/j/jibancat/desi-loa-prod-$(date +%Y%m%d)-v3/
```

### 6.3 LLS mode (sub-DLA + DLA, NHI ∈ [17.2, 22] or [19, 22])

LLS configs already set `MAX_DLAS=1, SINGLE_ABSORBER_MODEL=1, FILTER_LOW_LIKELIHOOD=0`
and select the right `pw_samples_a3_172_220_50000.mat` or `pw_samples_a3_190_220_50000.mat`.
To use the v3 model + τ-EB on top:

```bash
export LEARNED_FILE="/pscratch/sd/j/jibancat/prod533_5k_20260511/null_gp_test/converted/2lpt_loa124_nohcd_nobal_wide.h5"
export NUM_FOREST_LINES=3                # §10.9
export ENABLE_TAU_EB=1
export TAU_EB_OBJECTIVE=null
export EARLY_STOP_MODE=baseline

bash slurm/launch.sh slurm/configs/london0_y3_lls172.env \
    --outdir /pscratch/sd/j/jibancat/desi-mock-london0-lls172-$(date +%Y%m%d)-v3/

bash slurm/launch.sh slurm/configs/loa_y3_lls190.env \
    --outdir /pscratch/sd/j/jibancat/desi-loa-lls190-$(date +%Y%m%d)-v3/
```

LLS mode is ~3.3× the per-spectrum wall of multi-DLA mode (~54 node-hours per
million mock spectra at the v3 stack; LOA real ~78 nh; see §4.6).

### 6.4 Combine results

```bash
# Multi-DLA mock
python combine_processed_h5.py \
    --processed_dir "$OUTDIR" \
    --output_file   "$OUTDIR/combined.h5" \
    --mock

# Multi-DLA real LOA
python combine_processed_h5.py \
    --processed_dir "$OUTDIR" \
    --output_file   "$OUTDIR/combined.h5"
```

### 6.5 Run P/C eval (mock only)

```bash
# Molly-faithful (matches Molly's 2509 notebook headline)
python examples/molly_faithful_pc_plots.py \
    --catalog-dir "$OUTDIR" \
    --truth /global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/dla_cat.fits \
    --bal-cat /global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/bal_cat.fits \
    --no-bal --snr-min 2.0 --nhi-min 20.3 --gp-conf 0.99 --lyb-veto \
    --out "$OUTDIR/figures_molly/snr2_pdla0.99"
```

For Saclay use `hcd_truth_cat.fits` and the Saclay mockdir. For 2LPT, see truth
catalog at `<mockdir>/dla_cat.fits` (or equivalent).

---

## 7. Expected P/C numbers for the v3 high-purity stack

### 7.1 Classical DLAs (NHI ∈ [20.3, 23]), SNR_RED > 2

| Mock     | P_DLA ≥ 0.99    | P_DLA ≥ 0.999   | P_DLA ≥ 0.99999 |
|----------|-----------------|-----------------|------------------|
| London 8f | P=0.845, C=0.766 | P=0.855, C=0.740 | P=0.887, C=0.690 |
| Saclay 8f | P=0.871, C=0.771 | P=0.884, C=0.748 | P=0.901, C=0.677 |
| 2LPT 8f (baseline GP, projection) | P≈0.78, C≈0.91 (no v3 run yet on 2LPT) | — | — |

**For comparison — historical baseline (P_DLA ≥ 0.99, SNR_RED > 2, 26k London-0):**

| Stack | Purity | Completeness | Source |
|---|---:|---:|---|
| `model_epoch_920` + 10k QMC + FILTER=1 + multi-DLA (baseline) | 0.7504 | 0.8206 | `MOLLY_TABLES_SNR_CUTS.md` (the 26k baseline row) |
| same + lyb_veto post-process                                    | 0.7592 | 0.8198 | same |
| **v3 + PW14 50k + τ-EB (8f, this stack)**                       | **0.8452** | **0.7661** | §0 |

Δ vs historical baseline: **+9 pp P, −5.5 pp C**. The completeness regression is
the [20.3, 20.6) NHI bin failure under FILTER=1 (see top callout and
`docs/notes/2026-04-27_filter_completeness_explanation.md`).

**Headline at the recommended operating point (P_DLA ≥ 0.99, SNR_RED > 2):**
Purity ~0.85, Completeness ~0.77. The 8-pp completeness gap below 85 % is
partly *intrinsic at SNR > 2* (sampling-noise is 130× below the signal-null
gap; this is a model-side / forward-model limiter; see today's diagnostic) and
partly *attributable to FILTER=1*. The historical FILTER=0 baseline retains
~5 pp more completeness; switching the v3 stack to FILTER=0 has not been
measured but would likely recover most of the lost completeness at a
proportional purity cost.

### 7.2 Sub-DLA catalog (NHI ∈ [19, 20.3])

**Status: no production-scale sub-DLA P/C is available.**

The 2-way `[null, sub-DLA]` model-selection mode (§2.3) is the correct way to
measure sub-DLA P/C. The closest existing run is **Molly's old Saclay LLS-mode
catalog** (single absorber + PW14 NHI ∈ [17.2, 22], `model_epoch_920`),
post-processed at NHI ≥ 20.3 cut (i.e. measured for DLA P/C from an LLS catalog,
NOT sub-DLA P/C):

| Cut | Catalog | n_cat post-cuts | n_truth post-cuts | P_DLA cut | Purity | Completeness |
|---|---|---:|---:|---:|---:|---:|
| NHI ≥ 20.3, λ_rf ∈ [911, 1216] | Molly Saclay LLS-mode | 281 451 | 75 687 | ≥ 0.99 | 0.4075 | 0.8921 |
| NHI ≥ 20.3, λ_rf ∈ [911, 1216] | Molly Saclay LLS-mode | 281 451 | 75 687 | ≥ 0.99999 | 0.4114 | 0.8872 |

(Source: `prod533_5k_20260511/molly/saclay_lls_mode_old/lya_lyb/molly_summary.tsv`)

This is the LLS catalog at the DLA cut — its **low purity (~41 %)** is the
expected behaviour when DLAs are extracted from an LLS-mode catalog (the
prior boundary at NHI = 17.2 inflates sub-DLA-range posterior weight that
spills into the NHI ≥ 20.3 selection). Inverting the cut to extract sub-DLAs
only (NHI ∈ [19, 20.3]) has **not been computed** at production scale.

**The joint sub-DLA+DLA sweep `joint_dla_subdla_sweep/cell{A,B,C}` is the next
data point.** Cells A/B use PW14 NHI ∈ [19, 23], cell C uses PW14 NHI ∈ [17.2, 22].
All three run with `SINGLE_ABSORBER_MODEL=1` (2-way `[null, absorber]`) on London
mock-0 8 files (~6.6k spectra each). **Inference completed; P/C eval pending.**
Once eval lands, fill in this section apples-to-apples with §7.1 (same SNR cut,
same P_DLA cut family, same forest window, NHI cut [19, 20.3] for sub-DLA).

For sub-DLA P/C **on the v3 stack** specifically (i.e. v3 GP + PW14 50k + τ-EB in
2-way LLS mode + NHI cut [19, 20.3]), a production run is required — none exists yet.
The LLS-mode configs (§2.2) with the v3 overrides from §0 will produce this catalog;
the eval script `molly_faithful_subdla_pc_plots.py` does NOT exist yet (the existing
`molly_faithful_pc_plots.py` would need `--nhi-min 19.0 --nhi-max 20.3` and a
sub-DLA truth filter — that script extension is a separate task).

### 7.3 SNR > 1 (acceptable per project memory)

Completeness drops to ~65 % at SNR > 1 because the FILTER mechanism under-weights
marginal-z modes. See `MOLLY_TABLES_SNR_CUTS.md` SNR>1 table.

### 7.4 SNR > 4 (cosmology-grade subset; not the headline)

P ≈ 0.83 / C ≈ 0.90 at P_DLA ≥ 0.999 with the full stack — approximately 85/85.

### 7.5 LLS catalog (NHI ∈ [17.2, 20.3], from LLS-mode runs)

Not validated by the v3 stack harness yet. Use Pathway A CDDF
(`CDDF_analysis/calc_cddf.py`) on LLS-mode HDF5 posteriors, not the
catalog-time P/C tool, because LLS-mode emits a single-absorber catalog with
NHI floor 17.2.

---

## 8. Quick lookup: "I want to run X"

| Goal | Config | Outdir suffix |
|------|--------|---------------|
| DLA catalog on London-0 (production) | `slurm/configs/london0_y3.env` + §0 exports | `desi-mock-london0-prod-YYYYMMDD-v3` |
| DLA catalog on London-1 | (TODO: create `london1_y3.env`) | `desi-mock-london1-prod-YYYYMMDD-v3` |
| DLA catalog on Saclay-0 | `slurm/configs/saclay0_y3.env` + §0 exports | `desi-mock-saclay0-prod-YYYYMMDD-v3` |
| DLA catalog on Saclay-1 | (TODO: create `saclay1_y3.env`) | `desi-mock-saclay1-prod-YYYYMMDD-v3` |
| DLA catalog on 2LPT-0 | `slurm/configs/2lpt0_y3.env` + §0 exports | `desi-mock-2lpt0-prod-YYYYMMDD-v3` |
| DLA catalog on DESI LOA | `slurm/configs/loa_y3.env` + §0 exports | `desi-loa-prod-YYYYMMDD-v3` |
| LLS catalog NHI≥17.2 (any dataset) | `slurm/configs/<flavour>_y3_lls172.env` + v3+τ-EB exports | `…-lls172-YYYYMMDD-v3` |
| LLS catalog NHI≥19.0 (any dataset) | `slurm/configs/<flavour>_y3_lls190.env` + v3+τ-EB exports | `…-lls190-YYYYMMDD-v3` |

---

## 9. After all jobs finish

```bash
# 1. Combine per-file HDF5 → single combined.h5
python combine_processed_h5.py --processed_dir "$OUTDIR" --output_file "$OUTDIR/combined.h5" [--mock]

# 2. P/C eval (mock only, see §6.5)

# 3. Catalog-time post-process: lyb_veto (free ~+1.7 pp purity)
python -c "
from astropy.table import Table
from gpy_dla_detection.postprocess.lyb_veto import flag_lybeta
cat = Table.read('$OUTDIR/combined_dlacat.fits')
cat = flag_lybeta(cat, dz_match=0.005, targetid_col='TARGETID', z_col='Z_DLA', nhi_col='NHI')
cat = cat[~cat['LYBETA_FLAG']]
cat.write('$OUTDIR/combined_dlacat_lybveto.fits', overwrite=True)
"

# 4. CDDF / Ω_HI (population statistics)
# See docs/tutorial_population_statistics.md
```

---

## 10. Gotchas (READ BEFORE LAUNCHING)

### 10.1 Production sbatch scripts don't forward τ-EB / early_stop_mode

`slurm/submit_desi_mock.sh` and `slurm/submit_desi_loa.sh` do **not** pass
`--enable_tau_eb`, `--tau_eb_objective`, or `--early_stop_mode` to the python
CLI. The flags exist in `desi-DLAGP.py:parse()` and the env vars exist in the
exports, but the sbatch scripts' python command lines were last updated before
those flags landed.

Only `slurm/run_local.sh` forwards them (see lines 187-204).

**Fix options:**
- (a) Use `slurm/run_local.sh` on a salloc'd interactive node. Lose the queue
  fan-out but get correct hyperparams.
- (b) Patch `slurm/submit_desi_{mock,loa}.sh` to add three lines at the end of
  the python command:
  ```bash
  $(if [ "$ENABLE_TAU_EB" = "1" ]; then echo "--enable_tau_eb 1 --tau_eb_objective $TAU_EB_OBJECTIVE"; fi) \
  --early_stop_mode "${EARLY_STOP_MODE:-baseline}"
  ```
  before the trailing `&`. Then launch via `slurm/launch.sh`. **Not yet patched
  in `production_533`.**

This is a deployment blocker for any production run that needs τ-EB or A/D
early-stop modes via sbatch. Validate by looking at the actual `python …`
command in `slurm/submit_desi_{mock,loa}.sh` before believing the env vars
made it through.

### 10.2 NUM_DLA_SAMPLES must equal the .mat rowcount

`pw_samples_a3_190_220_50000.mat` is 50k rows. Running with `NUM_DLA_SAMPLES=10000`
will silently read only the first 10k samples — but the QMC weight normalization
assumes `1/N` where N is the *file* count. The math goes wrong. Always pair the
file with its rowcount.

### 10.3 Saclay mock-0 subdir is `juraLy8-124`, mock-1 is `jura-124`

Already encoded in the configs but worth knowing if you write a new one.

### 10.4 BAL inclusion vs exclusion

Inference: BAL QSOs are **included** (`BALMASK=false`). No pixel masking.
P/C eval: BAL QSOs are **excluded** via `--no-bal` flag in `molly_faithful_pc_plots.py`
(filters `BI_CIV > 0` from the truth + catalog). Always match this convention.

### 10.5 Z ≥ 1.96 cut is at inference time

The QSO catalog gets filtered in `desi-DLAGP.py` (look for `Z >= 1.96` or
similar near the `read_in_each_plots_*` logic). All numbers in §1.1 reflect
this cut. If you want to keep low-z QSOs, you have to patch the catalog
filter.

### 10.6 `EARLY_STOP_MODE` default is `baseline`, set via env

The `--early_stop_mode` CLI flag defaults to `os.environ.get("EARLY_STOP_MODE",
"baseline")`. If you don't set it, production behavior is unchanged
(`baseline`). Variants A and D are under evaluation and **not yet promoted to
production**.

### 10.7 The launcher refuses unsafe `OUTDIR`s

`slurm/launch.sh` refuses to submit if `OUTDIR` falls outside
`/pscratch/sd/j/jibancat/`, `/global/homes/j/jibancat/`, or
`/global/cfs/cdirs/desicollab/users/jibancat/`. This is by design; do not work
around it. See `docs/nersc_write_permissions.md`.

### 10.8 The current jupyter node has 22 inference slices running

As of 2026-05-13 12:00 PT, `nid004179` (job 52907557) is running 22 resume
slices inline. **Do not submit new jobs or kill these.** They populate
`prod533_5k_20260511/{london_v3_loa124_early_stop_A,early_stop_D,joint_dla_subdla_sweep/*}`.

### 10.9 `NUM_FOREST_LINES`: sbatch default (31) ≠ training value (3) ≠ `_base.env` (3)

There are three values for `NUM_FOREST_LINES` in this repo:

| Source | Value | Wiring |
|---|---:|---|
| `slurm/submit_desi_mock.sh:51`, `slurm/submit_desi_loa.sh:52` | **31** | sbatch entrypoint default |
| `slurm/configs/_base.env:35`, `slurm/run_local.sh` (via `_base.env`) | **3**  | local-mode entrypoint default |
| v3 GP training config (`/pscratch/.../v2_runs/2lpt_loa124_nohcd_nobal_52188321/config.json`) | **3** | what the v3 model was trained with |
| v2 trainer default (`gpy_dla_detection/training/dataset.py:223`, `de_forest_num_lines`) | **3** | trainer code default |

**For the v3 GP (`2lpt_loa124_nohcd_nobal_wide.h5`)**: training was at 3. Inference
should also be at 3 to avoid a train/inference mismatch. The local-mode path
already does this. **The sbatch path with the default `NUM_FOREST_LINES=31` is a
latent bug** — anything launching via `submit_desi_{mock,loa}.sh` without
exporting `NUM_FOREST_LINES=3` first will run inference at 31 against a model
trained at 3.

**Fix options:**
- (a) Always `export NUM_FOREST_LINES=3` before any v3-stack sbatch launch.
- (b) Patch the sbatch script defaults to `NUM_FOREST_LINES=3`.
- (c) If the trainer agent rebuilds the GP, explicitly retrain at
  `num_forest_lines=31` and document that in the new training-config artifact.
  Then change the inference default accordingly.

**For the legacy `model_epoch_920.h5`**: training config is not in this session's
file tree. The historical sbatch default of 31 may match its training; needs
verification by inspecting the matching trainer config (if `model_epoch_920` was
trained by the v2 trainer, it inherits the `num_forest_lines=3` trainer default,
in which case the sbatch default of 31 is also wrong for legacy production).
**Recommend the trainer-agent run a follow-up to enumerate the training
`num_forest_lines` for every shipped `.h5` GP and align inference defaults.**

---

## 11. Reproduce the current best result (copy-pasteable)

```bash
# === Step 0: env ===
source /usr/share/lmod/lmod/init/bash
export DESI_ROOT=/global/cfs/cdirs/desi
source /global/common/software/desi/desi_environment.sh main
cd /pscratch/sd/j/jibancat/desi_gpy_dla_detection

# === Step 1: v3 high-purity stack exports ===
# (See top-of-file KNOWN REGRESSION callout — this is NOT yet the recommended
# production baseline; it is the highest-purity config tested. Completeness
# regression in the [20.3, 20.6) NHI bin vs the historical FILTER=1 10k baseline.)
export LEARNED_FILE="/pscratch/sd/j/jibancat/prod533_5k_20260511/null_gp_test/converted/2lpt_loa124_nohcd_nobal_wide.h5"
export NUM_FOREST_LINES=3   # MUST match training; see §10.9
export DLA_SAMPLES_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_190_220_50000.mat"   # PW14 NHI ∈ [19, 22], NOT [19, 23]
export NUM_DLA_SAMPLES=50000
export ENABLE_TAU_EB=1
export TAU_EB_OBJECTIVE=null
export EARLY_STOP_MODE=baseline
export MAX_DLAS=3
export FILTER_LOW_LIKELIHOOD=1

OUTDIR="/pscratch/sd/j/jibancat/desi-mock-london0-prod-$(date +%Y%m%d)-v3"

# === Step 2: dry-run ===
bash slurm/launch.sh slurm/configs/london0_y3.env \
    --outdir "$OUTDIR" --dry-run --no-sleep | head

# === Step 3: PATCH submit_desi_mock.sh FIRST (§10.1, §10.9) ===
# (a) Add to the python command lines, before the trailing `&`:
#     --enable_tau_eb "$ENABLE_TAU_EB" \
#     --tau_eb_objective "$TAU_EB_OBJECTIVE" \
#     --early_stop_mode "$EARLY_STOP_MODE" \
# (b) Change NUM_FOREST_LINES default to 3 (currently 31) — see §10.9. OR
#     ensure `export NUM_FOREST_LINES=3` happens before sbatch.

# Either patch the file then:
bash slurm/launch.sh slurm/configs/london0_y3.env --outdir "$OUTDIR"

# OR use run_local.sh on a salloc'd node (forwarding already correct):
salloc -N 1 -C cpu -q interactive -t 04:00:00 -A desi
bash slurm/run_local.sh slurm/configs/london0_y3.env \
    --outdir "$OUTDIR" --parallel-files 32 --max-workers 8

# === Step 4: combine + P/C ===
python combine_processed_h5.py \
    --processed_dir "$OUTDIR" \
    --output_file   "$OUTDIR/combined.h5" \
    --mock

python examples/molly_faithful_pc_plots.py \
    --catalog-dir "$OUTDIR" \
    --truth /global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/dla_cat.fits \
    --bal-cat /global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/bal_cat.fits \
    --no-bal --snr-min 2.0 --nhi-min 20.3 --gp-conf 0.99 --lyb-veto \
    --out "$OUTDIR/figures_molly/snr2_pdla0.99"

# Expected at SNR>2, P_DLA≥0.99, lya_lyb [911, 1216] Å:
#   Purity  ~0.845
#   Completeness ~0.766
# (per docs/runs/2026-05-12_saclay_v3_loa124_results.md and the 8f reference run)
```
