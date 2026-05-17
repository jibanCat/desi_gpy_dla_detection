# Handoff — 2026-05-17 (login node)

> Consolidation session. The 2026-05-15/16 sessions ran a large batch of
> knob sweeps as `regular`-QOS sbatch jobs (all `COMPLETED`, queue now
> empty) but left the write-ups half-finished. This session: filled in
> the stale notes, wrote notes for the undocumented sweeps, and launched
> the MAX_LAMBDA cross-validation. **Regular-queue turnaround is now
> fast** (05-15/16 jobs ran within hours) — the 05-14 "10+ days deep"
> warning is stale.

## Headline science state

**Production knob picture (post +log(N) patch, 2-way cellC family,
London-0 5k, fixed molly recipe n_truth=581):**

| Knob | Decision | Evidence |
|---|---|---|
| **MAX_LAMBDA** | **1250** (was 1216.75) | lambda_fine F2: P=0.838/C=0.830, Pareto-best |
| MIN_LAMBDA | keep 911.75 | gp_range: blue moves inert-to-bad |
| MIN_Z_SEPARATION | keep 3000 km/s | min_z sweep: ≤noise, inert |
| NHI prior ceiling | extend to 22.5 | nhi_prior: P/C-neutral, cost-neutral, fixes high-NHI clipping |
| SINGLE_ABSORBER_MODEL | keep 1 | single_absorber: +12.6pp P, +21.4pp C vs mode 0 |
| +log(N) patch | keep ON | logn_patch A/B: −1.6pp P / +4.9pp C, favours CDDF LLS |
| 2-way vs 3-way | **2-way** (cellC) | subdla_3way: all 3-way cells lose completeness |

**The 1M production config** = 2-way single-absorber, MAX_LAMBDA=1250,
MIN_LAMBDA=911.75, MIN_Z_SEPARATION=3000, NHI prior [17.2, 22.5],
+log(N) patch ON, τ-EB null, PW 50k–100k. Still open: the p_DLA cut
convention (default 0.99 vs tightened — see 05-14 §"Priority 1") and
final PW count (50k vs 100k, ~1pp).

## In flight — 3 sweeps submitted 2026-05-17 (all `-q regular`)

All under `/pscratch/sd/j/jibancat/prod533_5k_20260511/`, auto-eval on
completion (watch each dir's `_chain.log` → `EVAL_DONE`, then `HEADLINE.tsv`).

1. **`lambda1250_crossval/`** (job `53076988`) — MAX_LAMBDA=1250 off-dist
   validation. 4 cells = paired 1216.75-vs-1250 on Saclay-0 + 2LPT-0, 50k
   PW. **Pickup**: if L_saclay>B_saclay and L_2lpt>B_2lpt on the Pareto
   front, 1250 is confirmed; append result to
   `docs/notes/2026-05-16_lambda_fine_and_gp_range.md`.

2. **`min_z_separation_sweep_50k/`** (jobs `53077531/533/534/535` + eval
   `53077536`) — MIN_Z_SEPARATION re-sweep at **50k spectra** (10× the 5k
   slice) to resolve M0-M3 above the noise floor; matters for downstream
   DLA-clustering. Run at MAX_LAMBDA=1250. **Pickup**: fill in
   `docs/notes/2026-05-15_min_z_separation_smoke.md` (or a new 50k note);
   if M1 (2000 km/s) holds its +2pp purity at 50k, reconsider the knob.

3. **`model_sweep/`** (job `53077686`) — 5 cells, baseline + 4 PR6
   `phase2_desi` trained GP models (V1 2lpt_loa124_m, V2 2lpt_loa0_m,
   V3 real-LOA no-dla, V4 real-LOA with-bal), current best config.
   **Key context**: the V0 baseline is the β-collapsed deprecated model
   (β=1.45, Garnett norm band) — see memory
   `project_baseline_model_beta_collapse`. **Pickup**: write a sweep note;
   if a healthy `_m` model wins, the 1M run should switch to it (and OFF
   the β-collapsed baseline).

## Sweep notes written/refreshed this session

| Note | Sweep | Verdict |
|---|---|---|
| `2026-05-15_lambda_range_smoke.md` | lambda_range L0-L3 | HELPS — red extension Pareto-improves |
| `2026-05-15_min_z_separation_smoke.md` | min_z M0-M3 | NO-CHANGE — keep 3000 km/s |
| `2026-05-15_nhi_prior_extension.md` | nhi E0/E1 | NO-CHANGE on P/C, ADOPT (fixes clipping) |
| `2026-05-16_lambda_fine_and_gp_range.md` | lambda_fine F0-F5 + gp_range | **MAX_LAMBDA=1250** |
| `2026-05-16_logn_patch_ab.md` | logn_patch P/G arms | patch confirmed: −1.6/+4.9pp |
| `2026-05-16_subdla_3way_sweep.md` | subdla U0-U3 | 2-way confirmed over 3-way |
| `2026-05-16_config_confirmations.md` | v1_model, single_absorber, determinism | noise floor ~0.3pp; v1-model swap not a headline win |

**Recipe-version caveat**: every 05-15/16 sweep was evaluated with the
*fixed* molly recipe (`2026-05-15_molly_eval_recipe_fix.md`: drop-ALL-BAL,
external snr/zcat → n_truth=581). The 05-14 cellC/cellD sweeps used the
*old* recipe (n_truth=618). The two generations are **not** directly
P/C-comparable; compare within a recipe generation only.

## Open items for next session

1. **Cross-val pickup** — read `lambda1250_crossval/HEADLINE.tsv`, append
   the result to `2026-05-16_lambda_fine_and_gp_range.md`, decide.
2. **V0 per-SNR** — `2026-05-16_config_confirmations.md` flags an unread
   item: does the v1-model swap recover the SNR>10 completeness loss?
   Read `v1_model_test/V0_v1model/pc_snr2_pdla99.md/molly_summary.tsv`.
3. **p_DLA cut convention** — still unpicked (05-14 Priority 1).
4. **PR** — uncommitted: +log(N) patch (`dla_gp.py`), molly recipe fix
   (`molly_faithful_pc_plots.py`), `fitwarning.py`, `production_runbook.md`,
   `run_local.sh`. Base branch `desi_y3` (memory `project_base_branch`).

---

# Handoff — 2026-05-14 EVENING (jupyter 52973367 on nid004315)

> **Patched the dla_gp.py log-evidence −log(N) bias** (path (a): kept
> the per-sample bake-in at line 212-214 / 425-429, added `+log(N)` to
> every downstream evidence formula — 7 sites, see "Patch summary"
> below). 94/94 network-free tests pass. Re-ran cellC + cellD sweeps
> (18 cells, ~4h wall) on London-0 5k with the patched code.
>
> **Headline (post-patch, default p_DLA ≥ 0.99)**:
>
> | Family | Headline winner | P | C |
> |---|---|---:|---:|
> | **2-way (recommended for CDDF LLS)** | **C7 (PW 100k)** | **0.776** | **0.892** |
> | 2-way (balanced) | C0 (baseline md=3) | 0.779 | 0.877 |
> | 3-way (best) | D7 (PW 100k) | 0.852 | 0.807 |
>
> **Patch's behavior shift**: log_evidence(DLA) ↑ log(N) ≈ 10.8 for
> N=50k. DLA-vs-null Bayes factor inflates uniformly → more spectra
> cross p_DLA ≥ 0.99 → +50–100% n_cat → P drops ~5pp, C rises ~5pp on
> 2-way, less on 3-way. **The patch is mathematically correct**; the
> conf-cut sweep on C0 confirms `p_DLA ≥ 1−1e-7` (= log_BF ≥ 15.4)
> recovers pre-patch operating point within 1pp.
>
> **C5/C6/C7 PW-count cells are now flat within ~1pp** (was ~3.5pp
> spread pre-patch in lockstep with `log(N)`) — confirms the bias is
> gone. **D7 (PW 100k) replaces pre-patch D1 (MAX_DLAS=4) as the 3-way
> Pareto winner** — the pre-patch D1 finding was bias artifact.
>
> **2-way (cellC) is the preferred family** for the user's CDDF LLS use
> case because the [17.2, 22] single-absorber model is directly usable
> for sub-DLA / LLS analysis without a post-hoc NHI gate. NHI gate
> (k>0) is **deferred** until NHI_ERR is validated separately.
>
> **Production-pick decision**: see "Decisions still open" + "Priority 1"
> + "Priority 2" below. User has the conf-cut convention (default 0.99
> with shifted operating point vs tightened 1−1e-7 to recover pre-patch)
> and the knob choice (C7 vs C0) to pick before the 1M run.

## Patch summary — dla_gp.py log-evidence +log(N)

**Bug**: `process_sample` at `dla_gp.py:212-214` (and `:425-429` in the
serial loop) pre-subtract `np.log(num_dla_samples)` from every per-sample
log-likelihood. The downstream MC estimator
`max(S) + log nanmean(probs)` then evaluates to
`log mean(exp(L_i)) − log(N)` instead of `log mean(exp(L_i))`, biasing
1-DLA log evidence DOWN by `log(N)` (≈ 10.8 for N=50k). At fixed N the
bias is constant; comparing cells at different N (e.g. C5/C6/C7 with PW
30k/80k/100k) the bias differs by `log(N_cell/N_baseline)`, which the
prior session verified empirically to <1%.

**Fix path (a)**: keep the per-sample bake-in (preserves the threshold
semantic at `dla_gp.py:135`, `initial_logL > null_evidence`, which
compares `S_i = L_i − log N` to the unbiased null log evidence) and add
`+ np.log(self.params.num_dla_samples)` to each log_likelihoods_dla
formula and to log_initial_logL inside the partition estimator.

**Touched lines** (all in `gpy_dla_detection/dla_gp.py`):

| Line | Branch |
|---|---|
| 463 | non-parallel `log_model_evidences` standard estimator (uses `+ lognorm`) |
| 672 | parallel early-stop on empty valid_mask |
| 732 | rejected-region `log_initial_logL` inside the partition |
| 814 | FILTER fix #5 1-DLA branch (uses `initial_logL`) |
| 839 | truncated branch `log_Z_trunc` |
| 865 | standard branch (filter_low_likelihood=False) |
| 898 | early-stop "D" mode pre-Occam `stop_lik` (kept on the same scale) |

Each touch adds exactly one line: `+ np.log(self.params.num_dla_samples)`.
Paired with a brief inline comment pointing at this patch note. The
truncated branch's `log_ratio = log(N) − log(n_initial)` correction at
line 846 is left as-is — it's a separate (pre-existing) design choice
about how the partition formula combines `Z_A`/`Z_B`, not part of the
verified `−log(N)` bias.

**Behavior implication**: `log_evidence(DLA) shifts up by +log(N)`
uniformly across all DLA k-models. `log_evidence(null)` from `null_gp.py`
does not have this bake-in and is unchanged. So every DLA-vs-null Bayes
factor moves by `+log(N) ≈ +10.8` for N=50k, i.e. `p(DLA|D)` is shifted
toward 1. **This may make the historical p_DLA ≥ 0.99 cut a much weaker
filter** (most spectra now sit very close to 1.0). The cellC/cellD
re-sweeps will tell us by how much; if the operating point needs to
move (e.g., to p_DLA ≥ 1 − 1/N), that becomes the next decision.

**Tests**: `pytest tests/test_cddf_mock.py tests/test_cddf_calibration.py
tests/test_generate_samples.py tests/test_voigt_v2_parity.py
tests/test_lyb_veto.py tests/test_smoke_target_contamination.py
tests/test_tau_eb_wiring.py` — 94 passed, 6 skipped. The DLA-MAP /
test_model / test_prior failures all hit `FileNotFoundError` on SDSS
spectra downloads (not patch-related; these were also failing before).

## Patched-sweep status (uncommitted, in /pscratch)

The post-patch sweeps add **C0** (2-way cellC baseline, MAX_DLAS=3) to
the 2-way sweep and **D0** (3-way baseline, MAX_DLAS=3, NHI [19,22]) to
the 3-way sweep, so HEADLINE.tsv contains an apples-to-apples baseline
for both families.

Configs:
- `/pscratch/sd/j/jibancat/prod533_5k_20260511/cellC_knob_sweep/configs/C0.env`
- `/pscratch/sd/j/jibancat/prod533_5k_20260511/cellD_knob_sweep/configs/D0.env`

Eval scripts updated to include C0/D0:
- `/pscratch/sd/j/jibancat/prod533_5k_20260511/cellC_knob_sweep/_eval_and_aggregate.sh`
- `/pscratch/sd/j/jibancat/prod533_5k_20260511/cellD_knob_sweep/_eval_and_aggregate.sh`

Chain runner (auto-batching 8-then-6 cells against 256 cores):
- `/pscratch/sd/j/jibancat/prod533_5k_20260511/_chain_sweeps.sh`

When the chain logs `ALL_BATCHES_DONE` to
`/pscratch/sd/j/jibancat/prod533_5k_20260511/_chain_sweeps.log`,
run the two eval scripts in any order to populate HEADLINE.tsv:

```bash
bash /pscratch/sd/j/jibancat/prod533_5k_20260511/cellC_knob_sweep/_eval_and_aggregate.sh
bash /pscratch/sd/j/jibancat/prod533_5k_20260511/cellD_knob_sweep/_eval_and_aggregate.sh
```

(Reads pre-existing `pc_snr2_pdla99.md` / will create them via
`molly_faithful_pc_plots.py` if the dir exists. Operating point unchanged
from previous runs: SNR>2, P_DLA≥0.99, lyb-veto, no-BAL,
λ_rf ∈ [911, 1216], NHI≥20.3 truth+predicted.)

## Smoke results (batch 1: C0, C1, C2, D0 — DONE 19:48)

Same molly eval recipe as before: SNR>2, P_DLA≥0.99, lyb-veto, no-BAL,
λ_rf ∈ [911, 1216], NHI≥20.3 truth+predicted, n_truth=618.

| Family | Cell | Pre-patch P/C | Post-patch P/C | ΔP | ΔC | n_cat (post→pre) |
|---|---|---|---|---:|---:|---:|
| 2-way | C0 baseline (md=3) | 0.8256 / 0.8304 | **0.7792 / 0.8772** | −4.7 | +4.7 | 5167 → 3268 (+58%) |
| 2-way | C1 (MAX_DLAS=4) | 0.7955 / 0.8187 | 0.7641 / 0.8713 | −3.1 | +5.3 | 5838 → 3891 (+50%) |
| 2-way | C2 (MAX_DLAS=5) | 0.8068 / 0.8304 | 0.7677 / 0.8889 | −3.9 | +5.9 | 6352 → 4397 (+44%) |
| 3-way | D0 baseline (md=3) | 0.8452 / 0.7661 | **0.8323 / 0.7982** | −1.3 | +3.2 | 2440 → 1242 (+96%) |

**Direction is exactly as predicted** by the patch: log_evidence(DLA) ↑
log(N), so DLA-vs-null Bayes factors uniformly inflate, more spectra
cross the p_DLA ≥ 0.99 cut, n_cat grows ~50–100%, P drops, C rises.
3-way absorbs the shift more gracefully (smaller swings) because its
separate SubDLA channel acts as a buffer for marginal DLA evidence.

## p_DLA cut sweep on patched C0 + D0 (recovers pre-patch operating point)

The patch is mathematically correct — relative ordering is preserved.
Tightening the p_DLA cut undoes the operating-point shift:

### C0 (post-patch, 2-way cellC baseline)

| p_DLA cut | P | C |
|---|---:|---:|
| 0.99 (old default) | 0.7792 | 0.8772 |
| 0.999 | 0.7888 | 0.8626 |
| 0.9999 | 0.8022 | 0.8538 |
| 0.99999 | 0.8073 | 0.8450 |
| 0.9999999 | 0.8169 | 0.8216 |

**0.9999999 (~7 nines) recovers pre-patch P/C within ~1pp.**

### D0 (post-patch, 3-way baseline)

| p_DLA cut | P | C |
|---|---:|---:|
| 0.99 (old default) | 0.8323 | 0.7982 |
| 0.999 | 0.8605 | 0.7573 |
| 0.9999 | 0.8849 | 0.7193 |
| 0.99999 | 0.8947 | 0.6959 |
| 0.9999999 | 0.9129 | 0.6433 |

**For 3-way, the new operating point at 0.99 is already Pareto-better
than pre-patch** (0.832/0.798 vs 0.845/0.766: +3.2pp C for −1.3pp P).
Tightening would over-correct.

### Equivalent log-Bayes-factor cut

Pre-patch `p_DLA ≥ 0.99` ⇔ `log BF ≥ log(99) ≈ 4.6` (in pre-patch
units, biased by −log N). Post-patch the unbiased threshold for the
same operating point is `log BF ≥ log(99) + log(N) ≈ 4.6 + 10.8 = 15.4`
for N=50k. Working in log-BF space is **N-invariant** and cleaner than
chasing decimal places of p_DLA.

Per-cell sweeps live in `cellC_knob_sweep/C0/pc_conf_*` and
`cellD_knob_sweep/D0/pc_conf_*`.

## Patched-sweep status — DONE 22:49

Chain timing:
- Batch 1 (C0,C1,C2,D0): launched 18:46, done 19:48 (62 min, no contention)
- Batch 2 (C3..C8 + D1..D2, 8 cells): launched 19:48, done 21:28 (100 min, 2× contention)
- Batch 3 (D3..D8, 6 cells): launched 21:28, done 22:49 (81 min, 1.5× contention)
- Total: 4h03m wall, 18 cells.

Final HEADLINE.tsv files at:
- `/pscratch/sd/j/jibancat/prod533_5k_20260511/cellC_knob_sweep/HEADLINE.tsv`
- `/pscratch/sd/j/jibancat/prod533_5k_20260511/cellD_knob_sweep/HEADLINE.tsv`

### Patched cellC sweep (2-way) — DONE 22:49

| Cell | Knob | P | C | ΔP vs C0 | ΔC vs C0 | n_cat |
|---|---|---:|---:|---:|---:|---:|
| **C0** | baseline (md=3) | **0.7792** | **0.8772** | ref | ref | 5167 |
| C1 | MAX_DLAS=4 | 0.7641 | 0.8713 | −1.5 | −0.6 | 5838 |
| C2 | MAX_DLAS=5 | 0.7677 | 0.8889 | −1.2 | **+1.2** | 6352 |
| C3 | NHI [17.2, 23] | 0.7755 | 0.8684 | −0.4 | −0.9 | 5841 |
| C4 | NHI [18, 22] | 0.7746 | 0.8743 | −0.5 | −0.3 | 5162 |
| C5 | PW 30k | 0.7661 | 0.8713 | −1.3 | −0.6 | 5279 |
| C6 | PW 80k | 0.7732 | 0.8772 | −0.6 | 0.0 | 5058 |
| **C7** | **PW 100k** | **0.7761** | **0.8918** | **−0.3** | **+1.5** | 5015 |
| C8 | n_init=10k | 0.7734 | 0.8684 | −0.6 | −0.9 | 5257 |

**Headline**: C7 (PW 100k) is a **marginal Pareto-dominator** (−0.3pp P,
+1.5pp C). C5/C6/C7 are now **flat within ~1pp**, confirming the
−log(N) bias is gone (pre-patch they spread by ~3.5pp on P and 1pp on C
in lockstep with `log(N)`). NHI prior tweaks (C3/C4) and MAX_DLAS≥4
(C1) are slightly worse than C0 baseline. n_truth=618 throughout.

### Patched cellD sweep (3-way) — DONE 22:49

| Cell | Knob | P | C | ΔP vs D0 | ΔC vs D0 | n_cat |
|---|---|---:|---:|---:|---:|---:|
| **D0** | baseline (md=3) | **0.8323** | **0.7982** | ref | ref | 2440 |
| D1 | MAX_DLAS=4 | 0.8303 | 0.8012 | −0.2 | +0.3 | 2452 |
| **D2** | MAX_DLAS=5 | **0.8354** | **0.8012** | **+0.3** | **+0.3** | 2450 |
| D3 | NHI [17.2, 22] | 0.8292 | 0.7807 | −0.3 | −1.7 | 3473 |
| **D4** | NHI [19, 23] | **0.8369** | **0.8099** | **+0.5** | **+1.2** | 3095 |
| D5 | PW 30k | 0.8283 | 0.8041 | −0.4 | +0.6 | 2473 |
| **D6** | PW 80k | **0.8445** | **0.8099** | **+1.2** | **+1.2** | 2432 |
| **D7** | **PW 100k** | **0.8519** | **0.8070** | **+2.0** | **+0.9** | 2416 |
| D8 | n_init=10k | 0.8283 | 0.8041 | −0.4 | +0.6 | 2527 |

**Headline**: **D7 (PW 100k) is the clear 3-way winner** (+2.0pp P,
+0.9pp C). D6 (PW 80k) and D4 (NHI [19, 23]) are also strict
Pareto-dominators of D0. D5/D6/D7 spread is now +0.6→+0.9→+0.9pp C and
−0.4→+1.2→+2.0pp P — a clean monotone P trend that pre-patch was
contaminated by the −log(N) bias (pre-patch D7 was P=0.874/C=0.752,
which over-stated P and under-stated C; post-patch D7 is genuinely the
best). D1 (MAX_DLAS=4) is now flat — pre-patch it looked like a
Pareto-dominator but that was bias artifact.

### Cross-family at default p_DLA ≥ 0.99

| Family | Best | P | C |
|---|---|---:|---:|
| 2-way (best on C) | C7 (PW 100k) | 0.776 | 0.892 |
| 2-way (best balanced) | C0 (baseline) | 0.779 | 0.877 |
| 3-way (best on P) | D7 (PW 100k) | 0.852 | 0.807 |
| 3-way (best balanced) | D6 (PW 80k) | 0.845 | 0.810 |

3-way wins ~7pp on purity at ~9pp completeness cost vs 2-way at the
default cut. The frontiers cross at no balanced point — the choice is
purely whether you prioritize C (use 2-way) or P (use 3-way).

## Production-run pickup priorities (for next-Claude)

### Priority 1 — pick the production p_DLA cut convention

Two equivalent options; pick one and document it in the runbook:

(a) **Stay at p_DLA ≥ 0.99** and accept the new operating point. For
    2-way cellC this means ~5pp more recall, ~5pp less purity. For
    3-way it's a strict Pareto improvement vs pre-patch.

(b) **Move to log-BF ≥ 15.4** (N-invariant) or equivalently
    p_DLA ≥ 1 − 1/(99·N). This recovers the pre-patch operating point
    almost exactly. Cleanest if 1M production is at fixed N=50k.

The user previously said completeness matters for CDDF LLS, which would
favour (a) for the 2-way family. Confirm with the user before coding it
into the production catalog post-processing.

### Priority 2 — pick the 2-way knob within cellC family

**Recommendation for the user's CDDF LLS use case** (where completeness
matters): **C7 (PW 100k)** at default p_DLA ≥ 0.99: P=0.776, C=0.892.
Marginal Pareto-dominator over C0 (+1.5pp C for −0.3pp P); ~20% extra
QMC cost vs PW 50k.

If you want the same operating point as pre-patch cellC (≈0.83/0.83):
- C7 + tightened cut to p_DLA ≥ 1−1e-7  →  ~P=0.81, C=0.83
- or stay at C0 + p_DLA ≥ 1−1e-7  →  P=0.817, C=0.822 (verified above)

C5/C6/C7 are now flat within 1pp — confirms the bias is gone. Other
knobs (C1, C2, C3, C4, C8) do not Pareto-dominate; reject them as
production candidates.

### Priority 2b — alternate 3-way recommendation if you change your mind

If purity matters more than CDDF LLS coverage and you switch to 3-way:
**D7 (PW 100k)** is the clear winner: P=0.852, C=0.807 (+2.0pp P,
+0.9pp C vs D0). D6 (PW 80k) and D4 (NHI [19,23]) are runners-up.

### Priority 3 — Saclay + 2LPT cross-validation of chosen config

User wants this **before** the 1M-spectrum production run. Reuse the
existing slurm configs:
- `slurm/configs/saclay0_y3.env` (if exists; otherwise adapt cellC0)
- `slurm/configs/2lpt_y3.env` (or adapt)
Run on Saclay mock-0 + 2LPT mock at 5k each. Cost ~0.3 nh per cell.

### Priority 4 — NHI consistency gate (k>0) — DEFERRED

User flagged that NHI_ERR has not been validated as a per-spectrum
calibrated uncertainty. **Skip** until a separate validation lands. To
validate when ready: pick a known-good DLA target, plot the per-spectrum
DLA-MAP NHI posterior, fit a Gaussian to the marginalised log-NHI dist,
compare its σ to the reported NHI_ERR column.

### Priority 5 — commit + open PR #7

PR base branch is `desi_y3` (per memory `project_base_branch`). Logical
commit chunks (in order):
1. **dla_gp.py +log(N) patch** (this session, ~10 changed lines + comments)
2. **C0/D0 baseline configs** + eval-script C0/D0 inclusion
3. **HANDOFF** + this session's findings note
4. (If we landed it) production runbook §3.7 with the new p_DLA
   convention + recalibration recipe.

## Files written in this session (uncommitted)

| File | Purpose |
|---|---|
| `gpy_dla_detection/dla_gp.py` | +log(N) bias-fix at 7 estimator sites |
| `cellC_knob_sweep/configs/C0.env` | 2-way cellC baseline (post-patch) |
| `cellD_knob_sweep/configs/D0.env` | 3-way baseline (post-patch) |
| `cellC_knob_sweep/_eval_and_aggregate.sh` | + C0 in iteration |
| `cellD_knob_sweep/_eval_and_aggregate.sh` | + D0 in iteration |
| `_chain_sweeps.sh` | auto-batch chain (3 batches → 18 cells) |
| `HANDOFF.md` | this section |

Outputs (off-repo, in /pscratch):
- `cellC_knob_sweep/{C0..C8}/dlacat-*.fits + processed/*.h5 + pc_snr2_pdla99.md`
- `cellD_knob_sweep/{D0..D8}/dlacat-*.fits + processed/*.h5 + pc_snr2_pdla99.md`
- `cellC_knob_sweep/C0/pc_conf_{0p999,...,0p9999999}/` — p_DLA cut sweep
- `cellD_knob_sweep/D0/pc_conf_{0p999,...,0p9999999}/` — p_DLA cut sweep
- `_chain_sweeps.log` — chain progress + ALL_BATCHES_DONE marker

## Compute env state

- jupyter `52973367` on `nid004315`, 256 CPUs, 503 GiB RAM
- expires `2026-05-15T00:08:32 PDT`
- regular sbatch queue still 10+ days deep — do NOT submit large sbatch
  jobs from here

---

# Handoff — 2026-05-14 11:20 PT (jupyter 52950547 on nid004213)

> Big day. Full PR #7 cleanup, two completed knob sweeps (2-way cellC +
> 3-way D-sweep), two background investigation agents that surfaced **a
> real bug** in `dla_gp.py` and a **3pp-purity tuning knob**. **No commits
> made today** — everything is uncommitted on `production_533` so the
> human can review the bug + sweep findings together.
>
> **EVENING UPDATE — D-sweep landed**. Two cells **Pareto-dominate the
> 3-way production baseline**: D1 (MAX_DLAS=4) at P=0.862 / C=0.769, and
> D4 (NHI [19, 23]) at P=0.855 / C=0.775. The D-sweep shows the OPPOSITE
> asymmetry from C-sweep: 3-way *rewards* MAX_DLAS↑ and NHI-prior↑, 2-way
> *penalizes* both. Mechanism: 3-way's separate SubDLA channel keeps the
> k-DLA denominator clean. Decision report:
> `docs/notes/2026-05-14_2way_vs_3way_decision.md` — recommends **3-way +
> D1+D4 stack** as production default with **cellC + NHI gate** as a
> documented opt-in. D9 stack test (D1+D4 combined, predicted P≈0.872 /
> C≈0.778) and Saclay/2LPT cross-val are the remaining experiments to
> finalize the recommendation.

## TL;DR for next-Claude

1. **PR #7 tasks 3, 5, 7 are DONE** (uncommitted): runbook §3.6 documents the
   FILTER=1 knobs, sub-DLA P/C 2×3 table on cellA/B/C is in
   `docs/notes/2026-05-14_subdla_pc_cellabc.md`, knob-tuning doc has a
   "refuted by 2×2" callout. Tasks 1 and 2 were already in `2e3642b`.

2. **CellC knob sweep (C1-C8)** ran on London-0 5k. Headline: **no knob
   Pareto-dominates cellC baseline** at the canonical operating point (P=0.83,
   C=0.83). Closest contender is PW 80k. Report:
   `docs/notes/2026-05-14_cellC_knob_sweep.md`.

3. **CRITICAL BUG SURFACED — `dla_gp.py:212-214 + 794-798/816/837-839`**.
   Per-sample log-likelihood is pre-divided by `-log(num_dla_samples)` at
   line 212-214, then the FILTER fix #5 / truncated / standard paths
   recompute log-evidence with `+ log(np.nanmean(...))` — the combination
   carries an extra `-log(N)` factor that biases 1-DLA log evidence DOWN as
   N grows. The C-sweep "more samples → worse" trend is the artifact of
   this bias interacting with the `p_DLA ≥ 0.99` threshold cut. Numerical
   falsification (background agent): predicted Δ = ±log(N_cell/N_baseline)
   matches observed to <1%. **Fix needed before any sample-count knob can be
   trusted.** Doesn't invalidate cellC vs production-baseline comparison
   (both N=50k).

4. **Top tuning knob found — NHI consistency gate** at the eval layer.
   `NHI_pred - 0.5·NHI_ERR ≥ 20.3` lifts cellC headline P 0.826 → **0.856
   (+3.0 pp)** at C 0.830 → 0.798 (−3.2 pp). Tested at k ∈ {0, 0.25, 0.5,
   0.75, 1.0}; clean Pareto sweep. 84% of the FPs in [20.3, 20.5) are
   sub-DLA bleed (true sub-DLAs whose MAP NHI scatters above 20.3 in the
   2-way model), which the gate excises. Re-eval recipe lives in the
   purity-diagnostic findings (see "Agent results" below). NOT yet
   incorporated into `examples/molly_faithful_pc_plots.py` as a real flag.

5. **D-sweep (3-way analog of cellC) DONE.** 8 cells × 5k London-0,
   aggregate cost 2.77 nh. Two strict Pareto-dominators of baseline:
   - **D1 (MAX_DLAS=4)**: P=0.862 (+1.7 pp), C=0.769 (+0.3 pp)
   - **D4 (NHI [19, 23])**: P=0.855 (+1.0 pp), C=0.775 (+0.9 pp)
   D2 (MAX_DLAS=5) gives marginal +0.8 pp P. D5/D6/D7 (PW count cells)
   show the same `-log(N)` bias artifact as C5/C6/C7 — bias-suspect.
   Report: `docs/notes/2026-05-14_cellD_knob_sweep.md`. Decision report:
   `docs/notes/2026-05-14_2way_vs_3way_decision.md`.

6. **Compute env**: jupyter `52950547` on `nid004213`, 256 CPUs, 503 GiB
   RAM. ~190 GiB used (8 D cells + buff/cache). Healthy.

## Today's evidence flow

### CellC sweep (C1-C8) — done

8 OAT variations on cellC baseline, all on London-0 5k. Headline:

| Cell | Knob | P | C | wall_min | node_hr |
|---|---|---:|---:|---:|---:|
| **cellC baseline** | — | **0.8256** | **0.8304** | (n/a) | (n/a) |
| C1 | MAX_DLAS=4 | 0.7955 | 0.8187 | 64.5 | 0.269 |
| C2 | MAX_DLAS=5 | 0.8068 | 0.8304 | 79.8 | 0.332 |
| C3 | NHI [17.2, 23] | 0.8011 | 0.8246 | 73.8 | 0.307 |
| C4 | NHI [18, 22] | 0.7994 | 0.8275 | 70.2 | 0.293 |
| C5 | PW 30k | 0.8103 | 0.8246 | 57.3 | 0.239 |
| C6 | PW 80k | 0.8179 | 0.8275 | 80.5 | 0.335 |
| C7 | PW 100k | 0.8174 | 0.8246 | 88.8 | 0.370 |
| C8 | n_init=10k | 0.8069 | 0.8187 | 76.2 | 0.318 |

Aggregate cost: 2.46 nh for all 8 × 5k. Outputs at
`/pscratch/sd/j/jibancat/prod533_5k_20260511/cellC_knob_sweep/`.

### NHI gate sweep on cellC — done

Eval-only re-run on cellC baseline HDF5. Gate:
`NHI_pred - k·NHI_ERR ≥ 20.3`.

| k | P | C | n_cat | TP |
|---:|---:|---:|---:|---:|
| 0 (baseline) | 0.8256 | 0.8304 | 467 | 340 |
| 0.25 | 0.8369 | 0.8099 | 452 | 332 |
| **0.5** | **0.8558** | 0.7982 | 438 | 327 |
| 0.75 | 0.8706 | 0.7865 | 425 | 322 |
| 1.0 | 0.8783 | 0.7807 | 420 | 320 |

Outputs at `/pscratch/sd/j/jibancat/prod533_5k_20260511/cellC_knob_sweep/_nhi_gate_k*/`.
Each step of k trades ~+1-2 pp P for ~−1 pp C — clean Pareto frontier. The
real CellC operating point can now be picked anywhere on the curve depending
on whether the deliverable wants to hit 85% purity (k ≈ 0.5) or stay
balanced near the original 83/83 point (k = 0).

### Sub-DLA P/C 2×3 — done (PR #7 task 5)

`docs/notes/2026-05-14_subdla_pc_cellabc.md`: cellA/B/C × {DLA, sub-DLA}
operating point. cellC wins on sub-DLA purity (0.43 vs 0.28-0.26) but all
three cells are far from the 85/70 sub-DLA aim — the 2-way model architecture
fundamentally constrains sub-DLA performance.

### D-sweep (3-way analog of cellC) — DONE

8 cells, mirror of C1-C8 on the 3-way production baseline (SINGLE_ABS=0,
NHI [19,22], PW 50k, SubDLA 10k, FILTER=1, τ-EB=on, MAX_DLAS=3 baseline).

| Cell | Knob | P | C | Δ P | Δ C |
|---|---|---:|---:|---:|---:|
| **3-way baseline** | — | 0.8452 | 0.7661 | ref | ref |
| **D1** | **MAX_DLAS=4** | **0.8623** | **0.7690** | **+1.7** | **+0.3** |
| D2 | MAX_DLAS=5 | 0.8534 | 0.7661 | +0.8 | 0.0 |
| D3 | NHI [17.2, 22] | 0.8552 | 0.7427 | +1.0 | −2.3 |
| **D4** | **NHI [19, 23]** | **0.8548** | **0.7749** | **+1.0** | **+0.9** |
| D5 | PW 30k *(bias-suspect)* | 0.8365 | 0.7778 | −0.9 | +1.2 |
| D6 | PW 80k *(bias-suspect)* | 0.8675 | 0.7661 | +2.2 | 0.0 |
| D7 | PW 100k *(bias-suspect)* | 0.8741 | 0.7515 | +2.9 | −1.5 |
| D8 | n_initial=10k | 0.8457 | 0.7690 | +0.05 | +0.3 |

Aggregate cost: **2.77 nh** for all 8 × 5k. **Bold = strict
Pareto-dominators.** `n_truth = 618` for every row.

Outputs: `/pscratch/sd/j/jibancat/prod533_5k_20260511/cellD_knob_sweep/`.

**Note on the asymmetry vs C-sweep**: cells with the same knob choices
behave OPPOSITELY in 2-way vs 3-way:

| Knob | 2-way (cellC) | 3-way (D-sweep) |
|---|---|---|
| MAX_DLAS=4 | C1: −3.0 / −1.2 (HURTS) | D1: +1.7 / +0.3 (PARETO WIN) |
| Wider NHI prior up | C3: −2.4 / −0.6 (HURTS) | D4: +1.0 / +0.9 (PARETO WIN) |

3-way's separate SubDLA channel acts as a "junk drawer" for low-NHI
evidence, keeping the k-DLA channels clean as capacity grows. 2-way
funnels everything into one absorber posterior, so extra capacity inflates
noise.

D9 (D1+D4 stack: MAX_DLAS=4 + NHI [19, 23]) NOT yet tested but predicted
linear-additive: **P ≈ 0.872, C ≈ 0.778** at ~0.36 nh per 5k. That is the
recommended next experiment.

### 2-way vs 3-way decision — DONE

Report: `docs/notes/2026-05-14_2way_vs_3way_decision.md`.

**Recommendation**: ship the **3-way + D1+D4 stack** as production
default, **cellC + NHI gate at k=0.5** as documented opt-in.

Frontiers (London-0 5k):
- 2-way: cellC + gate sweeps from (0.826/0.830) → (0.878/0.781) along
  k = 0..1.0
- 3-way: baseline (0.845/0.766) → D1 (0.862/0.769) → D4 (0.855/0.775)
  → D9 predicted (0.872/0.778)

At C ≥ 0.78, **2-way frontier dominates** on classical-DLA P/C. At
C < 0.78 the two are roughly Pareto-equivalent. Decision tips on
structural grounds: 3-way preserves separate `P(SubDLA)` channel, 84 %
of cellC's [20.3, 20.5) FPs are sub-DLA bleed (a structural cost the
gate can patch but not eliminate), 3-way rewards capacity knobs. cellC
NOT yet validated on Saclay/2LPT; 3-way IS (Saclay 0.871/0.771 per
2026-05-12).

## Background agent results

### Agent 1 — bug review (DONE)

**Verdict: real bug**, falsified by predicted vs observed Δlog_evidence
matching `-log(N)` to <1%. Locations:
- `gpy_dla_detection/dla_gp.py:212-214` — pre-subtracts `-log(num_dla_samples)`
  from every per-sample log-likelihood
- `gpy_dla_detection/dla_gp.py:794-798` (FILTER fix #5 1-DLA branch),
  `:816` (truncated path), `:837-839` (standard path) — all use
  `np.log(np.nanmean(initial_probs))` which double-counts the 1/N factor

Effect: 1-DLA log evidence biased DOWN by `log(N)`. Bigger N → lower p_DLA
→ fewer cat rows → higher P, lower C in lockstep with `log(N)`.

**Secondary finding**: legacy 50k file (`pw_samples_a3_172_220_50000.mat`,
Dec 29) was generated with a different QMC seed than my new 30k/80k/100k
files (today). The new files form a Halton prefix chain (30k ⊂ 80k ⊂ 100k
when seed=42), but the legacy 50k is independent. Adds ~0.007 dispersion on
top of the systematic bias. **If we patch the bug, also regenerate the 50k
file with the new seed=42 to make C5-C7 strictly nested.**

**Doesn't invalidate**:
- cellC vs production-baseline comparison (both N=50k → bias cancels)
- The cellC mechanism verdict (posterior-arithmetic stands)
- The Var[Δ_marg] verdict (still consistent with statistic-limited regime
  AFTER the bias is removed; in fact, removing the bias would make C5-C7
  even flatter)

**Settles**: regenerate 50k file, patch the `mean()` to `sum()/n_initial` (or
add the missing `+ log(N)` correction), re-run C5-C7. Predicted: P/C across
N becomes flat within 1σ.

### Agent 2 — [20.3, 20.5) purity diagnostic (DONE)

**Mechanism**: 84% of FPs in cellC's [20.3, 20.5) bin are **sub-DLA bleed**
(true sub-DLAs with NHI just below 20.3 whose MAP NHI scatters above
because of finite Voigt-fit uncertainty). 13% are pure noise (no truth in
spectrum), 3% Lyβ misIDs. The mechanism is the SYMMETRIC COST of cellC's
completeness win: the same posterior-arithmetic that pushes weak true DLAs
over p_dla=0.99 also pushes weak true sub-DLAs over.

**FP characteristics**:
- Median NHI_ERR for FPs = 0.101 dex; for TPs = 0.063 dex
- 90% of FPs have P_DLA = 1.0 to 5 places (raising gp-conf hits a floor)
- Median MODEL_P (per-DLA): FPs = 0.66, TPs = 0.01 (TPs are spectra with
  multi-DLA evidence split — MODEL_P cut hurts more than helps)

**Top knob — NHI consistency gate** (verified empirically above): clean
Pareto frontier, costs roughly 1pp C per 1-2pp P. Best operating points:
- k=0 (baseline): 0.83 / 0.83
- k=0.5: 0.86 / 0.80 — closest to 85/85 with balanced trade-off
- k=1.0: 0.88 / 0.78 — purity-first if downstream tolerates C loss

The gate generalizes across the cellC family (C5/C7/C8 all show similar
+5-10pp bin-P gains). Implementation: not yet a flag in
`examples/molly_faithful_pc_plots.py` — currently a 4-line monkey-patch
in the eval invocation. Should add `--nhi-consistency-k` flag if we want
to ship it.

## Files written today (all UNCOMMITTED)

| File | Purpose |
|---|---|
| `docs/production_runbook.md` (§3.6 added) | FILTER=1 knob docs (PR #7 task 3) |
| `docs/notes/2026-05-13_filter1_knob_tuning.md` (top callout added) | Refuted-by-2×2 (PR #7 task 7) |
| `docs/notes/2026-05-14_subdla_pc_cellabc.md` | Sub-DLA 2×3 (PR #7 task 5) |
| `docs/notes/2026-05-14_cellC_knob_sweep.md` | C-sweep report |
| `docs/notes/2026-05-14_cellD_knob_sweep.md` | D-sweep report |
| `docs/notes/2026-05-14_2way_vs_3way_decision.md` | Head-to-head decision |
| `tools/research/subdla_pc_eval.py` | Sub-DLA P/C wrapper |
| `slurm/run_local.sh` | `--filter_n_initial_floor` plumbing (8-line addition) |
| `data/dr12q/processed/pw_samples_a3_172_230_50000.mat` | New QMC: NHI [17.2, 23] |
| `data/dr12q/processed/pw_samples_a3_172_220_30000.mat` | New QMC: 30k @ [17.2, 22] |
| `data/dr12q/processed/pw_samples_a3_172_220_80000.mat` | New QMC: 80k @ [17.2, 22] |
| `data/dr12q/processed/pw_samples_a3_172_220_100000.mat` | New QMC: 100k @ [17.2, 22] |
| `data/dr12q/processed/pw_samples_a3_190_220_30000.mat` | New QMC: 30k @ [19, 22] (D-sweep) |
| `data/dr12q/processed/pw_samples_a3_190_220_80000.mat` | New QMC: 80k @ [19, 22] (D-sweep) |

Off-repo (in /pscratch):
- `cellC_knob_sweep/` (configs/C{1..8}.env, _launch.sh, _eval_and_aggregate.sh,
  _nhi_bin_table.py, C{1..8}/, HEADLINE.tsv, nhi_bins.tsv,
  _nhi_gate_k{0,0.25,0.5,0.75,1.0}/)
- `cellD_knob_sweep/` (configs/D{1..8}.env, _launch.sh, _eval_and_aggregate.sh,
  _nhi_bin_table.py, D{1..8}/ — populating)
- `joint_dla_subdla_sweep/subdla_pc_2x3.tsv` (sub-DLA results)

## Pickup priorities for next-Claude

### Priority 1 — Decide on the bug

The dla_gp.py log-evidence normalization affects all production runs that
have ever been committed. Two possible paths:

(a) **Patch + re-run**: subtract `-log(N)` from the FILTER fix #5 / truncated
/ standard branches (or replace `np.log(np.nanmean(...))` with
`np.log(np.nansum(...)) - np.log(n_in_branch)` consistently). Then re-run
the C5/C6/C7 cells to confirm the bias trend disappears. Predicted: all
PW-count cells fall within 1σ of cellC baseline.

(b) **Document and flag**: leave the code as-is (since the bug doesn't
affect production runs at fixed N=50k), add a code comment + runbook note
that NUM_DLA_SAMPLES variations have a systematic bias, and record this
in the model's "known issues" section.

The user previously expressed concern that production decisions might be
based on biased numbers. (a) is the safer path. Cost: re-run C5/C6/C7
≈ 1.5 nh + a regenerated 50k sample file with consistent seed.

### Priority 2 — D9 stack test (D1 + D4 combined)

Combine MAX_DLAS=4 + NHI prior [19, 23] into one cell. Predicted (linear
additive): P ≈ 0.872, C ≈ 0.778. Cost ~0.36 nh per 5k London-0. If the
prediction holds (or beats it), D9 becomes the production-default 3-way
config.

Config skeleton: copy `cellD_knob_sweep/configs/D1.env`, change
`DLA_SAMPLES_FILE` to `pw_samples_a3_190_230_50000.mat` (from D4).

### Priority 3 — D-sweep eval & 2-way vs 3-way decision — DONE

Reports already written:
- `docs/notes/2026-05-14_cellD_knob_sweep.md`
- `docs/notes/2026-05-14_2way_vs_3way_decision.md`

### Priority 4 — Saclay/2LPT cross-validation (PR #7 task 6)

Once the 2-way vs 3-way decision is made, run the chosen winner on Saclay
mock-0 + 2LPT mocks to verify off-distribution generalization. ~3-4 nh per
mock × 5k validation. Use existing slurm configs:
`slurm/configs/saclay0_y3.env`, `slurm/configs/2lpt_y3.env` (if exist).

### Priority 5 — NHI consistency gate as a real flag

Add `--nhi-consistency-k` to `examples/molly_faithful_pc_plots.py` (and
mirror it in `tools/research/subdla_pc_eval.py`). Default 0 (= current
behavior). Document in §7 of `docs/production_runbook.md`. The k=0.5
result should become a documented operating point.

### Priority 6 — Big commit

Once the bug + D-sweep + 2way-vs-3way decisions land, make ONE commit per
logical chunk:

1. Sub-DLA P/C tooling + report (`tools/research/subdla_pc_eval.py`,
   `docs/notes/2026-05-14_subdla_pc_cellabc.md`)
2. Runbook §3.6 + knob-tuning refutation callout
3. `slurm/run_local.sh` plumbing
4. New QMC sample files (`data/dr12q/processed/pw_samples_a3_*.mat`)
5. C-sweep + D-sweep reports + bug-finding note
6. (If patching the bug) the dla_gp.py fix as a separate clean commit

PR base branch is `desi_y3` per memory `project_base_branch`.

## Open questions for human

1. **Patch the dla_gp.py log-evidence bug** in this PR or as a follow-up?
   (Affects all sample-count knob comparisons; doesn't affect fixed-N
   production. The patch is ~3 lines.)
2. **Ship the NHI consistency gate at k=0.5** as a documented operating
   point? It cleanly approaches 85/85 but loses 3pp completeness.
3. **2-way vs 3-way default decision**: pending D-sweep results, but the
   sub-DLA bleed mechanism (84% of cellC's bin FPs) is already a strong
   argument for 3-way + post-hoc NHI gate as the headline catalog.

## Memory state

No new memory items needed today; existing memory items still load-bearing.
The bug finding could become a `feedback_dla_gp_log_evidence_bug` memory
once we decide whether to patch — for now it's documented in this handoff.

## Compute env state

- Jupyter: 52950547 on nid004213, 256 CPUs, 503 GiB RAM, expires unknown
  (check `scontrol show job 52950547`).
- Background waiter `bo5yeb3bk` listens for D-sweep completion, will fire
  per-cell + final all-done notifications.
- Two old-session waiters (`b4tqnr89c`, `bjdgpk2eh`) already completed.
- ~190 GiB RAM in use (8 D cells running + buff/cache).
- Regular sbatch queue: still 10+ days deep per yesterday's check; do NOT
  use sbatch for new work today.

---

# Handoff — 2026-05-13 21:30 PT (evening session, jupyter 52933605)

> **Evening session update (this top block)**: ran the FILTER=1 knob 2×2
> ablation, finished FILTER=0 to all 8 slices, wrote a `cellC` mechanism
> verdict, and built an aggregator P-vs-C plot. **Headline: the 2×2 ablation
> mostly refuted the knob-tuning hypothesis**. Knob 1 (`n_initial`=10k) gives
> only +0.6pp C vs baseline; knob 4 (empty-mask fall-through) is essentially
> a no-op. FILTER=0 (full 8 slices) gives +4pp C / −3pp P vs baseline — much
> smaller than the s1+s3-slice teaser (+12pp) implied. **Implication**:
> cellC's headline +7pp C / −2pp P stands as the best lever — and is *not*
> reproducible via FILTER=1 knob tuning. The completeness ceiling difference
> between FILTER=0 and FILTER=1 must come from something *other* than the
> coarse-scan-miss / early-stop mechanism in the original knob doc.
>
> See [`docs/notes/2026-05-13_cellC_mechanism_verdict.md`](docs/notes/2026-05-13_cellC_mechanism_verdict.md)
> for why cellC works (posterior-arithmetic dominance, not sample density).
> Uncommitted code changes in `desi-DLAGP.py`, `dlasearch.py`,
> `run_bayes_select.py`, `bayesian_model_selection.py`, `dla_gp.py`,
> `slurm/resume_local.sh` plumb two new flags `--filter_n_initial_floor`
> and `--filter_empty_mask_fallthrough`. Tests pass (80 pre-existing).
>
> **PR follow-ups for `production_533` (next session, in order)**:
>
> 1. Commit the knob-plumbing code (5 src files + resume_local.sh patch + 1
>    code-path fix for the empty-after-exclusion edge case in `dla_gp.py:691`).
> 2. Commit the cellC verdict note + this handoff update.
> 3. Add the `--filter_n_initial_floor` / `--filter_empty_mask_fallthrough`
>    documentation to the production runbook.
> 4. Decide: ship cellC as the default? Verdict note recommends "flag, not
>    default" pending Saclay + 2LPT + real-LOA cross-validation.
> 5. Add Sub-DLA P/C eval (Option B from `project_subdla_dla_joint_design`):
>    re-run `molly_faithful_pc_plots.py` with `--nhi-min 19.0` + truth filter
>    `NHI ∈ [19.0, 20.3]` on cellA/B/C to populate the (DLA, sub-DLA) ×
>    (cell) 2×3 table.
> 6. Cross-validate cellC on Saclay mock + 2LPT before promoting.
> 7. Drop the "knob 4 helps" hypothesis from `docs/notes/2026-05-13_filter1_knob_tuning.md`
>    or add a "refuted by 2×2 ablation" callout.
>
> ---

# Handoff — 2026-05-13 14:40 PT (jupyter 52907557 expires 15:44 PT)

> Big session. Var[Δ_marg] verdict landed, all 5 yesterday-in-flight runs
> resumed and finished, P/C eval done on all of them, 2-way joint-sweep
> cellC (single-absorber NHI [17.2, 22]) is a clear baseline-candidate
> winner, FILTER=1 knob tuning is the actionable lever to close the
> remaining completeness gap. 4 commits pushed to `production_533`
> (`c8ba76b..bb218c5`). PR #7 diff auto-updated.

## TL;DR for next-Claude

1. **Today's headline result**: cell C of the joint sweep
   (`SINGLE_ABSORBER_MODEL=1`, MAX_DLAS=3, NHI prior `[17.2, 22]`, PW 50k)
   gives **P=0.83 / C=0.83 at SNR>2, P_DLA≥0.99 vs dla_cat NHI≥20.3** —
   beats the FILTER=1 v3 baseline by **+7 pp completeness at only −2 pp
   purity**. This is the closest any tested config gets to balanced 85/85
   so far. Read [`docs/notes/2026-05-13_filter1_knob_tuning.md`](docs/notes/2026-05-13_filter1_knob_tuning.md)
   before deciding next steps.

2. **Var[Δ_marg] verdict (done today)**: pipeline is statistic-limited
   at production N=50k, not sampling-limited. σ_noise ≈ 0.1 vs signal
   gap ≈ 13. Drop bespoke MLMC / pocoMC for the SNR>2 ceiling. Read
   [`docs/notes/2026-05-13_var_delta_marg_diagnostic.md`](docs/notes/2026-05-13_var_delta_marg_diagnostic.md).

3. **FILTER=1 has tunable knobs** that should close the completeness
   gap vs FILTER=0 (which is the same Bayesian computation, slower). Per
   user reframe: don't switch to FILTER=0; tune FILTER=1. The
   knob doc [`docs/notes/2026-05-13_filter1_knob_tuning.md`](docs/notes/2026-05-13_filter1_knob_tuning.md)
   has a 2×2 ablation matrix ready to run.

4. **Compute environment is bad right now**: regular_milan_ss11 queue is
   **11 170 jobs deep**, sbatch start estimate is **2026-05-23** (10 days
   from now). All inference today ran inline on the jupyter compute node
   via `slurm/resume_local.sh` + `nohup` + `disown`. The session
   expires 15:44 PT — next compute should either be a fresh jupyter or
   `salloc -q interactive` (much faster queue).

5. **2 untouched workstreams continue elsewhere**: (a) trainer PR on
   GreatLakes is running the LOA-no-HCD-with-BAL retrain after PR #6
   fixed the v2 preload normalization bug; (b) PR #7 is still draft.

---

## Today's P/C results — single comparison table (SNR>2, P_DLA≥0.99, lyb-veto, no-BAL, full forest λ_rf∈[911, 1216])

| Variant | Purity | Completeness | cat rows | truth | vs baseline |
|---|---:|---:|---:|---:|---|
| **baseline** v3+PW14[19,22]+τ-EB+FILTER=1+md=3 | 0.8452 | 0.7661 | 1242 | 618 | reference |
| early_stop_A (no null-vs-current early-stop) | 0.8466 | 0.7749 | 1468 | 618 | +1pp C, more cat |
| early_stop_D (pre-Occam likelihood for null cmp) | 0.8466 | 0.7749 | 1399 | 618 | +1pp C, modest cat |
| NFL=31 (test trainer-mismatch hypothesis) | 0.8534 | 0.7661 | 1241 | 618 | **null** — reject |
| FILTER=0 (s1+s3 only, n_truth=54) | 0.8519 | **0.8846** | 144 | 54 | **+12pp C** (suggestive, undersized) |
| joint cellA: SINGLE_ABS=1, md=3, NHI[19,23], PW50k | 0.7906 | 0.8392 | 2668 | 618 | −5pp P, +8pp C |
| joint cellB: SINGLE_ABS=1, md=4, NHI[19,23], PW50k | 0.7950 | 0.8392 | 2912 | 618 | similar to A |
| **joint cellC**: SINGLE_ABS=1, md=3, NHI[17.2,22], LLS50k | **0.8256** | **0.8304** | 3268 | 618 | **−2pp P, +7pp C — best yet** |
| **FILTER=0** (full 8 slices, evening) | 0.8166 | 0.8070 | 1922 | 618 | **−2.9pp P, +4.1pp C** (small-sample teaser was misleading) |
| **k1=10k k4=off** (knob 1 only) | 0.8516 | 0.7719 | 1305 | 618 | +0.6pp P, +0.6pp C (≈ tie w/ baseline) |
| **k1=5k k4=on** (knob 4 only) | 0.8506 | 0.7661 | 1250 | 618 | +0.5pp P, **0pp C** (knob 4 is a no-op) |
| **k1=10k k4=on** (both knobs) | 0.8511 | 0.7690 | 1310 | 618 | +0.6pp P, +0.3pp C (knob 1 dominates) |

**Source**: `examples/molly_faithful_pc_plots.py` against London mock-0
`dla_cat.fits` (NHI≥20.3 truth). Per-variant log files:
`/pscratch/sd/j/jibancat/prod533_5k_20260511/resume_local_logs/pc_*.log`.

### How to interpret each row

- **early_stop A/D** (today's fix variants): Both lift completeness by
  ~1 pp without hurting purity much, at the cost of more catalog rows
  (more multi-DLA hypotheses survive). Marginal value. **Default
  EARLY_STOP_MODE=baseline still reasonable.**

- **NFL=31**: Tested whether NUM_FOREST_LINES mismatch with the
  trainer (user belief: trainer used 31) was the issue. Today's A1 agent
  verified the v3 GP was **actually trained at NFL=3** — so NFL=31 at
  inference is *more* lines than training. Result: indistinguishable
  from baseline. **Reject the hypothesis. Drop NFL=31.** The
  `submit_desi_{mock,loa}.sh` default of 31 is a latent bug.

- **FILTER=0** (only s1+s3 ran due to wall-time budget): +12 pp
  completeness in headline. Same-n_truth (54) comparison: 0.769 → 0.885
  C. Direction is consistent across all 4 P_DLA cuts. Statistical
  significance ~1.1 σ (n_truth=10 in [20.3, 20.6) bin). **Promising but
  needs full 8-slice confirmation.** See `docs/notes/2026-05-13_filter_nfl_confirmation.md`.

- **Joint sweep cells A/B/C** (today's joint catalog test — Option B
  from `project_subdla_dla_joint_design` memory): single DLA model
  with widened NHI prior. All three improve completeness +7-8 pp. **Cell
  C ([17.2, 22] prior, the LLS-extended one) wins on purity-completeness
  balance.** The [19, 23] prior (cells A/B) hurts purity more (−5 pp).
  These were run with `SINGLE_ABSORBER_MODEL=1` so the model is 2-way
  [null, k-DLA-with-widened-prior].

### NHI-bin-stratified completeness — cellC vs baseline

Computed via `_nhi_bin_compare.py` (which reuses molly_faithful helpers).
Same operating point: SNR>2, P_DLA≥0.99, lyb-veto, no-BAL, λ_rf ∈ [911, 1216].
Counts here are post-all-cuts (the molly `n_*_post_cuts` are pre-P_DLA-cut,
so they're larger; the per-bin C ratios are correct either way).

**Completeness per truth-NHI bin:**

| Bin | baseline | cellC [17.2, 22] | Δ |
|---|---:|---:|---:|
| [20.3, 20.5) | 62/108 = **0.574** | 76/108 = **0.704** | **+0.130** |
| [20.5, 21.0) | 129/158 = 0.816 | 138/158 = 0.873 | +0.057 |
| [21.0, 21.5) | 58/62  = 0.935 | 57/62  = 0.919 | −0.016 |
| [21.5, 22.0) | 13/14  = 0.929 | 13/14  = 0.929 | 0.000  |
| **overall**  | 262/342 = **0.766** | 284/342 = **0.830** | **+0.064** |

**Purity per predicted-NHI bin** (using cat NHI for the bin assignment):

| Bin | baseline | cellC [17.2, 22] | Δ |
|---|---:|---:|---:|
| [20.3, 20.5) | 42/72  = 0.583 | 57/95  = 0.600 | +0.017 |
| [20.5, 21.0) | 136/146 = 0.932 | 139/152 = 0.914 | −0.018 |
| [21.0, 21.5) | 63/70  = 0.900 | 66/74  = 0.892 | −0.008 |
| [21.5, 22.0) | 21/22  = 0.955 | 22/23  = 0.957 | +0.002 |
| **overall**  | 262/310 = **0.845** | 284/344 = **0.826** | **−0.019** |

**The cellC win is concentrated almost entirely in the [20.3, 20.5)
regression bin** (+13 pp completeness). Mid-NHI [20.5, 21.0) picks up
another +6 pp. Strong DLAs (NHI ≥ 21.0) are flat — those weren't broken.
Purity drops a uniform ~2 pp because the wider NHI prior +
`single_absorber_model=1` admits more cat candidates per spectrum; some
are spurious. Notably the weakest bin's purity actually *rises* slightly
(0.58 → 0.60).

**Why this maps onto the FILTER=1 knob-tuning story**: cellC uses the
SAME FILTER=1 algorithm with a wider NHI prior `[17.2, 22]` plus
`single_absorber_model=1`. The wider prior gives the FILTER=1 coarse
5000-sample scan a better chance of finding a sample near a weak truth's
high-likelihood mode → fewer spectra hit the "early-stop on empty
valid_mask" branch (`dla_gp.py:635`). This is mechanistically the same
fix as knob 1 (`n_initial` floor) and knob 4 (empty-mask fall-through)
in [`docs/notes/2026-05-13_filter1_knob_tuning.md`](docs/notes/2026-05-13_filter1_knob_tuning.md).
Cell C achieves the fix "for free" by widening the prior support; tuning
the FILTER=1 knobs directly should achieve the same [20.3, 20.5) recovery
**without** the −2 pp purity hit, because the knob tuning fixes the
coarse-scan miss without changing the prior support.

The script `_nhi_bin_compare.py` at the repo root is the analysis tool
that produced these tables. Untracked (matches the `_*.py` scratch
convention from earlier sessions). Re-run with the same DESI env preamble.

Other variants' per-bin completeness in [20.3, 20.5) for reference:
- early_stop_A: 0.602 (modest +3 pp vs baseline)
- early_stop_D: 0.593 (modest +2 pp)
- NFL=31: 0.574 (=baseline, no effect — confirms NFL is irrelevant)
- FILTER=0 (s1+s3 only, n_truth=6 in this bin): 0.833 (suggestive, undersized)
- cellA [19,23] md3: 0.722
- cellB [19,23] md4: 0.694

---

## What was committed today

| Commit | Title |
|---|---|
| `2c499a8` | feat: EARLY_STOP_MODE flag for multi-DLA inference + resume scripts |
| `86ad225` | diag: Var[Δ_marg] gating diagnostic + 2026-05-13 verdict note |
| `f08d63b` | docs: 2026-05-13 handoff — Var[Δ_marg] verdict, resume status, lessons |
| `c8ba76b` | docs: production runbook + model-side improvements suggestions |
| `bb218c5` | docs: runbook fixes + BAL scope + FILTER/NFL confirmation + model-side updates |

This handoff will be the 6th commit (push at end of session).

---

## Pickup priorities for next-Claude

### Priority 1 — Run the FILTER=1 knob 2×2 ablation

Read [`docs/notes/2026-05-13_filter1_knob_tuning.md`](docs/notes/2026-05-13_filter1_knob_tuning.md).
The 4-cell ablation:
- (n_initial floor, empty-mask fall-through) ∈ {(5000, no), (10000, no), (5000, yes), (10000, yes)}
- Goal: identify whether the FILTER=1 completeness gap is from coarse-scan miss
  (knob 1), early-stop on empty mask (knob 4), or both.
- Cost: 4 × 30 min wall in parallel = 30 min on one 256-CPU node.
- Code changes needed before running: parametrize `n_initial` and add a
  fall-through branch at `dla_gp.py:635`. Touches `dla_gp.py`,
  `run_bayes_select.py`, `desi-DLAGP.py`, `slurm/run_local.sh`.

This is the most direct path to closing the completeness regression
without paying FILTER=0's 2.4× cost.

### Priority 2 — Validate cellC as the new production baseline

Cell C looks like the best operating point in today's data (0.83/0.83
balanced). Before claiming it:
1. Verify on full London 26k (today was just 5k 8f).
2. Verify on Saclay and 2LPT (today was London-only).
3. Verify on real LOA (today was mock-only).
4. NHI-bin-stratified completeness — esp. [20.3, 20.6) — make sure cell C
   doesn't have a different pathology than the v3 baseline.

If cellC validates at full scale, it becomes a **strong candidate** to
ship as the production DLA catalog. Note: cellC's catalog row count is
3268 (vs baseline 1242), so the post-cut work is harder (more candidate
DLAs to filter). The lyb_veto + P_DLA≥0.99 already handle this in the
P/C eval; production catalog work needs to follow the same recipe.

### Priority 3 — Sub-DLA P/C eval on cellA/B/C against truth

Today's cellA/B/C P/C was at NHI ≥ 20.3 (classical DLAs). The whole
*point* of widening the NHI prior to [19, 22] or [17.2, 22] is to ALSO
get sub-DLA detection. Need to:
1. Re-run `examples/molly_faithful_pc_plots.py` with `--truth-nhi-min 19.0`
   `--nhi-min 19.0` and a truth filter `NHI ≤ 20.3` on each cell.
2. Compare against the LLS truth catalog `hcd_truth_cat.fits` (Saclay) or
   filtered `dla_cat.fits` (London).
3. Report the 2×3 table: (classical DLA, sub-DLA) × (cellA, cellB, cellC).

This is the "Option B" deliverable from `project_subdla_dla_joint_design`
memory.

### Priority 4 — Wait for trainer-PR retrain to land + retrain validation

The user has a retrain running on GreatLakes (`loa_no_hcd_with_bal` model
+ PR #6 v2-preload-bug fix). When it lands:
1. Convert the trained .h5 to inference format (`null_gp_test/converted/`
   pattern, see existing v3 conversion).
2. Run the 5k London smoke comparison: new-trained-model vs v3 model.
3. P/C delta at the canonical operating point.

### Lower priority / parked

- **BAL masking smoke test** ([`docs/notes/2026-05-13_bal_masking_scope.md`](docs/notes/2026-05-13_bal_masking_scope.md)):
  4-line config + 30 min run, falsifies whether `--balmask` lets us
  recover BAL QSOs in the catalog. Worth running but not blocking.
  Also fix `constants.bal_lines` duplicate-CIII bug (line 85 vs 95).
- **K-rank sweep** (Tier 1.2 in
  [`docs/notes/2026-05-13_model_side_improvements.md`](docs/notes/2026-05-13_model_side_improvements.md)):
  needs trainer-side action on GreatLakes after PR #6 merges.
- **Sub-DLA prior overlap test** (Tier 2.0): does `[19, 20.3]` sub-DLA
  prior improve DLA detection? Open question, untested.

---

## Compute env — what's running, what's expired, what's queue-bound

- **Current jupyter**: job 52907557 on `nid004210` → wait, that's wrong;
  it's `nid004179` per `scontrol show job 52907557`. 256 CPUs, 487 GB
  mem, expires **15:44:40 PT** (= ~1 hr from this handoff write time).
- **Background procs**: should be quiet now — all P/C eval done, all 5
  resume runs finished, all 2 confirmation runs finished. Verify via
  `pgrep -af desi-DLAGP.py | wc -l` (expect 0 or near-0).
- **Regular sbatch queue**: 11 170 jobs pending, start estimate 10 days
  out. **Do not submit large sbatch jobs today** — won't run.
- **Interactive queue** (`-q interactive`): much shorter wait. For
  next-session compute, prefer salloc on interactive over regular queue.
- **GreatLakes**: separate cluster, user's trainer PR is running there.

---

## File-by-file summary of today's commits

- `2c499a8`: EARLY_STOP_MODE plumbing (5 files in core + 2 new scripts in
  slurm/). Default "baseline" → bit-for-bit production unchanged.
- `86ad225`: Var[Δ_marg] re-analysis script + 2026-05-13 verdict note.
  ~70 s wall to reproduce.
- `f08d63b`: this morning's first handoff.
- `c8ba76b`: production runbook v1 + model-side improvements v1. Stale
  for hours — superseded by `bb218c5`.
- `bb218c5`: runbook corrections (regression callout, NUM_FOREST_LINES
  verified-trained-at-3, sub-DLA table, [19,22] prior verified, node-hours
  re-validated, 2-way subDLA mode documented), BAL scope, FILTER+NFL
  confirmation note, model-side improvements v2.

**Uncommitted at handoff write time**: this `HANDOFF.md` itself, the new
`docs/notes/2026-05-13_filter1_knob_tuning.md`. Will commit + push at the
end of the session.

---

## Memory state (per `/global/homes/j/jibancat/.claude/.../memory/MEMORY.md`)

No new memory items added today; existing memory items still load-bearing:
- `feedback-long-runs-need-sbatch` (updated this morning: 3 rules; the
  queue-vs-jupyter exception was vindicated today).
- `project-base-branch` (`desi_y3`).
- `feedback-snr-canonical` (SNR_RED > 2).
- `project-prior-dilution-finding` (now substantially refuted by today's
  Var[Δ_marg] verdict — should add a "superseded by 2026-05-13_var_delta_marg_diagnostic"
  note next session).
- `project-subdla-dla-joint-design` (today's cellC result corroborates).

---

## Key files to read in this order on next session

1. `HANDOFF.md` (this file)
2. `docs/notes/2026-05-13_filter1_knob_tuning.md` (the actionable knob doc)
3. `docs/notes/2026-05-13_var_delta_marg_diagnostic.md` (the verdict that redirects sampler work)
4. `docs/notes/2026-05-13_filter_nfl_confirmation.md` (today's confirmation runs)
5. `docs/production_runbook.md` (full production runbook with today's corrections)
6. `docs/notes/2026-05-13_model_side_improvements.md` (trainer PR roadmap)

---

## Open questions for human at next session

1. **Cell C ([17.2, 22] single-absorber-mode) as the new production
   baseline?** Today's evidence (0.83/0.83 vs baseline 0.85/0.77)
   strongly favors it. Validate on full London 26k + Saclay + 2LPT + real
   LOA before claiming.
2. ~~**FILTER=1 knob 2×2**: which knob actually matters? Run today's
   ablation matrix to find out.~~ **Done (evening session)**: knob 1 alone
   gives ~+0.6pp C, knob 4 is a no-op. **The 2×2 refuted the knob-tuning
   hypothesis as the dominant explanation for the FILTER=0 / FILTER=1 gap.**
   See evening-session top block + `aggregate_pc_scatter.py` plot.
3. **Sub-DLA P/C from cellA/B/C** — what's the [19, 20.3] truth match
   look like? This is the "is cellC also a usable sub-DLA catalog"
   question.
4. **EARLY_STOP_MODE A or D for production**? Today's data says +1 pp C
   for both, modest. Probably defer until cellC / FILTER=1 tuning lands.
5. **What IS the FILTER=0 vs FILTER=1 completeness gap if not knob 1/4?**
   The 2×2 cells leave a ~3pp completeness ceiling between FILTER=1 (best:
   k1=10k k4=off, C=0.772) and FILTER=0 (C=0.807). Mechanisms not yet
   tested: the `z_tol=0.02` knob 2, the `null_evidence` threshold knob 3,
   or the truncated-correction estimator (knob 5) at multi-DLA branches.

## Evening-session P/C table (canonical molly_faithful eval, SNR>2, P_DLA≥0.99, lyb-veto, no-BAL, λ_rf∈[911,1216], NHI≥20.3)

| Variant | Purity | Completeness | cat rows post-cuts | truth |
|---|---:|---:|---:|---:|
| baseline FILTER=1 (reference) | 0.8452 | 0.7661 | 1242 | 618 |
| FILTER=0 (this session, full 8 slices) | 0.8166 | **0.8070** | 1922 | 618 |
| k1=5k k4=on  (knob 4 only) | 0.8506 | 0.7661 | 1250 | 618 |
| k1=10k k4=off (knob 1 only — BEST knob) | **0.8516** | 0.7719 | 1305 | 618 |
| k1=10k k4=on (both knobs) | 0.8511 | 0.7690 | 1310 | 618 |
| cellC (for comparison) | 0.8256 | 0.8304 | 3268 | 618 |

Per-variant per-slice tables: each variant's `pc_snr2_pdla99.md` in its outdir.

## Node-hour cost — FILTER=0 vs FILTER=1

8-slice wall time (one slice processes ~600 spectra, 8 workers per slice):

| Variant | Min/slice (avg) | Min/slice (max) | Total compute (8 slices × 8 workers) |
|---|---:|---:|---:|
| baseline FILTER=1 (ran solo) | 27.6 min | 38.5 min | ~29 core-hours |
| **FILTER=0** (full, contention) | **102.5 min** | 138 min | **~109 core-hours** |
| k1=5k k4=on (contention) | 87.3 min | 112 min | ~93 core-hours |
| k1=10k k4=off (contention) | 56.9 min | 77 min | ~61 core-hours |
| k1=10k k4=on (contention) | 94.6 min | 119 min | ~101 core-hours |

**Caveats** (important — read before quoting numbers):
- Baseline FILTER=1 ran solo on the node. The 4 new variants ran *concurrently*
  (3 cells × 8 slices × 8 workers + FILTER=0 6 slices = ~240 cores active on
  256-core node). The 2×2 wall times are contention-inflated by ~2-3×.
- Knob 1 = 10k cells (k1_10k_*) ran faster than knob 4 alone (k1_5k_k4_on)
  despite doing more coarse-scan work — counterintuitive. Likely a slice-by-slice
  contention artifact, NOT a real per-spectrum cost ordering.
- **Best estimate of true cost ratio FILTER=0 / FILTER=1** (factoring out
  contention) ≈ 2–3×. For a 1M-spectrum production run this is roughly
  **~50 node-hours extra** to move from FILTER=1 to FILTER=0 (the baseline
  was ~29 node-hours/5k → ~5800 node-hours/1M).

Re-running any single cell *solo* on a fresh jupyter would be the clean way to
get the per-cell base cost. ~30-40 min per cell. Not done this session.

## Files added/modified this evening (uncommitted)

| File | Purpose |
|---|---|
| `desi-DLAGP.py` | `--filter_n_initial_floor`, `--filter_empty_mask_fallthrough` CLI flags |
| `dlasearch.py` | Plumb new flags to `DLAHolder` (both `hpx` and `mock` ctors) |
| `run_bayes_select.py` | `DLAHolder.__init__` accepts + stores knobs; `_process_spectrum` propagates them; both `bayes.model_selection` calls pass them |
| `gpy_dla_detection/bayesian_model_selection.py` | `model_selection()` signature + both `parallel_log_model_evidences()` calls forward the knobs |
| `gpy_dla_detection/dla_gp.py` | `parallel_log_model_evidences()` accepts `filter_n_initial_floor` (knob 1, line 575) + `filter_empty_mask_fallthrough` (knob 4, line 639); also handles the **edge case** where the non-empty valid_mask becomes empty after `valid_mask[:n_initial]=False` (more common with knob 1 ≥ 10k) — falls through to FILTER fix #5 1-DLA-only evidence with a warning |
| `slurm/resume_local.sh` | Pass `FILTER_N_INITIAL_FLOOR` + `FILTER_EMPTY_MASK_FALLTHROUGH` env vars as CLI flags |
| `examples/aggregate_pc_scatter.py` | NEW — sweep P_DLA cuts across multiple variants → one scatter plot, modeled on Molly's notebook |
| `docs/notes/2026-05-13_cellC_mechanism_verdict.md` | NEW — 5-section verdict on why cellC works: posterior-arithmetic dominance over sample-density penalty; ship as flag not default |
| `HANDOFF.md` (this file) | Evening-session top block + open-questions update + node-hour table |

Outputs (all under `/pscratch/sd/j/jibancat/prod533_5k_20260511/`):

- `london_v3_loa124_pw14_tau_eb_filter0/` — full 8-slice FILTER=0 + `pc_snr2_pdla99.md`
- `filter1_knob_2x2/{k1_5k_k4_on,k1_10k_k4_off,k1_10k_k4_on}/` — 2×2 ablation cells, each 8 slices + `pc_snr2_pdla99.md`
- `figures/pc_scatter/purity_vs_completeness_all_variants.{png,tsv}` — aggregator output (11 variants)
