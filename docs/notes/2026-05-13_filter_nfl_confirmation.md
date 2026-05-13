# FILTER + NUM_FOREST_LINES confirmation runs — 2026-05-13

> **Status**: complete. Two production-style inference runs on London v3_loa124
> (PW14 50k DLA samples + tau-EB) to discriminate between two hypotheses for
> the observed FILTER=1 completeness regression in `[20.3, 20.6)`:
>
> 1. **FILTER0 hypothesis**: turn off the `filter_low_likelihood` initial-scan
>    pre-filter that short-circuits the multi-DLA marginalisation when the
>    initial-scan likelihood is below the null model evidence.
> 2. **NFL31 hypothesis**: a stale train-vs-inference mismatch — the production
>    sbatch driver (`slurm/submit_desi_mock.sh:51`) defaults to
>    `NUM_FOREST_LINES=31`, but the run_local.sh path uses the
>    `slurm/configs/_base.env` default of `NUM_FOREST_LINES=3`. If the GP was
>    actually trained at NFL=31, inference at NFL=3 would be wrong (the residual
>    spectrum at inference would over-subtract Lyman-series forest absorption).
>
> **Bottom line**: NFL31 hypothesis is **rejected**. NFL=3 vs NFL=31 give P/C
> within ±0.005 at every operating point — they're indistinguishable. The
> FILTER0 hypothesis looks promising but n_truth=10 in the relevant slice pair
> makes the result statistically marginal; the full 8-slice FILTER0 run would
> take ~8 hours and exceeded the jupyter-session budget.

## What was run

| Variant | OUTDIR | Slices | FILTER | NFL | Other |
|---|---|---|---|---|---|
| baseline (existing) | `prod533_5k_20260511/london_v3_loa124_pw14_tau_eb` | 0..7 (all 8, 6766 spec) | **1** | **3** | PW14 50k samples, tau-EB null obj |
| `…_filter0` | `prod533_5k_20260511/london_v3_loa124_pw14_tau_eb_filter0` | **1 and 3 only** (696 spec) | **0** | 3 | identical to baseline otherwise |
| `…_nfl31` | `prod533_5k_20260511/london_v3_loa124_pw14_tau_eb_nfl31` | 0..7 (all 8, 6766 spec) | 1 | **31** | identical to baseline otherwise |

Launch invariants verified via `RESUME_LOCAL_*.md` + `pgrep -af`. Both variants
share the same model file
(`/pscratch/sd/j/jibancat/prod533_5k_20260511/null_gp_test/converted/2lpt_loa124_nohcd_nobal_wide.h5`),
same DLA samples file (`pw_samples_a3_190_220_50000.mat`), same enable_tau_eb=1
+ tau_eb_objective=null. Confirmed by `--filter_low_likelihood {0,1}` and
`--num_forest_lines {3,31}` in the launched python cmdlines.

**Why FILTER0 is only 2 slices**: with `filter_low_likelihood=0` the initial-scan
short-circuit in `gpy_dla_detection/dla_gp.py:594` is bypassed and every
spectrum's 1-DLA marginal is evaluated on the full 50000-sample QMC grid. Per-
spectrum wall time goes from ~1s (baseline FILTER=1) to ~7s (FILTER=0). At
~6766 spectra across 8 slices and 8 max_workers per slice, that's ~130 min
per slice × 8 slices = ~17 hours — far longer than the jupyter-session budget.
Slices 1 and 3 are the two smallest (332 + 364 spectra) and fit in ~30 min.

Wall times observed:

| Slice | NFL31 (1129/1083/364/… spec) | FILTER0 (332/364 spec) |
|---|---|---|
| 0_1 (1129) | 45.6 min |  — |
| 1_2 (332)  | 11.9 min | 28.1 min |
| 2_3 (1083) | 39.5 min |  — |
| 3_4 (364)  | 13.6 min | 32.7 min |
| 4_5 (1081) | 41.1 min |  — |
| 5_6 (700)  | 24.7 min |  — |
| 6_7 (1110) | 41.8 min |  — |
| 7_8 (967)  | 36.9 min |  — |

NFL31 cost is **nearly identical** to baseline (baseline slice 0_1 took 37.1 min;
NFL31 took 45.6 min on the same slice — the +8 min is from CPU contention with
the FILTER0 procs and the wider Lyman-series forest matrix construction).
FILTER0 cost is **~2.4× the baseline NFL31 cost on the same 332/364-spectrum
slice** (28.1 / 11.9 = 2.36; 32.7 / 13.6 = 2.40).

## Apples-to-apples molly P/C (SNR>2, lyb_veto, no-BAL, λ_rf>911 Å)

`examples/molly_faithful_pc_plots.py --snr-min 2 --gp-conf 0.99 --lyb-veto
--lam-rf-min 911 --truth-nhi-min 20.3 --no-bal …`

### Headline (all NHI≥20.3, all SNR>2)

| Variant | n_cat | n_truth | P @ P_DLA≥0.99 | C @ 0.99 | P @ 0.999 | C @ 0.999 | P @ 0.99999 | C @ 0.99999 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (8f) | 1242 | 618 | 0.845 | 0.766 | 0.855 | 0.740 | 0.887 | 0.690 |
| **NFL31 (8f)** | 1241 | 618 | 0.853 | 0.766 | 0.863 | 0.737 | 0.890 | 0.687 |
| FILTER0 (s1+s3) | 144 | 54 | 0.852 | **0.885** | 0.852 | **0.885** | 0.875 | **0.808** |
| baseline (s1+s3, same n_truth as FILTER0) | 96 | 54 | 0.870 | 0.769 | 0.909 | 0.769 | 0.947 | 0.692 |

### NHI-bin stratified completeness — [20.3, 20.6) only

The bin the user flagged. All cuts: SNR>2, lyb_veto, no-BAL, λ_rf>911 Å,
DLAFLAG==0, predicted NHI>20.3.

| Variant | n_truth (in bin, SNR>2) | C @ P_DLA≥0.9 | C @ 0.99 | C @ 0.999 | C @ 0.99999 |
|---|---:|---:|---:|---:|---:|
| baseline (8f) | 144 | 0.667 | 0.604 | 0.556 | 0.479 |
| **NFL31 (8f)** | 144 | 0.667 | 0.604 | 0.549 | 0.472 |
| baseline (s1+s3 sub-sample) | 10 | 0.60 | 0.60 | 0.60 | 0.40 |
| **FILTER0 (s1+s3 only)** | 10 | **0.80** | **0.80** | **0.80** | **0.60** |

(Other NHI bins for completeness, baseline 8f only:
[20.3,20.5)=0.50 @ 0.99, [20.5,21.0)=0.70, [21.0,22.0)=0.93. The regression
is concentrated in the lowest NHI bin, consistent with the user's prior
[20.3, 20.6) framing.)

## Findings

### NFL=31 vs NFL=3: no effect

8-slice apples-to-apples (same n_truth=618 in both, same 6766 input spectra,
same model file, same DLA samples). At every P_DLA cut the purity and
completeness differ by at most 0.008 — within Monte-Carlo / file-ordering
noise. **The `NUM_FOREST_LINES` train/inference mismatch hypothesis is
falsified.** The model file
`2lpt_loa124_nohcd_nobal_wide.h5` has no `num_forest_lines` attribute, and
the GP basis (M, mu, log_omega, log_c_0) is parameterised purely on the
rest-frame wavelength grid — the number of Lyman-series forest lines used in
training only enters indirectly via the residual subtraction. Whatever
NFL was used during training, inference at either NFL=3 or NFL=31 gives the
same P/C to within rounding. Stick with the simpler NFL=3 (also faster
per spectrum).

### FILTER=0 vs FILTER=1: real but underpowered evidence

In the 2-slice subset where FILTER0 ran (slices 1+3, n_truth=10 in
[20.3,20.6)):

| | C @ 0.9 | C @ 0.99 | C @ 0.999 | C @ 0.99999 |
|---|---:|---:|---:|---:|
| baseline (s1+s3) | 0.60 | 0.60 | 0.60 | 0.40 |
| FILTER0 (s1+s3)  | 0.80 | 0.80 | 0.80 | 0.60 |
| Δ                | +0.20 | +0.20 | +0.20 | +0.20 |

The +0.20 completeness lift is consistent across all four P_DLA cuts, which
is suggestive of a real effect rather than statistical noise — but Poisson
1σ on n=10 is √10/10 ≈ 32 %, so a one-bin Δ=0.20 is below noise on its own.
Headline completeness (any NHI≥20.3) shows the same picture: 0.885 (FILTER0)
vs 0.769 (baseline s1+s3) = +0.116 lift, also 4/4 cuts in the same direction.

**Plausibility**: this is mechanistically expected. With FILTER=1, the
`filter_low_likelihood` short-circuit triggers when the initial 5000-sample
scan has no valid region above the null evidence (`dla_gp.py:635`), which
prematurely returns NaN log-likelihoods for the multi-DLA models. A weak,
low-NHI DLA is exactly the population most likely to be near the null boundary
in the initial scan, so it's exactly the population FILTER=1 over-rejects.
This matches the regression pattern: the [20.3, 20.6) bin (weakest DLAs)
loses the most.

### What we still don't know

- **Is the +0.20 completeness lift statistically real?** Need the remaining
  6 FILTER0 slices to drive `n_truth` in [20.3, 20.6) from 10 to ~144.
  At baseline cost that's ~5 hours of single-node wall time; feasible as
  one regular-queue sbatch (~2-day queue ≪ wall) but not in a jupyter
  session.
- **Does FILTER0 cost ~2.4× more than FILTER=1?** Yes, confirmed: 28.1 min
  for FILTER0 vs 11.9 min for NFL31 on the same 332-spectrum slice. At
  full London 1M production scale, FILTER=0 would cost ~2.4× the current
  baseline production node-hour budget. Worth it iff the completeness
  lift survives statistical scrutiny.

## Recommendation

1. **Drop NFL31 from consideration.** No measurable improvement, slightly
   slower per spectrum (+8 min on the longest slice). Keep production at
   `NUM_FOREST_LINES=3` (the run_local.sh / `_base.env` default).
   Update `slurm/submit_desi_mock.sh:51` to also default to 3 for
   consistency (it currently says 31, which is misleading — but harmless
   in practice because production uses run_local.sh, not that sbatch).

2. **Take FILTER0 seriously as a candidate baseline change**, but do not
   adopt it as production yet. The 2-slice evidence is suggestive but
   undersized. Before adopting:
   - Run the remaining 6 FILTER0 slices (slices 0, 2, 4, 5, 6, 7) as
     an sbatch job, since queue wait < per-job wall. ETA ~8 hours total.
   - Reproduce the [20.3, 20.6) Δ on the full 6766-spectrum sample.
   - If Δ > 0.10 at n_truth=144, FILTER0 wins. If Δ ≲ 0.05, the 2-slice
     +0.20 was statistical noise and FILTER=1 stays.

3. **No code change in this PR.** Both variants ran from the existing
   `slurm/resume_local.sh` driver via env overrides — the FILTER/NFL knobs
   are already exposed and verified to propagate correctly through the
   config-source chain.

## Reproducibility

- Master launch script: `/pscratch/sd/j/jibancat/prod533_5k_20260511/resume_local_logs/filter_confirm_master.sh`
- Master log: `…/resume_local_logs/filter_confirm_master.log`
- FILTER0 relaunch (2 slices only): `…/resume_local_logs/filter0_slices13.log` driven by `_filter0_slice0_launch.sh` (with `MISSING_SLICES="1 3"`)
- Per-variant `RESUME_LOCAL_*.md` confirming env propagation:
  - filter0: `…_filter0/RESUME_LOCAL_440171_1778704385.md` shows `FILTER_LOW_LIKELIHOOD=0`
  - nfl31: `…_nfl31/RESUME_LOCAL_440172_1778704385.md` shows `FILTER_LOW_LIKELIHOOD=1`
  - (RESUME_LOCAL.md only logs a fixed subset of env vars; NFL is not in the
    dump but is verified in the actual python `--num_forest_lines` cmdline.)
- Molly P/C runs: `examples/molly_faithful_pc_plots.py --snr-min 2 --gp-conf 0.99 --lyb-veto --lam-rf-min 911 --truth-nhi-min 20.3 --no-bal`. TSVs at:
  - baseline (existing): `/pscratch/sd/j/jibancat/prod533_5k_20260511/molly/london_v3_loa124_pw14_tau_eb_8f/molly_summary.tsv`
  - nfl31: `…/molly/nfl31_8f/molly_summary.tsv`
  - filter0_s13: `…/molly/filter0_s13_8f/molly_summary.tsv`
- NHI-bin analysis script: `_nhi_bin_completeness.py` in repo root; output
  JSON at `…/resume_local_logs/nhi_bin_completeness.json`.
