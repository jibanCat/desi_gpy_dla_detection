# 2026-05-15 — λ-range (MAX_LAMBDA) smoke test

> **Status**: DONE (2026-05-15 18:03; results filled in 2026-05-17).
> **Verdict: HELPS.** Extending the GP modeling window redward of Lyα is a
> strict Pareto improvement; the optimum is near 1260 Å. Refined by the
> follow-up `lambda_fine_sweep` (see `2026-05-15_lambda_fine_sweep.md`),
> which found `MAX_LAMBDA=1250` (F2) is the Pareto-best point at
> P=0.838 / C=0.830.
>
> **Sweep root**: `/pscratch/sd/j/jibancat/prod533_5k_20260511/lambda_range_sweep/`
>
> **Recipe-version note**: this sweep was evaluated *after* the
> 2026-05-15 `molly_faithful_pc_plots.py` recipe fix
> (`2026-05-15_molly_eval_recipe_fix.md`) — drop-ALL-BAL, external
> snr/zcat. So n_truth = **581**, not the 618 quoted below in the
> pre-fix "Reference baseline" / "Operating point" sections. All four
> L-cells use the fixed recipe, so they are internally consistent; they
> are **not** directly comparable to the pre-fix cellC/cellD numbers.

## Hypothesis

Extending the GP rest-frame **modeling** window past the historical Lyα
boundary (`MAX_LAMBDA = 1216.75`) into the redward continuum gives the
GP additional pixels to constrain (a) the per-spectrum continuum
normalisation around the Lyα emission line, and (b) the redward
damping wing of strong DLAs (logN ≥ 20.3) whose Voigt profile
extends 5-20 Å past Lyα. If the redward residual is dominated by
clean continuum + the trained mu/M, this *helps* P/C. If it is
dominated by per-quasar emission-line variance (NV, SiII, OI, CII,
SiIV), intrinsic narrow-line absorbers, or BAL features, it *hurts*.

## Mechanism inspection (code reading, before launching)

The `MAX_LAMBDA` knob (`params.max_lambda`) feeds **two** things in
the inference pipeline:

1. **Spectrum slicing** at `null_gp.py:184`:
   ```python
   ind = (self.x >= self.params.min_lambda) & (self.x <= self.params.max_lambda)
   ```
   Determines which observed pixels are kept for the GP fit. Larger
   MAX_LAMBDA = more red-side pixels in the data vector.

2. **DLA z-search range** at `set_parameters.py:126-141`:
   ```python
   max_z_dla = min((max(wavelengths[ind]) / lya - 1) - max_z_cut,
                   z_qso - max_z_cut)
   ```
   The first term *would* grow with MAX_LAMBDA, **but** the second
   term (`z_qso − 3000 km/s`) caps it. So MAX_LAMBDA past Lyα does
   **not** add unphysical z_dla > z_qso candidates. Verified.

The trained model (`2lpt_loa124_nohcd_nobal_wide.h5`) covers
`rest_wavelengths ∈ [850.75, 1699.90]` at dλ=0.15 (5662 pixels), so
the `mu_interpolator`, `log_omega_interpolator`, and `M_interpolator`
do **not** extrapolate when MAX_LAMBDA is pushed up to ~1380 Å. This
was verified upstream in
`/pscratch/sd/j/jibancat/prod533_5k_20260511/null_gp_test/RESULTS.md`
("Grid validation" section).

`NORMALIZATION_MIN/MAX_LAMBDA = [1425, 1475]` is unchanged across all
cells — the per-spectrum median normalisation window is independent
and applied before the inference window slice.

`LOADING_MAX_LAMBDA = 1550` already covers all four test ranges
(1216.75, 1260, 1300, 1380) — no spectrum-loading change needed.

`tau_eb` calls `params.max_z_dla(...)` once per spectrum, so its
candidate-search range scales with MAX_LAMBDA via the same min() cap;
this is a coupled effect but not a confound.

`search_minlam/maxlam = 900/1230` (`constants.py:20-21`) drive the
"too-much-masked" guard in dlasearch.py and are independent of
MAX_LAMBDA.

The eval window stays at λ_rf ∈ [911, 1216] (predicted-DLA selection),
so we are testing **GP fit** width, not the catalog selection cut.

## Configs

Cloned from `cellC_knob_sweep/configs/C0.env` (post-patch 2-way
cellC baseline: PW 50k, MAX_DLAS=3, FILTER=1, τ-EB=on, NHI [17.2, 22],
SubDLA samples 100k). Single knob varied.

| Cell | MAX_LAMBDA | Δ from L0 (Å) | extra px* | rationale |
|------|-----------:|--------------:|----------:|-----------|
| L0   | 1216.75    | 0             | 0 (2033)  | baseline = clone of C0 |
| L1   | 1260       | +43.25        | +289 (+14%) | Lyα emission peak + N V 1240 |
| L2   | 1300       | +83.25        | +555 (+27%) | + Si II 1260, stops short of OI 1302 |
| L3   | 1380       | +163.25       | +1089 (+54%) | + OI 1302, SiII 1304, CII 1335; stress test |

*Pixel counts at dλ=0.15 Å.

## Operating point (eval recipe, identical to cellC sweep)

`molly_faithful_pc_plots.py` with: SNR_RED > 2, p_DLA ≥ 0.99,
lyb-veto on, `--no-bal`, λ_rf ∈ [911, 1216], NHI ≥ 20.3 truth+predicted,
n_truth = 618. Mock = London v5.9.5 mock-0, 5k slice (8 × spectra-16
files starting at index 0).

## Cost prediction

Wall-time scales roughly as O(n_pixels²) for the Woodbury inversion
plus a constant per-sample Voigt cost. From L0=2033 px → L3=3122 px,
expect L3 wall ~2.4× L0. With 4 cells × 8 parallel python procs/cell
sharing 256 cores (1:1 oversubscription budget), L0 ≈ ~60 min,
L3 ≈ ~140 min. Total ~140 min wall-clock for the slowest cell.

## Reference baseline

C0 (post-patch) at the same operating point: **P=0.7792, C=0.8772**
(n_cat = 5167, n_truth = 618). L0 should reproduce this within
sampling noise — it is a literal copy of C0 except living in a
different directory.

## P/C table — DONE

Source: `lambda_range_sweep/HEADLINE.tsv`. n_truth = 581 (fixed recipe).
Wall-time was not logged (ran on the jupyter node, no start/end stamps).

| Cell | knob | P | C | ΔP vs L0 | ΔC vs L0 | n_cat |
|------|------|---:|---:|---:|---:|---:|
| L0   | MAX_LAMBDA=1216.75 (baseline) | 0.7719 | 0.8173 | ref | ref | 4661 |
| L1   | MAX_LAMBDA=1260 | **0.8118** | **0.8545** | **+4.0** | **+3.7** | 4231 |
| L2   | MAX_LAMBDA=1300 | 0.8088 | 0.8514 | +3.7 | +3.4 | 4208 |
| L3   | MAX_LAMBDA=1380 | 0.8250 | 0.8173 | +5.3 | 0.0 | 4224 |

## Verdict — HELPS

Extending the GP modeling window redward of Lyα **strictly Pareto-improves**
P/C. L1 (1260) and L2 (1300) both gain ~+4pp purity *and* ~+3.5pp
completeness over L0 — the hypothesis's "clean continuum + trained mu/M"
branch wins; the per-quasar emission-line-variance branch does not dominate
out to ~1300 Å. n_cat *drops* ~9% (4661→4231) while completeness *rises*,
i.e. the extra red-side constraint kills false positives without losing
true DLAs.

L3 (1380) is the stress case: purity climbs further (+5.3pp) but
completeness falls back to baseline — pushing past the OI 1302 / SiII 1304
/ CII 1335 lines starts trading recall for precision, exactly the predicted
emission-line-variance regime. So the optimum is **interior**, near
1260–1300.

The follow-up `lambda_fine_sweep` (1228–1300 in fine steps) located the
balanced optimum at **MAX_LAMBDA=1250 (F2): P=0.838 / C=0.830** — the
recommended production value (carried into `lambda1250_crossval`).

## Files

- Configs: `lambda_range_sweep/configs/L{0,1,2,3}.env`
- Launcher: `lambda_range_sweep/_launch.sh` (1 cell)
- Chain runner: `lambda_range_sweep/_chain.sh` (4 cells in parallel,
  emits `ALL_DONE` then auto-runs eval)
- Eval+aggregate: `lambda_range_sweep/_eval_and_aggregate.sh`
- Headline: `lambda_range_sweep/HEADLINE.tsv` (created by eval)
