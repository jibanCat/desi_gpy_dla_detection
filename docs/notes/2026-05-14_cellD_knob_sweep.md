# 3-way knob sweep (D1-D8) — 2026-05-14 (London-0 5k validation)

> ⚠ **SUPERSEDED (2026-05-17).** Pre-+log(N)-patch sweep; its "D1+D4
> Pareto-dominate" conclusion was later shown to be a −log(N)-bias artifact
> (see HANDOFF). Numbers obsolete twice over (pre-patch, pre-DLAFLAG-fix).
> Current 3-way P/C: `cellD_knob_sweep/HEADLINE.tsv` (refreshed 2026-05-17).
> History only.

> **Status: COMPLETE.** 8 cells ran concurrently on jupyter `nid004213`,
> evaluated against London-0 truth. **Bottom line: D1 (MAX_DLAS=4) and D4
> (NHI prior [19, 23]) BOTH Pareto-dominate the 3-way production baseline.**
> The 3-way model is more sensitive to MAX_DLAS than the 2-way (cellC) model
> — opposite asymmetry to what the C-sweep showed, and explainable by the
> 3-way's separate SubDLA channel absorbing low-NHI evidence.
>
> **Caveat — PW-count results (D5/D6/D7) are bias-suspect** per the
> dla_gp.py `-log(N)` bug surfaced this morning. Treat sample-count
> numbers below as biased; MAX_DLAS and NHI-prior numbers are clean
> (same N=50k as baseline, bias cancels).

## Setup

- Mock: London-0 `jura-124`, full 5k (8 slices × 8 workers)
- GP model: `2lpt_loa124_nohcd_nobal_wide.h5`
- 3-way **production baseline** (`SINGLE_ABSORBER_MODEL=0`,
  `MAX_DLAS=3`, NHI [19, 22], PW 50k, SubDLA `subdla_samples.mat` 10k,
  `FILTER_LOW_LIKELIHOOD=1`, τ-EB=on, `n_initial=5000`): P=**0.8452**, C=**0.7661**
- All 8 D cells ran concurrently (2× CPU oversub) on a 256-core node;
  wall is contention-inflated, solo ≈ 50–60 % of reported.
- Configs: `/pscratch/sd/j/jibancat/prod533_5k_20260511/cellD_knob_sweep/configs/D{1..8}.env`
- Outputs: `cellD_knob_sweep/D{1..8}/`

## Headline P/C table

| Cell | Knob | Purity | Completeness | n_cat | wall (min) | node-hr | Δ P | Δ C |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **3-way baseline** | — | 0.8452 | 0.7661 | 1242 | (n/a) | (n/a) | ref | ref |
| **D1** | **MAX_DLAS=4** | **0.8623** | **0.7690** | 1283 | 81.8 | 0.341 | **+1.7** | **+0.3** |
| D2 | MAX_DLAS=5 | 0.8534 | 0.7661 | 1295 | 86.8 | 0.362 | +0.8 | 0.0 |
| D3 | NHI [17.2, 22] | 0.8552 | 0.7427 | 1927 | 81.7 | 0.340 | +1.0 | −2.3 |
| **D4** | **NHI [19, 23]** | **0.8548** | **0.7749** | 1542 | 81.7 | 0.340 | **+1.0** | **+0.9** |
| D5 | PW 30k *(bias-suspect)* | 0.8365 | 0.7778 | 1339 | 67.4 | 0.281 | −0.9 | +1.2 |
| D6 | PW 80k *(bias-suspect)* | 0.8675 | 0.7661 | 1176 | 85.6 | 0.357 | +2.2 | 0.0 |
| D7 | PW 100k *(bias-suspect)* | 0.8741 | 0.7515 | 1130 | 90.6 | 0.378 | +2.9 | −1.5 |
| D8 | n_initial=10k | 0.8457 | 0.7690 | 1307 | 88.6 | 0.369 | +0.05 | +0.3 |

`n_truth = 618` for every row. **Bold** = strict Pareto-dominator of the
baseline. Aggregate cost: **2.77 nh** for all 8 × 5k.

**Two cells Pareto-dominate the 3-way baseline**:

- **D1 (MAX_DLAS=4)**: +1.7 pp purity, +0.3 pp completeness. Adding a
  4th k-DLA branch lets the model absorb extra multi-DLA evidence without
  inflating noise — the 3-way's separate SubDLA channel keeps the
  denominator clean. **Strictly better than baseline at no real cost
  (+0.04 nh).**

- **D4 (NHI [19, 23])**: +1.0 pp purity, +0.9 pp completeness. The
  extended NHI ceiling catches the long upper tail (TP gains in
  [20.5, 21.0) and high-NHI bins) without adding noise at the lower
  boundary. ~+0.04 nh marginal cost.

These are real wins (same N=50k as baseline → unaffected by the dla_gp.py
`-log(N)` bug).

## NHI-bin-stratified completeness

Source: `_nhi_bin_table.py` against `cellD_knob_sweep/nhi_bins.tsv`.

| Cell | [20.3, 20.5) | [20.5, 21.0) | [21.0, 21.5) | [21.5, 22.0) | overall |
|---|---:|---:|---:|---:|---:|
| **3-way baseline** | 0.574 | 0.816 | 0.935 | 0.929 | 0.766 |
| D1 (md=4) | 0.574 | **0.823** | 0.935 | 0.929 | 0.769 |
| D2 (md=5) | 0.574 | 0.816 | 0.935 | 0.929 | 0.766 |
| D3 ([17.2, 22]) | 0.519 | 0.804 | 0.935 | 0.929 | 0.743 |
| **D4** ([19, 23]) | **0.583** | **0.829** | 0.935 | 0.929 | **0.775** |
| D5 (PW 30k) | **0.611** | **0.823** | 0.919 | 0.929 | **0.778** |
| D6 (PW 80k) | 0.565 | **0.823** | 0.935 | 0.929 | 0.766 |
| D7 (PW 100k) | 0.537 | 0.810 | 0.935 | 0.929 | 0.752 |
| D8 (n_init=10k) | 0.565 | **0.829** | 0.935 | 0.929 | 0.769 |

Per-bin purity (predicted-NHI bin):

| Cell | [20.3, 20.5) | [20.5, 21.0) | [21.0, 21.5) | [21.5, 22.0) |
|---|---:|---:|---:|---:|
| **3-way baseline** | 0.583 | **0.932** | 0.900 | 0.955 |
| D1 (md=4) | **0.638** | **0.932** | **0.926** | 0.909 |
| D2 (md=5) | 0.609 | 0.925 | 0.913 | 0.955 |
| D3 ([17.2, 22]) | 0.607 | **0.935** | 0.880 | 0.955 |
| D4 ([19, 23]) | 0.632 | 0.913 | 0.913 | **1.000** |
| D5 (PW 30k) | 0.615 | 0.908 | 0.908 | 0.913 |
| D6 (PW 80k) | **0.667** | 0.921 | 0.886 | **1.000** |
| D7 (PW 100k) | 0.649 | **0.932** | 0.899 | **1.000** |
| D8 (n_init=10k) | 0.600 | **0.932** | 0.889 | 0.909 |

The headline gains for D1 / D4 land in **[20.5, 21.0)** completeness
(D1: +0.7 pp, D4: +1.3 pp) and **[20.3, 20.5)** purity (D1: +5.5 pp,
D4: +5.0 pp). The weak-DLA bin still has C ≈ 0.57 even in the best D
cells — same intrinsic-difficulty story as cellC, just with different
absolute purity (3-way's NHI-conditioned posterior gives cleaner MAP
NHI distributions in this bin → less sub-DLA bleed than cellC).

## 2-way vs 3-way comparison — note alongside `2way_vs_3way_decision.md`

A few observations cellC-vs-D1/D4 worth flagging here:

- **D1 vs C1**: in 3-way, MAX_DLAS=4 is a **win** (+1.7/+0.3); in 2-way
  cellC, MAX_DLAS=4 is a **loss** (−3.0/−1.2). The asymmetry is
  mechanistic: 3-way's separate SubDLA channel absorbs low-NHI evidence,
  so adding k-DLA slots adds clean multi-DLA capacity; 2-way's funnel-all
  approach inflates noise as k grows.

- **D4 vs C3**: extended NHI ceiling [19, 23] is a 3-way **win**
  (+1.0/+0.9), in 2-way cellC same idea ([17.2, 23], C3) is a **loss**
  (−2.4/−0.6). Same asymmetry — 3-way's SubDLA channel keeps the lower
  boundary clean while the upper extension catches strong DLAs.

- **PW count interaction with the bug**: D5/D6/D7 show the same monotonic
  pattern as C5/C6/C7 (more samples → higher P, lower C, fewer cat rows).
  This is the `-log(N)` bias × P_DLA threshold artifact, NOT a real
  signal. The 30k cell shows P=0.84 (-0.9 pp) and the 100k cell shows
  P=0.87 (+2.9 pp) on otherwise-identical configs. Once the bug is fixed,
  these are likely all within ~1σ of baseline.

## Mechanism takeaways

- **3-way model rewards extra capacity**: MAX_DLAS=4 (D1) and extended
  upper NHI prior (D4) both Pareto-dominate. This is the OPPOSITE of
  cellC's behavior, where these knobs hurt. The 3-way's SubDLA channel
  acts as a "junk drawer" for low-NHI evidence, keeping the k-DLA
  channels clean.

- **NHI prior LOWER bound matters**: D3 ([17.2, 22]) hurts the 3-way C
  by 2.3 pp — extending into the LLS range overlaps the SubDLA prior
  ([19.1, 20]) and double-counts evidence in the [19, 20] regime. cellC
  avoids this because it has no separate SubDLA model.

- **n_initial floor (D8)** is essentially zero effect — confirms the
  2026-05-13 2×2 ablation finding that knob 1 is a no-op.

## Recommended best 3-way operating point

**D1 + D4 stacking** (MAX_DLAS=4 AND NHI prior [19, 23]) was not tested
in this OAT sweep. Both are independent dimensions and likely additive.
A "D9" cell combining both knobs is the natural next experiment if the
3-way path is chosen. Predicted (additive linear estimate):
P ≈ 0.872, C ≈ 0.778. Cost ≈ 0.36 nh per 5k.

For now, the **best-known 3-way single-knob cell is D4** (NHI [19, 23]),
because its completeness gain (+0.9 pp) is strictly larger than D1's
(+0.3 pp) at comparable purity. **D1 is best for purity-priority**
deliverables.

## Reproduction

```bash
# (Sample files: pw_samples_a3_190_220_30000.mat and
#  pw_samples_a3_190_220_80000.mat were generated today via
#  python -m gpy_dla_detection.generate_samples; the 100k file
#  pw_samples_a3_190_220_100000.mat already existed.)

# Launch one cell:
cd /pscratch/sd/j/jibancat/prod533_5k_20260511/cellD_knob_sweep
nohup bash _launch.sh D1 8 > logs/D1.log 2>&1 &

# Aggregate:
bash _eval_and_aggregate.sh all

# NHI-bin breakdown:
bash -c '
source /usr/share/lmod/lmod/init/bash
export DESI_ROOT=/global/cfs/cdirs/desi
source /global/common/software/desi/desi_environment.sh main
cd /pscratch/sd/j/jibancat/desi_gpy_dla_detection
python /pscratch/sd/j/jibancat/prod533_5k_20260511/cellD_knob_sweep/_nhi_bin_table.py
'
```

## Run log

All 8 cells ran concurrently on `nid004213` (256-core jupyter compute).

| Cell | wall (min) | dlacat |
|---|---:|---:|
| D1 | 81 | 8/8 |
| D2 | 86 | 8/8 |
| D3 | 81 | 8/8 |
| D4 | 81 | 8/8 |
| D5 | 67 | 8/8 |
| D6 | 85 | 8/8 |
| D7 | 90 | 8/8 |
| D8 | 88 | 8/8 |

Aggregate node-hours: **2.77 nh** for all 8 × 5k.
