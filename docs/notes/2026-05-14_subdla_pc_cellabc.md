# Sub-DLA P/C eval on joint sweep cells A/B/C — 2×3 table

> **PR #7 task 5 (Option B from `project_subdla_dla_joint_design`)**:
> the joint sweep cells A/B/C use a single DLA model with widened NHI prior
> rather than the 3-way `[Null, SubDLA, k-DLA]` baseline (see
> `docs/notes/2026-05-13_cellC_mechanism_verdict.md`). The whole *point* of
> widening the prior is to ALSO get sub-DLA detection out of the same run.
> This note populates the (DLA, sub-DLA) × (cellA, cellB, cellC) 2×3 table
> on London-0 5k validation, using the existing HDF5 + dlacat outputs (no
> new inference) at the canonical operating point.

## Setup

- Source cells: `/pscratch/sd/j/jibancat/prod533_5k_20260511/joint_dla_subdla_sweep/{cellA_md3_nhi19to23, cellB_md4_nhi19to23, cellC_md3_nhi172to22}/`
- Truth: London mock-0 `dla_cat.fits`, full 5k targets, S2N + Z_QSO from each cell's processed h5.
- Operating point: SNR_RED > 2, P_DLA ≥ 0.99, lyb-veto, `--no-bal`,
  λ_rf ∈ [911, 1216], z_qso ∈ [2, 4.25].
- Tool: `tools/research/subdla_pc_eval.py` — band-restricted P/C wrapper
  (predicted MAP NHI ∈ [lo, hi) **and** truth NHI ∈ [lo, hi) on both sides).
- Bins: **DLA** = NHI ∈ [20.3, ∞); **sub-DLA** = NHI ∈ [19.0, 20.3).
- Run with the DESI env preamble in `docs/development_map.md`. Raw TSV at
  `/pscratch/sd/j/jibancat/prod533_5k_20260511/joint_dla_subdla_sweep/subdla_pc_2x3.tsv`.

## 2×3 table — Purity / Completeness at P_DLA ≥ 0.99, SNR > 2

| Cell | Config | DLA P | DLA C | sub-DLA P | sub-DLA C |
|---|---|---:|---:|---:|---:|
| **cellA** | SINGLE_ABS=1, MAX_DLAS=3, NHI [19, 23], PW 50k | 0.793 | 0.839 | **0.284** | 0.568 |
| **cellB** | SINGLE_ABS=1, MAX_DLAS=4, NHI [19, 23], PW 50k | 0.797 | 0.839 | **0.260** | 0.590 |
| **cellC** | SINGLE_ABS=1, MAX_DLAS=3, NHI [17.2, 22], PW 50k | **0.828** | **0.830** | **0.426** | 0.565 |

Counts (TP / kept / truth):

| Cell | DLA TP/kept/truth | sub-DLA TP/kept/truth |
|---|---|---|
| cellA | 287 / 362 / 342 | 484 / 1707 / 852 |
| cellB | 287 / 360 / 342 | 503 / 1937 / 852 |
| cellC | 284 / 343 / 342 | 481 / 1129 / 852 |

The DLA-bin numbers reproduce the HANDOFF.md 2026-05-13 evening table
(within ±0.002 purity, identical completeness). The sub-DLA bin numbers
are **new** as of this note.

## Interpretation

**All three cells fall far short of the 85 / 70 sub-DLA target** stated
in `docs/development_map.md`. Sub-DLA completeness is in the 0.56–0.59
range across cells (≤ 70% target by ~10 pp), purity is 0.26–0.43 (≤ 85%
target by 40+ pp).

**The mechanism is the same posterior-arithmetic that makes cellC win on
the DLA bin** (`docs/notes/2026-05-13_cellC_mechanism_verdict.md`): under
`SINGLE_ABSORBER_MODEL=1` there is no per-bin probability — `p_dla` is one
scalar over the entire prior support. A spectrum with a borderline truth
NHI ≈ 20.0 absorber routinely passes p_dla ≥ 0.99 and emits a row whose
MAP NHI lands somewhere in the prior. That row goes into either the DLA
bin (if MAP NHI ≥ 20.3) or the sub-DLA bin (if MAP NHI ∈ [19, 20.3)). The
**predicted-NHI assignment is a thresholded readout, not a posterior
probability** — so high-confidence detections get spread across both bins
according to where the MAP point estimate lands.

Concretely:

- **cellA / cellB**: `n_kept` in the sub-DLA bin (1707, 1937) is ~5× the
  DLA bin (362, 360). The `[19, 23]` prior pushes MAP NHI estimates
  preferentially into the sub-DLA range. Purity tanks (0.28, 0.26)
  because most of those rows have no matched truth in [19.0, 20.3).
- **cellC**: with `[17.2, 22]` prior, `n_kept` in sub-DLA = 1129. Lower
  than A/B but still ~3× the DLA bin. Purity 0.43 — the highest of the
  three because the LLS-extended lower bound bleeds some MAP estimates
  into the [17.2, 19) LLS range (silently dropped by the eval), leaving a
  cleaner sub-DLA selection.
- **cellB MAX_DLAS=4** vs cellA MAX_DLAS=3: identical DLA-bin numbers,
  +2 pp sub-DLA completeness, −2 pp sub-DLA purity. The extra absorber
  slot picks up some additional sub-DLAs but at proportional FP cost.
  **Going to MAX_DLAS=4 for sub-DLA is a wash.**

## Why this approach is structurally limited for sub-DLAs

The 2-way model `[Null, k-abs]` with widened NHI prior is a deliberate
trade: it gives up the 3-way `[Null, SubDLA, k-DLA]` decomposition that
makes the DLA-bin classifier per-bin. The cost of that trade shows up
sharpest in the sub-DLA bin, where:

1. **No NHI prior conditioning**: under `SINGLE_ABSORBER_MODEL=1`, the
   posterior over NHI is conditioned only on data and the wide prior;
   there is no separate "this is most likely a sub-DLA, not a DLA"
   evidence channel.
2. **MAP-driven assignment**: the bin label comes from the MAP NHI scalar.
   For a borderline NHI ≈ 20.3 absorber the MAP can land on either side
   of the cut; this is a noisy classifier with no calibrated uncertainty
   bridging the bins.
3. **Sub-DLA truth is ~2.5× more numerous**: 852 sub-DLAs vs 342 DLAs
   in the post-cut mock-0 5k sample. Even a small per-spectrum FP rate
   becomes a large absolute count, dragging purity down.

The cleanest path back to a per-bin probability for sub-DLAs is the
3-way model (`SINGLE_ABSORBER_MODEL=0`) with the existing sub-DLA prior
`[19.1, 20.0]` — which is what the v3 baseline does, and which gives the
sub-DLA aggregator separate `model_posteriors[:, 1]` to threshold on.
That path was not run today; the 3-way DLA-bin numbers are in HANDOFF.md
2026-05-13 evening top block (baseline P=0.85 / C=0.77).

## Verdict relative to "ship cellC as the joint catalog"

For a downstream consumer that wants **one joint catalog** with usable
P/C in both NHI bins:

- **cellC is the best sub-DLA purity** (0.43) of the three at acceptable
  DLA-bin headline (0.83 / 0.83). But absolute sub-DLA P/C is well
  below the project aim, so a joint cellC catalog cannot be the
  *primary* sub-DLA deliverable — it would need an explicit "this is the
  DLA catalog; sub-DLA is best-effort" framing.
- **For a sub-DLA-primary deliverable**, run a separate 3-way job
  (baseline config) and post-hoc threshold `model_posteriors[:, 1]`. The
  separate sub-DLA pass costs ~1.0× a baseline production run (existing
  prior + sample files reused) and gives a calibrated per-bin P(SubDLA)
  rather than a MAP readout.

This corroborates the cellC verdict note's recommendation of
**"flag, not default"**: cellC remains the right joint catalog when
both bins are needed at one operating point, but the headline product
should still be the 3-way baseline plus a separate sub-DLA pass.

## Reproduction

```bash
bash -c '
source /usr/share/lmod/lmod/init/bash
export DESI_ROOT=/global/cfs/cdirs/desi
source /global/common/software/desi/desi_environment.sh main
cd /pscratch/sd/j/jibancat/desi_gpy_dla_detection
TRUTH=/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/dla_cat.fits
BAL=/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/bal_cat.fits
MOCKDIR=/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124
OUT=/pscratch/sd/j/jibancat/prod533_5k_20260511/joint_dla_subdla_sweep/subdla_pc_2x3.tsv
for cell_label in \
    "cellA_md3_nhi19to23 cellA" \
    "cellB_md4_nhi19to23 cellB" \
    "cellC_md3_nhi172to22 cellC"; do
    set -- $cell_label
    CD=/pscratch/sd/j/jibancat/prod533_5k_20260511/joint_dla_subdla_sweep/$1/
    LBL=$2
    for BIN in dla sub; do
        python3 tools/research/subdla_pc_eval.py \
            --catalog-dir "$CD" --truth "$TRUTH" --bal-cat "$BAL" --mockdir "$MOCKDIR" \
            --bin "$BIN" --no-bal --lyb-veto --lam-rf-min 911 --label "$LBL" \
            --out-tsv "$OUT"
    done
done
'
```
