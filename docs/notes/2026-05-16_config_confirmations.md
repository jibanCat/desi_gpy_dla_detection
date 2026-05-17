# 2026-05-16 — config-confirmation sweeps: v1_model, single_absorber, determinism

> **Status**: DONE. Three small sweeps that confirm (rather than tune)
> production choices. None changes a recommendation; they pin down the
> noise floor, confirm the single-absorber model, and check the trained
> GP model.
>
> Sweep roots under `/pscratch/sd/j/jibancat/prod533_5k_20260511/`:
> `v1_model_test/`, `single_absorber_sweep/`, `determinism_sweep/`.

All cells: post-patch C7-equivalent recipe (2-way unless noted, PW 100k,
NHI [17.2,22], τ-EB null), London-0 5k slice, fixed molly recipe
(SNR>2, p_DLA≥0.99, lyb-veto, no-BAL, λ_rf∈[911,1216], NHI≥20.3,
n_truth=581).

## 1. determinism_sweep — run-to-run noise floor (job 53017518)

Three byte-identical C7-config replicates.

| Cell | P | C |
|---|---:|---:|
| G0 | 0.8280 | 0.8050 |
| G1 | 0.8301 | 0.8019 |
| G2 | 0.8275 | 0.8019 |
| **spread** | **0.26 pp** | **0.31 pp** |

Within a single sbatch batch the 5k-slice P/C scatter is **≤ ~0.3pp**.
The pipeline is *not* bit-deterministic (τ-EB seed search + QMC Monte
Carlo integration introduce run-to-run variation), but the scatter is
small. Note: an earlier observation (memory `project_5k_noise_floor`)
saw ~1.2pp across *four* runs spanning different batches — so treat
**~0.3pp as the within-batch floor and ~1pp as the cross-batch floor**.
Any sweep delta below ~1pp should be treated as noise; finalist configs
need a 50k replicate to resolve sub-pp differences. These G0/G1/G2 cells
double as the patch-ON arm of `2026-05-16_logn_patch_ab.md`.

## 2. single_absorber_sweep — SINGLE_ABSORBER_MODEL on/off (job 53017517)

| Cell | knob | P | C | n_cat |
|---|---|---:|---:|---:|
| S0 | SINGLE_ABSORBER_MODEL=0 (multi-DLA / legacy-style) | 0.7127 | 0.5913 | 2338 |
| S1 | SINGLE_ABSORBER_MODEL=1 (2-way single-absorber, =C7) | 0.8387 | 0.8050 | 4516 |

The single-absorber model is **dramatically better**: +12.6pp purity,
+21.4pp completeness. The legacy-style multi-DLA mode (S0) badly
under-detects on this recipe (C=0.59). This is a hard confirmation of
the production choice `SINGLE_ABSORBER_MODEL=1` for the cellC family —
no ambiguity, no retune needed.

## 3. v1_model_test — trained-GP model swap (job 53013667)

One cell, **V0_v1model**: C7 in every knob except `LEARNED_FILE`, which
swaps the `2lpt_loa124_nohcd_nobal_wide.h5` research model for the v1
production model `learnlogs/model_epoch_920.h5`.

| Catalog | P | C | n_cat |
|---|---:|---:|---:|
| V0 (v1 model_epoch_920) | 0.8377 | 0.7988 | 3499 |
| C7 (2lpt research model) | 0.8323 | 0.8142 | — |
| LEGACY (v1 920, NHI [20,23]) | 0.7884 | 0.8421 | — |

**Motivation** (see `v1_model_test/README.md`): the high-SNR deep-dive
found C7 losing ~9pp completeness at SNR>10 vs the legacy v1-model
catalog. V0 tests whether the model swap recovers that loss.

**Headline result**: V0 does **not** recover completeness — its headline
C (0.799) is slightly *below* C7 (0.814) and well below LEGACY (0.842).
V0's purity is marginally above C7. So at the headline operating point
the v1 vs 2lpt model swap is roughly P/C-neutral-to-slightly-worse on
completeness.

**Caveat**: the deep-dive's concern was specifically the **SNR>10 bin**,
not the headline. The per-SNR breakdown for V0 (in
`v1_model_test/V0_v1model/pc_snr2_pdla99.md/molly_pc_vs_snr.png` /
`molly_summary.tsv`) should be inspected before concluding — if V0's
SNR>10 C matches LEGACY while its headline C is dragged down by
low-SNR bins, the model *is* the high-SNR lever. **Open item**: read the
V0 per-SNR matrix and compare the SNR>10 cell to the deep-dive's
C7=0.845 / LEGACY=0.948. Until then: the model swap is not a headline
win, and the high-SNR question is unresolved.
