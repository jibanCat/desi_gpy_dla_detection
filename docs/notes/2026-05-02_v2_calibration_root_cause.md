# v2 trainer rank-1 collapse — root-cause analysis

> **Status**: Finding 2026-05-02. PCA init is NOT the bug. Training
> dynamics (Adam + no weight_decay + cosine LR) are. Test fix in
> SLURM job 49227683.

## Headline correction to the earlier finding

The 2026-05-02 mid-day calibration writeup
(`2026-05-02_v2_trainer_calibration_finding.md`) blamed the v2
trainer's rank-1 mode-collapsed K on a per-pixel vs per-row PCA
NaN-fill choice. **That diagnosis was wrong.** Updated understanding:

1. The v2 PCA init was operating on **un-normalized fluxes** in my
   plot_pca_init_K.py script. The trainset.h5 stores fluxes that are
   pre-Lyα-normalized but NOT centered (per-spectrum medians range
   from -0.5 to 35 in [1310, 1325]). With that amplitude variance,
   PCA correctly finds "amplitude scale" as the rank-1 mode →
   misleading conclusion that the init itself was bad.

2. The actual training-time PCA init runs on the
   `load_preprocessed_h5` output, which DOES apply
   normalize → de-forest → center. With proper preprocessing,
   PCA init eff_rank = 1.7-3.1 (not 1.05). Healthy.

3. Even with the per-row vs per-column fill choice — both give
   essentially the same eff_rank on normalized data. The fill choice
   doesn't matter at this NaN density.

So the **PCA init is NOT the bug** in v2's mode collapse. The collapse
happens during training.

## What's actually broken

The v2 trainer ends with `trace_omega² / trace(K) = 0.2-3.4%` and
top eig 5-500× the second. The same trainset under v1 has trace_omega²
/ trace(K) = 84% and top eig 3.3× the second.

What's different in v2 training vs v1:

| | v1 | v2 |
|---|---|---|
| Optimizer | L-BFGS (2nd order) | Adam (1st order, momentum) |
| Weight decay | (none in objective) | `weight_decay=0.0` (default) |
| LR schedule | n/a (L-BFGS) | cosine `eta_min=1e-5` |
| Loss eval | per-spectrum analytical gradient + summed | vectorized batch + autograd |

The combination: Adam's momentum + zero weight decay + LR annealing
to ~1e-5 conspire to drive M growth in the dominant eigenvector
direction, then anneal LR before the basis can diversify into other
modes. ω² shrinks because the trainer can attribute more variance to
the (growing) M·M^T.

## What was correct in the earlier finding

The Mahalanobis χ² calibration check **is correctly diagnosing
over-fitting** — at the trained equilibrium, the residuals are
~5-7× smaller than the model's own predicted σ. So inference
*does* under-attribute residuals to noise, *does* over-attribute to
the M basis, and *does* miss DLA-as-residual events. The canonical
TID 120046865 misses (especially LOA-noHCD-withBAL with p_dla=0.037)
are real consequences.

## Test in flight

SLURM job **49227683** (queued, PD): re-train
`loa_no_dla_no_bal_pcafix_v2` from scratch with:
- `--scheduler none` (constant LR throughout 1500 epochs)
- `--weight-decay 1e-6` (constrain M growth)
- `--z-min 2.5 --z-max 4.25` (matches v1 trainset z range)

After it finishes (~3-5 h on GL):
1. Re-run `check_v2_model_calibration.py` → expect chi²/n closer to 1
2. Re-run K decomposition → expect trace_omega² / trace(K) > 0.5
3. Re-run canonical TID test → expect DLA detection p_dla > 0.5
4. If all three improve → root cause confirmed.

## Additional v1-compat fix

The v1 `preload_qsos.m` has two quality filters that v2 lacks:
- Bit 2: drop spectra where normalization median is NaN/zero
- Bit 3: drop spectra with < min_num_pixels valid pixels in
  [911, 1216] Å rest

Added both to `gpy_dla_detection/training/dataset.py:load_preprocessed_h5`
in commit `3bb9d67`. Default `min_valid_pixels_lyman=200` is
conservative (~10% of the 2030-pixel range at dlambda=0.15). v1
typically used higher.

These filters DON'T affect well-behaved spectra; they remove the few
pathological ones that pollute μ + PCA init. Probably orthogonal to
the rank-1 issue but still a real v1↔v2 gap that should be closed.

## Key take-aways for future investigations

1. **Always pipeline-normalize before diagnosing PCA**: running PCA
   on raw trainset.h5 fluxes gave a misleading rank-1 picture. The
   trainer doesn't actually do that — `load_preprocessed_h5` runs
   the normalize→deforest→center pipeline first.

2. **rank-1 at the trained equilibrium ≠ rank-1 at init**: v1 init
   was probably also rank-dominant at init (PCA always finds the
   biggest variance mode first), but training diversified it. v2
   training amplified the rank-1 instead. The dynamics matter.

3. **Adam + no weight decay + LR annealing is a known mode-collapse
   recipe** in the deep-learning literature. The legacy v1 used
   L-BFGS (no momentum, line-search step sizes) which avoids this
   trap. We picked Adam in v2 for vectorization-friendliness; that's
   a real cost.

4. **"Verified parity tests" don't catch dynamics regressions**:
   `test_objective_v2_parity.py` shows v2's NLL + gradient match v1's
   to 1e-9 on synthetic batches. But the parity tests don't run a
   1500-epoch optimization, so they couldn't have caught the
   mode-collapse behavior. We need a longer-horizon "training health"
   test that verifies trace_omega²/trace(K) stays healthy after N
   epochs on a real trainset subset.

## Artifacts

- `examples/plot_pca_init_K.py` — PCA init K viz (now uses normalized data)
- `docs/notes/2026-05-02_pca_init_K_loa_normalized.png` — LOA, eff_rank=1.78 (healthy)
- `docs/notes/2026-05-02_pca_init_K_2lpt_normalized.png` — 2lpt, eff_rank=3.11 (healthy)
- `docs/notes/2026-05-02_pca_init_K_loa_3way.png` — old plot on un-normalized data (kept for posterity)
- `docs/notes/2026-05-02_v2_loss_curves.png` — v2 converges, v1 oscillates per-batch
- `examples/check_v2_model_calibration.py` — Mahalanobis χ² + per-pixel resid check
- 4 calibration figures from the earlier session
- `docs/notes/2026-05-02_v2_trainer_calibration_finding.md` — earlier finding (PCA-init blamed; correction in this doc)
