# Trainer debug plan — rebuild v2 trainer from v1 reference

> Status: 2026-05-06. Started after `_corrected` v2 retrain regression
> (`docs/notes/2026-05-06_corrected_model_validation/REPORT.md`).
> All 6 retrains miss canonical TID 120046865 (p_DLA = 0.000–0.357 vs
> v1 = 0.520, prev v2 = 0.9999) and have non-physical corr(M·M^T)
> matrices despite χ²/n_valid in band.

## Why we're rebuilding instead of patching v2

`gpy_dla_detection/training/model_v2.py:108` initializes the GP basis
`M` with `torch.randn(num_pixels, k) * 0.05`. The v1 reference at
`gpy_dla_detection/learn_qso_model.py:432–436` uses
`coefficients[:, :k] * sqrt(latent[:k])` — PCA components from the
centered training fluxes scaled by the square root of their
eigenvalues. That is the physics prior the v2 rewrite walked away
from. Random init in a high-dim GP loss admits many degenerate
fixed points where the model "learns" to reconstruct each spectrum
through the basis, instead of capturing the population's covariance.

`gpy_dla_detection/training/objective_v2.py` uses pyTorch autograd
through a vectorized closure. The v1 reference at
`gpy_dla_detection/objective.py:97–191` uses hand-coded analytic
gradients (`dM`, `dlog_ω`, `dlog_c_0`, `dlog_τ_0`, `dlog_β`),
established over 8+ debug commits in 2025-02 and matching the MATLAB
reference at `https://github.com/jibanCat/gp_dla_detection_dr16q_public`
(`spectrum_loss.m`). Autograd cannot be cross-checked against the
MATLAB gold standard at the gradient level.

Patching v2 to fix these two would mean re-doing v1 by hand on top of
broken scaffolding. Cleaner: copy v1 verbatim into `training_v3/`
(done in the previous commit), prove it works, then carefully
introduce only the additions v2 was supposed to deliver (vectorization
across batch; saving normalization metadata in trained .h5).

## The four steps

### Step A — verbatim retrain reproduces v1 (and matches MATLAB)

**Goal**: confirm the verbatim v1 copy in `training_v3/`, run on a small
2lpt subset, produces:
1. Physically-plausible μ (Lyα emission peak in the right place; smooth
   continuum elsewhere).
2. ω that is small in the continuum, larger in the forest — not the
   inflated "absorbing all forest correlation" pattern we saw in
   `_corrected`.
3. corr(M·M^T) with smooth long-range eigenmodes (the Lyα-emission
   scale shows up as the dominant mode).
4. χ²/n_valid → 1 with z-score N(0,1).
5. Canonical TID 120046865 detection at p_DLA ≈ 1.

**Concrete tests to add** (`tests/test_v1_trainer_reference.py`):

A.1 **Numeric Jacobian sanity check** on one frozen 2lpt spectrum:
   Pick TID 120046865 (canonical) at the trainer's preprocessed grid.
   For each parameter, perturb by ±ε and compare central finite
   differences against the analytic gradient in `spectrum_loss`.
   - `∂L/∂M[i, j]` for a sample of (i, j) — should match to ~1e-7
   - `∂L/∂log_ω[i]` for a sample of i
   - `∂L/∂log_c_0`, `∂L/∂log_τ_0`, `∂L/∂log_β`
   Failure: bug in v1 itself (and probably in MATLAB too — escalate).
   Pass: v1's analytic gradients are the truth; they are the
   spec for any future vectorized rewrite.

A.2 **Loss equivalence vs MATLAB** on 5 frozen 2lpt spectra:
   - Run MATLAB `spectrum_loss.m` on each spectrum (parameters fixed
     at initial guesses: `c_0 = 0.1`, `τ_0 = 0.00246`, `β = 3.62`,
     `M` from PCA, `log_ω = log(std)`).
   - Run `training_v3/objective.py:spectrum_loss` on the same
     spectrum + parameters.
   - Loss values should match to ~1e-8.
   - Each gradient component should match to ~1e-8.
   The MATLAB code is gold standard; it has trained the v1 production
   models that DO detect DLAs. This is the ground-truth check.

A.3 **Short retrain on 2lpt** (~50 epochs, 5k spectra subset):
   Run `desi_learn_qsos_model.py` with the existing 2lpt mock-0 trainset.
   Plot loss curve; should decrease monotonically and stabilize.
   Compute calibration on 200 hold-out 2lpt spectra after training:
   χ²/n_valid mean and z-score std should be within range of MATLAB on
   the same data.
   Plot corr(M·M^T) — should be smooth, not noise-textured.

A.4 **Canonical TID inference** with the resulting `model_epoch_<N>.h5`:
   Run `examples/canonical_tid_per_model.py` (extended to take
   `--learned-file` overriding the model list). Expect p_DLA ≈ 1.0.

If A.1–A.4 all pass, Step A is complete. If any fails, the bug is in
v1 itself (or in the preprocessing pipeline), not in v2 — escalate
before continuing.

### Step B — vectorize the per-spectrum loop, with its own equivalence test

**Constraint** (per user): do NOT change the analytic gradient
formulas when vectorizing. The hand-derived per-spectrum gradients in
`spectrum_loss` are the truth; vectorization is purely a parallel
restructuring across the batch dim.

Vectorization challenges:
- Each spectrum has its own `valid_mask` (different masked pixels).
- Each spectrum has its own `lya_1pz` and `z_qso`.
- Implies batched cholesky / batched solves on potentially
  varying-shape inputs.

Strategy: pad all spectra to `n_pix = num_pixels` (the rest-frame
grid is fixed); apply the per-spectrum mask by setting masked
pixels to a large noise variance and zeroed flux/μ — they contribute
zero to the gradient. Use `torch.linalg.cholesky_ex` and
`solve_triangular` in batched mode.

**Test** (`tests/test_v1_trainer_vectorized_equivalence.py`):
   For a batch of 5 frozen 2lpt spectra, compute loss + each gradient
   tensor via:
   - per-spectrum loop in `training_v3/objective.py` (the verbatim v1)
   - vectorized version
   Loss + each gradient component must match element-by-element to
   ~1e-10 (it's the same math, just reshaped).

If equivalence test passes, vectorized version supersedes the loop in
`training_v3/objective.py`.

### Step C — longer retrain to confirm convergence & DLA recovery

After Step B, train for 1500 epochs on full 2lpt mock-0 + a real LOA
trainset, check:
- Loss curve smooth, plateau by ~500 epochs.
- Calibration χ²/n + z-score on hold-out: in range vs Step A.3 short
  retrain.
- DLA recovery on a held-out n=200 strong-DLA set vs truth catalog;
  match (or beat) v1 production's recall.
- corr(M·M^T) physical at full convergence.

Rough cost: ~1 GPU-hour per trainset on GreatLakes. CPU fallback is
~4× slower (per the user's budget remark, this is fine).

### Step D — production retrain decision

If Step C passes for both 2lpt and real LOA:
- Promote one of the new `_v3` models to production.
- Mark `gpy_dla_detection/training/_v2.py` as deprecated; remove in
  a follow-up cleanup.
- Land the per-spectrum normalization fix from 2026-05-01
  (commit `ad06a9f`) on top — it's an independent question
  whether v1's commented-out `normalize_spectra` was an oversight
  or intentional. (Note: `desi_learn_qsos_model.py:97–104` shows the
  call is commented out in the v1 DESI flow; the user's 2026-05-01
  finding said v1 production was correctly normalized. Reconcile in
  Step C: train both with-normalize and without-normalize and pick
  by χ² and DLA recovery.)

## What we're NOT doing in this PR

- Touching the v1 reference files at the source paths (frozen for
  diff).
- Touching the broken v2 files (frozen for inspection).
- Promoting any model to production — that's Step D, behind passing
  Step C.
- Re-running the `_corrected` retrains. Those are conclusively a
  regression; the SLURM jobs do not need to repeat.

## Reference paths

- v1 trainer (gold standard for reference, frozen):
  `gpy_dla_detection/learn_qso_model.py`,
  `gpy_dla_detection/objective.py`,
  `desi_learn_qsos_model.py`
- MATLAB reference (gold standard):
  `/home/mfho/MATLAB/gp_dla_detection_dr16q_public/{spectrum_loss.m, learn_qso_model.m}`
  (MATLAB is available on GreatLakes via `module load matlab/R2024b`).
- v2 broken (frozen for diff): `gpy_dla_detection/training/{model_v2,objective_v2,trainer_v2}.py`
- v3 working area (this PR): `gpy_dla_detection/training_v3/`
- 2lpt mock-0 spectra: `/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/`
- Canonical TID: 120046865, truth log NHI 21.263, in
  `spectra-16/7/789/spectra-16-789.fits`.

## Hard rules in effect

1. PCA / population-statistic init for `M`, `μ`, `log_ω`. No `randn`,
   no `zeros`. Abort on PCA failure.
2. Hand-coded analytic gradients matching MATLAB. No `loss.backward()`
   for parameters where v1 / MATLAB has the analytic form.
3. Copy-don't-reimagine: every change to `training_v3/*.py` is a
   reviewable diff against its starting point.
