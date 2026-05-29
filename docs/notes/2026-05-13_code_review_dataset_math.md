# Code review: dataset.py corr-noise fix

**Reviewer**: independent audit
**Date**: 2026-05-13
**Commit under review**: `aa36205` ("Corr-noise fix: reorder normalize→mask + tighten |med|<1e-2 threshold")
**Branch**: `claude/debug-trainer-from-v1`

Background docs read (not repeated here):
- `docs/notes/2026-05-12_2lpt_corr_noise_debug/findings.md` (discovery + probe results)
- `docs/notes/2026-05-12_training_pipeline_audit_vs_matlab/findings.md` (overall MATLAB↔Python audit)
- `~/MATLAB/gp_dla_detection_dr16q_public/{preload_qsos,learn_qso_model}.m`

## Order vs MATLAB

The reorder is faithful to MATLAB's effective ordering.

MATLAB normalize happens at preload time:
- `preload_qsos.m:41` computes `this_median = nanmedian(this_flux(ind));`
- `preload_qsos.m:63-64` writes the normalized arrays into the preload .mat:

  ```
  this_flux           = this_flux           / this_median;
  this_noise_variance = this_noise_variance / this_median^2;
  ```

Then MATLAB masks at train time, *after* re-loading those normalized arrays:
- `learn_qso_model.m:128`: `ind = (rest_noise_variances > max_noise_variance);` — the LHS is the array constructed at `learn_qso_model.m:111-112` from `all_noise_variance{i}`, which was already divided by `this_median^2` in `preload_qsos.m:64`. So the comparison is effectively `nv_raw / med^2 > 9`.

Python now matches at `gpy_dla_detection/training/dataset.py:356-372`: `_normalize_by_rest_median` runs first, then `_mask_high_noise_pixels`. Verdict: the order swap is correct and necessary for MATLAB parity. The old order (mask raw nv first) is a real divergence that let `med ∈ [1.5e-3, 1e-2]` spectra reach PCA — quantified in the discovery doc at 14.9× corr-init smoothness (0.0130 → 0.1939).

The audit doc was updated honestly at `docs/notes/2026-05-12_training_pipeline_audit_vs_matlab/findings.md:25-28` to flag the prior wrong ✓ on the `max_noise_variance` row.

## Threshold rationale

Threshold tightening to `|med| < 1e-2` is empirically grounded but the probe is narrower than the table in the discovery doc.

Authoritative probe output (`docs/notes/2026-05-12_2lpt_corr_noise_debug/outlier_tail_smoothness.json`):
- `small_pos` pool `[1.5e-3, 1e-2]` → smoothness `0.013004` after rejection (matches CLEAN `0.013004`)
- `large_pos` `[10, 30]` → `0.012999` (matches CLEAN)
- `extreme` `[50, 1e6]` → `0.012993` (matches CLEAN)
- `neg_ctrl` `[≤ 0]` → `0.013004` (matches CLEAN)

Median distribution at `findings.md:40-50` (2lpt loa-0 wide, 50k sample, norm band [1310, 1325]):
- ≤ 0: 0.51%
- (0, 1e-3): 0.008%
- [1e-3, 1e-2): **0.094%** (newly rejected at this commit)
- [1e-2, 1e-1): **1.92%** (still kept — unprobed)
- [1e-1, 10): 95.32% (bulk)
- ≥ 10: 2.15%

**On `5e-3` (looser cut)**: insufficient. The probe pool boundary is `[1.5e-3, 1e-2]` and the contamination signal at 14.9× came from that whole range. A `5e-3` cut would let the half of that pool with `med ∈ [5e-3, 1e-2]` through. Probe does not directly bisect inside the pool, but the asymmetry (95.32% of the preload sits at `med ≥ 1e-1`) makes a cut close to the bulk edge safer than one half-way inside the contaminating range. Reject `5e-3`.

**On `1e-1` (tighter cut)**: not falsified by current evidence, but **not justified** by the probe either. The probe explicitly skips `[1e-2, 1e-1)`. That tail is ~20× more populous (1.92% vs 0.094%) and would, if contaminating, cost ~5750 spectra per 300k batch. Whether to extend to `1e-1` is an open question worth one more probe run — easy to do by copying `examples/probe_outlier_tail_corr.py:71` and adding a `mid_pos: (1e-2, 1e-1)` pool. **(Could not verify either way from the existing evidence.)** Reasonable to ship the current cut and queue a follow-up probe rather than speculatively tightening.

**Verdict**: `1e-2` is the right cut given probed evidence. The `[1e-2, 1e-1)` band is an unprobed unknown — flag for follow-up but not a blocker.

Side observation at `dataset.py:177`: `np.abs(medians) < 1e-2` is technically redundant with `medians <= 0` (the only `|med| < 1e-2` cases not already caught by `medians <= 0` are positive ones), so writing `(medians > 0) & (medians < 1e-2)` would be equivalent. Current form is defensive and arguably clearer about intent. Nit.

## Numerical stability

The reorder introduces an interaction between `_normalize_by_rest_median` NaN-rows and `_mask_high_noise_pixels` that needed checking. It is safe.

Trace:

1. `_normalize_by_rest_median` (`dataset.py:189-193`): for `bad` spectra, `safe_med = 1.0`, then `fluxes_normed[bad] = np.nan` and `nv_normed[bad] = np.nan`. So entire rows are NaN for rejected spectra.

2. `_mask_high_noise_pixels` (`dataset.py:65-74`): `bad = noise_variances > max_noise_variance`. For NaN entries, `nan > 9.0` returns `False` (a deprecation/runtime warning may fire). So `np.where(False, nan, nan)` keeps NaN as-is. NaN propagates. No flux/nv corruption.

3. Downstream consumer `tests/phase2_train_desi.py:662`: `valid_masks = np.isfinite(centered) & np.isfinite(nv) & (nv > 0)`. NaN pixels → `valid_masks = False`. The objective at `gpy_dla_detection/training_v3/objective_vectorized.py:96-97` does `nv_safe = where(valid_mask, nv, ones)` and `y_safe = where(valid_mask, y, zeros)`. So NaN values are explicitly zeroed out before any arithmetic. NaN cannot propagate into gradients.

4. `_pca_init` (`tests/phase2_train_dr16.py:204-214`): for any row where `finite.any()` is False (i.e. all-NaN), `pca_input[i] = 0.0`. Then `np.nan_to_num(...)` belt-and-braces. Rejected spectra contribute a zero row to PCA, which is correct (they shouldn't contribute at all to the basis).

The reviewer's question 3 in the prompt asks "NaN comparisons return False; so NaN pixels are NOT masked → they propagate. Is this what we want?". Answer: yes. The mask uses `np.where(bad, nan, ...)` so a False from the comparison leaves the NaN-valued pixel untouched, and NaN is exactly what downstream code expects to drop the pixel from training (via `valid_masks`). The mask is idempotent on NaN. No bug.

One caveat worth knowing: numpy may emit a `RuntimeWarning: invalid value encountered in greater` when comparing NaN against 9.0 inside `_mask_high_noise_pixels`. The existing code in `_normalize_by_rest_median` (`dataset.py:158-161`) already wraps a similar comparison in `warnings.catch_warnings()`. Consider doing the same here for log hygiene at scale (nit, not a correctness issue).

## Test coverage

The new test at `tests/test_normalize_by_rest_median.py:126-167` is well-targeted: it sets `medians = [0.005, -0.5, 0.02, 100.0]` and asserts the right reject/keep decisions. It is exactly the regression-guard the discovery story implies.

Gaps:

- **Exact-boundary case**: no test for `med == 1e-2` (kept under `< 1e-2`) or `med == 1e-3` from the old rule. Boundary semantics — `< 1e-2` strict — should be explicit in a test (e.g. assert `med = 0.01` is KEPT, `med = 0.00999` is REJECTED).
- **Small-negative case**: no test for `med = -0.005` (rejected by `medians <= 0`, which is independent of the `1e-2` clause). The existing `-0.5` test covers negative, but does not stress the interaction with the magnitude clause.
- **NaN median**: `test_normalize_handles_bad_spectra` at `tests/test_normalize_by_rest_median.py:108-123` already covers NaN and `med = 0`; that path is fine.
- **Reorder interaction**: there is no integration test that asserts the *new ordering* of normalize→mask in `load_preprocessed_h5`. `test_load_preprocessed_h5_normalize_path_smoke` (line 261) tests end-to-end shape and μ but does not exercise a spectrum where `nv_raw < 9` but `nv_raw/med² > 9` (the MATLAB-equivalent rejection case). A targeted test that constructs a `med ∈ [1.5e-3, 1e-2]` spectrum with `nv_raw ~ 5` (which would have escaped the old mask-first pipeline) and asserts it ends up all-NaN after `load_preprocessed_h5` would directly assert the MATLAB-equivalence claim made in the commit message.

None of these gaps are critical because the probe at `examples/probe_outlier_tail_corr.py` doubles as an end-to-end test on real data. But unit tests are cheaper to run in CI.

## Issues found

- **(low)** Probe never tested `[1e-2, 1e-1)` band — 1.92% of the preload, ~20× more populous than the `[1e-3, 1e-2)` band that was confirmed contaminating. The current 1e-2 cut is the right call given probed evidence but the band immediately above is an unknown. Suggest extending `examples/probe_outlier_tail_corr.py:71` with a `mid_pos: (1e-2, 1e-1)` pool and re-running before the next production retrain. File: `examples/probe_outlier_tail_corr.py:70-75`.

- **(nit)** `np.abs(medians) < 1e-2` at `dataset.py:177` is redundant with the `(medians <= 0)` clause. Functionally identical; cosmetic only.

- **(nit)** `_mask_high_noise_pixels` at `dataset.py:71` may emit a `RuntimeWarning: invalid value encountered in greater` when called on NaN-filled rows after the reorder. `_normalize_by_rest_median` already suppresses an analogous warning. Consider mirroring that. File: `gpy_dla_detection/training/dataset.py:65-74`.

- **(low)** Test coverage misses (a) the exact boundary `med = 1e-2`, (b) the new normalize→mask integration case (raw nv passes 9, but normalized nv ≫ 9). Both are easy 5-line additions to `tests/test_normalize_by_rest_median.py`.

- **(nit)** Commit message says "Spectra with median == 0, NaN, negative, or |median| < 1e-2 are unusable — divide-by-tiny-median produces |flux| > 100 outliers" (`dataset.py:163-165`). The old comment had "> 1e3"; the new comment uses "> 100" (one order of magnitude smaller because the new threshold is 10× tighter). Self-consistent. Confirmed.

- **(severity: critical / high / medium)**: **none found**. The math is correct, the order matches MATLAB, the threshold is empirically motivated, NaN propagation through the reordered pipeline is safe.

## Recommendation

**Ship.**

The fix is:
1. **MATLAB-equivalent**: confirmed against `preload_qsos.m:63-64` and `learn_qso_model.m:128`.
2. **Empirically validated**: probe at `examples/probe_outlier_tail_corr.py` brings SMALL_POS smoothness from 0.1939 back to 0.0130, matching CLEAN.
3. **Numerically safe**: the NaN-row interaction between normalize and mask is well-behaved at every downstream consumer (`_pca_init`, `valid_masks`, `objective_vectorized`).
4. **Tested**: the regression-guard test is direct and the existing test file is comprehensive.

Suggested follow-ups (not blockers):
- One additional probe condition for `med ∈ [1e-2, 1e-1)` to put the unknown 1.92% tail to rest.
- One extra unit test asserting `med = 1e-2` is kept (boundary precision).
- One integration test for the new reorder: a `med = 5e-3, nv_raw = 5` spectrum should come out all-NaN, demonstrating that the mask now catches what only normalize used to.
- Optionally suppress the RuntimeWarning from `_mask_high_noise_pixels` on NaN inputs (log hygiene).

The current ordering and threshold should be locked in for any production retrains queued from this branch.
