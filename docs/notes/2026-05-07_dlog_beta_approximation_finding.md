# Finding: v1 + MATLAB share an approximation in `dlog_β`

> Discovered 2026-05-06 during Step A.1 (`tests/test_v1_spectrum_loss_jacobian.py`)
> of PR #6 (debug-trainer-from-v1). Resolution: introduce **v3.5** as a
> separate strict-gradient variant; keep **v1**, **v3** (verbatim+vectorized),
> and **MATLAB** unchanged.

## Summary

The hand-coded analytic gradient `dlog_β` in `gpy_dla_detection/objective.py:182–189`
(v1 Python) and `learn_qso_model_dr16q_public/spectrum_loss.m:91–95` (MATLAB)
both ignore the chromatic-frame term in `∂τ_total/∂β` when `num_forest_lines > 1`.

The approximation is **systematically biased by 0.5–2.5 %** depending on z_qso.
v1 production models trained successfully under it because (a) `log_β` is one
scalar parameter and the optimizer absorbs the bias into a slightly shifted
fixed point, and (b) the prior Turner+2024 `β = 3.62 ± 0.04` keeps β tightly
constrained near the truth.

## Math

Total optical depth in the GP forward model:

```
τ_total(i) = Σ_k τ_k(i) · 𝟙_{k-forest}(i)
τ_k(i)     = τ_0 · α_k · (1 + z_k(i))^β,   α_k = (λ_k · f_k)/(λ_lya · f_lya),
1 + z_k(i) = (λ_lya / λ_k) · (1 + z_lya(i)) ≡ r_k · (1 + z_lya(i))
```

Differentiate strictly:

```
∂τ_total/∂β = Σ_k τ_k(i) · log(1 + z_k(i)) · 𝟙_{k-forest}(i)
            = τ_total · log(1+z_lya)                                   ← TERM (A) — kept
              + Σ_{k>1} τ_k(i) · log(r_k) · 𝟙_{k-forest}(i)            ← TERM (B) — DROPPED
```

Term (B) is identically zero for `num_forest_lines = 1` (Lyα only). For
`num_forest_lines = 3`, log(r_k) is:

| line | λ (Å) | r_k = λ_lya/λ_k | log(r_k) |
|---|---:|---:|---:|
| Lyα | 1215.67 | 1.000 | 0.000 |
| Lyβ | 1025.72 | 1.185 | 0.170 |
| Lyγ | 972.54 | 1.250 | 0.224 |

## What v1 + MATLAB compute

```python
# objective.py:183, 188 (Python) — and matching spectrum_loss.m:88, 94 (MATLAB)
da_tau0 = omega2 * scaling_factor * lya_optical_depth * lya_absorption
da_beta = da_tau0 * torch.log(lya_1pz) * beta * indicator
```

`lya_optical_depth` is `τ_total` (Lyα + Lyβ + Lyγ summed). The expression
`lya_optical_depth * log(lya_1pz)` is term (A) only. Term (B) is silently
dropped.

## Empirical bias measured on the frozen 2lpt fixture

`tests/test_v1_spectrum_loss_jacobian.py` evaluates the analytic
`dlog_β` against central finite differences (eps = 1e-5) on the 6
frozen TIDs. Comparing v1 (approximate) vs v3.5 (strict) at the same
init point:

| TID | z_qso | v1 dlog_β | v3.5 dlog_β | bias (v1 vs strict) |
|---:|---:|---:|---:|---:|
| 237926    | 2.601 |   −354.79 |   −354.57 |   0.05 % |
| 120046865 | 2.962 |  2442.91  |  2460.60  |   0.72 % |
| 250915    | 2.879 |  2930.35  |  2951.91  |   0.74 % |
| 237575    | 3.225 |  2359.79  |  2409.86  |   2.12 % |
| 242431    | 3.502 |  1348.30  |  1381.15  |   2.44 % |
| 243225    | 3.782 |  6084.22  |  6179.97  |   1.57 % |

Bias scales with z_qso: at higher z, more pixels in the rest grid are
inside Lyβ + Lyγ forests (where term (B) contributes), so the missing
correction is larger. This matches the prediction from the math.

The strict-gradient version (`gpy_dla_detection/training_v3_5/objective.py`)
agrees with FD to ~6e-10 on `dlog_β` (machine-precision), confirming the
correction was the only remaining FD-vs-analytic discrepancy.

## Why v1 trained successfully under the approximation

1. **`log_β` is a single scalar parameter.** The optimizer converges to a
   fixed point where `dlog_β = 0`. With a 1–2 % systematically biased
   gradient, the fixed point is shifted by O(bias) from the true MLE; in
   absolute β units that's of order `0.02 · β = 0.07`. The Turner+2024
   prior `β ~ N(3.62, 0.04²)` keeps β within a tight range regardless.
2. **The other 4 gradients (`dM`, `dlog_ω`, `dlog_c_0`, `dlog_τ_0`) are
   correct** to FD precision in v1. They drive the bulk of training; the
   slightly biased `dlog_β` is a small perturbation on top.
3. **MATLAB v1 production has been validated empirically.** The trained
   `β` value is consistent with Turner+2024 expectations.

## Why we keep v1 + MATLAB unchanged AND introduce v3.5

The user's plan for PR #6 is to anchor v3 to v1's behaviour exactly, so
that any divergence (e.g. from vectorization in Step B) is detectable.
Fixing `dlog_β` would break the v1 ≡ MATLAB anchor, defeating Step A.2.

Instead:

| variant | gradient | role |
|---|---|---|
| **v1**       | approximate (term A only) | Frozen reference. Anchors the historical training behaviour. |
| **MATLAB**   | approximate (term A only) | Gold-standard reference. v1 ≡ MATLAB to machine precision (Step A.2). |
| **v3**       | approximate (= v1)        | Verbatim copy + vectorization (Step B). Stays approximate so v3 ≡ v1 ≡ MATLAB throughout. |
| **v3.5**     | **strict (terms A + B)**  | New separate variant. Tested in parallel with v1/v3/MATLAB on Steps A.2/A.3/A.4 to measure the **scientific impact** of the approximation. |

## What the v3.5 patch does

`gpy_dla_detection/training_v3_5/objective.py` — copied verbatim from v1,
then a single localized patch:

1. Inside the Lyman-line loop, accumulate `chromatic_correction` =
   `Σ_{k>1} τ_k · log(r_k) · 𝟙_{k-forest}` alongside `τ_total`.
2. In `dlog_β`, replace
   `da_β = da_τ_0 · log(lya_1pz) · β · 𝟙`  with
   `da_β = ω² · scaling · exp(-τ_total) · (τ_total · log(1+z_lya) + chromatic_correction) · β · 𝟙`.

`nlog_p`, `dM`, `dlog_ω`, `dlog_c_0`, `dlog_τ_0` are byte-identical to v1
output by construction. Only `dlog_β` differs.

## Comparison plan across the remaining steps

**A.2 — MATLAB cross-check.** Two parallel comparisons:
   - **v1 ≡ MATLAB** on all 5 gradients (loss + dM + dlog_ω + dlog_c_0
     + dlog_τ_0 + dlog_β) to machine precision. Both share the
     approximation by design.
   - **v3.5 vs MATLAB**: matches on 4 gradients to machine precision;
     differs on `dlog_β` by 0.5–2.5 % — the empirical signature of
     term (B). Confirms our patch is consistent with the math derivation.

**A.3 — Short retrain.** Two parallel runs on the 1300-spectrum
   stratified 2lpt set:
   - v1 (approximate gradient) — reference behaviour
   - v3.5 (strict gradient) — does it converge to a different `β`?
     A different `(M, μ, log_ω)`? Different loss curve shape?
   Compare:
   - β trajectory over epochs
   - final β values (expect Δβ ≤ 0.04, the prior σ)
   - corr(M·M^T) shape (expect ≈ identical, since dM is unchanged)
   - χ²/n_valid + z-score on hold-out (sensitive to whether β shift
     matters for goodness-of-fit)

**A.4 — Canonical TID inference.** After A.3 trainings finish, run
   `examples/canonical_tid_per_model.py` under both v1 and v3.5
   weights. Expect both to detect (`p_DLA ≈ 1`); compare MAP NHI bias
   to truth (21.26).

**Decision after A.4:**
   - If v3.5 trains to the same β, gives the same DLA recovery, and
     the same canonical-TID detection — the approximation is benign.
     v1 / MATLAB stay as production reference; v3.5 is a curiosity.
   - If v3.5 trains to a notably different β (Δ > 0.04 prior σ), or
     detects DLAs better — fix v1 + MATLAB in a follow-up PR. The
     production model may need to be re-derived under v3.5.

## Key files

- v1 (frozen): `gpy_dla_detection/objective.py:182–189`
- MATLAB (frozen): `/home/mfho/MATLAB/gp_dla_detection_dr16q_public/spectrum_loss.m:91–95`
- v3 (verbatim copy of v1): `gpy_dla_detection/training_v3/objective.py`
- **v3.5 (strict): `gpy_dla_detection/training_v3_5/objective.py`**
- Test (v1 — approximate, dlog_β tol 5e-2): `tests/test_v1_spectrum_loss_jacobian.py`
- **Test (v3.5 — strict, uniform tol 1e-4): `tests/test_v3_5_spectrum_loss_jacobian.py`**

## Numeric Jacobian sanity (Step A.1) — both pass

```
v1   spectrum_loss   max rel_err  log_c_0/τ_0/ω, M : 5.37e-05  (tol 1e-4) ✓
v1   spectrum_loss   max rel_err  log_β APPROX     : 2.44e-02  (tol 5e-2) ✓
v3.5 spectrum_loss   max rel_err  ALL PARAMS       : 5.37e-05  (tol 1e-4) ✓
                     including log_β              : 6.37e-10
```

## Future-PR scope

If A.4 shows v3.5 helps, file a follow-up PR that:
- Lands the strict-gradient term in `gpy_dla_detection/objective.py` (v1)
  with a one-line patch matching `training_v3_5/objective.py`.
- Patches the MATLAB reference for symmetry (optional — depends on whether
  MATLAB is still being maintained).
- Documents the change in production-model retrain notes.
- Supersedes this finding doc with the resolution.
