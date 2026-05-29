# β-drift investigation

## Summary

The Step C PR #6 trainers converge to β ≈ 2.6–3.1 (not the Turner+2024 prior μ = 3.62) because the **data genuinely prefers a lower β** under the published Turner statistical uncertainty σ = 0.04 — there is no bug in the prior implementation. The prior gradient is applied **once per Adam iteration** at `tests/phase2_train_desi.py:255-256`, which exactly matches MATLAB DR16 (`/home/mfho/MATLAB/gp_dla_detection_dr16q_public/objective.m:69-77`) and v1 production (`/home/mfho/desi_gpy_dla_detection/gpy_dla_detection/objective.py:70-71`). Exponential fits to the last 1000 iters of the four 2lpt trajectories give asymptotic β_∞ ∈ [2.80, 3.46] with time constants τ ∈ [1194, 2160] iter — 1500 iters is within ~0.1–0.4 of asymptote; even running to 5000+ iters would not produce β = 3.62. **Verdict: hypothesis (b)**, with small contribution from (c) (trajectory still climbing at iter 1500, but toward an asymptote ≠ 3.62 — not toward μ).

## Prior gradient implementation

`/home/mfho/desi_gpy_dla_detection/gpy_dla_detection/training_v3/objective_vectorized.py::spectrum_loss_batch` (lines 36–182) returns **data-only** gradients — there is no prior term inside this function. The docstring at lines 78–86 lists only data-likelihood outputs.

The Turner+2024 prior is added **once per Adam step** at `tests/phase2_train_desi.py:253-256`:

```
253:        # Turner+2024 priors on (τ_0, β). dlog_τ_0 += τ_0 (τ_0 - μ)/σ²
254:        # follows from chain rule on log-parameter prior.
255:        dlog_tau_0_acc = dlog_tau_0_acc + tau_0 * (tau_0 - TAU_0_PRIOR_MU) / TAU_0_PRIOR_SIGMA**2
256:        dlog_beta_acc = dlog_beta_acc + beta * (beta - BETA_PRIOR_MU) / BETA_PRIOR_SIGMA**2
```

Loop structure at `phase2_train_desi.py:214-270`:
- L229–251 — chunk loop accumulates *only data-likelihood gradient*.
- L253–261 — prior gradients added once.
- L263–268 — gradients written to `.grad`.
- L270 — single `optimizer.step()`.

**The prior is added once total, not N times, not zero times.** Each quantity at the call site is a scalar; there is no `for spectrum:` loop around lines 255–256. Hypothesis (a) is **falsified**.

## Effective prior strength at iter 1500

Four 2lpt models with histories, all n_spectra = 203,984, n_iter = 1500, lr = 0.005, σ_β = 0.04:

| Model        | β endpoint | β − μ_β  | d(NLL_prior)/d(log_β) = β(β−μ)/σ² |
|--------------|-----------:|---------:|----------------------------------:|
| loa0_wide_g  | 2.6912     | −0.9288  | −1562.2                           |
| loa0_wide_m  | 3.0862     | −0.5338  | −1029.6                           |
| loa124_g     | 2.5669     | −1.0531  | −1689.5                           |
| loa124_m     | 2.9694     | −0.6506  | −1207.4                           |

**Step ratio at iter 1499** — gold-standard near-equilibrium diagnostic for Adam:

| Model        | Δlog_β/iter (last) | / lr (=0.005) |
|--------------|-------------------:|--------------:|
| loa0_wide_g  | +9.3e-5            | 1.86 %        |
| loa0_wide_m  | +5.3e-5            | 1.06 %        |
| loa124_g     | +9.0e-5            | 1.80 %        |
| loa124_m     | +6.0e-5            | 1.20 %        |

Adam's effective step is `lr · m̂ / (√v̂ + ε)`. Step ratio ≈ 1 % means gradient is being suppressed by competing signals — **the signature of a near-Bayesian-equilibrium where data and prior gradients largely cancel**. The prior gradient at endpoint is ~1000–1700 in d/d log_β; the data gradient is approximately +1000–1700 (opposite sign). **The prior is NOT being washed out** — it contributes ~50 % of the gradient magnitude at the endpoint.

## MATLAB comparison

`/home/mfho/MATLAB/gp_dla_detection_dr16q_public/objective.m:42-77`:

```
42:  for i = 1:num_quasars
...
56:    f               = f               + this_f;
...
61:    dlog_beta       = dlog_beta       + this_dlog_beta;
62:
63:  end
...
65:  % apply prior for τ₀ (Kamble, et al. 2019) BOSS DR12Q prior
69:  dlog_tau_0 = dlog_tau_0 + ...
70:      tau_0 * (tau_0 - tau_0_mu) / tau_0_sigma^2;
...
73:  beta_mu     =   3.182;
74:  beta_sigma  =   0.074;
76:  dlog_beta = dlog_beta + ...
77:      beta * (beta - beta_mu) / beta_sigma^2;
```

And `gpy_dla_detection/objective.py:44-71`:

```
44:    for i in range(len(fluxes)):
...
60:        dlog_tau_0_accum += dlog_tau_0.detach()
61:        dlog_beta_accum += dlog_beta.detach()
...
63:    # ✅ Apply **DESI Y1 Prior** for log_tau_0 and log_beta
64:    tau_0_mu = 0.00246  # DESI Y1 Mean τ₀
65:    tau_0_sigma = 0.00014  # DESI Y1 Std τ₀
66:    beta_mu = 3.62
67:    beta_sigma = 0.04
...
70:    dlog_tau_0_accum += (tau_0 - tau_0_mu) / tau_0_sigma**2 * tau_0
71:    dlog_beta_accum += (beta - beta_mu) / beta_sigma**2 * beta
```

All three apply the prior **once total per gradient step**. No discrepancy.

## Trajectory shape

β trajectories are NOT monotonically rising. Two-phase structure (from `log_beta_history` arrays):

| Model        | Phase 1 (descent)            | Phase 2 (recovery)       | β_∞ (3-param exp fit) | τ (iter) |
|--------------|-----------------------------:|-------------------------:|----------------------:|---------:|
| loa0_wide_g  | β: 3.62 → 2.30 at iter 314   | climbs back to 2.69      | 2.97                  | 1313     |
| loa0_wide_m  | β: 3.62 → 2.77 at iter 193   | climbs back to 3.09      | 3.46                  | 2160     |
| loa124_g     | β: 3.62 → 2.23 at iter 367   | climbs back to 2.57      | 2.80                  | 1194     |
| loa124_m     | β: 3.62 → 2.65 at iter 220   | climbs back to 2.97      | 3.26                  | 1708     |

**Phase 1** (iter 0 → ~200–370): data gradient at PCA-init dominates → β plummets ~30–40 % below μ. Adam momentum carries β past its equilibrium.

**Phase 2** (iter ~300 → 1500): prior dominates the local imbalance → β climbs back. Exponential approach `β(t) = β_∞ − Δ · exp(−t/τ)` fit on last 1000 iter gives β_∞ ∈ [2.80, 3.46] across the four models. **None reach 3.62 even at t = ∞.** This matches v1 production's β = 2.41 on real LOA — independent corroboration that real data + 2lpt mock both prefer a lower β under the Turner prior.

## lr probe / cosine

At iter 1500 step ratio Δlog_β / lr ≈ 1 %. Adam in steady state delivers step magnitude at most lr per iter. Doubling lr → 0.01 would speed up Phase 1 and Phase 2 by ~2×, but **the asymptote β_∞ is set by gradient balance (data + prior), not by lr**. β_∞ is an attractor of the gradient flow; lr changes *speed*, not *position*.

Cosine decay would only **hurt** — as lr decays, Phase-2 climb slows monotonically, so at fixed iter budget the model lands at a *lower* β than with constant lr.

Mechanisms that would actually raise β_∞ toward 3.62:
1. Tighter prior σ (e.g. σ_β = 0.013, 3× tighter → 9× more pull would force loa0_wide_m's β_∞ ≈ 3.55). Violates Turner+2024 published σ.
2. Anchor c_0 (currently drifts 0.1 → 0.02; couples with β through `(c_0, M)` factor-analysis degeneracy documented at `phase2_train_desi.py:91-98`). The `--log-c-0-prior-sigma` knob at `phase2_train_desi.py:604-608` already exists for this.
3. Different prior μ (accept that Turner μ may not match 2lpt mock effective τ).

## Hypothesis verdict

**(b) data genuinely prefers β < 3.62**, with small partial contribution from (c).

1. Three independent implementations (MATLAB DR16 `objective.m:69-77`, v1 `gpy_dla_detection/objective.py:70-71`, PR #6 `tests/phase2_train_desi.py:255-256`) apply the prior identically: once per gradient step, `β(β−μ)/σ²` form. **(a) falsified.**
2. Trajectory shape is overshoot → asymptotic recovery, asymptote β_∞ ∈ [2.80, 3.46]. None reach 3.62. (c) is partially true — climb continues at iter 1500 — but extrapolation shows the plateau is below 3.62.
3. v1 production on real LOA → β = 2.41 with the same prior. Independent confirmation that real spectra also prefer lower β.
4. Step ratio 1–2 % of lr at iter 1500 = signature of near-equilibrium where data and prior largely cancel. If prior were broken, step ratio would be ≈ 1; if over-weighted, β would be pinned at exactly 3.62.

## Recommendations

1. **Accept β < 3.62 as a Bayesian feature**, not a bug. The Turner prior is informative; the data is informative too; the posterior compromise lies below μ. This is the correct behavior.

2. **Document the asymptote in the model card** emitted by `_save_readme` (`phase2_train_desi.py:310`). Add 3-param exp fit of β(t) over last 1000 iter and report β_∞. Note: "β_∞ < β_prior_μ is expected under σ_β = 0.04. v1 production on real LOA reproduces β = 2.41 under the identical prior."

3. **Run a sanity probe**: train with σ_β = 0.005 (impossibly tight) and verify β converges to 3.62 ± 0.01. Bounds the data-pull magnitude and confirms prior implementation is healthy.

4. **Try `--log-c-0-prior-sigma 0.1`** in next production retrain. c_0 drifts 0.1 → 0.02 (5× drop) and may be coupling with β. Anchoring c_0 may shift β_∞ closer to μ.

5. **Do not switch to cosine decay** (would push β_∞ lower). For faster convergence: double lr to 0.01 with brief warmup; halves τ from ~1500 → ~750 iter without moving β_∞.

6. **Do not "fix" the prior by dividing by N** — that would be the bug, not the absence of it. `prior_grad = β(β−μ)/σ²` added once is mathematically correct for loss = `−∑ᵢ log p(yᵢ|θ) + 0.5(θ−μ)²/σ²` (negative log posterior). Standard MAP-via-NLL convention; matches MATLAB DR16 and v1 production.
