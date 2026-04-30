# Bayesian-correctness synthesis — what we've tested, what's left, what to ship in PR #5

> Written 2026-04-29 EOD. Closes the question raised in
> `2026-04-27_bayesian_correctness_plan.md` ("is the multi-DLA Bayesian
> computation strictly correct?") with the test results that landed
> through 2026-04-29.
>
> Companion docs that contain the actual measurements:
> `findings.md`, `CURRENT_STATE.md`, `scale_out/summary_n54.csv`,
> `2026-04-25_filter_samples_sweep.md`, `2026-04-27_filter_completeness_explanation.md`,
> `2026-04-27_lybeta_persistence_hypotheses.md`,
> `2026-04-27_subdla_model_improvements.md`,
> `2026-04-27_london_pdla_scan_no_bal.md`,
> `2026-04-27_london_postprocess_p99_no_bal.md`.

---

## TL;DR

Of the four candidate causes for the historical +0.34 dex DLA bias and
spurious-Lyβ rate flagged in the original plan, **one was confirmed
(τ_eff prior mismatch — formally a *forward-model* defect, but distinct
from the LSF/num_lines hypothesis the original plan focused on), one
was confirmed for a *different regime* (DLA prior boundary at 20.3
drives sub-DLA / LLS bias), and two were ruled out** (LSF kernel,
num_lines, QMC density at high NHI). Sample density at the *prior
boundary* (sub-DLA / LLS regime) is still untested.

**The DLA-regime bias is closed to ~0 dex on n=18 targets across 3
mocks** (81% closure of median bias) with HCD-masked τ-EB (recipe
landed). The sub-DLA / LLS regime needs a separate fix (tasks #6/#7/#11)
and is **out of scope** for PR #5.

**Recommended PR #5 scope**: ship the τ-EB fix (production integration
+ the FILTER fix #5 already landed) and stop. The remaining ~0.05 dex
residual after τ-EB has multiple plausible causes that are individually
small; pinpointing them is a follow-up PR after we see retrained-GP
results.

---

## Hypothesis ledger

Five hypotheses were on the table. Numbers refer to the plan in
`2026-04-27_bayesian_correctness_plan.md` plus the τ_eff hypothesis
that was added after the original plan when the LSF tests came back null.

### H1 — LSF kernel mismatch (BOSS shape on DESI grid)
**Test**: Re-ran the full mock-target sweep with kernels A=BOSS-log-R2000,
B=DESI-linear-R3000, C=DESI+6-lines, D=no LSF (commits `3731b8c`,
`a052b50`, `de39e40`).
**Verdict**: **RULED OUT** for production DESI inference. DLA-regime MAP
is bit-identical across A/B/C/D — the saturated DLA core is wider than
any of these kernels. The earlier "R≈21000 over-sharpening" claim was a
grid-confusion bug on my side: production convolves on the DESI 0.8 Å
observed grid, not the 0.15 Å GP rest grid, so the production C-extension
kernel σ_eff = 0.49 Å gives R_eff ≈ 3400 (close to intended R=3000).
*Caveat*: kernel **does** matter in the sub-DLA / LLS regime (config-spread
0.029 → 0.384 dex on saclay 2385001246 between buggy and fixed kernel) —
but that regime's bias is dominated by H5 (prior boundary), so this is a
secondary effect.

### H2 — num_lines too few (Lyβ + Lyγ contribution underweighted)
**Test**: Configs B (3 lines) vs C (6 lines) in the same sweep.
**Verdict**: **RULED OUT** for DLA-regime targets. Bit-identical MAP at
n=3 vs n=6.
*Caveat*: The Lyβ misID rate question (lybeta_persistence) is separate —
that's about whether the *integration* over (z₂, NHI₂) gives spurious
2-DLA evidence at z_lyb, not whether the per-sample log-likelihood is
miscomputed. Currently mitigated by the `lyb_veto` postprocessor.

### H3 — QMC sample density at high NHI (sparse coverage near truth)
**Test**: Brute-force scan over 20k randomly-drawn samples from the
prior + targeted sample at exact truth NHI=21.263 (commit `e1cc94f`).
**Verdict**: **RULED OUT** for the canonical target's DLA component.
The sample at truth's exact NHI exists in the 100k set at Δ=0.0; the
brute-force MAP still lands at NHI=21.547 (+0.28 dex). This is not a
sampler-density problem — the GP+forward model is genuinely happier at
NHI=21.55 than at NHI=21.26 on this spectrum.
*Untested*: Sample density at the *sub-DLA prior boundary* (20.0 edge)
where MAP snaps to 20.3 for many sub-DLA / LLS truths. This is plausibly
a density problem but is conflated with H5.

### H4 — DLA prior pile-up at log_nhi=20.3 (Ho+2020 mixture α=0.97)
**Test**: Not yet tested as an ablation (would need to re-run with
α=0.3 and see whether spurious 2-DLA rate moves).
**Verdict**: **PARTIALLY CONFIRMED for sub-DLA / LLS regime, untested
for spurious 2-DLA**. The `uniform_min_log_nhi=20.0` boundary on the
DLA model forces sub-DLA / LLS truths to MAP=20.3, producing the
+0.5 to +2.7 dex biases observed in the n=54 scale-out (rows 8–19,
26–37, 44–55 in `scale_out/summary_n54.csv`). HCD-masked τ-EB does
**NOT** close these — they're orthogonal.
For 2-DLA spurious-Lyβ specifically: untested. The `lyb_veto`
postprocessor catches them empirically; whether prior reshape would
also help is open.

### H5 — τ_eff prior mismatch (Turner+2024 hardcoded τ_0=0.00246)
**Test added 2026-04-29 after H1/H2 came back null.** Fine NHI grid
scan at fixed truth z, sweeping τ_factor 0.25× → 3.0× production
(commit `de509f2`). EB vs full-marginalization (`caee3ed`). HCD-masked
EB single-target (`ca9dc8c`). Multi-target n=6 (`e9987db`). Scale-out
n=54 (commit `28231e7`, job 49020191).
**Verdict**: **CONFIRMED LEVER** and the dominant cause of DLA-regime
bias. Naive EB closes ~30% (HCD pixels look like extra forest absorption
to the τ fitter and hold τ low). HCD-masked EB closes 81% of median
DLA-regime bias (n=18 across 3 mocks: +0.240 → +0.045 dex).
Per-mock closure: 2LPT 54%, London 84%, Saclay 81%. The recipe matches
the standard Becker / Faucher-Giguère mean-flux convention.
*Caveat*: One of the n=6 targets (160089646) had production-bias
−0.34 dex which τ-EB pushed to −0.44 dex (got worse). Suggests τ-EB
isn't a uniformly safe transform; the residual scatter on individual
targets is non-trivial even when median closes well.

### H6 — finite-sample QMC integration noise (the Step 4 plan)
**Test**: Not done. Would require swapping integration estimator
(harmonic mean / importance sampling / nested sampling) on the same
spectra and seeing whether the spurious-Lyβ rate moves.
**Verdict**: **UNTESTED**. The original plan called for this as a
last-resort discriminator; current evidence says H5 is dominant for the
DLA regime, so this hasn't been prioritized. User has flagged this as
the long-run direction (custom nested sampler) — separate from PR #5.

### H7 — multivariate-Gaussian residual assumption (user's hypothesis)
**Verdict**: **UNTESTED**. Plausible candidate for the residual ~0.05
dex bias after τ-EB on the canonical target. Would need swapping the
likelihood for Student-t or Huber and checking whether the bias shrinks.
Non-trivial implementation; not in PR #5 scope.

### H8 — training data contamination (FILTER misclassification, residual HCDs in trainset)
**Test in flight**: NERSC training jobs `loa_no_dla_no_bal_52198069`
and `loa_no_hcd_with_bal_52198070` retrain the GP on truly-clean
forest after rejecting all known DLAs/HCDs. GreatLakes equivalents
just submitted (jobs 49037617, 49037618).
**Verdict**: **PENDING**. Will let us test whether retraining moves
the residual ~0.05 dex bias.

---

## Decision: what should ship in PR #5?

PR #5 has been about the LSF/Voigt question; that's now resolved (H1, H2
RULED OUT). The session shifted to τ_eff (H5), which is what's actually
fixing the bias. The current branch state:

- 22 commits since base
- HCD-masked τ-EB recipe validated at n=54 scale (81% median closure)
- FILTER fix #5 landed (matches FILTER=0 baseline to 0.7%, 10× speedup)
- voigt_v2 module + retraction note for the LSF claims
- Production integration of τ-EB **NOT YET DONE** (recipe lives in `examples/`)

### Three shipping options

**Option A — minimal**: Land what's already on the branch, document the
findings, leave production integration to a follow-up PR. Ships the
diagnostic tools and the FILTER fix; researchers can apply τ-EB
manually via `examples/check_tau_eb_robust_mask.py`.
- Pro: PR is reviewable today, FILTER fix #5 alone is worth landing.
- Con: production runs still use the biased τ_0 — the science result
  ("we know how to fix this") doesn't reach users.

**Option B — recommended**: Add τ-EB to the production inference path
behind a flag (`--enable-tau-eb-hcd-mask`, default OFF for backward
compat). FILTER fix #5 stays as-is. voigt_v2 stays as a research module.
- Pro: Production runs can opt into the fix immediately. Default-off
  means existing pipelines don't change behavior. The 81% closure
  result becomes actionable.
- Con: ~K× cost when enabled (estimated K=4–6 in current recipe; can
  reduce to K=3 if we pick {1.0, 2.0, 3.0} instead of full grid).
- Risk: the n=54 scale-out shows non-trivial per-target scatter, so
  blanket enabling is not yet safe — flag-default-OFF mitigates.

**Option C — broader**: Also add some sub-DLA / LLS work (FILTER fix
#5 already helps; add e.g. extending the DLA prior to log_nhi ≥ 19.5
to bypass H4 boundary). Bigger PR; more review surface.
- Pro: Closes more of the bias surface in one merge.
- Con: Sub-DLA improvement is genuinely a separate problem with its
  own design choices (extend prior? add a 19→20.3 model? add a
  posterior cut?) — coupling it to PR #5 expands review scope.

**My recommendation: Option B.** The τ-EB fix is a single, well-tested
change with a clear flag-gate. Sub-DLA work belongs in a separate PR
once we've decided on the design (the user's pinned project memory
`feedback_dla_prior_edge_bias.md` describes "extend prior + post-cut"
as the working approach — that's a separate ticket).

### What "Option B" means concretely

1. New module `gpy_dla_detection/tau_eb.py` — pure function
   `fit_tau_eb_hcd_mask(params, prior, learned_file, wave, flux, nv,
   mask, z_qso, *, prev_tau_0, prev_beta, tau_factors, mask_threshold_sigma)
   -> float (best_tau_0)`. Encapsulates the recipe.
2. Modify `run_bayes_select.DLAHolder.__init__` to accept
   `enable_tau_eb_hcd_mask: bool = False`, `tau_eb_factors: list = (0.5,
   1.0, 1.5, 2.0)`, `hcd_mask_threshold_sigma: float = 1.5`.
3. Modify `DLAHolder.process_qso` to call `fit_tau_eb_hcd_mask` per
   spectrum (when flag is set) and override `prev_tau_0` for the
   inference build.
4. CLI flag in `desi-DLAGP.py` and `dlasearch.py`.
5. Tests: parity test (flag off → identical to current), recipe
   reproducibility test on a frozen-input case.
6. Doc: `docs/tau_eb_hcd_mask.md` — explanation + figure (this is
   task 2 from the user's request).

### What stays out of PR #5

- Sub-DLA / LLS prior reshape (separate PR after design discussion).
- H6 (integration-method swap) — long-run sampler work; user has flagged
  as separate direction.
- H7 (Student-t residuals) — speculative, untested, would substantially
  complicate the likelihood code.
- H8 (retrained GP results) — in flight; PR can land before training
  completes since the τ-EB fix is orthogonal to the trained model.

---

## Recommended next step (one action)

**Implement Option B end-to-end on this branch** (task 3 in the user's
list). The τ-EB recipe is the highest-leverage change still on the
runway, and shipping it through the production CLI with a flag is the
right way to make the n=54 result actionable for downstream science
without forcing it on existing pipelines.

After that lands, PR #5 should be reviewable. The training results and
sub-DLA work can be separate follow-ups.
