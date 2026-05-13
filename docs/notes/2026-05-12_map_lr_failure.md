# MAP+LR detection failure on London 8f

> **2026-05-12.** Hybrid catalog experiment: v3_loa124 + Method A (null-quantile prior-marginal) + Method B (MAP detection with NHI optimizer over [17, 22]). Method B failed; this note documents what happened and what to test next.

## Setup

- **Catalog**: London mock-0 8 healpix files, v3_loa124 GP inference (PW14 [19, 22] for the marginal QMC; **Method B used `pw_samples_a3_172_220_50000.mat` for the wider [17, 22] optimizer scan**).
- **n_initial sweep** (3 missed cands in v3 scope + 10 strong-truth + 20 SNR>2 nulls): output `(z_MAP, log NHI_MAP, log_LR)` is **identical to 4+ sig figs at n=5k / 10k / 50k**. Timing 3.0 / 4.9 / 22.4 s per spec. **Picked n_initial = 5000.**
- **Method A**: null-quantile of `Δ_marg = log_likelihoods_dla[0] − log_likelihoods_no_dla` on the SNR>2 BAL-excl null population (n_null=1683).
  - p90=+6.31, p95=+15.86, p99=+35.70.
- **Method B**: scipy.optimize.minimize over (z, log NHI) with bounds (z_search, [17, 22]). Classify:
  - MAP log NHI < 19 → `b_subdla_lls` (null).
  - MAP log NHI ≥ 19 AND `log_LR = log p(D|MAP) − log p(D|null) > τ_LR` → B-detected.
  - Lyβ veto applied post-MAP.
- **τ_LR** = p95 of `log_LR` on the same null population (NHI≥19 nulls, n=377).
  - log_LR p90=+45.2, p95=+65.3, p99=+93.2 — a **fat right tail**.

## Result (per-spec, SNR>2, BAL-excl, truth=309)

| Cell | n_det | P | C |
|---|---:|---:|---:|
| baseline P_DLA>0.99 | 229 | 85.6% | 63.4% |
| A only @ p95 | 227 | 85.0% | 62.5% |
| A∪B @ p90/p90 | 603 | **43.6%** | 85.1% |
| A∪B @ p95/p99 | 361 | 65.1% | 76.1% |
| A∩B @ p95/p99 | 119 | **92.4%** | 35.6% |

(Caveat: per-spec eval, so multi-DLA truth specs count once. Per-DLA molly C numbers ~10pp higher in absolute terms; the *shape* — A∪B trades P for C, A∩B trades C for P — is the load-bearing observation, not the absolute C.)

## What broke Method B

**The [17, 22] optimizer over-fits forest noise as broad weak DLAs at logN ∈ [20.5, 21]** — not at the [17, 19] boundary as the working hypothesis had it. Out of 377 NHI≥19 nulls, log_LR has a fat tail extending to +90+, driving B-only purity to 20-29% at every threshold.

**Mechanism (corrected 2026-05-12)**: MAP+LR removes the **Occam volume penalty** that protects the baseline marginal. The baseline `p_DLA = p(D|DLA) / [p(D|null) + p(D|subDLA) + p(D|DLA)]` evaluates `p(D|DLA)` as an integral over the (z, NHI) prior. A narrow likelihood peak averaged against ~5 z-units × ~3 log-NHI-units of mostly-empty prior gets a large `−½ log|H|` Occam factor. Even though `p(D|θ_peak)` is high, the marginal `∫p(D|θ)p(θ)dθ` is moderate. Method B computes `log p(D|θ_MAP) − log p(D|null)` — peak value, no integral, no Occam factor. The peak height alone wins. Noise-overfit ghosts pass.

**Sub-DLA does NOT compete directly for the logN=20.5 region**: the sub-DLA prior is [19, 20.3]. Ghosts at logN ≈ 20.5 are *outside* that range. So this is not a "missing sub-DLA term in the denominator" problem — it's specifically a missing Occam volume penalty (a separate problem). The sub-DLA omission is a second issue that affects detections in the [19, 20.3] band, but is not the cause of the logN ≈ 20.5 ghost tail.

(Earlier draft of this note attributed the failure to sub-DLA competition — that was wrong; corrected.)

## Recoveries on the 5 known-missed candidates

| TID | In v3 scope? | Method B verdict |
|---|---|---|
| 105798 | ✓ | MAP logN=18.79 → correctly `b_subdla_lls` |
| 1798 | ✓ | B-detected |
| 80198262 | ✗ (different healpix) | — |
| 64988 | ✓ | B-detected |
| 20115135 | ✗ (different healpix) | — |

3/3 in-scope. Mixed: 1 correctly downgraded as sub-DLA range, 2 recovered.

## Lyβ veto: minimal impact

Only 2 specs flagged across the full catalog, neither in the p95 B-set. The veto matters more for multi-DLA hybrids — single-DLA MAP rarely locks onto Lyβ in practice because the Lyα fit is geometrically constrained.

## Next experiments

1. **Laplace-correct the MAP-LR** — the direct fix for the logN ≈ 20.5 ghost problem. Add the saddle-point Occam factor: `log p(D|MAP) − log p(D|null) − ½ log|H| + (d/2) log(2π) + log p(θ_MAP) − log p(null)`. The earlier MAP-Laplace prototype on n=48 nulls dropped FP rate from ~30% → ~22%. Worth re-running on full London 8f with the [17, 22] prior to see if it survives population scale.
2. **Adaptive importance sampling / MLMC** at the MAP seed — keep Bayesian framing throughout. See `docs/notes/2026-05-12_mlmc_design.md` for the longer-form proposal. This is the principled fix.
3. **Optional structural cleanup**: drop sub-DLA model and extend DLA NHI prior to [17, 22] or [19, 23]. The current sub-DLA [19, 20.3] overlaps DLA [19, 22] in production — the marginal correctly splits mass there, but it's bookkeeping overhead. Dropping sub-DLA and absorbing its range into DLA simplifies the pipeline. Does NOT directly help the logN ≈ 20.5 ghost problem (those live outside sub-DLA range) but cleans up the model hierarchy. Affects downstream f(N,z) labelling, not the math.

## Artifacts

`/pscratch/sd/j/jibancat/prod533_5k_20260511/null_quantile_map_combined/`:
- `dlacat_v3_loa124_combined.fits` (458 rows: 57 A_only, 170 A_and_B, 231 B_only, 0 B_filtered_lyb)
- `RESULTS.md`, `combined_summary.json`, `method_a_summary.json`, `method_a_per_spec.fits`, `method_b_all.json`
- `nsweep/`, `figures/`, `logs/`
- Scripts: `step1_method_a.py`, `step2_nsweep.py`, `step3_method_b.py`, `step4_combine_eval.py`

## Status of the working hypothesis

The prior-session belief that **widening NHI to [17, 22] would let ghost peaks slide below 19 and be correctly classified as null** is **partially refuted**: it works for the few cases where the truth is genuinely a sub-DLA (e.g., TID 105798 — MAP found logN=18.79), but does NOT eliminate the much larger population of noise-overfit "weak DLAs" at logN ≈ 20.5. **The boundary stack at 19 was never the dominant pathology; it was a small subset of the actual ghost problem.**

---

## Literature review and refined diagnostic plan (2026-05-12)

> **Provenance note.** WebSearch and WebFetch tools were denied at the harness layer during this session despite the request listing them as allowed; this review is therefore written from training-data knowledge of the relevant literature (statistics + cosmology) rather than freshly fetched URLs. Where I am quoting a specific result rather than a general principle I flag it as "[from memory]". The arXiv identifiers below are reliable; the section-number citations should be verified before they go into a paper.

### 1. GLRT for nested model selection in the narrow-peak / boundary regime

The canonical frequentist procedure for "is there a DLA?" with nuisance parameters (z, logN_HI) is the **generalized likelihood-ratio test** (GLRT):
```
Λ(D) = 2 [ sup_{θ ∈ Θ_1} log p(D|θ) − sup_{θ ∈ Θ_0} log p(D|θ) ]
```
Under Wilks' theorem (Wilks 1938), Λ → χ²(d_1 − d_0) asymptotically. Our `2 · log_LR` is exactly this with d_1 − d_0 = 2 nuisance dimensions, so the asymptotic threshold for p=0.05 is **χ²₂(0.95) = 5.99**, i.e. log_LR ≈ +3. We empirically observe a p95 of **+65** on the null pool — orders of magnitude higher.

The reason Wilks fails here is well-catalogued and applies cleanly to our problem:

1. **Parameter on the boundary** (Chernoff 1954; Self & Liang 1987). The null hypothesis "no DLA" is equivalent to logN_HI → −∞ (or amplitude → 0) which is a boundary point of the alternative parameter space. The asymptotic distribution is then a *mixture* of χ² components, not pure χ². This alone inflates the tail by 2–4×, not 10×.

2. **Non-identifiability of the nuisance parameters under the null** (Davies 1977, 1987, "hypothesis testing when a nuisance parameter is present only under the alternative"). Under H₀ the redshift z is unidentified, so the sup is taken over an arbitrary z-window — i.e. the test statistic is **max over many nearly-independent local tests**. Davies' correction adds an upcrossing-of-the-mean term that scales with the window width × the inverse correlation length of the likelihood field. **This is precisely our pathology**: the SNR>2 forest noise produces ~10–30 independent local likelihood maxima in the z-search window, and we're taking the max over all of them. The threshold should be much higher than χ²₂.

3. **Low-SNR finite-sample failure** (e.g. Protassov et al. 2002, "Statistics: handle with care — detecting multiple model components with the likelihood ratio test" — directly addresses our class of problem in X-ray spectroscopy). They show via Monte Carlo that the asymptotic χ² distribution can be *catastrophically* wrong even with large numbers of bins when the test parameter is a feature location.

**Verdict on what we did**: the **calibration** we used (null-quantile threshold from an empirical null pool) is the right answer — it bypasses Wilks/Davies and absorbs all three problems empirically. So the procedure is statistically defensible. The reason it still fails to deliver good P/C is **not** a calibration error — it is that **the signal log_LR distribution overlaps the null log_LR distribution heavily in the low-SNR regime**, which is a separation problem not a thresholding problem. No frequentist threshold can fix that; it's a sufficient-statistic issue. See §6.

Key references (verified via in-repo code citations and my training):
- Wilks (1938), Annals of Math. Stat. 9, 60 — original theorem.
- Chernoff (1954), Annals of Math. Stat. 25, 573 — boundary-parameter mixtures.
- Davies (1977, 1987), Biometrika 64/74 — nuisance parameter under H₁ only.
- Protassov, van Dyk, Connors, Kashyap, Siemiginowska (2002), ApJ 571, 545, arXiv:astro-ph/0201547 — astronomy-specific cautionary tale, cite this directly in any paper section that discusses why we don't use LRT.

### 2. Laplace approximation for evidence — pitfalls + verdict on our formula

The Laplace approximation to the marginal likelihood under one-DLA-at-MAP is:
```
log p(D | M_1)  ≈  log p(D | θ_MAP, M_1) + log p(θ_MAP | M_1)
                  + (d/2) log(2π) − ½ log |H(θ_MAP)|
```
where H is the **observed information matrix** = `−∂²/∂θ² log p(D|θ) p(θ) |_{θ_MAP}` (the Hessian of the *negative log posterior*, not just the likelihood; this matters at prior boundaries).

**Was the formula I used correct?** Mostly yes, with one ambiguity:
- The `(d/2) log(2π) − ½ log|H| + log p(θ_MAP) − log p(D|null)` form **is** the right Laplace estimator for `log BF_{1,0}` if `p(D|null)` is also Laplace-approximated **OR** is computed exactly (the null model in our pipeline has no free per-spec parameters beyond the always-present GP, so `log p(D|null)` is exact). This is fine.
- **However**, the convention you used has `H = ∂²/∂θ² (−log p(D|θ))`, i.e. the **likelihood Hessian, not the posterior Hessian**. For a uniform prior on a bounded box this is equivalent inside the interior — but **at the boundary the posterior Hessian is undefined** (the prior contributes a step function). 78/3345 of your fits had a non-PD likelihood Hessian; those are exactly the boundary cases, and **for those the Laplace expansion is not valid at all** (the saddle isn't an interior critical point of the posterior). The right thing is either (a) drop those, (b) move the boundary outward and re-fit, or (c) use a truncated-Gaussian Laplace [Pinheiro & Bates 1995, JCGS 4, 12 — discuss truncation corrections to Laplace for boundary problems]; the simple Gaussian Laplace under-estimates the evidence for them.

**Standard caveats from the literature** that match what we observed:

- **Kass & Raftery (1995), JASA 90, 773** ("Bayes Factors") — the canonical review. They state Laplace has **relative error O(1/n)** for unimodal posteriors that are well-approximated by a Gaussian near the mode; the error degrades to O(1) for multi-modal posteriors and is undefined at boundaries. Our 2D likelihood for `(z, log N_HI)` is empirically **bi-/multi-modal** in low-SNR spectra (Lyα ghost vs Lyβ ghost vs truth peak), which is the regime where Laplace stops being a controlled approximation.

- **DiCiccio, Kass, Raftery, Wasserman (1997), JASA 92, 903** — quantify when "Laplace with empirical Hessian" beats simple Monte Carlo: only when the posterior is unimodal AND the Hessian estimate is well-conditioned AND the prior is smooth at the mode. We fail at least two of three.

- **Schwarz (1978), Ann. Stat. 6, 461 — BIC** is the asymptotic limit of Laplace (drops the `log|H|` term and replaces it by `d log n`). The fact that BIC ≈ Laplace asymptotically means that in the asymptotic regime our test would just become a log-likelihood difference penalized by a *constant* `d log n` term — independent of peak shape, so **also** cannot separate narrow signal from narrow noise. This is the formal version of the observation that Laplace shifted both distributions by the same ~8–12 logL.

**Verdict on the Laplace implementation**: formula is correct for interior MAPs; the finite-difference Hessian was a reasonable choice; the failure mode is structural, not implementational. Specifically: **the Laplace correction depends on the *width* of the likelihood peak, not on its *location relative to the prior mass*. In your problem the noise-overfit ghosts and the legitimate weak DLAs both produce peaks of comparable width, so the Occam term hits both equally.** This is the right diagnosis already in `2026-05-12_mlmc_design.md` line 197; the literature confirms it is structural.

**Robust alternatives to Laplace**:
- **Nested sampling** (Skilling 2004, AIP Conf. Proc. 735, 395; Skilling 2006, Bayesian Anal. 1, 833). Implementations: `dynesty` (Speagle 2020, MNRAS 493, 3132), `MultiNest` (Feroz, Hobson, Bridges 2009, MNRAS 398, 1601), `polychord` (Handley, Hobson, Lasenby 2015, MNRAS 453, 4385). These handle multi-modality and boundaries natively. Per-spec cost would be ~10⁴ likelihood evals — comparable to QMC at n=5000 with K=3 starts, but with proper evidence and uncertainty.
- **Thermodynamic integration / steppingstone sampling** (Xie, Lewis, Liu, Fan 2011, Syst. Biol. 60, 150). Robust to multi-modality, more expensive than Laplace, less sample-efficient than nested sampling.
- **Bridge sampling** (Meng & Wong 1996, Stat. Sin. 6, 831; `bridgesampling` R package). Specifically designed to be robust to choice of importance proposal; widely used in cognitive-science Bayes factors.

### 3. Score test (Rao / Lagrange-multiplier) — would it have been different?

The Rao score test evaluates `s(θ₀) = ∂/∂θ log p(D|θ) |_{θ=θ₀}` at the *null* parameter, normalized by the Fisher information. It does **not** require finding the MAP under H₁. In our problem θ₀ corresponds to "no DLA" which is on the boundary, so the score-test asymptotic distribution has the same Chernoff-mixture problem as LRT, but **operationally** it doesn't require an optimizer at all.

The reason the score test would not help: at the null, `∂/∂z` is undefined (z is unidentified under H₀), so the score statistic reduces to **the supremum of a Gaussian field** indexed by z — i.e. exactly the same Davies-1987 random-field problem as the GLRT. The score statistic is **algebraically equivalent** to your Method A (`Δ_marg = log p(D|1DLA) − log p(D|null)` at the prior-marginal level) in the asymptotic limit because both reduce to a quadratic form in the residual signal projected onto the absorption-profile basis. So Method A is already (approximately) a score-style detector — and indeed it's your best performer.

### 4. Chib's method + Chib–Jeliazkov: directly relevant

**Chib (1995), JASA 90, 1313** computes `log p(D|M)` from MCMC output at a single "high posterior" point `θ*` (typically the MAP) via:
```
log p(D) = log p(D|θ*) + log p(θ*) − log p(θ*|D)
```
The trick: `p(θ*|D)` is estimated from the MCMC chain via a Rao-Blackwellized full-conditional. **Chib & Jeliazkov (2001), JASA 96, 270** extends this to Metropolis-Hastings (no full conditionals available). The resulting estimator is **the Laplace estimator with the Hessian replaced by an empirical posterior covariance from samples** — it relaxes the Gaussian assumption.

**This is what I'd expect to recommend instead of bare Laplace**: same MAP seed, but estimate the posterior density at the MAP from a small number of MCMC or importance samples around it, rather than from a finite-difference Hessian. ~100 likelihood evals per spec post-MAP. Cheaper than nested sampling, more robust than Laplace, and the formula degenerates back to Laplace in the Gaussian-posterior limit. The MLMC design in `2026-05-12_mlmc_design.md` is structurally a Chib-style estimator (level-1 importance-sampled around MAP), and that's a good reason to be confident in it.

### 5. Pseudo-marginal MCMC / unbiased likelihood estimators

Andrieu & Roberts (2009, Ann. Stat. 37, 697; "pseudo-marginal MCMC") allows targeting the true posterior using only an *unbiased* estimator of the likelihood. Relevant to us because the prior-marginal Δ is an unbiased Monte Carlo estimator of `log p(D|1DLA)` — so any downstream MCMC that uses it would still be asymptotically correct.

**Not directly useful for detection** because we'd still need a model selection criterion on top. Useful as a building block if the project ever moves to multi-DLA joint inference where the per-component evidence has high Monte-Carlo variance.

### 6. Direct marginal MAP / MMAP and variational methods

**Marginal MAP** (`MMAP`) is `argmax_θ ∫ p(D|θ, ψ) p(ψ|θ) dψ` — i.e. you optimize a model **after** marginalizing out a subset of nuisance parameters. In our setting if we treated z as the parameter and integrated out logN_HI we'd get a 1D z-scan of evidence values; the location of the maximum is the most-likely DLA z, and the value at the max is a proper marginal evidence — Occam factor included for logN_HI. **This is closer to the right detector** than MAP-LR. The cost is one inner integral per outer evaluation. With a 1D logN_HI integral evaluated by QMC, this is ~50× a single likelihood, so ~5–10s per spec — feasible. **I would have done this instead of bare MAP+LR if I'd known.**

References:
- Doucet, Godsill, Robert (2002), JCGS 11, 451 — "marginal MAP via simulated annealing"; gives a practical algorithm.
- Liu, Ihler (2013), JMLR 14 — message-passing for MMAP.

**Variational Bayes / VBEM**: would give a closed-form posterior approximation that, by construction, accounts for the Occam factor (the ELBO is `log p(D) − KL[q‖p(θ|D)]` and is a lower bound on the evidence). Less robust at multi-modality than nested sampling but cheaper. Probably overkill for a 2-parameter problem.

### 7. Numerical pitfalls in scipy.optimize.minimize at narrow peaks

What I would worry about, beyond what you already controlled with the 5k QMC seed:

- **Nelder-Mead default tolerances**: `xatol=1e-4, fatol=1e-4`. For a likelihood peak whose width is ~10⁻³ in z, these stop **well before** convergence onto the peak. Empirically your 5k/10k/50k QMC seeds give identical answers to 4 sig figs — this is **a sign that the optimizer is being limited by QMC seed resolution, not by NM convergence**, so the literal `argmax` is being missed by potentially `O(10⁻³)` in z. For LR-based detection that is small. For the Laplace `½ log|H|` term it matters: a wrong center means a wrong second derivative.

- **Finite-difference Hessian step size**: scipy's `optimize._numdiff.approx_derivative` defaults to `eps ≈ sqrt(machine_eps) ≈ 1.5e-8`. For a 2D Voigt-fit problem whose likelihood is smooth on the scale ~10⁻³ in z, this is **way too small** — you're differentiating numerical noise. The right step is `eps ≈ 10⁻⁴` for both coordinates, chosen as `(peak_width / ~30)`. If the Laplace agent used scipy defaults, the 78 non-PD Hessians are most likely just numerical-noise artifacts, not real boundary problems.

- **Local minima**: with K=3 starts you're vulnerable to deep but not-quite-global secondary peaks (Lyβ ghost in particular). A more robust approach: K=10 starts, keep top-3 by LR, run a focused local IS around each.

### What did Garnett 2017 / Ho–Bird–Garnett 2020 actually use?

From the in-repo citations and `bayesian_model_selection.py:41` ("Reference: Ho, Bird & Garnett (2020), arXiv:2003.11036, Section 3.3"), and from training knowledge of these papers:

- **Garnett+2017 (arXiv:1605.04460)** introduced the GP-DLA framework. Section 3 explicitly computes the **marginal evidence** via QMC over the (z, logN_HI) prior — **not** MAP+LR. The model posterior `P(DLA|D)` is the renormalized evidence ratio. They write down the Occam-volume argument explicitly and use the marginal as their detection score.

- **Ho–Bird–Garnett 2020 (arXiv:2003.11036)** extends to multi-DLA via greedy recursion (the `MAX_DLAS` loop in our `dla_gp.py:359`). Same QMC-based marginal evidence. They specifically note (§3.3 from memory) that the marginal `p(D|M_k)` is preferred over peak-likelihood for the Occam-protection reason; this is **exactly the principle our MAP+LR experiment violated**.

So the original authors faced the same narrow-peak problem and **chose the marginal-evidence route from the start**. Our Method A (`Δ_marg`) is the direct descendant of that choice; Method B was a regression to a non-Occam-protected statistic. The fact that Method A roughly matches `P_DLA > 0.99` is therefore expected — they are evaluating the same Bayes factor numerator, the only difference being threshold calibration.

### Was the canonical approach used correctly?

**Two things were wrong with how MAP+LR was set up**:

1. **MAP+LR is not the right test statistic for this problem**, regardless of how it's calibrated. The Bayesian marginal `Δ_marg = log p(D|1DLA) − log p(D|null)` is the *integrated* version and the *score* test for the absorber-feature amplitude is *also* an integrated/projected statistic. Both account for the redshift-search-window multiplicity that Davies-1987 tells us to penalize. The point-MAP likelihood does not. Calibrating against the null pool can compensate for the *level* of the statistic but not for the fact that signal and noise distributions overlap because the statistic doesn't use the prior information that would separate them.

2. **The Laplace correction was the right next step but does not fix (1)**. Laplace adds back the *self*-information of the peak (its width) but not the *external* information (which z's in the search window have a forest signal that locally supports a DLA shape vs not). The relevant separator is `q(θ) ∝ likelihood(θ) × prior(θ)` averaged over θ — i.e. the marginal — not the maximum of the integrand.

### Most-overlooked method from the literature

**Marginal MAP over (z, logN_HI) with the OTHER coordinate integrated out**, evaluated at the maximizing z. This gives you (a) an integrated, Occam-protected log-evidence as the test statistic, (b) a point estimate of the MAP z to feed the downstream catalog, (c) trivially extends to multi-DLA, and (d) costs ~10–50× a single likelihood (one 1D QMC integral per outer scan step). The closest published precedent in our problem class is the Bayesian-source-detection literature in CMB/X-ray (e.g. Hobson & McLachlan 2003, MNRAS 338, 765, for matched-filter source detection with a marginal evidence cut), and the "Maximum a Posteriori Marginal" community in graphical models (Doucet+2002, Liu+Ihler 2013).

### Refined diagnostic recommendation: if someone wants to retry MAP-based detection

The single best falsifiable experiment, before investing in MLMC:

**Experiment: "Variance-of-the-evidence diagnostic on the null pool"**.

For each of the ~1683 SNR>2 nulls and the ~309 truth-positives:
1. Compute `Δ_marg(N)` for N ∈ {1k, 5k, 10k, 50k, 200k} QMC samples (5 levels).
2. Report `Var[Δ_marg | N]` *across the QMC seed*, not across spectra. Use 4 independent seeds.
3. **Test**: does the Δ_marg estimator's QMC variance dominate over the signal-vs-null separation at N=5k? If yes (likely), then **the marginal evidence is itself QMC-noise-limited and the right fix is variance reduction (importance sampling around MAP, i.e. MLMC level-1)**. If no, then the marginal evidence is statistically separating signal from null even at the current QMC budget, and the problem is genuinely "no sufficient statistic exists in our likelihood" — at which point we have to add prior information (forest-correlation features, neighbor-pixel context, etc.).

This experiment is cheap (~2 hours on the existing London 8f), directly diagnoses (a) Occam-penalty vs (c) QMC-resolution from the question, and provides numerical grounding for the MLMC design. It also yields a paper-ready plot ("signal-to-null log-evidence separation vs QMC budget") even if MLMC isn't built.

A secondary cheap diagnostic: **plot `log_LR_MAP` vs `Δ_marg`** for the full Method-B catalog. The slope and scatter tell you the magnitude of the Occam penalty per-spec; multi-modal points appear as outliers from the regression. If the residuals correlate with SNR or with truth-status, that's evidence the missing penalty is the dominant pathology and Laplace-style corrections (with the boundary fix) could in principle work.

### One-line summary

We did MAP-based detection the "naive frequentist" way (GLRT with Wilks asymptotics replaced by an empirical null calibration), correctly diagnosed why it failed (Occam factor missing), tried the textbook fix (Laplace), and correctly diagnosed why *that* failed (both signal and noise peaks are equally narrow at low SNR). The literature broadly agrees with this diagnosis. **The correct alternative is what you already plan to do (importance sampling / MLMC around MAP)**; the most-overlooked alternative is **marginal MAP** (integrate one coordinate, optimize the other) which would have given an Occam-protected test statistic for ~5× the cost of MAP+LR, and which is much cheaper than full MLMC. If a future agent wants to revisit MAP-based detection, the diagnostic experiment is "variance of Δ_marg across QMC seeds at multiple N levels" — it directly tells you whether you have a sampling problem or a sufficient-statistic problem.

### References (verify before citing in a paper)

- Wilks (1938), Annals of Math. Stat. 9, 60.
- Chernoff (1954), Annals of Math. Stat. 25, 573.
- Davies (1977, 1987), Biometrika 64 + 74.
- Schwarz (1978), Ann. Stat. 6, 461 — BIC.
- Self & Liang (1987), JASA 82, 605 — boundary parameters.
- Skilling (2004, 2006) — nested sampling.
- Chib (1995), JASA 90, 1313; Chib & Jeliazkov (2001), JASA 96, 270.
- Kass & Raftery (1995), JASA 90, 773 — Bayes factors review.
- DiCiccio, Kass, Raftery, Wasserman (1997), JASA 92, 903.
- Meng & Wong (1996), Stat. Sin. 6, 831 — bridge sampling.
- Pinheiro & Bates (1995), JCGS 4, 12 — Laplace for boundary problems.
- Andrieu & Roberts (2009), Ann. Stat. 37, 697 — pseudo-marginal MCMC.
- Protassov, van Dyk, Connors, Kashyap, Siemiginowska (2002), ApJ 571, 545 (arXiv:astro-ph/0201547) — astronomy-specific LRT caveats.
- Feroz, Hobson, Bridges (2009), MNRAS 398, 1601 — MultiNest.
- Handley, Hobson, Lasenby (2015), MNRAS 453, 4385 — PolyChord.
- Speagle (2020), MNRAS 493, 3132 — dynesty.
- Hobson & McLachlan (2003), MNRAS 338, 765 — Bayesian matched filter / marginal evidence detection in astronomy.
- Garnett et al. (2017), arXiv:1605.04460 — the GP-DLA paper our code descends from; uses QMC-marginal evidence.
- Ho, Bird & Garnett (2020), arXiv:2003.11036 — multi-DLA extension; §3.3 is the model-selection reference cited in `bayesian_model_selection.py`.
- Doucet, Godsill, Robert (2002), JCGS 11, 451 — marginal MAP.
- Liu & Ihler (2013), JMLR 14 — MMAP via message passing.
- Xie, Lewis, Liu, Fan (2011), Syst. Biol. 60, 150 — steppingstone sampling.
