# REVIEW-ONLY (Phase A) — referee report: the geometric identifiability claim

**Claim under test.** "Sub-floor migration (pad, population A) and forest FP
(population B) are not separately identifiable; 16 of 75 pad directions within
1 deg of the [window u FP] span; smallest angle 0.0176 deg."

**Method (independent of the archived probe).** Jacobians of the production
fold (`CDDF_analysis.hbi_mcmc.forward.fold_mu`, worktree tip `9d73365`) by
**autodiff** (`jax.linearize` + vmapped JVPs), never hand-coded; rows
Poisson-Fisher-whitened (1/sqrt(mu)) over live cells; principal angles by
SVD-orthonormalization (the probe used QR). Reference point rebuilt from
scratch (window truth = `e4_probe.truth_f`; pad = own power-law continuation,
continuous at the floor; pad/FP scales set through the production fold
itself). Autodiff resolves the probe's `fp_operator` convention question by
construction — `d mu/d lam` carries `fp_w * fp_ell_eff * E` exactly.
Finite-difference validation: max rel err 2.9e-11 ... 3.4e-9 across psi_c /
psi_k / log_t / lam blocks; 2.2e-6 for theta (central-difference truncation
at step 2e-3, forced by the dead-bin sentinel |theta| = 2000 — not autodiff
error). Scripts `run_exp1..5.py`, all numbers in `results.json`
(+ per-experiment `out_exp*.json`).

---

## Findings

### E1 — Reproduce, then separate (3 mocks x 2 FP levels)

The quoted headline is reproduced **exactly** by the independent
implementation, and it is the pad vs [window u FP] pair:

| pair (2lpt0, 0.1-dex, probe convention) | min angle | n < 1 deg | q25 |
|---|---|---|---|
| pad vs [window u FP] | **0.0176 deg** | **16 / 75** | 12.0 deg |
| pad vs window | 0.283 deg | 15 / 75 | 14.0 deg |
| pad vs FP (**the A-vs-B question**) | **0.668 deg** | **1 / 75** | **89.8 deg** |
| pad vs ALL (+psi_c, psi_k, t) | 0.0 (exact) | 17 / 75 | 3.8 deg |
| FP vs [window u pad] | 0.090 deg | 4 / 174 | 81.4 deg |

**15 of the 16 sub-degree directions are pad-window, not pad-FP.** The
pad-FP block is the *best*-separated pair in the table: 71+ of its 75
principal angles sit near 90 deg. London-0 and Saclay-0 are numerically
indistinguishable (pad-FP min 0.666 / 0.634 deg; identical counts). Swapping
the FP reference level 1086.7 -> 14768 (pre-repair vs repaired anchor) moves
pad-FP only to 0.715 deg — the probe's amplitude-convention ambiguity is
immaterial for the geometry; the archived values are recovered at 1086.7,
confirming the archived probe's mu sat at the pre-repair FP amplitude.

### E2 — Column-selection artifacts (2lpt0)

* **FP columns.** Only 25 of 232 (c, s) cells have loa-0 counts; the other
  149 "informative" FP columns exist purely through the probe's lam floor
  (1e-4 x mean). Restricting to data-supported cells: pad-FP
  **0.668 -> 6.61 deg**, pad-[window u FP] 0.0176 -> 0.101 deg. The
  sub-degree pad-FP direction **is an artifact of floor-manufactured columns**.
* **psi_c columns.** Every informative psi_c column has window support (no
  zero-data molly cells on 2lpt0) — but the exact-zero pad-ALL direction is a
  **structural identity**, verified to 3e-16:
  `sum(pad cols) = sum_s psi_c[s, cell0]/(1 - C[s, cell0]) - sum(window cols in cell0)`.
  All pad bins clip into molly cell 0 = [19.5, 20.0) (`b_to_cell`), so a
  uniform pad rescale is *exactly* a completeness offset in the bottom molly
  cell plus a compensating rescale of the 19.5-20.0 window bins. This is a
  pad-(completeness + window) confusion — the FP has **zero** energy in it
  (least-squares attribution: 80% psi_c, 20% window, 0.0% FP).
* The FP-ALL exact zero is the t-vs-lam global scale identity
  (`sum_K t-cols = sum_cs loglam-cols`, 9e-17) — a known model redundancy the
  t prior controls.
* The 0.0176 deg direction itself is 97.4% window / 2.6% FP energy: a
  basis-truncation direction that FP columns only polish.

### E3 — Reference-point stability (24 variants, 2lpt0)

(T_A, T_B) x pad_slope {fitted, -1.5, 0} x fp_shape {n0, flat}:
pad-FP min angle spans **[0.591, 0.718] deg** (all columns) and
**[5.29, 7.96] deg** (data-supported); pad-[window u FP] spans
[0.0146, 0.0187] / [0.083, 0.111] deg. The geometry is a structural property
of the operator, not of the agent-chosen reference point.

### E4 — Prior curvature (Laplace in production sampled coordinates, 2lpt0)

Coordinates exactly as `model_a.model_a` samples (non-centered RW theta;
sigma_N = sigma_z = 0.5 held at their HalfNormal prior scales — conditional
Laplace; psi_* / t standardized; FP as (log lam_total, zero-sum shape /3));
prior Hessian = identity + Gamma(0.5, 1e-6) term; anchor =
`fp_counts ~ Poisson(ell_eff * lam)` expected Fisher; estimand gradients by
autodiff of the actual folded reductions. Marginal sds of the estimands:

| case (T_B = 14768, repaired anchor level) | sd(log10 T_A) | sd(log10 T_B) | corr |
|---|---|---|---|
| (a) likelihood only | 0.63 dex * | 0.077 dex * | +0.15 |
| (b) + loa-0 anchor | 0.58 dex * | 0.033 dex * | +0.09 |
| (c2) likelihood + full production priors (no anchor) | 0.054 dex | 0.016 dex | -0.15 |
| (c) likelihood + anchor + full priors | **0.054 dex** | **0.015 dex** | **-0.17** |

At T_B = 1086.7 (pre-repair level): (a) 0.54* / 0.92* dex; (c) **0.049 /
0.124 dex**, corr -0.14.

\* (a)/(b) are **lower bounds**: the likelihood Hessian has 273 (184) exact
null directions (dead-bin eps_z/eps_N, psi_c identity, t-lam scale); the
estimand gradients have small but nonzero null components (3e-4 ... 3e-3), so
those cases are structurally improper. This is also why the archived probe's
"22-dex" VIF marginals were meaningless — pinv marginals of a prior-less
singular Fisher measure the regulator, not the model. (It also produces the
apparent (a)->(b) non-monotonicity at the low FP level: adding the anchor
*resolves* 194 formerly-null directions into the marginal.)

**Reading:** the production prior — RW smoothness + proper psi_c + t priors —
supplies the identification the likelihood alone lacks (T_A: >= 0.6 dex ->
0.05 dex), and the loa-0 anchor pins T_B (0.077 -> 0.033 dex before priors).
The posterior correlation between the two totals is weak (|corr| < 0.25):
the A and B **totals** do not trade off against each other in the Gaussian
approximation.

### E5 — The ratified 0.2-dex basis (2lpt0)

On the analysis basis actually ratified (PI decision 3; 0.1 dex is
plotting-only), the headline structure collapses:

| pair | 0.1-dex (all cols) | 0.2-dex (all cols) | 0.2-dex (data-supported FP) |
|---|---|---|---|
| pad vs window | 0.283 deg, 15 < 1 deg | **7.87 deg, 0 < 1 deg** | — |
| pad vs [window u FP] | 0.0176 deg, 16 < 1 deg | **0.80 deg, 1 < 1 deg** | **1.88 deg, 0 < 1 deg** |
| pad vs FP | 0.668 deg | **1.43 deg** | **18.2 deg** |

The 15-direction sub-degree pad-window cluster was a fine-basis truncation
artifact (adjacent 0.1-dex bins are indistinguishable through the ~0.3-dex
response kernel). The exact psi_c completeness identity survives (it is
basis-independent), as expected.

---

## Verdict

**(a) Does the geometric pad-vs-FP non-identifiability survive an independent
implementation?** The *numbers* reproduce exactly (independent autodiff
implementation: 0.0176 deg, 16/75, pad-FP 0.668 deg — archived values
recovered to all quoted digits). The *claim* does not: pad-vs-FP is the
best-separated pair in the geometry (one fragile sub-degree direction out of
75; quartiles at 90 deg). "Population A and population B are not separately
identifiable" is not supported as a statement about the A-B pair.

**(b) Is "16 of 75 within 1 deg" honest?** It is a formally correct count for
pad vs [window u FP], but as evidence for A-vs-B confusion it conflates two
different things: 15 of the 16 directions are pad-window (basis truncation,
D1), and only ~1 involves the FP at all (and that one is 97% window by
energy). The scientifically-relevant pad-FP count is **1 of 75** — and 0 of
30 on the ratified basis.

**(c) Are the smallest angles artifacts of unsupported columns?** Yes, in two
distinct ways. The 0.668-deg pad-FP minimum needs the 149 FP columns
manufactured by the lam floor in zero-count loa-0 cells (data-supported:
6.6 deg; 18.2 deg on the 0.2-dex basis). The ~0-deg (archived 1.5e-6 deg)
class is *not* a column artifact but an **exact structural identity** —
uniform pad rescale == completeness offset in the clipped bottom molly cell +
window compensation (3e-16), plus the t-lam scale redundancy (9e-17). Both
identities are real features of the prior-less likelihood, both are
prior-controlled, and **neither involves the FP**.

**(d) Is the geometry stable?** Yes. Min angles move by < +-10% over 24
reference variants (T_A, T_B, pad slope, FP shape), are identical across the
three mocks, and are insensitive to the FP amplitude convention. The
*conclusions* are reference-stable; only the basis choice (0.1 vs 0.2 dex)
changes the picture qualitatively — in the direction of *better* separation
on the ratified basis.

**(e) Does prior curvature restore practical identification?** Yes. Under the
full production posterior (priors + anchor): sd(log10 T_A) ~ 0.05 dex,
sd(log10 T_B) ~ 0.015 dex (repaired FP level; 0.12 dex at the pre-repair
level), correlation |r| < 0.25. The likelihood alone leaves T_A structurally
unidentified (exact flat directions; >= 0.6 dex even projected), and the
identification of T_A is supplied almost entirely by the production prior
(the anchor pins T_B). Caveats: conditional Laplace (sigma_N, sigma_z fixed at
prior scales), local Gaussian approximation, evaluated on-mock at the probe's
reference — treat 0.05 dex as an order-of-magnitude statement, not a
reported uncertainty.

**Summary language.** *Structural* non-identifiability in the prior-less
likelihood: REAL, but it is pad-vs-(completeness x window) and t-vs-lam — not
pad-vs-FP. *Practical* non-identifiability under the production posterior:
NOT supported — both totals are identified at the few-hundredths-of-a-dex
level with weak mutual correlation, and the prior directions doing the work
(psi_c in the bottom molly cell; t; RW smoothness across the floor) should be
reported as the prior-sensitivity axes of any pad-related number. The
"sub-floor migration vs forest FP are not separately identifiable" headline
should be retired in favor of: "the pad amplitude is prior-identified, not
data-identified; its exact likelihood degeneracy is with the completeness
surface and the sub-20.0 window bins, and the FP is geometrically well
separated from it."
