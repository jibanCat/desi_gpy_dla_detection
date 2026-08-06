# FROZEN PHASE-A VERDICT — independent adversarial review of the 2026-08-05 session

Review branch `review/phaseA-adversarial-2026-08-05`, rooted at `9d73365`
(reviewed tip of `hbi-mcmc-threeroute`, preserved unchanged). Five independent
review tracks, each with an implementation that does not reuse the original
probes' calculation paths; all evidence committed under `review_phaseA/`
(per-track `findings.md` + machine-readable results + scripts + seeds).
Everything mock-derived; no real-DESI value. Inference-core test baseline at
the root: 811 passed (matches the handoff).

## The verdict

**1. FP-normalization repair (`7707c8e`): UPHELD — strengthened.** The defect
is confirmed by an independent counting derivation; the repaired fold equals
the counting expectation exactly (1.2e-16); the choice among repair options is
posterior-identical (9.3e-7). After correcting the comparator (below), the
repaired normalization is validated absolutely on the calibration twin at
+0.2%. Amendments: the `(1−η)` factor in the fold's own cited definition is
carried zero times (+0.58% bias on the FP term — confirmed definitional
inconsistency, fix awaits PI confirmation); `fp_w·ell_eff == n_sl` is an
algebraic tautology (pack-integrity check, not physics evidence);
`matching_contract.py:572–574` still documents the pre-repair convention; all
committed closure artifacts remain pre-repair and unmarked.

**2. Identifiability ("A and B are not separately identifiable"): REJECTED as
stated.** Every archived number reproduces exactly under an independent
autodiff implementation; the interpretation does not survive. 15 of the "16
within 1°" are pad↔window (fine-basis truncation; 0 of 30 sub-degree on the
ratified 0.2-dex basis); the 0.0176° direction is 97.4% window by energy;
pad↔FP is the *best*-separated pair (data-supported minimum 6.6°, 18.2° on the
ratified basis). The exact degeneracies are pad↔(completeness × sub-20.0
window) and t↔λ — neither involves the FP. What survives, restated: **the pad
amplitude is prior-identified, not data-identified** (sd(log₁₀T_A) ≈ 0.05 dex
under full production priors vs ≥ 0.6 dex likelihood-only); the FP total is
anchor-identified; the FP (c,s) shape is prior-dominated (149/174 live cells
carry zero calibration counts).

**3. Δdev = 41 → "0.6σ, a wrong model is undetectable": REJECTED.** The 0.6σ
derivation divides an LRT statistic by the sd of an absolute GOF deviance (a
category error); 41 is a noncentrality, not a test statistic; the empirical
null (parametric bootstrap, N = 120) has mean 23.2 ± 7.3, q99 = 40.7, and the
observed 85.6 has p ≤ 0.0083 with detection power ≈ 1.00. No single
σ-equivalent number is meaningful (the null is not pivotal). The defensible
residue: the misfit is invisible to the *absolute-GOF* check (power 0.058),
and the fitted FP total is badly wrong in *both* models without the anchor —
a parameter-identification problem, not model-misspecification invisibility.

**4. Geometry stability:** stable (< ±10% over 24 reference variants,
identical across mocks, insensitive to the FP-amplitude convention). The one
qualitative sensitivity is the basis width — in the direction of *better*
separation on the ratified basis. Prior curvature (full production priors +
anchor, Laplace) restores practical identification of both totals with weak
correlation (|r| < 0.25).

**5. SNR residual ("the leading residual is the SNR axis"): REJECTED.** The
committed by_snr numbers are silently window-restricted, and 85–95% of the
survey-only χ² is unpropagated loa-0 calibration sampling noise (89 events ×
~166-count footprints; var_cal/var_surv up to 17). Propagated: by_snr χ²/dof
3.6–5.9; the N̂ axis remains the leading residual (window by_nhat 8.9–9.7).
Cross-mock coherence is ~one observation (shared template). A genuine
signal-side survivor exists (p ≈ 0.002–0.02; SNR [2,3) under-predicted 17–19%
with FP off) — mechanism unresolved (completeness / kernel SNR dependence /
sub-19.0 promotions). No SNR model freedom is justified; the fix is
propagating calibration uncertainty into the gate statistics.

**6. FP ceiling ("μ_FP exceeds the mock's FP supply; no parameter can fix
this"): REJECTED as stated → recast.** Under the committed unmatched estimand
μ_FP is *below* the ceiling by 23–39%; the quoted comparator was the
floor-17.2 hostless class, which is ~92% genuine sub-floor detections (the
"lower bound" direction claim is inverted). Chance-corrected: μ_FP/supply =
1.002 (twin) / 1.447 / 1.307 (held-outs). The surviving negative result is a
**cross-mock transport failure**: ~31–45% over-prediction on the held-outs,
≈ 1.9–2.8σ once the template's own calibration noise is included, against a
t prior that under-covers it — and effectively one observation, since all
three mocks share the 89-event template draw.

**7. Mock transfer: UPHELD.** All 24 calibration blocks bit-identical across
packs (independently re-measured). London-0/Saclay-0 are prediction-transfer
tests only; any common calibration bias is invisible by construction.
Manuscript-safe wording in `mock_transfer_audit.md`.

**8. Closure: "nothing closes" STANDS** — window by_nhat 8.9–9.7 ≥ 3× the
ratified gate under the corrected variance model, on all three mocks, on the
N̂ axis — but the margins soften ~2.5× and the attribution changes: 2LPT-0's
total closes at +0.16% with a signal-shape redistribution inside the window;
the held-out overshoots are quantitatively the FP transport excess.

**9. A PI science ruling IS required** before any behavior-changing work (see
decisions below). **Recommended next action:** adopt the corrected statistical
accounting (calibration-noise-aware gate statistics) as the reporting frame,
re-found the injection/campaign decision on the *actual* degeneracy
(pad↔completeness, FP-shape prior-dominance, transport), and correct the
public record of the five rejected/reframed statements before any model
change.

## Claim status lists

**Upheld:** FP defect + repair (exact; twin-validated); closure failure on the
N̂ axis; mock calibration bit-identity / prediction-transfer-only;
`resp_N_fit_range` = binning knob (as-documented, relied on in passing).
**Weakened / revised:** FP "overshoot" → consistent with the template's
calibration noise, marginal on London-0 only; μ_FP-vs-supply → cross-mock
transport failure (~1.9–2.8σ, one-ish observation), twin validated; noiseless
pad under-recovery −7.7–8.5% → −15% ± 10% under noise (worse, and
anchor-dependent); "z-shape differs across the floor p = 0.001" → untouched,
but its consequence must be restated (the all-N FP SNR profile must NOT be
transferred: it makes closure 3–4× worse; any frozen shape must be the
in-support conditional).
**Rejected (recommend formal retraction in the public record):** "Δdev = 41 =
0.6σ / a wrong model is undetectable"; "16 of 75 pad directions within 1°" as
A-vs-B evidence; "populations A and B are not separately identifiable" as
stated; "the leading residual is the SNR axis"; "μ_FP exceeds the mock's
entire FP supply — no parameter can fix this" (direction and estimand).
**Unresolved:** mechanism of the signal-side SNR survivor; PI confirmation of
the (1−η) placement; `resp_clamp="hi"` defensibility (not adjudicated);
`t_sigma`'s FF-route→forward-route cross-link; the frozen-shape 16–21%
estimand sensitivity (likely stands, reinterpreted as prior-sensitivity — not
re-derived); kernel-support boundary §4 (not re-reviewed).

## PI decisions queued (Phase B blocked until ruled)

1. **Gate variance model** — propagate loa-0 calibration noise into the
   ratified gate statistics? (Changes the ratified gate's meaning; the review
   shows the current gate overstates significance wherever FP matters.)
2. **FP transport treatment** — the t prior under-covers the measured
   cross-mock bias; widen it, refit it, or treat transport as a stated
   systematic? (Substantive prior decision.)
3. **FP shape treatment** — 149/174 cells prior-dominated; if a frozen shape
   is ever adopted it must be the in-support conditional profile.
4. **(1−η) fold factor** — confirm restoring the stated definition (−0.58% on
   the FP term; one line + regression test, prepared but not applied).
5. **Injections / campaign re-founding** — option B's original justification
   (break the A↔B degeneracy) is void; the actual targets are the
   pad↔completeness degeneracy, the sub-19.0 promotion channel, and the FP
   shape calibration.
6. **Public-record corrections** — issue #30 body, the two checkpoint docs,
   and the five rejected statements.
7. **Manuscript scope** — the TeX documents the predecessor v3x pipeline;
   synchronizing = adopting Model A as the paper's method.

*Technical appendix: the five track reports under `review_phaseA/*/findings.md`
and their `results.json`/`summary.json` files (all replicate values, seeds,
commands, and site tables). Stale-claims inventory:
`review_phaseA/stale_claims_inventory.md`.*
