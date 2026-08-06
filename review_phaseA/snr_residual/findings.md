# REVIEW-ONLY (Phase A) — referee report: the post-repair SNR residual

*Recorded by the orchestrator from the referee agent's final report (the agent
environment could not write report files); `results.json` in this directory
carries the verdict machine-readably, with scripts s1–s5 and per-step JSONs.*

**Claim under test** (PI checkpoint §10 @ 9d73365): "The leading residual is
now the SNR axis: by_snr χ²/dof = 36.61/62.91/54.61 — worse than the n̂
gate — with a coherent monotone tilt."

## Exact reproduction

Packs `modelA_pack_{mock}_bw0p2_pad19p0_molly172.npz` rebuilt at 9d73365
(window lya_only, pad 19.0, molly172, bw 0.2; counts 88071/87840/86763),
production `forward_selftest.selftest(pack, resp_clamp="both")`, by_snr
marginal **restricted to n̂ ∈ [19.7, 21.6]**. Reproduces every digit: by_snr
χ²/dof 36.61/62.91/54.61; z rows +8.14/−1.35/+0.90/−4.00/−7.14/−9.16
(2LPT-0); window 22.22/28.39/25.77; by_z 9.94/9.72/7.85; full μ/obs
1.0016/1.0501/1.0281. **The checkpoint never states the by_snr numbers are
window-restricted** (full-grid values are 49.18/111.71/88.46). Independent
reconstruction (own aggregation + own FP formula) agrees to 1.1e-13.

## Verdict

**(i) "The leading residual is the SNR axis" does not survive.** In-window FP
events per stratum are [8,8,3,3,3,4]; measured var_cal/var_surv = 5.5–17.4.
With the loa-0 calibration noise propagated (delta method, confirmed by
bootstrap): by_snr χ²/dof **36.61→3.58, 62.91→5.86, 54.61→4.91**; max|z|
9.2–10.9 → **3.6–4.0** (all under the gate's 5); full-grid by_snr → 1.6/4.1/
2.9. Window by_nhat corrects to **9.36/9.65/8.93** (max|z| 5.7–6.4). **The N̂
axis remains the leading residual; by_snr was the arm most inflated by
unpropagated calibration noise.** Under the shared-template null the three
mocks' by_snr z* correlate at +0.81..+0.95; P(committed sign pattern on all
3 | perfect model) = 0.016 — cross-mock "coherence" is ~one observation, not
three.

**(ii) Origin = mixture.** ~85–95% of the survey-only by_snr χ² is loa-0
calibration sampling noise. A genuine survivor remains at joint p = 0.002
(shared-draw bootstrap) / 0.0195 (row-Jeffreys variant), and it is
**signal-side**: with FP off, the fold under-predicts SNR [2,3) by 17–19%
and over-predicts [7,∞) by 3.3–3.8% (z_nofp −4.5..−5.4 there; FP share 3%,
so no FP freedom can fix the top stratum). Excluded: truth stratification
(dX allocation matches truth_counts_bks to ≤1.4%), support holes (strata
[0,2) empty consistently; only note: stratum [3,4) straddles resp edge 3.5),
and FP SNR misallocation — reallocating the FP to the loa-0 **all-N** SNR
profile ([93,269,253,243,245,1275]; 54% in [7,∞), opposite of in-support)
makes χ² 3–4× **worse** (150/173/169): the FP SNR profile is strongly
N-conditional, so any future frozen-FP-shape option must freeze the
in-support conditional, not the all-N shape. Candidate mechanisms for the
survivor: per-stratum completeness error, kernel SNR dependence, or
unmodelled sub-19.0 promotions at low SNR (one-sided-support class).

**(iii) No SNR model freedom is justified.** The correct fix is propagating
calibration uncertainty into the gate statistics — the model already samples
lam_fp; only the selftest/gate plugs the raw 89-event point into a
survey-only z.

**(iv) Closure failure stands** (corrected window by_nhat 8.9–9.7, ≥ 3× the
ratified gate, on the N̂ axis). The FP-repair "overshoot" (−1.80/−3.45/
−2.61%) is largely the template's own calibration noise: full-total z
−0.46/−14.49/−8.17 → **−0.09/−2.77/−1.54** propagated; only London-0 is
marginal (−2.8σ), and the three mocks share the same 89-event draw.
Downgrade "the corrected FP overshoots" to "consistent with calibration
noise, marginal on London-0".

## Honesty note (agent-stated)

Two intermediate bootstrap variants (cell-level Jeffreys in s3; first joint
test in s4) were defective (+0.5 phantom mean × 29 zero cells/row) and are
superseded by s4-b2r/s5; documented in the JSONs.
