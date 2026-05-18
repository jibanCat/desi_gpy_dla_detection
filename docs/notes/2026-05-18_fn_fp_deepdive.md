# 2026-05-18 — FN/FP deep-dive: why the production run misses and over-detects

> **Status**: DONE (debug-QOS sbatch job 53133896, catalog-only analysis).
> **Verdict: the 85/85 gap is precisely diagnosed — it is systematic NHI
> overestimation of sub-DLAs at the 20.3 boundary.** 75% of false
> positives are *real* 19.0–20.3 sub-DLAs whose predicted NHI is pushed
> over 20.3; only ~6% of FPs are genuinely spurious. Threshold knobs only
> slide P against C along this boundary — the one lever that moves the
> joint operating point is **NHI debiasing**.
>
> Full breakdown + per-object FITS:
> `/pscratch/sd/j/jibancat/prod533_5k_20260511/fn_fp_deepdive/FINDINGS.md`

## Setup

Analysed `model_sweep/V1_2lpt124m/` — the production-candidate config
(V1 model, 2-way, MAX_LAMBDA=1250, PW 50k), London-0 5k, fixed molly
recipe. Truth-matched FN (33 missed truth DLAs) and FP (68 spurious
detections); P=0.804, C≈0.86–0.90 depending on the completeness basis.

## False negatives — 33 missed DLAs

Misses are weaker/noisier than recovered DLAs (SNR median 3.4 vs 5.2,
NHI median 20.40 vs 20.70):

- **64% are low-SNR (SNR<4)** and **64% are weak (NHI<20.5)** — largely
  an irreducible information floor, not a bug.
- **42% (14/33) produced a detection that a cut then rejected** — the
  addressable slice — but relaxing p_DLA/NHI to recover them lands in the
  same noisy 20.3–20.6 regime that creates FPs, so it just slides P↔C.
- **15% are blended pairs** (a second truth DLA within dz/(1+z)<0.015) —
  5× over-represented vs recovered DLAs. Structurally lost to the 2-way
  single-absorber model; matches the runbook's MAX_DLAS note (~6% of QSOs
  carry >3 absorbers).

## False positives — 68 spurious detections (purity 0.804)

- **75% (51/68) are coincident with a true sub-DLA (19.0 ≤ NHI < 20.3)** —
  i.e. *real* absorbers whose predicted NHI was over-estimated past the
  20.3 floor.
- **94% sit on a real absorber** (any NHI ≥ 17.2). Only **~6% (4/68) are
  genuinely spurious** (no truth absorber at all).
- **71% have NHI_pred just above the floor (<20.6)** vs 31% of true
  detections — the FP population is concentrated in the boundary bin.
- Continuum/emission FPs (near Lyα/NV), Lyβ leakage, deep-forest: each
  ≤15%, minor.

## Conclusion — the path to 85/85

Both deficits live in the **20.3–20.6 NHI boundary regime**, and they are
two faces of one defect: **systematic NHI overestimation near the
DLA/sub-DLA boundary** (consistent with `2026-05-17_nhi_flag_investigation.md`:
NHI_pred biased +0.06 dex high, NHI_ERR under-estimated ~1.4×).

- **Knob tuning cannot reach 85/85** — every threshold (p_DLA cut, NHI
  cut) just slides purity against completeness along this boundary. This
  is why the model sweep, p_DLA sweep, and lambda/PW sweeps all stalled
  ~3–5pp short.
- **The lever that moves the joint frontier is NHI debiasing.** Correcting
  the sub-DLA NHI overestimation reclassifies ~50 of the 68 FPs back to
  sub-DLA *without* costing completeness — a genuine purity gain, not a
  trade. The τ-EB NHI-debias recipe (`docs/tau_eb_hcd_mask.md`,
  PR #5 follow-up) already closed ~65% of the DLA-regime NHI bias in
  earlier work and is the obvious candidate.
- Blended pairs (~15% of FN) need an architecture change (MAX_DLAS>1 /
  multi-absorber) — a smaller, separate lever.

**Actionable next step**: apply an NHI-debias pass (τ-EB-style) and
re-measure P/C. This is the recommended pre-1M-launch work item — it is
the only thing identified that can close the 85/85 gap without a P↔C
trade.
