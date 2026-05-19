# PR #8 stacking — handoff & follow-up plans (2026-05-16)

Handoff for the real-LOA metal-line stacking work on branch
`claude/loa-dla-stacking` (PR #8 → `desi_y3`).

## State

PR #8 is **open, unreviewed**. It contains:
- `examples/stack_real_loa_dlas.py` — stacks real-LOA LLS / sub-DLA / DLA
  detections by N_HI bin in the absorber rest frame (spectra from the
  LoaArchive), median + 3σ-clip, redshift-scrambled control.
- `docs/notes/2026-05-15_stack_real_loa_dlas/LINE_LIST_REFERENCES.md` —
  the 35-line metal-line list, verified against Morton 2003 / Mas-Ribas
  2017, with forest-vs-redward reliability ranking.
- `.gitignore` — generated figures / per-object data stay local (real-LOA
  privacy).

**Result so far**: the CIV 1548/1551 doublet is detected at high
significance in the real sub-DLA and LLS stacks (CIV 1548 ~35σ vs an
empirical null in the LLS stack) and is flat (<2σ) in the z-scrambled
control. The Lyman-limit break rises monotonically with N_HI. So the
detections in the *tested* column-density range are real absorbers.

**Honest scope of that claim** (2026-05-19 verdict): the LLS bin
[17.2,19) is ~89 % log N_HI 18.5–19, so the evidence confirms the
**strong-LLS / sub-DLA-adjacent regime (log N_HI 18–19)**. The genuinely
low tail is *not* covered: [17.2,18.0) is fully NaN-clipped (only ~33–41
objects, below the 50-spectrum floor) and [18.0,18.5) is marginal
(n≈353, CIV S/N≈3–4). And the whole analysis uses P_DLA > 0.97, so it
validates the high-purity cut — it does **not** characterize the
marginal operating point (P_DLA 0.5–0.7). Significances are empirical-
null, not yet bootstrap-over-sightlines.

## Decision: metallicity is PARKED

Metallicity / metal-abundance analysis was researched this session (DLA
metallicity definitions, measurement methods, SDSS-vs-recent tracer
lines, a verified linear-COG S II recipe, and a full subfield deep-dive
— see the session memory). **It is not being pursued** — the priority is
cosmology (CDDF / dN/dX / Ω-type) rather than chemical abundances.

Key takeaways if anyone revisits it later:
- At DESI resolution (~69 km/s) Voigt-profile fitting is not feasible;
  metallicity from a stack means population-mean equivalent widths.
- The defensible cheap path: EW + a linear-curve-of-growth [S/H] from the
  S II 1250/1253/1259 triplet for strong-DLA bins only (S II is already
  in the line list). Verdict from verification: sound but ±0.2–0.3 dex,
  with hidden saturation, continuum placement, blending, and DESI
  color-selection / dust-obscuration bias as the rabbit holes.
- A headline metallicity result would require engaging all of those
  systematics properly — out of scope for now.

## Follow-up plans — improving the stacks (cosmology-relevant, not metals)

These sharpen PR #8 as a detection-reliability tool, no metallicity needed.

1. **Detection significance / error bars** — the current verdict is
   visual. Add bootstrap-over-sightlines errors and report the metal-line
   EW (esp. CIV) as an N-σ detection in real vs control. Turns "looks
   coherent" into a number. *Highest value.*
2. **Stack the marginal-purity detections** — current test used
   P_DLA > 0.97 (always-likely-real). Stack P_DLA ∈ [0.5, 0.7] separately:
   if those also show coherent CIV the marginal tail is real, if flat it
   is contaminated. Directly probes the operating point — the actual
   false-positive question.
3. **N_HI-resolved metal-line strength** — EW vs N_HI bin; a real
   absorber population shows EW rising with N_HI, a false-positive one
   would not. Second falsification axis.
4. **BAL exclusion** — selection does not drop BAL QSOs; BAL troughs can
   mimic absorption. Check for a BAL flag and exclude.
5. **FeII 1608** — flagship line just past the current 1600 Å rest cap;
   extend `REST_LAMBDA_MAX` to ~1620 Å + re-stack to include it.
6. **Code review** — PR #8 has had only a mid-development bug review; the
   final committed script + line-list doc have not been reviewed.
7. **Higher-N run** — `MAX_PER_BIN = 800` is a runtime cap; the LLS bins
   are noisy and would sharpen with more spectra.

Priority if picked up: **#1 + #2** together turn PR #8 from "suggestive
plots" into "low-N_HI detections are real at X-σ, including at the
marginal operating point" — the cosmology-relevant deliverable.

## Reproduce

```bash
python examples/stack_real_loa_dlas.py --zhist-only   # catalog-only, seconds
python examples/stack_real_loa_dlas.py                # full stack (~15 min)
python examples/stack_real_loa_dlas.py --plot-only    # re-render from cached npz
```
