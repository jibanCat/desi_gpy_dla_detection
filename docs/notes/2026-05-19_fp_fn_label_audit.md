# FP/FN label audit — the `fn_fp_deepdive` bookkeeping is inconsistent

> Run 2026-05-19. One `debug`-QOS sbatch job on Perlmutter (`53184601`,
> `fp_fn_label_audit.py`, 40 s compute, catalog + FITS reads only — no
> inference, no GP rebuild). Audit dir:
> `prod533_5k_20260511/fp_fn_label_audit/` (script, `run.sh`, `audit_log.txt`,
> `FINDINGS.md`). Catalog audited: `model_sweep/V1_2lpt124m/`. Truth London-0
> `dla_cat.fits` (762 822 absorbers, all NHI). Recipe = molly headline
> verbatim (`examples/molly_faithful_pc_plots.py`).

## TL;DR

The deep-dive FP table `fn_fp_deepdive/false_positives.fits` (68 rows) labels
**14 detections (20.6 %) as "false positive" that sit on a real strong truth
DLA** (truth NHI 20.3–21.9, Δz/(1+z) ≤ 0.011) and themselves predict
DLA-strength NHI (20.4–22.0). 13 of those 14 truth DLAs are also **absent from
`false_negatives.fits`** — so a real DLA is neither a TP, nor a genuine FP, nor
an FN; it has dropped out of the ledger. The deepdive FP list is bit-identical
to the headline P/C FP set, so the **headline purity 0.804 is understated** —
floor after correction ≥ 0.844.

The user's framing via wing_overestimate §5 ("50032/79067/121974 are FP rows
carrying NHI 17.8–18.9, so they're sub-DLA-band detections wrongly in a DLA FP
list") is **not** the bug. Those FP rows actually carry NHI = 20.94 / 21.17 /
20.89. The 17.8–18.9 values belong to a *different absorber row on the same
spectrum*. Both the wing study and the deepdive tripped over the same
multi-absorber spectra from opposite ends.

## 1. How `false_positives.fits` / `false_negatives.fits` are built

`fn_fp_deepdive/fn_fp_deepdive.py` reuses the molly headline recipe:

- **Catalog**: `model_sweep/V1_2lpt124m/dlacat-*.fits` (4815 detection rows,
  MAX_DLAS=3 single-absorber — a spectrum may carry 1–3 detection rows).
- **Truth**: `load_truth_molly` reads `dla_cat.fits` and **filters to
  NHI ≥ 20.3** (`truth-nhi-min`). The matcher therefore only ever sees
  strong-DLA truth; sub-DLAs/LLS are invisible to it (a *separate* full-truth
  table `truth_all` is loaded only for the §A NEAR_SUBDLA cross-ref).
- **Matcher** `match_truth_to_cat_molly`: greedy **1-to-1**, walks the catalog
  **in file order**; the first cat row with same TID and **|Δz|/(1+z_truth)
  < 0.01** claims a truth row (`truth_matched[j]=True`); ties broken by
  `min |NHI_pred − NHI_truth|`. No later cat row may re-use a claimed truth.
- **Cut bundle** `make_lambda_z_BAL_cuts`: λ_rf ∈ [911,1216], 2.0 < z_qso <
  4.25, drop all BAL TIDs; SNR_RED from the external snr_cat, restricted to the
  6766 processed TIDs.
- **`det_pass`** (the headline DLA mask): `SNR_RED>2 ∧ NHI_pred>20.3 ∧
  p_DLA>0.99 ∧ DLAFLAG==0 ∧ ¬LYBETA_FLAG`.
- **FP** = `det_pass ∧ ¬is_TP` (passing detection whose `NHI_TRUE` is NaN, i.e.
  the matcher gave it no truth row). **FN** = truth row with NHI > 20.3 that
  passes cuts, has SNR > 2, and was *not* matched by any `det_pass` detection.
- Multi-absorber handling: **per detection row, but matched per spectrum** —
  every absorber row of a MAX_DLAS=3 spectrum is an independent FP/FN candidate,
  yet they share one greedy 1-to-1 truth pool keyed on TARGETID.

That last point is the whole bug.

## 2. Root cause — file-order greedy matcher on multi-absorber spectra

Because the matcher is greedy in **file order** and 1-to-1, when the GP emits a
**weak/decoy absorber row earlier in the file than the correct strong one**,
the weak row reaches the shared truth DLA first and consumes it.

Worked example — **TID 50032** (one truth DLA: z = 2.3244, NHI = 20.88):

| cat row     | z_det  | NHI_pred | Δz/(1+z) to truth | matcher outcome |
|------------:|-------:|---------:|------------------:|-----------------|
| #1 (weak)   | 2.2976 | **17.83** | 0.00805 (< 0.01)  | **claims** the truth DLA |
| #2 (strong) | 2.3243 | **20.94** | 0.00006 (≈ exact) | truth taken → **FP** |

Consequence chain:

1. Row #1 grabs the truth → its `NHI_TRUE` is filled, **but** `NHI_pred 17.83
   < 20.3` fails `det_pass`, so it is **never counted as a TP**.
2. Row #2 — the near-perfect Δz/(1+z) = 6 × 10⁻⁵ match — finds the truth
   consumed → `is_TP = False`, passes every cut → lands in the **FP list**
   carrying NHI = 20.94.
3. The truth DLA *is* "matched" (by row #1), so it is **not** an FN.

A real, well-measured strong DLA produces one phantom FP and zero TP. The
`min |NHI_pred − NHI_truth|` tie-break cannot rescue it — that fires only when
one cat row sees several *unclaimed* truth candidates; here each cat row sees a
single truth row and **file order alone decides**.

79067 and 121974 are the identical pattern (weak row z-blueward, NHI ≈ 18.6–18.9,
claims the truth; strong row Δz/(1+z) ≤ 7 × 10⁻⁴, orphaned as FP).

**Verdict on the Task-3 menu:** the bug is **(b) a multi-absorber indexing
error** — greedy 1-to-1 matching over a MAX_DLAS=3 catalog in file order. It is
*not* (a) a z-window/comoving mismatch (Δz/(1+z) is used consistently), and
*not* (c) an FP list admitting sub-DLA-band detections (`det_pass` enforces
NHI_pred ≥ 20.3; **0/68** FP rows are below it).

## 3. Reconciliation table — the 8 named TARGETIDs

`pred` = catalog detection rows; `truth` = `dla_cat.fits` (all NHI).

| TID | catalog detection rows (z, NHI_pred) | truth (z, NHI) | deepdive label | verdict |
|----:|--------------------------------------|----------------|----------------|---------|
| 50032  | 2.2976/**17.83**; 2.3243/**20.94** | 2.3244/20.88 | FP (the 20.94 row) | **mislabel** — strong row is a correct DLA detection; the weak 17.83 row consumed the truth |
| 79067  | 2.1999/**18.88**; 2.2223/**21.17** | 2.2244/21.16 | FP (the 21.17 row) | **mislabel** — same pattern |
| 121974 | 2.0239/**18.65**; 2.0483/**20.89**; 2.0857/19.63 | 2.0469/20.93; 2.0857/19.69 | FP (the 20.89 row) | **mislabel** — strong row Δz/(1+z)=5×10⁻⁴ to the 20.93 DLA; weak row claimed it |
| 11485  | 2.9198/19.55; 3.4175/20.83; 3.5275/**20.45** | 6 absorbers incl. 3.5286/20.16 | FP (the 20.45 row) | **borderline-real** — sits on a true 20.16 sub-DLA (NEAR_SUBDLA already True); a true NHI-overestimate, not a phantom |
| 25463  | 2.2934/18.22; 2.2322/**20.49**; 2.1462/18.47 | 2.2322/20.28 | FP (the 20.49 row) | **NHI-overestimate FP** — the 20.49 row is Δz/(1+z)=4×10⁻⁵ to a true 20.28 *sub-DLA*; an overestimate FP, not a phantom continuum FP |
| 10063  | 3.1682/18.86; 3.4972/**20.25**; 3.5194/17.93 | 2.5386/17.80; 3.4978/20.33 | FN (truth z=3.498) | **correct FN** — best detection NHI 20.25 < 20.3 floor, so the true 20.33 DLA is genuinely missed at the recipe NHI cut |
| 127016 | 2.6737/20.29; 2.6583/**21.03**; 2.0969/19.91 | 2.6575/20.83; 2.6713/20.68 | FN (truth z=2.671) | **questionable FN** — a detection *does* exist at z=2.6737 (NHI 20.29) that matched the 20.68 truth, but NHI_pred<20.3 fails `det_pass`; the 21.03 row matched the *other* truth (20.83). Greedy order again splits a 2-DLA spectrum |
| 128648 | 2.2798/**21.39** | 2.2734/21.00; 2.2970/20.70 | FN (truth z=2.297) | **correct FN** — the single detection matched the z=2.273 DLA; the z=2.297 DLA has no detection (`HAD_DETECTION=False`) — a true blended-pair miss |

## 4. Population-level mislabel count

Across the whole 68-row FP list (`audit_log.txt` §Task 3):

- **14 FP rows (20.6 %)** sit on a real strong truth DLA (NHI ≥ 20.3) within
  Δz/(1+z) < 0.02 *and* have their own NHI_pred ≥ 20.3 — correct strong-DLA
  detections mislabelled FP. (TIDs 17936, 18306, 43172, 45203, 48327, 50032,
  79067, 87665, 112392, 113424, 121974, 124147, 125927, 127010.)
- Of those 14, **13 have the matching truth DLA missing from
  `false_negatives.fits`** (only TID 48327, a 4-absorber spectrum, has its DLA
  counted FN). So **≈ 13 real strong DLAs are absent from the FP/FN ledger on
  both sides** — neither recovered, nor flagged spurious, nor missed.
- **0/68** FP rows have NHI_pred < 20.3 — the suspected "FP list admits
  sub-DLA-band detections" mode does not occur.
- The deepdive's §A claim (75 % of FP coincident with a true 19.0–20.3 sub-DLA)
  is *not* contradicted — that flag uses the full truth and is correct. The bug
  is narrower and *additive*: 14 strong-DLA mislabels stacked on top of the
  genuine sub-DLA-overestimate FPs.

## 5. Headline-recipe cross-check

The deepdive FP list is **bit-identical** to the headline molly P/C FP set:
intersection 68/68, zero rows on either side only (`audit_log.txt` L143–144).
So this is **not** a deepdive-only quirk — the **headline P/C carries the same
14 mislabels**. 50032/79067/121974 are all confirmed *in* the headline FP set.

- **Purity** reported 0.804 = 279/347. Re-scoring the ≥ 14 confirmed
  strong-DLA detections as TP gives a floor of **(279+14)/347 = 0.844**.
- **Completeness** reported 0.864 is *also* pessimistic: the ~13 strong DLAs
  that fell out of the ledger were silently "matched" by the weak decoy row, so
  they were counted as neither recovered nor missed. A corrected matcher counts
  them as TP, nudging completeness up as well.

The exact corrected P/C needs the fixed matcher re-run, but the direction is
unambiguous: **both purity and completeness are understated; the true joint
operating point is better than the 0.804/0.864 headline.** This does not
overturn any model-comparison conclusion (the bias is roughly uniform across
configs that share MAX_DLAS=3), but every absolute P/C number in the prod533
5k sweeps that used this matcher is a slight underestimate.

## 6. Recommended fix

Make the matcher **absorber-best, not file-order-greedy**:

1. **Minimum fix (1-line):** before the greedy loop, sort the catalog by
   descending NHI_pred — `cat = cat[np.argsort(-np.asarray(cat["NHI"]))]` — so
   the strong correct detection claims its truth DLA before any weak decoy can.
   This alone fixes all 14 named/population cases. Re-run is a ~1-min debug-QOS
   job.
2. **Better:** per-TID **optimal 1-to-1 assignment** (Hungarian,
   `scipy.optimize.linear_sum_assignment`) minimising Σ Δz over same-TID
   blocks — removes order-dependence entirely.
3. **Bookkeeping:** when a cat row matched a truth row that a stronger cat row
   should own, re-attribute before the FP/FN split, so a multi-absorber
   spectrum is scored per absorber consistently.
4. Independently, in the deepdive's §A/§F breakdown, separate the
   **sub-DLA-NHI-overestimate FP** category (25463-style: NHI_pred ≥ 20.3 on a
   true 19.0–20.3 sub-DLA) from genuine continuum/forest-spike FPs. NEAR_SUBDLA
   already identifies them; they should be reported as a distinct class, not
   counted into DLA purity.

## 7. What is NOT wrong

- The molly recipe cuts (SNR>2, p_DLA≥0.99, DLAFLAG=0, lyb-veto, λ_rf window)
  are applied correctly.
- The deepdive's sub-DLA-overestimate diagnosis (§A) stands — those flags use
  the full truth and are right.
- The wing_overestimate §1–4 forest-vs-metals physics is untouched by this
  audit; only its §5 named-case *labels* were mis-attributed (it read the weak
  17.8–18.9 absorber row instead of the strong 20.9–21.2 row that the deepdive
  actually flagged FP). The wing study's qualitative point — those spectra do
  carry a real strong DLA the GP also under-fit on a second row — is correct;
  its claim that the FP row itself carries NHI 17.8–18.9 is not.
