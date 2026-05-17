# 2026-05-15 — molly_faithful_pc_plots.py recipe fix

## Summary

Patched `examples/molly_faithful_pc_plots.py` to match Molly Wolfson's
notebook (`/pscratch/sd/j/jibancat/molly/read_in_each_up_match_new_cats_2509.ipynb`)
on two methodology points that were silently producing inconsistent P/C numbers:

1. **Per-QSO SNR / Z_QSO source**: previously read from `processed-spectra-16-*.h5`
   files in `--catalog-dir`, which only covers spectra THIS run processed. Wrong
   for any cross-catalog evaluation (e.g. evaluating a legacy catalog against a
   different run's processed dir). **Fixed**: added `--snr-cat` + `--zcat` flags
   that load the canonical external FITS (full mock TID coverage). When h5 is
   present, intersect with that scope to keep 5k-slice runs evaluating against
   only their processed scope.
2. **BAL filter**: previously dropped only TIDs with `BI_CIV > 0` (~half of the
   bal_cat). Molly drops ALL bal_cat TIDs from both cat and truth. **Fixed**:
   default is now drop-all; `--bal-bi-civ-only` recovers the old behavior.

The trigger was the legacy 2025-09-12 catalog returning P=0.28 (vs ~78% expected).
Two compounding causes:
- I'd symlinked an unrelated 5k run's `processed/` into `legacy_baseline/`, so
  truth got decimated to the ~36k of 96k legacy cat TIDs that overlapped.
- Even after fixing the symlink, BAL convention was wrong by ~5pp on P.

## Bug detail

### Bug 1: per-QSO lookup scope

`build_per_qso_snr` (old version, lines 114-138) reads SNR_REDSIDE + z_qso from
`processed-spectra-16-*.h5` in `--catalog-dir`. `load_truth_molly` (lines
145-192) drops every truth row whose TARGETID is not in this lookup. When the
processed dir matches the catalog (the normal case for a fresh inference run),
this is correct — truth is restricted to QSOs the run actually processed. But
in cross-catalog evals (e.g., evaluating a Sept 2025 catalog with a May 2026
processed dir as the lookup), the lookup is INCONSISTENT with the catalog and
truth gets silently and badly subsetted.

### Bug 2: BAL filter

`main` (line 572-573, old version):
```python
bal = fitsio.read(args.bal_cat, ext=1, columns=["TARGETID", "BI_CIV"])
bal_tids = set(int(r["TARGETID"]) for r in bal if r["BI_CIV"] > 0)
```

Filters by `BI_CIV > 0`. The docstring at line 15 (old) said
"BAL removal = TARGETID ∈ bal_cat → dropped from BOTH cat AND truth" — i.e.
all bal_cat TIDs, no BI_CIV gate. **Implementation lied about its own
docstring.** Molly's notebook drops all bal_cat TIDs.

On London-0 mock-0:
- All bal_cat: 194,461 unique TIDs
- BI_CIV>0 only: 90,354 TIDs

Effect: ~5pp lower purity at the BI_CIV>0 setting because BAL contamination
in the kept-cat rows produces extra FPs.

## Patch

`examples/molly_faithful_pc_plots.py`:

1. **`build_per_qso_snr`** signature extended to
   `(catalog_dir, snr_cat_path=None, zcat_path=None, mockdir=None)`. Resolution:
   - external `--snr-cat` + `--zcat` (canonical, used when both exist)
     - intersect with h5-derived "processed scope" when h5 files exist in
       catalog-dir, so 5k slice runs still bound truth to the right scope
   - mockdir/snr_cat.fits + mockdir/zcat.fits (Saclay/2LPT have these inline)
   - processed h5 fallback (legacy behavior; warns about scope limits)

2. **BAL filter** in `main`:
   - default: drop ALL bal_cat TIDs (molly recipe)
   - `--bal-bi-civ-only`: legacy behavior (BI_CIV>0 only)

3. **Argparse**: added `--snr-cat`, `--bal-bi-civ-only`. `--zcat` already existed.

4. **Docstring**: clarified the BAL convention, the per-QSO lookup priority,
   and the slice-scope subtlety.

## Before / after on key cells

All numbers at p_DLA ≥ 0.99, SNR > 2, NHI ≥ 20.3, λ_rf [911, 1216], no-BAL,
DLAFLAG == 0, lyb-veto on, z_qso ∈ [2.0, 4.25].

### London-0 5k slice — cellC (2-way)

| Cell | Knob | Pre-patch P | Post-patch P | Pre-patch C | Post-patch C |
|---|---|---:|---:|---:|---:|
| C0 | baseline | 0.7792 | **0.8285** | 0.8772 | **0.8824** |
| C7 | PW 100k | 0.7761 | **0.8281** | 0.8918 | **0.8947** |

Δ ≈ +5pp P, ~+0.5pp C. The BAL convention shift (drop more BAL TIDs from
both sides) boosts the headline metric. The cellC family now sits firmly
above 80/80 with the C7 PW 100k cell at 82.8/89.5 — closer to the 85/85
target than any prior config.

### London-0 5k slice — cellD (3-way)

(See `cellD_knob_sweep/HEADLINE.tsv` for the full table; pattern is similar
but smaller deltas because 3-way absorbs FP shifts more gracefully.)

### Legacy 2025-09-12 — full London-0

| Recipe | P | C |
|---|---:|---:|
| Pre-patch (BI_CIV>0, mismatched processed dir) | **0.2828** | 0.8656 |
| Post-patch (drop-all-BAL, external snr_cat/zcat, lyb-veto) | **0.7949** | 0.8576 |

Matches user recall of ~78% pure / ~85% complete for the historical
production catalog.

## Eval-script call-site updates

The cellC/cellD/crossval eval scripts now pass:

```bash
--snr-cat /global/cfs/cdirs/desi/users/abrodze/DLA/dlatoolkit-catalogs/v20241030_mock/london-Y3-jura-mock0-snr-nobalmask.fits  # London
--zcat <mockdir>/zcat.fits
--mockdir <mockdir>
```

For Saclay / 2LPT runs (cross-val), `--snr-cat` is auto-discovered from
`<mockdir>/snr_cat.fits` since those mocks ship snr_cat inline.

## Files touched

- `examples/molly_faithful_pc_plots.py` — patch
- `/pscratch/sd/j/jibancat/prod533_5k_20260511/cellC_knob_sweep/_eval_and_aggregate.sh`
  — added `--snr-cat`, `--zcat` args
- `/pscratch/sd/j/jibancat/prod533_5k_20260511/cellD_knob_sweep/_eval_and_aggregate.sh`
  — same
- `/pscratch/sd/j/jibancat/prod533_5k_20260511/crossval/_eval_and_aggregate.sh`
  — note added (Saclay/2LPT auto-discover)
- `/pscratch/sd/j/jibancat/prod533_5k_20260511/legacy_baseline/_molly_exact_recipe.py`
  — standalone reproducer matching molly's notebook exactly (independent of
  the eval-script patch)

## Tests

After the patch, sanity ran on:
- C0 5k slice: P=0.8285, C=0.8824 (n_truth=581 — correct 5k scope)
- Legacy full mock with patched script: P=0.7949, C=0.8576 (matches user recall)

The processed-h5 scope intersection logic is essential — without it, a 5k
slice run would see truth = full-mock truth (~73k DLAs vs ~580 actually in
scope) and completeness collapses to ~0.007.

## Carry-over impact

Every P/C number in this workspace measured between 2026-04 and 2026-05-14
is on the OLD BI_CIV>0 + h5-only convention. They are internally consistent
for any pipeline-vs-itself comparison (knob sweeps, cell vs cell). They are
NOT directly comparable to molly's published numbers or to the patched
post-2026-05-15 numbers. **Always use the post-patch convention for
production decisions and cross-author comparisons.**

---

## Update — 2026-05-15 evening, audit-driven matcher fix

A second-pass faithfulness audit (audit agent #2) found a third
discrepancy that was not caught by the first audit:

### Bug 3: `gp_native.match_truth_to_cat` overcounts TPs by ~3pp

The matcher iterates **truth in NHI-descending order**, picks the
**closest-z unused cat** row. Molly's notebook iterates **cat in input
order**, picks the **closest-NHI unused truth** row (with `|Δz|/(1+z)
< 0.01` filter). Both are 1-to-1, but they assign different TPs on
multi-DLA spectra, and the overall TP count is ~3pp higher under the
NHI-descending matcher.

**Fix**: added `match_truth_to_cat_molly` to `examples/molly_faithful_pc_plots.py`
(verbatim port of the proven `_molly_exact_recipe.py` matcher). `main()`
now calls the new matcher.

### Side-effect: scope intersection became opt-in

The first patch's `--snr-cat` + `--zcat` machinery silently intersected
the per-QSO lookup with the run's `processed-spectra-16-*.h5` TID set,
which was correct for 5k slices but wrong for full-mock evals (legacy).
Added `--restrict-truth-to-processed` (default OFF). 5k slice eval scripts
pass it explicitly; full-mock evals don't.

Secondary tidy:
- `--snr-min` default 6.0 → 2.0 (matches molly's canonical operating point).
- `log_pdla` sweep prepended `-0.5` to match molly's published curve.
- Dead `apply_bal_cut` import removed.

### Final headline P/C — post-matcher-fix (2026-05-15 evening)

> **CORRECTION (2026-05-15 ~17:30)** — the table and conclusion in this
> section were written ~12:39, BEFORE the C/D cells were regenerated at
> 13:11. The regenerated values (live in `pc_pdla_sweep.tsv` and each
> cell's `molly_summary_combined.tsv`) at p_DLA ≥ 0.99 are:
>
> | Cell | P | C |
> |---|---:|---:|
> | C7 PW100k | **0.8323** | **0.8142** |
> | D7 PW100k | **0.8387** | **0.7245** |
> | C0 baseline | 0.8139 | 0.7988 |
> | D0 baseline | 0.7993 | 0.7028 |
> | Legacy 2025-09-12 | 0.8255 | 0.8350 |
>
> With these, C7 **clears 80/80** (83.2 P / 81.4 C) and beats legacy on
> purity (+0.7pp) at −2.1pp completeness. The per-cell numbers in the
> table immediately below are superseded — treat `pc_pdla_sweep.tsv` as
> authoritative. See the corrected "Headline-changing implication" below.

| Source | P | C | Δ vs prior (pre-matcher-fix) |
|---|---:|---:|---|
| Legacy 2025-09-12 (full London-0) | 0.7895 | 0.8518 | −0.5pp P, −0.6pp C |
| C0 baseline (5k slice) | 0.7803 | 0.8359 | −4.8pp P, −4.7pp C |
| C7 PW100k (5k slice) | 0.7908 | 0.8545 | −3.7pp P, −4.0pp C |
| D0 baseline (5k slice) | 0.7905 | 0.7245 | −8.7pp P, −7.7pp C |
| D7 PW100k (5k slice) | 0.8282 | 0.7461 | −6.9pp P, −6.2pp C |
| C7_saclay (cross-val) | 0.8047 | 0.8691 | −2.4pp P, −2.5pp C |
| C7_2lpt (cross-val) | 0.7769 | 0.8417 | −2.9pp P, −2.5pp C |
| D7_saclay (cross-val) | 0.7917 | 0.7600 | −7.5pp P, −6.9pp C |
| D7_2lpt (cross-val) | 0.7570 | 0.6750 | −9.4pp P, −8.3pp C |

The matcher fix has a larger effect on **3-way (cellD)** numbers (~−7pp
P) than on 2-way (cellC, ~−4pp P). 3-way's per-spectrum cat row count
is smaller (~250 detections per slice) so the matching geometry depends
more sensitively on the iteration order. 2-way has ~5× more cat rows
and the order matters less.

### Headline-changing implication

> **CORRECTED 2026-05-15 ~17:30** — the original text here used the
> pre-13:11 numbers (C7 0.7908/0.8545); see the correction block above.

With the post-13:11 numbers, C7 (the 2-way winner) sits at
**0.8323 / 0.8142** — it **clears the 80/80 target** (83.2 P / 81.4 C)
and edges past the historical Sept 2025 production catalog
(0.8255 / 0.8350) on purity (+0.7pp) while giving up ~2pp completeness.
The post-patch pipeline is roughly a wash vs legacy on the balanced
metric — a small purity gain, a small completeness loss. Larger gains,
if any, are expected in metrics not in this headline (NHI accuracy,
sub-DLA purity, low-SNR completeness).

**C7 is the only cell that clears 80/80 on both axes.** No cell reaches
85/85. D7 trades ~11pp completeness for ~1.3pp purity (0.8387 / 0.7245) —
useful for a high-purity catalog but well off the balanced target.
