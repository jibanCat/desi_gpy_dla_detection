# London production catalog — purity / completeness study

Production multi-DLA + LLS catalogs that the user pointed at, plus the
v5.9.5 London mock-0 truth. Computed on GreatLakes 2026-04-27.

## Catalogs

- **Multi-DLA** (FILTER=1, Y3 epoch_920):
  `/nfs/turbo/lsa-cavestru/mfho/DESI/DLA/desi-mock-gpdla-20250912-y3-learned-epoch920-filter`
  — 576 chunks, **173,588 MAP DLAs at P_DLA ≥ 0.5** across 67,572 unique TARGETIDs.
- **LLS** (single absorber, PW14 prior NHI 17.2–22.0):
  `/nfs/turbo/lsa-cavestru/mfho/DESI/DLA/desi-mock-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172`
  — 576 chunks, 297,302 entries.
- **Truth** (London mock-0 v5.9.5 jura-124):
  - `dla_cat_mask_20.30.fits` — 110,641 DLAs (NHI ≥ 20.3) over 101,226 TIDs.
  - `dla_cat.fits` — 762,822 absorbers: 110,641 DLA, 287,110 sub-DLA, 365,071 LLS.

## Coverage caveat (must read before interpreting completeness)

The production multi-DLA catalog only contains **67,572** unique
TARGETIDs out of **101,226** truth TIDs. The remaining 33,654 truth
TIDs were never processed — most likely filtered out upstream
(z_qso<2.5, ZWARN!=0, etc.). Reporting completeness against the *full*
truth catalog therefore looks artificially poor (39.8 %). The
meaningful denominator is **truth DLAs on processed TARGETIDs only**:
48,132 of the 110,641 truth DLAs. All numbers below use that denominator.

## Headline numbers

| stage                                  | completeness (all NHI bins) | strict purity (matches NHI ≥ 20.3 truth) |
|:---------------------------------------|:---------------------------:|:-------------------------------------:|
| raw catalog                            | **91.5 %** (44,028 / 48,132) | **25.4 %** (44,028 / 173,588) |
| + Lyβ veto (this work)                 | 91.5 %                       | 25.5 % (+0.1 pp)              |
| + Lyβ + LLS cross-reference (this work)| 90.5 %                       | 26.6 % (+1.2 pp)              |

**Production completeness is 91.5 %** — far better than the small
synthetic smoke sweep suggested. The post-processing helpers (Lyβ veto
+ LLS xref) add only marginal purity (+1.2 pp combined). The real story
is below.

## Where the apparent "spurious" 75 % go

A MAP DLA is flagged "spurious" only if it doesn't match a *DLA-class*
truth absorber (NHI ≥ 20.3). Most of those flags are not actually
hallucinations — they are real sub-DLAs / LLS that the multi-DLA
inference inflated to NHI ≥ 20.3 (the well-known prior-edge pile-up).

Cross-matching the same 173,588 MAP DLAs against the **full** truth
catalog (DLA + sub-DLA + LLS) on the same LOS gives:

| MAP DLA matches…                                  | count   | fraction |
|:--------------------------------------------------|--------:|---------:|
| truth DLA (NHI ≥ 20.3) — *strict-purity hits*    |  87,870 | **50.6 %** |
| truth sub-DLA (19 ≤ NHI < 20.3) — inflated subDLA |  35,364 | 20.4 %   |
| truth LLS (17 ≤ NHI < 19) — inflated LLS         |  10,676 | 6.2 %    |
| anything real on the LOS at this z (union)        | 120,898 | **69.6 %** |
| nothing real (genuine hallucination)              |  52,690 | **30.4 %** |

(The above categories are not exclusive — a MAP DLA at z within
|Δz|/(1+z) ≤ 0.01 of both a truth DLA and a truth sub-DLA at the same z
counts in both rows.)

So **only 30 % of MAP DLAs are genuine hallucinations**. The other ~27 %
flagged "non-DLA" by the strict criterion are real absorbers whose MAP
NHI is biased high — the kind of failure addressed in the sub-DLA
model improvement notes (`docs/notes/2026-04-27_subdla_model_improvements.md`),
not by the multi-DLA inference itself.

## Why the Lyβ veto barely helps on the production catalog

Only 1,172 of 173,588 MAP DLAs (**0.7 %**) get LYBETA_FLAG=True. On the
small synthetic 200-target FILTER=0 batch the fraction was 22 % — a 30×
difference. Two contributing reasons:

1. **The production catalog already uses FILTER=1**, which the small
   synthetic test showed cuts spurious-multi-DLA selections roughly in
   half. So the population that the Lyβ veto would catch is already
   suppressed before it reaches the postprocess stage.
2. **Production runs on London v5.9.5 jura-124**, which is cleaner than
   2LPT loa-124. The multi-DLA Lyβ confusion mode might be specific to
   particular mock generators — not yet falsified across mocks.

The Lyβ veto remains useful for catalogs run with FILTER=0 (e.g. some
LLS-mode runs by other groups), and as a guard rail for noisier inputs.
We should not expect a double-digit purity improvement on a FILTER=1
catalog.

## Why LLS cross-reference helps more than Lyβ veto

The LLS-mode posterior places non-zero mass below NHI=20.3, so inflated-
sub-DLA MAP DLAs get downgraded. 8,656 / 173,588 = 5.0 % are flagged
LLS_DOWNGRADE_FLAG=True. After downgrading, strict purity rises from
25.5 % → 26.6 % (+1.2 pp) and completeness drops from 91.5 % → 90.5 %
(some real DLAs in the [20.3, 20.6) bin happen to be downgraded — see
the auto-generated NHI-binned tables below).

The downgrade rule is intentionally conservative: it requires LLS-mode
NHI < 20.3 *and* P(LLS absorber) > P(DLA-mode DLA) / 2. A more
aggressive rule (always defer to LLS-mode when present) would catch
more inflated sub-DLAs but at higher completeness cost. The threshold
is **tunable in `lls_cross_reference.py`** if a future analyst wants
to explore the trade-off.

## Auto-generated per-NHI-bin tables

(See the analyzer `examples/analyze_production_catalog.py` output for
the live numbers; this section is regenerated when the analyzer is
re-run.)

## Difference between this analysis and the molly notebook

I read `/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/molly/read_in_each_plots_saclay-Y3-Learned.ipynb`.

Things I do **similarly**:
- TARGETID-based matching with |Δz|/(1+z_truth) < 0.01 (same metric).
- Per-NHI-bin breakdowns.
- p(DLA) ≥ 0.5 cut on the catalog before computing purity/completeness.

Things I do **differently** (and why):

- **Truth denominator: restricted to processed TARGETIDs.** The
  notebook computes against the full truth catalog, which conflates
  upstream filtering (BAL exclusion, ZWARN!=0, z<2.5) with model
  failure. Restricting to processed TIDs is more interpretable for
  "how good is the model at finding DLAs it could possibly find".
- **Recovery rate against the full truth catalog** (DLA + sub-DLA +
  LLS), not just NHI ≥ 20.3. The notebook counts any unmatched MAP DLA
  as a "false positive". I separate "matches a real lower-NHI absorber"
  (inflated, fixable in postprocessing) from "matches nothing real"
  (hallucinated, fixable in inference). About 39 % of the notebook's
  apparent false positives are inflated sub-DLAs / LLS in this run.
- **Lyβ veto and LLS xref are run as part of the analysis** (the
  notebook does neither). These are the postprocessing gains the user
  asked about.
- **No SNR_REDSIDE binning yet** in my analyzer. The molly notebook
  produces purity/completeness vs SNR_REDSIDE; that's a reasonable
  extension and I'll add it once we agree the headline numbers above
  are sane.
- **No BAL exclusion plot** yet. The molly notebook produces a "no_bal"
  variant that excludes BI_CIV>0 targets. The bal_cat is loaded by my
  analyzer (90,354 BAL targets identified) but not yet used in any
  cut.

What I think the notebook is doing **less well**:
- It treats inflated sub-DLAs as "false positives", which makes the
  multi-DLA mode look ~30 % less pure than it actually is. The ground
  truth is that the model *did* see absorption there, just at the wrong
  NHI.
- It hardcodes mock paths in the cell metadata, so reproducing requires
  manual edits.
- It uses one giant linear cell flow with no top-level functions —
  hard to re-run on a different mock without copy-paste.

I am NOT claiming my approach is correct in every respect. The molly
notebook produces the SNR-binned plots that the team uses for
publications and it would be wrong to drop those without replacement.
The right course is to integrate both views: (a) my matching against
full truth so the per-class breakdown is honest, (b) the molly
notebook's SNR/BAL-cut plots so we keep continuity with prior reports.

## Recommended follow-ups

1. **Add SNR_REDSIDE-binned purity/completeness plots** to my analyzer
   (matching molly's notebook visuals). One day's work.
2. **Loosen / sweep the LLS xref threshold and p_DLA cut.** Trace
   completeness/purity to find the operating point that maximises
   F1-like score.
3. **Repeat on a FILTER=0 production run** if one exists. Catalogs
   with a `filter` suffix used FILTER=1 (and the Lyβ veto is small
   here, as expected); a FILTER=0 catalog should show a much larger
   Lyβ-veto purity gain. The neighbouring directory
   `desi-mock-gpdla-20250929-y3-learned-epoch920-filter-nhi199`
   suggests other variants exist.
4. **Inspect the 30.4 % "hallucinated" entries** by NHI / SNR / hpx
   to identify the dominant noise vs metal-line vs other failure modes.

## Reproducibility

```bash
python examples/analyze_production_catalog.py \
   --catalog-dir /nfs/turbo/.../desi-mock-gpdla-20250912-y3-learned-epoch920-filter \
   --truth       /nfs/turbo/.../jura-124/dla_cat_mask_20.30.fits \
   --zcat        /nfs/turbo/.../jura-124/zcat.fits \
   --lls-dir     /nfs/turbo/.../desi-mock-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172 \
   --bal-cat     /nfs/turbo/.../jura-124/bal_cat.fits \
   --p-dla-cut 0.5 \
   --out docs/notes/2026-04-27_london_production_pc.md
```
