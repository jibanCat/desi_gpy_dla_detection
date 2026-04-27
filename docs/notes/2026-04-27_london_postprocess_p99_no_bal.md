# Post-processing efficacy on the London production catalog

Realistic operating point: **P_DLA ≥ 0.99**, **BAL LOS excluded** (BI_CIV>0).
This matches the historical convention used to report GP-DLA purity /
completeness on London mocks.

## Headline numbers

| stage                       | completeness | strict purity (1-to-1) | loose purity (matches ≥1 truth on LOS) |
|:----------------------------|:------------:|:----------------------:|:----------------------------------:|
| raw catalog                 | **81.8 %**   | 38.1 %                 | **76.0 %**                         |
| + Lyβ veto                  | 81.8 %       | 38.5 %                 | **76.7 %** (+0.7 pp)               |
| + Lyβ + LLS cross-reference | 81.2 % (-0.6 pp) | 39.7 %             | **79.2 %** (+3.2 pp from raw)      |

Headline takeaways:

- Loose purity 76.0 % matches the user's recollection of ~78 % under
  this operating point. The 1-to-1 strict purity 38.1 % is a stricter
  catalog-quality metric that counts duplicate / split detections as
  wrong. Both are reported so a downstream comparison with the molly
  notebook (which uses loose matching) is clean.
- **Lyβ veto** removes 910 / 94,458 = 1.0 % of MAP DLAs and gives
  +0.7 pp loose purity at zero completeness cost. Modest but free.
- **LLS cross-reference** removes 3,558 additional MAP DLAs (3.8 %
  of the catalog) and gives +2.5 pp additional loose purity (+3.2 pp
  total) for a 0.6 pp completeness cost. Favorable for any catalog
  use that values purity over completeness in the [20.3, 20.6) bin
  (where the LLS-mode posterior most often disagrees).

## Completeness vs NHI bin (after both post-processing steps)

| bin            |  total | matched |   rate |
|:--------------:|-------:|--------:|-------:|
| [20.3, 20.6)   | 16,565 |  11,886 | 71.8 % |
| [20.6, 21.0)   | 16,439 |  14,021 | 85.3 % |
| [21.0, 21.5)   |  9,085 |   8,095 | 89.1 % |
| [21.5, 23.5]   |  1,894 |   1,725 | 91.1 % |
| **all**        | 43,983 |  35,727 | **81.2 %** |

Completeness is steeply NHI-dependent — strong DLAs are recovered ≥90%,
weak DLAs near the prior edge ~72%. The LLS cross-reference dings
[20.3, 20.6) the most because that's the bin where DLA-mode and LLS-mode
posteriors most often disagree (consistent with the prior-edge effect
the user flagged).

## The two purity definitions, explained

**Strict 1-to-1 purity** (this analyzer): each MAP DLA can match at most
one truth DLA, and each truth DLA can be claimed by at most one MAP DLA.
If two MAP DLAs on the same LOS both fit the same truth DLA, only one
is "correct"; the other is a duplicate / split detection. This is the
metric used in catalog-quality literature (e.g. Wang+2021 CNN paper).

**Loose purity** (this analyzer + molly notebook): each MAP DLA is
"correct" if *any* truth DLA on the same LOS is within Δz/(1+z) ≤ 0.01.
Duplicates / splits both count as correct. More forgiving; the user's
historical 78 % number used this convention.

Both are reported here. **Strict is more honest about catalog quality;
loose makes for cleaner side-by-side comparison with prior reports.**

## Why these post-processing gains are smaller than the synthetic test

On the small 200-target synthetic FILTER=0 test the Lyβ veto removed
22 % of spurious detections; on this production FILTER=1 catalog it
removes 1 %. Two contributing reasons (also discussed in
`docs/notes/2026-04-27_lybeta_persistence_hypotheses.md`):

1. **FILTER=1 already suppresses most of the failure mode** the Lyβ
   veto targets. The veto stays useful as a guard rail and for
   FILTER=0 catalogs.
2. **London v5.9.5 jura-124 may be cleaner than 2LPT loa-124** in
   ways that haven't been characterised. Repeating on the Saclay
   juraLy8-124 production catalog would tell us whether the Lyβ-veto
   efficacy is mock-dependent.

The LLS cross-reference gain (+3.2 pp loose purity, +1.6 pp strict
purity) is the more substantial and reusable operating-point shift.

## Reproducibility

```bash
python examples/analyze_production_catalog.py \
   --catalog-dir /nfs/turbo/.../desi-mock-gpdla-20250912-y3-learned-epoch920-filter \
   --truth       /nfs/turbo/.../jura-124/dla_cat_mask_20.30.fits \
   --zcat        /nfs/turbo/.../jura-124/zcat.fits \
   --lls-dir     /nfs/turbo/.../desi-mock-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172 \
   --bal-cat     /nfs/turbo/.../jura-124/bal_cat.fits \
   --no-bal --p-dla-cut 0.99 \
   --out docs/notes/2026-04-27_london_postprocess_p99_no_bal.md
```
