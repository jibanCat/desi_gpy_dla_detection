# 2026-05-18 — closing the 20.3-boundary purity gap: NHI-debias and band-BF tests

> Two debug-node tests of the FN/FP deep-dive's conclusion that the 85/85
> gap is sub-DLA NHI overestimation at the 20.3 boundary
> (`2026-05-18_fn_fp_deepdive.md`).
>
> **Verdict:**
> - **NHI-debias (post-hoc point-estimate correction): does NOT work.**
>   It is just another 1:1 P↔C trade — the joint frontier does not move.
> - **Band Bayes factor: modestly useful.** A real discriminator
>   (AUC 0.726, ~2:1-favourable trade) — the best lever found — but not a
>   silver bullet. Recommended as a new informational `dlacat` column.

## Test 1 — NHI-debias pass (`nhi_debias_test/`, job 53134746)

Applied a debias correction to `NHI_pred` on the V1 production-candidate
run and re-measured P/C — both in-sample and leave-one-healpix-out
cross-validated.

| correction (CV) | ΔP | ΔC |
|---|---:|---:|
| constant shift | +2.7pp | −4.3pp |
| smooth NHI/SNR correction | −0.7pp | +2.2pp |

**Refuted.** Post-hoc NHI debiasing is just another P↔C threshold knob;
the joint frontier does not move (CV best ≈ 0.797 / 0.920, purity
slightly *worse* than baseline). In-sample and CV agree, so it is not a
circularity artifact.

**Why** — a population subtlety the deep-dive missed: the +0.06 dex NHI
bias is carried by the **strong-DLA end**; **near the 20.3 floor the bias
is ≈ 0**. So a debias correction pushes essentially nothing across the
boundary — it reclassified 0 of 68 FPs. And the FP sub-DLAs and the weak
*true* DLAs are interleaved in the same NHI_pred ∈ [20.3, 20.6] band, so
any shift large enough to drop the FPs drops the weak TPs too.

The deep-dive correctly located *where* the problem is (the 20.3–20.6
boundary), but the fix is **not** a point-estimate correction. It needs a
model-side change that sharpens the NHI *posterior* at the boundary, or a
joint sub-DLA+DLA forward model. (The NHI bias is still worth fixing for
N_HI / CDDF accuracy — just not as a route to 85/85.)

## Test 2 — band Bayes factor (`band_bf_test/`)

For each detection, re-aggregate the stored QMC samples into a
band-restricted evidence and take the ratio (no re-inference):

    log BF = log( Z[logNHI∈20.3,20.6] / Z[logNHI∈20.0,20.3] )

This uses the *evidence shape* — exactly the model-side information the
point-estimate debias lacks. On borderline detections (NHI_pred ∈
[20.2,20.7], n=391):

| group | n | median log BF |
|---|---:|---:|
| true DLA (NHI_true≥20.3) | 203 | +2.68 |
| true sub-DLA (19.0–20.3) | 114 | +0.42 |
| no-truth-match FP | 68 | +2.34 |

**AUC (true DLA vs true sub-DLA) = 0.726** — a real, modest discriminator.
At a `log BF ≥ +0.5` cut, borderline-set purity rises 0.640 → 0.755
(+11.4pp) while keeping 80% of true DLAs — a **~2:1-favourable** trade
(≈77 contaminants removed per ≈40 true DLAs lost), versus the strict 1:1
of the p_DLA / NHI cuts and the NHI-debias.

**Limitation**: AUC 0.726 is modest, and the band-BF does **not** separate
true DLAs from the no-truth-match FPs (those look DLA-like, median +2.34).
It catches the NHI-overestimated *sub-DLAs* — which are 75% of the FP
population — not the rarer pure FPs.

## Recommendation

1. **NHI-debias is not the pre-launch fix.** Drop it from the critical
   path (keep it as a separate N_HI-accuracy improvement).
2. **Add `BAND_LOGBF` as an informational `dlacat.fits` column** —
   `log(Z_[20.3,20.6]/Z_[20.0,20.3])` per detection, computed from the
   QMC samples. Treat it like `NHI_CONSISTENCY_FLAG` / `PDLA_SATURATED_FLAG`
   (not folded into `DLAFLAG`); a downstream user can apply `BAND_LOGBF ≥
   0.5` for a higher-purity boundary cut (~+11pp borderline purity, −20%
   borderline recall). It is the best discriminator found — but on its
   own it does not reach 85/85.
3. The real frontier-mover remains a **model-side** change: a sharper NHI
   posterior at the DLA/sub-DLA boundary, or a joint sub-DLA+DLA forward
   model. The band-BF is a partial, cheap proxy for that.
