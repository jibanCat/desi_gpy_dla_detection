# Phase-C2 production budget — MEASURED requirement, returned to the PI

Per rulings §12: **no documented PI-approved compute envelope for Phase C
exists** (searched: this repo's docs/, the notes repo, the rulings text —
2026-08-06). The default authorization rule therefore applies: Phase C1
is complete (design, pilot, bridge design, preimage, sizing), this budget
returns to the PI, and **Stage 2 has not been launched**. Everything
below is measured or pilot-re-measured; nothing assumes an unlimited
budget. The standing project rule (sbatch > ~500 CPU-h needs PI
sign-off; allocation cap ~5,000 CPU-h) is respected in both options.

## A. Response calibration campaign (frozen design, pilot-verified)

| item | value (measured basis) |
|---|---|
| injections (incl. completeness + 15% retries) | 2,533 |
| unit cost | 106 CPU-s/spec (pilot jobs 56605518/9; planning had 167) |
| finder CPU | ≈ 75 CPU-h |
| generation + margin + already-spent pilot (8.3) | ≈ **≤ 110 CPU-h total** |
| storage | ≈ 6 GB scratch (≈ 40 healpix arms) |
| jobs / wall | ~10 sbatch, wall ≈ 3–5 h end-to-end |
| achieved criteria (frozen §9) | σ(G3 pred) = 112 counts (≤ 116.7 required); power = 0.926 (≥ 0.90 at two-sided α = 0.01 vs the 450.25-count effect) |
| expected anchor yield | 96% pairs/injection (pilot-measured) |
| failure/retry allowance | 15% (one fast-fail step observed and self-covered in the pilot) |

## B. Independent-FP expansion (design committed; §15)

| target | new events | CPU-h (2.29/event measured) |
|---|---|---|
| FP total to ±5% | +311 (loa-0 mock-0, new healpix) | ≈ 710 |
| FP total to ±3% | +1,022 | ≈ 2,340 |
| Saclay control (jura-0), ~100 ev | | ≈ 230 |
| Saclay method-bias pair, ~100 ev | | ≈ 230 |
| London natural control, ~100 ev | | ≈ 230 |
| mock-1 loa-0 held-out, ~150 ev | | ≈ 340 |

## C. The two authorization options

| | option 5% | option 3% |
|---|---|---|
| response campaign | 110 | 110 |
| FP program | 1,740 | 3,370 |
| **Phase-C2 total** | **≈ 1,850 CPU-h** | **≈ 3,480 CPU-h** |
| share of the ~5,000 CPU-h cap | 37% | 70% |
| GPU | 0 | 0 |
| storage | ≈ 10 GB | ≈ 15 GB |

Recommendation (PI's decision): option 5% suffices for the Phase-C
scientific question — it makes calibration-sample uncertainty
subdominant for the shape/transport conclusions at ~⅓ of the allocation,
and the sequential stopping rule (FP design §7) can halt it early; the
3% option is insurance against the conditional-shape stability criteria
not converging, purchasable later as an increment (+1,630 CPU-h) under
the same frozen roles without redesign.

Power statement for the checkpoint: at the measured pilot variances the
FROZEN response-campaign design delivers σ(G3) = 112 counts and 0.926
power; if the PI authorizes nothing, the current response's G3-relevant
uncertainty (≈ 96 counts statistical from the single ~360-pair top-anchor
region, PLUS the unquantified clamped-region systematic that the pilot
observed at the −0.03…−0.14 dex scale) remains unmeasured and the frozen
Phase-B conclusion cannot be tested further.
