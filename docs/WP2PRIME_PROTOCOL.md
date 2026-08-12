# WP-2′ — frozen prospective protocol (v1, 2026-08-12; PI-approved)

One prospective, one-shot, frozen gate testing (i) mock-family transport of
the full ratified calibrated machinery at fold grade and (ii) the
pre-selected candidate reporting floor. Approved by PI rulings of
2026-08-12 (pass-2 checkpoint + §27 of the execution ruling). **Any
substantive change to the items below voids the approval (PI §28).**

## Frozen scientific choices

| item | value |
|---|---|
| candidate floor | **N̂ ≥ 20.0**, 0.2-dex reported bins from 20.0 |
| aligned secondary | N̂ ≥ 20.1 grain (frozen variant, reported alongside) |
| fallback / control | ≥ 20.3 (the ratified + holdout-certified domain) |
| held-out families | **London-0** and **Saclay** phaseB packs (calibration blocks = 2LPT-0-calibrated objects; observed counts/truth = each mock's own) |
| operator | `p1_natpair_ck/v1` (sha `6893a9ef…`), READ-ONLY; latent support truth ≥ 19.5 + per-mock M(<19.5) by the committed chain-bridge definition; K never renormalized; no kernel/granularity change |
| FP-transfer nuisance | the frozen in-pack `t_sigma` widths; expected direction pre-registered: cross-family FP over-supply ≈ 1.45× (London) / 1.31× (Saclay), Phase-A record |
| predictions frozen | per mock, committed BEFORE any observed count is read: (a) control domain [20.3, 21.5) reported grain; (b) candidate domain [20.0, 21.5) + aligned [20.1, 21.5); (c) per-z-bin ≥ 20.3 over B1–B4 ∩ [2.0, 3.5]; (d) group-level Layer-B statistic with t_sigma; (e) per-SNR-stratum residuals of observed [19.9, 20.1) — labeled DIAGNOSTIC, not a gate |
| scoring | Layer-B predictive p at the ratified 0.01 per mock; per-domain paired 0.2-dex χ²/dof ≤ 3 conditional diagnostic; Holm across {2 mocks} × {candidate, aligned} for the floor decision |
| read discipline | ONE read per mock; no retries; no post-read floor tuning (the fine-grid scan of 2026-08-12 is the tuning, completed pre-read) |

## Pre-registered decision rule

1. candidate PASS on BOTH families AND control PASS on both
   ⇒ **20.0 becomes the Paper-1 primary floor** (20.1-aligned variant
   reported as robustness);
2. candidate FAIL anywhere, control PASS on both
   ⇒ **20.3 primary**; [20.0, 20.3) demoted to labeled diagnostics;
3. control FAIL on either family
   ⇒ **hard stop, PI checkpoint** (family transport itself in question —
   larger than the floor decision).

## Engineering status (non-substantive; PI §28 allows)

- runner: pack-parameterized generalization of the committed
  `p1_refold.py` predict-then-close discipline (PACK becomes an argument;
  same guards: rebuild ≤ 1e-8, kernel integer identity, migration totals
  gate per mock);
- prerequisites to build before freezing predictions: London-0/Saclay
  completeness caches via the committed cache builder, and per-mock
  chain-bridge M counts;
- estimated cost: ≈ 0.2 CPU-h compute; execution follows on the canonical
  forward branch at a recorded SHA.

Statistical protocol unchanged from the PI-approved pass-2 definition;
this document is the committed protocol of record.
