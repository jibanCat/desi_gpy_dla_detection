# 2026-05-16 — +log(N) patch A/B validation (logn_patch_ab)

> **Status**: DONE (job 53021033, completed 2026-05-16 06:19).
> **Verdict: the 2026-05-14 dla_gp.py +log(N) patch behaves exactly as
> predicted** — it trades a small amount of purity for a larger
> completeness gain, and the effect is well above the run-to-run noise
> floor. The patch is mathematically correct (see HANDOFF "Patch summary")
> and this A/B confirms its empirical sign and magnitude.
>
> Sweep root:
> `/pscratch/sd/j/jibancat/prod533_5k_20260511/logn_patch_ab/`

## Design

Direct A/B of the `dla_gp.py` log-evidence `+log(N)` bias fix, holding
everything else at the C7 config (2-way, PW 100k, NHI [17.2,22], τ-EB
null, London-0 5k slice, fixed molly recipe).

- **patch-OFF** arm: P0/P1/P2 — three replicates run via the
  `/pscratch/sd/j/jibancat/gpdla_patchoff` git worktree, whose
  `dla_gp.py` is at the committed pre-patch state.
- **patch-ON** arm: G0/G1/G2 — three replicates, identical C7 config,
  patched `dla_gp.py`. (These are the `determinism_sweep` cells, reused.)

Three replicates per arm means the comparison is made against the
measured ~0.3pp run-to-run noise floor, not a single draw.

## Results

Refreshed 2026-05-17 (new DLAFLAG convention — NHI_INCONSISTENT no longer gated).

| cell | arm | P | C | n_cat |
|---|---|---:|---:|---:|
| P0 | patch-OFF | 0.8179 | 0.7926 | 2808 |
| P1 | patch-OFF | 0.8045 | 0.7771 | 2812 |
| P2 | patch-OFF | 0.8103 | 0.7802 | 2809 |
| G0 | patch-ON | 0.7913 | 0.8452 | 4519 |
| G1 | patch-ON | 0.7901 | 0.8390 | 4517 |
| G2 | patch-ON | 0.7855 | 0.8390 | 4520 |

**Per-arm means:**

| arm | n | purity | completeness | n_cat |
|---|---:|---:|---:|---:|
| patch-OFF | 3 | 0.8109 | 0.7833 | ~2810 |
| patch-ON  | 3 | 0.7890 | 0.8411 | ~4519 |
| **Δ (ON − OFF)** | | **−2.2 pp** | **+5.8 pp** | **+1709 (+61%)** |

## Interpretation

The patch raises `log_evidence(DLA)` by `+log(N)` uniformly, inflating
every DLA-vs-null Bayes factor, so more spectra cross the p_DLA≥0.99 cut.
n_cat grows +61% (2810→4519), completeness rises +5.8pp, purity falls
−2.2pp. This is the predicted direction and the within-arm spread
(P: 0.80–0.82 OFF, 0.786–0.791 ON; C: 0.777–0.793 OFF, 0.839–0.845 ON)
is ~1–2pp — so the −2.2/+5.8 shift is unambiguously real, not noise.

The patch is **net-favourable for the CDDF LLS use case**, where
completeness matters: it buys ~3× more recall headroom per pp of purity
given up. It is the correct fix regardless — the pre-patch code was
biased — but this A/B quantifies that adopting it (rather than tightening
the p_DLA cut to undo it) is the right call for this science goal.

## Cross-reference

- Patch description + touched lines: `HANDOFF.md` "Patch summary".
- The p_DLA-cut sweep that *recovers* the pre-patch operating point
  (log-BF ≥ 15.4): `HANDOFF.md` "p_DLA cut sweep".
- Noise-floor source: `2026-05-16_config_confirmations.md` (determinism).
