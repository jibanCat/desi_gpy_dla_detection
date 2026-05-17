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

| cell | arm | P | C | n_cat |
|---|---|---:|---:|---:|
| P0 | patch-OFF | 0.8403 | 0.7492 | 2808 |
| P1 | patch-OFF | 0.8448 | 0.7585 | 2812 |
| P2 | patch-OFF | 0.8472 | 0.7554 | 2809 |
| G0 | patch-ON | 0.8280 | 0.8050 | 4519 |
| G1 | patch-ON | 0.8301 | 0.8019 | 4517 |
| G2 | patch-ON | 0.8275 | 0.8019 | 4520 |

**Per-arm means** (`VERDICT.txt`):

| arm | n | purity | completeness | n_cat |
|---|---:|---:|---:|---:|
| patch-OFF | 3 | 0.8441 | 0.7544 | ~2810 |
| patch-ON  | 3 | 0.8285 | 0.8029 | ~4519 |
| **Δ (ON − OFF)** | | **−1.56 pp** | **+4.85 pp** | **+1709 (+61%)** |

## Interpretation

The patch raises `log_evidence(DLA)` by `+log(N)` uniformly, inflating
every DLA-vs-null Bayes factor, so more spectra cross the p_DLA≥0.99 cut.
n_cat grows +61% (2810→4519), completeness rises +4.85pp, purity falls
−1.56pp. This is the predicted direction and the within-arm spread
(P: 0.840–0.847 OFF, 0.828–0.830 ON; C: 0.749–0.759 OFF, 0.802–0.805 ON)
is ~1pp — so the −1.56/+4.85 shift is unambiguously real, not noise.

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
