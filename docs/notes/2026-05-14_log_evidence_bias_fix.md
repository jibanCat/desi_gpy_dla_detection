# 2026-05-14 — dla_gp.py log-evidence −log(N) bias fix

## Bug

`process_sample` (`dla_gp.py:212-214`, and `:425-429` in the serial loop)
pre-subtracts `np.log(num_dla_samples)` from every per-sample
log-likelihood. The downstream Monte-Carlo evidence estimator

    log Z ≈ max(S) + log( nanmean(exp(S − max(S))) )

then evaluates to `log mean(exp(L_i)) − log(N)` instead of
`log mean(exp(L_i))` — biasing the 1-DLA (and k-DLA) log evidence **down
by `log(N)`** (≈ 10.8 for N = 50k).

At fixed N the bias is a constant offset. But comparing cells run at
different sample counts (e.g. PW 30k / 80k / 100k) the bias differs by
`log(N_cell / N_baseline)`, which silently contaminates any sweep that
varies `NUM_DLA_SAMPLES`. The prior session verified this empirically:
the C5/C6/C7 PW-count cells spread ~3.5pp on purity *in lockstep with
`log(N)`* pre-patch, and went flat (~1pp) post-patch.

## Fix — path (a): keep the bake-in, add +log(N) downstream

The per-sample `−log(N)` bake-in is **kept** because it preserves the
threshold semantic at `dla_gp.py:135` (`initial_logL > null_evidence`),
which compares `S_i = L_i − log N` against the *unbiased* null log
evidence. Instead, `+ np.log(self.params.num_dla_samples)` is added to
each downstream evidence formula and to `log_initial_logL` inside the
partition estimator.

**Touched sites** (all in `gpy_dla_detection/dla_gp.py`), each adds
exactly one `+log(N)` term:

| Line (post-patch) | Branch |
|---|---|
| ~458 | non-parallel `log_model_evidences` standard estimator (`+ lognorm`) |
| ~668 | parallel early-stop on empty valid_mask |
| ~728 | rejected-region `log_initial_logL` inside the partition |
| ~810 | FILTER fix #5 1-DLA branch (`initial_logL`) |
| ~836 | truncated branch `log_Z_trunc` |
| ~862 | standard branch (`filter_low_likelihood=False`) |
| ~895 | early-stop "D" mode pre-Occam `stop_lik` |

The truncated branch's `log_ratio = log(N) − log(n_initial)` correction
is left as-is — it is a separate, pre-existing design choice about how
the partition formula combines `Z_A` / `Z_B`, not part of this bias.

## Behaviour implication

`log_evidence(DLA)` shifts up by `+log(N)` uniformly across all k-DLA
models. `log_evidence(null)` from `null_gp.py` has no such bake-in and is
unchanged. So every DLA-vs-null Bayes factor moves by `+log(N) ≈ +10.8`
(N=50k) — `p(DLA|D)` shifts toward 1. This makes the historical
`p_DLA ≥ 0.99` cut a weaker filter (more spectra cross it):
n_cat grows ~50–100%, completeness rises ~5pp, purity drops ~5pp on the
2-way family (less on 3-way). The patch is **mathematically correct** —
relative ordering of cells is preserved.

To recover the *pre-patch* operating point, tighten the cut: the
N-invariant equivalent is `log BF ≥ log(99) + log(N) ≈ 15.4` (N=50k), or
`p_DLA ≥ 1 − 1/(99·N)`. See `HANDOFF.md` "p_DLA cut sweep" — `~7 nines`
recovers the pre-patch P/C within ~1pp.

## Validation

- **A/B test**: `docs/notes/2026-05-16_logn_patch_ab.md` — 3 patch-OFF
  vs 3 patch-ON replicates, measured Δ = −1.6pp P / +4.9pp C, well above
  the ~0.3pp noise floor; direction and magnitude as predicted.
- **Tests**: 94 network-free tests pass (`test_cddf_mock`,
  `test_cddf_calibration`, `test_generate_samples`, `test_voigt_v2_parity`,
  `test_lyb_veto`, `test_smoke_target_contamination`, `test_tau_eb_wiring`).
- **Sweep flatness**: post-patch C5/C6/C7 (PW 30k/80k/100k) flat within
  ~1pp, confirming the N-dependent bias is gone.
