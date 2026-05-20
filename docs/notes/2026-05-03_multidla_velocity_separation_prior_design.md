# Tier 2 #2b — multi-DLA velocity-separation prior: design

> **Status**: Approved (2026-05-03) for Option A as next step; Option C
> kept as the ultimate goal for a follow-up PR.
> **Author**: Claude, 2026-05-03 session.
> **Goal**: Replace the hard 3000 km/s minimum-separation cutoff in
> the multi-DLA model with a smooth, clustering-informed prior on
> Δv between DLAs along the same line of sight.

## Problem characterization

The current multi-DLA model imposes a **hard floor** at
`min_z_separation = 3000 km/s` on every DLA pair, applied at
`gpy_dla_detection/dla_meanflux_gp.py:326-331` (and the analogous
parallel path at `:473-477`):

```python
ind = np.any(
    np.diff(np.sort(all_z_dlas, axis=0), axis=0)
    < self.min_z_separation,
    axis=0,
)
sample_log_likelihoods[ind, num_dlas] = np.nan
```

The implicit prior is therefore

```
p(Δv) ∝ Θ(|Δv| − 3000 km/s)
```

— zero below the cutoff, uniform above. This carries **no clustering
information**, which is wrong because DLAs cluster (`b_DLA ≈ 2`,
Pérez-Ràfols+2018, Pérez-Ràfols+2023) and pairs at small Δv are
correspondingly more probable than uniform.

### Observed failure mode

Multi-DLA conflation cases (e.g. TID 60167537: truth log NHI 20.6 fit at
log NHI 22.3) suggest the engine isn't penalizing or rewarding pair
configurations using clustering information — pairs are equally likely
at any Δv > 3000 km/s, so a single broad fit can absorb what should
have been two narrower DLAs.

## Three options of increasing scope

### Option A — empirical clustering table from 2LPT mock-0 truth (next PR)

**Inputs.** 2LPT mock-0 truth catalog at
`/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits`.
Contains all truth DLAs with `(TARGETID, Z_DLA, NHI)`.

**Build the prior table.**
1. Group DLAs by `TARGETID`. Keep LOS with ≥ 2 DLAs.
2. For each LOS pair, compute `Δv = c × |z2 − z1| / (1 + z̄)` km/s
   where `z̄ = (z1 + z2) / 2`.
3. Form a 1-D histogram `n_pairs(Δv)` on a velocity grid
   `[0, 60000) km/s` in 500 km/s bins.
4. Build the random expectation `n_random(Δv)` by bootstrap
   permutation: for each LOS, randomly draw two z values from the
   global z distribution (same number of pairs as truth). Repeat
   N=100 times and average.
5. Compute `1 + ξ(Δv) = n_pairs(Δv) / n_random(Δv)`.
6. Smooth (Savitzky-Golay or B-spline) to suppress shot noise at
   small Δv where pair counts are low.
7. Save as a `.npz` at `data/clustering/dla_pair_prior_2lpt.npz`
   with keys: `delta_v_centers`, `log_one_plus_xi`, `n_pairs_obs`,
   `n_pairs_random`, `provenance` (string with mock path + date).

**Wire at inference.** Replace the NaN mask in
`dla_meanflux_gp.py` with a multiplicative weight:

```python
# Pseudocode for the multi-DLA pair masking call site (:326-331).
# Two configurations to test, gated by a CLI flag --pair-prior-mode:
#
#   "mask"  — keep current Θ(Δv − 3000 km/s) cutoff. Baseline.
#   "table" — replace with multiplicative log-weight from the table.
#             For each pair Δv_ij in the sample,
#                 log_w = np.log1p(xi_table(Δv_ij))
#             Sum over all consecutive pairs in the sample;
#                 sample_log_likelihoods[i, num_dlas] += sum(log_w)
#             Pairs with Δv below the table's smallest finite bin
#             still get -inf (resolution floor).
```

The flag default in this PR is `"mask"` (no behavior change). User can
opt into `"table"` to validate.

**Pros.** Empirical, no cosmology dependency, fits the 1-PR scope. The
2LPT mock is the same realization the trainer was validated on, so we
know the underlying physics matches the mock generator (LyaCoLoRe
biased-tracer model with `b_DLA ≈ 2`).

**Cons.** Mock-vs-real divergence: 2LPT clustering may not match
real DESI DLAs, especially at small Δv where mock resolution effects
dominate. The bootstrap random expectation may have residual bias near
the survey window edges.

**Estimated effort.** ~3–5 days: 1 day for the measurement script, 1
day for the inference-side wiring + tests, 1–3 days for the validation
campaign + write-up.

### Option B — analytic biased-tracer model (skipped this PR)

Use the linear-bias model `ξ(r) = b_DLA² × ξ_matter(r)` with a fiducial
cosmology and `b_DLA = 2.0 ± 0.1` from eBOSS literature. Convert
`Δv ↔ r` via `r ≈ Δv / H(z̄)`. Same inference-side wiring as A; only
the table source changes.

Pros: principled, single tunable parameter, portable to other surveys.
Cons: requires CAMB/CLASS or pre-tabulated `ξ_matter`; sensitivity to
fiducial cosmology choice; the linear-bias approximation breaks down at
small r (where DLA-DLA pair clustering is dominated by halo-occupation
physics, not linear theory).

Decision: **deferred**. If Option A's empirical mock-derived table
shows the signal but raises mock-fidelity concerns, Option B is the
natural follow-up.

### Option C — hierarchical prior with marginalized bias (ultimate goal)

Same prior shape as Option B, but `b_DLA` becomes a per-spectrum (or
per-survey) nuisance parameter with a Gaussian hyperprior centered on
the eBOSS measurement. Bayes factor analytically marginalizes over
`b_DLA` (works because the prior shape is `1 + b² × ξ_fid` — quadratic
in `b`).

**Why this is the right end-state.**
1. Treats clustering uncertainty as an inference output, not an input.
2. The catalog gets a `b_DLA_posterior` column for free — independently
   interesting science (spatial bias of HCD-host halos).
3. Removes the mock-vs-real and fiducial-cosmology dependencies that
   weaken Options A and B.

**Why we don't ship it next.** Adds a new sampling axis (or analytic
marginalization machinery) to the multi-DLA evidence integral. Likely
2–3 weeks of design + implementation + validation. Best done after
Options A or B prove the basic infrastructure (table-based pair prior)
works at production scale.

## Implementation plan (Option A)

### Files touched

| File | Change |
|---|---|
| `examples/measure_dla_pair_clustering.py` (new) | Reads truth catalog, computes ξ(Δv), writes `data/clustering/dla_pair_prior_2lpt.npz`. |
| `data/clustering/dla_pair_prior_2lpt.npz` (new) | Tabulated prior; ~10 KB. Committed. |
| `gpy_dla_detection/clustering.py` (new) | `load_pair_prior(path)` → callable `xi(delta_v_km_s) -> 1 + xi`. |
| `gpy_dla_detection/dla_meanflux_gp.py` | `pair_prior_mode: str = "mask"` constructor arg. New helper `_apply_pair_prior(...)` replacing the mask block at `:326-331` and `:473-477`. |
| `desi-DLAGP.py` | Add `--pair-prior-mode {mask, table}` CLI flag (default `mask`). Add `--pair-prior-table` PATH (default `data/clustering/dla_pair_prior_2lpt.npz`). |
| `tests/test_pair_prior.py` (new) | Unit: log_w == 0 when xi == 0 everywhere (table-mode reduces to no-op); log_w == -inf below resolution floor; mask-mode bit-exact reproduces current behaviour. |

### Validation

Hold out 2LPT mock-1 truth as the validation set (the prior is built on
mock-0). On a sub-sample of n=100 LOS with truth `n_DLA ≥ 2`:

1. Run inference with `--pair-prior-mode mask` (baseline) and
   `--pair-prior-mode table`.
2. For each conflated case (truth pair within 5000 km/s, or single MAP
   DLA with NHI ≥ truth_NHI_sum + 0.5 dex), measure NHI bias.
3. **Success criterion**: ≥ 25 % reduction in median |Δlog_NHI| on
   conflated cases when switching from mask → table.
4. **No-regression criterion**: on n=100 single-DLA-truth LOS, the
   table mode must not increase false-positive rate beyond mask mode
   by more than 1 %.

Cross-check: also run with the table built from mock-1 (held-out
swap). Stability of the bias-reduction number across mock-0-built vs
mock-1-built tables tells us how mock-realisation-dependent the
result is.

### Validation outputs to commit

- `docs/notes/<date>_pair_prior_validation.md` — table comparing mask
  vs table on the n=100 set, with conflation rate, NHI bias, FPR.
- `docs/notes/<date>_pair_prior_xi_table.png` — plot of `1 + ξ(Δv)`
  with both mock-0 and mock-1 measurements, error bands.

## Decisions captured (from 2026-05-03 session)

| Question | Decision |
|---|---|
| Reference dataset for ξ measurement | 2LPT mock-0 truth (mock-1 held out for validation) |
| Velocity range to tabulate | 0–60000 km/s |
| Treatment of Δv < ~1500 km/s (resolution floor) | **Test both**: keep hard mask in `--pair-prior-mode mask`; let the data speak in `--pair-prior-mode table`. Compare. |
| Storage location | `data/clustering/` (in-repo, ~10 KB) |
| Validation target | n=100 multi-DLA-truth LOS from 2LPT mock-1; success = ≥ 25 % reduction in NHI bias on conflated cases |
| Option B (analytic biased-tracer) | Deferred. |
| Option C (hierarchical marginalized bias) | Ultimate goal — separate PR after Option A lands and validates. |

## What I will NOT do without further sign-off

- Modify `min_z_separation` defaults or signatures in `civ_gp.py` /
  `subdla_meanflux_gp.py` (the multi-DLA prior change is local to
  `dla_meanflux_gp.py`; the others stay as-is).
- Implement Option B (analytic) or Option C (hierarchical).
- Run any of the prior-measurement code on the truth catalogs before
  the design is reviewed.

## Files this builds on

- `docs/notes/2026-05-01_post_pr5_priorities.md` — original Tier 2 list,
  including the `b_DLA ≈ 2` reference and user's "scan + multi-DLA"
  framing.
- `gpy_dla_detection/dla_meanflux_gp.py:216, :326-331, :473-477` — the
  current hard-mask implementation.
