# Consolidated checkpoint — 2026-08-05

**Status: WORK IN PROGRESS. Nothing here is paper-facing.** No artifact in either lineage is
marked `paper_facing: true` (verified by walking every tracked JSON). No rung 10, no campaign,
no freeze, no tag, no lineage merge, no inference PR.

All values are **mock-derived** (2LPT-0 / London-0 / Saclay-0 / loa-0). No real-DESI value
appears in this repository.

---

## 1. Branch tips and worktrees

| branch | tip | role |
|---|---|---|
| `lls-subdla-cddf` | `1533333` | provenance, guards, tombstones, paper-safety layer |
| `hbi-mcmc-threeroute` | `bed0483` | inference core; kernel + FP repair |
| `hbi-readme-figs-fix` | `d943bfe` | tutorial series, PR #21 — separate, untouched |

Worktrees, reduced 9 → 3 and now at the intended endpoint:
`/home/mfho/desi_gpy_dla_detection` (primary), `/home/mfho/hbi_mcmc_wt`,
`/home/mfho/hbi_tutorial_wt`.

**The two lineages have not been merged into each other and must not be.**

### Commits

Guard lineage, on top of `0dfa060`:

| commit | subject |
|---|---|
| `5d8fda0` | Integrate `wip/tombstones-lineage` — retire four contaminated artifact identities |
| `dc8171c` | Reconcile the primary tree's untracked files — six are load-bearing, three were stale |
| `1533333` | Tombstone guards: close the registry against the filesystem; make the CLEAN recoverability claim checkable |

Inference lineage, on top of `ecc06cb`:

| commit | subject |
|---|---|
| `c4cd67c` | Integrate `wip/gate-ratification` — the authority record, and a guard with a MEASURED false-alarm rate |
| `bdae1a4` | Integrate `wip/adopt-basis-pad-window` — the 0.2-dex basis, and ONE authority source |
| `9461d5a` | Integrate `wip/spectral-window` — the analysis-window axis, and the cell that needed both |
| `a12385f` | window_study: the LATENT basis is a second axis, and the 0.2-dex cell is now expressible |
| `bed0483` | the `lya_lyb × 0.2-dex` cell: measured, and it does not change the closure picture |

Source branches are preserved at their tips and were not deleted: `wip/gate-ratification`
`2ef5445`, `wip/adopt-basis-pad-window` `7a9c0a7`, `wip/spectral-window` `3383939`,
`wip/tombstones-lineage` `0d95048`.

### Archived material

Removing a worktree destroys gitignored files permanently. Two sets were archived to
`/home/mfho/slurm_log_archive/` **before** removal, with a README recording provenance:

- `clustering_prior_wt_production_logs_2026-08-05.tar.gz` — 68 GreatLakes production log files
  (45 non-empty) that existed only in that checkout. `gpdla_gl_*.log` carry run provenance
  (LEARNED_FILE, MAX_LAMBDA, MAX_DLAS, NUM_DLA_SAMPLES, FILTER).
- `removed_untracked_calccddf_scripts_2026-08-05.tar.gz` — the three untracked
  `CDDF_analysis/hbi/calccddf_vs_hbi{,_artifact,_fig}.py` removed in `dc8171c`, preserved with
  full sha256 so the byte-identity claim stays independently checkable after deletion:

| file | bytes | sha256 | relation to git |
|---|---|---|---|
| `calccddf_vs_hbi_artifact.py` | 6029 | `01a437e9c6a5fb9b1d54f77d12bfb0e517b0cf506ed4ac02fac9930b9ac5b2ac` | byte-identical to `7ca9ea6` (verified against `git show`) |
| `calccddf_vs_hbi_fig.py` | 5234 | `76dc74b147dfc380442a0a8f66abe3e10de28c5db745cb7d99e33f1175c78602` | byte-identical to `7ca9ea6` (verified) |
| `calccddf_vs_hbi.py` | 12692 | `245ba4bc6c36…` | differs from `7ca9ea6` by 18 docstring lines only — the delta DELETES a "WIP — UNVALIDATED" banner |

Retired artifact identities carry stamped tombstones with verified byte counts and digests:
`lls_recovery_figures.json` 6407 / `84c0c802b44f…` (CLEAN);
`subdla_edge_systematic.json` 14192 / `b22c729d7540…` (DIRTY);
`subdla_floor_mc_band.json` 9340 / `4b37051ed0c7…` (DIRTY);
`subdla_mock_headline.json` 59484 / `b13adddae709…` (DIRTY).
All four re-verified by an independent referee.

---

## 2. The spectral-window × basis result — 0 of 24

The 2026-07-30 checkpoint was PARTIAL: no absolute number stated the adopted configuration
under the wider window, because `lya_lyb × 0.2-dex` needed code from two unmerged branches at
once. It is now measured.

**Nothing closes. 0/12 at 0.1 dex and 0/12 at 0.2 dex — 0 of 24 across the matched cross.**
Identical with and without the two unratified |z| arms
(`verdict_unchanged_without_the_unapproved_arms = true`).

Margins against the ratified `chi2_dof_max ≤ 3`:

| basis | best | worst |
|---|---|---|
| 0.1 dex | **9.8685×** (`london0 \| lya_only \| clamp=hi`, χ²/dof 29.605561515514744) | 35.5286× |
| 0.2 dex | **9.8009×** (same config, 29.402735695078178) | 30.3682× |

The adopted 0.2-dex basis improves χ²/dof in all 12 cells by **0.69 %–15.85 %** — on the best
configuration that is 0.7 % off a factor of ten. **Coarsening the latent basis does not move
the closure picture.** The wider `lya_lyb` window is worse than `lya_only` on all three mocks
at both clamps, unanimous 6/6 on the scale-free `rms_frac_dev` and 6/6 on χ²/dof — the same
verdict as at 0.1 dex.

Two results that complicate a naive "0.2-dex is better" reading:

- `rms_frac_dev` improves at 0.2 dex in 11 of 12 cells; the exception is the **best**
  configuration, which gets **worse** (0.084360 → 0.086348). Coarsening buys χ²/dof there
  almost entirely through the level, not the shape.
- The excluded high-N residual (≥ 21.6) moves **further from 1 in all 12** configurations
  (e.g. `2lpt0|lya_only` 1.1687 → 1.1979). The coarse basis trades in-window fit against the
  D2 tail.

Support-matching was checked, not assumed: per (mock, window), `counts_total`, `dX_total` and
`truth_total` are identical to every printed digit across bases; only `n_basis_bins` moves
(34 → 16). So χ²/dof is comparable across bases at fixed window; the counts confound is a
window effect only.

---

## 3. The FP normalisation defect — confirmed, and NOT sufficient

Model A's forward false-positive term is under-normalised by **exactly `pack.fp_ell_eff` ≈
13.59**. Audited, then attacked by two independent adversarial refuters with different lenses.

**Why it is certain** — three independent lines:

1. The loa-0 product's OWN definition (`build_loa0_fp_product.py:39-46`) is
   `μ_FP = (N_prod/N_sl_loa0)·N_FP_loa0·(1−η)` = `fp_w · ell_eff · lam_fp`.
   `forward.py:452` folds `fp_w · lam_fp` — the `×ell_eff` is absent.
2. `ell_eff·fp_w = 2255.0000 = n_sl_loa0` exactly, and `ell_eff·fp_w² = n_sl` per mock. A
   four-row dimensional table over the candidate units of `lam_fp` has **no consistent entry**.
   `ell_eff = ns0/vol_scale`, matching neither natural exposure convention.
3. `ell_eff` is **exactly inert** in the source route — measured with `ell_eff ∈ {1, 13.59,
   100}`, identical mean AND sd (2318.6449 ± 48.5914), because `Gamma(a,1/ℓ)·ℓ ≡ Gamma(a,1)`.
   A bookkeeping constant was promoted into a live Poisson exposure, where it does not cancel.

🔴 **Why no test could have caught it:** `pack.py:846-851`'s synthetic generator uses the SAME
convention, so no synthetic rung and no SBC could detect it, and no test pins
`w/ℓ == vol_scale`. **Any repair must fix the generator in the same change.**

The repair options are **identical in mean AND variance** — μ_FP ~ `Gamma(n+½, ℓ/w)`, depending
only on the ratio. Measured: 14,884 ± 1,592 vs 14,882 ± 1,609. An earlier claim that they
differ in variance is **wrong**. So the repair is a code-organisation choice, not a science one.

### 🔴 What did NOT survive — do not overclaim this fix

- **It does not close.** Window χ²/dof `56.58→22.22` (2lpt0), `40.16→28.39` (london0),
  `44.21→25.77` (saclay0) against a ratified gate of **3**. A scan over FP scale 0–40 finds
  **no scale** below ~19.6. FP-independent residual χ²/dof ≈ 20.
- **An arbitrary shape fits better.** Six alternative additive components renormalised to the
  same total: `exp(−c/1 bin)` — physically unmotivated, one parameter — gives window χ²/dof
  **19.95 / 14.92 / 15.44**, beating the loa-0 shape on **all three mocks**. The χ² gain shows
  ~13.6k counts are missing from the bottom two n̂ bins; it does **not** show they are forest
  false positives.
- **The counting identity holds only on the calibration twin.**
  `(obs−mu)/(mu_fp·(ell−1))` = **0.990** (2lpt0) but **0.677** (london0), **0.821** (saclay0).
- **The survey counts prefer the other explanation.** Exact non-negative Poisson MLE:
  Δdeviance for fixing `a_fp` at `ell_eff` versus free is **+2,519 / +3,478 / +3,089 on one
  dof**. The fix is **not identifiable from count data**; it rests entirely on the loa-0
  likelihood.

**What IS robust:** the deficit *is* the catalogue-unmatched population —
`(obs−mu)/census` = **0.977 / 0.963 / 1.057** on all three. Not a matching artifact:
quadrupling the velocity tolerance to a physically absurd 12,000 km/s removes only **17 %**.

**Scale caveat:** the entire loa-0 FP model is **89 detections from 3 healpix / 2,255
sightlines**, extrapolated by `vol_scale ≈ 165.9`. Poisson precision 10.6 %. Above N̂ = 20.2
loa-0 measures **zero** FP, so its shape is untestable in the DLA tier.

---

## 4. Defects found and fixed

### One-sided support — occurrences 8–11 of this project's signature class

| # | defect | measured |
|---|---|---|
| 8 | **Farr N_eff gate fail-open** (`model_a.py:507-509`): `n_cal = molly_n_tot.sum()` over ALL cells vs `n_obs` on the observed grid | `molly172` spans 17.2–inf vs nhat 19.5–22.4 → **4.4921 PASS**; support-matched to N≥19.5 → **1.6664 FAIL**; `const_extrap` (matched) → 1.6664 FAIL. **An armed production gate passes only via a convention choice.** NOT FIXED — reported. |
| 9 | **FF vs HBI ΔX**: same mock, same z-range, same SNR cut | FF divides by ΔX = 596,901 over 921,027 sightlines; the pack has 374,177 / 500,506. **2.46× in sightlines, 1.193× in ΔX.** NOT FIXED — reported. |
| 10 | **BAL veto one-sided in the FP weight** | `N_prod = 374,177` excludes BAL targets; `N_sl_loa0 = 2,255` does not (1,904 after veto); 19 of 89 detections sit on BAL targets. Support-matched rate 70/1904 = 0.03676 vs committed 89/2255 = 0.03947. NOT FIXED — reported. |
| 11 | **Tombstone registry one-sided** — guards iterated a whitelist, not the directory | A rogue 5th tombstone carrying `retirement.smuggled_R0 = 0.9137` (a float science value), a bogus defect code and `paper_facing: true` passed **31 tests green** with the new guards deselected. **FIXED** (`1533333`). |

Also: **`molly172` splices two definitions of "found"**. `cmp_min_pred_nhi` defaults to the
matrix's own first edge (`cddf_catalog_hbi.py:777-778`, `:1588-1589`, used at `:1607`), so
nhi195 ⇒ N̂>19.5 and nhi172 ⇒ N̂>17.2 — measured **1.6–3.6× apart** in [19.0,19.5), agreeing
bit-for-bit above 19.5. Load-bearing exactly on the pad. NOT FIXED — reported.

### Ratification and authority

- **Three fabricated-authority sites**, two live at session start, all now retracted. Decision 8
  ratified three things and called `|z| ≤ 5` **MALFORMED**; only `chi2_dof_max` is a ratified
  NUMBER. The four |z| arms are `RESTATED_NOT_RATIFIED` — they gate, nobody ratified them. The
  two span arms are UNRATIFIED and now **advisory only**, per the PI direction of 2026-08-05.
- 🔴 **The tree scanner had a 116-false-alarm rate on correctly-RETRACTED code**, including
  flagging retraction notes themselves as fabrications. Found only by pointing one stream's
  guard at another stream's current tip. Now **0 false alarms with detection retained**, both
  directions pinned by tests: CLEAN corpus 3 trees / 409 files / 20 claims / **0** violations;
  DIRTY corpus 3 trees / 385 files / 5 claims / **11** violations. The dirty half is the power
  check — a guard switched off would score a perfect false-alarm rate.
- **`reporting.py` was a complete parallel authority module** (own `PI_RATIFIED_ITEMS`, verbatim
  decision text, guard, stamp). Consolidated: statuses now derive from `ratification.py`, with
  `_assert_authority_agrees_with_ratification()` at import — the old guard catches a FABRICATED
  claim, this catches a SILENT DIVERGENCE.
- **`gate_tolerances_ratified` listed ratified COMMITMENTS as gate THRESHOLDS.** Only
  `chi2_dof_max` is a number in `GATE`; `fail_closed_framework` and `matched_configuration_sbc`
  are not. Fixed by intersecting with `GATE`.

### SBC

🔴 **The SBC ranked 11 quantities and declared 9.** The adopted-basis stream extended
`_reported_from_f` with the decision-1 window functionals
(`dndx/omega_report_197_216_allz`); gate's `_reported_names` predicted only
`evidence.reported_quantities`. `configuration_match` reported the mismatch — precisely what
matched-configuration SBC exists to catch. `_reported_names` now derives from
`_reported_from_f`, so declared == ranked by construction.

### Fail-open guards

Closed with mutation proof: the tombstone registry closure (above); the CLEAN record's
`recoverable_from_git`, previously verified by nothing and now checked against git including
that the producing script exists at the stamped commit (host-independent); a
`test_recording_host_still_holds_the_retired_artifacts` guard that fails on the recording host
and skips off it.

Repaired: the prose guard flattened RAW SOURCE and split on sentence terminators, but code has
almost none, so unrelated statements ran together and a correctly `_restated(...)` entry read as
an unqualified RATIFIED claim. It now scans prose only (docstrings, string literals, comments),
each literal its own unit. Mutation-verified in both comment and docstring forms.

Still open, reported not fixed: fabricating three of four endpoint commitments is caught only by
a host-gated `--check`; that detector depends on untracked scratch under one absolute path;
`window_study.phase_selftest` only **warns** where the sibling `adopted_config.pack_stamp_verdict`
**refuses** on stale packs.

---

## 5. Corrections to earlier claims

| claim | correction |
|---|---|
| "every config fails every arm by **10–35×**" | Wrong in **both tails**. Measured over 48 configurations: chi2-arm margin **7.82×–83.34× in-window** (median 38.99), **41.83×–738.75× full-grid** (median 302.14). In-window, **8 of 48 fall below 10×, 16 in [10,35], 24 above 35×**; full-grid, **0 below 35×**. Verdict unchanged (48/48 fail). |
| "fill the `lya_lyb × 0.2-dex` cell BEFORE any integration" | Impossible. The cell needs `basis_width` (adopted-basis branch) and the window axis (spectral-window branch) simultaneously; it became expressible only after both merged. |
| "858 tests passing" | That was a **named 15-module selection**, not the repo. The full suite carries **31 pre-existing failures and 29 collection errors** in both environments (missing `desispec`/`torch`, a network fetch, a 6.6e-12 float non-determinism), unrelated to this work — failure and error name sets byte-identical before and after. |
| the three FP repair options "differ in variance" | Wrong. μ_FP ~ `Gamma(n+½, ℓ/w)` depends only on the ratio `w/ℓ`; measured 14,884±1,592 vs 14,882±1,609. |
| `coarsen_basis` can produce the 0.2-dex production packs | No — its own docstring says SYNTHETIC/TEST; `extract_pack` builds real coarse packs from the catalogue. Re-gridding would be a component substitution. |

---

## 6. Reproducing the principal diagnostics

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# gpdla-hbi = inference (jax); gpdla = extraction / FF-FP / guards (no jax)

# spectral window x basis, both arms   (~4 min extract + ~5 s selftest per basis)
cd /home/mfho/hbi_mcmc_wt
python CDDF_analysis/hbi_mcmc/window_study.py --phase extract \
  --pack-dir /scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/window_study/packs \
  --basis-width 0.2
python -m CDDF_analysis.hbi_mcmc.window_study --phase selftest \
  --pack-dir /scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/window_study/packs \
  --basis-width 0.2

# the ratification tree scan (exit 1 on any fabricated claim)
python CDDF_analysis/hbi_mcmc/ratification.py --check CDDF_analysis docs

# guard layer
cd /home/mfho/desi_gpy_dla_detection
pytest tests/test_tombstones.py tests/test_subdla_forward_headline.py tests/test_unblind_audit.py -q
```

Each artifact carries its exact command in `metadata.rederive`; read that rather than
reconstructing one.

| artifact | stamp | verdict |
|---|---|---|
| `CDDF_analysis/hbi_mcmc/spectral_window_study.json` | `a12385f`, dirty=false | 12 configs, 0 closing |
| `CDDF_analysis/hbi_mcmc/spectral_window_study_bw0p2.json` | `a12385f`, dirty=false | 12 configs, 0 closing |
| `CDDF_analysis/hbi_mcmc/adopted_config_closure.json` | `8963e39`, dirty=false | 48 configs, 0 closing |
| `CDDF_analysis/hbi/tombstones/*.tombstone.json` | `6c9aafc` | four retired identities |

Packs: `/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/window_study/packs`
(0.1-dex, 6 packs) and `..._bw0p2` variants. A separate bit-identity re-extraction at HEAD is
kept at `.../window_study/packs_bw0p1_verify` as evidence that reuse is safe (32 arrays × 6
packs, 0 differ).

---

## 7. Remaining scientific limitations

1. **The forward model does not close**, by roughly an order of magnitude, in every
   configuration tested: 0/48 in the adopted-config cross, 0/24 in the window × basis cross.
2. **The FP normalisation fix is necessary, not sufficient, and not validated** as the physical
   account of the floor deficit (§3).
3. **The loa-0 FP model does not transfer.** Support-matched, it agrees with its own twin to
   **1.007** but over-predicts the held-outs: **0.700** (london0), **0.773** (saclay0). That
   error is |ln| = **0.42 / 0.32** against `t_sigma` = 0.127 / 0.165 / 0.10 — **2.5–4× more
   than the transfer prior can absorb.**
4. **The FP z-shape is imposed, not measured.** `forward.py:452` sets `mu_fp ∝ dX[k,s]`. The
   measured loa-0 z-shape differs by up to **35 %**: model 0.6017/0.2950/0.1033 vs measured
   0.6641/0.2693/0.0665 over z ∈ [2.0,2.5)/[2.5,3.0)/[3.0,3.5). The correct z- and SNR-resolved
   data already exist in `ff_fp_{mock}.json` (`strata`, partitioning exactly) and are discarded
   by `extract_pack.build_fp_block`. Since the z-shape difference is the ONLY handle separating
   sub-floor scatter-in from forest FP, A/B discrimination currently rests on a proxy wrong by a
   third.
5. **Response support**: measured anchors span 19.3360–19.5030 up to 21.0406–21.2164, so the
   21.6 ceiling contains **0.384–0.559 dex of extrapolated response**, and 100 % of the
   sub-floor population's kernel runs on a frozen/extrapolated covariate. `resp_N_fit_range` is
   a range of septile MEANS, not of the data. Nothing below N̂ = 19.5 has ever entered the fit
   (`znz_kernel.py:1675,1730`); the truncation inflates the measured mean up-bias by **3.0× at
   19.05, 2.1× at 19.45, 1.65× at 19.55**.
6. **One frozen 2LPT-0 kernel is shared by all three mocks** — cross-mock spread measures
   TRANSFER, never kernel uncertainty.
7. **Extending below 19.5 forces the completeness "found" definition to move with it.** These
   two cannot move independently without breaking the C·K factorisation.

## 8. Gated — not done, and not to be done without a PI ruling

No push of a new inference PR; no lineage merge; no freeze or tag; no rung 10; no major
campaign; no unblinding; nothing marked `paper_facing`. The ~17,000 CPU-h blue-end campaign is
NOT funded — the standing result is that a large blue-end-induced N_HI shift is unsupported at
current sensitivity, while shifts below ~0.09 dex remain insufficiently constrained.

Adopted working baseline, frozen for controlled comparison: floor 19.7, upper limit 21.6,
0.2-dex latent basis, provisional 19.0/molly172 pad, Lyα-only 1025–1216 Å primary with the
nominal Molly window as reference, dN/dX as the target observable, no Ω_HI headline, no LLS
population in Paper 1, 2LPT-0 as calibration and London-0 / Saclay-0 as transfer tests.
