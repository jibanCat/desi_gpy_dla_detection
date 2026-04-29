# 2026-04-29 — current state of the Voigt-LSF / Bayesian-correctness work

> Read this first. Companion docs in this directory cover individual experiments
> (`findings.md`, `report.md`); this is the executive summary as of end-of-day.

## Headline

The +0.34 dex N_HI bias on TID 120046865 is **not** an LSF problem (in
production, on DESI data). It is a **τ_eff prior mismatch**, fully
resolved on the strong-DLA target by an HCD-masked empirical-Bayes
τ_eff fit per spectrum.

**Multi-target validation (n=6)**: HCD-masked τ-EB closes the DLA-regime
bias to ~0 dex on all 4 DLA-regime targets tested (median bias +0.12 →
−0.02 dex). It does NOT close sub-DLA / LLS regime biases — but those
turn out to be from the DLA prior boundary (`uniform_min_log_nhi=20.0`
forces sub-DLA truths to map to ≥20.0), not τ_eff. Different problem,
fixed by the sub-DLA model (items #6, #7, #11) and FILTER fix #5.

A separate kernel-truncation defensive fix in `voigt_v2.py` does *not*
shift production results (DESI 0.8 Å grid, half_width=3 was already
sufficient there) but does shift sub-DLA / LLS regime results — see
sweep cross-comparison below.

---

## What the four committed tests on this branch establish

| Hypothesis | Test | Verdict | Commit |
|---|---|---|---|
| (1) LSF mismatch (BOSS-on-DESI) | Trace dλ in production: convolution actually applied at DESI 0.8 Å (not GP rest 0.15 Å). Production kernel σ_eff = 0.49 Å → R_eff ≈ 3400. | **Ruled out** for production DESI data. Earlier "R≈21000" claim was a confusion of grids. | `e1cc94f` |
| (2) Kernel half_width truncation | DESI-R3000 was truncated to 7-pixel kernel. Auto-size to ⌈4σ⌉. | Defensive fix (no production effect on 0.8 Å grid) but **shifts sub-DLA / LLS configs** — see sweep below. | `eda1930` |
| (3) QMC sampler density at high NHI | Brute-force scan over 20k samples; truth NHI=21.263 sample exists at Δ=0.0000. | **Ruled out** for this target's DLA component. | `e1cc94f` |
| (4) τ_eff scaling | Fine-NHI grid scan, τ_factor 0.25× → 3.0× production. | **Confirmed lever**. Production τ=1.0× gives MAP=21.60 (+0.34 dex bias). τ=1.5× gives MAP=21.50 (+0.24 dex). τ=3.0× gives MAP=21.225 ≈ truth. | `de509f2` |
| (5) τ_eff EB / marginalization | Compare production / EB / full-marg on K=6 grid. | **EB ≈ full-marg = 21.50 (+0.24 dex)**. Both close ~30% of the bias. Full marg adds no benefit over EB here. | `caee3ed` |
| (6) HCD-pixel masking before τ-EB | Mask pixels with residual < −1.5σ from null model, refit τ-EB on the remaining forest pixels. | **τ_best shifts 1.5 → 2.0; MAP NHI = 21.20, bias = −0.063 dex.** Closes 100% of the +0.34 dex bias on this n=1 target. | `ca9dc8c` |

The story: τ_eff (which acts on μ, M, AND Ω jointly via the A_lya
diagonal) is the major lever for high-NHI targets. The naive EB fit
is biased because it includes HCD pixels in the τ fit — they look like
extra forest absorption and hold τ down. The Becker / Faucher-Giguère
mean-flux convention (mask HCD pixels first, then fit τ) is the right
recipe.

## Voigt-LSF sweep — original buggy kernel vs fixed kernel

Both runs: 18 targets × 4 configs (A=BOSS-log-r2000, B=DESI-linear-r3000,
C=DESI+6 lines, D=no LSF) on 2lpt + saclay + london mocks. Master CSVs:

- Buggy: `/tmp/voigt_sweep_local/runs/master.csv` (same as
  `/nfs/turbo/.../voigt_sweep_48947439/runs/master.csv`).
- Fixed: `/nfs/turbo/.../voigt_sweep_fixed_kernel/runs/master.csv`.

**Numerical comparison** (8 rows have |Δ MAP| > 0.001 dex):

| target | regime | mock | finding |
|---|---|---|---|
| 2385001246 | sub-DLA | saclay | config-spread 0.029 → 0.384 dex (+1220%). Config D (no LSF) MAP shifts +0.36 dex. |
| 260170003 | LLS | 2lpt | config-spread 0.077 → 0.112 dex (+45%). |
| (others) | DLA | * | unchanged at <0.001 dex (saturated core wider than any kernel) |

DLA-regime nullity is robust (kernel doesn't matter; saturated core
dominates). **Sub-DLA / LLS regime is genuinely kernel-sensitive once
the kernel isn't artificially narrow** — earlier "kernel doesn't matter"
claim was over-broad.

Wall time: fixed kernel +14% (35-pixel DESI kernel vs 7-pixel BOSS).
Acceptable.

## What does and doesn't reproduce the historical bias

The historical +0.37 dex bias was on TID 120046865 with v1 production
(BOSS-log-R2000 kernel, Turner+2024 τ_0). Reproductions on this branch:

| Test | Result |
|---|---|
| v2 boss-log-r2000 brute-force MAP | logNHI 21.40–21.55 across sample subsets, +0.14 to +0.28 dex bias |
| Same target at production τ_0 (1.0×) on fine NHI grid | MAP=21.60, **+0.337 dex** ← matches historical |
| Inference (with FILTER) | NaN MAP, p_DLA=0.05 — FILTER initial-scan rejects the prior |

The FILTER step is broken on this target too — it returns p_DLA = 0.05
despite the data clearly containing a DLA (Δ(MAP − null) = +21.7 in log L).
This is the FILTER fix #5 issue (already on the task list).

So three independent things must be fixed before this target's bias goes
away in production:
1. **τ_eff treatment** — best fixed by per-spectrum HCD-masked EB fit.
2. **FILTER** — initial-scan is rejecting valid DLA candidates.
3. **Residual ~0.06–0.24 dex** — what's left after τ-EB. Likely μ shape
   in the wing or non-Gaussian residuals (user's untested hypothesis).

## Production / training pipeline state

**Preloads (4 ALL done, ready for training)**:

| dir | spectra | file | source | wall |
|---|---:|---:|---|---:|
| `production_preload_runs/loa_no_dla_no_bal_52198069/trainset.h5` | 300,008 | 5.44 GB | NERSC LOA | 4.4h |
| `production_preload_runs/loa_no_hcd_with_bal_52198070/trainset.h5` | 300,032 | 5.26 GB | NERSC LOA | 4.4h |
| `pscratch/.../v2_runs/2lpt_loa0_48938765/trainset.h5` | 299,811 | 5.8 GB | GreatLakes 2LPT mock | (see log) |
| `pscratch/.../v2_runs/2lpt_loa124_nohcd_nobal_48938766/trainset.h5` | (similar) | (similar) | GreatLakes 2LPT mock | (see log) |

All four are float32 with HDF5 compression (so 5–6 GB sizes are correct
for ~300k spectra × 3801 pixels). Schema verified: `fluxes`,
`noise_variance`, `rest_wavelengths`, `tids`, `zqso`, `bluesnr`, `redsnr`.

**Training runs (in NERSC `logs/v2_runs/`)**:
- `2lpt_loa0_52188320` and `2lpt_loa124_nohcd_nobal_52188321` — separate
  training jobs that consumed the GreatLakes preloads. Status: see those
  dirs / logs (need to confirm completion + epoch count separately).
- Several `e2e_loa_dbg_*` debug logs — likely failed early or cancelled.

## What's queued for the next session

- (a) ~~Saclay sub-DLA target deep-dive~~ **DONE**: confirmed the +0.36
  dex spread between configs in the sweep is multi-DLA solver noise,
  not real LSF sensitivity. All 3 kernels brute-force to MAP ≈ 21.22
  on a sub-DLA truth of 19.56 — that's a +1.7 dex bias from the
  DLA-prior boundary (`uniform_min_log_nhi=20.0`), not the kernel.
- (b) ~~Multi-target HCD-masking τ-EB~~ **DONE on n=6**: see results
  table above. HCD-masked EB closes DLA-regime bias to ~0 dex (median
  +0.12 → −0.02 dex over 4 DLA targets). Sub-DLA / LLS unchanged —
  that bias is prior-boundary, not τ_eff.
- (c) **Train GP on the no-DLA / no-BAL preload** — preload step done;
  training step pending. This is what would actually validate Step 4
  (does retraining on truly-clean forest data move the inference bias).

## Files of record

- `gpy_dla_detection/voigt_v2.py` — selectable-kernel Voigt + auto-width fix
- `examples/check_truth_vs_map_likelihood.py` — log L(truth) vs log L(MAP) brute-force
- `examples/check_tau_eff_sensitivity.py` — initial τ_eff sweep (discrete QMC, was misleading)
- `examples/check_tau_eff_fine_grid.py` — fine NHI-grid τ_eff scan (the right test)
- `examples/check_tau_eff_marginalization.py` — production / EB / marg comparison
- `examples/check_tau_eb_robust_mask.py` — HCD-masking before τ-EB
- `docs/voigt_demo/voigt_kernel_demo_dl08.png` — kernel demo at production grid
- `docs/notes/2026-04-29_voigt_lsf_sweep/findings.md` — full prior write-up with retractions

## Branch state

- Branch: `claude/voigt-lsf-fix`
- Last commit: `ca9dc8c`
- 7 commits since the original LSF sweep landed; all results corrected
  in-place via retractions in `findings.md`. No git history rewriting.
