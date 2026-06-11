# `injection/` — HCD injection-recovery campaign for the GP-DLA selection function

This package builds the **independent selection function** (response matrix `R`,
detection completeness `C_det`, N_HI bias `b_N`, false-positive rate `b_FP`) for
the GP-DLA finder by **injecting High-Column-Density absorbers (DLA / sub-DLA / LLS)
of known `(N_HI, z, SNR)` into real mock spectra, running the unmodified GP, and
comparing recovered vs injected truth.**

> **Why this exists.** The self-calibrated "diagonal O3" CDDF correction
> (`CDDF_analysis/cddf_forward/`) is *algebraically circular* on its own mock:
> `n_corr ≡ n_truth + 1` (the GP measurement cancels). A correction that returns
> the truth it is meant to test cannot validate the CDDF. The only non-circular
> selection function comes from injecting **known, independent** truth — that is
> this campaign. See `docs/` in the private notes repo for the circularity proof.

---

## 1. The injection mechanism (how a mock spectrum gets an absorber)

### Source = real mock spectra, clean sightlines
We inject into the **2LPT mock coadds**
`…/mock-0/loa-124/spectra-16/{hp//100}/{hp}/spectra-16-{hp}.fits` (DESI desispec
format: b/r/z cameras, observed-frame λ, flux, ivar, mask, fibermap). We restrict
to **HCD-free ∩ BAL-free** sightlines (`clean = zcat − hcd_truth_cat − bal_cat`,
≈640k of 1.21M) so the spectrum keeps its **real Lyα forest, continuum, and noise**
but has no pre-existing strong absorber or BAL trough to confuse recovery.

### The absorber = the GP's own Voigt transmission, multiplied into the flux
`gpy_dla_detection/inject_absorber.py::inject_voigt(wavelengths, flux, N_HI, z, num_lines)`:
```
T(λ) = VoigtProfile().compute_voigt_profile(wavelengths, N_HI, z, num_lines)   # = exp(−τ), the compiled C kernel _voigt.so
injected_flux = flux × T(λ)
```
- `VoigtProfile` is the **same `_voigt.so` the GP uses to MODEL a DLA** in
  `dla_gp.py`, so injecting = the GP's own forward model → recovery is a faithful
  test of the *inference*, not of a Voigt-model mismatch.
- `T(λ)` = damped Lyman series at `(1+z)·{1215.67, 1025.72, …}` Å (`num_lines=31`,
  matching the run), black core + broad damping wings scaling with `N_HI`.

**Load-bearing details (all test-pinned):**
- **`N_HI` is LINEAR** (`10**logN`), not log — the C kernel takes linear column density.
- **Pixel alignment**: the C kernel trims 3 pixels each edge (its instrument-LSF
  convolution); `inject_voigt` pads 3 log-spaced pixels each side so the trimmed
  profile realigns pixel-for-pixel — **no off-by-3 N_HI shift**.
- **`ivar`/noise/mask/fibermap untouched** — the absorber multiplies the *signal*
  only, so it sits on the sightline's **native noise**. The campaign is SNR-resolved
  by sightline *selection*, not by adding synthetic noise.
- **All b/r/z cameras are injected** (not just where Lyα lands). This is correct and
  equivalent to injecting into the coadd: the GP re-runs `coadd_cameras` on read
  (`dlasearch.py::process_spectra_group`), and since `T(λ)` is camera-independent at
  a given λ it factors out of the ivar-weighted coadd:
  `coadd(fᵢ·T) = T·coadd(fᵢ)`. We never modify the GP — injection is pure
  input-flux preprocessing UPSTREAM of the GP's own `coadd_cameras`.
  `verify_coadd_consistency(...)` asserts `coadd(injected) == T·coadd(original)` on
  the real data per run; the M4 round-trip test confirms the injected trough matches
  `dla_gp`'s own modeled trough on the resampled brz grid to **<0.5% EW**.

### Two methods (primary + cross-check)
- **(primary) `coadd_injection.inject_into_coadd`** — inject into real clean coadds
  (real forest/continuum/noise). `R` then carries real forest-driven systematics.
- **(cross-check) `gp_dla_draw.draw_gp_dla_spectrum`** — draw a spectrum from the
  null-GP generative model (`μ + low-rank covariance draw + per-pixel noise from a
  real ivar(λ) template`) and multiply by the same Voigt. Isolates pure inference
  self-consistency. **primary − cross-check = the forest contribution** (largest at NHI<19).

---

## 2. The campaign (what gets injected, and what we measure)

**Grid** (`campaign_grid.build_injection_grid`): `(logN_true × z_true × z_QSO_bin ×
SNR_bin)`, **dense in [17.2, 19.0]** (the LLS regime the single-absorber GP is weakest
in), moderate to 20.3, coarse to 22.5. `z_true` is clamped to the GP's *actual* Lyα-only
search window (Lyman-limit floor + 3000 km/s proximity/tail buffers + MAX_LAMBDA,
imported from `set_parameters` — not hardcoded). **SNR bins use the red-side SNR**
(`SNR_REDSIDE`), which is DLA-uncorrelated (forest SNR is anti-correlated with a DLA
since the DLA absorbs forest flux → biases completeness).

**z_QSO stratification (`zqso_bins`, default-on).** The N_HI bias and completeness can
depend on the absorber's **rest-frame position in the forest**, `λ_rest =
(1+z_DLA)/(1+z_QSO)·1215.67` (the GP null-model μ(λ_rest), the Lyman-series mean-flux
suppression, and the forest opacity all evolve across 912→1216 Å). The SNR-only draw
lands on whatever z_QSO is most common (→ a red-skewed λ_rest); stratifying the host
draw by z_QSO injects the SAME `(N, z_DLA)` on a range of hosts, so `λ_rest` is sampled
across the forest and the response matrix R can be **conditioned** on `zqso_bin` instead
of confounding it. The host selection for a subset also spans z_QSO (ranks healpix by
median z_QSO, picks evenly) rather than taking the most-populous. Effect on a 20-healpix
2LPT-0 draw: blue-forest fraction (λ_rest<1080 Å) rises **9% → 38%**, λ_rest reaches the
Lyman limit (917 Å). `--zqso_bins 0` disables. **Control rows** (`build_control_rows`,
`logN_true=NaN`, no injection) measure `b_FP`, labelled by `zqso_bin` so b_FP is
z-resolvable too.

**Campaigns** (all approved; ≤ 4000 SLURM CPU-h total):
- **A** — controlled single-absorber R-build (primary + cross-check) → `R`, `b_FP`.
- **B** — close pairs (Δv, ΔN) → blending/non-linearity systematic.
- **C** — natural-run closure (the existing FILTER-off run, no new compute).
- **D** — **inject a KNOWN non-PW100 truth distribution**, deconvolve with `R`, check
  recovery is unbiased across [17.2, 20.3]. This is the anti-circular gate.

**Measurements** (`measurements.py`, scored against the INJECTED manifest by
`inj_id` — *never* the natural 2LPT truth, so non-circular):
- `detection_completeness` → `C_det(logN_true, z, SNR)` (Beta-binomial CI).
- `nhi_bias` → `⟨logN_rec⟩ − logN_true` + scatter + full per-cell distribution,
  low-`n` flag (Malmquist-aware).
- `response_matrix` → off-diagonal `R[(N_rec,z_rec) | (N_true,z_true,SNR)]` + `b_FP`.

---

## 3. How to reproduce

Environment: the `gpdla` conda env (has `desispec`, `fitsio`, the compiled
`_voigt.so`). Inputs live at
`/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/`
(`zcat.fits`, `hcd_truth_cat.fits`, `bal_cat.fits`, `snr_cat.fits`, `spectra-16/`).

```bash
# 1. Generate the injectable tree (clean-select → grid → inject → write).
#    --n_healpix keeps a pilot fast; drop it for the full campaign.
conda run -n gpdla python injection/gen_injectables.py \
    --out  /scratch/.../pilot_campaign --target_injections 150 --n_controls 50 \
    --n_healpix 6 --snr_cut 2 --num_lines 31

# 2. Run the UNMODIFIED GP on the injectables (single-absorber FILTER-off, matching
#    the run being corrected). Restrict --qsocat to ONLY the injected TARGETIDs.
sbatch injection/run_gp_injection.sh   # -A yueyingn0, pinned BLAS, single_absorber_model=1

# 3. Measure recovered vs injected -> C_det, N_HI bias, R, b_FP + figures.
conda run -n gpdla python injection/measure_recovery.py \
    --campaign /scratch/.../pilot_campaign --processed <gp_out>/processed \
    --figdir <figs>
```

**Tests** (pinned, run in `gpdla`):
`pytest tests/test_inject_absorber.py tests/test_coadd_injection.py
tests/test_gp_dla_draw.py tests/test_campaign_grid.py tests/test_campaign_measurements.py`

## 4. Files
| file | role |
|------|------|
| `../gpy_dla_detection/inject_absorber.py` | low-level Voigt injection (`inject_voigt`) — a forward-model utility, lives with `voigt_fast` |
| `coadd_injection.py` | clean-table, `inject_into_coadd`, `write_campaign`, `verify_coadd_consistency` |
| `gp_dla_draw.py` | GP+DLA generative-draw cross-check |
| `campaign_grid.py` | grid + manifest + clean-sightline sampler + control rows |
| `measurements.py` | `C_det`, `nhi_bias`, `response_matrix` (+ `b_FP`) — scored vs injected truth |
| `gen_injectables.py` | orchestration: build the injectable tree (reproduce step 1) |
| `run_gp_injection.sh` | SLURM: run the unmodified GP on the injectables (step 2) |
| `measure_recovery.py` | orchestration: measurements + figures (step 3) |

---

## 5. Pilot results

Pilot: **145 injections + 48 controls** (50% NHI<19) into 6 HCD-free∩BAL-free
2LPT-0 healpix, run through the unmodified GP at the **byte-exact production
FILTER-off config** (`single_absorber_model=1, max_dlas=1, filter_low_likelihood=0,
τ-EB null, dlambda 0.15, num_forest_lines 31, 100k samples, k=30, release v2.8.5`).
**Measured cost: 80.6 CPU-s / spectrum** (193 spectra, 4.3 CPU-h, 23.8 min wall on
16 cores). Recovery is scored vs the INJECTED manifest by `inj_id` (non-circular),
read from the dlacat (`P_DLA` / `NHI` / `Z_DLA`).

![pilot recovery](../../desi_gpy_dla_notes/notes/figures/injection_pilot/injection_pilot_recovery.png)

What the injections look like (clean mock vs injected, the Voigt family, forest
context, and a Campaign-B close pair):

![injection variants](../../desi_gpy_dla_notes/notes/figures/injection_pilot/injection_variants_nhi.png)
![voigt family](../../desi_gpy_dla_notes/notes/figures/injection_pilot/injection_variants_voigt_family.png)
![close pair](../../desi_gpy_dla_notes/notes/figures/injection_pilot/injection_context_close_pair.png)

| injected logN | n | C_det (p>0.5) | ⟨NHI_rec − NHI_inj⟩ |
|---|---|---|---|
| 17.2–17.8 | 9 ea | 0.89 | **+1.3 to +1.6** |
| 18.0 | 8 | 0.88 | +0.84 |
| 18.4 | 7 | 1.00 | +0.37 |
| 19.0 | 7 | 1.00 | **+0.06** |
| 20.3 | 6 | 1.00 | +0.12 |
| 21.0–22.5 | 6 ea | 1.00 | ≈ ±0.05 |

Controls: **0 / 48 detected** → false-positive deposit `b_FP ≈ 0` at p>0.5.

**Two headline findings (both vindicate the campaign over the diagonal O3):**
1. **Detection completeness is HIGH (≈0.89 in the LLS regime, 1.00 for NHI≥18.2)** —
   the GP *detects* weak absorbers. This is the REAL completeness; the diagonal-O3
   C≈0.1 at NHI<19 was a fine-bin N-posterior-smearing artifact, not incompleteness.
2. **N_HI is over-recovered at low N (+1.3 to +1.6 dex at NHI<17.8), decaying through
   +0.84 (18.0) to ≈0 by NHI≈19** — the prior-pull: a weak Lyα line under-constrains
   N_HI, so the posterior collapses toward the PW14 prior mode. This is the dominant
   **off-diagonal migration** (an injected LLS at 17.4 is recovered near ~19) that the
   diagonal correction structurally cannot capture, and is exactly why the NHI<19 CDDF
   needs the response matrix R, not a diagonal C.

The R-build uses the dlacat MAP recovery + `NHI_ERR` + `P_DLA` (no dense posterior
storage), so the full A+B+C+D campaign (~68k injections ≈ **~1500 CPU-h** at the
measured 80.6 CPU-s/spec) fits comfortably under the 4000 CPU-h cap.

> **Provenance note.** An earlier pilot reported here was run at the WRONG config
> (τ-EB off, dlambda 0.25) on a manifest with a **sightline-reuse bug** that stacked
> up to 7 absorbers on one spectrum — both fixed (global one-injection-per-target;
> production-byte-exact GP config). The numbers above are the corrected re-run.
