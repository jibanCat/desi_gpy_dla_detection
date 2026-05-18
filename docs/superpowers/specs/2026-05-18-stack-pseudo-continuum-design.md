# Design — masked-spline pseudo-continuum for the real-LOA stacks

**Date:** 2026-05-18
**Topic:** post-stack pseudo-continuum fit for `examples/stack_real_loa_dlas.py` (PR #8)
**Status:** draft, awaiting review

---

## 1. Goal & scope

After the stack is built, fit a smooth **pseudo-continuum** to each
composite (metal lines masked), then divide — so coherent metal lines
sit on a flat unit baseline. This replaces the current per-zoom-panel
linear fit (`_local_continuum_norm`) with **one global masked spline**
per composite.

**In scope:** the pseudo-continuum fit + divide; normalized figures; a
fit-quality QC figure.

**Out of scope (later iterations):** equivalent-width measurement and
error bars; a matched non-absorber control composite. The existing
`_local_continuum_norm` is *removed* from the metal-zoom path and the
global pseudo-continuum used instead.

**Decided knobs** (from the brainstorming discussion):
- Spline engine: **fixed-knot `LSQUnivariateSpline` + Schlegel-style
  iterative sigma-rejection**.
- Wavelength coverage: **one spline over 945–1600 Å**; pseudo-continuum
  is undefined (`NaN`) below 945 Å — the Lyman-limit break stays on the
  existing raw, unnormalized `plot_lyman_limit` figure.

---

## 2. Why post-stack, why these choices

- The composite is high-S/N (median of hundreds of spectra); individual
  spectra are too noisy for a continuum fit, so the fit is done once,
  post-stack — the literature standard (York+2006, Mas-Ribas+2017, DESI
  DR2 2025). See `docs/notes/2026-05-18_stacking_continuum_and_lls_literature.md`.
- Fixed knots make the fit deterministic and reproducible (Mas-Ribas);
  the iterative rejection loop (Schlegel `bspline_iterfit` lineage)
  catches weak metal lines not in the mask table and the broad H I
  damping wings, which would otherwise pull the continuum down.
- 945 Å lower bound: blueward of that the Lyman-series lines crowd and
  converge into the 912 Å break — a genuine discontinuity the literature
  says not to spline across.

---

## 3. Algorithm

Operates on one composite curve `C(λ)` (a `BinStack.curve` or
`.curve_bal`, or a combined real/control curve) on the existing log-λ
rest grid, with `counts(λ)` the per-pixel contributing-spectrum count.

### Step 1 — line + coverage mask

`fit_ok(λ)` starts as `isfinite(C) & (counts >= 50) & (λ >= 945)`. Then
each line is masked within `±half_width(λ)`.

**Metal-line half-width — derived, not guessed.** The mask must cover the
*core* of the stacked metal line. A stacked line is a Gaussian broadened
by three terms in quadrature:

| Term | Value | Source |
|---|---|---|
| DESI instrumental LSF | σ_LSF ≈ 30 km/s | FWHM ≈ 70 km/s, the native pixel/LSF scale |
| per-object z_DLA error | σ_z ≈ 50 km/s | catalog `Z_DLA_ERR`: median 6.2e-4 → 47 km/s at z=3 (p75 → 101) |
| intrinsic metal velocity structure | σ_struct ≈ 80 km/s | DLA low-ion Δv ~ tens–150 km/s |

Quadrature sum → **`SIGMA_V ≈ 100 km/s`** (a documented constant). The
stacked metal line is then a Gaussian of width
`σ_stack(λ) = λ·SIGMA_V/c` — 0.40 Å at 1200 Å, 0.52 Å at 1550 Å. The
metal mask half-width is **`K_MASK_SIGMA · σ_stack(λ)`** with
`K_MASK_SIGMA = 3` (covers 99.7 % of a Gaussian core) — wavelength-scaled,
≈1.2 Å at 1200 Å. This is deliberately tighter than the ±10 Å of
York+2006 / DESI DR2 (ref. 2, 7 in the literature note): those are
*redward, well-separated* lines (Mg II, CIV) at lower-resolution BOSS
with larger z errors; our forest line list is densely packed (FeII
1143/1145 are 1.7 Å apart), so a ±10 Å mask would merge every forest
line into one block and leave no clean knot windows. The **iterative
rejection loop (Step 4) is the safety net** for line flux outside the
3σ core — so the static mask only needs to exclude the bulk.

**H I Lyman-series half-widths.** Lyα/Lyβ/Lyγ are *damped* (Lorentzian
wings growing with N_HI). Masking the full DLA-regime wings would erase
the forest metals, so the static H I mask covers core + near-wing only —
**Lyα ±25, Lyβ ±15, Lyγ ±8, Ly4/Ly5 ±5 Å** (≈ a few × the LLS-regime
core) — and the rejection loop removes the extended wing pixels (smooth
negative residuals). *Known limitation:* for the DLA bins the Lyα
damping wing is broad enough that the pseudo-continuum near 1190–1240 Å
is approximate; the metals there (SiII 1190/93, NI 1200, SiIII 1207) are
best trusted in the LLS / sub-DLA bins, which is the science focus
anyway. Flagged on the QC figure.

**Cross-check.** The QC step fits a Gaussian to the strong, isolated
CIV 1548 line in the composite and reports its measured σ; if it
disagrees with `σ_stack(1548)` by more than ~50 %, revise `SIGMA_V`.

### Step 2 — knot placement

Candidate interior knots are placed every `KNOT_SPACING ≈ 15 Å` across
945–1600 Å. The log-λ grid is ~0.3 Å/pixel, so this is ~one knot per
~50 pixels — comparable to Mas-Ribas (denser knots risk threading a knot
into an unmasked feature; sparser knots underfit the forest decrement).

Drop any candidate knot that has **no `fit_ok` pixel within
`±KNOT_SPACING`** — i.e. knots are kept only where there is clean data
to constrain them; this threads knots through the windows *between* the
metal lines (incl. SiII 1190/1193, NI 1200, SiIII 1207). Interior knots
must lie strictly inside the `fit_ok` data range (`LSQUnivariateSpline`
requirement). If fewer than ~4 knots survive, fall back to a low-order
polynomial and emit a warning (degenerate composite).

### Step 3 — initial spline fit

`spl = LSQUnivariateSpline(λ[fit_ok], C[fit_ok], t=interior_knots, k=3,
w=sqrt(counts[fit_ok]))`

Cubic (`k=3`); weight by `sqrt(counts)` so sparsely-covered pixels
(grid edges) carry less leverage.

### Step 4 — iterative sigma-rejection (Schlegel-style)

Across **all** iterations these are **fixed and never change**: the knot
vector `t`, the spline order `k = 3`, and the weight definition
`w = √counts`. **Nothing is tuned or optimized.** The *only* thing that
changes between iterations is the **set of pixels passed to the fit** —
it shrinks as outliers are removed.

Let `fit_ok` be the boolean pixel set from Steps 1–2. For `j = 1, 2, …`:

1. **Fit** `spl_j = LSQUnivariateSpline(λ[fit_ok], C[fit_ok], t, k=3,
   w=√counts[fit_ok])`.
2. **Residuals** on the currently-included pixels:
   `r = C[fit_ok] − spl_j(λ[fit_ok])`.
3. **Robust scatter** `σ = 1.4826 · MAD(r)`.
4. **New outliers** = included pixels with `|r| > REJECT_SIGMA·σ`
   (`REJECT_SIGMA = 5`). Unmasked weak absorption and H I damping-wing
   pixels show up as negative `r`; both signs are rejected.
5. If there are **no new outliers** → stop, `P = spl_j`. Otherwise
   remove the new outliers from `fit_ok` and **return to step 1** — a
   genuine re-fit: the identical `LSQUnivariateSpline` call with the
   *same* knots/order/weight definition, only the (now smaller) `x,y,w`
   arrays differ.
6. Stop unconditionally after `MAX_REJECT_ITER = 10` iterations.

So "re-fit" means: re-run the same spline solve on a smaller pixel set.
No parameter is searched between iterations.

**Knot bookkeeping** — the one way the knot set can change: if rejection
empties a knot interval (`LSQUnivariateSpline` requires ≥1 data point
between consecutive interior knots), that knot is dropped before the
next fit. This is forced by the data, not tuned.

### Step 5 — evaluate & normalize

`P(λ) = spl(λ)` for `945 ≤ λ ≤ 1600`, `NaN` elsewhere.
Normalized composite `C_norm(λ) = C(λ) / P(λ)`.

### Step 6 — QC

Record, per composite: number of knots, number of rejection iterations,
number of rejected pixels, and the **RMS of `C_norm − 1` over the
line-masked clean pixels** (should be ≈ the composite's per-pixel noise;
a large value flags a bad fit). Warn if `P` is non-monotone in a way
that implies ringing (any interior local extremum deeper than a
threshold).

---

## 4. Parameters (named constants, all tunable)

| Constant | Value | Meaning |
|---|---|---|
| `PCONT_LAMBDA_MIN` | 945.0 Å | blue end of the fit |
| `SIGMA_V` | 100 km/s | stacked-line width budget (LSF ⊕ z-err ⊕ velocity structure) |
| `K_MASK_SIGMA` | 3 | metal mask half-width = `K_MASK_SIGMA·σ_stack(λ)` |
| `HI_MASK_HALF` | {Lyα:25, Lyβ:15, Lyγ:8, Ly4:5, Ly5:5} Å | H I core+near-wing masks |
| `KNOT_SPACING` | 15.0 Å | interior knot spacing |
| `SPLINE_ORDER` | 3 | cubic |
| `REJECT_SIGMA` | 5.0 | rejection threshold |
| `MAX_REJECT_ITER` | 10 | rejection iteration cap |

---

## 5. Code integration

Per the repo's "extend existing files" style — all in
`examples/stack_real_loa_dlas.py`, no new module:

- `_continuum_mask(rest_grid, curve, counts)` → boolean `fit_ok`
  (Step 1).
- `fit_pseudo_continuum(rest_grid, curve, counts)` → `P` array
  (Steps 2–5); pure, deterministic, ~milliseconds.
- The pseudo-continuum is computed **at plot time** from the cached
  composite curves — it is *not* stored in the npz (cheap to recompute,
  keeps the npz and its provenance unchanged; `--plot-only` just works).
- `plot_metal_zoom` and `plot_control`: replace the per-panel
  `_local_continuum_norm(x, y, lines)` call with a slice of the global
  `C_norm`. `_local_continuum_norm` is deleted.
- `plot_overview`, `plot_lyman_limit`: unchanged — stay raw (see §7).

## 6. New / changed figures

- **`stack_pseudo_continuum_qc.png`** (new) — one panel per production
  NHI bin: the raw composite `C`, the fitted `P` overlaid, knot
  positions ticked, masked regions shaded. The eyeball check that the
  fit is sane.
- **`stack_metal_zoom_*.png`** — unchanged layout, but panels now show
  `C / P` (global pseudo-continuum) instead of the per-panel linear
  normalization.
- **`stack_control_*.png`** — same: real and control curves divided by
  their own pseudo-continua.
- `plot_lyman_limit` / `plot_overview` raw flux figures — unchanged
  (the LL break must stay unnormalized).

## 7. Resolved decisions & flagged spots

- **`plot_overview` stays raw** (decided) — it shows the forest
  decrement and the broad H I lines; no normalized variant added.
- **Knot spacing 15 Å** (decided as the starting value) — deliberately
  coarser than Mas-Ribas's ~4.5 Å because our forest line list is dense;
  if the QC figure shows the forest decrement underfit, drop to ~10 Å.
- **Lyα-vicinity crowding** (flagged, no decision needed) — SiIII 1207
  is only ~9 Å from Lyα; its ~±1.3 Å metal mask and the ±25 Å Lyα mask
  leave a thin clean sliver. Most likely spot to need a hand-tuned
  knot/window after the QC figure is inspected.

## 8. Validation — mock-injection test FIRST (TDD)

**Implementation step 1 is the mock test, before `fit_pseudo_continuum`
exists.** Build a synthetic composite with a *known* truth, write the
assertions, then implement the fitter until they pass.

### 8.1 Mock composite generator (`make_mock_composite()`, test helper)

On the real log-λ rest grid (700–1600 Å):

1. **Truth pseudo-continuum `P_true(λ)`** — analytic, smooth, known:
   a gentle redward slope/curvature × a broad smeared "QSO Lyα emission
   bump" (wide Gaussian ~1280 Å) × a smooth forest decrement (a sigmoid
   roll-off blueward of 1216 Å down to ~0.6). No sharp features.
2. **Injected absorption lines** — Gaussians at known rest wavelengths
   (≈8–10 drawn from `METAL_LINES`, spanning forest + redward: CIV,
   SiIV, SiII, FeII, OVI, …), each with a known central depth (0.05–0.4)
   and width `σ_stack(λ)`. Mock flux `C = P_true · (1 − Σ Gaussians)`.
3. **Two extra *weak* lines NOT in the mask table** — to exercise the
   rejection loop.
4. **Counts** array — realistic (~300–800), lower at the blue edge; a
   few pixels forced to 0 / NaN to exercise coverage masking.
5. **Noise** — per-pixel Gaussian, σ_pix known (~0.01–0.03)·/√counts.

Seeded RNG → deterministic.

### 8.2 Assertions (`tests/test_stack_pseudo_continuum.py`)

1. `P_fit` is `NaN` below 945 Å, finite over 945–1600 Å.
2. On line-free, well-covered pixels: `|P_fit/P_true − 1| < 2 %`.
3. `C/P_fit ≈ 1` on line-free pixels — RMS within ~1.5× input noise.
4. Each injected line survives: depth in `C/P_fit` recovers the input
   depth within ~15–20 % (the continuum did **not** eat the line).
5. The spline did **not** dip into lines: at each injected line centre
   `|P_fit/P_true − 1|` stays within tol (spline didn't follow the
   absorption).
6. Rejection loop works: the two weak *unmasked* lines do not pull
   `P_fit` down — assertion 2 still holds at their wavelengths.
7. Null case: no injected lines → `C/P_fit ≈ 1` everywhere within noise.
8. Coverage: `counts<50` / NaN pixels are excluded; no crash; degenerate
   all-NaN input returns all-NaN cleanly.
9. Determinism: identical input → identical output.

### 8.3 Visual

`stack_pseudo_continuum_qc.png` eyeball check on the *real* composite
once the SLURM stack run (job 50407357) finishes.

## 9. Minor cleanup (for testability)

`examples/stack_real_loa_dlas.py` currently runs `OUT_DIR.mkdir()` at
module import. Move it into a small `_ensure_outdir()` called from
`main()` / `compute_stacks` / `dump_zhist`, so `tests/` can import the
pure functions (`fit_pseudo_continuum`, `_continuum_mask`,
`make_mock_composite` lives in the test) without a filesystem side
effect. Precedent: `tests/test_voigt_sweep_targets.py` already imports
an `examples/` script.
