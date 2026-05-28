# Stacking literature — continuum normalization & LLS Lyman-limit recovery

Reference notes backing the 2026-05-18 changes to
`examples/stack_real_loa_dlas.py` (PR #8). Two literature surveys: how
astronomers normalize the continuum for QSO absorption-line composites,
and how they stack LLS sightlines and verify the Lyman-limit break.

---

## 1. Continuum normalization for stacked QSO absorption spectra

### Standard practice — two stages

1. **Per-spectrum *coarse flux normalization* BEFORE stacking.** Each
   spectrum is de-redshifted to the absorber rest frame and divided by a
   flux scalar measured in one or two emission-free rest-frame windows.
   This only equalizes scale — it is *not* a continuum-shape fit. Nobody
   does a full per-spectrum continuum fit for survey-quality (BOSS/eBOSS/
   DESI) data; individual spectra are too noisy.
2. **Continuum / pseudo-continuum fit AFTER stacking**, on the high-S/N
   composite, with metal-line regions masked. This is universal for
   DLA/LLS metal-line EW work.

The dominant post-stack method is a **masked cubic / B-spline** fit:
mask the metal lines (±10 Å, ±40 Å for Lyα), fit a smooth spline to the
masked composite, divide. "Divide the composite by a smoothed version of
itself" is legitimate and is essentially the DESI-DR2 standard — but
implemented as a *masked spline*, not a blind boxcar (an unmasked median
filter near strong/blended lines biases the continuum down and partially
subtracts line flux). Use median (not mean) combination if EWs are the
goal — the mean is biased by the few high-transmission sightlines.

### Key papers

- **Vanden Berk et al. 2001** (astro-ph/0105231) — SDSS composite QSO
  spectra; normalize per-spectrum by mean flux in an emission-free
  window, combine by median / geometric mean / arithmetic mean. Median
  preserves relative line EWs; geometric mean preserves the power-law
  continuum.
- **York et al. 2006** (A&A 454, 151) — SDSS Mg II / DLA absorber
  composites. Per-spectrum normalization by median flux in two windows
  flanking the feature; continuum isolated by dividing the absorber
  composite by a matched non-absorber control composite, masking lines,
  and fitting a **cubic B-spline (smoothness ≈ 1.1)** tuned to y = 1 in
  masked regions.
- **Pieri et al. 2010 / 2014** (1001.5282, 1309.6768) — composite of
  strong Lyα-forest absorbers; **stack first**, fit a **spline
  "pseudo-continuum"** to the stack (this paper coined "pseudo-continuum"
  for composites), metal lines emerge as residuals. Pieri's variant adds
  the pseudo-continuum decrement back additively instead of dividing.
- **Mas-Ribas et al. 2017** (1610.02711) — mean DLA metal-line spectrum,
  BOSS, ~27k DLAs. Per-spectrum normalization in **1300–1383 Å and
  1408–1500 Å**; composite continuum = **cubic spline, one knot per
  ~4.5 Å**, linear interpolation in the Lyα region; iterative correction
  for the DLA's effect on the underlying QSO continuum.
- **Mas-Ribas et al. 2018** (1801.02605) — documents that stacked-spectrum
  EWs are sensitive to the continuum recipe at the ~20 % level. Caution.
- **DESI DR2 composite of QSO absorption-line systems** (arXiv:2512.02992,
  2025) — modern DESI reference; carries the York-2006 recipe forward
  (control-composite division + masked cubic B-spline, smoothness ≈ 1.1).

### Bearing on `stack_real_loa_dlas.py`

The script's **per-spectrum** normalization (divide by the median in the
flat redward [1410, 1520] Å window) is exactly the correct *coarse*
pre-stack step — it matches Mas-Ribas+2017's approach. The zoom panels'
`_local_continuum_norm` (linear fit, line cores masked) is a light
post-stack continuum already. The proper upgrade, if population-mean EWs
are ever needed, is a **masked cubic B-spline pseudo-continuum on the
composite** (York 2006 / DESI DR2 recipe) — deferred follow-up, not done
in this PR.

---

## 2. LLS stacking & Lyman-limit break recovery

### Is the Lyman-limit break a legitimate LLS confirmation?

**Yes.** A composite of true LLS sightlines de-redshifted to the absorber
frame shows a coherent flux decrement turning on at the **911.76 Å** rest
Lyman limit; a stack of false positives shows no edge. This is exactly
how Fumagalli+2013 separates τ ≥ 2 from τ < 2 systems and how Pieri+2014
cross-checks forest absorbers.

Population-mean N_HI from the break is also literature-backed but needs
modeling: τ_LL = N_HI · σ_912 with σ_912 ≈ 6.30 × 10⁻¹⁸ cm² (τ_LL = 1 at
log N_HI ≈ 17.2; τ_LL = 2 at ≈ 17.5; essentially black for log N_HI
≳ 18). So the break depth is a useful **coarse N_HI estimator only in the
17.2–18 regime** and a "yes, optically thick" indicator above that. The
opacity deepens blueward (σ ∝ λ³), giving extra leverage down to ~800 Å.

### Recipe

De-redshift to absorber frame; coarse per-sightline flux normalization in
a clean window; **median** combine; **equal weight per sightline** (S/N
weighting biases toward weak-absorption sightlines — Worseck+2014 states
this explicitly). Model the decrement with two terms: exp(−τ_eff,Lyman)
for the Lyman-series lines above 912 Å × exp(−τ_LL) for the Lyman
continuum below.

### Key papers

- **Prochaska, Worseck & O'Meara 2009** (0910.0009) — first stacked-QSO
  measurement of IGM H I-ionizing opacity; 1800 SDSS QSOs, normalized at
  1450 Å, fit window from 905 Å.
- **Worseck et al. 2014** (1402.4154) — definitive mean-free-path
  methodology; stacks down to **~850 Å rest**, equal-weight, two-term
  opacity model, bootstrap σ_z for redshift-spread smearing.
- **Fumagalli, O'Meara & Prochaska 2013** (ApJ 775, 78) — most directly
  relevant: composite of ~20–38 LLSs with log N_HI ~17.5–19 in the
  absorber frame, **extends to ~800 Å rest**, transmitted flux blueward
  of 911.76 Å used to separate τ ≥ 2 from τ < 2.
- **Pieri et al. 2014** (1309.6768) — closest methodological template:
  composite of 242k strong Lyα-forest absorbers from BOSS, verified real
  via coherent metal lines + Lyman-series consistency; later version
  added sections on LLS contamination of the sample.
- **O'Meara et al. 2013**, **Prochaska, O'Meara & Worseck 2010** — LLS
  survey statistics and ℓ(z).
- **Lehner / Wotta et al. (COS CGM Compendium)** — define the pLLS
  (16.2 ≤ log N_HI < 17.2) and LLS (17.2–19) column-density bins.
- **Becker et al. 2021 / Zhu et al. 2023 / DESI Y1 MFP (2411.15838)** —
  modern Lyman-continuum-transmission fitting in stacked QSO spectra.

No dedicated DESI/eBOSS *intervening-LLS* composite paper exists for
2021–2025 — an absorber-frame LLS composite in DESI would be relatively
novel; Pieri+2014 (BOSS) is the closest precedent.

### Pitfalls

- **Redshift-spread smearing** — a spread in z_abs (and per-object z_abs
  error) smears the 911.76 Å edge into a gradual rolloff.
- **N_HI-spread smearing** — a mix of log N_HI 17.2–19 also gives a
  gradual decrement, not a step.
- **Lyman-series blanketing** — the 912–~945 Å region is depressed by
  converging Lyman-series lines; do **not** read the pre-break continuum
  level there — use ≳ 1000 Å.
- **Forest mean-flux decrement** — everything blueward of Lyα rest is
  depressed by the foreground forest, redshift-dependent; model/divide it
  out or stack in narrow z bins.
- **Foreground LLS** along the same sightline add a smooth non-coherent
  floor.
- Avoid the QSO proximity zone (Δz ≈ 0.4 of z_QSO).

### Reference list

Continuum / composite-spectrum methodology and absorber-stacking papers
(full citations, for the masked-spline pseudo-continuum work):

1. Vanden Berk, D. E., et al. 2001, "Composite Quasar Spectra from the
   SDSS," AJ 122, 549. https://arxiv.org/abs/astro-ph/0105231
2. York, D. G., Khare, P., Vanden Berk, D., et al. 2006, "Average
   Extinction Curves and Relative Abundances for QSO Absorption Line
   Systems at 1 ≤ z_abs < 2," MNRAS 367, 945.
   https://arxiv.org/abs/astro-ph/0601279
3. Pieri, M. M., Frank, S., Weinberg, D. H., Mathur, S., & York, D. G.
   2010, "The Composite Spectrum of Strong Lyα Forest Absorbers,"
   ApJL 724, L69. https://arxiv.org/abs/1001.5282
4. Pieri, M. M., Mortonson, M. J., Frank, S., et al. 2014, "Probing the
   CGM at High Redshift Using Composite BOSS Spectra of Strong Lyα
   Forest Absorbers," MNRAS 441, 1718. https://arxiv.org/abs/1309.6768
5. Mas-Ribas, L., Miralda-Escudé, J., Pérez-Ràfols, I., et al. 2017,
   "The Mean Metal-line Absorption Spectrum of DLAs in BOSS," ApJ 846, 4.
   https://arxiv.org/abs/1610.02711
6. Mas-Ribas, L., et al. 2018, "Metal-line absorption strength at low
   S/N," arXiv:1801.02605. https://arxiv.org/abs/1801.02605
7. Napolitano, L., et al. (DESI Collaboration) 2025, "The Composite
   Spectrum of QSO Absorption Line Systems in DESI DR2,"
   arXiv:2512.02992. https://arxiv.org/abs/2512.02992
8. Noterdaeme, P., Petitjean, P., Carithers, W. C., et al. 2012,
   "Column Density Distribution and Cosmological Mass Density of Neutral
   Gas: SDSS-III DR9," A&A 547, L1. https://arxiv.org/abs/1210.1213
9. Lee, K.-G., Suzuki, N., & Spergel, D. N. 2012, "Mean-flux-regulated
   PCA Continuum Fitting of SDSS Lyα Forest Spectra," AJ 143, 51.
   https://arxiv.org/abs/1108.6080
10. Prochaska, J. X., Worseck, G., & O'Meara, J. M. 2009, "A Direct
    Measurement of the IGM Opacity to H I Ionizing Photons," ApJL 705,
    L113. https://arxiv.org/abs/0910.0009
11. Prochaska, J. X., O'Meara, J. M., & Worseck, G. 2010, "A Definitive
    Survey for Lyman Limit Systems at z ~ 3.5 with SDSS," ApJ 718, 392.
12. O'Meara, J. M., et al. 2013, "The HST/ACS+WFC3 Survey for Lyman
    Limit Systems II," ApJ 765, 137.
13. Fumagalli, M., O'Meara, J. M., & Prochaska, J. X. 2013, "Dissecting
    Optically Thick Hydrogen at z ~ 3," ApJ 775, 78.
14. Worseck, G., et al. 2014, "The Giant Gemini GMOS Survey: the Mean
    Free Path of H I-Ionizing Photons," MNRAS 445, 1745.
    https://arxiv.org/abs/1402.4154
15. Zhu, Y., et al. 2023, "Direct Measurements of the Mean Free Path of
    Ionizing Photons over 5 < z < 6," ApJ 955, 115.
16. SDSS `idlutils` — Schlegel, D. J. & Burles, S., `bspline_iterfit`
    (B-spline with iterative rejection). https://github.com/sdss/idlutils
17. Weaver, B. A., et al., `pydl` — Python port of SDSS IDL utilities,
    incl. `pydl.pydlutils.bspline`. https://pydl.readthedocs.io
18. Virtanen, P., et al. 2020, "SciPy 1.0," Nature Methods 17, 261.
19. Astropy Collaboration 2022, "The Astropy Project … v5.0," ApJ 935,
    167. https://arxiv.org/abs/2206.14220

### Recommended rest floor — and the DESI coverage caveat

Literature floor: **~700 Å** (800 Å minimum). The script's floor was
moved 900 → **700 Å**. **Caveat specific to this sample:** at the
absorber redshifts here (median z_DLA ≈ 3.0, with a high-z tail), the
700–900 Å rest region maps to observed λ below the DESI ~3600 Å blue
cutoff for all but the highest-z absorbers — so the deep-blue pixels are
sparsely covered and the < 50-spectra cut NaN-clips them. The break is
best resolved in the **~850–960 Å** region, populated by the z_abs ≳ 3
tail. The lowest-N_HI LLS bin [17.2, 18.0) happens to sit at median
z_DLA ≈ 3.8, where 912 Å lands at ~4400 Å observed — comfortably inside
DESI coverage — so the break *is* recoverable for that bin despite its
small count (n ≈ 41).
