# QSO emission/absorption correlations: what's in real DESI spectra, what's in 2LPT mocks

**Summary.** Real quasar spectra carry a rich web of intra-population correlations
(Baldwin effect, Boroson-Green Eigenvector 1 / Sulentic quasar main sequence,
CIV blueshift trends, NV/Lyα and metallicity sequences, BAL imprints,
proximity effect, intervening DLA/LLS metal forests) that couple emission
strength, line profile, continuum slope, and absorption substructure across the
rest-frame UV. DESI synthetic mocks built from LyaColore + quickquasars inject
a single (or near-single) continuum template, a Lyα/HCD transmission field, and
randomly drawn DLA/BAL/metal contaminants — by construction they cannot
reproduce population-level cross-correlations between emission lines, between
emission and absorption, or between continuum shape and forest fluctuations.

---

## 1. Emission-line correlations across rest-frame UV

### 1a. Baldwin effect (continuum-luminosity ↔ line EW)

Baldwin (1977) observed an anti-correlation between CIV λ1549 rest-frame
equivalent width and continuum UV luminosity. The effect is now established
for many UV/optical broad lines, with line-specific slopes; SDSS samples of
~26k 1.5 < z < 5.1 QSOs confirm the global trend and show that SMBH mass
and L/L_Edd (Eddington ratio) are the better underlying drivers than L
alone (Xu et al. 2008, arXiv:0806.1787; Baskin & Laor 2004,
arXiv:astro-ph/0403365; Shemmer & Lieber 2015, arXiv:1503.07547;
Ge et al. 2016, arXiv:1608.02172).

**Why it matters for the GP kernel:** for fixed (rescaled) continuum, the
EW of CIV (and to a lesser extent Lyα, NV, SiIV, CIII], MgII) is
luminosity-dependent. A continuum-normalized GP trained on a real sample
sees this as a residual correlation between the line core pixels and the
inferred continuum amplitude.

### 1b. Boroson-Green Eigenvector 1 / Sulentic quasar main sequence

Boroson & Green (1992) applied PCA to ~80 PG quasars and discovered that a
single mode (EV1) — an anti-correlation between optical FeII strength
(R_FeII) and the peak of [OIII] λ5007, coupled to Hβ FWHM — explained
~29 % of the population variance. Sulentic and Marziani then extended this
into the "quasar main sequence" (Sulentic et al. 2000, 2007;
Marziani et al. 2018, arXiv:1802.05575;
Marziani et al. 2024 review, arXiv:2410.09262). The physical driver is
predominantly the Eddington ratio with secondary contributions from black
hole mass, orientation, and intrinsic metallicity (Shen & Ho 2014, Nature
513, 210; Sun & Shen 2015 / Marziani et al.).

**UV manifestations relevant to Lyα-forest QSO continua:**
- Population A (high L/L_Edd) — strong FeII, weak narrow [OIII], blueshifted CIV with prominent broad blue wings.
- Population B (low L/L_Edd) — symmetric or red-asymmetric CIV, weaker FeII, stronger narrow lines.

### 1c. CIV blueshift and BLR kinematics

CIV centroids in luminous QSOs are blueshifted by up to several thousand
km/s relative to low-ionization lines (e.g. MgII, Hβ), and the blueshift
correlates with line width and EW (Richards et al. 2011;
Sun et al. 2018, arXiv:1801.05111; Coatman et al. 2017;
Vietri et al. 2018, arXiv:1807.01978). Lyα and NV sit on the same
EV1 / disk-wind sequence: Lyα shows a (weaker) blue asymmetry, and the
NV λ1240 / Lyα ratio tracks BLR metallicity and luminosity
(Hamann & Ferland 1999; Jiang et al. 2008, arXiv:0802.4234;
Temple et al. 2021, arXiv:2106.01379).

### 1d. Anti-correlations relevant to the kernel

The classic EV1 anti-correlations are FeII ↔ [OIII] and FeII ↔ CIV core
strength. These manifest in stacked rest-frame UV/optical spectra as
*off-diagonal* covariance between widely separated wavelength windows —
exactly the structure a GP kernel on M·Mᵀ would encode.

---

## 2. Absorption-emission and absorption-absorption correlations

### 2a. Lyα forest ↔ continuum

The forest at z_abs < z_qso is a stochastic absorption modulation on the
intrinsic continuum. PCA continuum fits (Pâris et al. 2011,
arXiv:1104.2024; Suzuki 2006, ApJS) explicitly use the *correlation
between red-side line fluxes and the blue-side continuum shape* to
predict the unabsorbed continuum in the forest. The fact that this
correlation is detectable at all means that real spectra carry coupled
red-side ↔ blue-side covariance that mocks with a single continuum
template do not.

### 2b. Associated metal absorbers at z_abs ≈ z_qso

A non-trivial fraction of QSOs show narrow associated CIV, SiIV, NV, OVI,
and (at lower velocities) MgII at the systemic redshift; these are
intrinsic outflows / circumnuclear gas (Hamann & Ferland 1999 review;
Wild et al. 2008).

### 2c. Intervening DLA / LLS metal forests

Each intervening DLA at z_abs carries a metal-line system: CII λ1334,
SiII λ1260/1304/1526, FeII λ1608/2382/2600, AlII, etc., visible both
inside and outside the Lyα forest. Yang et al. 2022 (arXiv:2206.11385)
stacked the Lyα forest in eBOSS and detected up to 13 metal species
associated with relatively strong Lyα absorbers over 2 < z_abs < 4 —
the cross-correlation is significant and density/redshift dependent.
For BAO analyses DESI explicitly characterizes metal contamination of the
Lyα auto-correlation (Guy et al. 2024, arXiv:2404.03003).

### 2d. BAL and mini-BAL imprint

BALs (Δv ≥ 2000 km/s) appear in ~10-15 % of optically selected QSOs in
the strict definition and ~20-26 % under broader BI/AI thresholds
(Hewett & Foltz 2003; Knigge et al. 2008; Filiz Ak et al. 2014,
arXiv:1407.7532; for DESI EDR see Filbert et al. 2024, MNRAS 532, 3669).
BAL incidence and depth are correlated with EV1 location (population A,
weak [OIII], blueshifted CIV); the population is therefore *not*
orthogonal to emission-line properties — a fact that mocks injecting
BALs at random probability cannot capture.

### 2e. Proximity effect

Within roughly the quasar UV ionization sphere (Δz of a few × 10⁻²) the
forest is under-absorbed because the local UV background is dominated by
the host QSO (Bajtlik, Duncan & Ostriker 1988; Lu, Wolfe & Turnshek
1991; Faucher-Giguère et al.; recent DESI/eBOSS surveys reviewed in
Worseck et al. 2025, arXiv:2504.03848). This couples forest statistics
at λ → λ_Lyα(z_qso) to QSO luminosity — i.e. a deterministic
emission ↔ absorption correlation at the high-wavelength edge of the
forest window.

---

## 3. What the DESI Lyα synthetic mocks actually inject

The standard DESI mock pipeline is documented across two main papers and
the desisim source code:

- **LyaCoLoRe** (Farr et al. 2020, arXiv:1912.02763) — generates Gaussian
  random skewers from CoLoRe (lognormal + 2LPT) and maps them to
  transmitted-flux Lyα skewers via FGPA-style transforms, with
  large-scale bias parameters and BAO calibrated to match data.
- **Etourneau et al. 2024 / 2LPT mocks** (arXiv:2310.18996) — the
  eBOSS+DESI mock generation paper used for the DESI-DR1 BAO validation;
  also FGPA-on-Gaussian-fields, with HCD placement at high-density
  regions.
- **quickquasars** (desisim; see Herrera-Alcántar et al. 2024,
  arXiv:2401.00303, "Synthetic spectra for Lyα analysis in DESI") —
  the consumer that turns transmission skewers into observed DESI spectra:
  applies a continuum template (the `simqso`/`SIMQSO`-style template from
  desisim) with random emission-line amplitudes, draws DLAs (random or
  from the transmission file), draws BALs with empirical Niu (2020)
  templates at default rate 16 %, optionally paints metal forests
  (HCD-correlated SiII/SiIII/CIV through the `--metals-from-file`
  option), and adds DESI instrument noise + LSF.

What is therefore *present* in the mocks:
- Forest density / BAO at the right two-point statistics.
- HCD (DLAs, sub-DLAs / LLS) injected with correct dN/dz.
- BAL imprint at a population-averaged rate.
- Metals — at least SiII/SiIII coupled to neutral hydrogen.
- DESI noise model + per-camera LSF (simplified).

What is **not** correlated in the way real data are:
- BAL ↔ EV1 (BALs randomly assigned, not preferentially to pop-A
  blueshifted-CIV objects).
- DLA metal lines ↔ DLA N_HI/z (mock metal columns are a simple scaling).
- Emission-line ratios across the BLR (Lyα/NV/CIV/SiIV/CIII] are tied
  to a fixed template, not jointly sampled from the EV1 + Baldwin
  manifold).
- Continuum slope ↔ Lyα-forest mean flux at the QSO redshift
  (proximity-effect-like correlations).

---

## 4. What is missing — the gap that should show up in corr(M·Mᵀ)

The DESI mock pipeline papers themselves flag this gap. Etourneau et al.
2024 §6 and Herrera-Alcántar et al. 2024 §2 both note that emission-line
templates are population-averaged, BAL/DLA/metal injection rates are not
correlated with continuum properties, and that astrophysical
cross-correlations beyond the forest two-point function are not
guaranteed. Concretely:

1. **No EV1 / quasar-main-sequence variation.** Mocks use a fixed (or
   modestly perturbed) emission-line template; Population A vs B
   diversity, FeII↔[OIII], CIV blueshift correlations are absent.
2. **No Baldwin trend across luminosity bins.** Line EW does not track
   continuum L the way real data do.
3. **No realistic intrinsic + associated metals.** Mock metals are tied
   to the forest transmission, not to the QSO host metallicity, BLR
   metallicity, or the EV1 axis.
4. **BAL-emission decoupling.** BALs assigned at constant probability
   without correlation to CIV blueshift, EV1 location, or continuum slope.
5. **DLA metal lines.** I could not confirm from the mock papers I
   reviewed whether quickquasars paints the *full* DLA metal-line system
   (CII λ1334, SiII λ1260, FeII λ2382 …); the source in
   `desihub/desisim` and `desisim.dla` would be the authoritative check.
   *(speculative)* The literature implies these are at best simplified.
6. **Simplified LSF / sky-residual structure.** Real DESI spectra carry
   residual sky-subtraction features at telluric and 5577 Å OI windows
   that mocks do not model in the same fashion.
7. **No continuum-slope / forest-mean-flux coupling.** The continuum
   normalization and the forest are drawn independently, so the
   Pâris 2011-style red-side ↔ blue-side covariance is suppressed.

---

## 5. Synthesis

A GP continuum model trained on real DESI LOA spectra learns a low-rank
basis M whose covariance M·Mᵀ absorbs the intra-population structure of
real QSOs: Baldwin-like line-EW vs continuum modes, EV1-like joint
modulations of FeII/CIV/Lyα blueshift, BAL-like negative-flux modes at
preferred velocities, and a halo of metal-line covariances from
intervening DLA/LLS systems. Cross-pixel correlations therefore appear
between wavelengths that are not adjacent in λ but are physically
coupled — yielding rich off-diagonal structure in corr(M·Mᵀ). The same
GP trained on 2LPT / LyaColore mocks sees only what the mock pipeline
injected: a single continuum template (so low rank in the line regions),
randomly-drawn BAL/DLA/metal contaminants that are *uncorrelated with the
emission template*, and noise. The covariance is therefore expected to
be blockier (continuum block + roughly-diagonal noise) and to lack the
fine cross-coupling between emission, blueshift, and metal-bearing
absorption windows that the real-data kernel exhibits. The observed
difference between corr(M·Mᵀ) on LOA vs 2LPT is thus consistent with the
documented limitations of the DESI mock pipeline, not a bug in the GP
training.

---

## Open questions for further research

- Does `quickquasars` (`desisim.dla` / `desisim.metals`) paint the full
  DLA metal-line system (CII, SiII, FeII at z_abs) or only a small
  hydrogen-correlated subset (SiII λ1190/1193, SiIII λ1207)? The desisim
  source is the authoritative reference — worth a direct code read.
- Quantitative size of the EV1 / Baldwin contribution to the off-diagonal
  kernel: can we fit a 2-3 component EV1 manifold to the residual after
  subtracting the mock-style mean template, and check if that accounts
  for the LOA-vs-2LPT gap?
- Is the proximity effect detectable as a localized off-diagonal feature
  in corr(M·Mᵀ) at λ_rest ≈ 1216 Å × (1 - few × 10⁻²)?
- Would training on a luminosity-stratified subset of LOA collapse the
  Baldwin-driven covariance? This is a falsifiable test of the Baldwin
  contribution.
- How much of the LOA off-diagonal structure is from intervening DLA
  metal forests vs. from BLR emission correlations? Masking known
  DLA/BAL targets before training would disentangle these.

---

## Key references (arXiv IDs)

- Boroson & Green 1992, ApJS 80, 109 (no arXiv; original EV1).
- Sulentic et al. 2000, ARA&A 38, 521 (quasar main sequence framework).
- Hamann & Ferland 1999, ARA&A 37, 487 (BLR metallicity, intrinsic absorbers).
- Xu et al. 2008, arXiv:0806.1787 (SDSS CIV Baldwin effect).
- Baskin & Laor 2004, arXiv:astro-ph/0403365 (Baldwin/Eddington ratio).
- Ge et al. 2016, arXiv:1608.02172 (CIV Baldwin underlying driver).
- Jiang et al. 2008, arXiv:0802.4234 (N-rich QSOs, NV/Lyα).
- Pâris et al. 2011, arXiv:1104.2024 (QSO PCA, z~3 UV eigenspectra).
- Marziani et al. 2018, arXiv:1802.05575 (quasar main sequence review).
- Marziani et al. 2024 review, arXiv:2410.09262 (updated EV1).
- Sun et al. 2018, arXiv:1801.05111 (CIV blueshift variability).
- Filiz Ak et al. 2014, arXiv:1407.7532 (BAL variability / fractions).
- Yang et al. 2022, arXiv:2206.11385 (Lyα-associated metals in eBOSS).
- Farr et al. 2020, arXiv:1912.02763 (LyaCoLoRe).
- Etourneau et al. 2024, arXiv:2310.18996 (eBOSS/DESI mock data sets).
- Herrera-Alcántar et al. 2024, arXiv:2401.00303 (synthetic DESI Lyα).
- Guy et al. 2024, arXiv:2404.03003 (DESI Lyα contaminants).
- Cuceu et al. 2024, arXiv:2404.03004 (DESI 2024 Lyα BAO mock validation).
- DESI Collab. 2024, arXiv:2404.03001 (DESI 2024 IV: Lyα BAO).

---

## Caveats (flagged by the agent)

- Section 4 item 5: whether `quickquasars` paints the full DLA metal-line
  system vs. a hydrogen-correlated subset was not directly verifiable
  from the mock papers reviewed; reading `desisim.dla` / `desisim.metals`
  source is the next step.
- Section 1c: Richards et al. 2011 was widely cited in search results but
  the original abstract was not retrieved; the surrounding CIV-blueshift
  citations are independent.
