# DESI mock quickquasars chain: what's actually injected

## Summary

For the 2LPT loa-124 (v2.8.5, mock-0) chain used in this project, the
Lyman-series absorption (LYB, LY3, LY4, LY5) is **not** computed on-the-fly
from the H I skewer by `quickquasars`; instead, each higher-series line has
its own transmission HDU in the LyaCoLoRe / 2LPT transmission file,
generated from the same underlying density field as the Lyα skewer, and
`quickquasars` simply reads it, raises it to a tuned "strength" exponent,
and multiplies it into the flux. Metals (SiII/SiIII) are likewise carried
as per-line skewer HDUs and follow the H I density field (they are *not*
HCD-only). DLAs come from a truth catalog (`--dla file`) and are painted
as analytic Voigt profiles. BALs at `--balprob 0.16` are drawn from the
Niu (2020) empirical template library, independent of any QSO emission
property.

## The chain

```
CoLoRe Gaussian field  →  LyaCoLoRe (FGPA on Lyα + per-line FGPA on Lyβ/Ly3/4/5
                          + metal transmission tracing the same δ_H)
                       →  per-healpix transmission FITS with HDUs
                          {WAVELENGTH, METADATA, F_LYA, F_LYB, F_METALS, DLA?, …}
                       →  quickquasars  (continuum template + flux × T_LYA(λ)
                                         × T_LYB(λ)^s_LYB × T_LY3(λ)^s_LY3 …
                                         × T_metal(λ)^s_metal × Voigt(DLA)
                                         × BAL_template + DESI noise + LSF)
                       →  spectra-{healpix}.fits
```

For 2LPT the upstream generator is the LyaCoLoRe-derivative documented by
Etourneau+2024 (arXiv:2310.18996); the consumer code is
`desisim/scripts/quickquasars.py` + `desisim/lya_spectra.py` +
`desisim/bal.py` + `desisim/dla.py` (desisim API docs at
https://desisim.readthedocs.io/en/latest/api.html).

The exact CLI for our mock is:
```
quickquasars … --zbest --bbflux --zmin 1.7 --save-continuum --seed 0
  --from-catalog seed_zcat.fits
  --dla file
  --metals LYB LY3 LY4 LY5 SiII(1260) SiIII(1207) SiII(1193) SiII(1190)
  --metal-strengths 0.1901 0.0697 0.0335 0.0187 1.3e-03 3.5e-03 0.7e-03 1.4e-03
  --balprob 0.16
```

## Lyman series injection

**Where it lives.** Documentation for
`desisim.lya_spectra.apply_metals_transmission` explicitly warns *"This
function should not be used in London mocks with version > 2.0, since these
have their own metal transmission already in the files, and even the
'TRANSMISSION' HDU includes already Lyman beta."* (desisim 0.37 docs).
For the post-v2.0 LyaCoLoRe and 2LPT files in use today, the LYB / LY3 /
LY4 / LY5 transmission skewers are written by the LyaCoLoRe
`make_master.py` / save_master pipeline as additional FITS HDUs, computed
by applying an FGPA-style τ → e^-τ map *with the same underlying CoLoRe
density field* used for Lyα. They are therefore line-of-sight correlated
with the Lyα forest by construction (Farr+2020 arXiv:1912.02763 §3.2;
Etourneau+2024 arXiv:2310.18996 §3; Herrera-Alcántar+2024
arXiv:2401.00303 §2).

**How `quickquasars` applies them.** When the CLI lists
`--metals LYB LY3 LY4 LY5 … --metal-strengths s_LYB s_LY3 …`, the code
reads each named HDU and multiplies the QSO flux by `T_line^s_line`
pixel-wise (equivalently τ' = s · τ_skewer). The "strength" is therefore a
τ-rescaling exponent, *not* a fixed atomic-physics ratio (desisim docs for
`apply_metals_transmission`: "list of float strengths to apply to metals").

**Are the strengths physically motivated?**

The naive single-line atomic-physics ratio at fixed *observed* λ for an
optically thin line is

```
τ_LYβ / τ_LYα = (f_LYβ · λ_LYβ) / (f_LYα · λ_LYα)
              = (0.07914 · 1025.72) / (0.4164 · 1215.67) ≈ 0.160
```

(NIST oscillator strengths). The value used in the mock, 0.1901, is ~19%
higher. The reason is that LyaCoLoRe / 2LPT does **not** generate the
higher-series transmission from the Lyα τ scaled by an atomic ratio —
instead each line uses an independent FGPA fit with its own bias and
τ_eff(z), tuned so the mock 1D / 3D auto- and Lyα×Lyβ cross-correlations
reproduce eBOSS/DESI data (Etourneau+2024 §3.2; Herrera-Alcántar+2024 §2).
The downstream `--metal-strengths` lever then provides one extra global
scalar per line that the DESI mock team adjusts at the catalog-tuning
stage so the *final synthetic spectrum* matches the Lyα×Lyβ
cross-correlation in real data.

So 0.1901 should be read as an *effective* multiplier that absorbs (i)
the single-line optical-depth ratio, (ii) the redshift evolution of τ_eff
for each line averaged over the DESI forest, and (iii) any residual
mismatch in the upstream LyaCoLoRe FGPA tuning. The values for LY3
(0.0697), LY4 (0.0335), LY5 (0.0187) follow a roughly geometric decline
consistent with falling f-values + increasing IGM optical depth at the
higher-line observed wavelengths.

⚠ Caveat (agent): I could not retrieve the explicit calibration table
from Herrera-Alcántar+2024 because the arXiv HTML was web-fetch-blocked;
the canonical reference is §2 / Table 1 of arXiv:2401.00303.

## DLA injection

`--dla file` directs `quickquasars` to read the DLA truth catalog stored
as a separate HDU in the LyaCoLoRe transmission file (Farr+2020 §3.4,
"high column density systems"). LyaCoLoRe places DLAs by stochastic
sampling of high-δ peaks along each skewer using a Pérez-Ràfols+2018
N_HI(z, δ) distribution. `desisim.dla.insert_dlas` then paints each DLA as
an analytic Voigt profile using the Tepper-García (2006,
arXiv:astro-ph/0602124) Voigt-Hjerting approximation, with line parameters
(logN, z, b, λ_rest, f, γ) for Lyα (and Lyβ as a separate transition when
included). The DLA metadata HDU schema is
`{TARGETID, DLAID, Z_DLA, NHI}` — TARGETID links to the QSO catalog
(desisim API docs for `insert_dlas`).

## BAL injection

`--balprob 0.16` sets the per-QSO probability that a BAL template is
multiplied into the spectrum. BAL templates are the Niu (2020) empirical
library shipped with `desisim.bal`: each template is a normalized
CIV-BAL trough as a function of velocity, drawn from high-SNR SDSS DR14
spectra, and the library has been thinned to reproduce the DR14 BAL AI/BI
distributions
(https://desisim.readthedocs.io/en/0.37.0/_modules/desisim/bal.html;
desisim notebook nb/bal-templates). The 16 % default sits between the
strict BI-based fraction (~10–13 %; Filiz Ak+2014 arXiv:1407.7532) and the
broader AI-based fraction (~20–26 %); it matches the DESI EDR BAL fraction
of 12–20 % (Filbert+2024, MNRAS 532, 3669 = arXiv:2309.03434).

**Crucially**, the template is selected uniformly at random — there is no
coupling to CIV blueshift, EV1 position, or continuum slope of the
assigned QSO.

## Metal lines (SiII/SiIII)

The four metals in the CLI — SiII(1260), SiIII(1207), SiII(1193),
SiII(1190) — are the silicon transitions called out in Etourneau+2024
§3.3 and Herrera-Alcántar+2024 §2 as the most important for the DESI
Lyα×metal cross spectrum. They are stored as **per-line transmission
skewers** in the LyaCoLoRe output, computed from the same δ_H field used
for Lyα (linear bias, not HCD-localized). Hence they are **spread along
the line of sight following the neutral-hydrogen density**, not only where
DLAs sit. The `--metal-strengths` 1.3e-3 to 3.5e-3 are the same
τ-rescaling exponent as for the Lyman-series lines; the small absolute
values reflect that silicon has a much lower column-density-weighted
average τ than H I (Yang+2022 arXiv:2206.11385 stack gives
τ_SiIII ≈ 10⁻³ τ_Lyα at z ~ 2.5).

## What is NOT in the chain

A companion note documents the population-level correlations real DESI
quasars carry but the LyaCoLoRe / quickquasars chain does not. See
`../2026-05-13_qso_emission_absorption_correlations/findings.md` for the
full bibliography. One-line summary, *missing from the mock*:

- **BAL ↔ EV1 emission coupling** — `--balprob` is uncorrelated with CIV
  blueshift or pop A/B.
- **Baldwin effect** — line EW does not track continuum luminosity; the
  emission-line template is fixed/SIMQSO.
- **Intervening DLA full metal forests** — only Si II/Si III tracking δ_H;
  CII 1334, FeII 2382–2600, MgII 2796/2803 are absent.
- **CIV blueshift / Sulentic main sequence** — fixed template, no
  orientation or L/L_Edd diversity.
- **Continuum-slope ↔ forest mean flux** — continuum and δ_H drawn
  independently (Pâris 2011-style red↔blue covariance suppressed).
- **Proximity effect** — UV-background enhancement near the QSO is not
  modeled.
- **Associated metal absorbers at z_abs ≈ z_qso** — no intrinsic outflows
  beyond the BAL template.

## Implications for the GP corr(M·M^T) interpretation

A GP continuum trained on loa-124-anti-joined 2LPT spectra **sees Lyα–Lyβ
line-of-sight covariance** because the LYB transmission skewer is drawn
from the same δ_H field as Lyα and is multiplied into the same per-pixel
flux — so the kernel will indeed encode a non-trivial Lyα–Lyβ cross-pixel
correlation, with off-diagonal weight at (λ_Lyα(z_abs), λ_Lyβ(z_abs))
pairs. However, after rest-frame normalization and de-forest
preprocessing the amplitude is small: τ_LYB-strength × τ_eff(Lyβ at z) /
typical continuum is < 10 % of the Lyα-only forest variance on a
per-pixel basis, and after a k=30 PCA truncation it will compete with the
much louder continuum modes and noise blocks. So Lyα–Lyβ cross-coherence
*should* be present in corr(M·M^T), but only as a sub-leading off-diagonal
lobe in the blue-of-Lyα region; we should not expect it to dominate the
off-diagonal structure.

By contrast, the EV1 / Baldwin / CIV-blueshift cross-line correlations
that drive the loa-vs-mock corr(M·M^T) gap are wholly absent from the
chain, which is the load-bearing prediction of the companion note.

## References

- desisim source / docs: https://github.com/desihub/desisim ;
  https://desisim.readthedocs.io/en/stable/_modules/desisim/scripts/quickquasars.html ;
  https://desisim.readthedocs.io/en/latest/api.html
- LyaCoLoRe: https://github.com/igmhub/LyaCoLoRe
- Farr et al. 2020, "LyaCoLoRe", arXiv:1912.02763.
- Etourneau et al. 2024, "Mock data sets for the eBOSS and DESI Lyα
  forest surveys", arXiv:2310.18996.
- Herrera-Alcántar et al. 2025, "Synthetic spectra for Lyα forest
  analysis in DESI", arXiv:2401.00303 (JCAP 01, 141).
- Cuceu et al. 2025, "Validation of the DESI 2024 Lyα BAO analysis using
  synthetic datasets", arXiv:2404.03004 (JCAP 01, 148).
- Guy et al. 2024, "Characterization of contaminants in the Lyα
  auto-correlation with DESI", arXiv:2404.03003.
- Filbert et al. 2024, "BAL quasars in the DESI EDR", MNRAS 532, 3669
  (arXiv:2309.03434).
- Filiz Ak et al. 2014, arXiv:1407.7532.
- Tepper-García 2006, arXiv:astro-ph/0602124.
- Pérez-Ràfols et al. 2018 (DLA N_HI distribution used by LyaCoLoRe).
- Niu 2020 (BAL templates shipped with desisim.bal).
- Yang et al. 2022, arXiv:2206.11385.
- Companion note: `../2026-05-13_qso_emission_absorption_correlations/findings.md`.

## Caveats flagged by the agent

- The exact numerical values 0.1901 / 0.0697 / 0.0335 / 0.0187 for
  LYB/LY3/LY4/LY5 appear in our local production CLI but the matching
  calibration table from Herrera-Alcántar+2024 was not directly retrieved
  this session (arXiv HTML + GitHub raw fetches blocked). The
  interpretation as a tuned τ-exponent (vs. fixed atomic-physics ratio)
  is supported by the desisim source-level docs for
  `apply_metals_transmission` (explicit warning that London mocks v > 2.0
  ship the LYB + metal transmission in the same FITS file, and the
  strengths are tunable scaling factors).
- The statement that LyaCoLoRe per-line FGPA uses independent bias
  parameters (not a τ_α-rescaling) is reported via search-engine excerpts
  of Etourneau+2024 §3.2; the primary text was not directly retrieved.
- Niu (2020) is referenced in the desisim.bal module docstring as the BAL
  template source. A direct arXiv ID for Niu 2020 was not in the search
  results.
