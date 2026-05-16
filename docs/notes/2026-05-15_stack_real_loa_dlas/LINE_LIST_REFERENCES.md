# UV absorption-line list — sources & verification

Reference for the metal-line labels used in the real-LOA DLA/sub-DLA/LLS
stacking plots (`examples/stack_real_loa_dlas.py`, `METAL_LINES` dict).

## Provenance

- The core resonance lines (Lyman series, OI 1302, SiII 1260/1304/1527,
  CII 1334, SiIV 1394/1403, CIV 1548/1551, OVI 1032/1038) originate from
  the `absorber_IGM` dictionary in
  `desisim/lya_spectra.py` (DESI's own line list).
- The forest-region complex (FeII 1063/1097/1125/1143/1145, FeIII 1122,
  NI 1134/1200, PII 1152, SiII 1190/1193, SiIII 1206, CIII 977) and the
  later additions (SII 1250/1253, CII* 1335.71, NV 1238/1242) were
  cross-checked against the authoritative atomic-data references below.

## Sources

- **Morton 2003**, ApJS 149, 205 — *Atomic Data for Resonance Absorption
  Lines in the Spectra of Quasars* (vacuum wavelengths; the standard
  reference). https://iopscience.iop.org/article/10.1086/377639
- **Morton 2000**, ApJS 130, 403 — Atomic Data II.
- **Mas-Ribas et al. 2017**, ApJ 846, 4 — *The Mean Metal-line Absorption
  Spectrum of Damped Lyα Systems in BOSS* (arXiv:1610.02711). The
  definitive DLA-stack line list (50 lines + 13 blends); confirms the
  CII*, FeIII 1122, OVI, NV detections in DLA composites.
- **NIST Atomic Spectra Database** — https://physics.nist.gov/PhysRefData/ASD/

## Verification history

Two independent reviews (2026-05-15) checked every wavelength against
Morton 2003 / NIST / Mas-Ribas 2017:

- **All 36 wavelengths accurate to < 0.1 Å** — no numeric corrections.
- **CIII 1175.71 REMOVED** — it is the *excited-state* C III multiplet
  (lower level 2s2p ³P°, requires a populated metastable level), NOT a
  ground-state resonance line. It does not appear in intervening DLA
  absorption. The genuine ground-state C III resonance line is
  **CIII 977.02**, which is retained.
- **OI(989) = 988.77** is the OI wavelength; the ~989 Å feature is a
  blend (OI triplet 988.58/988.66/988.77 + NIII 989.80 + SiII 989.87).
  Labelled at the OI component.

## Verified line table (vacuum rest wavelengths, Å)

| Label | λ_vac | Class | Note |
|---|---|---|---|
| Ly5 | 937.80 | H I | Lyman series 6→1 |
| Ly4 | 949.74 | H I | Lyman series 5→1 |
| Lyγ | 972.54 | H I | Lyman series 4→1 |
| CIII(977) | 977.02 | resonance | ground-state C III |
| OI(989) | 988.77 | resonance (blend) | OI/NIII/SiII ~989 blend |
| Lyβ | 1025.72 | H I | Lyman series 3→1 |
| OVI(1032) | 1031.91 | **high-ion** | warm/hot gas; weak in stacks |
| OVI(1038) | 1037.61 | **high-ion** | warm/hot gas; weak in stacks |
| OI(1039) | 1039.23 | resonance | |
| FeII(1063) | 1063.18 | resonance | |
| FeII(1097) | 1096.88 | resonance | |
| FeIII(1123) | 1122.52 | resonance | FeIII — weak/marginal in DLAs |
| FeII(1125) | 1125.45 | resonance | |
| NI(1134) | 1134.17 | resonance | triplet 1134.17/1134.41/1134.98 |
| FeII(1143) | 1143.23 | resonance | |
| FeII(1145) | 1144.94 | resonance | |
| SiII(1190) | 1190.42 | resonance | SIII 1190.21 blends here |
| SiII(1193) | 1193.29 | resonance | |
| NI(1200) | 1200.22 | resonance | triplet 1199.55/1200.22/1200.71 |
| SiIII(1207) | 1206.50 | resonance | |
| Lyα | 1215.67 | H I | Lyman series 2→1 |
| NV(1239) | 1238.82 | **high-ion** | warm/hot gas; weak in stacks |
| NV(1243) | 1242.80 | **high-ion** | warm/hot gas; weak in stacks |
| SII(1251) | 1250.58 | resonance | SII metallicity triplet |
| SII(1254) | 1253.81 | resonance | SII metallicity triplet |
| SiII(1260) | 1260.42 | resonance | strong |
| OI(1302) | 1302.17 | resonance | strong |
| SiII(1304) | 1304.37 | resonance | |
| CII(1335) | 1334.53 | resonance | strong; ground-state |
| CII*(1336) | 1335.71 | **excited fine-structure** | traces cooling/SF, NOT column |
| SiIV(1394) | 1393.76 | high-ion | |
| SiIV(1403) | 1402.77 | high-ion | |
| SiII(1527) | 1526.71 | resonance | |
| CIV(1548) | 1548.20 | high-ion | |
| CIV(1551) | 1550.78 | high-ion | |

### Line classes

- **resonance** — ground-state low-ion lines; the cool-gas column-density
  tracers. The expected absorption signature of a real DLA.
- **high-ion** (OVI, NV, SiIV, CIV) — trace warm/ionized gas; weak in DLA
  stacks but genuinely detected in deep composites. CIV is the decisive
  sub-DLA/LLS false-positive discriminant (locked doublet ratio).
- **excited fine-structure** (CII* 1335.71) — populated by collisions in
  dense gas; a real, standard DLA-stack feature (Mas-Ribas 2017) but it
  traces excited gas / cooling rate, not HI column directly.

## Reliability: is a stacked dip genuinely at z_DLA?

A stacked metal-line dip is only trustworthy as a z_DLA feature if it
cannot be mimicked by absorption from gas at *other* redshifts. The
dividing line is **Lyα 1216 Å in the absorber rest frame**:

- **Redward of Lyα (> 1216 Å) — TRUSTWORTHY.** Outside the Lyα forest.
  A coherent dip here is the DLA's metal. Residual contamination is
  minor and incoherent (stray intervening metal systems, sky residuals).
  Lines: SiII 1260, OI 1302, SiII 1304, CII 1334, SiIV 1394/1403,
  SiII 1527, CIV 1548/1551.

- **Blueward of Lyα (< 1216 Å) — FOREST-FLOORED.** These sit in the Lyα
  forest. Intervening HI Lyα from random redshifts is uncorrelated with
  z_DLA, so in a stack it averages *incoherently* into a smooth depressed
  pseudo-continuum (this is why forest stacking works at all — Pieri+2010,
  Mas-Ribas+2017). The genuine z_DLA metal survives as a coherent dip on
  that floor, but per-system confirmation is unreliable here.
  - *Usable in a stack (EW-recoverable):* CIII 977, SiII 1190/1193,
    NI 1200, FeII 1144.
  - *Forest-confused — do NOT use to confirm z_DLA:* OI 989, OI 1039,
    FeII 1063/1097/1125, NI 1134, SiIII 1206 (only 9 Å from Lyα — blends
    with the DLA's own Lyα damping wing), OVI 1032/1038. **PII 1152 was
    dropped from the plot for this reason.**

How the literature mitigates it: incoherent averaging + local
pseudo-continuum fitting per line + doublet-ratio cross-checks
(SiIV 1394:1403, CIV 1548:1551 at fixed ratios) + keying DLA confirmation
to the strong *redward* low-ions. See Mas-Ribas+2017, Pieri+2014.

Practical rule for these plots: **trust the redward panels** (SiII 1260
through CIV 1548/1551) and the **z-scrambled control** as the decisive
checks; treat forest-region dips as supporting evidence only.

Extra references for this section:
- **Pieri et al. 2014**, MNRAS 441, 1718 — arXiv:1309.6768.
- **Pieri et al. 2010**, MNRAS 402, 1145 — arXiv:0911.5111.
- **Noterdaeme et al. 2014**, A&A 566, A24 — arXiv:1403.6608.
- **Morton 2003 erratum**, ApJS 151, 403 (2004).

## Known omissions (out of the current 900–1600 Å rest window)

- **FeII 1608.45** — flagship DLA FeII line, just past the 1600 Å cap.
  Extending `REST_LAMBDA_MAX` to ~1620 Å (needs a re-stack) would add it.
- **AlII 1670.79**, **AlIII 1854.7/1862.8**, **ZnII 2026/2062**,
  **CrII 2056/2062/2066**, **MgII 2796/2803** — all redward of 1600 Å.
  ZnII/CrII are the canonical DLA metallicity lines; a separate redward
  panel would be needed.
- **SIII 1190.21** blends into the SiII 1190.42 panel.
