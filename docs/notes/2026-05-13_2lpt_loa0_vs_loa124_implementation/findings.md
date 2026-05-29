# 2LPT mock implementation: loa-0 vs loa-124

> **Investigator constraint** (flagged by the writing agent): direct filesystem
> access to `/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/` was blocked
> for this run. The report below is built entirely from cross-references inside
> the repo (docstrings, manifest, unit tests, 3-way-compare report, training
> READMEs, de-forest source). To upgrade rows marked "uncertain" to confirmed,
> a follow-up session with read access to the mock tree (and ideally a peek at
> `desisim.lya_spectra`) is needed.

## TL;DR

- `loa-0` and `loa-124` are **two contamination variants of the same
  LyaCoLoRe / 2LPT v2.8.5 mock-0 base** that differ in *what was injected on
  top of the transmission skewer*:
  - `loa-0` = uncontaminated (Lyα forest + continuum only)
  - `loa-124` = contaminated (DLAs + sub-DLA/LLS + BALs + simple H-correlated
    metals overlaid)
- They share the underlying CoLoRe density skewers and TARGETID list.
  `tests/test_smoke_target_contamination.py` verifies TID 120046865 is in both
  files; the only difference at the DLA's apparent Lyα is a flux suppression
  from +0.31 (loa-0) → −0.04 (loa-124) inside ±20 Å.
- The kernel-structure observation (2LPT `corr(M·Mᵀ)` lacks Lyα↔Lyβ
  cross-correlation rungs visible on real LOA) is **not** because loa-0 vs
  loa-124 differ in higher-order Lyman series injection. Both inherit the same
  transmission skewer; the de-forest layer subtracts the deterministic mean
  Lyman-series optical depth (`num_forest_lines=31`) before the GP sees the
  data, so cross-rung physics shows up in the fluctuations regardless. The
  cross-rung gap is the more fundamental mock limitation already documented in
  `../2026-05-13_qso_emission_absorption_correlations/findings.md`.

## Directory layout (inferred from references; not visited directly)

```
/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/
└── mock-0/
    ├── loa-0/
    │   ├── zcat.fits                 (used in tests; no DLA/BAL truth)
    │   └── spectra-16/{hpx//100}/{hpx}/spectra-16-{hpx}.fits
    └── loa-124/
        ├── zcat.fits
        ├── bal_cat.fits              (referenced from MANIFEST.json)
        ├── hcd_truth_cat.fits        (referenced from preload script)
        └── spectra-16/{hpx//100}/{hpx}/spectra-16-{hpx}.fits
```

Healpix layout `spectra-16/{healpix//100}/{healpix}/` is the standard DESI
nside=16 hierarchy — see `preload_spectra/preload_2lpt_simple.py:62-65`.

Only `mock-0` appears in the repo for 2LPT v2.8.5. Sibling versions / mocks
weren't confirmed.

## What "loa-N" means

The repo uses `loa-N` as a quickquasars *contamination preset* label,
analogous to the London `jura-N` and Saclay `juraLy8-124` labels in older
notes (`CLAUDE.md` §11, `docs/nersc_greatlakes_mapping.md` §4):

| Tag | Convention used in this repo |
|---|---|
| `-0` suffix | uncontaminated baseline: transmission + continuum + DESI noise + LSF, **no** DLA / sub-DLA / BAL / metals overlaid |
| `-124` suffix | contaminated preset: DLAs and sub-DLAs (HCDs) drawn from a truth catalog with the right dN/dz, BALs at the population-averaged ~16 % rate, simple H-correlated metals, plus all of the `-0` content |

The "124" likely encodes a quickquasars contamination seed / config ID. The
"loa" prefix is most likely the LyaCoLoRe-pipeline label (analogous to
"jura" for London/Saclay). No primary-source definition of "loa" was found
in the repo; the meaning is inferred from usage.

## Per-component diff loa-0 vs loa-124

| Component | loa-0 | loa-124 | Evidence |
|---|---|---|---|
| Lyα forest (FGPA transmission) | yes | yes | identical underlying CoLoRe/2LPT skewers; `tests/test_smoke_target_contamination.py:79-85` confirms TID 120046865 is in both files |
| Higher-order Lyman series (Lyβ 1025.7, Lyγ 972.5, Lyδ 949.7, …) | **uncertain** — needs `desisim.lya_spectra` read | uncertain (likely same as loa-0) | de-forest layer adds them analytically regardless via `num_forest_lines=31` |
| DLAs (truth catalog) | no (forest-only by construction) | yes (drawn into spectrum + recorded in `hcd_truth_cat.fits`) | `docs/notes/2026-04-28_v2_3way_compare/report.md:11`; `tests/fixtures/2lpt_frozen/MANIFEST.json:3`; smoke contamination test |
| Sub-DLAs / LLS | no | yes (in `hcd_truth_cat.fits` with NHI ≥ 17) | `preload_spectra/preload_2lpt_simple.py:84-93` |
| BALs | no | yes (entries in `bal_cat.fits` with BI_CIV / AI_CIV) | `preload_spectra/preload_2lpt_simple.py:96-106` |
| Metal lines | no | yes, but only simple HCD-correlated metals (SiII/SiIII λ1207, etc.) — full DLA metal forest uncertain | `../2026-05-13_qso_emission_absorption_correlations/findings.md` §3, §4 item 5 |
| QSO continuum template | same `simqso`-style population template, randomly drawn emission-line amplitudes | same | `../2026-05-13_qso_emission_absorption_correlations/findings.md` §3 |
| DESI noise + per-camera LSF | yes | yes (same DESI sim layer) | `docs/notes/2026-04-28_v2_3way_compare/report.md` μ(λ), ω(λ) overlay |
| Seed / realization | same skewer / same TARGETIDs as loa-124 | same | smoke-test parity at TID 120046865; n=236,755 (loa-0, no filter) vs n=203,984 (loa-124, post HCD+BAL anti-join) — the ~33k difference matches the HCD+BAL contaminated subset |

## Evidence (file paths + relevant snippets)

1. **`docs/nersc_greatlakes_mapping.md:49-50`** (authoritative one-liner):
   > 2LPT v2.8.5 mock-0 uncontaminated  /nfs/turbo/.../mock-0/loa-0/  uncontaminated baseline for Voigt injection-recovery tests
   > 2LPT v2.8.5 mock-0 contaminated    .../loa-124/   DLA + metals + BAL contamination

2. **`preload_spectra/preload_2lpt_simple.py:1-30`**:
   > "Reads 2LPT mock spectra (loa-0 uncontaminated or loa-124 contaminated),
   > optionally filters out HCDs (DLA / sub-DLA / LLS) and BALs from the truth catalogs …"

3. **`tests/test_smoke_target_contamination.py:104-116`**:
   ```
   # Empirical numbers measured 2026-04-27 on this machine:
   #   loa-124 (contaminated):  mean ≈ -0.04
   #   loa-0   (uncontaminated): mean ≈ +0.31
   ```
   measured in a ±20 Å window around the truth DLA's Lyα (z=2.773, logNHI=21.26)
   on TARGETID 120046865.

4. **`docs/notes/2026-04-28_v2_3way_compare/report.md:165-179`** confirms that
   a GP trained on loa-0 vs a GP trained on loa-124 *with HCDs and BALs
   anti-joined out* produces functionally identical μ(λ), ω(λ), eigenspectra,
   and corr(M·Mᵀ). The anti-join works.

5. **`docs/notes/2026-05-11_desi_phase2_2lpt_loa{0,124_nohcd_nobal}_wide/README.md`**
   training cards: loa-0 wide trainset has 236,755 spectra after z/SNR/cap;
   loa-124 *nohcd_nobal* has 203,984 — the ~33k difference is the HCD+BAL
   contaminated subset getting filtered.

6. **`gpy_dla_detection/effective_optical_depth.py`** shows the de-forest model
   loops over `num_forest_lines` Lyman transitions and subtracts them
   analytically. Production sets `num_forest_lines=31` in
   `tests/phase2_train_desi.py`. So whether or not the mocks injected
   higher-order Lyman absorption, the de-forest layer fits all 31 of them.

7. **Inconsistency flagged**: `docs/production_models.md:161` originally read
   "2lpt loa-0 (DLAs + BALs included)" but every other reference in the repo
   says loa-0 is "uncontaminated by construction". **Fixed in this commit.**

## Implication for the GP corr(M·Mᵀ) interpretation

The user's hypothesis — "2LPT mocks may only inject Lyα, not the full Lyman
series, so the GP kernel can't pick up higher-order line cross-correlations"
— is **not the right framing** for why the corr plot looks the way it does:

1. **Both loa-0 and loa-124 inherit the same transmission skewer** from
   CoLoRe / 2LPT, which encodes neutral-H optical depth at multiple absorber
   redshifts simultaneously. When `quickquasars` turns that skewer into
   observed flux, transmitted flux at λ_obs depends on absorbers at multiple
   absorber redshifts (one per Lyman transition). The Lyα-↔-Lyβ cross-rung
   physics is *present in the data*, not "missing because the mock only
   injects Lyα" — see Farr et al. 2020 (arXiv:1912.02763) §4 and
   Herrera-Alcántar et al. 2024 (arXiv:2401.00303) §2.
2. **The de-forest layer subtracts the deterministic mean Lyman-series
   optical depth before fitting M.** With `num_forest_lines=31`, the GP only
   sees *fluctuations* around the Turner+2024 mean. Cross-correlations
   between forest pixels and Lyβ-/Lyγ-shifted pixels live in those
   fluctuations.
3. **What 2LPT genuinely lacks vs real LOA is the BLR/continuum covariance**
   — Baldwin effect, EV1 / quasar main sequence, CIV blueshift correlations,
   BAL ↔ emission-line coupling, intervening DLA metal forests beyond
   simplest H-correlated metals, continuum-slope ↔ forest-mean-flux coupling.
   These produce off-diagonal features in corr(M·Mᵀ) that the GP picks up on
   real LOA but cannot pick up on a mock where contaminants are drawn
   independently of the emission template. See
   `../2026-05-13_qso_emission_absorption_correlations/findings.md`.

A useful falsifying test of the "Lyman series is what's missing" hypothesis:
train the GP on loa-124 (contaminated) *without* the HCD/BAL anti-join, and
compare corr(M·Mᵀ) to the anti-joined loa-124 and to loa-0. The
2026-04-28 3-way-compare already partly does this (loa-0 vs
loa-124-nohcd-nobal) and finds them functionally identical → implicates
*intrinsic mock physics* (population-averaged emission template, no
EV1/Baldwin diversity, etc.) rather than residual contamination or
higher-Lyman-series treatment as the difference vs real LOA.

## Open questions (for a follow-up session with /nfs/turbo read access)

- Does `desisim.lya_spectra` / `quickquasars` actually paint per-pixel
  higher-order Lyman series absorption from the transmission skewer, or does
  it apply only Lyα and rely on the consumer's effective-optical-depth model
  for higher lines? Reading `desihub/desisim/py/desisim/lya_spectra.py` and
  the `transmission` HDU schema in any `truth-16-{hpx}.fits` would settle it.
- What does the "124" suffix encode literally — a `--config 124` CLI flag, a
  configuration-file revision number, a metals/BAL seed ID? Without
  `SOURCE.txt` access this cannot be confirmed.
- Are there sibling `loa-N` directories (e.g. `loa-1`, `loa-12`) that
  represent intermediate contamination presets (DLA-only, BAL-only,
  metals-only)? The repo only references `loa-0` and `loa-124`.
- Does `loa-0` actually contain zero BALs in zcat / no BAL truth catalog at
  all (clean by construction), or does it contain a `bal_cat.fits` with zero
  rows? `preload_2lpt_simple.py:96-99` warns rather than errors if the BAL
  catalog is missing, suggesting the loa-0 directory may simply lack the
  file — but a direct `ls /…/mock-0/loa-0/` would confirm.
