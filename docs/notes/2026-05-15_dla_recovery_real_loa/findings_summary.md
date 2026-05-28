# Real-LOA DLA-recovery test — aggregate summary

Date: 2026-05-15. Model under test: `loa_no_dla_no_bal_wide_m_normmask_3000iter`
(post-reorder LOA-trained GP, SLURM 50087967).

This is the **aggregate-only** summary. The full per-target table and the
100 per-target JSONs contain real DESI LOA TARGETIDs / per-object N_HI
and are kept local (gitignored), not committed.

## Purpose

In-distribution validation of the post-reorder LOA model on **real DESI
LOA spectra**. The 2lpt canonical-TID test (`dla_recovery_step_c.py`)
validated the 2lpt `_m_normmask` models in-distribution, but this LOA
model had only been tested out-of-distribution (on a 2lpt mock target).

## Setup

- **Targets**: 100 strong DLAs that v1 production confidently detected —
  NHI ∈ [20.3, 22.0], P_DLA ≥ 0.99, SNR_forest > 2, intervening
  (not proximate), stratified across NHI. 98 evaluated (2 not present
  in the LoaArchive).
- **Reference ("truth")**: v1 production — the `dlacat-loa-main-dark.fits`
  catalog IS v1's output (epoch_920 GP-DLA run).
- **Spectra**: real DESI LOA, from the LoaArchive `loa_full_z2_noR_v2.h5`.
- **Inference**: multi-DLA mode (max_dlas=3, single_absorber=False),
  10k QMC grid. Each detected DLA is matched to the catalog DLA by
  closest redshift before comparing MAP log N_HI.
- **Caveat**: the v1 catalog was produced in LLS single-absorber mode;
  this run is multi-DLA. For isolated strong DLAs the p_DLA and MAP N_HI
  are comparable across modes.

## Verdict — PASSED

- **Detection agreement**: the new model recovers **94/98 (96%)** at
  p_DLA > 0.5 and **91/98 (93%)** at p_DLA > 0.97 — on DLAs v1 detected
  at P_DLA ≥ 0.99.
- **Redshift match**: 92/98 targets have the new model's matched DLA
  within |Δz| < 0.01 of the catalog DLA (same absorber).
- **MAP log N_HI bias** (new − v1, well-matched): median **−0.043 dex**,
  mean −0.054, scatter (MAD) 0.076 dex. The new model's N_HI is
  marginally lower than v1 — well within typical N_HI uncertainties.

The post-reorder LOA model `loa_no_dla_no_bal_wide_m_normmask_3000iter`
reproduces v1 production's confident strong-DLA detections on real LOA
spectra. This completes the in-distribution validation flagged as the
open item in `docs/CURRENT_MODELS.md`.

## Reproduce

```bash
python examples/dla_recovery_real_loa.py              # full ~100-target run
python examples/dla_recovery_real_loa.py --limit 3    # smoke test
python examples/dla_recovery_real_loa.py --findings-only   # rebuild from cached JSONs
```
