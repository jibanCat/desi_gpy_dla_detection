# v3_loa0 GP-DLA on London mock-0, 8 spectra-16 files — Results

Run date: 2026-05-12. Same harness as the v3_loa124 reference run
(`run_local.sh` on Jupyter Perlmutter node, 8 files x 8 parallel,
window=8, max-workers=8). Inference wall: 14:13:14 -> 14:54:24 = **~41 min**
(v3_loa124 was 38 min on the same node — comparable).

## Settings

| Param | Value |
|---|---|
| LEARNED_FILE | `/pscratch/sd/j/jibancat/prod533_5k_20260511/null_gp_test/converted/2lpt_loa0_wide.h5` |
| Source config | `slurm/configs/london0_y3.env` (LEARNED_FILE + tau-EB env overrides) |
| MOCKDIR | `/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124` |
| MAX_DLAS | 3 |
| FILTER_LOW_LIKELIHOOD | 1 |
| NUM_DLA_SAMPLES | 50000 |
| DLA_SAMPLES_FILE | `data/dr12q/processed/pw_samples_a3_190_220_50000.mat` (PW14 [19,22]) |
| ENABLE_TAU_EB | 1 (objective="null") |
| prev_tau_0 / prev_beta | 0.00246 / 3.62 (Turner+2024) |
| BAL mask | OFF (BAL QSOs included, then excluded at molly evaluation) |
| Files processed | 8 (level2 0..8 -> spectra-16 0,1,2,3,8,9,10,11) |

## Headline P/C at SNR>2 (lyb_veto, NHI>20.3, BAL excluded)

| P_DLA cut | v3_loa124 (baseline) | **v3_loa0 (this run)** | v3_loa0 - v3_loa124 |
|---:|---:|---:|---:|
| >= 0.99    | 84.52 / 76.61 | **79.70 / 78.07** | -4.82 P / +1.46 C |
| >= 0.999   | 85.47 / 73.98 | **80.76 / 74.85** | -4.71 P / +0.87 C |
| >= 0.99999 | 88.72 / 69.01 | **83.33 / 67.25** | -5.39 P / -1.76 C |

Catalog sizes (post all cuts): v3_loa0 cat=1295 vs v3_loa124 cat=1242 (truth=618 same).

## Interpretation

**loa0 != loa124 on London.** v3_loa0 loses 4.7-5.4 pp purity at every threshold
relative to v3_loa124, with only small (+0.9-1.5 pp at the looser cuts) or
negative (-1.8 pp at 0.99999) completeness gains. The v3 architectural win on
London therefore is **NOT** purely an architecture effect; the **loa124 training
data composition (HCD/BAL exclusion on the loa124 subset) materially matters**.

Mechanistically: the larger v3_loa0 catalog (1295 vs 1242) means loa0 is more
permissive — more candidates pass the P_DLA threshold, but more of those extra
candidates are false positives (purity drops more than completeness rises). This
is consistent with loa0 having been trained on a less filtered subset and
therefore having a less discriminating null-vs-DLA likelihood ratio.

## Files

- Per-file dlacat FITS: `dlacat-v5.9.5-mockcat-{0..7}-{1..8}.fits`
- Per-spectrum h5: `processed/processed-spectra-16-{0,1,2,3,8,9,10,11}.h5`
- Logs: `logs/local_{0..7}_{1..8}.log`
- RUN_SETTINGS.md: full resolved param table
- Molly results: `/pscratch/sd/j/jibancat/prod533_5k_20260511/molly/london_v3_loa0_pw14_tau_eb_8f_gpconf{0.99,0.999,0.99999}/`
- Molly log: `/pscratch/sd/j/jibancat/prod533_5k_20260511/molly_v3_loa0.log`
