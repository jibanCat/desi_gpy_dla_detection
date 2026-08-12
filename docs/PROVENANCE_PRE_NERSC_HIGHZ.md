# Pre-NERSC high-z provenance record (2026-08-12; PI ruling §11/§17)

## Historical production finder (reference states)

Both real LOA productions ran from branch `nersc_production`, clean trees,
2026-06; both used the same GP model artifact and the same finder code
(the finder-path diff between the two commits is empty).

| | headline `loa_main_dark_v1` | CDDF `loa_cddf_main_dark_v1` |
|---|---|---|
| launch date | 2026-06-06 | 2026-06-09 |
| commit | `84fa6542f27604ce66404ac96c065460a88b3dca` | `bc6c4cb4f88fef0e11c30383a53de99c4643a2aa` |
| config | `loa_nersc_v1.env` (MAX_DLAS=4, FILTER=1, PW50k) | `loa_cddf_nersc.env` (MAX_DLAS=1, SINGLE_ABSORBER, FILTER=0, PW100k) |
| entry point | `submit_desi_loa_nersc.sh` → `desi-DLAGP.py` → `dlasearch.dlasearch_hpx` → `process_spectra_group` | same |
| environment | `desi_environment.sh main` (NERSC) | same |
| batches | `nersc_prod_loa_v1_20260606` | `nersc_cddf_loa_v1_20260609` (22/22 jobs, `cddf_job_ids.txt`) |
| QSO input | `QSO_cat_loa_main_dark_healpix_v2-altbal-20241115.fits`, z ∈ (2.0, 4.25) via `constants.py` | same |
| GP model | `2lpt_loa124_nohcd_nobal_wide_m/phase2_result.h5` | same |
| model sha256 (GL DEPLOYED copy) | `5e7a2691613f69da38ec828235d94a360ce6772363067a1bee017caecf232856` | same |

Model-hash caveat: the sha256 above is of the Great Lakes
`DEPLOYED_phase2_…` copy; the NERSC CFS copy is asserted identical by the
deployment convention and MUST be re-hashed by the pilot job before any
high-z inference (the pilot script logs `sha256sum $LEARNED_FILE` and
compares against this value).

## Proposed high-z finder (forward execution state)

* branch `hbi/forward-2026-08`, provenance base `prov/p1-refold-2026-08-08`
  (`3a65e2a`), which **contains `bc6c4cb` in its ancestry** (verified:
  `merge-base bc6c4cb 3a65e2a == bc6c4cb`).
* Same config files; the high-z run uses `loa_cddf_hz_v1.env`, which sources
  the frozen CDDF config unchanged and sets only
  `GPDLA_ZMIN_QSO=4.25 / GPDLA_ZMAX_QSO=7.0` (input population).
* Same model artifact (hash above; re-verified at job start).

## Finder-change audit (bc6c4cb → forward base 3a65e2a), classification

Complete diff of the finder path (`gpy_dla_detection/` incl. `lls/` and
`training/`, `dlasearch.py`, `constants.py`, `run_bayes_select.py`,
`desi-DLAGP.py`):

| change | class | note |
|---|---|---|
| `gpy_dla_detection/inject_absorber.py` NEW (+243) | P1 | never imported by any finder module (verified); injection tooling |
| `constants.py` comment rewrite at `zmax_qso` | P0 | value unchanged; comment only |
| `dlasearch.py` `np.in1d → np.isin` | P2 | **mock path only** (`dlasearch_mock`); the real healpix path never executes it; documented numpy-2.0 drop-in |
| `run_bayes_select.py` +4 lines | P1 | additive HDF5 output attr `filter_low_likelihood`; no consumed field changed |
| `desi-DLAGP.py` `--external_tid_list` (+32) | P1 | additive flag, default off, mock mode only |
| `submit_desi_loa_nersc.sh` exit-code propagation | P1 | engineering |
| `submit_desi_loa_nersc.sh` `--pixel_col` | **P1 defect, FIXED on this branch** (`4fafe51`) | the flag had no consumer anywhere (would crash argparse); now passed only when `PIXEL_COL` is set → LOA invocation byte-identical to historical |
| `loa_hpx_spec_counts.txt` removal + balancing tools | P1 | task load-balance only; per-spectrum outputs unaffected |
| forward-branch additions (`047acec`…) | P1 | env-overridable z window (defaults = historical), high-z tooling, diagnostics; none imported by the finder |

No P3 (scientific finder) change exists on this path.

## Regression test (PI §9; required because P1/P2 changes exist)

Forward-state finder (worktree at `3a65e2a`) run on real LOA spectra from
the local mirror, headline configuration (the config with locally stored
historical outputs), BLAS single-threaded; compared per-TARGETID against
the historical NERSC production files
`loa_main_dark_v1/processed/processed-main-dark-{24,147,150}.h5`
(90 sightlines: z_qso 2.0–4.15, low/high SNR, DLA/no-DLA, 10 multi-absorber
cases, low/high P_DLA). Tolerances pre-declared in the committed runner
(`finder_regression.py`): integers/flags/grid endpoints and MAP grid picks
EXACT; probabilities |Δ| ≤ 1e-6; log-likelihood family rel ≤ 1e-6;
error fields rel ≤ 1e-5.

**RESULT — three-way decomposition (artifacts at
`/scratch/cavestru_root/cavestru0/mfho/fable_reg_2026_08_12/`):**

1. **Forward code vs historical code, IDENTICAL GL environment (the code
   question):** the deterministic finder core is **BITWISE-IDENTICAL** —
   null-model likelihood chain exact, and the complete level-0
   sample-likelihood scan (all 50k samples × 90 sightlines) exact to the
   bit between `bc6c4cb` and `3a65e2a` (`ab_summary.json`, k0 max|d| = 0).
   All remaining differences live at multi-DLA levels k ≥ 1 and trace to
   `dla_gp.py:156 np.random.rand` — **unseeded stochastic base-sample
   resampling that is a pre-existing property of the historical production
   itself** (identical code in both commits; two runs of EITHER commit
   differ the same way). It is not a code change.
   CDDF-configuration A/B (MAX_DLAS=1, FILTER=0, PW100k — the exact
   Paper-1 production config, which exercises only the deterministic
   level): **ALL FIELDS BITWISE-EXACT on all 90 sightlines** (`ab_cddfcfg_summary.json`; jobs 57223976/57223977).
2. **Same code, GL environment vs the stored NERSC production outputs
   (the platform question):** the null chain is machine-exact (1e-14) but
   every DLA-branch per-sample likelihood shifts (median |ΔlogL| ≈ 0.1,
   same-index correlation 1.0000 — same sample grid, shifted absorption-
   profile values), giving |Δ p_DLA| ≤ 0.086 and occasional MAP grid-step
   moves. Attribution: the absorption-profile branch (`scipy.special.wofz`
   path) under a different scipy/numpy generation; the GL `gpdla` env is
   an unsupported pairing (scipy 1.14.1 vs numpy 2.4.4). **Consequence:
   finder outputs are environment-sensitive at this level; the high-z run
   must execute at NERSC in the recorded production environment, and the
   environment must be version-pinned and logged** (the June productions
   recorded only `desi_environment.sh main`).
3. **Run-to-run (the determinism question):** single-absorber/level-0
   quantities are deterministic; multi-DLA k ≥ 1 quantities carry inherent
   unseeded-RNG variance in ALL productions to date (documented here;
   headline-catalog property, not a Paper-1/CDDF-run property).

## Production-finder invariance gate (PI §10)

**PASS-B for the intended high-z execution**, on these grounds: the
forward-branch finder is bitwise output-identical to the historical
production code in a like-for-like environment for the Paper-1-relevant
(single-absorber/CDDF) configuration and for the deterministic level of
the headline configuration; every P1/P2 item is enumerated above; no P3
exists. The cross-environment sensitivity (2.) and the pre-existing k ≥ 1
stochasticity (3.) are properties of platform and of the historical
pipeline respectively, not of the audited change set — both are disclosed
and both are neutralized for the high-z run by executing at NERSC with the
pinned environment and (for CDDF purposes) the single-absorber
configuration.

## Identity statement (PI §11)

**OUTPUT-EQUIVALENT I/O-ADAPTED FINDER** — bitwise-equivalent scientific
core (level-0/single-absorber proven; P1-only additions elsewhere), with
the execution-environment condition stated above as a binding requirement
of the high-z run.
