# REAL-DATA PRODUCTION — REPRODUCIBILITY / ENVIRONMENT PRESCRIPTION
(PI ruling 2026-08-17 item 5; provisional pending the final checkpoint)

## Why this exists (measured motivation)

The Saclay A/B replication arm showed that rerunning the JUNE production
configuration in TODAY'S `gpdla` env moves detection probabilities by
dP rms = 0.011 at the acceptance margin — LARGER than the 50k->100k QMC
sample-count effect on matched detections. Population-projected, the
drift is confined outside the measured domain (net 0.00% at x-hat >=
19.5; -0.24% at >= 20.3, inside the transport scatter), but it is real:
environments drift, and un-pinned reruns are not reproducible at the
P_DLA-margin level. (Signature: numpy 2.4.4 with scipy 1.14.1 — a
version pairing scipy itself warns about.)

## The prescription (all items required before any real-data finder run)

1. **Frozen environment**: build a DEDICATED env from the lockfiles
   committed alongside this doc —
   `slurm/greatlakes/production/env_lock_gpdla_2026-08-17.txt` (+ the
   pip freeze `env_pip_gpdla_2026-08-17.txt`; `gpdla-hbi` pair for the
   fold/inference side) — and never update it for the lifetime of the
   production. Record `conda list --explicit` + `pip freeze` + python
   version in the run's BASELINE.env directory at launch.
2. **libcerf**: pin the runtime library
   (`~/.local/usr/local/lib64/libcerfcpp.so`, sha256
   `4bdb51d8...101435a7`) via LD_LIBRARY_PATH exactly as the inner
   scripts do; record the sha in BASELINE.env.
3. **Samples**: harmonized 100k QMC files —
   `pw_samples_a3_172_225_100000.mat` + `subdla_samples_a03_191_200_
   100000.mat` (NUM_*_SAMPLES=100000) — the configuration of the
   calibration family and the hz/BH production.
4. **Code**: a single provenance-tagged commit; CODE_DIRTY must be
   clean; BASELINE.env records the commit (the existing launcher
   behavior).
5. **Operator/pack path**: the pack is built by the v1.2 path (frozen
   response byte-identical; adopted response + carrier + phi_ref +
   TP/contract stamps) and `contract_guards_check` must PASS (G-A,
   G-CC, G-C hard; G-B per its PI ruling) BEFORE any read; the fold
   side uses `cc_fold_adopted` exclusively (fail-closed).
6. **Calibration-transfer bridge**: because the mock calibration
   catalogs were produced in the June-era env, any real-data run in the
   frozen env carries a calibration-transfer term. Its scale is
   MEASURED (this session's bridge): <= 0.25% on >= 20.3, 0.0% net in
   the measured domain — carried as a named systematic; a pinned-subset
   bridge rerun on the loa mock can tighten it if the PI wants.
7. **Known-drift ledger**: the unseeded k>=2 resampling
   (dla_gp.py) means catalog-level reruns are never bitwise even in a
   pinned env; acceptance-level reproducibility is the standard
   (measured above).

## Explicitly NOT authorized by this doc

Running the real-data science posterior; promoting any provisional
science choice; modifying H2/BH; any K freedom chosen from closure.


## Addendum 2026-08-26 (Paper-1 code review)
- Which lock is authoritative for which env: **gpdla-hbi → `env_lock_gpdla-hbi_2026-08-17.txt`** (explicit conda; the pip freeze of that env contains 90 `file:///` entries and is not installable); **gpdla → `env_pip_gpdla_2026-08-17.txt`** on the python of `env_lock_gpdla_2026-08-17.txt` (the explicit lock is a skeleton by construction: the scientific packages are pip-installed). Re-exported 2026-08-26: gpdla-hbi identical.
- Item 5 ("G-A hard"): on a REAL pack the truth-point partition is undefined (all-zero truth sentinel) and `contract_guards_check` now reports `NOT_APPLICABLE_REAL_PACK`; the guard of record on real data is `G_A_real_mode` evaluated inside `cc_real_posterior` (PASS on every frozen run). The frozen `.contract_guards.json` predates this and records FAIL for that reason.
- The paper-repo environment is now pinned too: `gp_dla_desi_y3/paper_figures/ENV_LOCK_2026-08-26.txt`.
- "Provisional" status: the prescription was applied unchanged through the 2026-08-26 freeze; lifting the word is a PI action.
