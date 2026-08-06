# FROZEN Stage-2 executable-state specification (rulings §4)

**This document IS the freeze.** The frozen state = the tree at the
LATEST commit touching this file (introduced at `5ee7202`; pre-launch
amendments update it — see the amendment log at the end; post-launch
amendments follow the quarantine protocol).
Every Stage-2 production job must launch from exactly that commit; every
production artifact stamps `git rev-parse HEAD` and it must match.
Behavior-changing corrections after launch follow the rulings' §4
protocol (defect report → quarantine of affected outputs → new freeze
commit → rerun); outputs from behaviorally different states are never
combined into one confirmatory estimator. Documentation-only commits do
not move the freeze but production jobs still launch from the freeze
commit itself.

## Code and branch

* Repository: `github.com/jibanCat/desi_gpy_dla_detection`, branch
  `calibration/phaseC-highN-fp-2026-08-06`; parent of the freeze commit:
  `99fb62e` (pre-production code complete). Working tree at freeze:
  clean except this document and its lockfiles.
* **Finder tree (sbatch `--chdir`)**: the primary worktree
  `/home/mfho/desi_gpy_dla_detection` @ `1533333` (`lls-subdla-cddf`),
  clean. Behavioral identity with this branch verified: the diff outside
  the unused `gpy_dla_detection/lls/` submodule is 3 docstring lines in
  `generate_samples.py`. The finder chain (`desi-DLAGP.py`,
  `run_bayes_select.py`, `dla_gp.py`) is byte-untouched by Phase C.
* Compiled artifact: `gpy_dla_detection/_voigt.so` sha256
  `2181f06c…96c0782` (copied from the primary tree; identical file used
  by the finder tree).

## Frozen analysis/config artifacts (sha256)

| artifact | sha256 |
|---|---|
| `docs/PHASEC_CALIB_DESIGN.md` | `02993342…a56794f5` |
| `docs/PHASEC_BRIDGE_DESIGN.md` (incl. the ratified amendment) | `39234776…36db9faa` |
| `diagnostics_phaseC/design_sizing/sizing.json` | `5682b527…f94c9779` |
| `diagnostics_phaseC/preimage/preimage.json` | `cfee1319…fe84e026` |
| `injection/gen_phaseC_resp.py` | `ac3560de…ea396b5c` |
| `injection/measure_phaseC_pairs.py` | `f3960f08…2e9af578` |
| `gpy_dla_detection/inject_absorber.py` | `4305c6a5…f63ff9e85` |
| `examples/molly_faithful_pc_plots.py` (THE matcher) | `881d8976…acd64cd1` |
| `slurm/greatlakes/production/phaseC_resp_gl_v1.env` | `e6e423e2…5ac73afd` |
| old response envelope `forward_response_2lpt0.npz` (scratch) | `def83ac4…c0a44e1b` |

## Estimand-defining versions and policies

* Finder configuration: the production chain
  `phaseC_resp_gl_v1.env → loa0_fp_gl_v1.env → 2lpt0_gl_v1.env →
  _base_gl.env` (learned model `2lpt_loa124_nohcd_nobal_wide_m`,
  MAX_DLAS=4, SINGLE_ABSORBER_MODEL=1, FILTER_LOW_LIKELIHOOD=1,
  NUM_FOREST_LINES=31, MIN_Z_SEPARATION=3000, MAX_LAMBDA=1250, …) —
  inherited unchanged; see the env file.
* Matcher: `match_truth_to_cat_molly`, `dz_rel = 0.01`, iteration
  `nhi_desc`, tie-break min|ΔNHI| — the production object.
* Multiple-candidate policy: greedy 1-to-1 with the above ordering;
  multi-candidate rates reported per cell (a bridge dimension).
* Op-mask for scoring: P_DLA > 0.99 strict, SNR_REDSIDE > 2 strict,
  DLAFLAG == 0, sentinel rows dropped pre-match, z_qso ∈ (2, 4.25),
  analysis window λ_rf ∈ [1025, 1216] with the 3000 km/s collar and
  3600 Å floor (the `make_lambda_z_BAL_cuts` direct geometry).
* Truth generation: `inject_absorber.inject_voigt` (round-trip
  validated), one absorber per sightline, `num_lines = 31`;
  substrate loa-124 (2LPT-0) with BAL veto; prodlike truth-HCD
  z-neighbor exclusion 5,000 km/s with redraw.
* Substrate catalogs (mock-0/loa-124): zcat.fits 121,328,640 B;
  snr_cat.fits 29,125,440 B; bal_cat.fits sha256 `5de118b6…806efd82`;
  hcd_truth_cat.fits sha256 `701c4422…43f3bf9`.
* Response/support schema: envelope `forward_response_*_phaseC.npz`
  v1 (extends `skewnormal_per_cell` with anchor sets, roles, support
  labels per PHASEC_CALIB_DESIGN §6); roles sidecar `phaseC_roles/v2`;
  pairs artifact `phaseC_pairs/v1`.

## Seeds and job mapping

* Production prodlike arm: generation seed **20260810**; holdout
  assignment seed = 20260810 + 777 (in-code); arm root
  `/scratch/cavestru_root/cavestru0/mfho/phaseC_resp/prod_v1`.
* Environment-probe (clean) arm: seed **20260811**, probe-scale 0.10,
  root `…/phaseC_resp/prod_env_probe_v1`.
* Substrate restriction: `--n-healpix 48` (storage budget ≈ 7 GB).
* Job mapping: one `launch_gl.sh` invocation per arm via
  `phaseC_resp_gl_v1.env` (`PHASEC_ARM` env), one sbatch per arm,
  positional window over the arm's healpix files, STEP=2 inner tiling;
  account cavestru1, partition standard, 12 h limit.
* Scoring is deterministic (no seeds).

## Test manifest at freeze (release gate)

The scientific release gate = these 8 suites in env `gpdla-hbi`:
`test_modelA_rungs, test_modelA_vs_legacy, test_modelA_pack,
test_gate_covariance, test_adopted_reporting, test_matching_contract,
test_modelA_forward_selftest, test_hbi_mcmc_toys` — **331 passed,
1 skipped** (the approved r5 release-cadence skip), 438 s, at the freeze
parent `99fb62e`. Per the rulings §9 the historical "939/1" aggregate is
retired as a baseline claim. Environment lockfiles:
`docs/env_locks/gpdla_pip_freeze_2026-08-06.txt` (171 pkgs),
`docs/env_locks/gpdla-hbi_pip_freeze_2026-08-06.txt` (177 pkgs);
python 3.11 conda envs at `/home/mfho/.conda/envs/{gpdla,gpdla-hbi}`.

## Amendment log (pre-launch only)

* A1 (pre-launch, NO production outputs existed): the scorer's
  per-anchor aggregation regrouped from exact-z keys to (logN anchor ×
  response z-cell × SNR stratum) with per-pair healpix carried for the
  bridge Ĉ_shared split (`phaseC_pairs/v2`) — required for continuous
  dX-drawn production z; the exact-z grouping would have fragmented
  production cells into singletons. Hash updated above; the pilot JSONs
  remain the committed v1 engineering record.
* A2 (pre-ANALYSIS; the GP jobs were launched but no output had been
  read): the bridge-statistic implementation
  `injection/build_phaseC_response.py` sha256
  `0bed6665…17c562023` — implements the FROZEN §4 criteria numbers
  verbatim (75 / 116.7 / z<3 / |z|<4 / 3σ completeness / LOF p≥0.01 /
  shared-inflation ratio 1.5), the old-side comparison at the deployed
  clamped covariate, the union refit, and the quarantine path. The GP
  jobs' behavior is untouched (analysis-side only). One recorded
  operationalization: the old envelope stores no per-anchor
  multi-candidate record, so criterion 3's multi-candidate clause tests
  new-side cells against the pooled new-side rate (rates reported raw).

## Budget manifest

The authorized envelope and itemized breakdown live in
`docs/PHASEC_HANDOFF.md` ("STAGE-2 AUTHORIZATION + BUDGET BREAKDOWN"):
hard ceiling 1,850 CPU-h; Stage-2A sub-ceiling 110 CPU-h; actual spend
appended there per campaign from sacct.
