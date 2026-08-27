# adopted_response — the builder of record for `adopted_response_v1p1.npz` (recovered 2026-08-26, PI ruling 4)

The adopted response operator (`track_c/stage0/adopted_response_v1p1.npz`, sha256 `8fb580b5…`) was built on 2026-08-16 by a
chain of session scratch scripts that were never committed. They were recovered verbatim on 2026-08-26 from the session
transcript and cross-checked byte-for-byte against the copies the session had committed to the private notes repo
(`figures/2026-08-17_stilt_diag/inputs/pi_diag/`); `RECOVERY_PROVENANCE.md` records every line offset, timestamp,
command and output. The scripts here are **byte-identical to the recovered originals** (`RECOVERY_SHA256SUMS.txt`) —
they are the analysis of record, not a re-implementation. `build_adopted_response.py` is the committed driver that
runs the chain exactly as `chain_recert.sh` did on 2026-08-16 and assembles the operator exactly as the assembly
heredoc did.

Chain (env `gpdla`, except `gb_audit.py` in `gpdla-hbi`):
1. `stage1b_events_full.py` — one catalogue load under the CURRENT `load_and_cut_catalog` (hierarchical tilt match since
   `0ecfeea`), op-cut detections with truth/tilt-host columns → `events_full.npz` (cells and `N_ref` from the frozen
   `forward_response_2lpt0.npz`).
2. `run_d2b.py` (+ `fitlib.py`, `run_d2b_lib.py`) — per-cell deg-2 untruncated skew-normal ML moment surfaces on 0.1-dex
   sub-bins (n ≥ 50), plus ONE shared cubic across the 9 cells (2 refit iterations); CV diagnostics → `d2b_variants.npz`
   (`ml_shared3__mu/sig/skew/rng`, `N_ref`) and `d2b_results.json`.
3. `v1_logo.py` — leave-one-group-out validation (15 folds) → `v1_logo_results.json`.
4. `boot_carrier.py` — 96-draw sightline multinomial bootstrap carrier (seed0 20260818, `default_rng(seed0+r)`), frozen anchor
   set, unit-weight gate (≤ 5e-3) → `adopted_carrier_ensemble.npz`.
5. `gb_audit.py` (gpdla-hbi) — G-B integer-exact / zero-mismatch audit of the kernel population.
6. assembly → `adopted_response_v1p1.npz` (schema `adopted_response/v1.1`).

Inputs (all frozen, hashed in `docs/PAPER1_FROZEN_MANIFEST.json` or on Turbo): the 2LPT-0 combined catalogue,
`hcd_truth_cat.fits`, `bal_cat.fits`, `snr_cat.fits`/`zcat.fits`, `figures_molly_nhi195/lya_only/molly_matrix.tsv`,
`forward_response_2lpt0.npz`. Outputs: see `docs/PAPER1_PROVENANCE_DAG.md` §7.4. Reproduction status: matrix §7.

Packaging note: `stage1b_events_full.py` hard-codes `REPO=/home/mfho/wt_forward_2026_08` and the DESI artifact paths
(`track_c_tf_loa._C0_*`, `ab_loa0_fp_baseline._resolve_molly`) — DESI policy, listed among the packaging blockers in
`docs/HBI_ARCHITECTURE.md`. The driver copies the scripts into the work dir so that their `HERE`-relative outputs never
land in the package directory.
