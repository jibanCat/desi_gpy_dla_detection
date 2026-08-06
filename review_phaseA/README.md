# Phase-A adversarial review — 2026-08-05 (fresh session)

REVIEW-ONLY branch. Nothing under `review_phaseA/` alters production behavior.

## Review isolation record (per protocol §4.1)

- Review branch: `review/phaseA-adversarial-2026-08-05`
- Root SHA: `9d73365fa0396e3f911e1fdea0a47f72f21a2a17` (= reviewed tip of
  `hbi-mcmc-threeroute`; original branch preserved unchanged)
- Worktree: `/home/mfho/wt_review_phaseA`
- Initial tree status at creation: clean (0 modified, 0 untracked)
- Guard lineage under review: `lls-subdla-cddf` @ `1533333` (untouched)
- Notes repo at review start: `c7f1a0c`
- Probe archive: `/home/mfho/slurm_log_archive/session_2026-08-05_probe_code_and_outputs.tar.gz`
  (extracted read-only into the session scratchpad; 59 scripts, 42 outputs)
- Environments: `gpdla-hbi` (jax, x64) / `gpdla`; `OMP/OPENBLAS/MKL_NUM_THREADS=1`
- Packs: `/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/PRESERVED_2026-07-28_small_artifacts/modelA_packs/modelA_pack_{2lpt0,london0,saclay0}_v11.npz`

## Contents

| path | track |
|---|---|
| `stale_claims_inventory.md` | where retracted/superseded claims still stand |
| `mock_transfer_audit.md` | what London-0 / Saclay-0 genuinely validate (verdict: prediction transfer only) |
| `fp_normalization/` | FP `ell_eff` defect + repair, independent dimensional review |
| `dev41_null/` | parametric-bootstrap null calibration of the Δdev=41 claim |
| `geometry/` | principal-angle / prior-curvature identifiability review |
| `snr_residual/` | by_snr residual diagnosis (incl. calibration-noise propagation) |
| `PHASE_A_VERDICT.md` | the frozen verdict (written last, before any Phase-B work) |

## History discipline

This branch preserves the three-way distinction required by the protocol:
1. what the previous session claimed — commits up to and including `9d73365`;
2. what the independent review found — commits on this branch under `review_phaseA/`;
3. what was subsequently repaired — a separate Phase-B branch, if and only if
   the frozen verdict permits it.

No amend, no rebase, no force-push. Corrections to provenance-bearing commits
go in new commits.
