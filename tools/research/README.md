# tools/research/

One-off investigation scripts used during specific analysis sessions. These are
checked in for reproducibility of session findings (see `docs/notes/`) but are
**not** part of the production pipeline. Treat them as supporting evidence for
the notes, not as APIs.

Each script is self-contained — it hard-codes the dataset/output paths it was
written for. Adapt those paths if rerunning on a different mock.

## Inventory

| Script | Session | Purpose |
|---|---|---|
| `test_unified_pdla.py` | 2026-05-12 | Per-spec re-aggregation of p_DLA to include the SubDLA column; verifies the "SubDLA-as-null siphons low-SNR weak-DLA mass" hypothesis. Output: `unified_pdla_test.json`. |
| `test_unified_pdla_perdla.py` | 2026-05-12 | Per-DLA molly-faithful evaluation of baseline vs unified p_DLA aggregation. Produces TWO tables (classical NHI≥20.3, sub-DLA NHI∈[19.1,20.0]) plus three sub-DLA-detector variants. Output: `unified_pdla_perdla.json`. |

## Reproducing the 2026-05-12 SubDLA mechanism finding

This is the canonical use case for `test_unified_pdla_perdla.py`.

**Inputs required** (already in place if you're on NERSC `/pscratch`):

- v3_loa124 London 8f inference: `/pscratch/sd/j/jibancat/prod533_5k_20260511/london_v3_loa124_pw14_tau_eb/processed/processed-spectra-16-*.h5` and `dlacat-*.fits`.
- Truth: `/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/dla_cat.fits`.
- BAL: `bal_cat.fits` in the same dir.

**Run**:

```bash
bash -c '
source /usr/share/lmod/lmod/init/bash
export DESI_ROOT=/global/cfs/cdirs/desi
source /global/common/software/desi/desi_environment.sh main
cd /pscratch/sd/j/jibancat/desi_gpy_dla_detection
python tools/research/test_unified_pdla_perdla.py
'
```

**Expected output** (per-DLA, BAL-excl, lya_lyb [911, 1216], SNR>2):

- Table 1 Classical DLA (truth NHI≥20.3, n_truth=322): baseline @ P_DLA≥0.99 → P=84.6% / C=83.5%; unified @ P_DLA≥0.99 → P=81.5% / C=86.3%.
- Table 2 Sub-DLA via DLA-model MAP (truth & pred ∈ [19, 20.3), n_truth=347): max P ≈ 19%, C ≈ 37% — **the wrong detector**, included to show that dropping SubDLA + extending DLA prior is worse than the dedicated SubDLA model.
- Table 3 Sub-DLA via P(SubDLA|D) per-spec (truth ∈ [19.1, 20.0), n_truth_pure=384): variants A/B/C identical (posteriors are normalized). Best operating point P=78% / C=4% at thr ≥ 0.99, or P=59% / C=57% at thr ≥ 0.5.

Numbers match `docs/notes/2026-05-12_mlmc_design.md` "Two-target validation" section.

## Adapting to a new run

Edit the constants at the top of each script:

```python
CAT_DIR   = "/pscratch/sd/j/jibancat/<run-name>"
TRUTH_FN  = "<path-to-dla_cat-or-hcd_truth.fits>"
BAL_FN    = "<path-to-bal_cat.fits>"
```

For Saclay mock-0, the inputs are at
`/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124/`
and the truth catalog is named `hcd_truth_cat.fits` instead of `dla_cat.fits`.

## Why these aren't in `examples/`

The scripts in `examples/` are the canonical, reusable evaluators
(molly-faithful, gp-native) maintained for production use. The scripts here
encode specific *experiments* with hard-coded paths and one-off analysis
choices. Promoting one to `examples/` requires generalizing the inputs to CLI
flags + adding a docstring + handling edge cases.
