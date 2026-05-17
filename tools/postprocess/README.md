# tools/postprocess/

Postprocessing utilities that turn raw inference-time `dlacat-*.fits` into
**production-ready DLA catalogs**. The canonical deliverable for
downstream science (dN/dX, f(N,z), Ω_HI, cross-correlations, …) is the
flagged dlacat — the per-spectrum HDF5 sample files in `processed/` are
intermediate products needed only for full Bayesian CDDF analysis.

## TL;DR for downstream consumers

```python
import fitsio
cat = fitsio.read("dlacat-*.fits", ext=1)

# "Clean" production filter — passed every inference + postprocess check
clean = cat[cat["DLAFLAG"] == 0]
```

That's it. `DLAFLAG == 0` is the recommended filter for population
statistics. For finer-grained filtering, see "Bitmask layout" below.

## Files

| File | Purpose |
|---|---|
| `add_dla_flags.py` | Adds the postprocess flag columns + folds them into `DLAFLAG`. Runs in place; idempotent. Designed to be the LAST step of the inference pipeline. |
| (this README) | Schema doc + recommended downstream usage |

## End-to-end pipeline

```
inference (slurm/run_local.sh)
   → produces dlacat-*.fits (per-spectrum-16-file)
   → produces processed/processed-spectra-16-*.h5 (per-spec, for CDDF)

postprocess (tools/postprocess/add_dla_flags.py)
   → adds boolean flag columns to dlacat-*.fits in place
   → folds quality flags into DLAFLAG bitmask

distribution (combine_processed_h5.py for CDDF; or ship dlacat as-is)
   → downstream science reads dlacat-*.fits and filters DLAFLAG==0
```

The postprocess step is mandatory for any catalog labeled "production".
Inference output without postprocess is intermediate.

## Usage

```bash
python tools/postprocess/add_dla_flags.py \
    --catalog-dir /pscratch/.../runs/cellC_C7/ \
    --bal-cat /global/cfs/.../jura-124/bal_cat.fits

# Tune the NHI consistency gate (default k=0.5):
python tools/postprocess/add_dla_flags.py \
    --catalog-dir <DIR> --bal-cat <BAL> \
    --nhi-consistency-k 0.0          # disable the gate
    --nhi-consistency-k 1.0          # stricter (drops ~more rows)

# Skip a flag entirely:
python tools/postprocess/add_dla_flags.py \
    --catalog-dir <DIR> --no-bal-flag --no-nhi-consistency
```

`--catalog-dir` is the run's output directory (containing `dlacat-*.fits`).
Run again at any time to refresh: previous postprocess bits on `DLAFLAG`
are cleared before re-setting (idempotent).

## Bitmask layout (DLAFLAG)

`DLAFLAG` is a bitmask. Bit definitions live in
[`fitwarning.py`](../../fitwarning.py) and are reproduced here for convenience.

| Bit | Constant | Set by | Meaning |
|---:|---|---|---|
| 0 (= 1) | `POTENTIAL_BAL` | `dlasearch.py` | DLA candidate's λ_DLA falls inside a BAL absorption window — likely false positive (inference-time geometric check) |
| 1 (= 2) | `BAD_ZFIT` | `dlasearch.py` | bad parabola fit to χ² as a function of refined z; also raised on `np.linalg.LinAlgError` during processing |
| 2 (= 4) | `BAD_NHIFIT` | `dlasearch.py` | bad parabola fit to χ² as a function of refined NHI; also raised on All-NaN slice |
| 3 (= 8) | `LYBETA_MISID` | postprocess | DLA is a likely Lyβ misidentification of a higher-z, higher-NHI DLA on the same LOS (`gpy_dla_detection.postprocess.lyb_veto`) |
| 4 (= 16) | `BAL_CAT_OVERLAP` | postprocess | TARGETID is in the mock/real `bal_cat.fits`. This is the strict "drop-all-bal_cat-TIDs" filter, not the inference-time `POTENTIAL_BAL` geometric check |
| 5 (= 32) | `NHI_INCONSISTENT` | postprocess | `NHI − k · NHI_ERR < 20.3` (default k=0.5). Lower 1σ of the NHI estimate falls below the canonical 20.3 catalog floor — DLA is not robustly above the floor |

### Quick filter recipes

```python
# "Clean" production catalog (recommended default for science)
clean = cat[cat["DLAFLAG"] == 0]

# Inference warnings only (ignore postprocess flags — pre-2026-05-15 default)
inference_clean = cat[(cat["DLAFLAG"] & 0x07) == 0]

# Postprocess flags only (ignore inference warnings)
post_clean = cat[(cat["DLAFLAG"] & 0x38) == 0]

# Per-flag filters using the boolean columns:
no_lyb = cat[~cat["LYBETA_FLAG"]]
no_bal = cat[~cat["BAL_FLAG"]]
nhi_robust = cat[~cat["NHI_CONSISTENCY_FLAG"]]
```

## Column inventory after postprocess

In addition to the 17 inference-output columns
(`TARGETID, RA, DEC, Z_QSO, SNR_FOREST, SNR_REDSIDE, DLAID, Z_DLA,
Z_DLA_ERR, NHI, NHI_ERR, DLAFLAG, P_DLA, P_NULL, LOGP_DLA, LOGP_NULL,
MODEL_P`), the following are added:

| Column | Type | Meaning |
|---|---|---|
| `LYBETA_FLAG` | bool | True ⇒ flagged by `flag_lybeta` |
| `LYBETA_PARENT_TID` | int64 | TARGETID of the higher-z parent (or `-1` if no flag) |
| `LYBETA_PARENT_Z` | float | z of the parent (or NaN) |
| `BAL_FLAG` | bool | True ⇒ TARGETID ∈ `bal_cat.fits` |
| `NHI_CONSISTENCY_FLAG` | bool | True ⇒ NHI − k · NHI_ERR < 20.3 |
| `PDLA_SATURATED_FLAG` | bool | True ⇒ P_DLA ≥ 1 − 1e-7 (informational, NOT in DLAFLAG) |

`PDLA_SATURATED_FLAG` marks rows where the p_DLA threshold becomes a
no-op (high-confidence detection at the saturated end of the
distribution). Useful for diagnosing the operating-point sensitivity but
is **not** a quality warning — keep these rows.

## Recommended downstream workflow

```python
import fitsio
import numpy as np

cat = fitsio.read("dlacat-v5.9.5-mockcat-0-1.fits", ext=1)

# Production filter
clean = cat[cat["DLAFLAG"] == 0]

# Apply the canonical headline cuts (matches molly's notebook, post-2026-05-15)
sel = (
    (clean["P_DLA"] >= 0.99) &
    (clean["NHI"] >= 20.3) &
    (clean["SNR_REDSIDE"] > 2)
)
production_dlas = clean[sel]
```

This is equivalent to running `examples/molly_faithful_pc_plots.py` with
the canonical recipe (no-BAL, lyb-veto on, NHI ≥ 20.3, SNR > 2,
p_DLA ≥ 0.99) — the eval script's filters are duplicates of the DLAFLAG
bits when run on a postprocessed catalog.

## Versioning

| Date | Change |
|---|---|
| 2026-05-15 | Initial 5-flag schema (LYBETA, BAL, NHI_CONSISTENCY, PDLA_SATURATED + DLAFLAG bitmask renumbering: bits 0-2 = inference, 3-5 = postprocess). Pre-2026-05-15 dlacats had only inference DLAFLAG with bits 3-5 reserved (`POTENTIAL_BAL`, `BAD_ZFIT`, `BAD_NHIFIT`); the renumber consolidates and removes the 3 unused template-fit bits. |

## See also

- [`fitwarning.py`](../../fitwarning.py) — bitmask source of truth
- [`gpy_dla_detection/postprocess/lyb_veto.py`](../../gpy_dla_detection/postprocess/lyb_veto.py)
- [`examples/molly_faithful_pc_plots.py`](../../examples/molly_faithful_pc_plots.py) — eval recipe
- [`docs/notes/2026-05-15_molly_eval_recipe_fix.md`](../../docs/notes/2026-05-15_molly_eval_recipe_fix.md) — eval recipe history
- [`docs/production_runbook.md`](../../docs/production_runbook.md) — full pipeline runbook
