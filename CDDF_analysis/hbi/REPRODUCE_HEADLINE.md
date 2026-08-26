# Reproducing the real-LOA CDDF headline (code walkthrough)

> **This reproduces the feed-forward (loa0 FF) headline, NOT Paper-1's frozen hierarchical measurement** (which is `docs/PAPER1_REPRODUCTION.md`). Its numpy/scipy pins predate `docs/REALDATA_ENV_PRESCRIPTION.md`; `environment-hbi.yml` now exists at the repo root.


**Scope:** re-derive the real-LOA DLA CDDF headline — integrated and per-z dN/dX and Ω_DLA
at log N_HI ≥ 20.3 (and ≥ 20.0) — from committed code. A **code/ops** doc, not a science
doc: it does **not** print real-LOA values (DESI-restricted; they live only in the private
notes repo — see [Verify](#4-verify-you-got-it-right)).

The headline uses the `loa0` false-positive estimator — a **config-only FP variant of
archival production job 52266001** (which ran `purity_mixture`): the *only* difference is
`cfg.fp_estimator`; forward kernel, completeness, catalog, cut bundle and code are identical.
Estimator (`cddf_catalog_hbi.py`) and inference (`gpy_dla_detection/`) are **byte-frozen** —
untouched by the reduction.

---

## 0. TL;DR (this machine, already-staged data)

```bash
cd /home/mfho/desi_gpy_dla_detection
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate gpdla
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE

# (a) loa0 HEADLINE — full schema (per-z + integrated + MC band + f(N) arrays) -> a JSON:
python CDDF_analysis/diagnostics/bal_metal_fp/arbiter/run_loa0_headline_full.py --n-mc 120 --gate
#     prints integrated dN/dX & Omega at >=20.0 and >=20.3; --gate also verifies the
#     purity_mixture single-knob equals archival job 52266001.

# (b) table point + the broad-trough BAL veto cross-check (fast, n_mc small):
python CDDF_analysis/diagnostics/bal_metal_fp/arbiter/apply_broadtrough_veto_headline.py --fp loa0 --n-mc 12
python CDDF_analysis/diagnostics/bal_metal_fp/arbiter/apply_broadtrough_veto_headline.py --fp purity_mixture --n-mc 12
```

Runtime ~50–80 s, ~1 GB RAM, single process/single node. The `--gate` / `--fp
purity_mixture` runs are the **provenance proof** (they must reproduce archival job
52266001 byte-identically). Compare the printed numbers to the committed provenance JSON
(§4).

---

## 1. Environment

`conda activate gpdla`. Verified working set (pin these if you rebuild the env):

| pkg | version |
|---|---|
| python | 3.11.15 |
| numpy | 2.4.4 |
| scipy | 1.17.1 |
| h5py | 3.16.0 |
| fitsio | 1.3.0 |
| astropy | 7.2.0 |

- **Thread pinning is required** — `OMP/OPENBLAS/MKL_NUM_THREADS=1`. More threads are
  *slower* here (BLAS oversubscription on small ops) and are not needed.
- `HDF5_USE_FILE_LOCKING=FALSE` avoids GPFS lock errors.
- The **`gpy_dla_detection/_voigt.so` C extension is NOT needed** on the headline reduce
  path (it imports zero voigt modules) — only *re-running GP inference* to regenerate the
  upstream catalogs needs it.

There is not yet a committed `environment.yml` — export one for a clean rebuild:
`conda env export -n gpdla > environment.yml`.

---

## 2. Data inputs you must stage (the manifest)

**None of these are in the git repo.** The reduction is fully path-driven; the paths are
module constants in the driver/arbiter (`DEF_*` in `ab_loa0_fp_baseline.py`, `_C0_*`/`_LOA_*`
in `track_c_tf_loa.py`, `V2_VAC`/`LOA0_LYAONLY` in the arbiter scripts). On another machine
you must stage all of these and repoint the constants.

### On `/scratch/cavestru_root/cavestru0/mfho/` — GreatLakes GPFS, **auto-purges ~60 days**

| file | size | role | stamped? |
|---|---|---|---|
| `cddf_o3_realdata/track_c/stage0/forward_response_2lpt0.npz` | 72 K | forward-response kernel (the Track-C calibration) | ✗ |
| `gl_prod_2lpt0_v1_20260526/figures_molly_nhi195/lya_only/molly_matrix.tsv` | 512 B | molly C/ρ completeness/purity matrix (lya-only, nhi195) | ✗ |
| `gl_prod_2lpt0_v1_20260526/combined_catalog/dlacat-v2.8.5-mockcat.fits` | 118 M | 2LPT-0 calibration dlacat | ✓ (`BASELINE.env CODE_COMMIT`) |
| `gl_loa0_fp_v1_20260615/outputs/loa0_fp_product_lyaonly1025.npz` | 11 K | **loa0 forest-FP product** (the headline FP) | self-doc, no commit; producer `build_loa0_fp_product.py` is committed |
| `cddf_o3_realdata/track_c/tf_loa/mockdir/{snr_cat,bal_cat}.fits` (+ `zcat` symlink) | 30 M | staged real-LOA pathlength / SNR / BAL | ✗ |

### On `/nfs/turbo/lsa-cavestru/mfho/` — persistent group storage

| file | size | role |
|---|---|---|
| `DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/{hcd_truth_cat,bal_cat,snr_cat,zcat}.fits` | ~210 M | 2LPT-0 truth/BAL/SNR for the frozen calibration |
| `DESI/gpdla_catalogs/loa_main_dark_v1/dlacat-loa-main-dark-v1.fits` | 134 M | **the real-LOA GP-DLA catalog** (headline input) |
| `DESI/loa/QSO_cat_loa_main_dark_healpix_v2-altbal.fits` | 969 M | v2 BAL VAC — **only** the broad-trough veto cross-check needs it; the base loa0/pm headline does **not** |

> ✅ **Persistent backup staged (2026-07-05):** the frozen-calibration set (kernel, molly,
> 2LPT-0 calib dlacat, loa0-FP product, staged mockdir) is copied + sha256-checksummed to
> **`/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/loa_headline_repro_bundle/`** (191 MB;
> see its `README.md` + `MANIFEST.sha256`, which maps each file → the `DEF_*` constant it
> feeds). If scratch purges, stage from there and repoint the `DEF_*`/`LOA0_LYAONLY` constants
> to the bundle. The three large real/mock catalogs (§below) already live persistently on turbo.

---

## 3. Step by step (fresh checkout)

```bash
# 1. code
git clone <repo> && cd desi_gpy_dla_detection && git checkout bal-metal-fp-tests   # or the merged desi_y3

# 2. env (see §1)
conda activate gpdla
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE

# 3. stage the §2 data and repoint the path constants:
#    ab_loa0_fp_baseline.py: DEF_CAT, DEF_TRUTH, DEF_BAL, DEF_LYAONLY_MOLLY, DEF_KERNEL, DEF_LOA0_PRODUCT
#    track_c_tf_loa.py:      _C0_* / _LOA_* module constants
#    arbiter scripts:        V2_VAC, LOA0_LYAONLY, SCRATCH_OUT, dlacat path
#    (a checksum-verified stage step is the recommended future hardening)

# 4. run (see §0). The routine prints integrated dN/dX & Omega and writes the full JSON.
python CDDF_analysis/diagnostics/bal_metal_fp/arbiter/run_loa0_headline_full.py --n-mc 120 --gate --out /path/out.json
```

**What the pipeline does internally** (all inside the byte-frozen estimator, no re-inference):
`build_frozen_calibration` (2LPT-0 forward kernel + z-resolved completeness g(N,z) + molly
C/ρ) → `build_loa_ingredients` (loads the real dlacat, applies cut bundle
`SNR>2 & P_DLA>0.99 & DLAFLAG==0` + lya-only window `lam_rf_min=1025`, builds the ΔX
pathlength) → `cfg.fp_estimator="loa0"` + loa0 FP product → `run_measurement` (MAP
marked-Poisson fit + gamma-draw MC band). `--fp purity_mixture` is identical except the FP model.

---

## 4. Verify you got it right

- **Determinism:** the MAP point is bit-identical across reruns, thread counts, and `--n-mc`
  (the MAP is n_mc-independent; the MC band is seed-controlled, `seed=0`). If your MAP moves
  between runs, something is non-deterministic — investigate before trusting it.
- **The single-knob provenance proof:** `run_loa0_headline_full.py --gate` (or
  `apply_broadtrough_veto_headline.py --fp purity_mixture`) must reproduce **archival job
  52266001's** integrated values byte-identically. If it doesn't match, your config has drifted
  (check `lam_rf_min`, `zbins`, `v2_z_fit_hi`, the cut bundle, the molly/kernel/product files).
- **The actual expected numbers** (headline dN/dX, Ω, per-z, band) are the committed,
  code-commit-stamped **`scripts/loa0_headline.json`** in the private notes repo
  (`desi_gpy_dla_notes/notes/2026-06-29_hbi_cddf_draft/`), cross-checked against the draft
  tables. Compare your run's JSON to that file. **Do not** put real-LOA result values in this
  (code) repo.

---

## 5. Known gaps / caveats

Surfaced by the 2026-07-04 code/reproducibility panel. The number is bit-reproducible here;
these are portability items:

1. **Path constants still point at scratch.** Backup bundle exists (§2); remaining polish is to
   repoint `DEF_*` to it (or add a `--data-root`/manifest stage step) so a fresh checkout uses
   the persistent copy.
2. **No env lockfile** — export `environment.yml`; the §1 pins are the working set.
3. **Stray cache:** `loa0_bal_fp_product.npz` in the arbiter dir is NOT the headline FP product
   (that loads `loa0_fp_product_lyaonly1025.npz`, §2). Ignore it.

Already hardened (no longer gaps): `code_commit` stamping in both routines, and pre-flight
guards that hard-fail on a stale/full-forest FP product or substituted molly (`n_sl_prod`
consistency + `lam_rf_min=1025`/grid provenance).

---

## 6. Files

| what | path |
|---|---|
| headline runner (full schema + figures data) | `CDDF_analysis/diagnostics/bal_metal_fp/arbiter/run_loa0_headline_full.py` |
| table point + broad-trough veto cross-check | `CDDF_analysis/diagnostics/bal_metal_fp/arbiter/apply_broadtrough_veto_headline.py` |
| driver (frozen calibration + measurement) | `CDDF_analysis/hbi/track_c_tf_loa.py` |
| estimator (BYTE-FROZEN) | `CDDF_analysis/hbi/cddf_catalog_hbi.py` |
| ingredients builder / default paths | `CDDF_analysis/hbi/ab_loa0_fp_baseline.py` |
| loa0 FP product builder | `CDDF_analysis/hbi/build_loa0_fp_product.py` |
| provenance record + full manifest (private) | `desi_gpy_dla_notes/notes/2026-06-29_hbi_cddf_draft/PROVENANCE.md` + `scripts/loa0_headline.json` |
