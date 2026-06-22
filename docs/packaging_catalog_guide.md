# Packaging a GP-DLA catalog — reference guide

How to turn a finished GP-DLA inference run into a shareable, fully-flagged,
reproducible absorber catalog (a `dlacat` bundle). This is the canonical
post-run procedure; it is codified in
[`slurm/nersc/production/package_catalog.sh`](../slurm/nersc/production/package_catalog.sh).

A finished run leaves, under `<RUNDIR>/outputs/`:
- per-task `dlacat-*-hpx-<a>-<b>.fits` (absorber rows, written at each task's window-end),
- `figures/processed/processed-*.h5` (per-healpix model output),
- `BASELINE.env` (resolved config + `CODE_COMMIT`, stamped at launch).

Packaging combines those into one catalog, adds quality flags, stamps provenance,
writes a README, and optionally copies the bundle to a shared location.

---

## The procedure (5 steps + 1 prerequisite)

| Step | Tool | Action |
|---|---|---|
| **0 (pre)** | [`tools/postprocess/build_bal_cat_from_qsocat.py`](../tools/postprocess/build_bal_cat_from_qsocat.py) | Build a **BAL-only** catalog from the altbal QSO catalog. **Required:** step 2 flags *every* `TARGETID` present in `--bal-cat` (drop-all-BAL), so passing the raw QSO catalog would flag all sightlines. Default `--col BI_CIV --thresh 0.0` (BAL ≡ `BI_CIV > 0`). |
| **1** | [`examples/combine_dlacat.py`](../examples/combine_dlacat.py) | Glob all per-task `dlacat-*-hpx-<int>-<int>.fits`, `vstack` → one catalog. With `--expect-positions N --fail-on-gap` it parses the filename ranges into a covered-healpix set and exits non-zero on any gap — the authoritative completeness gate (use it for real data, which has no truth catalog). |
| **2** | [`tools/postprocess/add_dla_flags.py`](../tools/postprocess/add_dla_flags.py) (+ [`gpy_dla_detection/postprocess/lyb_veto.py`](../gpy_dla_detection/postprocess/lyb_veto.py), [`fitwarning.py`](../fitwarning.py)) | Add flag columns and fold `DLAFLAG` postprocess bits (Lyβ-misID, BAL-overlap). `--lyb-veto-dz 0.005 --no-bf-band`. |
| **3** | `package_catalog.sh` (inline) | Clip `P_DLA`/`P_NULL` to `[0,1]`; stamp provenance header cards. |
| **4** | `package_catalog.sh` (inline) | Copy the run's `BASELINE.env` into the bundle. |
| **5** | [`slurm/nersc/production/_write_catalog_readme.py`](../slurm/nersc/production/_write_catalog_readme.py) | Write `README.md` (column dictionary, flag semantics, recommended cut). `--data-kind real` → no-truth caveat; `mock` → P/C-vs-truth table. |
| **share** | `package_catalog.sh` (inline) | `--share-to <dir>` copies FITS + README + BASELINE.env to a persistent share dir. |

`--data-kind` controls naming: **real** → `dlacat-<release>-<survey>-<program>.fits`
(no-truth README); **mock** → `dlacat-<release>-mockcat.fits` (mock-validation
README; byte-identical to the historical hand packaging).

---

## Commands

`package_catalog.sh` runs steps 1–5 + share. Build the BAL-only catalog first:

```bash
# (0) BAL-only catalog (BI_CIV>0) — one-time per run
python tools/postprocess/build_bal_cat_from_qsocat.py \
    --qsocat <ALTBAL_QSOCAT.fits> \
    --out    <RUNDIR>/outputs/<run>_bal_cat_bici_gt0.fits \
    --col BI_CIV --thresh 0.0

# (1–5 + share) package the finished run
bash slurm/nersc/production/package_catalog.sh \
    --rundir  <RUNDIR> \
    --release <release-label> \
    --bal-cat <RUNDIR>/outputs/<run>_bal_cat_bici_gt0.fits \
    --expect-positions <N_HEALPIX> \
    --data-kind <real|mock> [--survey main --program dark] \
    --share-to <persistent share dir>
```

**Run on a compute node, not a login node** — step 2 reads the (multi-GB) altbal
QSO catalog and the combined absorber table. On NERSC/Perlmutter, submit to the
shared partition (`sbatch -A desi -q shared -C cpu -t 0:30:00 -c 8 …`). Other
clusters: the equivalent short shared/standard allocation.

### Reproduce from scratch (if inference isn't done yet)

```bash
# always --dry-run first
bash slurm/nersc/production/launch_nersc.sh <flavour>.env \
    --start 0 --end <OUTER_MAX_INDEX> --window 512 --time 06:00:00 [--dry-run]
# then run the packaging block above. Resume a gap by re-launching
# --start <first_missing> --end <end>; per-task writes are idempotent.
```

---

## The output FITS (EXTNAME = `DLACAT`)

One row = one detected absorber; group by `TARGETID` (a k-absorber sightline
contributes k rows; non-detections are absent). Columns 1–17 come from
inference/combine; 18–23 are added by `add_dla_flags.py`.

| # | Column | Meaning |
|---|---|---|
| 1 | `TARGETID` | DESI TARGETID (join key) |
| 2–3 | `RA` `DEC` | QSO sky position (deg) |
| 4 | `Z_QSO` | QSO redshift |
| 5 | `SNR_FOREST` | mean S/N in the forest window (can be < 0) |
| 6 | `SNR_REDSIDE` | mean S/N in the red window — **the S/N used in cuts** |
| 7 | `DLAID` | `str(TARGETID)+"00"+n` per absorber n |
| 8–9 | `Z_DLA` `Z_DLA_ERR` | MAP absorber redshift; ERR `−1` = not computed |
| 10–11 | `NHI` `NHI_ERR` | MAP log₁₀(N_HI/cm⁻²); ERR `−1` = not computed |
| 12 | `DLAFLAG` | quality bitmask, `== 0` is clean (below) |
| 13–14 | `P_DLA` `P_NULL` | model posteriors, clipped to [0,1], sum to 1 |
| 15–16 | `LOGP_DLA` `LOGP_NULL` | log model posteriors |
| 17 | `MODEL_P` | `model_posteriors[1+num_subdla+n]` for this absorber |
| 18 | `LYBETA_FLAG` | likely Lyβ misID |
| 19–20 | `LYBETA_PARENT_TID` `LYBETA_PARENT_Z` | matched higher-z parent DLA; `−1`/`NaN` if none |
| 21 | `BAL_FLAG` | `TARGETID` ∈ the BAL catalog |
| 22 | `NHI_CONSISTENCY_FLAG` | informational (lower-1σ NHI below floor) |
| 23 | `PDLA_SATURATED_FLAG` | informational (`P_DLA ≥ 1−1e-7`) |

### `DLAFLAG` bitmask ([`fitwarning.py`](../fitwarning.py))

`DLAFLAG == 0` ⇒ all clear. Bits 0–2 are set during inference; bits 3–4 during
postprocess (re-stamped idempotently — postprocess bits are cleared then re-OR'd,
inference bits preserved).

| Bit | Val | Name | Condition | Set during |
|---|---|---|---|---|
| 0 | 1 | `POTENTIAL_BAL` | MAP DLA Lyα center inside a BAL trough (geometric) | inference |
| 1 | 2 | `BAD_ZFIT` | bad χ²(z) parabola fit / `LinAlgError` | inference |
| 2 | 4 | `BAD_NHIFIT` | bad χ²(NHI) parabola fit / All-NaN slice | inference |
| 3 | 8 | `LYBETA_MISID` | `= LYBETA_FLAG`: Lyβ misID of a higher-z, higher-NHI DLA on the same sightline (`dz ≤ --lyb-veto-dz`) | postprocess |
| 4 | 16 | `BAL_CAT_OVERLAP` | `= BAL_FLAG`: `TARGETID` in `--bal-cat` | postprocess |
| 5 | 32 | `NHI_INCONSISTENT` | **defined but NOT folded** into `DLAFLAG`; surfaced only via `NHI_CONSISTENCY_FLAG` | (not set) |

`NHI_CONSISTENCY_FLAG` and `PDLA_SATURATED_FLAG` are **informational — never folded
into `DLAFLAG`**. Apply them yourself only if wanted.

### Recommended operating-point cut

```python
sel = (cat["DLAFLAG"] == 0) & (cat["P_DLA"] > 0.99) & \
      (cat["SNR_REDSIDE"] > 2) & (cat["NHI"] > 20.0)   # >20.3 conservative; >19.0 incl. sub-DLAs
```
`SNR_REDSIDE > 2` is the canonical operating point. For real data, purity/
completeness are not measurable (no truth) — these cuts are the mock-validated
operating point.

### Provenance header cards (HDU1)

`package_catalog.sh` stamps: `CODECMT` (git commit that ran inference, from
`BASELINE.env`), `COMBTOOL`, `FLAGTOOL`, `LYBDZ`, `SRCRUN`, `PDLACLIP`,
`PKGDATE`, `NROWS`, `NUNQTID`. The combined-coverage cards written by
`combine_dlacat.py` (`NSLICES`/`NPOSCOV`/`NPOSMIS`) are dropped by the
`add_dla_flags.py` rewrite, so scripted bundles do not retain them; the gap gate
still runs at combine time.

---

## See also

- [`slurm/nersc/production/README.md`](../slurm/nersc/production/README.md) — production drivers, configs, "After a run finishes".
- `_write_catalog_readme.py` emits a per-bundle `README.md` (column dictionary + cut) alongside every catalog.
