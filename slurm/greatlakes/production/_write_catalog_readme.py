#!/usr/bin/env python3
"""_write_catalog_readme.py — emit a collaborator README for a packaged dlacat.

Called by package_catalog.sh (step 5). Run-specific values (row counts, source
run, commit) are read from the FITS header / passed in; the column dictionary,
flag semantics and recommended-cut guidance are static (the dlacat schema is
the same for every GP-DLA run). Validated P/C is optional (--purity/--compl):
the packager does not run the truth eval, so if omitted the README points the
reader at examples/molly_faithful_pc_plots.py instead.
"""
import argparse
import numpy as np
import fitsio


TEMPLATE = """# GP-DLA absorber catalog — {source_label}

**File:** `dlacat-{release}-mockcat.fits` (single FITS table, HDU 1, `EXTNAME=DLACAT`)

Combined Gaussian-Process DLA detection catalog. {data_kind}

| | |
|---|---|
| Rows | {nrows:,} (one row **per detected DLA candidate**) |
| Unique sightlines | {nuniq:,} TARGETIDs |
| Source run | {source_label} |
| Code version | commit **`{commit}`** of `desi_gpy_dla_detection` (FITS header `CODECMT`; also in `BASELINE.env`) |
| Provenance | `examples/combine_dlacat.py` + `tools/postprocess/add_dla_flags.py` (lyb dz={lybdz}); FITS header `COMBTOOL/FLAGTOOL/LYBDZ/SRCRUN/CODECMT` |

A row is one absorber, so a sightline with k DLAs contributes k rows (group by
`TARGETID`). Sightlines with no detection are absent.

**Files in this folder:** `dlacat-{release}-mockcat.fits` (catalog), `README.md`
(this file), `BASELINE.env` (exact resolved pipeline config — for reproducibility).

## Reading it
```python
from astropy.table import Table
cat = Table.read("dlacat-{release}-mockcat.fits")
# or:  import fitsio; cat = fitsio.read("dlacat-{release}-mockcat.fits", ext=1)
```

## Columns
Per-sightline: `TARGETID` (join key), `RA`, `DEC`, `Z_QSO`, `SNR_FOREST`,
`SNR_REDSIDE` (the S/N used in cuts).
Per-absorber: `DLAID`, `Z_DLA`(`_ERR`), `NHI`(`_ERR`) [log10 cm^-2],
`P_DLA` (detection confidence), `P_NULL`, `LOGP_DLA`, `LOGP_NULL`, `MODEL_P`.
Flags: `DLAFLAG` (bitmask; **`==0` is the clean set**), `LYBETA_FLAG`,
`LYBETA_PARENT_TID/Z`, `BAL_FLAG`, `NHI_CONSISTENCY_FLAG` (informational),
`PDLA_SATURATED_FLAG` (informational).

### `DLAFLAG` bitmask (`==0` ⇒ all clear)
bit0 `POTENTIAL_BAL` (overlaps a Lyα/N V BAL trough) · bit1 `BAD_ZFIT` ·
bit2 `BAD_NHIFIT` · bit3 `LYBETA_MISID` (=`LYBETA_FLAG`) ·
bit4 `BAL_CAT_OVERLAP` (=`BAL_FLAG`, sightline in bal_cat).
`NHI_CONSISTENCY_FLAG` / `PDLA_SATURATED_FLAG` are NOT folded in (apply yourself).

## Recommended selection
```python
import numpy as np
sel = (cat["DLAFLAG"] == 0) & (cat["P_DLA"] > 0.99) & \\
      (cat["SNR_REDSIDE"] > 2) & (cat["NHI"] > 20.3)
clean = cat[sel]
# to reproduce the validated P/C, also: Z_QSO in (2.0, 4.25) and Z_DLA within
# rest-frame lambda [911,1216] A of Z_QSO (the Lya-Lyb window).
```
`P_DLA` is the tunable knob: lower for completeness, raise for purity.

## Validated performance (vs the mock truth)
{pc_block}

## Caveats
- Mock validation catalog (truth known); numbers differ on real DESI spectra.
- BAL sightlines handled by *exclusion* (`BAL_FLAG`), not pixel masking.
- See `BASELINE.env` for the exact config; regenerate via the pipeline:
  inference (`desi-DLAGP.py`) → `examples/combine_dlacat.py` →
  `tools/postprocess/add_dla_flags.py` → `package_catalog.sh`.
- Questions: mfho@umich.edu.
"""

PC_WITH = """Matched per-TARGETID (|Δz|/(1+z) < 0.01, strongest-N_HI-first), BAL excluded,
Lyβ vetoed, at `P_DLA > 0.99`, `SNR_REDSIDE > 2`:

| N(H I) floor | Purity | Completeness |
|---|---:|---:|
| log N_HI > 20.3 | {purity} | {compl} |
"""

PC_WITHOUT = """Measure with `examples/molly_faithful_pc_plots.py` against the mock truth
(`hcd_truth_cat.fits` / `dla_cat.fits`), recipe: `--no-bal --lyb-veto
--snr-min 2 --nhi-min 20.3 --lam-rf-min 911 --lam-rf-max 1216` (NHI-descending
matcher). Re-run on this catalog to fill in the headline P/C."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--source-label", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--lyb-veto-dz", default="0.005")
    ap.add_argument("--purity", default=None)
    ap.add_argument("--compl", default=None)
    ap.add_argument("--data-kind",
                    default="It is a **validation/mock** catalog (truth known), not real-survey data.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = fitsio.read(a.fits, ext=1)
    nrows = len(d)
    nuniq = int(np.unique(d["TARGETID"]).size)

    if a.purity and a.compl:
        pc_block = PC_WITH.format(purity=a.purity, compl=a.compl)
    else:
        pc_block = PC_WITHOUT

    with open(a.out, "w") as f:
        f.write(TEMPLATE.format(
            source_label=a.source_label, release=a.release, data_kind=a.data_kind,
            nrows=nrows, nuniq=nuniq, commit=a.commit, lybdz=a.lyb_veto_dz,
            pc_block=pc_block,
        ))
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
