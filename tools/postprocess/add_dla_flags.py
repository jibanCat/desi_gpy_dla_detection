#!/usr/bin/env python3
"""tools/postprocess/add_dla_flags.py

Add downstream-friendly flag columns to a directory of dlacat-*.fits files
*in place*. Idempotent: re-running clears the previous postprocess bits
on DLAFLAG and the boolean columns are overwritten.

Flag columns added (booleans + supporting metadata):
  LYBETA_FLAG          bool   — DLA is a likely Lyβ misID of a higher-z DLA
                                on the same LOS. From
                                gpy_dla_detection.postprocess.lyb_veto.
  LYBETA_PARENT_TID    int64  — TARGETID of the parent (higher-z DLA).
  LYBETA_PARENT_Z      float  — z of the parent.
  BAL_FLAG             bool   — TARGETID is in bal_cat.fits (any row;
                                molly's recipe drops all bal_cat TIDs,
                                not just BI_CIV>0).
  NHI_CONSISTENCY_FLAG bool   — NHI - k * NHI_ERR < 20.3, i.e. the predicted
                                NHI's lower 1σ falls below the canonical
                                catalog cut. Informational only: this is a
                                purity↔completeness selection knob, NOT a
                                quality defect, so since 2026-05-17 it is
                                NOT folded into DLAFLAG (see
                                docs/notes/2026-05-17_nhi_flag_investigation.md).
  PDLA_SATURATED_FLAG  bool   — P_DLA ≥ 1 - pdla_saturation_threshold (default
                                ≥ 1 − 1e-7, equivalent to log_BF ≥ 15.4 for
                                N_DLA_SAMPLES = 50000). Informational only:
                                marks rows where the p_DLA cut is a no-op
                                (high-confidence detection). NOT folded
                                into DLAFLAG.
  BF_BAND              float32— local posterior mass P(logNHI ≥ 20.3 | local
                                data) for the absorber: from the QMC 1-DLA
                                per-sample log-likelihoods, restricted to
                                samples within ±z_window of the detection's
                                Z_DLA, the likelihood-weighted fraction with
                                logNHI ≥ 20.3. A prior-mass-corrected interval
                                Bayes factor expressed as a probability in
                                [0,1] — discriminates true DLAs from
                                NHI-overestimated sub-DLAs near the 20.3
                                boundary. Informational only, NOT folded into
                                DLAFLAG (it is a purity↔completeness selection
                                knob — see docs/notes/2026-05-18_band_bf_flag_design.md).
                                Requires the processed h5 + the DLA samples
                                .mat; skipped (NaN) with --no-bf-band or if
                                those are unavailable.
  BF_BAND_NLOCAL       int32  — number of QMC samples inside the local
                                z-window (BF_BAND's QMC-noise diagnostic;
                                small N ⇒ noisy BF_BAND).

DLAFLAG bitmask updates (see fitwarning.py for the full definition):
  bit 3 (LYBETA_MISID)     ← LYBETA_FLAG
  bit 4 (BAL_CAT_OVERLAP)  ← BAL_FLAG
  (NHI_CONSISTENCY_FLAG, PDLA_SATURATED_FLAG and BF_BAND are informational —
   they get their own columns and are NOT folded into DLAFLAG.)

After running this script, downstream consumers can use either:
  cat[cat["DLAFLAG"] == 0]                          # "clean" production cat
  cat[~cat["LYBETA_FLAG"] & ~cat["BAL_FLAG"]]       # finer-grained filtering
  cat[~cat["NHI_CONSISTENCY_FLAG"]]                 # optional high-purity cut
  cat[cat["BF_BAND"] >= 0.7]                        # optional boundary-purity cut

The script is meant to be the LAST step of the production pipeline:

    inference (slurm/run_local.sh)
        → produces dlacat-*.fits + processed/processed-spectra-16-*.h5
    postprocess (this script)
        → adds the 5 flag columns to each dlacat-*.fits in place
    eval / catalog distribution
        → downstream consumers read dlacat-*.fits and use the flags

Usage:
    python tools/postprocess/add_dla_flags.py \
        --catalog-dir /path/to/run/ \
        --bal-cat /path/to/bal_cat.fits \
        --dla-samples-file /path/to/pw_samples_a3_*.mat \
        [--processed-dir /path/to/run/processed] \
        [--bf-band-z-window 0.02] \
        [--nhi-consistency-k 0.5] \
        [--pdla-saturation 1e-7] \
        [--lyb-veto-dz 0.005] \
        [--no-lyb-veto] [--no-bal-flag] \
        [--no-nhi-consistency] [--no-pdla-saturated] [--no-bf-band]

BF_BAND needs the inference run's processed h5 (per-sample log-likelihoods)
and the QMC DLA samples .mat — pass --dla-samples-file (or --no-bf-band).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import fitsio
from astropy.table import Table

# Make `import gpy_dla_detection.…` work whether the repo is on PYTHONPATH or not.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from gpy_dla_detection.postprocess.lyb_veto import flag_lybeta
from fitwarning import DLAFLAG


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--catalog-dir", required=True,
                   help="Directory containing dlacat-*.fits files.")
    p.add_argument("--bal-cat", default=None,
                   help="Path to bal_cat.fits. Required unless --no-bal-flag.")
    p.add_argument("--nhi-consistency-k", type=float, default=0.5,
                   help="k in `NHI - k * NHI_ERR < 20.3` (default 0.5).")
    p.add_argument("--nhi-consistency-floor", type=float, default=20.3,
                   help="NHI floor for the consistency gate (default 20.3).")
    p.add_argument("--pdla-saturation", type=float, default=1e-7,
                   help="PDLA_SATURATED_FLAG = P_DLA >= 1 - this (default 1e-7).")
    p.add_argument("--lyb-veto-dz", type=float, default=0.005,
                   help="dz_match for the Lyβ veto (default 0.005).")
    p.add_argument("--no-lyb-veto", action="store_true",
                   help="Skip LYBETA_FLAG / LYBETA_PARENT_* columns.")
    p.add_argument("--no-bal-flag", action="store_true",
                   help="Skip BAL_FLAG column. Otherwise --bal-cat required.")
    p.add_argument("--no-nhi-consistency", action="store_true",
                   help="Skip NHI_CONSISTENCY_FLAG column.")
    p.add_argument("--no-pdla-saturated", action="store_true",
                   help="Skip PDLA_SATURATED_FLAG column.")
    p.add_argument("--no-bf-band", action="store_true",
                   help="Skip BF_BAND / BF_BAND_NLOCAL columns.")
    p.add_argument("--processed-dir", default=None,
                   help="Dir with processed-spectra-16-*.h5 (per-sample LL) "
                        "for BF_BAND. Default: <catalog-dir>/processed.")
    p.add_argument("--dla-samples-file", default=None,
                   help="QMC DLA samples .mat used by the inference run "
                        "(must match the run; provides logNHI per sample). "
                        "Required for BF_BAND unless --no-bf-band.")
    p.add_argument("--bf-band-z-window", type=float, default=0.02,
                   help="Half-width of the local z_DLA window for BF_BAND "
                        "(default 0.02; see the design note).")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-file progress lines.")
    return p.parse_args()


def _add_lybeta(tbl: Table, dz_match: float) -> tuple[int, int]:
    """Mutate `tbl` in place: add LYBETA_FLAG/PARENT_TID/PARENT_Z. Return
    (n_total_dla_rows, n_lyb_misid_flagged)."""
    flag_lybeta(
        tbl,
        targetid_col="TARGETID",
        z_col="Z_DLA",
        nhi_col="NHI",      # dlacat's NHI is already log10(N_HI)
        dz_match=dz_match,
        require_higher_nhi_parent=True,
    )
    return len(tbl), int(np.asarray(tbl["LYBETA_FLAG"], dtype=bool).sum())


def _add_bal_flag(tbl: Table, bal_tids: set[int]) -> int:
    """Mutate `tbl`: add BAL_FLAG. Return n_flagged."""
    tids = np.asarray(tbl["TARGETID"], dtype=np.int64)
    flag = np.isin(tids, np.fromiter(bal_tids, dtype=np.int64))
    tbl["BAL_FLAG"] = flag
    return int(flag.sum())


def _add_nhi_consistency(tbl: Table, k: float, floor: float) -> int:
    """Mutate `tbl`: add NHI_CONSISTENCY_FLAG. True == fails the gate."""
    nhi = np.asarray(tbl["NHI"], dtype=np.float64)
    nhi_err = np.asarray(tbl["NHI_ERR"], dtype=np.float64)
    flag = (nhi - k * nhi_err) < floor
    tbl["NHI_CONSISTENCY_FLAG"] = flag
    return int(flag.sum())


def _add_pdla_saturated(tbl: Table, threshold: float) -> int:
    """Mutate `tbl`: add PDLA_SATURATED_FLAG."""
    p = np.asarray(tbl["P_DLA"], dtype=np.float64)
    flag = p >= (1.0 - threshold)
    tbl["PDLA_SATURATED_FLAG"] = flag
    return int(flag.sum())


# --- BF_BAND: local posterior mass P(logNHI >= 20.3 | local) ----------------
_BF_BAND_NHI_THRESH = 20.3


def _load_bf_band_inputs(processed_dir: str, samples_file: str):
    """Load the QMC sample logNHI grid and, per TARGETID, the 1-DLA per-sample
    log-likelihoods + per-sample z_DLA. Returns (spec, log_nhi_samples) where
    spec maps tid -> (L float32[S], z_dla_samples float64[S]).

    Per-sample z_DLA is reconstructed as
        z_i = min_z_dla + (max_z_dla - min_z_dla) * offset_sample_i
    (min/max_z_dlas stored per spectrum; offset_samples in the .mat).
    """
    import h5py  # local import — only needed when BF_BAND is requested
    with h5py.File(samples_file, "r") as f:
        log_nhi_samples = np.asarray(f["log_nhi_samples"][:, 0], dtype=np.float64)
        offset_samples = np.asarray(f["offset_samples"][:, 0], dtype=np.float64)
    n_samples = log_nhi_samples.size

    spec: dict[int, tuple] = {}
    h5s = sorted(glob.glob(os.path.join(processed_dir,
                                        "processed-spectra-16-*.h5")))
    if not h5s:
        raise SystemExit(f"BF_BAND: no processed-spectra-16-*.h5 in {processed_dir}")
    for hp in h5s:
        with h5py.File(hp, "r") as f:
            tids = np.asarray(f["target_ids"][:], dtype=np.int64)
            slld = f["sample_log_likelihoods_dla"]   # (n_spec, S, max_dlas)
            if slld.shape[1] != n_samples:
                raise SystemExit(
                    f"BF_BAND: sample-count mismatch — {hp} has S={slld.shape[1]}"
                    f" but {samples_file} has {n_samples}. Wrong --dla-samples-file?")
            minz = np.asarray(f["min_z_dlas"][:], dtype=np.float64)
            maxz = np.asarray(f["max_z_dlas"][:], dtype=np.float64)
            for r in range(slld.shape[0]):
                tid = int(tids[r])
                if tid < 0:
                    continue
                L = np.asarray(slld[r, :, 0], dtype=np.float64)
                zlo, zhi = minz[r], maxz[r]
                if not (np.isfinite(L).any() and np.isfinite(zlo)
                        and np.isfinite(zhi) and zhi > zlo):
                    continue
                spec[tid] = (L.astype(np.float32),
                             zlo + (zhi - zlo) * offset_samples)
    return spec, log_nhi_samples


def _add_bf_band(tbl: Table, spec: dict, log_nhi_samples: np.ndarray,
                 z_window: float) -> int:
    """Mutate `tbl`: add BF_BAND (float32) and BF_BAND_NLOCAL (int32).

    BF_BAND = P(logNHI >= 20.3 | local) = likelihood-weighted fraction of QMC
    samples within |z - Z_DLA| <= z_window that have logNHI >= 20.3. NaN when
    the spectrum is absent from the processed h5 or the local window is empty.
    Informational only — NOT folded into DLAFLAG.
    """
    n = len(tbl)
    bf = np.full(n, np.nan, dtype=np.float64)
    nloc = np.zeros(n, dtype=np.int64)
    tids = np.asarray(tbl["TARGETID"], dtype=np.int64)
    zdla = np.asarray(tbl["Z_DLA"], dtype=np.float64)
    hi = log_nhi_samples >= _BF_BAND_NHI_THRESH
    for i in range(n):
        rec = spec.get(int(tids[i]))
        if rec is None:
            continue
        L, zs = rec
        loc = np.isfinite(L) & (np.abs(zs - zdla[i]) <= z_window)
        nl = int(loc.sum())
        nloc[i] = nl
        if nl == 0:
            continue
        Ll = L[loc].astype(np.float64)
        w = np.exp(Ll - Ll.max())            # unnormalised posterior weights
        denom = w.sum()
        if not np.isfinite(denom) or denom <= 0:
            continue
        bf[i] = float(w[hi[loc]].sum() / denom)
    tbl["BF_BAND"] = bf.astype(np.float32)
    tbl["BF_BAND_NLOCAL"] = nloc.astype(np.int32)
    return int(np.isfinite(bf).sum())


def _update_dlaflag_bitmask(
    tbl: Table,
    input_colnames: set[str] | None = None,
) -> int:
    """Fold the quality flag columns into the DLAFLAG bitmask.

    1. Clear all known postprocess bits (current + legacy schema) first, so a
       re-run is idempotent and a catalog stamped under an older schema is
       cleaned up. This INCLUDES the NHI_INCONSISTENT bit (5), which was
       folded into DLAFLAG before 2026-05-17 — clearing it un-NHI-gates an
       older catalog on re-postprocess.
    2. Set LYBETA_MISID / BAL_CAT_OVERLAP from their boolean columns.

    NHI_CONSISTENCY_FLAG is deliberately NOT folded in (it is an
    informational selection knob, not a quality defect — see fitwarning.py
    and docs/notes/2026-05-17_nhi_flag_investigation.md). PDLA_SATURATED_FLAG
    likewise stays out. Both remain as standalone columns.

    Parameters
    ----------
    tbl : astropy.table.Table
        The dlacat being updated in place.
    input_colnames : set[str] or None
        Snapshot of column names from the INPUT FITS file, taken in
        `process_one` BEFORE the _add_* helpers ran. The legacy-bit refusal
        (below) probes this snapshot, NOT `tbl.colnames` — otherwise the
        check is dead code (the helpers always add the four boolean columns
        in production, so `tbl.colnames` always satisfies the column-presence
        condition). If None (e.g. unit-test direct call), the function uses
        `tbl.colnames` as a best-effort snapshot — only safe when the caller
        is constructing `tbl` directly without going through `process_one`.

    Returns the number of rows where DLAFLAG changed.
    """
    if "DLAFLAG" not in tbl.colnames:
        # Older catalogs without DLAFLAG: create as int64 zero
        tbl["DLAFLAG"] = np.zeros(len(tbl), dtype=np.int64)
    flag = np.asarray(tbl["DLAFLAG"], dtype=np.int64).copy()
    before = flag.copy()

    # ---- Refuse to process pre-2026-05-15 (legacy-bit-numbering) catalogs ----
    # Under the legacy schema bits 3/4/5 were the inference-time warnings
    # POTENTIAL_BAL / BAD_ZFIT / BAD_NHIFIT (now bits 0/1/2). Clearing
    # _ALL_POSTPROCESS_BITS_TO_CLEAR on such a catalog would silently erase
    # real inference warnings. Heuristic: a catalog is "post-reshuffle" iff
    # any of the postprocess boolean columns was present in the INPUT FITS
    # (i.e. it has been postprocessed before OR shipped from the current
    # pipeline). A catalog with bits 3/4/5 set and none of those columns in
    # the INPUT is almost certainly legacy — refuse.
    #
    # NB: probe the input-time snapshot, NOT `tbl.colnames`. By the time this
    # runs in `process_one`, the _add_* helpers have already written all four
    # boolean columns onto `tbl`, so `tbl.colnames` always satisfies the
    # column-presence condition and the heuristic would be dead code.
    #
    # The mask is built from LITERAL bit positions (1<<3 | 1<<4 | 1<<5),
    # not from the current DLAFLAG enum symbols. The symbols
    # LYBETA_MISID / BAL_CAT_OVERLAP / NHI_INCONSISTENT happen to live at
    # those positions today, but any future renumbering would silently shift
    # the mask off the legacy positions. Bits 3/4/5 are the historical
    # POTENTIAL_BAL/BAD_ZFIT/BAD_NHIFIT positions — frozen at the
    # pre-2026-05-15 schema, do not symbolify.
    _LEGACY_INFERENCE_BIT_POSITIONS = np.int64((1 << 3) | (1 << 4) | (1 << 5))
    probe_colnames = (
        input_colnames if input_colnames is not None else set(tbl.colnames)
    )
    no_postproc_cols = not any(
        c in probe_colnames for c in
        ("LYBETA_FLAG", "BAL_FLAG", "NHI_CONSISTENCY_FLAG", "PDLA_SATURATED_FLAG")
    )
    legacy_bits_present = bool(
        (flag & _LEGACY_INFERENCE_BIT_POSITIONS).any()
    )
    if no_postproc_cols and legacy_bits_present:
        raise RuntimeError(
            "Refusing to postprocess what looks like a pre-2026-05-15 legacy "
            "catalog: DLAFLAG has bits 3/4/5 set and none of the postprocess "
            "boolean columns (LYBETA_FLAG/BAL_FLAG/NHI_CONSISTENCY_FLAG/"
            "PDLA_SATURATED_FLAG) are present in the input FITS. Under the "
            "legacy numbering those bits were inference warnings "
            "(POTENTIAL_BAL/BAD_ZFIT/BAD_NHIFIT) and clearing them would "
            "silently erase them. Remap the legacy bits to the new positions "
            "(0/1/2) first, or process a fresh-from-inference catalog under "
            "the current schema."
        )

    # Clear ALL known postprocess bits (current + legacy schema). Keeps the
    # inference-time bits intact. The legacy mask handles dlacats that were
    # postprocessed under an earlier bit numbering / the pre-2026-05-17
    # NHI_INCONSISTENT bit and are being re-flagged.
    flag &= ~np.int64(DLAFLAG._ALL_POSTPROCESS_BITS_TO_CLEAR)

    # Set postprocess bits from the boolean columns when present.
    # NHI_CONSISTENCY_FLAG is intentionally excluded — informational only.
    if "LYBETA_FLAG" in tbl.colnames:
        m = np.asarray(tbl["LYBETA_FLAG"], dtype=bool)
        flag[m] |= np.int64(DLAFLAG.LYBETA_MISID)
    if "BAL_FLAG" in tbl.colnames:
        m = np.asarray(tbl["BAL_FLAG"], dtype=bool)
        flag[m] |= np.int64(DLAFLAG.BAL_CAT_OVERLAP)

    tbl["DLAFLAG"] = flag
    return int((flag != before).sum())


def process_one(path: Path, args, bal_tids: set[int] | None,
                bf_spec: dict | None = None,
                bf_lognhi: np.ndarray | None = None,
                ) -> dict[str, int]:
    """Process one dlacat-*.fits file in place. Return per-file stats."""
    tbl = Table(fitsio.read(str(path), ext=1))
    stats: dict[str, int] = {"path": str(path), "n_rows": len(tbl)}

    # Snapshot input-file column names BEFORE any _add_* helper mutates `tbl`.
    # `_update_dlaflag_bitmask` uses this snapshot for its legacy-bit refusal
    # (the helpers below unconditionally add the four boolean columns when
    # invoked, so probing `tbl.colnames` AFTER they ran would make the
    # refusal dead code — see the docstring in _update_dlaflag_bitmask).
    input_colnames: set[str] = set(tbl.colnames)

    if not args.no_lyb_veto:
        n, n_flag = _add_lybeta(tbl, args.lyb_veto_dz)
        stats["lyb_flagged"] = n_flag
    if not args.no_bal_flag and bal_tids is not None:
        stats["bal_flagged"] = _add_bal_flag(tbl, bal_tids)
    if not args.no_nhi_consistency:
        stats["nhi_consistency_flagged"] = _add_nhi_consistency(
            tbl, args.nhi_consistency_k, args.nhi_consistency_floor)
    if not args.no_pdla_saturated:
        stats["pdla_saturated"] = _add_pdla_saturated(tbl, args.pdla_saturation)
    if not args.no_bf_band and bf_spec is not None:
        stats["bf_band_finite"] = _add_bf_band(
            tbl, bf_spec, bf_lognhi, args.bf_band_z_window)

    # Fold quality flags into DLAFLAG bitmask (excludes the informational
    # PDLA_SATURATED and NHI_CONSISTENCY flags). After this,
    # `cat[cat["DLAFLAG"] == 0]` is the "clean" downstream filter.
    stats["dlaflag_changed"] = _update_dlaflag_bitmask(tbl, input_colnames)
    flag = np.asarray(tbl["DLAFLAG"], dtype=np.int64)
    stats["dlaflag_zero"] = int((flag == 0).sum())

    # In-place rewrite via fitsio.write(..., clobber=True)
    fitsio.write(str(path), np.asarray(tbl), extname="DLAS", clobber=True)
    return stats


def main():
    args = parse_args()
    cat_dir = Path(args.catalog_dir)
    if not cat_dir.is_dir():
        raise SystemExit(f"--catalog-dir not a directory: {cat_dir}")

    files = sorted(cat_dir.glob("dlacat-*.fits"))
    if not files:
        raise SystemExit(f"no dlacat-*.fits in {cat_dir}")
    print(f"[postprocess] found {len(files)} dlacat-*.fits in {cat_dir}")

    bal_tids: set[int] | None = None
    if not args.no_bal_flag:
        if not args.bal_cat:
            raise SystemExit(
                "--bal-cat required (or pass --no-bal-flag to skip BAL_FLAG)")
        bal = fitsio.read(args.bal_cat, ext=1, columns=["TARGETID"])
        bal_tids = set(int(r["TARGETID"]) for r in bal)
        print(f"[postprocess] loaded {len(bal_tids)} BAL TIDs from "
              f"{args.bal_cat} (drop-all-BAL convention; matches molly)")

    # BF_BAND inputs (processed h5 + QMC samples) — loaded once.
    # Skipped (gracefully, not an error) if the inputs are unavailable, so an
    # existing run_local.sh postprocess call that omits --dla-samples-file
    # still works — it just won't get the BF_BAND column.
    bf_spec = None
    bf_lognhi = None
    if not args.no_bf_band:
        proc_dir = args.processed_dir or str(cat_dir / "processed")
        if not args.dla_samples_file:
            print("[postprocess] BF_BAND: SKIPPED — no --dla-samples-file "
                  "(pass it to compute BF_BAND, or --no-bf-band to silence).")
        elif not os.path.isdir(proc_dir):
            print(f"[postprocess] BF_BAND: SKIPPED — --processed-dir not "
                  f"found: {proc_dir}")
        else:
            print(f"[postprocess] BF_BAND: loading per-sample LL from "
                  f"{proc_dir} + samples {args.dla_samples_file} ...")
            bf_spec, bf_lognhi = _load_bf_band_inputs(proc_dir,
                                                      args.dla_samples_file)
            print(f"[postprocess] BF_BAND: {len(bf_spec)} spectra with usable "
                  f"1-DLA samples (z-window ±{args.bf_band_z_window})")

    totals = {"n_rows": 0, "lyb_flagged": 0, "bal_flagged": 0,
              "nhi_consistency_flagged": 0, "pdla_saturated": 0,
              "bf_band_finite": 0, "dlaflag_zero": 0, "dlaflag_changed": 0}
    for f in files:
        stats = process_one(f, args, bal_tids, bf_spec, bf_lognhi)
        if not args.quiet:
            extras = " ".join(f"{k}={v}" for k, v in stats.items()
                              if k not in ("path", "n_rows"))
            print(f"[postprocess] {f.name}: rows={stats['n_rows']} {extras}")
        for k, v in stats.items():
            if k in totals:
                totals[k] += v

    print()
    print(f"[postprocess] TOTALS across {len(files)} files:")
    print(f"  rows                       = {totals['n_rows']}")
    if not args.no_lyb_veto:
        n = totals['lyb_flagged']; tot = totals['n_rows']
        pct = 100.0 * n / tot if tot else 0.0
        print(f"  LYBETA_FLAG=True           = {n} ({pct:.2f}%)")
    if not args.no_bal_flag:
        n = totals['bal_flagged']; tot = totals['n_rows']
        pct = 100.0 * n / tot if tot else 0.0
        print(f"  BAL_FLAG=True              = {n} ({pct:.2f}%)")
    if not args.no_nhi_consistency:
        n = totals['nhi_consistency_flagged']; tot = totals['n_rows']
        pct = 100.0 * n / tot if tot else 0.0
        print(f"  NHI_CONSISTENCY_FLAG=True  = {n} ({pct:.2f}%)  "
              f"(k={args.nhi_consistency_k}, floor={args.nhi_consistency_floor})  "
              f"[informational, NOT in DLAFLAG]")
    if not args.no_pdla_saturated:
        n = totals['pdla_saturated']; tot = totals['n_rows']
        pct = 100.0 * n / tot if tot else 0.0
        print(f"  PDLA_SATURATED_FLAG=True   = {n} ({pct:.2f}%)  "
              f"(P_DLA ≥ 1 − {args.pdla_saturation:g})  [informational, NOT in DLAFLAG]")
    if not args.no_bf_band and bf_spec is not None:
        n = totals['bf_band_finite']; tot = totals['n_rows']
        pct = 100.0 * n / tot if tot else 0.0
        print(f"  BF_BAND finite             = {n} ({pct:.2f}%)  "
              f"(P(logNHI≥20.3|local), ±{args.bf_band_z_window} z-window)  "
              f"[informational, NOT in DLAFLAG]")
    n = totals['dlaflag_zero']; tot = totals['n_rows']
    pct = 100.0 * n / tot if tot else 0.0
    print(f"  DLAFLAG == 0 ('clean')     = {n} ({pct:.2f}%)  "
          f"[inference bits 0-2 + postprocess bits 3-4 all clear; "
          f"NHI_INCONSISTENT not gated]")
    print(f"[postprocess] done.")


if __name__ == "__main__":
    main()
