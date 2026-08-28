"""
examples/inject/coadd_injection.py
==================================
M3 injection-campaign I/O machinery (scientific-software / CS owner).

Three public entry points:

* :func:`build_clean_table` — the clean-sightline selection table
  ``clean = zcat.TARGETID − hcd_truth.TARGETID − bal_cat.TARGETID``, joined with
  ``snr_cat`` SNR and the nside=16 nested HEALPIX (DESI convention).
* :func:`inject_into_coadd` — read a desispec coadd, multiply selected fibers'
  flux by the Voigt transmission (``inject_voigt``) at the coadd's OBSERVED
  wavelengths, and write a NEW coadd in the SAME schema the GP reads, so
  ``dlasearch`` runs on it unchanged (ivar / mask / fibermap preserved).
* :func:`write_campaign` — orchestrate: group a manifest by healpix, inject into
  each source coadd, write the injectable healpix tree the production GP driver
  scans (``{out_root}/spectra-16/{hp//100}/{hp}/spectra-16-{hp}.fits`` + the
  ``truth-16`` companion), and a per-injection truth manifest for the
  measurement step.

Discipline
----------
* Reuse ``gpy_dla_detection.inject_absorber.inject_voigt`` — never reimplement
  the Voigt profile or its 3-pixel edge convention.
* Reuse ``desispec.io`` for read/write — never hand-roll the coadd FITS schema.
* The GP / ``dla_gp.py`` / inference are NEVER touched. Injection is purely
  input-flux preprocessing; the GP runs config-only on the injected tree.
* TARGETIDs are 19-digit DESI integers → kept strictly ``int64`` (no float
  coercion anywhere in the set algebra or the join).

Why inject into every camera
----------------------------
``dlasearch.process_spectra_group`` reads the b/r/z cameras and ``coadd_cameras``
them INSIDE the GP run, then scores on the coadded observed grid. We therefore
inject into each camera at that camera's own observed wavelengths. The Voigt
transmission is ≈ 1 far from the line, so injecting into every camera is exact
(the absorber only bites where it overlaps a camera's range) and avoids any
assumption about which camera the trough lands in.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

# Reuse the committed injection primitive (linear-N + 3-pixel edge handling).
from gpy_dla_detection.inject_absorber import inject_voigt

# DESI healpix resolution (nside=16, NESTED) for the spectra-16 tree.
_NSIDE = 16


# ---------------------------------------------------------------------------
# Small column helpers (tolerate RA/DEC naming variants across catalogs)
# ---------------------------------------------------------------------------


def _col(table, *names):
    """Return the first present column name from ``names`` (case as given)."""
    for n in names:
        if n in table.colnames:
            return n
    raise KeyError(f"none of {names} in table columns {list(table.colnames)}")


def _radec(table):
    ra = np.asarray(table[_col(table, "TARGET_RA", "RA")], dtype=np.float64)
    dec = np.asarray(table[_col(table, "TARGET_DEC", "DEC")], dtype=np.float64)
    return ra, dec


# ===========================================================================
# 1. Clean-sightline table
# ===========================================================================


def build_clean_table(zcat, hcd_truth, bal_cat, snr_cat):
    """Clean-sightline selection table for the injection campaign.

    ``clean = zcat.TARGETID − hcd_truth.TARGETID − bal_cat.TARGETID``: QSO
    sightlines with NO truth HCD absorber and NO BAL. The returned table carries,
    for each clean sightline, the QSO redshift, the ``snr_cat`` SNR columns, and
    the nside=16 nested HEALPIX (so :func:`write_campaign` can group by healpix
    exactly as the GP driver's ``spectra-16`` tree does).

    BAL caveat for M4: because BALs are EXCLUDED from the clean set, the response
    matrix R measured here is BAL-free. Production runs include BALs UNMASKED, so
    R must be applied with a separate BAL correction to the BAL-contaminated
    fraction of the real run at M4 (R_BAL-free is NOT valid on that fraction).

    Parameters
    ----------
    zcat, hcd_truth, bal_cat, snr_cat : astropy.table.Table
        The four 2LPT-0 catalogs. ``zcat`` needs TARGETID, Z, RA/DEC (or
        TARGET_RA/TARGET_DEC). ``hcd_truth`` and ``bal_cat`` need TARGETID.
        ``snr_cat`` needs TARGETID + SNR columns (SNR_FOREST / SNR_REDSIDE).

    Returns
    -------
    astropy.table.Table
        Columns: TARGETID (int64), Z, TARGET_RA, TARGET_DEC, HEALPIX (int64),
        plus every SNR column present in ``snr_cat`` (e.g. SNR_FOREST,
        SNR_REDSIDE). One row per clean sightline.
    """
    from astropy.table import Table

    z_tid = np.asarray(zcat["TARGETID"], dtype=np.int64)
    hcd_tid = np.asarray(hcd_truth["TARGETID"], dtype=np.int64)
    bal_tid = np.asarray(bal_cat["TARGETID"], dtype=np.int64)

    # Set difference on int64 ids (np.isin keeps exact integer comparison).
    contaminated = np.union1d(hcd_tid, bal_tid)
    clean_mask = ~np.isin(z_tid, contaminated)

    clean_tid = z_tid[clean_mask]
    ra, dec = _radec(zcat)
    ra, dec = ra[clean_mask], dec[clean_mask]
    z = np.asarray(zcat["Z"], dtype=np.float64)[clean_mask]

    # Healpix (nside=16, nested), the DESI spectra-16 assignment.
    import healpy as hp

    healpix = hp.ang2pix(_NSIDE, ra, dec, nest=True, lonlat=True).astype(np.int64)

    out = Table()
    out["TARGETID"] = clean_tid  # int64
    out["Z"] = z
    out["TARGET_RA"] = ra
    out["TARGET_DEC"] = dec
    out["HEALPIX"] = healpix

    # Join SNR columns from snr_cat by TARGETID (left join onto clean rows).
    snr_tid = np.asarray(snr_cat["TARGETID"], dtype=np.int64)
    order = np.argsort(snr_tid, kind="stable")
    snr_tid_sorted = snr_tid[order]
    pos = np.searchsorted(snr_tid_sorted, clean_tid)
    pos = np.clip(pos, 0, snr_tid_sorted.size - 1)
    found = snr_tid_sorted[pos] == clean_tid
    src_idx = order[pos]

    for name in snr_cat.colnames:
        if name == "TARGETID":
            continue
        col = np.asarray(snr_cat[name])
        joined = np.full(clean_tid.shape, np.nan, dtype=np.float64)
        joined[found] = col[src_idx[found]].astype(np.float64)
        out[name] = joined

    return out


# ===========================================================================
# 2. Coadd injector
# ===========================================================================


def _taueff_spec(spec):
    """A mean-flux model for the R-041C rescaling: a named model from
    ``injection.noise_preserving.TAUEFF_MODELS`` or a MEASURED table
    ``{"z": [...], "taueff": [...]}`` (e.g. tools/r041_mock_meanflux.py output),
    interpolated linearly in z (clamped at the table ends)."""
    from injection.noise_preserving import taueff
    if isinstance(spec, str):
        return taueff(spec)
    z = np.asarray(spec["z"], dtype=float); t = np.asarray(spec["taueff"], dtype=float)
    ok = np.isfinite(t)
    z, t = z[ok], t[ok]
    return lambda zz: np.interp(np.asarray(zz, dtype=float), z, t)


def _normalize_injections(injections: Iterable[Mapping]) -> list:
    """Coerce injection records to a uniform dict list.

    Each record needs ``target_id``, ``logN_true`` (log10 N_HI), ``z_true``
    (absorber redshift). ``num_lines`` optional (falls back to the call default).
    """
    norm = []
    for rec in injections:
        norm.append(
            {
                "target_id": int(rec["target_id"]),
                "logN_true": (None if rec.get("logN_true") is None else float(rec["logN_true"])),
                "z_true": (None if rec.get("z_true") is None else float(rec["z_true"])),
                "z_qso": (None if rec.get("z_qso") is None else float(rec["z_qso"])),
                "num_lines": (
                    int(rec["num_lines"]) if rec.get("num_lines") is not None else None
                ),
            }
        )
    return norm


# Forest-blend guard threshold: if the PRE-injection forest flux at the trough
# centre is below this fraction of the local pseudo-continuum, the sightline is
# already near-black there (a pre-existing strong forest/blend), so the injected
# absorber's response is not separable from the blend. We FLAG it (never drop
# silently) so the M4 measurement can exclude/annotate it.
_FOREST_BLEND_FRAC = 0.1


def _forest_flux_fraction(wave, flux, z_dla, *, half_window_ang=4.0):
    """Pre-injection forest flux at the trough centre, as a fraction of the local
    pseudo-continuum.

    Estimated on whichever camera covers ``(1 + z_dla) * Lyα``: the median flux
    in a small window AT the trough centre divided by a robust local continuum
    (the 90th percentile over a wider neighbourhood). Returns ``nan`` if the line
    centre falls outside every camera's range (the absorber is off-grid).
    """
    LYA = 1215.67
    lam0 = (1.0 + float(z_dla)) * LYA
    w = np.asarray(wave, dtype=np.float64)
    f = np.asarray(flux, dtype=np.float64)
    if lam0 < w[0] or lam0 > w[-1]:
        return np.nan
    centre = np.abs(w - lam0) < half_window_ang
    local = np.abs(w - lam0) < (10.0 * half_window_ang)
    if not np.any(centre) or not np.any(local):
        return np.nan
    cont = np.nanpercentile(f[local], 90)
    if not np.isfinite(cont) or cont <= 0:
        return np.nan
    centre_flux = np.nanmedian(f[centre])
    return float(centre_flux / cont)


def inject_into_coadd(
    coadd_in_path: str,
    coadd_out_path: str,
    injections: Iterable[Mapping],
    *,
    num_lines: int,
    blend_report: Optional[list] = None,
    method: str = "multiplicative",
    meanflux: Optional[Mapping] = None,
    seed_salt: str = "r041",
):
    """Inject Voigt absorbers into selected fibers of a desispec coadd.

    Reads ``coadd_in_path`` with ``desispec.io.read_spectra``, multiplies each
    injected fiber's flux (in EVERY camera, at that camera's observed
    wavelengths) by ``inject_voigt(wave, flux, 10**logN_true, z_true,
    num_lines)``, and writes a NEW coadd to ``coadd_out_path`` in the SAME schema
    — same bands, wave grids, ivar, mask, fibermap. The GP (``dlasearch``) runs
    on the output unchanged.

    Grid convention (M4-validated)
    ------------------------------
    Injection happens on each CAMERA's observed wavelength grid (~0.8 Å linear).
    ``dlasearch.process_spectra_group`` then coadds those cameras onto a SINGLE
    common **0.8 Å linear ``brz`` grid** (mock coadds carry no resolution on the
    coadd → ``coadd_cameras`` falls back to ``resample_spectra_lin_or_log`` using
    the ``truth-16`` resolution), and the GP scores on THAT grid. A round-trip
    validation (``tests/test_coadd_injection.py::test_m4_roundtrip_*``) confirms
    that injecting on the camera pitch survives this flux-conserving resample to
    <1 % in equivalent width across log N_HI ∈ [17.5, 20.3] — i.e. the GP recovers
    the SAME absorber, with no EW / N_HI bias from a grid mismatch — so no special
    pre-resampling to the brz grid is required.

    Multiple injections targeting the same fiber blend multiplicatively (the GP's
    own multi-DLA model blends the same way), so close pairs are supported.

    Parameters
    ----------
    coadd_in_path : str
        Source coadd (desispec ``Spectra`` FITS).
    coadd_out_path : str
        Destination path for the injected coadd.
    injections : iterable of mappings
        Each record: ``target_id`` (int), ``logN_true`` (log10 N_HI),
        ``z_true`` (absorber z), optional ``num_lines``.
    num_lines : int
        Default Lyman-series line count (matches the run's NUM_FOREST_LINES).
        Used when a record omits ``num_lines``.
    blend_report : list, optional
        If given, one diagnostic dict per attempted injection is APPENDED:
        ``{target_id, z_true, logN_true, forest_flux_frac, forest_blend}``.
        ``forest_blend`` is True when the PRE-injection forest flux at the trough
        centre is below ``_FOREST_BLEND_FRAC`` of the local pseudo-continuum (a
        pre-existing near-black forest/blend that would masquerade as the injected
        LLS response). Flagged, never dropped.

    Returns
    -------
    list[int]
        TARGETIDs that received at least one injection (those found in the coadd).
    """
    import desispec.io

    spec = desispec.io.read_spectra(coadd_in_path)
    recs = _normalize_injections(injections)

    # Map TARGETID -> fibermap row index.
    fib_tid = np.asarray(spec.fibermap["TARGETID"], dtype=np.int64)
    tid_to_row = {int(t): i for i, t in enumerate(fib_tid)}

    injected = []
    for rec in recs:
        tid = rec["target_id"]
        row = tid_to_row.get(tid)
        if row is None:
            # Fiber not in this coadd; skip (caller groups by healpix, so this is
            # a defensive guard, not the normal path).
            continue
        nlines = rec["num_lines"] if rec["num_lines"] is not None else int(num_lines)
        if rec["logN_true"] is None or rec["z_true"] is None:
            # R-041C mean-flux-only record (no absorber): handled by the noise-preserving
            # branch below; nothing to blend-check or multiply here.
            injected.append(tid)
            continue
        nhi_linear = 10.0 ** rec["logN_true"]

        # Forest-blend guard: measure the PRE-injection forest flux fraction at
        # the trough centre on whichever camera covers it (before we overwrite).
        frac = np.nan
        for cam in spec.bands:
            wave = np.asarray(spec.wave[cam], dtype=np.float64)
            cam_frac = _forest_flux_fraction(
                wave, spec.flux[cam][row], rec["z_true"]
            )
            if np.isfinite(cam_frac):
                frac = cam_frac
                break
        if blend_report is not None:
            blend_report.append(
                {
                    "target_id": int(tid),
                    "z_true": float(rec["z_true"]),
                    "logN_true": float(rec["logN_true"]),
                    "forest_flux_frac": float(frac),
                    "forest_blend": bool(
                        np.isfinite(frac) and frac < _FOREST_BLEND_FRAC
                    ),
                }
            )

        if method == "multiplicative":
            for cam in spec.bands:
                wave = np.asarray(spec.wave[cam], dtype=np.float64)
                flux_row = np.asarray(spec.flux[cam][row], dtype=np.float64)
                spec.flux[cam][row] = inject_voigt(
                    wave, flux_row, nhi_linear, rec["z_true"], num_lines=nlines
                )
        injected.append(tid)

    if method != "multiplicative":
        # R-041 (2026-08-28): noise-preserving injection (injection/noise_preserving.py),
        # applied per camera with that camera's own ivar/mask so the noise variance,
        # masks and grids are untouched; an optional mean-flux-only rescaling of the
        # forest SIGNAL (R-041C high-z extrapolation, ``meanflux`` = {"fiducial", "model",
        # "delta_z"}) is applied in the same operation. All injections on one fiber are
        # applied together (one transmission product; one seeded noise draw per camera).
        from injection.noise_preserving import inject_noise_preserving, meanflux_ratio, taueff

        by_row = {}
        for rec in recs:
            row = tid_to_row.get(rec["target_id"])
            if row is not None:
                by_row.setdefault(row, []).append(rec)
        for row, rr in by_row.items():
            absorbers = [
                {"nhi": 10.0 ** r["logN_true"], "z_dla": r["z_true"],
                 "num_lines": (r["num_lines"] if r["num_lines"] is not None else int(num_lines))}
                for r in rr if r["logN_true"] is not None
            ]
            zq = rr[0].get("z_qso")
            for cam in spec.bands:
                wave = np.asarray(spec.wave[cam], dtype=np.float64)
                flux_row = np.asarray(spec.flux[cam][row], dtype=np.float64)
                ivar_row = np.asarray(spec.ivar[cam][row], dtype=np.float64)
                mask_row = (np.asarray(spec.mask[cam][row]) if spec.mask is not None
                            else np.zeros(flux_row.size, dtype=np.uint32))
                r_mf = None
                if meanflux is not None and zq is not None:
                    dz = float(meanflux.get("delta_z", 0.0))
                    fid = _taueff_spec(meanflux["fiducial"]); alt = _taueff_spec(meanflux["model"])
                    r_mf = meanflux_ratio(wave, float(zq), lambda z: alt(np.asarray(z) + dz), fid)
                h = hashlib.sha256(f"{seed_salt}:{rr[0]['target_id']}:{cam}".encode()).digest()
                seed = int.from_bytes(h[:4], "little")
                spec.flux[cam][row] = inject_noise_preserving(
                    wave, flux_row, ivar_row, mask_row, absorbers,
                    z_qso=zq, r=r_mf, seed=seed, method=method,
                )

    os.makedirs(os.path.dirname(os.path.abspath(coadd_out_path)), exist_ok=True)
    desispec.io.write_spectra(coadd_out_path, spec)
    return injected


# ===========================================================================
# 3. Campaign writer
# ===========================================================================


def _hp_dir(root: str, healpix: int) -> str:
    return os.path.join(root, "spectra-16", str(healpix // 100), str(healpix))


def _manifest_rows(manifest):
    """Yield per-injection dicts from an astropy Table or list-of-dicts manifest."""
    try:
        colnames = manifest.colnames  # astropy Table
    except AttributeError:
        for r in manifest:
            yield dict(r)
        return
    for i in range(len(manifest)):
        yield {c: manifest[c][i] for c in colnames}


def write_campaign(
    manifest,
    clean_table,
    *,
    out_root: str,
    mockdir: str,
    num_lines: int,
    truth_manifest_name: str = "injection_truth.fits",
    method: str = "multiplicative",
    meanflux: Optional[Mapping] = None,
    seed_salt: str = "r041",
):
    """Orchestrate the injectable-tree build for a manifest of injections.

    Groups manifest rows by ``healpix``, reads each source coadd from
    ``{mockdir}/spectra-16/{hp//100}/{hp}/spectra-16-{hp}.fits``, injects the
    rows' absorbers, and writes the injected coadd into the SAME-layout tree
    under ``out_root`` (so the production GP driver scans it unchanged). The
    companion ``truth-16-{hp}.fits`` is carried over (copied) next to each
    injected coadd because ``dlasearch`` reads resolution data from it during
    resample. Finally writes a per-injection truth manifest (one row per
    injection) for the measurement step.

    This function does NOT run the GP — that is the gated SLURM step. It only
    produces the injectable inputs + the truth manifest.

    Parameters
    ----------
    manifest : astropy.table.Table or sequence of mappings
        One row per injection. Schema (campaign_grid): ``inj_id, campaign,
        method, target_id, healpix, z_qso, snr_bin, native_snr, logN_true,
        z_true, num_lines`` [+ optional close-pair fields].
    clean_table : astropy.table.Table or None
        The clean-sightline table (currently unused for the write; kept in the
        signature so callers pass provenance / future cross-checks).
    out_root : str
        Root of the injectable tree to create (``{out_root}/spectra-16/...``).
    mockdir : str
        Root of the SOURCE 2LPT mock tree (``{mockdir}/spectra-16/...``).
    num_lines : int
        Default Lyman-series line count (per-row ``num_lines`` overrides).
    truth_manifest_name : str
        Filename for the per-injection truth manifest written under ``out_root``.

    Returns
    -------
    str
        Path to the written truth manifest.
    """
    rows = list(_manifest_rows(manifest))

    # Group injections by healpix.
    by_hp: dict = {}
    for r in rows:
        by_hp.setdefault(int(r["healpix"]), []).append(r)

    # Collect per-injection forest-blend diagnostics keyed by (healpix, target_id)
    # so they can be merged back onto the truth-manifest rows. (Keyed by target id
    # within a healpix because a fiber appears once per source coadd.)
    blend_by_key: dict = {}

    for healpix, hp_rows in by_hp.items():
        src_dir = _hp_dir(mockdir, healpix)
        src_coadd = os.path.join(src_dir, f"spectra-16-{healpix}.fits")
        if not os.path.exists(src_coadd):
            raise FileNotFoundError(
                f"source coadd not found for healpix {healpix}: {src_coadd}"
            )

        dst_dir = _hp_dir(out_root, healpix)
        os.makedirs(dst_dir, exist_ok=True)
        dst_coadd = os.path.join(dst_dir, f"spectra-16-{healpix}.fits")

        injections = []
        for r in hp_rows:
            # CONTROL rows carry logN_true = NaN (clean no-injection sightlines that
            # measure b_FP).  They must NOT be injected: 10**NaN = NaN would blank the
            # whole control fiber to all-NaN, the GP then crashes on it ("All-NaN slice
            # → error flag"), it never reaches the dlacat, and b_FP collapses to a
            # FAKE zero (referee finding 2026-06-11).  Skip them so the control fiber
            # stays the clean source flux the GP is supposed to score for false positives.
            if bool(r.get("control", False)) or not np.isfinite(r.get("logN_true", np.nan)):
                continue
            nl = int(r["num_lines"]) if r.get("num_lines") is not None else None
            injections.append(
                {
                    "target_id": int(r["target_id"]),
                    "logN_true": float(r["logN_true"]),
                    "z_true": float(r["z_true"]),
                    "num_lines": nl,
                    "z_qso": (float(r["z_qso"]) if r.get("z_qso") is not None else None),
                }
            )
            # Campaign-B close pair: the SECOND absorber rides the SAME sightline.
            # inject_into_coadd blends multiple records on one fiber multiplicatively
            # (the GP's own multi-DLA model blends the same way), so emit it as a
            # second injection record rather than dropping it.  (validate_manifest
            # keeps ONE manifest row per target_id; the pair is two physical Voigt
            # absorbers on that single spectrum — exactly a close pair.)
            lN2 = r.get("logN_true2")
            z2 = r.get("z_true2")
            if lN2 is not None and z2 is not None and np.isfinite(lN2) and np.isfinite(z2):
                injections.append(
                    {
                        "target_id": int(r["target_id"]),
                        "logN_true": float(lN2),
                        "z_true": float(z2),
                        "num_lines": nl,
                        "z_qso": (float(r["z_qso"]) if r.get("z_qso") is not None else None),
                    }
                )
        blend_report: list = []
        inject_into_coadd(
            src_coadd, dst_coadd, injections, num_lines=num_lines,
            blend_report=blend_report, method=method, meanflux=meanflux,
            seed_salt=seed_salt,
        )
        for rep in blend_report:
            key = (healpix, int(rep["target_id"]))
            prev = blend_by_key.get(key)
            if prev is not None:
                # A Campaign-B close pair injects TWO records on one fiber; aggregate
                # their blend diagnostics over the sightline (most-blended wins) so
                # the truth manifest reflects BOTH absorbers, not just the last.
                rep = {
                    "target_id": rep["target_id"],
                    "z_true": rep["z_true"],
                    "logN_true": rep["logN_true"],
                    "forest_flux_frac": float(
                        np.nanmin([prev["forest_flux_frac"], rep["forest_flux_frac"]])
                    ),
                    "forest_blend": bool(prev["forest_blend"] or rep["forest_blend"]),
                }
            blend_by_key[key] = rep

        # Carry over the truth-16 companion (resolution data for resample).
        src_truth = os.path.join(src_dir, f"truth-16-{healpix}.fits")
        if os.path.exists(src_truth):
            dst_truth = os.path.join(dst_dir, f"truth-16-{healpix}.fits")
            shutil.copyfile(src_truth, dst_truth)

    # Write the per-injection truth manifest (with the forest-blend flags merged).
    os.makedirs(out_root, exist_ok=True)
    truth_path = os.path.join(out_root, truth_manifest_name)
    _write_truth_manifest(manifest, rows, truth_path, blend_by_key=blend_by_key)
    return truth_path


def _write_truth_manifest(manifest, rows, truth_path: str, *, blend_by_key=None):
    """Write a per-injection truth manifest (FITS) preserving int64 ids.

    ``blend_by_key`` (optional) maps ``(healpix, target_id) -> blend diagnostic``
    so the per-injection ``forest_blend`` / ``forest_flux_frac`` columns are
    written alongside the campaign columns (the M3 forest-blend guard). Rows with
    no diagnostic (e.g. control rows, or fibers absent from a coadd) get
    ``forest_blend=False`` / ``forest_flux_frac=nan``.
    """
    from astropy.table import Table

    # If the manifest is already an astropy Table, write it directly (preserves
    # dtypes incl. int64 target_id/healpix); else rebuild from row dicts.
    if hasattr(manifest, "colnames"):
        tbl = Table(manifest, copy=True)
    else:
        if not rows:
            tbl = Table()
        else:
            # Rows can be HETEROGENEOUS: Campaign-B pair rows carry the close-pair
            # fields (logN_true2/z_true2/dv_kms/_dlogN) that control rows (and
            # Campaign-A rows) lack.  Build the column set as the UNION over all
            # rows, in first-seen order, and fill a missing value with NaN (numeric)
            # so a mixed pair+control manifest writes without KeyError.
            cols = []
            for r in rows:
                for k in r.keys():
                    if k not in cols:
                        cols.append(k)
            data = {c: [r.get(c, np.nan) for r in rows] for c in cols}
            tbl = Table(data)
            for idcol in ("target_id", "healpix", "inj_id"):
                if idcol in tbl.colnames:
                    tbl[idcol] = np.asarray(tbl[idcol], dtype=np.int64)

    # Merge the forest-blend guard columns (one lookup per manifest row).
    if (
        blend_by_key is not None
        and len(tbl) > 0
        and "target_id" in tbl.colnames
        and "healpix" in tbl.colnames
    ):
        tids = np.asarray(tbl["target_id"], dtype=np.int64)
        hpx = np.asarray(tbl["healpix"], dtype=np.int64)
        frac = np.full(len(tbl), np.nan, dtype=np.float64)
        flag = np.zeros(len(tbl), dtype=bool)
        for i in range(len(tbl)):
            rep = blend_by_key.get((int(hpx[i]), int(tids[i])))
            if rep is not None:
                frac[i] = rep["forest_flux_frac"]
                flag[i] = bool(rep["forest_blend"])
        tbl["forest_flux_frac"] = frac
        tbl["forest_blend"] = flag

    if truth_path.endswith(".fits"):
        tbl.write(truth_path, overwrite=True, format="fits")
    else:
        tbl.write(truth_path, overwrite=True)
    return truth_path


def verify_coadd_consistency(coadd_in_path, coadd_out_path, injections, *,
                             num_lines, atol=0.01):
    """Assert the per-camera injection survives the GP's OWN coadd faithfully.

    Reads the original + injected coadds, runs the SAME ``desispec.coadd_cameras``
    the GP runs, and checks ``coadd(injected) == T(λ)·coadd(original)`` on the
    injected fibers — i.e. injecting per camera then letting the GP coadd is
    equivalent to injecting into the coadd (``T`` factors out of the ivar-weighted
    sum). Returns the worst-case ``max|coadd_out/coadd_in − T|`` over injected,
    absorbed, well-determined pixels (≈ float precision when consistent).

    Parameters
    ----------
    injections : mapping ``target_id -> [(N_HI_linear, z_dla), ...]``
        The absorbers injected into each fiber (LINEAR N_HI, as ``inject_voigt`` takes).
    """
    import desispec.io
    from desispec.coaddition import coadd_cameras

    def _read(path):
        sp = desispec.io.read_spectra(path)
        # Mock coadds have misaligned per-camera grids (no RESOLUTION HDU); the GP
        # resolves this with the truth-16 resample, and raw coadd_cameras() raises.
        # Fall back to a PER-CAMERA check (flux_out/flux_in == T(λ) in each band) —
        # the coadd-equivalence T·coadd(f)=coadd(f·T) is algebraic + M4-round-trip-tested.
        try:
            return coadd_cameras(sp)
        except Exception:
            return sp

    sp_in = _read(coadd_in_path)
    sp_out = _read(coadd_out_path)
    tids = np.asarray(sp_in.fibermap["TARGETID"], dtype=np.int64)
    worst = 0.0
    for tid, absorbers in injections.items():
        idx = np.where(tids == int(tid))[0]
        if idx.size == 0:
            continue
        i = int(idx[0])
        for band in sp_in.bands:
            wave = sp_in.wave[band]
            T = np.ones_like(wave, dtype=float)
            for nhi, z in absorbers:
                T = inject_voigt(wave, T, float(nhi), float(z), num_lines)
            fi = np.asarray(sp_in.flux[band][i], float)
            fo = np.asarray(sp_out.flux[band][i], float)
            m = (np.abs(fi) > 1e-3) & (T < 0.999)  # absorbed + well-determined pixels
            if m.any():
                worst = max(worst, float(np.max(np.abs(fo[m] / fi[m] - T[m]))))
    if worst > atol:
        raise AssertionError(
            f"coadd consistency FAILED: max|coadd_out/coadd_in − T| = {worst:.4g} "
            f"> atol={atol}. The per-camera injection does not survive coadd_cameras."
        )
    return worst
