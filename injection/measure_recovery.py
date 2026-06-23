#!/usr/bin/env python
"""Measure recovered-vs-INJECTED truth from a GP run on an injectable tree:
detection completeness C_det and N_HI bias b_N, then render diagnostic figures.
Step 3 of the campaign (see injection/README.md).

Non-circular by construction: recovery is scored against the INJECTION manifest
(inj_id → injected logN_true/z_true), never the natural 2LPT truth.

The off-diagonal response matrix R + b_FP (``measurements.response_matrix``) needs
the per-injection posterior DEPOSIT (a dense (N,z) mass list), which the dlacat
MAP-only recovery here does not carry — R is built in a separate step from the MAP
recovery + NHI_ERR + P_DLA (see README §5). This script reports C_det + bias only.
"""
import argparse, os, sys, glob
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))   # repo root (CDDF_analysis, gpy_dla_detection)
sys.path.insert(0, _HERE)                        # injection modules
from measurements import detection_completeness, nhi_bias


def _assemble_recovered(manifest, rows_iter):
    """Assemble recovered records keyed by inj_id from GP rows (pure, no I/O).

    ``manifest`` — sequence of mappings with ``target_id`` + ``inj_id``.
    ``rows_iter`` — iterable yielding ``(target_id, p_dla, logN_rec, z_rec)`` per
    GP-processed sightline.

    Returns ``{inj_id: {inj_id, p_dla, logN_rec, z_rec}}``.

    Guards against the silent **join-collapse** the campaign depends on NOT
    happening (M3 blocker): the TARGETID→inj_id map is one-to-one ONLY when the
    manifest assigns each clean sightline at most one injection (enforced by
    ``campaign_grid.validate_manifest``).  If the manifest reuses a target_id, the
    map would collapse several inj_ids onto one and recovery would be scored against
    the wrong injected truth — so we RAISE instead of silently dropping.  We also
    assert every matched GP row produced a DISTINCT recovered entry (a duplicate
    target_id in the GP output, e.g. a sightline processed twice, would otherwise
    overwrite a recovered record).  Partial coverage (fewer GP rows than manifest
    rows, e.g. a truncated/un-run healpix) is allowed — only collapse is fatal.
    """
    tid2inj = {}
    for r in manifest:
        tid = int(r["target_id"])
        if tid in tid2inj:
            raise ValueError(
                f"manifest reuses target_id {tid} (inj_id {tid2inj[tid]} and "
                f"{int(r['inj_id'])}) — the TARGETID→inj_id join would collapse and "
                f"recovery would score against the wrong injected truth. Each clean "
                f"sightline must host at most one injection (see validate_manifest)."
            )
        tid2inj[tid] = int(r["inj_id"])

    recovered = {}
    n_matched = 0
    for tid, p_abs, lognhi, zdla in rows_iter:
        inj_id = tid2inj.get(int(tid))
        if inj_id is None:
            continue  # GP processed a sightline not in the manifest — ignore
        n_matched += 1
        if inj_id in recovered:
            raise ValueError(
                f"two GP rows matched inj_id {inj_id} (target_id {int(tid)} appears "
                f"more than once in the GP output) — recovery would be overwritten."
            )
        recovered[inj_id] = dict(inj_id=inj_id, p_dla=float(p_abs),
                                 logN_rec=float(lognhi), z_rec=float(zdla))
    # Every matched GP row must have produced exactly one recovered record.
    assert len(recovered) == n_matched, (
        f"join collapsed: {n_matched} matched GP rows but only {len(recovered)} "
        f"recovered records"
    )
    return recovered


def _iter_processed_rows(processed_dir):
    """Yield ``(target_id, p_dla, logN_rec, z_rec)`` from every processed-*.h5.

    Single-absorber model: ``model_posteriors[:,1]`` = p(absorber);
    ``MAP_log_nhis`` / ``MAP_z_dlas`` are the recovered N_HI / z.  Truncated or
    schema-incomplete files are skipped (they yield nothing), so partial coverage
    degrades gracefully rather than crashing the measurement.
    """
    import h5py
    files = sorted(glob.glob(os.path.join(processed_dir, "processed-*-*.h5")))
    for fp in files:
        try:
            with h5py.File(fp, "r") as f:
                tids = np.asarray(f["target_ids"][:]).astype(np.int64).ravel()
                mp = f["model_posteriors"][:]
                p_abs = mp[:, 1] if mp.shape[1] >= 2 else mp[:, -1]
                lognhi = np.asarray(f["MAP_log_nhis"][:]).reshape(len(tids), -1)[:, 0]
                zdla = np.asarray(f["MAP_z_dlas"][:]).reshape(len(tids), -1)[:, 0]
        except (OSError, KeyError, ValueError, IndexError, TypeError):
            # Truncated / schema-incomplete / wrong-shape file → skip just this one
            # (a single bad healpix must not abort the whole measurement).
            continue
        for i, tid in enumerate(tids):
            yield int(tid), float(p_abs[i]), float(lognhi[i]), float(zdla[i])


def _dlacat_rows_from_table(tbl):
    """Yield ``(target_id, p_dla, logN_rec, z_rec)`` from a dlacat astropy Table.

    The mock GP run (``desi-DLAGP.py``) writes a ``dlacat-*.fits`` — ONE row per
    DETECTED absorber — with columns ``TARGETID, P_DLA, NHI, Z_DLA`` (README §5: the
    recovery source is the dlacat MAP recovery, no dense posterior storage).  A
    sightline that produced no detection is simply absent (scored as a non-detection
    by ``detection_completeness``).  If a TARGETID appears more than once (a
    multi-DLA dlacat), keep its HIGHEST-``P_DLA`` row so the single-absorber join
    stays one-to-one.
    """
    tid = np.asarray(tbl["TARGETID"]).astype(np.int64)
    p = np.asarray(tbl["P_DLA"], dtype=float)
    nhi = np.asarray(tbl["NHI"], dtype=float)
    z = np.asarray(tbl["Z_DLA"], dtype=float)
    best = {}
    for i in range(len(tid)):
        t = int(tid[i])
        if t not in best or p[i] > best[t][0]:
            best[t] = (float(p[i]), float(nhi[i]), float(z[i]))
    for t, (pp, nn, zz) in best.items():
        yield t, pp, nn, zz


def _find_dlacats(path):
    """Resolve ``path`` to a LIST of dlacat FITS: a file → ``[itself]``; a dir → ALL
    its ``dlacat-*.fits`` (a CHUNKED job array writes one dlacat per healpix range,
    e.g. ``dlacat-...-mockcat-0-10.fits``, ``...-10-20.fits`` — read them all)."""
    if os.path.isfile(path) and path.endswith(".fits"):
        return [path]
    return sorted(glob.glob(os.path.join(path, "dlacat-*.fits")))


def _iter_dlacat_rows(dlacat_paths):
    """Yield recovery rows from one or more dlacat FITS (concatenated, then deduped
    by max P_DLA in :func:`_dlacat_rows_from_table`).  Each healpix lives in exactly
    one chunk, so cross-chunk TARGETID collisions don't occur; the global dedup is a
    belt-and-suspenders guard."""
    from astropy.table import Table, vstack
    tabs = [Table.read(p) for p in dlacat_paths]
    if not tabs:
        return iter(())
    combined = vstack(tabs, metadata_conflicts="silent") if len(tabs) > 1 else tabs[0]
    return _dlacat_rows_from_table(combined)


def _load_recovered(processed, manifest):
    """Build recovered records keyed by inj_id by matching GP TARGETID → manifest.

    Recovered-record schema (measurements.py seam): {inj_id, p_dla, logN_rec, z_rec}.
    Accepts EITHER the mock run's ``dlacat-*.fits`` (a file, or a dir containing one
    — the production mock-mode output, README §5) OR a directory of per-healpix
    ``processed-*.h5`` (the legacy posterior-store layout).  Streams the rows
    through :func:`_assemble_recovered` (the join-collapse guard).
    """
    dlacats = _find_dlacats(processed)
    if dlacats:
        return _assemble_recovered(manifest, _iter_dlacat_rows(dlacats)), "dlacat"
    return _assemble_recovered(manifest, _iter_processed_rows(processed)), "processed"


def _pool_completeness_by_logN(cdet, *, prior=(0.5, 0.5)):
    """Pool the per-(logN,z,SNR) C_det cells to a C_det(logN_true) curve.

    SUMS n_injected / n_recovered across z & SNR per logN_true (count pooling, not a
    mean of fractions), then reports the binomial point k/n and a Jeffreys-Beta 68%
    CI on the pooled counts.  Returns (logN, C, C_lo68, C_hi68, n_injected) sorted.
    """
    from scipy.stats import beta as _beta
    lg = np.asarray(cdet["logN_true"], float)
    ninj = np.asarray(cdet["n_injected"], int)
    nrec = np.asarray(cdet["n_recovered"], int)
    out = {}
    for x, ni, ki in zip(lg, ninj, nrec):
        n, k = out.get(float(x), (0, 0))
        out[float(x)] = (n + int(ni), k + int(ki))
    xs = np.array(sorted(out), float)
    n = np.array([out[x][0] for x in xs], int)
    k = np.array([out[x][1] for x in xs], int)
    a0, b0 = float(prior[0]), float(prior[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    lo = _beta.ppf(0.16, k + a0, n - k + b0)
    hi = _beta.ppf(0.84, k + a0, n - k + b0)
    return xs, C, lo, hi, n


def _pool_bias_by_logN(bias):
    """Pool the per-(logN,SNR) N_HI-bias cells to b_N(logN_true) by n_used-weighting.

    Returns (logN, b_N, n_used) sorted by logN_true.  Cells with no survivors carry
    no weight; a logN with zero recovered survivors is omitted (no bias to report).
    """
    lg = np.asarray(bias["logN_true"], float)
    bN = np.asarray(bias["b_N"], float)
    nu = np.asarray(bias["n_used"], int)
    num, den = {}, {}
    for x, b, w in zip(lg, bN, nu):
        if w <= 0 or not np.isfinite(b):
            continue
        num[float(x)] = num.get(float(x), 0.0) + float(b) * int(w)
        den[float(x)] = den.get(float(x), 0) + int(w)
    xs = np.array(sorted(den), float)
    b = np.array([num[x] / den[x] for x in xs], float)
    w = np.array([den[x] for x in xs], int)
    return xs, b, w


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", required=True, help="injectable-tree root (has injection_truth.fits)")
    ap.add_argument("--processed", required=True,
                    help="GP output: the mock-run gp_out dir (or a dlacat-*.fits), "
                         "or a dir of per-healpix processed-*.h5")
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--p_thresh", type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(a.figdir, exist_ok=True)

    from astropy.table import Table
    man = Table.read(os.path.join(a.campaign, "injection_truth.fits"))
    manifest = [dict(zip(man.colnames, row)) for row in man]
    recovered, source = _load_recovered(a.processed, manifest)
    inj_ids = {int(r["inj_id"]) for r in manifest if not bool(r.get("control", False))}
    ctrl_ids = {int(r["inj_id"]) for r in manifest if bool(r.get("control", False))}
    n_inj_rec = len(inj_ids & set(recovered))
    n_ctrl_rec = len(ctrl_ids & set(recovered))
    if source == "dlacat":
        # The dlacat lists only DETECTIONS, so absence = a confident non-detection
        # (NOT an un-run sightline).  Report it that way: injection detections drive
        # C_det; control detections are the false-positive (b_FP) signal — 0 is good.
        print(f"[measure] dlacat: {n_inj_rec}/{len(inj_ids)} injections DETECTED "
              f"({len(inj_ids) - n_inj_rec} non-detections → C_det<1), "
              f"{n_ctrl_rec}/{len(ctrl_ids)} controls detected "
              f"(false positives; 0 ⇒ b_FP≈0). Coverage is implicit — the run "
              f"processed every qsocat target.")
    else:
        print(f"[measure] processed-h5: recovered {len(recovered)}/{len(manifest)} rows "
              f"(injections {n_inj_rec}/{len(inj_ids)}, controls {n_ctrl_rec}/{len(ctrl_ids)})")
        if inj_ids and n_inj_rec < len(inj_ids):
            frac = n_inj_rec / len(inj_ids)
            print(f"[measure] WARNING: {len(inj_ids) - n_inj_rec} injected sightlines "
                  f"({100 * (1 - frac):.1f}%) have NO GP output — check for un-run / "
                  f"truncated healpix before trusting C_det (a coverage gap reads as "
                  f"incompleteness).")

    cdet = detection_completeness(recovered, manifest, p_dla_thresh=a.p_thresh)
    bias = nhi_bias(recovered, manifest, p_dla_thresh=a.p_thresh)

    # Pool the per-(logN,z,SNR) cells down to the headline C_det(logN_true) curve by
    # SUMMING counts across z & SNR (honest pooling, not an average of fractions),
    # then a Jeffreys-Beta CI on the pooled counts.
    cx, cC, clo, chi, cn = _pool_completeness_by_logN(cdet)
    print("[C_det] pooled over z/SNR per injected logN_true:")
    for xi, Ci, ni in zip(cx, cC, cn):
        print(f"   logN={xi:.2f}  n={int(ni):3d}  C_det={Ci:.3f}")

    # Pool the per-(logN,SNR) bias cells to b_N(logN_true) by n_used-weighting.
    bx, bb, bn = _pool_bias_by_logN(bias)
    print("[bias] ⟨logN_rec − logN_true⟩ pooled over SNR per injected logN_true:")
    for xi, bi, ni in zip(bx, bb, bn):
        print(f"   logN={xi:.2f}  n={int(ni):3d}  bias={bi:+.3f}")

    # figures: C_det(logN_true) and N_HI bias(logN_true)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    if cx.size:
        # Clip yerr to ≥0: at k=n the binomial point C=1.0 exceeds the Beta upper
        # quantile (the documented one-sided-uncertainty case), which would make the
        # upper error bar negative.
        yerr = [np.maximum(cC - clo, 0.0), np.maximum(chi - cC, 0.0)]
        ax[0].errorbar(cx, cC, yerr=yerr, fmt="o-", capsize=3)
    ax[0].set_xlabel("log N_HI (injected)")
    ax[0].set_ylabel("detection completeness C_det")
    ax[0].axvline(20.3, ls=":", c="grey")
    ax[0].set_title("C_det (recovered vs INJECTED)")
    ax[0].set_ylim(0, 1.05)
    if bx.size:
        ax[1].plot(bx, bb, "s-")
    ax[1].axhline(0, ls="--", c="grey")
    ax[1].set_xlabel("log N_HI (injected)")
    ax[1].set_ylabel("⟨logN_rec⟩ − logN_true")
    ax[1].set_title("N_HI bias")
    ax[1].axvline(20.3, ls=":", c="grey")
    fig.suptitle("M3 injection pilot — recovery vs injected truth (non-circular)")
    fig.tight_layout()
    out = os.path.join(a.figdir, "injection_pilot_recovery.png")
    fig.savefig(out, dpi=110)
    print("[save] figure:", out)


if __name__ == "__main__":
    main()
