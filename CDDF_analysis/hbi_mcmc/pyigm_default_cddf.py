#!/usr/bin/env python
"""pyigm_default_cddf.py — the EXACT default `pyigm` f(N_HI, X) evaluated on the Paper-1
reporting bins with the Paper-1 redshift / path weighting (R-034, request package
2026-08-28, priority 1).

What "default" means here (established from the package, not from memory):
  * entry point: `pyigm.fN.fnmodel.FNModel.default_model()` — the only documented default
    f(N) constructor (used by pyigm's own plots, tau_eff and mcmc modules);
  * model: 'Hspline' — a monotone (PCHIP) Hermite spline of log10 f(N,X) at z = 2.4 through
    121 pivots (log N_HI = 12.0 … 24.0, 0.1 dex) read from `pyigm/data/fN/fN_spline_z24.fits.gz`
    ("default fN_model from Prochaska+14 … tested against XIDL code by JXP on 09 Nov 2014");
  * redshift evolution: log f(N,X)(z) = log f(N,X)(2.4) + gamma log10((1+z)/(1+2.4)) with
    gamma = 1.5, N-independent; declared validity zmnx = (0.5, 3.0) — `evaluate` REFUSES
    z outside it (and, at the master HEAD, drops into pdb first);
  * cosmology: astropy FlatLambdaCDM(H0=70, Om0=0.3) ("adopted in P14"); f(N,X) is per unit
    N_HI (cm^2) per unit absorption distance X in THAT cosmology.
Nothing is fitted or renormalised. Where the Paper-1 path lies above z = 3.0 (five of the
fifteen fine cells, the whole [3.0, 3.5) slice) the package cannot be evaluated on the
ordinary path; those values are produced by applying the model's OWN stated (1+z)^gamma form
beyond its declared range through an explicit zmnx override and are FLAGGED as such — a
documented extrapolation, never silent.

Matched evaluation: per 0.2-dex reporting bin, f_bin = int_bin f(N,X) dN / Delta N (200-point
trapezoid in log N, identical to the paper's `bin_average_per_N`), per fine redshift cell at the
cell centre, then path-weighted with the frozen pack's dX per cell (the weights the Paper-1
all-z / slice estimators use). The within-cell redshift variation is measured and reported
(sub-cell quadrature) rather than assumed away.

Cosmology convention: the package's f(N,X) is per unit X_P14 (Om = 0.3); Paper 1 uses
Om = 0.279 for its dX. f(N,X)_ours = f(N,X)_P14 x (dX/dz)_P14 / (dX/dz)_ours is a pure
convention conversion (not a renormalisation to data); both the as-returned and the
converted arrays are emitted and the per-cell factors recorded.

Runs in the isolated `pyigm-r034` environment (no science-repo imports). Real-data VALUES
never enter this file.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

Z_SLICES = [("z2.0-2.5", 2.0, 2.5), ("z2.5-3.0", 2.5, 3.0), ("z3.0-3.5", 3.0, 3.5)]
OM_PAPER1 = 0.279
OM_P14 = 0.3
NBINAVG = 200


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(cwd, *args):
    try:
        return subprocess.check_output(["git", *args], cwd=cwd).decode().strip()
    except Exception:
        return "unknown"


def dXdz(z, om):
    return (1.0 + z) ** 2 / np.sqrt(om * (1.0 + z) ** 3 + 1.0 - om)


def bin_average_per_N(fn_logf, lo, hi, n=NBINAVG):
    """int_bin f dN / Delta N for f given as log10 f(logN) (paper's convention, trapezoid in logN)."""
    out = []
    for a, b in zip(lo, hi):
        g = np.linspace(a, b, n)
        N = 10.0 ** g
        f = 10.0 ** fn_logf(g)
        out.append(np.trapezoid(f * N * np.log(10.0), g) / (10.0 ** b - 10.0 ** a))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-cddf-npz", required=True)
    ap.add_argument("--paper-cddf-z-npz", required=True)
    ap.add_argument("--pack", required=True, help="frozen real pack v2 (dX per fine z cell = the Paper-1 weights)")
    ap.add_argument("--pyigm-src", required=True, help="the pinned pyigm checkout the environment imports")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    import pyigm
    from pyigm.fN.fnmodel import FNModel
    from astropy import cosmology
    # the ordinary path installs a COPY into site-packages (pyigm's fnmodel.py locates its data
    # with imp.find_module, which an editable install breaks); pin the copy to the checkout by hash
    src = os.path.realpath(a.pyigm_src)
    installed = os.path.dirname(pyigm.__file__)
    data_file = os.path.join(installed, "data", "fN", "fN_spline_z24.fits.gz")
    for rel in ("fN/fnmodel.py", "data/fN/fN_spline_z24.fits.gz"):
        assert _sha(os.path.join(installed, rel)) == _sha(os.path.join(src, "pyigm", rel)), f"installed {rel} differs from the pinned checkout"
    prov = {
        "package": "pyigm (https://github.com/pyigm/pyigm)",
        "not_on_pypi": "pip cannot find 'pyigm' on PyPI (checked 2026-08-28); the documented install paths are docs/install.rst: `pip install --no-deps git+https://github.com/pyigm/pyigm.git` or `git clone` + `python setup.py develop` — i.e. the master HEAD",
        "commit": _git(src, "rev-parse", "HEAD"), "commit_date": _git(src, "log", "-1", "--format=%ci"),
        "describe": _git(src, "describe", "--tags", "--always"), "latest_tag": "v1.0",
        "setup_py_version_string": "0.1.dev0 (setup.py; the package defines no __version__)",
        "default_identical_at_v1p0": "yes — `git diff v1.0..HEAD -- pyigm/fN/fnmodel.py` touches only docstrings, a pdb.set_trace() before the zmnx ValueError, z clipping at 0 and mfp(); the data file is unchanged since 2015-12-08 (commit 9384258)",
        "install": "conda env pyigm-r034 (python 3.10, numpy/scipy/astropy from conda-forge) + `pip install linetools` + `pip install --no-deps <checkout>` (non-editable copy, as the documented path; an editable install breaks pyigm's imp.find_module data lookup); installed copy hash-pinned to the checkout; freeze in ENV_FREEZE.txt",
        "installed_location": installed,
        "entry_point": "pyigm.fN.fnmodel.FNModel.default_model(use_mcmc=False, cosmo=None)",
        "alternative_entry_points": "default_model(use_mcmc=True) needs a private MCMC chain under $DROPBOX_DIR (not distributed) — NOT an ordinary user path; FNModel('Gamma') requires the user to supply Inoue+14-style parameters — no default. The ordinary path is unambiguous.",
        "data_file": data_file, "data_file_sha256": _sha(data_file),
        "traced_literature_model": "Prochaska, Madau, O'Meara & Fumagalli 2014, MNRAS 438, 476 (P14) — per the code docstring ('default fN_model from Prochaska+14'); the FITS file itself carries no citation header (MWRFITS dummy header), so the attribution rests on the code",
    }
    # ---- instantiate the default exactly as a user does ----
    m = FNModel.default_model()
    cosmo_p14 = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
    prov.update({
        "model_type": m.fN_mtype, "zmnx_declared": list(map(float, m.zmnx)), "zpivot": float(m.zpivot), "gamma": float(m.gamma),
        "n_pivots": int(m.npivot), "pivots_logN_min_max": [float(m.pivots.min()), float(m.pivots.max())],
        "pivots_logN": np.asarray(m.pivots, float).round(6).tolist(), "log10_fNX_at_pivots_z2p4": np.asarray(m.param["sply"], float).tolist(),
        "spline": "scipy.interpolate.PchipInterpolator(pivots, sply) on log10 f(N,X); f in cm^2 per unit N per unit X",
        "redshift_evolution": "log10 f(N,X)(z) = spline(logN) + gamma*log10((1+z)/(1+zpivot)), gamma = 1.5 (N-independent); refused outside zmnx",
        "cosmology": {"as_coded": "astropy.cosmology.FlatLambdaCDM(H0=70, Om0=0.3) ('Adopted in P14')", "H0": 70.0, "Om0": 0.3,
                      "in_model_object": str(getattr(m, "cosmo", cosmo_p14))},
    })
    # a model with the identical spline/params but the validity range extended to cover the Paper-1 support
    m_ext = FNModel("Hspline", zmnx=(0.5, 3.5), pivots=np.asarray(m.pivots, float), param={"sply": np.asarray(m.param["sply"], float)},
                    zpivot=m.zpivot, gamma=m.gamma, cosmo=cosmo_p14)
    g = np.linspace(19.7, 22.4, 400)
    for z in (0.6, 1.5, 2.4, 2.95, 3.0):
        assert np.allclose(m.evaluate(g, z), m_ext.evaluate(g, z), rtol=0, atol=1e-12), "extended-range copy differs from the default inside zmnx"

    # ---- Paper-1 bins, weights ----
    C = np.load(a.paper_cddf_npz, allow_pickle=True)
    Cz = np.load(a.paper_cddf_z_npz, allow_pickle=True)
    P = np.load(a.pack, allow_pickle=True)
    lo, hi = np.asarray(C["nhi_lo"], float), np.asarray(C["nhi_hi"], float)
    tier = [str(t) for t in C["tier"]]
    zf = np.asarray(P["zf_edges"], float)
    dXk = np.asarray(P["dX"], float).sum(axis=1)
    zc = 0.5 * (zf[:-1] + zf[1:])
    med_allz = np.asarray(C["f_per_linear_N"], float)[:, 2]
    med_slice = {k: np.asarray(Cz[f"s{i+1}_f_per_linear_N"], float)[:, 2] for i, (k, _, _) in enumerate(Z_SLICES)}
    for i, (k, zl, zh) in enumerate(Z_SLICES):
        assert str(Cz[f"s{i+1}_key"][0]) == k and float(Cz[f"s{i+1}_z_lo"][0]) == zl and float(Cz[f"s{i+1}_z_hi"][0]) == zh
        assert np.allclose(np.asarray(Cz[f"s{i+1}_nhi_lo"], float), lo)
    domains = {"allz": (2.0, 3.5)}
    domains.update({k: (zl, zh) for k, zl, zh in Z_SLICES})

    # ---- per-cell evaluation ----
    fbin_cell = np.zeros((len(zc), len(lo)))            # bin-averaged f(N,X)_P14 per cell
    spine_cell = np.zeros((len(zc), g.size))
    for k, z in enumerate(zc):
        fbin_cell[k] = bin_average_per_N(lambda gg: m_ext.evaluate(gg, z).flatten(), lo, hi)
        spine_cell[k] = 10.0 ** m_ext.evaluate(g, z).flatten()
    in_range = zc <= m.zmnx[1] + 1e-12
    conv = dXdz(zc, OM_P14) / dXdz(zc, OM_PAPER1)      # f per unit X_ours = f per unit X_P14 * conv
    # within-cell redshift variation (sub-cell quadrature weighted by dX/dz in the Paper-1 cosmology)
    sub_diff = []
    for k, z in enumerate(zc):
        zz = np.linspace(zf[k], zf[k + 1], 9)
        w = dXdz(zz, OM_PAPER1)
        fs = np.array([bin_average_per_N(lambda gg: m_ext.evaluate(gg, zv).flatten(), lo, hi) for zv in zz])
        fsub = (fs * w[:, None]).sum(axis=0) / w.sum()
        sub_diff.append(float(np.max(np.abs(fsub / fbin_cell[k] - 1.0))))
    # slice path fractions from the frozen pack vs the paper npz (must agree)
    pf = {}
    for i, (k, zl, zh) in enumerate(Z_SLICES):
        sel = (zc > zl) & (zc < zh)
        pf[k] = float(dXk[sel].sum() / dXk.sum())
        assert abs(pf[k] - float(Cz[f"s{i+1}_path_fraction"][0])) < 1e-9, f"slice path fraction mismatch {k}"

    out_dom = {}
    npz = {"logN_edges": np.concatenate([lo, hi[-1:]]), "nhi_lo": lo, "nhi_hi": hi, "tier_paper1": np.array(tier),
           "spine_logN": g, "zf_edges": zf, "z_cell_centre": zc, "dX_cell": dXk, "cosmology_conversion_factor_cell": conv,
           "cell_inside_pyigm_zmnx": in_range,
           "axis_note": np.array(["per-domain arrays: (13 bins) in the order of nhi_lo; spine arrays: (400) on spine_logN; *_P14X = per unit X in pyigm's cosmology (Om 0.3) as returned; *_paper1X = converted to Paper-1's X convention (Om 0.279) by the per-cell factor"])}
    for name, (zl, zh) in domains.items():
        sel = (zc > zl) & (zc < zh)
        w = dXk * sel
        wsum = w.sum()
        f_p14 = (fbin_cell * w[:, None]).sum(axis=0) / wsum
        f_ours = (fbin_cell * (w * conv)[:, None]).sum(axis=0) / wsum
        sp_p14 = (spine_cell * w[:, None]).sum(axis=0) / wsum
        sp_ours = (spine_cell * (w * conv)[:, None]).sum(axis=0) / wsum
        med = med_allz if name == "allz" else med_slice[name]
        frac_out = float(w[~in_range].sum() / wsum)
        zeff = float((zc * w).sum() / wsum)
        # equivalence check: evaluating once at the path-weighted z_eff vs the path-weighted average
        f_zeff = bin_average_per_N(lambda gg: m_ext.evaluate(gg, zeff).flatten(), lo, hi)
        zeff_diff = float(np.max(np.abs(f_zeff / f_p14 - 1.0)))
        # pyigm's own l(X) integrator at z_eff, and our bin sum, as anchors
        lox_zeff = float(m_ext.calculate_lox(zeff, 20.3, 22.4, cosmo=cosmo_p14)) if True else None
        dndx_bins_p14 = float((f_p14 * (10.0 ** hi - 10.0 ** lo))[lo >= 20.3 - 1e-9].sum())
        dndx_bins_ours = float((f_ours * (10.0 ** hi - 10.0 ** lo))[lo >= 20.3 - 1e-9].sum())
        status = ("publication-ready (software benchmark; no fit, no renormalisation)" if frac_out == 0.0
                  else f"diagnostic — {100 * frac_out:.0f} % of this domain's path lies above pyigm's declared validity zmnx[1] = {m.zmnx[1]}; those cells use the model's own (1+z)^gamma form beyond its range (explicit override, flagged)")
        out_dom[name] = {
            "z_range": [zl, zh], "path_fraction_of_allz": (1.0 if name == "allz" else pf[name]), "z_eff_path_weighted": zeff,
            "fraction_of_path_outside_pyigm_zmnx": frac_out, "cells_used": int(sel.sum()), "status": status,
            "pyigm_binavg_f_per_linear_N_P14X": f_p14.tolist(), "pyigm_binavg_f_per_linear_N_paper1X": f_ours.tolist(),
            "pyigm_binavg_f_per_dex_P14X": (f_p14 * (10.0 ** hi - 10.0 ** lo) / (hi - lo)).tolist(),
            "paper1_median_f_per_linear_N": med.tolist(),
            "ratio_pyigm_over_paper1_P14X": (f_p14 / med).tolist(), "ratio_pyigm_over_paper1_paper1X": (f_ours / med).tolist(),
            "fractional_difference_paper1X": (f_ours / med - 1.0).tolist(),
            "log10_ratio_paper1X": np.log10(f_ours / med).tolist(),
            "flags_per_bin": [{"bin": [float(x), float(y)], "inside_spline_pivots": bool(x >= m.pivots.min() and y <= m.pivots.max()),
                               "z_extrapolated_fraction_of_path": frac_out, "paper1_tier_(ours, NOT folded in)": t} for x, y, t in zip(lo, hi, tier)],
            "within_cell_z_variation_max_rel": float(max(sub_diff[k] for k in np.where(sel)[0])),
            "z_eff_single_evaluation_vs_path_weighted_max_rel": zeff_diff,
            "anchors": {"pyigm_calculate_lox_ge20p3_to_22p4_at_zeff_P14X": lox_zeff, "pyigm_binsum_dndx_ge20p3_P14X": dndx_bins_p14,
                        "pyigm_binsum_dndx_ge20p3_paper1X": dndx_bins_ours,
                        "paper1_dndx_ge20p3_median": (float(C["dndx_allz_20p3"][2]) if name == "allz" else None)},
        }
        npz[f"{name}_pyigm_binavg_f_per_linear_N_P14X"] = f_p14
        npz[f"{name}_pyigm_binavg_f_per_linear_N_paper1X"] = f_ours
        npz[f"{name}_ratio_pyigm_over_paper1_paper1X"] = f_ours / med
        npz[f"{name}_ratio_pyigm_over_paper1_P14X"] = f_p14 / med
        npz[f"{name}_paper1_median_f_per_linear_N"] = med
        npz[f"{name}_spine_f_per_linear_N_P14X"] = sp_p14
        npz[f"{name}_spine_f_per_linear_N_paper1X"] = sp_ours
        npz[f"{name}_fraction_of_path_outside_pyigm_zmnx"] = np.array([frac_out])
        npz[f"{name}_status"] = np.array([status])
    # a plain-z reference: the model at its own pivot and at a few redshifts (for the record)
    ref = {str(z): bin_average_per_N(lambda gg: m_ext.evaluate(gg, z).flatten(), lo, hi).tolist() for z in (2.4, 2.0, 2.5, 3.0, 3.5)}

    env = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
    with open(os.path.join(a.out_dir, "ENV_FREEZE.txt"), "w") as fh:
        fh.write(env)
    out = {
        "role": "R-034 — the exact default pyigm f(N,X) on the Paper-1 reporting bins with the Paper-1 path weighting (software benchmark; sits BESIDE the frozen products; no fit, no renormalisation)",
        "status": "publication-ready for allz-in-range and the slices [2.0,2.5), [2.5,3.0); diagnostic (flagged extrapolation beyond pyigm's zmnx) for [3.0,3.5) and for the above-3.0 part of the all-z path — see per-domain status",
        "written_utc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": {"module": "CDDF_analysis/hbi_mcmc/pyigm_default_cddf.py", "commit": _git(os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"),
                      "argv": sys.argv, "python": sys.version.split()[0], "numpy": np.__version__, "conda_env": os.environ.get("CONDA_DEFAULT_ENV"), "env_freeze": "ENV_FREEZE.txt"},
        "pyigm_provenance": prov,
        "inputs": {k: {"path": p, "sha256": _sha(p)} for k, p in [("paper_cddf_npz", a.paper_cddf_npz), ("paper_cddf_z_npz", a.paper_cddf_z_npz), ("pack", a.pack)]},
        "conventions": {"bins": "the 13 Paper-1 0.2-dex reporting bins [19.7, 22.4) from fig_hbi_cddf.data.npz",
                        "bin_averaging": f"int_bin f(N,X) dN / Delta N, {NBINAVG}-point trapezoid in log N (paper's bin_average_per_N)",
                        "redshift_weighting": "per fine 0.1 cell of the frozen pack, evaluated at the cell centre, weighted by the pack's dX per cell (the Paper-1 all-z / slice estimator weights); within-cell variation measured (see per-domain)",
                        "units": "f per unit N_HI (cm^2) per unit absorption distance X; *_P14X in pyigm's X (Om 0.3, H0 70); *_paper1X converted by (dX/dz)_P14/(dX/dz)_paper1 per cell (Om 0.279)",
                        "cosmology_conversion_factor_range": [float(conv.min()), float(conv.max())],
                        "paper1_reference": "posterior median f per linear N of the same bin (frozen products)",
                        "no_fit_no_renormalisation": True, "tier_metadata": "ours, carried separately, not folded into the prediction"},
        "slice_path_fractions_from_pack": pf,
        "domains": out_dom,
        "pyigm_binavg_at_fixed_z_P14X_for_the_record": ref,
    }
    jp = os.path.join(a.out_dir, "R034_pyigm_default_cddf.json")
    with open(jp, "w") as fh:
        json.dump(out, fh, indent=1)
    np.savez(os.path.join(a.out_dir, "R034_pyigm_default_cddf.npz"), **npz)
    files = [jp, os.path.join(a.out_dir, "R034_pyigm_default_cddf.npz"), os.path.join(a.out_dir, "ENV_FREEZE.txt")]
    with open(os.path.join(a.out_dir, "SHA256SUMS"), "w") as fh:
        for p in files:
            fh.write(f"{_sha(p)}  {os.path.basename(p)}\n")
    print(open(os.path.join(a.out_dir, "SHA256SUMS")).read())
    for name, d in out_dom.items():
        r = np.array(d["ratio_pyigm_over_paper1_paper1X"])
        print(f"{name:9s} z_eff {d['z_eff_path_weighted']:.3f} out-of-range path {100 * d['fraction_of_path_outside_pyigm_zmnx']:.0f}%  ratio pyigm/P1 per bin:", np.round(r, 3).tolist())
        print(f"          within-cell max rel {d['within_cell_z_variation_max_rel']:.2e}; z_eff-single vs weighted max rel {d['z_eff_single_evaluation_vs_path_weighted_max_rel']:.2e}; anchors {d['anchors']}")


if __name__ == "__main__":
    main()
