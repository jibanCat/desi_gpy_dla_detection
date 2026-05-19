"""Stack real-LOA LLS / sub-DLA / DLA detections by NHI bin.

Goal: visually verify that low-NHI detections (LLS log NHI < 19, sub-DLA
19–20.3) show coherent metal-line absorption — the falsifying signature
for false-positive contamination is no coherent metals at any diagnostic
wavelength (especially the CIV 1548/1551 doublet, SiIV 1394/1403) — and,
for the LLS, that the composite shows the expected Lyman limit break
blueward of the 911.76 Å rest-frame H I limit.

Inputs
------
- DLA catalog:  /scratch/.../desi-loa-gpdla-...lls_run-nhi172/dlacat-loa-main-dark.fits
- Spectra:      /scratch/.../loa_archives/loa_full_z2_noR_v2.h5 (LoaArchive HDF5)
- BAL catalog:  QSO_cat_loa_main_dark_healpix_v3-altbal.fits — BI_CIV per
                TARGETID. The DLA catalog and the LoaArchive carry no BAL
                column, so BAL status is joined from here by TARGETID.

Methodology
-----------
- median stack, σ-clip 3σ per rest-frame pixel
- per-spectrum continuum: divide by the median in a flat redward window
  [1410, 1520] Å absorber-rest (fallback [1340, 1380]). This is the
  standard coarse per-spectrum flux normalization done BEFORE stacking
  (cf. Mas-Ribas+2017, York+2006); the composite-level continuum can be
  refined post-stack with a masked spline if EWs are needed later.
- log-λ grid at native ~0.0001 dex (69 km/s), 700–1600 Å absorber rest
- discard rest-frame pixels with <50 contributing spectra
- selection: P_DLA > 0.97, SNR_FOREST > 2, DLAFLAG=0, Z_QSO > 3,
  absorber in Lyα forest, not proximate

Bin sets
--------
- PRODUCTION: LLS merged to a single [17.2, 19) bin + three sub-DLA bins
  + two DLA bins. Used for the headline figures.
- DIAGNOSTIC: the LLS range resolved into three bins [17.2,18) [18,18.5)
  [18.5,19). Used for the extra LLS-resolved diagnostic figure.

BAL split
---------
Every stack is computed twice — non-BAL (BI_CIV = 0) and BAL (BI_CIV > 0)
— and a comparison figure overlays them per bin. BAL troughs sit near the
QSO-frame CIV and can mimic absorption; we usually work with the non-BAL
stack, but the comparison documents how different the BAL sample is.

Lyman limit
-----------
With the rest floor at 700 Å the composite reaches well blueward of the
911.76 Å Lyman limit. A true LLS population shows a coherent flux
decrement turning on at 912 Å; false positives show no edge. NOTE: at the
absorber redshifts here (z_abs ≲ z_QSO, z_QSO > 3) the 700–900 Å rest
region maps to observed λ below the DESI ~3600 Å blue cutoff for all but
the highest-z absorbers, so the deep-blue pixels are sparsely covered and
the <50-spectra cut NaN-clips them — the break is best seen in the
~850–960 Å region, populated by the z_abs ≳ 3 tail.

Control: each LLS / sub-DLA spectrum is ALSO stacked at a scrambled
redshift z + Δz (Δz random ±[0.15, 0.35]) — real metals locked to the
absorber redshift survive the real stack and wash out in the control.
A coherent CIV dip in the real stack but a flat control = real absorbers.

Outputs (docs/notes/2026-05-15_stack_real_loa_dlas/)
  stack_prod.png / stack_metal_zoom_prod.png        — production bins
  stack_lls_diag.png / stack_metal_zoom_lls_diag.png — 3 fine LLS bins
  stack_subdla.png / stack_metal_zoom_subdla.png    — sub-DLA [19, 20.3)
  stack_dla.png / stack_metal_zoom_dla.png          — DLA [20.3, 23)
  stack_lyman_limit.png                             — LL break recovery
  stack_bal_compare.png                             — non-BAL vs BAL
  stack_pseudo_continuum_qc.png                     — continuum-fit QC
  stack_control_lls.png / stack_control_subdla.png  — real vs control
  stack_curves_<purity>.npz   — cached curves + pseudo-continuum (`pcont`)
                                + provenance. Continuum-normalized stack
                                = curve / pcont.
"""
from __future__ import annotations

import json
import sys
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
from numpy.lib import recfunctions as rfn
import h5py
from scipy.interpolate import LSQUnivariateSpline
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DLACAT = "/scratch/cavestru_root/cavestru0/mfho/nersc/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/dlacat-loa-main-dark.fits"
LOA_ARCHIVE = "/scratch/cavestru_root/cavestru0/mfho/nersc/loa_archives/loa_full_z2_noR_v2.h5"
BAL_CATALOG = "/nfs/turbo/lsa-cavestru/mfho/DESI/loa/QSO_cat_loa_main_dark_healpix_v3-altbal.fits"
OUT_DIR = Path(__file__).resolve().parent.parent / "docs/notes/2026-05-15_stack_real_loa_dlas"


def tagged(basename, ext="png"):
    """Output filename tagged with the active purity preset, so the
    `high` and `marginal` runs do not clobber each other's outputs."""
    return f"{basename}_{PURITY}.{ext}"


def npz_path():
    """Path to the cached-curves npz for the active purity preset."""
    return OUT_DIR / tagged("stack_curves", "npz")


def _ensure_outdir():
    """Create the output directory. Called from main() — kept out of
    module import so tests can import the pure functions side-effect-free."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

# Rest-frame stack grid: log-λ, 700–1600 Å (floor extended from 900 → 700
# to reach blueward of the 911.76 Å Lyman limit), dλ ~ 0.6 Å at 1200 Å.
REST_LAMBDA_MIN = 700.0
REST_LAMBDA_MAX = 1600.0
DLOG_LAMBDA = 0.0001  # ~69 km/s, native BOSS/DESI
LYMAN_LIMIT = 911.76  # H I rest-frame Lyman limit (Å)
SIGMA_912 = 6.30e-18  # H I photoionization cross-section at 1 Ryd (cm²)

# --- bin sets -------------------------------------------------------------
# DIAGNOSTIC granularity: the LLS range is resolved into 3 bins. This is
# also the granularity at which spectra are read; the production LLS bin
# is the pooled union of the 3 fine LLS bins (no extra reads).
NHI_BINS = [
    (17.2, 18.0),   # LLS low
    (18.0, 18.5),   # LLS mid
    (18.5, 19.0),   # LLS high
    (19.0, 19.5),   # sub-DLA low
    (19.5, 20.0),   # sub-DLA mid
    (20.0, 20.3),   # sub-DLA high
    (20.3, 21.0),   # DLA mid
    (21.0, 23.0),   # DLA high
]
LLS_MERGED = (17.2, 19.0)            # single production LLS bin
# PRODUCTION binning: LLS merged to one bin.
NHI_BINS_PROD = [LLS_MERGED] + [b for b in NHI_BINS if b[0] >= 19.0]

LLS_BINS_FINE = [b for b in NHI_BINS if b[1] <= 19.0]                 # 3 bins
SUBDLA_BINS   = [b for b in NHI_BINS if b[0] >= 19.0 and b[1] <= 20.3]  # 3 bins
DLA_BINS      = [b for b in NHI_BINS if b[0] >= 20.3]                  # 2 bins

# One stable colour per distinct bin (production + fine LLS bins).
BIN_COLOR = {
    LLS_MERGED:    "#6a5acd",   # production LLS — indigo
    (17.2, 18.0):  "#b0a8d0",   # fine LLS — purples
    (18.0, 18.5):  "#8c7fc0",
    (18.5, 19.0):  "#6a5acd",
    (19.0, 19.5):  "#1f77b4",   # sub-DLA — blue/cyan/green
    (19.5, 20.0):  "#17becf",
    (20.0, 20.3):  "#2ca02c",
    (20.3, 21.0):  "#ff7f0e",   # DLA — orange/red
    (21.0, 23.0):  "#d62728",
}
# Bins for which the curves are persisted: all 8 fine bins + merged LLS.
STORED_BINS = NHI_BINS + [LLS_MERGED]
# Categories needing a redshift-scrambled control (low-NHI = contamination-prone).
# "lownhi" pools LLS + sub-DLA — the pooled stack used by --compare-purity.
CONTROL_CATEGORIES = {
    "lls":    LLS_BINS_FINE,
    "subdla": SUBDLA_BINS,
    "lownhi": LLS_BINS_FINE + SUBDLA_BINS,
}

# Selection — purity is a preset chosen by --purity (default "high",
# which reproduces the original P_DLA > 0.97 behaviour).
PURITY_PRESETS = {           # preset -> (p_lo, p_hi); keep p_lo < P_DLA <= p_hi
    "high":     (0.97, 1.01),
    "marginal": (0.50, 0.70),
}
PURITY = "high"              # module global; set from --purity in main()
SNR_FOREST_MIN = 2.0
Z_QSO_MIN = 3.0
Z_DLA_TO_QSO_MARGIN = 0.05    # exclude proximate absorbers
NHI_MIN = 17.2
NHI_MAX = 23.0
LYA = 1215.67
LYB = 1025.72

# Per-bin sample cap, applied independently to the non-BAL and BAL groups.
# /scratch HDF5 random reads are slow (~0.4s/row); batch-read sorted rows.
MAX_PER_BIN = 800
BATCH_SIZE = 200
# Continuum-window minimum pixel count + min good pixels per spectrum.
# Raised 5 → 30: a continuum from <30 pixels is too noisy to normalize by;
# such spectra are dropped (counted in n_skip_norm) rather than stacked
# with a poorly-determined continuum.
MIN_CONT_PIX = 30
MIN_GOOD_PIX = 100
# Output pixels whose bracketing source samples are farther apart than
# this (Å, absorber rest) fell in a masked gap np.interp bridged → NaN.
MAX_INTERP_GAP = 2.0

# --- pseudo-continuum fit (post-stack) ------------------------------------
# A masked fixed-knot cubic spline fit to each composite, divided out so
# metal lines sit on a flat baseline. See
# docs/superpowers/specs/2026-05-18-stack-pseudo-continuum-design.md
_C_KM_S = 299792.458
PCONT_LAMBDA_MIN = 945.0     # blue end of the spline fit (Å); below this
                             # the Lyman-series crowding / 912 Å break make
                             # the pseudo-continuum undefined.
SIGMA_V = 100.0              # stacked metal-line width budget (km/s):
                             # DESI LSF ~30 ⊕ z_DLA error ~50 (catalog
                             # Z_DLA_ERR median 6.2e-4 → 47 km/s at z=3)
                             # ⊕ metal velocity structure ~80, in quadrature.
K_MASK_SIGMA = 3.0           # metal mask half-width = K_MASK_SIGMA·σ_stack(λ)
HI_MASK_HALF = {             # H I core mask half-widths (Å)
    # Lyα ±12: a ±25 Å mask plus the SiII 1190/93 / NI 1200 / SiIII 1207
    # cluster left the spline unconstrained across ~50 Å and it overshot
    # onto the QSO-Lyα emission bump in the LLS bin (QC, 2026-05-18). ±12
    # covers the LLS/sub-DLA Lyα core; DLA damping wings are removed by
    # the rejection loop, not the static mask.
    "Lyα": 12.0, "Lyβ": 15.0, "Lyγ": 8.0, "Ly4": 5.0, "Ly5": 5.0,
}
KNOT_SPACING = 15.0          # interior knot spacing (Å)
SPLINE_ORDER = 3             # cubic
REJECT_SIGMA = 5.0           # iterative-rejection threshold (robust σ)
MAX_REJECT_ITER = 10         # rejection iteration cap

# Diagnostic metal lines (vacuum rest-frame Å). Lyman series + the strong
# low/intermediate-ion metals in the 1025–1216 Å forest region + the
# clean red-side diagnostics.
METAL_LINES = {
    "Ly5":         937.80,
    "Ly4":         949.74,
    "Lyγ":         972.54,
    "CIII(977)":   977.02,
    "OI(989)":     988.77,
    "Lyβ":         1025.72,
    "OVI(1032)":   1031.91,
    "OVI(1038)":   1037.61,
    "OI(1039)":    1039.23,
    "FeII(1063)":  1063.18,
    "FeII(1097)":  1096.88,
    "FeIII(1123)": 1122.52,
    "FeII(1125)":  1125.45,
    "NI(1134)":    1134.17,
    "FeII(1143)":  1143.23,
    "FeII(1145)":  1144.94,
    "SiII(1190)":  1190.42,
    "SiII(1193)":  1193.29,
    "NI(1200)":    1200.22,
    "SiIII(1207)": 1206.50,
    "Lyα":         1215.67,
    "NV(1239)":    1238.82,
    "NV(1243)":    1242.80,
    "SII(1251)":   1250.58,
    "SII(1254)":   1253.81,
    "SiII(1260)":  1260.42,
    "OI(1302)":    1302.17,
    "SiII(1304)":  1304.37,
    "CII(1335)":   1334.53,
    "CII*(1336)":  1335.71,
    "SiIV(1394)":  1393.76,
    "SiIV(1403)":  1402.77,
    "SiII(1527)":  1526.71,
    "CIV(1548)":   1548.20,
    "CIV(1551)":   1550.78,
}

# Zoom panels: (panel_title, center_λ, half_width_Å, [(line_name, λ), ...])
ZOOM_PANELS = [
    ("Lyγ 972 / CIII 977 / OI 989", 980.0, 24.0,
     [("Lyγ", 972.54), ("CIII", 977.02), ("OI", 988.77)]),
    ("Lyβ 1025.72",             1025.72, 22.0, [("Lyβ", 1025.72)]),
    ("OVI 1032/1038 + OI 1039", 1035.5,  16.0, [("OVI", 1031.91), ("OVI", 1037.61), ("OI", 1039.23)]),
    ("FeII 1063 / 1097",        1080.0,  24.0, [("FeII", 1063.18), ("FeII", 1096.88)]),
    ("FeIII 1123 / FeII 1125 / NI 1134", 1128.0, 16.0,
     [("FeIII", 1122.52), ("FeII", 1125.45), ("NI", 1134.17)]),
    ("FeII 1143/1145",          1144.0, 12.0,
     [("FeII", 1143.23), ("FeII", 1144.94)]),
    ("SiII 1190/1193",          1191.85, 13.0, [("SiII", 1190.42), ("SiII", 1193.29)]),
    ("NI 1200 / SiIII 1207",    1203.4,  13.0, [("NI", 1200.22), ("SiIII", 1206.50)]),
    ("SII 1251/1254 / SiII 1260", 1255.5, 16.0,
     [("SII", 1250.58), ("SII", 1253.81), ("SiII", 1260.42)]),
    ("OI 1302 / SiII 1304",     1303.3,  16.0, [("OI", 1302.17), ("SiII", 1304.37)]),
    ("CII 1335 / CII* 1336",    1335.1,  16.0, [("CII", 1334.53), ("CII*", 1335.71)]),
    ("SiIV 1394/1403",          1398.3,  18.0, [("SiIV", 1393.76), ("SiIV", 1402.77)]),
    ("SiII 1526.71",            1526.71, 16.0, [("SiII", 1526.71)]),
    ("CIV 1548/1551",           1549.5,  16.0, [("CIV", 1548.20), ("CIV", 1550.78)]),
]

# The most diagnostic metal-line panels for the marginal-vs-high purity
# comparison (a curated subset of ZOOM_PANELS).
_PURITY_COMPARE_TITLES = {
    "SiIV 1394/1403", "OI 1302 / SiII 1304", "CII 1335 / CII* 1336",
    "SII 1251/1254 / SiII 1260", "CIV 1548/1551",
}
PURITY_COMPARE_PANELS = [p for p in ZOOM_PANELS
                         if p[0] in _PURITY_COMPARE_TITLES]

# Metal species to suppress (markers + labels + zoom panels) per category.
EXCLUDE_SPECIES = {
    "all":    frozenset(),
    "lls":    frozenset(),
    "subdla": frozenset(),
    "dla":    frozenset({"CIII"}),
}

# Per-bin stack: non-BAL (curve/counts/n/pcont) + BAL (..._bal). `pcont`
# is the fitted pseudo-continuum; the continuum-normalized stack is
# `curve / pcont`.
BinStack = namedtuple(
    "BinStack", ["curve", "counts", "n", "pcont",
                 "curve_bal", "counts_bal", "n_bal", "pcont_bal"])


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def provenance_dict() -> dict:
    """The catalog paths + selection cuts + grid params that define the
    stack. Embedded in the npz so `--plot-only` can refuse a stale cache."""
    return {
        "dlacat": DLACAT,
        "loa_archive": LOA_ARCHIVE,
        "bal_catalog": BAL_CATALOG,
        "purity": PURITY,
        "p_dla_range": list(PURITY_PRESETS[PURITY]),
        "snr_forest_min": SNR_FOREST_MIN,
        "z_qso_min": Z_QSO_MIN,
        "z_dla_to_qso_margin": Z_DLA_TO_QSO_MARGIN,
        "nhi_min": NHI_MIN,
        "nhi_max": NHI_MAX,
        "max_per_bin": MAX_PER_BIN,
        "rest_lambda_min": REST_LAMBDA_MIN,
        "rest_lambda_max": REST_LAMBDA_MAX,
        "dlog_lambda": DLOG_LAMBDA,
        "min_cont_pix": MIN_CONT_PIX,
        "nhi_bins": [list(b) for b in NHI_BINS],
    }


def check_provenance(stored: dict, expect_preset: str = None) -> None:
    """Compare a cached npz's provenance to the current constants. With
    `expect_preset` set (used by --compare-purity), require the stored
    `purity` to equal it and compare all OTHER fields against the
    current settings; without it, every field — purity included — must
    match. Raise on mismatch unless `--force-plot` is passed."""
    current = provenance_dict()
    mismatches = []
    if expect_preset is not None:
        if stored.get("purity") != expect_preset:
            mismatches.append(f"  purity: cached={stored.get('purity')!r}  "
                              f"expected={expect_preset!r}")
        skip = {"purity", "p_dla_range"}
    else:
        skip = set()
    for key, cur_val in current.items():
        if key in skip:
            continue
        old_val = stored.get(key, "<absent>")
        if old_val != cur_val:
            mismatches.append(f"  {key}: cached={old_val!r}  current={cur_val!r}")
    if mismatches:
        msg = ("cached npz was built with different settings "
               "than the current script:\n" + "\n".join(mismatches))
        if "--force-plot" in sys.argv:
            print(f"[WARN] {msg}\n[WARN] --force-plot given; plotting anyway.",
                  flush=True)
        else:
            raise SystemExit(
                f"[ERROR] {msg}\n"
                "Re-run to regenerate, or pass --force-plot to plot the "
                "stale cache anyway.")
    else:
        print("provenance check: cached npz matches current settings.",
              flush=True)


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------

def load_catalog():
    print(f"loading catalog: {DLACAT}", flush=True)
    with fits.open(DLACAT) as f:
        cat = np.array(f["DLACAT"].data)
    print(f"  raw rows: {len(cat)}", flush=True)
    return cat


def load_bal_targetids() -> set:
    """TARGETIDs of QSOs flagged as BAL (BI_CIV > 0) in the altbal QSO
    catalog. The DLA catalog has no BAL column, so this is the join key."""
    print(f"loading BAL catalog: {BAL_CATALOG}", flush=True)
    with fits.open(BAL_CATALOG) as f:
        d = f["ZCATALOG"].data
        tid = np.asarray(d["TARGETID"]).astype(np.int64)
        bi = np.asarray(d["BI_CIV"]).astype(np.float64)
    is_bal = np.isfinite(bi) & (bi > 0.0)
    bal = set(int(t) for t in tid[is_bal])
    print(f"  BAL (BI_CIV > 0): {len(bal)} of {len(tid)} QSOs", flush=True)
    return bal


def select(cat: np.ndarray, bal_tids: set) -> np.ndarray:
    """Purity / SNR / redshift / forest-region selection. Appends an
    `IS_BAL` boolean field (BI_CIV > 0, joined from the altbal catalog)."""
    z_qso = cat["Z_QSO"]
    z_dla = cat["Z_DLA"]
    z_dla_lya_obs = (1 + z_dla) * LYA
    z_qso_lya_obs = (1 + z_qso) * LYA
    z_qso_lyb_obs = (1 + z_qso) * LYB
    in_forest = (z_dla_lya_obs > z_qso_lyb_obs) & (z_dla_lya_obs < z_qso_lya_obs)
    not_proximate = z_dla < (z_qso - Z_DLA_TO_QSO_MARGIN)

    p_lo, p_hi = PURITY_PRESETS[PURITY]
    in_purity = (cat["P_DLA"] > p_lo) & (cat["P_DLA"] <= p_hi)
    keep = (
        in_purity
        & (cat["SNR_FOREST"] > SNR_FOREST_MIN)
        & (cat["DLAFLAG"] == 0)
        & (z_qso > Z_QSO_MIN)
        & in_forest
        & not_proximate
        & (cat["NHI"] >= NHI_MIN)
        & (cat["NHI"] <= NHI_MAX)
    )
    print(f"  purity={PURITY} P_DLA∈({p_lo},{p_hi}]:  {in_purity.sum()}", flush=True)
    print(f"  SNR_FOREST > {SNR_FOREST_MIN}:   {(cat['SNR_FOREST'] > SNR_FOREST_MIN).sum()}", flush=True)
    print(f"  DLAFLAG == 0:       {(cat['DLAFLAG'] == 0).sum()}", flush=True)
    print(f"  Z_QSO > {Z_QSO_MIN}:        {(z_qso > Z_QSO_MIN).sum()}", flush=True)
    print(f"  in Lyα forest:      {in_forest.sum()}", flush=True)
    print(f"  not proximate:      {not_proximate.sum()}", flush=True)
    print(f"  NHI ∈ [{NHI_MIN}, {NHI_MAX}]:  "
          f"{((cat['NHI'] >= NHI_MIN) & (cat['NHI'] <= NHI_MAX)).sum()}", flush=True)
    print(f"  ALL combined:       {keep.sum()}", flush=True)

    cat = cat[keep]
    is_bal = np.array([int(t) in bal_tids for t in cat["TARGETID"]], dtype=bool)
    print(f"  BAL (BI_CIV > 0):   {is_bal.sum()} of {len(cat)} "
          f"({100.0 * is_bal.mean():.1f}%)", flush=True)
    cat = rfn.append_fields(cat, "IS_BAL", is_bal, usemask=False)
    return cat


def dump_zhist(cat):
    """Per-bin redshift diagnostics — Z_QSO, Z_DLA, and the stacking-frame
    ratio (1+Z_QSO)/(1+Z_DLA). The ratio sets where each QSO's Lyα
    emission lands in the absorber rest frame, so a bin-to-bin shift in
    its distribution explains differing redward continuum shape. Catalog-
    only — no spectrum reads."""
    quantities = [
        ("Z_QSO",  cat["Z_QSO"],                              "Z_QSO"),
        ("Z_DLA",  cat["Z_DLA"],                              "Z_DLA (absorber)"),
        ("ratio",  (1 + cat["Z_QSO"]) / (1 + cat["Z_DLA"]),   "(1+Z_QSO)/(1+Z_DLA)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    summary = ["# Per-bin redshift summary (median values)",
               "# NHI_lo NHI_hi   n   med_Z_QSO med_Z_DLA med_ratio "
               "med_QSOLyA_in_absframe[Å]"]
    for qi, (qname, qval, qlabel) in enumerate(quantities):
        ax = axes[qi]
        lo_e = np.nanpercentile(qval, 0.5)
        hi_e = np.nanpercentile(qval, 99.5)
        bins_edges = np.linspace(lo_e, hi_e, 50)
        for (lo, hi) in NHI_BINS:
            m = (cat["NHI"] >= lo) & (cat["NHI"] < hi)
            if m.sum() == 0:
                continue
            ax.hist(qval[m], bins=bins_edges, histtype="step", lw=1.4,
                    color=BIN_COLOR[(lo, hi)], density=True,
                    label=f"[{lo:.1f},{hi:.1f}) n={int(m.sum())}")
        ax.set_xlabel(qlabel)
        ax.set_ylabel("normalized density")
        ax.set_title(qlabel)
        ax.grid(alpha=0.2)
        if qi == 0:
            ax.legend(fontsize=7, framealpha=0.9)
    fig.suptitle("Per-NHI-bin redshift distributions — real-LOA selected "
                 "sample. A shifted ratio distribution moves the QSO Lyα "
                 "emission bump in the absorber rest frame.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / tagged("zhist"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR / tagged('zhist')}", flush=True)

    for (lo, hi) in NHI_BINS:
        m = (cat["NHI"] >= lo) & (cat["NHI"] < hi)
        if m.sum() == 0:
            summary.append(f"{lo:.2f} {hi:.2f}   0   - - - -")
            continue
        zq = np.median(cat["Z_QSO"][m])
        zd = np.median(cat["Z_DLA"][m])
        ratio = np.median((1 + cat["Z_QSO"][m]) / (1 + cat["Z_DLA"][m]))
        qso_lya_absframe = LYA * ratio
        summary.append(f"{lo:.2f} {hi:.2f}   {int(m.sum())}   "
                       f"{zq:.4f}  {zd:.4f}  {ratio:.4f}  {qso_lya_absframe:.2f}")
    (OUT_DIR / tagged("zhist_summary", "txt")).write_text(
        "\n".join(summary) + "\n")
    print(f"[saved] {OUT_DIR / tagged('zhist_summary', 'txt')}", flush=True)
    print("\n".join(summary), flush=True)


# ---------------------------------------------------------------------------
# resample + stack core
# ---------------------------------------------------------------------------

def _resample_spectrum(f, iv, m, wave_obs, z, rest_grid):
    """Mask bad pixels, shift to rest frame at redshift z, continuum
    normalize, resample onto rest_grid. Returns resampled array or None.

    `f` is consumed (modified in place) — pass a copy if you need it again."""
    bad = (m != 0) | (iv <= 0) | ~np.isfinite(f)
    f[bad] = np.nan

    rest = wave_obs / (1.0 + z)
    cont = None
    # Primary window [1410, 1520] Å absorber-rest is genuinely flat
    # continuum: well redward of the QSO Lyα emission bump, clear of
    # strong DLA metal lines (SiIV 1403 / SiII 1527 just outside), and
    # wide (110 Å ≈ 550 native pixels) so sky-line masking essentially
    # never leaves < MIN_CONT_PIX good pixels. [1340, 1380] is a fallback
    # for the rare highly-masked or very-high-z (z_DLA > 5.5, red edge
    # clipped) case; if both fail the spectrum is dropped.
    for win in [(1410.0, 1520.0), (1340.0, 1380.0)]:
        win_mask = (rest > win[0]) & (rest < win[1]) & np.isfinite(f)
        if win_mask.sum() >= MIN_CONT_PIX:
            c = np.nanmedian(f[win_mask])
            if c > 0 and np.isfinite(c):
                cont = c
                break
    if cont is None:
        return None
    f_norm = f / cont

    ok = np.isfinite(f_norm)
    if ok.sum() < MIN_GOOD_PIX:
        return None
    rest_ok = rest[ok]
    f_ok = f_norm[ok]
    in_range = (rest_ok > REST_LAMBDA_MIN - 5) & (rest_ok < REST_LAMBDA_MAX + 5)
    if in_range.sum() < 50:
        return None
    rest_ok = rest_ok[in_range]
    f_ok = f_ok[in_range]
    resampled = np.interp(rest_grid, rest_ok, f_ok, left=np.nan, right=np.nan)
    resampled[(rest_grid < rest_ok[0]) | (rest_grid > rest_ok[-1])] = np.nan
    # np.interp linearly bridges masked-pixel gaps. NaN out any output
    # pixel whose two bracketing source samples are > MAX_INTERP_GAP Å
    # apart — i.e. a real masked gap was silently interpolated across.
    bracket = np.searchsorted(rest_ok, rest_grid)
    bracket = np.clip(bracket, 1, len(rest_ok) - 1)
    gap = rest_ok[bracket] - rest_ok[bracket - 1]
    resampled[gap > MAX_INTERP_GAP] = np.nan
    return resampled


def _sigma_clip_median(stack):
    """3σ-clip per pixel, return (median_curve, count_per_pixel).
    Pixels with <50 contributing spectra get NaN."""
    n_pix = stack.shape[1]
    if stack.shape[0] == 0:
        return np.full(n_pix, np.nan), np.zeros(n_pix, dtype=int)
    med = np.nanmedian(stack, axis=0)
    mad = np.nanmedian(np.abs(stack - med), axis=0) * 1.4826
    bad = np.abs(stack - med) > 3 * mad[None, :]
    clipped = np.where(bad, np.nan, stack)
    curve = np.nanmedian(clipped, axis=0)
    counts = np.sum(np.isfinite(clipped), axis=0)
    curve[counts < 50] = np.nan
    return curve, counts


def _stack_pair(rest_grid, raw, is_bal):
    """Split a raw resampled stack into non-BAL and BAL groups, σ-clip
    median each, and fit the pseudo-continuum of each. Returns a
    BinStack carrying both curves and their pseudo-continua."""
    nb = ~is_bal
    curve, counts = _sigma_clip_median(raw[nb])
    curve_b, counts_b = _sigma_clip_median(raw[is_bal])
    pcont = fit_pseudo_continuum(rest_grid, curve, counts)
    pcont_b = fit_pseudo_continuum(rest_grid, curve_b, counts_b)
    return BinStack(curve, counts, int(nb.sum()), pcont,
                    curve_b, counts_b, int(is_bal.sum()), pcont_b)


def read_bin_spectra(cat_bin, arch_tid_to_row, archive, rest_grid, bin_name,
                     control_dz=None):
    """Batch-read a bin's spectra and resample. Returns
    (real_stack, control_stack_or_None, is_bal) — `is_bal` is the BAL
    flag aligned row-for-row with the returned stacks. The non-BAL and
    BAL groups are each independently capped at MAX_PER_BIN."""
    flux = archive["flux"]
    ivar = archive["ivar"]
    mask = archive["mask"]
    wave_obs = archive["wavelength"][:]

    # Cap each BAL group independently so the (minority) BAL stack is not
    # starved by a global cap dominated by non-BAL rows.
    rng = np.random.default_rng(42)
    is_bal_full = cat_bin["IS_BAL"].astype(bool)
    keep_parts = []
    for grp_mask in (~is_bal_full, is_bal_full):
        idx = np.where(grp_mask)[0]
        if len(idx) > MAX_PER_BIN:
            idx = rng.choice(idx, MAX_PER_BIN, replace=False)
        keep_parts.append(idx)
    keep_idx = np.sort(np.concatenate(keep_parts))
    cat_bin = cat_bin[keep_idx]
    if control_dz is not None:
        control_dz = control_dz[keep_idx]

    tids = cat_bin["TARGETID"].astype(np.int64)
    row_idx = np.array([arch_tid_to_row.get(int(t), -1) for t in tids],
                       dtype=np.int64)
    found = row_idx >= 0
    n_skip_tid = int((~found).sum())
    cat_bin = cat_bin[found]
    row_idx = row_idx[found]
    if control_dz is not None:
        control_dz = control_dz[found]
    is_bal = cat_bin["IS_BAL"].astype(bool)
    n = len(row_idx)
    print(f"  bin={bin_name}: {n} spectra to read "
          f"(non-BAL={int((~is_bal).sum())}, BAL={int(is_bal.sum())}, "
          f"skip-no-TID={n_skip_tid})", flush=True)

    n_pix = len(rest_grid)
    real = np.full((n, n_pix), np.nan, dtype=np.float32)
    ctrl = (np.full((n, n_pix), np.nan, dtype=np.float32)
            if control_dz is not None else None)
    n_skip_norm = 0

    sort_perm = np.argsort(row_idx)
    sorted_idx = row_idx[sort_perm]

    t_total = time.time()
    for batch_start in range(0, n, BATCH_SIZE):
        t0 = time.time()
        batch_end = min(batch_start + BATCH_SIZE, n)
        b_idx = sorted_idx[batch_start:batch_end]
        f_batch = flux[b_idx].astype(np.float64)
        iv_batch = ivar[b_idx].astype(np.float64)
        m_batch = mask[b_idx]

        for j, src_i in enumerate(range(batch_start, batch_end)):
            orig_i = sort_perm[src_i]
            z_dla = float(cat_bin[orig_i]["Z_DLA"])
            res = _resample_spectrum(f_batch[j].copy(), iv_batch[j],
                                     m_batch[j], wave_obs, z_dla, rest_grid)
            if res is None:
                n_skip_norm += 1
            else:
                real[orig_i] = res
            if ctrl is not None:
                z_ctrl = z_dla + float(control_dz[orig_i])
                res_c = _resample_spectrum(f_batch[j].copy(), iv_batch[j],
                                           m_batch[j], wave_obs, z_ctrl,
                                           rest_grid)
                if res_c is not None:
                    ctrl[orig_i] = res_c
        print(f"    batch {batch_start//BATCH_SIZE+1}/"
              f"{(n+BATCH_SIZE-1)//BATCH_SIZE}: "
              f"{batch_end-batch_start} specs in {time.time()-t0:.1f}s",
              flush=True)

    print(f"  bin={bin_name}: total {time.time()-t_total:.1f}s, "
          f"skip-no-norm={n_skip_norm}", flush=True)
    return real, ctrl, is_bal


def build_archive_tid_map(archive):
    print(f"building TARGETID→row map ({archive['catalog'].shape[0]} rows)…",
          flush=True)
    tids = archive["catalog"]["TARGETID"][:]
    return {int(t): i for i, t in enumerate(tids)}


def compute_stacks():
    """Read the archive, build per-bin stacks (non-BAL + BAL), the
    production-binned stacks, and the combined LLS/sub-DLA real-vs-control
    stacks. Slow (~15–30 min on /scratch)."""
    cat = load_catalog()
    bal_tids = load_bal_targetids()
    cat = select(cat, bal_tids)
    dump_zhist(cat)

    rest_grid = 10 ** np.arange(np.log10(REST_LAMBDA_MIN),
                                np.log10(REST_LAMBDA_MAX), DLOG_LAMBDA)
    print(f"rest grid: {rest_grid[0]:.2f}..{rest_grid[-1]:.2f} Å, "
          f"n={len(rest_grid)}", flush=True)

    control_bins = set()
    for cat_bins in CONTROL_CATEGORIES.values():
        control_bins.update(cat_bins)

    with h5py.File(LOA_ARCHIVE, "r") as archive:
        tid_to_row = build_archive_tid_map(archive)

        raw_real = {}     # (lo,hi) -> raw real stack array
        raw_ctrl = {}     # (lo,hi) -> raw control stack array
        raw_isbal = {}    # (lo,hi) -> BAL flag aligned to raw_real rows
        rng = np.random.default_rng(7)
        for (lo, hi) in NHI_BINS:
            sub = cat[(cat["NHI"] >= lo) & (cat["NHI"] < hi)]
            print(f"\nbin NHI ∈ [{lo}, {hi}): n_candidates={len(sub)}",
                  flush=True)
            if len(sub) == 0:
                continue
            want_ctrl = (lo, hi) in control_bins
            cdz = None
            if want_ctrl:
                # Random ±[0.15, 0.35] redshift offset per spectrum. At
                # z~3 that is ~11000–26000 km/s — far larger than any zoom
                # panel, so a real metal line shifts well clear of its
                # panel and the per-spectrum-random offsets fully smear it
                # out in the control median.
                cdz = (rng.uniform(0.15, 0.35, len(sub))
                       * rng.choice([-1.0, 1.0], len(sub)))
            real, ctrl, is_bal = read_bin_spectra(
                sub, tid_to_row, archive, rest_grid, f"[{lo},{hi})",
                control_dz=cdz)
            raw_real[(lo, hi)] = real
            raw_isbal[(lo, hi)] = is_bal
            if ctrl is not None:
                raw_ctrl[(lo, hi)] = ctrl

    # Per-bin stacks (non-BAL + BAL) for the 8 fine bins.
    per_bin = {}
    for b in NHI_BINS:
        if b not in raw_real:
            continue
        per_bin[b] = _stack_pair(rest_grid, raw_real[b], raw_isbal[b])

    # Production LLS bin = pooled union of the 3 fine LLS bins.
    lls_raw = [raw_real[b] for b in LLS_BINS_FINE if b in raw_real]
    lls_bal = [raw_isbal[b] for b in LLS_BINS_FINE if b in raw_isbal]
    if lls_raw:
        per_bin[LLS_MERGED] = _stack_pair(rest_grid,
                                          np.concatenate(lls_raw, axis=0),
                                          np.concatenate(lls_bal, axis=0))

    # Combined real-vs-control stacks per low-NHI category (non-BAL only —
    # the decisive false-positive plot uses the clean sample).
    combined = {}   # name -> (real_curve, real_counts, ctrl_curve, ctrl_counts, n)
    for name, cat_bins in CONTROL_CATEGORIES.items():
        reals, ctrls = [], []
        for b in cat_bins:
            if b in raw_real and b in raw_ctrl:
                nb = ~raw_isbal[b]
                reals.append(raw_real[b][nb])
                ctrls.append(raw_ctrl[b][nb])
        if not reals:
            continue
        pooled_real = np.concatenate(reals, axis=0)
        pooled_ctrl = np.concatenate(ctrls, axis=0)
        rc, rn = _sigma_clip_median(pooled_real)
        cc, cn = _sigma_clip_median(pooled_ctrl)
        pcont_r = fit_pseudo_continuum(rest_grid, rc, rn)
        pcont_c = fit_pseudo_continuum(rest_grid, cc, cn)
        combined[name] = (rc, rn, pcont_r, cc, cn, pcont_c, len(pooled_real))
        print(f"combined {name}: {len(pooled_real)} non-BAL spectra "
              f"(real + scrambled control)", flush=True)

    # counts log
    with (OUT_DIR / tagged("counts", "txt")).open("w") as fh:
        fh.write(f"# purity={PURITY}, SNR_FOREST > {SNR_FOREST_MIN}, "
                 f"DLAFLAG=0, Z_QSO > {Z_QSO_MIN}, in-forest, not-proximate\n")
        fh.write(f"# MAX_PER_BIN={MAX_PER_BIN} (per BAL group)\n")
        fh.write("# NHI_lo NHI_hi n_candidates n_nonbal n_bal\n")
        for (lo, hi) in NHI_BINS:
            n_cand = int(((cat["NHI"] >= lo) & (cat["NHI"] < hi)).sum())
            bs = per_bin.get((lo, hi))
            n_nb = bs.n if bs else 0
            n_b = bs.n_bal if bs else 0
            fh.write(f"{lo:.2f} {hi:.2f} {n_cand} {n_nb} {n_b}\n")
    return rest_grid, per_bin, combined


def prod_bins_view(per_bin):
    """The production-binned subset of `per_bin` (merged LLS + sub-DLA +
    DLA), in NHI_BINS_PROD order."""
    return {b: per_bin[b] for b in NHI_BINS_PROD if b in per_bin}


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def save_curves(rest_grid, per_bin, combined):
    payload = {"rest_grid": rest_grid,
               "provenance": np.array(json.dumps(provenance_dict()))}
    for (lo, hi), bs in per_bin.items():
        key = f"{lo:.2f}_{hi:.2f}"
        payload[f"curve_{key}"] = bs.curve
        payload[f"counts_{key}"] = bs.counts
        payload[f"ntot_{key}"] = np.int64(bs.n)
        payload[f"pcont_{key}"] = bs.pcont
        payload[f"curvebal_{key}"] = bs.curve_bal
        payload[f"countsbal_{key}"] = bs.counts_bal
        payload[f"ntotbal_{key}"] = np.int64(bs.n_bal)
        payload[f"pcontbal_{key}"] = bs.pcont_bal
    for name, (rc, rn, pcont_r, cc, cn, pcont_c, n) in combined.items():
        payload[f"comb_real_{name}"] = rc
        payload[f"comb_realcnt_{name}"] = rn
        payload[f"comb_pcontreal_{name}"] = pcont_r
        payload[f"comb_ctrl_{name}"] = cc
        payload[f"comb_ctrlcnt_{name}"] = cn
        payload[f"comb_pcontctrl_{name}"] = pcont_c
        payload[f"comb_n_{name}"] = np.int64(n)
    out = npz_path()
    np.savez(out, **payload)
    print(f"[saved] {out}", flush=True)


def load_curves(path, expect_preset=None):
    path = Path(path)
    d = np.load(path, allow_pickle=False)
    if "provenance" not in d:
        raise SystemExit(
            f"[ERROR] cached {path.name} predates the provenance/BAL "
            "format — re-run to regenerate.")
    check_provenance(json.loads(str(d["provenance"])), expect_preset)
    rest_grid = d["rest_grid"]
    per_bin = {}
    for (lo, hi) in STORED_BINS:
        key = f"{lo:.2f}_{hi:.2f}"
        if f"curve_{key}" not in d:
            continue
        per_bin[(lo, hi)] = BinStack(
            d[f"curve_{key}"], d[f"counts_{key}"], int(d[f"ntot_{key}"]),
            d[f"pcont_{key}"],
            d[f"curvebal_{key}"], d[f"countsbal_{key}"],
            int(d[f"ntotbal_{key}"]), d[f"pcontbal_{key}"])
    combined = {}
    for name in CONTROL_CATEGORIES:
        if f"comb_real_{name}" not in d:
            continue
        combined[name] = (d[f"comb_real_{name}"], d[f"comb_realcnt_{name}"],
                           d[f"comb_pcontreal_{name}"],
                           d[f"comb_ctrl_{name}"], d[f"comb_ctrlcnt_{name}"],
                           d[f"comb_pcontctrl_{name}"],
                           int(d[f"comb_n_{name}"]))
    return rest_grid, per_bin, combined


# ---------------------------------------------------------------------------
# pseudo-continuum
# ---------------------------------------------------------------------------

_HI_KEYS = frozenset(HI_MASK_HALF)  # METAL_LINES keys that are H I lines


def _continuum_mask(rest_grid, curve, counts):
    """Boolean `fit_ok` — pixels usable for the pseudo-continuum fit:
    finite, well-covered (≥50 spectra), redward of PCONT_LAMBDA_MIN, and
    outside every line's mask window. Metal masks are wavelength-scaled
    (K_MASK_SIGMA × σ_stack(λ)); H I lines use the wider HI_MASK_HALF."""
    fit_ok = (np.isfinite(curve) & (np.asarray(counts) >= 50)
              & (rest_grid >= PCONT_LAMBDA_MIN))
    for name, w in METAL_LINES.items():
        if name in _HI_KEYS:
            half = HI_MASK_HALF[name]
        else:
            half = K_MASK_SIGMA * w * SIGMA_V / _C_KM_S
        fit_ok = fit_ok & (np.abs(rest_grid - w) > half)
    return fit_ok


def _safe_lsq_spline(x, y, w, knots):
    """LSQUnivariateSpline that thins knots until the Schoenberg-Whitney
    condition is satisfied — on ValueError it drops the knot in the
    sparsest interval and retries. Falls back to a single polynomial
    piece (no interior knots) if all knots fail."""
    knots = list(knots)
    for _ in range(len(knots) + 1):
        if not knots:
            return LSQUnivariateSpline(x, y, t=[], k=SPLINE_ORDER, w=w)
        try:
            return LSQUnivariateSpline(x, y, t=knots, k=SPLINE_ORDER, w=w)
        except ValueError:
            near = [int(np.sum(np.abs(x - t) <= KNOT_SPACING)) for t in knots]
            knots.pop(int(np.argmin(near)))
    return LSQUnivariateSpline(x, y, t=[], k=SPLINE_ORDER, w=w)


def fit_pseudo_continuum(rest_grid, curve, counts, return_info=False):
    """Masked fixed-knot cubic spline + Schlegel-style iterative
    sigma-rejection. Returns P(λ) — the pseudo-continuum — NaN below
    PCONT_LAMBDA_MIN and where the fit is degenerate. With
    return_info=True also returns a diagnostics dict.

    The knot vector, spline order and weight definition are fixed across
    rejection iterations; only the set of fitted pixels shrinks."""
    rest_grid = np.asarray(rest_grid, float)
    curve = np.asarray(curve, float)
    counts = np.asarray(counts, float)
    n_pix = len(rest_grid)
    P = np.full(n_pix, np.nan)
    info = {"n_knots": 0, "n_iter": 0, "n_rejected": 0, "rms": np.nan}

    fit_ok0 = _continuum_mask(rest_grid, curve, counts)
    if int(fit_ok0.sum()) < 4 * SPLINE_ORDER:
        return (P, info) if return_info else P

    weights = np.sqrt(np.maximum(counts, 0.0))
    ok = fit_ok0.copy()
    spl = None
    for it in range(MAX_REJECT_ITER + 1):
        x, y, w = rest_grid[ok], curve[ok], weights[ok]
        if len(x) < 4 * SPLINE_ORDER:
            break
        cand = np.arange(x[0] + KNOT_SPACING, x[-1] - KNOT_SPACING / 2.0,
                         KNOT_SPACING)
        knots = np.array([k for k in cand
                          if np.any(np.abs(x - k) <= KNOT_SPACING)])
        knots = knots[(knots > x[0]) & (knots < x[-1])]
        spl = _safe_lsq_spline(x, y, w, knots)
        info["n_knots"] = len(spl.get_knots()) - 2  # minus the 2 endpoints
        info["n_iter"] = it
        resid = y - spl(x)
        med = np.median(resid)
        sigma = 1.4826 * np.median(np.abs(resid - med))
        if sigma <= 0:
            break
        new_bad = np.abs(resid - med) > REJECT_SIGMA * sigma
        if not new_bad.any():
            break
        ok_idx = np.where(ok)[0]
        ok[ok_idx[new_bad]] = False

    if spl is None:
        return (P, info) if return_info else P
    x = rest_grid[ok]
    # Evaluate P only over the post-rejection fitted-pixel span (not the
    # full 945-1600 A): clamping to [x[0], x[-1]] avoids cubic extrapolation
    # past the data if rejection trims the blue/red edge.
    inside = (rest_grid >= max(PCONT_LAMBDA_MIN, x[0])) & (rest_grid <= x[-1])
    P[inside] = spl(rest_grid[inside])
    info["n_rejected"] = int(fit_ok0.sum() - ok.sum())
    clean = fit_ok0 & np.isfinite(P)
    if clean.any():
        info["rms"] = float(np.sqrt(np.nanmean((curve[clean] / P[clean] - 1.0) ** 2)))
    return (P, info) if return_info else P


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def _species_of(line_name):
    """'CIII(1176)' -> 'CIII'; 'Lyα' -> 'Lyα'."""
    return line_name.split("(")[0]


def _filtered_panels(exclude):
    """ZOOM_PANELS with `exclude`d species removed. Panels left with no
    lines are dropped; partially-filtered panels get a regenerated title."""
    out = []
    for title, center, half, lines in ZOOM_PANELS:
        surv = [(sp, w) for sp, w in lines if sp not in exclude]
        if not surv:
            continue
        if len(surv) != len(lines):
            title = " / ".join(f"{sp} {w:.0f}" for sp, w in surv)
        out.append((title, center, half, surv))
    return out


def _draw_metal_labels(ax, y_lo, y_hi, lam_lo, lam_hi, exclude=frozenset()):
    """Vertical metal-line markers with staggered labels (3-level y
    rotation so neighbouring labels don't overlap)."""
    span = y_hi - y_lo
    levels = [y_hi - 0.04 * span, y_hi - 0.13 * span, y_hi - 0.22 * span]
    in_range = sorted((w, n) for n, w in METAL_LINES.items()
                      if lam_lo < w < lam_hi and _species_of(n) not in exclude)
    for k, (w, name) in enumerate(in_range):
        ax.axvline(w, color="grey", lw=0.5, ls="--", alpha=0.5)
        ax.text(w, levels[k % 3], name, rotation=90, fontsize=6.5,
                ha="center", va="top", color="dimgrey")


def plot_overview(rest_grid, per_bin, bins, fname, subtitle,
                  exclude=frozenset()):
    """Full REST_LAMBDA_MIN–REST_LAMBDA_MAX overview for the given NHI
    bins (non-BAL stacks)."""
    fig, ax = plt.subplots(figsize=(15, 6))
    for (lo, hi) in bins:
        if (lo, hi) not in per_bin:
            continue
        bs = per_bin[(lo, hi)]
        n_eff = (int(np.nanmedian(bs.counts[bs.counts > 0]))
                 if (bs.counts > 0).any() else 0)
        ax.plot(rest_grid, bs.curve, color=BIN_COLOR[(lo, hi)], lw=1.2,
                alpha=0.85,
                label=f"log NHI [{lo:.1f}, {hi:.1f})  n={bs.n} "
                      f"(median_pix={n_eff})")
    ax.axhline(1.0, color="k", lw=0.5, alpha=0.4)
    ax.axvline(LYMAN_LIMIT, color="navy", lw=0.9, ls=":", alpha=0.7)
    ax.text(LYMAN_LIMIT - 4, 1.62, "Lyman limit 912 Å", rotation=90,
            fontsize=7, ha="right", va="top", color="navy")
    ax.set_xlim(REST_LAMBDA_MIN, REST_LAMBDA_MAX)
    ax.set_ylim(0.0, 1.7)
    _draw_metal_labels(ax, 0.0, 1.7, REST_LAMBDA_MIN, REST_LAMBDA_MAX, exclude)
    ax.set_xlabel(r"absorber rest-frame wavelength [Å]")
    ax.set_ylabel("median stacked flux (norm. to 1 at 1410–1520 Å rest)")
    ax.set_title(
        f"{subtitle} — purity={PURITY}, "
        f"SNR_forest > {SNR_FOREST_MIN}, z_QSO > {Z_QSO_MIN}, "
        "Lyα-forest detection only, non-BAL")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR / fname}", flush=True)



def plot_metal_zoom(rest_grid, per_bin, bins, fname, suptitle,
                    exclude=frozenset()):
    """Per-metal-line zoom panels with local-continuum normalization and
    adaptive y-limits. Coherent dip at a line centre = real absorption."""
    panels = _filtered_panels(exclude)
    n_panels = len(panels)
    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.6 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (title, center, half, lines) in zip(axes, panels):
        lo_w, hi_w = center - half, center + half
        sel = (rest_grid >= lo_w) & (rest_grid <= hi_w)
        x = rest_grid[sel]
        panel_min = 1.0
        for (lo, hi) in bins:
            if (lo, hi) not in per_bin:
                continue
            bs = per_bin[(lo, hi)]
            y = (bs.curve / bs.pcont)[sel]
            ax.plot(x, y, color=BIN_COLOR[(lo, hi)], lw=1.3, alpha=0.85,
                    label=f"NHI [{lo:.1f},{hi:.1f})")
            if np.isfinite(y).any():
                panel_min = min(panel_min, np.nanpercentile(y, 1))
        for ln_name, ln_w in lines:
            ax.axvline(ln_w, color="k", lw=0.7, ls="--", alpha=0.6)
        y_lo = max(0.25, panel_min - 0.05)
        ax.set_ylim(y_lo, 1.06)
        for ln_name, ln_w in lines:
            ax.text(ln_w, 1.05, ln_name, rotation=90, fontsize=7,
                    ha="center", va="top", color="k")
        ax.axhline(1.0, color="grey", lw=0.5, alpha=0.5)
        ax.set_xlim(lo_w, hi_w)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("rest-frame λ [Å]", fontsize=8)
        ax.set_ylabel("flux / local continuum", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    axes[0].legend(loc="lower left", fontsize=7, framealpha=0.9)
    fig.suptitle(suptitle, fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR / fname}", flush=True)


def lls_recovery_model(rest_grid, log_nhi):
    """Single-absorber Lyman-limit transmission T(λ) = exp(−τ_LL) for an
    H I column log10(N_HI). Blueward of 911.76 Å the LL optical depth is
    τ_LL(λ) = N_HI·σ_912·(λ/911.76)^3 — the ν^−3 photoionization cross-
    section of Prochaska, Worseck & O'Meara 2009 (arXiv:0910.0009),
    σ_912 = 6.30e-18 cm². T = 1 redward of the limit. This is the
    recovery curve for ONE LLS; a real stacked LLS population, plus the
    foreground IGM + Lyman-series opacity, sits below it."""
    tau = np.zeros_like(rest_grid, dtype=float)
    blue = rest_grid < LYMAN_LIMIT
    tau[blue] = (10.0 ** log_nhi) * SIGMA_912 * (rest_grid[blue] / LYMAN_LIMIT) ** 3
    return np.exp(-tau)


def plot_lyman_limit(rest_grid, per_bin, fname):
    """Lyman limit break recovery: the 740–1050 Å absorber-rest region for
    the production NHI bins (non-BAL). A true LLS / sub-DLA / DLA
    population shows a coherent flux decrement turning on at the 911.76 Å
    Lyman limit; the depth deepens with NHI (and saturates near-black
    above log NHI ~18). No coherent edge = false positives. Dashed black
    curves overplot the single-absorber τ_LL ∝ (λ/912)^3 recovery model
    (Prochaska, Worseck & O'Meara 2009) for reference log N_HI."""
    fig, ax = plt.subplots(figsize=(13, 6))
    lo_w, hi_w = 740.0, 1050.0
    sel = (rest_grid >= lo_w) & (rest_grid <= hi_w)
    x = rest_grid[sel]
    for (lo, hi) in NHI_BINS_PROD:
        if (lo, hi) not in per_bin:
            continue
        bs = per_bin[(lo, hi)]
        ax.plot(x, bs.curve[sel], color=BIN_COLOR[(lo, hi)], lw=1.4,
                alpha=0.9, label=f"log NHI [{lo:.1f}, {hi:.1f})  n={bs.n}")
    # Overplot single-absorber LL-recovery models exp(−τ_LL), τ_LL ∝
    # (λ/912)^3 (Prochaska, Worseck & O'Meara 2009). Plotted only blueward
    # of the limit. The observed stack sits BELOW these — it also carries
    # foreground IGM + Lyman-series opacity not in this single-LLS model.
    xb = x[x <= LYMAN_LIMIT]
    for log_nhi, ls in [(17.2, ":"), (17.5, "--"), (18.0, "-.")]:
        Tb = lls_recovery_model(xb, log_nhi)
        tau912 = (10.0 ** log_nhi) * SIGMA_912
        ax.plot(xb, Tb, color="black", lw=1.2, ls=ls, alpha=0.75,
                label=f"LLS model: log NHI={log_nhi}  (τ₉₁₂={tau912:.1f})")
    # Lyman limit + Lyman-series blanketing band (912–~940 Å, depressed by
    # converging Lyman-series lines — do NOT read the pre-break level here).
    ax.axvline(LYMAN_LIMIT, color="navy", lw=1.2, ls=":", alpha=0.85)
    ax.text(LYMAN_LIMIT, 1.28, "Lyman limit 911.76 Å", rotation=90,
            fontsize=8, ha="right", va="top", color="navy")
    ax.axvspan(LYMAN_LIMIT, 945.0, color="grey", alpha=0.12)
    ax.text(928.0, 1.28, "Lyman-series\nblanketing", fontsize=7,
            ha="center", va="top", color="dimgrey")
    ax.axhline(1.0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlim(lo_w, hi_w)
    ax.set_ylim(0.0, 1.35)
    ax.set_xlabel("absorber rest-frame wavelength [Å]")
    ax.set_ylabel("median stacked flux (norm. to 1 at 1410–1520 Å rest)")
    ax.set_title(
        "Lyman limit break recovery — production NHI bins, non-BAL. "
        "Flux blueward of 912 Å should drop coherently for real H I "
        "absorbers (deeper / blacker with NHI). Deep-blue pixels are "
        "sparsely covered (DESI blue cutoff) and NaN-clipped below 50 "
        "spectra.")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR / fname}", flush=True)


def plot_bal_compare(rest_grid, per_bin, fname):
    """Non-BAL vs BAL median stacks, one panel per production NHI bin.
    BAL troughs sit near the QSO-frame CIV; this documents how different
    the BAL sample's absorber-frame composite is from the non-BAL one."""
    bins = [b for b in NHI_BINS_PROD if b in per_bin]
    n_panels = len(bins)
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.8 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (lo, hi) in zip(axes, bins):
        bs = per_bin[(lo, hi)]
        ax.plot(rest_grid, bs.curve, color="#1f77b4", lw=1.1, alpha=0.9,
                label=f"non-BAL  n={bs.n}")
        ax.plot(rest_grid, bs.curve_bal, color="#d62728", lw=1.1, alpha=0.9,
                label=f"BAL (BI_CIV>0)  n={bs.n_bal}")
        ax.axhline(1.0, color="k", lw=0.5, alpha=0.4)
        ax.axvline(LYMAN_LIMIT, color="navy", lw=0.8, ls=":", alpha=0.6)
        _draw_metal_labels(ax, 0.0, 1.7, REST_LAMBDA_MIN, REST_LAMBDA_MAX)
        ax.set_xlim(REST_LAMBDA_MIN, REST_LAMBDA_MAX)
        ax.set_ylim(0.0, 1.7)
        ax.set_title(f"log NHI [{lo:.1f}, {hi:.1f})", fontsize=9)
        ax.set_xlabel("rest-frame λ [Å]", fontsize=8)
        ax.set_ylabel("median stacked flux", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
        ax.grid(alpha=0.2)
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    fig.suptitle("Non-BAL vs BAL stacks per production NHI bin. BAL QSOs "
                 "(BI_CIV > 0) are usually excluded; large non-BAL/BAL "
                 "differences near CIV/SiIV would justify that.",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR / fname}", flush=True)


def plot_control(rest_grid, combined, name, fname, exclude=frozenset()):
    """Real vs redshift-scrambled-control zoom panels for one low-NHI
    category. The decisive plot: a coherent dip in the real stack that is
    absent in the control = the detections are real absorbers."""
    if name not in combined:
        print(f"[skip] no combined stack for {name}", flush=True)
        return
    rc, rn, pcont_r, cc, cn, pcont_c, n = combined[name]
    norm_real = rc / pcont_r
    norm_ctrl = cc / pcont_c
    panels = _filtered_panels(exclude)
    n_panels = len(panels)
    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.6 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (title, center, half, lines) in zip(axes, panels):
        lo_w, hi_w = center - half, center + half
        sel = (rest_grid >= lo_w) & (rest_grid <= hi_w)
        x = rest_grid[sel]
        y_real = norm_real[sel]
        y_ctrl = norm_ctrl[sel]
        ax.plot(x, y_real, color="#d62728", lw=1.5, alpha=0.9, label="real")
        ax.plot(x, y_ctrl, color="#888888", lw=1.3, alpha=0.8,
                label="z-scrambled control")
        panel_min = 1.0
        for yy in (y_real, y_ctrl):
            if np.isfinite(yy).any():
                panel_min = min(panel_min, np.nanpercentile(yy, 1))
        for ln_name, ln_w in lines:
            ax.axvline(ln_w, color="k", lw=0.7, ls="--", alpha=0.6)
            ax.text(ln_w, 1.05, ln_name, rotation=90, fontsize=7,
                    ha="center", va="top", color="k")
        ax.axhline(1.0, color="grey", lw=0.5, alpha=0.5)
        ax.set_xlim(lo_w, hi_w)
        ax.set_ylim(max(0.25, panel_min - 0.05), 1.06)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("rest-frame λ [Å]", fontsize=8)
        ax.set_ylabel("flux / local continuum", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    label = {"lls": "LLS [17.2, 19)", "subdla": "sub-DLA [19, 20.3)",
             "lownhi": "low-NHI (LLS + sub-DLA)"}[name]
    fig.suptitle(
        f"Real vs redshift-scrambled control — combined {label}, "
        f"n={n} non-BAL spectra. A coherent dip in RED (real) absent in "
        "GREY (control) = real absorbers (CIV 1548/1551 is the decisive "
        "panel). NOTE: the LLS bin is ~89% log NHI 18.5–19 — this confirms "
        "the strong-LLS regime; the [17.2,18.5) tail is coverage-limited "
        "and untested. P_DLA > 0.97 only — the marginal operating point "
        "is not probed here.",
        fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR / fname}", flush=True)


def plot_pseudo_continuum_qc(rest_grid, per_bin, fname):
    """QC: per production NHI bin, the raw composite with its fitted
    pseudo-continuum overlaid, masked regions shaded, knot count + fit
    RMS in the panel title. The eyeball check that the fit is sane."""
    bins = [b for b in NHI_BINS_PROD if b in per_bin]
    n_panels = len(bins)
    ncols = 2
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.2 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (lo, hi) in zip(axes, bins):
        bs = per_bin[(lo, hi)]
        P, info = fit_pseudo_continuum(rest_grid, bs.curve, bs.counts,
                                       return_info=True)
        fit_ok = _continuum_mask(rest_grid, bs.curve, bs.counts)
        ax.plot(rest_grid, bs.curve, color="#444444", lw=0.8, alpha=0.8,
                label="composite")
        ax.plot(rest_grid, P, color="#d62728", lw=1.5, alpha=0.9,
                label="pseudo-continuum")
        # shade the masked (non-fit) regions redward of the fit floor
        masked = (~fit_ok) & (rest_grid >= PCONT_LAMBDA_MIN)
        ax.fill_between(rest_grid, 0, 1, where=masked, transform=ax.get_xaxis_transform(),
                        color="grey", alpha=0.12, step="mid")
        ax.axvline(PCONT_LAMBDA_MIN, color="navy", lw=0.8, ls=":", alpha=0.7)
        ax.axhline(1.0, color="k", lw=0.5, alpha=0.3)
        ax.set_xlim(REST_LAMBDA_MIN, REST_LAMBDA_MAX)
        ax.set_ylim(0.0, 1.7)
        ax.set_title(f"log NHI [{lo:.1f}, {hi:.1f})  n={bs.n}  "
                     f"knots={info['n_knots']}  rejected={info['n_rejected']}  "
                     f"RMS={info['rms']:.3f}", fontsize=9)
        ax.set_xlabel("absorber rest-frame λ [Å]", fontsize=8)
        ax.set_ylabel("stacked flux", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
        ax.grid(alpha=0.2)
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    fig.suptitle("Pseudo-continuum QC — masked fixed-knot cubic spline "
                 "(grey = masked from the fit; dotted = 945 Å fit floor).",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR / fname}", flush=True)


def plot_purity_comparison(rest_grid, comb_high, comb_marg, fname):
    """Marginal-purity vs high-purity, pooled low-NHI (LLS + sub-DLA).
    Three pseudo-continuum-normalized curves per metal-line panel:
    marginal-real, marginal z-scrambled control, high-purity-real
    (reference). Marginal-real tracking high-real ⇒ the marginal
    detections are real; marginal-real flat like its control ⇒ the
    marginal operating point is contaminated. `comb_*` are the
    `combined["lownhi"]` 7-tuples (rc, rn, pcont_r, cc, cn, pcont_c, n)."""
    rc_h, _, pc_h, _, _, _, n_h = comb_high
    rc_m, _, pc_m, cc_m, _, pcc_m, n_m = comb_marg
    norm_high = rc_h / pc_h
    norm_marg = rc_m / pc_m
    norm_mctrl = cc_m / pcc_m
    panels = PURITY_COMPARE_PANELS
    n_panels = len(panels)
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(19, 4.6 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (title, center, half, lines) in zip(axes, panels):
        lo_w, hi_w = center - half, center + half
        sel = (rest_grid >= lo_w) & (rest_grid <= hi_w)
        x = rest_grid[sel]
        ax.plot(x, norm_marg[sel], color="#d62728", lw=1.6, alpha=0.9,
                label=f"marginal real  n={n_m}")
        ax.plot(x, norm_mctrl[sel], color="#888888", lw=1.3, alpha=0.8,
                label="marginal z-scrambled control")
        ax.plot(x, norm_high[sel], color="#1f77b4", lw=1.3, alpha=0.85,
                ls="--", label=f"high-purity real  n={n_h}")
        for ln_name, ln_w in lines:
            ax.axvline(ln_w, color="k", lw=0.7, ls="--", alpha=0.6)
            ax.text(ln_w, 1.05, ln_name, rotation=90, fontsize=7,
                    ha="center", va="top", color="k")
        ax.axhline(1.0, color="grey", lw=0.5, alpha=0.5)
        ax.set_xlim(lo_w, hi_w)
        ax.set_ylim(0.55, 1.1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("rest-frame λ [Å]", fontsize=8)
        ax.set_ylabel("flux / pseudo-continuum", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.suptitle("Marginal-purity (P_DLA 0.5–0.7) vs high-purity "
                 "(P_DLA > 0.97) — pooled low-NHI. Marginal-real tracking "
                 "high-real ⇒ marginal detections real; marginal-real flat "
                 "like its control ⇒ marginal operating point contaminated.",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR / fname}", flush=True)


def render_all(rest_grid, per_bin, combined):
    prod = prod_bins_view(per_bin)

    # Production-binned headline figures (LLS merged to one bin).
    plot_overview(rest_grid, prod, NHI_BINS_PROD, tagged("stack_prod"),
                  "Real-LOA production bins (LLS merged / sub-DLA / DLA)",
                  exclude=EXCLUDE_SPECIES["all"])
    plot_metal_zoom(rest_grid, prod, NHI_BINS_PROD,
                    tagged("stack_metal_zoom_prod"),
                    "Metal-line zoom — production NHI bins",
                    exclude=EXCLUDE_SPECIES["all"])

    # Diagnostic: LLS resolved into 3 fine bins.
    plot_overview(rest_grid, per_bin, LLS_BINS_FINE, tagged("stack_lls_diag"),
                  "Real-LOA LLS resolved (3 fine bins, log NHI 17.2–19)",
                  exclude=EXCLUDE_SPECIES["lls"])
    plot_metal_zoom(rest_grid, per_bin, LLS_BINS_FINE,
                    tagged("stack_metal_zoom_lls_diag"),
                    "Metal-line zoom — LLS resolved (3 fine bins)",
                    exclude=EXCLUDE_SPECIES["lls"])

    # Sub-DLA / DLA focus figures (production bins).
    for tag, bins, label in [
        ("subdla", SUBDLA_BINS, "Real-LOA sub-DLAs (log NHI 19–20.3)"),
        ("dla", DLA_BINS, "Real-LOA DLAs (log NHI ≥ 20.3)"),
    ]:
        exc = EXCLUDE_SPECIES[tag]
        plot_overview(rest_grid, per_bin, bins, tagged(f"stack_{tag}"), label,
                      exclude=exc)
        plot_metal_zoom(rest_grid, per_bin, bins,
                        tagged(f"stack_metal_zoom_{tag}"),
                        f"Metal-line zoom — {label}", exclude=exc)

    # Lyman limit break recovery + BAL comparison + continuum QC.
    plot_lyman_limit(rest_grid, per_bin, tagged("stack_lyman_limit"))
    plot_bal_compare(rest_grid, per_bin, tagged("stack_bal_compare"))
    plot_pseudo_continuum_qc(rest_grid, prod,
                             tagged("stack_pseudo_continuum_qc"))

    # Decisive real-vs-control plots.
    for name in CONTROL_CATEGORIES:
        plot_control(rest_grid, combined, name,
                     tagged(f"stack_control_{name}"),
                     exclude=EXCLUDE_SPECIES.get(name, frozenset()))


def _arg_value(flag, default):
    """Value of `--flag VALUE` in sys.argv, else `default`."""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def main():
    global PURITY
    PURITY = _arg_value("--purity", "high")
    if PURITY not in PURITY_PRESETS:
        raise SystemExit(f"--purity must be one of {list(PURITY_PRESETS)}; "
                         f"got {PURITY!r}")
    _ensure_outdir()
    # `--compare-purity`: load the cached high + marginal npz and render
    # the marginal-vs-high comparison figure (no stacking, seconds).
    if "--compare-purity" in sys.argv:
        high_p = OUT_DIR / "stack_curves_high.npz"
        marg_p = OUT_DIR / "stack_curves_marginal.npz"
        missing = [str(p) for p in (high_p, marg_p) if not p.exists()]
        if missing:
            raise SystemExit("--compare-purity needs both npz; missing: "
                             + ", ".join(missing))
        rg_h, _, comb_h = load_curves(high_p, expect_preset="high")
        _rg_m, _, comb_m = load_curves(marg_p, expect_preset="marginal")
        if "lownhi" not in comb_h or "lownhi" not in comb_m:
            raise SystemExit("npz lacks the 'lownhi' combined category — "
                             "re-run both --purity presets to regenerate.")
        plot_purity_comparison(rg_h, comb_h["lownhi"], comb_m["lownhi"],
                               "stack_purity_comparison.png")
        return
    # `--zhist-only`: just the per-bin redshift diagnostics (catalog-only,
    # runs in seconds — no spectrum reads).
    if "--zhist-only" in sys.argv:
        cat = load_catalog()
        bal_tids = load_bal_targetids()
        cat = select(cat, bal_tids)
        dump_zhist(cat)
        return
    if "--plot-only" in sys.argv:
        if not npz_path().exists():
            raise SystemExit(f"no cached curves at {npz_path()}; "
                             "run without --plot-only first")
        print(f"loading cached curves from {npz_path()}", flush=True)
        rest_grid, per_bin, combined = load_curves(npz_path(),
                                                   expect_preset=PURITY)
    else:
        rest_grid, per_bin, combined = compute_stacks()
        save_curves(rest_grid, per_bin, combined)
    render_all(rest_grid, per_bin, combined)


if __name__ == "__main__":
    main()
