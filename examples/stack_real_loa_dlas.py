"""Stack real-LOA LLS / sub-DLA / DLA detections by NHI bin.

Goal: visually verify that low-NHI detections (LLS log NHI < 19, sub-DLA
19–20.3) show coherent metal-line absorption — the falsifying signature
for false-positive contamination is no coherent metals at any diagnostic
wavelength (especially the CIV 1548/1551 doublet, SiIV 1394/1403).

Inputs
------
- DLA catalog:  /scratch/.../desi-loa-gpdla-...lls_run-nhi172/dlacat-loa-main-dark.fits
- Spectra:      /scratch/.../loa_archives/loa_full_z2_noR_v2.h5 (LoaArchive)

Methodology (per docs/notes/2026-05-15_stack_methodology summary):
- median stack, σ-clip 3σ per rest-frame pixel
- per-spectrum continuum: divide by median in a flat redward window
  [1410, 1520] Å absorber-rest (fallback [1340, 1380])
- log-λ grid at native ~0.0001 dex (69 km/s)
- discard rest-frame pixels with <50 contributing spectra
- selection: P_DLA > 0.97, SNR_FOREST > 2, DLAFLAG=0, Z_QSO > 3,
  absorber in Lyα forest, not proximate

Control: each LLS / sub-DLA spectrum is ALSO stacked at a scrambled
redshift z + Δz (Δz random ±[0.04, 0.10]) — real metals locked to the
absorber redshift survive the real stack and wash out in the control.
A coherent CIV dip in the real stack but a flat control = real absorbers.

Outputs (docs/notes/2026-05-15_stack_real_loa_dlas/):
  stack_all.png / stack_metal_zoom_all.png            — all 8 NHI bins
  stack_lls.png / stack_metal_zoom_lls.png            — LLS [17.2, 19)
  stack_subdla.png / stack_metal_zoom_subdla.png      — sub-DLA [19, 20.3)
  stack_dla.png / stack_metal_zoom_dla.png            — DLA [20.3, 23)
  stack_control_lls.png / stack_control_subdla.png    — real vs control
  stack_curves.npz                                    — cached curves
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import h5py
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DLACAT = "/scratch/cavestru_root/cavestru0/mfho/nersc/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/dlacat-loa-main-dark.fits"
LOA_ARCHIVE = "/scratch/cavestru_root/cavestru0/mfho/nersc/loa_archives/loa_full_z2_noR_v2.h5"
OUT_DIR = Path("/home/mfho/desi_gpy_dla_detection/docs/notes/2026-05-15_stack_real_loa_dlas")
OUT_DIR.mkdir(parents=True, exist_ok=True)
NPZ_PATH = OUT_DIR / "stack_curves.npz"

# Rest-frame stack grid: log-λ, 900–1600 Å, dλ ~ 0.6 Å at 1200 Å
REST_LAMBDA_MIN = 900.0
REST_LAMBDA_MAX = 1600.0
DLOG_LAMBDA = 0.0001  # ~69 km/s, native BOSS/DESI

# NHI bins spanning the full nhi172 catalog range [17.2, 23].
# LLS < 19.0, sub-DLA 19.0–20.3, DLA >= 20.3 (canonical threshold).
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
BIN_COLORS = [
    "#b0a8d0", "#8c7fc0", "#6a5acd",   # LLS — purples
    "#1f77b4", "#17becf", "#2ca02c",   # sub-DLA — blue/cyan/green
    "#ff7f0e", "#d62728",              # DLA — orange/red
]
LLS_BINS    = [b for b in NHI_BINS if b[1] <= 19.0]
SUBDLA_BINS = [b for b in NHI_BINS if b[0] >= 19.0 and b[1] <= 20.3]
DLA_BINS    = [b for b in NHI_BINS if b[0] >= 20.3]
# Categories needing a redshift-scrambled control (low-NHI = contamination-prone)
CONTROL_CATEGORIES = {"lls": LLS_BINS, "subdla": SUBDLA_BINS}

# Selection
P_DLA_MIN = 0.97
SNR_FOREST_MIN = 2.0
Z_QSO_MIN = 3.0
Z_DLA_TO_QSO_MARGIN = 0.05    # exclude proximate absorbers
LYA = 1215.67
LYB = 1025.72

# Per-bin sample cap. /scratch HDF5 random reads are slow (~0.4s/row);
# batch-read sorted rows to amortize.
MAX_PER_BIN = 800
BATCH_SIZE = 200
# Continuum-window minimum pixel count + min good pixels per spectrum.
MIN_CONT_PIX = 5
MIN_GOOD_PIX = 100
# Output pixels whose bracketing source samples are farther apart than
# this (Å, absorber rest) fell in a masked gap np.interp bridged → NaN.
MAX_INTERP_GAP = 2.0

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

# Metal species to suppress (markers + labels + zoom panels) per category.
# Sub-DLA labels are kept in full — the 20.0–20.3 bin is DLA-like and
# does show the low-ion metals (FeII / PII), so suppressing them hid
# real features. CIII is still dropped from DLA-only plots per request.
# Lyman-series and other species are unaffected.
EXCLUDE_SPECIES = {
    "all":    frozenset(),
    "lls":    frozenset(),
    "subdla": frozenset(),
    "dla":    frozenset({"CIII"}),
}


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------

def load_catalog():
    print(f"loading catalog: {DLACAT}", flush=True)
    with fits.open(DLACAT) as f:
        cat = np.array(f["DLACAT"].data)
    print(f"  raw rows: {len(cat)}", flush=True)
    return cat


def select(cat: np.ndarray) -> np.ndarray:
    """Purity / SNR / redshift / forest-region selection. NHI floor 17.2."""
    z_qso = cat["Z_QSO"]
    z_dla = cat["Z_DLA"]
    z_dla_lya_obs = (1 + z_dla) * LYA
    z_qso_lya_obs = (1 + z_qso) * LYA
    z_qso_lyb_obs = (1 + z_qso) * LYB
    in_forest = (z_dla_lya_obs > z_qso_lyb_obs) & (z_dla_lya_obs < z_qso_lya_obs)
    not_proximate = z_dla < (z_qso - Z_DLA_TO_QSO_MARGIN)

    keep = (
        (cat["P_DLA"] > P_DLA_MIN)
        & (cat["SNR_FOREST"] > SNR_FOREST_MIN)
        & (cat["DLAFLAG"] == 0)
        & (z_qso > Z_QSO_MIN)
        & in_forest
        & not_proximate
        & (cat["NHI"] >= 17.2)
        & (cat["NHI"] <= 23.0)
    )
    print(f"  P_DLA > {P_DLA_MIN}:        {(cat['P_DLA'] > P_DLA_MIN).sum()}", flush=True)
    print(f"  SNR_FOREST > {SNR_FOREST_MIN}:   {(cat['SNR_FOREST'] > SNR_FOREST_MIN).sum()}", flush=True)
    print(f"  DLAFLAG == 0:       {(cat['DLAFLAG'] == 0).sum()}", flush=True)
    print(f"  Z_QSO > {Z_QSO_MIN}:        {(z_qso > Z_QSO_MIN).sum()}", flush=True)
    print(f"  in Lyα forest:      {in_forest.sum()}", flush=True)
    print(f"  not proximate:      {not_proximate.sum()}", flush=True)
    print(f"  NHI ∈ [17.2, 23]:   {((cat['NHI'] >= 17.2) & (cat['NHI'] <= 23.0)).sum()}", flush=True)
    print(f"  ALL combined:       {keep.sum()}", flush=True)
    return cat[keep]


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
            color = BIN_COLORS[NHI_BINS.index((lo, hi))]
            ax.hist(qval[m], bins=bins_edges, histtype="step", lw=1.4,
                    color=color, density=True,
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
    fig.savefig(OUT_DIR / "zhist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR / 'zhist.png'}", flush=True)

    for (lo, hi) in NHI_BINS:
        m = (cat["NHI"] >= lo) & (cat["NHI"] < hi)
        if m.sum() == 0:
            summary.append(f"{lo:.2f} {hi:.2f}   0   - - - -")
            continue
        zq = np.median(cat["Z_QSO"][m])
        zd = np.median(cat["Z_DLA"][m])
        ratio = np.median((1 + cat["Z_QSO"][m]) / (1 + cat["Z_DLA"][m]))
        # QSO Lyα emission position in absorber rest frame
        qso_lya_absframe = LYA * ratio
        summary.append(f"{lo:.2f} {hi:.2f}   {int(m.sum())}   "
                       f"{zq:.4f}  {zd:.4f}  {ratio:.4f}  {qso_lya_absframe:.2f}")
    (OUT_DIR / "zhist_summary.txt").write_text("\n".join(summary) + "\n")
    print(f"[saved] {OUT_DIR / 'zhist_summary.txt'}", flush=True)
    print("\n".join(summary), flush=True)


# ---------------------------------------------------------------------------
# resample + stack core
# ---------------------------------------------------------------------------

def _resample_spectrum(f, iv, m, wave_obs, z, rest_grid):
    """Mask bad pixels, shift to rest frame at redshift z, local-continuum
    normalize, resample onto rest_grid. Returns resampled array or None.

    `f` is consumed (modified in place) — pass a copy if you need it again."""
    bad = (m != 0) | (iv <= 0) | ~np.isfinite(f)
    f[bad] = np.nan

    rest = wave_obs / (1.0 + z)
    cont = None
    # Primary window [1410, 1520] Å absorber-rest is genuinely flat
    # continuum: well redward of the QSO Lyα emission bump (~1313 Å in
    # the absorber frame), clear of strong DLA metal lines (SiIV 1403 /
    # SiII 1527 just outside), and wide (110 Å ≈ 550 native pixels) so
    # sky-line masking essentially never leaves < MIN_CONT_PIX good
    # pixels. [1340, 1380] is a fallback for the rare highly-masked or
    # very-high-z (z_DLA > 5.5, red edge clipped) case; if both fail the
    # spectrum is dropped and counted in n_skip_norm.
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
    med = np.nanmedian(stack, axis=0)
    mad = np.nanmedian(np.abs(stack - med), axis=0) * 1.4826
    bad = np.abs(stack - med) > 3 * mad[None, :]
    clipped = np.where(bad, np.nan, stack)
    curve = np.nanmedian(clipped, axis=0)
    counts = np.sum(np.isfinite(clipped), axis=0)
    curve[counts < 50] = np.nan
    return curve, counts


def read_bin_spectra(cat_bin, arch_tid_to_row, archive, rest_grid, bin_name,
                     control_dz=None):
    """Batch-read a bin's spectra and resample. Returns the raw resampled
    stack array (n_spec, n_pix); if `control_dz` is given (per-spectrum
    redshift offset), also returns a control stack resampled at z+Δz.

    Returns (real_stack, control_stack_or_None)."""
    flux = archive["flux"]
    ivar = archive["ivar"]
    mask = archive["mask"]
    wave_obs = archive["wavelength"][:]

    rng = np.random.default_rng(42)
    if len(cat_bin) > MAX_PER_BIN:
        sel = rng.choice(len(cat_bin), MAX_PER_BIN, replace=False)
        cat_bin = cat_bin[sel]
        if control_dz is not None:
            control_dz = control_dz[sel]

    tids = cat_bin["TARGETID"].astype(np.int64)
    row_idx = np.array([arch_tid_to_row.get(int(t), -1) for t in tids],
                       dtype=np.int64)
    found = row_idx >= 0
    n_skip_tid = int((~found).sum())
    cat_bin = cat_bin[found]
    row_idx = row_idx[found]
    if control_dz is not None:
        control_dz = control_dz[found]
    n = len(row_idx)
    print(f"  bin={bin_name}: {n} spectra to read (skip-no-TID={n_skip_tid})",
          flush=True)

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
    return real, ctrl


def build_archive_tid_map(archive):
    print(f"building TARGETID→row map ({archive['catalog'].shape[0]} rows)…",
          flush=True)
    tids = archive["catalog"]["TARGETID"][:]
    return {int(t): i for i, t in enumerate(tids)}


def compute_stacks():
    """Read the archive, build per-bin stacks + combined LLS/sub-DLA
    real-vs-control stacks. Slow (~45 min on /scratch)."""
    cat = load_catalog()
    cat = select(cat)
    dump_zhist(cat)

    rest_grid = 10 ** np.arange(np.log10(REST_LAMBDA_MIN),
                                np.log10(REST_LAMBDA_MAX), DLOG_LAMBDA)
    print(f"rest grid: {rest_grid[0]:.2f}..{rest_grid[-1]:.2f} Å, "
          f"n={len(rest_grid)}", flush=True)

    # Which categories want a control? (low-NHI bins)
    control_bins = set()
    for cat_bins in CONTROL_CATEGORIES.values():
        control_bins.update(cat_bins)

    with h5py.File(LOA_ARCHIVE, "r") as archive:
        tid_to_row = build_archive_tid_map(archive)

        per_bin = {}             # (lo,hi) -> (curve, counts, n_total)
        raw_real = {}            # (lo,hi) -> raw real stack array
        raw_ctrl = {}            # (lo,hi) -> raw control stack array
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
                # panel (a few 1000 km/s), so a real metal line shifts
                # well clear of its panel and the per-spectrum-random
                # offsets fully smear it out in the control median.
                cdz = (rng.uniform(0.15, 0.35, len(sub))
                       * rng.choice([-1.0, 1.0], len(sub)))
            real, ctrl = read_bin_spectra(
                sub, tid_to_row, archive, rest_grid, f"[{lo},{hi})",
                control_dz=cdz)
            curve, counts = _sigma_clip_median(real)
            per_bin[(lo, hi)] = (curve, counts, len(sub))
            raw_real[(lo, hi)] = real
            if ctrl is not None:
                raw_ctrl[(lo, hi)] = ctrl

    # Combined real-vs-control stacks per low-NHI category
    combined = {}   # name -> (real_curve, real_counts, ctrl_curve, ctrl_counts, n)
    for name, cat_bins in CONTROL_CATEGORIES.items():
        reals = [raw_real[b] for b in cat_bins if b in raw_real]
        ctrls = [raw_ctrl[b] for b in cat_bins if b in raw_ctrl]
        if not reals:
            continue
        pooled_real = np.concatenate(reals, axis=0)
        pooled_ctrl = np.concatenate(ctrls, axis=0)
        rc, rn = _sigma_clip_median(pooled_real)
        cc, cn = _sigma_clip_median(pooled_ctrl)
        combined[name] = (rc, rn, cc, cn, len(pooled_real))
        print(f"combined {name}: {len(pooled_real)} spectra "
              f"(real + scrambled control)", flush=True)

    # counts log
    with (OUT_DIR / "counts.txt").open("w") as fh:
        fh.write(f"# P_DLA > {P_DLA_MIN}, SNR_FOREST > {SNR_FOREST_MIN}, "
                 f"DLAFLAG=0, Z_QSO > {Z_QSO_MIN}, in-forest, not-proximate\n")
        fh.write(f"# MAX_PER_BIN={MAX_PER_BIN}\n# NHI_lo NHI_hi n_candidates\n")
        for (lo, hi) in NHI_BINS:
            n_cand = int(((cat["NHI"] >= lo) & (cat["NHI"] < hi)).sum())
            fh.write(f"{lo:.2f} {hi:.2f} {n_cand}\n")
    return rest_grid, per_bin, combined


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def save_curves(rest_grid, per_bin, combined):
    payload = {"rest_grid": rest_grid}
    for (lo, hi), (curve, counts, n_total) in per_bin.items():
        key = f"{lo:.2f}_{hi:.2f}"
        payload[f"curve_{key}"] = curve
        payload[f"counts_{key}"] = counts
        payload[f"ntot_{key}"] = np.int64(n_total)
    for name, (rc, rn, cc, cn, n) in combined.items():
        payload[f"comb_real_{name}"] = rc
        payload[f"comb_realcnt_{name}"] = rn
        payload[f"comb_ctrl_{name}"] = cc
        payload[f"comb_ctrlcnt_{name}"] = cn
        payload[f"comb_n_{name}"] = np.int64(n)
    np.savez(NPZ_PATH, **payload)
    print(f"[saved] {NPZ_PATH}", flush=True)


def load_curves():
    d = np.load(NPZ_PATH)
    rest_grid = d["rest_grid"]
    per_bin = {}
    for (lo, hi) in NHI_BINS:
        key = f"{lo:.2f}_{hi:.2f}"
        if f"curve_{key}" not in d:
            continue
        per_bin[(lo, hi)] = (d[f"curve_{key}"], d[f"counts_{key}"],
                             int(d[f"ntot_{key}"]))
    combined = {}
    for name in CONTROL_CATEGORIES:
        if f"comb_real_{name}" not in d:
            continue
        combined[name] = (d[f"comb_real_{name}"], d[f"comb_realcnt_{name}"],
                           d[f"comb_ctrl_{name}"], d[f"comb_ctrlcnt_{name}"],
                           int(d[f"comb_n_{name}"]))
    return rest_grid, per_bin, combined


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
    """Full 900–1600 Å overview for the given NHI bins."""
    fig, ax = plt.subplots(figsize=(15, 6))
    for (lo, hi) in bins:
        if (lo, hi) not in per_bin:
            continue
        color = BIN_COLORS[NHI_BINS.index((lo, hi))]
        curve, counts, n_total = per_bin[(lo, hi)]
        n_eff = int(np.nanmedian(counts[counts > 0])) if (counts > 0).any() else 0
        ax.plot(rest_grid, curve, color=color, lw=1.2, alpha=0.85,
                label=f"log NHI [{lo:.1f}, {hi:.1f})  n={n_total} "
                      f"(median_pix={n_eff})")
    ax.axhline(1.0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlim(REST_LAMBDA_MIN, REST_LAMBDA_MAX)
    ax.set_ylim(0.0, 1.7)
    _draw_metal_labels(ax, 0.0, 1.7, REST_LAMBDA_MIN, REST_LAMBDA_MAX, exclude)
    ax.set_xlabel(r"absorber rest-frame wavelength [Å]")
    ax.set_ylabel("median stacked flux (norm. to 1 at 1410–1520 Å rest)")
    ax.set_title(
        f"{subtitle} — high-purity (P_DLA > {P_DLA_MIN}), "
        f"SNR_forest > {SNR_FOREST_MIN}, z_QSO > {Z_QSO_MIN}, "
        "Lyα-forest detection only")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR / fname}", flush=True)


def _local_continuum_norm(x, y, lines, core_half=2.5):
    """Divide y by a linear fit to the in-window continuum (line cores
    within ±core_half Å excluded). Reveals few-% lines on a sloped base."""
    cont_mask = np.isfinite(y)
    for _, ln_w in lines:
        cont_mask &= np.abs(x - ln_w) > core_half
    if cont_mask.sum() >= 5:
        coef = np.polyfit(x[cont_mask], y[cont_mask], 1)
        return y / np.polyval(coef, x)
    med = np.nanmedian(y)
    return y / med if (med and np.isfinite(med)) else y


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
            color = BIN_COLORS[NHI_BINS.index((lo, hi))]
            curve, counts, n_total = per_bin[(lo, hi)]
            y = _local_continuum_norm(x, curve[sel].astype(np.float64), lines)
            ax.plot(x, y, color=color, lw=1.3, alpha=0.85,
                    label=f"NHI [{lo:.1f},{hi:.1f})")
            if np.isfinite(y).any():
                panel_min = min(panel_min, np.nanpercentile(y, 1))
        for ln_name, ln_w in lines:
            ax.axvline(ln_w, color="k", lw=0.7, ls="--", alpha=0.6)
        # adaptive y-limits: floor at the deepest line (clamped), head at 1.06
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


def plot_control(rest_grid, combined, name, fname, exclude=frozenset()):
    """Real vs redshift-scrambled-control zoom panels for one low-NHI
    category. The decisive plot: a coherent dip in the real stack that is
    absent in the control = the detections are real absorbers."""
    if name not in combined:
        print(f"[skip] no combined stack for {name}", flush=True)
        return
    rc, rn, cc, cn, n = combined[name]
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
        y_real = _local_continuum_norm(x, rc[sel].astype(np.float64), lines)
        y_ctrl = _local_continuum_norm(x, cc[sel].astype(np.float64), lines)
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
    label = {"lls": "LLS [17.2, 19)", "subdla": "sub-DLA [19, 20.3)"}[name]
    fig.suptitle(
        f"Real vs redshift-scrambled control — combined {label}, "
        f"n={n} spectra. A coherent dip in RED (real) absent in GREY "
        "(control) confirms the detections are real absorbers, not "
        "false positives. CIV 1548/1551 is the decisive panel.",
        fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR / fname}", flush=True)


def render_all(rest_grid, per_bin, combined):
    plot_overview(rest_grid, per_bin, NHI_BINS, "stack_all.png",
                  "Real-LOA LLS / sub-DLA / DLA, all NHI bins",
                  exclude=EXCLUDE_SPECIES["all"])
    plot_metal_zoom(rest_grid, per_bin, NHI_BINS, "stack_metal_zoom_all.png",
                    "Metal-line zoom — all NHI bins",
                    exclude=EXCLUDE_SPECIES["all"])
    for tag, bins, label in [
        ("lls", LLS_BINS, "Real-LOA LLS only (log NHI 17.2–19)"),
        ("subdla", SUBDLA_BINS, "Real-LOA sub-DLAs only (log NHI 19–20.3)"),
        ("dla", DLA_BINS, "Real-LOA DLAs only (log NHI ≥ 20.3)"),
    ]:
        exc = EXCLUDE_SPECIES[tag]
        plot_overview(rest_grid, per_bin, bins, f"stack_{tag}.png", label,
                      exclude=exc)
        plot_metal_zoom(rest_grid, per_bin, bins, f"stack_metal_zoom_{tag}.png",
                        f"Metal-line zoom — {label}", exclude=exc)
    for name in CONTROL_CATEGORIES:
        plot_control(rest_grid, combined, name, f"stack_control_{name}.png",
                     exclude=EXCLUDE_SPECIES.get(name, frozenset()))


def main():
    # `--zhist-only`: just the per-bin redshift diagnostics (catalog-only,
    # runs in seconds — no spectrum reads).
    if "--zhist-only" in sys.argv:
        cat = load_catalog()
        cat = select(cat)
        dump_zhist(cat)
        return
    if "--plot-only" in sys.argv:
        if not NPZ_PATH.exists():
            raise SystemExit(f"no cached curves at {NPZ_PATH}; "
                             "run without --plot-only first")
        print(f"loading cached curves from {NPZ_PATH}", flush=True)
        rest_grid, per_bin, combined = load_curves()
    else:
        rest_grid, per_bin, combined = compute_stacks()
        save_curves(rest_grid, per_bin, combined)
    render_all(rest_grid, per_bin, combined)


if __name__ == "__main__":
    main()
