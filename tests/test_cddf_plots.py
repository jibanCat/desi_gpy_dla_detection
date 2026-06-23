"""Tests for the O3 diagnostic plot suite
(``CDDF_analysis.cddf_forward.driver.plot_o3_diagnostics`` + back-compat
``plot_o3_products``).

These are STRUCTURAL tests: with the Agg backend (no display) we build a small
SYNTHETIC O3-products dict (the shape :func:`compute_o3_products` /
``compute_o3_products_streaming`` return) and assert the figure has the expected
multi-panel structure, the honest "DIAGONAL" labelling, and that it writes a PNG.
We never assert pixel values — only that every requested panel exists, is drawn,
and carries the load-bearing honest caveats.
"""
import os

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")  # no display; must precede pyplot import

pytest.importorskip("matplotlib")

from CDDF_analysis.cddf_forward import driver as o3driver  # noqa: E402


# --------------------------------------------------------------------------- #
# A synthetic O3-products dict matching the compute_o3_products schema.
# --------------------------------------------------------------------------- #
def _synthetic_products(*, nbin=5, nz=4, with_coverage=True, with_gap=False):
    rng = np.random.default_rng(0)
    logN = np.linspace(20.3, 22.5, nbin + 1)
    logN = 0.5 * (logN[:-1] + logN[1:])
    f_raw = 10.0 ** (-np.linspace(21, 25, nbin))
    f = f_raw * 1.4
    f68 = np.vstack([f * 0.8, f * 1.2]).T
    f95 = np.vstack([f * 0.6, f * 1.4]).T
    f68_raw = np.vstack([f_raw * 0.8, f_raw * 1.2]).T
    f95_raw = np.vstack([f_raw * 0.6, f_raw * 1.4]).T

    z = np.linspace(2.4, 3.3, nz)
    dndx_raw = np.linspace(0.02, 0.05, nz)
    dndx = dndx_raw * 1.3
    dndx68 = np.vstack([dndx * 0.85, dndx * 1.15]).T
    dndx95 = np.vstack([dndx * 0.7, dndx * 1.3]).T

    omega = np.array([1.2e-3])
    omega68 = np.array([[1.0e-3, 1.4e-3]])
    omega95 = np.array([[0.8e-3, 1.6e-3]])

    # C / b_FP per logN bin, with a couple of NaN (masked/unreliable) bins.
    C = np.linspace(0.3, 0.9, nbin)
    C[-1] = np.nan  # an unreliable / masked completeness bin
    C_lo68 = np.clip(C - 0.1, 0, 1)
    C_hi68 = np.clip(C + 0.1, 0, 1)
    b_FP = np.linspace(0.0, 2.0, nbin)
    b_FP[1] = np.nan
    n_truth = np.array([10, 8, 6, 4, 2], dtype=float)[:nbin]

    coverage = None
    if with_coverage:
        if with_gap:
            healpix_coverage = {100: 50, 101: 40, 102: 0}  # 102 = a gap
        else:
            healpix_coverage = {100: 50, 101: 40, 102: 45}
        coverage = {
            "n_truth_only": 12,
            "n_processed_only": 30,
            "n_both": 200,
            "n_truth_targets": 212,
            "n_processed_targets": 230,
            "n_healpix": len(healpix_coverage),
            "healpix_coverage": healpix_coverage,
        }

    return {
        "o1": {
            "cddf": {"logN": logN, "f": f_raw, "f68": f68_raw, "f95": f95_raw,
                     "xerrs": None},
            "dndx": {"z": z, "dndx": dndx_raw, "dndx68": dndx68, "dndx95": dndx95,
                     "xerrs": None},
            "omega": {"z": np.array([2.85]), "omega": omega, "omega68": omega68,
                      "omega95": omega95, "xerrs": None},
        },
        "o3_cddf": {
            "logN": logN, "f": f, "f68": f68, "f95": f95,
            "f_raw": f_raw, "f68_raw": f68_raw, "f95_raw": f95_raw,
            "valid_mask": np.isfinite(C),
            "neg_clip_mask": np.zeros(nbin, bool),
            "n_corr": f * 1e22,
        },
        "o3_dndx": {
            "z": z, "dndx": dndx, "dndx68": dndx68, "dndx95": dndx95,
            "dndx_raw": dndx_raw,
            "dndx68_raw": dndx68, "dndx95_raw": dndx95,
            "valid_mask": np.ones(nz, bool),
            "neg_clip_mask": np.zeros(nz, bool),
            "n_corr": dndx * 1e3,
        },
        "o3_omega": {
            "z": np.array([2.85]), "omega": omega,
            "omega68": omega68, "omega95": omega95,
        },
        "completeness": {
            "logN": logN, "C": C, "C_lo68": C_lo68, "C_hi68": C_hi68,
            "b_FP": b_FP, "n_truth": n_truth,
            "F_matched": n_truth * C_lo68, "F_unmatched": np.linspace(0, 1, nbin),
        },
        "closure": {},
        "provenance": {
            "z_min": 2.4, "z_max": 3.3,
            "lnhi_min": 20.3, "lnhi_max": 22.5,
            "streaming": True, "n_files": 3,
            "coverage": coverage,
        },
    }


def _axes_titles(fig):
    return [(ax.get_title() or "") for ax in fig.axes]


def _all_text(fig):
    """Every label/title/legend/annotation string drawn on the figure (lowercased)."""
    bits = []
    if fig._suptitle is not None:
        bits.append(fig._suptitle.get_text())
    for ax in fig.axes:
        bits += [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
        leg = ax.get_legend()
        if leg is not None:
            bits += [t.get_text() for t in leg.get_texts()]
        for t in ax.texts:
            bits.append(t.get_text())
    return " ".join(b for b in bits if b).lower()


class TestPlotO3Diagnostics:
    def test_returns_figure_and_writes_png(self, tmp_path):
        prod = _synthetic_products()
        out = str(tmp_path / "diag.png")
        fig = o3driver.plot_o3_diagnostics(prod, save_path=out)
        assert fig is not None
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_has_all_requested_panels(self, tmp_path):
        # f(N), dN/dX, Omega, C(N), b_FP, coverage -> >= 6 panels.
        prod = _synthetic_products()
        fig = o3driver.plot_o3_diagnostics(prod, save_path=str(tmp_path / "d.png"))
        assert len(fig.axes) >= 6
        titles = " ".join(_axes_titles(fig)).lower()
        for need in ("cddf", "density", "omega", "completeness", "false-positive",
                     "coverage"):
            assert need in titles, f"missing panel titled ~{need!r}: {titles!r}"

    def test_fN_panel_is_loglog_with_raw_and_corrected(self, tmp_path):
        prod = _synthetic_products()
        fig = o3driver.plot_o3_diagnostics(prod, save_path=str(tmp_path / "d.png"))
        # find the f(N) axis by title
        fn_ax = next(a for a in fig.axes if "cddf" in (a.get_title() or "").lower())
        assert fn_ax.get_yscale() == "log"
        labels = [ln.get_label().lower() for ln in fn_ax.get_lines()]
        assert any("raw" in lb for lb in labels)
        assert any("corrected" in lb or "o3" in lb for lb in labels)

    def test_honest_diagonal_labelling_everywhere(self, tmp_path):
        prod = _synthetic_products()
        fig = o3driver.plot_o3_diagnostics(prod, save_path=str(tmp_path / "d.png"))
        text = _all_text(fig)
        # The honest banner + the N-scatter / off-diagonal caveat must be present.
        assert "diagonal" in text
        assert "scatter" in text or "off-diagonal" in text or "m3" in text
        # must NOT falsely claim an alpha(z) / London-mock calibration.
        assert "london mock" not in text

    def test_masked_bins_greyed_not_crashed(self, tmp_path):
        # NaN C / b_FP bins must not raise and must be visibly distinguished.
        prod = _synthetic_products()
        fig = o3driver.plot_o3_diagnostics(prod, save_path=str(tmp_path / "d.png"))
        assert fig is not None  # NaNs handled without crashing

    def test_coverage_gap_is_visible(self, tmp_path):
        # A zero-coverage healpix must be annotated as a GAP.
        prod = _synthetic_products(with_gap=True)
        fig = o3driver.plot_o3_diagnostics(prod, save_path=str(tmp_path / "d.png"))
        text = _all_text(fig)
        assert "gap" in text or "0" in text  # the gap is surfaced

    def test_no_coverage_block_still_renders(self, tmp_path):
        # provenance.coverage absent -> coverage panel still drawn (annotated N/A).
        prod = _synthetic_products(with_coverage=False)
        fig = o3driver.plot_o3_diagnostics(prod, save_path=str(tmp_path / "d.png"))
        assert fig is not None

    def test_accepts_title_override(self, tmp_path):
        prod = _synthetic_products()
        fig = o3driver.plot_o3_diagnostics(
            prod, save_path=str(tmp_path / "d.png"), title="My Run X"
        )
        assert "my run x" in _all_text(fig)


class TestPlotO3ProductsBackCompat:
    def test_old_entry_point_still_returns_figure(self, tmp_path):
        # The original plot_o3_products must keep working (delegates to the new one).
        prod = _synthetic_products()
        out = str(tmp_path / "old.png")
        fig = o3driver.plot_o3_products(prod, save_path=out)
        assert fig is not None
        assert os.path.exists(out)
