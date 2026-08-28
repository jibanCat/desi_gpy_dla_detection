"""A5 regression (R-041A, 2026-08-28): every completeness cell overridden by the H2 patch must
have its MC Beta draw centred on the adopted value — including the explicit --gap-c cell, which
previously kept the FROZEN counts and therefore drew around the frozen k/n."""
import json
import os
import types

import numpy as np
import pytest


def _mm():
    mm = types.SimpleNamespace()
    mm.nhi_edges = np.array([19.5, 20.0, 20.3, 20.5, 21.0, 21.5, 22.0, np.inf])
    mm.snr_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, np.inf])
    n_s, n_n = 8, 7
    mm.completeness = np.full((n_s, n_n), 0.5)
    mm.cmp_nfound = np.full((n_s, n_n), 300.0)      # frozen counts: k/n = 0.3 everywhere
    mm.cmp_nfid = np.full((n_s, n_n), 1000.0)
    return mm


def test_gap_c_draw_is_centred_on_the_adopted_value(monkeypatch, tmp_path):
    from CDDF_analysis.hbi import track_c_tf_hz as HZ
    from CDDF_analysis.hbi.cddf_catalog_hbi import _draw_beta_cell
    # a synthetic H2 table: only the gap cell is overridden through gap_c
    monkeypatch.setattr(HZ, "h2_c_table", lambda window, gap_treatment="frozen": ({"[20.3,20.5)": (None, None, None)}, {"synthetic": True}))
    mm = _mm()
    HZ.patch_mm_with_h2(mm, "lya", None, "frozen", gap_c=0.62, gap_c_neff=40.0)
    j = 2                                            # the [20.3, 20.5) cell
    rng = np.random.default_rng(0)
    draws = np.array([_draw_beta_cell(rng, mm.cmp_nfound, mm.cmp_nfid)[3, j] for _ in range(4000)])
    assert abs(draws.mean() - 0.62) < 0.01           # centred on gap_c, not on the frozen 0.3
    assert mm.completeness[3, j] == 0.62 and mm.cmp_nfid[3, j] == 40.0
    assert mm.cmp_nfound[0, j] == 300.0              # SNR < 2 rows untouched


def test_measured_cells_draw_centred_on_k_over_n(monkeypatch):
    from CDDF_analysis.hbi import track_c_tf_hz as HZ
    from CDDF_analysis.hbi.cddf_catalog_hbi import _draw_beta_cell
    monkeypatch.setattr(HZ, "h2_c_table", lambda window, gap_treatment="frozen": ({"[20.5,21.0)": (0.7, 70, 100)}, {"synthetic": True}))
    mm = _mm()
    HZ.patch_mm_with_h2(mm, "lya", None, "frozen")
    rng = np.random.default_rng(1)
    draws = np.array([_draw_beta_cell(rng, mm.cmp_nfound, mm.cmp_nfid)[4, 3] for _ in range(4000)])
    assert abs(draws.mean() - 0.7) < 0.01


def test_neff_from_record_reproduces_the_68pct_width(tmp_path):
    from CDDF_analysis.hbi import track_c_tf_hz as HZ
    rec = {"posterior": {"C_gap_p16_50_84": [0.4067, 0.4962, 0.5925]}}
    p = tmp_path / "rec.json"; p.write_text(json.dumps(rec))
    n_eff = HZ._neff_from_cgap_record(str(p))
    m = 0.4962; sd = np.sqrt(m * (1 - m) / (n_eff + 1))
    assert abs(sd - 0.5 * (0.5925 - 0.4067)) < 1e-6
