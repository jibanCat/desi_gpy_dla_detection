# Masked-Spline Pseudo-Continuum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a post-stack masked cubic-spline pseudo-continuum fit to `examples/stack_real_loa_dlas.py`, so the stacked composites are divided by a smooth global pseudo-continuum and metal lines sit on a flat baseline.

**Architecture:** A fixed-knot cubic spline (`scipy.interpolate.LSQUnivariateSpline`) fit to the composite with metal/H I lines masked, wrapped in a Schlegel-style iterative sigma-rejection loop. Computed at plot time from the cached composite curves (the npz is unchanged). The per-zoom-panel linear `_local_continuum_norm` is replaced by this global fit. A mock-injection unit test is built first.

**Tech Stack:** Python, numpy, scipy (`LSQUnivariateSpline`), matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-05-18-stack-pseudo-continuum-design.md`

---

## File Structure

- **Modify** `examples/stack_real_loa_dlas.py` — add pseudo-continuum constants, `_ensure_outdir()`, `_continuum_mask()`, `_safe_lsq_spline()`, `fit_pseudo_continuum()`, `plot_pseudo_continuum_qc()`; rewire `plot_metal_zoom()` and `plot_control()`; delete `_local_continuum_norm()`.
- **Create** `tests/test_stack_pseudo_continuum.py` — mock composite generator + unit tests.

All pseudo-continuum logic lives in the existing examples script (repo style: extend existing files, no new module). Tests import it via `from examples.stack_real_loa_dlas import ...`.

---

## Task 1: Testability cleanup — move the module-level mkdir

**Files:**
- Modify: `examples/stack_real_loa_dlas.py` (the `OUT_DIR` block near line 50, and `main()` near line 755)

Importing the module currently runs `OUT_DIR.mkdir()` at import — a filesystem side effect that pollutes the repo when `tests/` imports it. Move it into a function called from `main()`.

- [ ] **Step 1: Remove the module-level mkdir**

In `examples/stack_real_loa_dlas.py`, find:

```python
OUT_DIR = Path(__file__).resolve().parent.parent / "docs/notes/2026-05-15_stack_real_loa_dlas"
OUT_DIR.mkdir(parents=True, exist_ok=True)
NPZ_PATH = OUT_DIR / "stack_curves.npz"
```

Replace with:

```python
OUT_DIR = Path(__file__).resolve().parent.parent / "docs/notes/2026-05-15_stack_real_loa_dlas"
NPZ_PATH = OUT_DIR / "stack_curves.npz"


def _ensure_outdir():
    """Create the output directory. Called from main() — kept out of
    module import so tests can import the pure functions side-effect-free."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 2: Call `_ensure_outdir()` at the start of `main()`**

Find `def main():` and insert `_ensure_outdir()` as the first statement of the function body (before the `--zhist-only` check):

```python
def main():
    _ensure_outdir()
    # `--zhist-only`: just the per-bin redshift diagnostics (catalog-only,
```

- [ ] **Step 3: Verify the module imports without creating the directory**

Run:
```bash
cd /home/mfho/desi_gpy_dla_detection
python -c "import importlib, sys; sys.path.insert(0,'.'); import examples.stack_real_loa_dlas; print('import OK, no mkdir side effect')"
python -c "import ast; ast.parse(open('examples/stack_real_loa_dlas.py').read()); print('syntax OK')"
```
Expected: both print OK.

- [ ] **Step 4: Commit**

```bash
git add examples/stack_real_loa_dlas.py
git commit -m "stack: move OUT_DIR mkdir out of module import for testability"
```

---

## Task 2: Pseudo-continuum constants + `_continuum_mask`

**Files:**
- Modify: `examples/stack_real_loa_dlas.py` (add constants after the `MAX_INTERP_GAP` block ~line 99; add functions in a new section before the plotting section)
- Create: `tests/test_stack_pseudo_continuum.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_stack_pseudo_continuum.py`:

```python
"""Unit tests for the masked-spline pseudo-continuum fit in
examples/stack_real_loa_dlas.py — built on a synthetic composite with a
known truth pseudo-continuum and known injected absorption lines.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.stack_real_loa_dlas import (  # noqa: E402
    REST_LAMBDA_MIN, REST_LAMBDA_MAX, DLOG_LAMBDA, PCONT_LAMBDA_MIN,
    SIGMA_V, _C_KM_S, METAL_LINES, _continuum_mask,
)


def _rest_grid():
    return 10 ** np.arange(np.log10(REST_LAMBDA_MIN),
                           np.log10(REST_LAMBDA_MAX), DLOG_LAMBDA)


def make_mock_composite(rest_grid, *, inject=True, seed=0):
    """Synthetic composite with a known truth pseudo-continuum.

    Returns (curve, counts, P_true, lines) where `lines` is a list of
    (lambda0, depth, sigma) for each injected absorption line."""
    rng = np.random.default_rng(seed)
    lam = rest_grid
    # truth pseudo-continuum: gentle slope * smeared QSO Lya bump * forest decrement
    slope = 1.0 + 0.10 * (lam - 1200.0) / 900.0
    bump = 1.0 + 0.25 * np.exp(-0.5 * ((lam - 1280.0) / 60.0) ** 2)
    forest = 0.6 + 0.4 / (1.0 + np.exp(-(lam - 1180.0) / 25.0))
    P_true = slope * bump * forest
    # counts: ramp up from the blue edge, with one mid-band coverage hole
    counts = np.clip(50.0 + 0.9 * (lam - 700.0), 0.0, 800.0)
    counts[(lam > 1080.0) & (lam < 1090.0)] = 0.0
    # injected absorption
    absorption = np.zeros_like(lam)
    lines = []
    if inject:
        specs = [(1031.91, 0.20), (1063.18, 0.12), (1143.23, 0.10),
                 (1190.42, 0.18), (1260.42, 0.22), (1334.53, 0.15),
                 (1393.76, 0.25), (1548.20, 0.35),
                 (1117.0, 0.06), (1450.0, 0.05)]  # last 2: NOT in METAL_LINES
        for lam0, depth in specs:
            sig = lam0 * SIGMA_V / _C_KM_S
            absorption += depth * np.exp(-0.5 * ((lam - lam0) / sig) ** 2)
            lines.append((lam0, depth, sig))
    curve = P_true * (1.0 - absorption)
    noise = np.where(counts > 0,
                     0.02 / np.sqrt(np.maximum(counts, 1.0) / 400.0), 0.0)
    curve = curve + rng.normal(0.0, np.maximum(noise, 1e-6))
    curve[counts < 50] = np.nan
    return curve, counts, P_true, lines


def test_continuum_mask_excludes_lines_and_blue_end():
    rg = _rest_grid()
    curve, counts, _, lines = make_mock_composite(rg)
    fit_ok = _continuum_mask(rg, curve, counts)
    # nothing below PCONT_LAMBDA_MIN is kept
    assert not fit_ok[rg < PCONT_LAMBDA_MIN].any()
    # the CIV 1548 metal line centre is masked out
    civ = np.argmin(np.abs(rg - 1548.20))
    assert not fit_ok[civ]
    # a clean window (1500 A, no METAL_LINES within a few A) is kept
    clean = np.argmin(np.abs(rg - 1500.0))
    assert fit_ok[clean]
    # the coverage hole [1080,1090] is excluded
    assert not fit_ok[(rg > 1081.0) & (rg < 1089.0)].any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/mfho/desi_gpy_dla_detection && python -m pytest tests/test_stack_pseudo_continuum.py -v`
Expected: FAIL — `ImportError: cannot import name 'PCONT_LAMBDA_MIN'` (constants/function not defined yet).

- [ ] **Step 3: Add the constants**

In `examples/stack_real_loa_dlas.py`, after the `MAX_INTERP_GAP = 2.0` line and its comment block, add:

```python
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
HI_MASK_HALF = {             # H I core+near-wing mask half-widths (Å)
    "Lyα": 25.0, "Lyβ": 15.0, "Lyγ": 8.0, "Ly4": 5.0, "Ly5": 5.0,
}
KNOT_SPACING = 15.0          # interior knot spacing (Å)
SPLINE_ORDER = 3             # cubic
REJECT_SIGMA = 5.0           # iterative-rejection threshold (robust σ)
MAX_REJECT_ITER = 10         # rejection iteration cap
```

- [ ] **Step 4: Add `_continuum_mask`**

In `examples/stack_real_loa_dlas.py`, add a new section just before the `# ---- plotting ----` divider comment:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_stack_pseudo_continuum.py -v`
Expected: PASS (1 test).

- [ ] **Step 6: Commit**

```bash
git add examples/stack_real_loa_dlas.py tests/test_stack_pseudo_continuum.py
git commit -m "stack: add pseudo-continuum constants + _continuum_mask + mock test"
```

---

## Task 3: `fit_pseudo_continuum` — spline fit (no rejection yet)

**Files:**
- Modify: `examples/stack_real_loa_dlas.py` (pseudo-continuum section; add `import` for scipy)
- Modify: `tests/test_stack_pseudo_continuum.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stack_pseudo_continuum.py`:

```python
from examples.stack_real_loa_dlas import fit_pseudo_continuum  # noqa: E402


def _offline_mask(rest_grid, lines, pad_sigma=6.0):
    """Pixels far from every injected line and away from the 945 Å edge."""
    ok = rest_grid >= 960.0
    for lam0, _depth, sig in lines:
        ok = ok & (np.abs(rest_grid - lam0) > pad_sigma * sig)
    return ok


def test_pcont_nan_below_945_finite_above():
    rg = _rest_grid()
    curve, counts, _, _ = make_mock_composite(rg)
    P = fit_pseudo_continuum(rg, curve, counts)
    assert np.all(np.isnan(P[rg < PCONT_LAMBDA_MIN]))
    mid = (rg > 1000.0) & (rg < 1550.0)
    assert np.isfinite(P[mid]).mean() > 0.99


def test_pcont_recovers_truth_off_lines():
    rg = _rest_grid()
    curve, counts, P_true, lines = make_mock_composite(rg)
    P = fit_pseudo_continuum(rg, curve, counts)
    off = _offline_mask(rg, lines) & np.isfinite(P) & (counts >= 50)
    rel = np.abs(P[off] / P_true[off] - 1.0)
    assert np.nanmedian(rel) < 0.02
    assert np.nanpercentile(rel, 95) < 0.04


def test_pcont_null_case_flat():
    rg = _rest_grid()
    curve, counts, _, _ = make_mock_composite(rg, inject=False)
    P = fit_pseudo_continuum(rg, curve, counts)
    norm = curve / P
    ok = np.isfinite(norm) & (rg > 960.0) & (rg < 1590.0)
    assert abs(np.nanmedian(norm[ok]) - 1.0) < 0.01
    assert np.nanstd(norm[ok]) < 0.05


def test_pcont_lines_survive_normalization():
    rg = _rest_grid()
    curve, counts, _, lines = make_mock_composite(rg)
    P = fit_pseudo_continuum(rg, curve, counts)
    norm = curve / P
    for lam0, depth, sig in lines:
        if lam0 < PCONT_LAMBDA_MIN:
            continue
        core = np.abs(rg - lam0) < 2.0 * sig
        measured = 1.0 - np.nanmin(norm[core])
        assert measured > 0.6 * depth, f"line {lam0} eaten: {measured} vs {depth}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_stack_pseudo_continuum.py -v`
Expected: the 4 new tests FAIL — `ImportError: cannot import name 'fit_pseudo_continuum'`.

- [ ] **Step 3: Add the scipy import**

In `examples/stack_real_loa_dlas.py`, in the import block at the top, after `import h5py`, add:

```python
from scipy.interpolate import LSQUnivariateSpline
```

- [ ] **Step 4: Add `_safe_lsq_spline` and `fit_pseudo_continuum`**

In the pseudo-continuum section (after `_continuum_mask`), add:

```python
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
    inside = (rest_grid >= max(PCONT_LAMBDA_MIN, x[0])) & (rest_grid <= x[-1])
    P[inside] = spl(rest_grid[inside])
    info["n_rejected"] = int(fit_ok0.sum() - ok.sum())
    clean = fit_ok0 & np.isfinite(P)
    if clean.any():
        info["rms"] = float(np.sqrt(np.nanmean((curve[clean] / P[clean] - 1.0) ** 2)))
    return (P, info) if return_info else P
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_stack_pseudo_continuum.py -v`
Expected: PASS (5 tests). If `test_pcont_recovers_truth_off_lines` fails marginally, the forest decrement may be underfit — note it for the QC step but the 2%/4% tolerances should hold with 15 Å knots.

- [ ] **Step 6: Commit**

```bash
git add examples/stack_real_loa_dlas.py tests/test_stack_pseudo_continuum.py
git commit -m "stack: add fit_pseudo_continuum (masked fixed-knot cubic spline)"
```

---

## Task 4: Iterative sigma-rejection — verify it catches unmasked lines

**Files:**
- Modify: `tests/test_stack_pseudo_continuum.py`

The rejection loop is already implemented in Task 3's `fit_pseudo_continuum`. This task adds the tests that *prove* it works — that the two injected lines absent from `METAL_LINES` (1117 Å, 1450 Å) do not pull the continuum down.

- [ ] **Step 1: Write the failing/verifying tests**

Append to `tests/test_stack_pseudo_continuum.py`:

```python
def test_rejection_catches_unmasked_lines():
    """1117 Å and 1450 Å are injected but NOT in METAL_LINES, so the
    static mask misses them — the rejection loop must keep the continuum
    from dipping into them."""
    rg = _rest_grid()
    curve, counts, P_true, _ = make_mock_composite(rg)
    P = fit_pseudo_continuum(rg, curve, counts)
    for lam0 in (1117.0, 1450.0):
        i = np.argmin(np.abs(rg - lam0))
        rel = abs(P[i] / P_true[i] - 1.0)
        assert rel < 0.03, f"continuum dipped into unmasked line {lam0}: {rel}"


def test_rejection_actually_rejects():
    rg = _rest_grid()
    curve, counts, _, _ = make_mock_composite(rg)
    _P, info = fit_pseudo_continuum(rg, curve, counts, return_info=True)
    assert info["n_rejected"] > 0
    assert info["n_iter"] >= 1


def test_spline_does_not_follow_masked_lines():
    """At each masked injected line centre the spline must stay on the
    truth continuum, not dip toward the absorption."""
    rg = _rest_grid()
    curve, counts, P_true, lines = make_mock_composite(rg)
    P = fit_pseudo_continuum(rg, curve, counts)
    for lam0, _depth, _sig in lines:
        if lam0 < 1000.0 or lam0 in (1117.0, 1450.0):
            continue
        i = np.argmin(np.abs(rg - lam0))
        if not np.isfinite(P[i]):
            continue
        assert abs(P[i] / P_true[i] - 1.0) < 0.04, f"spline dipped at {lam0}"
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_stack_pseudo_continuum.py -v`
Expected: PASS (8 tests total). `test_rejection_catches_unmasked_lines` proves the loop works; if it fails, `REJECT_SIGMA` is too loose or the loop is not removing enough — but with 5σ and depth-0.05–0.06 lines on ~0.02 noise the lines are >2σ deep over several pixels and get rejected.

- [ ] **Step 3: Commit**

```bash
git add tests/test_stack_pseudo_continuum.py
git commit -m "stack: tests for pseudo-continuum sigma-rejection"
```

---

## Task 5: Edge cases + determinism

**Files:**
- Modify: `tests/test_stack_pseudo_continuum.py`

- [ ] **Step 1: Write the tests**

Append to `tests/test_stack_pseudo_continuum.py`:

```python
def test_all_nan_input_returns_all_nan():
    rg = _rest_grid()
    curve = np.full(len(rg), np.nan)
    counts = np.zeros(len(rg))
    P = fit_pseudo_continuum(rg, curve, counts)
    assert np.all(np.isnan(P))


def test_low_coverage_pixels_excluded():
    """A pixel block with counts < 50 must not break the fit and must be
    excluded from fit_ok."""
    rg = _rest_grid()
    curve, counts, _, _ = make_mock_composite(rg)
    counts = counts.copy()
    counts[(rg > 1300.0) & (rg < 1320.0)] = 10.0  # below the 50 floor
    fit_ok = _continuum_mask(rg, curve, counts)
    assert not fit_ok[(rg > 1301.0) & (rg < 1319.0)].any()
    P = fit_pseudo_continuum(rg, curve, counts)
    assert np.isfinite(P[(rg > 1340.0) & (rg < 1360.0)]).all()


def test_determinism():
    rg = _rest_grid()
    curve, counts, _, _ = make_mock_composite(rg, seed=3)
    P1 = fit_pseudo_continuum(rg, curve, counts)
    P2 = fit_pseudo_continuum(rg, curve, counts)
    np.testing.assert_array_equal(np.nan_to_num(P1), np.nan_to_num(P2))
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_stack_pseudo_continuum.py -v`
Expected: PASS (11 tests total).

- [ ] **Step 3: Run the full pseudo-continuum test file**

Run: `python -m pytest tests/test_stack_pseudo_continuum.py -q`
Expected: 11 passed. (The change touches only the new pseudo-continuum
code path, so no other repo tests are affected.)

- [ ] **Step 4: Commit**

```bash
git add tests/test_stack_pseudo_continuum.py
git commit -m "stack: edge-case + determinism tests for pseudo-continuum"
```

---

## Task 6: Rewire `plot_metal_zoom` + `plot_control`, delete `_local_continuum_norm`

**Files:**
- Modify: `examples/stack_real_loa_dlas.py` (`plot_metal_zoom`, `plot_control`, delete `_local_continuum_norm`)

`plot_metal_zoom` and `plot_control` currently call `_local_continuum_norm(x, y, lines)` — a per-panel linear fit. Replace with a slice of the global pseudo-continuum-normalized composite.

- [ ] **Step 1: Rewire `plot_metal_zoom`**

In `plot_metal_zoom`, the per-bin loop currently reads:

```python
        for (lo, hi) in bins:
            if (lo, hi) not in per_bin:
                continue
            bs = per_bin[(lo, hi)]
            y = _local_continuum_norm(x, bs.curve[sel].astype(np.float64), lines)
            ax.plot(x, y, color=BIN_COLOR[(lo, hi)], lw=1.3, alpha=0.85,
                    label=f"NHI [{lo:.1f},{hi:.1f})")
            if np.isfinite(y).any():
                panel_min = min(panel_min, np.nanpercentile(y, 1))
```

Replace with:

```python
        for (lo, hi) in bins:
            if (lo, hi) not in per_bin:
                continue
            bs = per_bin[(lo, hi)]
            P = fit_pseudo_continuum(rest_grid, bs.curve, bs.counts)
            y = (bs.curve / P)[sel]
            ax.plot(x, y, color=BIN_COLOR[(lo, hi)], lw=1.3, alpha=0.85,
                    label=f"NHI [{lo:.1f},{hi:.1f})")
            if np.isfinite(y).any():
                panel_min = min(panel_min, np.nanpercentile(y, 1))
```

- [ ] **Step 2: Rewire `plot_control`**

In `plot_control`, the panel body currently reads:

```python
        y_real = _local_continuum_norm(x, rc[sel].astype(np.float64), lines)
        y_ctrl = _local_continuum_norm(x, cc[sel].astype(np.float64), lines)
```

Replace with (compute each pseudo-continuum once, before the panel loop). Find the line `rc, rn, cc, cn, n = combined[name]` near the top of `plot_control` and add right after it:

```python
    rc, rn, cc, cn, n = combined[name]
    P_real = fit_pseudo_continuum(rest_grid, rc, rn)
    P_ctrl = fit_pseudo_continuum(rest_grid, cc, cn)
    norm_real = rc / P_real
    norm_ctrl = cc / P_ctrl
```

Then replace the two `_local_continuum_norm` lines inside the panel loop with:

```python
        y_real = norm_real[sel]
        y_ctrl = norm_ctrl[sel]
```

- [ ] **Step 3: Delete `_local_continuum_norm`**

Remove the entire `_local_continuum_norm` function definition from `examples/stack_real_loa_dlas.py` (it now has no callers).

- [ ] **Step 4: Verify no remaining references**

Run:
```bash
cd /home/mfho/desi_gpy_dla_detection
grep -n "_local_continuum_norm" examples/stack_real_loa_dlas.py || echo "no references — OK"
python -c "import ast; ast.parse(open('examples/stack_real_loa_dlas.py').read()); print('syntax OK')"
```
Expected: "no references — OK" and "syntax OK".

- [ ] **Step 5: Smoke-test the rewired plotters**

Create a temporary check (do not commit this file) `/tmp/smoke_plot.py`:

```python
import sys; sys.path.insert(0, "/home/mfho/desi_gpy_dla_detection")
import numpy as np, tempfile
from pathlib import Path
import examples.stack_real_loa_dlas as stk

rg = 10 ** np.arange(np.log10(stk.REST_LAMBDA_MIN),
                     np.log10(stk.REST_LAMBDA_MAX), stk.DLOG_LAMBDA)
from tests.test_stack_pseudo_continuum import make_mock_composite
curve, counts, _, _ = make_mock_composite(rg)
bs = stk.BinStack(curve, counts, 500, curve, counts, 80)
per_bin = {b: bs for b in stk.NHI_BINS_PROD}
combined = {"lls": (curve, counts, curve, counts, 400),
            "subdla": (curve, counts, curve, counts, 400)}
with tempfile.TemporaryDirectory() as d:
    stk.OUT_DIR = Path(d)
    stk.plot_metal_zoom(rg, per_bin, stk.NHI_BINS_PROD, "z.png", "smoke")
    stk.plot_control(rg, combined, "lls", "c.png")
    assert (Path(d) / "z.png").stat().st_size > 0
    assert (Path(d) / "c.png").stat().st_size > 0
print("plot smoke OK")
```

Run: `python /tmp/smoke_plot.py`
Expected: "plot smoke OK".

- [ ] **Step 6: Commit**

```bash
git add examples/stack_real_loa_dlas.py
git commit -m "stack: normalize metal-zoom + control plots by the global pseudo-continuum"
```

---

## Task 7: QC figure `plot_pseudo_continuum_qc`

**Files:**
- Modify: `examples/stack_real_loa_dlas.py` (add `plot_pseudo_continuum_qc`, call it in `render_all`, update the module docstring outputs list)

- [ ] **Step 1: Add `plot_pseudo_continuum_qc`**

In `examples/stack_real_loa_dlas.py`, add this function just before `def render_all`:

```python
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
```

- [ ] **Step 2: Register it in `render_all`**

In `render_all`, find the line `plot_bal_compare(rest_grid, per_bin, "stack_bal_compare.png")` and add immediately after it:

```python
    plot_pseudo_continuum_qc(rest_grid, prod, "stack_pseudo_continuum_qc.png")
```

(`prod` is the production-bin view already defined at the top of `render_all`.)

- [ ] **Step 3: Update the module docstring outputs list**

In the module docstring (`Outputs (docs/notes/...)` block near the top), add this line under `stack_bal_compare.png`:

```
  stack_pseudo_continuum_qc.png                     — continuum-fit QC
```

- [ ] **Step 4: Smoke-test the QC plot**

Create `/tmp/smoke_qc.py` (do not commit):

```python
import sys; sys.path.insert(0, "/home/mfho/desi_gpy_dla_detection")
import numpy as np, tempfile
from pathlib import Path
import examples.stack_real_loa_dlas as stk
from tests.test_stack_pseudo_continuum import make_mock_composite

rg = 10 ** np.arange(np.log10(stk.REST_LAMBDA_MIN),
                     np.log10(stk.REST_LAMBDA_MAX), stk.DLOG_LAMBDA)
curve, counts, _, _ = make_mock_composite(rg)
bs = stk.BinStack(curve, counts, 500, curve, counts, 80)
per_bin = {b: bs for b in stk.NHI_BINS_PROD}
with tempfile.TemporaryDirectory() as d:
    stk.OUT_DIR = Path(d)
    stk.plot_pseudo_continuum_qc(rg, per_bin, "qc.png")
    assert (Path(d) / "qc.png").stat().st_size > 0
print("QC smoke OK")
```

Run: `python /tmp/smoke_qc.py`
Expected: "QC smoke OK".

- [ ] **Step 5: Run the full pseudo-continuum test file once more**

Run: `python -m pytest tests/test_stack_pseudo_continuum.py -q`
Expected: 11 passed.

- [ ] **Step 6: Commit**

```bash
git add examples/stack_real_loa_dlas.py
git commit -m "stack: add pseudo-continuum QC figure"
```

---

## Self-review notes

- **Spec coverage:** §3 algorithm → Tasks 2–4; §4 constants → Task 2; §5 integration → Tasks 6–7; §6 figures → Tasks 6–7; §8 mock test → Tasks 2–5; §9 cleanup → Task 1. All covered.
- **Type consistency:** `fit_pseudo_continuum(rest_grid, curve, counts, return_info=False)` is defined in Task 3 and used with that signature in Tasks 6 (2-arg form) and 7 (`return_info=True`). `_continuum_mask(rest_grid, curve, counts)` consistent in Tasks 2 and 7. `BinStack` fields (`curve`, `counts`, `n`) match the existing namedtuple.
- **No placeholders:** every code step has complete code.
- **Note for executor:** the SLURM stack run (job 50407357) produces the real `stack_curves.npz`. Once it lands, run `python examples/stack_real_loa_dlas.py --plot-only` to regenerate the figures with the pseudo-continuum and eyeball `stack_pseudo_continuum_qc.png` — this is the §8.3 visual check, outside the automated tests.
