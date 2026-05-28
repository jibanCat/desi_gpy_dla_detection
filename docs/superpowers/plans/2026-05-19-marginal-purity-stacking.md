# Marginal-Purity Stacking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--purity` preset selector to `examples/stack_real_loa_dlas.py` so the marginal-purity (P_DLA ∈ [0.5,0.7]) absorber detections can be stacked and compared to the high-purity stack — the operating-point false-positive test.

**Architecture:** A `PURITY` preset (`high` / `marginal`) drives the P_DLA selection; each preset writes a preset-tagged npz + figures. A pooled low-N_HI `combined` category and a persisted pseudo-continuum become first-class npz products. A `--compare-purity` mode loads both npz and renders one 3-curve comparison figure.

**Tech Stack:** Python, numpy, scipy, matplotlib, h5py, astropy, pytest.

**Spec:** `docs/superpowers/specs/2026-05-19-marginal-purity-stacking-design.md`

---

## File Structure

- **Modify** `examples/stack_real_loa_dlas.py` — all production logic (the repo keeps the stacking pipeline in this one script).
- **Modify** `slurm/greatlakes/stack_real_loa.sh` — `PURITY` env pass-through.
- **Create** `tests/test_stack_purity.py` — preset / path / persistence / comparison tests (imports the script like `tests/test_stack_pseudo_continuum.py`).

Tasks are ordered by dependency: presets → tagged paths → pooled category → persisted pseudo-continuum → comparison figure → SLURM.

---

## Task 1: Purity presets + `--purity` parsing

**Files:**
- Modify: `examples/stack_real_loa_dlas.py`
- Create: `tests/test_stack_purity.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_stack_purity.py`:

```python
"""Tests for the marginal-purity stacking additions to
examples/stack_real_loa_dlas.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import examples.stack_real_loa_dlas as stk  # noqa: E402


def _synthetic_catalog(p_dla_values):
    """A DLACAT-shaped structured array; every row passes every cut
    except P_DLA, which is set per-row from `p_dla_values`."""
    n = len(p_dla_values)
    dt = np.dtype([("TARGETID", "<i8"), ("Z_QSO", "<f4"), ("Z_DLA", "<f4"),
                   ("P_DLA", "<f4"), ("SNR_FOREST", "<f4"),
                   ("DLAFLAG", "<i4"), ("NHI", "<f4")])
    cat = np.zeros(n, dtype=dt)
    cat["TARGETID"] = np.arange(1, n + 1)
    cat["Z_QSO"] = 4.0
    cat["Z_DLA"] = 3.5          # in-forest + not-proximate for z_qso=4.0
    cat["SNR_FOREST"] = 5.0
    cat["DLAFLAG"] = 0
    cat["NHI"] = 20.0
    cat["P_DLA"] = np.asarray(p_dla_values, dtype=np.float32)
    return cat


def test_purity_preset_selection(monkeypatch):
    cat = _synthetic_catalog([0.30, 0.55, 0.65, 0.75, 0.98, 0.995])

    monkeypatch.setattr(stk, "PURITY", "marginal")
    kept = stk.select(cat, set())
    assert sorted(kept["P_DLA"].round(3)) == [0.55, 0.65]

    monkeypatch.setattr(stk, "PURITY", "high")
    kept = stk.select(cat, set())
    assert sorted(kept["P_DLA"].round(3)) == [0.98, 0.995]


def test_provenance_carries_purity(monkeypatch):
    monkeypatch.setattr(stk, "PURITY", "marginal")
    prov = stk.provenance_dict()
    assert prov["purity"] == "marginal"
    assert prov["p_dla_range"] == [0.50, 0.70]
    assert "p_dla_min" not in prov
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/mfho/desi_gpy_dla_detection && python -m pytest tests/test_stack_purity.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'PURITY'`.

- [ ] **Step 3: Replace the `P_DLA_MIN` constant with purity presets**

In `examples/stack_real_loa_dlas.py`, find:

```python
# Selection
P_DLA_MIN = 0.97
SNR_FOREST_MIN = 2.0
```

Replace with:

```python
# Selection — purity is a preset chosen by --purity (default "high",
# which reproduces the original P_DLA > 0.97 behaviour).
PURITY_PRESETS = {           # preset -> (p_lo, p_hi); keep p_lo < P_DLA <= p_hi
    "high":     (0.97, 1.01),
    "marginal": (0.50, 0.70),
}
PURITY = "high"              # module global; set from --purity in main()
SNR_FOREST_MIN = 2.0
```

- [ ] **Step 4: Apply the preset range in `select()`**

In `select()`, find:

```python
    keep = (
        (cat["P_DLA"] > P_DLA_MIN)
        & (cat["SNR_FOREST"] > SNR_FOREST_MIN)
```

Replace with:

```python
    p_lo, p_hi = PURITY_PRESETS[PURITY]
    in_purity = (cat["P_DLA"] > p_lo) & (cat["P_DLA"] <= p_hi)
    keep = (
        in_purity
        & (cat["SNR_FOREST"] > SNR_FOREST_MIN)
```

Then find:

```python
    print(f"  P_DLA > {P_DLA_MIN}:        {(cat['P_DLA'] > P_DLA_MIN).sum()}", flush=True)
```

Replace with:

```python
    print(f"  purity={PURITY} P_DLA∈({p_lo},{p_hi}]:  {in_purity.sum()}", flush=True)
```

- [ ] **Step 5: Update `provenance_dict()`**

In `provenance_dict()`, find the line `"p_dla_min": P_DLA_MIN,` and replace it with:

```python
        "purity": PURITY,
        "p_dla_range": list(PURITY_PRESETS[PURITY]),
```

- [ ] **Step 6: Update the other `P_DLA_MIN` references**

In `compute_stacks`, find:

```python
        fh.write(f"# P_DLA > {P_DLA_MIN}, SNR_FOREST > {SNR_FOREST_MIN}, "
                 f"DLAFLAG=0, Z_QSO > {Z_QSO_MIN}, in-forest, not-proximate\n")
```

Replace with:

```python
        fh.write(f"# purity={PURITY}, SNR_FOREST > {SNR_FOREST_MIN}, "
                 f"DLAFLAG=0, Z_QSO > {Z_QSO_MIN}, in-forest, not-proximate\n")
```

In `plot_overview`, find:

```python
    ax.set_title(
        f"{subtitle} — high-purity (P_DLA > {P_DLA_MIN}), "
        f"SNR_forest > {SNR_FOREST_MIN}, z_QSO > {Z_QSO_MIN}, "
        "Lyα-forest detection only, non-BAL")
```

Replace with:

```python
    ax.set_title(
        f"{subtitle} — purity={PURITY}, "
        f"SNR_forest > {SNR_FOREST_MIN}, z_QSO > {Z_QSO_MIN}, "
        "Lyα-forest detection only, non-BAL")
```

Then confirm no other references remain:
```bash
grep -n "P_DLA_MIN" examples/stack_real_loa_dlas.py || echo "no P_DLA_MIN refs — OK"
```
Expected: "no P_DLA_MIN refs — OK".

- [ ] **Step 7: Parse `--purity` in `main()`**

In `examples/stack_real_loa_dlas.py`, add this helper just before `def main():`:

```python
def _arg_value(flag, default):
    """Value of `--flag VALUE` in sys.argv, else `default`."""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default
```

Then in `main()`, find:

```python
def main():
    _ensure_outdir()
```

Replace with:

```python
def main():
    global PURITY
    PURITY = _arg_value("--purity", "high")
    if PURITY not in PURITY_PRESETS:
        raise SystemExit(f"--purity must be one of {list(PURITY_PRESETS)}; "
                         f"got {PURITY!r}")
    _ensure_outdir()
```

- [ ] **Step 8: Run tests + syntax check**

Run:
```bash
python -c "import ast; ast.parse(open('examples/stack_real_loa_dlas.py').read()); print('syntax OK')"
python -m pytest tests/test_stack_purity.py tests/test_stack_pseudo_continuum.py -q
```
Expected: "syntax OK"; `tests/test_stack_purity.py` 2 passed, `tests/test_stack_pseudo_continuum.py` 11 passed.

- [ ] **Step 9: Commit**

```bash
git add examples/stack_real_loa_dlas.py tests/test_stack_purity.py
git commit -m "stack: --purity preset selector (high / marginal P_DLA range)"
```

---

## Task 2: Preset-tagged output paths

**Files:**
- Modify: `examples/stack_real_loa_dlas.py`
- Modify: `tests/test_stack_purity.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stack_purity.py`:

```python
def test_tagged_and_npz_path(monkeypatch):
    monkeypatch.setattr(stk, "PURITY", "marginal")
    assert stk.tagged("stack_prod") == "stack_prod_marginal.png"
    assert stk.tagged("counts", "txt") == "counts_marginal.txt"
    assert stk.npz_path().name == "stack_curves_marginal.npz"
    monkeypatch.setattr(stk, "PURITY", "high")
    assert stk.npz_path().name == "stack_curves_high.npz"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stack_purity.py::test_tagged_and_npz_path -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'tagged'`.

- [ ] **Step 3: Add `tagged()` and `npz_path()`; drop the `NPZ_PATH` constant**

In `examples/stack_real_loa_dlas.py`, find:

```python
OUT_DIR = Path(__file__).resolve().parent.parent / "docs/notes/2026-05-15_stack_real_loa_dlas"
NPZ_PATH = OUT_DIR / "stack_curves.npz"


def _ensure_outdir():
```

Replace with:

```python
OUT_DIR = Path(__file__).resolve().parent.parent / "docs/notes/2026-05-15_stack_real_loa_dlas"


def tagged(basename, ext="png"):
    """Output filename tagged with the active purity preset, so the
    `high` and `marginal` runs do not clobber each other's outputs."""
    return f"{basename}_{PURITY}.{ext}"


def npz_path():
    """Path to the cached-curves npz for the active purity preset."""
    return OUT_DIR / tagged("stack_curves", "npz")


def _ensure_outdir():
```

- [ ] **Step 4: Use `npz_path()` in `save_curves`**

In `save_curves`, find:

```python
    np.savez(NPZ_PATH, **payload)
    print(f"[saved] {NPZ_PATH}", flush=True)
```

Replace with:

```python
    out = npz_path()
    np.savez(out, **payload)
    print(f"[saved] {out}", flush=True)
```

- [ ] **Step 5: Make `load_curves` take an explicit path**

In `load_curves`, find:

```python
def load_curves():
    d = np.load(NPZ_PATH, allow_pickle=False)
    if "provenance" not in d:
        raise SystemExit(
            f"[ERROR] cached {NPZ_PATH.name} predates the provenance/BAL "
            "format — re-run without --plot-only to regenerate.")
    check_provenance(json.loads(str(d["provenance"])))
```

Replace with:

```python
def load_curves(path):
    path = Path(path)
    d = np.load(path, allow_pickle=False)
    if "provenance" not in d:
        raise SystemExit(
            f"[ERROR] cached {path.name} predates the provenance/BAL "
            "format — re-run to regenerate.")
    check_provenance(json.loads(str(d["provenance"])))
```

- [ ] **Step 6: Preset-tag the zhist / counts outputs**

In `dump_zhist`, find:

```python
    fig.savefig(OUT_DIR / "zhist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR / 'zhist.png'}", flush=True)
```

Replace with:

```python
    fig.savefig(OUT_DIR / tagged("zhist"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR / tagged('zhist')}", flush=True)
```

In `dump_zhist`, find:

```python
    (OUT_DIR / "zhist_summary.txt").write_text("\n".join(summary) + "\n")
    print(f"[saved] {OUT_DIR / 'zhist_summary.txt'}", flush=True)
```

Replace with:

```python
    (OUT_DIR / tagged("zhist_summary", "txt")).write_text(
        "\n".join(summary) + "\n")
    print(f"[saved] {OUT_DIR / tagged('zhist_summary', 'txt')}", flush=True)
```

In `compute_stacks`, find:

```python
    with (OUT_DIR / "counts.txt").open("w") as fh:
```

Replace with:

```python
    with (OUT_DIR / tagged("counts", "txt")).open("w") as fh:
```

- [ ] **Step 7: Preset-tag the figure names in `render_all`**

Replace the entire body of `render_all` (everything after the `def render_all(rest_grid, per_bin, combined):` line) with:

```python
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
```

- [ ] **Step 8: Update `main()` to use `npz_path()`**

In `main()`, find:

```python
    if "--plot-only" in sys.argv:
        if not NPZ_PATH.exists():
            raise SystemExit(f"no cached curves at {NPZ_PATH}; "
                             "run without --plot-only first")
        print(f"loading cached curves from {NPZ_PATH}", flush=True)
        rest_grid, per_bin, combined = load_curves()
```

Replace with:

```python
    if "--plot-only" in sys.argv:
        if not npz_path().exists():
            raise SystemExit(f"no cached curves at {npz_path()}; "
                             "run without --plot-only first")
        print(f"loading cached curves from {npz_path()}", flush=True)
        rest_grid, per_bin, combined = load_curves(npz_path())
```

- [ ] **Step 9: Run tests + syntax check**

Run:
```bash
python -c "import ast; ast.parse(open('examples/stack_real_loa_dlas.py').read()); print('syntax OK')"
grep -n "NPZ_PATH" examples/stack_real_loa_dlas.py || echo "no NPZ_PATH refs — OK"
python -m pytest tests/test_stack_purity.py tests/test_stack_pseudo_continuum.py -q
```
Expected: "syntax OK"; "no NPZ_PATH refs — OK"; test_stack_purity 3 passed, test_stack_pseudo_continuum 11 passed.

- [ ] **Step 10: Commit**

```bash
git add examples/stack_real_loa_dlas.py tests/test_stack_purity.py
git commit -m "stack: preset-tag the npz + figure outputs"
```

---

## Task 3: Pooled low-N_HI combined category + provenance preset check

**Files:**
- Modify: `examples/stack_real_loa_dlas.py`
- Modify: `tests/test_stack_purity.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stack_purity.py`:

```python
def test_control_categories_has_lownhi():
    assert "lownhi" in stk.CONTROL_CATEGORIES
    assert (set(stk.CONTROL_CATEGORIES["lownhi"])
            == set(stk.LLS_BINS_FINE) | set(stk.SUBDLA_BINS))
    assert "lownhi" in {"lls", "subdla", "lownhi"}  # named control category


def test_check_provenance_preset_mismatch(monkeypatch):
    monkeypatch.setattr(stk, "PURITY", "high")
    stored = stk.provenance_dict()          # a 'high' provenance dict
    # expecting 'marginal' must raise
    monkeypatch.setattr(sys, "argv", ["x"])  # no --force-plot
    with pytest.raises(SystemExit):
        stk.check_provenance(stored, expect_preset="marginal")
    # expecting 'high' must pass (no raise)
    stk.check_provenance(stored, expect_preset="high")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stack_purity.py::test_control_categories_has_lownhi tests/test_stack_purity.py::test_check_provenance_preset_mismatch -v`
Expected: FAIL — `lownhi` not in `CONTROL_CATEGORIES`; `check_provenance` takes no `expect_preset`.

- [ ] **Step 3: Add the `lownhi` control category**

In `examples/stack_real_loa_dlas.py`, find:

```python
# Categories needing a redshift-scrambled control (low-NHI = contamination-prone)
CONTROL_CATEGORIES = {"lls": LLS_BINS_FINE, "subdla": SUBDLA_BINS}
```

Replace with:

```python
# Categories needing a redshift-scrambled control (low-NHI = contamination-prone).
# "lownhi" pools LLS + sub-DLA — the pooled stack used by --compare-purity.
CONTROL_CATEGORIES = {
    "lls":    LLS_BINS_FINE,
    "subdla": SUBDLA_BINS,
    "lownhi": LLS_BINS_FINE + SUBDLA_BINS,
}
```

- [ ] **Step 4: Add the `lownhi` label in `plot_control`**

In `plot_control`, find:

```python
    label = {"lls": "LLS [17.2, 19)", "subdla": "sub-DLA [19, 20.3)"}[name]
```

Replace with:

```python
    label = {"lls": "LLS [17.2, 19)", "subdla": "sub-DLA [19, 20.3)",
             "lownhi": "low-NHI (LLS + sub-DLA)"}[name]
```

- [ ] **Step 5: Add `expect_preset` to `check_provenance`**

Replace the whole `check_provenance` function with:

```python
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
```

- [ ] **Step 6: Thread `expect_preset` through `load_curves`**

In `load_curves`, find:

```python
def load_curves(path):
    path = Path(path)
    d = np.load(path, allow_pickle=False)
    if "provenance" not in d:
        raise SystemExit(
            f"[ERROR] cached {path.name} predates the provenance/BAL "
            "format — re-run to regenerate.")
    check_provenance(json.loads(str(d["provenance"])))
```

Replace with:

```python
def load_curves(path, expect_preset=None):
    path = Path(path)
    d = np.load(path, allow_pickle=False)
    if "provenance" not in d:
        raise SystemExit(
            f"[ERROR] cached {path.name} predates the provenance/BAL "
            "format — re-run to regenerate.")
    check_provenance(json.loads(str(d["provenance"])), expect_preset)
```

In `main()`, find:

```python
        rest_grid, per_bin, combined = load_curves(npz_path())
```

Replace with:

```python
        rest_grid, per_bin, combined = load_curves(npz_path(),
                                                   expect_preset=PURITY)
```

- [ ] **Step 7: Run tests + syntax check**

Run:
```bash
python -c "import ast; ast.parse(open('examples/stack_real_loa_dlas.py').read()); print('syntax OK')"
python -m pytest tests/test_stack_purity.py tests/test_stack_pseudo_continuum.py -q
```
Expected: "syntax OK"; test_stack_purity 5 passed, test_stack_pseudo_continuum 11 passed.

- [ ] **Step 8: Commit**

```bash
git add examples/stack_real_loa_dlas.py tests/test_stack_purity.py
git commit -m "stack: pooled low-NHI control category + preset-aware provenance check"
```

---

## Task 4: Persist the pseudo-continuum in the npz

**Files:**
- Modify: `examples/stack_real_loa_dlas.py`
- Modify: `tests/test_stack_purity.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stack_purity.py`:

```python
def _mock_curve(rg, seed=0):
    rng = np.random.default_rng(seed)
    slope = 1.0 + 0.05 * (rg - 1200.0) / 900.0
    curve = slope + rng.normal(0.0, 0.01, len(rg))
    curve[rg < 760.0] = np.nan
    counts = np.clip(60.0 + 0.9 * (rg - 700.0), 0.0, 800.0)
    return curve.astype(float), counts.astype(float)


def test_pcont_persists_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(stk, "OUT_DIR", tmp_path)
    monkeypatch.setattr(stk, "PURITY", "high")
    rg = 10 ** np.arange(np.log10(stk.REST_LAMBDA_MIN),
                         np.log10(stk.REST_LAMBDA_MAX), stk.DLOG_LAMBDA)
    curve, counts = _mock_curve(rg)
    # 80 noisy copies (60 non-BAL + 20 BAL) so the non-BAL group clears
    # the 50-spectrum coverage floor and `bs.curve` is a real stack.
    raw = (np.tile(curve, (80, 1))
           + np.random.default_rng(1).normal(0.0, 0.01, (80, len(rg))))
    is_bal = np.array([False] * 60 + [True] * 20)
    bs = stk._stack_pair(rg, raw, is_bal)
    assert np.isfinite(bs.curve[(rg > 1000) & (rg < 1500)]).any()
    # pcont must be the deterministic fit of the non-BAL curve
    assert np.allclose(np.nan_to_num(bs.pcont),
                       np.nan_to_num(stk.fit_pseudo_continuum(rg, bs.curve,
                                                              bs.counts)))
    per_bin = {stk.NHI_BINS[0]: bs}
    P = stk.fit_pseudo_continuum(rg, curve, counts)
    combined = {"lownhi": (curve, counts, P, curve, counts, P, 2)}
    stk.save_curves(rg, per_bin, combined)
    _rg, per_bin2, comb2 = stk.load_curves(stk.npz_path())
    bs2 = per_bin2[stk.NHI_BINS[0]]
    assert np.array_equal(np.nan_to_num(bs2.pcont), np.nan_to_num(bs.pcont))
    assert np.array_equal(np.nan_to_num(comb2["lownhi"][2]),
                          np.nan_to_num(P))
    assert len(comb2["lownhi"]) == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stack_purity.py::test_pcont_persists_round_trip -v`
Expected: FAIL — `_stack_pair` takes 2 args / `BinStack` has no `pcont`.

- [ ] **Step 3: Extend the `BinStack` namedtuple**

In `examples/stack_real_loa_dlas.py`, find:

```python
# Per-bin stack: non-BAL curve/counts/n + BAL curve/counts/n.
BinStack = namedtuple(
    "BinStack", ["curve", "counts", "n", "curve_bal", "counts_bal", "n_bal"])
```

Replace with:

```python
# Per-bin stack: non-BAL (curve/counts/n/pcont) + BAL (..._bal). `pcont`
# is the fitted pseudo-continuum; the continuum-normalized stack is
# `curve / pcont`.
BinStack = namedtuple(
    "BinStack", ["curve", "counts", "n", "pcont",
                 "curve_bal", "counts_bal", "n_bal", "pcont_bal"])
```

- [ ] **Step 4: Compute `pcont` in `_stack_pair`**

Replace the whole `_stack_pair` function with:

```python
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
```

- [ ] **Step 5: Update the `_stack_pair` call sites + `combined` in `compute_stacks`**

In `compute_stacks`, find:

```python
    # Per-bin stacks (non-BAL + BAL) for the 8 fine bins.
    per_bin = {}
    for b in NHI_BINS:
        if b not in raw_real:
            continue
        per_bin[b] = _stack_pair(raw_real[b], raw_isbal[b])

    # Production LLS bin = pooled union of the 3 fine LLS bins.
    lls_raw = [raw_real[b] for b in LLS_BINS_FINE if b in raw_real]
    lls_bal = [raw_isbal[b] for b in LLS_BINS_FINE if b in raw_isbal]
    if lls_raw:
        per_bin[LLS_MERGED] = _stack_pair(np.concatenate(lls_raw, axis=0),
                                          np.concatenate(lls_bal, axis=0))
```

Replace with:

```python
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
```

In `compute_stacks`, find:

```python
        rc, rn = _sigma_clip_median(pooled_real)
        cc, cn = _sigma_clip_median(pooled_ctrl)
        combined[name] = (rc, rn, cc, cn, len(pooled_real))
```

Replace with:

```python
        rc, rn = _sigma_clip_median(pooled_real)
        cc, cn = _sigma_clip_median(pooled_ctrl)
        pcont_r = fit_pseudo_continuum(rest_grid, rc, rn)
        pcont_c = fit_pseudo_continuum(rest_grid, cc, cn)
        combined[name] = (rc, rn, pcont_r, cc, cn, pcont_c, len(pooled_real))
```

- [ ] **Step 6: Serialize `pcont` in `save_curves`**

In `save_curves`, find:

```python
    for (lo, hi), bs in per_bin.items():
        key = f"{lo:.2f}_{hi:.2f}"
        payload[f"curve_{key}"] = bs.curve
        payload[f"counts_{key}"] = bs.counts
        payload[f"ntot_{key}"] = np.int64(bs.n)
        payload[f"curvebal_{key}"] = bs.curve_bal
        payload[f"countsbal_{key}"] = bs.counts_bal
        payload[f"ntotbal_{key}"] = np.int64(bs.n_bal)
    for name, (rc, rn, cc, cn, n) in combined.items():
        payload[f"comb_real_{name}"] = rc
        payload[f"comb_realcnt_{name}"] = rn
        payload[f"comb_ctrl_{name}"] = cc
        payload[f"comb_ctrlcnt_{name}"] = cn
        payload[f"comb_n_{name}"] = np.int64(n)
```

Replace with:

```python
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
```

- [ ] **Step 7: Deserialize `pcont` in `load_curves`**

In `load_curves`, find:

```python
        per_bin[(lo, hi)] = BinStack(
            d[f"curve_{key}"], d[f"counts_{key}"], int(d[f"ntot_{key}"]),
            d[f"curvebal_{key}"], d[f"countsbal_{key}"],
            int(d[f"ntotbal_{key}"]))
    combined = {}
    for name in CONTROL_CATEGORIES:
        if f"comb_real_{name}" not in d:
            continue
        combined[name] = (d[f"comb_real_{name}"], d[f"comb_realcnt_{name}"],
                           d[f"comb_ctrl_{name}"], d[f"comb_ctrlcnt_{name}"],
                           int(d[f"comb_n_{name}"]))
```

Replace with:

```python
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
```

- [ ] **Step 8: Use the stored `pcont` in `plot_metal_zoom`**

In `plot_metal_zoom`, find:

```python
            bs = per_bin[(lo, hi)]
            P = fit_pseudo_continuum(rest_grid, bs.curve, bs.counts)
            y = (bs.curve / P)[sel]
```

Replace with:

```python
            bs = per_bin[(lo, hi)]
            y = (bs.curve / bs.pcont)[sel]
```

- [ ] **Step 9: Use the stored `pcont` in `plot_control`**

In `plot_control`, find:

```python
    rc, rn, cc, cn, n = combined[name]
    P_real = fit_pseudo_continuum(rest_grid, rc, rn)
    P_ctrl = fit_pseudo_continuum(rest_grid, cc, cn)
    norm_real = rc / P_real
    norm_ctrl = cc / P_ctrl
```

Replace with:

```python
    rc, rn, pcont_r, cc, cn, pcont_c, n = combined[name]
    norm_real = rc / pcont_r
    norm_ctrl = cc / pcont_c
```

- [ ] **Step 10: Document the normalized stack in the module docstring**

In the module docstring, find the line:

```
  stack_curves.npz                                  — cached curves + provenance
```

Replace with:

```
  stack_curves_<purity>.npz   — cached curves + pseudo-continuum (`pcont`)
                                + provenance. Continuum-normalized stack
                                = curve / pcont.
```

- [ ] **Step 11: Run tests + syntax check**

Run:
```bash
python -c "import ast; ast.parse(open('examples/stack_real_loa_dlas.py').read()); print('syntax OK')"
python -m pytest tests/test_stack_purity.py tests/test_stack_pseudo_continuum.py -q
```
Expected: "syntax OK"; test_stack_purity 6 passed, test_stack_pseudo_continuum 11 passed.

- [ ] **Step 12: Commit**

```bash
git add examples/stack_real_loa_dlas.py tests/test_stack_purity.py
git commit -m "stack: persist the pseudo-continuum in the npz (curve/pcont = normalized stack)"
```

---

## Task 5: Comparison figure + `--compare-purity` mode

**Files:**
- Modify: `examples/stack_real_loa_dlas.py`
- Modify: `tests/test_stack_purity.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stack_purity.py`:

```python
def test_purity_comparison_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(stk, "OUT_DIR", tmp_path)
    rg = 10 ** np.arange(np.log10(stk.REST_LAMBDA_MIN),
                         np.log10(stk.REST_LAMBDA_MAX), stk.DLOG_LAMBDA)
    curve, counts = _mock_curve(rg)
    P = stk.fit_pseudo_continuum(rg, curve, counts)
    comb = (curve, counts, P, curve, counts, P, 300)
    stk.plot_purity_comparison(rg, comb, comb, "cmp.png")
    assert (tmp_path / "cmp.png").stat().st_size > 0


def test_purity_compare_panels_nonempty():
    assert len(stk.PURITY_COMPARE_PANELS) >= 4
    titles = {p[0] for p in stk.PURITY_COMPARE_PANELS}
    assert "CIV 1548/1551" in titles
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stack_purity.py::test_purity_comparison_smoke tests/test_stack_purity.py::test_purity_compare_panels_nonempty -v`
Expected: FAIL — `PURITY_COMPARE_PANELS` / `plot_purity_comparison` undefined.

- [ ] **Step 3: Add the `PURITY_COMPARE_PANELS` constant**

In `examples/stack_real_loa_dlas.py`, immediately after the `ZOOM_PANELS = [ ... ]` list (after its closing `]`), add:

```python
# The most diagnostic metal-line panels for the marginal-vs-high purity
# comparison (a curated subset of ZOOM_PANELS).
_PURITY_COMPARE_TITLES = {
    "SiIV 1394/1403", "OI 1302 / SiII 1304", "CII 1335 / CII* 1336",
    "SII 1251/1254 / SiII 1260", "CIV 1548/1551",
}
PURITY_COMPARE_PANELS = [p for p in ZOOM_PANELS
                         if p[0] in _PURITY_COMPARE_TITLES]
```

- [ ] **Step 4: Add `plot_purity_comparison`**

In `examples/stack_real_loa_dlas.py`, add this function just before `def render_all`:

```python
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
```

- [ ] **Step 5: Add the `--compare-purity` mode to `main()`**

In `main()`, find:

```python
    _ensure_outdir()
    # `--zhist-only`: just the per-bin redshift diagnostics (catalog-only,
```

Replace with:

```python
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
```

- [ ] **Step 6: Run tests + syntax check**

Run:
```bash
python -c "import ast; ast.parse(open('examples/stack_real_loa_dlas.py').read()); print('syntax OK')"
python -m pytest tests/test_stack_purity.py tests/test_stack_pseudo_continuum.py -q
```
Expected: "syntax OK"; test_stack_purity 8 passed, test_stack_pseudo_continuum 11 passed.

- [ ] **Step 7: Commit**

```bash
git add examples/stack_real_loa_dlas.py tests/test_stack_purity.py
git commit -m "stack: --compare-purity mode + marginal-vs-high comparison figure"
```

---

## Task 6: SLURM `PURITY` pass-through

**Files:**
- Modify: `slurm/greatlakes/stack_real_loa.sh`

- [ ] **Step 1: Pass `PURITY` through to the script**

In `slurm/greatlakes/stack_real_loa.sh`, find:

```bash
echo
echo "=== running full stack ==="
python -u examples/stack_real_loa_dlas.py
```

Replace with:

```bash
PURITY="${PURITY:-high}"
echo
echo "=== running full stack (purity=$PURITY) ==="
python -u examples/stack_real_loa_dlas.py --purity "$PURITY"
```

- [ ] **Step 2: Document the marginal submission in the header comment**

In `slurm/greatlakes/stack_real_loa.sh`, find:

```bash
# Single-threaded Python; bottleneck is random HDF5 reads off /scratch.
# Expect ~45-90 min wall when /scratch is responsive; walltime is set to
# 10h to absorb the ~10x slowdowns seen under /scratch I/O contention.
```

Replace with:

```bash
# Single-threaded Python; bottleneck is random HDF5 reads off /scratch.
# Expect ~45-90 min wall when /scratch is responsive; walltime is set to
# 10h to absorb the ~10x slowdowns seen under /scratch I/O contention.
#
# Purity preset via the PURITY env var (default "high"). Run both:
#   sbatch slurm/greatlakes/stack_real_loa.sh                          # high
#   sbatch --export=ALL,PURITY=marginal slurm/greatlakes/stack_real_loa.sh
# then, once both npz exist (fast, no archive reads):
#   python examples/stack_real_loa_dlas.py --compare-purity
```

- [ ] **Step 3: Verify the script syntax**

Run: `bash -n slurm/greatlakes/stack_real_loa.sh && echo "sbatch script OK"`
Expected: "sbatch script OK".

- [ ] **Step 4: Commit**

```bash
git add slurm/greatlakes/stack_real_loa.sh
git commit -m "slurm: PURITY env pass-through for the stacking job"
```

---

## Self-review notes

- **Spec coverage:** §2 presets → Task 1; §3 tagged outputs → Task 2; §4a pooled `lownhi` → Task 3; §4b persisted pseudo-continuum → Task 4; §5 comparison + `--compare-purity` → Task 5; §6 CLI flow → Tasks 1+5; §7 SLURM → Task 6; §8 tests → Tasks 1-5 (`test_stack_purity.py`). All covered.
- **Type consistency:** `BinStack` is the 8-field form (`curve, counts, n, pcont, curve_bal, counts_bal, n_bal, pcont_bal`) from Task 4 onward; the `combined` tuple is the 7-element form (`rc, rn, pcont_r, cc, cn, pcont_c, n`) from Task 4 onward — Task 5's `plot_purity_comparison` unpacks exactly that. `_stack_pair(rest_grid, raw, is_bal)`, `load_curves(path, expect_preset=None)`, `check_provenance(stored, expect_preset=None)`, `tagged(basename, ext="png")`, `npz_path()` signatures are consistent across tasks.
- **No placeholders:** every code step shows complete code.
- **Note for the executor:** after Task 6, both presets must be (re-)run on GreatLakes — the existing `stack_curves.npz` predates the preset tag and the `pcont` / `lownhi` schema. Run `high` and `marginal`, then `--compare-purity`, then eyeball `stack_purity_comparison.png`. Generated npz/figures stay gitignored (real-LOA privacy).
