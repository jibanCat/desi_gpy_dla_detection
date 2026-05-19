# Design — marginal-purity stacking

**Date:** 2026-05-19
**Topic:** stack the marginal-purity (P_DLA ∈ [0.5,0.7]) absorber detections in `examples/stack_real_loa_dlas.py` (PR #8 follow-up)
**Status:** draft, awaiting review

---

## 1. Goal & scope

The existing stacking analysis selects `P_DLA > 0.97` — a high-purity
cut, "always likely real" by construction. It validates the cut; it
does **not** characterize the marginal operating point. This feature
stacks the **marginal-purity detections (P_DLA ∈ [0.5,0.7])** and
compares them to the high-purity stack and the z-scrambled control: if
the marginal stack shows coherent CIV like the high-purity stack →
the marginal detections are real; if flat like its own control →
contaminated.

**In scope:** a `--purity` preset selector, preset-tagged outputs, a
pooled low-N_HI combined stack, a `--compare-purity` figure, and
persisting the pseudo-continuum / continuum-normalized stack in the npz.

**Out of scope:** bootstrap-over-sightlines error bars; per-N_HI-bin
purity breakdown (the comparison is pooled low-N_HI); EW tables.

**Decided in brainstorming:** purity presets with two runs + one
comparison figure; the comparison figure is a metal-zoom, 3 curves
(marginal-real / marginal-control / high-purity-real), pooled over the
LLS + sub-DLA range.

---

## 2. Purity presets

Replace the hardcoded `P_DLA_MIN = 0.97` with a named preset:

```python
PURITY_PRESETS = {           # preset -> (p_lo, p_hi); select keeps p_lo < P_DLA <= p_hi
    "high":     (0.97, 1.01),  # P_DLA > 0.97 — current default behaviour
    "marginal": (0.50, 0.70),  # marginal operating point
}
PURITY = "high"              # module global; set from --purity in main()
```

`select()` uses `p_lo, p_hi = PURITY_PRESETS[PURITY]` and the keep
condition becomes `(cat["P_DLA"] > p_lo) & (cat["P_DLA"] <= p_hi)`
(replacing `cat["P_DLA"] > P_DLA_MIN`). For `high` this is identical to
the current `> 0.97` (P_DLA ≤ 1, so the 1.01 upper bound is inert) —
**fully backward-compatible**.

`PURITY` is set once at the top of `main()` by parsing `--purity
<preset>` from `sys.argv` (default `"high"`); an invalid value is a
`SystemExit` with the valid presets listed. All preset-dependent
quantities (npz path, figure names, provenance, plot-title text) read
the `PURITY` global / the active range.

`provenance_dict()` gains `"purity": PURITY` and `"p_dla_range":
list(PURITY_PRESETS[PURITY])` (replacing the current `"p_dla_min"`).

---

## 3. Preset-tagged outputs

So the two runs do not clobber each other, every generated file gets a
`_<preset>` tag:

- npz: `stack_curves_<preset>.npz` (was `stack_curves.npz`).
- figures: `stack_prod_<preset>.png`, `stack_metal_zoom_prod_<preset>.png`,
  `stack_lls_diag_<preset>.png`, `stack_subdla_<preset>.png`,
  `stack_dla_<preset>.png`, `stack_lyman_limit_<preset>.png`,
  `stack_bal_compare_<preset>.png`, `stack_pseudo_continuum_qc_<preset>.png`,
  `stack_control_<cat>_<preset>.png`, and the metal-zoom variants.
- diagnostics: `zhist_<preset>.png`, `zhist_summary_<preset>.txt`,
  `counts_<preset>.txt`.

Implementation: a helper `tagged(basename, ext)` →
`f"{basename}_{PURITY}.{ext}"`; `render_all` / `dump_zhist` /
`compute_stacks` build their output names through it. `npz_path()` →
`OUT_DIR / f"stack_curves_{PURITY}.npz"` replaces the `NPZ_PATH`
constant.

The current un-suffixed `stack_curves.npz` becomes orphaned; it is
gitignored and regenerated as `stack_curves_high.npz`, so no action
beyond re-running the `high` preset (also required for §4 below).

---

## 4. npz data products: pooled low-N_HI stack + persisted pseudo-continuum

### 4a. Pooled low-N_HI combined stack

The comparison figure needs a real + z-scrambled-control stack pooled
over the **whole low-N_HI range** (LLS + sub-DLA). Pooling must happen
on the raw per-spectrum arrays at stack time — you cannot correctly
median-pool two finished median curves.

Add a pooled category to `CONTROL_CATEGORIES`:

```python
CONTROL_CATEGORIES = {
    "lls":    LLS_BINS_FINE,
    "subdla": SUBDLA_BINS,
    "lownhi": LLS_BINS_FINE + SUBDLA_BINS,   # pooled — for the purity comparison
}
```

`compute_stacks` already builds `combined[name]` for every
`CONTROL_CATEGORIES` entry by pooling the raw real/control arrays of its
bins — so `combined["lownhi"]` is produced with no new pooling logic
(all six fine bins are already control bins, so their raw control
arrays exist). It is persisted in the npz as `comb_*_lownhi` by the
existing `save_curves` loop and read back by `load_curves`.

`plot_control`'s `label` dict gains `"lownhi": "low-NHI (LLS+sub-DLA)"`
so the per-category control loop in `render_all` renders a
`stack_control_lownhi_<preset>.png` too (harmless, mildly useful).

### 4b. Persisted pseudo-continuum + continuum-normalized stack

Today `fit_pseudo_continuum` is recomputed at plot time and never
saved, so the continuum-normalized stack exists only inside the figure
PNGs. To make the normalized composite a reusable data product (model
overplots, EW measurement, sharing numbers rather than re-deriving from
a figure):

- `compute_stacks` computes the pseudo-continuum
  `P = fit_pseudo_continuum(rest_grid, curve, counts)` **once** for
  every persisted composite — each per-bin non-BAL `curve` and BAL
  `curve_bal`, and each `combined` real/control curve — and carries it
  alongside the curve. `BinStack` gains `pcont` and `pcont_bal` fields;
  each `combined` tuple gains `pcont_real` and `pcont_ctrl`.
- `save_curves` writes the `P` arrays (`pcont_<key>`,
  `pcont_real_<cat>`, `pcont_ctrl_<cat>`); `load_curves` reads them
  back. The continuum-normalized stack is `curve / pcont` — a one-line
  derivation, documented in the module docstring and the npz-schema
  comment.
- The normalizing plot functions (`plot_metal_zoom`, `plot_control`,
  the `--compare-purity` figure) consume the carried/stored `P` instead
  of recomputing — `fit_pseudo_continuum` then has a single call site
  (`compute_stacks`). `plot_pseudo_continuum_qc` is the one exception:
  it re-runs `fit_pseudo_continuum(..., return_info=True)` because it
  needs the diagnostics dict (knots, rejections, RMS), not just `P`.
  `fit_pseudo_continuum` is deterministic, so routing plotting through
  the stored `P` is behaviour-preserving.
- `P` arrays are data fields, not settings — `provenance_dict()` is
  unchanged. Both presets are re-run from scratch (already required by
  §4a), so every npz carries `pcont`; no old-format fallback is needed.

---

## 5. Comparison figure + `--compare-purity` mode

New mode `--compare-purity` in `main()`: it does not stack — it loads
the two cached npz and renders one figure.

- Requires `stack_curves_high.npz` **and** `stack_curves_marginal.npz`
  to exist; otherwise `SystemExit` naming the missing file(s).
- `load_curves` is generalized to take an explicit path and an optional
  `expect_preset`: `load_curves(path, expect_preset=None)`. With
  `expect_preset` set it asserts the npz provenance `purity` matches and
  checks the non-purity provenance fields (catalog, cuts, grid, bins)
  against the current `provenance_dict()`; mismatch → `SystemExit`
  (overridable with `--force-plot`, as today).
- New `plot_purity_comparison(rest_grid, comb_high, comb_marg, fname)`:
  takes the `combined["lownhi"]` tuple from each preset — which now
  carries the stored pseudo-continuum (`pcont_real`, `pcont_ctrl`, per
  §4b). For each panel in `PURITY_COMPARE_PANELS` it overlays three
  curves — marginal-real (red), marginal z-scrambled control (grey),
  high-purity-real (blue, reference) — each normalized by its stored
  `pcont` (`curve / pcont`), with the line markers. Output:
  `stack_purity_comparison.png` (un-tagged — it spans both presets).

`PURITY_COMPARE_PANELS` is a curated subset of `ZOOM_PANELS` — the most
diagnostic lines: SiIV 1394/1403, OI 1302, CII 1335, SiII 1260,
CIV 1548/1551. Normalization uses the stored pseudo-continuum, the same
`P` that `plot_metal_zoom` uses (§4b) — no recomputation.

The figure's caption states the read: marginal-real tracking
high-purity-real ⇒ marginal detections real; marginal-real flat like
its own control ⇒ marginal operating point contaminated. The marginal
sample may be small — the n of each pooled stack is shown in the legend.

---

## 6. CLI / run flow

```
python examples/stack_real_loa_dlas.py --purity high       # → stack_curves_high.npz + *_high figures
python examples/stack_real_loa_dlas.py --purity marginal   # → stack_curves_marginal.npz + *_marginal figures
python examples/stack_real_loa_dlas.py --compare-purity     # → stack_purity_comparison.png  (seconds, no archive reads)
```

`--purity` combines with the existing `--zhist-only` / `--plot-only`
(those operate on the active preset's npz). `--compare-purity` is
mutually exclusive with stacking and ignores `--purity`.

---

## 7. SLURM

`slurm/greatlakes/stack_real_loa.sh` gains a `PURITY` env var
(default `high`) passed through as `--purity "$PURITY"`. The
marginal-purity stack is a second submission with
`--export=ALL,PURITY=marginal`. The `--compare-purity` step is cheap
(no archive reads, seconds) — run it locally / on a login node after
both npz exist, or as a tiny tail step in the marginal job.

Both presets must be (re-)run: the `high` npz must be regenerated to
carry the new `comb_*_lownhi` category (§4).

---

## 8. Tests

Extend the test suite (new file `tests/test_stack_purity.py`, importing
the script like `tests/test_stack_pseudo_continuum.py` does):

1. **Preset selection.** Build a small synthetic DLACAT-shaped
   structured array spanning P_DLA 0.0–1.0; with `PURITY="marginal"`,
   `select()` keeps exactly the rows with `0.5 < P_DLA ≤ 0.7` (and the
   other cuts); with `PURITY="high"`, exactly `P_DLA > 0.97`. Confirms
   the preset range is applied and `high` is unchanged.
2. **Provenance round-trip.** `provenance_dict()` carries `purity` +
   `p_dla_range`; `check_provenance` rejects an npz whose stored
   `purity` differs from `expect_preset`.
3. **Comparison-figure smoke test.** Construct two synthetic
   `combined["lownhi"]` tuples (from the mock composite helper) and a
   tmp `OUT_DIR`; `plot_purity_comparison` writes a non-empty png.
4. **Pseudo-continuum persistence round-trip.** `save_curves` then
   `load_curves` on a synthetic stack recovers the `pcont` arrays
   bit-for-bit; the recovered `pcont` equals `fit_pseudo_continuum` on
   the stored curve (single-source-of-truth check); `curve / pcont` is
   finite and ≈1 off-line.

The existing 11 pseudo-continuum tests must still pass (the `select()`
signature/behaviour change is additive — `high` preset is unchanged;
`fit_pseudo_continuum` itself is untouched).

---

## 9. Code-change map

All in `examples/stack_real_loa_dlas.py` unless noted:

- Constants: add `PURITY_PRESETS`, `PURITY`, `PURITY_COMPARE_PANELS`;
  remove `P_DLA_MIN` (replaced by the preset range).
- `provenance_dict()` — swap `p_dla_min` → `purity` + `p_dla_range`.
- `select()` — apply the preset range.
- `npz_path()` helper + `tagged()` helper; replace the `NPZ_PATH`
  constant and literal figure names.
- `compute_stacks`, `dump_zhist` — preset-tagged `counts`/`zhist`
  outputs.
- `CONTROL_CATEGORIES` — add `"lownhi"`; `plot_control` label dict.
- `BinStack` — add `pcont` + `pcont_bal` fields; `combined` tuples —
  add `pcont_real` + `pcont_ctrl` (§4b).
- `compute_stacks` — compute `P` via `fit_pseudo_continuum` for every
  composite and carry it in `BinStack` / `combined`.
- `save_curves` / `load_curves` — serialize the `pcont_*` arrays;
  `load_curves(path, expect_preset=None)` also generalized for §5.
- `plot_metal_zoom`, `plot_control` — normalize by the carried `pcont`
  instead of calling `fit_pseudo_continuum`.
- `plot_purity_comparison()` — new.
- `main()` — parse `--purity`; add `--compare-purity` mode.
- module docstring — document `curve / pcont` = the normalized stack.
- `slurm/greatlakes/stack_real_loa.sh` — `PURITY` env pass-through.
- `tests/test_stack_purity.py` — new.

---

## 10. Open / decided

- **Decided:** marginal range `0.5 < P_DLA ≤ 0.7`; comparison pooled
  over LLS + sub-DLA; pseudo-continuum normalization applied in the
  comparison panels; default preset `high` (backward-compatible).
- **Known caveat:** the marginal sample size is unknown until the run;
  the pooled low-N_HI stack (6 fine bins co-added) mitigates noise. If
  the marginal pooled stack has too few spectra for the <50-per-pixel
  floor, the figure will show NaN gaps — acceptable, and reported via
  the legend n.
- **Not done:** bootstrap error bars on the comparison (a later
  follow-up, already tracked on PR #8).
