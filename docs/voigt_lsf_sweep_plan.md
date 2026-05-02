# Voigt LSF + num_lines hypothesis-test sweep — plan and tools

## Why

Y3 mocks show a **+0.37 dex N_HI bias** on the canonical regression target
(2LPT TID 120046865, truth log NHI = 21.26 → MAP 21.63 with the production
Y3 model). Three (non-exclusive) hypotheses for the bias:

1. **LSF mismatch**: production C extension hardcodes the BOSS R=2000 LSF
   on a *log-λ* grid. Applied to DESI's *linear-λ* grid this is too narrow
   in pixel velocity by factor ~3 (eBOSS preset) to ~5 (Y3 preset).
   Under-broadening → too-sharp model → fitter compensates with high N_HI
   to widen the damping wings.
2. **Insufficient Lyman lines**: production uses `num_lines=3` (Lyα+β+γ).
   Higher orders (Lyδ, Lyε, ...) contribute non-trivially in some
   wavelength regions and have their own per-line damping constants.
3. **Mock-physics differences** (data-side, not inference-side):
   - **2LPT v2.8.5**: newest, expected most correct DLA implementation.
   - **Saclay v4.7.5**: also relatively recent.
   - **London v5.9.5**: known bug — `quickquasars` doesn't take into account
     the different damping constants for the Lyman series lines, instead
     rescales by oscillator strength and shifts the DLA feature to other
     Lyman lines (and uses this approach for metals too — comment from
     external analyst). The `dlaplus` branch fixes this but adds metals
     based on optical depth, so metals are added as if every Lyman line
     is Lyα (incorrectly).

This experiment isolates contributions (1) and (2) at inference time, and
quantifies (3) by comparing the same configurations across the three mock
generators.

## Experiment design

### 4 Voigt configurations (cross-product with mocks × NHI regimes)

| tag | kernel | num_lines | what it tests |
|---|---|---:|---|
| **A** | `boss-log-r2000` | 3 | production baseline |
| **B** | `desi-linear-r3000` | 3 | LSF fix isolated |
| **C** | `desi-linear-r3000` | 6 | LSF fix + more Lyman lines |
| **D** | `none` | 3 | bare Voigt — diagnostic |

A vs B isolates the LSF effect. B vs C isolates the num_lines effect.
D vs B isolates whether the LSF kernel itself does anything at the line
core (D removes the LSF entirely).

### 3 NHI regimes (user request)

User goal: *accurate log NHI ≥ 20 (DLA), okay-bias for the rest*. Test
each regime separately:

| regime | log NHI range | physics |
|---|---|---|
| LLS | [17.2, 19.0) | Doppler-core-dominated; no damping wings |
| sub-DLA | [19.0, 20.3) | transition regime |
| DLA | [20.3, 23.0] | damping-wing-dominated; LSF most relevant |

### 3 mocks × 3 regimes × 4 configs × N_PER_BIN targets

Default N_PER_BIN = 5 → 5 × 3 × 4 × 3 = **180 inferences**. At
~30–60 sec/inference on A100 → **1.5–3 hours** wall.

Targets are picked to:
- Have **exactly one** truth absorber in the search window (avoids
  multi-DLA confounds).
- Be **mid-forest** (z_qso − 0.5 ≤ z_dla ≤ z_qso − 0.05), not at
  prior edges.
- Pass an SNR cut.

## Files in this PR

| Path | Purpose |
|---|---|
| `gpy_dla_detection/voigt_v2.py` | Selectable-kernel Voigt forward model. Speedup: convolution swapped from Python loop to `np.convolve` (200× faster, 1.4× v1 baseline). |
| `tests/test_voigt_lsf_correctness.py` | 25 unit tests: kernel correctness, per-line damping, NHI-regime scaling, boundary cases. |
| `tests/profile/profile_voigt.py` | CPU profiler for v1 vs v2 vs torch-CPU port. |
| `tests/profile/results/voigt_profile.md` | Profiling result table + per-spectrum projection. |
| `examples/pick_voigt_sweep_targets.py` | Stratified target picker (3 mocks × 3 regimes). |
| `examples/voigt_lsf_sweep.py` | Multi-process sweep runner (one fresh process per config × target — needed because the kernel swap mutates a global). |
| `examples/analyze_voigt_sweep.py` | Read master.csv, write report.md with figures. |
| `examples/voigt_kernel_demo.py` | Standalone demonstration: shows what kernel choice does to a forward-modelled DLA in 4 NHI regimes. |
| `slurm/greatlakes/voigt_lsf_sweep.sh` | One-shot SLURM submit (pick + sweep + analyze in one job). |
| `tests/test_voigt_sweep_targets.py` | Unit tests for the picker — synthetic mocks under tmpdir. |

## Demo figure: what the LSF kernel actually does

`python examples/voigt_kernel_demo.py --out demo.png`

Produces a 4-panel comparison (LLS / sub-DLA / DLA / strong DLA) of the
bare Voigt vs `boss-log-r2000` vs `desi-linear-r3000`. Visual confirmation
that the kernel difference is at the **line core** — for DLA-regime
profiles, damping wings dominate so the kernel matters less than expected;
for sub-DLA / LLS, the kernel determines almost everything.

## Profiling result (from the new `np.convolve` speedup)

| variant | μs/Voigt call (n_pix=600, num_lines=3) | speedup vs v1 |
|---|---:|---:|
| v1 production C ext | 90 | (reference) |
| v2 `boss-log-r2000` | 127 | **0.71×** v1 |
| v2 `desi-linear-r3000` | 140 | 0.64× |
| v2 `none` (no convolution) | 120 | 0.75× |
| v2 num_lines=31 | 882 | 0.10× (linear in lines) |
| v2 torch CPU (no GPU) | 235 | 0.38× — wofz isn't accelerated |

**v2 is now ~70% the speed of v1**, down from ~10% before the np.convolve
fix. Per-spectrum projection: at 30k Voigt calls/spectrum (FILTER=1 + 1 DLA),
v2 is 3.6 s vs 2.7 s for v1 — manageable difference.

## How to run the sweep

```bash
# On GreatLakes (one job does everything):
ssh greatlakes
cd /home/mfho/desi_gpy_dla_detection
git fetch && git checkout claude/voigt-lsf-fix && git pull
sbatch slurm/greatlakes/voigt_lsf_sweep.sh

# Default: N_PER_BIN=5 → 180 inferences. Override:
sbatch --export=ALL,N_PER_BIN=2 slurm/greatlakes/voigt_lsf_sweep.sh   # 72 inferences (~30 min)
sbatch --export=ALL,N_PER_BIN=10 slurm/greatlakes/voigt_lsf_sweep.sh  # 360 inferences (~3 h)
```

Output:
```
/nfs/turbo/.../voigt_sweep_<jobid>/
  targets.tsv               ← picked targets
  runs/                     ← per-(target, config) HDF5s
    {mock}_{regime}_tid{N}_{config}.h5
    master.csv              ← summary across all runs
  report/
    report.md               ← analysis + figures
docs/notes/<date>_voigt_lsf_sweep/
  (same content, copied for git)
```

## Expected outcomes

If the LSF mismatch is the dominant bias driver:
- Config B (DESI LSF) should reduce the median ΔlogNHI in the **DLA regime**
  by ~0.2–0.4 dex relative to config A on 2LPT and Saclay. London should
  show similar but possibly noisier results because of the mock-generator
  bug.
- Configs B and D should be **similar** in the sub-DLA / LLS regimes
  (line core only, kernel choice matters less).
- Config C (more Lyman lines) might add a small further reduction —
  the higher-order Lyman lines bleed into the **Lyβ wing** of strong
  DLAs which is at λ_obs ~ 0.84 × Lyα.

If the LSF mismatch is **not** the dominant driver:
- B's bias ≈ A's. The signal would then be in the mock-physics or QMC
  prior, and we'd need to look at Step 2/3/4 of the Bayesian-correctness
  plan (`docs/notes/2026-04-27_bayesian_correctness_plan.md`).

## Caveats

- LLS / sub-DLA absorbers in mocks may use simpler physics than DLAs.
  Comparing config A bias across mocks for the LLS regime tests *mock
  generation*, not inference correctness — interpret carefully.
- 4 configs × 9 (mock × regime) cells × 5 targets is **45 targets per
  config** — adequate for median bias with ~0.05 dex uncertainty on the
  median, but tight on tail behaviour. Increase N_PER_BIN if needed.
- The Voigt forward model is the only thing changing across configs.
  The GP continuum (μ + Mω + ...) is unchanged. So this experiment
  cannot distinguish a *Voigt* bias from a *continuum* bias if both
  exist in the same direction.
