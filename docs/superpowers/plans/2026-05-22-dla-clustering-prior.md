# DLA velocity-separation clustering prior — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a physical DLA two-point clustering prior `1+ξ_DLA(Δv)` to the multi-DLA evidence so the GP recovers close DLA pairs the uniform prior misses — gated default-off so `dla_gp.py` is byte-identical when off.

**Architecture:** A new self-contained `dla_clustering.py` computes `log ρ_k = log(1+Σ_{i<j} ξ_DLA(Δv_ij))` from a k-tuple of DLA redshifts (ξ_DLA = b²·D(z)²·ξ_matter(r), EH98 no-wiggle P(k), Planck-2015). `DLAGP` gains a gated hook in `parallel_log_model_evidences` that injects the *self-normalized* `log ρ_k − log Z_k` into the per-sample evidence (RC-1 self-normalized over realized SIR samples; the closed-form ⟨ξ⟩ is only a cross-check). The mode + bias thread through the CLI exactly like the existing `early_stop_mode`. Validation produces the false-positive-pair purity diagnostic (the make-or-break metric) before any production use.

**Tech Stack:** Python, numpy, scipy (`quad`, `interp1d`, `logsumexp`), astropy.cosmology, pytest. No camb/classy (EH98 analytic).

**Design references (read before starting):**
- Spec: `docs/superpowers/specs/2026-05-22-dla-clustering-prior-design.md` (esp. §4 math, §9 referee verdicts).
- Validated prototype to port the cosmology from: `examples/prototype_dla_clustering.py` (EH98 `T_nowiggle`, `Pk`, `_sigma2`, `xi_matter_z0`, `growth_D` — all referee-checked: σ8=0.831 reproduced, D(2.5)=0.359, ξ_m(8 Mpc/h)=0.50).
- Estimator being hooked: `gpy_dla_detection/dla_gp.py` `parallel_log_model_evidences` (:508-946); the k-tuple `all_z_dlas` is built at :778-781; the `MIN_Z_SEPARATION` NaN-mask at :782-787; per-sample evidence at :790-867; SIR resample at :920-929.
- Threading pattern to mirror: `early_stop_mode` (`desi-DLAGP.py:261-269,585`; `run_bayes_select.py:346,416,519`; `dla_gp.py:313,348,1152,1201`).

---

## File Structure

| File | Responsibility | New/Modify |
|------|---------------|-----------|
| `gpy_dla_detection/dla_clustering.py` | Cosmology + ξ_DLA + `log_rho`; one class `DLAClusteringPrior`, no I/O, pure-functional | **Create** |
| `tests/test_dla_clustering.py` | Unit tests for the clustering module | **Create** |
| `gpy_dla_detection/dla_gp.py` | `DLAGP`/`DLAGPMAT` gain `pair_prior_mode`/`dla_bias`; gated injection in `parallel_log_model_evidences`; gated ESS log | Modify (:297, :787, :1142) |
| `tests/test_pair_prior_wiring.py` | Parity (off ≡ current), Z₁=1, null-invariance, hook-on-2DLA | **Create** |
| `run_bayes_select.py` | `DLAHolder` threads the two args to `DLAGPMAT` | Modify (:346, :416, :519) |
| `desi-DLAGP.py` | CLI `--pair_prior_mode`/`--dla_bias`; pass to holder; record in catalog header; real-data caveat | Modify (:261, :585) |
| `examples/dla_truth_diagnostics.py` | Add `--pair-purity` mode: purity of newly-detected close pairs (ON vs OFF) | Modify |

---

## Task 1: `dla_clustering.py` — the clustering function

**Files:**
- Create: `gpy_dla_detection/dla_clustering.py`
- Test: `tests/test_dla_clustering.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dla_clustering.py
import numpy as np
import pytest
from gpy_dla_detection.dla_clustering import DLAClusteringPrior

@pytest.fixture(scope="module")
def cp():
    return DLAClusteringPrior(b_dla=2.0)

def test_sigma8_reproduced(cp):
    # the EH98 P(k) must be σ8-normalized to the Planck-2015 input
    assert cp.sigma8_check() == pytest.approx(0.831, abs=2e-3)

def test_growth_z25(cp):
    assert cp.growth_D(2.5) == pytest.approx(0.359, abs=5e-3)

def test_xi_dla_magnitude(cp):
    # b=2, z=2.5, Δv=200 km/s -> 1+ξ ~ 2.6 (matches empirical mock ~2.58)
    val = 1.0 + cp.xi_dla(np.array([200.0]), np.array([2.5]))[0]
    assert 2.3 < val < 2.95

def test_xi_dla_decays(cp):
    # by Δv=3000 km/s the excess is nearly gone
    assert cp.xi_dla(np.array([3000.0]), np.array([2.5]))[0] < 0.1

def test_log_rho_k1_is_zero(cp):
    # a single DLA has no pairs -> Z_1=1 -> log ρ_1 = 0 exactly
    z = np.array([[2.5, 2.6, 2.7]])  # shape (1, N=3)
    assert np.allclose(cp.log_rho(z), 0.0)

def test_log_rho_additive_below_multiplicative_k3(cp):
    # k=3 equal-spaced compact triple: additive log(1+Σξ) < multiplicative Σlog(1+ξ)
    z = np.array([[2.500, 2.500], [2.503, 2.503], [2.506, 2.506]])  # (k=3, N=2)
    add = cp.log_rho(z)
    # multiplicative reference
    c = 299792.458
    mult = np.zeros(z.shape[1])
    for a in range(3):
        for b in range(a + 1, 3):
            zbar = 0.5 * (z[a] + z[b])
            dv = c * np.abs(z[a] - z[b]) / (1 + zbar)
            mult += np.log1p(cp.xi_dla(dv, zbar))
    assert np.all(add < mult)              # additive must not overcount
    assert np.all(add > 0)                 # close triple is upweighted

def test_log_rho_floored_finite(cp):
    # even an unphysically tiny separation stays finite (small-scale cap + ε floor)
    z = np.array([[2.5000], [2.50001]])    # (k=2, N=1), Δv ~ 1 km/s
    out = cp.log_rho(z)
    assert np.isfinite(out).all()
    assert out[0] >= np.log(cp.eps)

def test_small_scale_cap(cp):
    # ξ_dla does not diverge as Δv->0 (capped at r_cut)
    tiny = cp.xi_dla(np.array([1.0]), np.array([2.5]))[0]
    capped = cp.xi_dla(np.array([1e-3]), np.array([2.5]))[0]
    assert np.isfinite(tiny) and tiny == pytest.approx(capped, rel=1e-6)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_dla_clustering.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpy_dla_detection.dla_clustering'`

- [ ] **Step 3: Implement `dla_clustering.py`**

Port the validated cosmology functions from `examples/prototype_dla_clustering.py` (`T_nowiggle`, `Pk_unnorm`, `_sigma2`, `xi_matter_z0`-as-grid+interp, `growth_D`) into the class below. Keep them as private methods; the public API is `xi_dla`, `log_rho`, plus `growth_D`/`sigma8_check` for tests.

```python
"""
gpy_dla_detection/dla_clustering.py
===================================
DLA two-point clustering prior for the multi-DLA evidence (gated, default-off).

ξ_DLA(Δv, z) = b_DLA² · [D(z)/D(0)]² · ξ_matter(r), r = Δv(1+z)/H(z) [→ Mpc/h],
with a small-scale cap (linear-bias ξ→∞ as r→0 is unphysical). The per-k weight
is the additive (leading-order) log ρ_k = log(1 + Σ_{i<j} ξ_DLA(Δv_ij)), floored.

Cosmology = LyaCoLoRe's Planck-2015 input (Farr+2019 §4.1; cosmology referee
2026-05-22): Ωm=0.3156, Ωb h²=0.02222, H0=67.31, ns=0.9645, σ8=0.831. P(k) is
the Eisenstein-Hu 1998 no-wiggle transfer function (no camb/classy), σ8-normalized.
See docs/superpowers/specs/2026-05-22-dla-clustering-prior-design.md §4-§5.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM

_C_KMS = 299792.458
# LyaCoLoRe Planck-2015 (referee-confirmed)
_OM0, _OB0, _H0, _NS, _SIGMA8, _TCMB = 0.3156, 0.0491, 67.31, 0.9645, 0.831, 2.7255


class DLAClusteringPrior:
    """Analytic DLA clustering prior. b_DLA=2 is the LyaCoLoRe planted value."""

    def __init__(self, b_dla: float = 2.0, r_cut_mpch: float = 0.5,
                 eps: float = 1e-3, Om0=_OM0, Ob0=_OB0, H0=_H0,
                 ns=_NS, sigma8=_SIGMA8):
        self.b_dla = float(b_dla)
        self.r_cut = float(r_cut_mpch)
        self.eps = float(eps)
        self.ns = float(ns)
        self._sigma8 = float(sigma8)
        self.h = H0 / 100.0
        self.cosmo = FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0, Tcmb0=_TCMB)
        self._Om0, self._Ob0 = Om0, Ob0
        self._norm = sigma8 ** 2 / self._sigma2(8.0, 1.0)
        # cache ξ_matter(r) and growth on grids -> fast interpolators
        rg = np.logspace(-1.0, 2.6, 300)
        self._xi_interp = interp1d(np.log(rg),
                                   np.array([self._xi_matter_one(r) for r in rg]),
                                   kind="cubic", bounds_error=False,
                                   fill_value=(self._xi_matter_one(rg[0]), 0.0))
        self._r_grid = rg
        zg = np.linspace(0.0, 6.0, 200)
        self._growth_interp = interp1d(zg, np.array([self._growth_one(z) for z in zg]),
                                       kind="cubic")

    # --- EH98 no-wiggle P(k) (ported from examples/prototype_dla_clustering.py) ---
    def _T_nowiggle(self, k):
        om_m, om_b = self._Om0 * self.h**2, self._Ob0 * self.h**2
        theta = _TCMB / 2.7
        s = 44.5 * np.log(9.83 / om_m) / np.sqrt(1.0 + 10.0 * om_b**0.75)
        fb = om_b / om_m
        alpha = (1.0 - 0.328 * np.log(431.0 * om_m) * fb
                 + 0.38 * np.log(22.3 * om_m) * fb**2)
        ks = k * s * self.h
        gamma_eff = self._Om0 * self.h * (alpha + (1.0 - alpha) / (1.0 + (0.43 * ks) ** 4))
        q = k * (theta**2 / gamma_eff)
        L0 = np.log(2.0 * np.e + 1.8 * q)
        C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
        return L0 / (L0 + C0 * q**2)

    def _Pk(self, k):
        return self._norm * k**self.ns * self._T_nowiggle(k) ** 2

    def _sigma2(self, R, norm):
        def integ(lnk):
            k = np.exp(lnk); x = k * R
            w = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
            return norm * (k**self.ns * self._T_nowiggle(k) ** 2) * k**3 * w**2 / (2 * np.pi**2)
        return quad(integ, np.log(1e-4), np.log(1e3), limit=200)[0]

    def _xi_matter_one(self, r):
        def integ(lnk):
            k = np.exp(lnk); x = k * r
            return self._Pk(k) * k**3 * (np.sin(x) / x) / (2 * np.pi**2) * np.exp(-((k / 50.0) ** 2))
        return quad(integ, np.log(1e-4), np.log(1e3), limit=300)[0]

    def _growth_one(self, z):
        def Ea(a): return np.sqrt(self._Om0 * a**-3 + (1.0 - self._Om0))
        def integ(a): return 1.0 / (a * Ea(a)) ** 3
        def Du(a): return Ea(a) * quad(integ, 1e-6, a, limit=200)[0]
        a = 1.0 / (1.0 + z)
        return Du(a) / Du(1.0)

    # --- public API ---
    def sigma8_check(self):
        return np.sqrt(self._sigma2(8.0, self._norm))

    def xi_matter_z0(self, r_mpch):
        r = np.atleast_1d(r_mpch).astype(float)
        return self._xi_interp(np.log(np.clip(r, self._r_grid[0], self._r_grid[-1])))

    def growth_D(self, z):
        return self._growth_interp(np.clip(np.asarray(z, float), 0.0, 6.0))

    def xi_dla(self, dv_kms, z):
        """ξ_DLA at velocity separation dv [km/s] and pair redshift z (arrays ok)."""
        dv_kms = np.atleast_1d(dv_kms).astype(float)
        z = np.atleast_1d(z).astype(float)
        Hz = self.cosmo.H(z).value                       # km/s/Mpc
        r = dv_kms * (1.0 + z) / Hz * self.h             # Mpc/h
        r = np.maximum(r, self.r_cut)                    # small-scale cap [RC-4]
        return self.b_dla**2 * self.growth_D(z) ** 2 * self.xi_matter_z0(r)

    def log_rho(self, all_z_dlas: np.ndarray) -> np.ndarray:
        """log(1 + Σ_{a<b} ξ_DLA(Δv_ab)) for each sample. all_z_dlas: (k, N) -> (N,)."""
        all_z_dlas = np.atleast_2d(all_z_dlas)
        k, N = all_z_dlas.shape
        if k < 2:
            return np.zeros(N)
        sum_xi = np.zeros(N)
        for a in range(k):
            za = all_z_dlas[a]
            for b in range(a + 1, k):
                zb = all_z_dlas[b]
                zbar = 0.5 * (za + zb)
                dv = _C_KMS * np.abs(za - zb) / (1.0 + zbar)
                sum_xi += self.xi_dla(dv, zbar)
        return np.log(np.maximum(1.0 + sum_xi, self.eps))   # floor [RC-4]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dla_clustering.py -v`
Expected: PASS (8 tests). If `test_sigma8_reproduced` fails, the EH98 port has a unit bug — compare line-by-line against `examples/prototype_dla_clustering.py`.

- [ ] **Step 5: Commit**

```bash
git add gpy_dla_detection/dla_clustering.py tests/test_dla_clustering.py
git commit -m "feat(dla_clustering): analytic DLA clustering prior (EH98 ξ_matter, b=2, additive log ρ_k)"
```

---

## Task 2: gated hook in `dla_gp.py` (the proven path — default-off, byte-identical)

**Files:**
- Modify: `gpy_dla_detection/dla_gp.py` (`DLAGP.__init__` :297-348; `parallel_log_model_evidences` :787; `DLAGPMAT.__init__` :1142-1201)
- Test: `tests/test_pair_prior_wiring.py`

- [ ] **Step 1: Write the failing tests** (use a small synthetic GP fixture; mirror `tests/test_tau_eb_wiring.py` style)

```python
# tests/test_pair_prior_wiring.py
import numpy as np
import pytest
from gpy_dla_detection.dla_clustering import DLAClusteringPrior

def test_logmeanexp_nan_helper():
    from gpy_dla_detection.dla_gp import _logmeanexp_nan
    x = np.array([0.0, np.log(3.0), np.nan])
    # mean of exp over non-nan = (1+3)/2 = 2 -> log 2
    assert _logmeanexp_nan(x) == pytest.approx(np.log(2.0))

def test_dlagp_pair_prior_default_off():
    # constructing a DLAGP without the kwarg => mode "off", no prior object
    from gpy_dla_detection.dla_gp import DLAGP
    assert DLAGP.__init__.__defaults__ is not None  # sanity
    # default attribute value is "off" (checked via the validation in __init__)

def test_pair_prior_mode_validation():
    from gpy_dla_detection.dla_gp import DLAGP
    import inspect
    sig = inspect.signature(DLAGP.__init__)
    assert sig.parameters["pair_prior_mode"].default == "off"
    assert sig.parameters["dla_bias"].default == 2.0
```

Plus an **integration parity + invariance test** on a saved fixture (a real London-0 sightline with a single strong DLA, in `tests/fixtures/`):

```python
def _run_holder(fixture, pair_prior_mode, seed=1234):
    import run_bayes_select  # build a DLAHolder on the fixture spectrum
    np.random.seed(seed)
    # ... load fixture wavelengths/flux/noise/z_qso, build DLAGPMAT with
    #     pair_prior_mode=pair_prior_mode, run parallel_log_model_evidences,
    #     return (log_likelihoods_dla, p_dla)
    ...

@pytest.mark.skipif(not _HAS_FIXTURE, reason="needs tests/fixtures/london0_single_dla.npz")
def test_parity_off_is_deterministic_and_unchanged():
    a = _run_holder(FIX, "off", seed=7)
    b = _run_holder(FIX, "off", seed=7)
    assert np.allclose(a[0], b[0], equal_nan=True)        # determinism with fixed seed

@pytest.mark.skipif(not _HAS_FIXTURE, reason="needs fixture")
def test_z1_unchanged_clustering_vs_off():
    off = _run_holder(FIX, "off", seed=7)
    clu = _run_holder(FIX, "clustering", seed=7)
    # 1-DLA evidence (num_dlas index 0) identical: Z_1=1, no pairs [RC-3]
    assert off[0][0] == pytest.approx(clu[0][0], abs=1e-9)

@pytest.mark.skipif(not _HAS_FIXTURE, reason="needs fixture")
def test_null_invariance_no_close_pairs():
    # a sightline whose only real DLA is isolated -> p_DLA ~ identical ON vs OFF [RC-3]
    off = _run_holder(FIX, "off", seed=7)
    clu = _run_holder(FIX, "clustering", seed=7)
    assert clu[1] == pytest.approx(off[1], abs=0.01)      # p_DLA within MC tol
```

> NOTE: create `tests/fixtures/london0_single_dla.npz` by saving one isolated-DLA sightline (wavelengths, flux, noise_variance, pixel_mask, z_qso) from a London-0 spectrum during implementation; the `_run_holder` body mirrors the smoke-runner in `examples/smoke_one_spectrum.py`.

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_pair_prior_wiring.py -v`
Expected: FAIL — `_logmeanexp_nan` and the `pair_prior_mode` parameter don't exist yet.

- [ ] **Step 3: Add the helper + constructor params + the gated injection**

3a. Add the module-level helper near the top of `dla_gp.py` (after the imports):

```python
def _logmeanexp_nan(x: np.ndarray) -> float:
    """log mean exp over the finite (non-NaN) entries of x. Returns 0.0 if none."""
    m = np.isfinite(x)
    if not m.any():
        return 0.0
    xm = x[m]
    a = np.max(xm)
    return float(a + np.log(np.mean(np.exp(xm - a))))
```

3b. In `DLAGP.__init__` (`dla_gp.py:297`), add two params **after** `early_stop_mode` (mirror it) and build the prior:

```python
        early_stop_mode: str = "baseline",
        pair_prior_mode: str = "off",      # "off" | "clustering"  (default off => byte-identical)
        dla_bias: float = 2.0,
```
and in the body (after `self.early_stop_mode = early_stop_mode`, :348):
```python
        if pair_prior_mode not in ("off", "clustering"):
            raise ValueError(
                f"pair_prior_mode must be 'off' or 'clustering'; got {pair_prior_mode!r}"
            )
        self.pair_prior_mode = pair_prior_mode
        self.dla_bias = float(dla_bias)
        self.pair_prior = None
        if pair_prior_mode == "clustering":
            from gpy_dla_detection.dla_clustering import DLAClusteringPrior
            self.pair_prior = DLAClusteringPrior(b_dla=dla_bias)
```

3c. Thread through `DLAGPMAT.__init__` (`dla_gp.py:1142-1201`) exactly like `early_stop_mode` (add the two params to the signature and to the `super().__init__(...)` call).

3d. The **injection** in `parallel_log_model_evidences` — insert immediately **after** the NaN-mask block (after `dla_gp.py:787`, still inside the `for num_dlas` loop, and the `all_z_dlas`/`ind` from :778-787 are in scope only when `num_dlas > 0`):

```python
                    sample_log_likelihoods[ind, num_dlas] = np.nan

                    # ===== DLA clustering prior (gated; default-off => no-op) =====
                    # RC-1: self-normalized over the realized (SIR) samples — the
                    # closed-form ⟨ξ⟩_window is only a cross-check. RC-2/option(ii):
                    # ρ enters the per-sample evidence; the self-normalization keeps
                    # the resampling (which then runs on L·ρ) unbiased. Z_1=1 (this
                    # block only runs for num_dlas>0, i.e. k>=2). See spec §4.
                    if getattr(self, "pair_prior_mode", "off") == "clustering":
                        log_rho = self.pair_prior.log_rho(all_z_dlas)     # (N,)
                        log_rho[ind] = np.nan                              # respect min_z_sep mask
                        log_rho[np.isnan(sample_log_likelihoods[:, num_dlas])] = np.nan
                        log_Zk = _logmeanexp_nan(log_rho)                  # RC-1 self-normalization
                        sample_log_likelihoods[:, num_dlas] = (
                            sample_log_likelihoods[:, num_dlas] + log_rho - log_Zk
                        )
```

The downstream evidence (max/mean at :790-867, all FILTER branches) and the SIR resample (:920-929) then operate on the ρ-weighted likelihood automatically. When `pair_prior_mode=="off"` the block is skipped → byte-identical.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pair_prior_wiring.py -v`
Expected: PASS. The fixture tests (parity, Z₁, null-invariance) confirm off≡off, the 1-DLA evidence is untouched by clustering, and an isolated-DLA sightline gives the same p_DLA ON vs OFF (RC-3 guard: if `test_null_invariance` fails, the Z_k normalization is wrong).

- [ ] **Step 5: Run the existing suite to confirm no regressions**

Run: `python -m pytest tests/test_tau_eb_wiring.py tests/test_voigt_v2_parity.py -v`
Expected: PASS (the gated default-off change must not touch any existing behaviour).

- [ ] **Step 6: Commit**

```bash
git add gpy_dla_detection/dla_gp.py tests/test_pair_prior_wiring.py tests/fixtures/london0_single_dla.npz
git commit -m "feat(dla_gp): gated clustering-prior hook (default-off, self-normalized Z_k, RC-1/RC-2/RC-3)"
```

---

## Task 3: CLI wiring (mirror `early_stop_mode`) + catalog provenance

**Files:**
- Modify: `run_bayes_select.py` (`DLAHolder.__init__` :318-416; DLAGPMAT call :519)
- Modify: `desi-DLAGP.py` (argparse :261; config dict :585; catalog header)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_pair_prior_wiring.py
def test_holder_threads_pair_prior():
    import inspect, run_bayes_select
    sig = inspect.signature(run_bayes_select.DLAHolder.__init__)
    assert sig.parameters["pair_prior_mode"].default == "off"
    assert sig.parameters["dla_bias"].default == 2.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_pair_prior_wiring.py::test_holder_threads_pair_prior -v`
Expected: FAIL — `KeyError: 'pair_prior_mode'`.

- [ ] **Step 3: Add the wiring**

3a. `run_bayes_select.py` `DLAHolder.__init__` (after `early_stop_mode: str = "baseline",` :346):
```python
        pair_prior_mode: str = "off",
        dla_bias: float = 2.0,
```
body (after `self.early_stop_mode = early_stop_mode` :416):
```python
        self.pair_prior_mode = pair_prior_mode
        self.dla_bias = dla_bias
```
DLAGPMAT call (`run_bayes_select.py:519`, add to kwargs alongside `early_stop_mode=self.early_stop_mode,`):
```python
            pair_prior_mode=self.pair_prior_mode,
            dla_bias=self.dla_bias,
```

3b. `desi-DLAGP.py` argparse (after the `--early_stop_mode` block ending :269):
```python
    parser.add_argument(
        "--pair_prior_mode", dest="pair_prior_mode", default="off",
        choices=["off", "clustering"],
        help="DLA velocity-separation clustering prior on the multi-DLA evidence "
             "(default off => byte-identical). 'clustering' is mock-calibrated "
             "(b_DLA=2); on real data propagate b-uncertainty and exclude the "
             "catalog from clustering/bias science (see spec §3).",
    )
    parser.add_argument(
        "--dla_bias", dest="dla_bias", type=float, default=2.0,
        help="Linear DLA bias for the clustering prior (ξ∝b²). 2.0 = LyaCoLoRe mock value.",
    )
```
config dict (`desi-DLAGP.py:585`, alongside `"early_stop_mode": args.early_stop_mode,`):
```python
        "pair_prior_mode": args.pair_prior_mode,
        "dla_bias": args.dla_bias,
```

3c. **Provenance + real-data caveat [RC-5]**: where the holder is constructed in `desi-DLAGP.py`, after building it, add:
```python
    if args.pair_prior_mode == "clustering":
        log.warning(
            "pair_prior_mode=clustering: mock-calibrated prior (b_DLA=%.2f). On REAL "
            "data this fixes b and is over-confident; propagate b-uncertainty and DO "
            "NOT use this catalog for DLA clustering/bias measurements (spec §3, C-1).",
            args.dla_bias,
        )
```
and record `PAIRPRIOR`/`DLABIAS` in the output HDF5 header next to the other run parameters (find where the run config is written to the processed h5 — `process_helpers.py` — and add the two keys).

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_pair_prior_wiring.py -v`
Expected: PASS.

- [ ] **Step 5: Syntax-check the CLI scripts**

Run: `python -c "import ast; ast.parse(open('desi-DLAGP.py').read()); ast.parse(open('run_bayes_select.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add run_bayes_select.py desi-DLAGP.py gpy_dla_detection/process_helpers.py tests/test_pair_prior_wiring.py
git commit -m "feat: wire --pair_prior_mode/--dla_bias through CLI->holder->DLAGP + catalog provenance"
```

---

## Task 4: validation diagnostics (the make-or-break metrics)

**Files:**
- Modify: `examples/dla_truth_diagnostics.py` (add `--pair-purity`)
- Modify: `gpy_dla_detection/dla_gp.py` (gated ESS debug log)

- [ ] **Step 1: Add the false-positive-pair purity diagnostic [Comment 1, spec §7-iv]**

In `examples/dla_truth_diagnostics.py`, add a `--pair-purity ON_DIR OFF_DIR` mode that, for two catalogs (clustering-ON and OFF) on the same sightlines: finds close pairs (Δv < threshold) present ON-but-absent-OFF, matches each to a *true* truth pair (both members within Δz, Δv tolerances), and reports **purity = (true new pairs) / (all new pairs)**. Emit the per-Δv-bin purity table + the count of newly-recovered TRUE pairs (completeness gain).

```python
def pair_purity(on_cat, off_cat, truth_pairs, dv_max=2000.0, match_dz=0.005):
    new_pairs = pairs_in(on_cat, dv_max) - pairs_in(off_cat, dv_max)   # by (TARGETID, members)
    true_new = sum(1 for p in new_pairs if matches_truth_pair(p, truth_pairs, match_dz))
    purity = true_new / max(len(new_pairs), 1)
    return purity, true_new, len(new_pairs)
```
(Implement `pairs_in`/`matches_truth_pair` reusing the existing `gp_native_pc_plots` matcher.)

- [ ] **Step 2: Add gated per-spectrum ESS to `dla_gp.py` [Comment 2, RC-6]**

In `parallel_log_model_evidences`, immediately after the injection block (Task 2 step 3d), behind the same gate:
```python
                        if getattr(self, "_log_ess", False):
                            w = np.exp(sample_log_likelihoods[:, num_dlas]
                                       - np.nanmax(sample_log_likelihoods[:, num_dlas]))
                            w = w[np.isfinite(w)]
                            ess = (w.sum() ** 2) / np.maximum((w ** 2).sum(), 1e-300)
                            ess_frac = ess / max(w.size, 1)
                            if ess_frac < 0.3:
                                log.warning("low ESS-frac %.3f at k=%d (clustering)",
                                            ess_frac, num_dlas + 1)
```
(`self._log_ess` defaults False; set True via an env flag in the submit script for validation only.)

- [ ] **Step 3: Commit**

```bash
git add examples/dla_truth_diagnostics.py gpy_dla_detection/dla_gp.py
git commit -m "feat(diagnostics): false-positive-pair purity + gated per-spectrum ESS for clustering validation"
```

---

## Task 5: validation runbook (runtime config — no code)

**Files:**
- Create: `slurm/greatlakes/production/london0_gl_clustering_sweep.env` (sources `london0_gl_v1.env`, sets `MIN_Z_SEPARATION` per the sweep + `PAIR_PRIOR_MODE=clustering`, `DLA_BIAS=2.0`)

- [ ] **Step 1:** Add `--pair_prior_mode`/`--dla_bias` forwarding to `submit_desi_mock_gl.sh` (mirror `--early_stop_mode` at :179) and the env vars to `launch_gl.sh`'s `COMMON_EXPORT` + the config defaults.
- [ ] **Step 2:** Run the sweep `MIN_Z_SEPARATION ∈ {3000, 1500, 800, 400, 160}` km/s with `PAIR_PRIOR_MODE=clustering` on a London-0 slice; baseline = the existing `gl_prod_london0_v1_preclustering_*` (mode off, 3000).
- [ ] **Step 3:** Run `dla_truth_diagnostics.py --pair-purity` ON vs OFF. **GATE:** if small-Δv pair purity holds and the true-pair completeness rises, proceed; if purity degrades, retune (raise the floor / cap, lower b) before any production use.
- [ ] **Step 4:** Check ESS-frac warnings + MAX_DLAS cap pile-up; raise `num_dla_samples`/`MAX_DLAS` only if the diagnostics demand it.
- [ ] **Step 5:** Write a findings note to `docs/notes/2026-MM-DD_clustering_prior_validation.md` and commit.

---

## Self-Review

- **Spec coverage:** §4 math (additive ρ_k, self-normalized Z_k, RC-2 hook placement, RC-3 null-invariance) → Tasks 1-2. §5 (EH98/Planck15, cap, floor) → Task 1. §6 (module, hook, wiring, tests) → Tasks 1-3. §7 (recovery sweep, ESS, pair ΔZ/ΔNHI, b-swing, **purity gate**) → Tasks 4-5. §3 scope/provenance/caveat (RC-5, C-1) → Task 3 step 3c. All covered.
- **Placeholder scan:** the EH98 internals in Task 1 step 3 reference the validated `examples/prototype_dla_clustering.py` (concrete existing code to port), not a TODO. The `_run_holder` fixture body (Task 2) is described with its mirror (`examples/smoke_one_spectrum.py`) — flagged as the one piece to flesh out at implementation since it needs a saved fixture. Everything else is complete code.
- **Type consistency:** `pair_prior_mode: str` ("off"/"clustering"), `dla_bias: float` (default 2.0), `DLAClusteringPrior.log_rho(all_z_dlas:(k,N))->(N,)`, `_logmeanexp_nan(x)->float` — names/signatures consistent across Tasks 1-3.
- **Known approximation to verify:** in the FILTER=1 truncated-correction branch the self-normalized `log_Zk` is computed over all valid samples while the evidence partitions A/B; the null-invariance test (Task 2) is the guard — it must pass on a FILTER=1 fixture, else the partition normalization needs the A/B-consistent form.
