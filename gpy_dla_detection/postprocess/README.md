# Post-processing — Lyβ veto + LLS cross-reference

Post-inference helpers that flag MAP DLAs which are most likely
**spurious** (not a real DLA) and let downstream science choose to
ignore them. Nothing is deleted — only flag columns are added.

The pipeline produces two flavours of catalog per release:

- **DLA-mode** (multi-DLA, NHI prior [20.0, 23.0]) — the headline DLA
  catalog, used for cosmology and BAO.
- **LLS-mode** (single absorber, PW14 prior NHI [17.2, 22.0]) — used
  for f(N), dN/dX, Ω_HI population statistics.

The DLA-mode catalog has two known false-positive populations that the
helpers in this directory address:

1. **Lyβ misidentification.** A real DLA at z_real produces Lyα
   absorption at λ_obs = (1 + z_real) · λ_Lyα and Lyβ absorption at the
   same observed wavelength. A multi-DLA finder that searches for "Lyα
   features" can pick up the Lyβ feature as a *separate* DLA at the
   apparent redshift `z_lybeta_apparent = (λ_Lyβ/λ_Lyα)·(1 + z_real) − 1
   ≈ 0.844·(1 + z_real) − 1`. Even though the production Voigt model
   includes Lyβ + Lyγ in each DLA's profile, the multi-DLA evidence
   integral can still favor the spurious second DLA — see the docstring
   in `lyb_veto.py` for the mechanism.

2. **Sub-threshold absorbers inflated to the prior edge.** The DLA prior
   places no posterior mass below 20.0, so a real LLS or sub-DLA can be
   fit at log NHI ≈ 20.0–20.3 in DLA-mode. LLS-mode runs the same
   spectrum with a prior that *can* place mass at 17.2–20.3, so a
   high-confidence LLS-mode posterior at log NHI < 20.3 means DLA-mode
   was probably wrong.

## Usage

```python
from astropy.table import Table
from gpy_dla_detection.postprocess.lyb_veto import flag_lybeta
from gpy_dla_detection.postprocess.lls_cross_reference import cross_reference_lls

# Load the catalogs produced by the production pipeline
dla_cat = Table.read("dlacat-iron-main-dark-hpx-0-3000.fits")
lls_cat = Table.read("dlacat-iron-main-dark-hpx-0-3000-lls.fits")

# Step 1: flag Lyβ misidentifications (within-catalog operation)
dla_cat = flag_lybeta(dla_cat, dz_match=0.005)

# Step 2: cross-reference against LLS-mode posteriors
dla_cat = cross_reference_lls(dla_cat, lls_cat,
                              dz_match=0.01, lls_threshold=20.3)

# Optional clean DLA catalog — drop both classes of likely-spurious entries
clean = dla_cat[~dla_cat["LYBETA_FLAG"] & ~dla_cat["LLS_DOWNGRADE_FLAG"]]
clean.write("dlacat-iron-main-dark-clean.fits", overwrite=True)
```

## What the flag columns mean

| Column                | Type    | Meaning                                                        |
|-----------------------|--------:|----------------------------------------------------------------|
| `LYBETA_FLAG`         | bool    | True ⇒ this row is the Lyβ-shifted apparent z of another DLA on the same LOS, with strictly higher NHI. Most likely a spurious second DLA. |
| `LYBETA_PARENT_TID`   | int64   | TARGETID of the parent DLA we matched to (= same TARGETID; useful for joining/debugging). |
| `LYBETA_PARENT_Z`     | float   | z of the parent DLA. The "apparent z" is `0.844·(1+parent_z) − 1`. |
| `LLS_LOG_NHI`         | float   | MAP log NHI from LLS-mode at the matched (TARGETID, z). NaN if no LLS-mode match. |
| `LLS_P_ABSORBER`      | float   | p(absorber) from LLS-mode at the matched z. NaN if no match. |
| `LLS_DOWNGRADE_FLAG`  | bool    | True ⇒ LLS-mode prefers log NHI < 20.3 with non-trivial confidence ⇒ this DLA is more likely a sub-DLA/LLS. |

Both helpers are **conservative**: they only flag, never delete. The
science user decides whether to filter. Recommended downstream policy
for a high-purity catalog is `~LYBETA_FLAG & ~LLS_DOWNGRADE_FLAG`.
For population statistics (CDDF, Ω_HI), the user has historically
preferred to keep all rows and propagate the full posterior — see the
project documentation under `CDDF_analysis/`.

## Tunable thresholds

| Parameter                        | Default | Notes                                                                |
|----------------------------------|--------:|----------------------------------------------------------------------|
| `flag_lybeta(dz_match=)`         | 0.005   | ≈ 7 px at DESI dlambda=0.15 Å. Tighter is safer; looser catches more. |
| `flag_lybeta(require_higher_nhi_parent=)` | True | Lyβ confusion makes the *spurious* DLA NHI lower than the parent DLA — keep True unless studying degenerate cases. |
| `cross_reference_lls(dz_match=)` | 0.01    | LLS posteriors can be slightly offset from DLA posteriors; loose match is fine. |
| `cross_reference_lls(lls_threshold=)` | 20.3 | Conventional DLA boundary. Below this, LLS-mode disagrees with DLA-mode. |

## Hypothesis tests run against this code

- `tests/test_smoke_target_contamination.py` — verifies the truth-data
  on a known smoke target (sanity guard only).
- `tests/test_lyb_veto.py` — synthetic test: inject a parent DLA at
  z=2.7 and a "child" at z = 0.844 × 3.7 − 1 = 2.122; verify the helper
  flags the child only.
- A real-data analysis on `out/smoke/batch/eboss_filter0_n10000`
  (FILTER=0 produces ~50 % spurious rate) showed **34.8 % of spurious
  detections are explained by Lyβ-of-a-real-DLA** at this dz_match.
  The remaining 65 % are sub-threshold absorbers (handled by the LLS
  cross-reference) or genuine multi-DLA degeneracies.

## Why this is needed even though Voigt already models Lyβ

Each absorber in the multi-DLA model contributes its own Lyα + Lyβ +
Lyγ optical depth via `voigt_v2.voigt_absorption(num_lines≥2)`. So a
single DLA at z=2.7 already produces the correct Lyβ trough at z_app=
2.12 in the model. The multi-DLA confusion arises in the *evidence
integral over candidate (z, NHI) for a SECOND absorber*: the QMC
samples at z_app ≈ 2.12 fit the data twice (DLA1 already contributed
absorption there), so the fitter shrinks the second NHI to ~20.3 and
the joint M_DLA(2) likelihood at that sample is comparable to the
clean M_DLA(1) likelihood. Marginalised over the (z2, NHI2) prior,
M_DLA(2) can edge out M_DLA(1) by a small Bayes factor. FILTER=1
(truncated sampling) suppresses this somewhat (see the FILTER=1
algorithm notes), but does not eliminate it — hence the catalog-time
veto.
