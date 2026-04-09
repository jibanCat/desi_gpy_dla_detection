"""
generate_samples.py — QMC sample generation for DLA/sub-DLA/LLS inference.

Overview
--------
Generates Quasi-Monte Carlo (QMC) sample grids used to numerically integrate
DLA model evidence:

    p(D | k-absorber model) ≈ (1/N) Σᵢ p(D | z_i, log_NHI_i)

where (z_i, log_NHI_i) are drawn from an absorber prior.

This module converts the logic in ``notebooks/GenerateSamples_PW14.ipynb``
into a reusable, testable, and CLI-callable function.

Run modes and recommended parameter ranges
------------------------------------------
  DLA run   (multi):  log NHI ∈ [20.3, 23]  — use dla_samples_a03.mat (Ho+2020)
  Sub-DLA   (single): log NHI ∈ [19, 20.3]  — generate with this module
  LLS       (single): log NHI ∈ [17.2, 19]  — generate with this module

Column-density prior
--------------------
The prior on log₁₀(N_HI) is a mixture of:
  - The Prochaska et al. (2014) CDDF (arxiv 1402.0548) — a monotonic cubic
    Hermite spline fit to Table 2 (Spline Model, Figure 7)
  - A uniform component over the requested NHI range

  p(logN) = alpha * p_PW14(logN) + (1 - alpha) * Uniform(min, max)

where alpha=0.97 by default (97% data-driven, 3% uniform).

Redshift offsets
----------------
Redshift samples are generated as uniform [0, 1] Halton offsets.
At inference time, these are scaled to the actual search window:

  z_DLA_i = z_min + (z_max - z_min) * offset_i

Output file format
------------------
An HDF5 file (MATLAB v7.3 compatible) with the following datasets, each of
shape (N, 1) for compatibility with the existing pipeline's h5py loading code:

  log_nhi_samples      — log₁₀(N_HI) samples, shape (N, 1)
  nhi_samples          — 10^log_nhi_samples, shape (N, 1)
  offset_samples       — z_DLA Halton offsets in [0, 1], shape (N, 1)
  alpha                — mixture weight, shape (1, 1)
  fit_min_log_nhi      — lower bound of NHI range, shape (1, 1)
  fit_max_log_nhi      — upper bound of NHI range, shape (1, 1)
  uniform_min_log_nhi  — same as fit_min_log_nhi, shape (1, 1)
  uniform_max_log_nhi  — same as fit_max_log_nhi, shape (1, 1)

Usage (Python)
--------------
    from gpy_dla_detection.generate_samples import generate_pw14_samples

    # Sub-DLA run
    generate_pw14_samples(
        num_samples=50000,
        min_log_nhi=19.0,
        max_log_nhi=20.3,
        output_path="subdla_samples_pw14.mat",
    )

    # LLS run
    generate_pw14_samples(
        num_samples=50000,
        min_log_nhi=17.2,
        max_log_nhi=19.0,
        output_path="lls_samples_pw14.mat",
    )

Usage (CLI)
-----------
    python -m gpy_dla_detection.generate_samples \\
        --mode subdla --output subdla_samples_pw14.mat

    python -m gpy_dla_detection.generate_samples \\
        --mode lls --output lls_samples_pw14.mat

    python -m gpy_dla_detection.generate_samples \\
        --min-log-nhi 19.0 --max-log-nhi 20.3 --num-samples 50000 \\
        --output custom_samples.mat

References
----------
Prochaska et al. (2014) https://arxiv.org/abs/1402.0548
Ho, Bird & Garnett (2020) https://arxiv.org/abs/2003.11036
"""

import argparse
import numpy as np
import h5py
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.stats.qmc import Halton


# ---------------------------------------------------------------------------
# Prochaska+2014 CDDF spline (Table 2, Spline Model — Figure 7)
# ---------------------------------------------------------------------------
# Node locations in log10 N_HI (cm^-2)
_LOGNHI_NODES = np.array([12.0, 15.0, 17.0, 18.0, 20.0, 21.0, 21.5, 22.0])
# Corresponding log10 f(N_HI, X)
_LOGF_NODES   = np.array([-9.72, -14.41, -17.94, -19.39,
                           -21.28, -22.82, -23.95, -25.50])

_cddf_spline = PchipInterpolator(_LOGNHI_NODES, _LOGF_NODES)


def f_pw14(log_nhi: np.ndarray) -> np.ndarray:
    """
    Prochaska+2014 CDDF value f(N_HI, X) at the given log10 N_HI.

    Uses a monotonic cubic Hermite spline (PchipInterpolator) fit to
    Table 2 (Spline Model) of Prochaska et al. (2014), arxiv 1402.0548.

    Parameters
    ----------
    log_nhi : array_like
        log10(N_HI) values [cm^-2].

    Returns
    -------
    np.ndarray
        f(N_HI, X) values in linear units.
    """
    log_nhi = np.asarray(log_nhi)
    log_nhi_clip = np.clip(log_nhi, _LOGNHI_NODES[0], _LOGNHI_NODES[-1])
    return 10.0 ** _cddf_spline(log_nhi_clip)


def build_pw14_prior(
    min_log_nhi: float,
    max_log_nhi: float,
    alpha: float = 0.97,
    n_grid: int = 50_000,
):
    """
    Build a normalized prior PDF on log10(N_HI) using the Prochaska+2014 CDDF.

    The prior is a mixture:
        p(logN) = alpha * p_PW14(logN) + (1 - alpha) * Uniform(min, max)

    Parameters
    ----------
    min_log_nhi : float
        Lower bound on log10(N_HI).
    max_log_nhi : float
        Upper bound on log10(N_HI).
    alpha : float, optional
        Mixture weight for the PW14 component (default 0.97).
    n_grid : int, optional
        Number of grid points for the numerical CDF (default 50000).

    Returns
    -------
    pdf : callable
        Normalized prior PDF, callable as ``pdf(log_nhi)``.
    inverse_cdf : callable
        Inverse CDF, callable as ``inverse_cdf(u)`` for u ∈ [0, 1].
    """

    def _unnormalized_pw14_pdf(log_nhi):
        """p(logN) ∝ f(N) × N × ln(10)"""
        log_nhi = np.asarray(log_nhi)
        N = 10.0 ** log_nhi
        return f_pw14(log_nhi) * N * np.log(10.0)

    # Normalize the PW14 part over [min_log_nhi, max_log_nhi]
    Z_pw14, _ = quad(_unnormalized_pw14_pdf, min_log_nhi, max_log_nhi)

    def pdf(log_nhi):
        log_nhi = np.asarray(log_nhi)
        pw = _unnormalized_pw14_pdf(log_nhi) / Z_pw14
        width = max_log_nhi - min_log_nhi
        uniform = ((log_nhi >= min_log_nhi) & (log_nhi <= max_log_nhi)) / width
        return alpha * pw + (1.0 - alpha) * uniform

    # Build inverse CDF via dense grid + interpolation
    x = np.linspace(min_log_nhi, max_log_nhi, n_grid)
    y = pdf(x)
    cdf = np.cumsum(y)
    cdf /= cdf[-1]  # normalize to [0, 1]
    inverse_cdf = interp1d(
        cdf, x, bounds_error=False, assume_sorted=True,
        fill_value=(x[0], x[-1])
    )

    return pdf, inverse_cdf


def generate_pw14_samples(
    num_samples: int = 50_000,
    min_log_nhi: float = 19.0,
    max_log_nhi: float = 20.3,
    alpha: float = 0.97,
    output_path: str = None,
    seed: int = 42,
) -> dict:
    """
    Generate QMC DLA/sub-DLA/LLS parameter samples using the Prochaska+2014 prior.

    Samples (z-offsets, log_NHI) are generated using a 2D Halton sequence
    (scrambled, for better uniformity).  The z-offset dimension gives the
    position of the absorber within the search window; the NHI dimension is
    mapped through the inverse CDF of the PW14 prior mixture.

    Parameters
    ----------
    num_samples : int
        Number of QMC samples to generate (default 50000).
    min_log_nhi : float
        Lower bound of log10(N_HI) prior range.
        Sub-DLA: 19.0, LLS: 17.2, DLA: 20.3
    max_log_nhi : float
        Upper bound of log10(N_HI) prior range.
        Sub-DLA: 20.3, LLS: 19.0, DLA: 23.0
    alpha : float
        Mixture weight for the PW14 component (default 0.97).
    output_path : str, optional
        If provided, write the samples to this HDF5 ``.mat`` file.
        The format is compatible with the existing pipeline's loading code
        in DLASamplesMAT (dla_samples.py).
    seed : int
        Random seed for the Halton scrambling (default 42).

    Returns
    -------
    dict with keys:
        offset_samples  — z-offset QMC samples, shape (N,)
        log_nhi_samples — log10(N_HI) samples, shape (N,)
        nhi_samples     — 10^log_nhi_samples, shape (N,)
        alpha           — scalar mixture weight
        min_log_nhi     — lower bound used
        max_log_nhi     — upper bound used
    """
    # 2D Halton sequence: dim 0 → NHI, dim 1 → z-offset
    # Pass seed directly to the Halton sampler (not np.random.seed, which has no effect here)
    sampler = Halton(d=2, scramble=True, seed=seed)
    halton = sampler.random(num_samples)  # shape (N, 2)

    _, inverse_cdf = build_pw14_prior(min_log_nhi, max_log_nhi, alpha=alpha)

    log_nhi_samples = inverse_cdf(halton[:, 0])
    offset_samples  = halton[:, 1]

    samples = {
        "offset_samples":     offset_samples,
        "log_nhi_samples":    log_nhi_samples,
        "nhi_samples":        10.0 ** log_nhi_samples,
        "alpha":              float(alpha),
        "min_log_nhi":        float(min_log_nhi),
        "max_log_nhi":        float(max_log_nhi),
    }

    if output_path is not None:
        save_samples_to_mat(samples, output_path)

    return samples


def save_samples_to_mat(samples: dict, output_path: str) -> None:
    """
    Save QMC samples to an HDF5 file in the format expected by DLASamplesMAT.

    The output format matches the existing ``dla_samples_a03.mat`` file so
    that the inference pipeline (DLASamplesMAT) can load it without changes.
    All datasets are stored with shape (N, 1) or (1, 1) to match the
    MATLAB convention used by the original files.

    Parameters
    ----------
    samples : dict
        Output of ``generate_pw14_samples``.
    output_path : str
        Path to the output ``.mat`` / HDF5 file.
    """
    N = len(samples["offset_samples"])
    with h5py.File(output_path, "w") as f:
        f.create_dataset("log_nhi_samples",     data=samples["log_nhi_samples"].reshape(N, 1))
        f.create_dataset("nhi_samples",         data=samples["nhi_samples"].reshape(N, 1))
        f.create_dataset("offset_samples",      data=samples["offset_samples"].reshape(N, 1))
        f.create_dataset("alpha",               data=np.array([[samples["alpha"]]]))
        f.create_dataset("fit_min_log_nhi",     data=np.array([[samples["min_log_nhi"]]]))
        f.create_dataset("fit_max_log_nhi",     data=np.array([[samples["max_log_nhi"]]]))
        f.create_dataset("uniform_min_log_nhi", data=np.array([[samples["min_log_nhi"]]]))
        f.create_dataset("uniform_max_log_nhi", data=np.array([[samples["max_log_nhi"]]]))
    print(f"Saved {N} samples to {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

# Preset configurations for common run modes
_PRESETS = {
    "subdla": {"min_log_nhi": 19.0,  "max_log_nhi": 20.3, "label": "sub-DLA"},
    "lls":    {"min_log_nhi": 17.2,  "max_log_nhi": 19.0, "label": "LLS"},
    "dla":    {"min_log_nhi": 20.3,  "max_log_nhi": 23.0, "label": "DLA"},
}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate QMC absorber parameter samples using the Prochaska+2014 "
            "CDDF prior.  Outputs an HDF5 file compatible with DLASamplesMAT."
        )
    )
    parser.add_argument(
        "--mode",
        choices=list(_PRESETS.keys()),
        default=None,
        help=(
            "Preset run mode: 'subdla' (logNHI 19–20.3), "
            "'lls' (logNHI 17.2–19), or 'dla' (logNHI 20.3–23).  "
            "Overrides --min-log-nhi and --max-log-nhi."
        ),
    )
    parser.add_argument(
        "--min-log-nhi", type=float, default=19.0,
        help="Lower bound of log10(N_HI) range (default 19.0).",
    )
    parser.add_argument(
        "--max-log-nhi", type=float, default=20.3,
        help="Upper bound of log10(N_HI) range (default 20.3).",
    )
    parser.add_argument(
        "--num-samples", type=int, default=50_000,
        help="Number of QMC samples to generate (default 50000).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.97,
        help="Mixture weight for PW14 component (default 0.97).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for Halton scrambling (default 42).",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output HDF5 file path (e.g. subdla_samples_pw14.mat).",
    )
    args = parser.parse_args()

    if args.mode is not None:
        preset = _PRESETS[args.mode]
        min_log_nhi = preset["min_log_nhi"]
        max_log_nhi = preset["max_log_nhi"]
        print(f"Using preset '{args.mode}' ({preset['label']}): "
              f"log NHI ∈ [{min_log_nhi}, {max_log_nhi}]")
    else:
        min_log_nhi = args.min_log_nhi
        max_log_nhi = args.max_log_nhi

    generate_pw14_samples(
        num_samples=args.num_samples,
        min_log_nhi=min_log_nhi,
        max_log_nhi=max_log_nhi,
        alpha=args.alpha,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
