#!/usr/bin/env python
"""Streamlined GP-DLA training entry point.

Replaces the legacy ``desi_learn_qsos_model.py`` for new training runs.
Math is byte-stable equivalent to the legacy code via autograd
(parity-tested in ``tests/test_objective_v2_parity.py``); under the hood
this entry point uses ``gpy_dla_detection.training.objective_v2``
(vectorized) and ``trainer_v2`` (clean Adam loop with autograd
backward), which Layer 3 measured at ~14× faster than the legacy loop
on CPU.

Usage example (NERSC LOA training)::

    python train_gp.py \\
        --preloaded-file /pscratch/.../preload-loa-gpdla-*/gp_interp_trainset.h5 \\
        --catalog-file /pscratch/.../data/loa/gp_trainset_loa.fits \\
        --z-min 2.5 --z-max 4.25 \\
        --max-spectra 300000 \\
        --num-pca-components 30 \\
        --num-epochs 800 \\
        --batch-size 12500 \\
        --learning-rate 0.005 \\
        --output-dir learnlogs_v2/

Defaults match the production NERSC config in
``slurm_train/submit_train_gp_loa_full.sh``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

# Make repo importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpy_dla_detection.training.dataset import (  # noqa: E402
    load_preprocessed_h5, load_targetids_from_catalog,
)
from gpy_dla_detection.training.model_v2 import GPModelV2  # noqa: E402
from gpy_dla_detection.training.trainer_v2 import TrainConfig, train  # noqa: E402
from gpy_dla_detection.voigt import (  # noqa: E402
    transition_wavelengths as TRANSITION_WAVELENGTHS_NP,
    oscillator_strengths as OSCILLATOR_STRENGTHS_NP,
)


def _initial_M_from_pca(centered_fluxes: np.ndarray, k: int) -> np.ndarray:
    """Initial M = top-k PCA components × √eigenvalues.

    Mirrors v1 MATLAB ``learn_qso_model.m`` (lines 197-216) and the
    legacy v1 Python ``learn_qso_model.py:fill_nan_with_median``:

      - For each spectrum (ROW), replace NaN pixels with that
        spectrum's median flux (NOT per-pixel population median —
        per-pixel fill artificially aligns NaN-padded spectra).
      - PCA on the filled centered fluxes.
      - M = top-k components × sqrt(eigenvalues).

    Important: the input ``centered_fluxes`` MUST be the post-pipeline
    fluxes (i.e. after normalize → de-forest → center, as produced by
    ``load_preprocessed_h5``). Running PCA on raw un-normalized
    trainset.h5 fluxes gives a misleading rank-1 basis because the
    per-spectrum amplitude variance dominates.
    """
    fluxes = centered_fluxes.copy()
    n_quasars, n_pix = fluxes.shape
    for i in range(n_quasars):
        row = fluxes[i, :]
        finite = np.isfinite(row)
        if finite.any():
            row[~finite] = np.nanmedian(row[finite])
        else:
            row[:] = 0.0
    pca = PCA(n_components=k)
    pca.fit(fluxes)
    coefficients = pca.components_.T          # (n_pix, k)
    eigvals = pca.explained_variance_          # (k,)
    print(f"[pca-init] eigvals top-5 = {eigvals[:5]}")
    print(f"[pca-init] eff_rank = trace_MMT/max_eig = "
          f"{float((coefficients**2 * eigvals).sum()) / float(eigvals[0]):.2f}  "
          f"(>5 = healthy, ~1 = collapsed)")
    return (coefficients * np.sqrt(eigvals)).astype(np.float32)


def _initial_log_omega(centered_fluxes: np.ndarray, default: float = 0.1) -> np.ndarray:
    """Initial log_omega = log(per-pixel std).

    Pixels where all spectra are NaN-padded (e.g. rest-grid edges) yield
    nanstd → NaN → log(NaN) → NaN. .clip(min=...) does NOT mask NaN, so
    we explicitly substitute the median of finite per-pixel std (and fall
    back to ``default`` if no pixel has a finite std).
    """
    per_pix_std = np.nanstd(centered_fluxes, axis=0)
    # Replace inf and 0 (or near-0) with NaN so they're treated uniformly
    # below.
    per_pix_std = np.where(np.isfinite(per_pix_std) & (per_pix_std > 1e-12),
                            per_pix_std, np.nan)
    n_bad = int(np.isnan(per_pix_std).sum())
    if n_bad > 0:
        finite = per_pix_std[np.isfinite(per_pix_std)]
        fill = float(np.median(finite)) if finite.size else default
        per_pix_std = np.where(np.isnan(per_pix_std), fill, per_pix_std)
        print(f"[init] log_omega: {n_bad}/{len(per_pix_std)} pixels had "
              f"non-finite/zero std; filled with median={fill:.3e}")
    return np.log(per_pix_std).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--preloaded-file", required=True,
                   help="Path to gp_interp_trainset.h5 produced by the preload pipeline.")
    p.add_argument("--catalog-file", default=None,
                   help="Optional FITS catalog with TARGETID column (legacy nonBAL-nonDLA filter).")
    # Filtering
    p.add_argument("--z-min", type=float, default=2.5)
    p.add_argument("--z-max", type=float, default=4.25)
    p.add_argument("--min-snr", type=float, default=0.0)
    p.add_argument("--max-spectra", type=int, default=300_000)
    # Model
    p.add_argument("--num-pca-components", type=int, default=30)
    # Optimization
    p.add_argument("--num-epochs", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=12_500)
    p.add_argument("--learning-rate", type=float, default=5e-3)
    p.add_argument("--num-forest-lines", type=int, default=3)
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "step", "none"])
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="L2 weight decay on optimizer (default 0.0; 1e-6 to "
                        "constrain M growth and prevent rank-1 collapse).")
    p.add_argument("--save-every", type=int, default=10)
    # Per-spectrum median normalization in [norm_min_lambda, norm_max_lambda]
    # is ON by default (Garnett+2017 [1310, 1325] Å rest). v2 trainsets do
    # NOT include normalization; this step has to happen at training-load
    # time for now. Pass --no-normalize to disable for ablation.
    p.add_argument("--no-normalize", dest="apply_normalize",
                   action="store_false", default=True,
                   help="Disable per-spectrum median normalization "
                        "(reproduces the buggy pre-2026-05-01 behavior).")
    p.add_argument("--norm-min-lambda", type=float, default=1310.0,
                   help="Per-spectrum normalization window min Å rest. "
                        "Default Garnett+2017 [1310, 1325].")
    p.add_argument("--norm-max-lambda", type=float, default=1325.0)
    p.add_argument("--min-valid-pixels-lyman", type=int, default=200,
                   help="Drop spectra with fewer than this many valid pixels "
                        "in the Lyman-modelling range [911, 1216] Å rest "
                        "(v1 preload_qsos.m bit 3 equivalent). Set to 0 to "
                        "disable the filter.")
    p.add_argument("--lyman-min-lambda", type=float, default=911.0,
                   help="Lower bound of the Lyman science range used for "
                        "the min-valid-pixels filter.")
    p.add_argument("--lyman-max-lambda", type=float, default=1216.0,
                   help="Upper bound of the Lyman science range used for "
                        "the min-valid-pixels filter.")
    # Y1 (Turner+2024) Gaussian prior on (τ₀, β) is ON by default; pass
    # --no-y1-prior to drop it (e.g. for ablation studies).
    p.add_argument("--no-y1-prior", dest="apply_y1_prior",
                   action="store_false", default=True)
    # I/O
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None,
                   help="cuda or cpu; auto-detect if omitted")
    args = p.parse_args()

    # 1) Load training data.
    catalog_targetids = None
    if args.catalog_file:
        catalog_targetids = load_targetids_from_catalog(args.catalog_file)
        print(f"[main] catalog filter: {len(catalog_targetids)} TARGETIDs")

    ts = load_preprocessed_h5(
        args.preloaded_file,
        z_min=args.z_min, z_max=args.z_max, min_snr=args.min_snr,
        max_spectra=args.max_spectra,
        catalog_targetids=catalog_targetids,
        apply_normalize=args.apply_normalize,
        norm_min_lambda=args.norm_min_lambda,
        norm_max_lambda=args.norm_max_lambda,
        min_valid_pixels_lyman=args.min_valid_pixels_lyman,
        lyman_min_lambda=args.lyman_min_lambda,
        lyman_max_lambda=args.lyman_max_lambda,
        dtype=torch.float32,
    )
    print(f"[main] loaded {ts.n_spectra} spectra × {ts.n_pix} pixels")

    # 2) Initialise PCA-based M and log_omega from the loaded fluxes.
    centered_fluxes_np = ts.fluxes.numpy()
    initial_M = _initial_M_from_pca(centered_fluxes_np, args.num_pca_components)
    initial_log_omega = _initial_log_omega(centered_fluxes_np)
    print(f"[main] PCA init M: shape {initial_M.shape}, σ {initial_M.std():.3e}")

    # 3) Build the model.
    # Pass rest_wavelengths and mu through so the saved H5 includes the
    # metadata the legacy inference loader expects.
    #
    # Carry the normalization region forward so save_h5_model writes it
    # into the .h5 → inference picks it up automatically (see
    # null_gp.NullGPMAT.__init__ for the read-side). When --no-normalize
    # was passed, write NaN as a sentinel: the inference loader will
    # detect NaN, skip the params mutation, and warn the user that the
    # model was trained un-normalized so they need to set the
    # normalization region explicitly. Distinguishes from legacy v1 .h5
    # files (no fields at all) which still fall back silently.
    if args.apply_normalize:
        norm_min_for_model = args.norm_min_lambda
        norm_max_for_model = args.norm_max_lambda
    else:
        norm_min_for_model = float("nan")
        norm_max_for_model = float("nan")
    model = GPModelV2(
        num_pixels=ts.n_pix, k=args.num_pca_components,
        init_M=initial_M, init_log_omega=initial_log_omega,
        rest_wavelengths=ts.rest_wavelengths,
        mu=ts.mu,  # may be None if --no-center
        normalization_min_lambda=norm_min_for_model,
        normalization_max_lambda=norm_max_for_model,
    )

    # Sanity: every initial parameter must be finite, otherwise the
    # entire training run produces NaN from epoch 0.
    bad = []
    for name, p in model.named_parameters():
        finite = torch.isfinite(p)
        if not bool(finite.all().item()):
            bad.append((name, int((~finite).sum().item()), int(p.numel())))
    if bad:
        for name, n_bad, n_total in bad:
            print(f"[init] ERROR: {name} has {n_bad}/{n_total} non-finite values")
        sys.exit("[init] non-finite parameters at initialisation; aborting")

    # 4) Train.
    cfg = TrainConfig(
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_forest_lines=args.num_forest_lines,
        scheduler=args.scheduler,
        weight_decay=args.weight_decay,
        save_every=args.save_every,
        apply_y1_prior=args.apply_y1_prior,
        seed=args.seed,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    transition_wavelengths = torch.tensor(TRANSITION_WAVELENGTHS_NP, dtype=torch.float32)
    oscillator_strengths = torch.tensor(OSCILLATOR_STRENGTHS_NP, dtype=torch.float32)

    train(
        model,
        ts.fluxes, ts.lya_1pzs, ts.noise_variances, ts.z_qsos,
        transition_wavelengths, oscillator_strengths,
        Path(args.output_dir), cfg,
    )

    print("[main] training complete.")


if __name__ == "__main__":
    main()
