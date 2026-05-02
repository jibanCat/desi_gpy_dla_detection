"""Minimal GP model — same five learnable parameters as the legacy
``GaussianProcessModel``, but without the per-batch save / per-batch
plot machinery that hurt training throughput in Layer 3.

Usage::

    from gpy_dla_detection.training.model_v2 import GPModelV2
    model = GPModelV2(num_pixels=600, k=30,
                     init_M=initial_M, init_log_omega=initial_log_omega)

The forward pass is intentionally NOT defined here — call
``vectorized_nll(...)`` from ``objective_v2`` directly with the model's
parameters. This keeps the model class a pure parameter container and
the training loop fully transparent.
"""

from __future__ import annotations

from typing import Optional

import math
import numpy as np
import torch
from torch import nn


class GPModelV2(nn.Module):
    """Pure parameter container; same surface as the legacy model.

    Parameters
    ----------
    num_pixels : int
        Number of rest-frame wavelength pixels (n in the GP math).
    k : int
        Low-rank emission basis dimension.
    init_M : np.ndarray | torch.Tensor of shape (n, k), optional
        Initial M (e.g. PCA components × √eigvals on training data).
        If None, initialise to small random.
    init_log_omega : np.ndarray | torch.Tensor of shape (n,), optional
        Initial log_omega; if None, initialise to log(0.1).
    init_log_c_0 : float
        Initial log c_0 (default log(0.1)).
    init_log_tau_0 : float
        Initial log τ_0 (default log(0.00246), DESI Y1).
    init_log_beta : float
        Initial log β (default log(3.62), DESI Y1).
    dtype : torch.dtype
        Defaults to float32 to match the legacy production trainer.

    Notes
    -----
    The DR16Q-public MATLAB used ``initial_c_0 = 0.1, tau_0 = 0.00554,
    beta = 3.182`` (Kamble+2019). The DESI Y3 production code uses Turner+2024
    values. Defaults here are Turner+2024 to match the current production
    pipeline.
    """

    def __init__(
        self,
        num_pixels: int,
        k: int,
        *,
        init_M: Optional[torch.Tensor] = None,
        init_log_omega: Optional[torch.Tensor] = None,
        init_log_c_0: float = math.log(0.1),
        init_log_tau_0: float = math.log(0.00246),
        init_log_beta: float = math.log(3.62),
        rest_wavelengths: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        max_noise_variance: float = 9.0,
        normalization_min_lambda: float = 1310.0,
        normalization_max_lambda: float = 1325.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.num_pixels = num_pixels
        self.k = k

        # Non-trainable metadata kept on the model so the H5 saver can dump
        # everything the legacy inference loader expects (rest_wavelengths,
        # mu, max_noise_variance).
        if rest_wavelengths is None:
            rest_wavelengths = torch.linspace(911.0, 1216.0, num_pixels, dtype=dtype)
        elif isinstance(rest_wavelengths, np.ndarray):
            rest_wavelengths = torch.from_numpy(rest_wavelengths).to(dtype)
        else:
            rest_wavelengths = rest_wavelengths.to(dtype)
        self.register_buffer("rest_wavelengths", rest_wavelengths.clone().detach())

        if mu is None:
            mu = torch.zeros(num_pixels, dtype=dtype)
        elif isinstance(mu, np.ndarray):
            mu = torch.from_numpy(mu).to(dtype)
        else:
            mu = mu.to(dtype)
        self.register_buffer("mu", mu.clone().detach())

        self.max_noise_variance = float(max_noise_variance)
        # Normalization region used at PRELOAD/TRAINING time. Stored on the
        # model so the inference pipeline can pick up the right window when
        # loading this trained model. Default [1310, 1325] matches the
        # Garnett+2017 (arXiv:1605.04460) recommendation; v1 production
        # used [1425, 1475] but v2 trainsets don't extend that red.
        self.normalization_min_lambda = float(normalization_min_lambda)
        self.normalization_max_lambda = float(normalization_max_lambda)

        if init_M is None:
            init_M = torch.randn(num_pixels, k, dtype=dtype) * 0.05
        elif isinstance(init_M, np.ndarray):
            init_M = torch.from_numpy(init_M).to(dtype)
        else:
            init_M = init_M.to(dtype)

        if init_log_omega is None:
            init_log_omega = torch.full((num_pixels,), math.log(0.1), dtype=dtype)
        elif isinstance(init_log_omega, np.ndarray):
            init_log_omega = torch.from_numpy(init_log_omega).to(dtype)
        else:
            init_log_omega = init_log_omega.to(dtype)

        self.M = nn.Parameter(init_M.clone().detach())
        self.log_omega = nn.Parameter(init_log_omega.clone().detach())
        self.log_c_0 = nn.Parameter(torch.tensor(init_log_c_0, dtype=dtype))
        self.log_tau_0 = nn.Parameter(torch.tensor(init_log_tau_0, dtype=dtype))
        self.log_beta = nn.Parameter(torch.tensor(init_log_beta, dtype=dtype))

    def state_dict_for_h5(self):
        """Flat dict suitable for h5py dump.

        Keys mirror the legacy ``save_h5_file`` output layout so that
        downstream code (e.g. the inference pipeline reading
        ``learnlogs/model_epoch_NNN.h5``) can load v2-trained models
        unchanged.

        Required by the legacy inference loader (gpy_dla_detection/dla_gp.py
        line 1042-1056 and null_gp.py line 453-457):
            rest_wavelengths, mu, M, log_omega, log_c_0, log_tau_0, log_beta
        """
        return {
            "M": self.M.detach().cpu().numpy(),
            "log_omega": self.log_omega.detach().cpu().numpy(),
            "log_c_0": float(self.log_c_0.detach().cpu().item()),
            "log_tau_0": float(self.log_tau_0.detach().cpu().item()),
            "log_beta": float(self.log_beta.detach().cpu().item()),
            "rest_wavelengths": self.rest_wavelengths.detach().cpu().numpy(),
            "mu": self.mu.detach().cpu().numpy(),
            "max_noise_variance": float(self.max_noise_variance),
            "normalization_min_lambda": float(self.normalization_min_lambda),
            "normalization_max_lambda": float(self.normalization_max_lambda),
            "num_pixels": self.num_pixels,
            "k": self.k,
        }
