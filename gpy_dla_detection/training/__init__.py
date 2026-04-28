"""Streamlined GP training pipeline.

This subpackage is an alternative to ``gpy_dla_detection.objective`` and
``gpy_dla_detection.learn_qso_model``. The legacy modules are byte-stable
to the DR16Q-public MATLAB reference (Layer 4 parity test) and remain
untouched; this subpackage exists for new work where the per-spectrum
Python loop, manual gradient accumulation, and per-epoch I/O overhead
are not desirable.

Modules:
- ``dataset``: load preprocessed gp_interp_trainset.h5 (legacy + newer
  schemas) and apply train-time mask + de-forest + center.
- ``model_v2``: pure-parameter ``GPModelV2`` container with the same
  five learnable tensors as the legacy GP, plus rest_wavelengths/mu/
  max_noise_variance buffers for inference-loader compatibility.
- ``objective_v2``: vectorized batched NLL with autograd backward
  and optional Y1 (Turner+2024) Gaussian prior on (τ₀, β).
- ``trainer_v2``: streamlined Adam loop with per-N-epoch checkpointing,
  resume support, and legacy-compatible H5 saves.
"""
