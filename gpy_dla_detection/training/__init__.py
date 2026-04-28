"""Streamlined GP training pipeline.

This subpackage is an alternative to ``gpy_dla_detection.objective`` and
``gpy_dla_detection.learn_qso_model``. The legacy modules are byte-stable
to the DR16Q-public MATLAB reference (Layer 4 parity test) and remain
untouched; this subpackage exists for new work where the per-spectrum
Python loop, manual gradient accumulation, and per-epoch I/O overhead
are not desirable.

Modules:
- ``objective_v2``: vectorized NLL across a batch of spectra, autograd-friendly
- (more to come: dataset, trainer, profiling helpers)
"""
