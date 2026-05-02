"""Small shared helpers for the v1 (legacy) and v2 trained-model loaders.

Currently houses the per-spectrum normalization-window override path
that was duplicated across ``null_gp.NullGPMAT``, ``dla_gp.DLAGPMAT``
and ``subdla_gp.SubDLAGPMAT``. Keeping it here so a future fourth GP
class (e.g. a BAL-only variant) doesn't drift.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

from .set_parameters import Parameters


def apply_normalization_from_h5(
    params: Parameters,
    learned: Any,
    *,
    verbose: bool = True,
) -> None:
    """Sync ``params.normalization_min/max_lambda`` to whatever the v2
    trained model recorded in its ``.h5``.

    Three cases:
      * Field absent (legacy v1 ``.h5``) — leave ``params`` untouched.
      * Field is finite — mutate ``params`` in place to match the
        training window. Print a one-line note if it actually changed
        (skipped silently when the request matches the existing window).
      * Field is NaN — model was trained with ``--no-normalize``. Issue
        a ``RuntimeWarning`` telling the caller they need to set
        ``params.normalization_*_lambda`` themselves; do NOT mutate.

    ``learned`` is the open ``h5py.File`` handle returned by
    ``h5py.File(learned_file, "r")``.
    """
    if "normalization_min_lambda" not in learned:
        return  # legacy v1 .h5 — silent fall-through

    new_min = float(learned["normalization_min_lambda"][()])
    new_max = float(learned["normalization_max_lambda"][()])

    if math.isnan(new_min) or math.isnan(new_max):
        warnings.warn(
            f"Loaded GP model was trained with --no-normalize "
            f"(normalization_*_lambda is NaN in the .h5). Inference "
            f"will still normalize input spectra using the current "
            f"params window [{params.normalization_min_lambda:.1f}, "
            f"{params.normalization_max_lambda:.1f}] Å, which likely "
            f"does NOT match the model's training setup. Set "
            f"params.normalization_min_lambda / _max_lambda explicitly "
            f"before set_data() if you want to control this.",
            RuntimeWarning,
            stacklevel=3,
        )
        return

    if (new_min != params.normalization_min_lambda
        or new_max != params.normalization_max_lambda):
        if verbose:
            print(f"  → overriding params.normalization window from "
                  f".h5: [{params.normalization_min_lambda:.1f}, "
                  f"{params.normalization_max_lambda:.1f}] Å → "
                  f"[{new_min:.1f}, {new_max:.1f}] Å")
        params.normalization_min_lambda = new_min
        params.normalization_max_lambda = new_max
