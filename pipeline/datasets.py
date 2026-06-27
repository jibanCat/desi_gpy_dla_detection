"""Per-dataset primary-input bundles for the HBI results-store pipeline (task 3 of
``pipeline/IMPLEMENTATION_PLAN.md``).

A ``DatasetInputs`` is the immutable primary-input descriptor for one dataset: the
GP-DLA absorber catalog directory, the injected/placeholder truth, the BAL catalog,
the mock catalog dir (snr_cat/zcat), and the privacy class (``"mock"`` vs
``"real-LOA"``). The defaults are taken VERBATIM from the producers' current module
constants so a pipeline run is byte-identical to running the producer by hand:

  * ``2lpt0``    — the 2LPT-0 mock baseline. Catalog/truth/bal come from
    ``CDDF_analysis.hbi.ab_loa0_fp_baseline`` (``DEF_CAT``/``DEF_TRUTH``/``DEF_BAL``),
    re-exported through ``track_c_tf_loa`` as ``_C0_*``. This is the calibration
    dataset every kernel/completeness/FP stage is built on, and (in public mode) the
    catalog the headline reduction is run on.
  * ``real_loa`` — the real DESI-LOA catalog (the science target, NO injected truth).
    Constants are ``track_c_tf_loa._LOA_*``. ``privacy="real-LOA"`` → every leaf lands
    under ``real_loa/`` and is unstageable by the privacy guard.
  * ``2lpt1`` / ``london0`` — held-out cross-mock validation datasets. Stubs here:
    they point at their ``track_c_tf_{2lpt1,london0}`` module defaults when those
    modules exist, else fall back to the 2LPT-0 constants with a note. Only ``2lpt0``
    and ``real_loa`` are required to be correct for this PR.

Reading these from the producer modules (not re-typing the paths) is deliberate: if a
producer's default input path changes, the pipeline tracks it automatically and the
config hash / provenance reflect the real input.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Producer constant sources (read VERBATIM — never re-typed).
from CDDF_analysis.hbi import ab_loa0_fp_baseline as AB
from CDDF_analysis.hbi import track_c_tf_loa as TF

__all__ = ["DatasetInputs", "DATASETS", "dataset_inputs"]


@dataclass(frozen=True)
class DatasetInputs:
    """Immutable primary-input bundle for one dataset.

    Attributes
    ----------
    name        : short dataset id (the store's ``{dataset}`` path component)
    catalog_dir : the GP-DLA absorber catalog directory (the data being reduced)
    truth       : injected/placeholder truth FITS (mock truth, or an empty real-LOA
                  placeholder)
    bal         : BAL catalog FITS (for BAL exclusion)
    mockdir     : dir holding snr_cat.fits / zcat.fits (None ⇒ derive from truth dir)
    privacy     : "mock" | "real-LOA" — the leaf privacy subtree + shareability
    """

    name: str
    catalog_dir: str
    truth: str
    bal: str
    mockdir: str | None
    privacy: str  # "mock" | "real-LOA"

    @property
    def resolved_mockdir(self) -> str:
        """The mock catalog dir, deriving it from the truth dir when not set
        (matches ``ab_loa0_fp_baseline``/NB5: ``C0_MOCKDIR = dirname(C0_TRUTH)``)."""
        return self.mockdir if self.mockdir else os.path.dirname(self.truth)


# 2LPT-0 — the calibration / public-headline mock. Constants VERBATIM from the
# producers (AB.DEF_* == TF._C0_*; the mockdir is dirname(truth), as NB5 derives it).
_2LPT0 = DatasetInputs(
    name="2lpt0",
    catalog_dir=AB.DEF_CAT,
    truth=AB.DEF_TRUTH,
    bal=AB.DEF_BAL,
    mockdir=os.path.dirname(AB.DEF_TRUTH),
    privacy="mock",
)

# Real DESI-LOA — the science target, no injected truth (empty placeholder). Constants
# VERBATIM from track_c_tf_loa._LOA_*. privacy="real-LOA" is contagious.
_REAL_LOA = DatasetInputs(
    name="real_loa",
    catalog_dir=TF._LOA_CAT,
    truth=TF._LOA_TRUTH,
    bal=TF._LOA_BAL,
    mockdir=TF._LOA_MOCKDIR,
    privacy="real-LOA",
)


def _stub_crossmock(name: str, module_suffix: str) -> DatasetInputs:
    """Build a cross-mock dataset bundle from its ``track_c_tf_<suffix>`` module
    defaults when that module exists, else fall back to the 2LPT-0 constants.

    These are stubs (the PR only requires 2lpt0 + real_loa to be correct); when the
    held-out cross-mock reduction modules land, their ``_LOA_*``/``_C0_*`` constants
    become the source of truth automatically.
    """
    try:
        mod = __import__(
            f"CDDF_analysis.hbi.track_c_tf_{module_suffix}",
            fromlist=["_C0_CAT"],
        )
    except Exception:
        # module not present yet — stub against 2LPT-0 so the registry is well-formed.
        return DatasetInputs(
            name=name,
            catalog_dir=AB.DEF_CAT,
            truth=AB.DEF_TRUTH,
            bal=AB.DEF_BAL,
            mockdir=os.path.dirname(AB.DEF_TRUTH),
            privacy="mock",
        )
    # prefer the module's own held-out catalog constants when present.
    cat = getattr(mod, "_LOA_CAT", getattr(mod, "_C0_CAT", AB.DEF_CAT))
    truth = getattr(mod, "_LOA_TRUTH", getattr(mod, "_C0_TRUTH", AB.DEF_TRUTH))
    bal = getattr(mod, "_LOA_BAL", getattr(mod, "_C0_BAL", AB.DEF_BAL))
    mockdir = getattr(mod, "_LOA_MOCKDIR", None) or os.path.dirname(truth)
    return DatasetInputs(
        name=name, catalog_dir=cat, truth=truth, bal=bal,
        mockdir=mockdir, privacy="mock",
    )


DATASETS: dict[str, DatasetInputs] = {
    "2lpt0": _2LPT0,
    "real_loa": _REAL_LOA,
    "2lpt1": _stub_crossmock("2lpt1", "2lpt1"),
    "london0": _stub_crossmock("london0", "london0"),
}


def dataset_inputs(name: str) -> DatasetInputs:
    """Return the ``DatasetInputs`` bundle for ``name``; raise a clear error listing
    the known datasets otherwise."""
    try:
        return DATASETS[name]
    except KeyError:
        known = ", ".join(sorted(DATASETS))
        raise KeyError(
            f"unknown dataset {name!r}; known datasets: {known}."
        ) from None
