"""loader.py -- tidy, guarded loader for the DLA headline artifact.

Returns a ``HeadlineData`` dataclass of numpy arrays that the DLA / sub-DLA plotters
consume.  Paths are PARAMETERS with defaults (never scattered string literals).  By
default it runs the provenance guard (must be RE_DERIVABLE) and the schema validator
before returning anything.

It also DERIVES three per-z-bin flags so plotters cannot forget the (distinct!)
extrapolation regimes -- see ``ZBinFlags`` below.  Nothing here prints a science value.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import provenance as prov
from . import schema as _schema

# ---------------------------------------------------------------------------
# Default artifact paths + their generating routines (repo-relative).
# ---------------------------------------------------------------------------
_TF_LOA_DIR = "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/tf_loa"

# loa0 FP-model headline (the DLA headline), correctly stamped at HEAD (d496f42).
DEFAULT_LOA0_ARTIFACT = os.path.join(_TF_LOA_DIR, "track_c_tf_loa_loa0_restamped.json")
DEFAULT_LOA0_ROUTINE = (
    "CDDF_analysis/diagnostics/bal_metal_fp/arbiter/run_loa0_headline_full.py"
)

# purity_mixture archival comparison (the other arm of the FP-model bracket), f1784fc.
DEFAULT_PURITY_ARTIFACT = os.path.join(_TF_LOA_DIR, "track_c_tf_loa.json")
DEFAULT_PURITY_ROUTINE = "CDDF_analysis/hbi/track_c_tf_loa.py"

_QK = ("MAP", "q025", "q16", "q84", "q975", "std")


@dataclass
class ZBinFlags:
    """Three DISTINCT per-z-bin regime flags.  They do NOT coincide; the loader

    surfaces each so a plotter cannot collapse them:

      beyond_calibration    == metadata.z_extrapolated (as stamped): the frozen
                               completeness g(N,z) has no 2LPT-0 truth support here
                               (lower edge above max_truth_z, or zero truth count).
      beyond_v2_fit         := zbin_lo >= metadata.v2_z_fit_hi: the mean-flux /
                               effective-optical-depth model was fit only up to this
                               ceiling; above it the forward model is extrapolated.
      partial_truth_support := zbin_lo < max_truth_z < zbin_hi: the bin straddles the
                               truth cap (calibrated over part of the bin only).
    """

    zbin_lo: np.ndarray            # (n_zbin,)
    zbin_hi: np.ndarray            # (n_zbin,)
    z_centers: np.ndarray          # (n_zbin,)
    beyond_calibration: np.ndarray  # (n_zbin,) bool  == z_extrapolated
    beyond_v2_fit: np.ndarray       # (n_zbin,) bool
    partial_truth_support: np.ndarray  # (n_zbin,) bool
    z_thin: np.ndarray              # (n_zbin,) bool  (thin but calibrated)
    max_truth_z: float = float("nan")
    v2_z_fit_hi: float = float("nan")
    # indices where beyond_v2_fit but NOT beyond_calibration -- the case that must not hide.
    v2_beyond_but_calibrated: tuple = ()


@dataclass
class HeadlineData:
    """Tidy headline container.  Arrays hold real-LOA values IN MEMORY ONLY --

    never print/tabulate them; plotters draw from them and outputs are stripped."""

    artifact_path: str
    code_commit: Optional[str]
    fp_estimator: Optional[str]
    limits: tuple                          # ('20.0', '20.3')
    zbins: np.ndarray                      # (6,) edges
    z_centers: np.ndarray                  # (5,)
    zflags: ZBinFlags
    truth_counts_perz: np.ndarray          # (5,) int
    logN_centers: np.ndarray               # (52,)
    # integrated[limit][obs] -> dict of 6 scalar quantiles (MAP,q025,q16,q84,q975,std)
    integrated: dict = field(default_factory=dict)
    # perz[limit][obs] -> dict of 6 arrays, each (5,)
    perz: dict = field(default_factory=dict)
    # fN[key] -> array: f/f68_lo/f68_hi/f95_lo/f95_hi (5,52); z (5,); z_idx (5,)
    fN: dict = field(default_factory=dict)
    provenance: Optional[prov.ProvenanceResult] = None
    schema_report: Optional[_schema.SchemaReport] = None

    def band(self, limit: str, obs: str, kind: str = "integrated"):
        """Convenience accessor.  ``kind`` in {'integrated','perz'}."""
        return (self.integrated if kind == "integrated" else self.perz)[limit][obs]


def _derive_zflags(zbins, z_extrapolated, z_thin, max_truth_z, v2_z_fit_hi) -> ZBinFlags:
    zbins = np.asarray(zbins, float)
    lo, hi = zbins[:-1], zbins[1:]
    centers = 0.5 * (lo + hi)
    beyond_cal = np.asarray(z_extrapolated, bool)
    beyond_v2 = lo >= float(v2_z_fit_hi)
    mtz = float(max_truth_z)
    partial = (lo < mtz) & (mtz < hi)
    v2_but_cal = tuple(int(i) for i in np.where(beyond_v2 & ~beyond_cal)[0])
    return ZBinFlags(
        zbin_lo=lo, zbin_hi=hi, z_centers=centers,
        beyond_calibration=beyond_cal, beyond_v2_fit=beyond_v2,
        partial_truth_support=partial, z_thin=np.asarray(z_thin, bool),
        max_truth_z=mtz, v2_z_fit_hi=float(v2_z_fit_hi),
        v2_beyond_but_calibrated=v2_but_cal,
    )


def _perz_arrays(node):
    """{quantile: (n_zbin,) array} from a measurement[lim][obs]['perz'] list."""
    perz = node["perz"]
    return {q: np.array([e[q] for e in perz], float) for q in _QK}


def load_headline(
    artifact_path: str = DEFAULT_LOA0_ARTIFACT,
    routine_path: Optional[str] = DEFAULT_LOA0_ROUTINE,
    guard: bool = True,
    require=prov.PASS_STATUSES,
    validate: bool = True,
    warn_v2: bool = True,
) -> HeadlineData:
    """Load, guard, validate, and tidy a headline artifact.

    Parameters
    ----------
    artifact_path : path to the stamped headline JSON (default: loa0 headline).
    routine_path  : the generating routine (needed for the ORPHANED check on
                    artifacts that lack metadata.rederive, e.g. the track_c family).
    guard         : run the provenance guard and RAISE unless status in ``require``.
    validate      : run the schema validator (raise SchemaError on drift).
    warn_v2       : loudly surface any z-bin that is beyond_v2_fit yet NOT
                    beyond_calibration (the regime that must not pass silently).
    """
    pres = None
    if guard:
        pres = prov.check_artifact(artifact_path, routine_path=routine_path, allowed=require)

    with open(artifact_path) as f:
        d = json.load(f)

    srep = _schema.validate_headline_schema(d) if validate else None

    md = d["metadata"]
    zbins = np.asarray(d["zbins"], float)
    pfn = d["perz_fN"]

    zflags = _derive_zflags(
        zbins=zbins,
        z_extrapolated=md.get("z_extrapolated", pfn["z_extrapolated"]),
        z_thin=md.get("z_thin", pfn["z_thin"]),
        max_truth_z=md.get("max_truth_z", float("nan")),
        v2_z_fit_hi=md.get("v2_z_fit_hi", float("nan")),
    )
    if warn_v2 and zflags.v2_beyond_but_calibrated:
        idx = zflags.v2_beyond_but_calibrated
        edges = [(float(zflags.zbin_lo[i]), float(zflags.zbin_hi[i])) for i in idx]
        print(
            "[loader] WARNING: z-bin(s) "
            + ", ".join(f"#{i} [{lo:.2f},{hi:.2f}]" for i, (lo, hi) in zip(idx, edges))
            + f" are beyond the v2 fit ceiling (v2_z_fit_hi={zflags.v2_z_fit_hi:g}) but are "
            "NOT flagged beyond_calibration (z_extrapolated). The forward mean-flux model "
            "is extrapolated there even though completeness has (partial) truth support -- "
            "carry the v2-extrapolation systematic; the MC band does NOT cover it."
        )

    limits = tuple(_schema.LIMITS)
    integrated, perz = {}, {}
    for lim in limits:
        integrated[lim] = {obs: dict(d["measurement"][lim][obs]["integrated"]) for obs in ("dndx", "omega")}
        perz[lim] = {obs: _perz_arrays(d["measurement"][lim][obs]) for obs in ("dndx", "omega")}

    fN = {
        "logN_centers": np.asarray(pfn["logN_centers"], float),
        "f": np.array([e["f"] for e in pfn["perz"]], float),
        "f68_lo": np.array([e["f68_lo"] for e in pfn["perz"]], float),
        "f68_hi": np.array([e["f68_hi"] for e in pfn["perz"]], float),
        "f95_lo": np.array([e["f95_lo"] for e in pfn["perz"]], float),
        "f95_hi": np.array([e["f95_hi"] for e in pfn["perz"]], float),
        "z": np.array([e["z"] for e in pfn["perz"]], float),
        "z_idx": np.array([e["z_idx"] for e in pfn["perz"]], int),
        "extrapolated": np.array([e["extrapolated"] for e in pfn["perz"]], bool),
        "thin": np.array([e["thin"] for e in pfn["perz"]], bool),
    }

    return HeadlineData(
        artifact_path=artifact_path,
        code_commit=md.get("code_commit"),
        fp_estimator=md.get("fp_estimator"),
        limits=limits,
        zbins=zbins,
        z_centers=zflags.z_centers,
        zflags=zflags,
        truth_counts_perz=np.asarray(pfn["truth_counts_perz"], int),
        logN_centers=fN["logN_centers"],
        integrated=integrated,
        perz=perz,
        fN=fN,
        provenance=pres,
        schema_report=srep,
    )
