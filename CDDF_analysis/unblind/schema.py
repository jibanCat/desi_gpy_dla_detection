"""schema.py -- structural validator for the real-LOA / mock headline JSON.

Validates key presence and array shapes ONLY.  It never inspects a value, so it is
privacy-safe to run on the real-LOA artifact.  It fails loudly on any drift from the
verified schema so a downstream plotter can never silently mis-read a re-shaped file.

Verified schema (track_c_tf_loa* family):

  top-level                 : measurement, metadata, perz_fN, zbins
  measurement               : keys '20.0', '20.3'
  measurement[lim]          : keys 'dndx', 'omega'
  measurement[lim][obs]     : 'integrated' (dict) + 'perz' (list, one per z-bin)
  integrated / perz[i]      : keys MAP, q025, q16, q84, q975, std
  zbins                     : list length N_ZBIN_EDGES (=6 -> 5 z-bins)
  perz_fN.logN_centers      : length 52
  perz_fN.zbins             : length 6
  perz_fN.z_extrapolated    : length 5 (bool)
  perz_fN.z_thin            : length 5 (bool)
  perz_fN.truth_counts_perz : length 5 (int)
  perz_fN.perz              : list length 5 of dicts with keys
                              extrapolated, f, f68_hi, f68_lo, f95_hi, f95_lo, thin, z, z_idx
  perz_fN.perz[i].f*        : length-52 arrays

Extra keys (e.g. perz_fN.band_method, perz_fN.floor) are tolerated; only drift in the
required keys/shapes fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# expected constants (assert exactly -- drift is the thing we want to catch)
N_LOGN = 52
N_ZBIN_EDGES = 6
N_ZBIN = N_ZBIN_EDGES - 1  # 5

LIMITS = ("20.0", "20.3")
OBSERVABLES = ("dndx", "omega")
QUANTILE_KEYS = frozenset({"MAP", "q025", "q16", "q84", "q975", "std"})
PERZ_FN_KEYS = frozenset(
    {"extrapolated", "f", "f68_hi", "f68_lo", "f95_hi", "f95_lo", "thin", "z", "z_idx"}
)
FN_ARRAY_KEYS = ("f", "f68_hi", "f68_lo", "f95_hi", "f95_lo")


class SchemaError(ValueError):
    """Raised on any drift from the verified headline schema."""


@dataclass
class SchemaReport:
    """Shapes only -- carries NO science values, safe to print."""

    limits: tuple = ()
    n_zbin_edges: int = 0
    n_zbin: int = 0
    n_logN: int = 0
    n_fN_perz: int = 0
    checks: list = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["schema OK:"]
        lines += [f"  - {c}" for c in self.checks]
        return "\n".join(lines)


def _require(cond: bool, msg: str):
    if not cond:
        raise SchemaError(msg)


def _has_len(x, n: int, name: str):
    _require(hasattr(x, "__len__"), f"{name}: expected a sized array, got {type(x).__name__}")
    _require(len(x) == n, f"{name}: expected length {n}, got {len(x)}")


def validate_headline_schema(d: dict) -> SchemaReport:
    """Validate the headline JSON structure/shapes; raise SchemaError on drift.

    Returns a SchemaReport of shape counts (no values)."""
    rep = SchemaReport()

    _require(isinstance(d, dict), "top level is not a JSON object")
    for k in ("measurement", "metadata", "perz_fN", "zbins"):
        _require(k in d, f"missing top-level key {k!r}")
    rep.checks.append("top-level keys present: measurement, metadata, perz_fN, zbins")

    # zbins ---------------------------------------------------------------
    _has_len(d["zbins"], N_ZBIN_EDGES, "zbins")
    rep.n_zbin_edges = len(d["zbins"])
    rep.n_zbin = rep.n_zbin_edges - 1
    rep.checks.append(f"zbins has {N_ZBIN_EDGES} edges -> {N_ZBIN} z-bins")

    # measurement ---------------------------------------------------------
    m = d["measurement"]
    _require(isinstance(m, dict), "measurement is not an object")
    for lim in LIMITS:
        _require(lim in m, f"measurement missing limit {lim!r}")
        for obs in OBSERVABLES:
            _require(obs in m[lim], f"measurement[{lim!r}] missing observable {obs!r}")
            node = m[lim][obs]
            for part in ("integrated", "perz"):
                _require(part in node, f"measurement[{lim!r}][{obs!r}] missing {part!r}")
            # integrated: quantile dict
            _require(
                QUANTILE_KEYS.issubset(node["integrated"].keys()),
                f"measurement[{lim!r}][{obs!r}].integrated missing quantile keys "
                f"{sorted(QUANTILE_KEYS - set(node['integrated'].keys()))}",
            )
            # perz: one entry per z-bin, each a quantile dict
            _has_len(node["perz"], N_ZBIN, f"measurement[{lim!r}][{obs!r}].perz")
            for i, entry in enumerate(node["perz"]):
                _require(
                    QUANTILE_KEYS.issubset(entry.keys()),
                    f"measurement[{lim!r}][{obs!r}].perz[{i}] missing quantile keys "
                    f"{sorted(QUANTILE_KEYS - set(entry.keys()))}",
                )
    rep.limits = LIMITS
    rep.checks.append(
        f"measurement[{LIMITS}] x {OBSERVABLES}: integrated + perz[{N_ZBIN}] with "
        f"quantile keys {sorted(QUANTILE_KEYS)}"
    )

    # perz_fN -------------------------------------------------------------
    p = d["perz_fN"]
    _require(isinstance(p, dict), "perz_fN is not an object")
    for k in ("logN_centers", "zbins", "z_extrapolated", "z_thin", "truth_counts_perz", "perz"):
        _require(k in p, f"perz_fN missing key {k!r}")
    _has_len(p["logN_centers"], N_LOGN, "perz_fN.logN_centers")
    _has_len(p["zbins"], N_ZBIN_EDGES, "perz_fN.zbins")
    _has_len(p["z_extrapolated"], N_ZBIN, "perz_fN.z_extrapolated")
    _has_len(p["z_thin"], N_ZBIN, "perz_fN.z_thin")
    _has_len(p["truth_counts_perz"], N_ZBIN, "perz_fN.truth_counts_perz")
    _has_len(p["perz"], N_ZBIN, "perz_fN.perz")
    rep.n_logN = len(p["logN_centers"])
    rep.n_fN_perz = len(p["perz"])
    for i, entry in enumerate(p["perz"]):
        _require(
            PERZ_FN_KEYS.issubset(entry.keys()),
            f"perz_fN.perz[{i}] missing keys {sorted(PERZ_FN_KEYS - set(entry.keys()))}",
        )
        for key in FN_ARRAY_KEYS:
            _has_len(entry[key], N_LOGN, f"perz_fN.perz[{i}].{key}")
    rep.checks.append(
        f"perz_fN: logN_centers[{N_LOGN}], zbins[{N_ZBIN_EDGES}], "
        f"z_extrapolated/z_thin/truth_counts_perz[{N_ZBIN}], perz[{N_ZBIN}] "
        f"each with f/f68/f95 arrays length {N_LOGN}"
    )

    return rep
