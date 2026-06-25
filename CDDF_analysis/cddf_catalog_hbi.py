"""Back-compat shim — module moved to CDDF_analysis.hbi.cddf_catalog_hbi (2026-06 reorg).
Importing this old path returns the SAME object as the new path. Prefer the new path."""
import sys
from CDDF_analysis.hbi import cddf_catalog_hbi as _moved
sys.modules[__name__] = _moved
