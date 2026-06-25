"""Back-compat shim — module moved to CDDF_analysis.hbi.run_remp_kernel (2026-06 reorg).
Importing this old path returns the SAME object as the new path. Prefer the new path."""
import sys
from CDDF_analysis.hbi import run_remp_kernel as _moved
sys.modules[__name__] = _moved
