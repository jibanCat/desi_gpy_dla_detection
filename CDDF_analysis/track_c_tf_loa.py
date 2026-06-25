"""Back-compat shim — module moved to CDDF_analysis.hbi.track_c_tf_loa (2026-06 reorg).
Importing this old path returns the SAME object as the new path. Prefer the new path."""
import sys
from CDDF_analysis.hbi import track_c_tf_loa as _moved
sys.modules[__name__] = _moved
