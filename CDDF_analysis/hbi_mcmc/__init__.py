"""CDDF_analysis.hbi_mcmc — NumPyro/NUTS MCMC inference module (three-route plan, Track C).

Seed of the new MCMC-based HBI inference package. Pure-synthetic validation ladder
lives in ``validation/`` (rungs 1-3); shared MCMC/recycling diagnostics in
``diagnostics.py``. No survey data, no coupling to the frozen GP inference path.

IMPORTANT: this package enables JAX 64-bit mode at import time (below). All
submodules assume float64; import this package (or any submodule, which triggers
this __init__) before creating any jax arrays.
"""
import jax

# Must run before any jax array is created anywhere in the process.
jax.config.update("jax_enable_x64", True)

__version__ = "0.0.1"
