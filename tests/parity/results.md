# Layer 4 — MATLAB ↔ Python parity for spectrum_loss

Reference: `/home/mfho/gp_dla_detection_dr16q_public/spectrum_loss.m`
Python:    `gpy_dla_detection/objective.py::spectrum_loss`

Tolerance: rtol = atol = 1e-12

| output | shape | max\|Δ\| | max rel\|Δ\| | pass |
|---|---|---:|---:|:---:|
| `nlog_p` | () | 7.105e-15 | 3.636e-16 | ✅ |
| `dM` | (32, 4) | 1.835e-13 | 2.238e-12 | ✅ |
| `dlog_omega` | (32,) | 5.551e-16 | 3.338e-15 | ✅ |
| `dlog_c_0` | () | 0.000e+00 | 0.000e+00 | ✅ |
| `dlog_tau_0` | () | 1.776e-15 | 1.865e-15 | ✅ |
| `dlog_beta` | () | 7.105e-15 | 1.983e-15 | ✅ |

**Overall: PASS**
