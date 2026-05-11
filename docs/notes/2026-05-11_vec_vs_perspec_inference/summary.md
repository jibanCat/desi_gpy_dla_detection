# Inference consistency: vec vs per-spec on canonical TID 120046865

Cross-domain test: SDSS-DR16-trained model on a DESI 2lpt spectrum.
What we're checking: do **both DR16 retrains** (vec full and per-spec
full) produce the same DLAHolder output? Absolute values are not the
point — relative agreement is.

Truth: TID 120046865 log_NHI = 21.263 (z_qso = 2.9620).

| metric | vec_full | per_spec | Δ |
|---|---:|---:|---:|
| p_DLA | 0.086514 | 0.083611 | +2.90e-03 |
| MAP z_DLA | nan | nan | +nan |
| MAP log NHI | nan | nan | +nan |
| elapsed (s) | 24.9 | 30.3 | — |

## Model posteriors (per absorber count)

Layout: [Null, SubDLA, 1DLA, 2DLA, 3DLA, 4DLA] (max_dlas=4, single_absorber_model=False).

| idx | vec_full | per_spec | Δ |
|---|---:|---:|---:|
| 0 | 9.134781e-01 | 9.163820e-01 | -2.90e-03 |
| 1 | 7.526140e-06 | 7.466257e-06 | +5.99e-08 |
| 2 | 8.651435e-02 | 8.361053e-02 | +2.90e-03 |
| 3 | nan | nan | +nan |
| 4 | nan | nan | +nan |
| 5 | nan | nan | +nan |
