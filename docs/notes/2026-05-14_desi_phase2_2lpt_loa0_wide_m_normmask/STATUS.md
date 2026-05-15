# ⏳ IN-FLIGHT (post-reorder retrain)

| Field | Value |
|---|---|
| SLURM job | 50212866 |
| Dataset | 2lpt loa-0 |
| Target n_iters | 1500 |
| Norm band | [1425/1475] Å rest |
| Pipeline | **post-reorder** (`dataset.py` reorder + `|med| < 1e-2` threshold; commit aa36205+) |
| ETA | ~2026-05-14 evening |

This directory exists but training is incomplete. When training finishes,
`phase2_train_desi.py` will write `phase2_result.h5`, `phase2_result.npz`,
and overwrite this STATUS with a proper model-card README via
`examples/reemit_step_c_readmes.py`.

**Until then, do NOT use any partial checkpoint as a production model.**
The .pt checkpoints in
`/scratch/cavestru_root/cavestru0/mfho/phase2_desi/<run_name>/checkpoints/`
are intermediate Adam iterates without trained-hyper metadata. See
`docs/CURRENT_MODELS.md` for currently-recommended models to use.

Monitor: `tail -f slurm/greatlakes/phase2_desi_retrain_50212866.log`
