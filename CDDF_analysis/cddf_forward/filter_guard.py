"""FILTER-off guard for the CDDF pipeline.

The Monte-Carlo CDDF estimator is only valid on FILTER-off runs
(``FILTER_LOW_LIKELIHOOD = 0``).  FILTER-on truncates per-sample DLA evidence for
low-likelihood samples, which biases the per-bin probabilistic counts the CDDF
integrates over — so the CDDF must refuse to run on a FILTER-on catalog.

SCHEMA FINDING (M0)
-------------------
The processed-HDF5 schema does **not** persist the FILTER flag.  The production
writer ``run_bayes_select.DLAHolder.save_results`` (run_bayes_select.py:602) and
the legacy ``gpy_dla_detection.process_helpers.save_results_to_hdf5``
(process_helpers.py:243) both write only ``pair_prior_mode`` and ``dla_bias`` as
root-group attributes — ``filter_low_likelihood`` is never recorded in the file
or in any dataset.  Consequently this guard cannot read the flag from the
processed file; the FILTER setting must be supplied explicitly by the driver
(from the run's ``.env`` / sbatch config).  Persisting the FILTER flag in the
HDF5 schema is a follow-up for the driver milestone.
"""


def assert_filter_off(filter_low_likelihood: int, *, ctx: str = "") -> None:
    """Raise ``ValueError`` unless the run was FILTER-off.

    Parameters
    ----------
    filter_low_likelihood : int
        The ``FILTER_LOW_LIKELIHOOD`` setting of the run that produced the
        processed HDF5.  ``0`` (FILTER-off) is the only CDDF-valid value;
        anything truthy (``1`` / ``True``) raises.  Supplied explicitly because
        the schema does not persist it (see module docstring).
    ctx : str, optional
        Caller-side context prepended to the error message.

    Raises
    ------
    ValueError
        If ``filter_low_likelihood`` is non-zero (FILTER-on) OR cannot be
        determined (``None`` / NaN / non-integer) — an unknown FILTER setting is
        treated as unsafe and refused.
    """
    prefix = f"{ctx}: " if ctx else ""
    # An unknown flag (None / NaN / non-coercible) must be treated as unsafe:
    # the guard exists to refuse a possibly-FILTER-on catalog, so "don't know"
    # is refused, not silently passed or crashed with a confusing message.
    try:
        if filter_low_likelihood is None or (
            isinstance(filter_low_likelihood, float)
            and filter_low_likelihood != filter_low_likelihood  # NaN
        ):
            raise ValueError("unknown")
        flag = int(filter_low_likelihood)
    except (TypeError, ValueError):
        raise ValueError(
            prefix
            + "the CDDF is only valid on FILTER-off runs, but the FILTER setting "
            + f"is unknown (FILTER_LOW_LIKELIHOOD={filter_low_likelihood!r}). "
            + "Refusing: supply the run's FILTER_LOW_LIKELIHOOD (0 for FILTER-off) "
            + "explicitly from its .env/config."
        )
    if flag != 0:
        raise ValueError(
            prefix
            + "the CDDF is only valid on FILTER-off runs, but "
            + f"FILTER_LOW_LIKELIHOOD={flag} (FILTER-on). "
            + "FILTER-on truncates low-likelihood per-sample DLA evidence, which "
            + "biases the probabilistic per-bin counts the CDDF integrates over. "
            + "Re-run inference with FILTER_LOW_LIKELIHOOD=0, or pass a FILTER-off "
            + "catalog."
        )


def read_filter_flag(processed_file):
    """Read the persisted ``filter_low_likelihood`` root attr from a processed HDF5.

    Returns the int flag if present, or ``None`` for legacy files written before
    ``run_bayes_select.save_results`` started persisting it (in which case the
    caller must supply the FILTER setting explicitly).  Makes the catalog
    self-describing so the guard can be asserted FROM THE FILE rather than trusted
    from the driver.
    """
    import h5py

    with h5py.File(processed_file, "r") as f:
        if "filter_low_likelihood" in f.attrs:
            return int(f.attrs["filter_low_likelihood"])
    return None


def assert_filter_off_from_file(processed_file, *, supplied=None, ctx=""):
    """Guard using the persisted flag when present, else the ``supplied`` value.

    If the file records ``filter_low_likelihood``, that wins (and, if ``supplied``
    is given and disagrees, raise — the driver was mislabelled).  If the file has
    no attr (legacy), fall back to ``supplied`` (which must then be provided).
    """
    file_flag = read_filter_flag(processed_file)
    if file_flag is not None:
        if supplied is not None and int(supplied) != file_flag:
            raise ValueError(
                f"{ctx + ': ' if ctx else ''}supplied filter_low_likelihood="
                f"{supplied!r} disagrees with the file's persisted value {file_flag}."
            )
        assert_filter_off(file_flag, ctx=ctx)
    else:
        if supplied is None:
            raise ValueError(
                f"{ctx + ': ' if ctx else ''}this processed file predates the "
                "persisted FILTER flag; supply filter_low_likelihood explicitly "
                "(0 for FILTER-off) from the run's .env."
            )
        assert_filter_off(supplied, ctx=ctx)
