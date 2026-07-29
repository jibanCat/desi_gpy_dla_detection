"""B16 contamination of the LLS ell(X) truth denominator — executable documentation.

Background
----------
B16 is the z-leaky truth reduction: the truth ``f(N)`` array counts absorbers at ALL
redshifts while the pathlength ``X_sum`` it is divided by spans only ``cfg.zbins``. The
z-bin index is computed and applied to ``dndx_total`` but not to ``f_truth``.

It lives in TWO functions, not one:

  * ``CDDF_analysis/hbi/cddf_tilt_closure.py::tilted_truth_reductions``  :144-146 (leaky)
    vs :168-170 (``dndx_total``, masked with ``zidx >= 0``)   -- the originally reported site
  * ``CDDF_analysis/hbi/cddf_catalog_hbi.py::truth_reductions``          :2140-2142 (leaky)
    vs :2153-2155 (``dndx_total``, masked with ``t_zidx >= 0``) -- the SECOND site, which the
    2026-07-13 blast-radius audit missed entirely

The LLS ``ell(X)`` R0 is contaminated through the SECOND site:

    joint_drop_count_validation.py:101   tr = truth_reductions(...)
    joint_drop_count_validation.py:146   f_truth = np.asarray(tr["f_truth"], float)
    joint_drop_count_validation.py:148   true_ell = nansum(f_truth[sel] * dNb[sel])
    joint_drop_count_validation.py:199   r0_band_q50 = band_q50 / true_ell

The blast-radius audit recorded ``ell(X) R0 ~ 0.82`` as CLEAN and "``dndx_total``-derived".
It is ``f_truth``-derived, hence leaky. This test pins that, using ONLY committed artifacts.

What this test asserts
----------------------
``ell(X)`` over a band and ``dN/dX`` over the same band are the SAME estimand — the incidence
per unit absorption distance. So the z-masked ``ell`` truth must equal the ``dndx_total``
truth over the same band. The committed artifacts disagree by exactly the leak factor, which
is the contamination, measured:

    joint_mock_validation.json::result.true_ell           = 0.2628520  (f_truth, LEAKY)
    lls_mock_validation.json::...v1.dndx_tru_172_195      = 0.2487742  (dndx_total, CLEAN)
    leak                                                   = 1.056588

Corrected (denominator-only) LLS ell(X) recoveries on 2LPT-0:

    r0_band_q50   [17.2,19.5)  0.817644 -> 0.863913
    r0_canonical  [17.2,19.5)  0.812580 -> 0.858562
    r0_175        [17.5,19.5)  0.480230 -> 0.507595   (untracked lls_recovery_figures.json)

Survives the B16 fix
--------------------
A fix at source is in flight in a separate workstream. These tests are written to stay GREEN
across it, because the load-bearing assertion is a **state invariant**:

    r0_band_q50 * (true_ell / dndx_tru_172_195)  ==  0.863913

holds in BOTH worlds. Pre-fix the first factor is 0.8176 and the ratio is 1.0566; post-fix
(after re-deriving both JSONs) the first factor is ~0.8639 and the ratio is 1.0. The identity
is exactly "measure ell(X) recovery against the z-masked truth", which is the corrected claim.

``test_reports_b16_state`` detects and reports which regime the committed artifacts are in and
never fails. Only ``EXPECT_CORRECTED_*`` are hard-pinned. Do NOT "fix" a failure by editing
those constants — a real move means the LLS ell(X) recovery itself changed.

Reads committed JSON only — no fitsio, no mock data, no compute. Runs anywhere.
"""
from __future__ import annotations

import json
import os

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HBI = os.path.join(_REPO, "CDDF_analysis", "hbi")

JOINT_JSON = os.path.join(_HBI, "joint_mock_validation.json")
LLS_JSON = os.path.join(_HBI, "lls_mock_validation.json")

# measured 2026-07-28 by re-running the committed path (bit-for-bit reproduction) and
# recomputing f_truth with the `t_zidx >= 0` mask that truth_reductions omits, against the
# artifacts as committed at 66088d7 / 4d3e16d.
EXPECT_LEAK = 1.0565884686739537
CORRECTED_R0_BAND_Q50 = 0.863913
CORRECTED_R0_CANONICAL = 0.858562
# truth rows in the [17.2,19.5) bin support, 2LPT-0, SNR>2: 131559 all-z / 124513 in-window
EXPECT_LEAK_ROWS = (131559, 124513)


def _load(path):
    if not os.path.exists(path):
        pytest.skip(f"committed artifact missing: {path}")
    with open(path) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def joint():
    return _load(JOINT_JSON)["result"]


@pytest.fixture(scope="module")
def lls():
    return _load(LLS_JSON)["results"]


def test_reports_b16_state(joint, lls):
    """Report, never fail: which regime are the committed artifacts in?

    The ell(X) truth and the dN/dX truth are the SAME estimand over the SAME band
    ([17.2,19.5) incidence per unit absorption distance), so their ratio is exactly the B16
    leak. 1.0566 => artifacts predate the fix; 1.0 => they were re-derived after it."""
    leak = joint["true_ell"] / lls["loa0"]["v1"]["dndx_tru_172_195"]
    if leak == pytest.approx(EXPECT_LEAK, rel=1e-6):
        state = "PRE-FIX (leaky) — ell(X) R0 in these JSONs must be read x1.0566"
    elif leak == pytest.approx(1.0, rel=1e-6):
        state = "POST-FIX (z-masked) — ell(X) R0 in these JSONs is already corrected"
    else:
        state = f"UNRECOGNISED leak {leak!r}"
    print(f"\n[B16 state] true_ell/dndx_tru_172_195 = {leak!r} -> {state}")
    assert 0.99 <= leak <= 1.07, (
        f"B16 leak factor {leak!r} is outside both known regimes (1.0 or {EXPECT_LEAK}). "
        f"The LLS truth reduction changed in an unexpected way — investigate before quoting "
        f"any ell(X) number.")


def test_dndx_truth_is_shared_across_fp_estimators_and_paths(lls):
    """Sanity on the CLEAN leg: the truth denominator does not depend on the FP estimator
    or the estimator path, so the cross-check value is unambiguous."""
    vals = {f"{m}/{p}": lls[m][p]["dndx_tru_172_195"]
            for m in ("loa0", "purity_mixture") for p in ("v1", "v3x")}
    ref = vals["loa0/v1"]
    for k, v in vals.items():
        assert v == pytest.approx(ref, rel=1e-12), f"{k} truth differs from loa0/v1: {vals}"


def test_ell_recovery_against_the_zmasked_truth_is_the_documented_value(joint, lls):
    """THE load-bearing assertion, and the one that survives the B16 fix.

    ell(X) recovery measured against the z-masked truth is 0.8639 on [17.2,19.5), not the
    0.8176 the blast-radius audit filed as clean. Written as an identity in `leak` so it
    holds whether or not the committed JSONs have been re-derived (see module docstring).
    Tolerance 1e-3 absorbs the ~0.1% numerator shift a source fix adds via _shape_priors."""
    leak = joint["true_ell"] / lls["loa0"]["v1"]["dndx_tru_172_195"]
    assert joint["derived"]["r0_band_q50"] * leak == pytest.approx(
        CORRECTED_R0_BAND_Q50, abs=1e-3), (
        "LLS ell(X) R0[17.2,19.5) vs the z-masked truth is no longer 0.8639 — do NOT edit "
        "this constant; the recovery itself moved.")
    assert joint["derived"]["r0_canonical"] * leak == pytest.approx(
        CORRECTED_R0_CANONICAL, abs=1e-3)
    # and it is NOT the number the audit filed as clean
    assert abs(CORRECTED_R0_BAND_Q50 - 0.8176435427171277) > 0.04


def test_correcting_the_truth_does_not_rescue_band_coverage(joint, lls):
    """The z-masked truth is still OUTSIDE the shape-marginalized band, so the LLS
    'truth_in_band = False' conclusion is B16-immune. Removes any temptation to read the
    correction as a rescue of ell(X)."""
    q16, _, q84 = joint["derived"]["band_q16_q50_q84"]
    corrected_truth = lls["loa0"]["v1"]["dndx_tru_172_195"]
    assert joint["derived"]["truth_in_band"] is False
    assert not (q16 <= corrected_truth <= q84), (
        f"corrected truth {corrected_truth} unexpectedly inside band [{q16}, {q84}] — "
        f"the 'band excludes truth' finding would need re-stating.")


def test_lambda_mfp_headline_is_not_the_contaminated_quantity(joint):
    """The lambda_mfp truth leg is a direct sum over the HCD truth catalogue
    (joint_drop_count_validation.py:105-131), never f_truth, so R0_lambda stays ~1.
    Its ESTIMATE leg inherits a +0.026 dex shape-prior anchor shift worth +0.124%
    (0.983515 -> 0.984735, measured 2026-07-28) — small, but not exactly zero, which is why
    the blast-radius audit's stated REASON for cleanliness is wrong even though its verdict
    holds."""
    assert joint["headline"]["r0_lambda_mfp"] == pytest.approx(0.9835145764785819, rel=1e-9)
    assert abs(0.984735 - joint["headline"]["r0_lambda_mfp"]) < 0.005


def test_b16_stamp_does_not_declare_the_ell_rows_clean(joint, lls):
    """Guard against RE-INTRODUCING the corrected error via a ``metadata.b16`` stamp.

    A stamp may not call this artifact clean while its ``true_ell`` is still leaky.
    ``r0_band_q50`` and ``r0_canonical`` are ell(X) recoveries whose denominator is an
    ``f_truth`` integral; "it has no Omega leaf" does NOT make them clean — that is the
    units-based reasoning this whole test module exists to refute. Either the artifact is
    re-derived (leak -> 1.0) or the stamp must not say CLEAN.
    """
    meta = _load(JOINT_JSON).get("metadata", {})
    b16 = meta.get("b16")
    if not b16:
        pytest.skip("no metadata.b16 stamp on joint_mock_validation.json yet")
    leak = joint["true_ell"] / lls["loa0"]["v1"]["dndx_tru_172_195"]
    still_leaky = leak > 1.0001
    blob = json.dumps(b16).upper()
    claims_clean = "CLEAN" in str(b16.get("status", "")).upper() or "AUDITED CLEAN" in blob
    assert not (still_leaky and claims_clean), (
        f"metadata.b16 declares status={b16.get('status')!r} while true_ell is STILL LEAKY "
        f"(leak={leak!r}). r0_band_q50={joint['derived']['r0_band_q50']!r} and "
        f"r0_canonical={joint['derived']['r0_canonical']!r} are ell(X) recoveries built from "
        f"tr['f_truth'] (joint_drop_count_validation.py:146-148), NOT from dndx_total. "
        f"Correct values against the z-masked truth: {CORRECTED_R0_BAND_Q50} and "
        f"{CORRECTED_R0_CANONICAL}. Either re-derive the artifact or mark the ell rows "
        f"INVALIDATED_PENDING_REDERIVE — do not stamp them CLEAN. "
        f"See CRITICAL_FINDINGS_B16_BLAST_RADIUS.md correction [C1]/[C5].")


def test_docstring_row_counts_are_recorded():
    """Pin the measured z-leak on the [17.2,19.5) truth rows so the magnitude is not lost."""
    n_all, n_win = EXPECT_LEAK_ROWS
    assert n_all > n_win
    # the leak factor is exactly the row-count ratio: same X_sum in both numerator and
    # denominator, so it cancels.
    assert n_all / n_win == pytest.approx(EXPECT_LEAK, rel=2e-6)


# ===========================================================================
# The sub-DLA per-bin rows: a SECOND mislabelled B16 blast radius
# ===========================================================================
"""
The b16 stamp on ``subdla_mock_validation{,_forward}.json`` says::

    "per_bin.*[*].f_tru (CONTAMINATED -- it IS f_truth)",
    "integrated.*/dndx_* and per_bin.*[*].dndx_* (CLEAN)",

The FIRST line is right.  The SECOND is WRONG, and it is wrong in the DANGEROUS
direction: it declares clean a row that is not.

``subdla_loa0_validation.py::run_mode`` reads ONE truth object and derives both
per-bin columns from it::

    :208   f_tru   = np.asarray(base["t0"]["f_truth"], float)     # LEAKY
    :217   ft      = np.nansum(f_tru[sel])                        # -> per_bin f_tru
    :219   dndx_t  = np.nansum(f_tru[sel] * dN_b[sel])            # -> per_bin dndx_tru
    :220   r0      = dndx_e / dndx_t                              # -> per_bin r0

so per-bin ``dndx_tru`` is a UNIT CONVERSION of the leaky ``f_truth`` (``sel`` picks
exactly one fine bin, so ``dndx_tru == f_tru * dN_b`` identically), and per-bin ``r0``
inherits it.  Only the INTEGRATED ``dndx_*`` come from ``t0["dndx_total"]``, which
carries the ``t_zidx >= 0`` mask and IS clean -- which is why the artifact's own
post-fix re-derivation records ``dndx_tru_195_203_UNCHANGED``.

A prior reading of this ran the inference BACKWARDS -- it saw the exact identity
``f_tru == dndx_tru / (10**bhi - 10**blo)`` and concluded that ``f_tru`` is a harmless
unit conversion of a clean ``dndx_tru``, therefore quotable.  The identity is
symmetric and settles nothing about direction.  What settles it is the CLOSURE TEST
below: summing the 8 per-bin rows over [19.5,20.3) must reproduce the integrated
value for that band.  On the ESTIMATOR side it does, to 1e-16.  On the TRUTH side it
overshoots by 5.64% -- the B16 leak.  Same N support, same X_sum, same file; the only
difference is the z mask.
"""

SUBDLA_JSONS = ("subdla_mock_validation_forward.json", "subdla_mock_validation.json")
SUBDLA_BAND = (19.5, 20.3)
#: sum(per_bin dndx_tru) / integrated dndx_tru_195_203, as committed (leaky per-bin).
EXPECT_PERBIN_TRUTH_LEAK = 1.056365947727909
#: the CLEAN band truth from t0["dndx_total"], recorded UNCHANGED by the post-fix
#: re-derivation in metadata.b16.rederived_post_b16.loa0.
CLEAN_DNDX_TRU_195_203 = 0.09272816200828467


def _subdla(name):
    return _load(os.path.join(_HBI, name))


@pytest.mark.parametrize("name", SUBDLA_JSONS)
def test_subdla_per_bin_f_tru_and_dndx_tru_are_the_same_object(name):
    """PIN THE DERIVATION IDENTITY: per-bin dndx == f * (10**bhi - 10**blo), exactly,
    on BOTH the truth and the estimator column.  One fine bin per row, so this is
    arithmetic, not a coincidence -- and it holds regardless of which of the two is
    leaky, so it can NEVER be used on its own to argue either is clean."""
    doc = _subdla(name)
    n = 0
    for mode, rows in doc["per_bin"].items():
        for r in rows:
            width = 10.0 ** r["bhi"] - 10.0 ** r["blo"]
            for fk, dk in (("f_tru", "dndx_tru"), ("f_est", "dndx_est")):
                assert r[dk] == pytest.approx(r[fk] * width, rel=1e-11), (
                    f"{name}:{mode} [{r['blo']},{r['bhi']}) {fk}/{dk}")
                n += 1
    assert n >= 32, f"{name}: only {n} per-bin comparisons"


@pytest.mark.parametrize("name", SUBDLA_JSONS)
def test_subdla_per_bin_ESTIMATOR_closes_on_the_integrated_estimator(name):
    """CONTROL.  The estimator column sums across the 8 rows to the integrated
    estimator to machine precision.  This is what rules out a support/binning
    explanation for the truth-side mismatch below: same rows, same band, same file."""
    doc = _subdla(name)
    for mode, rows in doc["per_bin"].items():
        s = sum(r["dndx_est"] for r in rows)
        integ = doc["integrated"][mode]["dndx_est_195_203"]
        assert s == pytest.approx(integ, rel=1e-12), f"{name}:{mode}"


@pytest.mark.parametrize("name", SUBDLA_JSONS)
def test_subdla_per_bin_TRUTH_does_not_close_and_that_is_the_leak(name):
    """The decisive test.  sum(per-bin dndx_tru) over [19.5,20.3) must equal the
    integrated dndx_tru_195_203.  It does not: it is high by the B16 z-leak, because
    the per-bin column is f_truth-derived and the integrated one is dndx_total-derived.

    Written to stay GREEN across a future re-derivation: post-fix the ratio becomes
    1.0 and the artifact is consistent.  What is HARD-PINNED in both worlds is the
    clean band truth, which the fix leaves UNCHANGED.
    """
    doc = _subdla(name)
    ratios = []
    for mode, rows in doc["per_bin"].items():
        integ = doc["integrated"][mode]["dndx_tru_195_203"]
        assert integ == pytest.approx(CLEAN_DNDX_TRU_195_203, rel=1e-12), (
            f"{name}:{mode}: the CLEAN band truth moved -- that is a real change, "
            "not a stamp issue; do not edit this constant to make it pass")
        ratios.append(sum(r["dndx_tru"] for r in rows) / integ)
    assert len(set(round(x, 12) for x in ratios)) == 1, (
        f"{name}: truth is FP-mode independent, so every mode must give one ratio: "
        f"{ratios}")
    ratio = ratios[0]
    if ratio == pytest.approx(1.0, rel=1e-9):
        pytest.skip(f"{name}: per-bin truth re-derived post-B16 (ratio 1.0) -- "
                    "the stamp correction below still applies")
    assert ratio == pytest.approx(EXPECT_PERBIN_TRUTH_LEAK, rel=1e-9), (
        f"{name}: per-bin/integrated truth ratio {ratio!r}")
    assert ratio > 1.0, "the leak inflates truth; a deficit would be a different bug"


@pytest.mark.parametrize("name", SUBDLA_JSONS)
def test_subdla_b16_stamp_does_not_declare_the_per_bin_dndx_clean(name):
    """THE CORRECTION.  The stamp must not describe per-bin dndx / r0 as CLEAN: that
    wrongly certifies a contaminated row as quotable.  Per-bin f_tru IS f_truth and
    the stamp already says so; per-bin dndx_tru and r0 are derived from it and must
    say so too, while the INTEGRATED dndx stays correctly marked clean."""
    keys = _subdla(name)["metadata"]["b16"]["affected_keys"]
    # Read the VERDICT, not any occurrence of the word: a corrected line legitimately
    # says "...previously listed CLEAN, which is WRONG...". The verdict is the FIRST
    # verdict token on the line. (Same disclaimer-vs-scanner trap as c596ff7.)
    def verdict(line):
        hits = [(line.find(w), w) for w in ("CONTAMINATED", "UNAFFECTED", "CLEAN")
                if line.find(w) >= 0]
        return min(hits)[1] if hits else None

    verdicts = {k: verdict(k) for k in keys}
    assert None not in verdicts.values(), f"{name}: unlabelled key line: {verdicts}"

    # the true statements must survive
    assert any("f_tru" in k and v == "CONTAMINATED" for k, v in verdicts.items()), keys
    assert any("integrated" in k and "dndx_tru" in k and v == "CLEAN"
               for k, v in verdicts.items()), keys
    # the false one must be gone: no line may give a per_bin TRUTH dndx / r0 a
    # non-CONTAMINATED verdict, and one line must name them.
    named = 0
    for k, v in verdicts.items():
        if "per_bin" in k and ("dndx_tru" in k or "].r0" in k):
            named += 1
            assert v == "CONTAMINATED", (
                f"{name}: per-bin dndx_tru is f_truth*dN_b "
                f"(subdla_loa0_validation.py:219) and overshoots the clean band truth "
                f"by {EXPECT_PERBIN_TRUTH_LEAK:.4f}x -- verdict {v!r} is wrong: {k!r}")
        # the blanket pre-correction wording, which swept per_bin into the CLEAN line
        assert not ("per_bin" in k and "dndx_*" in k and v == "CLEAN"), (
            f"{name}: blanket 'per_bin ... dndx_* (CLEAN)' is back: {k!r}")
    assert named >= 1, (
        f"{name}: the stamp must NAME per-bin dndx_tru and r0, which inherit the leak "
        f"through their denominator: {keys}")


def test_lls_per_bin_dndx_really_is_clean_and_stays_marked_so():
    """CONTRAST + guard against over-correcting.  The LLS routine builds its per-bin
    dndx from t0['dndx_total'] differences (lls_loa0_validation.py:134-139), NOT from
    f_truth, and its artifact carries no per-bin f_tru row at all -- so its 'dndx
    CLEAN' line is TRUE and must not be swept up by the correction above."""
    doc = _load(LLS_JSON)
    keys = doc["metadata"]["b16"]["affected_keys"]
    assert any("dndx" in k and "CLEAN" in k for k in keys), keys
    for mode in doc["results"].values():
        for path in ("v1", "v3x"):
            if path not in mode or not isinstance(mode[path], dict):
                continue
            for r in mode[path].get("per_bin", []):
                assert "f_tru" not in r, "LLS per-bin gained an f_tru row; re-audit"
