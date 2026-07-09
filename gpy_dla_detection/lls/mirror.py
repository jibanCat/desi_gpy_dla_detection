"""lyc_make_mirror_mock.py — M1: a MIRROR 2LPT mock with the Lyman-limit drop added.

The 2LPT spectra carry quickquasars' HCD absorption as Lyman-SERIES lines only (no bound-free
912 A break; source-verified). Since optical depth is additive (tau_total = tau_lines + tau_LL),
re-doing the HCD absorption with the improved (break-aware) Voigt is IDENTICAL to just
multiplying each existing spectrum by exp(-tau_LL) for its truth HCDs. So we do exactly that —
NO quickquasars re-run — and write a mirror spectra-16 tree the GP finder reads unchanged. Then
the break-aware finder (SubDLAGPMATLymanBreak) can be re-inferred on the mirror (M3) and its LLS
purity compared to the line-only control (M4).

Noise propagation (default, `--legacy-noise` OFF).  A quickquasars pixel is
    F_obs = S + n,   n ~ N(0, sigma^2),   sigma^2 = 1/ivar,
i.e. the reported flux is the true source S plus ONE noise realization whose variance the pixel's
ivar advertises.  The naive `flux *= T` (the legacy path) yields `T*S + T*n`: the SIGNAL is
attenuated correctly but the noise realization is attenuated too, so the absorbed region is
artificially quiet (true variance T^2/ivar) while ivar still advertises 1/ivar — a break-aware
finder then sees an unrealistically clean drop.  We fix this by regenerating statistically correct
noise consistent with ivar:
    F_new = T*F_obs + eps,   eps ~ N(0, (1 - T^2)*sigma^2)  (independent of n),
so that E[F_new] = T*S (signal correctly attenuated) and Var(F_new) = T^2*sigma^2 + (1-T^2)*sigma^2
= sigma^2 = 1/ivar (noise consistent with the UNCHANGED ivar).  ivar is left byte-identical; only
FLUX is rewritten.  eps is drawn from a per-(seed, TARGETID, camera) RNG so the mirror is
bit-reproducible for a fixed --seed and independent of how many healpix are processed.
    ASSUMPTION (measured, not assumed): at the DESI blue edge (~3600-4300 A) the pixel noise is
SKY/READ-dominated.  Regressing var = A_sky + B*S in 50 A bins (so the sky floor is held fixed) on
2LPT-0 loa-124 gives f_src = B*S/(A+B*S) = 2-9% AT THE MEDIAN FOREST FLUX (var tracks wavelength/
sky: corr(var,wave) ~ -0.6, corr(var,S) ~ +0.1; a ~30x flux swing moves var by only 8-15%).  So
attenuating the source leaves the pixel variance (=1/ivar) essentially unchanged, which is why we
keep ivar and only restore the noise realization.
    ATTACH THE FLUX LEVEL TO f_src.  It is 2-6% on the faint forest pixels where LLS breaks land,
~13-16% at BRIGHT (continuum-level) forest pixels, and a global affine fit over the whole blue band
confounds the steep sky-vs-wavelength trend into B and inflates it.  Neglecting the source term
means we restore var to A + B*S instead of the true A + B*(T*S), i.e. the mirror is
        var_skyfix / var_target - 1  =  B*S*(1-T) / (A + B*T*S)   >=  0   ALWAYS.
It therefore OVER-noises and NEVER under-noises: 0.4-3.6% where breaks actually sit, worst case
~27% in the (bright pixel) x (saturated break) corner.  Over-noising can only SUPPRESS marginal
detections, so completeness/purity measured here are LOWER BOUNDS -- the safe direction for a
detection claim.  Note the opposite sign for an INCIDENCE: ell ~ D/(C*exposure), so an
under-estimated completeness C inflates ell.  That residual is second order because the
over-noising is largest exactly where S is large, i.e. on bright/high-S/N sightlines where C
already saturates.  A strictly source-aware run would set var_new = A(lambda) + T*B(lambda)*S and
ivar_new = 1/var_new; that needs per-wavelength (A,B) (the sky floor triples below 3800 A) i.e. the
DESI ETC, not recoverable from the mock -- hence the sky-dominated form is the production choice.
    NOISE STRUCTURE.  Corr(F_base, F_mirror) = T exactly (the noise realization is REUSED, which is
mandatory: at T=1 the mirror must be byte-identical to the base).  Any paired base-vs-mirror
analysis must therefore use PAIRED (sightline-bootstrap) errors, not independent ones.  Also, since
eps is Gaussian and the 4th cumulant is additive, excess_kurt(mirror) = T^4 * excess_kurt(base):
the mirror is THINNER-tailed than heavy real noise, so per-pixel false-positive rates on it are
optimistic, while a matched filter over >~5 px is well calibrated (realized 3.5-sigma FP is
0.90-0.99x nominal).
    USABLE BLUE CUTOFF.  Pixels exist to 3600 A, but the sky floor triples below ~3800 A: a
best-case matched filter reaches median S/N >= 5 only for break edges >~ 3660 A (logN >= 17.5) or
>~ 3710 A (logN = 17.2), i.e. z_abs >~ 3.0-3.07.  Do not claim usable breaks below z_abs ~ 3.0.
    BETA / CIRCULARITY.  The injection uses sigma ~ nu^-beta with beta = BETA_LL = 3.0 by default,
which is the classical Kramers ASYMPTOTE, not the near-threshold slope (exact: 8/3 = 2.667; the
effective index over the observable wing is 2.68-2.71).  The FROZEN finder
`gpy_dla_detection/voigt_lls.py::tau_LLS_break` hard-wires (lambda_rest/912)^3, so a beta=3 mirror
hands the finder EXACTLY its own break shape and any purity/completeness calibrated on it is blind
to the ~2.6% wing mismatch that real breaks carry.  Pass ``--beta 2.70`` (or 2.75) to inject the
physical shape and de-circularize the calibration.  The default stays 3.0 only because the stamped
mock closures are self-consistent with a beta=3 injection.
    Masked pixels (ivar<=0): left UNTOUCHED in the new path (no defined variance to restore).  T->0
(fully absorbed): F_new -> eps ~ N(0, sigma^2), i.e. pure noise at the advertised sigma (no NaN, no
negative variance).  T==1 (above the observed limit): F_new == F_obs exactly (eps==0).
    DOWNSTREAM INVALIDATION: the break-aware Tier-1 validations run on the OLD (legacy-noise) mirror
(SLURM 52949574 / 52950580) are calibrated on artificially quiet below-break noise and MUST be
re-run on a mirror built with the default (noise-corrected) path.
    Legacy behaviour (`--legacy-noise`): the original `flux *= T` on every pixel (masked included),
kept byte-identical so prior mirror mocks remain reproducible.
Caveat 2 (sigma_912 convention): the injector (CDDF_analysis.lyc, sigma_912=6.35e-18 Verner+1996)
and the model the finder fits (voigt_lls.tau_LLS_break, sigma_912=1/10^17.2=6.31e-18, the LLS
threshold definition) differ by a flat 0.64% in tau -> ~0.0028 dex in recovered logN. Negligible
vs the ~0.06 dex prior-edge bias; for a production-grade mirror, align the two to 1/10^17.2 so the
injected break equals the fitted break exactly.

Run (subset): python -m gpy_dla_detection.lls.mirror --limit-healpix 2 --out /scratch/.../mirror
"""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.table import Table

_REPO = Path(__file__).resolve().parents[2]  # gpy_dla_detection/lls/ -> repo root
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from preload_spectra.preload_2lpt_simple import _spec_path
from CDDF_analysis.lyc import lyc_transmission, BETA_LL

DEF_MOCK = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124"

# camera -> stable index for per-sightline RNG seeding (keeps B/R/Z noise independent)
_CAM_INDEX = {"B": 0, "R": 1, "Z": 2}


def row_rng(seed: int, targetid: int, cam_index: int) -> np.random.Generator:
    """Deterministic per-(seed, TARGETID, camera) RNG.

    Deriving the stream from the (targetid, camera) key — not from a single global generator
    advanced in iteration order — makes each sightline's injected noise bit-reproducible AND
    independent of which/how-many healpix are processed (so a subset run matches the full run).
    """
    return np.random.default_rng([int(seed), int(targetid), int(cam_index)])


def attenuate_with_noise(flux, ivar, transmission, rng):
    """Attenuate flux by ``transmission`` (T=exp(-tau_LL)) with STATISTICALLY CORRECT noise.

    Sky/read-dominated regime (measured; see module docstring): the pixel variance 1/ivar is
    independent of the source, so the mirror flux is
        F_new = T*F_obs + eps,   eps ~ N(0, (1 - T^2)/ivar),
    which has mean T*S (signal attenuated) and variance 1/ivar (noise consistent with the
    UNCHANGED ivar).  ivar is not modified.  Edge handling:
      * ivar<=0 or non-finite (masked): pixel left UNTOUCHED.
      * T>=1 (above the observed Lyman limit) or T not finite treated as 1: pixel unchanged
        (eps term vanishes since 1-T^2=0).
      * T->0: F_new -> eps ~ N(0, 1/ivar); never NaN, variance never negative.
    ``rng.standard_normal`` is drawn over the FULL pixel grid (fixed length) so the stream depends
    only on the array size, keeping the result deterministic for a given ``rng``.
    """
    flux = np.asarray(flux, float)
    ivar = np.asarray(ivar, float)
    T = np.asarray(transmission, float)
    T = np.where(np.isfinite(T), np.clip(T, 0.0, 1.0), 1.0)

    good = np.isfinite(ivar) & (ivar > 0) & np.isfinite(flux)
    sigma2 = np.zeros_like(flux)
    sigma2[good] = 1.0 / ivar[good]
    extra_var = np.clip(1.0 - T * T, 0.0, None) * sigma2        # >= 0 always
    draw = rng.standard_normal(flux.shape)                       # full-grid, size-only dependence
    eps = draw * np.sqrt(extra_var)

    out = flux.copy()
    out[good] = T[good] * flux[good] + eps[good]                 # masked pixels stay untouched
    return out


def load_hcd_by_tid(mockdir: Path, nhi_min=17.2):
    hcd = Table.read(mockdir / "hcd_truth_cat.fits")
    N = np.asarray(hcd["NHI"], float); z = np.asarray(hcd["Z"], float); tid = np.asarray(hcd["TARGETID"])
    keep = N >= nhi_min
    N, z, tid = N[keep], z[keep], tid[keep]
    order = np.argsort(tid); tid, z, N = tid[order], z[order], N[order]
    uniq, start = np.unique(tid, return_index=True); end = np.r_[start[1:], len(tid)]
    return {int(t): (z[s:e], N[s:e]) for t, s, e in zip(uniq, start, end)}


def _healpix_list(mockdir: Path, limit):
    base = mockdir / "spectra-16"
    hp = []
    for grp in sorted(base.iterdir()):
        if grp.is_dir():
            for hd in sorted(grp.iterdir()):
                if hd.is_dir():
                    hp.append(int(hd.name))
    hp = sorted(hp)
    return hp[:limit] if limit else hp


def _code_commit():
    """HEAD, with a `-dirty` marker when the tree has uncommitted changes: a stamp taken over a
    dirty tree must be self-evidently provisional, not silently wrong."""
    try:
        return subprocess.check_output(["git", "describe", "--always", "--dirty"],
                                       cwd=Path(__file__).resolve().parents[2],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def inject_file(specfile: Path, out_file: Path, hcd_by_tid, zq_of,
                legacy_noise: bool = False, seed: int = 0, beta: float = BETA_LL):
    """Copy spectra-16 file and inject the Lyman-limit drop into B/R/Z_FLUX per target's HCDs.

    Default (legacy_noise=False): attenuate flux by exp(-tau_LL) AND regenerate statistically
    correct noise consistent with the (unchanged) ivar — see ``attenuate_with_noise`` / module
    docstring.  legacy_noise=True reproduces the original ``flux *= T`` on every pixel byte-for-byte.
    ivar is never modified.  The noise draw uses a per-(seed, TARGETID, camera) RNG so output is
    bit-reproducible for a fixed ``seed``.

    ``beta`` is the cross-section index (sigma ~ nu^-beta).  Default BETA_LL=3.0 matches the frozen
    finder's hard-wired ^3 — pass 2.70 to inject the physical near-threshold shape and break that
    circularity (see module docstring).

    The output primary header is STAMPED (LYCNOISE / LYCSEED / LYCBETA / LYCCOMM) so a mirror mock
    on disk is never ambiguous between the legacy and the noise-corrected build.
    """
    os.makedirs(out_file.parent, exist_ok=True)
    shutil.copy2(specfile, out_file)
    n_inj = 0
    with fits.open(out_file, mode="update") as hdul:
        h = hdul[0].header
        h["LYCMIRR"] = (True, "Lyman-continuum mirror mock (gpy_dla_detection.lls.mirror)")
        h["LYCNOISE"] = ("legacy" if legacy_noise else "corrected",
                         "legacy=flux*T (noise attenuated); corrected=+eps")
        h["LYCSEED"] = (int(seed), "RNG seed for the noise-restoration draw")
        h["LYCBETA"] = (float(beta), "sigma ~ nu^-beta cross-section index")
        h["LYCCOMM"] = (_code_commit()[:60], "code commit that built this mirror")
        tids = np.asarray(hdul["FIBERMAP"].data["TARGETID"])
        for cam in ("B", "R", "Z"):
            fk, wk, ik = f"{cam}_FLUX", f"{cam}_WAVELENGTH", f"{cam}_IVAR"
            if fk not in hdul or wk not in hdul:
                continue
            wave = np.asarray(hdul[wk].data, float)
            flux = hdul[fk].data
            has_ivar = ik in hdul
            ivar = np.asarray(hdul[ik].data, float) if has_ivar else None
            if not legacy_noise and not has_ivar:
                print(f"  [warn] {out_file.name} {cam}_IVAR missing; falling back to legacy "
                      f"(noise-attenuated) injection for this camera")
            cam_index = _CAM_INDEX.get(cam, 0)
            for row, t in enumerate(tids):
                t = int(t)
                if t not in hcd_by_tid:
                    continue
                zk, nk = hcd_by_tid[t]
                zqso = zq_of.get(t, None)
                if zqso is not None:
                    m = zk < zqso
                    zk, nk = zk[m], nk[m]
                if zk.size == 0:
                    continue
                T = lyc_transmission(wave, zk, nk, beta=beta)
                if legacy_noise or not has_ivar:
                    flux[row] = flux[row] * T
                else:
                    rng = row_rng(seed, t, cam_index)
                    flux[row] = attenuate_with_noise(flux[row], ivar[row], T, rng)
                if cam == "B":
                    n_inj += 1
            hdul[fk].data = flux
        hdul.flush()
    return n_inj


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mockdir", default=DEF_MOCK)
    ap.add_argument("--out", required=True, help="mirror mock root (a new dir; spectra-16 written under it)")
    ap.add_argument("--limit-healpix", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0,
                    help="master RNG seed for the noise regeneration (bit-reproducible per seed)")
    ap.add_argument("--beta", type=float, default=BETA_LL,
                    help="cross-section index sigma~nu^-beta. Default 3.0 matches the FROZEN "
                         "finder voigt_lls.tau_LLS_break (circular); pass 2.70 for the physical "
                         "near-threshold shape and a de-circularized calibration.")
    ap.add_argument("--legacy-noise", action="store_true",
                    help="reproduce the ORIGINAL flux*=exp(-tau_LL) (noise attenuated too, ivar "
                         "unchanged). Default OFF: regenerate statistically correct noise. Prior "
                         "OLD-mirror results (SLURM 52949574/52950580) used this legacy path.")
    args = ap.parse_args()
    M = Path(args.mockdir); OUT = Path(args.out)
    os.makedirs(OUT, exist_ok=True)
    # symlink the non-spectra products (zcat, truth cats) so the mirror is a usable mock dir
    for name in ("zcat.fits", "hcd_truth_cat.fits", "bal_cat.fits", "snr_cat.fits", "zcat_gauss_400.fits"):
        src = M / name; dst = OUT / name
        if src.exists() and not dst.exists():
            try:
                os.symlink(src, dst)
            except OSError:
                pass
    zc = Table.read(M / "zcat.fits")
    zq_of = dict(zip(np.asarray(zc["TARGETID"]).tolist(), np.asarray(zc["Z"], float).tolist()))
    hcd_by_tid = load_hcd_by_tid(M)
    print(f"[hcd] {len(hcd_by_tid)} sightlines carry >=1 HCD (logN>=17.2)")
    hplist = _healpix_list(M, args.limit_healpix)
    mode = "LEGACY (noise attenuated)" if args.legacy_noise else "noise-corrected"
    print(f"[mirror] injecting LyC drop [{mode}, seed={args.seed}] into {len(hplist)} healpix -> {OUT}")
    tot = 0
    for i, hp in enumerate(hplist):
        sf = _spec_path(M, int(hp))
        of = _spec_path(OUT, int(hp))
        if not sf.exists():
            continue
        n = inject_file(sf, of, hcd_by_tid, zq_of, beta=args.beta,
                        legacy_noise=args.legacy_noise, seed=args.seed)
        tot += n
        # copy the sibling truth-16 (needed for camera resolution + TRUE_CONT on read)
        tsrc = Path(str(sf).replace("spectra-16-", "truth-16-"))
        tdst = Path(str(of).replace("spectra-16-", "truth-16-"))
        if tsrc.exists() and not tdst.exists():
            try:
                os.symlink(tsrc, tdst)
            except OSError:
                shutil.copy2(tsrc, tdst)
        print(f"  [{i+1}/{len(hplist)}] healpix {hp}: injected break into {n} sightlines")
    print(f"[done] mirror mock at {OUT}; {tot} sightlines given a Lyman-limit break")


if __name__ == "__main__":
    main()
