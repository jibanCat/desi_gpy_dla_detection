"""VALIDATION-ONLY / DIAGNOSTIC-ONLY.

Convergence estimators for the low-z HBI CP-3 convergence-methods study
(OPUS-3, 2026-09-11).  Nothing here is production code, nothing here is
imported by any pipeline, and nothing here writes outside
/home/mfho/lowz_clean_work_2026-09-11/phase3_convergence/.

Definitions are the ones fixed in PREDECLARATION.md §2:
  rhat_unsplit_asrun   - the exact formula at cc_real_posterior.py:155-170
  rhat_split_classic   - classic Gelman-Rubin on 2m half-chains
  rhat_rank            - rank-normalised split-Rhat (Vehtari+2021)
  rhat_fold            - folded rank-normalised split-Rhat
  rhat_true            - max(rhat_rank, rhat_fold)
  ess_bulk / ess_tail  - Vehtari+2021 sec.3 (Geyer initial monotone positive)
  mcse_median          - Vehtari+2021 sec.4
  geweke_z             - Geweke (1992) first 10% vs last 50%
  halfdiff_mad         - (median 2nd half - median 1st half)/MAD, per chain
  ebfmi                - Betancourt (2016) per chain
"""
import numpy as np
import xarray as xr
from scipy.stats import norm
import arviz as az

# ArviZ 0.23.4 is the REFERENCE implementation of Vehtari et al. (2021).
# It is used for rhat_true / rhat_folded / ess_bulk / ess_tail / mcse_median;
# the hand-written versions below are retained and reported as an independent
# cross-check (see xcheck()).


def _ds(x):
    return xr.Dataset({"v": (("chain", "draw"), np.asarray(x, float))})


def az_rhat_true(x):
    """max(rank-normalised split-Rhat, folded rank-normalised split-Rhat)."""
    return float(az.rhat(_ds(x), method="rank").v)


def az_rhat_folded(x):
    return float(az.rhat(_ds(x), method="folded").v)


def az_ess_bulk(x):
    return float(az.ess(_ds(x), method="bulk").v)


def az_ess_tail(x):
    return float(az.ess(_ds(x), method="tail").v)


def az_mcse_median(x):
    return float(az.mcse(_ds(x), method="median").v)


# --------------------------------------------------------------- helpers
def _split(x):
    """(m, n) -> (2m, n//2) half-chains."""
    x = np.asarray(x, float)
    m, n = x.shape
    h = n // 2
    return np.concatenate([x[:, :h], x[:, n - h:]], axis=0)


def _gr(x):
    """Classic Gelman-Rubin on the chains AS GIVEN (no splitting)."""
    x = np.asarray(x, float)
    m, n = x.shape
    if m < 2 or n < 2:
        return float("nan")
    W = x.var(axis=1, ddof=1).mean()
    if not np.isfinite(W) or W <= 0:
        return float("nan")
    B = x.mean(axis=1).var(ddof=1) * n
    return float(np.sqrt(((n - 1) / n * W + B / n) / W))


def rhat_unsplit_asrun(x):
    """EXACTLY cc_real_posterior.py:155-170 (B/n added, not B)."""
    x = np.asarray(x, float)
    m, n = x.shape
    if m < 2:
        return None
    W = x.var(axis=1, ddof=1).mean()
    Bv = x.mean(axis=1).var(ddof=1) * n
    return float(np.sqrt(((n - 1) / n * W + Bv / n) / W))


def _rank_normalise(x):
    """Average-rank then inverse-normal (Vehtari+2021 eq. 13-14)."""
    x = np.asarray(x, float)
    flat = x.ravel()
    order = flat.argsort(kind="stable")
    ranks = np.empty(flat.size, float)
    ranks[order] = np.arange(1, flat.size + 1, dtype=float)
    # average ties
    uniq, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
    if cnt.max() > 1:
        sums = np.zeros(uniq.size)
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
    S = flat.size
    z = norm.ppf((ranks - 3.0 / 8.0) / (S - 0.25))
    return z.reshape(x.shape)


def rhat_rank(x):
    return _gr(_rank_normalise(_split(x)))


def rhat_fold(x):
    x = np.asarray(x, float)
    folded = np.abs(x - np.median(x))
    return _gr(_rank_normalise(_split(folded)))


def rhat_split_classic(x):
    return _gr(_split(x))


def rhat_true(x):
    a, b = rhat_rank(x), rhat_fold(x)
    return float(np.nanmax([a, b]))


# --------------------------------------------------------------- ESS
def _ess_raw(x):
    """ESS of the (already split / already transformed) chains, Vehtari sec.3."""
    x = np.asarray(x, float)
    m, n = x.shape
    if m < 2 or n < 4:
        return float("nan")
    if np.allclose(x, x.flat[0]):
        return float(m * n)
    # per-chain autocovariance by FFT
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    acov = np.empty((m, n))
    for i in range(m):
        c = x[i] - x[i].mean()
        F = np.fft.rfft(c, nfft)
        a = np.fft.irfft(F * np.conjugate(F), nfft)[:n]
        acov[i] = a / n
    chain_var = acov[:, 0] * n / (n - 1.0)
    W = chain_var.mean()
    B = x.mean(axis=1).var(ddof=1) * n
    var_plus = (n - 1.0) / n * W + B / n
    if var_plus <= 0:
        return float("nan")
    rho_t = np.empty(n)
    rho_t[0] = 1.0
    for t in range(1, n):
        rho_t[t] = 1.0 - (W - (acov[:, t] * n / (n - 1.0)).mean()) / var_plus
    # Geyer initial positive sequence on the paired sums
    t = 1
    P = []
    while t + 1 < n:
        p = rho_t[t] + rho_t[t + 1]
        if p <= 0:
            break
        P.append(p)
        t += 2
    if not P:
        return float(m * n)
    P = np.array(P)
    # initial monotone: enforce non-increasing
    for i in range(1, P.size):
        if P[i] > P[i - 1]:
            P[i] = P[i - 1]
    tau = -1.0 + 2.0 * P.sum()
    tau = max(tau, 1.0 / np.log10(m * n))
    return float(m * n / tau)


def ess_bulk(x):
    return _ess_raw(_rank_normalise(_split(x)))


def ess_tail(x):
    x = np.asarray(x, float)
    q05, q95 = np.quantile(x, [0.05, 0.95])
    lo = _ess_raw(_split((x <= q05).astype(float)))
    hi = _ess_raw(_split((x >= q95).astype(float)))
    return float(np.nanmin([lo, hi]))


def _ess_quantile(x, p):
    x = np.asarray(x, float)
    q = np.quantile(x, p)
    return _ess_raw(_split((x <= q).astype(float)))


def mcse_median(x):
    """Vehtari+2021 sec.4.3: MCSE of the median via the ESS of the median
    indicator, propagated through the empirical quantile function."""
    x = np.asarray(x, float)
    S = x.size
    ess = _ess_quantile(x, 0.5)
    if not np.isfinite(ess) or ess <= 0:
        return float("nan")
    a = 0.5
    # Vehtari's Bayesian bootstrap-free approximation
    p = norm.cdf([-1.0, 1.0]) - 0.5          # +-1 s.e. of Beta(a*ess, (1-a)*ess)
    sd = np.sqrt(a * (1 - a) / ess)
    lo = max(0.0, a - sd)
    hi = min(1.0, a + sd)
    xs = np.sort(x)
    ql = np.interp(lo, (np.arange(S) + 0.5) / S, xs)
    qh = np.interp(hi, (np.arange(S) + 0.5) / S, xs)
    _ = p
    return float((qh - ql) / 2.0)


# --------------------------------------------------------------- stationarity
def _spectral_var0(c):
    """Spectral density at frequency 0 via Geyer's initial positive sequence."""
    c = np.asarray(c, float)
    n = c.size
    y = c - c.mean()
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    F = np.fft.rfft(y, nfft)
    a = np.fft.irfft(F * np.conjugate(F), nfft)[:n] / n
    if a[0] <= 0:
        return 0.0
    rho = a / a[0]
    t, P = 1, []
    while t + 1 < n:
        p = rho[t] + rho[t + 1]
        if p <= 0:
            break
        P.append(p)
        t += 2
    tau = 1.0 + 2.0 * (np.sum(P) if P else 0.0)
    return float(a[0] * tau / n)


def geweke_z(chain, first=0.1, last=0.5):
    chain = np.asarray(chain, float)
    n = chain.size
    a = chain[: int(first * n)]
    b = chain[int((1 - last) * n):]
    va, vb = _spectral_var0(a), _spectral_var0(b)
    d = va + vb
    if d <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(d))


def halfdiff_mad(chain):
    chain = np.asarray(chain, float)
    n = chain.size
    h = n // 2
    mad = np.median(np.abs(chain - np.median(chain)))
    if mad <= 0:
        return float("nan")
    return float((np.median(chain[n - h:]) - np.median(chain[:h])) / mad)


def ebfmi(energy):
    e = np.asarray(energy, float)
    if e.size < 3:
        return float("nan")
    v = e.var(ddof=1)
    if v <= 0:
        return float("nan")
    return float(np.sum(np.diff(e) ** 2) / ((e.size - 1) * v))


# --------------------------------------------------------------- driver
def all_diagnostics(x, energy=None):
    """x : (m, n) per-chain scalar matrix."""
    x = np.asarray(x, float)
    m, n = x.shape
    out = dict(
        n_chains=int(m), n_draws_per_chain=int(n),
        rhat_unsplit_asrun=rhat_unsplit_asrun(x),
        rhat_split_classic=rhat_split_classic(x),
        rhat_rank_bulkonly=rhat_rank(x),
        rhat_fold=az_rhat_folded(x),
        rhat_true=az_rhat_true(x),
        ess_bulk=az_ess_bulk(x),
        ess_tail=az_ess_tail(x),
        mcse_median=az_mcse_median(x),
        xcheck_rhat_true_own=rhat_true(x),
        xcheck_ess_bulk_own=ess_bulk(x),
        geweke_z_max=float(np.nanmax([abs(geweke_z(c)) for c in x])),
        halfdiff_mad_max=float(np.nanmax([abs(halfdiff_mad(c)) for c in x])),
        chain_medians=[float(np.median(c)) for c in x],
        pooled_q16_q50_q84=[float(v) for v in np.percentile(x, [16, 50, 84])],
    )
    if energy is not None:
        E = np.asarray(energy, float)
        out["ebfmi_per_chain"] = [round(ebfmi(e), 4) for e in E]
        out["ebfmi_min"] = float(np.nanmin(out["ebfmi_per_chain"]))
    return out
