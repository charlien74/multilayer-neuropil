"""
Fast discrete mutual-information estimation for many pairs of spike trains.

Design
------
* Spikes are discretised into per-bin integer states (counts clipped to an
  alphabet of size K, K=2 meaning binarised).
* Every entropy is a function of an integer count histogram only, so all
  per-count terms (n*log n, n*psi(n), n*G(n)) are pre-tabulated once with
  scipy and the numba kernel does nothing but table lookups and adds.
* One numba kernel computes, for each pair, the plug-in MI plus three
  analytic bias corrections and (optionally) a circular-shift null.

Estimators
----------
plugin  : maximum-likelihood / naive.  Biased upward by ~(Kxy-Kx-Ky+1)/(2N).
mm      : Miller-Madow. H += (m-1)/(2N), m = number of occupied states.
psi     : Grassberger 1988. H = psi(N) - (1/N) sum n_i psi(n_i).
          A smooth version of Miller-Madow (expands to it for large n_i).
grass   : Grassberger 2003. H = log N - (1/N) sum n_i G(n_i),
          G(n) = psi(n) + (-1)^n/2 * [psi((n+1)/2) - psi(n/2)].
          Best of the analytic corrections when many counts are small.

On top of those:
shift_correct()      : subtract the mean MI of a circular-shift null, and
                       return a permutation p-value. Most robust option for
                       pairwise MI, and gives you significance for free.
quadratic_extrap()   : Strong/Panzeri extrapolation of I(N) = I_inf + a/N + b/N^2
                       from subsets of the data.
learning_curve()     : the diagnostic you should always look at before
                       trusting any of the above.

Units are bits by default.

Quickstart
----------
>>> S = bin_population(spike_times_list, 0.0, 600.0, dt=0.005, clip=1)  # (n_neurons, n_bins)
>>> tab = CountTables(S.shape[1])          # build once, reuse for every call
>>> r = mi_pairs(S, lag=1, n_null=200, tables=tab)   # all upper-triangle pairs
>>> mi = r["mi_corrected"][:, METHODS.index("grass")]
>>> sig, q = bh_fdr(r["p"][:, METHODS.index("grass")])
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from scipy.special import digamma

LOG2 = np.log(2.0)

__all__ = [
    "bin_spikes",
    "bin_population",
    "CountTables",
    "mi_pairs",
    "shift_correct",
    "quadratic_extrap",
    "learning_curve",
    "METHODS",
]

METHODS = ("plugin", "mm", "psi", "grass")


# --------------------------------------------------------------------------
# discretisation
# --------------------------------------------------------------------------
def bin_spikes(spike_times, t_start, t_stop, dt, clip=1):
    """Bin one spike train into clipped integer counts.

    Parameters
    ----------
    spike_times : (n_spikes,) array of times, same units as dt
    clip        : max count per bin. clip=1 -> binary words.

    Returns
    -------
    (n_bins,) uint8 array of states in {0, ..., clip}
    """
    n_bins = int(np.floor((t_stop - t_start) / dt + 1e-9))
    st = np.asarray(spike_times, dtype=np.float64)
    st = st[(st >= t_start) & (st < t_start + n_bins * dt)]
    idx = ((st - t_start) / dt).astype(np.int64)
    counts = np.bincount(idx, minlength=n_bins)[:n_bins]
    return np.minimum(counts, clip).astype(np.uint8)


def bin_population(spike_time_list, t_start, t_stop, dt, clip=1):
    """Bin a list of spike trains -> (n_neurons, n_bins) uint8, C-contiguous."""
    rows = [bin_spikes(s, t_start, t_stop, dt, clip) for s in spike_time_list]
    return np.ascontiguousarray(np.stack(rows))


# --------------------------------------------------------------------------
# pre-tabulated per-count terms
# --------------------------------------------------------------------------
class CountTables:
    """Lookup tables for counts 0..n_max. Build once, reuse for every pair."""

    __slots__ = ("n_max", "nlogn", "npsi", "nG")

    def __init__(self, n_max: int):
        n = np.arange(n_max + 1, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.nlogn = np.where(n > 0, n * np.log(np.maximum(n, 1.0)), 0.0)
            psi = digamma(np.maximum(n, 1.0))
            self.npsi = np.where(n > 0, n * psi, 0.0)
            # Grassberger 2003 G(n)
            alt = np.where(n > 0, 0.5 * (-1.0) ** n *
                           (digamma((n + 1) / 2.0) - digamma(np.maximum(n, 1e-12) / 2.0)), 0.0)
            self.nG = np.where(n > 0, n * (psi + alt), 0.0)
        self.n_max = n_max

    def as_tuple(self):
        return self.nlogn, self.npsi, self.nG


# --------------------------------------------------------------------------
# numba kernels
# --------------------------------------------------------------------------
@njit(cache=True, fastmath=True, inline="always")
def _entropies(hist, N, psiN, logN, nlogn, npsi, nG):
    """Four entropy estimates (nats) from an integer histogram."""
    s_log = 0.0
    s_psi = 0.0
    s_G = 0.0
    m = 0
    for k in range(hist.shape[0]):
        c = hist[k]
        if c > 0:
            m += 1
            s_log += nlogn[c]
            s_psi += npsi[c]
            s_G += nG[c]
    inv = 1.0 / N
    h_plug = logN - s_log * inv
    h_mm = h_plug + (m - 1) * 0.5 * inv
    h_psi = psiN - s_psi * inv
    h_gr = logN - s_G * inv
    return h_plug, h_mm, h_psi, h_gr


@njit(cache=True, fastmath=True, inline="always")
def _mi_one(S, i, j, K, lag, shift, hist, mx, my, psiN, logN, nlogn, npsi, nG, out, row):
    """MI of neuron i at t vs neuron j at t+lag(+shift), written into out[row]."""
    T = S.shape[1]
    N = T - lag
    hist[:] = 0
    for t in range(N):
        u = t + lag + shift
        if u >= T:
            u -= T
        hist[S[i, t] * K + S[j, u]] += 1

    mx[:] = 0
    my[:] = 0
    for a in range(K):
        base = a * K
        for b in range(K):
            c = hist[base + b]
            mx[a] += c
            my[b] += c

    hx = _entropies(mx, N, psiN, logN, nlogn, npsi, nG)
    hy = _entropies(my, N, psiN, logN, nlogn, npsi, nG)
    hxy = _entropies(hist, N, psiN, logN, nlogn, npsi, nG)
    for e in range(4):
        out[row, e] = hx[e] + hy[e] - hxy[e]


@njit(cache=True, parallel=True, fastmath=True)
def _mi_pairs_kernel(S, pairs, K, lag, shifts, nlogn, npsi, nG,
                     mi, null_mean, null_sd, null_ge):
    """mi: (P,4). shifts: (P, n_null) int64, 0 columns allowed for no null."""
    P = pairs.shape[0]
    n_null = shifts.shape[1]
    T = S.shape[1]
    N = T - lag
    psiN = npsi[N] / N          # psi(N) recovered from the table
    logN = nlogn[N] / N         # log(N)

    for p in prange(P):
        i = pairs[p, 0]
        j = pairs[p, 1]
        hist = np.zeros(K * K, dtype=np.int64)
        mx = np.zeros(K, dtype=np.int64)
        my = np.zeros(K, dtype=np.int64)
        obs = np.zeros((1, 4), dtype=np.float64)

        _mi_one(S, i, j, K, lag, 0, hist, mx, my, psiN, logN, nlogn, npsi, nG, obs, 0)
        for e in range(4):
            mi[p, e] = obs[0, e]

        if n_null > 0:
            s1 = np.zeros(4, dtype=np.float64)
            s2 = np.zeros(4, dtype=np.float64)
            ge = np.zeros(4, dtype=np.int64)
            nul = np.zeros((1, 4), dtype=np.float64)
            for r in range(n_null):
                _mi_one(S, i, j, K, lag, shifts[p, r], hist, mx, my,
                        psiN, logN, nlogn, npsi, nG, nul, 0)
                for e in range(4):
                    v = nul[0, e]
                    s1[e] += v
                    s2[e] += v * v
                    if v >= obs[0, e]:
                        ge[e] += 1
            for e in range(4):
                mu = s1[e] / n_null
                null_mean[p, e] = mu
                var = s2[e] / n_null - mu * mu
                null_sd[p, e] = np.sqrt(var) if var > 0.0 else 0.0
                null_ge[p, e] = ge[e]


# --------------------------------------------------------------------------
# python API
# --------------------------------------------------------------------------
def _prep(states, pairs):
    S = np.ascontiguousarray(states, dtype=np.uint8)
    if S.ndim != 2:
        raise ValueError("states must be (n_neurons, n_bins)")
    if pairs is None:
        n = S.shape[0]
        iu = np.triu_indices(n, k=1)
        pairs = np.column_stack(iu)
    return S, np.ascontiguousarray(pairs, dtype=np.int64)


def mi_pairs(states, pairs=None, K=None, lag=0, n_null=0, tables=None,
             rng=None, min_shift=None, bits=True):
    """Bias-corrected MI for many pairs at once.

    Parameters
    ----------
    states  : (n_neurons, n_bins) uint8, output of bin_population
    pairs   : (P, 2) int array of neuron indices; default = all upper-triangle pairs
    K       : alphabet size; default states.max()+1
    lag     : MI( x_t , y_{t+lag} ), in bins
    n_null  : number of circular-shift surrogates per pair (0 = none)
    min_shift : minimum |shift| in bins; default n_bins//10. Must exceed the
              timescale of genuine dependence, or the null will absorb signal.

    Returns
    -------
    dict with 'mi', and if n_null>0 also 'null_mean', 'null_sd', 'p',
    'mi_corrected'. Each is (P, 4), columns in METHODS order.
    """
    S, pairs = _prep(states, pairs)
    n_neurons, T = S.shape
    if K is None:
        K = int(S.max()) + 1
    N = T - lag
    if N < 2:
        raise ValueError("lag too large")

    if tables is None:
        tables = CountTables(N)
    elif tables.n_max < N:
        raise ValueError("tables too small for this N")
    nlogn, npsi, nG = tables.as_tuple()

    P = pairs.shape[0]
    if n_null > 0:
        rng = np.random.default_rng(rng)
        lo = T // 10 if min_shift is None else int(min_shift)
        lo = max(1, min(lo, T // 2 - 1))
        shifts = rng.integers(lo, T - lo, size=(P, n_null)).astype(np.int64)
    else:
        shifts = np.zeros((P, 0), dtype=np.int64)

    mi = np.zeros((P, 4))
    null_mean = np.zeros((P, 4))
    null_sd = np.zeros((P, 4))
    null_ge = np.zeros((P, 4), dtype=np.int64)

    _mi_pairs_kernel(S, pairs, K, lag, shifts, nlogn, npsi, nG,
                     mi, null_mean, null_sd, null_ge)

    scale = 1.0 / LOG2 if bits else 1.0
    res = {"mi": mi * scale, "pairs": pairs, "N": N, "K": K,
           "methods": METHODS}
    if n_null > 0:
        res["null_mean"] = null_mean * scale
        res["null_sd"] = null_sd * scale
        res["mi_corrected"] = (mi - null_mean) * scale
        res["p"] = (1.0 + null_ge) / (1.0 + n_null)
    return res


def shift_correct(states, pairs=None, n_null=200, method="grass", **kw):
    """Convenience wrapper: shift-corrected MI and p-values for one estimator."""
    e = METHODS.index(method)
    r = mi_pairs(states, pairs, n_null=n_null, **kw)
    return r["mi_corrected"][:, e], r["p"][:, e], r["mi"][:, e]


def quadratic_extrap(states, pairs=None, fractions=(1.0, 0.5, 0.25),
                     method="plugin", deg=2, **kw):
    """Strong/Panzeri extrapolation: fit I_hat(N) = I_inf + a/N + b/N^2.

    Each fraction f is evaluated on all floor(1/f) disjoint contiguous blocks
    and averaged, which keeps local temporal structure intact.
    """
    S, pairs = _prep(states, pairs)
    e = METHODS.index(method)
    T = S.shape[1]
    tables = kw.pop("tables", None) or CountTables(T)

    inv_n, vals = [], []
    for f in fractions:
        nb = max(1, int(round(1.0 / f)))
        L = T // nb
        acc = np.zeros(pairs.shape[0])
        for b in range(nb):
            seg = np.ascontiguousarray(S[:, b * L:(b + 1) * L])
            acc += mi_pairs(seg, pairs, tables=tables, **kw)["mi"][:, e]
        vals.append(acc / nb)
        inv_n.append(1.0 / L)

    u = np.asarray(inv_n)
    Y = np.asarray(vals)                       # (n_fractions, P)
    coef = np.polyfit(u, Y, deg=min(deg, len(u) - 1))
    return coef[-1], Y, u                      # intercept = I_inf


def scan_bin_width(spike_time_list, t_start, t_stop, dts, pairs=None, clip=1,
                   lag_bins=0, n_null=100, method="grass", rng=0):
    """Sweep bin width and report everything needed to choose one.

    For each dt returns: MI per bin, the shift-null mean (= empirical bias floor),
    shift-corrected MI, MI *rate* in bits/s (the quantity comparable across dt),
    the sampling ratio N/K^2, and the fraction of bins that would be clipped.
    """
    recs = []
    for dt in dts:
        S = bin_population(spike_time_list, t_start, t_stop, dt, clip=clip)
        raw = bin_population(spike_time_list, t_start, t_stop, dt, clip=255)
        K = int(S.max()) + 1
        e = METHODS.index(method)
        r = mi_pairs(S, pairs, K=K, lag=lag_bins, n_null=n_null, rng=rng)
        recs.append({
            "dt": dt,
            "n_bins": S.shape[1],
            "K": K,
            "N_over_states": S.shape[1] / K ** 2,
            "clipped_frac": float(np.mean(raw > clip)),
            "mi_raw": r["mi"][:, e],
            "null_mean": r["null_mean"][:, e],
            "mi_corrected": r["mi_corrected"][:, e],
            "rate_bits_per_s": r["mi_corrected"][:, e] / dt,
            "p": r["p"][:, e],
        })
    return recs


def bh_fdr(p, alpha=0.05):
    """Benjamini-Hochberg. Returns (reject, qvalues). Use this across pairs/lags."""
    p = np.asarray(p, dtype=float).ravel()
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out <= alpha, out


def learning_curve(states, pairs=None, fractions=(1.0, 0.75, 0.5, 0.35, 0.25, 0.15),
                   method="grass", **kw):
    """MI vs amount of data. A flat curve means you are sampling-limited-free;
    a curve still falling as data increases means the estimate is bias-dominated
    and no analytic correction should be trusted."""
    S, pairs = _prep(states, pairs)
    e = METHODS.index(method)
    T = S.shape[1]
    tables = kw.pop("tables", None) or CountTables(T)
    out = np.zeros((len(fractions), pairs.shape[0]))
    Ns = np.zeros(len(fractions), dtype=np.int64)
    for k, f in enumerate(fractions):
        L = max(2, int(T * f))
        nb = max(1, T // L)
        acc = np.zeros(pairs.shape[0])
        for b in range(nb):
            seg = np.ascontiguousarray(S[:, b * L:(b + 1) * L])
            acc += mi_pairs(seg, pairs, tables=tables, **kw)["mi"][:, e]
        out[k] = acc / nb
        Ns[k] = L
    return Ns, out
