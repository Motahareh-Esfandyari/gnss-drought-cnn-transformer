# =========================================================================
#  GNSS-Drought : Drought Identification from GNSS Time Series (ISPRS 2026)
#  -------------------------------------------------------------------------
#  Module  : ssa.py
#  Purpose : Singular Spectrum Analysis of a 1D series. Builds the
#            trajectory matrix, computes the SVD, and groups the
#            eigentriples into trend / annual / semi-annual components
#            by the dominant frequency of each reconstructed component
#            (the grouping described in Section 3.2 of the paper).
#  -------------------------------------------------------------------------
#  Author  : Motahareh Esfandyari-Kaloukan
#  Paper   : doi.org/10.5194/isprs-annals-XI-3-2026-625-2026
# =========================================================================

import numpy as np


def _hankelize(X):
    """Diagonal (Hankel) averaging: turn a reconstructed trajectory
    matrix back into a 1D series."""
    L, K = X.shape
    N = L + K - 1
    out = np.zeros(N)
    counts = np.zeros(N)
    for i in range(L):
        out[i:i + K] += X[i, :]
        counts[i:i + K] += 1.0
    return out / counts


def _dominant_freq(x, fs):
    """Dominant frequency of a series in cycles per year
    (fs = samples per year, 365.25 for daily data)."""
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    spec[0] = 0.0
    return freqs[int(np.argmax(spec))]


def ssa_decompose(y, L=365, n_components=20, fs=365.25):
    """SSA decomposition of y with window length L.

    Returns a dict with the 'trend', 'annual', 'semi' and 'residual'
    series (each the same length as y) and the per-component dominant
    frequencies, so the grouping can be inspected.

    Grouping rule (cycles per year):
        trend       f < 0.4
        annual      0.6 <= f <= 1.4
        semi-annual 1.6 <= f <= 2.4
        residual    everything else
    """
    y = np.asarray(y, dtype=float)
    N = len(y)
    K = N - L + 1

    # Trajectory matrix (L x K)
    X = np.lib.stride_tricks.sliding_window_view(y, L).T

    # Economy SVD; only the leading eigentriples are needed
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    d = min(n_components, len(s))

    comps = {"trend": np.zeros(N), "annual": np.zeros(N),
             "semi": np.zeros(N), "residual": np.zeros(N)}
    freqs = []

    for i in range(d):
        Xi = s[i] * np.outer(U[:, i], Vt[i, :])
        gi = _hankelize(Xi)
        f = _dominant_freq(gi, fs)
        freqs.append(f)
        if f < 0.4:
            comps["trend"] += gi
        elif 0.6 <= f <= 1.4:
            comps["annual"] += gi
        elif 1.6 <= f <= 2.4:
            comps["semi"] += gi
        else:
            comps["residual"] += gi

    # Everything beyond the kept eigentriples also belongs to the residual
    comps["residual"] += y - (comps["trend"] + comps["annual"]
                              + comps["semi"] + comps["residual"])
    return comps, np.array(freqs)
