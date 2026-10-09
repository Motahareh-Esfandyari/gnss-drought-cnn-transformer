# =========================================================================
#  GNSS-Drought : Drought Identification from GNSS Time Series (ISPRS 2026)
#  -------------------------------------------------------------------------
#  Script  : preprocessing.py
#  Purpose : Python version of PreProcessing.m. Reads the daily vertical
#            series of station P242, removes outliers with a moving
#            z-score (+/- 3 sigma, Eq. of Sec. 3.1), fills short gaps by
#            linear interpolation, detects offsets with the A-mode
#            chi-square test (alpha = 0.02), and decomposes the cleaned
#            series with SSA into trend / annual / semi-annual components.
#            Writes ssaresult.csv for feature_extraction.py.
#  Usage   : python preprocessing.py P242.cwu.igs14.csv
#  -------------------------------------------------------------------------
#  Author  : Motahareh Esfandyari-Kaloukan
#  Paper   : doi.org/10.5194/isprs-annals-XI-3-2026-625-2026
# =========================================================================

import sys
import numpy as np
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt

from ssa import ssa_decompose

WINDOW = 30          # moving window for the z-score filter [days]
Z_THR = 3.0          # outlier threshold [sigmas]
ALPHA = 0.02         # significance level of the offset test
SSA_L = 365          # SSA window length [days]


def read_series(path):
    """Read a NOTA/UNAVCO csv. Expects a date (or decimal-year) column
    and a vertical column; extra columns are ignored."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    datecol = df.columns[0]
    vercol = [c for c in df.columns if "ver" in c or "up" in c][-1]
    try:
        dates = pd.to_datetime(df[datecol])
    except (ValueError, TypeError):
        # decimal years
        dy = df[datecol].astype(float)
        dates = pd.to_datetime((dy - 1970.0) * 365.25 * 86400.0, unit="s")
    ver = df[vercol].astype(float)
    s = pd.Series(ver.values, index=dates).sort_index()
    # one sample per day on a regular grid
    s = s.resample("1D").mean()
    return s


def remove_outliers(s):
    """Moving z-score filter: flag |y - movmean| > Z_THR * movstd."""
    m = s.rolling(WINDOW, center=True, min_periods=5).mean()
    sd = s.rolling(WINDOW, center=True, min_periods=5).std()
    bad = (s - m).abs() > Z_THR * sd
    out = s.copy()
    out[bad] = np.nan
    return out, int(bad.sum())


def fill_gaps(s):
    """Linear interpolation of the gaps (Eq. 1 of the paper)."""
    return s.interpolate(method="linear", limit_direction="both")


def detect_offsets(y, t, alpha=ALPHA, min_gap=180):
    """A-mode offset detection (Sec. 3.1). Candidate epochs are taken
    from the largest steps of the smoothed series; each candidate step
    is estimated by least squares on the detrended series and kept if
    its chi-square statistic exceeds the (1 - alpha) quantile with
    df = 1. A white-noise covariance is used here; for the full
    white + flicker version build Q with the routines in the MATLAB
    lib/ folder (makePL) and replace the identity below.
    """
    n = len(y)
    # detrend (linear + annual + semi-annual), as in the MATLAB code
    A = np.column_stack([
        np.ones(n), t,
        np.cos(2 * np.pi * t), np.sin(2 * np.pi * t),
        np.cos(4 * np.pi * t), np.sin(4 * np.pi * t),
    ])
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    v = y - A @ x

    # candidate epochs: largest jumps of the 30-day smoothed residual
    sm = pd.Series(v).rolling(WINDOW, center=True, min_periods=5).mean().values
    jump = np.abs(np.diff(sm))
    order = np.argsort(jump)[::-1]
    candidates, taken = [], np.zeros(n, bool)
    for k in order[:200]:
        if not taken[max(0, k - min_gap):k + min_gap].any():
            candidates.append(k)
            taken[k] = True
        if len(candidates) >= 10:
            break

    sigma2 = np.var(v, ddof=A.shape[1])
    thr = chi2.ppf(1 - alpha, df=1)
    offsets = []
    for k in sorted(candidates):
        step = np.zeros(n)
        step[k + 1:] = 1.0
        g = step - step.mean()
        amp = (g @ v) / (g @ g)
        stat = amp**2 * (g @ g) / sigma2
        if stat > thr:
            offsets.append((k, amp))
            v = v - amp * (step - step.mean())
    return offsets, v + (A @ x)  # corrected series (model added back)


def main(path):
    s = read_series(path)
    print(f"read {len(s)} daily epochs: {s.index[0].date()} .. {s.index[-1].date()}")

    s, nbad = remove_outliers(s)
    print(f"outliers removed: {nbad}")
    s = fill_gaps(s)

    t = (s.index - s.index[0]).days.values / 365.25
    y = s.values

    offsets, y = detect_offsets(y, t)
    print(f"significant offsets (alpha={ALPHA}): {len(offsets)}")
    for k, amp in offsets:
        print(f"  {s.index[k].date()}  amplitude {amp:+.2f} mm")

    comps, freqs = ssa_decompose(y - y.mean(), L=SSA_L)
    print("SSA eigentriple frequencies [cpy]:",
          np.round(freqs, 2))

    out = pd.DataFrame({
        "date": s.index.strftime("%Y-%m-%d"),
        "trend": comps["trend"],
        "annual": comps["annual"],
        "semi": comps["semi"],
    })
    out.to_csv("ssaresult.csv", index=False)
    print("wrote ssaresult.csv")

    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(s.index, y, lw=0.5); ax[0].set_ylabel("cleaned [mm]")
    ax[1].plot(s.index, comps["trend"]); ax[1].set_ylabel("trend")
    ax[2].plot(s.index, comps["annual"], lw=0.5); ax[2].set_ylabel("annual")
    ax[3].plot(s.index, comps["semi"], lw=0.5); ax[3].set_ylabel("semi-annual")
    fig.suptitle("P242 vertical series and SSA components")
    fig.tight_layout()
    fig.savefig("ssa_components.png", dpi=150)
    print("wrote ssa_components.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "P242.cwu.igs14.csv")
