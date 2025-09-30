"""
utils.py
========
Internal helper functions for pyDLVH.

These are not part of the public API and may change without notice.
"""
from __future__ import annotations

import numpy as np


def _freedman_diaconis_bins(*, data: np.ndarray, max_bins: int = 200) -> np.ndarray:
    """
    Suggest bin edges using the Freedman–Diaconis rule.

    Parameters
    ----------
    data : np.ndarray
        Input 1D or ND array (flattened internally).
    max_bins : int, default=200
        Maximum number of bins allowed.

    Returns
    -------
    bins : np.ndarray
        Array of bin edges covering the data range.

    Notes
    -----
    h = 2 * IQR(x) / n^(1/3). If h <= 0 or range == 0, fallback.
    """
    x = np.asarray(data).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        raise ValueError("Empty data passed to _freedman_diaconis_bins.")
    xmin = float(np.min(x))
    xmax = float(np.max(x))

    # Handle constant array (range == 0) → make a tiny 2-edge range
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        raise ValueError("Non-finite range in _freedman_diaconis_bins.")
    if xmax <= xmin:
        eps = 1e-6 if xmin == 0.0 else abs(xmin) * 1e-6
        return np.array([xmin, xmin + eps], dtype=float)

    if n < 2:
        return np.array([xmin, xmax], dtype=float)

    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    h = 2.0 * iqr / (n ** (1.0 / 3.0)) if iqr > 0 else 0.0

    if h <= 0:
        std = float(np.std(x))
        h = (2.0 * std) / (n ** (1.0 / 3.0)) if std > 0 else (xmax - xmin)

    nbins = int(np.ceil((xmax - xmin) / h)) if h > 0 else 1
    nbins = min(max(nbins, 1), max_bins)

    return np.linspace(xmin, xmax, nbins + 1)


def _auto_bins(*, arr: np.ndarray, max_bins: int = 200) -> np.ndarray:
    """
    Suggest optimal bin edges for a 1D histogram.

    Parameters
    ----------
    arr : np.ndarray
        Input array to compute bin edges for.
    max_bins : int, default=200
        Maximum number of bins.

    Returns
    -------
    bins : np.ndarray
        Bin edges for arr.
    """
    return _freedman_diaconis_bins(data=arr, max_bins=max_bins)


def _suffix_cumsum2d(counts: np.ndarray) -> np.ndarray:
    """Fast suffix cumulative sum for 2D arrays.
    Given differential counts C[i,j] on an (Nd×Nl) grid, returns S where
    S[i,j] = sum_{p>=i, q>=j} C[p,q].
    """
    # reverse both axes → prefix cumsum → reverse back
    s = counts[::-1, ::-1].cumsum(axis=0).cumsum(axis=1)[::-1, ::-1]
    return s