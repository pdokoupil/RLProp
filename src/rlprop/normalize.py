"""Optional per-objective normalization of marginal gains.

The paper notes that putting objectives "on the same scale" is what makes the
proportional ratios meaningful. These transforms are applied per objective (per
row of the marginal-gain matrix), across the candidate items.

All implementations are pure NumPy on purpose: ``pip install rlprop-rs`` should pull
in nothing but NumPy. The names mirror the familiar scikit-learn scalers.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _minmax(row: np.ndarray) -> np.ndarray:
    lo, hi = row.min(), row.max()
    if hi - lo < _EPS:
        return np.zeros_like(row)
    return (row - lo) / (hi - lo)


def _standard(row: np.ndarray) -> np.ndarray:
    std = row.std()
    if std < _EPS:
        return np.zeros_like(row)
    return (row - row.mean()) / std


def _robust(row: np.ndarray) -> np.ndarray:
    med = np.median(row)
    q1, q3 = np.percentile(row, [25, 75])
    iqr = q3 - q1
    if iqr < _EPS:
        return np.zeros_like(row)
    return (row - med) / iqr


def _quantile(row: np.ndarray) -> np.ndarray:
    """Empirical-CDF transform to uniform [0, 1] (like QuantileTransformer)."""
    n = row.size
    if n <= 1:
        return np.zeros_like(row)
    order = row.argsort()
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    return ranks / (n - 1)


_METHODS = {
    "none": lambda row: row,
    "minmax": _minmax,
    "standard": _standard,
    "robust": _robust,
    "quantile": _quantile,
}

METHODS = tuple(_METHODS)


def normalize_mgains(mgains: np.ndarray, method: str | None) -> np.ndarray:
    """Normalize each objective's marginal-gain row with ``method``.

    ``method`` may be ``None`` / ``"none"``, ``"minmax"``, ``"standard"``,
    ``"robust"``, or ``"quantile"``.
    """
    if method is None or method == "none":
        return mgains
    try:
        fn = _METHODS[method]
    except KeyError:
        raise ValueError(
            f"Unknown normalization {method!r}; choose one of {METHODS}."
        ) from None
    out = np.empty_like(mgains)
    for o in range(mgains.shape[0]):
        out[o] = fn(mgains[o])
    return out
