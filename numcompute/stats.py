"""Batch and streaming statistical summaries."""

from collections import deque
import numpy as np


def _to_array(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=float)


def mean(X, axis=None) -> np.ndarray:
    return np.nanmean(_to_array(X), axis=axis)


def median(X, axis=None) -> np.ndarray:
    return np.nanmedian(_to_array(X), axis=axis)


def std(X, axis=None, ddof: int = 0) -> np.ndarray:
    return np.nanstd(_to_array(X), axis=axis, ddof=ddof)


def minimum(X, axis=None) -> np.ndarray:
    return np.nanmin(_to_array(X), axis=axis)


def maximum(X, axis=None) -> np.ndarray:
    return np.nanmax(_to_array(X), axis=axis)


def quantiles(X, q=(0.25, 0.5, 0.75), axis=None) -> np.ndarray:
    return np.nanpercentile(_to_array(X), np.array(q) * 100, axis=axis)


def histogram(X, bins: int = 10, bins_range=None):
    arr = _to_array(X)
    arr = arr[~np.isnan(arr)]
    return np.histogram(arr, bins=bins, range=bins_range)


def stats(X, axis=None, ddof: int = 0) -> dict:
    q25, q75 = quantiles(X, q=(0.25, 0.75), axis=axis)
    return {
        "mean": mean(X, axis=axis),
        "median": median(X, axis=axis),
        "std": std(X, axis=axis, ddof=ddof),
        "min": minimum(X, axis=axis),
        "max": maximum(X, axis=axis),
        "q25": q25,
        "q75": q75,
    }


class StreamingStats:
    """Chunk-based running statistics with optional rolling window."""

    def __init__(self, bins=10, bins_range=None, window_size=None):
        self.bins = bins
        self.bins_range = bins_range
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.n_ = None
        self.mean_ = None
        self.m2_ = None
        self.min_ = None
        self.max_ = None
        self._values = deque(maxlen=self.window_size) if self.window_size else []
        return self

    def update_stats(self, X_chunk):
        X = _to_array(X_chunk)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.window_size:
            for row in X:
                self._values.append(row.copy())
            return self._recompute_window()

        count = np.sum(~np.isnan(X), axis=0).astype(float)
        chunk_sum = np.nansum(X, axis=0)
        chunk_mean = np.divide(chunk_sum, count, out=np.zeros_like(chunk_sum), where=count > 0)
        centered = np.where(np.isnan(X), 0.0, X - chunk_mean)
        chunk_m2 = np.sum(centered ** 2, axis=0)
        chunk_min = np.nanmin(X, axis=0)
        chunk_max = np.nanmax(X, axis=0)

        if self.mean_ is None:
            self.n_ = count
            self.mean_ = chunk_mean
            self.m2_ = chunk_m2
            self.min_ = chunk_min
            self.max_ = chunk_max
        else:
            total = self.n_ + count
            delta = chunk_mean - self.mean_
            active = count > 0
            self.mean_[active] = self.mean_[active] + delta[active] * count[active] / total[active]
            self.m2_[active] = (
                self.m2_[active]
                + chunk_m2[active]
                + delta[active] ** 2 * self.n_[active] * count[active] / total[active]
            )
            self.n_ = total
            self.min_ = np.fmin(self.min_, chunk_min)
            self.max_ = np.fmax(self.max_, chunk_max)

        if not self.window_size:
            self._values.extend(X.tolist())
        return self

    def _recompute_window(self):
        data = np.asarray(list(self._values), dtype=float)
        self.n_ = np.sum(~np.isnan(data), axis=0).astype(float)
        self.mean_ = np.nanmean(data, axis=0)
        self.m2_ = np.nanvar(data, axis=0) * np.maximum(self.n_, 1.0)
        self.min_ = np.nanmin(data, axis=0)
        self.max_ = np.nanmax(data, axis=0)
        return self

    def result(self, q=(0.25, 0.5, 0.75)):
        if self.mean_ is None:
            return {}
        arr = np.asarray(list(self._values), dtype=float)
        var = np.divide(self.m2_, np.maximum(self.n_, 1.0), out=np.zeros_like(self.m2_))
        counts, edges = histogram(arr.ravel(), bins=self.bins, bins_range=self.bins_range)
        return {
            "count": self.n_.copy(),
            "mean": self.mean_.copy(),
            "variance": var,
            "std": np.sqrt(var),
            "min": self.min_.copy(),
            "max": self.max_.copy(),
            "quantiles": quantiles(arr, q=q, axis=0),
            "histogram": (counts, edges),
        }
