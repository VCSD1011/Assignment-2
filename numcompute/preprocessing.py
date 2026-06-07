"""Streaming preprocessing utilities for NumCompute.

The classes keep the original fit/transform API and add partial_fit so they can
be updated safely one chunk at a time in streaming pipelines.
"""

import numpy as np

if __name__ == "__main__":
    from utils import BaseTransformer
else:
    from numcompute.utils import BaseTransformer


class StandardScaler(BaseTransformer):
    """Z-score scaler with batch and incremental fitting.

    partial_fit uses a vectorised running mean/variance update, ignoring NaN
    values independently per feature.
    """

    def __init__(self):
        super().__init__()
        self.mean_ = None
        self.var_ = None
        self.n_seen_ = None
        self._m2 = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.nanmean(X, axis=0)
        self.var_ = np.nanvar(X, axis=0)
        counts = np.sum(~np.isnan(X), axis=0).astype(float)
        self.n_seen_ = counts
        self._m2 = self.var_ * np.maximum(counts, 1.0)
        self.var_ = np.where(self.var_ == 0.0, 1e-8, self.var_)
        self._is_fitted = True
        return self

    def partial_fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        chunk_count = np.sum(~np.isnan(X), axis=0).astype(float)
        safe_chunk = np.where(np.isnan(X), 0.0, X)
        chunk_sum = np.sum(safe_chunk, axis=0)
        chunk_mean = np.divide(
            chunk_sum,
            chunk_count,
            out=np.zeros_like(chunk_sum, dtype=float),
            where=chunk_count > 0,
        )
        centered = np.where(np.isnan(X), 0.0, X - chunk_mean)
        chunk_m2 = np.sum(centered ** 2, axis=0)

        if self.mean_ is None:
            self.mean_ = chunk_mean
            self.n_seen_ = chunk_count
            self._m2 = chunk_m2
        else:
            total = self.n_seen_ + chunk_count
            delta = chunk_mean - self.mean_
            active = chunk_count > 0
            new_mean = self.mean_.copy()
            new_mean[active] = self.mean_[active] + delta[active] * (
                chunk_count[active] / total[active]
            )
            new_m2 = self._m2.copy()
            new_m2[active] = (
                self._m2[active]
                + chunk_m2[active]
                + delta[active] ** 2
                * self.n_seen_[active]
                * chunk_count[active]
                / total[active]
            )
            self.mean_ = new_mean
            self._m2 = new_m2
            self.n_seen_ = total

        self.var_ = np.divide(
            self._m2,
            np.maximum(self.n_seen_, 1.0),
            out=np.zeros_like(self._m2, dtype=float),
        )
        self.var_ = np.where(self.var_ == 0.0, 1e-8, self.var_)
        self._is_fitted = True
        return self

    def transform(self, X):
        super().transform(X)
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / np.sqrt(self.var_)


class MinMaxScaler(BaseTransformer):
    """Scale features into a fixed range, with streaming min/max updates."""

    def __init__(self, f_range=(0.0, 1.0)):
        super().__init__()
        self.f_range = f_range
        self.X_min = None
        self.X_max = None
        self.X_range = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.X_min = np.nanmin(X, axis=0)
        self.X_max = np.nanmax(X, axis=0)
        self.X_range = self.X_max - self.X_min
        self.X_range = np.where(self.X_range == 0.0, 1.0, self.X_range)
        self._is_fitted = True
        return self

    def partial_fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        chunk_min = np.nanmin(X, axis=0)
        chunk_max = np.nanmax(X, axis=0)
        if self.X_min is None:
            self.X_min = chunk_min
            self.X_max = chunk_max
        else:
            self.X_min = np.fmin(self.X_min, chunk_min)
            self.X_max = np.fmax(self.X_max, chunk_max)
        self.X_range = self.X_max - self.X_min
        self.X_range = np.where(self.X_range == 0.0, 1.0, self.X_range)
        self._is_fitted = True
        return self

    def transform(self, X):
        super().transform(X)
        X = np.asarray(X, dtype=float)
        X_std = (X - self.X_min) / self.X_range
        range_min, range_max = self.f_range
        return X_std * (range_max - range_min) + range_min


class OneHotEncoder(BaseTransformer):
    """Incremental one-hot encoder.

    New categories discovered during partial_fit are appended to each feature's
    category list. transform keeps a stable column order based on discovery.
    """

    def __init__(self, handle_unknown="error"):
        super().__init__()
        if handle_unknown not in ["error", "ignore"]:
            raise ValueError("handle_unknown must be 'error' or 'ignore'")
        self.handle_unknown = handle_unknown
        self.categories = None

    def fit(self, X, y=None):
        self.categories = None
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        X = np.asarray(X, dtype=object)
        X = np.atleast_2d(X)
        if self.categories is None:
            self.categories = []
            for j in range(X.shape[1]):
                self.categories.append(np.unique(X[:, j]))
        else:
            for j in range(X.shape[1]):
                merged = np.concatenate([self.categories[j], np.unique(X[:, j])])
                self.categories[j] = np.unique(merged)
        self._is_fitted = True
        return self

    def transform(self, X):
        super().transform(X)
        X = np.asarray(X, dtype=object)
        X = np.atleast_2d(X)
        encoded_columns = []
        for j in range(X.shape[1]):
            col = X[:, j, np.newaxis]
            cats = self.categories[j]
            mask = col == cats
            if self.handle_unknown == "error" and not np.all(np.any(mask, axis=1)):
                raise ValueError(f"Found unknown categories in column {j} during transform.")
            encoded_columns.append(mask.astype(float))
        return np.hstack(encoded_columns) if encoded_columns else np.empty((X.shape[0], 0))


class Imputer(BaseTransformer):
    """Replace missing values using a constant or streaming feature means."""

    def __init__(self, strategy="constant", fill_value=0.0):
        super().__init__()
        if strategy not in ["mean", "constant"]:
            raise ValueError("Strategy must be 'mean' or 'constant'.")
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics = None
        self._sum = None
        self._count = None

    def fit(self, X, y=None):
        self.statistics = None
        self._sum = None
        self._count = None
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_features = X.shape[1]

        if self.strategy == "constant":
            self.statistics = np.full(n_features, self.fill_value, dtype=float)
        else:
            chunk_sum = np.nansum(X, axis=0)
            chunk_count = np.sum(~np.isnan(X), axis=0).astype(float)
            if self._sum is None:
                self._sum = chunk_sum
                self._count = chunk_count
            else:
                self._sum += chunk_sum
                self._count += chunk_count
            self.statistics = np.divide(
                self._sum,
                self._count,
                out=np.zeros_like(self._sum, dtype=float),
                where=self._count > 0,
            )
        self._is_fitted = True
        return self

    def transform(self, X):
        super().transform(X)
        X = np.asarray(X, dtype=float)
        return np.where(np.isnan(X), self.statistics, X)


SimpleImputer = Imputer
