"""Streaming ensemble classifier built from NumCompute decision trees."""

import numpy as np

try:
    from numcompute.tree import DecisionTreeClassifier
except ImportError:
    from tree import DecisionTreeClassifier


class EnsembleClassifier:
    """Bagging-style ensemble of decision trees with partial_fit support."""

    def __init__(
        self,
        n_estimators=5,
        max_depth=5,
        min_samples_split=2,
        max_features=None,
        criterion="gini",
        bootstrap=True,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.estimators_ = [
            DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                max_features=max_features,
                criterion=criterion,
                random_state=None if random_state is None else random_state + i,
            )
            for i in range(n_estimators)
        ]
        self.classes_ = None

    def fit(self, X, y):
        self._reset_estimators()
        return self.partial_fit(X, y)

    def partial_fit(self, X_chunk, y_chunk):
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)
        if X_chunk.ndim == 1:
            X_chunk = X_chunk.reshape(-1, 1)
        self.classes_ = np.unique(y_chunk) if self.classes_ is None else np.unique(np.concatenate([self.classes_, np.unique(y_chunk)]))

        n = X_chunk.shape[0]
        for tree in self.estimators_:
            if self.bootstrap and n > 0:
                idx = self._rng.integers(0, n, size=n)
                tree.partial_fit(X_chunk[idx], y_chunk[idx])
            else:
                tree.partial_fit(X_chunk, y_chunk)
        return self

    def predict(self, X):
        if not self.estimators_ or self.estimators_[0].root_ is None:
            raise RuntimeError("EnsembleClassifier must be fitted before predict().")
        predictions = np.vstack([tree.predict(X) for tree in self.estimators_]).T
        final = []
        for row in predictions:
            values, counts = np.unique(row, return_counts=True)
            final.append(values[np.argmax(counts)])
        return np.array(final)

    def _reset_estimators(self):
        self._rng = np.random.default_rng(self.random_state)
        self.estimators_ = [
            DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                criterion=self.criterion,
                random_state=None if self.random_state is None else self.random_state + i,
            )
            for i in range(self.n_estimators)
        ]
        self.classes_ = None
