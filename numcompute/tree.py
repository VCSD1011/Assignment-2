

from dataclasses import dataclass
import numpy as np


@dataclass
class _Node:
    prediction: object
    feature_index: int = None
    threshold: float = None
    left: object = None
    right: object = None
    is_leaf: bool = True


class DecisionTreeClassifier:


    def __init__(self, max_depth=5, min_samples_split=2, max_features=None, criterion="gini", random_state=None):
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.root_ = None
        self.classes_ = None
        self._X_seen = None
        self._y_seen = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.classes_ = np.unique(y)
        self.root_ = self._build_tree(X, y, depth=0)
        self._X_seen = X.copy()
        self._y_seen = y.copy()
        return self

    def partial_fit(self, X_chunk, y_chunk):
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)
        if X_chunk.ndim == 1:
            X_chunk = X_chunk.reshape(-1, 1)
        if self._X_seen is None:
            self._X_seen = X_chunk.copy()
            self._y_seen = y_chunk.copy()
        else:
            self._X_seen = np.vstack([self._X_seen, X_chunk])
            self._y_seen = np.concatenate([self._y_seen, y_chunk])
        self.classes_ = np.unique(self._y_seen)
        self.root_ = self._build_tree(self._X_seen, self._y_seen, depth=0)
        return self

    def predict(self, X):
        if self.root_ is None:
            raise RuntimeError("DecisionTreeClassifier must be fitted before predict().")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self._predict_row(row, self.root_) for row in X])

    def _build_tree(self, X, y, depth):
        prediction = self._majority_class(y)
        if (
            depth >= self.max_depth
            or X.shape[0] < self.min_samples_split
            or np.unique(y).size == 1
        ):
            return _Node(prediction=prediction)

        feature_index, threshold, gain = self._best_split(X, y)
        if feature_index is None or gain <= 0:
            return _Node(prediction=prediction)

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        return _Node(
            prediction=prediction,
            feature_index=feature_index,
            threshold=threshold,
            left=self._build_tree(X[left_mask], y[left_mask], depth + 1),
            right=self._build_tree(X[right_mask], y[right_mask], depth + 1),
            is_leaf=False,
        )

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        features = np.arange(n_features)
        if self.max_features is not None:
            if isinstance(self.max_features, float):
                k = max(1, int(np.ceil(self.max_features * n_features)))
            else:
                k = min(n_features, int(self.max_features))
            features = self._rng.choice(features, size=k, replace=False)

        parent_impurity = self._impurity(y)
        best_gain, best_feature, best_threshold = 0.0, None, None
        for feature in features:
            values = X[:, feature]
            thresholds = np.unique(values[~np.isnan(values)])
            if thresholds.size <= 1:
                continue
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2.0
            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask
                if not left_mask.any() or not right_mask.any():
                    continue
                left_weight = np.sum(left_mask) / n_samples
                right_weight = 1.0 - left_weight
                gain = parent_impurity - (
                    left_weight * self._impurity(y[left_mask])
                    + right_weight * self._impurity(y[right_mask])
                )
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold
        return best_feature, best_threshold, best_gain

    def _impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        if self.criterion == "gini":
            return 1.0 - np.sum(probs ** 2)
        return -np.sum(probs * np.log2(probs + 1e-12))

    @staticmethod
    def _majority_class(y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _predict_row(self, row, node):
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction
