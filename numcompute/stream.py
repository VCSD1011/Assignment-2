"""Stream training orchestration for NumCompute."""

import sys
import time
import numpy as np

try:
    from numcompute.metrics import StreamingClassificationMetrics
except ImportError:
    from metrics import StreamingClassificationMetrics


class StreamTrainer:

    def __init__(self, estimator, labels=None, metric_tracker=None):
        self.estimator = estimator
        self.metric_tracker = metric_tracker or StreamingClassificationMetrics(labels=labels)
        self.logs_ = []
        self.chunk_index_ = 0
        self.total_correct_ = 0
        self.total_seen_ = 0

    def fit_chunk(self, X, y):
        """Incrementally train the estimator on one chunk."""
        start = time.perf_counter()
        if hasattr(self.estimator, "partial_fit"):
            self.estimator.partial_fit(X, y)
        elif hasattr(self.estimator, "fit"):
            self.estimator.fit(X, y)
        else:
            raise ValueError("estimator must implement partial_fit() or fit().")

        elapsed = time.perf_counter() - start
        log = {
            "chunk": self.chunk_index_,
            "phase": "fit",
            "n_samples": int(np.asarray(X).shape[0]),
            "elapsed_seconds": elapsed,
            "memory_bytes": self._memory_bytes(X, y),
        }
        self.logs_.append(log)
        self.chunk_index_ += 1
        return log

    def score_chunk(self, X, y):
        """Predict and update streaming metrics on one chunk."""
        y_pred = self.estimator.predict(X)
        self.metric_tracker.update(y, y_pred)
        result = self.metric_tracker.result()

        y = np.asarray(y)
        y_pred = np.asarray(y_pred)
        chunk_accuracy = float(np.mean(y == y_pred)) if y.size else 0.0
        self.total_correct_ += int(np.sum(y == y_pred))
        self.total_seen_ += int(y.size)
        cumulative_accuracy = self.total_correct_ / self.total_seen_ if self.total_seen_ else 0.0

        log = {
            "chunk": self.chunk_index_ - 1,
            "phase": "score",
            "n_samples": int(y.size),
            "chunk_accuracy": chunk_accuracy,
            "cumulative_accuracy": cumulative_accuracy,
            "memory_bytes": self._memory_bytes(X, y),
            "metrics": result,
        }
        self.logs_.append(log)
        return log

    def fit_score_chunk(self, X, y):
        """Convenience method: fit one chunk, then score the same chunk."""
        fit_log = self.fit_chunk(X, y)
        score_log = self.score_chunk(X, y)
        return fit_log, score_log

    @staticmethod
    def _memory_bytes(*arrays):
        total = 0
        for arr in arrays:
            arr_np = np.asarray(arr)
            total += arr_np.nbytes if hasattr(arr_np, "nbytes") else sys.getsizeof(arr)
        return int(total)
