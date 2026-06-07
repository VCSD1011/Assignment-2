"""Batch and streaming metrics for NumCompute."""

from collections import deque
import numpy as np


def accuracy(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(y_true == y_pred)) if y_true.size else 0.0


def confusion_matrix(y_true, y_pred, labels=None):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if labels is None:
        labels = np.array([0, 1])
    labels = np.array(labels)
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            matrix[i, j] = int(np.sum((y_true == true_label) & (y_pred == pred_label)))
    return matrix


def precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp, fp = cm[1, 1], cm[0, 1]
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp, fn = cm[1, 1], cm[1, 0]
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def mse(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2)) if y_true.size else 0.0


def roc_curve(y_true, y_scores):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores, dtype=float)
    if y_true.size == 0:
        return np.array([0.0]), np.array([0.0]), np.array([np.inf])
    desc_idx = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_idx]
    thresholds = y_scores[desc_idx]
    tp_cumsum = np.cumsum(y_true_sorted == 1)
    fp_cumsum = np.cumsum(y_true_sorted != 1)
    total_pos = tp_cumsum[-1]
    total_neg = fp_cumsum[-1]
    tpr = tp_cumsum / total_pos if total_pos > 0 else tp_cumsum * 0.0
    fpr = fp_cumsum / total_neg if total_neg > 0 else fp_cumsum * 0.0
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    thresholds = np.concatenate(([thresholds[0] + 1], thresholds))
    return fpr, tpr, thresholds


def auc(fpr, tpr):
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz_fn(tpr, fpr))


class StreamingClassificationMetrics:
    """Accumulate classification metrics across chunks.

    Parameters
    ----------
    labels : array-like, optional
        Class labels used to build the confusion matrix.
    window_size : int, optional
        If provided, result() is computed over the latest window_size samples.
    """

    def __init__(self, labels=None, window_size=None):
        self.labels = np.array([0, 1]) if labels is None else np.array(labels)
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.cm_ = np.zeros((len(self.labels), len(self.labels)), dtype=int)
        self.correct_ = 0
        self.total_ = 0
        self.y_true_ = []
        self.y_pred_ = []
        self.y_score_ = []
        self._window = deque(maxlen=self.window_size) if self.window_size else None
        return self

    def update(self, y_true_chunk, y_pred_chunk, y_score_chunk=None):
        y_true_chunk = np.asarray(y_true_chunk)
        y_pred_chunk = np.asarray(y_pred_chunk)
        if y_true_chunk.shape[0] != y_pred_chunk.shape[0]:
            raise ValueError("y_true_chunk and y_pred_chunk must have the same length.")

        if self.window_size:
            scores = y_score_chunk if y_score_chunk is not None else [None] * len(y_true_chunk)
            for yt, yp, ys in zip(y_true_chunk, y_pred_chunk, scores):
                self._window.append((yt, yp, ys))
            return self._rebuild_from_window()

        self.cm_ += confusion_matrix(y_true_chunk, y_pred_chunk, labels=self.labels)
        self.correct_ += int(np.sum(y_true_chunk == y_pred_chunk))
        self.total_ += int(y_true_chunk.size)
        self.y_true_.extend(y_true_chunk.tolist())
        self.y_pred_.extend(y_pred_chunk.tolist())
        if y_score_chunk is not None:
            self.y_score_.extend(np.asarray(y_score_chunk, dtype=float).tolist())
        return self

    def _rebuild_from_window(self):
        items = list(self._window)
        self.cm_[:] = 0
        self.correct_ = 0
        self.total_ = len(items)
        self.y_true_ = [i[0] for i in items]
        self.y_pred_ = [i[1] for i in items]
        self.y_score_ = [i[2] for i in items if i[2] is not None]
        if items:
            self.cm_ += confusion_matrix(self.y_true_, self.y_pred_, labels=self.labels)
            self.correct_ = int(np.sum(np.asarray(self.y_true_) == np.asarray(self.y_pred_)))
        return self

    def result(self):
        acc = self.correct_ / self.total_ if self.total_ else 0.0
        if len(self.labels) == 2:
            p = precision(self.y_true_, self.y_pred_)
            r = recall(self.y_true_, self.y_pred_)
            f = f1(self.y_true_, self.y_pred_)
        else:
            per_class_p = []
            per_class_r = []
            for i in range(len(self.labels)):
                tp = self.cm_[i, i]
                fp = np.sum(self.cm_[:, i]) - tp
                fn = np.sum(self.cm_[i, :]) - tp
                per_class_p.append(tp / (tp + fp) if (tp + fp) else 0.0)
                per_class_r.append(tp / (tp + fn) if (tp + fn) else 0.0)
            p = float(np.mean(per_class_p)) if per_class_p else 0.0
            r = float(np.mean(per_class_r)) if per_class_r else 0.0
            f = 2 * p * r / (p + r) if (p + r) else 0.0

        out = {
            "accuracy": float(acc),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "confusion_matrix": self.cm_.copy(),
            "total": self.total_,
        }
        if len(self.y_score_) == len(self.y_true_) and len(self.y_score_) > 0:
            fpr, tpr, _ = roc_curve(self.y_true_, self.y_score_)
            out["auc"] = auc(fpr, tpr)
        return out


RollingClassificationMetrics = StreamingClassificationMetrics
