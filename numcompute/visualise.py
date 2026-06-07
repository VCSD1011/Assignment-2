"""Reusable visualisation helpers for streaming experiments."""

import matplotlib.pyplot as plt
import numpy as np


def _finish_plot(save_path=None, show=True):
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return plt.gcf()


def plot_metric_over_time(metric_values, title="Metric over time", ylabel="Metric", save_path=None, show=True):
    """Plot a metric such as accuracy across streaming chunks."""
    values = np.asarray(metric_values, dtype=float)
    plt.figure()
    plt.plot(np.arange(1, values.size + 1), values, marker="o")
    plt.title(title)
    plt.xlabel("Chunk")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    return _finish_plot(save_path, show)


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"), save_path=None, show=True):
    """Compare two streaming metric sequences on the same chart."""
    m1 = np.asarray(metric1, dtype=float)
    m2 = np.asarray(metric2, dtype=float)
    plt.figure()
    plt.plot(np.arange(1, m1.size + 1), m1, marker="o", label=labels[0])
    plt.plot(np.arange(1, m2.size + 1), m2, marker="s", label=labels[1])
    plt.title("Model comparison over chunks")
    plt.xlabel("Chunk")
    plt.ylabel("Metric value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    return _finish_plot(save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, save_path=None, show=True):
    """Visualise latest-chunk predicted labels against true labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    x = np.arange(y_true.size)
    plt.figure()
    plt.scatter(x, y_true, label="Ground truth", marker="o")
    plt.scatter(x, y_pred, label="Prediction", marker="x")
    plt.title("Predictions vs ground truth")
    plt.xlabel("Sample index")
    plt.ylabel("Class label")
    plt.legend()
    plt.grid(True, alpha=0.3)
    return _finish_plot(save_path, show)
