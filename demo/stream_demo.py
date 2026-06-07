

import os
import sys
from pathlib import Path

import numpy as np

# Allow running the file directly from the project root without installing package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from numcompute.ensemble import EnsembleClassifier
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler
from numcompute.stream import StreamTrainer
from numcompute.visualise import plot_metric_over_time


def make_stream(n_samples=240, n_features=4, chunk_size=30, random_state=42):
    """Create binary classification data and yield it in chunks."""
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))

    # Non-trivial target rule so the tree ensemble has something to learn.
    logits = 1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.5 * X[:, 2]
    y = (logits > 0).astype(int)

    for start in range(0, n_samples, chunk_size):
        end = start + chunk_size
        yield X[start:end], y[start:end]


def main():
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", EnsembleClassifier(
            n_estimators=5,
            max_depth=4,
            max_features=0.75,
            random_state=7,
        )),
    ])

    trainer = StreamTrainer(estimator=pipeline, labels=[0, 1])
    cumulative_accuracies = []

    print("Streaming Assignment 2.2 demo")
    print("Chunk | Samples | Chunk Accuracy | Cumulative Accuracy")
    print("------|---------|----------------|--------------------")

    for chunk_id, (X_chunk, y_chunk) in enumerate(make_stream(), start=1):
        trainer.fit_chunk(X_chunk, y_chunk)
        score_log = trainer.score_chunk(X_chunk, y_chunk)
        cumulative_accuracies.append(score_log["cumulative_accuracy"])

        print(
            f"{chunk_id:5d} | "
            f"{score_log['n_samples']:7d} | "
            f"{score_log['chunk_accuracy']:.3f}          | "
            f"{score_log['cumulative_accuracy']:.3f}"
        )

    final_metrics = trainer.metric_tracker.result()
    print("\nFinal streaming metrics")
    for key, value in final_metrics.items():
        if key == "confusion_matrix":
            print(f"{key}:\n{value}")
        else:
            print(f"{key}: {value}")

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / "streaming_accuracy.png"
    plot_metric_over_time(
        cumulative_accuracies,
        title="Cumulative Accuracy Across Streaming Chunks",
        ylabel="Accuracy",
        save_path=str(plot_path),
        show=False,
    )
    print(f"\nSaved plot: {plot_path}")


if __name__ == "__main__":
    main()
