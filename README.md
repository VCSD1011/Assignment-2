# NumCompute Streaming – Assignment 2.2

Individual update by **Chandrasekaran V.**

NumCompute is a small machine learning and data-processing library built from scratch using Python and NumPy. This Assignment 2.2 version updates the previous Assignment 2.1 team project into an **individual streaming-learning project**. The main change is that the core components now support chunk-based updates through `partial_fit()` and streaming metric/statistic accumulation.

---

## Assignment 2.2 Focus

The project now supports streaming settings where data arrives in chunks instead of one complete batch. The updated library includes:

- Incremental preprocessing using `partial_fit()`
- Streaming descriptive statistics using `update_stats()`
- Streaming classification metrics using `update()`, `result()`, and `reset()`
- A depth-limited decision tree classifier with chunk updates
- A bagging-style ensemble classifier with streaming adaptation
- A pipeline that passes each chunk through transformers and models
- A `StreamTrainer` that logs per-chunk accuracy, cumulative accuracy, memory use, and timing
- Reusable matplotlib visualisation functions for streaming experiments

---

## Project Structure

```text
numcompute/
├── preprocessing.py   # StandardScaler, MinMaxScaler, OneHotEncoder, Imputer with partial_fit
├── stats.py           # Batch stats + StreamingStats
├── metrics.py         # Batch metrics + StreamingClassificationMetrics
├── pipeline.py        # Pipeline with fit, transform, predict, partial_fit
├── tree.py            # DecisionTreeClassifier
├── ensemble.py        # EnsembleClassifier
├── stream.py          # StreamTrainer
├── visualise.py       # Plotting functions
├── io.py              # CSV loading helpers from Assignment 2.1
├── sort_search.py     # Sorting/searching algorithms from Assignment 2.1
├── rank.py            # Ranking and percentile utilities from Assignment 2.1
├── optim.py           # Gradient/Jacobian utilities from Assignment 2.1
└── utils.py           # BaseEstimator and BaseTransformer
```

---

## Quick Example

```python
import numpy as np
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler
from numcompute.ensemble import EnsembleClassifier
from numcompute.stream import StreamTrainer

X1 = np.array([[0, 0], [0, 1], [1, 0]], dtype=float)
y1 = np.array([0, 0, 1])

X2 = np.array([[1, 1], [2, 2], [2, 3]], dtype=float)
y2 = np.array([1, 1, 1])

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", EnsembleClassifier(n_estimators=3, max_depth=3, random_state=1)),
])

trainer = StreamTrainer(pipe, labels=[0, 1])
trainer.fit_chunk(X1, y1)
print(trainer.score_chunk(X1, y1))

trainer.fit_chunk(X2, y2)
print(trainer.score_chunk(X2, y2))
```

---

## Streaming APIs

### Preprocessing

```python
scaler = StandardScaler()
scaler.partial_fit(X_chunk_1)
scaler.partial_fit(X_chunk_2)
X_scaled = scaler.transform(X_latest)
```

`StandardScaler` maintains running mean and variance. `MinMaxScaler` updates feature minima and maxima. `Imputer` updates missing-value estimates, and `OneHotEncoder` expands categories as new chunks arrive.

### Pipeline

```python
pipe.partial_fit(X_chunk, y_chunk)
y_pred = pipe.predict(X_chunk)
```

Intermediate steps are updated and transformed one by one. The final model receives the processed chunk.

### Metrics

```python
metrics = StreamingClassificationMetrics(labels=[0, 1])
metrics.update(y_true_chunk, y_pred_chunk)
print(metrics.result())
metrics.reset()
```

The result includes accuracy, precision, recall, F1, confusion matrix, total sample count, and AUC when score values are provided.

### Statistics

```python
from numcompute.stats import StreamingStats

s = StreamingStats(bins=10)
s.update_stats(X_chunk_1)
s.update_stats(X_chunk_2)
print(s.result())
```

The result includes count, mean, variance, standard deviation, min, max, quantiles, and histogram.

---

## Testing

Run all tests from the project root:

```bash
pytest -q
```

Current local result after the Assignment 2.2 update:

```text
46 passed
```

The test suite covers previous Assignment 2.1 functionality and new Assignment 2.2 streaming features.

---



The code follows readable Python style with clear class names, method names, docstrings, and comments where useful.
