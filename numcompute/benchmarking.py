import time
import contextlib
import numpy as np


# 1. Low-level timer

@contextlib.contextmanager
def timer(label=""):
   
    record = {"elapsed": None}
    start = time.perf_counter()
    try:
        yield record
    finally:
        record["elapsed"] = time.perf_counter() - start
        if label:
            print(f"[timer] {label}: {record['elapsed']:.6f}s")


# 2. Single-function benchmark

def benchmark(fn, n_runs=100, warmup=5, label=""):

    # 2.1 Input validation
    if not callable(fn):
        raise TypeError(f"'fn' must be callable. Got: {type(fn)}")
    if not isinstance(n_runs, int) or n_runs < 1:
        raise ValueError(f"'n_runs' must be a positive integer. Got: {n_runs}")
    if not isinstance(warmup, int) or warmup < 0:
        raise ValueError(f"'warmup' must be a non-negative integer. Got: {warmup}")

    # 2.2 Warmup (results discarded)
    for _ in range(warmup):
        fn()

    # 2.3 Timed runs (fully vectorised timing with perf_counter)
    times = np.empty(n_runs, dtype=float)
    for i in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times[i] = time.perf_counter() - t0

    return {
        "label":   label,
        "n_runs":  n_runs,
        "mean_s":  float(np.mean(times)),
        "std_s":   float(np.std(times)),
        "min_s":   float(np.min(times)),
        "max_s":   float(np.max(times)),
        "total_s": float(np.sum(times)),
        "times":   times,
    }


# 3. Head-to-head comparison

def compare(label, vectorised_fn, loop_fn, n_runs=100, warmup=5):
    
    _print_comparison(label, vec_stats, loop_stats, speedup)

    return {
        "label":      label,
        "vectorised": vec_stats,
        "loop":       loop_stats,
        "speedup":    speedup,
    }


def _print_comparison(label, vec_stats, loop_stats, speedup):

    col = 20  # column width for alignment
    sep = "-" * 65
    print(f"\n{'='*65}")
    print(f"  Benchmark: {label}")
    print(sep)
    print(f"  {'Metric':<{col}} {'Vectorised':>15} {'Loop':>15}")
    print(sep)

    rows = [
        ("Mean time",  f"{vec_stats['mean_s']*1e3:.4f} ms",  f"{loop_stats['mean_s']*1e3:.4f} ms"),
        ("Std dev",    f"{vec_stats['std_s']*1e3:.4f} ms",   f"{loop_stats['std_s']*1e3:.4f} ms"),
        ("Min time",   f"{vec_stats['min_s']*1e3:.4f} ms",   f"{loop_stats['min_s']*1e3:.4f} ms"),
        ("Max time",   f"{vec_stats['max_s']*1e3:.4f} ms",   f"{loop_stats['max_s']*1e3:.4f} ms"),
        ("Runs",       str(vec_stats['n_runs']),              str(loop_stats['n_runs'])),
    ]

    for name, v_val, l_val in rows:
        print(f"  {name:<{col}} {v_val:>15} {l_val:>15}")

    print(sep)
    winner = "Vectorised" if speedup >= 1.0 else "Loop"
    ratio  = speedup if speedup >= 1.0 else 1.0 / speedup
    print(f"  Speedup: {winner} is {ratio:.2f}x faster")
    print(f"{'='*65}\n")


# 4. BenchmarkSuite — collect and report multiple comparisons

class BenchmarkSuite:


    def __init__(self, name="Benchmark Suite"):
        self.name = name
        self._cases = []      # list of (label, vec_fn, loop_fn)
        self._results = []    # populated by run_all()

    def add(self, label, vectorised_fn, loop_fn):

        if not callable(vectorised_fn) or not callable(loop_fn):
            raise TypeError("Both 'vectorised_fn' and 'loop_fn' must be callable.")
        self._cases.append((label, vectorised_fn, loop_fn))
        return self

    def run_all(self, n_runs=100, warmup=5):

        self._results.clear()
        for label, vec_fn, loop_fn in self._cases:
            result = compare(label, vec_fn, loop_fn, n_runs=n_runs, warmup=warmup)
            self._results.append(result)
        return self

    def summary(self):
     
        if not self._results:
            print("No results yet. Call .run_all() first.")
            return []

        col_l = 30   # label column width
        col_v = 14   # value column width
        sep = "=" * (col_l + col_v * 3 + 6)

        print(f"\n{sep}")
        print(f"  {self.name}")
        print(sep)
        print(f"  {'Operation':<{col_l}} {'Vec (ms)':>{col_v}} {'Loop (ms)':>{col_v}} {'Speedup':>{col_v}}")
        print("-" * (col_l + col_v * 3 + 6))

        for r in self._results:
            v_ms = r["vectorised"]["mean_s"] * 1e3
            l_ms = r["loop"]["mean_s"] * 1e3
            sp   = r["speedup"]
            winner_marker = "✓" if sp >= 1.0 else "✗"
            print(
                f"  {r['label']:<{col_l}} "
                f"{v_ms:>{col_v}.4f} "
                f"{l_ms:>{col_v}.4f} "
                f"{sp:>{col_v-1}.2f}x {winner_marker}"
            )

        avg_speedup = float(np.mean([r["speedup"] for r in self._results]))
        print("-" * (col_l + col_v * 3 + 6))
        print(f"  {'Average speedup':<{col_l}} {'':>{col_v}} {'':>{col_v}} {avg_speedup:>{col_v-1}.2f}x")
        print(f"{sep}\n")

        return self._results


# 5. Default benchmark suite — demonstrates NumCompute vs Python loops

def run_default_benchmarks(n_rows=5_000, n_cols=20, n_runs=100):

    rng = np.random.default_rng(seed=42)
    X   = rng.standard_normal((n_rows, n_cols))

    print(f"\n{'#'*65}")
    print(f"  NumCompute Default Benchmark Suite")
    print(f"  Array shape: ({n_rows}, {n_cols})  |  n_runs: {n_runs}")
    print(f"{'#'*65}")

    suite = BenchmarkSuite(name=f"NumCompute Vectorised vs Loop  [{n_rows}×{n_cols}]")

    # 5.1 Column mean
    def _loop_col_mean():
        result = np.empty(n_cols)
        for j in range(n_cols):
            total = 0.0
            for i in range(n_rows):
                total += X[i, j]
            result[j] = total / n_rows
        return result

    suite.add(
        label="Column mean",
        vectorised_fn=lambda: np.mean(X, axis=0),
        loop_fn=_loop_col_mean,
    )

    # 5.2 Column standard deviation
    def _loop_col_std():
        result = np.empty(n_cols)
        for j in range(n_cols):
            col = X[:, j]
            mu = sum(col) / n_rows
            result[j] = (sum((v - mu) ** 2 for v in col) / n_rows) ** 0.5
        return result

    suite.add(
        label="Column std dev",
        vectorised_fn=lambda: np.std(X, axis=0),
        loop_fn=_loop_col_std,
    )

    # 5.3 Min-Max normalisation
    def _loop_minmax():
        result = np.empty_like(X)
        for j in range(n_cols):
            col_min = min(X[i, j] for i in range(n_rows))
            col_max = max(X[i, j] for i in range(n_rows))
            denom   = col_max - col_min if col_max != col_min else 1.0
            for i in range(n_rows):
                result[i, j] = (X[i, j] - col_min) / denom
        return result

    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    denom   = np.where(col_max - col_min != 0, col_max - col_min, 1.0)

    suite.add(
        label="Min-Max normalisation",
        vectorised_fn=lambda: (X - col_min) / denom,
        loop_fn=_loop_minmax,
    )

    # 5.4 Row-wise dot product (matrix multiply)
    w = rng.standard_normal(n_cols)

    def _loop_dot():
        result = np.empty(n_rows)
        for i in range(n_rows):
            s = 0.0
            for j in range(n_cols):
                s += X[i, j] * w[j]
            result[i] = s
        return result

    suite.add(
        label="Row-wise dot product",
        vectorised_fn=lambda: X @ w,
        loop_fn=_loop_dot,
    )

    # 5.5 Softmax (numerically stable: subtract row max)
    def _loop_softmax():
        result = np.empty_like(X)
        for i in range(n_rows):
            row_max = max(X[i, j] for j in range(n_cols))
            exps    = [np.exp(X[i, j] - row_max) for j in range(n_cols)]
            s       = sum(exps)
            for j in range(n_cols):
                result[i, j] = exps[j] / s
        return result

    def _vec_softmax():
        shifted = X - X.max(axis=1, keepdims=True)
        e       = np.exp(shifted)
        return e / e.sum(axis=1, keepdims=True)

    suite.add(
        label="Softmax (stable)",
        vectorised_fn=_vec_softmax,
        loop_fn=_loop_softmax,
    )

    # 5.6 Sorting each column
    def _loop_sort_cols():
        result = np.empty_like(X)
        for j in range(n_cols):
            col = list(X[:, j])
            col.sort()
            result[:, j] = col
        return result

    suite.add(
        label="Sort each column",
        vectorised_fn=lambda: np.sort(X, axis=0),
        loop_fn=_loop_sort_cols,
    )

    # 5.7 L2 row norms
    def _loop_l2_norms():
        result = np.empty(n_rows)
        for i in range(n_rows):
            result[i] = sum(X[i, j] ** 2 for j in range(n_cols)) ** 0.5
        return result

    suite.add(
        label="Row L2 norms",
        vectorised_fn=lambda: np.linalg.norm(X, axis=1),
        loop_fn=_loop_l2_norms,
    )

    # Run everything and print summary
    suite.run_all(n_runs=n_runs, warmup=5)
    suite.summary()

    return suite


# Entry point

if __name__ == "__main__":
    # Run with a smaller array for quick local testing
    run_default_benchmarks(n_rows=2_000, n_cols=10, n_runs=50)
