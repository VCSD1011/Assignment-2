import numpy as np

def stable_sort(data):
    """Stable sort wrapper using NumPy's stable sort kind."""
    return np.sort(data, kind='stable')

def multi_key_sort(data, columns):
    """Sort by multiple columns by creating a structured array."""
    # Using lexsort for multi-key sorting; it sorts from last key to first
    keys = tuple(data[:, col] for col in reversed(columns))
    indices = np.lexsort(keys)
    return data[indices]



def topk(values, k, largest=True, return_indices=True):
    """Vectorised top-k using np.argpartition for O(n) performance."""
    values = np.asanyarray(values)
    if largest:
        idx = np.argpartition(values, -k)[-k:]
        idx = idx[np.argsort(values[idx])][::-1]  # Sort the partition
    else:
        idx = np.argpartition(values, k)[:k]
        idx = idx[np.argsort(values[idx])]
    return (values[idx], idx) if return_indices else values[idx]

def quickselect(a, k):
    """Recursive quickselect for educational purposes."""
    a = np.array(a)
    if len(a) == 1: return a[0]
    pivot = a[np.random.randint(0, len(a))]
    
    lows = a[a < pivot]
    highs = a[a > pivot]
    pivots = a[a == pivot]
    
    if k < len(lows):
        return quickselect(lows, k)
    elif k < len(lows) + len(pivots):
        return pivots[0]
    else:
        return quickselect(highs, k - len(lows) - len(pivots))
    
    
def binary_search(sorted_array, x):
    """Returns insertion index and existence boolean."""
    idx = np.searchsorted(sorted_array, x)
    exists = bool(idx < len(sorted_array) and sorted_array[idx] == x)
    return idx, exists