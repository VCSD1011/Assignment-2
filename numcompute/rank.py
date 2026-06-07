import numpy as np

def rank(data, method='average'):
    """
    Assigns ranks to data, handling ties based on the specified method.
    
    Args:
        data: Array-like input.
        method: 'average', 'dense', or 'ordinal'.
        
    Returns:
        np.ndarray: Ranks of the input data.
    """
    data = np.asanyarray(data)
    if data.size == 0:
        raise ValueError("Cannot rank an empty array.")

    # Get indices that would sort the array
    sorter = np.argsort(data)
    # Inverse permutation to map sorted ranks back to original positions
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)

    if method == 'ordinal':
        return (inv + 1).astype(float)

    # Find unique values and their counts/starting positions for tie-handling
    unique_vals, root_indices, counts = np.unique(data[sorter], return_index=True, return_counts=True)

    if method == 'dense':
        # cumulative sum of unique values mapped back via inverse indices
        return (np.arange(len(unique_vals)) + 1)[np.searchsorted(unique_vals, data)]
    
    if method == 'average':
        # Average rank: start_index + (count - 1) / 2 + 1
        avg_ranks = root_indices + (counts - 1) / 2.0 + 1
        return avg_ranks[np.searchsorted(unique_vals, data)]

    raise ValueError(f"Unknown ranking method: {method}")

def percentile(data, q, interpolation='linear'):
    """
    Compute the q-th percentile of the data.
    
    Args:
        data: Input array.
        q: Percentile or sequence of percentiles (0-100).
        interpolation: 'linear', 'lower', 'higher', 'midpoint'.
    """
    data = np.asanyarray(data)
    # Using numpy's built-in percentile as it is highly optimized and vectorized
    return np.percentile(data, q, method=interpolation)