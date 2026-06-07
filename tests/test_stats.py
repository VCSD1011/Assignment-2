from env import *
from numcompute.stats import *
import unittest

class TestStatsModule(unittest.TestCase):

    def setUp(self):
        # Sample data with and without NaNs
        self.data_simple = [1, 2, 3, 4, 5]
        self.data_with_nan = [1, 2, np.nan, 4, 5]
        self.data_2d = [
            [1, 2, 3],
            [4, np.nan, 6]
        ]

    ## --- Individual Function Tests ---

    def test_mean(self):
        # Should ignore NaNs: (1+2+4+5)/4 = 3.0
        self.assertEqual(mean(self.data_with_nan), 3.0)
        self.assertEqual(mean(self.data_simple), 3.0)

    def test_median(self):
        # Median of [1, 2, 4, 5] is 3.0
        self.assertEqual(median(self.data_with_nan), 3.0)

    def test_std(self):
        # Standard deviation check with ddof=0
        res = std([1, 2, 3]) # mean=2, var = ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = 2/3
        self.assertAlmostEqual(res, np.sqrt(2/3))

    def test_min_max(self):
        self.assertEqual(minimum(self.data_with_nan), 1.0)
        self.assertEqual(maximum(self.data_with_nan), 5.0)

    def test_quantiles(self):
        # Testing 50th percentile (median)
        res = quantiles(self.data_simple, q=[0.5])
        self.assertEqual(res[0], 3.0)

    ## --- Multi-dimensional / Axis Tests ---

    def test_axis_handling(self):
        # Column-wise mean for [[1, 2, 3], [4, nan, 6]]
        # result should be [2.5, 2.0, 4.5]
        res = mean(self.data_2d, axis=0)
        np.testing.assert_array_equal(res, [2.5, 2.0, 4.5])

    ## --- Specialized Function Tests ---

    def test_histogram(self):
        # Histogram should return counts and bin edges
        counts, edges = histogram(self.data_with_nan, bins=2)
        self.assertEqual(len(counts), 2)
        # Total counts should be 4 (NaN is removed)
        self.assertEqual(np.sum(counts), 4)

    def test_stats_dictionary(self):
        # Comprehensive check for the dictionary output
        res = stats(self.data_simple)
        expected_keys = {'mean', 'median', 'std', 'min', 'max', 'q25', 'q75'}
        self.assertTrue(expected_keys.issubset(res.keys()))
        self.assertEqual(res['mean'], 3.0)
        self.assertEqual(res['min'], 1.0)

    ## --- Edge Cases ---

    def test_empty_input(self):
        # NumPy usually warns or returns NaN for empty slices
        with self.assertWarns(RuntimeWarning):
            res = mean([])
            self.assertTrue(np.isnan(res))

if __name__ == '__main__':
    unittest.main()