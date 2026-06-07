from env import *
from numcompute.rank import rank, percentile

class TestRank(unittest.TestCase):
    
    def test_rank_ordinal(self):
        # Unique values, simple order
        data = [10, 40, 30, 20]
        result = rank(data, method='ordinal')
        np.testing.assert_array_equal(result, [1, 4, 3, 2])

    def test_rank_average_ties(self):
        # Tie handling: (1+2)/2 = 1.5 for the two '10's
        data = [10, 10, 20, 30]
        result = rank(data, method='average')
        np.testing.assert_array_equal(result, [1.5, 1.5, 3.0, 4.0])

    def test_rank_dense_ties(self):
        # Dense ranking: ties get same rank, next rank is next integer
        data = [10, 10, 20, 30]
        result = rank(data, method='dense')
        np.testing.assert_array_equal(result, [1, 1, 2, 3])

    def test_percentile_linear(self):
        data = [1, 2, 3, 4, 5]
        # 50th percentile is the median
        self.assertEqual(percentile(data, 50), 3.0)

    def test_percentile_interpolation(self):
        data = [1, 10]
        # Linear (default) would be 5.5, 'lower' should be 1
        self.assertEqual(percentile(data, 50, interpolation='lower'), 1)

    def test_all_identical_elements(self):
        # Important edge case for rubric
        data = [5, 5, 5, 5]
        result = rank(data, method='average')
        # (1+2+3+4)/4 = 2.5
        np.testing.assert_array_equal(result, [2.5, 2.5, 2.5, 2.5])