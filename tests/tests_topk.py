from env import *
from numcompute.sort_search import topk, quickselect

class TestTopK(unittest.TestCase):
    def test_topk_largest_indices(self):
        data = np.array([10, 50, 20, 40, 30])
        vals, idx = topk(data, k=3, largest=True, return_indices=True)
        np.testing.assert_array_equal(vals, [50, 40, 30])
        np.testing.assert_array_equal(idx, [1, 3, 4])

    def test_topk_smallest(self):
        data = np.array([10, 50, 20, 40, 30])
        vals = topk(data, k=2, largest=False, return_indices=False)
        np.testing.assert_array_equal(vals, [10, 20])

    def test_quickselect_consistency(self):
        data = [45, 1, 10, 30, 25]
        # 0-indexed: 2nd smallest (k=1) should be 10
        result = quickselect(data, 1)
        self.assertEqual(result, 10)

    def test_topk_extreme_k(self):
        # Test when k equals the length of the array
        data = np.array([1, 2, 3])
        vals = topk(data, k=3)
        self.assertEqual(len(vals[0]), 3)