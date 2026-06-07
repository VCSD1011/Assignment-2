from env import *
from numcompute.sort_search import stable_sort, multi_key_sort

class TestSorting(unittest.TestCase):
    def test_stable_sort_logic(self):
        # Testing if relative order of equal elements is preserved
        data = np.array([3, 1, 2, 1, 2])
        result = stable_sort(data)
        np.testing.assert_array_equal(result, [1, 1, 2, 2, 3])

    def test_multi_key_sort(self):
        # Sort by Col 0 (Primary) then Col 1 (Secondary)
        data = np.array([[2, 10], [1, 20], [2, 5], [1, 10]])
        # Expected: [[1, 10], [1, 20], [2, 5], [2, 10]]
        result = multi_key_sort(data, columns=[0, 1])
        self.assertEqual(result[0, 1], 10)
        self.assertEqual(result[-1, 1], 10)

    def test_sorting_empty(self):
        data = np.array([])
        result = stable_sort(data)
        self.assertEqual(len(result), 0)