from env import *
from numcompute.sort_search import binary_search

class TestSearching(unittest.TestCase):
    def test_search_exists(self):
        arr = np.array([10, 20, 30, 40, 50])
        idx, found = binary_search(arr, 40)
        self.assertEqual(idx, 3)
        self.assertTrue(found)

    def test_search_not_found(self):
        arr = np.array([10, 20, 30, 40, 50])
        idx, found = binary_search(arr, 25)
        self.assertEqual(idx, 2) # Insertion index
        self.assertFalse(found)

    def test_search_boundaries(self):
        arr = np.array([10, 20, 30])
        # Test searching for something smaller than min
        idx_low, _ = binary_search(arr, 5)
        self.assertEqual(idx_low, 0)
        # Test searching for something larger than max
        idx_high, _ = binary_search(arr, 100)
        self.assertEqual(idx_high, 3)