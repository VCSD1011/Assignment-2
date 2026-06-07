from env import *
from numcompute.preprocessing import OneHotEncoder

class TestOneHotEncoder(unittest.TestCase):

    def setUp(self):
        self.encoder_error = OneHotEncoder(handle_unknown='error')
        self.encoder_ignore = OneHotEncoder(handle_unknown='ignore')
        
        # Mixed types (often represented as object or string arrays in NumPy)
        self.X_train = np.array([
            ['cat', 1],
            ['dog', 2],
            ['cat', 3]
        ], dtype=object)
        
    def test_standard_encoding(self):
        """Tests that categories are mapped correctly."""
        X_encoded = self.encoder_error.fit_transform(self.X_train)
        
        # Feature 0 unique: ['cat', 'dog']
        # Feature 1 unique: [1, 2, 3]
        # Expected Output Shape: (3 samples, 2 + 3 = 5 encoded columns)
        expected = np.array([
            [1., 0.,  1., 0., 0.], # cat, 1
            [0., 1.,  0., 1., 0.], # dog, 2
            [1., 0.,  0., 0., 1.]  # cat, 3
        ])
        
        assert_array_equal(X_encoded, expected)

    def test_handle_unknown_error(self):
        """Tests that unseen test data raises an error on default settings."""
        self.encoder_error.fit(self.X_train)
        
        X_test_unseen = np.array([['bird', 1]], dtype=object)
        
        with self.assertRaises(ValueError):
            self.encoder_error.transform(X_test_unseen)

    def test_handle_unknown_ignore(self):
        """Tests that unseen data drops to zero vectors when ignore is flagged."""
        self.encoder_ignore.fit(self.X_train)
        
        # 'bird' is unknown in col 0. '4' is unknown in col 1.
        X_test_unseen = np.array([['bird', 4]], dtype=object)
        X_encoded = self.encoder_ignore.transform(X_test_unseen)
        
        # The result should be an array of all zeros
        expected = np.array([[0., 0.,  0., 0., 0.]])
        assert_array_equal(X_encoded, expected)
        
    def test_transform_before_fit_raises_error(self):
        """Tests the BaseTransformer safety mechanism."""
        with self.assertRaises(RuntimeError):
            self.encoder_error.transform(self.X_train)

if __name__ == '__main__':
    unittest.main()