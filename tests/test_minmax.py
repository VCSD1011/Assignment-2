from env import *
from numcompute.preprocessing import MinMaxScaler


class TestMinMaxScaler(unittest.TestCase):

    def setUp(self):
        """Initializes scalers and standard data before each test."""
        self.scaler_default = MinMaxScaler() # Defaults to (0, 1)
        self.scaler_custom = MinMaxScaler(f_range=(-1.0, 1.0))
        
        self.X_normal = np.array([
            [1.0, -1.0],
            [2.0,  0.0],
            [3.0,  1.0]
        ])
        
    def test_default_range_scaling(self):
        """Tests that data scales properly to the default [0, 1] range."""
        X_scaled = self.scaler_default.fit_transform(self.X_normal)
        
        expected = np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0]
        ])
        assert_allclose(X_scaled, expected)

    def test_custom_range_scaling(self):
        """Tests that the scaler successfully maps to a custom defined range."""
        X_scaled = self.scaler_custom.fit_transform(self.X_normal)
        
        # Mapping [1, 2, 3] to [-1, 1] results in [-1, 0, 1]
        expected = np.array([
            [-1.0, -1.0],
            [ 0.0,  0.0],
            [ 1.0,  1.0]
        ])
        assert_allclose(X_scaled, expected)

    def test_zero_variance_edge_case(self):
        """Tests that columns with identical values do not cause division by zero."""
        X_zero_var = np.array([
            [5.0, 2.0],
            [5.0, 4.0],
            [5.0, 6.0]
        ])
        
        X_scaled = self.scaler_default.fit_transform(X_zero_var)
        
        # The first column is constant. The mathematical fallback we wrote 
        # should collapse it to the minimum of the feature range (0.0).
        expected_col_0 = np.array([0.0, 0.0, 0.0])
        assert_array_equal(X_scaled[:, 0], expected_col_0)
        
    def test_nan_handling(self):
        """Tests that NaNs are ignored during minimum and maximum calculation."""
        X_with_nan = np.array([
            [1.0, 2.0],
            [np.nan, 4.0],
            [5.0, 6.0]
        ])
        
        self.scaler_default.fit(X_with_nan)
        
        # The min and max of column 0 should be 1.0 and 5.0 (ignoring the NaN)
        self.assertEqual(self.scaler_default.X_min[0], 1.0)
        self.assertEqual(self.scaler_default.X_max[0], 5.0)

    def test_transform_before_fit_raises_error(self):
        """Tests the BaseTransformer safety mechanism."""
        with self.assertRaises(RuntimeError):
            self.scaler_default.transform(self.X_normal)

if __name__ == '__main__':
    unittest.main()