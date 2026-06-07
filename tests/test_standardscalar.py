from env import * # ensure full env tests are implemented
from numcompute.preprocessing import *



class TestStandardScaler(unittest.TestCase):

    def setUp(self):
        """Init: uses same data for each test"""
        self.scaler = StandardScaler()
        self.X_normal = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        
    def test_standard_scaling(self):
        """Tests standard z-score normalization without edge cases."""
        X_scaled = self.scaler.fit_transform(self.X_normal)
        
        # Expected means: [3.0, 4.0]
        # Expected stds: [1.63299, 1.63299] (population std, ddof=0)
        
        # Check that the means of the scaled data are exactly 0
        assert_allclose(np.mean(X_scaled, axis=0), [0.0, 0.0], atol=1e-7)
        # Check that the variances are exactly 1
        assert_allclose(np.std(X_scaled, axis=0), [1.0, 1.0], atol=1e-7)

    def test_zero_variance_edge_case(self):
        """Tests that columns with identical values do not cause division by zero."""
        X_zero_var = np.array([
            [5.0, 1.0],
            [5.0, 2.0],
            [5.0, 3.0]
        ])
        
        X_scaled = self.scaler.fit_transform(X_zero_var)
        
        # The first column should just be centered at 0 (since variance was 0, we divided by 1.0)
        expected_col_0 = np.array([0.0, 0.0, 0.0])
        assert_array_equal(X_scaled[:, 0], expected_col_0)
        
    def test_nan_handling(self):
        """Tests that NaNs are ignored during parameter calculation."""
        X_with_nan = np.array([
            [1.0, 2.0],
            [np.nan, 4.0],
            [5.0, 6.0]
        ])
        
        self.scaler.fit(X_with_nan)
        
        # The mean of column 0 should be exactly 3.0 (ignoring the NaN)
        self.assertEqual(self.scaler.mean_[0], 3.0)
        
    def test_transform_before_fit_raises_error(self):
        """Tests our BaseTransformer safety mechanism."""
        with self.assertRaises(RuntimeError):
            self.scaler.transform(self.X_normal)

if __name__ == '__main__':
    unittest.main()