from env import *
from numcompute.optim import grad, jacobian

class TestOptim(unittest.TestCase):

    def test_grad_central_and_forward(self):
        """Tests scalar function gradient: f(x, y) = x^2 + 3y"""
        def f(vec):
            return vec[0]**2 + 3 * vec[1]
            
        x = np.array([2.0, 5.0])
        
        # Analytical derivative: df/dx = 2x, df/dy = 3
        # At x=[2, 5], gradient = [4, 3]
        expected_grad = np.array([4.0, 3.0])
        
        grad_central = grad(f, x, method='central')
        grad_forward = grad(f, x, method='forward')
        
        # Central difference should be highly accurate
        assert_allclose(grad_central, expected_grad, atol=1e-5)
        assert_allclose(grad_forward, expected_grad, atol=1e-4)

    def test_jacobian(self):
        """Tests vector function Jacobian: F(x, y) = [x^2*y, 5x + sin(y)]"""
        def F(vec):
            x, y = vec[0], vec[1]
            return np.array([x**2 * y, 5 * x + np.sin(y)])
            
        x_eval = np.array([2.0, np.pi])
        
        # Analytical Jacobian:
        # [ 2xy,       x^2    ]
        # [ 5,         cos(y) ]
        # At x=[2, pi]:
        # [ 4*pi,      4      ]
        # [ 5,         -1     ]
        expected_J = np.array([
            [4 * np.pi, 4.0],
            [5.0, -1.0]
        ])
        
        J_numerical = jacobian(F, x_eval, method='central')
        
        assert_allclose(J_numerical, expected_J, atol=1e-5)

    def test_invalid_method(self):
        """Ensures the API gracefully rejects bad method strings."""
        def f(vec): return np.sum(vec)
        x = np.array([1.0, 2.0])
        
        with self.assertRaises(ValueError):
            grad(f, x, method='backward')

if __name__ == '__main__':
    unittest.main()