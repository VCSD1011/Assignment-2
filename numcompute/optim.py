import numpy as np

def grad(f, x, h=1e-5, method='central'):
    """Estimates the gradient of a scalar-valued function f at vector x.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    
    if method not in ['central', 'forward']:
        raise ValueError("method must be 'central' or 'forward'")
        
    g = np.zeros_like(x)
    
    f_x = f(x) if method == 'forward' else None # pre-calculate f(x) for the forward method to save compute time
    
    # Loop over the dimensions
    for i in range(n):
        h_vec = np.zeros_like(x)
        h_vec[i] = h
        
        if method == 'central':
            # f(x + h) - f(x - h) / 2h
            g[i] = (f(x + h_vec) - f(x - h_vec)) / (2.0 * h)
        else:
            # f(x + h) - f(x) / h
            g[i] = (f(x + h_vec) - f_x) / h
            
    return g

def jacobian(F, x, h=1e-5, method='central'):

    x = np.asarray(x, dtype=float)
    n = x.size
    
    if method not in ['central', 'forward']:
        raise ValueError("method must be 'central' or 'forward'")
        
    F_x = F(x)
    F_x = np.asarray(F_x, dtype=float)
    m = F_x.size
    
    # Initialize the Jacobian matrix (m outputs x n inputs)
    J = np.zeros((m, n))
    
    for i in range(n):
        h_vec = np.zeros_like(x)
        h_vec[i] = h
        
        if method == 'central':
            J[:, i] = (F(x + h_vec) - F(x - h_vec)) / (2.0 * h)
        else:
            J[:, i] = (F(x + h_vec) - F_x) / h
            
    return J
