import numpy as np

class BaseEstimator:
    """Base class for all estimators in NumCompute.
    """
    def __init__(self):
        self._is_fitted = False

    def fit(self, X, y=None):
        """Fits the estimator to the data.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            y (np.ndarray, optional): Target values. Defaults to None.
            
        Returns:
            self: The fitted instance.
        """
        # Validation logic goes here
        
        self._is_fitted = True # ensures .fit() has been called prior to other action
        return self

class BaseTransformer(BaseEstimator):
    """Base class for transformers (scalers, imupters, etc.)."""
    
    def transform(self, X):
        """Applies transformation to the input data. 
        
        When implementing, first call super.transform() to inherit code execution.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Transformed data.
        """
        if not self._is_fitted:
            raise RuntimeError("Estimator must be fitted before transforming.")
        return X

    def fit_transform(self, X, y=None):
        """Fits and transforms in a single vectorized call."""
        return self.fit(X, y).transform(X)